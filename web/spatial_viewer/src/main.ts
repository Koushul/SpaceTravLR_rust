import "./style.css";

import { Deck } from "@deck.gl/core";
import { OrthographicView } from "@deck.gl/core";
import { LineLayer, ScatterplotLayer } from "@deck.gl/layers";
import {
  applyBetadataColorsPerCluster,
  applyColors,
  colormapLegendGradientCss,
  type ColormapId,
} from "./colormaps";
import { rgbForCategoryIndex } from "./cellTypePalette";

const CT_UNKNOWN = 65535;

const ORTHO_CONTROLLER = {
  dragPan: true,
  scrollZoom: { speed: 0.01, smooth: false },
  touchZoom: true,
  doubleClickZoom: true,
  dragRotate: false,
  touchRotate: false,
  keyboard: true,
} as const;

async function fetchF32(path: string): Promise<Float32Array> {
  const r = await fetch(path);
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}`);
  }
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

async function fetchU32(path: string): Promise<Uint32Array> {
  const r = await fetch(path);
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}`);
  }
  const buf = await r.arrayBuffer();
  return new Uint32Array(buf);
}

interface Meta {
  n_obs: number;
  n_vars: number;
  spatial_obsm_key: string;
  layer: string;
  cluster_annot: string;
  bounds: { min_x: number; max_x: number; min_y: number; max_y: number };
  umap_obsm_key?: string | null;
  umap_bounds?: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  } | null;
  cell_type_column?: string | null;
  cell_type_categories?: string[];
  network_loaded?: boolean;
  network_species?: string | null;
  /** `Cluster` = seed-only lasso; `CellID` = spatial CNN per-cell β */
  betadata_row_id?: string | null;
  perturb_ready?: boolean;
  /** True while betabase / PerturbRuntime is loading in the server background. */
  perturb_loading?: boolean;
  perturb_error?: string | null;
  /** 0–100 while loading betabase or running a perturb job (see /api/meta). */
  perturb_progress_percent?: number | null;
  perturb_progress_label?: string | null;
  adata_path: string;
  betadata_dir: string;
  network_dir?: string | null;
  run_toml?: string | null;
  /** False until a .h5ad is loaded (CLI or Dataset paths). */
  dataset_ready?: boolean;
}

interface SessionConfigureResponse {
  ok: boolean;
  message: string;
  meta: Meta;
}

interface UmapFieldResponse {
  nx: number;
  ny: number;
  grid_x: number[];
  grid_y: number[];
  u: number[];
  v: number[];
  cell_u?: number[];
  cell_v?: number[];
}

interface CellContextResponse {
  focus_gene: string;
  cell_index: number;
  modulators: {
    regulators: string[];
    ligands: string[];
    receptors: string[];
    tfl_ligands: string[];
    tfl_regulators: string[];
    lr_pairs: string[];
    tfl_pairs: string[];
  };
  neighbors: {
    index: number;
    distance_sq: number;
    max_support_score?: number | null;
    lr_edges: {
      ligand: string;
      receptor: string;
      lig_expr_sender: number;
      rec_expr_neighbor: number;
      support_score: number;
      linked_tf?: string;
      linked_tf_expr?: number;
    }[];
  }[];
  sender_regulator_exprs: { gene: string; expr: number }[];
  sender_ligand_exprs: { gene: string; expr: number }[];
  neighbor_query?: string | null;
  radius_used?: number | null;
  neighbors_in_query?: number | null;
}

interface InteractionLineDatum {
  sourcePosition: [number, number, number];
  targetPosition: [number, number, number];
  color?: [number, number, number, number];
}

interface QuiverSegDatum extends InteractionLineDatum {
  width?: number;
}

function fitOrthographic(
  width: number,
  height: number,
  b: Meta["bounds"],
): { target: [number, number]; zoom: number } {
  const pad = 0.06;
  const dw = (b.max_x - b.min_x) * (1 + pad * 2) || 1;
  const dh = (b.max_y - b.min_y) * (1 + pad * 2) || 1;
  const cx = (b.min_x + b.max_x) / 2;
  const cy = (b.min_y + b.max_y) / 2;
  const scale = Math.min(width / dw, height / dh);
  const zoom = Math.log2(Math.max(scale, 1e-6));
  return { target: [cx, cy], zoom };
}

function meanForMask(values: Float32Array | null, mask: Uint8Array): string {
  if (!values) return "—";
  let s = 0;
  let c = 0;
  for (let i = 0; i < values.length; i++) {
    if (mask[i]) {
      s += values[i]!;
      c++;
    }
  }
  if (c === 0) return "—";
  return (s / c).toPrecision(5);
}

function globalMean(values: Float32Array | null): string {
  if (!values || values.length === 0) return "—";
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i]!;
  return (s / values.length).toPrecision(5);
}

function selectInRect(
  deck: Deck,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  positions: Float32Array,
  n: number,
  mask: Uint8Array,
  allowed: (i: number) => boolean,
): void {
  mask.fill(0);
  const vps = deck.getViewports();
  if (!vps.length) return;
  const vp = vps[0]!;
  const a = vp.unproject([x0, y0], { topLeft: true });
  const b = vp.unproject([x1, y1], { topLeft: true });
  const wx0 = Math.min(a[0], b[0]);
  const wx1 = Math.max(a[0], b[0]);
  const wy0 = Math.min(a[1], b[1]);
  const wy1 = Math.max(a[1], b[1]);
  for (let i = 0; i < n; i++) {
    const px = positions[i * 2]!;
    const py = positions[i * 2 + 1]!;
    if (
      px >= wx0 &&
      px <= wx1 &&
      py >= wy0 &&
      py <= wy1 &&
      allowed(i)
    ) {
      mask[i] = 1;
    }
  }
}

async function main() {
  const root = document.querySelector<HTMLDivElement>("#app")!;
  root.innerHTML = `
    <div class="app-layout">
      <aside class="app-sidebar" id="appSidebar">
        <div class="sidebar-top">
          <button
            type="button"
            class="secondary sidebar-collapse-btn"
            id="toggleToolbar"
            aria-expanded="true"
            title="Collapse or expand control panel"
          >
            Hide controls
          </button>
        </div>
        <div class="sidebar-scroll">
      <details class="session-panel" id="sessionPanel">
        <summary class="session-summary">Dataset paths</summary>
        <div class="session-grid">
          <label class="session-field session-field-span2"
            ><span class="session-label">AnnData (.h5ad)</span>
            <input
              type="text"
              id="sessionAdataPath"
              spellcheck="false"
              autocomplete="off"
              placeholder="/path/to/data.h5ad"
          /></label>
          <label class="session-field"
            ><span class="session-label">Expression layer</span>
            <input type="text" id="sessionLayer" placeholder="imputed_count"
          /></label>
          <label class="session-field"
            ><span class="session-label">Cluster column</span>
            <input type="text" id="sessionClusterAnnot" placeholder="cell_type"
          /></label>
          <label class="session-field session-field-span2"
            ><span class="session-label">Network dir (optional)</span>
            <input
              type="text"
              id="sessionNetworkDir"
              spellcheck="false"
              autocomplete="off"
              placeholder="Leave empty for default GRN search"
          /></label>
          <label class="session-field session-field-span2"
            ><span class="session-label">Run TOML (optional)</span>
            <input
              type="text"
              id="sessionRunToml"
              spellcheck="false"
              autocomplete="off"
              placeholder="…/run_dir/spacetravlr_run_repro.toml — enables perturbation + betadata from that directory"
          /></label>
          <div class="session-actions session-field-span2">
            <button type="button" class="primary" id="sessionApply">
              Load dataset
            </button>
            <span class="session-busy hidden" id="sessionBusy">Loading…</span>
          </div>
        </div>
      </details>
      <div class="control-section">
        <h2 class="control-section-title">Color &amp; data</h2>
      <div class="toolbar toolbar-sidebar">
      <label>Color source
        <select id="colorSource">
          <option value="expression">Expression</option>
          <option value="betadata" id="colorSourceBetaOpt">Betadata</option>
          <option value="perturb" id="colorSourcePerturbOpt" class="hidden">
            Perturbation (KO)
          </option>
        </select>
      </label>
      <div class="perturb-toolbar hidden" id="perturbPanel">
        <label>Gene to perturb
          <input id="perturbGene" list="geneHints" placeholder="var symbol" />
        </label>
        <label>Target expr
          <input type="number" id="perturbExpr" value="0" step="any" />
        </label>
        <label>Where to apply
          <select id="perturbScope">
            <option value="all">All cells</option>
            <option value="selection">Current selection</option>
            <option value="cell_type">One cell type (annotation)</option>
            <option value="cluster">One cluster id</option>
          </select>
        </label>
        <label id="perturbCellTypeWrap" class="hidden"
          >Cell type
          <select id="perturbCellType"><option value="">—</option></select>
        </label>
        <label id="perturbClusterWrap" class="hidden"
          >Cluster id (<code>--cluster-annot</code>)
          <input type="number" id="perturbClusterId" min="0" step="1" />
        </label>
        <button type="button" id="clearPerturb" class="secondary">
          Clear perturb Δ
        </button>
      </div>
      <label id="exprGeneWrap">Gene (var)
        <input id="exprGene" list="geneHints" placeholder="e.g. CD3E" />
        <datalist id="geneHints"></datalist>
      </label>
      <label id="betaGeneWrap" class="hidden">Betadata target
        <select id="betaGene"><option value="">—</option></select>
      </label>
      <label id="betaColWrap" class="hidden">Coefficient
        <select id="betaCol"><option value="">—</option></select>
      </label>
      <label>Colormap
        <select id="cmap">
          <option value="viridis">Viridis</option>
          <option value="magma">Magma</option>
          <option value="diverging">Diverging (RdBu)</option>
        </select>
      </label>
      <label class="toolbar-cell-size"
        >Cell size <span id="cellSizeVal">4</span> px
        <input
          type="range"
          id="cellSize"
          min="0.5"
          max="24"
          step="0.5"
          value="4"
        />
      </label>
      <label id="layoutToggleWrap" class="hidden"
        >Layout
        <select id="layoutMode">
          <option value="spatial">Spatial</option>
          <option value="umap">UMAP</option>
        </select>
      </label>
      <button type="button" id="loadColor">Load / refresh</button>
      <label><input type="checkbox" id="brushToggle" /> Rect select</label>
      <button type="button" id="clearSel" class="secondary">Clear selection</button>
      <p class="toolbar-hint">
        Click a cell to select one · Shift+click to add or remove. Betadata feathers are
        <strong>seed-only</strong> (<code>Cluster</code> rows, mapped by
        <code>--cluster-annot</code>) or <strong>spatial</strong> (<code>CellID</code>, per-cell
        β); status line shows which was detected.
      </p>
      </div>
      </div>
    <details class="transition-details hidden" id="umapQuiverPanel">
      <summary class="transition-summary">UMAP transition quiver</summary>
      <div class="transition-inner">
        <p class="transition-hint">
          Same pipeline as Python <code>VirtualTissue.plot_arrows</code> →
          <code>Cartography.plot_umap_quiver</code> (see
          <a
            href="https://spacetravlr.readthedocs.io/en/latest/ligand_perturbation.html"
            target="_blank"
            rel="noopener"
            >Ligand Perturbation tutorial</a
          >): server runs perturbation (gene, target, scope above), builds δ, KNN transition on UMAP,
          bins to a grid, scales arrows. Does <strong>not</strong> reuse <strong>Load / refresh</strong>
          colors. <strong>Quick sanity</strong>: single-gene δ to target, no GRN. Use
          <strong>UMAP</strong> layout to see arrows.
        </p>
        <div class="transition-grid">
          <label
            ><code>n_neighbors</code>
            <input type="number" id="transNeighbors" min="5" max="500" value="150" />
          </label>
          <label
            >Temperature <code>T</code>
            <input type="number" id="transT" value="0.05" step="0.005" />
          </label>
          <label
            ><code>grid_scale</code>
            <input type="number" id="transGridScale" value="1" step="0.1" min="0.1" />
          </label>
          <label
            ><code>vector_scale</code>
            <input type="number" id="transVecScale" value="0.85" step="0.05" min="0.01" />
          </label>
          <label
            ><code>rescale</code> (δ before transition)
            <input type="number" id="transDeltaRescale" value="1" step="0.1" />
          </label>
          <label
            ><code>threshold</code> (grid arrow mag.)
            <input type="number" id="transMagThresh" value="0" step="0.001" min="0" />
          </label>
          <label class="transition-span2"
            ><input type="checkbox" id="transRemoveNull" checked />
            <code>remove_null</code></label
          >
          <label class="transition-span2"
            ><input type="checkbox" id="transUnitDirs" />
            <code>normalize</code> (unit directions on UMAP)</label
          >
          <label class="transition-span2"
            ><input type="checkbox" id="transQuickKo" /> Quick sanity (single-gene δ to target;
            no GRN)</label
          >
          <label class="transition-span2"
            ><input type="checkbox" id="transFullGraph" /> Dense colΔCor (slow; small
            <em>n</em> only)</label
          >
          <label
            >Full-graph max cells
            <input type="number" id="transFullMax" min="64" max="8192" value="4096" />
          </label>
          <div class="trans-limit-wrap hidden" id="transLimitWrap">
            <label class="trans-limit-label"
              ><input type="checkbox" id="transLimitClusters" />
              <code>limit_clusters</code> — δ only from selected types (others → 0, like Python)</label
            >
            <label class="trans-highlight-label"
              ><code>highlight_cell_types</code> (hold ⌘/Ctrl to multi-select)
              <select
                id="transHighlightTypes"
                class="trans-highlight-select"
                multiple
                size="5"
              ></select>
            </label>
          </div>
        </div>
        <div class="transition-actions">
          <button
            type="button"
            class="primary"
            id="computeQuiverBtn"
            title="Runs KO/perturb on the server (same settings as above), then draws UMAP arrows"
          >
            Run perturb + UMAP quiver
          </button>
          <button type="button" class="secondary" id="perturbSummaryBtn">
            Perturbation summary
          </button>
          <button type="button" class="secondary" id="clearQuiverBtn">Clear quiver</button>
        </div>
        <p class="quiver-display-hint">
          <strong>Quiver display</strong> (instant, no recompute): arrow length, line width, head size,
          grid stride. Recompute when changing <code>n_neighbors</code>, <code>T</code>,
          <code>grid_scale</code> (tutorials often use <code>2</code>), <code>rescale</code>,
          <code>limit_clusters</code>, etc. Field math follows <code>cartography.py</code> /
          <code>shift.py</code>, not <code>gene_factory.py</code>.
        </p>
        <div class="transition-grid quiver-display-grid">
          <label class="toolbar-cell-size"
            >Vis. length <span id="quiverVisScaleVal">100</span>%
            <input
              type="range"
              id="quiverVisScale"
              min="10"
              max="300"
              step="5"
              value="100"
            />
          </label>
          <label class="toolbar-cell-size"
            >Line width <span id="quiverLineWVal">2</span> px
            <input
              type="range"
              id="quiverLineW"
              min="0.5"
              max="8"
              step="0.5"
              value="2"
            />
          </label>
          <label class="toolbar-cell-size"
            >Head size <span id="quiverHeadVal">28</span>%
            <input type="range" id="quiverHeadFrac" min="10" max="50" step="1" value="28" />
          </label>
          <label class="toolbar-cell-size"
            >Grid stride <span id="quiverStrideVal">1</span>
            <input type="range" id="quiverStride" min="1" max="6" step="1" value="1" />
          </label>
        </div>
        <div id="perturbSummaryBody" class="hidden" style="margin-top:6px;"></div>
        <p class="transition-note hidden" id="transUmapOnlyHint">
          Quiver data loaded — switch layout to <strong>UMAP</strong> to display arrows.
        </p>
      </div>
    </details>
    <details class="filter-details hidden" id="cellTypePanel">
      <summary class="filter-summary">Cell types &amp; overlay</summary>
      <div class="cell-type-bar-inner">
        <span class="cell-type-bar-title"
          >Column: <span id="cellTypeColName"></span></span
        >
        <label class="cell-type-overlay"
          ><input type="checkbox" id="cellTypeOverlay" /> Color by cell type</label
        >
        <div
          class="cell-type-checks"
          id="cellTypeFilters"
          title="Unchecked types are dimmed and excluded from click and rectangle selection"
        ></div>
      </div>
    </details>
    <details class="interaction-details hidden" id="interactionPanel">
      <summary class="interaction-summary">GRN neighbors (sender cell)</summary>
      <div class="interaction-details-inner">
        <p class="interaction-hint">
          Select one cell and enable <strong>Interaction lens</strong>. Shows GRN-supported L→R
          links to spatial kNN or radius neighbors (line color scales with support score).
        </p>
        <div class="interaction-controls interaction-controls-row1">
          <label class="interaction-mode-label"
            >Neighbor query
            <select id="interactionModeSel">
              <option value="knn">Sender + kNN</option>
              <option value="radius">Sender + radius</option>
            </select>
          </label>
        </div>
        <div class="interaction-controls">
          <label class="interaction-toggle"
            ><input type="checkbox" id="interactionLens" /> Interaction lens</label
          >
          <label
            >Focus gene
            <input
              id="focusGeneCtx"
              list="geneHints"
              placeholder="Model target / receiver gene"
            />
          </label>
          <label id="neighborKWrap"
            >k neighbors
            <input type="number" id="neighborK" min="1" max="200" value="24" />
          </label>
          <label id="neighborRadiusWrap" class="hidden"
            >Radius (same units as coordinates)
            <input
              type="number"
              id="neighborRadius"
              min="0.0001"
              step="any"
              value="120"
            />
          </label>
          <button type="button" id="refreshContext" class="secondary">
            Refresh context
          </button>
        </div>
        <div class="interaction-body" id="interactionBody"></div>
      </div>
    </details>
        </div>
      </aside>
      <div class="app-stage">
        <div class="stage-strip">
          <div id="colorBarWrap" class="color-bar-wrap hidden">
            <div class="color-bar-title" id="colorBarTitle"></div>
            <div class="color-bar-track">
              <div class="color-bar-gradient" id="colorBarGradient"></div>
            </div>
            <div class="color-bar-labels">
              <span id="colorBarLo"></span>
              <span id="colorBarHi"></span>
            </div>
          </div>
          <div class="stage-jitter-group">
            <label
              class="stage-jitter-toggle"
              title="Very subtle motion on points (visual only; rect select uses true coordinates)"
            >
              <input type="checkbox" id="cellJitterToggle" checked /> Jitter
            </label>
            <input
              type="range"
              id="cellJitterAmp"
              class="stage-jitter-slider"
              min="0"
              max="100"
              value="100"
              title="Jitter amplitude"
            />
          </div>
          <div class="stats" id="stats"></div>
        </div>
        <div class="main" id="main">
          <div id="deck-root"></div>
          <div class="brush-overlay" id="brushOverlay"></div>
        </div>
        <div class="status" id="status"></div>
        <div class="status-progress-wrap hidden" id="statusProgressWrap">
          <div class="status-progress-fill" id="statusProgressFill"></div>
        </div>
      </div>
    </div>
  `;

  const styleHidden = document.createElement("style");
  styleHidden.textContent = `.hidden { display: none !important; }`;
  document.head.appendChild(styleHidden);

  const statusEl = root.querySelector<HTMLDivElement>("#status")!;
  const statusProgressWrap =
    root.querySelector<HTMLDivElement>("#statusProgressWrap")!;
  const statusProgressFill =
    root.querySelector<HTMLDivElement>("#statusProgressFill")!;
  const statsEl = root.querySelector<HTMLDivElement>("#stats")!;
  const cellJitterToggle =
    root.querySelector<HTMLInputElement>("#cellJitterToggle")!;
  const cellJitterAmp =
    root.querySelector<HTMLInputElement>("#cellJitterAmp")!;
  const appSidebar = root.querySelector<HTMLElement>("#appSidebar")!;
  const toggleToolbarBtn =
    root.querySelector<HTMLButtonElement>("#toggleToolbar")!;
  const colorBarWrap = root.querySelector<HTMLDivElement>("#colorBarWrap")!;
  const colorBarGradient =
    root.querySelector<HTMLDivElement>("#colorBarGradient")!;
  const colorBarLo = root.querySelector<HTMLSpanElement>("#colorBarLo")!;
  const colorBarHi = root.querySelector<HTMLSpanElement>("#colorBarHi")!;
  const colorBarTitle =
    root.querySelector<HTMLDivElement>("#colorBarTitle")!;
  const deckContainer = root.querySelector<HTMLDivElement>("#deck-root")!;
  const brushOverlay = root.querySelector<HTMLDivElement>("#brushOverlay")!;
  const colorSource = root.querySelector<HTMLSelectElement>("#colorSource")!;
  const exprGene = root.querySelector<HTMLInputElement>("#exprGene")!;
  const geneHints = root.querySelector<HTMLDataListElement>("#geneHints")!;
  const betaGene = root.querySelector<HTMLSelectElement>("#betaGene")!;
  const betaCol = root.querySelector<HTMLSelectElement>("#betaCol")!;
  const cmapSel = root.querySelector<HTMLSelectElement>("#cmap")!;
  const exprGeneWrap = root.querySelector<HTMLLabelElement>("#exprGeneWrap")!;
  const betaGeneWrap = root.querySelector<HTMLLabelElement>("#betaGeneWrap")!;
  const betaColWrap = root.querySelector<HTMLLabelElement>("#betaColWrap")!;
  const loadBtn = root.querySelector<HTMLButtonElement>("#loadColor")!;
  const brushToggle = root.querySelector<HTMLInputElement>("#brushToggle")!;
  const clearSel = root.querySelector<HTMLButtonElement>("#clearSel")!;
  const cellSizeInput = root.querySelector<HTMLInputElement>("#cellSize")!;
  const cellSizeVal = root.querySelector<HTMLSpanElement>("#cellSizeVal")!;
  const cellTypePanel = root.querySelector<HTMLDetailsElement>("#cellTypePanel")!;
  const cellTypeOverlayEl =
    root.querySelector<HTMLInputElement>("#cellTypeOverlay")!;
  const cellTypeFilters = root.querySelector<HTMLDivElement>("#cellTypeFilters")!;
  const cellTypeColNameEl =
    root.querySelector<HTMLSpanElement>("#cellTypeColName")!;
  const interactionPanel =
    root.querySelector<HTMLDetailsElement>("#interactionPanel")!;
  const interactionLensEl =
    root.querySelector<HTMLInputElement>("#interactionLens")!;
  const focusGeneCtx = root.querySelector<HTMLInputElement>("#focusGeneCtx")!;
  const neighborKInput = root.querySelector<HTMLInputElement>("#neighborK")!;
  const refreshContextBtn =
    root.querySelector<HTMLButtonElement>("#refreshContext")!;
  const interactionBodyEl =
    root.querySelector<HTMLDivElement>("#interactionBody")!;
  const interactionModeSel =
    root.querySelector<HTMLSelectElement>("#interactionModeSel")!;
  const neighborKWrap =
    root.querySelector<HTMLLabelElement>("#neighborKWrap")!;
  const neighborRadiusWrap =
    root.querySelector<HTMLLabelElement>("#neighborRadiusWrap")!;
  const neighborRadiusInput =
    root.querySelector<HTMLInputElement>("#neighborRadius")!;
  const colorSourceBetaOpt = root.querySelector<HTMLOptionElement>(
    "#colorSourceBetaOpt",
  )!;
  const colorSourcePerturbOpt = root.querySelector<HTMLOptionElement>(
    "#colorSourcePerturbOpt",
  )!;
  const perturbPanel = root.querySelector<HTMLDivElement>("#perturbPanel")!;
  const perturbGene = root.querySelector<HTMLInputElement>("#perturbGene")!;
  const perturbExpr = root.querySelector<HTMLInputElement>("#perturbExpr")!;
  const perturbScope = root.querySelector<HTMLSelectElement>("#perturbScope")!;
  const perturbCellTypeWrap =
    root.querySelector<HTMLLabelElement>("#perturbCellTypeWrap")!;
  const perturbCellType =
    root.querySelector<HTMLSelectElement>("#perturbCellType")!;
  const perturbClusterWrap =
    root.querySelector<HTMLLabelElement>("#perturbClusterWrap")!;
  const perturbClusterId =
    root.querySelector<HTMLInputElement>("#perturbClusterId")!;
  const clearPerturbBtn =
    root.querySelector<HTMLButtonElement>("#clearPerturb")!;
  const layoutToggleWrap =
    root.querySelector<HTMLLabelElement>("#layoutToggleWrap")!;
  const layoutModeEl = root.querySelector<HTMLSelectElement>("#layoutMode")!;
  const sessionPanel =
    root.querySelector<HTMLDetailsElement>("#sessionPanel")!;
  const sessionAdataPath =
    root.querySelector<HTMLInputElement>("#sessionAdataPath")!;
  const sessionLayer = root.querySelector<HTMLInputElement>("#sessionLayer")!;
  const sessionClusterAnnot =
    root.querySelector<HTMLInputElement>("#sessionClusterAnnot")!;
  const sessionNetworkDir =
    root.querySelector<HTMLInputElement>("#sessionNetworkDir")!;
  const sessionRunToml = root.querySelector<HTMLInputElement>("#sessionRunToml")!;
  const sessionApplyBtn =
    root.querySelector<HTMLButtonElement>("#sessionApply")!;
  const sessionBusyEl = root.querySelector<HTMLSpanElement>("#sessionBusy")!;
  const mainEl = root.querySelector<HTMLDivElement>("#main")!;
  const umapQuiverPanel =
    root.querySelector<HTMLDetailsElement>("#umapQuiverPanel")!;
  const transNeighbors =
    root.querySelector<HTMLInputElement>("#transNeighbors")!;
  const transT = root.querySelector<HTMLInputElement>("#transT")!;
  const transGridScale =
    root.querySelector<HTMLInputElement>("#transGridScale")!;
  const transVecScale =
    root.querySelector<HTMLInputElement>("#transVecScale")!;
  const transDeltaRescale =
    root.querySelector<HTMLInputElement>("#transDeltaRescale")!;
  const transMagThresh =
    root.querySelector<HTMLInputElement>("#transMagThresh")!;
  const transRemoveNull =
    root.querySelector<HTMLInputElement>("#transRemoveNull")!;
  const transUnitDirs =
    root.querySelector<HTMLInputElement>("#transUnitDirs")!;
  const transQuickKo =
    root.querySelector<HTMLInputElement>("#transQuickKo")!;
  const transFullGraph =
    root.querySelector<HTMLInputElement>("#transFullGraph")!;
  const transFullMax = root.querySelector<HTMLInputElement>("#transFullMax")!;
  const transLimitWrap = root.querySelector<HTMLDivElement>("#transLimitWrap")!;
  const transLimitClusters =
    root.querySelector<HTMLInputElement>("#transLimitClusters")!;
  const transHighlightTypes =
    root.querySelector<HTMLSelectElement>("#transHighlightTypes")!;
  const computeQuiverBtn =
    root.querySelector<HTMLButtonElement>("#computeQuiverBtn")!;
  const clearQuiverBtn =
    root.querySelector<HTMLButtonElement>("#clearQuiverBtn")!;
  const transUmapOnlyHint =
    root.querySelector<HTMLParagraphElement>("#transUmapOnlyHint")!;
  const perturbSummaryBtn =
    root.querySelector<HTMLButtonElement>("#perturbSummaryBtn")!;
  const perturbSummaryBody =
    root.querySelector<HTMLDivElement>("#perturbSummaryBody")!;
  const quiverVisScale =
    root.querySelector<HTMLInputElement>("#quiverVisScale")!;
  const quiverVisScaleVal =
    root.querySelector<HTMLSpanElement>("#quiverVisScaleVal")!;
  const quiverLineW = root.querySelector<HTMLInputElement>("#quiverLineW")!;
  const quiverLineWVal =
    root.querySelector<HTMLSpanElement>("#quiverLineWVal")!;
  const quiverHeadFrac =
    root.querySelector<HTMLInputElement>("#quiverHeadFrac")!;
  const quiverHeadVal =
    root.querySelector<HTMLSpanElement>("#quiverHeadVal")!;
  const quiverStride = root.querySelector<HTMLInputElement>("#quiverStride")!;
  const quiverStrideVal =
    root.querySelector<HTMLSpanElement>("#quiverStrideVal")!;

  function escapeHtml(s: string): string {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function lrSupportColor(
    support: number,
    maxS: number,
  ): [number, number, number, number] {
    if (!(maxS > 0) || !Number.isFinite(support)) {
      return [100, 200, 255, 185];
    }
    const t = Math.min(1, Math.max(0, support / maxS));
    const r = Math.round(55 + t * 200);
    const g = Math.round(140 + t * 115);
    const b = Math.round(255 - t * 70);
    const a = Math.round(140 + t * 115);
    return [r, g, b, a];
  }

  function refillTransHighlightTypes() {
    transHighlightTypes.innerHTML = "";
    for (const c of cellCategories) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      transHighlightTypes.appendChild(opt);
    }
  }

  function syncInteractionPanelLayout() {
    const radius = interactionModeSel.value === "radius";
    neighborKWrap.classList.toggle("hidden", radius);
    neighborRadiusWrap.classList.toggle("hidden", !radius);
  }

  const setStatus = (msg: string, err = false) => {
    statusEl.textContent = msg;
    statusEl.classList.toggle("error", err);
  };

  function syncProgressBar(percent: number | null | undefined) {
    if (percent == null || !Number.isFinite(percent)) {
      statusProgressWrap.classList.add("hidden");
      statusProgressFill.style.width = "0%";
      return;
    }
    const p = Math.min(100, Math.max(0, percent));
    statusProgressWrap.classList.remove("hidden");
    statusProgressFill.style.width = `${p}%`;
  }

  function applyMetaProgressToUi(m: Meta) {
    const pct = m.perturb_progress_percent;
    const lbl = (m.perturb_progress_label ?? "").trim();
    if (pct != null && Number.isFinite(pct)) {
      syncProgressBar(pct);
      setStatus(lbl ? `${lbl} · ${pct}%` : `Working… ${pct}%`);
    }
  }

  async function withMetaProgressPoll<T>(work: Promise<T>): Promise<T> {
    const id = window.setInterval(() => {
      void (async () => {
        try {
          const mr = await fetch("/api/meta");
          if (!mr.ok) return;
          const m = (await mr.json()) as Meta;
          applyMetaProgressToUi(m);
        } catch {
          /* ignore */
        }
      })();
    }, 200);
    try {
      return await work;
    } finally {
      clearInterval(id);
      syncProgressBar(null);
    }
  }

  let meta!: Meta;
  let n = 0;
  let cellCategories: string[] = [];
  let cellTypeColumnLabel: string | null = null;
  let cellTypeCodes: Uint16Array | null = null;
  let typeFilterChecked: boolean[] = [];
  let positionsSpatial!: Float32Array;
  let positionsUmap: Float32Array | null = null;
  let positions!: Float32Array;
  let jitterPositions: Float32Array | null = null;
  let cellJitterRaf = 0;
  let perturbMetaPollTimer: ReturnType<typeof setTimeout> | null = null;
  let clusterIds: Uint32Array | null = null;
  let interactionSenderIndex: number | null = null;
  const interactionNeighborSet = new Set<number>();
  let interactionLineData: InteractionLineDatum[] = [];
  let quiverFieldCache: UmapFieldResponse | null = null;
  let quiverSegData: QuiverSegDatum[] = [];
  let baseColors!: Uint8Array;
  let colors!: Uint8Array;
  let selected!: Uint8Array;
  let activeValues: Float32Array | null = null;
  let rangeLo = 0;
  let rangeHi = 1;
  let scaleLine = "Scale: —";
  let lastColorSource: "expression" | "betadata" | "perturb" | null = null;
  let perturbDisplayGene = "";
  let deck: Deck | undefined;

  toggleToolbarBtn.addEventListener("click", () => {
    const collapsed = appSidebar.classList.toggle("sidebar-collapsed");
    toggleToolbarBtn.textContent = collapsed ? "Show controls" : "Hide controls";
    toggleToolbarBtn.setAttribute(
      "aria-expanded",
      collapsed ? "false" : "true",
    );
  });

  function compositeSelectionIntoDisplayColors() {
    colors.set(baseColors);
    for (let i = 0; i < n; i++) {
      if (!selected[i]) continue;
      const o = i * 4;
      colors[o] = Math.min(
        255,
        Math.round(baseColors[o]! * 0.45 + 255 * 0.55),
      );
      colors[o + 1] = Math.min(
        255,
        Math.round(baseColors[o + 1]! * 0.45 + 210 * 0.55),
      );
      colors[o + 2] = Math.min(
        255,
        Math.round(baseColors[o + 2]! * 0.45 + 90 * 0.55),
      );
      colors[o + 3] = 255;
    }
  }

  function collectSelectedIndices(): number[] {
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      if (selected[i]) out.push(i);
    }
    return out;
  }

  function formatColorTick(x: number): string {
    if (!Number.isFinite(x)) return "—";
    const ax = Math.abs(x);
    if (ax >= 1e4 || (ax > 0 && ax < 1e-2)) return x.toExponential(2);
    return x.toPrecision(4);
  }

  function dataMinMax(values: Float32Array): { lo: number; hi: number } {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i]!;
      if (!Number.isFinite(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };
    if (hi <= lo) hi = lo + 1e-9;
    return { lo, hi };
  }

  function updateColorBar() {
    const overlayOn =
      cellTypeOverlayEl.checked &&
      !!cellTypeCodes &&
      cellCategories.length > 0;
    if (overlayOn || !activeValues) {
      colorBarWrap.classList.add("hidden");
      return;
    }
    colorBarWrap.classList.remove("hidden");
    const cmap = cmapSel.value as ColormapId;
    colorBarGradient.style.backgroundImage = colormapLegendGradientCss(cmap);
    let lo = rangeLo;
    let hi = rangeHi;
    if (lastColorSource === "betadata" && clusterIds) {
      const mm = dataMinMax(activeValues);
      lo = mm.lo;
      hi = mm.hi;
    }
    colorBarLo.textContent = formatColorTick(lo);
    colorBarHi.textContent = formatColorTick(hi);
    colorBarTitle.textContent = scaleLine;
  }

  const updateStats = () => {
    if (!selected || n === 0) return;
    let selCount = 0;
    for (let i = 0; i < n; i++) {
      if (selected[i]) selCount++;
    }
    const anySel = selCount > 0;
    const mSel = meanForMask(activeValues, selected);
    const mAll = globalMean(activeValues);
    const selLine = `Selected: <strong>${selCount}</strong> cell(s)`;
    statsEl.innerHTML = anySel
      ? `<div>${selLine}</div><div>Selection mean: <strong>${mSel}</strong></div><div>Global mean: ${mAll}</div>`
      : `<div>${selLine}</div><div>Global mean: <strong>${mAll}</strong></div>`;
    updateColorBar();
  };

  const syncPerturbScopeFields = () => {
    const s = perturbScope.value;
    perturbCellTypeWrap.classList.toggle("hidden", s !== "cell_type");
    perturbClusterWrap.classList.toggle("hidden", s !== "cluster");
  };

  const syncColorModeUi = () => {
    const b = colorSource.value === "betadata";
    const p = colorSource.value === "perturb";
    exprGeneWrap.classList.toggle("hidden", b || p);
    betaGeneWrap.classList.toggle("hidden", !b);
    betaColWrap.classList.toggle("hidden", !b);
    perturbPanel.classList.toggle("hidden", !p);
    loadBtn.textContent = p ? "Run perturbation" : "Load / refresh";
    if (p) syncPerturbScopeFields();
  };

  colorSource.addEventListener("change", () => {
    if (colorSource.value !== "perturb" && lastColorSource === "perturb") {
      activeValues = null;
      lastColorSource = null;
      perturbDisplayGene = "";
    }
    syncColorModeUi();
    refreshVisualization();
  });
  perturbScope.addEventListener("change", syncPerturbScopeFields);

  function cellSelectableByType(i: number): boolean {
    if (!cellTypeCodes || typeFilterChecked.length === 0) return true;
    const c = cellTypeCodes[i]!;
    if (c === CT_UNKNOWN) return true;
    if (c >= typeFilterChecked.length) return true;
    return typeFilterChecked[c] === true;
  }

  function pruneSelectionToAllowedTypes() {
    for (let i = 0; i < n; i++) {
      if (selected[i] && !cellSelectableByType(i)) selected[i] = 0;
    }
  }

  function fillBaseFromCellTypes() {
    if (!cellTypeCodes) return;
    const nc = cellCategories.length;
    for (let i = 0; i < n; i++) {
      const c = cellTypeCodes[i]!;
      const o = i * 4;
      if (c === CT_UNKNOWN) {
        baseColors[o] = 88;
        baseColors[o + 1] = 88;
        baseColors[o + 2] = 95;
        baseColors[o + 3] = 255;
      } else {
        const [r, g, b] = rgbForCategoryIndex(c, nc);
        baseColors[o] = r;
        baseColors[o + 1] = g;
        baseColors[o + 2] = b;
        baseColors[o + 3] = 255;
      }
    }
  }

  function applyDisabledTypeDimming() {
    if (!cellTypeCodes || typeFilterChecked.length === 0) return;
    if (typeFilterChecked.every((x) => x)) return;
    for (let i = 0; i < n; i++) {
      const c = cellTypeCodes[i]!;
      if (c === CT_UNKNOWN) continue;
      if (c < typeFilterChecked.length && typeFilterChecked[c]) continue;
      const o = i * 4;
      baseColors[o] = Math.min(255, Math.round(baseColors[o]! * 0.22));
      baseColors[o + 1] = Math.min(
        255,
        Math.round(baseColors[o + 1]! * 0.22),
      );
      baseColors[o + 2] = Math.min(
        255,
        Math.round(baseColors[o + 2]! * 0.22),
      );
    }
  }

  function applyInteractionContextDimming() {
    if (interactionSenderIndex === null || !interactionLensEl.checked) return;
    for (let i = 0; i < n; i++) {
      if (i === interactionSenderIndex || interactionNeighborSet.has(i)) continue;
      const o = i * 4;
      baseColors[o] = Math.round(baseColors[o]! * 0.36);
      baseColors[o + 1] = Math.round(baseColors[o + 1]! * 0.36);
      baseColors[o + 2] = Math.round(baseColors[o + 2]! * 0.36);
    }
  }

  function rebuildQuiverFromCache(): number {
    quiverSegData.length = 0;
    const data = quiverFieldCache;
    if (!data) return 0;
    const nx = data.nx;
    const ny = data.ny;
    const visMul = Math.max(0.05, (Number(quiverVisScale.value) || 100) / 100);
    const shaftW = Math.min(12, Math.max(0.5, Number(quiverLineW.value) || 2));
    const headW = Math.min(14, shaftW * 1.2);
    const headFrac = Math.max(
      0.08,
      Math.min(0.55, (Number(quiverHeadFrac.value) || 28) / 100),
    );
    const stride = Math.max(1, Math.min(12, Math.trunc(Number(quiverStride.value) || 1)));
    const qc: [number, number, number, number] = [235, 98, 52, 235];
    let arrowCount = 0;
    for (let ix = 0; ix < nx; ix += stride) {
      const gx = data.grid_x[ix]!;
      for (let iy = 0; iy < ny; iy += stride) {
        const gy = data.grid_y[iy]!;
        const k = ix * ny + iy;
        const u = data.u[k]! * visMul;
        const v = data.v[k]! * visMul;
        const len = Math.hypot(u, v);
        if (len < 1e-12) continue;
        arrowCount++;
        const dx = u / len;
        const dy = v / len;
        const hl = Math.min(len * headFrac, len * 0.98);
        const Tx = gx + u;
        const Ty = gy + v;
        const Bx = Tx - hl * dx;
        const By = Ty - hl * dy;
        const px = -dy;
        const py = dx;
        const hw = hl * 0.48;
        const Lx = Bx + hw * px;
        const Ly = By + hw * py;
        const Rx = Bx - hw * px;
        const Ry = By - hw * py;
        quiverSegData.push(
          {
            sourcePosition: [gx, gy, 0],
            targetPosition: [Bx, By, 0],
            color: qc,
            width: shaftW,
          },
          {
            sourcePosition: [Lx, Ly, 0],
            targetPosition: [Tx, Ty, 0],
            color: qc,
            width: headW,
          },
          {
            sourcePosition: [Rx, Ry, 0],
            targetPosition: [Tx, Ty, 0],
            color: qc,
            width: headW,
          },
        );
      }
    }
    return arrowCount;
  }

  function syncQuiverDisplayLabels() {
    quiverVisScaleVal.textContent = String(quiverVisScale.value);
    quiverLineWVal.textContent = String(quiverLineW.value);
    quiverHeadVal.textContent = String(quiverHeadFrac.value);
    quiverStrideVal.textContent = String(quiverStride.value);
  }

  function stopCellJitterLoop() {
    if (cellJitterRaf !== 0) {
      cancelAnimationFrame(cellJitterRaf);
      cellJitterRaf = 0;
    }
  }

  function fillCellJitterBuffer(tSec: number) {
    if (!jitterPositions || n === 0) return;
    const b =
      layoutModeEl.value === "umap" && meta.umap_bounds
        ? meta.umap_bounds
        : meta.bounds;
    const span = Math.max(
      b.max_x - b.min_x,
      b.max_y - b.min_y,
      1e-9,
    );
    const sliderT = Math.max(0, Math.min(100, Number(cellJitterAmp.value) || 0)) / 100;
    const amp = span * 0.003 * sliderT;
    for (let i = 0; i < n; i++) {
      const g = i * 0.813492075;
      const jx = Math.sin(tSec * 1.12 + g);
      const jy = Math.cos(tSec * 0.97 + g * 1.71);
      jitterPositions[i * 2] = positions[i * 2]! + amp * jx;
      jitterPositions[i * 2 + 1] = positions[i * 2 + 1]! + amp * jy;
    }
  }

  function tickCellJitter() {
    if (!cellJitterToggle.checked || !deck || n === 0) {
      cellJitterRaf = 0;
      return;
    }
    if (!document.hidden) {
      fillCellJitterBuffer(performance.now() * 0.001);
      rebuildLayer();
    }
    cellJitterRaf = requestAnimationFrame(tickCellJitter);
  }

  function startCellJitterLoop() {
    stopCellJitterLoop();
    if (!cellJitterToggle.checked || !deck || n === 0 || !jitterPositions)
      return;
    cellJitterRaf = requestAnimationFrame(tickCellJitter);
  }

  const rebuildLayer = () => {
    if (!deck) return;
    const raw = Number(cellSizeInput.value);
    const px = Number.isFinite(raw)
      ? Math.min(48, Math.max(0.5, raw))
      : 4;
    const posBuf =
      cellJitterToggle.checked && jitterPositions ? jitterPositions : positions;
    const scatterLayer = new ScatterplotLayer({
      id: "cells",
      data: {
        length: n,
        attributes: {
          getPosition: { value: posBuf, size: 2 },
          getFillColor: { value: colors, size: 4, normalized: true },
        },
      },
      pickable: true,
      radiusUnits: "pixels",
      radiusScale: 1,
      radiusMinPixels: px,
      radiusMaxPixels: px,
      stroked: false,
      billboard: true,
    });
    const layers: (LineLayer | ScatterplotLayer)[] = [];
    if (interactionLineData.length > 0) {
      layers.push(
        new LineLayer({
          id: "lr-context-lines",
          data: interactionLineData,
          getSourcePosition: (d: InteractionLineDatum) => d.sourcePosition,
          getTargetPosition: (d: InteractionLineDatum) => d.targetPosition,
          getColor: (d: InteractionLineDatum) =>
            d.color ?? [95, 210, 255, 210],
          getWidth: 2.5,
          widthUnits: "pixels",
          pickable: false,
          parameters: { depthTest: false },
        }),
      );
    }
    if (layoutModeEl.value === "umap" && quiverSegData.length > 0) {
      layers.push(
        new LineLayer({
          id: "umap-quiver",
          data: quiverSegData,
          getSourcePosition: (d: QuiverSegDatum) => d.sourcePosition,
          getTargetPosition: (d: QuiverSegDatum) => d.targetPosition,
          getColor: (d: QuiverSegDatum) => d.color ?? [235, 98, 52, 220],
          getWidth: (d: QuiverSegDatum) => d.width ?? 2,
          widthUnits: "pixels",
          pickable: false,
          parameters: { depthTest: false },
        }),
      );
    }
    layers.push(scatterLayer);
    const hasQ = quiverFieldCache !== null;
    const onUmap = layoutModeEl.value === "umap";
    transUmapOnlyHint.classList.toggle("hidden", !hasQ || onUmap);

    deck.setProps({
      layers,
      getTooltip: (info) => {
        if (info.index == null || info.index < 0) return null;
        let extra = "";
        if (cellTypeCodes && cellCategories.length > 0) {
          const code = cellTypeCodes[info.index]!;
          const label =
            code === CT_UNKNOWN
              ? "(unknown)"
              : (cellCategories[code] ?? "?");
          extra = `\n${cellTypeColumnLabel ?? "cell_type"}: ${label}`;
        }
        let tip = `Cell #${info.index}${extra}`;
        if (
          lastColorSource === "perturb" &&
          activeValues &&
          info.index < activeValues.length
        ) {
          tip += `\nΔ ${perturbDisplayGene || "?"}: ${activeValues[info.index]!.toPrecision(4)}`;
        }
        return { text: tip };
      },
    });
  };

  const refreshVisualization = () => {
    if (!selected || n === 0) return;
    const cmap = cmapSel.value as ColormapId;
    const overlayOn =
      cellTypeOverlayEl.checked &&
      !!cellTypeCodes &&
      cellCategories.length > 0;
    if (overlayOn) {
      fillBaseFromCellTypes();
      scaleLine = `Color: cell type (${cellTypeColumnLabel ?? "cell_type"})`;
    } else if (activeValues) {
      if (lastColorSource === "perturb") {
        const rr = applyColors(activeValues, baseColors, n, cmap);
        rangeLo = rr.lo;
        rangeHi = rr.hi;
        scaleLine = `Perturbation Δ ${perturbDisplayGene || "?"} [${rangeLo.toPrecision(4)}, ${rangeHi.toPrecision(4)}]`;
      } else if (lastColorSource === "betadata" && clusterIds) {
        applyBetadataColorsPerCluster(
          activeValues,
          clusterIds,
          baseColors,
          n,
          cmap,
        );
        scaleLine = "Scale: per-cluster (all-zero clusters → gray)";
      } else {
        const rr = applyColors(activeValues, baseColors, n, cmap);
        rangeLo = rr.lo;
        rangeHi = rr.hi;
        scaleLine =
          lastColorSource === "betadata" && !clusterIds
            ? `Scale: global [${rangeLo.toPrecision(4)}, ${rangeHi.toPrecision(4)}] (no cluster ids)`
            : `Scale: global [${rangeLo.toPrecision(4)}, ${rangeHi.toPrecision(4)}]`;
      }
    } else {
      const rr = applyColors(null, baseColors, n, cmap);
      rangeLo = rr.lo;
      rangeHi = rr.hi;
      scaleLine = "Scale: —";
    }
    applyDisabledTypeDimming();
    applyInteractionContextDimming();
    pruneSelectionToAllowedTypes();
    compositeSelectionIntoDisplayColors();
    rebuildLayer();
    updateStats();
  };

  function clearInteractionContextVisuals() {
    interactionSenderIndex = null;
    interactionNeighborSet.clear();
    interactionLineData.length = 0;
  }

  function clearInteractionContextFull() {
    clearInteractionContextVisuals();
    interactionBodyEl.innerHTML = "";
    refreshVisualization();
  }

  function singleSelectedIndex(): number | null {
    let found: number | null = null;
    for (let i = 0; i < n; i++) {
      if (!selected[i]) continue;
      if (found !== null) return null;
      found = i;
    }
    return found;
  }

  function renderInteractionPanel(data: CellContextResponse) {
    const regRows = data.sender_regulator_exprs
      .slice(0, 80)
      .map(
        (r) =>
          `<tr><td>${escapeHtml(r.gene)}</td><td class="num">${r.expr.toPrecision(3)}</td></tr>`,
      )
      .join("");
    const ligRows = data.sender_ligand_exprs
      .slice(0, 80)
      .map(
        (r) =>
          `<tr><td>${escapeHtml(r.gene)}</td><td class="num">${r.expr.toPrecision(3)}</td></tr>`,
      )
      .join("");
    let maxEdge = 0;
    for (const nb of data.neighbors) {
      for (const e of nb.lr_edges) {
        const sc =
          e.support_score ??
          Math.sqrt(
            Math.max(0, e.lig_expr_sender * e.rec_expr_neighbor),
          );
        maxEdge = Math.max(maxEdge, sc);
      }
    }
    const neighBlocks = data.neighbors
      .map((nb) => {
        if (nb.lr_edges.length === 0) {
          return `<div class="interaction-neigh"><strong>Cell ${nb.index}</strong> — no LR above threshold</div>`;
        }
        const er = nb.lr_edges
          .map((e) => {
            const chain =
              e.linked_tf != null
                ? ` <span class="chain-hint">(${escapeHtml(e.linked_tf)} → ligand, nichenet)</span>`
                : "";
            const sc =
              e.support_score ??
              Math.sqrt(
                Math.max(0, e.lig_expr_sender * e.rec_expr_neighbor),
              );
            const barW =
              maxEdge > 0 ? Math.min(100, (sc / maxEdge) * 100) : 0;
            return `<li class="lr-edge-li"><span class="lr-edge-bar" style="width:${barW}%"></span><span class="lr-edge-text"><span class="lr-pair">${escapeHtml(e.ligand)} → ${escapeHtml(e.receptor)}</span> · L=${e.lig_expr_sender.toPrecision(3)} R=${e.rec_expr_neighbor.toPrecision(3)} · √LR=${sc.toPrecision(3)}${chain}</span></li>`;
          })
          .join("");
        return `<div class="interaction-neigh"><strong>Neighbor ${nb.index}</strong> (dist² ${nb.distance_sq.toPrecision(4)})<ul class="lr-edge-list">${er}</ul></div>`;
      })
      .join("");
    const q = data.neighbor_query;
    const rq = data.radius_used;
    const nq = data.neighbors_in_query;
    let qstr = "";
    if (q === "radius" && rq != null) {
      qstr = ` · radius <strong>${rq}</strong> · <strong>${nq ?? data.neighbors.length}</strong> neighbors`;
    } else if (q === "knn") {
      qstr = ` · kNN · <strong>${nq ?? data.neighbors.length}</strong> neighbors`;
    }
    interactionBodyEl.innerHTML = `
      <p class="interaction-meta">Sender <strong>#${data.cell_index}</strong> · focus <strong>${escapeHtml(data.focus_gene)}</strong>${qstr}</p>
      <div class="interaction-cols">
        <div><h4>TFs regulating focus (sender)</h4>
        <table class="interaction-table"><thead><tr><th>TF</th><th class="num">expr</th></tr></thead><tbody>${regRows || "<tr><td colspan='2'>—</td></tr>"}</tbody></table></div>
        <div><h4>Ligands (sender, ranked set)</h4>
        <table class="interaction-table"><thead><tr><th>Ligand</th><th class="num">expr</th></tr></thead><tbody>${ligRows || "<tr><td colspan='2'>—</td></tr>"}</tbody></table></div>
      </div>
      <h4>Neighbors & supported L→R pairs</h4>
      ${neighBlocks || "<p>—</p>"}`;
  }

  async function fetchAndApplyInteractionContext(cellIdx: number) {
    const fg = focusGeneCtx.value.trim();
    if (!fg) {
      setStatus("Enter focus gene for interaction context", true);
      interactionBodyEl.innerHTML =
        '<p class="interaction-empty">Enter a focus gene, then select one cell or click Refresh.</p>';
      return;
    }
    const nk = Math.min(
      200,
      Math.max(1, Number(neighborKInput.value) || 24),
    );
    const mode = interactionModeSel.value;
    const payload: Record<string, unknown> = {
      cell_index: cellIdx,
      focus_gene: fg,
      neighbor_k: nk,
      neighbor_mode: mode === "radius" ? "radius" : "knn",
    };
    if (mode === "radius") {
      payload.radius = Math.max(
        1e-9,
        Number(neighborRadiusInput.value) || 120,
      );
    }
    interactionBodyEl.innerHTML =
      '<p class="interaction-loading">Computing context…</p>';
    try {
      const r = await fetch("/api/network/cell-context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as CellContextResponse;
      interactionSenderIndex = data.cell_index;
      interactionNeighborSet.clear();
      for (const nb of data.neighbors) {
        interactionNeighborSet.add(nb.index);
      }
      let maxN = 0;
      for (const nb of data.neighbors) {
        const s = nb.max_support_score;
        if (s != null && Number.isFinite(s)) maxN = Math.max(maxN, s);
      }
      if (maxN <= 0) {
        for (const nb of data.neighbors) {
          for (const e of nb.lr_edges) {
            const sc =
              e.support_score ??
              Math.sqrt(
                Math.max(0, e.lig_expr_sender * e.rec_expr_neighbor),
              );
            maxN = Math.max(maxN, sc);
          }
        }
      }
      interactionLineData.length = 0;
      for (const nb of data.neighbors) {
        if (nb.lr_edges.length === 0) continue;
        const i = nb.index;
        const ms = nb.max_support_score;
        const strength =
          ms != null && ms > 0
            ? ms
            : Math.max(
                ...nb.lr_edges.map((e) =>
                  e.support_score ??
                  Math.sqrt(
                    Math.max(0, e.lig_expr_sender * e.rec_expr_neighbor),
                  ),
                ),
              );
        interactionLineData.push({
          sourcePosition: [
            positions[data.cell_index * 2]!,
            positions[data.cell_index * 2 + 1]!,
            0,
          ],
          targetPosition: [positions[i * 2]!, positions[i * 2 + 1]!, 0],
          color: lrSupportColor(strength, maxN),
        });
      }
      renderInteractionPanel(data);
      refreshVisualization();
      setStatus(
        `Context: sender #${cellIdx} · ${data.focus_gene} · ${interactionLineData.length} neighbor LR link(s)`,
      );
    } catch (e) {
      clearInteractionContextVisuals();
      refreshVisualization();
      interactionBodyEl.innerHTML = `<p class="interaction-error">${escapeHtml(String(e))}</p>`;
      setStatus(String(e), true);
    }
  }

  function syncPerturbPanelsFromMeta() {
    colorSourcePerturbOpt.classList.toggle("hidden", !meta.perturb_ready);
    umapQuiverPanel.classList.toggle(
      "hidden",
      !meta.perturb_ready || !meta.umap_obsm_key,
    );
  }

  function stopPerturbMetaPoll() {
    if (perturbMetaPollTimer !== null) {
      clearTimeout(perturbMetaPollTimer);
      perturbMetaPollTimer = null;
    }
  }

  function schedulePerturbMetaPoll() {
    stopPerturbMetaPoll();
    const tick = () => {
      perturbMetaPollTimer = null;
      void (async () => {
        try {
          const mr = await fetch("/api/meta");
          if (!mr.ok) return;
          const m = (await mr.json()) as Meta;
          meta = m;
          syncPerturbPanelsFromMeta();
          if (meta.perturb_error) {
            syncProgressBar(null);
            const base = (statusEl.textContent ?? "").replace(
              /\s*·?\s*perturbation.*$/i,
              "",
            );
            setStatus(`${base} · perturbation failed: ${meta.perturb_error}`, true);
            return;
          }
          if (meta.perturb_loading) {
            applyMetaProgressToUi(meta);
            perturbMetaPollTimer = setTimeout(tick, 500);
            return;
          }
          syncProgressBar(null);
          if (meta.perturb_ready) {
            const base = (statusEl.textContent ?? "").replace(
              /\s*·?\s*perturbation.*$/i,
              "",
            );
            setStatus(`${base} · perturbation ready`);
          }
        } catch {
          /* ignore */
        }
      })();
    };
    perturbMetaPollTimer = setTimeout(tick, 400);
  }

  function syncInteractionFromSelection() {
    if (!meta.network_loaded || !interactionLensEl.checked) return;
    const one = singleSelectedIndex();
    if (one !== null) {
      void fetchAndApplyInteractionContext(one);
    } else {
      clearInteractionContextFull();
    }
  }

  async function initDataset(metaOverride?: Meta): Promise<boolean> {
    stopPerturbMetaPoll();
    syncProgressBar(null);
    stopCellJitterLoop();
    jitterPositions = null;
    cellTypeOverlayEl.checked = false;
    if (deck) {
      deck.finalize();
      deck = undefined;
    }
    layoutModeEl.value = "spatial";
    layoutToggleWrap.classList.add("hidden");
    positionsUmap = null;
    activeValues = null;
    lastColorSource = null;
    perturbDisplayGene = "";
    colorSource.value = "expression";
    exprGene.value = "";
    betaCol.innerHTML = '<option value="">—</option>';
    interactionSenderIndex = null;
    interactionNeighborSet.clear();
    interactionLineData.length = 0;
    quiverFieldCache = null;
    quiverSegData.length = 0;
    transUmapOnlyHint.classList.add("hidden");
    interactionBodyEl.innerHTML = "";
    interactionLensEl.checked = false;
    setStatus("Loading dataset…");
    try {
      if (metaOverride) meta = metaOverride;
      else {
        const mr = await fetch("/api/meta");
        if (!mr.ok) throw new Error(await mr.text());
        meta = (await mr.json()) as Meta;
      }
    } catch (e) {
      setStatus(`Failed to load metadata: ${e}`, true);
      return false;
    }

    n = meta.n_obs;
    sessionAdataPath.value = meta.adata_path;
    sessionLayer.value = meta.layer;
    sessionClusterAnnot.value = meta.cluster_annot;
    sessionNetworkDir.value = meta.network_dir ?? "";
    sessionRunToml.value = meta.run_toml ?? "";

    cellCategories = meta.cell_type_categories ?? [];
    cellTypeColumnLabel = meta.cell_type_column ?? null;
    cellTypeCodes = null;
    typeFilterChecked = cellCategories.map(() => true);

    cellTypePanel.classList.add("hidden");
    cellTypeFilters.innerHTML = "";
    perturbCellType.innerHTML = '<option value="">—</option>';

    if (meta.cell_type_column) {
      try {
        const cr = await fetch("/api/cell_type/codes");
        if (cr.ok) {
          const buf = await cr.arrayBuffer();
          const arr = new Uint16Array(buf);
          if (arr.length === n) cellTypeCodes = arr;
        }
      } catch {
        /* optional */
      }
    }

    if (cellTypeColumnLabel && cellCategories.length > 0) {
      cellTypePanel.classList.remove("hidden");
      cellTypeColNameEl.textContent = cellTypeColumnLabel;
      cellTypeFilters.innerHTML = cellCategories
        .map(
          (name, idx) =>
            `<label class="cell-type-item"><input type="checkbox" data-ct-idx="${idx}" checked /> ${escapeHtml(name)}</label>`,
        )
        .join("");
      perturbCellType.innerHTML =
        '<option value="">— pick —</option>' +
        cellCategories
          .map(
            (name, idx) =>
              `<option value="${idx}">${escapeHtml(name)}</option>`,
          )
          .join("");
    }
    refillTransHighlightTypes();
    transLimitWrap.classList.toggle(
      "hidden",
      !meta.cell_type_column || cellCategories.length === 0,
    );

    cellTypeOverlayEl.checked = !!(
      cellTypeCodes &&
      cellCategories.length > 0
    );

    const ready = meta.dataset_ready !== false && meta.n_obs > 0;
    if (!ready) {
      layoutToggleWrap.classList.add("hidden");
      umapQuiverPanel.classList.add("hidden");
      interactionPanel.classList.add("hidden");
      colorSourceBetaOpt.classList.add("hidden");
      colorSourcePerturbOpt.classList.add("hidden");
      setStatus(
        "No dataset loaded — set .h5ad (and optional run TOML) under Dataset paths, then Load dataset.",
      );
      sessionPanel.open = true;
      return true;
    }

    {
      const parts = [
        `n=${n} cells`,
        `${meta.spatial_obsm_key}`,
        `layer ${meta.layer}`,
        `clusters: ${meta.cluster_annot}`,
      ];
      if (meta.network_loaded && meta.network_species) {
        parts.splice(1, 0, `GRN ${meta.network_species}`);
      }
      if (meta.betadata_row_id === "Cluster") {
        parts.push("β seed-only (Cluster)");
      } else       if (meta.betadata_row_id === "CellID") {
        parts.push("β spatial (CellID)");
      }
      if (meta.perturb_loading) {
        parts.push("perturbation loading (may take minutes)…");
      }
      setStatus(parts.join(" · "));
    }
    interactionPanel.classList.toggle("hidden", !meta.network_loaded);
    interactionModeSel.value = "knn";
    syncInteractionPanelLayout();
    const hasBetadata = !!meta.betadata_row_id;
    colorSourceBetaOpt.classList.toggle("hidden", !hasBetadata);
    colorSourcePerturbOpt.classList.toggle("hidden", !meta.perturb_ready);
    umapQuiverPanel.classList.toggle(
      "hidden",
      !meta.perturb_ready || !meta.umap_obsm_key,
    );
    perturbSummaryBody.classList.add("hidden");

    try {
      const [spR, clR] = await Promise.allSettled([
        fetchF32("/api/spatial"),
        fetchU32("/api/clusters"),
      ]);
      if (spR.status !== "fulfilled") {
        setStatus(`Spatial load failed: ${String(spR.reason)}`, true);
        return false;
      }
      positionsSpatial = spR.value;
      positions = positionsSpatial;
      clusterIds = null;
      if (clR.status === "fulfilled" && clR.value.length === n) {
        clusterIds = clR.value;
      } else if (clR.status === "fulfilled") {
        setStatus(
          `Cluster id length ${clR.value.length} != n_obs ${n}; betadata uses global scale`,
          true,
        );
      }
    } catch (e) {
      setStatus(`Load failed: ${e}`, true);
      return false;
    }

    if (positionsSpatial.length !== n * 2) {
      setStatus(
        `Expected ${n * 2} floats in spatial, got ${positionsSpatial.length}`,
        true,
      );
      return false;
    }

    if (meta.umap_obsm_key && meta.umap_bounds) {
      try {
        const u = await fetchF32("/api/umap");
        if (u.length === n * 2) {
          positionsUmap = u;
          layoutToggleWrap.classList.remove("hidden");
          const opt = layoutModeEl.options[1];
          if (opt) opt.textContent = meta.umap_obsm_key;
        } else {
          console.warn(
            `UMAP length ${u.length} != ${n * 2}; layout toggle disabled`,
          );
        }
      } catch {
        console.warn("UMAP coordinates unavailable");
      }
    }

    baseColors = new Uint8Array(n * 4);
    colors = new Uint8Array(n * 4);
    selected = new Uint8Array(n);
    jitterPositions = new Float32Array(n * 2);

    const w0 = Math.max(mainEl.clientWidth, 32);
    const h0 = Math.max(mainEl.clientHeight, 32);
    const vs0 = fitOrthographic(w0, h0, meta.bounds);

    deck = new Deck({
      parent: deckContainer,
      width: w0,
      height: h0,
      views: new OrthographicView({ id: "ortho", flipY: false }),
      initialViewState: {
        target: [vs0.target[0], vs0.target[1], 0],
        zoom: vs0.zoom,
      },
      viewState: null,
      controller: ORTHO_CONTROLLER,
      getCursor: ({ isDragging, isHovering }) =>
        isDragging ? "grabbing" : isHovering ? "pointer" : "grab",
      layers: [],
      onClick: (info, evt) => {
        if (brushToggle.checked) return;
        const idx = info.index;
        if (idx == null || idx < 0 || idx >= n) return;
        if (!cellSelectableByType(idx)) return;
        const dom = (evt as { srcEvent?: PointerEvent }).srcEvent;
        const shift = !!(dom && "shiftKey" in dom && dom.shiftKey);
        if (shift) {
          selected[idx] = selected[idx] ? 0 : 1;
        } else {
          selected.fill(0);
          selected[idx] = 1;
        }
        compositeSelectionIntoDisplayColors();
        rebuildLayer();
        updateStats();
        syncInteractionFromSelection();
      },
    });

    try {
      const br = await fetch("/api/betadata/genes");
      if (br.ok) {
        const genes = (await br.json()) as string[];
        betaGene.innerHTML =
          '<option value="">— pick —</option>' +
          genes.map((g) => `<option value="${g}">${g}</option>`).join("");
      }
    } catch {
      /* optional */
    }

    syncColorModeUi();
    refreshVisualization();
    if (cellJitterToggle.checked) startCellJitterLoop();
    if (meta.perturb_loading && !meta.perturb_error) {
      schedulePerturbMetaPoll();
    }
    return true;
  }

  const resizeDeck = () => {
    if (!deck) return;
    const w = Math.max(mainEl.clientWidth, 32);
    const h = Math.max(mainEl.clientHeight, 32);
    deck.setProps({ width: w, height: h });
  };

  function fitDeckToBounds(b: Meta["bounds"]) {
    if (!deck) return;
    const w = Math.max(mainEl.clientWidth, 32);
    const h = Math.max(mainEl.clientHeight, 32);
    const vs = fitOrthographic(w, h, b);
    deck.setProps({
      viewState: null,
      initialViewState: {
        target: [vs.target[0], vs.target[1], 0],
        zoom: vs.zoom,
        transitionDuration: 180,
      },
    });
  }

  layoutModeEl.addEventListener("change", () => {
    if (!positionsUmap) return;
    const wantUmap = layoutModeEl.value === "umap";
    positions = wantUmap ? positionsUmap : positionsSpatial;
    const b =
      wantUmap && meta.umap_bounds ? meta.umap_bounds : meta.bounds;
    fitDeckToBounds(b);
    rebuildLayer();
    const one = singleSelectedIndex();
    if (meta.network_loaded && interactionLensEl.checked && one !== null) {
      void fetchAndApplyInteractionContext(one);
    }
  });

  interactionLensEl.addEventListener("change", () => {
    if (!interactionLensEl.checked) {
      clearInteractionContextFull();
    } else {
      syncInteractionFromSelection();
    }
  });
  refreshContextBtn.addEventListener("click", () => {
    syncInteractionFromSelection();
  });

  interactionModeSel.addEventListener("change", () => {
    syncInteractionPanelLayout();
    if (interactionLensEl.checked) {
      syncInteractionFromSelection();
    }
  });

  cellTypeOverlayEl.addEventListener("change", () => {
    refreshVisualization();
  });

  cellTypeFilters.addEventListener("change", (ev) => {
    const t = ev.target as HTMLInputElement;
    if (!t.matches("input[data-ct-idx]")) return;
    const idx = Number(t.dataset.ctIdx);
    if (Number.isFinite(idx) && idx >= 0 && idx < typeFilterChecked.length) {
      typeFilterChecked[idx] = t.checked;
    }
    refreshVisualization();
  });

  sessionApplyBtn.addEventListener("click", async () => {
    sessionApplyBtn.disabled = true;
    sessionBusyEl.classList.remove("hidden");
    try {
      const r = await fetch("/api/session/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          adata_path: sessionAdataPath.value.trim(),
          layer: sessionLayer.value.trim(),
          cluster_annot: sessionClusterAnnot.value.trim(),
          network_dir: sessionNetworkDir.value.trim(),
          run_toml: sessionRunToml.value.trim(),
        }),
      });
      const text = await r.text();
      if (!r.ok) throw new Error(text);
      const j = JSON.parse(text) as SessionConfigureResponse;
      const ok = await initDataset(j.meta);
      if (ok) {
        setStatus(`Loaded ${meta.n_obs} cells · ${meta.spatial_obsm_key}`);
      }
    } catch (e) {
      setStatus(String(e), true);
    } finally {
      sessionApplyBtn.disabled = false;
      sessionBusyEl.classList.add("hidden");
    }
  });

  await initDataset();

  const ro = new ResizeObserver(() => {
    resizeDeck();
    rebuildLayer();
  });
  ro.observe(mainEl);

  let geneSearchTimer = 0;
  exprGene.addEventListener("input", () => {
    window.clearTimeout(geneSearchTimer);
    geneSearchTimer = window.setTimeout(async () => {
      const p = exprGene.value.trim();
      if (p.length < 2) {
        geneHints.innerHTML = "";
        return;
      }
      try {
        const r = await fetch(
          `/api/genes?prefix=${encodeURIComponent(p)}&limit=40`,
        );
        if (!r.ok) return;
        const list = (await r.json()) as string[];
        geneHints.innerHTML = list.map((g) => `<option value="${g}">`).join("");
      } catch {
        /* ignore */
      }
    }, 200);
  });

  betaGene.addEventListener("change", async () => {
    betaCol.innerHTML = '<option value="">—</option>';
    const g = betaGene.value;
    if (!g) return;
    try {
      const r = await fetch(`/api/betadata/columns?gene=${encodeURIComponent(g)}`);
      if (!r.ok) throw new Error(await r.text());
      const cols = (await r.json()) as string[];
      betaCol.innerHTML =
        '<option value="">— pick —</option>' +
        cols.map((c) => `<option value="${c}">${c}</option>`).join("");
    } catch (e) {
      setStatus(`Columns: ${e}`, true);
    }
  });

  function perturbScopePayload():
    | {
        ok: true;
        scope: {
          type: string;
          indices?: number[];
          category?: number;
          cluster_id?: number;
        };
      }
    | { ok: false; msg: string } {
    const scopeVal = perturbScope.value;
    if (scopeVal === "all") {
      return { ok: true, scope: { type: "all" } };
    }
    if (scopeVal === "selection") {
      const idx = collectSelectedIndices();
      if (idx.length === 0) {
        return {
          ok: false,
          msg: "Select cells first, or change “Where to apply”",
        };
      }
      return { ok: true, scope: { type: "indices", indices: idx } };
    }
    if (scopeVal === "cell_type") {
      const cat = Number(perturbCellType.value);
      if (!Number.isFinite(cat) || perturbCellType.value === "") {
        return { ok: false, msg: "Pick a cell type" };
      }
      return { ok: true, scope: { type: "cell_type", category: cat } };
    }
    const cid = Number(perturbClusterId.value);
    if (!Number.isFinite(cid)) {
      return { ok: false, msg: "Enter cluster id (from --cluster-annot)" };
    }
    return {
      ok: true,
      scope: { type: "cluster", cluster_id: Math.trunc(cid) },
    };
  }

  const loadActiveChannel = async () => {
    try {
      if (colorSource.value === "perturb") {
        await runPerturbFromUi();
        return;
      }
      if (colorSource.value === "expression") {
        const g = exprGene.value.trim();
        if (!g) {
          setStatus("Enter a gene symbol", true);
          return;
        }
        activeValues = await fetchF32(
          `/api/gene/expression?gene=${encodeURIComponent(g)}`,
        );
      } else {
        const g = betaGene.value;
        const col = betaCol.value;
        if (!g || !col) {
          setStatus("Pick betadata target and coefficient column", true);
          return;
        }
        activeValues = await fetchF32(
          `/api/betadata/values?gene=${encodeURIComponent(g)}&column=${encodeURIComponent(col)}`,
        );
      }
      if (activeValues.length !== n) {
        throw new Error(`length ${activeValues.length} != n_obs ${n}`);
      }
      lastColorSource =
        colorSource.value === "expression" ? "expression" : "betadata";
      refreshVisualization();
      setStatus(`Loaded ${colorSource.value} (${activeValues.length} values)`);
    } catch (e) {
      setStatus(`Load failed: ${e}`, true);
    }
  };

  async function runPerturbFromUi() {
    if (meta.perturb_error) {
      setStatus(`Perturbation unavailable: ${meta.perturb_error}`, true);
      return;
    }
    if (meta.perturb_loading) {
      setStatus(
        "Perturbation engine is still loading; wait until the status line shows “perturbation ready”.",
        true,
      );
      return;
    }
    if (!meta.perturb_ready) {
      setStatus("Perturbation needs server --run-toml", true);
      return;
    }
    const gene = perturbGene.value.trim();
    if (!gene) {
      setStatus("Enter gene to perturb", true);
      return;
    }
    const desired = Number(perturbExpr.value);
    const sp = perturbScopePayload();
    if (!sp.ok) {
      setStatus(sp.msg, true);
      return;
    }
    const scopeVal = perturbScope.value;
    setStatus("Running perturbation (may take a while)…");
    try {
      const r = await withMetaProgressPoll(
        fetch("/api/perturb/preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            gene,
            desired_expr: Number.isFinite(desired) ? desired : 0,
            scope: sp.scope,
          }),
        }).then(async (res) => {
          if (!res.ok) throw new Error(await res.text());
          return res;
        }),
      );
      const buf = await r.arrayBuffer();
      activeValues = new Float32Array(buf);
      if (activeValues.length !== n) {
        throw new Error(`length ${activeValues.length} != n_obs ${n}`);
      }
      lastColorSource = "perturb";
      perturbDisplayGene = gene;
      cmapSel.value = "diverging";
      refreshVisualization();
      setStatus(`Perturbation Δ · ${gene} · ${scopeVal}`);
    } catch (e) {
      setStatus(String(e), true);
    }
  }

  async function computeUmapTransitionField() {
    if (meta.perturb_error) {
      setStatus(`Perturbation unavailable: ${meta.perturb_error}`, true);
      return;
    }
    if (meta.perturb_loading) {
      setStatus(
        "Perturbation engine is still loading; wait for “perturbation ready” in the status bar.",
        true,
      );
      return;
    }
    if (!meta.perturb_ready) {
      setStatus("Perturbation needs server --run-toml", true);
      return;
    }
    if (!meta.umap_obsm_key) {
      setStatus("No UMAP embedding in this dataset", true);
      return;
    }
    const gene = perturbGene.value.trim();
    if (!gene) {
      setStatus("Enter gene in the perturbation row", true);
      return;
    }
    const sp = perturbScopePayload();
    if (!sp.ok) {
      setStatus(sp.msg, true);
      return;
    }
    const desired = Number(perturbExpr.value);
    if (transLimitClusters.checked) {
      const hi = Array.from(transHighlightTypes.selectedOptions).map((o) => o.value);
      if (hi.length === 0) {
        setStatus(
          "limit_clusters: select one or more types in highlight_cell_types",
          true,
        );
        return;
      }
    }
    const body = {
      gene,
      desired_expr: Number.isFinite(desired) ? desired : 0,
      scope: sp.scope,
      n_neighbors: Math.min(
        500,
        Math.max(5, Math.trunc(Number(transNeighbors.value) || 150)),
      ),
      temperature: Number(transT.value) || 0.05,
      remove_null: transRemoveNull.checked,
      unit_directions: transUnitDirs.checked,
      grid_scale: Number(transGridScale.value) || 1,
      vector_scale: Number(transVecScale.value) || 0.85,
      delta_rescale: Number(transDeltaRescale.value) || 1,
      magnitude_threshold: Math.max(0, Number(transMagThresh.value) || 0),
      use_full_graph: transFullGraph.checked,
      full_graph_max_cells: Math.min(
        8192,
        Math.max(64, Math.trunc(Number(transFullMax.value) || 4096)),
      ),
      quick_ko_sanity: transQuickKo.checked,
      limit_clusters: transLimitClusters.checked,
      highlight_cell_types: transLimitClusters.checked
        ? Array.from(transHighlightTypes.selectedOptions).map((o) => o.value)
        : [],
    };
    setStatus(
      transQuickKo.checked
        ? "Computing UMAP quiver (quick single-gene δ)…"
        : "Computing UMAP transition field…",
    );
    try {
      const r = await withMetaProgressPoll(
        fetch("/api/perturb/umap-field", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }).then(async (res) => {
          if (!res.ok) throw new Error(await res.text());
          return res;
        }),
      );
      const data = (await r.json()) as UmapFieldResponse;
      const nx = data.nx;
      const ny = data.ny;
      if (data.u.length !== nx * ny || data.v.length !== nx * ny) {
        throw new Error("quiver length mismatch");
      }
      quiverFieldCache = data;
      const nArrows = rebuildQuiverFromCache();
      syncQuiverDisplayLabels();
      rebuildLayer();
      const mode = transQuickKo.checked ? "quick δ" : "full GRN";
      setStatus(
        `UMAP quiver (${mode}): ${nArrows} arrows (${nx}×${ny} grid, ${quiverSegData.length} segments)`,
      );
    } catch (e) {
      quiverFieldCache = null;
      quiverSegData.length = 0;
      rebuildLayer();
      setStatus(String(e), true);
    }
  }

  computeQuiverBtn.addEventListener("click", () =>
    void computeUmapTransitionField(),
  );
  clearQuiverBtn.addEventListener("click", () => {
    quiverFieldCache = null;
    quiverSegData.length = 0;
    rebuildLayer();
    setStatus("Cleared UMAP quiver");
  });

  function onQuiverDisplayInput() {
    syncQuiverDisplayLabels();
    if (!quiverFieldCache) return;
    rebuildQuiverFromCache();
    rebuildLayer();
  }

  quiverVisScale.addEventListener("input", onQuiverDisplayInput);
  quiverLineW.addEventListener("input", onQuiverDisplayInput);
  quiverHeadFrac.addEventListener("input", onQuiverDisplayInput);
  quiverStride.addEventListener("input", onQuiverDisplayInput);

  perturbSummaryBtn.addEventListener("click", async () => {
    if (meta.perturb_error) {
      setStatus(`Perturbation unavailable: ${meta.perturb_error}`, true);
      return;
    }
    if (meta.perturb_loading) {
      setStatus(
        "Perturbation engine is still loading; wait for “perturbation ready”.",
        true,
      );
      return;
    }
    if (!meta.perturb_ready) {
      setStatus("Perturbation needs server --run-toml", true);
      return;
    }
    const gene = perturbGene.value.trim();
    if (!gene) {
      setStatus("Enter gene in the perturbation row", true);
      return;
    }
    const sp = perturbScopePayload();
    if (!sp.ok) {
      setStatus(sp.msg, true);
      return;
    }
    const desired = Number(perturbExpr.value);
    perturbSummaryBody.classList.remove("hidden");
    perturbSummaryBody.innerHTML =
      '<p class="interaction-loading">Computing perturbation summary…</p>';
    setStatus("Perturbation summary…");
    try {
      const r = await withMetaProgressPoll(
        fetch("/api/perturb/summary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            gene,
            desired_expr: Number.isFinite(desired) ? desired : 0,
            scope: sp.scope,
          }),
        }).then(async (res) => {
          if (!res.ok) throw new Error(await res.text());
          return res;
        }),
      );
      const d = await r.json();
      const geneRows = (d.top_affected_genes ?? [])
        .slice(0, 30)
        .map(
          (g: { gene: string; mean_delta: number; max_abs_delta: number }) =>
            `<tr><td>${escapeHtml(g.gene)}</td><td class="num">${g.mean_delta.toPrecision(3)}</td><td class="num">${g.max_abs_delta.toPrecision(3)}</td></tr>`,
        )
        .join("");
      perturbSummaryBody.innerHTML = `
        <p class="interaction-meta">Gene <strong>${escapeHtml(d.gene)}</strong> · ${d.n_obs} cells ·
          mean Δ <strong>${d.mean_delta.toPrecision(4)}</strong> · max |Δ| <strong>${d.max_abs_delta.toPrecision(4)}</strong></p>
        <p class="interaction-meta">↑ ${d.n_positive} · ↓ ${d.n_negative} · = ${d.n_zero}</p>
        <h4 style="margin:6px 0 2px;">Top affected genes</h4>
        <table class="interaction-table" style="font-size:0.82em;">
          <thead><tr><th>Gene</th><th class="num">mean Δ</th><th class="num">max |Δ|</th></tr></thead>
          <tbody>${geneRows || "<tr><td colspan='3'>—</td></tr>"}</tbody>
        </table>`;
      setStatus(
        `Perturbation summary: ${d.top_affected_genes?.length ?? 0} affected genes`,
      );
    } catch (e) {
      perturbSummaryBody.innerHTML = `<p class="interaction-error">${escapeHtml(String(e))}</p>`;
      setStatus(String(e), true);
    }
  });

  loadBtn.addEventListener("click", () => void loadActiveChannel());
  clearPerturbBtn.addEventListener("click", () => {
    activeValues = null;
    lastColorSource = null;
    perturbDisplayGene = "";
    quiverFieldCache = null;
    quiverSegData.length = 0;
    refreshVisualization();
    rebuildLayer();
    setStatus("Cleared perturbation coloring");
  });
  cellJitterToggle.addEventListener("change", () => {
    if (cellJitterToggle.checked) startCellJitterLoop();
    else {
      stopCellJitterLoop();
      rebuildLayer();
    }
  });

  cellSizeInput.addEventListener("input", () => {
    cellSizeVal.textContent = cellSizeInput.value;
    rebuildLayer();
  });
  cmapSel.addEventListener("change", () => {
    refreshVisualization();
  });

  clearSel.addEventListener("click", () => {
    selected.fill(0);
    clearInteractionContextVisuals();
    interactionBodyEl.innerHTML = "";
    refreshVisualization();
  });

  let brushDown: { x: number; y: number } | null = null;
  let rectEl: HTMLDivElement | null = null;

  brushToggle.addEventListener("change", () => {
    brushOverlay.classList.toggle("active", brushToggle.checked);
    if (!brushToggle.checked) {
      brushDown = null;
      if (rectEl) {
        rectEl.remove();
        rectEl = null;
      }
    }
  });

  brushOverlay.addEventListener("mousedown", (ev) => {
    if (!brushToggle.checked) return;
    const r = brushOverlay.getBoundingClientRect();
    brushDown = { x: ev.clientX - r.left, y: ev.clientY - r.top };
    rectEl = document.createElement("div");
    rectEl.className = "brush-rect";
    rectEl.style.left = `${brushDown.x}px`;
    rectEl.style.top = `${brushDown.y}px`;
    rectEl.style.width = "0";
    rectEl.style.height = "0";
    brushOverlay.appendChild(rectEl);
  });

  brushOverlay.addEventListener("mousemove", (ev) => {
    if (!brushDown || !rectEl) return;
    const r = brushOverlay.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const y = ev.clientY - r.top;
    const x0 = Math.min(brushDown.x, x);
    const y0 = Math.min(brushDown.y, y);
    const w = Math.abs(x - brushDown.x);
    const h = Math.abs(y - brushDown.y);
    rectEl.style.left = `${x0}px`;
    rectEl.style.top = `${y0}px`;
    rectEl.style.width = `${w}px`;
    rectEl.style.height = `${h}px`;
  });

  brushOverlay.addEventListener("mouseup", (ev) => {
    if (!brushDown || !rectEl) return;
    if (!deck || n === 0) {
      rectEl.remove();
      rectEl = null;
      brushDown = null;
      return;
    }
    const r = brushOverlay.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const y = ev.clientY - r.top;
    selectInRect(deck, brushDown.x, brushDown.y, x, y, positions, n, selected, (i) =>
      cellSelectableByType(i),
    );
    rectEl.remove();
    rectEl = null;
    brushDown = null;
    compositeSelectionIntoDisplayColors();
    rebuildLayer();
    updateStats();
    syncInteractionFromSelection();
  });

  brushOverlay.addEventListener("mouseleave", () => {
    if (rectEl) {
      rectEl.remove();
      rectEl = null;
    }
    brushDown = null;
  });
}

main().catch((e) => {
  console.error(e);
  document.body.textContent = String(e);
});

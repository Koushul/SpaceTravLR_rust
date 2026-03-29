import "./style.css";

import {
  Deck,
  OrthographicView,
  type OrthographicViewState,
} from "@deck.gl/core";
import { LineLayer, ScatterplotLayer } from "@deck.gl/layers";
import {
  applyBetadataColorsPerCluster,
  applyColors,
  colormapLegendGradientCss,
  type ColormapId,
} from "./colormaps";
import { rgbForCategoryIndex } from "./cellTypePalette";
import {
  attachMcpCaptureSink,
  attachMcpCollectInteractionsSink,
  attachMcpControlSink,
  attachMcpPerturbRunSink,
  attachMcpReceivedLigandSink,
  attachMcpSignatureUmapSink,
  attachMcpSplashNetworkSink,
  bootstrapMcp,
  type McpCaptureRequest,
  type McpCollectInteractionsRequest,
  type McpPerturbRunRequest,
  type McpReceivedLigandRequest,
  type McpSignatureUmapRequest,
  type McpSplashNetworkRequest,
} from "./mcpBridge";
import {
  renderSplashNetwork,
  type SplashForceParams,
  type SplashNetworkJson,
  type SplashNetworkLayout,
} from "./splashNetwork";

let globalApiBase = "";

function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const p = path.startsWith("/") ? path : `/${path}`;
  const b = globalApiBase.replace(/\/$/, "");
  return b ? `${b}${p}` : p;
}

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
  const r = await fetch(apiUrl(path));
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}`);
  }
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

async function postF32(path: string, body: unknown): Promise<Float32Array> {
  const r = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${t}`);
  }
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

async function fetchU32(path: string): Promise<Uint32Array> {
  const r = await fetch(apiUrl(path));
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
  /** 0–1000 permille; prefer over `perturb_progress_percent` for progress bar / label when set. */
  perturb_progress_permille?: number | null;
  perturb_progress_label?: string | null;
  adata_path: string;
  betadata_dir: string;
  network_dir?: string | null;
  run_toml?: string | null;
  /** False until a .h5ad is loaded (CLI or Dataset paths). */
  dataset_ready?: boolean;
  /** Training spatial config + received-ligand channel names when perturb runtime is loaded. */
  spatial_model?: {
    weighted_ligand_scale_factor: number;
    spatial_radius: number;
    contact_distance: number;
    spatial_dim: number;
    received_ligand_n_channels: number;
    received_ligand_columns_sample: string[];
    tfl_ligand_n_channels: number;
  } | null;
}

interface SessionConfigureResponse {
  ok: boolean;
  message: string;
  meta: Meta;
}

interface CollectedInteractionRow {
  interaction: string;
  gene: string;
  beta: number;
  interaction_type: string;
}

interface CollectInteractionsApiResponse {
  interactions: CollectedInteractionRow[];
  n_reported: number;
  n_total: number;
  capped: boolean;
}

interface PairLrRow {
  target_gene: string;
  interaction: string;
  beta_cell_a: number;
  beta_cell_b: number;
  score: number;
}

interface PairLrApiResponse {
  cell_a: number;
  cell_b: number;
  betadata_row_id?: string | null;
  rows: PairLrRow[];
  n_genes_scanned: number;
}

function metaDatasetSignature(m: Meta): string {
  return [
    m.dataset_ready === false ? "0" : "1",
    m.n_obs,
    m.n_vars,
    m.adata_path,
    m.layer,
    m.cluster_annot,
    m.network_dir ?? "",
    m.run_toml ?? "",
  ].join("\u001f");
}

function cellTypeSignature(m: Meta): string {
  return [
    m.cell_type_column ?? "",
    ...(m.cell_type_categories ?? []),
  ].join("\u001f");
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
  /** Server wrote quiver lines to this path when `export_svg` was true (e.g. under `/tmp` on Unix). */
  svg_export_path?: string | null;
}

/** POST /api/umap/signature_field — gene-set sum on cells, KNN + gradient quiver (VirtualTissue-style). */
interface UmapSignatureFieldResponse extends UmapFieldResponse {
  signature_per_cell: number[];
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
    distance?: number;
    cell_type?: string | null;
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

function canvasToScaledPngBase64(
  canvas: HTMLCanvasElement,
  maxW?: number,
): string {
  const w = canvas.width;
  const h = canvas.height;
  let tw = w;
  let th = h;
  if (maxW != null && maxW > 0 && w > maxW) {
    tw = maxW;
    th = Math.max(1, Math.round((h * maxW) / w));
  }
  const out = document.createElement("canvas");
  out.width = tw;
  out.height = th;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(canvas, 0, 0, tw, th);
  const dataUrl = out.toDataURL("image/png");
  const i = dataUrl.indexOf(",");
  if (i < 0) throw new Error("invalid data URL");
  return dataUrl.slice(i + 1);
}

async function waitAnimationFrames(count: number): Promise<void> {
  for (let i = 0; i < count; i++) {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve());
    });
  }
}

function globalMean(values: Float32Array | null): string {
  if (!values || values.length === 0) return "—";
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i]!;
  return (s / values.length).toPrecision(5);
}

async function main() {
  const mcp = await bootstrapMcp();
  globalApiBase = mcp.apiBase;

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
      <details class="session-panel hidden" id="mcpPanel">
        <summary class="session-summary">MCP</summary>
        <p class="session-hint" style="margin:0.35rem 0;font-size:0.85em;color:var(--st-muted);">
          Send a snapshot of the current viewer state to the assistant (inline MCP host).
        </p>
        <button type="button" class="secondary" id="mcpReportContextBtn">
          Send context to chat
        </button>
      </details>
      <div class="control-section">
        <h2 class="control-section-title">Color &amp; data</h2>
      <div class="toolbar toolbar-sidebar">
      <label>Color source
        <select id="colorSource">
          <option value="expression">Expression</option>
          <option value="betadata" id="colorSourceBetaOpt">Betadata</option>
          <option value="received_ligand">Received ligand</option>
          <option value="perturb" id="colorSourcePerturbOpt" class="hidden">
            Perturbation (KO)
          </option>
        </select>
      </label>
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
      <div id="recvLigWrap" class="recv-lig-wrap hidden">
        <label
          >Source
          <select id="recvLigSource">
            <option value="adata">From expression (Gaussian)</option>
            <option value="model" id="recvLigSourceModelOpt">From model (training)</option>
          </select>
        </label>
        <label
          >Ligand gene(s)
          <input
            id="recvLigGenes"
            list="recvLigGeneHints"
            placeholder="TGFA, IL2 (comma-sep) or one column for model"
          />
          <datalist id="recvLigGeneHints"></datalist>
        </label>
        <label id="recvLigMatrixWrap" class="hidden"
          >Model matrix
          <select id="recvLigMatrix">
            <option value="lr">LR received</option>
            <option value="tfl">TFL received</option>
          </select>
        </label>
        <label id="recvLigRadiusWrap"
          >Radius (coord. units)
          <input type="number" id="recvLigRadius" min="0.0001" step="any" value="120" />
        </label>
        <label id="recvLigScaleWrap"
          >Scale factor
          <input type="number" id="recvLigScale" min="0" step="any" value="1" />
        </label>
        <label id="recvLigGridWrap" class="recv-lig-check"
          ><input type="checkbox" id="recvLigGrid" checked /> Grid accelerate (O(N²) off)</label
        >
        <label id="recvLigGridFactorWrap"
          >Grid factor
          <input type="number" id="recvLigGridFactor" min="0.05" max="2" step="0.05" value="0.5" />
        </label>
        <label
          >Aggregate channels
          <select id="recvLigAgg">
            <option value="sum">sum</option>
            <option value="max">max</option>
            <option value="mean">mean</option>
          </select>
        </label>
        <p class="recv-lig-hint" id="recvLigHint">
          Spatial Gaussian received signal from neighbor ligand expression (same rule as training).
          Model path needs <strong>perturb_ready</strong> and a column name from training.
        </p>
      </div>
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
      <div class="toolbar-load-row">
        <button type="button" id="loadColor">Load / refresh</button>
        <button type="button" id="cancelJobsBtn" class="secondary" title="Stop perturb jobs and hide stuck loading (background betabase load exits on its own)">
          Cancel jobs
        </button>
      </div>
      <p class="toolbar-hint">
        With <strong>Interaction lens</strong> on, click a cell to set the sender for GRN context.
        Betadata feathers are <strong>seed-only</strong> (<code>Cluster</code>) or
        <strong>spatial</strong> (<code>CellID</code>); status line shows which was detected.
      </p>
      </div>
      </div>
    <details class="transition-details hidden" id="umapQuiverPanel">
      <summary class="transition-summary">Perturbation &amp; UMAP quiver</summary>
      <div class="transition-inner">
        <p class="transition-hint">
          Same pipeline as Python <code>VirtualTissue.plot_arrows</code> →
          <code>Cartography.plot_umap_quiver</code> (<a
            href="https://spacetravlr.readthedocs.io/en/latest/ligand_perturbation.html"
            target="_blank"
            rel="noopener"
            >tutorial</a
          >). Set the perturbation target here; <strong>Color source → Perturbation (KO)</strong> +
          <strong>Load / refresh</strong> colors cells by Δ using these fields. <strong>Quick
            sanity</strong> skips GRN (single-gene δ). Use <strong>UMAP</strong> layout for arrows.
        </p>
        <p class="transition-note hidden" id="perturbUmapMissingHint">
          No 2D UMAP in this dataset — use <strong>Load / refresh</strong> with color =
          Perturbation on spatial; the quiver button is disabled until UMAP exists.
        </p>
        <h3 class="transition-subhead">Perturbation target</h3>
        <div class="transition-grid">
          <label class="perturb-field" for="perturbGene"
            >Gene to perturb
            <input
              id="perturbGene"
              aria-label="Gene to perturb"
              list="geneHints"
              placeholder="var symbol"
            />
          </label>
          <label class="perturb-field" for="perturbExpr"
            >Target expr
            <input
              type="number"
              id="perturbExpr"
              aria-label="Target expression after perturbation"
              value="0"
              step="any"
            />
          </label>
          <label class="perturb-field" for="perturbScope"
            >Where to apply
            <select id="perturbScope" aria-label="Perturbation scope">
              <option value="all">All cells</option>
              <option value="cell_type">One cell type (annotation)</option>
              <option value="cluster">One cluster id</option>
            </select>
          </label>
          <label id="perturbCellTypeWrap" class="hidden"
            >Cell type
            <select id="perturbCellType" aria-label="Perturb cell type category">
              <option value="">—</option>
            </select>
          </label>
          <label id="perturbClusterWrap" class="hidden"
            >Cluster id (<code>--cluster-annot</code>)
            <input
              type="number"
              id="perturbClusterId"
              aria-label="Perturb cluster id"
              min="0"
              step="1"
            />
          </label>
          <label
            >Perturb iterations
            <input
              type="number"
              id="perturbNProp"
              aria-label="Perturbation propagation iterations"
              min="1"
              max="32"
              value="3"
              step="1"
            />
          </label>
        </div>
        <h3 class="transition-subhead">Transition field (server)</h3>
        <div class="transition-grid">
          <label
            ><code>n_neighbors</code>
            <input type="number" id="transNeighbors" min="5" max="500" value="200" />
          </label>
          <label
            >Temperature <code>T</code>
            <input type="number" id="transT" value="0.06" step="0.005" />
          </label>
          <label
            ><code>grid_scale</code>
            <input type="number" id="transGridScale" value="2" step="0.1" min="0.1" />
          </label>
          <label
            ><code>vector_scale</code>
            <input type="number" id="transVecScale" value="1.35" step="0.05" min="0.01" />
          </label>
          <label
            ><code>rescale</code> (δ before transition)
            <input type="number" id="transDeltaRescale" value="4" step="0.1" />
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
          <button type="button" id="clearPerturb" class="secondary">
            Clear perturb Δ
          </button>
          <button
            type="button"
            class="primary"
            id="computeQuiverBtn"
            title="Runs perturbation on the server with the target above, draws UMAP arrows, and saves SVG under /tmp when successful"
          >
            Run perturb + UMAP quiver
          </button>
          <button type="button" class="secondary" id="perturbSummaryBtn">
            Perturbation summary
          </button>
          <button type="button" class="secondary" id="clearQuiverBtn">Clear quiver</button>
        </div>
        <div class="signature-umap-wrap">
          <p class="signature-umap-title">Gene signature on UMAP</p>
          <p class="signature-umap-hint">
            Sum of layer expression across genes → KNN on the UMAP grid → gradient arrows (same idea as
            <code>VirtualTissue.signature2gradient</code>). Optional mask zeros arrows where the perturbation
            quiver is zero.
          </p>
          <label class="signature-umap-genes"
            >Signature genes (comma-separated)
            <input
              id="sigUmapGenes"
              type="text"
              placeholder="e.g. IL2RA, IL2RB, IFNG"
            />
          </label>
          <label
            >KNN (grid interpolation)
            <input type="number" id="sigUmapKnn" min="3" max="200" step="1" value="30" />
          </label>
          <label class="sig-umap-check"
            ><input type="checkbox" id="sigUmapMaskPerturb" /> Mask with perturb quiver (needs
            <strong>perturb_ready</strong>; uses gene + scope + quick δ row above)</label
          >
          <div class="transition-actions signature-umap-actions">
            <button type="button" class="primary" id="computeSigUmapBtn">
              Compute signature quiver
            </button>
            <button type="button" class="secondary" id="clearSigUmapBtn">
              Clear signature quiver
            </button>
            <button type="button" class="secondary" id="colorBySigUmapBtn">
              Color cells by signature
            </button>
          </div>
        </div>
        <div class="splash-net-wrap" id="splashNetWrap">
          <div class="splash-net-controls">
            <div class="splash-net-controls-head">
              <button
                type="button"
                class="secondary splash-net-controls-toggle"
                id="splashNetControlsToggle"
                aria-expanded="true"
                title="Show or hide gene and layout settings"
              >
                Hide settings
              </button>
              <button
                type="button"
                class="secondary splash-net-exit-fs"
                id="splashNetExitFullscreenBtn"
                title="Leave full-screen splash view (Esc)"
              >
                Exit full screen
              </button>
            </div>
            <div class="splash-net-controls-inner" id="splashNetControlsInner">
              <p class="splash-net-title">Splash signal network (A → B)</p>
              <p class="splash-net-hint">
                Mean <code>splash()</code> derivative ∂(target)/∂(source) per edge (same cell mask as
                perturb scope). <strong>Gene B</strong> must be a trained target.
                <strong>Layered</strong> is left-to-right; <strong>Force</strong> is draggable. Hover to
                highlight links. Scroll to zoom.
              </p>
              <div class="splash-net-fields">
                <label class="perturb-field"
                  >Gene A (upstream)
                  <input type="text" id="splashNetGeneA" placeholder="e.g. TGFB1" autocomplete="off" />
                </label>
                <label class="perturb-field"
                  >Gene B (trained target)
                  <input type="text" id="splashNetGeneB" placeholder="e.g. COL1A1" autocomplete="off" />
                </label>
                <label class="perturb-field splash-net-layout-field"
                  >Layout
                  <select id="splashNetLayout">
                    <option value="layered" selected>Layered (A → B)</option>
                    <option value="force">Force</option>
                  </select>
                </label>
                <label class="splash-net-slider"
                  >Context hops <span id="splashNetHopsVal">1</span>
                  <input type="range" id="splashNetHops" min="0" max="4" step="1" value="1" />
                </label>
                <label class="splash-net-slider"
                  >Max nodes <span id="splashNetMaxNodesVal">24</span>
                  <input type="range" id="splashNetMaxNodes" min="6" max="64" step="2" value="24" />
                </label>
                <div class="splash-net-progress-wrap hidden" id="splashNetProgressWrap" aria-live="polite">
                  <div class="splash-net-progress-head">
                    <span class="splash-net-progress-title">Computing splash graph</span>
                    <span class="splash-net-progress-label" id="splashNetProgressLabel">0%</span>
                  </div>
                  <div class="splash-net-progress-track">
                    <div class="splash-net-progress-fill" id="splashNetProgressFill"></div>
                  </div>
                </div>
                <div class="splash-net-force-fields hidden" id="splashNetForceFields">
                  <p class="splash-net-force-title">Force layout</p>
                  <label class="splash-net-slider"
                    >Link distance (strong) <span id="splashNetForceLinkMinVal">36</span> px
                    <input
                      type="range"
                      id="splashNetForceLinkMin"
                      min="15"
                      max="90"
                      step="1"
                      value="36"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Link stretch (weak) <span id="splashNetForceLinkSpanVal">120</span> px
                    <input
                      type="range"
                      id="splashNetForceLinkSpan"
                      min="20"
                      max="250"
                      step="2"
                      value="120"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Link strength <span id="splashNetForceStrengthVal">0.35</span>
                    <input
                      type="range"
                      id="splashNetForceStrength"
                      min="5"
                      max="100"
                      step="1"
                      value="35"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Repulsion <span id="splashNetForceChargeVal">220</span>
                    <input
                      type="range"
                      id="splashNetForceCharge"
                      min="40"
                      max="500"
                      step="5"
                      value="220"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Collision padding <span id="splashNetForceCollideVal">14</span> px
                    <input
                      type="range"
                      id="splashNetForceCollide"
                      min="2"
                      max="40"
                      step="1"
                      value="14"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Cooling (α decay ×10⁴) <span id="splashNetForceAlphaDecayVal">228</span>
                    <input
                      type="range"
                      id="splashNetForceAlphaDecay"
                      min="80"
                      max="500"
                      step="2"
                      value="228"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Friction (velocity decay) <span id="splashNetForceVelocityVal">0.40</span>
                    <input
                      type="range"
                      id="splashNetForceVelocity"
                      min="15"
                      max="85"
                      step="1"
                      value="40"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Drag reheat <span id="splashNetForceDragAlphaVal">0.35</span>
                    <input
                      type="range"
                      id="splashNetForceDragAlpha"
                      min="10"
                      max="70"
                      step="1"
                      value="35"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Link solver passes <span id="splashNetForceLinkIterVal">1</span>
                    <input
                      type="range"
                      id="splashNetForceLinkIter"
                      min="1"
                      max="8"
                      step="1"
                      value="1"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Zoom min <span id="splashNetForceZoomMinVal">0.35</span>×
                    <input
                      type="range"
                      id="splashNetForceZoomMin"
                      min="20"
                      max="90"
                      step="1"
                      value="35"
                    />
                  </label>
                  <label class="splash-net-slider"
                    >Zoom max <span id="splashNetForceZoomMaxVal">3.0</span>×
                    <input
                      type="range"
                      id="splashNetForceZoomMax"
                      min="150"
                      max="600"
                      step="10"
                      value="300"
                    />
                  </label>
                </div>
              </div>
              <p class="splash-net-scope-hint">
                Cell mask: same as <strong>Perturbation target</strong> scope above.
              </p>
              <div class="transition-actions splash-net-actions">
                <button type="button" class="primary" id="splashNetComputeBtn">
                  Compute splash network
                </button>
                <button
                  type="button"
                  class="secondary"
                  id="splashNetFullscreenBtn"
                  title="Fill the window with the splash panel. Press Esc to exit."
                >
                  Full screen
                </button>
              </div>
              <p class="splash-net-message hidden" id="splashNetMessage"></p>
            </div>
          </div>
          <div id="splashNetChart" class="splash-net-chart"></div>
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
            >Vis. length <span id="quiverVisScaleVal">185</span>%
            <input
              type="range"
              id="quiverVisScale"
              min="10"
              max="300"
              step="5"
              value="185"
            />
          </label>
          <label class="toolbar-cell-size"
            >Line width <span id="quiverLineWVal">2.5</span> px
            <input
              type="range"
              id="quiverLineW"
              min="0.5"
              max="8"
              step="0.5"
              value="2.5"
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
          title="Unchecked types are dimmed on the plot"
        ></div>
      </div>
    </details>
    <details class="ccc-panel filter-details hidden" id="cccInteractionsPanel">
      <summary class="filter-summary">β interactions (parallel collect)</summary>
      <div class="ccc-inner">
        <p class="ccc-hint">
          Rust + Rayon scans <code>*_betadata.feather</code> like Python
          <code>Betabase.collect_interactions</code> — see
          <a
            href="https://spacetravlr.readthedocs.io/en/latest/ligand_receptor_interactions.html"
            target="_blank"
            rel="noopener"
            >ligand–receptor tutorial</a
          >.
        </p>
        <label class="ccc-show-plot"
          ><input type="checkbox" id="cccShowPlot" checked /> Show horizontal bar chart</label
        >
        <div id="cccChartWrap" class="ccc-chart-wrap">
          <div class="ccc-chart-title" id="cccChartTitle">Top interactions</div>
          <div id="cccBars" class="ccc-bars"></div>
        </div>
        <div class="ccc-grid">
          <label
            >Filter
            <select id="cccFilterMode">
              <option value="cell_type">Cell type</option>
              <option value="cluster">Cluster id</option>
            </select>
          </label>
          <label id="cccCellTypeWrap"
            >Type
            <select id="cccCellType"><option value="">—</option></select>
          </label>
          <label id="cccClusterWrap" class="hidden"
            >Cluster id
            <input type="number" id="cccClusterId" min="0" step="1" value="0" />
          </label>
          <label
            >Aggregate
            <select id="cccAggregate">
              <option value="mean">mean</option>
              <option value="min">min</option>
              <option value="max">max</option>
              <option value="sum">sum</option>
              <option value="positive">positive</option>
              <option value="negative">negative</option>
            </select>
          </label>
          <label
            >Plot
            <select id="cccPlotKind">
              <option value="ligand-receptor">Ligand–receptor</option>
              <option value="all">All types</option>
            </select>
          </label>
          <label
            >Top K
            <input type="number" id="cccTopK" min="5" max="40" value="15" step="1" />
          </label>
        </div>
        <div class="ccc-actions">
          <button type="button" id="cccComputeBtn" class="primary">Collect interactions</button>
          <span class="session-busy hidden" id="cccBusy">Scanning…</span>
        </div>
        <p class="ccc-footnote" id="cccFootnote"></p>
      </div>
    </details>
    <details class="pair-lr-panel filter-details hidden" id="pairLrPanel">
      <summary class="filter-summary">Pair cells — top L–R β</summary>
      <div class="pair-lr-inner">
        <p class="pair-lr-hint">
          Uses the same feather row mapping as parallel collect (<strong>Cluster</strong> vs
          <strong>CellID</strong> from metadata). Neighbor radius is in <strong>spatial coordinate
          units</strong> (µm if your slide is in µm). Works in <strong>Spatial</strong> layout only.
        </p>
        <label class="pair-lr-toggle"
          ><input type="checkbox" id="pairLrToggle" /> Pick two cells on the plot</label
        >
        <div class="pair-lr-grid">
          <label
            >Radius
            <input
              type="number"
              id="pairLrRadius"
              min="0.0001"
              step="any"
              value="300"
            />
          </label>
          <label
            >Top K
            <input type="number" id="pairLrTopK" min="5" max="60" value="20" step="1" />
          </label>
        </div>
        <div class="pair-lr-actions">
          <button type="button" id="pairLrClearBtn">Clear selection</button>
          <span class="session-busy hidden" id="pairLrBusy">Scanning β…</span>
        </div>
        <p class="pair-lr-status" id="pairLrStatus"></p>
        <div id="pairLrChartWrap" class="pair-lr-chart-wrap">
          <div class="pair-lr-chart-title perturb-cell-pair" id="pairLrChartTitle">
            Top ligand–receptor β for pair
          </div>
          <div id="pairLrBars" class="pair-lr-bars"></div>
        </div>
        <p class="pair-lr-footnote" id="pairLrFootnote"></p>
      </div>
    </details>
    <details class="interaction-details hidden" id="interactionPanel">
      <summary class="interaction-summary">GRN neighbors (sender cell)</summary>
      <div class="interaction-details-inner">
        <p class="interaction-hint">
          Enable <strong>Interaction lens</strong>, enter a focus gene, then <strong>click a cell</strong>
          on the plot (sender). Shows GRN-supported L→R links to kNN or radius neighbors.
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
      <div
        class="sidebar-resizer"
        id="sidebarResizer"
        role="separator"
        aria-orientation="vertical"
        title="Drag to resize control panel"
      ></div>
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
              title="Very subtle motion on points (visual only)"
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
          <details
            id="cellTypeLegendWrap"
            class="cell-type-legend-wrap hidden"
            open
          >
            <summary
              id="cellTypeLegendSummary"
              class="cell-type-legend-summary"
            >
              Cell types
            </summary>
            <div id="cellTypeLegendBody" class="cell-type-legend-body"></div>
          </details>
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
  const sidebarResizer = root.querySelector<HTMLDivElement>("#sidebarResizer")!;
  const toggleToolbarBtn =
    root.querySelector<HTMLButtonElement>("#toggleToolbar")!;
  const SIDEBAR_WIDTH_LS = "spatialViewerSidebarWidthPx";
  const colorBarWrap = root.querySelector<HTMLDivElement>("#colorBarWrap")!;
  const colorBarGradient =
    root.querySelector<HTMLDivElement>("#colorBarGradient")!;
  const colorBarLo = root.querySelector<HTMLSpanElement>("#colorBarLo")!;
  const colorBarHi = root.querySelector<HTMLSpanElement>("#colorBarHi")!;
  const colorBarTitle =
    root.querySelector<HTMLDivElement>("#colorBarTitle")!;
  const deckContainer = root.querySelector<HTMLDivElement>("#deck-root")!;
  const colorSource = root.querySelector<HTMLSelectElement>("#colorSource")!;
  const exprGene = root.querySelector<HTMLInputElement>("#exprGene")!;
  const geneHints = root.querySelector<HTMLDataListElement>("#geneHints")!;
  const betaGene = root.querySelector<HTMLSelectElement>("#betaGene")!;
  const betaCol = root.querySelector<HTMLSelectElement>("#betaCol")!;
  const cmapSel = root.querySelector<HTMLSelectElement>("#cmap")!;
  const exprGeneWrap = root.querySelector<HTMLLabelElement>("#exprGeneWrap")!;
  const betaGeneWrap = root.querySelector<HTMLLabelElement>("#betaGeneWrap")!;
  const betaColWrap = root.querySelector<HTMLLabelElement>("#betaColWrap")!;
  const recvLigWrap = root.querySelector<HTMLDivElement>("#recvLigWrap")!;
  const recvLigSource = root.querySelector<HTMLSelectElement>("#recvLigSource")!;
  const recvLigSourceModelOpt = root.querySelector<HTMLOptionElement>(
    "#recvLigSourceModelOpt",
  )!;
  const recvLigGenes = root.querySelector<HTMLInputElement>("#recvLigGenes")!;
  const recvLigGeneHints =
    root.querySelector<HTMLDataListElement>("#recvLigGeneHints")!;
  const recvLigMatrixWrap =
    root.querySelector<HTMLLabelElement>("#recvLigMatrixWrap")!;
  const recvLigMatrix = root.querySelector<HTMLSelectElement>("#recvLigMatrix")!;
  const recvLigRadiusWrap =
    root.querySelector<HTMLLabelElement>("#recvLigRadiusWrap")!;
  const recvLigRadius = root.querySelector<HTMLInputElement>("#recvLigRadius")!;
  const recvLigScaleWrap =
    root.querySelector<HTMLLabelElement>("#recvLigScaleWrap")!;
  const recvLigScale = root.querySelector<HTMLInputElement>("#recvLigScale")!;
  const recvLigGridWrap =
    root.querySelector<HTMLLabelElement>("#recvLigGridWrap")!;
  const recvLigGrid = root.querySelector<HTMLInputElement>("#recvLigGrid")!;
  const recvLigGridFactorWrap =
    root.querySelector<HTMLLabelElement>("#recvLigGridFactorWrap")!;
  const recvLigGridFactor =
    root.querySelector<HTMLInputElement>("#recvLigGridFactor")!;
  const recvLigAgg = root.querySelector<HTMLSelectElement>("#recvLigAgg")!;
  const loadBtn = root.querySelector<HTMLButtonElement>("#loadColor")!;
  const cancelJobsBtn = root.querySelector<HTMLButtonElement>("#cancelJobsBtn")!;
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
  const perturbNProp =
    root.querySelector<HTMLInputElement>("#perturbNProp")!;
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
  const mcpPanel = root.querySelector<HTMLDetailsElement>("#mcpPanel")!;
  const mcpReportContextBtn =
    root.querySelector<HTMLButtonElement>("#mcpReportContextBtn")!;
  const cccInteractionsPanel =
    root.querySelector<HTMLDetailsElement>("#cccInteractionsPanel")!;
  const cccShowPlot = root.querySelector<HTMLInputElement>("#cccShowPlot")!;
  const cccChartWrap = root.querySelector<HTMLDivElement>("#cccChartWrap")!;
  const cccChartTitle = root.querySelector<HTMLDivElement>("#cccChartTitle")!;
  const cccBars = root.querySelector<HTMLDivElement>("#cccBars")!;
  const cccFilterMode = root.querySelector<HTMLSelectElement>("#cccFilterMode")!;
  const cccCellTypeWrap =
    root.querySelector<HTMLLabelElement>("#cccCellTypeWrap")!;
  const cccCellType = root.querySelector<HTMLSelectElement>("#cccCellType")!;
  const cccClusterWrap =
    root.querySelector<HTMLLabelElement>("#cccClusterWrap")!;
  const cccClusterId = root.querySelector<HTMLInputElement>("#cccClusterId")!;
  const cccAggregate = root.querySelector<HTMLSelectElement>("#cccAggregate")!;
  const cccPlotKind = root.querySelector<HTMLSelectElement>("#cccPlotKind")!;
  const cccTopK = root.querySelector<HTMLInputElement>("#cccTopK")!;
  const cccComputeBtn =
    root.querySelector<HTMLButtonElement>("#cccComputeBtn")!;
  const cccBusy = root.querySelector<HTMLSpanElement>("#cccBusy")!;
  const cccFootnote = root.querySelector<HTMLParagraphElement>("#cccFootnote")!;
  const pairLrPanel = root.querySelector<HTMLDetailsElement>("#pairLrPanel")!;
  const pairLrToggle = root.querySelector<HTMLInputElement>("#pairLrToggle")!;
  const pairLrRadius = root.querySelector<HTMLInputElement>("#pairLrRadius")!;
  const pairLrTopK = root.querySelector<HTMLInputElement>("#pairLrTopK")!;
  const pairLrClearBtn =
    root.querySelector<HTMLButtonElement>("#pairLrClearBtn")!;
  const pairLrBusy = root.querySelector<HTMLSpanElement>("#pairLrBusy")!;
  const pairLrStatus = root.querySelector<HTMLParagraphElement>("#pairLrStatus")!;
  const pairLrChartWrap =
    root.querySelector<HTMLDivElement>("#pairLrChartWrap")!;
  const pairLrChartTitle =
    root.querySelector<HTMLDivElement>("#pairLrChartTitle")!;
  const pairLrBars = root.querySelector<HTMLDivElement>("#pairLrBars")!;
  const pairLrFootnote =
    root.querySelector<HTMLParagraphElement>("#pairLrFootnote")!;
  function syncCccFilterFields() {
    const m = cccFilterMode.value;
    cccCellTypeWrap.classList.toggle("hidden", m !== "cell_type");
    cccClusterWrap.classList.toggle("hidden", m !== "cluster");
  }
  syncCccFilterFields();
  cccFilterMode.addEventListener("change", syncCccFilterFields);
  cccShowPlot.addEventListener("change", () => {
    syncCccChartVisibility();
    renderCccBarChart();
  });
  cccPlotKind.addEventListener("change", () => renderCccBarChart());
  cccTopK.addEventListener("change", () => renderCccBarChart());
  if (mcp.mcpApp) {
    mcpPanel.classList.remove("hidden");
  }
  const mainEl = root.querySelector<HTMLDivElement>("#main")!;
  const cellTypeLegendWrap =
    root.querySelector<HTMLDetailsElement>("#cellTypeLegendWrap")!;
  const cellTypeLegendBody =
    root.querySelector<HTMLDivElement>("#cellTypeLegendBody")!;
  const cellTypeLegendSummary =
    root.querySelector<HTMLElement>("#cellTypeLegendSummary")!;
  const umapQuiverPanel =
    root.querySelector<HTMLDetailsElement>("#umapQuiverPanel")!;
  const perturbUmapMissingHint = root.querySelector<HTMLParagraphElement>(
    "#perturbUmapMissingHint",
  )!;
  const transNeighbors =
    root.querySelector<HTMLInputElement>("#transNeighbors")!;
  const transT = root.querySelector<HTMLInputElement>("#transT")!;
  const transGridScale =
    root.querySelector<HTMLInputElement>("#transGridScale")!;
  const transVecScale =
    root.querySelector<HTMLInputElement>("#transVecScale")!;
  const sigUmapGenes = root.querySelector<HTMLInputElement>("#sigUmapGenes")!;
  const sigUmapKnn = root.querySelector<HTMLInputElement>("#sigUmapKnn")!;
  const sigUmapMaskPerturb =
    root.querySelector<HTMLInputElement>("#sigUmapMaskPerturb")!;
  const splashNetGeneA = root.querySelector<HTMLInputElement>("#splashNetGeneA")!;
  const splashNetGeneB = root.querySelector<HTMLInputElement>("#splashNetGeneB")!;
  const splashNetHops = root.querySelector<HTMLInputElement>("#splashNetHops")!;
  const splashNetHopsVal = root.querySelector<HTMLSpanElement>("#splashNetHopsVal")!;
  const splashNetMaxNodes = root.querySelector<HTMLInputElement>("#splashNetMaxNodes")!;
  const splashNetMaxNodesVal =
    root.querySelector<HTMLSpanElement>("#splashNetMaxNodesVal")!;
  const splashNetComputeBtn =
    root.querySelector<HTMLButtonElement>("#splashNetComputeBtn")!;
  const splashNetFullscreenBtn =
    root.querySelector<HTMLButtonElement>("#splashNetFullscreenBtn")!;
  const splashNetExitFullscreenBtn =
    root.querySelector<HTMLButtonElement>("#splashNetExitFullscreenBtn")!;
  const splashNetControlsToggle =
    root.querySelector<HTMLButtonElement>("#splashNetControlsToggle")!;
  const splashNetWrap = root.querySelector<HTMLDivElement>("#splashNetWrap")!;
  const splashNetMessage =
    root.querySelector<HTMLParagraphElement>("#splashNetMessage")!;
  const splashNetChart = root.querySelector<HTMLDivElement>("#splashNetChart")!;
  const splashNetLayout = root.querySelector<HTMLSelectElement>("#splashNetLayout")!;
  const splashNetForceFields =
    root.querySelector<HTMLDivElement>("#splashNetForceFields")!;
  const splashNetProgressWrap =
    root.querySelector<HTMLDivElement>("#splashNetProgressWrap")!;
  const splashNetProgressFill =
    root.querySelector<HTMLDivElement>("#splashNetProgressFill")!;
  const splashNetProgressLabel =
    root.querySelector<HTMLSpanElement>("#splashNetProgressLabel")!;
  const splashNetForceLinkMin =
    root.querySelector<HTMLInputElement>("#splashNetForceLinkMin")!;
  const splashNetForceLinkMinVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceLinkMinVal")!;
  const splashNetForceLinkSpan =
    root.querySelector<HTMLInputElement>("#splashNetForceLinkSpan")!;
  const splashNetForceLinkSpanVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceLinkSpanVal")!;
  const splashNetForceStrength =
    root.querySelector<HTMLInputElement>("#splashNetForceStrength")!;
  const splashNetForceStrengthVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceStrengthVal")!;
  const splashNetForceCharge =
    root.querySelector<HTMLInputElement>("#splashNetForceCharge")!;
  const splashNetForceChargeVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceChargeVal")!;
  const splashNetForceCollide =
    root.querySelector<HTMLInputElement>("#splashNetForceCollide")!;
  const splashNetForceCollideVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceCollideVal")!;
  const splashNetForceAlphaDecay =
    root.querySelector<HTMLInputElement>("#splashNetForceAlphaDecay")!;
  const splashNetForceAlphaDecayVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceAlphaDecayVal")!;
  const splashNetForceVelocity =
    root.querySelector<HTMLInputElement>("#splashNetForceVelocity")!;
  const splashNetForceVelocityVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceVelocityVal")!;
  const splashNetForceDragAlpha =
    root.querySelector<HTMLInputElement>("#splashNetForceDragAlpha")!;
  const splashNetForceDragAlphaVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceDragAlphaVal")!;
  const splashNetForceLinkIter =
    root.querySelector<HTMLInputElement>("#splashNetForceLinkIter")!;
  const splashNetForceLinkIterVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceLinkIterVal")!;
  const splashNetForceZoomMin =
    root.querySelector<HTMLInputElement>("#splashNetForceZoomMin")!;
  const splashNetForceZoomMinVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceZoomMinVal")!;
  const splashNetForceZoomMax =
    root.querySelector<HTMLInputElement>("#splashNetForceZoomMax")!;
  const splashNetForceZoomMaxVal =
    root.querySelector<HTMLSpanElement>("#splashNetForceZoomMaxVal")!;
  const computeSigUmapBtn =
    root.querySelector<HTMLButtonElement>("#computeSigUmapBtn")!;
  const clearSigUmapBtn =
    root.querySelector<HTMLButtonElement>("#clearSigUmapBtn")!;
  const colorBySigUmapBtn =
    root.querySelector<HTMLButtonElement>("#colorBySigUmapBtn")!;
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

  function syncPerturbPanelsFromMeta() {
    colorSourcePerturbOpt.classList.toggle("hidden", !meta.perturb_ready);
    umapQuiverPanel.classList.toggle("hidden", !meta.perturb_ready);
    const canQuiver = !!meta.umap_obsm_key;
    computeQuiverBtn.disabled = !canQuiver;
    splashNetComputeBtn.disabled = !meta.perturb_ready;
    perturbUmapMissingHint.classList.toggle(
      "hidden",
      canQuiver || !meta.perturb_ready,
    );
    fillRecvLigGeneHintsFromMeta();
    syncRecvLigModelOptionAvailability();
  }
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
    const pm = m.perturb_progress_permille;
    const pct = m.perturb_progress_percent;
    const lbl = (m.perturb_progress_label ?? "").trim();
    const barPct =
      pm != null && Number.isFinite(pm)
        ? Math.min(100, Math.max(0, pm / 10))
        : pct != null && Number.isFinite(pct)
          ? pct
          : null;
    const pctLabel =
      pm != null && Number.isFinite(pm)
        ? (pm / 10).toFixed(1)
        : pct != null && Number.isFinite(pct)
          ? String(pct)
          : null;
    if (barPct != null) {
      syncProgressBar(barPct);
      setStatus(
        lbl
          ? `${lbl} · ${pctLabel}%`
          : `Working… ${pctLabel}%`,
      );
    }
  }

  async function withMetaProgressPoll<T>(work: Promise<T>): Promise<T> {
    const id = window.setInterval(() => {
      void (async () => {
        try {
          const mr = await fetch(apiUrl("/api/meta"));
          if (!mr.ok) return;
          const m = (await mr.json()) as Meta;
          applyMetaProgressToUi(m);
        } catch {
          /* ignore */
        }
      })();
    }, 150);
    try {
      return await work;
    } finally {
      clearInterval(id);
      try {
        const mr = await fetch(apiUrl("/api/meta"));
        if (mr.ok) {
          const m = (await mr.json()) as Meta;
          applyMetaProgressToUi(m);
          const pm = m.perturb_progress_permille;
          const pct = m.perturb_progress_percent;
          if (
            (pm != null && Number.isFinite(pm)) ||
            (pct != null && Number.isFinite(pct))
          ) {
            await new Promise((r) => setTimeout(r, 200));
          }
        }
      } catch {
        /* ignore */
      }
      syncProgressBar(null);
    }
  }

  let meta: Meta = {
    n_obs: 0,
    n_vars: 0,
    spatial_obsm_key: "",
    layer: "",
    cluster_annot: "",
    bounds: { min_x: 0, max_x: 1, min_y: 0, max_y: 1 },
    adata_path: "",
    betadata_dir: "",
    dataset_ready: false,
  };
  let lastSyncedDatasetSignature = "";
  let lastCellTypeSig = "";
  let datasetHotReloadLock = false;
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
  let pairCellA: number | null = null;
  let pairCellB: number | null = null;
  const pairNeighborSet = new Set<number>();
  let pairLrRows: PairLrRow[] = [];
  let interactionLineData: InteractionLineDatum[] = [];
  let quiverFieldCache: UmapFieldResponse | null = null;
  let quiverSegData: QuiverSegDatum[] = [];
  let sigQuiverFieldCache: UmapSignatureFieldResponse | null = null;
  let sigQuiverSegData: QuiverSegDatum[] = [];
  let signaturePerCellCache: Float32Array | null = null;
  let splashNetSimCleanup: (() => void) | null = null;
  let lastSplashNetworkJson: SplashNetworkJson | null = null;
  let baseColors!: Uint8ClampedArray;
  let colors!: Uint8ClampedArray;
  let activeValues: Float32Array | null = null;
  let rangeLo = 0;
  let rangeHi = 1;
  let scaleLine = "Scale: —";
  let lastColorSource:
    | "expression"
    | "betadata"
    | "perturb"
    | "received_ligand"
    | "gene_signature"
    | null = null;
  let perturbDisplayGene = "";
  let recvLigandLabel = "";
  let deck: Deck<OrthographicView> | undefined;

  try {
    const savedW = localStorage.getItem(SIDEBAR_WIDTH_LS);
    const n = savedW ? Number(savedW) : NaN;
    if (Number.isFinite(n) && n >= 200 && n <= 900) {
      appSidebar.style.width = `${n}px`;
    }
  } catch {
    /* ignore */
  }

  let sidebarResizeActive = false;
  sidebarResizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    sidebarResizeActive = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", (e) => {
    if (!sidebarResizeActive || appSidebar.classList.contains("sidebar-collapsed")) return;
    const rect = root.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const maxW = Math.min(720, rect.width * 0.72);
    const nw = Math.min(Math.max(x, 220), maxW);
    appSidebar.style.width = `${nw}px`;
  });
  window.addEventListener("mouseup", () => {
    if (!sidebarResizeActive) return;
    sidebarResizeActive = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    try {
      const w = Math.round(appSidebar.getBoundingClientRect().width);
      if (w >= 200) localStorage.setItem(SIDEBAR_WIDTH_LS, String(w));
    } catch {
      /* ignore */
    }
  });

  toggleToolbarBtn.addEventListener("click", () => {
    const collapsed = appSidebar.classList.toggle("sidebar-collapsed");
    sidebarResizer.style.display = collapsed ? "none" : "";
    toggleToolbarBtn.textContent = collapsed ? "Show controls" : "Hide controls";
    toggleToolbarBtn.setAttribute(
      "aria-expanded",
      collapsed ? "false" : "true",
    );
  });

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
    if (n === 0) return;
    const mAll = globalMean(activeValues);
    statsEl.innerHTML = `<div>Global mean: <strong>${mAll}</strong></div>`;
    updateColorBar();
  };

  function updateCellTypeLegend() {
    const show =
      cellTypeOverlayEl.checked &&
      !!cellTypeCodes &&
      cellCategories.length > 0 &&
      n > 0;
    if (!show) {
      cellTypeLegendWrap.classList.add("hidden");
      return;
    }
    cellTypeLegendWrap.classList.remove("hidden");
    const codes = cellTypeCodes!;
    const nc = cellCategories.length;
    const title =
      (cellTypeColumnLabel && cellTypeColumnLabel.trim()) || "Cell types";
    cellTypeLegendSummary.textContent = title;
    const rows: string[] = [];
    for (let i = 0; i < nc; i++) {
      const name = cellCategories[i] ?? `(${i})`;
      const on =
        typeFilterChecked.length === 0 ||
        i >= typeFilterChecked.length ||
        typeFilterChecked[i];
      const [r0, g0, b0] = rgbForCategoryIndex(i, nc);
      const r = on ? r0 : Math.min(255, Math.round(r0 * 0.22));
      const g = on ? g0 : Math.min(255, Math.round(g0 * 0.22));
      const b = on ? b0 : Math.min(255, Math.round(b0 * 0.22));
      const rowClass = on
        ? "cell-type-legend-row"
        : "cell-type-legend-row cell-type-legend-row--dimmed";
      rows.push(
        `<div class="${rowClass}"><span class="cell-type-legend-swatch" style="background:rgb(${r},${g},${b})"></span><span class="cell-type-legend-label">${escapeHtml(name)}</span></div>`,
      );
    }
    let hasUnknown = false;
    for (let i = 0; i < n; i++) {
      if (codes[i] === CT_UNKNOWN) {
        hasUnknown = true;
        break;
      }
    }
    if (hasUnknown) {
      rows.push(
        `<div class="cell-type-legend-row cell-type-legend-row--unknown"><span class="cell-type-legend-swatch" style="background:rgb(88,88,95)"></span><span class="cell-type-legend-label">Unknown</span></div>`,
      );
    }
    cellTypeLegendBody.innerHTML = rows.join("");
  }

  const syncPerturbScopeFields = () => {
    const s = perturbScope.value;
    perturbCellTypeWrap.classList.toggle("hidden", s !== "cell_type");
    perturbClusterWrap.classList.toggle("hidden", s !== "cluster");
  };

  function syncRecvLigPanelsFromSource() {
    const fromModel = recvLigSource.value === "model";
    recvLigMatrixWrap.classList.toggle("hidden", !fromModel);
    recvLigRadiusWrap.classList.toggle("hidden", fromModel);
    recvLigScaleWrap.classList.toggle("hidden", fromModel);
    recvLigGridWrap.classList.toggle("hidden", fromModel);
    recvLigGridFactorWrap.classList.toggle("hidden", fromModel);
    recvLigAgg.disabled = fromModel;
    recvLigAgg.closest("label")?.classList.toggle("hidden", fromModel);
  }

  function fillRecvLigGeneHintsFromMeta() {
    const sm = meta.spatial_model;
    if (!sm?.received_ligand_columns_sample?.length) {
      recvLigGeneHints.innerHTML = "";
      return;
    }
    recvLigGeneHints.innerHTML = sm.received_ligand_columns_sample
      .map((g) => `<option value="${escapeHtml(g)}"></option>`)
      .join("");
  }

  function applyRecvLigDefaultsFromMeta() {
    const sm = meta.spatial_model;
    if (sm) {
      recvLigRadius.value = String(sm.spatial_radius);
      recvLigScale.value = String(sm.weighted_ligand_scale_factor);
    } else {
      recvLigRadius.value = "120";
      recvLigScale.value = "1";
    }
  }

  function syncRecvLigModelOptionAvailability() {
    const ok = !!meta.perturb_ready;
    recvLigSourceModelOpt.disabled = !ok;
    if (!ok && recvLigSource.value === "model") {
      recvLigSource.value = "adata";
      syncRecvLigPanelsFromSource();
    }
  }

  const syncColorModeUi = () => {
    const b = colorSource.value === "betadata";
    const p = colorSource.value === "perturb";
    const rl = colorSource.value === "received_ligand";
    exprGeneWrap.classList.toggle("hidden", b || p || rl);
    betaGeneWrap.classList.toggle("hidden", !b);
    betaColWrap.classList.toggle("hidden", !b);
    recvLigWrap.classList.toggle("hidden", !rl);
    loadBtn.textContent = p
      ? "Load perturb Δ"
      : rl
        ? "Load received ligand"
        : "Load / refresh";
    if (p) syncPerturbScopeFields();
    if (rl) {
      syncRecvLigModelOptionAvailability();
      syncRecvLigPanelsFromSource();
    }
  };

  colorSource.addEventListener("change", () => {
    if (lastColorSource === "gene_signature") {
      activeValues = null;
      lastColorSource = null;
      signaturePerCellCache = null;
    }
    if (colorSource.value !== "perturb" && lastColorSource === "perturb") {
      activeValues = null;
      lastColorSource = null;
      perturbDisplayGene = "";
    }
    if (
      colorSource.value !== "received_ligand" &&
      lastColorSource === "received_ligand"
    ) {
      activeValues = null;
      lastColorSource = null;
      recvLigandLabel = "";
    }
    syncColorModeUi();
    refreshVisualization();
  });

  recvLigSource.addEventListener("change", () => {
    syncRecvLigPanelsFromSource();
  });
  perturbScope.addEventListener("change", syncPerturbScopeFields);

  function cellSelectableByType(i: number): boolean {
    if (!cellTypeCodes || typeFilterChecked.length === 0) return true;
    const c = cellTypeCodes[i]!;
    if (c === CT_UNKNOWN) return true;
    if (c >= typeFilterChecked.length) return true;
    return typeFilterChecked[c] === true;
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

  const sigQuiverRgb: [number, number, number, number] = [65, 203, 200, 228];

  function rebuildSignatureQuiverFromCache(): number {
    sigQuiverSegData.length = 0;
    const data = sigQuiverFieldCache;
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
        sigQuiverSegData.push(
          {
            sourcePosition: [gx, gy, 0],
            targetPosition: [Bx, By, 0],
            color: sigQuiverRgb,
            width: shaftW,
          },
          {
            sourcePosition: [Lx, Ly, 0],
            targetPosition: [Tx, Ty, 0],
            color: sigQuiverRgb,
            width: headW,
          },
          {
            sourcePosition: [Rx, Ry, 0],
            targetPosition: [Tx, Ty, 0],
            color: sigQuiverRgb,
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
          getFillColor: {
            value: colors,
            size: 4,
            type: "unorm8",
          },
        },
      },
      pickable: true,
      radiusUnits: "pixels",
      radiusScale: 1,
      radiusMinPixels: px,
      radiusMaxPixels: px,
      stroked: false,
      billboard: true,
      parameters: { depthWriteEnabled: false },
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
    if (
      pairLrToggle.checked &&
      layoutModeEl.value === "spatial" &&
      pairCellA !== null &&
      pairNeighborSet.size > 0
    ) {
      const neighIdx = Array.from(pairNeighborSet);
      layers.push(
        new ScatterplotLayer<number>({
          id: "pair-lr-neighbors",
          data: neighIdx,
          getPosition: (ci) => [
            posBuf[ci * 2]!,
            posBuf[ci * 2 + 1]!,
            0,
          ],
          getFillColor: [247, 201, 201, 110],
          radiusUnits: "pixels",
          radiusMinPixels: px + 2.5,
          radiusMaxPixels: px + 2.5,
          stroked: false,
          billboard: true,
          pickable: false,
          parameters: { depthTest: false, depthWriteEnabled: false },
        }),
      );
    }
    layers.push(scatterLayer);
    if (
      pairLrToggle.checked &&
      layoutModeEl.value === "spatial" &&
      (pairCellA !== null || pairCellB !== null)
    ) {
      type PairMarker = { cellIndex: number; kind: "a" | "b" };
      const markerData: PairMarker[] = [];
      if (pairCellA !== null)
        markerData.push({ cellIndex: pairCellA, kind: "a" });
      if (pairCellB !== null)
        markerData.push({ cellIndex: pairCellB, kind: "b" });
      layers.push(
        new ScatterplotLayer<PairMarker>({
          id: "pair-lr-markers",
          data: markerData,
          getPosition: (d) => [
            posBuf[d.cellIndex * 2]!,
            posBuf[d.cellIndex * 2 + 1]!,
            0,
          ],
          getFillColor: (d) =>
            d.kind === "a"
              ? [255, 200, 72, 228]
              : [236, 112, 154, 228],
          getLineColor: [255, 255, 255, 255],
          lineWidthMinPixels: 2,
          stroked: true,
          radiusUnits: "pixels",
          radiusMinPixels: px + 5,
          radiusMaxPixels: px + 5,
          billboard: true,
          pickable: false,
          parameters: { depthTest: false, depthWriteEnabled: false },
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
    if (layoutModeEl.value === "umap" && sigQuiverSegData.length > 0) {
      layers.push(
        new LineLayer({
          id: "umap-signature-quiver",
          data: sigQuiverSegData,
          getSourcePosition: (d: QuiverSegDatum) => d.sourcePosition,
          getTargetPosition: (d: QuiverSegDatum) => d.targetPosition,
          getColor: (d: QuiverSegDatum) => d.color ?? [65, 203, 200, 220],
          getWidth: (d: QuiverSegDatum) => d.width ?? 2,
          widthUnits: "pixels",
          pickable: false,
          parameters: { depthTest: false },
        }),
      );
    }
    const hasQ = quiverFieldCache !== null || sigQuiverFieldCache !== null;
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
        if (
          lastColorSource === "received_ligand" &&
          activeValues &&
          info.index < activeValues.length
        ) {
          tip += `\nReceived ligand ${recvLigandLabel || "?"}: ${activeValues[info.index]!.toPrecision(4)}`;
        }
        if (
          lastColorSource === "gene_signature" &&
          activeValues &&
          info.index < activeValues.length
        ) {
          tip += `\nSignature Σexpr: ${activeValues[info.index]!.toPrecision(4)}`;
        }
        return { text: tip };
      },
    });
  };

  const refreshVisualization = () => {
    if (n === 0 || !baseColors || !colors) {
      updateCellTypeLegend();
      return;
    }
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
      } else if (lastColorSource === "received_ligand") {
        const rr = applyColors(activeValues, baseColors, n, cmap);
        rangeLo = rr.lo;
        rangeHi = rr.hi;
        scaleLine = `Received ligand ${recvLigandLabel || "?"} [${rangeLo.toPrecision(4)}, ${rangeHi.toPrecision(4)}]`;
      } else if (lastColorSource === "gene_signature") {
        const rr = applyColors(activeValues, baseColors, n, cmap);
        rangeLo = rr.lo;
        rangeHi = rr.hi;
        scaleLine = `Gene signature (Σ expr) [${rangeLo.toPrecision(4)}, ${rangeHi.toPrecision(4)}]`;
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
    colors.set(baseColors);
    rebuildLayer();
    updateStats();
    updateCellTypeLegend();
  };

  function pairLrRadiusValue(): number {
    const r = Number(pairLrRadius.value);
    return Number.isFinite(r) && r > 0 ? r : 300;
  }

  function recomputePairNeighborSetFromA() {
    pairNeighborSet.clear();
    if (pairCellA === null || n === 0) return;
    const rad = pairLrRadiusValue();
    const r2 = rad * rad;
    const pos = positionsSpatial;
    const c = pairCellA;
    const ax = pos[c * 2]!;
    const ay = pos[c * 2 + 1]!;
    for (let j = 0; j < n; j++) {
      if (j === c) continue;
      const dx = pos[j * 2]! - ax;
      const dy = pos[j * 2 + 1]! - ay;
      if (dx * dx + dy * dy <= r2) pairNeighborSet.add(j);
    }
  }

  function clearPairLrSelection(clearToggle: boolean) {
    pairCellA = null;
    pairCellB = null;
    pairNeighborSet.clear();
    pairLrRows = [];
    pairLrBars.innerHTML = "";
    pairLrFootnote.textContent = "";
    pairLrChartTitle.textContent = "Top ligand–receptor β for pair";
    pairLrStatus.textContent = "";
    if (clearToggle) pairLrToggle.checked = false;
    rebuildLayer();
  }

  function renderPairLrBarChart() {
    const top = Math.max(5, Math.trunc(Number(pairLrTopK.value) || 20));
    const rows = pairLrRows.slice(0, top);
    if (rows.length === 0) {
      pairLrChartTitle.textContent = "Top ligand–receptor β for pair";
      pairLrBars.innerHTML =
        '<p class="pair-lr-empty">No rows yet — pick cell A, then a neighbor as cell B.</p>';
      return;
    }
    const maxS = Math.max(...rows.map((r) => r.score), 1e-30);
    pairLrChartTitle.textContent = `Top L–R β (max |β| per row) — cells #${pairCellA} & #${pairCellB}`;
    pairLrBars.innerHTML = rows
      .map((r) => {
        const label =
          r.interaction.length > 42
            ? `${r.interaction.slice(0, 40)}…`
            : r.interaction;
        const w = Math.min(100, (100 * r.score) / maxS);
        const posA = r.beta_cell_a >= 0;
        return `<div class="pair-lr-bar-row">
  <div class="pair-lr-bar-meta">
    <span class="pair-lr-bar-label" title="${escapeHtml(r.interaction)}">${escapeHtml(label)}</span>
    <span class="pair-lr-bar-gene">${escapeHtml(r.target_gene)}</span>
  </div>
  <div class="pair-lr-bar-track">
    <div class="pair-lr-bar-fill ${posA ? "pair-lr-pos" : "pair-lr-neg"}" style="width:${w.toFixed(2)}%"></div>
  </div>
  <span class="pair-lr-bar-val" title="β at A / β at B">${r.beta_cell_a.toExponential(2)} / ${r.beta_cell_b.toExponential(2)}</span>
</div>`;
      })
      .join("");
  }

  async function fetchPairLrForPair(a: number, b: number) {
    pairLrBusy.classList.remove("hidden");
    pairLrFootnote.textContent = "";
    try {
      const topN = Math.max(5, Math.trunc(Number(pairLrTopK.value) || 20));
      const r = await fetch(apiUrl("/api/betadata/pair_lr"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cell_a: a, cell_b: b, top_n: topN }),
      });
      const text = await r.text();
      if (!r.ok) throw new Error(text);
      const j = JSON.parse(text) as PairLrApiResponse;
      pairLrRows = j.rows;
      const rid = j.betadata_row_id ?? meta.betadata_row_id ?? "";
      pairLrFootnote.textContent = `Scanned ${j.n_genes_scanned} target genes · β rows keyed by ${rid || "?"}`;
      renderPairLrBarChart();
    } catch (e) {
      pairLrRows = [];
      pairLrFootnote.textContent = String(e);
      renderPairLrBarChart();
    } finally {
      pairLrBusy.classList.add("hidden");
    }
  }

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
        const distBit =
          nb.distance != null && Number.isFinite(nb.distance)
            ? ` · dist ${nb.distance.toPrecision(4)}`
            : "";
        const ctBit =
          nb.cell_type != null && nb.cell_type !== ""
            ? ` · ${escapeHtml(nb.cell_type)}`
            : "";
        return `<div class="interaction-neigh"><strong>Neighbor ${nb.index}</strong>${ctBit} (dist² ${nb.distance_sq.toPrecision(4)}${distBit})<ul class="lr-edge-list">${er}</ul></div>`;
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
        '<p class="interaction-empty">Enter a focus gene, enable Interaction lens, click a cell on the plot, or click Refresh.</p>';
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
      const r = await fetch(apiUrl("/api/network/cell-context"), {
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
          const mr = await fetch(apiUrl("/api/meta"));
          if (!mr.ok) return;
          const m = (await mr.json()) as Meta;
          meta = m;
          syncPerturbPanelsFromMeta();
          syncColorModeUi();
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
            perturbMetaPollTimer = setTimeout(tick, 250);
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
    perturbMetaPollTimer = setTimeout(tick, 250);
  }

  function syncInteractionFromSender() {
    if (!meta.network_loaded || !interactionLensEl.checked) return;
    if (interactionSenderIndex !== null) {
      void fetchAndApplyInteractionContext(interactionSenderIndex);
    } else {
      clearInteractionContextFull();
    }
  }

  function readUiNPropagation(): number {
    const v = Math.trunc(Number(perturbNProp.value) || 4);
    return Math.min(32, Math.max(1, v));
  }

  let lastCccInteractions: CollectedInteractionRow[] = [];

  function formatCccInteractionLabel(raw: string): string {
    const s = raw.startsWith("beta_") ? raw.slice(5) : raw;
    return s.replaceAll("$", "–").replaceAll("#", "→");
  }

  function syncCccChartVisibility() {
    cccChartWrap.classList.toggle("hidden", !cccShowPlot.checked);
  }

  function renderCccBarChart() {
    syncCccChartVisibility();
    if (!cccShowPlot.checked) {
      cccBars.innerHTML = "";
      return;
    }
    const topK = Math.min(
      40,
      Math.max(5, Math.trunc(Number(cccTopK.value) || 15)),
    );
    const kind = cccPlotKind.value;
    let rows = lastCccInteractions;
    if (kind === "ligand-receptor") {
      rows = rows.filter((r) => r.interaction_type === "ligand-receptor");
    }
    const sorted = [...rows].sort(
      (a, b) => Math.abs(b.beta) - Math.abs(a.beta),
    );
    const pick = sorted.slice(0, topK);
    if (pick.length === 0) {
      cccChartTitle.textContent = "No rows to plot";
      cccBars.innerHTML =
        '<p class="ccc-empty">Run collect with a different filter or aggregate, or enable “All types”.</p>';
      return;
    }
    const maxAbs = Math.max(
      1e-12,
      ...pick.map((r) => Math.abs(r.beta)),
    );
    cccChartTitle.textContent =
      kind === "ligand-receptor"
        ? `Top ${pick.length} ligand–receptor (|β|)`
        : `Top ${pick.length} interactions (|β|)`;
    cccBars.innerHTML = pick
      .map((r) => {
        const label = formatCccInteractionLabel(r.interaction);
        const sub = `${escapeHtml(r.gene)} · ${escapeHtml(r.interaction_type)}`;
        const w = (Math.abs(r.beta) / maxAbs) * 100;
        const pos = r.beta >= 0;
        return `<div class="ccc-bar-row">
  <div class="ccc-bar-meta">
    <span class="ccc-bar-label" title="${escapeHtml(r.interaction)}">${escapeHtml(label)}</span>
    <span class="ccc-bar-gene">${sub}</span>
  </div>
  <div class="ccc-bar-track">
    <div class="ccc-bar-fill ${pos ? "ccc-pos" : "ccc-neg"}" style="width:${w.toFixed(2)}%"></div>
  </div>
  <span class="ccc-bar-val">${r.beta.toExponential(3)}</span>
</div>`;
      })
      .join("");
  }

  async function runCollectInteractionsApi(opts: {
    aggregate: string;
    filter: string;
    cell_type?: string;
    cluster_id?: number;
    max_genes: number;
    push_summary_to_chat: boolean;
  }) {
    cccBusy.classList.remove("hidden");
    cccComputeBtn.disabled = true;
    cccFootnote.textContent = "";
    try {
      const body: Record<string, unknown> = {
        aggregate: opts.aggregate,
        filter: opts.filter,
        max_genes: opts.max_genes,
      };
      if (opts.filter === "cell_type" && opts.cell_type)
        body.cell_type = opts.cell_type;
      if (opts.filter === "cluster")
        body.cluster_id = opts.cluster_id ?? 0;
      const r = await fetch(apiUrl("/api/betadata/collect_interactions"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const text = await r.text();
      if (!r.ok) throw new Error(text);
      const j = JSON.parse(text) as CollectInteractionsApiResponse;
      lastCccInteractions = j.interactions;
      cccFootnote.textContent = `${j.n_reported} rows returned (${j.n_total} before response cap${j.capped ? "; output truncated" : ""}).`;
      renderCccBarChart();
      setStatus(`Collected ${j.n_total} interaction rows`);
      if (opts.push_summary_to_chat && mcp.mcpApp) {
        const lr = j.interactions
          .filter((x) => x.interaction_type === "ligand-receptor")
          .sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta))
          .slice(0, 14);
        const lines = [
          "**β interactions (ligand–receptor, top by |β|)**",
          `aggregate=${opts.aggregate} filter=${opts.filter}`,
          "",
          "| L–R | target | β |",
          "| --- | --- | --- |",
          ...lr.map(
            (row) =>
              `| ${formatCccInteractionLabel(row.interaction)} | ${row.gene} | ${row.beta.toExponential(3)} |`,
          ),
        ];
        try {
          await mcp.mcpApp.sendMessage({
            role: "user",
            content: [{ type: "text", text: lines.join("\n") }],
          });
        } catch {
          /* ignore */
        }
      }
    } catch (e) {
      cccFootnote.textContent = String(e);
      setStatus(String(e), true);
    } finally {
      cccBusy.classList.add("hidden");
      cccComputeBtn.disabled = false;
    }
  }

  async function runCollectInteractionsFromUi() {
    const mode = cccFilterMode.value;
    if (mode === "cell_type") {
      const idx = Number(cccCellType.value);
      const label = cellCategories[idx];
      if (!label) {
        setStatus("Pick a cell type for β collect", true);
        return;
      }
      await runCollectInteractionsApi({
        aggregate: cccAggregate.value,
        filter: "cell_type",
        cell_type: label,
        max_genes: 2048,
        push_summary_to_chat: false,
      });
    } else {
      const cid = Math.trunc(Number(cccClusterId.value));
      if (!Number.isFinite(cid) || cid < 0) {
        setStatus("Enter a valid cluster id", true);
        return;
      }
      await runCollectInteractionsApi({
        aggregate: cccAggregate.value,
        filter: "cluster",
        cluster_id: cid,
        max_genes: 2048,
        push_summary_to_chat: false,
      });
    }
  }

  cccComputeBtn.addEventListener("click", () => void runCollectInteractionsFromUi());

  async function executePerturbPreview(
    gene: string,
    desired: number,
    scope: unknown,
    nProp: number,
  ): Promise<boolean> {
    if (meta.perturb_error) {
      setStatus(`Perturbation unavailable: ${meta.perturb_error}`, true);
      return false;
    }
    if (meta.perturb_loading) {
      setStatus(
        "Perturbation engine is still loading; wait until the status line shows “perturbation ready”.",
        true,
      );
      return false;
    }
    if (!meta.perturb_ready) {
      setStatus("Perturbation needs server --run-toml", true);
      return false;
    }
    if (!gene.trim()) {
      setStatus("Enter gene to perturb", true);
      return false;
    }
    setStatus("Running perturbation (may take a while)…");
    try {
      const res = await withMetaProgressPoll(
        fetch(apiUrl("/api/perturb/preview"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            gene: gene.trim(),
            desired_expr: Number.isFinite(desired) ? desired : 0,
            scope,
            n_propagation: nProp,
          }),
        }).then(async (rr) => {
          if (!rr.ok) throw new Error(await rr.text());
          return rr;
        }),
      );
      const buf = await res.arrayBuffer();
      activeValues = new Float32Array(buf);
      if (activeValues.length !== n) {
        throw new Error(`length ${activeValues.length} != n_obs ${n}`);
      }
      lastColorSource = "perturb";
      perturbDisplayGene = gene.trim();
      cmapSel.value = "diverging";
      refreshVisualization();
      return true;
    } catch (e) {
      setStatus(String(e), true);
      return false;
    }
  }

  async function initDataset(metaOverride?: Meta): Promise<boolean> {
    datasetHotReloadLock = true;
    try {
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
    recvLigandLabel = "";
    colorSource.value = "expression";
    exprGene.value = "";
    betaCol.innerHTML = '<option value="">—</option>';
    interactionSenderIndex = null;
    interactionNeighborSet.clear();
    interactionLineData.length = 0;
    quiverFieldCache = null;
    quiverSegData.length = 0;
    sigQuiverFieldCache = null;
    sigQuiverSegData.length = 0;
    signaturePerCellCache = null;
    transUmapOnlyHint.classList.add("hidden");
    interactionBodyEl.innerHTML = "";
    interactionLensEl.checked = false;
    pairLrToggle.checked = false;
    pairCellA = null;
    pairCellB = null;
    pairNeighborSet.clear();
    pairLrRows = [];
    pairLrBars.innerHTML = "";
    pairLrFootnote.textContent = "";
    pairLrStatus.textContent = "";
    pairLrChartTitle.textContent = "Top ligand–receptor β for pair";
    lastCccInteractions = [];
    cccBars.innerHTML = "";
    cccFootnote.textContent = "";
    cccChartTitle.textContent = "Top interactions";
    syncCccChartVisibility();
    setStatus("Loading dataset…");
    try {
      if (metaOverride) meta = metaOverride;
      else {
        const mr = await fetch(apiUrl("/api/meta"));
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
    applyRecvLigDefaultsFromMeta();

    cellCategories = meta.cell_type_categories ?? [];
    cellTypeColumnLabel = meta.cell_type_column ?? null;
    cellTypeCodes = null;
    typeFilterChecked = cellCategories.map(() => true);

    cellTypePanel.classList.add("hidden");
    cellTypeFilters.innerHTML = "";
    perturbCellType.innerHTML = '<option value="">—</option>';

    if (meta.cell_type_column) {
      try {
        const cr = await fetch(apiUrl("/api/cell_type/codes"));
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
      cccCellType.innerHTML = cellCategories
        .map(
          (name, idx) =>
            `<option value="${idx}">${escapeHtml(name)}</option>`,
        )
        .join("");
      if (cccCellType.options.length > 0) cccCellType.selectedIndex = 0;
    } else {
      cccCellType.innerHTML = '<option value="">—</option>';
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
      syncPerturbPanelsFromMeta();
      interactionPanel.classList.add("hidden");
      cccInteractionsPanel.classList.add("hidden");
      pairLrPanel.classList.add("hidden");
      colorSourceBetaOpt.classList.add("hidden");
      colorSourcePerturbOpt.classList.add("hidden");
      setStatus(
        "No dataset loaded — set .h5ad (and optional run TOML) under Dataset paths, then Load dataset.",
      );
      sessionPanel.open = true;
      lastSyncedDatasetSignature = metaDatasetSignature(meta);
      lastCellTypeSig = cellTypeSignature(meta);
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
    cccInteractionsPanel.classList.toggle("hidden", !hasBetadata);
    pairLrPanel.classList.toggle("hidden", !hasBetadata);
    colorSourcePerturbOpt.classList.toggle("hidden", !meta.perturb_ready);
    syncPerturbPanelsFromMeta();
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

    baseColors = new Uint8ClampedArray(n * 4);
    colors = new Uint8ClampedArray(n * 4);
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
      } satisfies OrthographicViewState,
      viewState: null,
      controller: ORTHO_CONTROLLER,
      getCursor: ({ isDragging, isHovering }) =>
        isDragging ? "grabbing" : isHovering ? "pointer" : "grab",
      layers: [],
      onClick: (info) => {
        const idx = info.index;
        if (idx == null || idx < 0 || idx >= n) return;
        if (!cellSelectableByType(idx)) return;

        if (pairLrToggle.checked && meta.betadata_row_id) {
          if (layoutModeEl.value !== "spatial") {
            setStatus(
              "Pair L–R β: switch layout to Spatial (neighbor radius uses tissue coordinates).",
              true,
            );
            return;
          }

          if (pairCellB !== null) {
            pairCellA = idx;
            pairCellB = null;
            pairLrRows = [];
            recomputePairNeighborSetFromA();
            pairLrStatus.textContent = `Cell A = #${idx} (${pairNeighborSet.size} within radius). Pick cell B.`;
            renderPairLrBarChart();
            rebuildLayer();
            return;
          }

          if (pairCellA === null) {
            pairCellA = idx;
            pairCellB = null;
            pairLrRows = [];
            recomputePairNeighborSetFromA();
            pairLrStatus.textContent = `Cell A = #${idx} (${pairNeighborSet.size} neighbors in radius). Click a neighbor for cell B.`;
            renderPairLrBarChart();
            rebuildLayer();
            return;
          }

          if (idx === pairCellA) {
            clearPairLrSelection(false);
            pairLrStatus.textContent = "Cleared. Click cell A to start again.";
            renderPairLrBarChart();
            return;
          }

          if (!pairNeighborSet.has(idx)) {
            setStatus(
              "Pair L–R: choose a cell in the pink neighbor halo (or increase radius).",
              true,
            );
            return;
          }

          const a = pairCellA;
          pairCellB = idx;
          pairLrStatus.textContent = `Cells #${a} & #${idx} — loading β…`;
          rebuildLayer();
          void fetchPairLrForPair(a, idx).then(() => rebuildLayer());
          return;
        }

        if (!interactionLensEl.checked || !meta.network_loaded) return;
        interactionSenderIndex = idx;
        void fetchAndApplyInteractionContext(idx);
      },
    });

    try {
      const br = await fetch(apiUrl("/api/betadata/genes"));
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
    lastSyncedDatasetSignature = metaDatasetSignature(meta);
    lastCellTypeSig = cellTypeSignature(meta);
    return true;
    } finally {
      datasetHotReloadLock = false;
    }
  }

  async function refreshCellTypeInfo(m: Meta) {
    cellCategories = m.cell_type_categories ?? [];
    cellTypeColumnLabel = m.cell_type_column ?? null;
    cellTypeCodes = null;
    typeFilterChecked = cellCategories.map(() => true);

    cellTypePanel.classList.add("hidden");
    cellTypeFilters.innerHTML = "";
    perturbCellType.innerHTML = '<option value="">—</option>';

    if (m.cell_type_column) {
      try {
        const cr = await fetch(apiUrl("/api/cell_type/codes"));
        if (cr.ok) {
          const buf = await cr.arrayBuffer();
          const arr = new Uint16Array(buf);
          if (arr.length === n) cellTypeCodes = arr;
        }
      } catch { /* optional */ }
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
      cccCellType.innerHTML = cellCategories
        .map(
          (name, idx) =>
            `<option value="${idx}">${escapeHtml(name)}</option>`,
        )
        .join("");
      if (cccCellType.options.length > 0) cccCellType.selectedIndex = 0;
    } else {
      cccCellType.innerHTML = '<option value="">—</option>';
    }
    refillTransHighlightTypes();
    transLimitWrap.classList.toggle(
      "hidden",
      !m.cell_type_column || cellCategories.length === 0,
    );

    cellTypeOverlayEl.checked = !!(cellTypeCodes && cellCategories.length > 0);

    meta = m;
    lastCellTypeSig = cellTypeSignature(m);
    refreshVisualization();
  }

  async function pollServerDatasetIfChanged() {
    if (datasetHotReloadLock) return;
    try {
      const mr = await fetch(apiUrl("/api/meta"));
      if (!mr.ok) return;
      const m = (await mr.json()) as Meta;
      const sig = metaDatasetSignature(m);
      if (sig !== lastSyncedDatasetSignature) {
        setStatus("Dataset changed on server — refreshing…");
        await initDataset(m);
        lastSyncedDatasetSignature = metaDatasetSignature(m);
        lastCellTypeSig = cellTypeSignature(m);
        return;
      }
      const ctSig = cellTypeSignature(m);
      if (ctSig !== lastCellTypeSig) {
        setStatus("Cell type labels updated — refreshing…");
        await refreshCellTypeInfo(m);
      }
    } catch {
      /* ignore */
    }
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
      } satisfies OrthographicViewState,
    });
  }

  function applyLayoutChoice(wantUmap: boolean) {
    if (wantUmap && !positionsUmap) return;
    layoutModeEl.value = wantUmap ? "umap" : "spatial";
    positions = wantUmap ? positionsUmap! : positionsSpatial;
    const b =
      wantUmap && meta.umap_bounds ? meta.umap_bounds : meta.bounds;
    fitDeckToBounds(b);
    rebuildLayer();
    if (
      meta.network_loaded &&
      interactionLensEl.checked &&
      interactionSenderIndex !== null
    ) {
      void fetchAndApplyInteractionContext(interactionSenderIndex);
    }
  }

  layoutModeEl.addEventListener("change", () => {
    if (!positionsUmap) return;
    applyLayoutChoice(layoutModeEl.value === "umap");
  });

  interactionLensEl.addEventListener("change", () => {
    if (interactionLensEl.checked) {
      pairLrToggle.checked = false;
      clearPairLrSelection(false);
    }
    if (!interactionLensEl.checked) {
      clearInteractionContextFull();
    } else {
      syncInteractionFromSender();
    }
  });

  pairLrToggle.addEventListener("change", () => {
    if (pairLrToggle.checked) {
      interactionLensEl.checked = false;
      clearInteractionContextFull();
      pairLrStatus.textContent =
        "Spatial layout: click cell A, then a pink-highlighted neighbor.";
      renderPairLrBarChart();
    } else {
      clearPairLrSelection(false);
    }
    rebuildLayer();
  });

  pairLrClearBtn.addEventListener("click", () => {
    clearPairLrSelection(false);
    if (pairLrToggle.checked) {
      pairLrStatus.textContent =
        "Click cell A, then a neighbor within the radius.";
    }
    renderPairLrBarChart();
  });

  pairLrRadius.addEventListener("input", () => {
    if (!pairLrToggle.checked || pairCellA === null) return;
    recomputePairNeighborSetFromA();
    pairLrStatus.textContent = `Cell A = #${pairCellA} (${pairNeighborSet.size} neighbors in radius).`;
    rebuildLayer();
  });

  pairLrTopK.addEventListener("change", () => renderPairLrBarChart());
  refreshContextBtn.addEventListener("click", () => {
    syncInteractionFromSender();
  });

  interactionModeSel.addEventListener("change", () => {
    syncInteractionPanelLayout();
    if (interactionLensEl.checked) {
      syncInteractionFromSender();
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
      const r = await fetch(apiUrl("/api/session/configure"), {
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
          apiUrl(
            `/api/genes?prefix=${encodeURIComponent(p)}&limit=40`,
          ),
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
      const r = await fetch(
        apiUrl(`/api/betadata/columns?gene=${encodeURIComponent(g)}`),
      );
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
          name?: string;
        };
      }
    | { ok: false; msg: string } {
    const scopeVal = perturbScope.value;
    if (scopeVal === "all") {
      return { ok: true, scope: { type: "all" } };
    }
    if (scopeVal === "cell_type") {
      const cat = Number(perturbCellType.value);
      if (!Number.isFinite(cat) || perturbCellType.value === "") {
        return { ok: false, msg: "Pick a cell type" };
      }
      const lab = cellCategories[cat];
      if (!lab?.trim()) {
        return { ok: false, msg: "Unknown cell type selection" };
      }
      return { ok: true, scope: { type: "cell_type_name", name: lab } };
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

  function syncSplashNetSliderLabels() {
    splashNetHopsVal.textContent = String(
      Math.min(4, Math.max(0, Math.trunc(Number(splashNetHops.value) || 1))),
    );
    splashNetMaxNodesVal.textContent = String(
      Math.min(64, Math.max(6, Math.trunc(Number(splashNetMaxNodes.value) || 24))),
    );
  }
  splashNetHops.addEventListener("input", syncSplashNetSliderLabels);
  splashNetMaxNodes.addEventListener("input", syncSplashNetSliderLabels);
  syncSplashNetSliderLabels();

  function updateSplashForceFieldsVisibility() {
    const force = splashNetLayout.value === "force";
    splashNetForceFields.classList.toggle("hidden", !force);
  }

  function syncSplashForceLabels() {
    splashNetForceLinkMinVal.textContent = String(
      Math.min(90, Math.max(15, Math.trunc(Number(splashNetForceLinkMin.value) || 36))),
    );
    splashNetForceLinkSpanVal.textContent = String(
      Math.min(250, Math.max(20, Math.trunc(Number(splashNetForceLinkSpan.value) || 120))),
    );
    const str = Math.min(100, Math.max(5, Math.trunc(Number(splashNetForceStrength.value) || 35)));
    splashNetForceStrengthVal.textContent = (str / 100).toFixed(2);
    splashNetForceChargeVal.textContent = String(
      Math.min(500, Math.max(40, Math.trunc(Number(splashNetForceCharge.value) || 220))),
    );
    splashNetForceCollideVal.textContent = String(
      Math.min(40, Math.max(2, Math.trunc(Number(splashNetForceCollide.value) || 14))),
    );
    const ad = Math.min(500, Math.max(80, Math.trunc(Number(splashNetForceAlphaDecay.value) || 228)));
    splashNetForceAlphaDecayVal.textContent = String(ad);
    const vel = Math.min(85, Math.max(15, Math.trunc(Number(splashNetForceVelocity.value) || 40)));
    splashNetForceVelocityVal.textContent = (vel / 100).toFixed(2);
    const dra = Math.min(70, Math.max(10, Math.trunc(Number(splashNetForceDragAlpha.value) || 35)));
    splashNetForceDragAlphaVal.textContent = (dra / 100).toFixed(2);
    splashNetForceLinkIterVal.textContent = String(
      Math.min(8, Math.max(1, Math.trunc(Number(splashNetForceLinkIter.value) || 1))),
    );
    const zm = Math.min(90, Math.max(20, Math.trunc(Number(splashNetForceZoomMin.value) || 35)));
    splashNetForceZoomMinVal.textContent = (zm / 100).toFixed(2);
    const zx = Math.min(600, Math.max(150, Math.trunc(Number(splashNetForceZoomMax.value) || 300)));
    splashNetForceZoomMaxVal.textContent = (zx / 100).toFixed(1);
  }

  function readSplashForceParams(): SplashForceParams {
    const linkMin = Math.min(90, Math.max(15, Math.trunc(Number(splashNetForceLinkMin.value) || 36)));
    const linkSpan = Math.min(250, Math.max(20, Math.trunc(Number(splashNetForceLinkSpan.value) || 120)));
    const str = Math.min(100, Math.max(5, Math.trunc(Number(splashNetForceStrength.value) || 35))) / 100;
    const charge = -Math.min(500, Math.max(40, Math.trunc(Number(splashNetForceCharge.value) || 220)));
    const collide = Math.min(40, Math.max(2, Math.trunc(Number(splashNetForceCollide.value) || 14)));
    const alphaDecay =
      Math.min(500, Math.max(80, Math.trunc(Number(splashNetForceAlphaDecay.value) || 228))) / 10_000;
    const velocityDecay =
      Math.min(85, Math.max(15, Math.trunc(Number(splashNetForceVelocity.value) || 40))) / 100;
    const dragAlphaTarget =
      Math.min(70, Math.max(10, Math.trunc(Number(splashNetForceDragAlpha.value) || 35))) / 100;
    const linkIterations = Math.min(8, Math.max(1, Math.trunc(Number(splashNetForceLinkIter.value) || 1)));
    let zoomScaleMin =
      Math.min(90, Math.max(20, Math.trunc(Number(splashNetForceZoomMin.value) || 35))) / 100;
    let zoomScaleMax =
      Math.min(600, Math.max(150, Math.trunc(Number(splashNetForceZoomMax.value) || 300))) / 100;
    if (zoomScaleMin >= zoomScaleMax) {
      zoomScaleMax = zoomScaleMin + 0.05;
    }
    return {
      linkDistanceMin: linkMin,
      linkDistanceSpan: linkSpan,
      linkStrength: str,
      charge,
      collidePadding: collide,
      alphaDecay,
      velocityDecay,
      dragAlphaTarget,
      linkIterations,
      zoomScaleMin,
      zoomScaleMax,
    };
  }

  const splashForceInputs: HTMLInputElement[] = [
    splashNetForceLinkMin,
    splashNetForceLinkSpan,
    splashNetForceStrength,
    splashNetForceCharge,
    splashNetForceCollide,
    splashNetForceAlphaDecay,
    splashNetForceVelocity,
    splashNetForceDragAlpha,
    splashNetForceLinkIter,
    splashNetForceZoomMin,
    splashNetForceZoomMax,
  ];
  for (const el of splashForceInputs) {
    el.addEventListener("input", () => {
      syncSplashForceLabels();
      if (splashNetLayout.value === "force" && lastSplashNetworkJson) {
        mountSplashNetworkChart(lastSplashNetworkJson);
      }
    });
  }
  syncSplashForceLabels();
  updateSplashForceFieldsVisibility();

  function mountSplashNetworkChart(data: SplashNetworkJson) {
    if (splashNetSimCleanup) {
      splashNetSimCleanup();
      splashNetSimCleanup = null;
    }
    const w = splashNetChart.clientWidth || 560;
    const layout = splashNetLayout.value as SplashNetworkLayout;
    const fullscreen = splashNetWrap.classList.contains(
      "splash-net-wrap--fullscreen",
    );
    const fsCollapsed =
      fullscreen &&
      splashNetWrap.classList.contains("splash-net-controls-collapsed");
    const chromeReserve = fullscreen ? (fsCollapsed ? 52 : 220) : 0;
    const hBase = fullscreen
      ? Math.max(
          400,
          splashNetChart.clientHeight ||
            Math.round(window.innerHeight - chromeReserve),
        )
      : layout === "layered"
        ? Math.max(440, Math.min(920, 32 + data.nodes.length * 11))
        : 420;
    splashNetSimCleanup = renderSplashNetwork(splashNetChart, data, {
      width: w,
      height: hBase,
      layout,
      force: layout === "force" ? readSplashForceParams() : undefined,
    });
  }

  function syncSplashNetControlsUi() {
    const fs = splashNetWrap.classList.contains("splash-net-wrap--fullscreen");
    const collapsed = splashNetWrap.classList.contains(
      "splash-net-controls-collapsed",
    );
    splashNetFullscreenBtn.textContent = fs ? "Exit full screen" : "Full screen";
    splashNetExitFullscreenBtn.textContent = "Exit full screen";
    if (fs) {
      splashNetControlsToggle.textContent = collapsed ? "Show settings" : "Hide settings";
      splashNetControlsToggle.setAttribute(
        "aria-expanded",
        collapsed ? "false" : "true",
      );
    } else {
      splashNetControlsToggle.textContent = "Hide settings";
      splashNetControlsToggle.setAttribute("aria-expanded", "true");
    }
  }

  function setSplashNetFullscreen(on: boolean) {
    splashNetWrap.classList.toggle("splash-net-wrap--fullscreen", on);
    if (on) {
      splashNetWrap.classList.add("splash-net-controls-collapsed");
    } else {
      splashNetWrap.classList.remove("splash-net-controls-collapsed");
    }
    document.body.style.overflow = on ? "hidden" : "";
    syncSplashNetControlsUi();
    if (lastSplashNetworkJson) {
      requestAnimationFrame(() =>
        requestAnimationFrame(() => mountSplashNetworkChart(lastSplashNetworkJson!)),
      );
    }
  }

  splashNetFullscreenBtn.addEventListener("click", () => {
    setSplashNetFullscreen(
      !splashNetWrap.classList.contains("splash-net-wrap--fullscreen"),
    );
  });

  splashNetExitFullscreenBtn.addEventListener("click", () => {
    setSplashNetFullscreen(false);
  });

  splashNetControlsToggle.addEventListener("click", () => {
    if (!splashNetWrap.classList.contains("splash-net-wrap--fullscreen")) return;
    splashNetWrap.classList.toggle("splash-net-controls-collapsed");
    syncSplashNetControlsUi();
    if (lastSplashNetworkJson) {
      requestAnimationFrame(() =>
        requestAnimationFrame(() => mountSplashNetworkChart(lastSplashNetworkJson!)),
      );
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!splashNetWrap.classList.contains("splash-net-wrap--fullscreen")) return;
    setSplashNetFullscreen(false);
  });

  splashNetLayout.addEventListener("change", () => {
    updateSplashForceFieldsVisibility();
    if (lastSplashNetworkJson) mountSplashNetworkChart(lastSplashNetworkJson);
  });

  async function runSplashNetworkFromUi() {
    if (!meta.perturb_ready) {
      setStatus("Splash network needs perturb_ready (--run-toml)", true);
      return;
    }
    const ga = splashNetGeneA.value.trim();
    const gb = splashNetGeneB.value.trim();
    if (!ga || !gb) {
      setStatus("Enter gene A and gene B", true);
      return;
    }
    if (ga === gb) {
      setStatus("Gene A and B must differ", true);
      return;
    }
    const sp = perturbScopePayload();
    if (!sp.ok) {
      setStatus(sp.msg, true);
      return;
    }
    const body: Record<string, unknown> = {
      gene_a: ga,
      gene_b: gb,
      scope: sp.scope,
      surround_hops: Math.min(4, Math.max(0, Math.trunc(Number(splashNetHops.value) || 1))),
      max_nodes: Math.min(64, Math.max(6, Math.trunc(Number(splashNetMaxNodes.value) || 24))),
    };
    setStatus("Computing splash network (all trained targets)…");
    splashNetMessage.classList.add("hidden");
    const pollAc = new AbortController();
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    const stopSplashProgressPoll = () => {
      if (pollTimer != null) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      pollAc.abort();
    };
    const setSplashProgressUi = (permille: number) => {
      const pct = Math.min(100, Math.max(0, Math.round(permille / 10)));
      splashNetProgressFill.style.width = `${pct}%`;
      splashNetProgressLabel.textContent = `${pct}%`;
    };
    splashNetProgressWrap.classList.remove("hidden");
    setSplashProgressUi(0);
    pollTimer = window.setInterval(async () => {
      try {
        const pr = await fetch(apiUrl("/api/perturb/splash_progress"), {
          signal: pollAc.signal,
        });
        if (!pr.ok) return;
        const j = (await pr.json()) as { active: boolean; permille: number };
        setSplashProgressUi(Number(j.permille) || 0);
      } catch {
        /* aborted or transient */
      }
    }, 130);
    try {
      const r = await fetch(apiUrl("/api/perturb/splash_network"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(`${r.status}: ${t}`);
      }
      const data = (await r.json()) as SplashNetworkJson;
      lastSplashNetworkJson = data;
      setSplashProgressUi(1000);
      mountSplashNetworkChart(data);
      if (data.message?.trim()) {
        splashNetMessage.textContent = data.message;
        splashNetMessage.classList.remove("hidden");
      }
      const pathStr =
        data.path_found && data.path?.length
          ? data.path.join(" → ")
          : "no directed path";
      setStatus(
        `Splash network: ${data.n_cells_used} cells · ${data.nodes.length} genes · ${pathStr}`,
        false,
      );
    } catch (e) {
      setStatus(`Splash network: ${e}`, true);
    } finally {
      stopSplashProgressPoll();
      window.setTimeout(() => {
        splashNetProgressWrap.classList.add("hidden");
        splashNetProgressFill.style.width = "0%";
        splashNetProgressLabel.textContent = "0%";
      }, 350);
    }
  }

  splashNetComputeBtn.addEventListener("click", () => {
    void runSplashNetworkFromUi();
  });

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
      } else if (colorSource.value === "received_ligand") {
        const raw = recvLigGenes.value.trim();
        if (!raw) {
          setStatus("Enter ligand gene(s) or one model column name", true);
          return;
        }
        const geneTokens = raw
          .split(/[,;\s]+/)
          .map((s) => s.trim())
          .filter(Boolean);
        const fromModel = recvLigSource.value === "model";
        if (fromModel) {
          if (!meta.perturb_ready) {
            setStatus("Model path needs perturb_ready (--run-toml)", true);
            return;
          }
          const col = geneTokens[0];
          if (!col) {
            setStatus("Enter one received-ligand column name for model", true);
            return;
          }
          activeValues = await postF32("/api/spatial/received_ligand", {
            source: "model",
            genes: [col],
            matrix: recvLigMatrix.value,
          });
          recvLigandLabel = `${recvLigMatrix.value.toUpperCase()}: ${col}`;
        } else {
          if (geneTokens.length === 0) {
            setStatus("Enter at least one ligand gene symbol", true);
            return;
          }
          const radius = Number(recvLigRadius.value);
          const scale = Number(recvLigScale.value);
          const gridFactor = Number(recvLigGridFactor.value);
          const body: Record<string, unknown> = {
            source: "adata",
            genes: geneTokens,
            matrix: recvLigMatrix.value,
            use_grid: recvLigGrid.checked,
            aggregate: recvLigAgg.value,
          };
          if (Number.isFinite(radius) && radius > 0) body.radius = radius;
          if (Number.isFinite(scale)) body.scale_factor = scale;
          if (Number.isFinite(gridFactor) && gridFactor > 0) {
            body.grid_factor = gridFactor;
          }
          activeValues = await postF32(
            "/api/spatial/received_ligand",
            body,
          );
          recvLigandLabel =
            geneTokens.length === 1
              ? `${geneTokens[0]!} (received)`
              : `${recvLigAgg.value}(${geneTokens.join(", ")})`;
        }
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
      if (colorSource.value === "expression") {
        lastColorSource = "expression";
      } else if (colorSource.value === "received_ligand") {
        lastColorSource = "received_ligand";
      } else {
        lastColorSource = "betadata";
      }
      refreshVisualization();
      setStatus(`Loaded ${colorSource.value} (${activeValues.length} values)`);
    } catch (e) {
      setStatus(`Load failed: ${e}`, true);
    }
  };

  if (mcp.openSession?.adata_path) {
    sessionAdataPath.value = mcp.openSession.adata_path;
    sessionLayer.value = mcp.openSession.layer;
    sessionClusterAnnot.value = mcp.openSession.cluster_annot;
    sessionNetworkDir.value = mcp.openSession.network_dir;
    sessionRunToml.value = mcp.openSession.run_toml;
    sessionBusyEl.classList.remove("hidden");
    try {
      const r = await fetch(apiUrl("/api/session/configure"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          adata_path: mcp.openSession.adata_path,
          layer: mcp.openSession.layer.trim() || "imputed_count",
          cluster_annot:
            mcp.openSession.cluster_annot.trim() || "cell_type",
          network_dir: mcp.openSession.network_dir.trim(),
          run_toml: mcp.openSession.run_toml.trim(),
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
      await initDataset();
    } finally {
      sessionBusyEl.classList.add("hidden");
    }
  } else {
    await initDataset();
  }

  window.setInterval(() => void pollServerDatasetIfChanged(), 2000);

  attachMcpControlSink((args) => {
    if (typeof args.status_message === "string" && args.status_message.trim()) {
      setStatus(args.status_message.trim());
    }
    let needColorUiRefresh = false;
    let willAsyncLoad = false;
    const cs = args.color_source;
    if (
      cs === "expression" ||
      cs === "betadata" ||
      cs === "perturb" ||
      cs === "received_ligand"
    ) {
      colorSource.value = cs;
      needColorUiRefresh = true;
    }
    if (
      typeof args.received_ligand_genes === "string" &&
      args.received_ligand_genes.trim()
    ) {
      recvLigGenes.value = args.received_ligand_genes.trim();
      colorSource.value = "received_ligand";
      needColorUiRefresh = true;
    }
    if (
      args.received_ligand_source === "adata" ||
      args.received_ligand_source === "model"
    ) {
      recvLigSource.value = args.received_ligand_source;
      syncRecvLigPanelsFromSource();
      needColorUiRefresh = true;
    }
    if (
      args.received_ligand_matrix === "lr" ||
      args.received_ligand_matrix === "tfl"
    ) {
      recvLigMatrix.value = args.received_ligand_matrix;
      needColorUiRefresh = true;
    }
    const rlr = args.received_ligand_radius;
    if (typeof rlr === "number" && Number.isFinite(rlr) && rlr > 0) {
      recvLigRadius.value = String(rlr);
      needColorUiRefresh = true;
    }
    const rls = args.received_ligand_scale;
    if (typeof rls === "number" && Number.isFinite(rls)) {
      recvLigScale.value = String(rls);
      needColorUiRefresh = true;
    }
    if (typeof args.received_ligand_use_grid === "boolean") {
      recvLigGrid.checked = args.received_ligand_use_grid;
      needColorUiRefresh = true;
    }
    const rgf = args.received_ligand_grid_factor;
    if (typeof rgf === "number" && Number.isFinite(rgf) && rgf > 0) {
      recvLigGridFactor.value = String(rgf);
      needColorUiRefresh = true;
    }
    if (
      args.received_ligand_aggregate === "sum" ||
      args.received_ligand_aggregate === "max" ||
      args.received_ligand_aggregate === "mean"
    ) {
      recvLigAgg.value = args.received_ligand_aggregate;
      needColorUiRefresh = true;
    }
    if (
      typeof args.expression_gene === "string" &&
      args.expression_gene.trim()
    ) {
      exprGene.value = args.expression_gene.trim();
      colorSource.value = "expression";
      needColorUiRefresh = true;
      if (args.apply_expression === true) {
        willAsyncLoad = true;
        void loadActiveChannel();
      }
    } else if (args.apply_expression === true) {
      willAsyncLoad = true;
      void loadActiveChannel();
    }
    if (args.apply_received_ligand === true) {
      willAsyncLoad = true;
      void loadActiveChannel();
    }
    if (
      needColorUiRefresh ||
      args.apply_received_ligand === true ||
      args.apply_expression === true
    ) {
      syncColorModeUi();
    }
    if (!willAsyncLoad && needColorUiRefresh) refreshVisualization();
    if (typeof args.focus_gene_context === "string") {
      focusGeneCtx.value = args.focus_gene_context.trim();
    }
  });

  async function runMcpCaptureRender(req: McpCaptureRequest) {
    const app = mcp.mcpApp;
    if (!app) return;
    try {
      if (!deck || n === 0) {
        await app.sendMessage({
          role: "user",
          content: [
            {
              type: "text",
              text: "[Spatial viewer] No deck to capture — load a dataset in the viewer first.",
            },
          ],
        });
        setStatus("Capture skipped — load a dataset first", true);
        return;
      }
      rebuildLayer();
      const redraw = (
        deck as { redraw?: (reason?: string) => void }
      ).redraw;
      redraw?.("mcp-capture");
      await waitAnimationFrames(3);
      const canvas = deckContainer.querySelector("canvas");
      if (!canvas) throw new Error("WebGL canvas not found under #deck-root");
      const b64 = canvasToScaledPngBase64(canvas, req.max_width);
      const layout = layoutModeEl.value === "umap" ? "UMAP" : "spatial";
      const quiverHint =
        layoutModeEl.value === "umap" &&
        (quiverSegData.length > 0 || sigQuiverSegData.length > 0)
          ? `\n- UMAP quiver segments: perturb ${quiverSegData.length} · signature ${sigQuiverSegData.length}`
          : "";
      const text = [
        "**Spatial viewer render (PNG)**",
        req.caption.trim() ? `Note: ${req.caption.trim()}` : null,
        `- Layout: ${layout}`,
        `- n_obs=${meta.n_obs} · color mode: ${colorSource.value}`,
        colorSource.value === "expression"
          ? `- Expression gene: ${exprGene.value.trim() || "—"}`
          : null,
        colorSource.value === "betadata"
          ? `- Betadata: ${betaGene.value || "—"} / ${betaCol.value || "—"}`
          : null,
        colorSource.value === "received_ligand"
          ? `- Received ligand: ${recvLigandLabel || recvLigGenes.value.trim() || "—"} (${recvLigSource.value})`
          : null,
        colorSource.value === "perturb"
          ? `- Perturb gene: ${perturbDisplayGene || perturbGene.value.trim() || "—"}`
          : null,
        lastColorSource === "gene_signature"
          ? `- Gene signature coloring: ${sigUmapGenes.value.trim() || "—"}`
          : null,
        quiverHint.trim() ? quiverHint : null,
        `- Dataset path: ${meta.adata_path || "—"}`,
      ]
        .filter((x): x is string => typeof x === "string" && x.length > 0)
        .join("\n");
      const blocks = [
        { type: "text" as const, text },
        { type: "image" as const, data: b64, mimeType: "image/png" },
      ];
      try {
        await app.updateModelContext({ content: blocks });
      } catch {
        /* host may not support context images */
      }
      const sm = await app.sendMessage({ role: "user", content: blocks });
      if (sm.isError) {
        setStatus("Host did not accept screenshot (ui/message)", true);
      } else {
        setStatus("Screenshot sent to chat for the assistant");
      }
    } catch (e) {
      const msg = `[Spatial viewer] Capture failed: ${e}`;
      setStatus(msg, true);
      try {
        await mcp.mcpApp?.sendMessage({
          role: "user",
          content: [{ type: "text", text: msg }],
        });
      } catch {
        /* ignore */
      }
    }
  }

  attachMcpCaptureSink((req) => void runMcpCaptureRender(req));

  function buildUmapTransitionBodyFromUi(
    gene: string,
    desired: number,
    scope: Record<string, unknown>,
    nProp: number,
  ) {
    return {
      gene,
      desired_expr: Number.isFinite(desired) ? desired : 0,
      scope,
      n_propagation: nProp,
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
      export_svg: true,
    };
  }

  async function fetchUmapFieldAndApply(
    body: ReturnType<typeof buildUmapTransitionBodyFromUi>,
    statusBusy: string,
    statusOk: (
      nx: number,
      ny: number,
      nArrows: number,
      svgPath?: string | null,
    ) => string,
  ): Promise<boolean> {
    setStatus(statusBusy);
    try {
      const r = await withMetaProgressPoll(
        fetch(apiUrl("/api/perturb/umap-field"), {
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
      setStatus(
        statusOk(nx, ny, nArrows, data.svg_export_path ?? null),
      );
      return true;
    } catch (e) {
      quiverFieldCache = null;
      quiverSegData.length = 0;
      rebuildLayer();
      setStatus(String(e), true);
      return false;
    }
  }

  async function computeSignatureUmapField(): Promise<boolean> {
    if (!meta.umap_obsm_key) {
      setStatus("No UMAP embedding in this dataset", true);
      return false;
    }
    const raw = sigUmapGenes.value.trim();
    if (!raw) {
      setStatus("Enter signature genes (comma-separated)", true);
      return false;
    }
    const genes = raw.split(/[,;\s]+/).map((s) => s.trim()).filter(Boolean);
    if (genes.length === 0) {
      setStatus("Enter at least one gene symbol", true);
      return false;
    }
    const mask = sigUmapMaskPerturb.checked;
    if (mask) {
      if (meta.perturb_error) {
        setStatus(`Perturbation unavailable: ${meta.perturb_error}`, true);
        return false;
      }
      if (meta.perturb_loading) {
        setStatus("Wait for perturbation engine to finish loading", true);
        return false;
      }
      if (!meta.perturb_ready) {
        setStatus("Mask requires perturb_ready (--run-toml)", true);
        return false;
      }
      if (!perturbGene.value.trim()) {
        setStatus("Mask: enter perturb gene in the row above", true);
        return false;
      }
    }
    const body: Record<string, unknown> = {
      genes,
      n_knn: Math.min(200, Math.max(3, Math.trunc(Number(sigUmapKnn.value) || 30))),
      grid_scale: Number(transGridScale.value) || 1,
      vector_scale: Number(transVecScale.value) || 0.85,
      magnitude_threshold: Math.max(0, Number(transMagThresh.value) || 0),
      gradient_gain: 2,
      mask_with_perturb_quiver: mask,
      mask_quick_ko: transQuickKo.checked,
      export_svg: true,
    };
    if (mask) {
      const sp = perturbScopePayload();
      if (!sp.ok) {
        setStatus(sp.msg, true);
        return false;
      }
      body.mask_perturb = {
        gene: perturbGene.value.trim(),
        desired_expr: Number.isFinite(Number(perturbExpr.value))
          ? Number(perturbExpr.value)
          : 0,
        scope: sp.scope,
        n_propagation: readUiNPropagation(),
      };
    }
    setStatus("Computing UMAP gene signature field…");
    try {
      const r = await fetch(apiUrl("/api/umap/signature_field"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as UmapSignatureFieldResponse;
      const nx = data.nx;
      const ny = data.ny;
      if (data.u.length !== nx * ny || data.v.length !== nx * ny) {
        throw new Error("signature quiver length mismatch");
      }
      if (data.signature_per_cell.length !== n) {
        throw new Error(
          `signature_per_cell length ${data.signature_per_cell.length} != n_obs ${n}`,
        );
      }
      sigQuiverFieldCache = data;
      signaturePerCellCache = new Float32Array(data.signature_per_cell);
      const nArrows = rebuildSignatureQuiverFromCache();
      syncQuiverDisplayLabels();
      rebuildLayer();
      let msg = `UMAP signature quiver: ${nArrows} arrows (${nx}×${ny}) · ${genes.slice(0, 5).join(", ")}${genes.length > 5 ? "…" : ""}`;
      if (data.svg_export_path) msg += ` · SVG → ${data.svg_export_path}`;
      setStatus(msg);
      return true;
    } catch (e) {
      sigQuiverFieldCache = null;
      sigQuiverSegData.length = 0;
      signaturePerCellCache = null;
      rebuildLayer();
      setStatus(String(e), true);
      return false;
    }
  }

  async function runUmapQuiverAfterMcpPerturb(
    gene: string,
    desired: number,
    scope: Record<string, unknown>,
    nProp: number,
  ): Promise<boolean> {
    if (meta.perturb_error) {
      setStatus(`Perturbation unavailable: ${meta.perturb_error}`, true);
      return false;
    }
    if (meta.perturb_loading) {
      setStatus(
        "Perturbation engine is still loading; wait for “perturbation ready”.",
        true,
      );
      return false;
    }
    if (!meta.perturb_ready) {
      setStatus("Perturbation needs server --run-toml", true);
      return false;
    }
    if (!meta.umap_obsm_key) {
      setStatus("No UMAP embedding in this dataset", true);
      return false;
    }
    if (positionsUmap) {
      applyLayoutChoice(true);
    }
    const body = buildUmapTransitionBodyFromUi(gene, desired, scope, nProp);
    const quick = transQuickKo.checked;
    return fetchUmapFieldAndApply(
      body,
      quick
        ? "Computing UMAP quiver (MCP, quick single-gene δ)…"
        : "Computing UMAP transition field (MCP)…",
      (nx, ny, nArrows, svgPath) => {
        const mode = quick ? "quick δ" : "full GRN";
        let msg = `UMAP quiver (${mode}, MCP): ${nArrows} arrows (${nx}×${ny} grid, ${quiverSegData.length} segments)`;
        if (svgPath) msg += ` · SVG → ${svgPath}`;
        return msg;
      },
    );
  }

  function mcpScopeToPerturbBody(
    req: McpPerturbRunRequest,
  ):
    | {
        ok: true;
        scope: Record<string, unknown>;
      }
    | { ok: false; msg: string } {
    const sc = req.scope ?? "all";
    if (sc === "all") {
      return { ok: true, scope: { type: "all" } };
    }
    if (sc === "cell_type") {
      const lab = req.cell_type_label.trim();
      if (!lab) {
        return { ok: false, msg: "cell_type_label required when scope=cell_type" };
      }
      if (!cellCategories.some((c) => c === lab)) {
        return {
          ok: false,
          msg: `Unknown cell_type_label ${JSON.stringify(lab)} (not in dataset categories)`,
        };
      }
      return { ok: true, scope: { type: "cell_type_name", name: lab } };
    }
    if (sc === "cluster") {
      return {
        ok: true,
        scope: { type: "cluster", cluster_id: Math.trunc(req.cluster_id) },
      };
    }
    if (interactionSenderIndex === null) {
      return {
        ok: false,
        msg:
          "scope=selection requires a clicked sender cell (enable Interaction lens and click a cell)",
      };
    }
    return {
      ok: true,
      scope: { type: "indices", indices: [interactionSenderIndex] },
    };
  }

  async function runMcpPerturbFromBridge(req: McpPerturbRunRequest) {
    const sp = mcpScopeToPerturbBody(req);
    if (!sp.ok) {
      setStatus(sp.msg, true);
      if (mcp.mcpApp?.sendMessage && req.push_summary_to_chat) {
        try {
          await mcp.mcpApp.sendMessage({
            role: "user",
            content: [
              { type: "text", text: `[Spatial viewer] Perturb skipped: ${sp.msg}` },
            ],
          });
        } catch {
          /* ignore */
        }
      }
      return;
    }
    const nProp = req.n_propagation ?? readUiNPropagation();
    const ok = await executePerturbPreview(
      req.gene,
      req.desired_expr ?? 0,
      sp.scope,
      nProp,
    );
    if (ok && req.run_umap_quiver) {
      await runUmapQuiverAfterMcpPerturb(
        req.gene.trim(),
        req.desired_expr ?? 0,
        sp.scope,
        nProp,
      );
    }
    if (
      ok &&
      req.push_summary_to_chat &&
      mcp.mcpApp &&
      activeValues &&
      activeValues.length > 0
    ) {
      const arr = activeValues;
      let mn = arr[0]!;
      let mx = arr[0]!;
      let s = 0;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i]!;
        mn = Math.min(mn, v);
        mx = Math.max(mx, v);
        s += v;
      }
      const mean = s / arr.length;
      try {
        await mcp.mcpApp.sendMessage({
          role: "user",
          content: [
            {
              type: "text",
              text: [
                `**Perturbation Δ** (${req.gene})`,
                `- min: ${mn}`,
                `- max: ${mx}`,
                `- mean: ${mean}`,
                `- n_cells: ${arr.length}`,
              ].join("\n"),
            },
          ],
        });
      } catch {
        /* ignore */
      }
    }
  }

  async function runMcpCollectFromBridge(req: McpCollectInteractionsRequest) {
    let cellTypeLabel = req.cell_type.trim();
    if (req.filter === "cell_type" && !cellTypeLabel && cellCategories.length > 0) {
      cellTypeLabel = cellCategories[0] ?? "";
    }
    if (req.filter === "cell_type" && !cellTypeLabel) {
      const msg = "MCP collect_interactions: cell_type label required";
      setStatus(msg, true);
      if (mcp.mcpApp?.sendMessage && req.push_summary_to_chat) {
        try {
          await mcp.mcpApp.sendMessage({
            role: "user",
            content: [{ type: "text", text: `[Spatial viewer] ${msg}` }],
          });
        } catch {
          /* ignore */
        }
      }
      return;
    }
    await runCollectInteractionsApi({
      aggregate: req.aggregate,
      filter: req.filter,
      cell_type: req.filter === "cell_type" ? cellTypeLabel : undefined,
      cluster_id: req.filter === "cluster" ? req.cluster_id : undefined,
      max_genes: req.max_genes,
      push_summary_to_chat: req.push_summary_to_chat,
    });
  }

  function applyReceivedLigandRequestFromMcp(req: McpReceivedLigandRequest) {
    recvLigGenes.value = req.genes.join(", ");
    recvLigSource.value = req.source;
    recvLigMatrix.value = req.matrix;
    if (req.radius != null && Number.isFinite(req.radius) && req.radius > 0) {
      recvLigRadius.value = String(req.radius);
    }
    if (req.scale_factor != null && Number.isFinite(req.scale_factor)) {
      recvLigScale.value = String(req.scale_factor);
    }
    if (typeof req.use_grid === "boolean") {
      recvLigGrid.checked = req.use_grid;
    }
    if (
      req.grid_factor != null &&
      Number.isFinite(req.grid_factor) &&
      req.grid_factor > 0
    ) {
      recvLigGridFactor.value = String(req.grid_factor);
    }
    if (
      req.aggregate === "sum" ||
      req.aggregate === "max" ||
      req.aggregate === "mean"
    ) {
      recvLigAgg.value = req.aggregate;
    }
    syncRecvLigPanelsFromSource();
    colorSource.value = "received_ligand";
    syncColorModeUi();
  }

  function runMcpReceivedLigandFromBridge(req: McpReceivedLigandRequest) {
    applyReceivedLigandRequestFromMcp(req);
    void loadActiveChannel();
  }

  attachMcpPerturbRunSink((req) => void runMcpPerturbFromBridge(req));
  attachMcpCollectInteractionsSink((req) =>
    void runMcpCollectFromBridge(req),
  );
  attachMcpReceivedLigandSink((req) =>
    void runMcpReceivedLigandFromBridge(req),
  );
  attachMcpSignatureUmapSink((req: McpSignatureUmapRequest) => {
    sigUmapGenes.value = req.genes.join(", ");
    if (req.n_knn != null) sigUmapKnn.value = String(req.n_knn);
    sigUmapMaskPerturb.checked = req.mask_with_perturb_quiver === true;
    void computeSignatureUmapField();
  });
  attachMcpSplashNetworkSink((req: McpSplashNetworkRequest) => {
    splashNetGeneA.value = req.gene_a;
    splashNetGeneB.value = req.gene_b;
    if (req.surround_hops != null) {
      splashNetHops.value = String(req.surround_hops);
    }
    if (req.max_nodes != null) {
      splashNetMaxNodes.value = String(req.max_nodes);
    }
    syncSplashNetSliderLabels();
    if (req.scope === "all") {
      perturbScope.value = "all";
    } else if (req.scope === "cell_type" && req.cell_type_label) {
      perturbScope.value = "cell_type";
      const i = cellCategories.findIndex(
        (c) => c.trim() === req.cell_type_label!.trim(),
      );
      if (i >= 0) perturbCellType.value = String(i);
    } else if (req.scope === "cluster" && req.cluster_id != null) {
      perturbScope.value = "cluster";
      perturbClusterId.value = String(req.cluster_id);
    }
    syncPerturbScopeFields();
    void runSplashNetworkFromUi();
  });

  mcpReportContextBtn.addEventListener("click", async () => {
    if (!mcp.mcpApp) return;
    const lines = [
      "**Spatial viewer**",
      `- Dataset: ${meta.adata_path} (${meta.n_obs} cells)`,
      `- Color source: ${colorSource.value}`,
      `- Expression gene: ${exprGene.value.trim() || "—"}`,
      `- Betadata: ${betaGene.value || "—"} / ${betaCol.value || "—"}`,
      `- Received ligand: ${recvLigGenes.value.trim() || "—"} (${recvLigSource.value}) · ${recvLigandLabel || "—"}`,
      `- UMAP signature genes: ${sigUmapGenes.value.trim() || "—"} · quiver loaded: ${sigQuiverFieldCache ? "yes" : "no"}`,
      `- Splash network: ${splashNetGeneA.value.trim() || "—"} → ${splashNetGeneB.value.trim() || "—"}`,
    ];
    if (interactionSenderIndex !== null) {
      lines.push(`- Interaction sender cell index: ${interactionSenderIndex}`);
    }
    const summary = lines.join("\n");
    try {
      setStatus("Sending context to chat…");
      await mcp.mcpApp.callServerTool({
        name: "spatial_viewer_report_context",
        arguments: { summary },
      });
      setStatus("Context sent to chat");
    } catch (e) {
      setStatus(String(e), true);
    }
  });

  async function runPerturbFromUi() {
    const gene = perturbGene.value.trim();
    const desired = Number(perturbExpr.value);
    const sp = perturbScopePayload();
    if (!sp.ok) {
      setStatus(sp.msg, true);
      return;
    }
    const scopeVal = perturbScope.value;
    const ok = await executePerturbPreview(
      gene,
      desired,
      sp.scope,
      readUiNPropagation(),
    );
    if (ok) setStatus(`Perturbation Δ · ${gene} · ${scopeVal}`);
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
    const body = buildUmapTransitionBodyFromUi(
      gene,
      desired,
      sp.scope as Record<string, unknown>,
      readUiNPropagation(),
    );
    const quick = transQuickKo.checked;
    await fetchUmapFieldAndApply(
      body,
      quick
        ? "Computing UMAP quiver (quick single-gene δ)…"
        : "Computing UMAP transition field…",
      (nx, ny, nArrows, svgPath) => {
        const mode = quick ? "quick δ" : "full GRN";
        let msg = `UMAP quiver (${mode}): ${nArrows} arrows (${nx}×${ny} grid, ${quiverSegData.length} segments)`;
        if (svgPath) msg += ` · SVG → ${svgPath}`;
        return msg;
      },
    );
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

  computeSigUmapBtn.addEventListener("click", () =>
    void computeSignatureUmapField(),
  );
  clearSigUmapBtn.addEventListener("click", () => {
    sigQuiverFieldCache = null;
    sigQuiverSegData.length = 0;
    signaturePerCellCache = null;
    if (lastColorSource === "gene_signature") {
      activeValues = null;
      lastColorSource = null;
      refreshVisualization();
    }
    rebuildLayer();
    setStatus("Cleared UMAP signature quiver");
  });
  colorBySigUmapBtn.addEventListener("click", () => {
    if (!signaturePerCellCache || signaturePerCellCache.length !== n) {
      setStatus("Compute signature quiver first (loads per-cell Σ expression)", true);
      return;
    }
    activeValues = signaturePerCellCache;
    lastColorSource = "gene_signature";
    refreshVisualization();
    setStatus("Cell colors: gene signature (sum of genes in layer)");
  });

  function onQuiverDisplayInput() {
    syncQuiverDisplayLabels();
    let touched = false;
    if (quiverFieldCache) {
      rebuildQuiverFromCache();
      touched = true;
    }
    if (sigQuiverFieldCache) {
      rebuildSignatureQuiverFromCache();
      touched = true;
    }
    if (touched) rebuildLayer();
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
        fetch(apiUrl("/api/perturb/summary"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            gene,
            desired_expr: Number.isFinite(desired) ? desired : 0,
            scope: sp.scope,
            n_propagation: readUiNPropagation(),
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

  cancelJobsBtn.addEventListener("click", async () => {
    try {
      const r = await fetch(apiUrl("/api/cancel"), { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
      setStatus("Cancel sent — perturb jobs stop at next iteration; UI loading cleared.");
      schedulePerturbMetaPoll();
    } catch (e) {
      setStatus(String(e), true);
    }
  });
}

main().catch((e) => {
  console.error(e);
  document.body.textContent = String(e);
});

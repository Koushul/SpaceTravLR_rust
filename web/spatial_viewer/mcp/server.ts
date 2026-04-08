import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  registerAppResource,
  registerAppTool,
  RESOURCE_MIME_TYPE,
} from "@modelcontextprotocol/ext-apps/server";
import { z } from "zod";

import {
  buildPerturbScopeApiBody,
  buildPerturbScopeBody,
  buildReferenceCentroidScopeBody,
  fetchMetaWith,
  formatStatus,
  makeConnectDomainList,
  normalizeApiBase,
  parseDelimitedGenes,
  type MetaSnapshot,
} from "./mcpHttp.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, "..");
const mcpHtmlPath = path.join(rootDir, "dist", "mcp-app.html");

const RESOURCE_URI = "ui://spacetravlr/spatial-viewer.html";

export type SpatialMcpDeps = {
  fetch: typeof fetch;
  defaultApiBase: string;
  connectDomainList: () => string[];
  readMcpHtml: () => Promise<string>;
};

const openInputSchema = {
  adata_path: z
    .string()
    .describe("Path to the .h5ad file (same machine as spatial_viewer server)"),
  layer: z.string().optional().describe("AnnData layer (default imputed_count)"),
  cluster_annot: z
    .string()
    .optional()
    .describe("obs column for clusters (default cell_type)"),
  network_dir: z
    .string()
    .optional()
    .describe("Directory with GRN parquet files (optional)"),
  run_toml: z
    .string()
    .optional()
    .describe("spacetravlr_run_repro.toml for betadata/perturb (optional)"),
  perturb_overlay: z
    .string()
    .optional()
    .describe(
      "Optional TOML merged into the run TOML when loading perturb runtime (same idea as spacetravlr-perturb --config).",
    ),
  api_base_url: z
    .string()
    .optional()
    .describe(
      "Base URL of spatial_viewer HTTP API (default SPATIAL_VIEWER_API_BASE or http://127.0.0.1:8080). Start Rust with --allow-cors when using MCP iframe.",
    ),
};

const captureInputSchema = {
  max_width: z
    .number()
    .int()
    .positive()
    .max(4096)
    .optional()
    .describe("Max image width in pixels (height scaled); omit for native resolution."),
  caption: z
    .string()
    .optional()
    .describe("Short note for the assistant (shown next to the image)."),
};

const perturbRunInputSchema = {
  gene: z.string().describe("Gene symbol to perturb (must be in model var_names)"),
  desired_expr: z
    .number()
    .optional()
    .describe("Target expression after perturbation (default 0)"),
  scope: z
    .enum(["all", "selection", "cell_type", "cluster"])
    .optional()
    .describe("Where to apply the perturbation"),
  cell_type_label: z
    .string()
    .optional()
    .describe(
      "Annotation label when scope=cell_type (e.g. Epithelial). Server unions every cluster with that exact name for the KO.",
    ),
  cluster_id: z
    .number()
    .int()
    .optional()
    .describe("Cluster id from cluster_annot when scope=cluster"),
  n_propagation: z
    .number()
    .int()
    .min(1)
    .max(32)
    .optional()
    .describe("GRN propagation depth (overrides run TOML default)"),
  push_summary_to_chat: z
    .boolean()
    .optional()
    .describe(
      "If true, the viewer sends a short Δ summary (min/max/mean) to the chat via ui/message after the run.",
    ),
  run_umap_quiver: z
    .boolean()
    .optional()
    .describe(
      "If true, after perturbation preview the viewer runs UMAP transition quiver (same as Run perturb + UMAP quiver). " +
        "Uses current transition-panel options; leave limit_clusters unchecked in the UI to show arrows on all cells.",
    ),
};

const collectInteractionsInputSchema = {
  aggregate: z
    .enum(["mean", "min", "max", "sum", "positive", "negative"])
    .optional()
    .describe("How to aggregate β across selected cells (Python Betabase.collect_interactions)"),
  filter: z
    .enum(["cell_type", "cluster"])
    .optional()
    .describe("Restrict to one annotation category or one cluster id"),
  cell_type: z
    .string()
    .optional()
    .describe("Category label when filter=cell_type"),
  cluster_id: z
    .number()
    .int()
    .optional()
    .describe("Cluster id when filter=cluster"),
  max_genes: z
    .number()
    .int()
    .min(1)
    .max(4096)
    .optional()
    .describe("Max target-gene feather files to scan in parallel (default 2048)"),
  push_summary_to_chat: z
    .boolean()
    .optional()
    .describe("Send top ligand–receptor rows as markdown to the chat"),
};

const controlInputSchema = {
  expression_gene: z
    .string()
    .optional()
    .describe("Gene symbol for expression coloring (sets color source to expression)"),
  color_source: z
    .enum(["expression", "betadata", "perturb", "received_ligand"])
    .optional()
    .describe("Active color mode in the viewer"),
  apply_expression: z
    .boolean()
    .optional()
    .describe("If true with expression_gene, run Load color after setting gene"),
  received_ligand_genes: z
    .string()
    .optional()
    .describe(
      "Comma-separated ligand symbols (adata / Gaussian) or one training column name (model); sets color to received_ligand",
    ),
  received_ligand_source: z
    .enum(["adata", "model"])
    .optional()
    .describe("Compute from expression+spatial vs slice training GeneMatrix"),
  received_ligand_matrix: z
    .enum(["lr", "tfl"])
    .optional()
    .describe("Which received-ligand block when source=model"),
  received_ligand_radius: z
    .number()
    .positive()
    .optional()
    .describe("Gaussian radius in spatial coordinate units (adata path)"),
  received_ligand_scale: z
    .number()
    .optional()
    .describe("weighted_ligand scale factor (adata path)"),
  received_ligand_use_grid: z
    .boolean()
    .optional()
    .describe("Use grid acceleration for adata path (default true in UI)"),
  received_ligand_grid_factor: z
    .number()
    .positive()
    .optional()
    .describe("Grid cell size as fraction of radius"),
  received_ligand_aggregate: z
    .enum(["sum", "max", "mean"])
    .optional()
    .describe("Aggregate multiple ligand channels (adata path only)"),
  apply_received_ligand: z
    .boolean()
    .optional()
    .describe("If true, run Load received ligand after applying fields above"),
  betadata_gene: z
    .string()
    .optional()
    .describe(
      "Betadata target gene (must match *_betadata.feather stem); sets color source to betadata and selects that target",
    ),
  betadata_column: z
    .string()
    .optional()
    .describe(
      "Coefficient column name from GET /api/betadata/columns?gene=… (e.g. beta_VEGFA or LIG$REC); required for apply_betadata unless exactly one plottable column exists",
    ),
  apply_betadata: z
    .boolean()
    .optional()
    .describe(
      "If true, fetch per-cell β via GET /api/betadata/values and refresh the spatial layer (same as UI Load / refresh). Use with betadata_gene and betadata_column for CellID (spatial) models.",
    ),
  focus_gene_context: z
    .string()
    .optional()
    .describe("Focus gene for interaction / LR context panel"),
  status_message: z
    .string()
    .optional()
    .describe("Short message shown in the viewer status bar (LLM narration)"),
};

const signatureUmapToolInputSchema = {
  genes: z
    .string()
    .min(1)
    .describe(
      "Comma-separated gene symbols; signature per cell = sum of expression in the dataset layer (SpaceTravLR VirtualTissue-style)",
    ),
  n_knn: z
    .number()
    .int()
    .min(3)
    .max(200)
    .optional()
    .describe("KNN neighbors for interpolating signature onto the UMAP grid (default 30)"),
  mask_with_perturb_quiver: z
    .boolean()
    .optional()
    .describe(
      "If true, arrows are zero where the perturbation transition quiver is zero; set perturb gene, scope, and quick-KO in the viewer before calling",
    ),
};

const splashNetworkToolInputSchema = {
  gene_a: z.string().min(1).describe("Upstream / source gene (modulator in splash graph)"),
  gene_b: z
    .string()
    .min(1)
    .describe("Downstream gene; must be a trained target (has *_betadata.feather)"),
  surround_hops: z
    .number()
    .int()
    .min(0)
    .max(4)
    .optional()
    .describe("Undirected hops around the A→B shortest path to add context genes (default 1)"),
  max_nodes: z
    .number()
    .int()
    .min(6)
    .max(64)
    .optional()
    .describe("Cap on number of genes in the subgraph (default 24)"),
  scope: z.enum(["all", "cell_type", "cluster"]).optional().describe("Cell mask for averaging splash"),
  cell_type_label: z
    .string()
    .optional()
    .describe("Annotation label when scope=cell_type"),
  cluster_id: z.number().int().optional().describe("Cluster id when scope=cluster"),
};

const receivedLigandToolInputSchema = {
  genes: z
    .string()
    .min(1)
    .describe(
      "Comma/space-separated ligand gene symbols (adata) or a single column name from training received-ligand matrix (model)",
    ),
  source: z
    .enum(["adata", "model"])
    .optional()
    .describe("default adata: recompute weighted neighbors; model needs perturb_ready"),
  matrix: z
    .enum(["lr", "tfl"])
    .optional()
    .describe("Training matrix slice when source=model (default lr)"),
  radius: z.number().positive().optional(),
  scale_factor: z.number().optional(),
  use_grid: z.boolean().optional(),
  grid_factor: z.number().positive().optional(),
  aggregate: z.enum(["sum", "max", "mean"]).optional(),
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function createSpatialViewerMcpServer(deps: SpatialMcpDeps): McpServer {
  const fetchImpl = deps.fetch;
  const defaultApiBase = deps.defaultApiBase;
  const connectDomainList = deps.connectDomainList;
  const readMcpHtml = deps.readMcpHtml;

  const fetchMeta = (api: string) => fetchMetaWith(fetchImpl, api);

  const server = new McpServer({
    name: "spatial-viewer",
    version: "1.0.0",
  });

registerAppTool(
  server,
  "show_spatial_viewer",
  {
    title: "Spatial viewer",
    description:
      "Open the SpaceTravLR spatial transcriptomics viewer in the chat UI. Requires spatial_viewer (Rust) running with --allow-cors and matching api_base_url. After calling this, use spatial_viewer_wait_ready to confirm the dataset (and optionally perturbation runtime) has finished loading before running further analysis.",
    inputSchema: openInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const structured = {
      api_base_url: normalizeApiBase(
        args.api_base_url ? String(args.api_base_url) : undefined,
        defaultApiBase,
      ),
      adata_path: args.adata_path,
      layer: args.layer ?? "",
      cluster_annot: args.cluster_annot ?? "",
      network_dir: args.network_dir ?? "",
      run_toml: args.run_toml ?? "",
      perturb_overlay: args.perturb_overlay ?? "",
      _spatialTool: "open" as const,
    };
    return {
      content: [
        {
          type: "text" as const,
          text: `Spatial viewer: loading ${args.adata_path} (API ${structured.api_base_url}).`,
        },
      ],
      structuredContent: structured,
    };
  },
);

registerAppTool(
  server,
  "spatial_viewer_capture_render",
  {
    title: "Spatial viewer — screenshot for assistant",
    description:
      "Capture the current Deck.gl view (spatial scatter, expression/betadata coloring, or UMAP + quiver when that layout is active) as a PNG. The inline MCP app posts the image into the chat (ui/message + model context) so the assistant can see it. Requires the viewer UI open and a loaded dataset.",
    inputSchema: captureInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const max_width = args.max_width;
    const caption = args.caption ?? "";
    const structured = {
      _spatialTool: "capture" as const,
      max_width,
      caption,
      api_base_url: normalizeApiBase(undefined, defaultApiBase),
    };
    return {
      content: [
        {
          type: "text" as const,
          text:
            "Capture requested. The spatial viewer (if open) will send a PNG to the chat for the assistant. If nothing appears, open show_spatial_viewer first and ensure a dataset is loaded.",
        },
      ],
      structuredContent: structured,
    };
  },
);

registerAppTool(
  server,
  "spatial_viewer_run_perturb",
  {
    title: "Spatial viewer — run GRN perturbation",
    description:
      "Runs in-silico perturbation in the open viewer (same as UI Load with color=perturb). Requires spatial_viewer with --run-toml and perturb_ready (check with spatial_viewer_wait_ready first). The iframe executes POST /api/perturb/preview — this takes 30–120 s depending on dataset size. Use spatial_viewer_check_progress to poll progress if needed. The viewer sends a Δ summary if push_summary_to_chat is true. " +
      "Set scope=cell_type with cell_type_label matching an annotation name (e.g. Epithelial); the server unions all clusters that share that label for a scoped KO. " +
      "Set run_umap_quiver=true to also compute the UMAP quiver (POST /api/perturb/umap-field with export_svg); the server saves an SVG under /tmp on Unix and the viewer status line shows the path. Uncheck limit_clusters in the UI if you want all cells.",
    inputSchema: perturbRunInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const structured = {
      _spatialTool: "perturb_run" as const,
      gene: args.gene,
      desired_expr: args.desired_expr ?? 0,
      scope: args.scope ?? "all",
      cell_type_label: args.cell_type_label ?? "",
      cluster_id: args.cluster_id ?? 0,
      n_propagation: args.n_propagation,
      push_summary_to_chat: args.push_summary_to_chat === true,
      run_umap_quiver: args.run_umap_quiver === true,
      api_base_url: normalizeApiBase(undefined, defaultApiBase),
    };
    return {
      content: [
        {
          type: "text" as const,
          text: `Perturbation requested for ${args.gene}${args.run_umap_quiver ? " + UMAP quiver" : ""}. The viewer will run when the UI is open.`,
        },
      ],
      structuredContent: structured,
    };
  },
);

registerAppTool(
  server,
  "spatial_viewer_collect_interactions",
  {
    title: "Spatial viewer — collect β interactions",
    description:
      "Scans betadata feathers in parallel (Rust/Rayon) like Python Betabase.collect_interactions: aggregated β per target gene × modulator edge for cells of one type or cluster. Takes 5–30 s depending on number of feathers. Results appear in the viewer bar chart; optional chat summary. Requires dataset_ready=true and betadata_dir configured.",
    inputSchema: collectInteractionsInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const structured = {
      _spatialTool: "collect_interactions" as const,
      aggregate: args.aggregate ?? "mean",
      filter: args.filter ?? "cell_type",
      cell_type: args.cell_type ?? "",
      cluster_id: args.cluster_id ?? 0,
      max_genes: args.max_genes ?? 2048,
      push_summary_to_chat: args.push_summary_to_chat === true,
      api_base_url: normalizeApiBase(undefined, defaultApiBase),
    };
    return {
      content: [
        {
          type: "text" as const,
          text:
            "Collect interactions requested. The viewer (if open) will POST /api/betadata/collect_interactions using parallel feather scans.",
        },
      ],
      structuredContent: structured,
    };
  },
);

registerAppTool(
  server,
  "spatial_viewer_control",
  {
    title: "Spatial viewer — UI control",
    description:
      "Update the open spatial viewer (gene, color mode, status). Call after show_spatial_viewer. Same UI resource. " +
      "For betadata spatial coloring: set betadata_gene, betadata_column (coefficient), and apply_betadata=true so GET /api/betadata/values runs and cells are colored by per-cell β (CellID) or cluster-mapped β (Cluster).",
    inputSchema: controlInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const structured = {
      ...args,
      _spatialTool: "control" as const,
      api_base_url: normalizeApiBase(undefined, defaultApiBase),
    };
    return {
      content: [
        {
          type: "text" as const,
          text: `Viewer control: ${JSON.stringify(args)}`,
        },
      ],
      structuredContent: structured,
    };
  },
);

registerAppTool(
  server,
  "spatial_viewer_received_ligand",
  {
    title: "Spatial viewer — received ligand coloring",
    description:
      "Sets the viewer to Color → Received ligand and loads values via POST /api/spatial/received_ligand. " +
      "adata: Gaussian weighted sum of neighbor ligand expression (same rule as training; optional grid). " +
      "model: one column from rw_ligands_init / rw_tfligands_init (requires perturb_ready + --run-toml).",
    inputSchema: receivedLigandToolInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const genes = parseDelimitedGenes(String(args.genes));
    const structured = {
      _spatialTool: "received_ligand" as const,
      genes,
      source: args.source ?? "adata",
      matrix: args.matrix ?? "lr",
      radius: args.radius,
      scale_factor: args.scale_factor,
      use_grid: args.use_grid,
      grid_factor: args.grid_factor,
      aggregate: args.aggregate ?? "sum",
      api_base_url: normalizeApiBase(undefined, defaultApiBase),
    };
    return {
      content: [
        {
          type: "text" as const,
          text: `Received ligand: ${genes.join(", ")} (${structured.source}).`,
        },
      ],
      structuredContent: structured,
    };
  },
);

registerAppTool(
  server,
  "spatial_viewer_signature_umap",
  {
    title: "Spatial viewer — gene signature UMAP quiver",
    description:
      "Runs POST /api/umap/signature_field in the open viewer: KNN-smoothed Σ-expression on the UMAP grid, gradient arrows (Python virtual_tissue.signature2gradient). " +
      "Switch layout to UMAP to see teal arrows (perturb quiver stays orange). Optional mask uses current perturb row.",
    inputSchema: signatureUmapToolInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const genes = parseDelimitedGenes(String(args.genes));
    const structured = {
      _spatialTool: "signature_umap" as const,
      genes,
      n_knn: args.n_knn,
      mask_with_perturb_quiver: args.mask_with_perturb_quiver === true,
      api_base_url: normalizeApiBase(undefined, defaultApiBase),
    };
    return {
      content: [
        {
          type: "text" as const,
          text: `UMAP gene signature quiver: ${genes.join(", ")}.`,
        },
      ],
      structuredContent: structured,
    };
  },
);

registerAppTool(
  server,
  "spatial_viewer_splash_network",
  {
    title: "Spatial viewer — splash gene network (D3)",
    description:
      "Computes mean splash() derivatives ∂(target)/∂(modulator) over selected cells, finds a directed path gene_a → gene_b, and opens the interactive D3 force layout in the viewer. " +
      "gene_b must be a trained target. Uses POST /api/perturb/splash_network. Requires perturb_ready.",
    inputSchema: splashNetworkToolInputSchema,
    _meta: {
      ui: {
        resourceUri: RESOURCE_URI,
      },
    },
  },
  async (args) => {
    const gene_a = String(args.gene_a ?? "").trim();
    const gene_b = String(args.gene_b ?? "").trim();
    const structured = {
      _spatialTool: "splash_network" as const,
      gene_a,
      gene_b,
      surround_hops: args.surround_hops ?? 1,
      max_nodes: args.max_nodes ?? 24,
      scope: args.scope,
      cell_type_label: args.cell_type_label,
      cluster_id: args.cluster_id,
      api_base_url: normalizeApiBase(undefined, defaultApiBase),
    };
    return {
      content: [
        {
          type: "text" as const,
          text: `Splash network: ${gene_a} → ${gene_b} (${structured.surround_hops} hop context, max ${structured.max_nodes} nodes).`,
        },
      ],
      structuredContent: structured,
    };
  },
);

server.registerTool(
  "spatial_viewer_report_context",
  {
    description:
      "Posts a short viewer summary into the conversation. Usually invoked from the viewer UI (Send context to chat).",
    inputSchema: {
      summary: z
        .string()
        .describe("Markdown or plain text summary for the assistant"),
      detail: z
        .string()
        .optional()
        .describe("Optional extra JSON or structured notes"),
    },
  },
  async ({ summary, detail }) => {
    const text =
      detail && detail.trim().length > 0
        ? `${summary}\n\n${detail}`
        : summary;
    return {
      content: [{ type: "text" as const, text }],
    };
  },
);

server.registerTool(
  "spatial_viewer_cluster_expression",
  {
    description:
      "Get mean gene expression per cluster for a list of genes (≤200 at a time, ~1 s per batch). Useful for annotating clusters by checking known marker genes. Returns cluster IDs, n_cells_per_cluster, and mean expression per gene. Requires dataset_ready=true.",
    inputSchema: {
      genes: z.array(z.string()).min(1).max(200).describe("Gene symbols to query"),
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ genes, api_base_url }: { genes: string[]; api_base_url?: string }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    try {
      const res = await fetchImpl(`${api}/api/cluster/mean_expression`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ genes }),
      });
      if (!res.ok) {
        const msg = await res.text();
        return { content: [{ type: "text" as const, text: `Error ${res.status}: ${msg}` }] };
      }
      const data = await res.json();
      return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_label_clusters",
  {
    description:
      "Assign human-readable cell-type labels to integer clusters. The viewer UI will auto-update to show the new names. Pass a map of cluster_id (as string) → label.",
    inputSchema: {
      labels: z
        .record(z.string(), z.string())
        .describe('Map of cluster_id → cell-type label, e.g. {"0":"B cells","1":"T cells"}'),
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ labels, api_base_url }: { labels: Record<string, string>; api_base_url?: string }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    try {
      const res = await fetchImpl(`${api}/api/meta/label_clusters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labels }),
      });
      if (!res.ok) {
        const msg = await res.text();
        return { content: [{ type: "text" as const, text: `Error ${res.status}: ${msg}` }] };
      }
      const data = await res.json();
      return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_perturb_summary",
  {
    description:
      "Run GRN perturbation and return a structured summary of the top 50 affected genes with mean_delta and max_abs_delta. BLOCKING: takes 30–120 s depending on dataset size and n_propagation. Ensure perturb_ready=true first (use spatial_viewer_wait_ready). Does NOT update the viewer UI — use spatial_viewer_run_perturb for that.",
    inputSchema: {
      gene: z.string().describe("Gene symbol to KO"),
      desired_expr: z.number().optional().describe("Target expression (default 0 for KO)"),
      scope: z
        .enum(["all", "selection", "cell_type", "cluster"])
        .optional()
        .describe("Perturbation scope (selection = viewer sender cell or cell_indices)"),
      cell_type_label: z.string().optional().describe("Cell type category when scope=cell_type"),
      cluster_id: z.number().int().optional().describe("Cluster ID when scope=cluster"),
      cell_indices: z
        .array(z.number().int().nonnegative())
        .optional()
        .describe("Explicit cell row indices when scope=selection"),
      n_propagation: z.number().int().min(1).max(32).optional().describe("GRN propagation depth"),
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({
    gene,
    desired_expr,
    scope,
    cell_type_label,
    cluster_id,
    cell_indices,
    n_propagation,
    api_base_url,
  }: {
    gene: string;
    desired_expr?: number;
    scope?: string;
    cell_type_label?: string;
    cluster_id?: number;
    cell_indices?: number[];
    n_propagation?: number;
    api_base_url?: string;
  }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);

    try {
      const meta = await fetchMeta(api);
      if (!meta.perturb_ready) {
        const hint = meta.perturb_loading
          ? `Perturbation runtime is still loading (${
              meta.perturb_progress_permille != null && Number.isFinite(meta.perturb_progress_permille)
                ? `${(meta.perturb_progress_permille / 10).toFixed(1)}%`
                : `${meta.perturb_progress_percent ?? 0}%`
            } — ${meta.perturb_progress_label ?? ""}). Use spatial_viewer_wait_ready(require_perturb=true) first.`
          : "Perturbation is not configured. Start the server with --run-toml.";
        return { content: [{ type: "text" as const, text: `NOT READY: ${hint}` }] };
      }
    } catch { /* server may not have /api/meta yet — try the summary anyway */ }

    const built = await buildPerturbScopeApiBody(
      fetchImpl,
      api,
      scope,
      cell_type_label,
      cluster_id,
      cell_indices,
    );
    if (built.error) {
      return { content: [{ type: "text" as const, text: `BAD SCOPE: ${built.error}` }] };
    }
    const reqBody: Record<string, unknown> = {
      gene,
      desired_expr: desired_expr ?? 0,
      scope: built.scope,
    };
    if (n_propagation != null) reqBody.n_propagation = n_propagation;
    const t0 = Date.now();
    try {
      const res = await fetchImpl(`${api}/api/perturb/summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
      });
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      if (!res.ok) {
        const msg = await res.text();
        return { content: [{ type: "text" as const, text: `Error ${res.status} (after ${elapsed}s): ${msg}` }] };
      }
      const data = await res.json();
      return {
        content: [
          { type: "text" as const, text: `Perturbation summary for ${gene} (completed in ${elapsed}s):\n${JSON.stringify(data, null, 2)}` },
        ],
      };
    } catch (e: any) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      return { content: [{ type: "text" as const, text: `Fetch error after ${elapsed}s: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_perturb_reference_similarity",
  {
    description:
      "POST /api/perturb/reference_similarity: after full GRN perturbation, cosine similarity of each perturbed cell's expression to a reference cell-type centroid (before vs after). Use reference= T_follicular_helper and exclude_perturb_cells_from_reference=true to ask if Tfh became more like other Tfh. BLOCKING like perturb_summary. genes[] optional (marker panel); omit for all model genes.",
    inputSchema: {
      gene: z.string().describe("Perturbed gene symbol"),
      desired_expr: z.number().optional().describe("Target expression level"),
      scope: z
        .enum(["all", "selection", "cell_type", "cluster"])
        .optional()
        .describe("Which cells receive the perturbation"),
      cell_type_label: z.string().optional().describe("Perturbation scope when scope=cell_type"),
      cluster_id: z.number().int().optional().describe("Perturbation scope when scope=cluster"),
      cell_indices: z.array(z.number().int().nonnegative()).optional().describe("When scope=selection"),
      n_propagation: z.number().int().min(1).max(32).optional().describe("GRN propagation depth"),
      reference_scope: z.enum(["all", "cell_type", "cluster"]).describe("Cell set for centroid: all, one cell_type, or one cluster"),
      reference_cell_type_label: z.string().optional().describe("Reference cell type name when reference_scope=cell_type"),
      reference_cluster_id: z.number().int().optional().describe("Reference cluster when reference_scope=cluster"),
      genes: z.array(z.string()).optional().describe("Gene symbols for cosine subspace; omit = all model genes"),
      exclude_perturb_cells_from_reference: z
        .boolean()
        .optional()
        .describe("Default true: omit perturb-target cells from centroid (recommended when ref type = perturbed type)"),
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async (args: Record<string, unknown> & { api_base_url?: string }) => {
    const api = normalizeApiBase(String(args.api_base_url ?? "").trim() || undefined, defaultApiBase);
    const gene = String(args.gene ?? "").trim();
    if (!gene) {
      return { content: [{ type: "text" as const, text: "gene is required." }] };
    }
    try {
      const meta = await fetchMeta(api);
      if (!meta.perturb_ready) {
        return {
          content: [
            {
              type: "text" as const,
              text: "perturb_ready is false — use spatial_viewer_wait_ready(require_perturb=true).",
            },
          ],
        };
      }
    } catch {
      /* continue */
    }
    const cellIdx =
      Array.isArray(args.cell_indices) && args.cell_indices.length > 0
        ? args.cell_indices.filter((x: unknown): x is number => typeof x === "number")
        : undefined;
    const built = await buildPerturbScopeApiBody(
      fetchImpl,
      api,
      typeof args.scope === "string" ? args.scope : undefined,
      typeof args.cell_type_label === "string" ? args.cell_type_label : undefined,
      typeof args.cluster_id === "number" ? args.cluster_id : undefined,
      cellIdx,
    );
    if (built.error) {
      return { content: [{ type: "text" as const, text: `BAD SCOPE: ${built.error}` }] };
    }
    const reference = buildReferenceCentroidScopeBody(
      typeof args.reference_scope === "string" ? args.reference_scope : undefined,
      typeof args.reference_cell_type_label === "string"
        ? args.reference_cell_type_label
        : undefined,
      typeof args.reference_cluster_id === "number" ? args.reference_cluster_id : undefined,
    );
    const reqBody: Record<string, unknown> = {
      gene,
      desired_expr: typeof args.desired_expr === "number" ? args.desired_expr : 0,
      scope: built.scope,
      reference,
      exclude_perturb_cells_from_reference:
        args.exclude_perturb_cells_from_reference === false ? false : true,
    };
    if (args.n_propagation != null) reqBody.n_propagation = args.n_propagation;
    if (Array.isArray(args.genes) && args.genes.length > 0) {
      reqBody.genes = args.genes.map((x) => String(x).trim()).filter(Boolean);
    }
    const t0 = Date.now();
    try {
      const res = await fetchImpl(`${api}/api/perturb/reference_similarity`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
      });
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      if (!res.ok) {
        const msg = await res.text();
        return { content: [{ type: "text" as const, text: `Error ${res.status} (after ${elapsed}s): ${msg}` }] };
      }
      const data = await res.json();
      return {
        content: [
          {
            type: "text" as const,
            text: `Reference similarity (${elapsed}s). mean_delta_cosine > 0 means more like reference after perturb.\n${JSON.stringify(data, null, 2)}`,
          },
        ],
      };
    } catch (e: any) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      return { content: [{ type: "text" as const, text: `Fetch error after ${elapsed}s: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_splash_network_json",
  {
    description:
      "HTTP-only: POST /api/perturb/splash_network and return raw JSON (no iframe). Same parameters as app tool spatial_viewer_splash_network. " +
      "Splash() derivative graph between two genes (mean over cells in scope). BLOCKING; requires perturb_ready.",
    inputSchema: {
      ...splashNetworkToolInputSchema,
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async (args: Record<string, unknown> & { api_base_url?: string }) => {
    const api = normalizeApiBase(String(args.api_base_url ?? "").trim() || undefined, defaultApiBase);
    const gene_a = String(args.gene_a ?? "").trim();
    const gene_b = String(args.gene_b ?? "").trim();
    if (!gene_a || !gene_b) {
      return { content: [{ type: "text" as const, text: "gene_a and gene_b are required." }] };
    }
    try {
      const meta = await fetchMeta(api);
      if (!meta.perturb_ready) {
        return {
          content: [
            {
              type: "text" as const,
              text: "perturb_ready is false — start spatial_viewer with --run-toml and wait for load.",
            },
          ],
        };
      }
    } catch {
      /* continue */
    }
    const scope = buildPerturbScopeBody(
      typeof args.scope === "string" ? args.scope : undefined,
      typeof args.cell_type_label === "string" ? args.cell_type_label : undefined,
      typeof args.cluster_id === "number" ? args.cluster_id : undefined,
    );
    const body: Record<string, unknown> = {
      gene_a,
      gene_b,
      scope,
      surround_hops:
        typeof args.surround_hops === "number" && Number.isFinite(args.surround_hops)
          ? args.surround_hops
          : 1,
      max_nodes:
        typeof args.max_nodes === "number" && Number.isFinite(args.max_nodes)
          ? args.max_nodes
          : 24,
    };
    const t0 = Date.now();
    try {
      const res = await fetchImpl(`${api}/api/perturb/splash_network`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      if (!res.ok) {
        const msg = await res.text();
        return {
          content: [{ type: "text" as const, text: `Error ${res.status} (after ${elapsed}s): ${msg}` }],
        };
      }
      const data = await res.json();
      return {
        content: [
          {
            type: "text" as const,
            text: `Splash network ${gene_a} → ${gene_b} (${elapsed}s):\n${JSON.stringify(data, null, 2)}`,
          },
        ],
      };
    } catch (e: any) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      return { content: [{ type: "text" as const, text: `Fetch error after ${elapsed}s: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_perturb_neighbor_sanity",
  {
    description:
      "Single-cell scoped GRN perturbation sanity check: applies desired_expr to exactly one cell (optional require_cluster_id, e.g. Tfh cluster from cell_type_int), runs the same perturbation as /api/perturb/preview, then compares mean |Δ| in spatial neighbors (within neighbor_radius, default = max LR training radius) vs remote cells. BLOCKING (same cost as a full perturb pass). Use after perturb_ready. cell_index is 0-based AnnData row order (matches viewer indices).",
    inputSchema: {
      gene: z.string().describe("Gene to perturb (e.g. IL21)"),
      cell_index: z.number().int().min(0).describe("0-based cell row index to receive the perturbation"),
      desired_expr: z.number().optional().describe("Target expression on that cell (default 0)"),
      n_propagation: z.number().int().min(1).max(32).optional().describe("GRN propagation depth"),
      neighbor_radius: z
        .number()
        .optional()
        .describe("Euclidean cutoff in same units as training spatial coords (default: max ligand radius from run)"),
      require_cluster_id: z
        .number()
        .int()
        .optional()
        .describe("If set, fail unless clusters[cell_index] equals this id (e.g. T_follicular_helper cluster int)"),
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({
    gene,
    cell_index,
    desired_expr,
    n_propagation,
    neighbor_radius,
    require_cluster_id,
    api_base_url,
  }: {
    gene: string;
    cell_index: number;
    desired_expr?: number;
    n_propagation?: number;
    neighbor_radius?: number;
    require_cluster_id?: number;
    api_base_url?: string;
  }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);

    try {
      const meta = await fetchMeta(api);
      if (!meta.perturb_ready) {
        const hint = meta.perturb_loading
          ? `Perturbation runtime is still loading. Use spatial_viewer_wait_ready(require_perturb=true) first.`
          : "Perturbation is not configured. Start the server with --run-toml.";
        return { content: [{ type: "text" as const, text: `NOT READY: ${hint}` }] };
      }
    } catch { /* try request anyway */ }

    const reqBody: Record<string, unknown> = {
      gene,
      cell_index,
      desired_expr: desired_expr ?? 0,
    };
    if (n_propagation != null) reqBody.n_propagation = n_propagation;
    if (neighbor_radius != null) reqBody.neighbor_radius = neighbor_radius;
    if (require_cluster_id != null) reqBody.require_cluster_id = require_cluster_id;

    const t0 = Date.now();
    try {
      const res = await fetchImpl(`${api}/api/perturb/neighbor_sanity`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
      });
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      if (!res.ok) {
        const msg = await res.text();
        return { content: [{ type: "text" as const, text: `Error ${res.status} (after ${elapsed}s): ${msg}` }] };
      }
      const data = await res.json();
      return {
        content: [
          {
            type: "text" as const,
            text: `Neighbor sanity for ${gene} @ cell ${cell_index} (${elapsed}s):\n${JSON.stringify(data, null, 2)}`,
          },
        ],
      };
    } catch (e: any) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      return { content: [{ type: "text" as const, text: `Fetch error after ${elapsed}s: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_get_meta",
  {
    description:
      "Get the current dataset metadata (n_obs, n_vars, cluster_annot, cell_type_categories, perturb_ready, etc.). Returns both raw JSON and a human-readable status summary.",
    inputSchema: {
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ api_base_url }: { api_base_url?: string }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    try {
      const m = await fetchMeta(api);
      const summary = formatStatus(m);
      return {
        content: [
          { type: "text" as const, text: summary + "\n\n" + JSON.stringify(m, null, 2) },
        ],
      };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_check_progress",
  {
    description:
      "Quick status check: is the dataset loaded? Is betadata/perturbation ready or still loading (with %)? Is a perturbation job running? Returns a concise human-readable status line. Use this to poll progress during long operations.",
    inputSchema: {
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ api_base_url }: { api_base_url?: string }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    try {
      const m = await fetchMeta(api);
      return { content: [{ type: "text" as const, text: formatStatus(m) }] };
    } catch (e: any) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Server unreachable (${api}): ${e.message}\nIs spatial_viewer running? Start with: cargo run --features spatial-viewer --bin spatial_viewer -- --run-toml <path> --static-dir web/spatial_viewer/dist --allow-cors --bind 127.0.0.1 --port 8080`,
          },
        ],
      };
    }
  },
);

server.registerTool(
  "spatial_viewer_wait_ready",
  {
    description:
      "Polls the spatial_viewer server until the dataset is loaded (and optionally perturbation runtime is ready). Returns a status summary when ready, or after timeout. Use this after show_spatial_viewer or after starting the server to confirm everything is loaded before performing analysis. Betadata/perturbation loading typically takes 30–180 s depending on dataset size.",
    inputSchema: {
      require_perturb: z
        .boolean()
        .optional()
        .describe("If true (default), wait until perturb_ready=true. If false, only wait for dataset_ready."),
      timeout_seconds: z
        .number()
        .int()
        .min(5)
        .max(600)
        .optional()
        .describe("Max seconds to wait (default 300 = 5 min)"),
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ require_perturb, timeout_seconds, api_base_url }: {
    require_perturb?: boolean; timeout_seconds?: number; api_base_url?: string;
  }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    const needPerturb = require_perturb !== false;
    const timeoutMs = (timeout_seconds ?? 300) * 1000;
    const t0 = Date.now();
    let lastMeta: MetaSnapshot | null = null;
    let lastPct = -1;

    while (Date.now() - t0 < timeoutMs) {
      try {
        lastMeta = await fetchMeta(api);

        if (lastMeta.dataset_ready && (!needPerturb || lastMeta.perturb_ready)) {
          const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
          return {
            content: [
              {
                type: "text" as const,
                text: `READY (waited ${elapsed}s)\n${formatStatus(lastMeta)}`,
              },
            ],
          };
        }

        if (lastMeta.perturb_error) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Perturbation load FAILED: ${lastMeta.perturb_error}\n${formatStatus(lastMeta)}`,
              },
            ],
          };
        }

        const pct = lastMeta.perturb_progress_percent ?? 0;
        if (pct !== lastPct) lastPct = pct;
      } catch {
        // server not yet up — keep waiting
      }

      await sleep(2000);
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const status = lastMeta ? formatStatus(lastMeta) : "Server unreachable";
    return {
      content: [
        {
          type: "text" as const,
          text: `TIMEOUT after ${elapsed}s — not fully ready yet.\n${status}`,
        },
      ],
    };
  },
);

server.registerTool(
  "spatial_viewer_cancel_jobs",
  {
    description:
      "Cancel all running or queued jobs on the spatial_viewer server (perturbation preview, summary, UMAP quiver, background betadata loading). " +
      "Use this when an operation appears stuck, is taking too long, or you want to start a different analysis. " +
      "After cancelling, check status with spatial_viewer_check_progress and start fresh.",
    inputSchema: {
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ api_base_url }: { api_base_url?: string }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    try {
      const res = await fetchImpl(`${api}/api/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        const msg = await res.text();
        return { content: [{ type: "text" as const, text: `Error ${res.status}: ${msg}` }] };
      }
      const data = await res.json();
      await sleep(500);
      let statusLine = "";
      try {
        const m = await fetchMeta(api);
        statusLine = "\n\nCurrent status:\n" + formatStatus(m);
      } catch { /* ignore */ }
      return {
        content: [
          {
            type: "text" as const,
            text: `Cancel requested: ${data.message ?? "ok"}${statusLine}\n\nNote: background loading was suppressed. To reload perturbation runtime, use show_spatial_viewer or reload the dataset.`,
          },
        ],
      };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_session_snapshot",
  {
    description:
      "Fetches GET /api/meta and GET /api/viewer_state together (dataset + UI-reported color/perturb/sender cell). Use for agent-in-the-loop context.",
    inputSchema: {
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ api_base_url }: { api_base_url?: string }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    try {
      const [mr, vr] = await Promise.all([
        fetchImpl(`${api}/api/meta`),
        fetchImpl(`${api}/api/viewer_state`),
      ]);
      const meta = mr.ok ? ((await mr.json()) as Record<string, unknown>) : { error: await mr.text() };
      const viewer_state = vr.ok
        ? ((await vr.json()) as Record<string, unknown>)
        : { error: await vr.text() };
      const snap = { meta, viewer_state };
      return {
        content: [{ type: "text" as const, text: JSON.stringify(snap, null, 2) }],
        structuredContent: snap,
      };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_betadata_pair_lr",
  {
    description:
      "POST /api/betadata/pair_lr — top ligand–receptor β explaining communication between two cell indices (AnnData row order).",
    inputSchema: {
      cell_a: z.number().int().min(0).describe("Cell index A (0-based)"),
      cell_b: z.number().int().min(0).describe("Cell index B (0-based)"),
      top_n: z.number().int().min(1).max(200).optional(),
      max_genes: z.number().int().min(1).max(4096).optional(),
      api_base_url: z.string().optional(),
    },
  },
  async (args: Record<string, unknown> & { api_base_url?: string }) => {
    const api = normalizeApiBase(String(args.api_base_url ?? "").trim() || undefined, defaultApiBase);
    const cell_a = Number(args.cell_a);
    const cell_b = Number(args.cell_b);
    try {
      const res = await fetchImpl(`${api}/api/betadata/pair_lr`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cell_a,
          cell_b,
          top_n: args.top_n ?? 25,
          max_genes: args.max_genes ?? 2048,
        }),
      });
      if (!res.ok) {
        return { content: [{ type: "text" as const, text: `Error ${res.status}: ${await res.text()}` }] };
      }
      const data = await res.json();
      return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

server.registerTool(
  "spatial_viewer_search_genes",
  {
    description:
      "GET /api/genes?prefix= — validates gene symbols exist in the dataset before expensive perturb calls.",
    inputSchema: {
      prefix: z.string().min(1).describe("Gene symbol prefix"),
      limit: z.number().int().min(1).max(2000).optional(),
      api_base_url: z.string().optional(),
    },
  },
  async ({
    prefix,
    limit,
    api_base_url,
  }: {
    prefix: string;
    limit?: number;
    api_base_url?: string;
  }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    const lim = limit ?? 40;
    try {
      const res = await fetchImpl(
        `${api}/api/genes?prefix=${encodeURIComponent(prefix.trim())}&limit=${lim}`,
      );
      if (!res.ok) {
        return { content: [{ type: "text" as const, text: `Error ${res.status}: ${await res.text()}` }] };
      }
      const list = (await res.json()) as string[];
      return {
        content: [{ type: "text" as const, text: JSON.stringify({ prefix, genes: list }, null, 2) }],
      };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

const perturbPlanJobSchema = z.object({
  gene: z.string(),
  desired_expr: z.number().optional(),
  scope: z.enum(["all", "selection", "cell_type", "cluster"]).optional(),
  cell_type_label: z.string().optional(),
  cluster_id: z.number().int().optional(),
  cell_indices: z.array(z.number().int().nonnegative()).optional(),
  n_propagation: z.number().int().min(1).max(32).optional(),
});

server.registerTool(
  "spatial_viewer_run_perturb_plan",
  {
    description:
      "Runs POST /api/perturb/summary serially for each job (max 20). BLOCKING total time ≈ sum of individual summaries. Use for multi-gene screening from the agent.",
    inputSchema: {
      jobs: z.array(perturbPlanJobSchema).min(1).max(20),
      api_base_url: z.string().optional(),
    },
  },
  async ({
    jobs,
    api_base_url,
  }: {
    jobs: z.infer<typeof perturbPlanJobSchema>[];
    api_base_url?: string;
  }) => {
    const api = normalizeApiBase(api_base_url, defaultApiBase);
    try {
      const meta = await fetchMeta(api);
      if (!meta.perturb_ready) {
        return {
          content: [
            {
              type: "text" as const,
              text: "NOT READY: use spatial_viewer_wait_ready(require_perturb=true).",
            },
          ],
        };
      }
    } catch {
      /* continue */
    }
    const parts: string[] = [];
    let i = 0;
    for (const job of jobs) {
      i += 1;
      const built = await buildPerturbScopeApiBody(
        fetchImpl,
        api,
        job.scope,
        job.cell_type_label,
        job.cluster_id,
        job.cell_indices,
      );
      if (built.error) {
        parts.push(`Job ${i} (${job.gene}): SKIP — ${built.error}`);
        continue;
      }
      const reqBody: Record<string, unknown> = {
        gene: job.gene.trim(),
        desired_expr: job.desired_expr ?? 0,
        scope: built.scope,
      };
      if (job.n_propagation != null) reqBody.n_propagation = job.n_propagation;
      const t0 = Date.now();
      try {
        const res = await fetchImpl(`${api}/api/perturb/summary`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(reqBody),
        });
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        if (!res.ok) {
          parts.push(`Job ${i} (${job.gene}): Error ${res.status} (${elapsed}s) ${await res.text()}`);
          continue;
        }
        const data = await res.json();
        parts.push(`Job ${i} (${job.gene}) ${elapsed}s:\n${JSON.stringify(data, null, 2)}`);
      } catch (e: any) {
        parts.push(`Job ${i} (${job.gene}): ${e.message}`);
      }
    }
    return { content: [{ type: "text" as const, text: parts.join("\n\n---\n\n") }] };
  },
);

server.registerTool(
  "spatial_viewer_export_feather",
  {
    description:
      "POST /api/perturb/export_feather — returns Feather bytes (simulated expression). Response is summarized (byte length); use curl against the same endpoint to save a file locally.",
    inputSchema: {
      gene: z.string(),
      desired_expr: z.number().optional(),
      scope: z.enum(["all", "selection", "cell_type", "cluster"]).optional(),
      cell_type_label: z.string().optional(),
      cluster_id: z.number().int().optional(),
      cell_indices: z.array(z.number().int().nonnegative()).optional(),
      n_propagation: z.number().int().min(1).max(32).optional(),
      api_base_url: z.string().optional(),
    },
  },
  async (args: Record<string, unknown> & { api_base_url?: string }) => {
    const api = normalizeApiBase(String(args.api_base_url ?? "").trim() || undefined, defaultApiBase);
    const gene = String(args.gene ?? "").trim();
    if (!gene) return { content: [{ type: "text" as const, text: "gene is required." }] };
    try {
      const meta = await fetchMeta(api);
      if (!meta.perturb_ready) {
        return {
          content: [
            { type: "text" as const, text: "perturb_ready is false — use spatial_viewer_wait_ready(require_perturb=true)." },
          ],
        };
      }
    } catch {
      /* continue */
    }
    const cellIdx =
      Array.isArray(args.cell_indices) && args.cell_indices.length > 0
        ? args.cell_indices.filter((x: unknown): x is number => typeof x === "number")
        : undefined;
    const built = await buildPerturbScopeApiBody(
      fetchImpl,
      api,
      typeof args.scope === "string" ? args.scope : undefined,
      typeof args.cell_type_label === "string" ? args.cell_type_label : undefined,
      typeof args.cluster_id === "number" ? args.cluster_id : undefined,
      cellIdx,
    );
    if (built.error) {
      return { content: [{ type: "text" as const, text: `BAD SCOPE: ${built.error}` }] };
    }
    const reqBody: Record<string, unknown> = {
      gene,
      desired_expr: typeof args.desired_expr === "number" ? args.desired_expr : 0,
      scope: built.scope,
    };
    if (args.n_propagation != null) reqBody.n_propagation = args.n_propagation;
    const t0 = Date.now();
    try {
      const res = await fetchImpl(`${api}/api/perturb/export_feather`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
      });
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      if (!res.ok) {
        return {
          content: [{ type: "text" as const, text: `Error ${res.status} (${elapsed}s): ${await res.text()}` }],
        };
      }
      const buf = await res.arrayBuffer();
      return {
        content: [
          {
            type: "text" as const,
            text:
              `Feather export OK (${elapsed}s): ${buf.byteLength} bytes for gene ${gene}. ` +
              `Save with: curl -sS -X POST ${api}/api/perturb/export_feather -H 'Content-Type: application/json' ` +
              `-d '${JSON.stringify(reqBody).replace(/'/g, "'\\''")}' -o simulated.feather`,
          },
        ],
        structuredContent: { gene, bytes: buf.byteLength, elapsed_s: Number(elapsed) },
      };
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Fetch error: ${e.message}` }] };
    }
  },
);

registerAppResource(
  server,
  "Spatial viewer UI",
  RESOURCE_URI,
  {
    description: "Deck.gl spatial + betadata viewer",
    _meta: {
      ui: {
        csp: { connectDomains: connectDomainList() },
      },
    },
  },
  async () => {
    const text = await readMcpHtml();
    const domains = connectDomainList();
    return {
      contents: [
        {
          uri: RESOURCE_URI,
          mimeType: RESOURCE_MIME_TYPE,
          text,
          _meta: {
            ui: {
              csp: { connectDomains: domains },
            },
          },
        },
      ],
    };
  },
);

  return server;
}

export async function startSpatialViewerMcpStdio(): Promise<void> {
  const defaultApiBase = process.env.SPATIAL_VIEWER_API_BASE?.trim() || "http://127.0.0.1:8080";
  const server = createSpatialViewerMcpServer({
    fetch: globalThis.fetch,
    defaultApiBase,
    connectDomainList: () =>
      makeConnectDomainList(defaultApiBase, process.env.SPATIAL_VIEWER_CONNECT_ORIGINS),
    readMcpHtml: () => readFile(mcpHtmlPath, "utf-8"),
  });
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

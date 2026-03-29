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

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, "..");
const mcpHtmlPath = path.join(rootDir, "dist", "mcp-app.html");

const RESOURCE_URI = "ui://spacetravlr/spatial-viewer.html";

const defaultApiBase =
  process.env.SPATIAL_VIEWER_API_BASE?.trim() || "http://127.0.0.1:8080";

function connectOrigin(apiBase: string): string {
  try {
    return new URL(apiBase).origin;
  } catch {
    return "http://127.0.0.1:8080";
  }
}

function connectDomainList(): string[] {
  const base = connectOrigin(defaultApiBase);
  const extra = process.env.SPATIAL_VIEWER_CONNECT_ORIGINS?.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (!extra?.length) return [base];
  return [...new Set([base, ...extra])];
}

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
    .enum(["expression", "betadata", "perturb"])
    .optional()
    .describe("Active color mode in the viewer"),
  apply_expression: z
    .boolean()
    .optional()
    .describe("If true with expression_gene, run Load color after setting gene"),
  focus_gene_context: z
    .string()
    .optional()
    .describe("Focus gene for interaction / LR context panel"),
  status_message: z
    .string()
    .optional()
    .describe("Short message shown in the viewer status bar (LLM narration)"),
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface MetaSnapshot {
  n_obs: number;
  n_vars: number;
  dataset_ready: boolean;
  perturb_ready: boolean;
  perturb_loading: boolean;
  perturb_progress_percent?: number;
  perturb_progress_label?: string;
  perturb_error?: string;
  cell_type_categories?: string[];
  cell_type_column?: string;
  cluster_annot?: string;
  adata_path?: string;
  betadata_dir?: string;
  network_loaded?: boolean;
  network_species?: string;
  betadata_row_id?: string;
  [k: string]: unknown;
}

async function fetchMeta(api: string): Promise<MetaSnapshot> {
  const res = await fetch(`${api}/api/meta`);
  if (!res.ok) throw new Error(`/api/meta ${res.status}: ${await res.text()}`);
  return (await res.json()) as MetaSnapshot;
}

function formatStatus(m: MetaSnapshot): string {
  const lines: string[] = [];
  if (!m.dataset_ready) {
    lines.push("Dataset: not loaded yet");
  } else {
    lines.push(
      `Dataset: ready — ${m.n_obs} cells, ${m.n_vars} genes` +
        (m.adata_path ? ` (${m.adata_path})` : ""),
    );
    if (m.cluster_annot) lines.push(`  clusters: ${m.cluster_annot}`);
    if (m.cell_type_categories?.length)
      lines.push(`  cell types: ${m.cell_type_categories.join(", ")}`);
    if (m.network_loaded) lines.push(`  GRN: ${m.network_species ?? "loaded"}`);
    if (m.betadata_dir) lines.push(`  betadata: ${m.betadata_dir}`);
    if (m.betadata_row_id) lines.push(`  betadata row id: ${m.betadata_row_id}`);
  }

  if (m.perturb_loading) {
    const pct = m.perturb_progress_percent ?? 0;
    const label = m.perturb_progress_label ?? "";
    lines.push(`Perturbation: loading ${pct}% — ${label}`);
  } else if (m.perturb_error) {
    lines.push(`Perturbation: ERROR — ${m.perturb_error}`);
  } else if (m.perturb_ready) {
    lines.push("Perturbation: ready");
  } else {
    lines.push("Perturbation: not configured (no --run-toml)");
  }
  return lines.join("\n");
}

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
    const api =
      (args.api_base_url && String(args.api_base_url).trim()) || defaultApiBase;
    const structured = {
      api_base_url: api.replace(/\/$/, ""),
      adata_path: args.adata_path,
      layer: args.layer ?? "",
      cluster_annot: args.cluster_annot ?? "",
      network_dir: args.network_dir ?? "",
      run_toml: args.run_toml ?? "",
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
      api_base_url: defaultApiBase.replace(/\/$/, ""),
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
      "Set run_umap_quiver=true to also compute the UMAP quiver on all cells (uncheck limit_clusters in the UI first).",
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
      api_base_url: defaultApiBase.replace(/\/$/, ""),
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
      api_base_url: defaultApiBase.replace(/\/$/, ""),
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
      "Update the open spatial viewer (gene, color mode, status). Call after show_spatial_viewer. Same UI resource.",
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
      api_base_url: defaultApiBase.replace(/\/$/, ""),
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
    const api = (api_base_url?.trim() || defaultApiBase).replace(/\/$/, "");
    try {
      const res = await fetch(`${api}/api/cluster/mean_expression`, {
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
    const api = (api_base_url?.trim() || defaultApiBase).replace(/\/$/, "");
    try {
      const res = await fetch(`${api}/api/meta/label_clusters`, {
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
      scope: z.enum(["all", "cell_type", "cluster"]).optional().describe("Perturbation scope"),
      cell_type_label: z.string().optional().describe("Cell type category when scope=cell_type"),
      cluster_id: z.number().int().optional().describe("Cluster ID when scope=cluster"),
      n_propagation: z.number().int().min(1).max(32).optional().describe("GRN propagation depth"),
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ gene, desired_expr, scope, cell_type_label, cluster_id, n_propagation, api_base_url }: {
    gene: string; desired_expr?: number; scope?: string; cell_type_label?: string;
    cluster_id?: number; n_propagation?: number; api_base_url?: string;
  }) => {
    const api = (api_base_url?.trim() || defaultApiBase).replace(/\/$/, "");

    try {
      const meta = await fetchMeta(api);
      if (!meta.perturb_ready) {
        const hint = meta.perturb_loading
          ? `Perturbation runtime is still loading (${meta.perturb_progress_percent ?? 0}% — ${meta.perturb_progress_label ?? ""}). Use spatial_viewer_wait_ready(require_perturb=true) first.`
          : "Perturbation is not configured. Start the server with --run-toml.";
        return { content: [{ type: "text" as const, text: `NOT READY: ${hint}` }] };
      }
    } catch { /* server may not have /api/meta yet — try the summary anyway */ }

    const s = scope ?? "all";
    let scopeObj: any;
    if (s === "cell_type" && cell_type_label != null && String(cell_type_label).trim() !== "") {
      scopeObj = { type: "cell_type_name", name: String(cell_type_label).trim() };
    } else if (s === "cluster" && cluster_id != null) {
      scopeObj = { type: "cluster", cluster_id };
    } else {
      scopeObj = { type: "all" };
    }
    const reqBody: any = { gene, desired_expr: desired_expr ?? 0, scope: scopeObj };
    if (n_propagation != null) reqBody.n_propagation = n_propagation;
    const t0 = Date.now();
    try {
      const res = await fetch(`${api}/api/perturb/summary`, {
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
  "spatial_viewer_get_meta",
  {
    description:
      "Get the current dataset metadata (n_obs, n_vars, cluster_annot, cell_type_categories, perturb_ready, etc.). Returns both raw JSON and a human-readable status summary.",
    inputSchema: {
      api_base_url: z.string().optional().describe("API base URL"),
    },
  },
  async ({ api_base_url }: { api_base_url?: string }) => {
    const api = (api_base_url?.trim() || defaultApiBase).replace(/\/$/, "");
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
    const api = (api_base_url?.trim() || defaultApiBase).replace(/\/$/, "");
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
    const api = (api_base_url?.trim() || defaultApiBase).replace(/\/$/, "");
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
    const api = (api_base_url?.trim() || defaultApiBase).replace(/\/$/, "");
    try {
      const res = await fetch(`${api}/api/cancel`, {
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
    const text = await readFile(mcpHtmlPath, "utf-8");
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

const transport = new StdioServerTransport();
await server.connect(transport);

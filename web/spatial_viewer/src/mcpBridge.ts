import {
  App,
  PostMessageTransport,
  applyDocumentTheme,
  applyHostFonts,
  applyHostStyleVariables,
} from "@modelcontextprotocol/ext-apps";

export type McpOpenSession = {
  api_base_url: string;
  adata_path: string;
  layer: string;
  cluster_annot: string;
  network_dir: string;
  run_toml: string;
};

export type McpCaptureRequest = {
  max_width?: number;
  caption: string;
};

export type McpPerturbRunRequest = {
  gene: string;
  desired_expr: number;
  scope: "all" | "selection" | "cell_type" | "cluster";
  cell_type_label: string;
  cluster_id: number;
  n_propagation?: number;
  push_summary_to_chat: boolean;
  /** After Δ preview, run UMAP quiver with current transition-panel settings (leave limit_clusters off for all cells). */
  run_umap_quiver?: boolean;
};

export type McpCollectInteractionsRequest = {
  aggregate: string;
  filter: "cell_type" | "cluster";
  cell_type: string;
  cluster_id: number;
  max_genes: number;
  push_summary_to_chat: boolean;
};

const pendingControl: Record<string, unknown>[] = [];
let controlSink: ((args: Record<string, unknown>) => void) | null = null;

const pendingCapture: McpCaptureRequest[] = [];
let captureSink: ((req: McpCaptureRequest) => void | Promise<void>) | null =
  null;

const pendingPerturbRun: McpPerturbRunRequest[] = [];
let perturbRunSink: ((req: McpPerturbRunRequest) => void | Promise<void>) | null =
  null;

const pendingCollect: McpCollectInteractionsRequest[] = [];
let collectInteractionsSink:
  | ((req: McpCollectInteractionsRequest) => void | Promise<void>)
  | null = null;

function emitControl(args: Record<string, unknown>) {
  if (controlSink) controlSink(args);
  else pendingControl.push({ ...args });
}

export function attachMcpControlSink(
  fn: (args: Record<string, unknown>) => void,
): void {
  controlSink = fn;
  for (const p of pendingControl) fn(p);
  pendingControl.length = 0;
}

function emitCapture(req: McpCaptureRequest) {
  if (captureSink) void Promise.resolve(captureSink(req));
  else pendingCapture.push(req);
}

export function attachMcpCaptureSink(
  fn: (req: McpCaptureRequest) => void | Promise<void>,
): void {
  captureSink = fn;
  for (const p of pendingCapture) void Promise.resolve(fn(p));
  pendingCapture.length = 0;
}

function emitPerturbRun(req: McpPerturbRunRequest) {
  if (perturbRunSink) void Promise.resolve(perturbRunSink(req));
  else pendingPerturbRun.push(req);
}

export function attachMcpPerturbRunSink(
  fn: (req: McpPerturbRunRequest) => void | Promise<void>,
): void {
  perturbRunSink = fn;
  for (const p of pendingPerturbRun) void Promise.resolve(fn(p));
  pendingPerturbRun.length = 0;
}

function emitCollectInteractions(req: McpCollectInteractionsRequest) {
  if (collectInteractionsSink) void Promise.resolve(collectInteractionsSink(req));
  else pendingCollect.push(req);
}

export function attachMcpCollectInteractionsSink(
  fn: (req: McpCollectInteractionsRequest) => void | Promise<void>,
): void {
  collectInteractionsSink = fn;
  for (const p of pendingCollect) void Promise.resolve(fn(p));
  pendingCollect.length = 0;
}

function parsePerturbRunFromStructured(
  sc: Record<string, unknown>,
): McpPerturbRunRequest | null {
  if (sc._spatialTool !== "perturb_run") return null;
  const gene = typeof sc.gene === "string" ? sc.gene.trim() : "";
  if (!gene) return null;
  const de = sc.desired_expr;
  const desired_expr =
    typeof de === "number" && Number.isFinite(de) ? de : 0;
  const scopeRaw = typeof sc.scope === "string" ? sc.scope : "all";
  const scope =
    scopeRaw === "selection" ||
    scopeRaw === "cell_type" ||
    scopeRaw === "cluster"
      ? (scopeRaw as McpPerturbRunRequest["scope"])
      : "all";
  const cell_type_label =
    typeof sc.cell_type_label === "string" ? sc.cell_type_label : "";
  const cid = sc.cluster_id;
  const cluster_id =
    typeof cid === "number" && Number.isFinite(cid) ? Math.trunc(cid) : 0;
  const np = sc.n_propagation;
  const n_propagation =
    typeof np === "number" && Number.isFinite(np)
      ? Math.min(32, Math.max(1, Math.trunc(np)))
      : undefined;
  const push_summary_to_chat = sc.push_summary_to_chat === true;
  const run_umap_quiver = sc.run_umap_quiver === true;
  return {
    gene,
    desired_expr,
    scope,
    cell_type_label,
    cluster_id,
    n_propagation,
    push_summary_to_chat,
    run_umap_quiver,
  };
}

function parseCollectInteractionsFromStructured(
  sc: Record<string, unknown>,
): McpCollectInteractionsRequest | null {
  if (sc._spatialTool !== "collect_interactions") return null;
  const aggregate =
    typeof sc.aggregate === "string" && sc.aggregate.trim()
      ? sc.aggregate.trim()
      : "mean";
  const filterRaw = typeof sc.filter === "string" ? sc.filter : "cell_type";
  const filter: McpCollectInteractionsRequest["filter"] =
    filterRaw === "cluster" ? "cluster" : "cell_type";
  const cell_type =
    typeof sc.cell_type === "string" ? sc.cell_type.trim() : "";
  const cid = sc.cluster_id;
  const cluster_id =
    typeof cid === "number" && Number.isFinite(cid) ? Math.trunc(cid) : 0;
  const mg = sc.max_genes;
  const max_genes =
    typeof mg === "number" && Number.isFinite(mg)
      ? Math.min(4096, Math.max(1, Math.trunc(mg)))
      : 2048;
  const push_summary_to_chat = sc.push_summary_to_chat === true;
  return {
    aggregate,
    filter,
    cell_type,
    cluster_id,
    max_genes,
    push_summary_to_chat,
  };
}

function parseCaptureFromStructured(
  sc: Record<string, unknown>,
): McpCaptureRequest | null {
  if (sc._spatialTool !== "capture") return null;
  const mw = sc.max_width;
  const max_width =
    typeof mw === "number" && Number.isFinite(mw) && mw > 0
      ? Math.min(4096, Math.trunc(mw))
      : undefined;
  const cap = typeof sc.caption === "string" ? sc.caption : "";
  return { max_width, caption: cap };
}

export function isMcpApp(): boolean {
  try {
    return (
      window.location.origin === "null" && window.parent !== window.self
    );
  } catch {
    return false;
  }
}

function mergeOpenPayload(
  structured: Record<string, unknown> | undefined,
  args: Record<string, unknown>,
): McpOpenSession | null {
  const s = structured ?? {};
  const adata =
    (typeof s.adata_path === "string" && s.adata_path) ||
    (typeof args.adata_path === "string" && args.adata_path);
  if (!adata) return null;
  const api =
    (typeof s.api_base_url === "string" && s.api_base_url.trim()) ||
    (typeof args.api_base_url === "string" && args.api_base_url.trim()) ||
    "http://127.0.0.1:8080";
  return {
    api_base_url: api.replace(/\/$/, ""),
    adata_path: String(adata),
    layer:
      typeof s.layer === "string"
        ? s.layer
        : typeof args.layer === "string"
          ? args.layer
          : "",
    cluster_annot:
      typeof s.cluster_annot === "string"
        ? s.cluster_annot
        : typeof args.cluster_annot === "string"
          ? args.cluster_annot
          : "",
    network_dir:
      typeof s.network_dir === "string"
        ? s.network_dir
        : typeof args.network_dir === "string"
          ? args.network_dir
          : "",
    run_toml:
      typeof s.run_toml === "string"
        ? s.run_toml
        : typeof args.run_toml === "string"
          ? args.run_toml
          : "",
  };
}

const STREAM_KEYS = [
  "adata_path",
  "layer",
  "cluster_annot",
  "network_dir",
  "run_toml",
  "api_base_url",
] as const;

export async function bootstrapMcp(): Promise<{
  apiBase: string;
  mcpApp: App | null;
  openSession: McpOpenSession | null;
}> {
  if (!isMcpApp()) {
    return { apiBase: "", mcpApp: null, openSession: null };
  }

  let streamBar: HTMLDivElement | null = null;
  let streamFill: HTMLDivElement | null = null;
  const ensureStreamUi = () => {
    if (streamBar) return;
    streamBar = document.createElement("div");
    streamBar.id = "mcp-stream-bar";
    streamBar.style.cssText =
      "position:fixed;bottom:0;left:0;right:0;height:4px;background:#222;z-index:99999;";
    streamFill = document.createElement("div");
    streamFill.style.cssText = "height:100%;width:0%;background:#388bfd;";
    streamBar.appendChild(streamFill);
    document.body.appendChild(streamBar);
  };

  const updateStream = (args: Record<string, unknown> | undefined) => {
    if (!args) return;
    let n = 0;
    for (const k of STREAM_KEYS) {
      const v = args[k];
      if (v != null && String(v).trim().length > 0) n++;
    }
    const pct = (n / STREAM_KEYS.length) * 100;
    ensureStreamUi();
    if (streamFill) streamFill.style.width = `${pct}%`;
  };

  const removeStreamUi = () => {
    streamBar?.remove();
    streamBar = null;
    streamFill = null;
  };

  const app = new App({ name: "spatial-viewer", version: "1.0.0" });

  let resolvedOpen: McpOpenSession | null = null;
  let resolveOpen!: (v: McpOpenSession | null) => void;
  const openPromise = new Promise<McpOpenSession | null>((r) => {
    resolveOpen = r;
  });

  const tryResolveOpen = (
    structured: Record<string, unknown> | undefined,
    args: Record<string, unknown>,
  ) => {
    if (resolvedOpen) return;
    const m = mergeOpenPayload(structured, args);
    if (m) {
      resolvedOpen = m;
      removeStreamUi();
      resolveOpen(m);
    }
  };

  app.onhostcontextchanged = (ctx) => {
    if (ctx.theme) applyDocumentTheme(ctx.theme);
    if (ctx.styles?.variables)
      applyHostStyleVariables(ctx.styles.variables);
    if (ctx.styles?.css?.fonts) applyHostFonts(ctx.styles.css.fonts);
  };

  app.onteardown = async (_params, _extra) => ({});

  app.ontoolinputpartial = (p) => updateStream(p.arguments);

  app.ontoolinput = (p) => {
    const args = (p.arguments ?? {}) as Record<string, unknown>;
    updateStream(args);
    const hasPath =
      typeof args.adata_path === "string" && args.adata_path.trim() !== "";
    const controlLike =
      args.expression_gene != null ||
      args.color_source != null ||
      args.apply_expression === true ||
      typeof args.focus_gene_context === "string" ||
      typeof args.status_message === "string";
    if (!hasPath && controlLike) {
      emitControl(args);
      return;
    }
    tryResolveOpen(undefined, args);
  };

  app.ontoolresult = (r) => {
    const sc = r.structuredContent as Record<string, unknown> | undefined;
    if (!sc) return;
    if (sc._spatialTool === "control") {
      emitControl(sc);
      return;
    }
    const cap = parseCaptureFromStructured(sc);
    if (cap) {
      emitCapture(cap);
      return;
    }
    const pr = parsePerturbRunFromStructured(sc);
    if (pr) {
      emitPerturbRun(pr);
      return;
    }
    const ci = parseCollectInteractionsFromStructured(sc);
    if (ci) {
      emitCollectInteractions(ci);
      return;
    }
    tryResolveOpen(sc, {});
  };

  await app.connect(
    new PostMessageTransport(window.parent, window.parent),
  );

  const openSession = await Promise.race([
    openPromise,
    new Promise<McpOpenSession | null>((r) =>
      setTimeout(() => r(null), 30_000),
    ),
  ]);

  removeStreamUi();

  return {
    apiBase:
      openSession?.api_base_url.replace(/\/$/, "") ||
      "http://127.0.0.1:8080",
    mcpApp: app,
    openSession,
  };
}

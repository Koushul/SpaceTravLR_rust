let globalApiBase = "";

export function setGlobalApiBase(b: string): void {
  globalApiBase = b;
}

export function getGlobalApiBase(): string {
  return globalApiBase;
}

export function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const p = path.startsWith("/") ? path : `/${path}`;
  const b = globalApiBase.replace(/\/$/, "");
  return b ? `${b}${p}` : p;
}

export async function fetchF32(path: string): Promise<Float32Array> {
  const r = await fetch(apiUrl(path));
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}`);
  }
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

export async function postF32(path: string, body: unknown): Promise<Float32Array> {
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

export async function fetchU32(path: string): Promise<Uint32Array> {
  const r = await fetch(apiUrl(path));
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}`);
  }
  const buf = await r.arrayBuffer();
  return new Uint32Array(buf);
}

export interface ViewerUiStatePayload {
  color_source: string | null;
  expr_gene: string | null;
  perturb_gene: string | null;
  perturb_expr: number | null;
  perturb_scope: string | null;
  perturb_cell_type: string | null;
  perturb_cluster_id: number | null;
  interaction_sender_index: number | null;
  pair_cell_a: number | null;
  pair_cell_b: number | null;
}

export async function postViewerUiState(body: ViewerUiStatePayload): Promise<void> {
  try {
    await fetch(apiUrl("/api/viewer_state"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    /* ignore — localhost sync best-effort */
  }
}

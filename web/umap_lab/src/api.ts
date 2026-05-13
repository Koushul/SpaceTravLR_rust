export type LoadResponse = {
  path: string
  n_cells: number
  n_pca_available: number
  color_column: string | null
  color_categories: string[]
  color_codes: number[]
  ef_construction: number
  obs_columns: string[]
}

export type UmapResponse = {
  x: number[]
  y: number[]
  timings_sec: [string, number][]
}

export async function apiLoad(body: {
  path: string
  n_top_hvg?: number
  n_pca_components?: number
}): Promise<LoadResponse> {
  const res = await fetch("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<LoadResponse>
}

export async function apiUmap(body: Record<string, number | undefined>): Promise<UmapResponse> {
  const res = await fetch("/api/umap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<UmapResponse>
}

export type LeidenResponse = {
  labels: string[]
  categories: string[]
  codes: number[]
  n_clusters: number
  elapsed_sec: number
}

export async function apiLeiden(body: { resolution: number }): Promise<LeidenResponse> {
  const res = await fetch("/api/leiden", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<LeidenResponse>
}

export async function apiLeidenSubcluster(body: {
  parent_code: number
  resolution: number
}): Promise<LeidenResponse> {
  const res = await fetch("/api/leiden/subcluster", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<LeidenResponse>
}

export async function apiLeidenReset(): Promise<LeidenResponse> {
  const res = await fetch("/api/leiden/reset", { method: "POST" })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<LeidenResponse>
}

export type GeneExpressionResponse = {
  gene: string
  values: number[]
  vmin: number
  vmax: number
}

export async function apiGene(body: { gene: string }): Promise<GeneExpressionResponse> {
  const res = await fetch("/api/gene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<GeneExpressionResponse>
}

export type ColorByResponse = {
  column: string
  categories: string[]
  codes: number[]
}

export async function apiColorBy(body: { column: string }): Promise<ColorByResponse> {
  const res = await fetch("/api/color_by", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<ColorByResponse>
}

export type MaltResponse = {
  outdir: string
  csv_path: string
  csv_columns: string[]
  elapsed_sec: number
}

export async function apiMalt(body: {
  reference_path: string
  groupby?: string
  outdir?: string
  no_leiden_map?: boolean
}): Promise<MaltResponse> {
  const res = await fetch("/api/malt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<MaltResponse>
}

export type MaltOptimizedResponse = {
  column: string
  categories: string[]
  codes: number[]
  n_subsample: number
  n_total: number
  min_cluster_count: number
  elapsed_sec: number
}

export async function apiMaltOptimized(body: {
  reference_path: string
  groupby?: string
  no_leiden_map?: boolean
}): Promise<MaltOptimizedResponse> {
  const res = await fetch("/api/malt_optimized", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<MaltOptimizedResponse>
}

export async function apiExportCsv(annotations: Record<string, string>): Promise<Blob> {
  const res = await fetch("/api/export_csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ annotations }),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.blob()
}

export type LoadCsvResponse = {
  column: string
  categories: string[]
  codes: number[]
  n_matched: number
  n_missing: number
}

export async function apiLoadCsv(body: {
  csv_path: string
  column: string
}): Promise<LoadCsvResponse> {
  const res = await fetch("/api/load_csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || res.statusText)
  }
  return res.json() as Promise<LoadCsvResponse>
}

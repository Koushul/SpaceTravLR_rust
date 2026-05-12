export type LoadResponse = {
  path: string
  n_cells: number
  n_pca_available: number
  color_column: string | null
  color_categories: string[]
  color_codes: number[]
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

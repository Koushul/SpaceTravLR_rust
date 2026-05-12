import { Loader2Icon } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { apiLoad, apiLeiden, apiUmap, type LeidenResponse, type LoadResponse, type UmapResponse } from "@/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { drawScatter2d, findNearestPoint, hslForCategory, type ScatterState } from "@/drawScatter"

function clampMinDistSpread(minDist: number, spread: number): [number, number] {
  const md = Math.max(minDist, 1e-6)
  const sp = Math.max(spread, 1e-6)
  return [Math.min(md, sp), Math.max(sp, md)]
}

type Phase = "idle" | "loading_file" | "running_umap" | "running_leiden" | "ready" | "error"

export default function App() {
  const [path, setPath] = useState("")
  const [phase, setPhase] = useState<Phase>("idle")
  const [error, setError] = useState<string | null>(null)
  const [meta, setMeta] = useState<LoadResponse | null>(null)
  const [umap, setUmap] = useState<UmapResponse | null>(null)

  const [nNeighbors, setNNeighbors] = useState(15)
  const [minDist, setMinDist] = useState(0.5)
  const [nEpochs, setNEpochs] = useState(500)
  const [efConstruction, setEfConstruction] = useState(200)
  const [nPca, setNPca] = useState(50)
  const [spread, setSpread] = useState(1)
  const [lr, setLr] = useState(1)
  const [nTopHvg, setNTopHvg] = useState(2000)

  const [leidenRes, setLeidenRes] = useState(0.5)
  const [leiden, setLeiden] = useState<LeidenResponse | null>(null)
  const [useLeidenColors, setUseLeidenColors] = useState(true)

  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; cluster: string; count: number } | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const scatterStateRef = useRef<ScatterState | null>(null)

  const activeCategories = leiden && useLeidenColors ? leiden.categories : (meta?.color_categories ?? [])
  const activeCodes = leiden && useLeidenColors ? leiden.codes : (meta?.color_codes ?? [])
  const activeNCat = activeCategories.length

  const clusterCounts = useRef<Map<number, number>>(new Map())
  useEffect(() => {
    const m = new Map<number, number>()
    for (const c of activeCodes) {
      m.set(c, (m.get(c) ?? 0) + 1)
    }
    clusterCounts.current = m
  }, [activeCodes])

  const runLeiden = useCallback(async (resolution: number) => {
    if (!umap) return
    setPhase("running_leiden")
    try {
      const res = await apiLeiden({ resolution })
      setLeiden(res)
      setSelectedCluster(null)
      setPhase("ready")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [umap])

  const runUmap = useCallback(async () => {
    if (!meta) return
    const [md2, sp2] = clampMinDistSpread(minDist, spread)
    if (md2 !== minDist) setMinDist(md2)
    if (sp2 !== spread) setSpread(sp2)
    setPhase("running_umap")
    setError(null)
    try {
      const res = await apiUmap({
        n_neighbors: nNeighbors,
        min_dist: md2,
        n_epochs: nEpochs,
        ef_construction: efConstruction,
        n_pca_components: Math.min(nPca, meta.n_pca_available),
        spread: sp2,
        umap_learning_rate: lr,
      })
      setUmap(res)
      setSelectedCluster(null)
      setPhase("ready")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [
    meta,
    nNeighbors,
    minDist,
    nEpochs,
    efConstruction,
    nPca,
    spread,
    lr,
  ])

  const triggerUmap = useCallback(() => {
    if (!meta) return
    void runUmap()
  }, [meta, runUmap])

  const loadFile = async () => {
    const p = path.trim()
    if (!p) {
      setError("Enter a path to an .h5ad file")
      setPhase("error")
      return
    }
    setPhase("loading_file")
    setError(null)
    setUmap(null)
    setMeta(null)
    setLeiden(null)
    try {
      const m = await apiLoad({
        path: p,
        n_top_hvg: nTopHvg,
        n_pca_components: nPca,
      })
      setMeta(m)
      setNPca((v) => Math.min(v, m.n_pca_available))
      setPhase("ready")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }

  const didInitialRun = useRef(false)
  useEffect(() => {
    if (!meta?.path) {
      didInitialRun.current = false
      return
    }
    if (didInitialRun.current) return
    didInitialRun.current = true
    void runUmap()
  }, [meta?.path, runUmap])

  const prevUmapRef = useRef<UmapResponse | null>(null)
  useEffect(() => {
    if (!umap || umap === prevUmapRef.current) return
    prevUmapRef.current = umap
    void runLeiden(leidenRes)
  }, [umap, leidenRes, runLeiden])

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap || !umap) return

    const paint = () => {
      const codes = activeCodes
      const nCat = activeNCat
      const n = umap.x.length
      const cc = new Uint32Array(n)
      for (let i = 0; i < n; i++) {
        cc[i] = codes[i] ?? 0
      }
      const state = drawScatter2d(canvas, umap.x, umap.y, cc, nCat, selectedCluster)
      scatterStateRef.current = state
    }

    requestAnimationFrame(paint)
    const ro = new ResizeObserver(() => requestAnimationFrame(paint))
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [umap, activeCodes, activeNCat, selectedCluster])

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const state = scatterStateRef.current
    if (!canvas || !state || !umap) {
      setHoverInfo(null)
      return
    }
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top

    const idx = findNearestPoint(state, umap.x, umap.y, cx, cy, 12)
    if (idx == null) {
      setHoverInfo(null)
      return
    }
    const cluster = activeCodes[idx] ?? 0
    const label = activeCategories[cluster] ?? String(cluster)
    const count = clusterCounts.current.get(cluster) ?? 0
    setHoverInfo({ x: e.clientX, y: e.clientY, cluster: label, count })
  }, [umap, activeCodes, activeCategories])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const state = scatterStateRef.current
    if (!canvas || !state || !umap) return
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top

    const idx = findNearestPoint(state, umap.x, umap.y, cx, cy, 12)
    if (idx == null) {
      setSelectedCluster(null)
      return
    }
    const cluster = activeCodes[idx] ?? 0
    setSelectedCluster((prev) => (prev === cluster ? null : cluster))
  }, [umap, activeCodes])

  const handleCanvasLeave = useCallback(() => {
    setHoverInfo(null)
  }, [])

  const busy = phase === "loading_file" || phase === "running_umap"
  const leidenBusy = phase === "running_leiden"

  return (
    <div className="flex h-full min-h-svh min-h-0 flex-col gap-4 p-4 lg:flex-row lg:gap-6">
      <div className="flex w-full shrink-0 flex-col gap-4 lg:w-[380px]">
        <Card>
          <CardHeader>
            <CardTitle>UMAP lab</CardTitle>
            <CardDescription>
              Rust umap-rs + HNSW (same path as{" "}
              <code className="text-xs">rust_preprocess</code>). Load PCA from{" "}
              <code className="text-xs">obsm[&apos;X_pca&apos;]</code> or compute once.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="h5ad">AnnData path (.h5ad)</Label>
              <Input
                id="h5ad"
                placeholder="/path/to/file.h5ad"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                disabled={busy}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="hvg">n_top_hvg (only if PCA is computed)</Label>
              <Input
                id="hvg"
                type="number"
                min={500}
                max={8000}
                step={100}
                value={nTopHvg}
                onChange={(e) => setNTopHvg(Number(e.target.value) || 2000)}
                disabled={busy}
              />
            </div>
            <Button
              type="button"
              onClick={() => void loadFile()}
              disabled={busy}
              className="w-full"
            >
              {phase === "loading_file" ? (
                <>
                  <Loader2Icon data-icon="inline-start" className="animate-spin" />
                  Loading…
                </>
              ) : (
                "Load & first UMAP"
              )}
            </Button>
            {error ? (
              <p className="text-destructive text-sm break-words">{error}</p>
            ) : null}
          </CardContent>
          {meta ? (
            <CardFooter className="flex flex-col items-start gap-2 border-t">
              <div className="text-muted-foreground flex flex-wrap gap-2 text-xs">
                <Badge variant="secondary">{meta.n_cells.toLocaleString()} cells</Badge>
                <Badge variant="secondary">
                  PCA {meta.n_pca_available}D
                  {meta.color_column ? (
                    <>
                      {" "}
                      · {meta.color_column}
                    </>
                  ) : null}
                </Badge>
              </div>
            </CardFooter>
          ) : null}
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>UMAP parameters</CardTitle>
            <CardDescription>
              UMAP re-runs on pointer up. <code className="text-xs">min_dist</code> must be ≤{" "}
              <code className="text-xs">spread</code> (values auto-adjust to satisfy umap-rs).
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <ParamSlider
              label={`n_neighbors (${nNeighbors})`}
              value={nNeighbors}
              min={2}
              max={150}
              step={1}
              onChange={setNNeighbors}
              onCommit={triggerUmap}
              disabled={!meta || busy}
            />
            <ParamSlider
              label={`min_dist (${minDist.toFixed(2)})`}
              value={minDist}
              min={0.01}
              max={0.99}
              step={0.01}
              onChange={setMinDist}
              onCommit={triggerUmap}
              disabled={!meta || busy}
            />
            <ParamSlider
              label={`n_epochs (${nEpochs})`}
              value={nEpochs}
              min={50}
              max={1200}
              step={10}
              onChange={setNEpochs}
              onCommit={triggerUmap}
              disabled={!meta || busy}
            />
            <ParamSlider
              label={`ef_construction (${efConstruction})`}
              value={efConstruction}
              min={16}
              max={400}
              step={4}
              onChange={setEfConstruction}
              onCommit={triggerUmap}
              disabled={!meta || busy}
            />
            <ParamSlider
              label={`n_pca_components (${Math.min(nPca, meta?.n_pca_available ?? nPca)})`}
              value={Math.min(nPca, meta?.n_pca_available ?? nPca)}
              min={2}
              max={Math.max(2, meta?.n_pca_available ?? 50)}
              step={1}
              onChange={(v) => setNPca(v)}
              onCommit={triggerUmap}
              disabled={!meta || busy}
            />
            <ParamSlider
              label={`spread (${spread.toFixed(2)})`}
              value={spread}
              min={0.1}
              max={1}
              step={0.02}
              onChange={setSpread}
              onCommit={triggerUmap}
              disabled={!meta || busy}
            />
            <ParamSlider
              label={`learning_rate (${lr.toFixed(2)})`}
              value={lr}
              min={0.1}
              max={5}
              step={0.05}
              onChange={setLr}
              onCommit={triggerUmap}
              disabled={!meta || busy}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Leiden clustering
              {leidenBusy ? <Loader2Icon className="size-4 animate-spin" /> : null}
            </CardTitle>
            <CardDescription>
              Community detection on the UMAP fuzzy graph. Re-runs on slider commit.
              {leiden ? (
                <span className="ml-1 font-medium">
                  {leiden.n_clusters} clusters · {leiden.elapsed_sec.toFixed(2)}s
                </span>
              ) : null}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <ParamSlider
              label={`resolution (${leidenRes.toFixed(2)})`}
              value={leidenRes}
              min={0.1}
              max={3.0}
              step={0.05}
              onChange={setLeidenRes}
              onCommit={() => void runLeiden(leidenRes)}
              disabled={!umap || busy || leidenBusy}
            />
            <div className="flex items-center gap-2">
              <Button
                variant={useLeidenColors ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  setUseLeidenColors(!useLeidenColors)
                  setSelectedCluster(null)
                }}
                disabled={!leiden}
              >
                {useLeidenColors ? "Showing Leiden colors" : "Showing file colors"}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          {phase === "running_umap" && meta ? (
            <Badge variant="outline" className="gap-1">
              <Loader2Icon className="size-3 animate-spin" />
              Re-running UMAP…
            </Badge>
          ) : null}
          {umap?.timings_sec?.length ? (
            <span className="text-muted-foreground text-xs">
              {umap.timings_sec
                .map(([k, s]) => `${k}: ${s.toFixed(2)}s`)
                .join(" · ")}
            </span>
          ) : null}
        </div>
        <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
          <div
            ref={wrapRef}
            className="border-border bg-card relative min-h-[min(72vh,720px)] w-full min-w-0 flex-1 overflow-hidden rounded-xl border"
          >
            <canvas
              ref={canvasRef}
              className="size-full block"
              onMouseMove={handleCanvasMouseMove}
              onClick={handleCanvasClick}
              onMouseLeave={handleCanvasLeave}
            />
            {hoverInfo ? (
              <div
                className="bg-popover text-popover-foreground pointer-events-none fixed z-50 rounded-md border px-3 py-1.5 text-xs shadow-md"
                style={{ left: hoverInfo.x + 12, top: hoverInfo.y - 8 }}
              >
                <span className="font-medium">Cluster {hoverInfo.cluster}</span>
                <span className="text-muted-foreground ml-2">{hoverInfo.count.toLocaleString()} pts</span>
              </div>
            ) : null}
          </div>

          {activeCategories.length > 0 && umap ? (
            <div className="mt-2 flex max-h-32 flex-wrap gap-x-3 gap-y-1 overflow-y-auto rounded-lg border p-2">
              {activeCategories.map((cat, i) => {
                const count = clusterCounts.current.get(i) ?? 0
                const isSelected = selectedCluster === i
                const isDimmed = selectedCluster != null && !isSelected
                return (
                  <button
                    key={`${cat}-${i}`}
                    className="flex items-center gap-1.5 rounded px-1 py-0.5 text-xs transition-opacity hover:bg-muted"
                    style={{ opacity: isDimmed ? 0.35 : 1 }}
                    onClick={() => setSelectedCluster((prev) => (prev === i ? null : i))}
                  >
                    <span
                      className="inline-block size-2.5 shrink-0 rounded-full"
                      style={{ background: hslForCategory(i, activeNCat) }}
                    />
                    <span className="truncate max-w-[120px]">{cat}</span>
                    <span className="text-muted-foreground tabular-nums">{count.toLocaleString()}</span>
                  </button>
                )
              })}
            </div>
          ) : null}
        </div>
        {!meta ? (
          <p className="text-muted-foreground text-sm">
            Start the server:{" "}
            <code className="text-xs">
              cargo run --features umap-lab --bin umap_lab
            </code>{" "}
            then <code className="text-xs">npm run dev</code> in{" "}
            <code className="text-xs">web/umap_lab</code> (API proxied to :8765).
          </p>
        ) : null}
      </div>
    </div>
  )
}

function ParamSlider(props: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  onCommit: () => void
  disabled?: boolean
}) {
  const { label, value, min, max, step, onChange, onCommit, disabled } = props
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs">{label}</Label>
      </div>
      <Slider
        disabled={disabled}
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(v) => {
          const n = typeof v === "number" ? v : (v[0] ?? value)
          onChange(n)
        }}
        onValueCommitted={onCommit}
      />
    </div>
  )
}

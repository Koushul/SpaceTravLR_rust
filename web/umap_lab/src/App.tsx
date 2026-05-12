import { Loader2Icon } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { apiLoad, apiUmap, type LoadResponse, type UmapResponse } from "@/api"
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
import { drawScatter2d } from "@/drawScatter"

function clampMinDistSpread(minDist: number, spread: number): [number, number] {
  const md = Math.max(minDist, 1e-6)
  const sp = Math.max(spread, 1e-6)
  return [Math.min(md, sp), Math.max(sp, md)]
}

type Phase = "idle" | "loading_file" | "running_umap" | "ready" | "error"

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

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)

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

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap || !umap) return

    const paint = () => {
      const codes = meta?.color_codes ?? []
      const nCat = meta?.color_categories?.length ?? 0
      const n = umap.x.length
      const cc = new Uint32Array(n)
      for (let i = 0; i < n; i++) {
        cc[i] = codes[i] ?? 0
      }
      drawScatter2d(canvas, umap.x, umap.y, cc, nCat)
    }

    paint()
    const ro = new ResizeObserver(() => paint())
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [umap, meta])

  const busy = phase === "loading_file" || phase === "running_umap"

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
              max={3}
              step={0.05}
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
        <div
          ref={wrapRef}
          className="border-border bg-card min-h-[min(72vh,720px)] w-full min-w-0 flex-1 overflow-hidden rounded-xl border"
        >
          <canvas ref={canvasRef} className="size-full block" />
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

import { CheckIcon, ChevronDownIcon, Loader2Icon, PencilIcon } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import {
  apiColorBy,
  apiGene,
  apiLoad,
  apiLoadCsv,
  apiLeiden,
  apiLeidenReset,
  apiLeidenSubcluster,
  apiMalt,
  apiUmap,
  type LeidenResponse,
  type LoadResponse,
  type MaltResponse,
  type UmapResponse,
} from "@/api"
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
import { buildClusterPalette, type ClusterPalette } from "@/clusterPalette"
import { drawScatter2d, findNearestPoint, hslForCategory, type ClusterCentroid, type ClusterLabel, type DrawScatterOpts, type ScatterState } from "@/drawScatter"

const EMPTY_CLUSTER_PALETTE: ClusterPalette = { fillCss: [], rgb: [] }

function SubtlePanel(props: {
  title: string
  children: React.ReactNode
}) {
  const { title, children } = props
  return (
    <details className="border-border/50 bg-card/25 text-foreground/90 group rounded-lg border">
      <summary className="text-muted-foreground hover:bg-muted/30 flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2 text-xs font-medium tracking-wide select-none [&::-webkit-details-marker]:hidden">
        <span>{title}</span>
        <ChevronDownIcon className="size-3.5 shrink-0 opacity-45 transition-transform duration-200 group-open:rotate-180" />
      </summary>
      <div className="border-border/35 space-y-3 border-t px-3 py-3">{children}</div>
    </details>
  )
}

function ParamHint(props: { text: string }) {
  const { text } = props
  return (
    <abbr
      className="text-muted-foreground hover:text-foreground ml-0.5 inline cursor-help align-super text-[10px] font-bold leading-none no-underline"
      title={text}
    >
      ?
    </abbr>
  )
}

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
  const [efConstruction, setEfConstruction] = useState(30)
  const [nPca, setNPca] = useState(50)
  const [spread, setSpread] = useState(0.5)
  const [lr, setLr] = useState(1)
  const [nTopHvg, setNTopHvg] = useState(2000)

  const [leidenRes, setLeidenRes] = useState(0.5)
  const [subclusterRes, setSubclusterRes] = useState(0.5)
  const [leiden, setLeiden] = useState<LeidenResponse | null>(null)
  const [leidenBaselineReady, setLeidenBaselineReady] = useState(false)
  const [useLeidenColors, setUseLeidenColors] = useState(true)

  const [geneQuery, setGeneQuery] = useState("")
  const [geneColor, setGeneColor] = useState<{
    gene: string
    values: number[]
    vmin: number
    vmax: number
  } | null>(null)
  const [colorByGene, setColorByGene] = useState(false)
  const [geneBusy, setGeneBusy] = useState(false)
  const [plotJitter, setPlotJitter] = useState(false)
  const [jitterAmp, setJitterAmp] = useState(40)
  const [pointSizePx, setPointSizePx] = useState(1)
  const lastJitterTRef = useRef(0)

  const [obsColumns, setObsColumns] = useState<string[]>([])
  const [activeColorColumn, setActiveColorColumn] = useState<string | null>(null)
  const [colorBusy, setColorBusy] = useState(false)

  const [maltRefPath, setMaltRefPath] = useState("")
  const [maltGroupby, setMaltGroupby] = useState("")
  const [maltOutdir, setMaltOutdir] = useState("/tmp/malt_results")
  const [maltBusy, setMaltBusy] = useState(false)
  const [maltResult, setMaltResult] = useState<MaltResponse | null>(null)

  const [csvColumns, setCsvColumns] = useState<string[]>([])
  const [csvPath, setCsvPath] = useState("")

  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; cluster: string; count: number } | null>(null)

  const annotationsRef = useRef<Map<string, string>>(new Map())
  const [annotationsVer, setAnnotationsVer] = useState(0)
  const [editingCluster, setEditingCluster] = useState<number | null>(null)
  const [editText, setEditText] = useState("")
  const [showLabelsOnPlot, setShowLabelsOnPlot] = useState(true)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const scatterStateRef = useRef<ScatterState | null>(null)
  const [centroidsForOverlay, setCentroidsForOverlay] = useState<ClusterCentroid[]>([])
  const editInputRef = useRef<HTMLInputElement>(null)

  const activeCategories = leiden && useLeidenColors ? leiden.categories : (meta?.color_categories ?? [])
  const activeCodes = leiden && useLeidenColors ? leiden.codes : (meta?.color_codes ?? [])
  const activeNCat = activeCategories.length

  const [clusterPalette, setClusterPalette] = useState<ClusterPalette>(EMPTY_CLUSTER_PALETTE)
  useEffect(() => {
    if (activeNCat <= 0) {
      setClusterPalette(EMPTY_CLUSTER_PALETTE)
      return
    }
    let cancelled = false
    const t = window.setTimeout(() => {
      if (cancelled) return
      try {
        setClusterPalette(buildClusterPalette(activeNCat))
      } catch {
        setClusterPalette(EMPTY_CLUSTER_PALETTE)
      }
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [activeNCat])

  const clusterCounts = useRef<Map<number, number>>(new Map())
  useEffect(() => {
    const m = new Map<number, number>()
    for (const c of activeCodes) {
      m.set(c, (m.get(c) ?? 0) + 1)
    }
    clusterCounts.current = m
  }, [activeCodes])

  const getDisplayLabel = useCallback(
    (cat: string) => annotationsRef.current.get(cat) ?? cat,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [annotationsVer],
  )

  const commitAnnotation = useCallback((cat: string, newLabel: string) => {
    const trimmed = newLabel.trim()
    if (trimmed && trimmed !== cat) {
      annotationsRef.current.set(cat, trimmed)
    } else {
      annotationsRef.current.delete(cat)
    }
    setAnnotationsVer((v) => v + 1)
    setEditingCluster(null)
  }, [])

  const startEditing = useCallback((clusterIdx: number) => {
    const cat = activeCategories[clusterIdx]
    if (!cat) return
    setEditingCluster(clusterIdx)
    setEditText(annotationsRef.current.get(cat) ?? cat)
  }, [activeCategories])

  const clusterLabelsForCanvas = useCallback((): ClusterLabel[] => {
    if (!showLabelsOnPlot) return []
    return activeCategories.map((cat, i) => ({
      code: i,
      label: getDisplayLabel(cat),
      fillCss: clusterPalette.fillCss[i] ?? hslForCategory(i, activeNCat),
    }))
  }, [activeCategories, activeNCat, clusterPalette, getDisplayLabel, showLabelsOnPlot])

  const runLeiden = useCallback(async (resolution: number) => {
    if (!umap) return
    setPhase("running_leiden")
    try {
      const res = await apiLeiden({ resolution })
      setLeiden(res)
      setLeidenBaselineReady(true)
      setSelectedCluster(null)
      setPhase("ready")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [umap])

  const runLeidenSubcluster = useCallback(async () => {
    if (!umap || selectedCluster == null) return
    setPhase("running_leiden")
    setError(null)
    try {
      const res = await apiLeidenSubcluster({
        parent_code: selectedCluster,
        resolution: subclusterRes,
      })
      setLeiden(res)
      setPhase("ready")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [umap, selectedCluster, subclusterRes])

  const resetLeidenSubclusters = useCallback(async () => {
    if (!umap || !leidenBaselineReady) return
    setError(null)
    try {
      const res = await apiLeidenReset()
      setLeiden(res)
      setSelectedCluster(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [umap, leidenBaselineReady])

  const getScatterOpts = useCallback(
    (jitterT: number): DrawScatterOpts => ({
      pointRadiusMul: pointSizePx,
      jitter: plotJitter
        ? { enabled: true, amp01: jitterAmp / 100, tSec: jitterT }
        : null,
      continuous:
        colorByGene && geneColor
          ? { values: geneColor.values, vmin: geneColor.vmin, vmax: geneColor.vmax }
          : null,
      clusterLabels: clusterLabelsForCanvas(),
    }),
    [pointSizePx, plotJitter, jitterAmp, colorByGene, geneColor, clusterLabelsForCanvas],
  )

  const fetchGeneExpression = useCallback(async () => {
    const g = geneQuery.trim()
    if (!g || !umap) return
    setGeneBusy(true)
    setError(null)
    try {
      const res = await apiGene({ gene: g })
      if (res.values.length !== umap.x.length) {
        throw new Error("Gene vector length does not match embedding")
      }
      setGeneColor({
        gene: res.gene,
        values: res.values,
        vmin: res.vmin,
        vmax: res.vmax,
      })
      setColorByGene(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    } finally {
      setGeneBusy(false)
    }
  }, [geneQuery, umap])

  const switchColorColumn = useCallback(async (column: string) => {
    if (!meta) return
    setColorBusy(true)
    setError(null)
    try {
      const res = await apiColorBy({ column })
      setMeta((prev) => prev ? {
        ...prev,
        color_column: res.column,
        color_categories: res.categories,
        color_codes: res.codes,
      } : prev)
      setActiveColorColumn(res.column)
      setUseLeidenColors(false)
      setColorByGene(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setColorBusy(false)
    }
  }, [meta])

  const runMalt = useCallback(async () => {
    if (!meta) return
    const ref = maltRefPath.trim()
    if (!ref) {
      setError("Enter a reference .h5ad path for MALT")
      return
    }
    setMaltBusy(true)
    setError(null)
    try {
      const res = await apiMalt({
        reference_path: ref,
        groupby: maltGroupby.trim() || undefined,
        outdir: maltOutdir.trim() || undefined,
      })
      setMaltResult(res)
      setCsvPath(res.csv_path)
      setCsvColumns(res.csv_columns)
      if (res.csv_columns.length > 0) {
        const col = res.csv_columns.find((c) => c.startsWith("malt_label")) ?? res.csv_columns[0]
        const csvRes = await apiLoadCsv({ csv_path: res.csv_path, column: col })
        setMeta((prev) => prev ? {
          ...prev,
          color_column: csvRes.column,
          color_categories: csvRes.categories,
          color_codes: csvRes.codes,
        } : prev)
        setActiveColorColumn(csvRes.column)
        setUseLeidenColors(false)
        setColorByGene(false)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setMaltBusy(false)
    }
  }, [meta, maltRefPath, maltGroupby, maltOutdir])

  const loadCsvColumn = useCallback(async (column: string) => {
    const cp = csvPath.trim()
    if (!cp || !meta) return
    setColorBusy(true)
    setError(null)
    try {
      const res = await apiLoadCsv({ csv_path: cp, column })
      setMeta((prev) => prev ? {
        ...prev,
        color_column: res.column,
        color_categories: res.categories,
        color_codes: res.codes,
      } : prev)
      setActiveColorColumn(res.column)
      setUseLeidenColors(false)
      setColorByGene(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setColorBusy(false)
    }
  }, [csvPath, meta])

  const fetchUmapForMeta = useCallback(
    async (m: LoadResponse, snap?: { ef_construction?: number }) => {
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
        ef_construction: snap?.ef_construction ?? efConstruction,
        n_pca_components: Math.min(nPca, m.n_pca_available),
        spread: sp2,
        umap_learning_rate: lr,
      })
      setUmap(res)
      setLeidenBaselineReady(false)
      setSelectedCluster(null)
      setGeneColor(null)
      setColorByGene(false)
      setPhase("ready")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  },
  [
    nNeighbors,
    minDist,
    nEpochs,
    efConstruction,
    nPca,
    spread,
    lr,
  ],
)

  const runUmap = useCallback(async () => {
    if (!meta) return
    await fetchUmapForMeta(meta)
  }, [meta, fetchUmapForMeta])

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
    setLeiden(null)
    setLeidenBaselineReady(false)
    setGeneColor(null)
    setColorByGene(false)
    try {
      const m = await apiLoad({
        path: p,
        n_top_hvg: nTopHvg,
        n_pca_components: nPca,
      })
      setMeta(m)
      setUmap(null)
      setNPca((v) => Math.min(v, m.n_pca_available))
      setEfConstruction(m.ef_construction)
      setObsColumns(m.obs_columns)
      setActiveColorColumn(m.color_column)
      setMaltResult(null)
      setCsvColumns([])
      setCsvPath("")
      setPhase("ready")
      void fetchUmapForMeta(m, { ef_construction: m.ef_construction })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }

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

    let raf = 0

    const paintFrame = (jitterT: number) => {
      lastJitterTRef.current = jitterT
      const n = umap.x.length
      const cc = new Uint32Array(n)
      for (let i = 0; i < n; i++) {
        cc[i] = activeCodes[i] ?? 0
      }
      const state = drawScatter2d(
        canvas,
        umap.x,
        umap.y,
        cc,
        activeNCat,
        selectedCluster,
        clusterPalette,
        getScatterOpts(jitterT),
      )
      scatterStateRef.current = state
      if (state?.centroids && !plotJitter) {
        setCentroidsForOverlay(state.centroids)
      }
    }

    const paintOnce = () => {
      paintFrame(plotJitter ? performance.now() * 0.001 : 0)
      const st = scatterStateRef.current
      if (st?.centroids) setCentroidsForOverlay(st.centroids)
    }

    requestAnimationFrame(paintOnce)
    const ro = new ResizeObserver(() =>
      requestAnimationFrame(() => paintFrame(plotJitter ? performance.now() * 0.001 : 0)),
    )
    ro.observe(wrap)

    const loop = () => {
      paintFrame(performance.now() * 0.001)
      raf = requestAnimationFrame(loop)
    }
    if (plotJitter) {
      raf = requestAnimationFrame(loop)
    }

    return () => {
      ro.disconnect()
      if (raf) cancelAnimationFrame(raf)
    }
  }, [
    umap,
    activeCodes,
    activeNCat,
    selectedCluster,
    clusterPalette,
    getScatterOpts,
    plotJitter,
  ])

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

    const idx = findNearestPoint(
      state,
      umap.x,
      umap.y,
      cx,
      cy,
      12,
      getScatterOpts(lastJitterTRef.current),
    )
    if (idx == null) {
      setHoverInfo(null)
      return
    }
    const cluster = activeCodes[idx] ?? 0
    const rawLabel = activeCategories[cluster] ?? String(cluster)
    const label = getDisplayLabel(rawLabel)
    const count = clusterCounts.current.get(cluster) ?? 0
    let detail = label !== rawLabel ? `${label} (${rawLabel})` : label
    if (colorByGene && geneColor) {
      const v = geneColor.values[idx]
      if (typeof v === "number" && Number.isFinite(v)) {
        detail = `${detail} · ${geneColor.gene} ${v.toFixed(2)}`
      }
    }
    setHoverInfo({ x: e.clientX, y: e.clientY, cluster: detail, count })
  }, [umap, activeCodes, activeCategories, colorByGene, geneColor, getScatterOpts, getDisplayLabel])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const state = scatterStateRef.current
    if (!canvas || !state || !umap) return
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top

    const idx = findNearestPoint(
      state,
      umap.x,
      umap.y,
      cx,
      cy,
      12,
      getScatterOpts(lastJitterTRef.current),
    )
    if (idx == null) {
      setSelectedCluster(null)
      return
    }
    const cluster = activeCodes[idx] ?? 0
    setSelectedCluster((prev) => (prev === cluster ? null : cluster))
  }, [umap, activeCodes, getScatterOpts])

  const handleCanvasLeave = useCallback(() => {
    setHoverInfo(null)
  }, [])

  const busy = phase === "loading_file" || phase === "running_umap"
  const leidenBusy = phase === "running_leiden"

  const showPlotBusyOverlay =
    phase === "loading_file" ||
    (meta != null && umap == null && phase !== "error")

  return (
    <div className="flex h-full min-h-svh flex-col gap-4 p-4 lg:flex-row lg:gap-6">
      <div className="flex w-full min-w-0 shrink-0 flex-col gap-4 lg:max-h-svh lg:w-[340px] lg:overflow-y-auto lg:overflow-x-hidden lg:pb-4">
        <Card size="sm">
          <CardHeader className="min-w-0">
            <CardTitle>UMAP lab</CardTitle>
            <CardDescription className="text-xs leading-snug">
              Load PCA from <code className="text-xs">obsm[&apos;X_pca&apos;]</code> or compute once.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-1">
              <Label htmlFor="h5ad" className="text-xs">
                AnnData path (.h5ad)
              </Label>
              <ParamHint text="Path on the machine running the API server (not your browser). Larger files or dense X matrices take longer to open." />
            </div>
              <Input
                id="h5ad"
                placeholder="/path/to/file.h5ad"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                disabled={busy}
              />
            </div>
            <Button
              type="button"
              onClick={() => void loadFile()}
              disabled={busy}
              size="sm"
              className="w-full"
            >
              {phase === "loading_file" ? (
                <>
                  <Loader2Icon data-icon="inline-start" className="animate-spin" />
                  Loading…
                </>
              ) : (
                "Load & run UMAP"
              )}
            </Button>
            {error ? (
              <p className="text-destructive text-xs break-words">{error}</p>
            ) : null}
          </CardContent>
          {meta ? (
            <CardFooter className="flex flex-col items-start gap-1.5 border-t">
              <div className="text-muted-foreground flex flex-wrap gap-1.5 text-xs">
                <Badge variant="secondary">{meta.n_cells.toLocaleString()} cells</Badge>
                <Badge variant="secondary">
                  PCA {meta.n_pca_available}D
                  {meta.color_column ? ` · ${meta.color_column}` : null}
                </Badge>
              </div>
            </CardFooter>
          ) : null}
        </Card>

        <SubtlePanel title="Embedding & UMAP">
          <p className="text-muted-foreground text-[11px] leading-snug">
            Sliders re-run UMAP when you release the handle. <code className="text-[10px]">min_dist ≤ spread</code> enforced.
            Re-tuning min_dist, spread, epochs, or learning rate reuses the HNSW neighbor graph when neighbors, PCA dimensions, and ef_construction are unchanged (much faster).
          </p>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1">
              <Label htmlFor="hvg" className="text-[11px] text-muted-foreground">
                n_top_hvg (only if PCA is computed)
              </Label>
              <ParamHint text="How many highly variable genes feed PCA when obsm['X_pca'] is missing. Higher keeps more signal but slows PCA slightly; ignored when PCA is loaded from the file." />
            </div>
            <Input
              id="hvg"
              type="number"
              min={500}
              max={8000}
              step={100}
              value={nTopHvg}
              onChange={(e) => setNTopHvg(Number(e.target.value) || 2000)}
              disabled={busy}
              className="h-8 text-xs"
            />
          </div>
          <ParamSlider
            label={`n_neighbors (${nNeighbors})`}
            hint="Larger k preserves more global structure and smooths fine clusters; it also slows HNSW build and query. Changing this recomputes the neighbor graph."
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
            hint="Controls how tightly points pack in 2D: lower clumps similar cells, higher spreads them apart. Does not rebuild HNSW when neighbors and PCA settings are unchanged."
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
            hint="More epochs refine the embedding layout but increase optimization time. Neighbor search time is unchanged."
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
            hint="HNSW build quality: higher values improve neighbor accuracy and slow index construction; lower values speed iteration when exploring parameters."
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
            hint="Each extra PCA dimension multiplies distance work in HNSW (build and query). Fewer dims are faster; more dims can recover finer biology if signal remains."
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
            hint="Together with min_dist, sets how wide gaps between groups can become in the embedding. Does not trigger HNSW recomputation unless neighbors or PCA inputs change."
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
            hint="Step size for UMAP’s stochastic layout optimizer. Too large can look noisy or unstable; too small converges slowly without affecting neighbor search."
            value={lr}
            min={0.1}
            max={5}
            step={0.05}
            onChange={setLr}
            onCommit={triggerUmap}
            disabled={!meta || busy}
          />
        </SubtlePanel>

        <SubtlePanel title="Leiden clustering">
          {leidenBusy ? (
            <div className="text-muted-foreground flex items-center gap-2 text-[11px]">
              <Loader2Icon className="size-3.5 animate-spin" />
              Running…
            </div>
          ) : null}
          <p className="text-muted-foreground text-[11px] leading-snug">
            {leiden
              ? `${leiden.n_clusters} clusters · ${leiden.elapsed_sec.toFixed(2)}s · `
              : null}
            <a className="text-foreground/80 underline-offset-2 hover:underline" href="https://github.com/andrewliebchen/iwanthueAPI" target="_blank" rel="noreferrer">iWantHue</a> palette.
          </p>
          <ParamSlider
            label={`resolution (${leidenRes.toFixed(2)})`}
            hint="Higher resolution splits the graph into more clusters (finer communities); lower merges clusters. Only affects Leiden, not UMAP coordinates."
            value={leidenRes}
            min={0.1}
            max={3.0}
            step={0.05}
            onChange={setLeidenRes}
            onCommit={() => void runLeiden(leidenRes)}
            disabled={!umap || busy || leidenBusy}
          />
          <ParamSlider
            label={`subcluster resolution (${subclusterRes.toFixed(2)})`}
            hint="Resolution for re-running Leiden inside the selected cluster only. Higher yields more sub-labels within that region."
            value={subclusterRes}
            min={0.1}
            max={3.0}
            step={0.05}
            onChange={setSubclusterRes}
            onCommit={() => {
              if (selectedCluster != null && leiden) void runLeidenSubcluster()
            }}
            disabled={!umap || busy || leidenBusy || !leiden || selectedCluster == null}
          />
          <Button
            type="button"
            variant="secondary"
            size="xs"
            className="w-full"
            onClick={() => void runLeidenSubcluster()}
            disabled={!umap || busy || leidenBusy || !leiden || selectedCluster == null}
          >
            Re-run Leiden on selected cluster
          </Button>
          <Button
            type="button"
            variant="outline"
            size="xs"
            className="w-full"
            title="Restore labels from the last full-graph Leiden (undoes subclusters)."
            onClick={() => void resetLeidenSubclusters()}
            disabled={
              !umap || busy || leidenBusy || !leiden || !leidenBaselineReady
            }
          >
            Reset subclusters
          </Button>
          <Button
            variant={useLeidenColors ? "default" : "outline"}
            size="xs"
            className="w-full"
            onClick={() => {
              setUseLeidenColors(!useLeidenColors)
              setSelectedCluster(null)
            }}
            disabled={!leiden}
          >
            {useLeidenColors ? "Leiden colors" : "File colors"}
          </Button>
        </SubtlePanel>

        <SubtlePanel title="Color by obs column">
          <p className="text-muted-foreground text-[11px] leading-snug">
            Pick any column from the loaded AnnData <code className="text-[10px]">obs</code> to color by.
            {csvColumns.length > 0 ? " Or pick a column from the MALT CSV below." : null}
          </p>
          {obsColumns.length > 0 ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="obs-col-select" className="text-[11px] text-muted-foreground">
                obs column
              </Label>
              <select
                id="obs-col-select"
                className="border-input bg-background text-foreground h-8 w-full rounded-md border px-2 text-xs"
                value={activeColorColumn ?? ""}
                disabled={colorBusy || busy}
                onChange={(e) => {
                  const v = e.target.value
                  if (v) void switchColorColumn(v)
                }}
              >
                <option value="" disabled>
                  Select column…
                </option>
                {obsColumns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <p className="text-muted-foreground text-[11px]">Load a dataset first.</p>
          )}
          {csvColumns.length > 0 ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="csv-col-select" className="text-[11px] text-muted-foreground">
                MALT CSV column
              </Label>
              <select
                id="csv-col-select"
                className="border-input bg-background text-foreground h-8 w-full rounded-md border px-2 text-xs"
                value=""
                disabled={colorBusy || busy}
                onChange={(e) => {
                  const v = e.target.value
                  if (v) void loadCsvColumn(v)
                }}
              >
                <option value="" disabled>
                  Select CSV column…
                </option>
                {csvColumns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          {colorBusy ? (
            <div className="text-muted-foreground flex items-center gap-2 text-[11px]">
              <Loader2Icon className="size-3.5 animate-spin" />
              Loading…
            </div>
          ) : null}
        </SubtlePanel>

        <SubtlePanel title="MALT label transfer">
          <p className="text-muted-foreground text-[11px] leading-snug">
            Run Marker-Aware Label Transfer from a reference AnnData. The query is the currently loaded dataset.
            After completion, MALT labels are loaded automatically.
          </p>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1">
              <Label htmlFor="malt-ref" className="text-[11px] text-muted-foreground">
                Reference .h5ad path
              </Label>
              <ParamHint text="Path to a reference .h5ad file with cell type labels in obs (e.g. cell_type, final_annotation)." />
            </div>
            <Input
              id="malt-ref"
              placeholder="/path/to/reference.h5ad"
              value={maltRefPath}
              onChange={(e) => setMaltRefPath(e.target.value)}
              disabled={maltBusy || !meta}
              className="h-8 text-xs"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1">
              <Label htmlFor="malt-groupby" className="text-[11px] text-muted-foreground">
                groupby (optional)
              </Label>
              <ParamHint text="Reference obs column for labels. Leave blank to auto-detect (cell_type, final_annotation, …)." />
            </div>
            <Input
              id="malt-groupby"
              placeholder="cell_type"
              value={maltGroupby}
              onChange={(e) => setMaltGroupby(e.target.value)}
              disabled={maltBusy || !meta}
              className="h-8 text-xs"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="malt-outdir" className="text-[11px] text-muted-foreground">
              Output directory
            </Label>
            <Input
              id="malt-outdir"
              value={maltOutdir}
              onChange={(e) => setMaltOutdir(e.target.value)}
              disabled={maltBusy || !meta}
              className="h-8 text-xs"
            />
          </div>
          <Button
            type="button"
            size="sm"
            className="w-full"
            onClick={() => void runMalt()}
            disabled={maltBusy || !meta || !maltRefPath.trim()}
          >
            {maltBusy ? (
              <>
                <Loader2Icon data-icon="inline-start" className="animate-spin" />
                Running MALT…
              </>
            ) : (
              "Run MALT"
            )}
          </Button>
          {maltResult ? (
            <div className="text-muted-foreground space-y-1 text-[11px]">
              <p>
                Finished in {maltResult.elapsed_sec.toFixed(1)}s
              </p>
              <p className="truncate" title={maltResult.csv_path}>
                CSV: <code className="text-[10px]">{maltResult.csv_path}</code>
              </p>
              <p>
                Columns: {maltResult.csv_columns.join(", ")}
              </p>
            </div>
          ) : null}
        </SubtlePanel>

        <SubtlePanel title="Plot appearance">
          <p className="text-muted-foreground text-[11px] leading-snug">
            Gene values are read from <code className="text-[10px]">X</code> (reloads file). Color scale uses 2–98% percentiles.
          </p>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1">
              <Label htmlFor="gene" className="text-[11px] text-muted-foreground">
                Gene symbol
              </Label>
              <ParamHint text="Fetches expression from X on the server (can re-read the h5ad). Does not change UMAP positions; only the color overlay." />
            </div>
            <div className="flex gap-1.5">
              <Input
                id="gene"
                placeholder="e.g. CD3E"
                value={geneQuery}
                onChange={(e) => setGeneQuery(e.target.value)}
                disabled={!meta || busy || geneBusy}
                className="h-8 min-w-0 flex-1 text-xs"
                onKeyDown={(e) => {
                  if (e.key === "Enter") void fetchGeneExpression()
                }}
              />
              <Button
                type="button"
                size="xs"
                variant="secondary"
                className="h-8 shrink-0 px-2"
                disabled={!umap || busy || geneBusy || !geneQuery.trim()}
                onClick={() => void fetchGeneExpression()}
              >
                {geneBusy ? <Loader2Icon className="size-3.5 animate-spin" /> : "Load"}
              </Button>
            </div>
          </div>
          <label className="text-muted-foreground flex cursor-pointer items-center gap-2 text-[11px]">
            <input
              type="checkbox"
              className="border-muted-foreground/50 accent-foreground size-3.5 rounded border"
              checked={colorByGene}
              disabled={!geneColor}
              onChange={(e) => setColorByGene(e.target.checked)}
            />
            <span>Color by gene</span>
            <ParamHint text="Toggles the gene color scale on or off without reloading expression from disk." />
          </label>
          {geneColor ? (
            <button
              type="button"
              className="text-muted-foreground self-start text-[11px] underline-offset-2 hover:text-foreground hover:underline"
              onClick={() => {
                setGeneColor(null)
                setColorByGene(false)
              }}
            >
              Clear gene
            </button>
          ) : null}
          <label className="text-muted-foreground flex cursor-pointer items-center gap-2 text-[11px]">
            <input
              type="checkbox"
              className="border-muted-foreground/50 accent-foreground size-3.5 rounded border"
              checked={plotJitter}
              disabled={!umap}
              onChange={(e) => setPlotJitter(e.target.checked)}
            />
            <span>Subtle jitter (spatial-viewer style)</span>
            <ParamHint text="Adds a small random screen-space nudge so dense blobs separate visually; underlying UMAP coordinates are unchanged." />
          </label>
          <ParamSlider
            label={`Jitter strength (${jitterAmp}%)`}
            hint="Larger values move points farther in pixel space for readability when jitter is enabled."
            value={jitterAmp}
            min={5}
            max={100}
            step={5}
            onChange={setJitterAmp}
            onCommit={() => {}}
            disabled={!umap || !plotJitter}
          />
          <ParamSlider
            label={`Point size (×${pointSizePx.toFixed(2)})`}
            hint="Scales drawn marker radius on the canvas for dense vs sparse presentations; does not affect data."
            value={pointSizePx}
            min={0.35}
            max={4}
            step={0.05}
            onChange={setPointSizePx}
            onCommit={() => {}}
            disabled={!umap}
          />
        </SubtlePanel>
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
            {showLabelsOnPlot && centroidsForOverlay.length > 0 && umap && !showPlotBusyOverlay ? (
              <div className="pointer-events-none absolute inset-0 z-20">
                {centroidsForOverlay.map((c) => {
                  const cat = activeCategories[c.code]
                  if (!cat) return null
                  const isEditing = editingCluster === c.code
                  return (
                    <div
                      key={c.code}
                      className="pointer-events-auto absolute"
                      style={{
                        left: c.cx,
                        top: c.cy,
                        transform: "translate(-50%, -140%)",
                      }}
                    >
                      {isEditing ? (
                        <form
                          className="flex items-center gap-1"
                          onSubmit={(e) => {
                            e.preventDefault()
                            commitAnnotation(cat, editText)
                          }}
                        >
                          <input
                            ref={editInputRef}
                            autoFocus
                            className="border-input bg-background/95 text-foreground w-32 rounded border px-1.5 py-0.5 text-xs shadow-md outline-none backdrop-blur-sm focus:ring-1 focus:ring-ring"
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            onBlur={() => commitAnnotation(cat, editText)}
                            onKeyDown={(e) => {
                              if (e.key === "Escape") setEditingCluster(null)
                            }}
                          />
                          <button
                            type="submit"
                            className="bg-background/80 text-muted-foreground hover:text-foreground rounded p-0.5 shadow-sm backdrop-blur-sm"
                          >
                            <CheckIcon className="size-3" />
                          </button>
                        </form>
                      ) : (
                        <button
                          className="group/edit flex items-center gap-0.5 rounded-sm px-1 py-0.5 opacity-0 transition-opacity hover:opacity-100"
                          style={{ opacity: 0 }}
                          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.opacity = "1" }}
                          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.opacity = "0" }}
                          title={`Annotate "${cat}"`}
                          onClick={(e) => {
                            e.stopPropagation()
                            startEditing(c.code)
                          }}
                        >
                          <PencilIcon className="size-3 text-foreground/70 drop-shadow" />
                        </button>
                      )}
                    </div>
                  )
                })}
              </div>
            ) : null}
            {showPlotBusyOverlay ? (
              <div className="bg-background/88 absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 px-6 text-center backdrop-blur-[2px]">
                <Loader2Icon className="text-muted-foreground size-9 animate-spin" />
                <p className="text-foreground text-sm font-medium">
                  {phase === "loading_file"
                    ? "Loading AnnData…"
                    : "Computing UMAP…"}
                </p>
                <p className="text-muted-foreground max-w-md text-xs leading-relaxed">
                  {phase === "loading_file"
                    ? "The server may run PCA and preprocessing (often several seconds). The plot stays dimmed until the first embedding is ready."
                    : "Embedding cells in 2D. This can take a while for large datasets."}
                </p>
              </div>
            ) : null}
            {hoverInfo ? (
              <div
                className="bg-popover text-popover-foreground pointer-events-none fixed z-50 rounded-md border px-3 py-1.5 text-xs shadow-md"
                style={{ left: hoverInfo.x + 12, top: hoverInfo.y - 8 }}
              >
                <span className="font-medium">
                  {hoverInfo.cluster.includes(" · ") ? hoverInfo.cluster : `Cluster ${hoverInfo.cluster}`}
                </span>
                <span className="text-muted-foreground ml-2">{hoverInfo.count.toLocaleString()} pts</span>
              </div>
            ) : null}
          </div>

          {activeCategories.length > 0 && umap ? (
            <div className="mt-2 flex max-h-24 flex-wrap gap-x-3 gap-y-1 overflow-y-auto rounded-lg border p-2">
              <label className="text-muted-foreground flex w-full cursor-pointer items-center gap-1.5 text-[10px]">
                <input
                  type="checkbox"
                  className="border-muted-foreground/50 accent-foreground size-3 rounded border"
                  checked={showLabelsOnPlot}
                  onChange={(e) => setShowLabelsOnPlot(e.target.checked)}
                />
                Labels on plot (click label to annotate)
              </label>
              {activeCategories.map((cat, i) => {
                const count = clusterCounts.current.get(i) ?? 0
                const isSelected = selectedCluster === i
                const isDimmed = selectedCluster != null && !isSelected
                const display = getDisplayLabel(cat)
                return (
                  <button
                    key={`${cat}-${i}`}
                    className="flex items-center gap-1.5 rounded px-1 py-0.5 text-xs transition-opacity hover:bg-muted"
                    style={{ opacity: isDimmed ? 0.35 : 1 }}
                    onClick={() => setSelectedCluster((prev) => (prev === i ? null : i))}
                  >
                    <span
                      className="inline-block size-2.5 shrink-0 rounded-full"
                      style={{
                        background: clusterPalette.fillCss[i] ?? hslForCategory(i, activeNCat),
                      }}
                    />
                    <span className="truncate max-w-[120px]">{display}</span>
                    <span className="text-muted-foreground tabular-nums">{count.toLocaleString()}</span>
                  </button>
                )
              })}
            </div>
          ) : null}
          {colorByGene && geneColor && umap ? (
            <div className="text-muted-foreground mt-2 flex flex-wrap items-center gap-2 border-t border-transparent pt-1 text-[10px]">
              <span className="shrink-0 font-medium text-foreground/80">{geneColor.gene}</span>
              <div
                className="h-2 min-w-[100px] flex-1 rounded-sm border border-foreground/10"
                style={{
                  background: "linear-gradient(90deg,#440154,#21908d,#fde725)",
                }}
              />
              <span className="tabular-nums">
                {geneColor.vmin.toPrecision(3)} … {geneColor.vmax.toPrecision(3)}
              </span>
            </div>
          ) : null}
        </div>
        {!meta ? (
          <p className="text-muted-foreground text-sm">
            Start the server (use <code className="text-xs">--release</code> for fast HNSW):{" "}
            <code className="text-xs">
              cargo run --release --features umap-lab --bin umap_lab
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
  hint?: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  onCommit: () => void
  disabled?: boolean
}) {
  const { label, hint, value, min, max, step, onChange, onCommit, disabled } = props
  return (
    <div className="flex flex-col gap-1">
      <div className="flex min-w-0 items-baseline gap-1">
        <Label className="min-w-0 flex-1 text-xs leading-snug break-words">{label}</Label>
        {hint ? <ParamHint text={hint} /> : null}
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

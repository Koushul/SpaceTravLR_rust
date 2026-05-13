import { CheckIcon, ChevronDownIcon, DownloadIcon, Loader2Icon, PencilIcon } from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  apiColorBy,
  apiExportCsv,
  apiGene,
  apiLoad,
  apiLoadCsv,
  apiLeiden,
  apiLeidenReset,
  apiLeidenSubcluster,
  apiMagicLeiden,
  apiMalt,
  apiMaltOptimized,
  apiSignatureExpression,
  apiSignatureSets,
  apiStatus,
  apiUmap,
  type LeidenResponse,
  type LoadResponse,
  type MaltResponse,
  type MaltOptimizedResponse,
  type ReferenceSignatureSet,
  type ReferenceSignatureSummary,
  type SignatureExpressionResponse,
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

const EXPRESSION_SOURCE_LABEL: Record<"x" | "normalized_count" | "imputed_count", string> = {
  x: "X",
  normalized_count: "normalized_count",
  imputed_count: "MAGIC imputed",
}

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
  const [geneListOpen, setGeneListOpen] = useState(false)
  const [plotSpace, setPlotSpace] = useState<"umap" | "spatial">("umap")
  const [geneColor, setGeneColor] = useState<{
    gene: string
    values: number[]
    vmin: number
    vmax: number
  } | null>(null)
  const [colorByGene, setColorByGene] = useState(false)
  const [geneBusy, setGeneBusy] = useState(false)
  const [signatureSets, setSignatureSets] = useState<ReferenceSignatureSet[]>([])
  const [signatureSpecies, setSignatureSpecies] = useState("human")
  const [signatureId, setSignatureId] = useState("")
  const [signatureSearch, setSignatureSearch] = useState("")
  const [signatureSearchOpen, setSignatureSearchOpen] = useState(false)
  const [signatureBusy, setSignatureBusy] = useState(false)
  const [magicBusy, setMagicBusy] = useState(false)
  const [magicImputedReady, setMagicImputedReady] = useState(false)
  type GeneExpressionSource = "x" | "normalized_count" | "imputed_count"
  const [geneExpressionSource, setGeneExpressionSource] =
    useState<GeneExpressionSource>("x")
  const [signatureColor, setSignatureColor] = useState<SignatureExpressionResponse | null>(null)
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

  const embedXY = useMemo(() => {
    if (
      plotSpace === "spatial" &&
      meta?.has_spatial &&
      meta.spatial_x.length === meta.n_cells &&
      meta.spatial_y.length === meta.n_cells
    ) {
      return { x: meta.spatial_x, y: meta.spatial_y, kind: "spatial" as const }
    }
    if (umap) {
      return { x: umap.x, y: umap.y, kind: "umap" as const }
    }
    return null
  }, [plotSpace, meta, umap])

  useEffect(() => {
    if (!meta?.has_spatial && plotSpace === "spatial") {
      setPlotSpace("umap")
    }
  }, [meta?.has_spatial, plotSpace])

  useEffect(() => {
    let cancelled = false
    if (!meta) {
      setSignatureSets([])
      setSignatureId("")
      setSignatureColor(null)
      return
    }
    setSignatureBusy(true)
    apiSignatureSets()
      .then((res) => {
        if (cancelled) return
        setSignatureSets(res.sets)
        const best = res.sets.reduce<ReferenceSignatureSet | null>((acc, set) => {
          const n = set.signatures.reduce((sum, sig) => sum + sig.present_genes.length, 0)
          const accN = acc?.signatures.reduce((sum, sig) => sum + sig.present_genes.length, 0) ?? -1
          return n > accN ? set : acc
        }, null)
        setSignatureSpecies((prev) =>
          best && !res.sets.some((set) => set.species === prev) ? best.species : prev,
        )
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setSignatureBusy(false)
      })
    return () => {
      cancelled = true
    }
  }, [meta?.path, meta?.n_cells])

  const activeSignatureSet = useMemo(
    () => signatureSets.find((set) => set.species === signatureSpecies) ?? signatureSets[0] ?? null,
    [signatureSets, signatureSpecies],
  )

  const activeSignatureOptions = useMemo<ReferenceSignatureSummary[]>(() => {
    return activeSignatureSet?.signatures.filter((sig) => sig.present_genes.length > 0) ?? []
  }, [activeSignatureSet])

  useEffect(() => {
    if (activeSignatureOptions.length === 0) {
      setSignatureId("")
      return
    }
    if (signatureId && !activeSignatureOptions.some((sig) => sig.id === signatureId)) {
      setSignatureId("")
    }
  }, [activeSignatureOptions, signatureId])

  const selectedSignature = useMemo(() => {
    return activeSignatureOptions.find((sig) => sig.id === signatureId) ?? null
  }, [activeSignatureOptions, signatureId])

  const signatureSearchResults = useMemo(() => {
    const q = signatureSearch.trim().toLowerCase()
    const tokens = q.split(/\s+/).filter(Boolean)
    const hits: Array<{
      set: ReferenceSignatureSet
      sig: ReferenceSignatureSummary
      score: number
    }> = []
    for (const set of signatureSets) {
      for (const sig of set.signatures) {
        if (sig.present_genes.length === 0) continue
        const haystack = [
          set.species,
          sig.id,
          sig.label,
          sig.category,
          sig.description,
          sig.present_genes.join(" "),
        ].join(" ").toLowerCase()
        if (tokens.length > 0 && !tokens.every((token) => haystack.includes(token))) continue
        const label = sig.label.toLowerCase()
        const score =
          (q && label.startsWith(q) ? 1000 : 0) +
          (q && label.includes(q) ? 250 : 0) +
          Math.min(sig.present_genes.length, 50)
        hits.push({ set, sig, score })
      }
    }
    return hits
      .sort((a, b) => b.score - a.score || a.sig.label.localeCompare(b.sig.label))
      .slice(0, 40)
  }, [signatureSearch, signatureSets])

  const selectSignature = useCallback((set: ReferenceSignatureSet, sig: ReferenceSignatureSummary) => {
    setSignatureSpecies(set.species)
    setSignatureId(sig.id)
    setSignatureSearch(`${sig.label} (${set.species})`)
    setSignatureSearchOpen(false)
    setGeneQuery("")
  }, [])

  const geneSuggestions = useMemo(() => {
    const names = selectedSignature?.present_genes ?? meta?.var_names ?? []
    const q = geneQuery.trim().toLowerCase()
    if (!q) {
      return names.slice(0, selectedSignature ? 100 : 40)
    }
    const out: string[] = []
    for (const n of names) {
      if (n.toLowerCase().includes(q)) {
        out.push(n)
        if (out.length >= 50) break
      }
    }
    return out
  }, [meta?.var_names, geneQuery, selectedSignature])

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

  const refreshMagicStatus = useCallback(async () => {
    try {
      const s = await apiStatus()
      setMagicImputedReady(s.magic_imputed_ready)
    } catch {
      setMagicImputedReady(false)
    }
  }, [])

  const runMagicLeiden = useCallback(async () => {
    if (!umap || !leiden) return
    setMagicBusy(true)
    setError(null)
    try {
      const res = await apiMagicLeiden()
      setMagicImputedReady(res.magic_imputed_ready)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    } finally {
      setMagicBusy(false)
    }
  }, [umap, leiden])

  const runLeiden = useCallback(async (resolution: number) => {
    if (!umap) return
    setPhase("running_leiden")
    try {
      const res = await apiLeiden({ resolution })
      setLeiden(res)
      setLeidenBaselineReady(true)
      setSelectedCluster(null)
      setPhase("ready")
      void refreshMagicStatus()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [umap, refreshMagicStatus])

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
      void refreshMagicStatus()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [umap, selectedCluster, subclusterRes, refreshMagicStatus])

  const resetLeidenSubclusters = useCallback(async () => {
    if (!umap || !leidenBaselineReady) return
    setError(null)
    try {
      const res = await apiLeidenReset()
      setLeiden(res)
      setSelectedCluster(null)
      void refreshMagicStatus()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase("error")
    }
  }, [umap, leidenBaselineReady, refreshMagicStatus])

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

  const fetchGeneExpression = useCallback(
    async (explicitGene?: string, sourceOverride?: GeneExpressionSource) => {
      const g = (explicitGene ?? geneQuery).trim()
      if (!g || !meta) return
      const src = sourceOverride ?? geneExpressionSource
      setGeneBusy(true)
      setError(null)
      try {
        const res = await apiGene({
          gene: g,
          source: src === "x" ? undefined : src,
        })
        if (res.values.length !== meta.n_cells) {
          throw new Error("Gene vector length does not match number of cells")
        }
        setGeneColor({
          gene: res.gene,
          values: res.values,
          vmin: res.vmin,
          vmax: res.vmax,
        })
        setSignatureColor(null)
        setColorByGene(true)
        setGeneListOpen(false)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
        setPhase("error")
      } finally {
        setGeneBusy(false)
      }
    },
    [geneQuery, meta, geneExpressionSource],
  )

  const fetchSignatureExpression = useCallback(
    async (sourceOverride?: GeneExpressionSource) => {
      if (!meta || !activeSignatureSet || !selectedSignature) return
      const src = sourceOverride ?? geneExpressionSource
      setSignatureBusy(true)
      setError(null)
      try {
        const res = await apiSignatureExpression({
          species: activeSignatureSet.species,
          id: selectedSignature.id,
          expression_source: src === "x" ? undefined : src,
        })
        if (res.values.length !== meta.n_cells) {
          throw new Error("Signature vector length does not match number of cells")
        }
        setGeneColor({
          gene: res.label,
          values: res.values,
          vmin: res.vmin,
          vmax: res.vmax,
        })
        setSignatureColor(res)
        setColorByGene(true)
        setGeneListOpen(false)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
        setPhase("error")
      } finally {
        setSignatureBusy(false)
      }
    },
    [activeSignatureSet, meta, selectedSignature, geneExpressionSource],
  )

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
      setSignatureColor(null)
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
        setSignatureColor(null)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setMaltBusy(false)
    }
  }, [meta, maltRefPath, maltGroupby, maltOutdir])

  const runMaltOptimized = useCallback(async () => {
    if (!meta) return
    const ref = maltRefPath.trim()
    if (!ref) {
      setError("Enter a reference .h5ad path for MALT")
      return
    }
    setMaltBusy(true)
    setError(null)
    try {
      const res: MaltOptimizedResponse = await apiMaltOptimized({
        reference_path: ref,
        groupby: maltGroupby.trim() || undefined,
      })
      setMaltResult({
        outdir: "(in-memory)",
        csv_path: "(projected)",
        csv_columns: [res.column],
        elapsed_sec: res.elapsed_sec,
      })
      setMeta((prev) => prev ? {
        ...prev,
        color_column: res.column,
        color_categories: res.categories,
        color_codes: res.codes,
      } : prev)
      setActiveColorColumn(res.column)
      setUseLeidenColors(false)
      setColorByGene(false)
      setSignatureColor(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setMaltBusy(false)
    }
  }, [meta, maltRefPath, maltGroupby])

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
      setSignatureColor(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setColorBusy(false)
    }
  }, [csvPath, meta])

  const handleExportCsv = useCallback(async () => {
    try {
      const annotations: Record<string, string> = {}
      annotationsRef.current.forEach((v, k) => { annotations[k] = v })
      const blob = await apiExportCsv(annotations)
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "umap_lab_export.csv"
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

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
      setSignatureColor(null)
      setGeneExpressionSource("x")
      setPhase("ready")
      void refreshMagicStatus()
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
    refreshMagicStatus,
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
    setSignatureColor(null)
    setGeneExpressionSource("x")
    setMagicImputedReady(false)
    try {
      const m = await apiLoad({
        path: p,
        n_top_hvg: nTopHvg,
        n_pca_components: nPca,
      })
      setMeta(m)
      setUmap(null)
      setPlotSpace("umap")
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

  useEffect(() => {
    if (magicImputedReady) return
    if (geneExpressionSource !== "imputed_count") return
    if (!geneColor) return
    setGeneExpressionSource("x")
    if (signatureColor && activeSignatureSet && selectedSignature) {
      void fetchSignatureExpression("x")
    } else if (!signatureColor) {
      void fetchGeneExpression(geneColor.gene, "x")
    }
  }, [
    magicImputedReady,
    geneExpressionSource,
    geneColor,
    signatureColor,
    activeSignatureSet,
    selectedSignature,
    fetchGeneExpression,
    fetchSignatureExpression,
  ])

  const prevUmapRef = useRef<UmapResponse | null>(null)
  useEffect(() => {
    if (!umap || umap === prevUmapRef.current) return
    prevUmapRef.current = umap
    void runLeiden(leidenRes)
  }, [umap, leidenRes, runLeiden])

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap || !embedXY) return

    let raf = 0

    const paintFrame = (jitterT: number) => {
      lastJitterTRef.current = jitterT
      const n = embedXY.x.length
      const cc = new Uint32Array(n)
      for (let i = 0; i < n; i++) {
        cc[i] = activeCodes[i] ?? 0
      }
      const state = drawScatter2d(
        canvas,
        embedXY.x,
        embedXY.y,
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
    embedXY,
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
    if (!canvas || !state || !embedXY) {
      setHoverInfo(null)
      return
    }
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top

    const idx = findNearestPoint(
      state,
      embedXY.x,
      embedXY.y,
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
  }, [embedXY, activeCodes, activeCategories, colorByGene, geneColor, getScatterOpts, getDisplayLabel])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const state = scatterStateRef.current
    if (!canvas || !state || !embedXY) return
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top

    const idx = findNearestPoint(
      state,
      embedXY.x,
      embedXY.y,
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
  }, [embedXY, activeCodes, getScatterOpts])

  const handleCanvasLeave = useCallback(() => {
    setHoverInfo(null)
  }, [])

  const busy = phase === "loading_file" || phase === "running_umap"
  const leidenBusy = phase === "running_leiden"

  const showPlotBusyOverlay =
    phase === "loading_file" ||
    (meta != null && embedXY == null && phase !== "error")

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
                {meta.has_spatial ? (
                  <Badge variant="outline" className="border-dashed">
                    {meta.spatial_key ?? "spatial"}
                  </Badge>
                ) : null}
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
            type="button"
            variant="secondary"
            size="xs"
            className="w-full"
            title="Run within-cluster MAGIC (rust_preprocess) and write imputed_count to a temp copy for coloring."
            onClick={() => void runMagicLeiden()}
            disabled={!umap || busy || leidenBusy || magicBusy || !leiden}
          >
            {magicBusy ? (
              <span className="flex items-center justify-center gap-1.5">
                <Loader2Icon className="size-3.5 animate-spin" />
                MAGIC…
              </span>
            ) : (
              "Run MAGIC (Leiden clusters)"
            )}
          </Button>
          {magicImputedReady ? (
            <p className="text-muted-foreground text-[10px] leading-snug">
              MAGIC layer ready — use expression layer in plot appearance to color by imputed counts.
            </p>
          ) : null}
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
          <div className="flex gap-1.5">
            {leiden ? (
              <Button
                type="button"
                size="sm"
                className="flex-1"
                onClick={() => void runMaltOptimized()}
                disabled={maltBusy || !meta || !maltRefPath.trim()}
              >
                {maltBusy ? (
                  <>
                    <Loader2Icon data-icon="inline-start" className="animate-spin" />
                    Running…
                  </>
                ) : (
                  "Run MALT (optimized)"
                )}
              </Button>
            ) : null}
            <Button
              type="button"
              size="sm"
              variant={leiden ? "outline" : "default"}
              className={leiden ? "flex-1" : "w-full"}
              onClick={() => void runMalt()}
              disabled={maltBusy || !meta || !maltRefPath.trim()}
            >
              {maltBusy && !leiden ? (
                <>
                  <Loader2Icon data-icon="inline-start" className="animate-spin" />
                  Running…
                </>
              ) : (
                leiden ? "Full MALT" : "Run MALT"
              )}
            </Button>
          </div>
          {leiden ? (
            <p className="text-muted-foreground text-[10px]">
              Optimized: subsamples to min cluster size ({leiden.categories.length} clusters), maps via PCA-KNN
            </p>
          ) : null}
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

        <SubtlePanel title="Color UMAP / plot appearance">
          <p className="text-muted-foreground text-[11px] leading-snug">
            Gene and signature values default to <code className="text-[10px]">X</code>. After running MAGIC, choose{" "}
            <code className="text-[10px]">normalized_count</code> or <code className="text-[10px]">imputed_count</code>{" "}
            from the expression layer control. Color scales use 2–98% percentiles.
          </p>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1">
              <Label htmlFor="gene" className="text-[11px] text-muted-foreground">
                Gene symbol
              </Label>
              <ParamHint text="Type to filter var_names; pick from suggestions or press Enter. Uses the selected expression layer (X, normalized_count, or imputed_count after MAGIC)." />
            </div>
            <div className="relative flex gap-1.5">
              <Input
                id="gene"
                placeholder="e.g. CD3E"
                value={geneQuery}
                onChange={(e) => {
                  setGeneQuery(e.target.value)
                  setGeneListOpen(true)
                }}
                onFocus={() => setGeneListOpen(true)}
                onBlur={() => {
                  window.setTimeout(() => setGeneListOpen(false), 180)
                }}
                disabled={!meta || busy || geneBusy}
                className="h-8 min-w-0 flex-1 text-xs"
                onKeyDown={(e) => {
                  if (e.key === "Enter") void fetchGeneExpression()
                }}
                autoComplete="off"
              />
              <Button
                type="button"
                size="xs"
                variant="secondary"
                className="h-8 shrink-0 px-2"
                disabled={!meta || busy || geneBusy || !geneQuery.trim()}
                onClick={() => void fetchGeneExpression()}
              >
                {geneBusy ? <Loader2Icon className="size-3.5 animate-spin" /> : "Load"}
              </Button>
              {geneListOpen && geneSuggestions.length > 0 ? (
                <ul className="border-border bg-popover text-popover-foreground absolute top-full right-0 left-0 z-40 mt-0.5 max-h-48 overflow-y-auto rounded-md border py-0.5 text-xs shadow-md">
                  {geneSuggestions.map((name) => (
                    <li key={name}>
                      <button
                        type="button"
                        className="hover:bg-muted/80 block w-full px-2 py-1 text-left"
                        onMouseDown={(e) => {
                          e.preventDefault()
                          setGeneQuery(name)
                          void fetchGeneExpression(name)
                        }}
                      >
                        {name}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </div>
          <div className="flex flex-col gap-1.5 rounded-md border border-border/50 p-2">
            <div className="flex items-center gap-1">
              <Label htmlFor="signature-search" className="text-[11px] text-muted-foreground">
                Reference signature search
              </Label>
              <ParamHint text="Fuzzy-search all human and mouse signatures. Selecting a signature limits the gene autocomplete above to genes from that signature that are present in var_names." />
            </div>
            <div className="relative">
              <Input
                id="signature-search"
                placeholder="e.g. hypoxia, T cell, apoptosis, kidney"
                value={signatureSearch}
                disabled={!meta || busy || signatureBusy || signatureSets.length === 0}
                className="h-8 text-xs"
                onChange={(e) => {
                  setSignatureSearch(e.target.value)
                  setSignatureSearchOpen(true)
                }}
                onFocus={() => setSignatureSearchOpen(true)}
                onBlur={() => {
                  window.setTimeout(() => setSignatureSearchOpen(false), 180)
                }}
                autoComplete="off"
              />
              {signatureSearchOpen && signatureSearchResults.length > 0 ? (
                <ul className="border-border bg-popover text-popover-foreground absolute top-full right-0 left-0 z-40 mt-0.5 max-h-56 overflow-y-auto rounded-md border py-0.5 text-xs shadow-md">
                  {signatureSearchResults.map(({ set, sig }) => (
                    <li key={`${set.species}:${sig.id}`}>
                      <button
                        type="button"
                        className="hover:bg-muted/80 block w-full px-2 py-1.5 text-left"
                        onMouseDown={(e) => {
                          e.preventDefault()
                          selectSignature(set, sig)
                        }}
                      >
                        <span className="block text-foreground">{sig.label}</span>
                        <span className="text-muted-foreground block text-[10px]">
                          {set.species} · {sig.category.replaceAll("_", " ")} · {sig.present_genes.length}/{sig.genes.length} genes
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
            <Button
              type="button"
              size="xs"
              variant="secondary"
              className="h-8 w-full"
              disabled={!meta || busy || signatureBusy || !selectedSignature}
              onClick={() => void fetchSignatureExpression()}
            >
              {signatureBusy ? <Loader2Icon className="size-3.5 animate-spin" /> : "Load signature"}
            </Button>
            {selectedSignature ? (
              <div className="space-y-1">
                <p className="text-muted-foreground text-[10px] leading-snug">
                  {activeSignatureSet?.species} · {selectedSignature.category.replaceAll("_", " ")} ·{" "}
                  {selectedSignature.present_genes.length} present, {selectedSignature.missing_genes.length} missing.{" "}
                  {selectedSignature.description}
                </p>
                <div className="flex max-h-20 flex-wrap gap-1 overflow-y-auto">
                  {selectedSignature.present_genes.map((gene) => (
                    <button
                      key={gene}
                      type="button"
                      className="border-border bg-muted/40 hover:bg-muted rounded border px-1.5 py-0.5 text-[10px] text-foreground/85"
                      disabled={!meta || busy || geneBusy}
                      onClick={() => {
                        setGeneQuery(gene)
                        void fetchGeneExpression(gene)
                      }}
                    >
                      {gene}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  className="text-muted-foreground text-[10px] underline-offset-2 hover:text-foreground hover:underline"
                  onClick={() => {
                    setSignatureId("")
                    setSignatureSearch("")
                  }}
                >
                  Clear signature gene filter
                </button>
              </div>
            ) : (
              <p className="text-muted-foreground text-[10px] leading-snug">
                Search and select a signature to restrict the gene dropdown to its present marker genes.
              </p>
            )}
            {signatureColor ? (
              <p className="text-muted-foreground text-[10px] leading-snug">
                Active signature: <span className="text-foreground/80">{signatureColor.label}</span>{" "}
                using {signatureColor.present_genes.length}/{signatureColor.genes.length} genes.
              </p>
            ) : null}
          </div>
          <label className="text-muted-foreground flex cursor-pointer items-center gap-2 text-[11px]">
            <input
              type="checkbox"
              className="border-muted-foreground/50 accent-foreground size-3.5 rounded border"
              checked={colorByGene}
              disabled={!geneColor}
              onChange={(e) => setColorByGene(e.target.checked)}
            />
            <span>Color by expression</span>
            <ParamHint text="Toggles the current gene or signature color scale on or off without reloading expression from disk." />
          </label>
          {geneColor ? (
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-1">
                <Label htmlFor="expr-source" className="text-[11px] text-muted-foreground">
                  Expression layer
                </Label>
                <ParamHint text="X is the AnnData matrix. normalized_count and imputed_count are read from layers; imputed_count requires Run MAGIC (Leiden clusters) first." />
              </div>
              <select
                id="expr-source"
                className="border-input bg-background text-foreground h-8 w-full rounded-md border px-2 text-xs"
                value={geneExpressionSource}
                disabled={busy || geneBusy || signatureBusy}
                onChange={(e) => {
                  const v = e.target.value as GeneExpressionSource
                  setGeneExpressionSource(v)
                  if (signatureColor && activeSignatureSet && selectedSignature) {
                    void fetchSignatureExpression(v)
                  } else if (!signatureColor) {
                    void fetchGeneExpression(geneColor.gene, v)
                  }
                }}
              >
                <option value="x">X</option>
                <option value="normalized_count">normalized_count</option>
                <option value="imputed_count" disabled={!magicImputedReady}>
                  imputed_count (MAGIC)
                </option>
              </select>
            <button
              type="button"
              className="text-muted-foreground self-start text-[11px] underline-offset-2 hover:text-foreground hover:underline"
              onClick={() => {
                setGeneColor(null)
                setColorByGene(false)
                setSignatureColor(null)
                setGeneExpressionSource("x")
              }}
            >
              Clear expression color
            </button>
          </div>
          ) : null}
          <label className="text-muted-foreground flex cursor-pointer items-center gap-2 text-[11px]">
            <input
              type="checkbox"
              className="border-muted-foreground/50 accent-foreground size-3.5 rounded border"
              checked={plotJitter}
              disabled={!embedXY}
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
            disabled={!embedXY || !plotJitter}
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
            disabled={!embedXY}
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
            {meta?.has_spatial ? (
              <div className="absolute top-2 left-2 z-30 flex rounded-md bg-background/80 p-0.5 text-[10px] shadow-sm ring-1 ring-border backdrop-blur">
                <button
                  type="button"
                  className={`rounded px-2 py-1 transition-colors ${plotSpace === "umap" ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => setPlotSpace("umap")}
                  title="UMAP embedding"
                >
                  UMAP
                </button>
                <button
                  type="button"
                  className={`rounded px-2 py-1 transition-colors ${plotSpace === "spatial" ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => setPlotSpace("spatial")}
                  title={`obsm['${meta.spatial_key ?? "spatial"}']`}
                >
                  Spatial
                </button>
              </div>
            ) : null}
            <canvas
              ref={canvasRef}
              className="size-full block"
              onMouseMove={handleCanvasMouseMove}
              onClick={handleCanvasClick}
              onMouseLeave={handleCanvasLeave}
            />
            {meta ? (
              <button
                type="button"
                className="absolute top-2 right-2 z-30 flex items-center gap-1 rounded-md bg-background/80 px-2 py-1 text-[11px] text-muted-foreground shadow-sm ring-1 ring-border backdrop-blur hover:bg-background hover:text-foreground transition-colors"
                onClick={() => void handleExportCsv()}
                title="Download annotations as CSV"
              >
                <DownloadIcon className="size-3.5" />
                Export CSV
              </button>
            ) : null}
            {showLabelsOnPlot && centroidsForOverlay.length > 0 && embedXY && !showPlotBusyOverlay ? (
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

          {activeCategories.length > 0 && embedXY ? (
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
          {colorByGene && geneColor && embedXY ? (
            <div className="text-muted-foreground mt-2 flex flex-wrap items-center gap-2 border-t border-transparent pt-1 text-[10px]">
              <span className="shrink-0 font-medium text-foreground/80">
                {geneColor.gene}
                <span className="text-muted-foreground font-normal">
                  {" "}
                  · {EXPRESSION_SOURCE_LABEL[geneExpressionSource]}
                </span>
              </span>
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

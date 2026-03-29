import * as d3 from "d3";

export type SplashNetworkNode = {
  id: string;
  on_path: boolean;
  role: string;
};

export type SplashNetworkLink = {
  source: string;
  target: string;
  weight: number;
  abs_weight: number;
  /** Mean trained β for modulator→target over splash cell scope (when betadata file exists). */
  beta_mean?: number | null;
};

export type SplashNetworkJson = {
  gene_a: string;
  gene_b: string;
  n_cells_used: number;
  path: string[] | null;
  path_found: boolean;
  nodes: SplashNetworkNode[];
  links: SplashNetworkLink[];
  message?: string | null;
  surround_hops: number;
  max_nodes: number;
};

export type SplashNetworkLayout = "force" | "layered";

export type SplashForceParams = {
  linkDistanceMin: number;
  linkDistanceSpan: number;
  linkStrength: number;
  charge: number;
  collidePadding: number;
  alphaDecay: number;
  velocityDecay: number;
  dragAlphaTarget: number;
  linkIterations: number;
  zoomScaleMin: number;
  zoomScaleMax: number;
};

export const SPLASH_FORCE_DEFAULTS: SplashForceParams = {
  linkDistanceMin: 36,
  linkDistanceSpan: 120,
  linkStrength: 0.35,
  charge: -220,
  collidePadding: 14,
  alphaDecay: 0.0228,
  velocityDecay: 0.4,
  dragAlphaTarget: 0.35,
  linkIterations: 1,
  zoomScaleMin: 0.35,
  zoomScaleMax: 3,
};

type SimNode = d3.SimulationNodeDatum &
  SplashNetworkNode & { x?: number; y?: number; fx?: number | null; fy?: number | null };

type SimLink = d3.SimulationLinkDatum<SimNode> & {
  weight: number;
  abs_weight: number;
  beta_mean?: number | null;
  linkIndex: number;
};

const ROLE_COLORS: Record<string, string> = {
  source: "#d4a84b",
  sink: "#e8956c",
  path: "#4a9ebc",
  context: "#6b8fa3",
};

const ROLE_LABELS: Record<string, string> = {
  source: "Source (gene A)",
  sink: "Target (gene B)",
  path: "On directed path",
  context: "Context (surround)",
};

function nodeFill(d: SimNode): string {
  return ROLE_COLORS[d.role] ?? "#94a3b8";
}

function linkPath(d: SimLink, curve: boolean): string {
  const s = d.source as SimNode;
  const t = d.target as SimNode;
  const sx = s.x ?? 0;
  const sy = s.y ?? 0;
  const tx = t.x ?? 0;
  const ty = t.y ?? 0;
  const i = d.linkIndex;
  if (!curve) return `M${sx},${sy}L${tx},${ty}`;
  const mx = (sx + tx) / 2;
  const my = (sy + ty) / 2;
  const dx = tx - sx;
  const dy = ty - sy;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const rad = (i % 2 === 0 ? 1 : -1) * Math.min(48, len * 0.22);
  const cx = mx + nx * rad;
  const cy = my + ny * rad;
  return `M${sx},${sy} Q${cx},${cy} ${tx},${ty}`;
}

function edgeLabelPoint(d: SimLink, curve: boolean): [number, number] {
  const s = d.source as SimNode;
  const t = d.target as SimNode;
  const sx = s.x ?? 0;
  const sy = s.y ?? 0;
  const tx = t.x ?? 0;
  const ty = t.y ?? 0;
  const i = d.linkIndex;
  if (!curve) return [(sx + tx) / 2, (sy + ty) / 2];
  const mx = (sx + tx) / 2;
  const my = (sy + ty) / 2;
  const dx = tx - sx;
  const dy = ty - sy;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const rad = (i % 2 === 0 ? 1 : -1) * Math.min(48, len * 0.22);
  const cx = mx + nx * rad;
  const cy = my + ny * rad;
  const u = 0.5;
  const x = (1 - u) ** 2 * sx + 2 * (1 - u) * u * cx + u ** 2 * tx;
  const y = (1 - u) ** 2 * sy + 2 * (1 - u) * u * cy + u ** 2 * ty;
  return [x, y - 3];
}

function buildForwardRev(
  links: SplashNetworkLink[],
  nodeIds: Set<string>,
): { forward: Map<string, string[]>; rev: Map<string, string[]> } {
  const forward = new Map<string, string[]>();
  const rev = new Map<string, string[]>();
  for (const l of links) {
    if (!nodeIds.has(l.source) || !nodeIds.has(l.target)) continue;
    if (!forward.has(l.source)) forward.set(l.source, []);
    forward.get(l.source)!.push(l.target);
    if (!rev.has(l.target)) rev.set(l.target, []);
    rev.get(l.target)!.push(l.source);
  }
  return { forward, rev };
}

function bfsDistFrom(
  start: string,
  nodeIds: Set<string>,
  adj: Map<string, string[]>,
): Map<string, number> {
  const dist = new Map<string, number>();
  if (!nodeIds.has(start)) return dist;
  const q: string[] = [start];
  dist.set(start, 0);
  let qi = 0;
  while (qi < q.length) {
    const u = q[qi++]!;
    const du = dist.get(u)!;
    for (const v of adj.get(u) ?? []) {
      if (!nodeIds.has(v)) continue;
      const nv = du + 1;
      if (!dist.has(v) || nv < dist.get(v)!) {
        dist.set(v, nv);
        q.push(v);
      }
    }
  }
  return dist;
}

function bfsDistPredToGoal(
  goal: string,
  nodeIds: Set<string>,
  rev: Map<string, string[]>,
): Map<string, number> {
  return bfsDistFrom(goal, nodeIds, rev);
}

function layerColumns(
  data: SplashNetworkJson,
  nodeIds: Set<string>,
  forward: Map<string, string[]>,
  rev: Map<string, string[]>,
): Map<string, number> {
  const fromA = bfsDistFrom(data.gene_a, nodeIds, forward);
  const toB = bfsDistPredToGoal(data.gene_b, nodeIds, rev);
  let maxA = 0;
  for (const d of fromA.values()) maxA = Math.max(maxA, d);
  const raw = new Map<string, number>();
  for (const id of nodeIds) {
    const da = fromA.get(id);
    if (da !== undefined) {
      raw.set(id, da);
      continue;
    }
    const db = toB.get(id);
    if (db !== undefined) {
      raw.set(id, Math.max(0, maxA - db));
      continue;
    }
    raw.set(id, Math.max(0, Math.round(maxA / 2)));
  }
  const uniq = [...new Set(raw.values())].sort((a, b) => a - b);
  const remap = new Map<number, number>();
  uniq.forEach((c, i) => remap.set(c, i));
  const out = new Map<string, number>();
  for (const [id, c] of raw) out.set(id, remap.get(c)!);
  return out;
}

function neighborSet(
  focus: string,
  links: { source: string; target: string }[],
): Set<string> {
  const s = new Set<string>([focus]);
  for (const l of links) {
    if (l.source === focus) s.add(l.target);
    if (l.target === focus) s.add(l.source);
  }
  return s;
}

function applyFocusHighlight(
  focus: string | null,
  linkSel: d3.Selection<SVGPathElement, SimLink, SVGGElement, unknown>,
  hitSel: d3.Selection<SVGPathElement, SimLink, SVGGElement, unknown>,
  betaLabelSel: d3.Selection<SVGTextElement, SimLink, SVGGElement, unknown> | null,
  nodeCircles: d3.Selection<SVGCircleElement, SimNode, SVGGElement, unknown>,
  allLinks: SimLink[],
) {
  const pairs = allLinks.map((l) => ({
    source: (l.source as SimNode).id,
    target: (l.target as SimNode).id,
  }));
  const neigh = focus === null ? null : neighborSet(focus, pairs);
  const dim = 0.12;
  const hi = 1;
  const op = (d: SimLink) => {
    if (!focus) return hi;
    const a = (d.source as SimNode).id;
    const b = (d.target as SimNode).id;
    return a === focus || b === focus ? hi : dim;
  };
  linkSel.style("opacity", op);
  hitSel.style("opacity", op);
  betaLabelSel?.style("opacity", op);
  nodeCircles.style("opacity", (d) => {
    if (!focus) return hi;
    return neigh?.has(d.id) ? hi : dim;
  });
}

function drawLegend(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  roles: string[],
) {
  const entries = roles.filter((r) => ROLE_LABELS[r]);
  let y = 0;
  for (const r of entries) {
    g.append("rect")
      .attr("x", 0)
      .attr("y", y)
      .attr("width", 12)
      .attr("height", 12)
      .attr("rx", 2)
      .attr("fill", ROLE_COLORS[r] ?? "#888");
    g.append("text")
      .attr("x", 18)
      .attr("y", y + 10)
      .attr("fill", "#b8c0cc")
      .attr("font-size", 10)
      .text(ROLE_LABELS[r] ?? r);
    y += 18;
  }
}

export function renderSplashNetwork(
  mount: HTMLElement,
  data: SplashNetworkJson,
  opts?: {
    width?: number;
    height?: number;
    layout?: SplashNetworkLayout;
    force?: Partial<SplashForceParams>;
  },
): () => void {
  mount.innerHTML = "";
  const layout: SplashNetworkLayout = opts?.layout ?? "layered";
  const forceParams: SplashForceParams =
    layout === "force"
      ? { ...SPLASH_FORCE_DEFAULTS, ...opts?.force }
      : SPLASH_FORCE_DEFAULTS;
  const margin = { top: 36, right: 8, bottom: 8, left: 8 };
  const legendW = 150;
  const w =
    (opts?.width ?? (mount.clientWidth || 560)) - margin.left - margin.right - legendW;
  const h = (opts?.height ?? 400) - margin.top - margin.bottom;

  const svg = d3
    .select(mount)
    .append("svg")
    .attr("class", "splash-net-svg")
    .attr("width", w + margin.left + margin.right + legendW)
    .attr("height", h + margin.top + margin.bottom)
    .attr(
      "viewBox",
      `0 0 ${w + margin.left + margin.right + legendW} ${h + margin.top + margin.bottom}`,
    );

  const gRoot = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const uid = `sn-${Math.random().toString(36).slice(2, 9)}`;
  const defs = svg.append("defs");
  const mkArrow = (id: string, fill: string) => {
    defs
      .append("marker")
      .attr("id", id)
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 18)
      .attr("refY", 0)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", fill);
  };
  mkArrow(`splash-arrow-pos-${uid}`, "#4ade80");
  mkArrow(`splash-arrow-neg-${uid}`, "#fb7185");
  mkArrow(`splash-arrow-grey-${uid}`, "#9aa5b4");

  const titleG = gRoot.append("g").attr("class", "splash-net-title-g");
  titleG
    .append("text")
    .attr("x", w / 2)
    .attr("y", -12)
    .attr("text-anchor", "middle")
    .attr("fill", "#e6edf3")
    .attr("font-size", 13)
    .attr("font-weight", 600)
    .attr("font-style", "italic")
    .text(`${data.gene_a} → ${data.gene_b} (splash)`);
  titleG
    .append("text")
    .attr("x", w / 2)
    .attr("y", 2)
    .attr("text-anchor", "middle")
    .attr("fill", "#8b949e")
    .attr("font-size", 9)
    .text(() => {
      const base =
        data.path_found && data.path?.length
          ? `Directed path · ${data.n_cells_used} cells`
          : `${data.n_cells_used} cells · model-local couplings`;
      const betaHint = data.links.some(
        (l) => l.beta_mean != null && Number.isFinite(Number(l.beta_mean)),
      )
        ? " · β = mean trained coefficient on same cell mask"
        : "";
      return base + betaHint;
    });

  const legendG = gRoot
    .append("g")
    .attr("transform", `translate(${w + 14}, ${8})`);
  legendG.append("text").attr("fill", "#c9d1d9").attr("font-size", 10).text("Legend");
  drawLegend(legendG.append("g").attr("transform", "translate(0,16)"), [
    "source",
    "sink",
    "path",
    "context",
  ]);

  const nodeById = new Map<string, SimNode>();
  for (const n of data.nodes) {
    nodeById.set(n.id, { ...n });
  }
  const nodeIds = new Set(nodeById.keys());

  const linksRaw: SimLink[] = [];
  let linkIndex = 0;
  for (const l of data.links) {
    const s = nodeById.get(l.source);
    const t = nodeById.get(l.target);
    if (!s || !t) continue;
    linksRaw.push({
      source: s,
      target: t,
      weight: l.weight,
      abs_weight: l.abs_weight,
      beta_mean: l.beta_mean ?? null,
      linkIndex: linkIndex++,
    });
  }
  const withBetaForLabels = linksRaw.filter(
    (l) => l.beta_mean != null && Number.isFinite(Number(l.beta_mean)),
  );

  const maxW = d3.max(linksRaw, (x) => x.abs_weight) || 1;
  const strokeFor = (abs: number, layered: boolean) =>
    layered ? 0.6 + 3.2 * Math.sqrt(abs / maxW) : 0.8 + 4 * Math.sqrt(abs / maxW);

  const nodes = Array.from(nodeById.values());

  if (layout === "layered") {
    const { forward, rev } = buildForwardRev(data.links, nodeIds);
    const colOf = layerColumns(data, nodeIds, forward, rev);
    const byCol = new Map<number, SimNode[]>();
    for (const n of nodes) {
      const c = colOf.get(n.id) ?? 0;
      if (!byCol.has(c)) byCol.set(c, []);
      byCol.get(c)!.push(n);
    }
    for (const arr of byCol.values()) arr.sort((a, b) => a.id.localeCompare(b.id));
    const nCol = Math.max(1, byCol.size);
    const padX = 28;
    const padY = 24;
    const xForCol = (c: number) =>
      nCol <= 1 ? w / 2 : padX + (c / (nCol - 1)) * (w - 2 * padX);
    for (const [c, arr] of byCol) {
      const x = xForCol(c);
      arr.forEach((n, i) => {
        const nIn = arr.length;
        const span = h - 2 * padY;
        const y = nIn === 1 ? h / 2 : padY + (i / (nIn - 1 || 1)) * span;
        n.x = x;
        n.y = y;
        n.fx = x;
        n.fy = y;
      });
    }
  } else {
    for (const n of nodes) {
      n.x = w / 2 + (Math.random() - 0.5) * 80;
      n.y = h / 2 + (Math.random() - 0.5) * 80;
    }
  }

  const zoomG = gRoot.append("g");
  const linkG = zoomG.append("g").attr("class", "splash-links");
  const hitLinkG = zoomG.append("g").attr("class", "splash-link-hits");
  const labelG = zoomG.append("g").attr("class", "splash-beta-labels");
  const nodeG = zoomG.append("g").attr("class", "splash-nodes");

  const layered = layout === "layered";
  const curveLinks = false;
  const linkSel = linkG
    .selectAll<SVGPathElement, SimLink>("path")
    .data(linksRaw)
    .join("path")
    .attr("fill", "none")
    .attr("pointer-events", "none")
    .attr("stroke", (d) =>
      layered
        ? "rgba(154,165,180,0.55)"
        : d.weight >= 0
          ? "rgba(74,222,128,0.75)"
          : "rgba(251,113,133,0.8)",
    )
    .attr("stroke-opacity", layered ? 0.85 : 0.9)
    .attr("stroke-width", (d) => strokeFor(d.abs_weight, layered))
    .attr("marker-end", (d) =>
      layered
        ? `url(#splash-arrow-grey-${uid})`
        : d.weight >= 0
          ? `url(#splash-arrow-pos-${uid})`
          : `url(#splash-arrow-neg-${uid})`,
    );

  const hitSel = hitLinkG
    .selectAll<SVGPathElement, SimLink>("path")
    .data(linksRaw)
    .join("path")
    .attr("fill", "none")
    .attr("stroke", "transparent")
    .attr("stroke-width", 20)
    .style("cursor", "pointer")
    .on("mouseenter", (_ev, d) => {
      const src = (d.source as SimNode).id;
      const tgt = (d.target as SimNode).id;
      const bm = d.beta_mean;
      const betaLine =
        bm != null && Number.isFinite(Number(bm))
          ? `<br/>mean β (${tgt} model): ${Number(bm).toPrecision(4)}`
          : `<br/><span style="color:#8b949e">no matching β column for <code>${src}</code> in <code>${tgt}</code> betadata</span>`;
      tip
        .html(
          `<strong>${src} → ${tgt}</strong><br/>splash ∂: ${Number(d.weight).toPrecision(4)} · |∂|: ${Number(d.abs_weight).toPrecision(4)}${betaLine}`,
        )
        .style("opacity", 1);
    })
    .on("mousemove", (ev) => {
      tip.style("left", `${ev.offsetX + 12}px`).style("top", `${ev.offsetY + 12}px`);
    })
    .on("mouseleave", () => {
      tip.style("opacity", 0);
    });

  const betaLabelSel =
    withBetaForLabels.length > 0
      ? labelG
          .selectAll<SVGTextElement, SimLink>("text")
          .data(withBetaForLabels)
          .join("text")
          .attr("fill", "#f0e6a6")
          .attr("font-size", 8)
          .attr("text-anchor", "middle")
          .attr("pointer-events", "none")
          .text((d) => `β ${Number(d.beta_mean).toPrecision(3)}`)
      : null;

  const drag = d3
    .drag<SVGGElement, SimNode>()
    .on("start", (ev, d) => {
      if (layout === "layered") return;
      if (!ev.active) sim!.alphaTarget(forceParams.dragAlphaTarget).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (ev, d) => {
      if (layout === "layered") {
        d.fx = ev.x;
        d.fy = ev.y;
        d.x = ev.x;
        d.y = ev.y;
        tick();
      } else {
        d.fx = ev.x;
        d.fy = ev.y;
      }
    })
    .on("end", (ev, d) => {
      if (layout === "layered") {
        d.fx = d.x;
        d.fy = d.y;
        return;
      }
      if (!ev.active) sim!.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    });

  const nodeRadius = (d: SimNode) => (d.role === "source" || d.role === "sink" ? 11 : 8);

  const tip = d3
    .select(mount)
    .append("div")
    .attr("class", "splash-net-tooltip")
    .style("opacity", 0);

  const nodeSel = nodeG
    .selectAll<SVGGElement, SimNode>("g")
    .data(nodes)
    .join("g")
    .call(drag)
    .on("mouseenter", (_ev, d) => {
      applyFocusHighlight(
        d.id,
        linkSel,
        hitSel,
        betaLabelSel,
        nodeSel.select("circle"),
        linksRaw,
      );
      tip
        .html(
          `<strong>${d.id}</strong><br/>${ROLE_LABELS[d.role] ?? d.role}<br/>on path: ${d.on_path ? "yes" : "no"}`,
        )
        .style("opacity", 1);
    })
    .on("mousemove", (ev) => {
      tip.style("left", `${ev.offsetX + 12}px`).style("top", `${ev.offsetY + 12}px`);
    })
    .on("mouseleave", () => {
      applyFocusHighlight(
        null,
        linkSel,
        hitSel,
        betaLabelSel,
        nodeSel.select("circle"),
        linksRaw,
      );
      tip.style("opacity", 0);
    });

  const circles = nodeSel
    .append("circle")
    .attr("r", nodeRadius)
    .attr("fill", (d) => nodeFill(d))
    .attr("stroke", "#0b1220")
    .attr("stroke-width", 2)
    .attr("opacity", 1);

  nodeSel
    .append("text")
    .attr("text-anchor", "middle")
    .attr("dy", 28)
    .attr("fill", "#cbd5e1")
    .attr("font-size", 10)
    .attr("font-style", "italic")
    .attr("font-family", "Georgia, 'Times New Roman', serif")
    .text((d) => (d.id.length > 14 ? `${d.id.slice(0, 12)}…` : d.id));

  function tick() {
    const dfn = (d: SimLink) => linkPath(d, curveLinks);
    linkSel.attr("d", dfn);
    hitSel.attr("d", dfn);
    betaLabelSel
      ?.attr("x", (d) => edgeLabelPoint(d, curveLinks)[0])
      .attr("y", (d) => edgeLabelPoint(d, curveLinks)[1]);
    nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
  }

  let sim: d3.Simulation<SimNode, undefined> | null = null;
  if (layout === "force") {
    const fp = forceParams;
    sim = d3
      .forceSimulation<SimNode>(nodes)
      .alphaDecay(fp.alphaDecay)
      .velocityDecay(fp.velocityDecay)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(linksRaw)
          .id((d) => d.id)
          .distance(
            (d) => fp.linkDistanceMin + fp.linkDistanceSpan * (1 - d.abs_weight / maxW),
          )
          .strength(fp.linkStrength)
          .iterations(Math.max(1, Math.min(8, Math.round(fp.linkIterations)))),
      )
      .force("charge", d3.forceManyBody<SimNode>().strength(fp.charge))
      .force("center", d3.forceCenter(w / 2, h / 2))
      .force(
        "collision",
        d3.forceCollide<SimNode>().radius((d) => nodeRadius(d) + fp.collidePadding),
      );

    sim.on("tick", tick);
  } else {
    tick();
  }

  const zMin = layout === "force" ? forceParams.zoomScaleMin : 0.35;
  const zMax = layout === "force" ? forceParams.zoomScaleMax : 3;
  const zoom = d3
    .zoom<SVGSVGElement, unknown>()
    .scaleExtent([Math.min(zMin, zMax), Math.max(zMin, zMax)])
    .on("zoom", (ev) => {
      zoomG.attr("transform", ev.transform.toString());
    });

  svg.call(zoom);

  return () => {
    sim?.stop();
    tip.remove();
  };
}

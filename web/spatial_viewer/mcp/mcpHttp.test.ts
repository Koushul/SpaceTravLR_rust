import { describe, expect, it, vi } from "vitest";

import {
  buildPerturbScopeApiBody,
  buildPerturbScopeBody,
  buildReferenceCentroidScopeBody,
  fetchMetaWith,
  makeConnectDomainList,
  normalizeApiBase,
  parseDelimitedGenes,
} from "./mcpHttp.js";

describe("normalizeApiBase", () => {
  it("trims slashes and uses fallback", () => {
    expect(normalizeApiBase(undefined, "http://127.0.0.1:8080")).toBe("http://127.0.0.1:8080");
    expect(normalizeApiBase("  http://h:9/foo/  ", "x")).toBe("http://h:9/foo");
    expect(normalizeApiBase("", "http://a")).toBe("http://a");
  });
});

describe("makeConnectDomainList", () => {
  it("merges base origin with extra CSV origins uniquely", () => {
    expect(makeConnectDomainList("http://127.0.0.1:8080", undefined)).toEqual([
      "http://127.0.0.1:8080",
    ]);
    expect(
      makeConnectDomainList("http://127.0.0.1:8080/path/", "http://other:1, http://127.0.0.1:8080 "),
    ).toEqual(["http://127.0.0.1:8080", "http://other:1"]);
  });
});

describe("parseDelimitedGenes", () => {
  it("splits on comma, semicolon, and whitespace", () => {
    expect(parseDelimitedGenes("A, B;C  D")).toEqual(["A", "B", "C", "D"]);
    expect(parseDelimitedGenes("  ")).toEqual([]);
  });
});

describe("buildPerturbScopeBody (Rust PerturbScopeBody JSON tag=type)", () => {
  it.each([
    [undefined, undefined, undefined, { type: "all" }],
    ["all", "T", 3, { type: "all" }],
    ["cell_type", "  Epithelial ", undefined, { type: "cell_type_name", name: "Epithelial" }],
    ["cluster", "x", 7, { type: "cluster", cluster_id: 7 }],
    ["cell_type", "", undefined, { type: "all" }],
    ["cell_type", "   ", undefined, { type: "all" }],
    ["cluster", "T", undefined, { type: "all" }],
  ] as const)("scope=%s label=%s cluster=%s -> %j", (scope, label, cid, expected) => {
    expect(buildPerturbScopeBody(scope, label, cid)).toEqual(expected);
  });
});

describe("buildReferenceCentroidScopeBody", () => {
  it("defaults reference_scope to cell_type when undefined", () => {
    expect(buildReferenceCentroidScopeBody(undefined, undefined, undefined)).toEqual({
      type: "all",
    });
  });

  it("uses cell_type_name when label present", () => {
    expect(buildReferenceCentroidScopeBody("cell_type", "B_cell", undefined)).toEqual({
      type: "cell_type_name",
      name: "B_cell",
    });
  });

  it("uses cluster when reference_scope is cluster", () => {
    expect(buildReferenceCentroidScopeBody("cluster", "x", 2)).toEqual({
      type: "cluster",
      cluster_id: 2,
    });
  });

  it("all reference_scope yields all", () => {
    expect(buildReferenceCentroidScopeBody("all", "ignored", 9)).toEqual({ type: "all" });
  });
});

describe("fetchMetaWith", () => {
  it("throws on non-OK with status text", async () => {
    const fetchImpl = vi.fn(async () => new Response("bad", { status: 503 }));
    await expect(fetchMetaWith(fetchImpl as unknown as typeof fetch, "http://x")).rejects.toThrow(
      "/api/meta 503: bad",
    );
  });

  it("returns parsed JSON on 200", async () => {
    const body = {
      n_obs: 10,
      n_vars: 20,
      dataset_ready: true,
      perturb_ready: false,
      perturb_loading: false,
    };
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
    await expect(
      fetchMetaWith(fetchImpl as unknown as typeof fetch, "http://127.0.0.1:8080"),
    ).resolves.toEqual(body);
    expect(fetchImpl).toHaveBeenCalledWith("http://127.0.0.1:8080/api/meta");
  });
});

describe("HTTP contract snapshots vs Rust structs", () => {
  it("PerturbPreviewBody-style JSON for summary", () => {
    const scope = buildPerturbScopeBody("cell_type", "Tfh", undefined);
    const body = {
      gene: "IL21",
      desired_expr: 0,
      scope,
      n_propagation: 4,
    };
    expect(JSON.parse(JSON.stringify(body))).toEqual({
      gene: "IL21",
      desired_expr: 0,
      scope: { type: "cell_type_name", name: "Tfh" },
      n_propagation: 4,
    });
  });

  it("PerturbReferenceSimilarityBody-style flattened JSON", () => {
    const perturbScope = buildPerturbScopeBody("all", undefined, undefined);
    const reference = buildReferenceCentroidScopeBody("cell_type", "T_follicular_helper", undefined);
    const body = {
      gene: "X",
      desired_expr: 0,
      scope: perturbScope,
      reference,
      exclude_perturb_cells_from_reference: true,
      genes: ["BCL6", "CD40LG"],
    };
    expect(body).toMatchObject({
      reference: { type: "cell_type_name", name: "T_follicular_helper" },
      exclude_perturb_cells_from_reference: true,
    });
  });

  it("SplashNetworkBody-style JSON", () => {
    const scope = buildPerturbScopeBody("cluster", "x", 3);
    const body = {
      gene_a: "A",
      gene_b: "B",
      scope,
      surround_hops: 2,
      max_nodes: 12,
    };
    expect(JSON.parse(JSON.stringify(body))).toEqual({
      gene_a: "A",
      gene_b: "B",
      scope: { type: "cluster", cluster_id: 3 },
      surround_hops: 2,
      max_nodes: 12,
    });
  });

  it("NeighborSanityBody-style JSON", () => {
    const body = {
      gene: "IL21",
      cell_index: 42,
      desired_expr: 0,
      n_propagation: 3,
      neighbor_radius: 120.5,
      require_cluster_id: 7,
    };
    expect(JSON.parse(JSON.stringify(body))).toEqual(body);
  });
});

describe("buildPerturbScopeApiBody", () => {
  it("uses explicit indices for selection", async () => {
    const fetchImpl = vi.fn() as typeof fetch;
    const r = await buildPerturbScopeApiBody(
      fetchImpl,
      "http://127.0.0.1:8080",
      "selection",
      undefined,
      undefined,
      [3, 3, 5],
    );
    expect(r.error).toBeUndefined();
    expect(r.scope).toEqual({ type: "indices", indices: [3, 5] });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("fetches viewer_state for selection when indices omitted", async () => {
    const fetchImpl = vi.fn(async (url: string) => {
      expect(url).toContain("/api/viewer_state");
      return new Response(JSON.stringify({ interaction_sender_index: 99 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as unknown as typeof fetch;
    const r = await buildPerturbScopeApiBody(
      fetchImpl,
      "http://127.0.0.1:8080",
      "selection",
      undefined,
      undefined,
      undefined,
    );
    expect(r.error).toBeUndefined();
    expect(r.scope).toEqual({ type: "indices", indices: [99] });
  });

  it("errors when selection has no indices and no viewer sender", async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })) as unknown as typeof fetch;
    const r = await buildPerturbScopeApiBody(
      fetchImpl,
      "http://127.0.0.1:8080",
      "selection",
      undefined,
      undefined,
      undefined,
    );
    expect(r.error).toMatch(/scope=selection/);
    expect(r.scope).toEqual({ type: "all" });
  });
});

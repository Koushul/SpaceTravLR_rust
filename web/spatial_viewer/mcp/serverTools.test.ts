import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createSpatialViewerMcpServer } from "./server.js";

const API = "http://127.0.0.1:8080";

function metaBase(over: Record<string, unknown> = {}) {
  return JSON.stringify({
    n_obs: 1,
    n_vars: 1,
    dataset_ready: true,
    perturb_ready: false,
    perturb_loading: false,
    ...over,
  });
}

async function withTestServer(
  fetchImpl: typeof fetch,
  run: (client: Client) => Promise<void>,
): Promise<void> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const mcp = createSpatialViewerMcpServer({
    fetch: fetchImpl,
    defaultApiBase: API,
    connectDomainList: () => [API],
    readMcpHtml: async () => "<html/>",
  });
  await mcp.connect(serverTransport);
  const client = new Client({ name: "test", version: "1.0.0" }, { capabilities: {} });
  await client.connect(clientTransport);
  try {
    await run(client);
  } finally {
    await client.close();
    await mcp.close();
  }
}

describe("spatial viewer MCP server tools", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("spatial_viewer_get_meta returns summary and JSON", async () => {
    const fetchImpl = vi.fn(async (input: RequestInfo | URL) => {
      const u = String(input);
      if (u.endsWith("/api/meta")) {
        return new Response(metaBase({ perturb_ready: true, cluster_annot: "c" }), { status: 200 });
      }
      return new Response("nope", { status: 404 });
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const res = await client.callTool({
        name: "spatial_viewer_get_meta",
        arguments: {},
      });
      const text = (res.content as { type: string; text: string }[]).find((c) => c.type === "text")
        ?.text;
      expect(text).toContain("Dataset: ready");
      expect(text).toContain('"cluster_annot": "c"');
      expect(fetchImpl).toHaveBeenCalledWith(`${API}/api/meta`);
    });
  });

  it("spatial_viewer_check_progress reports unreachable on fetch failure", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("ECONNREFUSED");
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const res = await client.callTool({
        name: "spatial_viewer_check_progress",
        arguments: {},
      });
      const text = (res.content as { type: string; text: string }[])[0].text;
      expect(text).toContain("Server unreachable");
      expect(text).toContain("ECONNREFUSED");
    });
  });

  it("spatial_viewer_cluster_expression POSTs genes and surfaces API errors", async () => {
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const u = String(input);
      if (u.endsWith("/api/cluster/mean_expression")) {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({ genes: ["CD3E", "CD4"] });
        return new Response("unknown gene", { status: 400 });
      }
      return new Response("nope", { status: 404 });
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const res = await client.callTool({
        name: "spatial_viewer_cluster_expression",
        arguments: { genes: ["CD3E", "CD4"] },
      });
      const text = (res.content as { type: string; text: string }[])[0].text;
      expect(text).toContain("Error 400");
      expect(text).toContain("unknown gene");
    });
  });

  it("rejects cluster_expression with more than 200 genes (Zod)", async () => {
    const fetchImpl = vi.fn();
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const genes = Array.from({ length: 201 }, (_, i) => `G${i}`);
      const res = await client.callTool({
        name: "spatial_viewer_cluster_expression",
        arguments: { genes },
      });
      expect(res.isError).toBe(true);
      const text = (res.content as { type: string; text: string }[])[0].text;
      expect(text).toContain("too_big");
      expect(text).toContain("200");
    });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("spatial_viewer_perturb_summary returns NOT READY when perturb_ready is false", async () => {
    const fetchImpl = vi.fn(async (input: RequestInfo | URL) => {
      const u = String(input);
      if (u.endsWith("/api/meta")) {
        return new Response(metaBase({ perturb_ready: false }), { status: 200 });
      }
      return new Response("nope", { status: 404 });
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const res = await client.callTool({
        name: "spatial_viewer_perturb_summary",
        arguments: { gene: "X" },
      });
      const text = (res.content as { type: string; text: string }[])[0].text;
      expect(text).toContain("NOT READY");
      expect(text).toContain("--run-toml");
    });
    expect(fetchImpl).not.toHaveBeenCalledWith(
      expect.stringContaining("/api/perturb/summary"),
      expect.anything(),
    );
  });

  it("spatial_viewer_perturb_summary POST body matches Rust PerturbPreviewBody", async () => {
    let posted: string | undefined;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const u = String(input);
      if (u.endsWith("/api/meta")) {
        return new Response(metaBase({ perturb_ready: true }), { status: 200 });
      }
      if (u.endsWith("/api/perturb/summary")) {
        posted = String(init?.body);
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }
      return new Response("nope", { status: 404 });
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      await client.callTool({
        name: "spatial_viewer_perturb_summary",
        arguments: {
          gene: "G",
          scope: "cluster",
          cluster_id: 5,
          n_propagation: 2,
          desired_expr: 0.5,
        },
      });
    });
    expect(JSON.parse(posted!)).toEqual({
      gene: "G",
      desired_expr: 0.5,
      scope: { type: "cluster", cluster_id: 5 },
      n_propagation: 2,
    });
  });

  it("spatial_viewer_perturb_reference_similarity sets exclude flag and reference", async () => {
    let posted: string | undefined;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const u = String(input);
      if (u.endsWith("/api/meta")) {
        return new Response(metaBase({ perturb_ready: true }), { status: 200 });
      }
      if (u.endsWith("/api/perturb/reference_similarity")) {
        posted = String(init?.body);
        return new Response(JSON.stringify({}), { status: 200 });
      }
      return new Response("nope", { status: 404 });
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      await client.callTool({
        name: "spatial_viewer_perturb_reference_similarity",
        arguments: {
          gene: "IL21",
          reference_scope: "cell_type",
          reference_cell_type_label: "Tfh",
          exclude_perturb_cells_from_reference: false,
        },
      });
    });
    expect(JSON.parse(posted!)).toMatchObject({
      gene: "IL21",
      scope: { type: "all" },
      reference: { type: "cell_type_name", name: "Tfh" },
      exclude_perturb_cells_from_reference: false,
    });
  });

  it("spatial_viewer_splash_network_json requires gene_a and gene_b after trim", async () => {
    const fetchImpl = vi.fn();
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const res = await client.callTool({
        name: "spatial_viewer_splash_network_json",
        arguments: { gene_a: "A", gene_b: "   " },
      });
      const text = (res.content as { type: string; text: string }[])[0].text;
      expect(text).toContain("gene_a and gene_b are required");
    });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("spatial_viewer_splash_network_json POST includes scope", async () => {
    let posted: string | undefined;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const u = String(input);
      if (u.endsWith("/api/meta")) {
        return new Response(metaBase({ perturb_ready: true }), { status: 200 });
      }
      if (u.endsWith("/api/perturb/splash_network")) {
        posted = String(init?.body);
        return new Response(JSON.stringify({ nodes: [] }), { status: 200 });
      }
      return new Response("nope", { status: 404 });
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      await client.callTool({
        name: "spatial_viewer_splash_network_json",
        arguments: {
          gene_a: "SRC",
          gene_b: "SNK",
          scope: "cell_type",
          cell_type_label: "Epi",
        },
      });
    });
    expect(JSON.parse(posted!)).toEqual({
      gene_a: "SRC",
      gene_b: "SNK",
      scope: { type: "cell_type_name", name: "Epi" },
      surround_hops: 1,
      max_nodes: 24,
    });
  });

  it("spatial_viewer_perturb_neighbor_sanity POST body", async () => {
    let posted: string | undefined;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const u = String(input);
      if (u.endsWith("/api/meta")) {
        return new Response(metaBase({ perturb_ready: true }), { status: 200 });
      }
      if (u.endsWith("/api/perturb/neighbor_sanity")) {
        posted = String(init?.body);
        return new Response(JSON.stringify({ interpretation: "ok" }), { status: 200 });
      }
      return new Response("nope", { status: 404 });
    });
    await withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      await client.callTool({
        name: "spatial_viewer_perturb_neighbor_sanity",
        arguments: {
          gene: "G",
          cell_index: 3,
          neighbor_radius: 50,
          require_cluster_id: 1,
        },
      });
    });
    expect(JSON.parse(posted!)).toEqual({
      gene: "G",
      cell_index: 3,
      desired_expr: 0,
      neighbor_radius: 50,
      require_cluster_id: 1,
    });
  });

  it("spatial_viewer_wait_ready polls until ready with 2s interval", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let metaCalls = 0;
    const fetchImpl = vi.fn(async (input: RequestInfo | URL) => {
      const u = String(input);
      if (!u.endsWith("/api/meta")) return new Response("nope", { status: 404 });
      metaCalls++;
      const ready = metaCalls >= 3;
      return new Response(
        metaBase({
          dataset_ready: ready,
          perturb_ready: false,
        }),
        { status: 200 },
      );
    });

    const p = withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const toolPromise = client.callTool({
        name: "spatial_viewer_wait_ready",
        arguments: { require_perturb: false, timeout_seconds: 30 },
      });
      await vi.advanceTimersByTimeAsync(1);
      await vi.advanceTimersByTimeAsync(2000);
      await vi.advanceTimersByTimeAsync(2000);
      await vi.advanceTimersByTimeAsync(2000);
      const res = await toolPromise;
      const text = (res.content as { type: string; text: string }[])[0].text;
      expect(text).toContain("READY");
      expect(metaCalls).toBeGreaterThanOrEqual(3);
    });
    await p;
  });

  it("spatial_viewer_wait_ready times out", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fetchImpl = vi.fn(async (input: RequestInfo | URL) => {
      const u = String(input);
      if (!u.endsWith("/api/meta")) return new Response("nope", { status: 404 });
      return new Response(metaBase({ dataset_ready: false }), { status: 200 });
    });

    const p = withTestServer(fetchImpl as unknown as typeof fetch, async (client) => {
      const toolPromise = client.callTool({
        name: "spatial_viewer_wait_ready",
        arguments: { require_perturb: false, timeout_seconds: 7 },
      });
      await vi.advanceTimersByTimeAsync(8000);
      const res = await toolPromise;
      const text = (res.content as { type: string; text: string }[])[0].text;
      expect(text).toContain("TIMEOUT");
    });
    await p;
  });
});

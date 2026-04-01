---
name: spatial-viewer-mcp
description: Drives the SpaceTravLR spatial viewer via MCP (ext-apps UI + HTTP API tools). Use when the user works with spatial_viewer, MCP spatial-viewer tools, Deck.gl spatial transcriptomics UI, in-silico perturbation from chat, betadata collect_interactions, cluster labeling, or asks how to wire Cursor/MCP to SpaceTravLR_rust.
---

# SpaceTravLR spatial viewer MCP

## When to use

Apply this skill when automating or assisting with:

- Opening the inline spatial viewer (`show_spatial_viewer`) and waiting for loads
- Screenshots (`spatial_viewer_capture_render`), UI control (`spatial_viewer_control`), perturbation from the UI (`spatial_viewer_run_perturb`)
- Server-side queries that do not require the iframe (`spatial_viewer_get_meta`, `spatial_viewer_cluster_expression`, `spatial_viewer_perturb_summary`, etc.)
- Cancelling long jobs or debugging “not ready” / CORS / API base URL mismatches

**Source of truth for tool names and parameters:** `web/spatial_viewer/mcp/server.ts`. Before calling any MCP tool, read the client’s tool descriptor JSON (schema) if available; do not guess required fields.

## Repo layout

| Piece | Path |
| --- | --- |
| MCP stdio entry (run this) | `web/spatial_viewer/mcp/stdio.ts` |
| MCP server implementation (tools + `ui://` resource) | `web/spatial_viewer/mcp/server.ts` |
| Viewer frontend build | `web/spatial_viewer/` → `npm run build` or `npm run build:all` (MCP bundle) |
| Rust HTTP API + static hosting | `src/bin/spatial_viewer.rs` |

## Prerequisites

1. **Build viewer** (from repo root): `cd web/spatial_viewer && npm install && npm run build:all && cd ../..`
2. **Run Rust API** with CORS for MCP iframe origins:

```bash
cargo run --features spatial-viewer --bin spatial_viewer -- \
  --allow-cors --bind 127.0.0.1 --port 8080 \
  --static-dir web/spatial_viewer/dist
```

Add `--run-toml path/to/spacetravlr_run_repro.toml` when perturbation / betadata features are needed.

3. **Run MCP server** (cwd = `web/spatial_viewer`): `npm run mcp:serve` or `npx tsx mcp/stdio.ts` (loads `mcp/server.ts` without side effects on import).

Register in the MCP client with working directory `web/spatial_viewer` and args `tsx`, `mcp/stdio.ts` (or `npm run mcp:serve`). Set **`SPATIAL_VIEWER_API_BASE`** on the MCP process if the API is not `http://127.0.0.1:8080`. For extra allowed connect origins (CSP), use **`SPATIAL_VIEWER_CONNECT_ORIGINS`** (comma-separated); see `connectDomainList()` in `server.ts`.

## Typical agent workflow

1. Ensure the Rust server is up (`spatial_viewer_check_progress` or `spatial_viewer_get_meta`).
2. **`show_spatial_viewer`**: pass `adata_path` (host-resolved path), optional `layer`, `cluster_annot`, `network_dir`, `run_toml`, and `api_base_url` if not default.
3. **`spatial_viewer_wait_ready`**: after opening the UI, wait for `dataset_ready` (and `perturb_ready` unless `require_perturb: false`). Loading often takes tens of seconds to a few minutes.
4. Run analysis tools; for long perturbations, poll **`spatial_viewer_check_progress`** or use **`spatial_viewer_cancel_jobs`** if stuck.

## Tools (summary)

**App tools** (structured content goes to the open `ui://spacetravlr/spatial-viewer.html` iframe; require the UI resource active):

| Tool | Role |
| --- | --- |
| `show_spatial_viewer` | Open viewer + session paths / API base |
| `spatial_viewer_capture_render` | PNG of current Deck.gl view to chat |
| `spatial_viewer_run_perturb` | In-iframe perturbation preview (+ optional UMAP quiver, chat Δ summary) |
| `spatial_viewer_collect_interactions` | Betadata β aggregation for selection/type/cluster |
| `spatial_viewer_control` | Gene / color mode / status bar; **`betadata_gene`**, **`betadata_column`**, **`apply_betadata`** → GET `/api/betadata/values` and spatial cell colors (CellID = per-cell β) |
| `spatial_viewer_received_ligand` | Weighted received-ligand layer (POST `/api/spatial/received_ligand`) |
| `spatial_viewer_signature_umap` | Gene-set signature quiver on UMAP (POST `/api/umap/signature_field`; VirtualTissue-style) |
| `spatial_viewer_splash_network` | Splash derivative network A→B (D3 force graph in viewer; POST `/api/perturb/splash_network`) |

**Server tools** (MCP process talks to Rust `fetch` directly; work without relying on iframe message routing):

| Tool | Role |
| --- | --- |
| `spatial_viewer_get_meta` | Full `/api/meta` + human summary |
| `spatial_viewer_check_progress` | Short status line |
| `spatial_viewer_wait_ready` | Poll until ready or timeout |
| `spatial_viewer_cluster_expression` | POST `/api/cluster/mean_expression` (≤200 genes) |
| `spatial_viewer_label_clusters` | POST `/api/meta/label_clusters` |
| `spatial_viewer_perturb_summary` | POST `/api/perturb/summary` — **blocking** 30–120s+; does not repaint the UI |
| `spatial_viewer_perturb_reference_similarity` | POST `/api/perturb/reference_similarity` — cosine to a reference cell-type centroid before/after GRN perturb (**blocking**); use to test “more Tfh-like” |
| `spatial_viewer_perturb_neighbor_sanity` | POST `/api/perturb/neighbor_sanity` — single-cell perturbation; compares spatial neighbors vs remote **blocking** |
| `spatial_viewer_cancel_jobs` | POST `/api/cancel` |
| `spatial_viewer_report_context` | Text from UI “Send context to chat” into the thread |
| `spatial_viewer_splash_network_json` | POST `/api/perturb/splash_network` — raw JSON graph in chat (no iframe); same inputs as app tool; blocking like one perturb iteration |

**Pairwise betadata (HTTP only):** `POST /api/betadata/pair_lr` with `{ "cell_a": i, "cell_b": j, "top_n": 25 }` returns top ligand–receptor β for that cell pair (same Cluster vs CellID row mapping as collect). The viewer section **Pair cells — top L–R β** calls this from **Spatial** layout with a neighbor-radius pick for cell B.

**Received ligand (HTTP):** `POST /api/spatial/received_ligand` with JSON body `{ "source": "adata"|"model", "genes": [...], "matrix": "lr"|"tfl", "radius"?, "scale_factor"?, "use_grid"?, "grid_factor"?, "aggregate"? }` returns `application/octet-stream` of little-endian `f32` per cell (`n_obs` values). **adata**: recomputes Gaussian weighted ligand signal from expression + spatial (optional `aggregate` across multiple ligands). **model** / **runtime**: requires `perturb_ready`; `genes` must be a single column name present in the training received-ligand matrix. UI: Color source **Received ligand**; MCP: `spatial_viewer_received_ligand` or `spatial_viewer_control` with `color_source: "received_ligand"` and optional `received_ligand_*` fields.

**Reference similarity (“more Tfh-like?”) (HTTP + MCP):** `POST /api/perturb/reference_similarity` with JSON `{ ...PerturbPreviewBody fields (gene, desired_expr, scope, n_propagation?), "reference": { "type": "cell_type_name", "name": "T_follicular_helper" } | cluster | all, "genes"?: ["BCL6",...], "exclude_perturb_cells_from_reference"?: true }`. Runs the same GRN perturbation as `/api/perturb/summary`, then for each cell in the **perturb scope** computes **cosine similarity** to the **mean expression vector** of **reference** cells (same gene columns as the runtime, or the `genes` subset). Default **`exclude_perturb_cells_from_reference`: true** drops perturb-target cells from the centroid so e.g. Tfh are compared to *other* Tfh. Response includes `mean_cosine_before`, `mean_cosine_after`, `mean_delta_cosine` (positive ⇒ shifted toward the reference profile in that subspace). MCP: `spatial_viewer_perturb_reference_similarity`.

**Gene signature UMAP (HTTP + UI):** `POST /api/umap/signature_field` with `{ "genes": ["G1","G2",...], "n_knn"?, "grid_scale"?, "vector_scale"?, "magnitude_threshold"?, "gradient_gain"?, "mask_with_perturb_quiver"?, "mask_quick_ko"?, "mask_perturb"?: { "gene", "desired_expr", "scope", "n_propagation"? }, "export_svg"? }` returns JSON `{ nx, ny, grid_x, grid_y, u, v, signature_per_cell[], svg_export_path? }` (same grid layout as `/api/perturb/umap-field`). Implements the **KNN → grid → gradient → sqrt-normalize → scale** pipeline from `SpaceTravLR/virtual_tissue.py` `signature2gradient`. Optional **mask** zeros signature arrows where the perturbation transition field is zero. UI: **Perturbation & UMAP quiver** → **Gene signature on UMAP**; MCP: `spatial_viewer_signature_umap`.

**Splash network (HTTP + UI):** `POST /api/perturb/splash_network` with `{ "gene_a", "gene_b", "scope": PerturbScope (same tag union as perturb preview), "surround_hops": 0–4, "max_nodes": 6–64 }`. Server runs **`compute_splash_all`** on baseline WL + expression, aggregates each ∂(target)/∂(modulator) over cells in scope, builds a directed edge list, finds a shortest path **A → B**, then expands **undirected hops** around that path and trims to **max_nodes**. Response JSON: `nodes[]` (`id`, `on_path`, `role`), `links[]` (`source`, `target`, `weight`, `abs_weight`), `path`, `path_found`, `n_cells_used`, optional `message`. UI: **Splash signal network** panel (D3 force + zoom). MCP app: `spatial_viewer_splash_network`; MCP server (raw JSON in thread): `spatial_viewer_splash_network_json`.

## Conventions and pitfalls

- **`api_base_url`**: Optional on many tools; defaults to `SPATIAL_VIEWER_API_BASE` or `http://127.0.0.1:8080`. Keep Rust, MCP env, and any `show_spatial_viewer` argument aligned.
- **Paths** (`adata_path`, `network_dir`, `run_toml`) are read by the **machine running `spatial_viewer`**, not the chat client.
- **`spatial_viewer_run_perturb`** supports `scope`: `all`, `selection`, `cell_type`, `cluster` (see Zod schema in `server.ts`). **`spatial_viewer_perturb_summary`** only supports `all`, `cell_type`, `cluster`.
- Perturbation UI/API requires `--run-toml` and successful load (`perturb_ready`).
- Do not enable `--allow-cors` on internet-facing deployments without additional controls.

## Related docs

Project overview and CLI flags: repository `README.md` (MCP / CORS section).

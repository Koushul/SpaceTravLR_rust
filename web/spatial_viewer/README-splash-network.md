# Splash signal network (gene A → gene B)

This document summarizes the **splash derivative network** feature in the spatial viewer: what it implements, what it means biologically, and how it connects to the rest of **SpaceTravLR**.

## What we implemented

### Server (Rust, `spatial_viewer` binary)

- **Endpoint:** `POST /api/perturb/splash_network`
- **Requirements:** Dataset loaded, **`perturb_ready`** (training run TOML + loaded **Betabase** and baseline weighted ligands), same as in-silico perturbation.
- **Computation:**
  1. For every **trained target gene**, run **`splash()`** on the **baseline** state: current expression matrix plus **initial** received-ligand matrices (`rw_ligands_init`, `rw_tfligands_init`), with the same `beta_scale_factor` and `beta_cap` as perturbation.
  2. For each edge **modulator → target**, aggregate splash entries over cells in the chosen **scope** (all cells, one cell-type label, or one cluster), using the **mean** of the derivative across those cells.
  3. Build a **directed graph**: an edge from gene *m* to trained target *t* exists when that mean derivative is above a small threshold (relative to the largest magnitude in the graph).
  4. Find a **shortest directed path** from **gene A** to **gene B** (up to 16 hops).
  5. Expand the subgraph with **undirected** hops around path nodes (**surround_hops**, 0–4), then **trim** to **max_nodes** (6–64), keeping path-related genes and the strongest remaining incident edges.
- **Response:** JSON with `nodes` (id, `on_path`, `role`: source / sink / path / context), `links` (signed `weight`, `abs_weight`), optional `path`, `path_found`, `n_cells_used`, and sometimes a human-readable `message` if no directed path exists.

The heavy lifting reuses the library entry point **`compute_splash_all`**, which mirrors the same splash step used inside **`perturb_with_targets`** (GRN propagation).

### Viewer UI (TypeScript + D3)

- New panel under **Perturbation & UMAP quiver**: **Gene A**, **Gene B**, sliders for **context hops** and **max nodes**, and **Compute splash network**.
- The **cell mask** matches the **perturbation scope** (all / cell type / cluster) so users stay aligned with how they already subset cells for KO previews.
- **D3** force-directed layout: link width scales with |derivative|, **green** vs **pink** arrows for **positive** vs **negative** local sensitivity, draggable nodes, scroll zoom, simple tooltips.

### MCP

- **App tool** `spatial_viewer_splash_network`: opens or updates the embedded viewer and sends structured content so the UI fills genes, optional scope, hops, and max nodes, then runs the same compute.
- **Server tool** `spatial_viewer_splash_network_json`: calls the HTTP API directly and returns the JSON graph for agents that do not rely on the iframe.

---

## Biological meaning

### What splash measures

**Splash** computes **partial derivatives** of each **trained model’s predicted target expression** with respect to each **modulator** in that model: transcription factors, ligand and receptor sides of LR pairs, and TFL (transcription-factor–ligand) style terms. In short, for a small hypothetical change in modulator *m* (holding other inputs fixed in the local linearization), splash approximates how much **target** *t* would move **per unit** of that change, **in the current expression and neighborhood signaling context**.

So each **directed edge** *m* → *t* in the network is **not** a generic “gene–gene correlation.” It is **mechanistic within the fitted SpaceTravLR model**: “under this run’s betas and this cell’s (or subset’s mean) state, perturbing *m*’s channel in the model pushes *t* up or down with this signed sensitivity.”

### Path from A to B

A **directed path** **A → … → B** is a chain of such local sensitivities. Biologically, you can read it as a **hypothetical propagation story** in the **learned** program: nudging upstream programs (ligands, receptors, TFs as encoded in the model) that the trainer associated with moving **B**, possibly through intermediate trained targets. It is **not** guaranteed to be the true causal chain in vivo; it is the **model’s** Jacobian structure averaged over your chosen cells.

If **no directed path** is found, the API still returns a **neighborhood subgraph** around A and B in the **undirected** sense of the derivative graph, so you can inspect **strong local couplings** even when a strict A→B chain is missing.

### Why gene B must be a trained target

Splash rows are produced **per target gene that has a trained BetaFrame** (a `*_betadata.feather` / loaded frame). **Gene A** only needs to appear as a **modulator** in some model. **Gene B** must be a **target** so that incoming edges represent **∂B/∂·** in the model. If B is not trained, there is no splash row for B and the endpoint is not defined in this framework.

---

## Connection to SpaceTravLR

| Piece | Role |
|--------|------|
| **Betabase / BetaFrame** | Stores trained coefficients; **`splash()`** in Rust matches **`BetaFrame.splash()`** in Python (same partials for TF, LR, TFL terms). |
| **Perturbation pipeline** | Each propagation iteration recomputes splash on the fly and applies **δy ≈ splash · δx** (`perturb_all_cells`). The network view is essentially **inspecting one slice** of that Jacobian at **baseline**, aggregated over a cell set. |
| **Weighted ligands** | Splash depends on **received ligand** fields; the API uses the same **initial** WL matrices as the start of a perturb run, so the graph reflects **spatial signaling context** at load time. |
| **Python / docs** | Repository **README** and SpaceTravLR docs describe splash as the derivative of expression w.r.t. modulators for perturbation; this feature **visualizes** those couplings between two genes of interest. |
| **Virtual tissue / signatures** | Not the same math as signature UMAP gradients, but complementary: signatures summarize **expression programs** on embeddings; splash networks summarize **trained regulatory sensitivity** between genes. |

In one line: **SpaceTravLR learns local gene–program couplings; splash quantifies them; the splash network UI and API turn that into an interpretable, scoped graph from a chosen source gene to a chosen trained target, with MCP and HTTP access for automation.**

---

## Quick reference

- **Build UI + MCP bundle:** `cd web/spatial_viewer && npm run build:all`
- **API:** `POST /api/perturb/splash_network` with `gene_a`, `gene_b`, `scope` (same JSON shape as perturb preview scope), `surround_hops`, `max_nodes`
- **MCP tools:** `spatial_viewer_splash_network` (app) and `spatial_viewer_splash_network_json` (HTTP); see `web/spatial_viewer/mcp/server.ts` and `.agents/skills/spatial-viewer-mcp/SKILL.md`

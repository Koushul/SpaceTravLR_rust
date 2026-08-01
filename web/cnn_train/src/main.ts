import "./style.css";
import init, { train_pack, smoke_train_ms } from "../pkg/spacetravlr_cnn_wasm.js";

type Info = {
  h5ad: string;
  default_genes: string[];
  spatial_dim: number;
  max_ligands: number;
  wasm_epochs: number;
  backend: string;
};

type ClusterResult = {
  cluster_id: number;
  n_cells: number;
  mse_epochs: number[];
  diverged: boolean;
  wall_ms: number;
};

type GeneResult = {
  gene: string;
  clusters: ClusterResult[];
  wall_ms: number;
};

const app = document.querySelector<HTMLDivElement>("#app")!;

app.innerHTML = `
  <h1 class="brand">SpaceTravLR</h1>
  <p class="lede">
    Browser WebAssembly CNN trainer (Burn NdArray). The server prepares Lasso anchors and
    spatial maps from your AnnData; Adam epochs run in WASM in this tab.
  </p>
  <section class="panel">
    <div class="row">
      <div class="field">
        <label for="genes">Genes</label>
        <input id="genes" type="text" value="AICDA,CD74" />
      </div>
      <div class="field" style="flex:0 1 8rem">
        <label for="epochs">WASM epochs</label>
        <input id="epochs" type="number" min="1" max="100" value="8" />
      </div>
    </div>
    <div class="actions">
      <button id="btn-smoke" class="secondary" type="button">WASM smoke</button>
      <button id="btn-prepare" class="secondary" type="button">Prepare packs</button>
      <button id="btn-train" type="button">Train in WASM</button>
    </div>
    <p class="meta" id="meta">Loading…</p>
  </section>
  <section class="panel">
    <div class="log" id="log"></div>
    <div id="results"></div>
  </section>
`;

const logEl = document.querySelector<HTMLDivElement>("#log")!;
const metaEl = document.querySelector<HTMLParagraphElement>("#meta")!;
const resultsEl = document.querySelector<HTMLDivElement>("#results")!;
const genesInput = document.querySelector<HTMLInputElement>("#genes")!;
const epochsInput = document.querySelector<HTMLInputElement>("#epochs")!;
const btnSmoke = document.querySelector<HTMLButtonElement>("#btn-smoke")!;
const btnPrepare = document.querySelector<HTMLButtonElement>("#btn-prepare")!;
const btnTrain = document.querySelector<HTMLButtonElement>("#btn-train")!;

function log(line: string) {
  logEl.textContent += (logEl.textContent ? "\n" : "") + line;
  logEl.scrollTop = logEl.scrollHeight;
}

function setBusy(busy: boolean) {
  btnSmoke.disabled = busy;
  btnPrepare.disabled = busy;
  btnTrain.disabled = busy;
}

function renderResults(genes: GeneResult[]) {
  resultsEl.innerHTML = genes
    .map((g) => {
      const cards = g.clusters
        .map((c) => {
          const last = c.mse_epochs[c.mse_epochs.length - 1];
          const first = c.mse_epochs[0];
          const max = Math.max(...c.mse_epochs.filter((x) => Number.isFinite(x)), 1e-6);
          const bars = c.mse_epochs
            .map((m, i) => {
              const pct = Number.isFinite(m) ? Math.min(100, (m / max) * 100) : 0;
              return `<div class="bar-row"><span>ep ${i + 1}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><span>${Number.isFinite(m) ? m.toFixed(4) : "NaN"}</span></div>`;
            })
            .join("");
          return `<div class="gene-card"><h3>${g.gene} · cluster ${c.cluster_id} · ${c.n_cells} cells${c.diverged ? " · DIVERGED" : ""}</h3><p class="meta">${c.wall_ms} ms · MSE ${first?.toFixed(4) ?? "—"} → ${last?.toFixed(4) ?? "—"}</p><div class="bars">${bars}</div></div>`;
        })
        .join("");
      return `<div><h2 style="font-size:1.15rem;margin:1.2rem 0 0.2rem">${g.gene} <span class="meta">(${g.wall_ms} ms total)</span></h2>${cards}</div>`;
    })
    .join("");
}

async function loadInfo() {
  const r = await fetch("/api/info");
  const info = (await r.json()) as Info;
  genesInput.value = info.default_genes.join(",");
  epochsInput.value = String(info.wasm_epochs);
  metaEl.textContent = `${info.backend} · spatial_dim=${info.spatial_dim} · max_ligands=${info.max_ligands} · ${info.h5ad}`;
}

async function main() {
  try {
    await init();
    log("WASM module loaded.");
    await loadInfo();
  } catch (e) {
    metaEl.textContent = `WASM init failed: ${e}`;
    log(String(e));
  }

  btnSmoke.addEventListener("click", () => {
    setBusy(true);
    try {
      const ms = smoke_train_ms();
      log(`WASM smoke train: ${ms} ms`);
    } catch (e) {
      log(`smoke failed: ${e}`);
    } finally {
      setBusy(false);
    }
  });

  btnPrepare.addEventListener("click", async () => {
    setBusy(true);
    log("Preparing packs on server (Lasso + spatial maps)…");
    try {
      const r = await fetch("/api/prepare", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          genes: genesInput.value,
          epochs: Number(epochsInput.value) || 8,
          native_cnn: false,
        }),
      });
      const text = await r.text();
      if (!r.ok) throw new Error(text);
      const status = JSON.parse(text);
      log(`Packs ready for ${status.genes?.join(", ")} (${status.elapsedMs ?? status.elapsed_ms} ms prep)`);
      for (const g of status.results ?? []) {
        log(`  ${g.gene}: ${g.nClusters ?? g.n_clusters} clusters`);
      }
    } catch (e) {
      log(`prepare failed: ${e}`);
    } finally {
      setBusy(false);
    }
  });

  btnTrain.addEventListener("click", async () => {
    setBusy(true);
    resultsEl.innerHTML = "";
    const genes = genesInput.value
      .split(",")
      .map((g) => g.trim())
      .filter(Boolean);
    const epochs = Number(epochsInput.value) || 8;
    log(`Preparing + WASM-training ${genes.join(", ")} for ${epochs} epochs…`);
    try {
      const prep = await fetch("/api/prepare", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ genes: genes.join(","), epochs, native_cnn: false }),
      });
      if (!prep.ok) throw new Error(await prep.text());
      log("Packs ready. Fetching bincode packs into WASM…");

      const out: GeneResult[] = [];
      for (const gene of genes) {
        const pr = await fetch(`/api/pack?gene=${encodeURIComponent(gene)}`);
        if (!pr.ok) throw new Error(await pr.text());
        const buf = new Uint8Array(await pr.arrayBuffer());
        log(`  ${gene}: pack ${buf.byteLength} bytes → WASM train…`);
        const t0 = performance.now();
        const result = train_pack(buf) as GeneResult;
        const dt = Math.round(performance.now() - t0);
        log(`  ${gene}: done in ${dt} ms (wasm wall ${result.wall_ms} ms), ${result.clusters.length} clusters`);
        out.push(result);
      }
      renderResults(out);
      log("All genes finished.");
    } catch (e) {
      log(`train failed: ${e}`);
    } finally {
      setBusy(false);
    }
  });
}

main();

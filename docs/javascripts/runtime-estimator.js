// ── Calibration (runtime formula) ───────────────────────────────────────────
const PER_GENE_BASE = 26.259;
const CELLS_EXP = 0.85;
const CELLS_REF = 200;
const SPATIAL_EXP = 1.5;
const SPATIAL_REF = 16;
const EPOCHS_REF = 100;
// ln(2): normalize so N_eff = N_cnn when N = N_cnn (replaces hard min(N, N_cnn) cap)
const CNN_CELLS_LOG_NORM = Math.LN2;
// < 1 dampens cost of cells beyond N_cnn (0.4 ≈ strong tail; 1.0 = plain log only)
const CNN_CELLS_SATURATION_POWER = 0.4;

// ── Slider ranges (edit min / max / step / default here) ────────────────────
const LOG_SLIDER_MAX = 1000; // internal 0…1000 scale for log-scaled sliders

const N_OBS = { min: 100, max: 1_000_000, default: 50_00 };
const N_GENES = { min: 1, max: 8_000, step: 10, default: 3500 };
const SPATIAL_DIM = { min: 8, max: 256, step: 4, default: 32 };
const EPOCHS = { min: 1, max: 300, step: 1, default: 100 };
const N_PARALLEL = { min: 1, max: 128, step: 1, default: 9 };
const EXT_WORKERS = { min: 0, max: 100, step: 1, default: 5 };
const CNN_MAX_CELLS = { min: 500, max: 10_000, step: 100, default: 3_000 };
const GPU_SPEEDUP = { min: 1, max: 100, step: 5, default: 15 };

/** Soft cell load: ~linear below N_cnn; above N_cnn extra cells add diminishing cost. */
function effectiveCells(nObs, nCnn) {
  if (nObs <= 0 || nCnn <= 0) return 0;
  const logRatio = Math.log1p(nObs / nCnn) / CNN_CELLS_LOG_NORM;
  return nCnn * logRatio ** CNN_CELLS_SATURATION_POWER;
}

function runtimeSec(state) {
  const w = Math.max(1, state.wLocal + state.wExt);
  const nEff = effectiveCells(state.nObs, state.nCnn);
  const sGpu = Math.max(0.1, state.sGpu);

  return (
    (state.nGenes / (w * sGpu)) *
    PER_GENE_BASE *
    (nEff / CELLS_REF) ** CELLS_EXP *
    (state.spatialDim / SPATIAL_REF) ** SPATIAL_EXP *
    (state.epochs / EPOCHS_REF)
  );
}

function formatDuration(sec) {
  if (!Number.isFinite(sec) || sec < 0) return { main: "—", sub: "" };
  if (sec < 120) return { main: `${sec.toFixed(0)} s`, sub: "" };
  if (sec < 86400) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return {
      main: h > 0 ? `${h} h ${m} m` : `${m} m`,
      sub: `${sec.toFixed(0)} s`,
    };
  }
  const d = sec / 86400;
  const h = Math.floor((sec % 86400) / 3600);
  return {
    main: `${d.toFixed(1)} days`,
    sub: h > 0 ? `${Math.floor(sec / 3600)} h total · ${sec.toFixed(0)} s` : `${sec.toFixed(0)} s`,
  };
}

function logToLinear(slider, min, max) {
  const t = slider / LOG_SLIDER_MAX;
  return Math.round(min * (max / min) ** t);
}

function linearToLog(value, min, max) {
  const clamped = Math.max(min, Math.min(max, value));
  return Math.round(
    (LOG_SLIDER_MAX * Math.log(clamped / min)) / Math.log(max / min)
  );
}

function sliderRow(id, label, min, max, step, value, format) {
  return `
    <div class="st-runtime-estimator__control">
      <div class="st-runtime-estimator__head">
        <span class="st-runtime-estimator__name">${label}</span>
        <span class="st-runtime-estimator__val" data-out="${id}">${format(value)}</span>
      </div>
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" />
    </div>`;
}

function mountRuntimeEstimator(root) {
  if (!root || root.dataset.mounted === "1") return;

  const defaults = {
    nObs: N_OBS.default,
    nGenes: N_GENES.default,
    spatialDim: SPATIAL_DIM.default,
    epochs: EPOCHS.default,
    wLocal: N_PARALLEL.default,
    wExt: EXT_WORKERS.default,
    nCnn: CNN_MAX_CELLS.default,
    sGpu: GPU_SPEEDUP.default,
  };

  root.dataset.mounted = "1";
  root.innerHTML = `
    <div class="st-runtime-estimator__card">
      <div class="st-runtime-estimator__hero">
        <p class="st-runtime-estimator__label">Estimated SpaceTravLR flight time</p>
        <p class="st-runtime-estimator__time" data-out="hero-main">—</p>
        <p class="st-runtime-estimator__subtime" data-out="hero-sub"></p>
      </div>
      <div class="st-runtime-estimator__grid">
        ${sliderRow(
          "st-re-n",
          "<code>n_obs</code> cells",
          linearToLog(N_OBS.min, N_OBS.min, N_OBS.max),
          LOG_SLIDER_MAX,
          1,
          linearToLog(defaults.nObs, N_OBS.min, N_OBS.max),
          (v) => logToLinear(v, N_OBS.min, N_OBS.max).toLocaleString()
        )}
        ${sliderRow("st-re-g", "Genes <code>G</code>", N_GENES.min, N_GENES.max, N_GENES.step, defaults.nGenes, (v) => Number(v).toLocaleString())}
        ${sliderRow("st-re-d", "<code>spatial_dim</code>", SPATIAL_DIM.min, SPATIAL_DIM.max, SPATIAL_DIM.step, defaults.spatialDim, (v) => v)}
        ${sliderRow("st-re-e", "<code>epochs</code>", EPOCHS.min, EPOCHS.max, EPOCHS.step, defaults.epochs, (v) => v)}
        ${sliderRow("st-re-wl", "<code>n_parallel</code>", N_PARALLEL.min, N_PARALLEL.max, N_PARALLEL.step, defaults.wLocal, (v) => v)}
        ${sliderRow("st-re-we", "Jobs submitted", EXT_WORKERS.min, EXT_WORKERS.max, EXT_WORKERS.step, defaults.wExt, (v) => v)}
        ${sliderRow("st-re-ncnn", "<code>cnn_max_cells</code>", CNN_MAX_CELLS.min, CNN_MAX_CELLS.max, CNN_MAX_CELLS.step, defaults.nCnn, (v) => Number(v).toLocaleString())}
        ${sliderRow("st-re-sgpu", "GPU speedup <code>s</code>", GPU_SPEEDUP.min, GPU_SPEEDUP.max, GPU_SPEEDUP.step, defaults.sGpu, (v) => `${Number(v).toFixed(1)}×`)}
      </div>

    </div>`;

  const els = {
    n: root.querySelector("#st-re-n"),
    g: root.querySelector("#st-re-g"),
    d: root.querySelector("#st-re-d"),
    e: root.querySelector("#st-re-e"),
    wl: root.querySelector("#st-re-wl"),
    we: root.querySelector("#st-re-we"),
    ncnn: root.querySelector("#st-re-ncnn"),
    sgpu: root.querySelector("#st-re-sgpu"),
    heroMain: root.querySelector('[data-out="hero-main"]'),
    heroSub: root.querySelector('[data-out="hero-sub"]'),
  };

  function readState() {
    return {
      nObs: logToLinear(Number(els.n.value), N_OBS.min, N_OBS.max),
      nGenes: Number(els.g.value),
      spatialDim: Number(els.d.value),
      epochs: Number(els.e.value),
      wLocal: Number(els.wl.value),
      wExt: Number(els.we.value),
      nCnn: Number(els.ncnn.value),
      sGpu: Number(els.sgpu.value),
    };
  }

  function update() {
    const s = readState();
    const sec = runtimeSec(s);
    const dur = formatDuration(sec);
    const w = Math.max(1, s.wLocal + s.wExt);

    els.heroMain.textContent = dur.main;
    els.heroSub.textContent =
      dur.sub ||
      `W = ${w} · ${s.nGenes} genes · N_eff ≈ ${Math.round(effectiveCells(s.nObs, s.nCnn)).toLocaleString()} cells`;

    const formats = {
      "st-re-n": () => s.nObs.toLocaleString(),
      "st-re-g": () => s.nGenes.toLocaleString(),
      "st-re-d": () => String(s.spatialDim),
      "st-re-e": () => String(s.epochs),
      "st-re-wl": () => String(s.wLocal),
      "st-re-we": () => String(s.wExt),
      "st-re-ncnn": () => s.nCnn.toLocaleString(),
      "st-re-sgpu": () => `${s.sGpu.toFixed(1)}×`,
    };

    for (const [id, fmt] of Object.entries(formats)) {
      const out = root.querySelector(`[data-out="${id}"]`);
      if (out) out.textContent = fmt();
    }
  }

  root.querySelectorAll('input[type="range"]').forEach((inp) => {
    inp.addEventListener("input", update);
  });

  update();
}

function initRuntimeEstimators() {
  document.querySelectorAll(".st-runtime-estimator").forEach((el) => {
    if (el.id === "st-runtime-estimator") mountRuntimeEstimator(el);
  });
}

document$.subscribe(initRuntimeEstimators);

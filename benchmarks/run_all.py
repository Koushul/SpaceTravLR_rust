"""Top-level orchestrator for the scientific scaling study.

Steps
-----
1. Subsample the atera_human_cervix.h5ad source at each requested cell count.
2. For every size, run Rust preprocess and (size-permitting) Python scanpy preprocess
   via the `spacetravlr` binary.
3. For every size, run the CNN training bench in Rust (`scaling_bench`) and Python
   (`bench_cnn_python.py`).
4. At a single mid-size, run a dropout sensitivity sweep against both backends.
5. Persist all results into a single JSON file under `benchmarks/results/`.

The script tolerates partial failures: any subprocess crash is recorded into the JSON
under its own slot so the final plots can still show the comparison up to the point
where Python crashed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent


def load_config(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def step(name: str):
    print(f"\n=== {name} === {time.strftime('%H:%M:%S')}", flush=True)


def run_subsample(n: int, source: str, out_dir: Path, seed: int, stratify: str, uv_with: list) -> dict:
    out = out_dir / f"atera_n{n}.h5ad"
    cmd = [
        "uv",
        "run",
        "--isolated",
        "--python",
        "3.11",
    ]
    for w in uv_with:
        cmd += ["--with", w]
    cmd += [
        "python",
        str(BENCH / "subsample.py"),
        "--n", str(n),
        "--out", str(out),
        "--source", source,
        "--seed", str(seed),
        "--stratify", stratify,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-1500:], "wall_seconds": elapsed}
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        info = {"raw_stdout": proc.stdout[-500:]}
    info["wall_seconds_outer"] = elapsed
    info["ok"] = True
    return info


def run_preprocess(h5: Path, out_dir: Path, binary: Path, mode: str, timeout: int) -> dict:
    cmd = [
        sys.executable,
        str(BENCH / "bench_preprocess.py"),
        "--input", str(h5),
        "--out-dir", str(out_dir),
        "--binary", str(binary),
        "--mode", mode,
        "--timeout", str(timeout),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-1500:], "stdout": proc.stdout[-500:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": f"parse: {e}", "raw": proc.stdout[-500:]}


def run_cnn_rust(binary: Path, n: int, cfg: dict, dropout: float = 0.0) -> dict:
    cmd = [
        str(binary),
        "--n", str(n),
        "--spatial-dim", str(cfg["spatial_dim"]),
        "--n-modulators", str(cfg["n_modulators"]),
        "--n-clusters", str(cfg["n_clusters"]),
        "--epochs", str(cfg["epochs"]),
        "--minibatch", str(cfg["minibatch_size"]),
        "--lr", str(cfg["learning_rate"]),
        "--dropout", str(dropout),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-1500:], "wall_seconds": time.time() - t0}
    try:
        d = json.loads(proc.stdout.strip().splitlines()[-1])
        d["ok"] = True
        return d
    except Exception as e:
        return {"ok": False, "error": f"parse: {e}", "raw": proc.stdout[-500:]}


def run_cnn_python(n: int, cfg: dict, python_bin: Path, dropout: float = 0.0, device: str = "auto") -> dict:
    cmd = [
        str(python_bin),
        str(BENCH / "bench_cnn_python.py"),
        "--n", str(n),
        "--spatial-dim", str(cfg["spatial_dim"]),
        "--n-modulators", str(cfg["n_modulators"]),
        "--n-clusters", str(cfg["n_clusters"]),
        "--epochs", str(cfg["epochs"]),
        "--minibatch", str(cfg["minibatch_size"]),
        "--lr", str(cfg["learning_rate"]),
        "--dropout", str(dropout),
        "--device", device,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "error": "timeout", "wall_seconds": time.time() - t0}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-2000:], "wall_seconds": time.time() - t0}
    try:
        d = json.loads(proc.stdout.strip().splitlines()[-1])
        d["ok"] = True
        return d
    except Exception as e:
        return {"ok": False, "error": f"parse: {e}", "raw": proc.stdout[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(BENCH / "config.json"))
    ap.add_argument("--results-json", type=str, default=str(BENCH / "results/results.json"))
    ap.add_argument("--sizes", type=str, default="",
                    help="Comma-separated override list, e.g. 700,2000,7000")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--skip-dropout", action="store_true")
    ap.add_argument("--skip-subsample", action="store_true")
    ap.add_argument("--cnn-device-python", type=str, default="auto")
    ap.add_argument("--fresh", action="store_true",
                    help="Discard any existing results.json")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.sizes:
        sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = list(cfg["subsample_sizes"])

    data_dir = ROOT / "benchmarks/data"
    results_dir = ROOT / "benchmarks/results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    binary = ROOT / "target/release/spacetravlr"
    rust_cnn_binary = ROOT / "target/release/scaling_bench"
    assert binary.exists(), f"missing {binary}; run cargo build --release"
    assert rust_cnn_binary.exists(), f"missing {rust_cnn_binary}; run cargo build --release --bin scaling_bench"

    UV_BASIC = ["numpy<2", "h5py"]
    torch_venv = ROOT / "benchmarks/.venv-torch"
    py_bin = torch_venv / "bin/python"
    if not py_bin.exists():
        step("Bootstrap PyTorch venv (one-time)")
        subprocess.run(
            ["uv", "venv", "--python", "3.11", str(torch_venv)], check=True
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(py_bin),
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
                "--extra-index-url",
                "https://pypi.org/simple",
                "numpy<2",
                "torch==2.4.1",
            ],
            check=True,
        )

    out_json = Path(args.results_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists() and not args.fresh:
        with open(out_json) as f:
            try:
                results = json.load(f)
            except Exception:
                results = None
    else:
        results = None
    if results is None:
        results = {
            "config": cfg,
            "sizes_run": [],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "subsamples": {},
            "preprocess": {},
            "cnn_scaling": {},
            "dropout": {},
        }
    results.setdefault("subsamples", {})
    results.setdefault("preprocess", {})
    results.setdefault("cnn_scaling", {})
    results.setdefault("dropout", {})
    results["sizes_run"] = sorted(set([*results.get("sizes_run", []), *sizes]))
    results["config"] = cfg

    def flush():
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    step("Subsamples")
    h5_paths = {}
    for n in sizes:
        h5 = data_dir / f"atera_n{n}.h5ad"
        if h5.exists():
            results["subsamples"][str(n)] = {
                "ok": True,
                "cached": True,
                "out": str(h5),
                "size_gb": h5.stat().st_size / 1e9,
            }
        else:
            info = run_subsample(
                n,
                source=cfg["source_h5ad"],
                out_dir=data_dir,
                seed=cfg["rng_seed"],
                stratify=cfg["stratify_obs_column"],
                uv_with=UV_BASIC,
            )
            results["subsamples"][str(n)] = info
        h5_paths[n] = h5
        flush()
        print(f"  n={n} -> {results['subsamples'][str(n)]}", flush=True)

    if not args.skip_preprocess:
        step("Preprocess (rust + python)")
        skip_py_geq = int(cfg["preprocess"].get("skip_python_for_sizes_geq", 10**9))
        for n in sizes:
            h5 = h5_paths.get(n)
            if h5 is None or not h5.exists():
                continue
            out_dir = results_dir / f"preprocess_n{n}"
            mode = "rust" if n >= skip_py_geq else "both"
            existing = results["preprocess"].get(str(n))
            need_rust = not (existing and (existing.get("rust") or {}).get("ok"))
            need_py = mode == "both" and not (existing and (existing.get("python") or {}).get("ok"))
            run_mode = None
            if need_rust and need_py:
                run_mode = "both"
            elif need_rust:
                run_mode = "rust"
            elif need_py:
                run_mode = "python"
            if run_mode is None:
                print(f"  n={n} preprocess (cached)", flush=True)
                continue
            r = run_preprocess(h5, out_dir, binary, run_mode, cfg["preprocess"]["per_run_timeout_seconds"])
            r["mode_attempted"] = run_mode
            if existing:
                if "rust" in existing and "rust" not in r:
                    r["rust"] = existing["rust"]
                if "python" in existing and "python" not in r:
                    r["python"] = existing["python"]
            results["preprocess"][str(n)] = r
            flush()
            print(f"  n={n} preprocess -> mode={run_mode} rust_ok={r.get('rust', {}).get('ok')} py_ok={r.get('python', {}).get('ok')}", flush=True)

    if not args.skip_cnn:
        step("CNN scaling (rust + python)")
        cnn_cfg = cfg["cnn_scaling"]
        for n in sizes:
            existing = results["cnn_scaling"].get(str(n)) or {}
            r_rust = existing.get("rust")
            r_py = existing.get("python")
            if not (r_rust and r_rust.get("ok")):
                r_rust = run_cnn_rust(rust_cnn_binary, n, cnn_cfg)
            if not (r_py and r_py.get("ok")):
                r_py = run_cnn_python(n, cnn_cfg, py_bin, device=args.cnn_device_python)
            results["cnn_scaling"][str(n)] = {"rust": r_rust, "python": r_py}
            flush()
            r_t = r_rust.get("total_seconds")
            p_t = r_py.get("total_seconds")
            print(f"  n={n} cnn rust={r_t}s python={p_t}s", flush=True)

    if not args.skip_dropout:
        step("Dropout sensitivity")
        d_cfg = cfg["dropout_experiment"]
        d_n = d_cfg["n_cells"]
        cnn_like = {k: d_cfg[k] for k in ("spatial_dim", "n_modulators", "n_clusters", "epochs", "minibatch_size", "learning_rate")}
        for drop in d_cfg["dropouts"]:
            key = f"{drop}"
            existing = results["dropout"].get(key) or {}
            r_rust = existing.get("rust")
            r_py = existing.get("python")
            if not (r_rust and r_rust.get("ok")):
                r_rust = run_cnn_rust(rust_cnn_binary, d_n, cnn_like, dropout=drop)
            if not (r_py and r_py.get("ok")):
                r_py = run_cnn_python(d_n, cnn_like, py_bin, dropout=drop, device=args.cnn_device_python)
            results["dropout"][key] = {"rust": r_rust, "python": r_py}
            flush()
            print(f"  dropout={drop} rust_mse={r_rust.get('final_mse')} py_mse={r_py.get('final_mse')}", flush=True)

    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    flush()
    print(f"\nResults written to: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

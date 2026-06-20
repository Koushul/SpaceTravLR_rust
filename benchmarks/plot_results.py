"""Plot Rust vs Python scaling from the results JSON.

Produces under `benchmarks/results/plots/`:

- `preprocess_scaling.png`  — wall time vs cell count (rust + python; log-log)
- `preprocess_memory.png`   — peak RSS vs cell count
- `cnn_scaling.png`         — CNN total seconds vs cell count, both impls
- `cnn_throughput.png`      — cells/sec vs cell count
- `dropout_time.png`        — training time vs input dropout fraction
- `dropout_loss.png`        — final MSE vs input dropout fraction

Each plot writes a small companion CSV next to the .png for downstream re-use.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_get(d, *path, default=None):
    cur = d
    for p in path:
        if cur is None or not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _sorted_int_keys(d):
    return sorted([int(k) for k in d.keys()])


def plot_preprocess(results, out_dir: Path):
    pp = results.get("preprocess", {})
    if not pp:
        return
    ns = _sorted_int_keys(pp)
    rust_t = []
    py_t = []
    rust_rss = []
    py_rss = []
    rust_ok = []
    py_ok = []
    for n in ns:
        e = pp[str(n)]
        r = e.get("rust", {}) or {}
        p = e.get("python", {}) or {}
        rust_t.append(r.get("wall_seconds") if r.get("ok") else None)
        py_t.append(p.get("wall_seconds") if p.get("ok") else None)
        rust_rss.append((r.get("max_rss_kb") or 0) / 1024.0 if r.get("ok") else None)
        py_rss.append((p.get("max_rss_kb") or 0) / 1024.0 if p.get("ok") else None)
        rust_ok.append(bool(r.get("ok")))
        py_ok.append(bool(p.get("ok")))

    csv_path = out_dir / "preprocess.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_cells", "rust_seconds", "python_seconds", "rust_rss_mb", "python_rss_mb", "rust_ok", "python_ok"])
        for n, rt, pt, rr, pr, ro, po in zip(ns, rust_t, py_t, rust_rss, py_rss, rust_ok, py_ok):
            w.writerow([n, rt, pt, rr, pr, int(ro), int(po)])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    rust_ns = [n for n, v in zip(ns, rust_t) if v is not None]
    rust_vs = [v for v in rust_t if v is not None]
    py_ns = [n for n, v in zip(ns, py_t) if v is not None]
    py_vs = [v for v in py_t if v is not None]
    if rust_ns:
        ax.plot(rust_ns, rust_vs, marker="o", color="#d6604d", lw=2.0, label="Rust (single_rust + magic-impute crate)")
    if py_ns:
        ax.plot(py_ns, py_vs, marker="s", color="#4393c3", lw=2.0, label="Python (Scanpy + magic-impute)")
    for n, po, pt in zip(ns, py_ok, py_t):
        if not po:
            ax.scatter([n], [max(py_vs) if py_vs else 1.0], marker="x", color="#4393c3", s=120)
            ax.annotate("py crash/oom", (n, max(py_vs) if py_vs else 1.0), fontsize=8, color="#4393c3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Wall time (seconds, log scale)")
    ax.set_title("Preprocess + impute: Rust vs Python (scanpy) — atera_human_cervix")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "preprocess_scaling.png", dpi=150)
    plt.close(fig)

    if any(rr is not None for rr in rust_rss) or any(pr is not None for pr in py_rss):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        rmem_ns = [n for n, v in zip(ns, rust_rss) if v]
        rmem_vs = [v for v in rust_rss if v]
        pmem_ns = [n for n, v in zip(ns, py_rss) if v]
        pmem_vs = [v for v in py_rss if v]
        if rmem_ns:
            ax.plot(rmem_ns, rmem_vs, marker="o", color="#d6604d", lw=2.0, label="Rust peak RSS")
        if pmem_ns:
            ax.plot(pmem_ns, pmem_vs, marker="s", color="#4393c3", lw=2.0, label="Python peak RSS")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cells")
        ax.set_ylabel("Max RSS (MB, log scale)")
        ax.set_title("Preprocess peak memory: Rust vs Python")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "preprocess_memory.png", dpi=150)
        plt.close(fig)


def plot_cnn(results, out_dir: Path):
    cs = results.get("cnn_scaling", {})
    if not cs:
        return
    ns = _sorted_int_keys(cs)
    r_t, p_t, r_thr, p_thr, r_mse, p_mse = [], [], [], [], [], []
    for n in ns:
        e = cs[str(n)]
        r = e.get("rust", {}) or {}
        p = e.get("python", {}) or {}
        rt = r.get("total_seconds") if r.get("ok") else None
        pt = p.get("total_seconds") if p.get("ok") else None
        epochs_r = r.get("epochs", 1) or 1
        epochs_p = p.get("epochs", 1) or 1
        r_t.append(rt)
        p_t.append(pt)
        r_thr.append((n * epochs_r / rt) if rt else None)
        p_thr.append((n * epochs_p / pt) if pt else None)
        r_mse.append(r.get("final_mse") if r.get("ok") else None)
        p_mse.append(p.get("final_mse") if p.get("ok") else None)

    with open(out_dir / "cnn_scaling.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_cells", "rust_seconds", "python_seconds", "rust_cells_per_sec", "python_cells_per_sec", "rust_final_mse", "python_final_mse"])
        for n, rt, pt, rs, ps, rm, pm in zip(ns, r_t, p_t, r_thr, p_thr, r_mse, p_mse):
            w.writerow([n, rt, pt, rs, ps, rm, pm])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if any(rt for rt in r_t):
        ax.plot(ns, r_t, marker="o", color="#d6604d", lw=2.0, label="Rust burn (NdArray autodiff CPU)")
    if any(pt for pt in p_t):
        ax.plot(ns, p_t, marker="s", color="#4393c3", lw=2.0,
                label="Python PyTorch (A100 CUDA)" if any(safe_get(cs[str(n)], "python", "backend") == "cuda" for n in ns) else "Python PyTorch")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells")
    ax.set_ylabel("CNN training wall time (seconds)")
    ax.set_title("32x32 CellularNicheNetwork training — Rust vs Python scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cnn_scaling.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if any(rs for rs in r_thr):
        ax.plot(ns, r_thr, marker="o", color="#d6604d", lw=2.0, label="Rust")
    if any(ps for ps in p_thr):
        ax.plot(ns, p_thr, marker="s", color="#4393c3", lw=2.0, label="Python")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Training throughput (cells * epoch / sec)")
    ax.set_title("CNN training throughput vs sample size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cnn_throughput.png", dpi=150)
    plt.close(fig)


def plot_dropout(results, out_dir: Path):
    dd = results.get("dropout", {})
    if not dd:
        return
    drops = sorted([float(k) for k in dd.keys()])
    r_t, p_t, r_mse, p_mse = [], [], [], []
    for d in drops:
        e = dd[f"{d}"]
        r = e.get("rust", {}) or {}
        p = e.get("python", {}) or {}
        r_t.append(r.get("total_seconds") if r.get("ok") else None)
        p_t.append(p.get("total_seconds") if p.get("ok") else None)
        r_mse.append(r.get("final_mse") if r.get("ok") else None)
        p_mse.append(p.get("final_mse") if p.get("ok") else None)
    with open(out_dir / "dropout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dropout", "rust_seconds", "python_seconds", "rust_final_mse", "python_final_mse"])
        for d, rt, pt, rm, pm in zip(drops, r_t, p_t, r_mse, p_mse):
            w.writerow([d, rt, pt, rm, pm])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if any(rt for rt in r_t):
        ax.plot(drops, r_t, marker="o", color="#d6604d", lw=2.0, label="Rust")
    if any(pt for pt in p_t):
        ax.plot(drops, p_t, marker="s", color="#4393c3", lw=2.0, label="Python")
    ax.set_xlabel("Input spatial-map dropout fraction")
    ax.set_ylabel("Training wall time (seconds)")
    ax.set_title("CNN training time vs input sparsity / dropout")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dropout_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if any(m for m in r_mse):
        ax.plot(drops, r_mse, marker="o", color="#d6604d", lw=2.0, label="Rust")
    if any(m for m in p_mse):
        ax.plot(drops, p_mse, marker="s", color="#4393c3", lw=2.0, label="Python")
    ax.set_xlabel("Input spatial-map dropout fraction")
    ax.set_ylabel("Final MSE after fixed epochs")
    ax.set_title("CNN robustness to input dropout (lower MSE = better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dropout_loss.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="benchmarks/results/results.json")
    ap.add_argument("--out-dir", type=str, default="benchmarks/results/plots")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_preprocess(results, out_dir)
    plot_cnn(results, out_dir)
    plot_dropout(results, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()

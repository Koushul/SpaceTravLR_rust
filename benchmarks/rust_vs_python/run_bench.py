#!/usr/bin/env python3
"""
Drive Rust (`bench_steps`) and Python (`bench_steps.py`) per-step microbenchmarks
and print a side-by-side comparison table. Compares this crate to the Python
reference implementation at https://github.com/jishnu-lab/SpaceTravLR .

Example:

    cd benchmarks/rust_vs_python
    ./run_bench.py --n-cells 256 --repeats 3

Or from the repo root:

    python3 benchmarks/rust_vs_python/run_bench.py --repo-root . --n-cells 512

Use `--uv` to run the Python side with `uv run` (first run downloads PEP 723
dependencies into `tmp/uv_cache`, preferring CPU PyTorch). Otherwise use your
current interpreter and install `numpy`, `scipy`, `numba`, `torch`, `group-lasso`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


STEPS = (
    "received_ligands",
    "spatial_features",
    "xyc2spatial",
    "group_lasso",
    "train_one_gene",
)


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "Cargo.toml").is_file() and (p / "src" / "bin" / "bench_steps.rs").is_file():
            return p
    raise FileNotFoundError(
        "Could not locate SpaceTravLR_rust repo root (Cargo.toml + src/bin/bench_steps.rs)."
    )


def rust_bench_binary(repo_root: Path) -> Path | None:
    release = repo_root / "target" / "release" / "bench_steps"
    debug = repo_root / "target" / "debug" / "bench_steps"
    if release.is_file():
        return release
    if debug.is_file():
        return debug
    return None


def run_rust(
    repo_root: Path,
    step: str,
    args_ns: argparse.Namespace,
) -> dict[str, Any]:
    exe = rust_bench_binary(repo_root)
    if exe is not None:
        cmd = [
            str(exe),
            "--step",
            _rust_step_flag(step),
            "--n-cells",
            str(args_ns.n_cells),
            "--seed",
            str(args_ns.seed),
            "--n-ligands",
            str(args_ns.n_ligands),
            "--n-features",
            str(args_ns.n_features),
            "--n-clusters",
            str(args_ns.n_clusters),
            "--extent",
            str(args_ns.extent),
            "--radius",
            str(args_ns.radius),
            "--spatial-dim",
            str(args_ns.spatial_dim),
            "--epochs",
            str(args_ns.epochs),
            "--n-iter",
            str(args_ns.n_iter),
            "--repeats",
            str(args_ns.repeats),
        ]
    else:
        cmd = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--bin",
            "bench_steps",
            "--manifest-path",
            str(repo_root / "Cargo.toml"),
            "--",
            "--step",
            _rust_step_flag(step),
            "--n-cells",
            str(args_ns.n_cells),
            "--seed",
            str(args_ns.seed),
            "--n-ligands",
            str(args_ns.n_ligands),
            "--n-features",
            str(args_ns.n_features),
            "--n-clusters",
            str(args_ns.n_clusters),
            "--extent",
            str(args_ns.extent),
            "--radius",
            str(args_ns.radius),
            "--spatial-dim",
            str(args_ns.spatial_dim),
            "--epochs",
            str(args_ns.epochs),
            "--n-iter",
            str(args_ns.n_iter),
            "--repeats",
            str(args_ns.repeats),
        ]
    return _run_json(cmd, cwd=repo_root)


def _rust_step_flag(step: str) -> str:
    # clap ValueEnum uses PascalCase variants; pass kebab-case matching derive.
    return step.replace("_", "-")


def run_python(
    script: Path,
    step: str,
    args_ns: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    common = [
        "--step",
        step,
        "--n-cells",
        str(args_ns.n_cells),
        "--seed",
        str(args_ns.seed),
        "--n-ligands",
        str(args_ns.n_ligands),
        "--n-features",
        str(args_ns.n_features),
        "--n-clusters",
        str(args_ns.n_clusters),
        "--extent",
        str(args_ns.extent),
        "--radius",
        str(args_ns.radius),
        "--spatial-dim",
        str(args_ns.spatial_dim),
        "--epochs",
        str(args_ns.epochs),
        "--n-iter",
        str(args_ns.n_iter),
        "--repeats",
        str(args_ns.repeats),
    ]
    if args_ns.use_uv:
        cmd = ["uv", "run", str(script), *common]
        extra: dict[str, str] = {
            "UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cpu",
            "UV_CACHE_DIR": str(repo_root / "tmp" / "uv_cache"),
        }
        return _run_json(cmd, cwd=script.parent, env=extra)
    cmd = [args_ns.python, str(script), *common]
    return _run_json(cmd, cwd=script.parent)


def _run_json(cmd: list[str], cwd: Path | None, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    run_env = None
    if env is not None:
        run_env = os.environ.copy()
        run_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return json.loads(line)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Compare Rust vs Python SpaceTravLR step benchmarks.")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to SpaceTravLR_rust (default: search upward from this script).",
    )
    p.add_argument(
        "--steps",
        default=",".join(STEPS),
        help=f"Comma-separated subset of steps (default: all). Choices: {', '.join(STEPS)}.",
    )
    p.add_argument("--n-cells", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-ligands", type=int, default=32)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--extent", type=float, default=5_000.0)
    p.add_argument("--radius", type=float, default=300.0)
    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for bench_steps.py (install PEP 723 deps yourself, or use --uv).",
    )
    p.add_argument(
        "--uv",
        action="store_true",
        help="Run bench_steps.py via `uv run` (uses repo tmp/uv_cache + CPU PyTorch index; first run may download packages).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object per line instead of a table.",
    )
    args = p.parse_args()
    args.use_uv = bool(args.uv)

    repo_root = args.repo_root
    if repo_root is None:
        repo_root = find_repo_root(here)
    else:
        repo_root = repo_root.resolve()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in steps:
        if s not in STEPS:
            sys.stderr.write(f"Unknown step {s!r}; allowed: {STEPS}\n")
            sys.exit(2)

    py_script = here / "bench_steps.py"
    if not py_script.is_file():
        sys.stderr.write(f"Missing {py_script}\n")
        sys.exit(2)

    rows: list[tuple[str, float, float, float | None]] = []
    for step in steps:
        r = run_rust(repo_root, step, args)
        py = run_python(py_script, step, args, repo_root)
        tw = float(r["wall_s"])
        pw = float(py["wall_s"])
        ratio = (pw / tw) if tw > 0 else None
        rows.append((step, tw, pw, ratio))
        if args.json:
            print(json.dumps({"step": step, "rust": r, "python": py}))

    if args.json:
        return

    w_step = max(len("step"), max(len(x[0]) for x in rows))
    print()
    print(
        f"{'step'.ljust(w_step)}  {'rust_s':>10}  {'python_s':>10}  {'python/rust':>12}  (higher = Rust faster)"
    )
    print(f"{'-' * w_step}  {'-' * 10}  {'-' * 10}  {'-' * 12}")
    for step, tw, pw, ratio in rows:
        rs = f"{tw:.6f}"
        ps = f"{pw:.6f}"
        rr = f"{ratio:.2f}x" if ratio is not None and ratio == ratio else "n/a"
        print(f"{step.ljust(w_step)}  {rs:>10}  {ps:>10}  {rr:>12}")
    print()
    print(f"n_cells={args.n_cells} repeats={args.repeats} seed={args.seed} repo={repo_root}")
    print()


if __name__ == "__main__":
    main()

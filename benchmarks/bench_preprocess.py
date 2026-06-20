"""Preprocessing bench: time `spacetravlr --rust-process-h5ad` vs `--process-h5ad` (scanpy).

For each subsample we call the `spacetravlr` release binary twice — once for the pure
Rust path and once for the embedded Scanpy + magic-impute path via uv — capturing
wall time, peak RSS (via /usr/bin/time -v when available), and exit status. The Scanpy
path is allowed to fail (the question is *when* it crashes); we mark such runs with
`{"ok": false, "error": "..."}` and move on.

Usage::

    python bench_preprocess.py --input atera_n7000.h5ad --out-dir out --binary ./target/release/spacetravlr
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _has_gnu_time() -> bool:
    return shutil.which("/usr/bin/time") is not None


def _parse_time_v(stderr: str) -> dict:
    out = {}
    for line in stderr.splitlines():
        m = re.match(r"\s*Maximum resident set size \(kbytes\):\s*(\d+)", line)
        if m:
            out["max_rss_kb"] = int(m.group(1))
        m = re.match(r"\s*Elapsed \(wall clock\) time.*:\s*(.+)$", line)
        if m:
            out["elapsed_str"] = m.group(1).strip()
    return out


def run_one(cmd: list, log_path: Path, timeout_s: int) -> dict:
    """Run a subprocess with optional /usr/bin/time -v; capture timing, RSS, exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    use_time = _has_gnu_time()
    full = (["/usr/bin/time", "-v"] + cmd) if use_time else cmd
    t0 = time.time()
    err_text = ""
    try:
        with open(log_path, "wb") as fout:
            proc = subprocess.run(
                full,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
        elapsed = time.time() - t0
        err_text = proc.stderr.decode("utf-8", "replace")
        with open(log_path.with_suffix(".stderr.log"), "w") as f:
            f.write(err_text)
        info = _parse_time_v(err_text) if use_time else {}
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "wall_seconds": elapsed,
            "stdout_log": str(log_path),
            "stderr_log": str(log_path.with_suffix(".stderr.log")),
            **info,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "ok": False,
            "error": "timeout",
            "wall_seconds": elapsed,
            "timeout_s": timeout_s,
            "stdout_log": str(log_path),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "wall_seconds": time.time() - t0,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Subsampled .h5ad file")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--binary", type=str, default="./target/release/spacetravlr")
    p.add_argument(
        "--mode",
        choices=["rust", "python", "both"],
        default="both",
        help="Which preprocess path(s) to run",
    )
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--label", type=str, default="")
    args = p.parse_args()

    inp = Path(args.input).resolve()
    if not inp.exists():
        print(json.dumps({"ok": False, "error": f"missing input {inp}"}))
        return 1
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_rust = out_root / "rust"
    out_py = out_root / "python"
    out_rust.mkdir(exist_ok=True)
    out_py.mkdir(exist_ok=True)

    binary = Path(args.binary).resolve()
    assert binary.exists(), f"binary not found: {binary}"

    result = {
        "input": str(inp),
        "label": args.label,
        "input_size_gb": inp.stat().st_size / 1e9,
    }

    env_common = os.environ.copy()
    env_common.setdefault("SPACETRAVLR_FORCE_CPU", "1")
    env_common.setdefault("RUST_BACKTRACE", "1")
    env_common.setdefault("OMP_NUM_THREADS", "8")

    if args.mode in ("rust", "both"):
        cmd = [
            str(binary),
            "--plain",
            "--rust-process-h5ad",
            "--h5ad",
            str(inp),
            "--process-output-dir",
            str(out_rust),
            "--rust-n-top-hvg",
            "2000",
            "--rust-n-neighbors",
            "15",
        ]
        log = out_rust / "spacetravlr_rust_process.log"
        rust_out_h5 = out_rust / (inp.stem + "_rust_processed.h5ad")
        if rust_out_h5.exists():
            try:
                rust_out_h5.unlink()
            except Exception:
                pass
        r = run_one(cmd, log, args.timeout)
        if rust_out_h5.exists():
            try:
                r["output_size_gb"] = rust_out_h5.stat().st_size / 1e9
                rust_out_h5.unlink()
            except Exception:
                pass
        result["rust"] = r

    if args.mode in ("python", "both"):
        cmd = [
            str(binary),
            "--plain",
            "--process-h5ad",
            "--h5ad",
            str(inp),
            "--process-output-dir",
            str(out_py),
        ]
        log = out_py / "spacetravlr_python_process.log"
        py_out_h5 = out_py / (inp.stem + "_processed.h5ad")
        if py_out_h5.exists():
            try:
                py_out_h5.unlink()
            except Exception:
                pass
        r = run_one(cmd, log, args.timeout)
        if py_out_h5.exists():
            try:
                r["output_size_gb"] = py_out_h5.stat().st_size / 1e9
                py_out_h5.unlink()
            except Exception:
                pass
        result["python"] = r

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Execute validation notebook code cells via the bootstrapped analysis Python."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))
from nb_common import BOOT_PY, boot_env, MC38_SITE  # noqa: E402


def run_cell(source: str, cwd: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(source)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [str(BOOT_PY), tmp_path],
            cwd=cwd,
            env=boot_env(),
            capture_output=True,
            text=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Cell failed (exit {proc.returncode})")


def execute_notebook(path: Path, cwd: Path) -> None:
    nb = json.loads(path.read_text())
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            chunks.append(source)
    if not chunks:
        raise RuntimeError("No code cells")
    run_cell("\n\n".join(chunks), cwd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebooks", nargs="*", default=None)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast subset (01, 05, 06, 07) for smoke testing",
    )
    args = ap.parse_args()

    if args.quick:
        names = [
            "01_core_multislice_validation.ipynb",
            "05_paper_findings.ipynb",
            "06_cnn_guide_enrichment.ipynb",
            "07_validation_dashboard.ipynb",
        ]
    else:
        names = args.notebooks or sorted(p.name for p in NB_DIR.glob("*.ipynb"))
    failed: list[str] = []
    for name in names:
        path = NB_DIR / name
        if not path.exists():
            print(f"SKIP missing {name}")
            failed.append(name)
            continue
        print(f"=== {name} ===", flush=True)
        try:
            execute_notebook(path, ROOT)
            print(f"OK {name}")
        except Exception as exc:
            print(f"FAIL {name}: {exc}", file=sys.stderr)
            failed.append(name)

    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")
    print("All notebooks OK.")


if __name__ == "__main__":
    main()

"""Shared helpers for SPAC-seq / SpaceTravLR validation Jupyter notebooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
MC38_SITE = Path(
    "/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/mc38_visiumhd/pyuser/lib/python3.11/site-packages"
)
BOOT_PY = Path(
    os.environ.get(
        "SPACETRAVLR_PYTHON",
        "/software/rhel9/manual/install/rust/1.89.0/python3.11/bin/python3",
    )
)


def boot_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SPACETRAVLR_PY_BOOT"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    extra = os.pathsep.join(str(p) for p in (MC38_SITE, ROOT, SCRIPTS))
    env["PYTHONPATH"] = f"{extra}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else extra
    return env


def bootstrap() -> Path:
    """Add analysis paths to the active notebook kernel."""
    os.chdir(ROOT)
    for p in (str(SCRIPTS), str(ROOT), str(MC38_SITE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return ROOT


def default_config() -> dict[str, str]:
    return {
        "tag": "tuned",
        "spatial_tag": "spatial_v3",
        "pred_dir": "results/predictions_tuned",
        "pred_dir_cnn": "results/predictions_cnn",
        "betadata_dir": "runs/baseline_pooled_seed",
        "betadata_dir_cnn": "runs/baseline_pooled_cnn",
        "baseline_h5ad": "data/pooled/baseline_ntc.h5ad",
    }


def run_script(script: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a validation script in the bootstrapped analysis Python."""
    cmd = [str(BOOT_PY), str(SCRIPTS / script), *args]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=boot_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        msg = f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        raise RuntimeError(msg)
    return proc


def load_json(rel: str | Path) -> dict:
    path = ROOT / rel
    return json.loads(path.read_text()) if path.exists() else {}


def load_csv(rel: str | Path):
    import pandas as pd

    path = ROOT / rel
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def show_figures(glob_pattern: str, max_images: int = 16) -> list[Path]:
    """Display PNG figures in Jupyter; return paths for non-interactive runs."""
    paths = sorted(ROOT.glob(glob_pattern))[:max_images]
    shown: list[Path] = []
    try:
        from IPython.display import Image, display
    except ImportError:
        return paths
    for path in paths:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            continue
        display(Image(filename=str(path)))
        shown.append(path)
    return shown


def artifact_status(paths: list[str]) -> dict[str, bool]:
    return {p: (ROOT / p).exists() for p in paths}

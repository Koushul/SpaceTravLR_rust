"""Re-exec under Rust Python with mc38 site-packages and no user-site."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PY = Path(os.environ.get("SPACETRAVLR_PYTHON", "/software/rhel9/manual/install/rust/1.89.0/python3.11/bin/python3"))
SITE = Path("/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/mc38_visiumhd/pyuser/lib/python3.11/site-packages")
ROOT = Path(__file__).resolve().parent.parent


def ensure_boot() -> None:
    if os.environ.get("SPACETRAVLR_PY_BOOT") == "1":
        return
    env = os.environ.copy()
    env["SPACETRAVLR_PY_BOOT"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    extra = f"{SITE}:{ROOT}"
    env["PYTHONPATH"] = f"{extra}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else extra
    os.execve(str(PY), [str(PY), *sys.argv], env)

#!/usr/bin/env python3
"""Point overlays that use TF modulators at a shared CellOracle priors feather (skip re-inference)."""
from pathlib import Path

import toml

HERE = Path(__file__).resolve().parent
OVER = HERE / "overlays"
RUNS = HERE / "runs"
PRIOR = RUNS / "tf_only__seed" / "celloracle_tf_priors.feather"


def main() -> None:
    if not PRIOR.is_file():
        raise SystemExit(f"missing priors (train tf_only__seed first): {PRIOR}")
    p = str(PRIOR)
    for f in sorted(OVER.glob("*.toml")):
        cfg = toml.load(f)
        grn = cfg.get("grn", {})
        if grn.get("use_tf_modulators", True):
            cfg.setdefault("grn", {})["tf_priors_feather"] = p
        with open(f, "w") as fp:
            toml.dump(cfg, fp)
        print("patched", f.name)


if __name__ == "__main__":
    main()

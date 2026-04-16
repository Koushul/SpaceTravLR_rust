#!/usr/bin/env python3
import copy
import os
from pathlib import Path

import toml

ROOT = Path(__file__).resolve().parents[2]
BASE_CFG = ROOT / "spaceship_config.toml"
OUT_DIR = Path(__file__).resolve().parent / "overlays"
RUNS = Path(__file__).resolve().parent / "runs"

H5AD = "/ix/djishnu/shared/djishnu_kor11/training_data_revision/snrna_human_tonsil.h5ad"
NETWORK = str(ROOT / "data")

MODS = {
    "tf_only": (True, False, False),
    "tf_lr": (True, True, False),
    "tf_lr_ltfl": (True, True, True),
    "lr_only": (False, True, False),
    "ltfl_only": (False, False, True),
}


def main() -> None:
    base = toml.load(BASE_CFG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    for name, (tf, lr, tfl) in MODS.items():
        for spatial in ("seed", "full"):
            cfg = copy.deepcopy(base)
            dcfg = cfg.setdefault("data", {})
            dcfg["adata_path"] = H5AD
            # Must match betadata `Cluster` keys (integer cluster ids in this dataset).
            dcfg["cluster_annot"] = "cell_type_int"
            cfg.setdefault("grn", {})
            cfg["grn"]["network_data_dir"] = NETWORK
            cfg["grn"]["use_tf_modulators"] = tf
            cfg["grn"]["use_lr_modulators"] = lr
            cfg["grn"]["use_tfl_modulators"] = tfl
            cfg["grn"]["max_ligands"] = 40
            cfg.setdefault("training", {})
            cfg["training"]["mode"] = spatial
            cfg["training"]["epochs"] = 12 if spatial == "full" else 3
            cfg.setdefault("execution", {})
            cfg["execution"]["n_parallel"] = 4
            run_name = f"{name}__{spatial}"
            cfg["execution"]["output_dir"] = str(RUNS / run_name)
            stem = Path(H5AD).stem
            cfg.setdefault("training", {})
            cfg["training"]["genes"] = ["IL21", "BCL6"]
            fname = OUT_DIR / f"{run_name}.toml"
            with open(fname, "w") as f:
                toml.dump(cfg, f)
            print("wrote", fname)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end shared tissue CNN pipeline for SlideTags human tonsil."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_GENES = ["AICDA", "CD74", "CD3D", "MS4A6A", "LYZ"]


def pick_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda"
    except Exception:
        pass
    return "cpu"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, default=ROOT / "../../data/h5ad/SlideTags_human_tonsil.h5ad")
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--pretrain-epochs", type=int, default=40)
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--genes", default=",".join(DEFAULT_GENES))
    p.add_argument("--skip-pretrain-variants", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.device = pick_device(args.device)
    gene_list = [g.strip() for g in args.genes.split(",") if g.strip()]

    py = sys.executable
    data_dir = ROOT / "data"
    out = ROOT / "outputs"

    run([py, "split_tonsil.py", "--h5ad", str(args.h5ad.resolve()), "--out-dir", str(data_dir)])

    run(
        [
            py,
            "build_cache.py",
            "--h5ad",
            str(data_dir / "tonsil_finetune.h5ad"),
            "--cache",
            str(out / "finetune_cache.npz"),
            "--spatial-dim",
            str(args.spatial_dim),
            "--force-genes",
            args.genes,
        ]
    )

    variants = ["base", "deep", "wide"]
    for variant in variants:
        run(
            [
                py,
                "pretrain_cnn.py",
                "--h5ad",
                str(data_dir / "tonsil_train.h5ad"),
                "--cache",
                str(out / "train_cache.npz"),
                "--out-dir",
                str(out / "pretrain"),
                "--spatial-dim",
                str(args.spatial_dim),
                "--epochs",
                str(args.pretrain_epochs),
                "--variant",
                variant,
                "--device",
                args.device,
                "--force-genes",
                args.genes,
            ]
        )

    # Pick best variant by transfer linear-probe R² on finetune half
    variant_eval = out / "variant_transfer.json"
    run(
        [
            py,
            "evaluate_variants.py",
            "--train-cache",
            str(out / "train_cache.npz"),
            "--finetune-cache",
            str(out / "finetune_cache.npz"),
            "--pretrain-dir",
            str(out / "pretrain"),
            "--probe-genes",
            args.genes,
            "--out",
            str(variant_eval),
            "--device",
            args.device,
        ]
    )
    if variant_eval.exists():
        vt = json.loads(variant_eval.read_text())
        best = vt.get("best_by_transfer", "base")
        scores = {v: vt["variants"][v]["mean_transfer_r2"] for v in vt.get("variants", {})}
    else:
        scores = {}
        for variant in variants:
            meta_path = out / "pretrain" / f"pretrain_{variant}_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                scores[variant] = meta.get("final_mse", float("inf"))
        best = min(scores, key=scores.get) if scores else "base"
    encoder = out / "pretrain" / f"tissue_encoder_{best}.pt"
    print(f"Selected encoder variant: {best} (transfer scores={scores})")

    run(
        [
            py,
            "finetune_genes.py",
            "--h5ad",
            str(data_dir / "tonsil_finetune.h5ad"),
            "--cache",
            str(out / "finetune_cache.npz"),
            "--encoder",
            str(encoder),
            "--out-dir",
            str(out / "finetune"),
            "--genes",
            args.genes,
            "--spatial-dim",
            str(args.spatial_dim),
            "--epochs",
            str(args.finetune_epochs),
            "--device",
            args.device,
            "--force-genes",
            args.genes,
        ]
    )

    run(
        [
            py,
            "evaluate_halves.py",
            "--train-cache",
            str(out / "train_cache.npz"),
            "--finetune-cache",
            str(out / "finetune_cache.npz"),
            "--encoder",
            str(encoder),
            "--heads-dir",
            str(out / "finetune" / "gene_heads"),
            "--device",
            args.device,
        ]
    )

    meta_paths = [out / "pretrain" / f"pretrain_{v}_meta.json" for v in variants]
    run(
        [
            py,
            "plot_results.py",
            "--finetune-json",
            str(out / "finetune" / "gene_performance.json"),
            "--pretrain-meta",
            *[str(m) for m in meta_paths if m.exists()],
            "--finetune-cache",
            str(out / "finetune_cache.npz"),
            "--encoder",
            str(encoder),
            "--figures",
            str(ROOT / "figures"),
            "--device",
            args.device,
        ]
    )

    summary = {
        "encoder_variant": best,
        "variant_transfer_r2": scores,
        "genes": gene_list,
        "figures": str(ROOT / "figures"),
        "betadata_dir": str(out / "finetune" / "betadata"),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

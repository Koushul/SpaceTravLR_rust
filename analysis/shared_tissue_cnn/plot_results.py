"""Plots: performance comparison, Grad-CAM, activation maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import SpatialCache, build_spatial_cache, cluster_maps_for_cells
from finetune_genes import encode_all, load_encoder
from models import TissueVisionEncoder


def grad_cam(
    encoder: TissueVisionEncoder,
    spatial_map: torch.Tensor,
    spatial_features: torch.Tensor,
    target_layer: str = "conv3",
) -> np.ndarray:
    """Grad-CAM on conv3 output for a single cell."""
    encoder.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_mod, _inp, out):
        activations["conv3"] = out

    def bwd_hook(_mod, _gin, gout):
        gradients["conv3"] = gout[0]

    handle_f = encoder.conv3.register_forward_hook(fwd_hook)
    handle_b = encoder.conv3.register_backward_hook(bwd_hook)

    sm = spatial_map.unsqueeze(0) if spatial_map.dim() == 3 else spatial_map
    sf = spatial_features.unsqueeze(0) if spatial_features.dim() == 1 else spatial_features
    sm = sm.requires_grad_(True)
    feat = encoder(sm, sf)
    score = feat.sum()
    encoder.zero_grad(set_to_none=True)
    score.backward()

    handle_f.remove()
    handle_b.remove()

    act = activations["conv3"].detach()[0]
    grad = gradients["conv3"].detach()[0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=0))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()


def integrated_gradients(
    encoder: TissueVisionEncoder,
    spatial_map: torch.Tensor,
    spatial_features: torch.Tensor,
    steps: int = 32,
) -> np.ndarray:
    """Integrated Gradients attribution on spatial map input."""
    encoder.eval()
    sm = spatial_map.unsqueeze(0) if spatial_map.dim() == 3 else spatial_map
    sf = spatial_features.unsqueeze(0) if spatial_features.dim() == 1 else spatial_features
    baseline = torch.zeros_like(sm)
    total_grad = torch.zeros_like(sm)
    for step in range(1, steps + 1):
        alpha = step / steps
        interp = (baseline + alpha * (sm - baseline)).clone().requires_grad_(True)
        interp.retain_grad()
        feat = encoder(interp, sf)
        score = feat.sum()
        encoder.zero_grad(set_to_none=True)
        score.backward()
        total_grad += interp.grad.detach()
    attr = (sm - baseline) * total_grad / steps
    cam = attr[0, 0].abs()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy()


def plot_per_cluster_heatmap(finetune_json: Path, out_path: Path) -> None:
    rows = json.loads(finetune_json.read_text())
    if not rows:
        return
    genes = [r["gene"] for r in rows]
    all_clusters = sorted({c for r in rows for c in r.get("per_cluster_r2", {})})
    if not all_clusters:
        return
    mat = np.full((len(genes), len(all_clusters)), np.nan)
    for i, r in enumerate(rows):
        for j, cl in enumerate(all_clusters):
            mat[i, j] = r.get("per_cluster_r2", {}).get(cl, np.nan)
    fig, ax = plt.subplots(figsize=(max(6, len(all_clusters) * 0.55), max(3, len(genes) * 0.6)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_clusters)), all_clusters, rotation=45, ha="right")
    ax.set_yticks(range(len(genes)), genes)
    ax.set_title("Per-cluster R² (shared CNN + gene MLP, finetune half)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_performance(
    finetune_json: Path,
    pretrain_meta_paths: list[Path],
    out_path: Path,
) -> None:
    fin = json.loads(finetune_json.read_text())
    genes = [r["gene"] for r in fin]
    r2_cnn = [r["r2_cnn"] for r in fin]
    r2_lasso = [r["r2_lasso"] for r in fin]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(genes))
    w = 0.35
    ax.bar(x - w / 2, r2_lasso, w, label="Lasso (finetune half)", color="#6baed6")
    ax.bar(x + w / 2, r2_cnn, w, label="Shared CNN + gene MLP", color="#fd8d3c")
    ax.set_xticks(x, genes, rotation=30, ha="right")
    ax.set_ylabel("In-sample R² (finetune half)")
    ax.set_title("Per-gene performance on held-out tissue half")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if pretrain_meta_paths:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for mp in pretrain_meta_paths:
            if not mp.exists():
                continue
            meta = json.loads(mp.read_text())
            hist = meta.get("history") or []
            if not hist and "checkpoint" in meta:
                ckpt = torch.load(meta["checkpoint"], map_location="cpu")
                hist = ckpt.get("history", [])
            if not hist:
                continue
            epochs = [h["epoch"] for h in hist]
            mse = [h["mse"] for h in hist]
            ax2.plot(epochs, mse, label=meta.get("variant", mp.stem))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE (multi-gene pretrain)")
        ax2.set_title("CNN pretrain curves (train half)")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(out_path.with_name("pretrain_curves.png"), dpi=150)
        plt.close(fig2)


def plot_cnn_interpretation(
    cache: SpatialCache,
    encoder_ckpt: Path,
    out_dir: Path,
    n_examples: int = 6,
    device: str = "cpu",
) -> None:
    dev = torch.device(device)
    encoder = load_encoder(encoder_ckpt, dev)
    maps = cluster_maps_for_cells(cache)
    sf = cache.spatial_features

    rng = np.random.default_rng(1)
    idx = rng.choice(len(cache.obs_names), size=min(n_examples, len(cache.obs_names)), replace=False)

    fig, axes = plt.subplots(n_examples, 4, figsize=(12, 2.2 * n_examples))
    if n_examples == 1:
        axes = np.array([axes])

    for row, i in enumerate(idx):
        sm = torch.from_numpy(maps[i : i + 1]).to(dev)
        sfi = torch.from_numpy(sf[i : i + 1]).to(dev)
        spatial = sm[0, 0].cpu().numpy()
        cam = grad_cam(encoder, sm, sfi)
        ig = integrated_gradients(encoder, sm, sfi)

        from scipy.ndimage import zoom

        h, w = spatial.shape
        cam_up = zoom(cam, (h / cam.shape[0], w / cam.shape[1]), order=1)
        ig_up = zoom(ig, (h / ig.shape[0], w / ig.shape[1]), order=1)

        ax0, ax1, ax2, ax3 = axes[row]
        ax0.imshow(spatial, cmap="viridis")
        ax0.set_title(f"Inv-dist map\n{cache.cluster_labels[cache.clusters[i]]}")
        ax0.axis("off")
        ax1.imshow(cam_up, cmap="hot")
        ax1.set_title("Grad-CAM (conv3)")
        ax1.axis("off")
        ax2.imshow(ig_up, cmap="hot")
        ax2.set_title("Integrated Gradients")
        ax2.axis("off")
        ax3.imshow(spatial, cmap="gray", alpha=0.5)
        ax3.imshow(cam_up, cmap="hot", alpha=0.55)
        ax3.set_title("Grad-CAM overlay")
        ax3.axis("off")

    fig.suptitle("Shared tissue CNN interpretation (finetune half cells)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "gradcam_examples.png", dpi=150)
    plt.close(fig)

    # Mean activation map across cells
    feats = encode_all(encoder, cache, dev, batch_size=256)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    mean_map = maps[:, 0].mean(axis=0)
    ax3.imshow(mean_map, cmap="magma")
    ax3.set_title(f"Mean own-cluster spatial map\nencoder feat std={feats.std():.3f}")
    ax3.axis("off")
    fig3.tight_layout()
    fig3.savefig(out_dir / "mean_spatial_activation.png", dpi=150)
    plt.close(fig3)


def compare_variants(
    train_cache: Path,
    variant_ckpts: dict[str, Path],
    finetune_h5ad: Path,
    out_path: Path,
    spatial_dim: int = 16,
    device: str = "cpu",
) -> None:
    """Compare CNN variants by finetune-half reconstruction MSE after pretrain."""
    from pretrain_cnn import pretrain

    dev = torch.device(device)
    cache = SpatialCache.load(train_cache)
    scores: dict[str, float] = {}
    for name, ckpt in variant_ckpts.items():
        if not ckpt.exists():
            continue
        enc = load_encoder(ckpt, dev)
        fin_cache = build_spatial_cache(finetune_h5ad, spatial_dim=spatial_dim)
        feats = encode_all(enc, fin_cache, dev)
        # proxy transfer: feature variance + linear probe on first gene
        y = fin_cache.expr_log1p[:, 0]
        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=1.0)
        ridge.fit(feats, y)
        pred = ridge.predict(feats)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        scores[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if not scores:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    ax.bar(names, vals, color=["#74c476", "#9e9ac8"][: len(names)])
    ax.set_ylabel("Linear probe R² (1st HVG, finetune half)")
    ax.set_title("CNN variant transferability")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    write_json = out_path.with_suffix(".json")
    write_json.write_text(json.dumps(scores, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finetune-json", type=Path, default=Path("outputs/finetune/gene_performance.json"))
    p.add_argument("--pretrain-meta", type=Path, nargs="*", default=[Path("outputs/pretrain/pretrain_base_meta.json")])
    p.add_argument("--finetune-cache", type=Path, default=Path("outputs/finetune_cache.npz"))
    p.add_argument("--encoder", type=Path, default=Path("outputs/pretrain/tissue_encoder_deep.pt"))
    p.add_argument("--figures", type=Path, default=Path("figures"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    if args.finetune_json.exists():
        plot_performance(
            args.finetune_json,
            args.pretrain_meta,
            args.figures / "gene_performance_finetune_half.png",
        )
        plot_per_cluster_heatmap(args.finetune_json, args.figures / "per_cluster_r2_heatmap.png")

    if args.finetune_cache.exists() and args.encoder.exists():
        cache = SpatialCache.load(args.finetune_cache)
        plot_cnn_interpretation(cache, args.encoder, args.figures, device=args.device)

    halves_json = Path("outputs/half_comparison.json")
    if halves_json.exists():
        rows = json.loads(halves_json.read_text())
        if rows:
            fig, ax = plt.subplots(figsize=(7, 4))
            genes = [r["gene"] for r in rows]
            x = np.arange(len(genes))
            w = 0.35
            ax.bar(x - w / 2, [r["r2_train_half"] for r in rows], w, label="Train half (transfer)", color="#9ecae1")
            ax.bar(x + w / 2, [r["r2_finetune_half"] for r in rows], w, label="Finetune half (fit)", color="#fdae6b")
            ax.set_xticks(x, genes, rotation=30, ha="right")
            ax.set_ylabel("R² (gene MLP on frozen CNN)")
            ax.set_title("Train vs finetune half generalization")
            ax.legend()
            ax.axhline(0, color="k", lw=0.5)
            fig.tight_layout()
            fig.savefig(args.figures / "train_vs_finetune_half.png", dpi=150)
            plt.close(fig)

    print(f"figures -> {args.figures}")


if __name__ == "__main__":
    main()

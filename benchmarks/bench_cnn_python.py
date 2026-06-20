"""PyTorch CNN scaling bench, mirroring `src/bin/scaling_bench.rs`.

Builds the same synthetic training task (32x32 spatial maps + modulator features +
cluster one-hots + linear target) and trains SpaceTravLR's
`CellularNicheNetwork` (PyTorch) for `--epochs` epochs.

Designed for head-to-head comparison with the Rust `scaling_bench` binary at every
sample size used in the scaling study.

Output: one JSON line on stdout with timing per epoch and final MSE.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset


class CellularNicheNetworkLite(nn.Module):
    """PyTorch clone of SpaceTravLR's CellularNicheNetwork CNN (training-equivalent).

    Replicated locally so this benchmark does not depend on heavy SpaceTravLR
    imports (pyro, jscatter, enlighten) — only PyTorch.
    """

    def __init__(self, n_modulators: int, n_clusters: int, anchors: np.ndarray):
        super().__init__()
        self.dim = n_modulators + 1
        self.register_buffer("anchors", torch.from_numpy(anchors).float())
        self.conv_layers = nn.Sequential(
            weight_norm(nn.Conv2d(1, 16, kernel_size=3, padding="same")),
            nn.BatchNorm2d(16),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(2, 2),
            weight_norm(nn.Conv2d(16, 32, kernel_size=3, padding="same")),
            nn.BatchNorm2d(32),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(2, 2),
            weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding="same")),
            nn.BatchNorm2d(64),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.spatial_features_mlp = nn.Sequential(
            nn.Linear(n_clusters, 16),
            nn.PReLU(init=0.1),
            nn.Linear(16, 32),
            nn.PReLU(init=0.1),
            nn.Linear(32, 64),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.PReLU(init=0.1),
            nn.Linear(64, self.dim),
        )
        self.output_activation = nn.Sigmoid()

    def get_betas(self, spatial_maps, spatial_features):
        out = self.conv_layers(spatial_maps)
        sp_out = self.spatial_features_mlp(spatial_features)
        out = out + sp_out
        betas = self.mlp(out)
        betas = self.output_activation(betas)
        return betas * self.anchors

    def forward(self, spatial_maps, inputs_x, spatial_features):
        betas = self.get_betas(spatial_maps, spatial_features)
        return (
            torch.matmul(inputs_x.unsqueeze(1), betas[:, 1:].unsqueeze(2)).squeeze(1).squeeze(1)
            + betas[:, 0]
        )


def synth_inputs(n, spatial_dim, n_modulators, n_clusters, dropout, seed):
    rng = np.random.default_rng(seed)
    sm = rng.uniform(0.0, 1.0, size=(n, 1, spatial_dim, spatial_dim)).astype(np.float32)
    if dropout > 0:
        keep = rng.uniform(0.0, 1.0, size=sm.shape).astype(np.float32)
        sm = np.where(keep < dropout, 0.0, sm).astype(np.float32)
    x = rng.uniform(0.0, 1.0, size=(n, n_modulators)).astype(np.float32)
    sf = np.zeros((n, n_clusters), dtype=np.float32)
    sf[np.arange(n), np.arange(n) % n_clusters] = 1.0
    coefs = (0.05 + 0.02 * np.arange(min(n_modulators, 6))).astype(np.float32)
    y = 0.1 + x[:, : len(coefs)] @ coefs
    y = y + 0.02 * sm.reshape(n, -1).mean(axis=1)
    y = (y + rng.uniform(0.0, 0.05, size=n)).astype(np.float32)
    return sm, x, sf, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=700)
    p.add_argument("--spatial-dim", type=int, default=32)
    p.add_argument("--n-modulators", type=int, default=16)
    p.add_argument("--n-clusters", type=int, default=12)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    sm, x, sf, y = synth_inputs(
        args.n, args.spatial_dim, args.n_modulators, args.n_clusters, args.dropout, args.seed
    )
    sm_t = torch.from_numpy(sm)
    x_t = torch.from_numpy(x)
    sf_t = torch.from_numpy(sf)
    y_t = torch.from_numpy(y)

    ds = TensorDataset(sm_t, x_t, sf_t, y_t)
    loader = DataLoader(
        ds,
        batch_size=args.minibatch,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    anchors = np.concatenate([[0.5], 0.1 * np.ones(args.n_modulators)]).astype(np.float32)
    model = CellularNicheNetworkLite(args.n_modulators, args.n_clusters, anchors).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    epoch_seconds = []
    mse_history = []
    diverged = False

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total = time.time()
    for ep in range(args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        model.train()
        run = 0.0
        nb = 0
        for sm_b, x_b, sf_b, y_b in loader:
            sm_b = sm_b.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            sf_b = sf_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(sm_b, x_b, sf_b)
            loss = loss_fn(pred, y_b)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            opt.step()
            run += float(loss.item())
            nb += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds.append(time.time() - t0)
        mse_history.append(run / max(1, nb))
        if diverged:
            break

    total_seconds = time.time() - t_total
    out = {
        "impl_": "python-pytorch",
        "backend": ("cuda" if device.type == "cuda" else "cpu"),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_cells": args.n,
        "spatial_dim": args.spatial_dim,
        "n_modulators": args.n_modulators,
        "n_clusters": args.n_clusters,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "total_seconds": total_seconds,
        "epoch_seconds": epoch_seconds,
        "final_mse": mse_history[-1] if mse_history else float("nan"),
        "mse_history": mse_history,
        "diverged": diverged,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

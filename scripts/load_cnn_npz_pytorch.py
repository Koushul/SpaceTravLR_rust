#!/usr/bin/env python3
"""
Load SpaceTravLR CNN weights exported from Rust (.npz) into PyTorch.

Example:
  python scripts/load_cnn_npz_pytorch.py \
    --npz /tmp/slideseq_brain/models/GENE_cnn_weights.npz \
    --cluster 0
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT_IDENTITY = 0
_ACT_SIGMOID = 1
_ACT_TANH = 2
_ACT_SIGMOID_X2 = 3


class CellularNicheNetworkTorch(nn.Module):
    def __init__(
        self, n_modulators: int, n_clusters: int, output_activation: int = _ACT_SIGMOID
    ) -> None:
        super().__init__()
        dim = n_modulators + 1
        self.output_activation = int(output_activation)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)

        self.spatial_l1 = nn.Linear(n_clusters, 16)
        self.spatial_l2 = nn.Linear(16, 32)
        self.spatial_l3 = nn.Linear(32, 64)

        self.head_l1 = nn.Linear(64, 64)
        self.head_l2 = nn.Linear(64, dim)

        self.register_buffer("anchors", torch.ones(dim, dtype=torch.float32))

    def get_betas(self, spatial_maps: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        x = F.prelu(self.bn1(self.conv1(spatial_maps)), torch.tensor(0.1, device=spatial_maps.device))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.prelu(self.bn2(self.conv2(x)), torch.tensor(0.1, device=spatial_maps.device))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.prelu(self.bn3(self.conv3(x)), torch.tensor(0.1, device=spatial_maps.device))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).reshape(spatial_maps.shape[0], 64)

        s = F.prelu(self.spatial_l1(spatial_features), torch.tensor(0.1, device=spatial_maps.device))
        s = F.prelu(self.spatial_l2(s), torch.tensor(0.1, device=spatial_maps.device))
        s = self.spatial_l3(s)

        out = x + s
        out = F.prelu(self.head_l1(out), torch.tensor(0.1, device=spatial_maps.device))
        out = self.head_l2(out)
        if self.output_activation == _ACT_IDENTITY:
            betas = out
        elif self.output_activation == _ACT_TANH:
            betas = torch.tanh(out)
        elif self.output_activation == _ACT_SIGMOID_X2:
            betas = torch.sigmoid(out) * 2.0
        else:
            betas = torch.sigmoid(out)
        return betas * self.anchors.unsqueeze(0)

    def forward(
        self, spatial_maps: torch.Tensor, inputs_x: torch.Tensor, spatial_features: torch.Tensor
    ) -> torch.Tensor:
        betas = self.get_betas(spatial_maps, spatial_features)
        beta0 = betas[:, 0]
        beta_rest = betas[:, 1:]
        return beta0 + (beta_rest * inputs_x).sum(dim=1)


def _t(npz: Dict[str, np.ndarray], key: str) -> torch.Tensor:
    if key not in npz:
        raise KeyError(f"Missing key in npz: {key}")
    return torch.from_numpy(np.asarray(npz[key], dtype=np.float32))


def load_cluster_model(npz_path: str, cluster_id: int) -> CellularNicheNetworkTorch:
    data = np.load(npz_path, allow_pickle=False)
    p = f"c{cluster_id:04d}_"

    if "meta_cnn_output_activation" in data.files:
        out_act = int(np.asarray(data["meta_cnn_output_activation"]).reshape(-1)[0])
    else:
        out_act = _ACT_SIGMOID

    anchors = _t(data, p + "anchors")
    n_modulators = int(anchors.numel() - 1)

    spatial_l1_w = _t(data, p + "spatial_l1_weight")
    n_clusters = int(spatial_l1_w.shape[0])

    model = CellularNicheNetworkTorch(n_modulators, n_clusters, out_act)

    with torch.no_grad():
        model.conv1.weight.copy_(_t(data, p + "conv1_weight"))
        model.conv1.bias.copy_(_t(data, p + "conv1_bias"))
        model.conv2.weight.copy_(_t(data, p + "conv2_weight"))
        model.conv2.bias.copy_(_t(data, p + "conv2_bias"))
        model.conv3.weight.copy_(_t(data, p + "conv3_weight"))
        model.conv3.bias.copy_(_t(data, p + "conv3_bias"))

        model.bn1.weight.copy_(_t(data, p + "bn1_gamma"))
        model.bn1.bias.copy_(_t(data, p + "bn1_beta"))
        model.bn1.running_mean.copy_(_t(data, p + "bn1_running_mean"))
        model.bn1.running_var.copy_(_t(data, p + "bn1_running_var"))

        model.bn2.weight.copy_(_t(data, p + "bn2_gamma"))
        model.bn2.bias.copy_(_t(data, p + "bn2_beta"))
        model.bn2.running_mean.copy_(_t(data, p + "bn2_running_mean"))
        model.bn2.running_var.copy_(_t(data, p + "bn2_running_var"))

        model.bn3.weight.copy_(_t(data, p + "bn3_gamma"))
        model.bn3.bias.copy_(_t(data, p + "bn3_beta"))
        model.bn3.running_mean.copy_(_t(data, p + "bn3_running_mean"))
        model.bn3.running_var.copy_(_t(data, p + "bn3_running_var"))

        # Burn linear weights are [in, out]. PyTorch expects [out, in].
        model.spatial_l1.weight.copy_(_t(data, p + "spatial_l1_weight").T)
        model.spatial_l1.bias.copy_(_t(data, p + "spatial_l1_bias"))
        model.spatial_l2.weight.copy_(_t(data, p + "spatial_l2_weight").T)
        model.spatial_l2.bias.copy_(_t(data, p + "spatial_l2_bias"))
        model.spatial_l3.weight.copy_(_t(data, p + "spatial_l3_weight").T)
        model.spatial_l3.bias.copy_(_t(data, p + "spatial_l3_bias"))

        model.head_l1.weight.copy_(_t(data, p + "head_l1_weight").T)
        model.head_l1.bias.copy_(_t(data, p + "head_l1_bias"))
        model.head_l2.weight.copy_(_t(data, p + "head_l2_weight").T)
        model.head_l2.bias.copy_(_t(data, p + "head_l2_bias"))

        model.anchors.copy_(anchors)

    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Load exported SpaceTravLR CNN weights into PyTorch")
    ap.add_argument("--npz", required=True, help="Path to *_cnn_weights.npz")
    ap.add_argument("--cluster", type=int, default=0, help="Cluster ID to load (default: 0)")
    args = ap.parse_args()

    model = load_cluster_model(args.npz, args.cluster)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded cluster {args.cluster} model from {args.npz}")
    print(f"n_params={n_params}, anchors={model.anchors.shape[0]}")


if __name__ == "__main__":
    main()


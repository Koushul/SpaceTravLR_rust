#!/usr/bin/env python3
"""
Load SpaceTravLR Rust-exported CNN weights (.npz) into PyTorch and run inference.

The Rust binary writes one file per gene under output_dir/CNN_weights/<GENE>_cnn_weights.npz
when [model_export].save_cnn_weights is true or when using --save-cnn-weights.

Layout (per cluster id C, prefix c0000_, c0001_, ...):
  cluster_ids, meta_spatial_dim, meta_n_clusters, meta_n_modulators
  cNNNN_conv[123]_weight / _bias, cNNNN_bn[123]_{gamma,beta,running_mean,running_var}
  cNNNN_spatial_l{123}_{weight,bias}, cNNNN_head_l{12}_{weight,bias}, cNNNN_anchors
  cNNNN_{conv1,conv2,conv3,spatial_l1,spatial_l2,head_l1}_prelu_alpha  (length-1 f32, learnable)

Forward matches Rust `CellularNicheNetwork` (src/model.rs): 3x3 same-pad convs, BN, PReLU,
maxpool 2x2, adaptive avg pool 1x1, spatial MLP on neighbor-count features, residual add, head MLP,
then output activation (identity / sigmoid / tanh / sigmoid×2; see meta_cnn_output_activation in .npz) * anchors.
Each PReLU is a learnable, single-parameter (`num_parameters=1`) module initialised at 0.1, matching
Python `nn.PReLU(init=0.1)`. Older Rust binaries did not store the slopes; if the `*_prelu_alpha`
keys are missing the loader falls back to the 0.1 init value.

meta_cnn_output_activation: uint32 0 = identity, 1 = sigmoid (default if key missing), 2 = tanh, 3 = 2·sigmoid → (0,2).

Dependencies: numpy, torch
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PRELU_INIT = 0.1
_BN_EPS = 1e-5
_ACT_IDENTITY = 0
_ACT_SIGMOID = 1
_ACT_TANH = 2
_ACT_SIGMOID_X2 = 3
_CNN_SPP_FLAT = 64 * (1 + 4 + 16)


def _prelu(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return F.prelu(x, alpha.to(device=x.device, dtype=x.dtype))


def cluster_prefixes(npz: np.lib.npyio.NpzFile) -> list[int]:
    found: set[int] = set()
    for k in npz.files:
        m = re.match(r"c(\d+)_conv1_weight$", k)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def _meta(npz: np.lib.npyio.NpzFile) -> Tuple[int, int, int, int]:
    if "meta_spatial_dim" not in npz.files:
        raise ValueError(
            "This .npz has no meta_spatial_dim; re-export with a current SpaceTravLR_rust binary."
        )
    h = int(np.asarray(npz["meta_spatial_dim"]).reshape(-1)[0])
    nc = int(np.asarray(npz["meta_n_clusters"]).reshape(-1)[0])
    nm = int(np.asarray(npz["meta_n_modulators"]).reshape(-1)[0])
    if "meta_cnn_output_activation" in npz.files:
        act = int(np.asarray(npz["meta_cnn_output_activation"]).reshape(-1)[0])
    else:
        act = _ACT_SIGMOID
    return h, nc, nm, act


class RustCellularNicheCNN(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_modulators: int,
        spatial_dim: int,
        output_activation: int = _ACT_SIGMOID,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.n_clusters = n_clusters
        self.n_modulators = n_modulators
        self.output_activation = int(output_activation)
        ic = int(input_channels)
        self.conv1 = nn.Conv2d(ic, 16, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16, eps=_BN_EPS)
        self.prelu_conv1 = nn.PReLU(num_parameters=1, init=_PRELU_INIT)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32, eps=_BN_EPS)
        self.prelu_conv2 = nn.PReLU(num_parameters=1, init=_PRELU_INIT)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64, eps=_BN_EPS)
        self.prelu_conv3 = nn.PReLU(num_parameters=1, init=_PRELU_INIT)
        self.spatial_l1 = nn.Linear(n_clusters, 16)
        self.prelu_spatial_l1 = nn.PReLU(num_parameters=1, init=_PRELU_INIT)
        self.spatial_l2 = nn.Linear(16, 32)
        self.prelu_spatial_l2 = nn.PReLU(num_parameters=1, init=_PRELU_INIT)
        self.spatial_l3 = nn.Linear(32, 64)
        self.head_l1 = nn.Linear(64, 64)
        self.prelu_head_l1 = nn.PReLU(num_parameters=1, init=_PRELU_INIT)
        self.head_l2 = nn.Linear(64, n_modulators + 1)
        self.spp_proj = nn.Linear(_CNN_SPP_FLAT, 64)
        self.register_buffer("anchors", torch.zeros(n_modulators + 1))

    def load_from_npz(self, d: np.lib.npyio.NpzFile, prefix: str) -> None:
        def t32(name: str) -> torch.Tensor:
            return torch.from_numpy(np.asarray(d[name], dtype=np.float32))

        self.conv1.weight.data.copy_(t32(f"{prefix}conv1_weight"))
        if f"{prefix}conv1_bias" in d.files:
            self.conv1.bias.data.copy_(t32(f"{prefix}conv1_bias"))
        self.conv2.weight.data.copy_(t32(f"{prefix}conv2_weight"))
        self.conv2.bias.data.copy_(t32(f"{prefix}conv2_bias"))
        self.conv3.weight.data.copy_(t32(f"{prefix}conv3_weight"))
        self.conv3.bias.data.copy_(t32(f"{prefix}conv3_bias"))

        self.bn1.weight.data.copy_(t32(f"{prefix}bn1_gamma"))
        self.bn1.bias.data.copy_(t32(f"{prefix}bn1_beta"))
        self.bn1.running_mean.copy_(t32(f"{prefix}bn1_running_mean"))
        self.bn1.running_var.copy_(t32(f"{prefix}bn1_running_var"))
        self.bn2.weight.data.copy_(t32(f"{prefix}bn2_gamma"))
        self.bn2.bias.data.copy_(t32(f"{prefix}bn2_beta"))
        self.bn2.running_mean.copy_(t32(f"{prefix}bn2_running_mean"))
        self.bn2.running_var.copy_(t32(f"{prefix}bn2_running_var"))
        self.bn3.weight.data.copy_(t32(f"{prefix}bn3_gamma"))
        self.bn3.bias.data.copy_(t32(f"{prefix}bn3_beta"))
        self.bn3.running_mean.copy_(t32(f"{prefix}bn3_running_mean"))
        self.bn3.running_var.copy_(t32(f"{prefix}bn3_running_var"))

        self.spatial_l1.weight.data.copy_(t32(f"{prefix}spatial_l1_weight"))
        self.spatial_l1.bias.data.copy_(t32(f"{prefix}spatial_l1_bias"))
        self.spatial_l2.weight.data.copy_(t32(f"{prefix}spatial_l2_weight"))
        self.spatial_l2.bias.data.copy_(t32(f"{prefix}spatial_l2_bias"))
        self.spatial_l3.weight.data.copy_(t32(f"{prefix}spatial_l3_weight"))
        self.spatial_l3.bias.data.copy_(t32(f"{prefix}spatial_l3_bias"))

        self.head_l1.weight.data.copy_(t32(f"{prefix}head_l1_weight"))
        self.head_l1.bias.data.copy_(t32(f"{prefix}head_l1_bias"))
        self.head_l2.weight.data.copy_(t32(f"{prefix}head_l2_weight"))
        self.head_l2.bias.data.copy_(t32(f"{prefix}head_l2_bias"))

        if f"{prefix}spp_proj_weight" in d.files:
            w = np.asarray(d[f"{prefix}spp_proj_weight"], dtype=np.float32)
            self.spp_proj.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(w.T)))
            self.spp_proj.bias.data.copy_(t32(f"{prefix}spp_proj_bias"))

        self.anchors.copy_(t32(f"{prefix}anchors"))

        for module, key in (
            (self.prelu_conv1, "conv1_prelu_alpha"),
            (self.prelu_conv2, "conv2_prelu_alpha"),
            (self.prelu_conv3, "conv3_prelu_alpha"),
            (self.prelu_spatial_l1, "spatial_l1_prelu_alpha"),
            (self.prelu_spatial_l2, "spatial_l2_prelu_alpha"),
            (self.prelu_head_l1, "head_l1_prelu_alpha"),
        ):
            full = f"{prefix}{key}"
            if full in d.files:
                module.weight.data.copy_(t32(full))

    def get_betas(self, spatial_maps: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        x = spatial_maps
        x = self.conv1(x)
        x = self.bn1(x)
        x = _prelu(x, self.prelu_conv1.weight)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = _prelu(x, self.prelu_conv2.weight)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = _prelu(x, self.prelu_conv3.weight)
        x = F.max_pool2d(x, 2, stride=2)
        p1 = F.adaptive_avg_pool2d(x, (1, 1))
        p2 = F.adaptive_avg_pool2d(x, (2, 2))
        p3 = F.adaptive_avg_pool2d(x, (4, 4))
        b = x.shape[0]
        spp = torch.cat([p1.reshape(b, -1), p2.reshape(b, -1), p3.reshape(b, -1)], dim=1)
        x = self.spp_proj(spp)

        s = self.spatial_l1(spatial_features)
        s = _prelu(s, self.prelu_spatial_l1.weight)
        s = self.spatial_l2(s)
        s = _prelu(s, self.prelu_spatial_l2.weight)
        s = self.spatial_l3(s)

        out = x + s
        out = self.head_l1(out)
        out = _prelu(out, self.prelu_head_l1.weight)
        out = self.head_l2(out)
        if self.output_activation == _ACT_IDENTITY:
            pass
        elif self.output_activation == _ACT_TANH:
            out = torch.tanh(out)
        elif self.output_activation == _ACT_SIGMOID_X2:
            out = torch.sigmoid(out) * 2.0
        else:
            out = torch.sigmoid(out)
        return out * self.anchors.unsqueeze(0)

    def forward(
        self,
        spatial_maps: torch.Tensor,
        spatial_features: torch.Tensor,
        modulator_expression: torch.Tensor,
    ) -> torch.Tensor:
        betas = self.get_betas(spatial_maps, spatial_features)
        b0 = betas[:, 0:1]
        rest = betas[:, 1:]
        y = (rest * modulator_expression).sum(dim=1, keepdim=True) + b0
        return y.squeeze(1)


@dataclass
class LoadedRustGeneCNNs:
    spatial_dim: int
    n_clusters: int
    n_modulators: int
    output_activation: int
    models: Dict[int, RustCellularNicheCNN]

    def eval(self) -> None:
        for m in self.models.values():
            m.eval()


def load_gene_npz(path: str, device: str | torch.device | None = None) -> LoadedRustGeneCNNs:
    device = device or "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    data = np.load(path, allow_pickle=False)
    try:
        ids = cluster_prefixes(data)
        if not ids:
            raise ValueError(f"no c*_conv1_weight keys in {path}")

        spatial_dim, n_clusters, n_modulators, out_act = _meta(data)

        pfx0 = f"c{ids[0]:04d}_"
        _cw = np.asarray(data[f"{pfx0}conv1_weight"])
        inch = int(_cw.shape[1]) if _cw.ndim == 4 else 1

        models: Dict[int, RustCellularNicheCNN] = {}
        for cid in ids:
            prefix = f"c{cid:04d}_"
            m = RustCellularNicheCNN(
                n_clusters, n_modulators, spatial_dim, out_act, input_channels=inch
            )
            m.load_from_npz(data, prefix)
            m.to(device)
            m.eval()
            models[cid] = m
        return LoadedRustGeneCNNs(
            spatial_dim, n_clusters, n_modulators, out_act, models
        )
    finally:
        data.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Load Rust CNN .npz and run a dummy forward pass.")
    ap.add_argument("npz", type=str, help="Path to <GENE>_cnn_weights.npz")
    ap.add_argument("--cluster", type=int, default=None, help="Cluster id (default: first in file)")
    ap.add_argument("--batch", type=int, default=2, help="Batch size for demo tensors")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    bundle = load_gene_npz(args.npz, device=args.device)
    cid = args.cluster if args.cluster is not None else min(bundle.models.keys())
    if cid not in bundle.models:
        raise SystemExit(f"cluster {cid} not in {list(bundle.models.keys())}")

    m = bundle.models[cid]
    b = args.batch
    h = m.spatial_dim
    ic = m.conv1.in_channels
    sm = torch.randn(b, ic, h, h, device=args.device, dtype=torch.float32)
    sf = torch.randn(b, m.n_clusters, device=args.device, dtype=torch.float32)
    x = torch.randn(b, m.n_modulators, device=args.device, dtype=torch.float32)

    with torch.no_grad():
        y = m(sm, sf, x)
    print(
        f"cluster={cid} spatial_dim={bundle.spatial_dim} n_clusters={bundle.n_clusters} "
        f"n_modulators={bundle.n_modulators} output_activation={bundle.output_activation}"
    )
    print(f"demo output shape {tuple(y.shape)} sample: {y[: min(3, b)].cpu().numpy()}")


if __name__ == "__main__":
    main()

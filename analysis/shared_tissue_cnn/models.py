"""Shared tissue CNN + per-gene MLP (mirrors SpaceTravLR VisionEncoder / HeadMLP split)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

PRELU_INIT = 0.1
CNN_SPP_FLAT_DIM = 64 * (1 + 4 + 16)


def spatial_pyramid_pool(x: torch.Tensor) -> torch.Tensor:
    """Adaptive avg pool pyramid (1x1, 2x2, 4x4) -> concat -> 64*(1+4+16)."""
    p1 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    p2 = F.adaptive_avg_pool2d(x, (2, 2)).flatten(1)
    p3 = F.adaptive_avg_pool2d(x, (4, 4)).flatten(1)
    return torch.cat([p1, p2, p3], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3)
        self.prelu = nn.PReLU(num_parameters=1, init=PRELU_INIT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return F.max_pool2d(x, kernel_size=2, stride=2)


class SpatialMLP(nn.Module):
    def __init__(self, n_clusters: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_clusters, 16),
            nn.PReLU(num_parameters=1, init=PRELU_INIT),
            nn.Linear(16, 32),
            nn.PReLU(num_parameters=1, init=PRELU_INIT),
            nn.Linear(32, out_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        return self.net(spatial_features)


class TissueVisionEncoder(nn.Module):
    """Shared CNN per tissue: spatial map -> 64-d embedding."""

    def __init__(
        self,
        n_clusters: int,
        in_channels: int = 1,
        feature_dim: int = 64,
        variant: Literal["base", "deep", "wide"] = "base",
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.variant = variant

        if variant == "wide":
            ch1, ch2, ch3, ch4 = 32, 64, 128, 128
        else:
            ch1, ch2, ch3, ch4 = 16, 32, 64, 64

        self.conv1 = ConvBlock(in_channels, ch1)
        self.conv2 = ConvBlock(ch1, ch2)
        self.conv3 = ConvBlock(ch2, ch3)
        out_ch = ch3
        if variant in ("deep", "wide"):
            self.conv4 = ConvBlock(ch3, ch4)
            out_ch = ch4
        else:
            self.conv4 = None

        spp_in = out_ch * (1 + 4 + 16)
        self.spp_proj = nn.Linear(spp_in, feature_dim)
        self.spatial_mlp = SpatialMLP(n_clusters, feature_dim)

    def encode_maps(self, spatial_maps: torch.Tensor) -> torch.Tensor:
        x = self.conv1(spatial_maps)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.conv4 is not None:
            x = self.conv4(x)
        spp = spatial_pyramid_pool(x)
        return self.spp_proj(spp)

    def forward(
        self,
        spatial_maps: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        vision = self.encode_maps(spatial_maps)
        context = self.spatial_mlp(spatial_features)
        return vision + context


class GeneHeadMLP(nn.Module):
    """Per-gene head on frozen tissue features -> betas scaled by lasso anchors."""

    def __init__(self, feature_dim: int, n_betas: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.PReLU(num_parameters=1, init=PRELU_INIT),
            nn.Linear(64, n_betas),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def apply_output_activation(raw: torch.Tensor, mode: str = "sigmoidx2") -> torch.Tensor:
    if mode == "identity":
        return raw
    if mode == "sigmoid":
        return torch.sigmoid(raw)
    if mode == "tanh":
        return torch.tanh(raw)
    if mode == "sigmoidx2":
        return torch.sigmoid(raw) * 2.0
    raise ValueError(f"unknown activation {mode}")


def linear_readout_y(betas: torch.Tensor, inputs_x: torch.Tensor) -> torch.Tensor:
    beta0 = betas[:, :1]
    beta_rest = betas[:, 1:]
    return beta0.squeeze(1) + (beta_rest * inputs_x).sum(dim=1)


class PretrainHead(nn.Module):
    """Discarded after CNN pretraining; predicts multi-gene expression."""

    def __init__(self, feature_dim: int, n_genes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_genes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

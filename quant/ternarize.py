# GitHub/ternaryedge-sdk/quant/ternarize.py
"""
Ternary quantization (TWN-style) utilities and layers.

Implements layer-wise or channel-wise ternarization of weights to {-alpha, 0, +alpha}
with threshold Δ = t * E(|W|) (default t=0.7), and a straight-through estimator (STE)
for backprop.

References (conceptual):
- Ternary Weight Networks (Li & Liu, 2016)
- Balanced-ternary mapping: sign(W) ∈ {-1, 0, +1} with learned/estimated scale α
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TernaryConfig:
    t: float = 0.7  # threshold factor for Δ = t * mean(|W|)
    per_channel: bool = False  # set True for channel-wise α/Δ (Conv: out_channels)
    channel_dim: int = 0  # dimension along which channels are defined
    enable: bool = True  # master switch (disable -> pass-through weights)


def _compute_delta_alpha(
    w: torch.Tensor, cfg: TernaryConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute (Δ, α, mask) for ternarization.

    Δ = t * mean(|W|)  (layer-wise or channel-wise)
    mask = 1 where |W| > Δ else 0
    α = mean(|W| over mask), with safe denominator
    """
    abs_w = w.abs()

    if cfg.per_channel:
        # Mean over all dims except channel_dim
        reduce_dims = [d for d in range(w.dim()) if d != cfg.channel_dim]
        mean_abs = abs_w.mean(dim=reduce_dims, keepdim=True)
        delta = cfg.t * mean_abs
        mask = (abs_w > delta).to(w.dtype)
        num = (abs_w * mask).sum(dim=reduce_dims, keepdim=True)
        den = mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1.0)
        alpha = num / den
    else:
        mean_abs = abs_w.mean()
        delta = cfg.t * mean_abs
        mask = (abs_w > delta).to(w.dtype)
        num = (abs_w * mask).sum()
        den = mask.sum().clamp(min=1.0)
        alpha = num / den

    return delta, alpha, mask


def ternarize_weight(
    w: torch.Tensor, cfg: TernaryConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward ternarization mapping:
      W_t = α * sign(W) if |W| > Δ else 0
    Returns (W_q (with STE), α, Δ).
    """
    if not cfg.enable:
        # Pass-through (no quantization)
        return (
            w,
            torch.tensor(1.0, device=w.device, dtype=w.dtype),
            torch.tensor(0.0, device=w.device, dtype=w.dtype),
        )

    with torch.no_grad():
        delta, alpha, mask = _compute_delta_alpha(w, cfg)
        sign = torch.sign(w)
        # Map exact zeros in sign (if any) to 0 (already 0), fine.

        w_tern = alpha * sign * mask  # {-α, 0, +α}

    # Straight-Through Estimator (identity gradient wrt original weights)
    w_q = w_tern.detach() - w.detach() + w
    return w_q, alpha.detach(), delta.detach()


class TernaryQuantizer(nn.Module):
    """
    Module wrapper to ternarize a weight tensor on-the-fly during forward.
    Use inside custom layers or to wrap existing parameters.
    """

    def __init__(
        self,
        t: float = 0.7,
        per_channel: bool = False,
        channel_dim: int = 0,
        enable: bool = True,
    ):
        super().__init__()
        self.cfg = TernaryConfig(
            t=t, per_channel=per_channel, channel_dim=channel_dim, enable=enable
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w_q, _, _ = ternarize_weight(w, self.cfg)
        return w_q

    def set_enable(self, enable: bool) -> None:
        self.cfg.enable = enable


class TernaryLinear(nn.Linear):
    """
    Drop-in Linear layer with ternary weights (TWN-style).
    Bias remains full precision by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        t: float = 0.7,
        per_channel: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        # For Linear, per_channel=True means per-output-feature scaling (channel_dim=0 for [out, in])
        self.quant = TernaryQuantizer(
            t=t, per_channel=per_channel, channel_dim=0, enable=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, _, _ = ternarize_weight(self.weight, self.quant.cfg)
        return F.linear(x, w_q, self.bias)


class TernaryConv2d(nn.Conv2d):
    """
    Drop-in Conv2d layer with ternary weights (TWN-style).
    Bias remains full precision by default.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        t: float = 0.7,
        per_channel: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        # For Conv2d, per_channel=True means per-out-channel scaling (channel_dim=0 for [out, in, kH, kW])
        self.quant = TernaryQuantizer(
            t=t, per_channel=per_channel, channel_dim=0, enable=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, _, _ = ternarize_weight(self.weight, self.quant.cfg)
        return F.conv2d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def disable_quant(model: nn.Module) -> None:
    """
    Utility to disable ternary quantization for all supported submodules (evaluation / debugging).
    """
    for m in model.modules():
        if isinstance(m, (TernaryQuantizer, TernaryLinear, TernaryConv2d)):
            if hasattr(m, "quant"):  # TernaryLinear/Conv2d
                m.quant.set_enable(False)
            elif hasattr(m, "cfg"):  # TernaryQuantizer
                m.set_enable(False)


def enable_quant(model: nn.Module) -> None:
    """
    Utility to enable ternary quantization for all supported submodules (training/inference).
    """
    for m in model.modules():
        if isinstance(m, (TernaryQuantizer, TernaryLinear, TernaryConv2d)):
            if hasattr(m, "quant"):
                m.quant.set_enable(True)
            elif hasattr(m, "cfg"):
                m.set_enable(True)

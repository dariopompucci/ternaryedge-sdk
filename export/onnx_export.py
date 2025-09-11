# GitHub/ternaryedge-sdk/export/onnx_export.py
"""
Utilities to export trained ternary models to ONNX with plain FP ops.
We "bake in" ternary weights into vanilla Conv/Linear so runtimes don't
need custom kernels.

Usage:
  python -m export.onnx_export --ckpt ./models/mnist_ternary_cnn.pt --onnx ./models/mnist_ternary_cnn.onnx
"""

from __future__ import annotations
import argparse
import copy
import os
from typing import Tuple

import torch
import torch.nn as nn

from quant.ternarize import (
    TernaryConv2d,
    TernaryLinear,
    ternarize_weight,
    TernaryConfig,
)

# We reuse the SmallTernaryCNN defined in training.train to rebuild the model
try:
    from training.train import SmallTernaryCNN
except Exception:
    SmallTernaryCNN = None


def _convert_module(m: nn.Module) -> nn.Module:
    """
    Convert TernaryConv2d/TernaryLinear to vanilla nn.Conv2d/nn.Linear with
    pre-quantized weights. Other modules are deep-copied.
    """
    if isinstance(m, TernaryConv2d):
        # Grab config from the quantizer attached to the module
        cfg = m.quant.cfg if hasattr(m, "quant") else TernaryConfig()
        with torch.no_grad():
            w_q, _, _ = ternarize_weight(m.weight, cfg)
        conv = nn.Conv2d(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=(m.bias is not None),
        )
        conv.weight.data.copy_(w_q)
        if m.bias is not None:
            conv.bias.data.copy_(m.bias.data)
        return conv

    if isinstance(m, TernaryLinear):
        cfg = m.quant.cfg if hasattr(m, "quant") else TernaryConfig()
        with torch.no_grad():
            w_q, _, _ = ternarize_weight(m.weight, cfg)
        lin = nn.Linear(m.in_features, m.out_features, bias=(m.bias is not None))
        lin.weight.data.copy_(w_q)
        if m.bias is not None:
            lin.bias.data.copy_(m.bias.data)
        return lin

    # Generic container types: recursively convert children
    for name, child in list(m.named_children()):
        setattr(m, name, _convert_module(child))
    return m


def convert_to_exportable(model: nn.Module) -> nn.Module:
    """
    Recursively replace ternary layers with FP layers holding ternary weights.
    """
    model = copy.deepcopy(model).eval()
    model = _convert_module(model)
    return model


def load_model_from_ckpt(ckpt_path: str) -> Tuple[nn.Module, dict]:
    """
    Rebuild the training-time model and load weights.
    Requires SmallTernaryCNN to be importable from training.train.
    """
    if SmallTernaryCNN is None:
        raise RuntimeError("Could not import SmallTernaryCNN from training.train")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    quant = True if not cfg.get("no_quant", False) else False
    t = cfg.get("t", 0.7)
    per_channel = cfg.get("per_channel", True)

    model = SmallTernaryCNN(quant=quant, t=t, per_channel=per_channel)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def export_mnist_checkpoint_to_onnx(ckpt_path: str, onnx_path: str) -> None:
    model, cfg = load_model_from_ckpt(ckpt_path)
    model = convert_to_exportable(model).cpu().eval()

    dummy = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    dynamic_axes = {"input": {0: "N"}, "logits": {0: "N"}}

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX → {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export ternary MNIST model to ONNX")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to training checkpoint (*.pt)"
    )
    parser.add_argument("--onnx", type=str, required=True, help="Output ONNX path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.onnx) or ".", exist_ok=True)
    export_mnist_checkpoint_to_onnx(args.ckpt, args.onnx)


if __name__ == "__main__":
    main()

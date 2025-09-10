# GitHub/ternaryedge-sdk/backends/packing.py
"""
Reference packing utilities for ternary tensors {-1,0,+1} into a 2-bit stream.

Encoding (2-bit):
  00 -> 0
  01 -> +1
  10 -> -1
  11 -> reserved

Packs 4 trits per byte (little groups of 2 bits).
These are reference CPU utils; optimized kernels can come later (SIMD).
"""

from __future__ import annotations
import math
import torch


def encode_trits(x: torch.Tensor) -> torch.Tensor:
    """
    Map {-1,0,+1} -> {2,0,1} then to 2-bit values {10b,00b,01b}.
    We produce a uint8 tensor of the same shape with small integers 0..2.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dtype.is_floating_point:
        x = x.sign()  # ensure -1/0/+1 from floats
    vals = torch.empty_like(x, dtype=torch.uint8)
    vals[x == 0] = 0  # 00
    vals[x == 1] = 1  # 01
    vals[x == -1] = 2 # 10
    return vals


def pack_trits(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten x and pack into bytes, 4 trits per byte (2 bits each).
    Returns a uint8 tensor of length ceil(N/4).
    """
    vals = encode_trits(x).flatten().to(torch.uint8)
    n = vals.numel()
    out_len = (n + 3) // 4
    out = torch.zeros(out_len, dtype=torch.uint8)

    for i in range(n):
        byte_idx = i // 4
        shift = (i % 4) * 2
        out[byte_idx] |= (vals[i] & 0b11) << shift
    return out


def unpack_trits(packed: torch.Tensor, n_trits: int) -> torch.Tensor:
    """
    Inverse of pack_trits. Returns a 1D tensor of length n_trits with values in {-1,0,+1}.
    """
    if not torch.is_tensor(packed):
        packed = torch.as_tensor(packed, dtype=torch.uint8)
    out = torch.zeros(n_trits, dtype=torch.int8)

    for i in range(n_trits):
        byte_idx = i // 4
        shift = (i % 4) * 2
        code = (packed[byte_idx] >> shift) & 0b11
        if code == 0:
            out[i] = 0
        elif code == 1:
            out[i] = 1
        elif code == 2:
            out[i] = -1
        else:
            # reserved (treat as zero)
            out[i] = 0
    return out

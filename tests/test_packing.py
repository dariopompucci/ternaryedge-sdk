# GitHub/ternaryedge-sdk/tests/test_packing.py
import torch
from backends.packing import pack_trits, unpack_trits


def test_pack_unpack_roundtrip():
    x = torch.tensor([1, 0, -1, 1, -1, 0, 0, 1, -1, 1, 0, 0], dtype=torch.int8)
    packed = pack_trits(x)
    y = unpack_trits(packed, n_trits=x.numel()).to(torch.int8)
    assert torch.all(x == y), f"Roundtrip failed: {x} vs {y}"

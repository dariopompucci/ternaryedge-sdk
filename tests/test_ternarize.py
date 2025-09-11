# GitHub/ternaryedge-sdk/tests/test_ternarize.py
import torch
from quant.ternarize import TernaryConfig, ternarize_weight


def test_twn_properties_layerwise():
    w = torch.randn(256)
    cfg = TernaryConfig(t=0.7, per_channel=False)
    w_q, alpha, delta = ternarize_weight(w, cfg)

    # Values should be in {-alpha, 0, +alpha}
    uniq = torch.unique(w_q)
    # alpha can be scalar tensor; handle device/dtype
    a = float(alpha)
    allowed = set([0.0, +a, -a])
    assert all(
        float(v) in allowed for v in uniq
    ), f"Found values outside ternary set: {uniq}"


def test_twn_threshold_effect():
    w = torch.tensor([0.01, 0.1, 1.0, -2.0, 0.0, -0.05])
    cfg = TernaryConfig(t=0.7, per_channel=False)
    w_q, alpha, delta = ternarize_weight(w, cfg)
    # All entries with |w| <= delta must map to 0
    zero_mask = w.abs() <= float(delta)
    assert torch.all(w_q[zero_mask] == 0), "Values under threshold should be zeroed"

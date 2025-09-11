cat > quant/__init__.py <<'PY'
from .ternarize import (
    TernaryConfig as TernaryConfig,
    TernaryQuantizer as TernaryQuantizer,
    TernaryLinear as TernaryLinear,
    TernaryConv2d as TernaryConv2d,
    ternarize_weight as ternarize_weight,
    enable_quant as enable_quant,
    disable_quant as disable_quant,
)

__all__ = [
    "TernaryConfig",
    "TernaryQuantizer",
    "TernaryLinear",
    "TernaryConv2d",
    "ternarize_weight",
    "enable_quant",
    "disable_quant",
]
PY

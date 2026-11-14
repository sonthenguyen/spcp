"""spcp: Stable Principal Component Pursuit (M = L + S + N)."""
from ._stablepcp import StablePCP, trim_rank

__all__ = ["StablePCP", "trim_rank"]
__version__ = "0.1.1"

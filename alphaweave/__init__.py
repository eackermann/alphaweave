"""alphaweave - Weave data. Craft alpha."""

__version__ = "0.0.1"

# Public API: top-level conveniences
from .core.frame import Frame

# Namespace-style access (optional but recommended)
from . import core
from . import data
from . import engine
from . import strategy
from . import results
from . import utils
from . import indicators
from . import signals

__all__ = [
    "Frame",
    "core",
    "data",
    "engine",
    "strategy",
    "results",
    "utils",
    "indicators",
    "signals",
]

"""Compatibility imports for the model-specific modules.

New code should import implementations from ``kamp_models.models``.
"""

from .foundation_common import (
    FoundationComparisonResult,
    FoundationVariant,
    resolve_device,
)
from .models import run_chronos2, run_moirai2, run_timesfm3

__all__ = [
    "FoundationComparisonResult",
    "FoundationVariant",
    "resolve_device",
    "run_timesfm3",
    "run_chronos2",
    "run_moirai2",
]

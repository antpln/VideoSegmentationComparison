"""Predictor runner registry."""

from .sam2 import SAM2_RUNNERS
from .edgetam import EDGETAM_RUNNERS

__all__ = [
    "SAM2_RUNNERS",
    "EDGETAM_RUNNERS",
]

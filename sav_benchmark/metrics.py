"""Evaluation metrics for predicted mask sequences."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


def iou_binary(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def j_and_proxy_jf(pred_seq_masks: Iterable[Optional[np.ndarray]], gt_seq_masks: Iterable[Optional[np.ndarray]]) -> tuple[Optional[float], Optional[float]]:
    """Return the mean Jaccard score and a proxy J&F value for a sequence."""
    scores: List[float] = []
    for pred, gt in zip(pred_seq_masks, gt_seq_masks):
        if pred is None or gt is None:
            continue
        if pred.ndim == 3:
            # Collapse multi-channel predictions to a single binary mask.
            pred = np.any(pred.astype(bool), axis=0)
        scores.append(iou_binary(pred, gt))
    if not scores:
        return None, None
    j_score = float(np.mean(scores))
    return j_score, j_score

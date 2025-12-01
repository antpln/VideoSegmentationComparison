"""Evaluation metrics for predicted mask sequences."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import cv2  # type: ignore[import]


def iou_binary(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def j_and_proxy_jf(pred_seq_masks: Iterable[Optional[np.ndarray]], gt_seq_masks: Iterable[Optional[np.ndarray]]) -> Tuple[Optional[float], Optional[float]]:
    """Return the mean Jaccard score and boundary F-measure (DAVIS J&F)."""
    scores: List[float] = []
    f_scores: List[float] = []
    for pred, gt in zip(pred_seq_masks, gt_seq_masks):
        if gt is None:
            continue
        if pred is None:
            scores.append(0.0)
            f_scores.append(0.0)
            continue
        if pred.ndim == 3:
            # Collapse multi-channel predictions to a single binary mask.
            pred = np.any(pred.astype(bool), axis=0)
        pred = pred.astype(bool)
        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred.astype(np.uint8),
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        scores.append(iou_binary(pred, gt))
        f_scores.append(_boundary_f_measure(pred, gt))
    if not scores:
        return None, None
    j_score = float(np.mean(scores))
    f_score = float(np.mean(f_scores))
    return j_score, f_score


def _boundary_f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    pred_uint8 = pred.astype(np.uint8)
    gt_uint8 = gt.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = cv2.morphologyEx(pred_uint8, cv2.MORPH_GRADIENT, kernel)
    gt_boundary = cv2.morphologyEx(gt_uint8, cv2.MORPH_GRADIENT, kernel)

    diag = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
    dilation_radius = max(1, int(round(bound_th * diag)))
    dilation_kernel = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), np.uint8)

    pred_dil = cv2.dilate(pred_boundary, dilation_kernel)
    gt_dil = cv2.dilate(gt_boundary, dilation_kernel)

    pred_match = pred_boundary & gt_dil
    gt_match = gt_boundary & pred_dil

    pred_boundary_sum = pred_boundary.sum()
    gt_boundary_sum = gt_boundary.sum()

    if pred_boundary_sum == 0 and gt_boundary_sum == 0:
        return 1.0
    if pred_boundary_sum == 0 or gt_boundary_sum == 0:
        return 0.0

    precision = pred_match.sum() / float(pred_boundary_sum)
    recall = gt_match.sum() / float(gt_boundary_sum)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

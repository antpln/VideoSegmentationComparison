"""Prompt extraction utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def mask_centroid(binary_mask: np.ndarray) -> Optional[List[int]]:
    """Return the integer centroid of the foreground mask when present."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    cx = int(xs.mean())
    cy = int(ys.mean())
    return [cx, cy]


def extract_points_from_mask(binary_mask: np.ndarray, num_points: int = 5) -> Tuple[List[List[int]], List[int]]:
    """Sample up to ``num_points`` positive clicks that cover the mask."""
    try:
        from scipy import ndimage  # type: ignore[import]
    except ImportError:
        centroid = mask_centroid(binary_mask)
        if centroid is None:
            return [], []
        return [centroid], [1]

    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return [], []

    points: List[List[int]] = []
    labels: List[int] = []

    if num_features == 1:
        pos = np.where(binary_mask)
        if len(pos[0]) == 0:
            return [], []

        centroid = mask_centroid(binary_mask)
        if centroid:
            points.append(centroid)
            labels.append(1)

        available = len(pos[0])
        if available > 1:
            sample_count = min(num_points - len(points), available - 1)
            if sample_count > 0:
                # Uniformly sample additional positives for the same component.
                idxs = np.random.choice(available, sample_count, replace=False)
                for idx in idxs:
                    y, x = pos[0][idx], pos[1][idx]
                    points.append([int(x), int(y)])
                    labels.append(1)
    else:
        for component_idx in range(1, min(num_features + 1, num_points + 1)):
            component_mask = labeled_mask == component_idx
            centroid = mask_centroid(component_mask)
            if centroid:
                points.append(centroid)
                labels.append(1)

    return points, labels


def extract_bbox_from_mask(binary_mask: np.ndarray) -> Optional[List[int]]:
    """Compute the tight axis-aligned bounding box for the mask."""
    if binary_mask is None or not np.any(binary_mask):
        return None

    coords = np.column_stack(np.where(binary_mask))
    if coords.size == 0:
        return None

    y_coords, x_coords = coords[:, 0], coords[:, 1]
    x1, y1 = int(x_coords.min()), int(y_coords.min())
    x2, y2 = int(x_coords.max()), int(y_coords.max())
    return [x1, y1, x2, y2]

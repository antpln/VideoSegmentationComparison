"""Dataset discovery and mask loading helpers for the SA-V benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2  # type: ignore[import]
import numpy as np


def read_video_ids(split_dir: Path, split_name: str) -> List[str]:
    """Return video identifiers listed in ``{split_name}.txt`` or inferred from frames."""
    list_file = split_dir / f"{split_name}.txt"
    if list_file.exists():
        with open(list_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    # Fall back to scanning the JPEG directory when the manifest is missing.
    frame_root = split_dir / "JPEGImages_24fps"
    if frame_root.exists():
        return sorted(p.name for p in frame_root.iterdir() if p.is_dir())
    return []


def list_frames_24fps(split_dir: Path, video_id: str) -> List[Path]:
    """Return sorted RGB frame paths for a given video."""
    frame_dir = split_dir / "JPEGImages_24fps" / video_id
    return sorted(frame_dir.glob("*.jpg"))


def parse_idx(stem: str) -> int | None:
    """Parse a filename stem into an integer frame index when possible."""
    try:
        return int(stem)
    except ValueError:
        return None


def list_annotated_indices_6fps(split_dir: Path, video_id: str) -> Dict[str, List[int]]:
    """Return annotated frame indices for each object in a video."""
    anno_root = split_dir / "Annotations_6fps" / video_id
    out: Dict[str, List[int]] = {}
    if not anno_root.exists():
        return out

    for obj_dir in sorted(p for p in anno_root.iterdir() if p.is_dir()):
        indices: List[int] = []
        for mask_path in sorted(obj_dir.glob("*.png")):
            idx = parse_idx(mask_path.stem)
            if idx is not None:
                indices.append(idx)
        if indices:
            # Preserve the frame indices for each object separately.
            out[obj_dir.name] = indices
    return out


def load_mask_png(path: Path) -> np.ndarray | None:
    """Return a boolean mask loaded from a PNG file."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        # Convert RGBA/BGR masks down to a single channel for thresholding.
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask > 0


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

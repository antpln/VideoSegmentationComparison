"""Synthetic data generation for quick smoke tests."""

from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np

from .data_io import ensure_dir


def create_synthetic_test_data(split_dir: Path) -> None:
    """Generate a tiny three-frame dataset for smoke testing the pipeline."""
    split_dir = Path(split_dir)
    ensure_dir(split_dir / "JPEGImages_24fps" / "test_video")
    ensure_dir(split_dir / "Annotations_6fps" / "test_video" / "1")

    height, width = 320, 480
    for idx in range(3):
        # Compose a simple moving rectangle so propagation has non-trivial motion.
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        x1, y1 = 100 + idx * 20, 100 + idx * 10
        x2, y2 = x1 + 80, y1 + 60
        frame[y1:y2, x1:x2] = [255, 128, 64]
        frame_path = split_dir / "JPEGImages_24fps" / "test_video" / f"{idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)

    for idx in range(3):
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1 = 100 + idx * 20, 100 + idx * 10
        x2, y2 = x1 + 80, y1 + 60
        mask[y1:y2, x1:x2] = 255
        mask_path = split_dir / "Annotations_6fps" / "test_video" / "1" / f"{idx:05d}.png"
        cv2.imwrite(str(mask_path), mask)

    # The manifest mimics the layout of real SA-V splits.
    with open(split_dir / "sav_val.txt", "w", encoding="utf-8") as handle:
        handle.write("test_video\n")

"""Dataset discovery and mask loading helpers for the SA-V benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import shutil
import subprocess
import glob

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

    # Fall back to video directories (video_fps_24 or videos_fps_24)
    for vid_dir_name in ("video_fps_24", "videos_fps_24"):
        vid_root = split_dir / vid_dir_name
        if vid_root.exists():
            # Expect per-video directories or mp4 files; infer ids from folder names or filenames
            ids = []
            for p in sorted(vid_root.iterdir()):
                if p.is_dir():
                    ids.append(p.name)
                elif p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv"}:
                    ids.append(p.stem)
            if ids:
                return ids
    return []


def _extract_video_to_tmp_frames(video_path: Path, fps: int = 24) -> Path:
    """Extract an MP4 to a temporary directory of JPEG frames using ffmpeg.

    Returns the temporary directory Path (caller owns cleanup).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=f"sv_frames_{video_path.stem}_"))
    out_pattern = str(tmpdir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        out_pattern,
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("ffmpeg not found; please install ffmpeg or provide JPEGImages_24fps directories")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"ffmpeg failed extracting {video_path}: {e}")
    return tmpdir


def list_frames_24fps(split_dir: Path, video_id: str) -> List[Path]:
    """Return sorted RGB frame paths for a given video.

    Preference order:
    1. split_dir/JPEGImages_24fps/<video_id>/*.jpg
    2. split_dir/video_fps_24/<video_id>.mp4 (or first mp4 in dir)
    3. split_dir/videos_fps_24/<video_id>.mp4
    """
    # 1) JPEG directory (DAVIS-style)
    jpeg_dir = split_dir / "JPEGImages_24fps" / video_id
    if jpeg_dir.exists():
        return sorted(jpeg_dir.glob("*.jpg"))

    # 2) video_fps_24 directory with per-video mp4 or folder
    for vid_dir_name in ("video_fps_24", "videos_fps_24"):
        vid_root = split_dir / vid_dir_name
        if not vid_root.exists():
            continue
        # Prefer a subdirectory named after the video_id containing mp4s
        candidate_dir = vid_root / video_id
        if candidate_dir.exists() and candidate_dir.is_dir():
            mp4s = sorted(candidate_dir.glob("*.mp4"))
            if mp4s:
                tmp = _extract_video_to_tmp_frames(mp4s[0], fps=24)
                return sorted(tmp.glob("*.jpg"))
        # Otherwise look for a standalone mp4 named <video_id>.mp4 under vid_root
        single_mp4 = vid_root / f"{video_id}.mp4"
        if single_mp4.exists():
            tmp = _extract_video_to_tmp_frames(single_mp4, fps=24)
            return sorted(tmp.glob("*.jpg"))
        # Otherwise, if vid_root contains mp4s, pick the one matching the stem
        mp4s = sorted(vid_root.glob("*.mp4"))
        for mp4 in mp4s:
            if mp4.stem == video_id:
                tmp = _extract_video_to_tmp_frames(mp4, fps=24)
                return sorted(tmp.glob("*.jpg"))

    return []


def parse_idx(stem: str) -> Optional[int]:
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


def load_mask_png(path: Path) -> Optional[np.ndarray]:
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

"""Shared helpers for running suites of SAM/EdgeTAM experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from sav_benchmark.main import main as benchmark_main


DEFAULT_MODELS = "sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox"


@dataclass
class ExperimentConfig:
    name: str
    description: str
    overrides: Dict[str, object]


def parse_common_args(description: str, argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--split_dir", required=True, help="Path to the SA-V split root")
    parser.add_argument("--weights_dir", default=".", help="Directory containing weight files")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model list")
    parser.add_argument(
        "--out_root",
        default="./extended_experiments",
        help="Base directory where experiment outputs will be written",
    )
    parser.add_argument("--limit_videos", type=int, default=0, help="Limit number of videos (0 = all)")
    parser.add_argument(
        "--limit_objects",
        type=int,
        default=0,
        help="Limit number of objects per video (0 = all)",
    )
    parser.add_argument("--save_overlays", type=int, default=0, help="Write overlay videos when set to 1")
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision to request from the benchmark",
    )
    parser.add_argument("--max_clip_frames", type=int, default=0)
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN for lower memory use")
    parser.add_argument("--compile_models", action="store_true", help="Enable torch.compile in runners")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--compile_backend", default=None)
    parser.add_argument("--shuffle_videos", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def _base_cli(args: argparse.Namespace) -> List[str]:
    cli = [
        "--split_dir",
        args.split_dir,
        "--weights_dir",
        args.weights_dir,
        "--models",
        args.models,
        "--limit_videos",
        str(args.limit_videos),
        "--limit_objects",
        str(args.limit_objects),
        "--save_overlays",
        str(args.save_overlays),
        "--precision",
        args.precision,
        "--max_clip_frames",
        str(args.max_clip_frames),
    ]
    if args.disable_cudnn:
        cli.append("--disable_cudnn")
    if args.compile_models:
        cli.append("--compile_models")
    if args.compile_backend:
        cli.extend(["--compile_backend", args.compile_backend])
    if args.compile_mode:
        cli.extend(["--compile_mode", args.compile_mode])
    if args.shuffle_videos:
        cli.append("--shuffle_videos")
    if args.seed is not None:
        cli.extend(["--seed", str(args.seed)])
    return cli


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _apply_overrides(cli: List[str], overrides: Dict[str, object]) -> None:
    for key, value in overrides.items():
        if key == "out_dir":
            continue  # handled separately
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
        elif value is not None:
            cli.extend([flag, str(value)])


def run_experiment_suite(
    args: argparse.Namespace,
    experiments: Sequence[ExperimentConfig],
    summary_filename: str,
) -> Optional[Path]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    combined_rows: List[Dict[str, str]] = []
    combined_fields: Optional[List[str]] = None
    base_cli = _base_cli(args)

    for exp in experiments:
        exp_out_dir = out_root / exp.name
        cli = list(base_cli)
        cli.extend(["--out_dir", str(exp_out_dir)])
        _apply_overrides(cli, exp.overrides)
        print(f"\n[RUN] Experiment '{exp.name}' -> {exp_out_dir}")
        csv_path = Path(benchmark_main(cli))
        rows = _load_csv_rows(csv_path)
        if not rows:
            print(f"  [WARN] No rows produced for {exp.name}.")
            continue
        if combined_fields is None and rows:
            combined_fields = list(rows[0].keys()) + ["experiment_name", "experiment_description"]
        for row in rows:
            row["experiment_name"] = exp.name
            row["experiment_description"] = exp.description
            combined_rows.append(row)

    if combined_rows and combined_fields:
        combined_path = out_root / summary_filename
        with open(combined_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=combined_fields)
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"\nCombined CSV: {combined_path}")
        return combined_path

    print("\nNo combined CSV written (no experiment rows).")
    return None

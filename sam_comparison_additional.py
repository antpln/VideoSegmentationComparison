#!/usr/bin/env python3
"""Run preset experiment variants for the SAM vs EdgeTAM comparison."""

from __future__ import annotations

from sam_experiment_utils import (
    ExperimentConfig,
    parse_common_args,
    run_experiment_suite,
)


def main() -> None:
    args = parse_common_args("Run additional SAM/EdgeTAM experiments.")
    experiments = [
        ExperimentConfig(
            name="imgsz512_baseline",
            description="Matches the baseline run but forces 512x512 inference",
            overrides={"imgsz": 512},
        ),
        ExperimentConfig(
            name="imgsz768_stride6",
            description="768 resolution with annotations every 6th 24fps frame (4 Hz)",
            overrides={"imgsz": 768, "annotation_stride_frames": 6},
        ),
    ]
    run_experiment_suite(args, experiments, "additional_experiments_summary.csv")


if __name__ == "__main__":
    main()

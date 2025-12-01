#!/usr/bin/env python3
"""Run annotation-stride sweeps for SAM vs EdgeTAM experiments."""

from __future__ import annotations

from sam_experiment_utils import (
    ExperimentConfig,
    parse_common_args,
    run_experiment_suite,
)


def main() -> None:
    args = parse_common_args("Sweep annotation frequencies for 768 and 1024 resolutions.")
    experiments = [
        ExperimentConfig(
            name="imgsz768_stride8",
            description="768 resolution with annotations every 8th 24fps frame (3 Hz)",
            overrides={"imgsz": 768, "annotation_stride_frames": 8, "input_frame_stride": 8},
        ),
        ExperimentConfig(
            name="imgsz1024_stride4",
            description="1024 resolution with annotations every 4th 24fps frame (6 Hz)",
            overrides={"imgsz": 1024, "annotation_stride_frames": 4, "input_frame_stride": 4},
        ),
        ExperimentConfig(
            name="imgsz1024_stride6",
            description="1024 resolution with annotations every 6th 24fps frame (4 Hz)",
            overrides={"imgsz": 1024, "annotation_stride_frames": 6, "input_frame_stride": 6},
        ),
        ExperimentConfig(
            name="imgsz1024_stride8",
            description="1024 resolution with annotations every 8th 24fps frame (3 Hz)",
            overrides={"imgsz": 1024, "annotation_stride_frames": 8, "input_frame_stride": 8},
        ),
    ]
    run_experiment_suite(args, experiments, "stride_sweep_summary.csv")


if __name__ == "__main__":
    main()

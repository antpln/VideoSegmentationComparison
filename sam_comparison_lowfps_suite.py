#!/usr/bin/env python3
"""Run the low-FPS vs resolution experiments discussed for SAM/EdgeTAM."""

from __future__ import annotations

from sam_experiment_utils import (
    ExperimentConfig,
    parse_common_args,
    run_experiment_suite,
)


def main() -> None:
    args = parse_common_args(
        "Compare 24fps vs 12fps inputs and 768 vs 512 resolutions at 6fps annotations."
    )
    # SA-V annotations are spaced every 4th 24fps frame (~6 fps), so we set
    # annotation_stride_frames=4 to evaluate exactly those samples.
    experiments = [
        ExperimentConfig(
            name="imgsz768_input24_anno6",
            description="Baseline 768 inference at 24fps input, scoring every 6th frame",
            overrides={"imgsz": 768, "input_frame_stride": 1, "annotation_stride_frames": 4},
        ),
        ExperimentConfig(
            name="imgsz768_input12_anno6",
            description="768 inference fed 12fps input (stride 2) with 6fps annotations",
            overrides={"imgsz": 768, "input_frame_stride": 2, "annotation_stride_frames": 4},
        ),
        ExperimentConfig(
            name="imgsz512_input24_anno6",
            description="512 inference at 24fps input, 6fps annotations",
            overrides={"imgsz": 512, "input_frame_stride": 1, "annotation_stride_frames": 4},
        ),
    ]
    run_experiment_suite(args, experiments, "lowfps_suite_summary.csv")


if __name__ == "__main__":
    main()

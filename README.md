# Video Segmentation Comparison

Minimal tooling to compare SAM2 and EdgeTAM video segmentation pipelines. Supported prompts are limited to points and bounding boxes for both model families.

Runners are wired through a light abstraction in `sav_benchmark/runners/base.py`. Each
model family now subclasses `Model` (see `SAM2` and `EdgeTAM`) and registers its
supported prompts on import, so adding a new runner only requires overriding the
relevant prompt methods.


## Benchmark against a dataset

Assuming you have the SA-V split on disk and the required weights in `./sam2` and `./EdgeTAM/checkpoints`:

```bash
python sam_comparison.py \
  --split_dir /path/to/sav_val \
  --models sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox \
  --out_dir ./benchmark_outputs \
  --save_overlays 1 \
  --compile_models
```

The script writes a summary CSV to the chosen output directory and, when `--save_overlays 1`, saves per-object overlay videos. Add `--compile_backend` or `--compile_mode` to experiment with different `torch.compile` settings.

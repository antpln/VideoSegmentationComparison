# Video Segmentation Comparison

Quick runners to compare SAM2 and EdgeTAM on the SA-V dataset (points and bbox prompts).

## What you need
- SA-V split on disk (`JPEGImages_24fps` and `Annotations_6fps`)
- Weights: SAM2 (`sam2.1_*.pt`) and EdgeTAM (`EdgeTAM/checkpoints/edgetam.pt`)
- Python with PyTorch, OpenCV, and Ultralytics installed

## Run a fast sanity check
```bash
python sam_comparison.py --test_mode --models sam2_base_points
```
Writes synthetic results to `./test_outputs`.

## Benchmark on SA-V
```bash
python sam_comparison.py \
  --split_dir /path/to/sav_val \
  --models sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox \
  --out_dir ./benchmark_outputs \
  --save_overlays 1
```
Outputs `sav_benchmark_summary.csv` (and overlays when requested) in the chosen directory.

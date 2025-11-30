# Video Segmentation Comparison

Minimal tooling to compare SAM2 and EdgeTAM video segmentation pipelines. Supported prompts are limited to points and bounding boxes for both model families.

Runners are wired through a light abstraction in `sav_benchmark/runners/base.py`. Each
model family now subclasses `Model` (see `SAM2` and `EdgeTAM`) and registers its
supported prompts on import, so adding a new runner only requires overriding the
relevant prompt methods.

For edge deployments prefer TorchScript/JIT exports through Ultralytics’ built-in
exporters when running the SAM2 family. EdgeTAM remains a PyTorch-only tracker and
cannot be traced directly.

To produce a TorchScript artifact for SAM2 via Ultralytics:

```bash
python - <<'PY'
from ultralytics.models.sam import SAM2VideoPredictor
predictor = SAM2VideoPredictor(overrides={
    'model': 'sam2.1_b.pt',
    'task': 'track',
    'mode': 'predict',
    'imgsz': 1024,
    'device': 'cuda',
    'save': False,
})
predictor.export(format='torchscript', imgsz=1024, dynamic=True, out='sam2.1_b.torchscript.pt')
PY
```

The resulting `*.torchscript.pt` file can be deployed on edge hardware that
supports TorchScript runtimes.


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
# Video Segmentation Comparison

Quick runners to compare SAM2 and EdgeTAM on the SA-V dataset (points and bbox prompts).

## What you need
- SA-V split on disk (`JPEGImages_24fps` and `Annotations_6fps`)
- Weights: SAM2 (`sam2.1_*.pt`) and EdgeTAM (`EdgeTAM/checkpoints/edgetam.pt`)
- Python with PyTorch, OpenCV, and Ultralytics installed

## Branches

- **`main`**: Accuracy benchmarking (J, J&F metrics)
- **`nsight`**: GPU profiling with NVIDIA Nsight Systems (FPS, memory, kernel timing)

## Quick Start - Accuracy Benchmarking (main branch)

### Run a fast sanity check
```bash
python sam_comparison.py --test_mode --models sam2_base_points
```
Writes synthetic results to `./test_outputs`.

### Benchmark on SA-V
```bash
python sam_comparison.py \
  --split_dir /path/to/sav_val \
  --models sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox \
  --out_dir ./benchmark_outputs \
  --save_overlays 1
```
Outputs `sav_benchmark_summary.csv` (and overlays when requested) in the chosen directory.

## GPU Profiling (nsight branch)

For comprehensive GPU performance profiling with NVIDIA Nsight Systems:

```bash
# Switch to nsight branch
git checkout nsight

# Test setup
./run_test.sh

# Run full profiling experiment
./run_full_experiment.sh --split_dir /path/to/sav_val

# See EXPERIMENT_WORKFLOW.md for complete walkthrough
```

**Documentation:**
- **`EXPERIMENT_WORKFLOW.md`**: Complete step-by-step workflow for running experiments
- **`QUICKSTART.md`**: Quick setup guide
- **`PROFILING_SETUP.md`**: Overview of profiling infrastructure
- **`PROFILE_README.md`**: Detailed command-line reference

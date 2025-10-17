# Video Segmentation Comparison

Minimal tooling to compare SAM2 and EdgeTAM video segmentation pipelines. Supported prompts are limited to points and bounding boxes for both model families.

Runners are wired through a light abstraction in `sav_benchmark/runners/base.py`. Each
model family now subclasses `Model` (see `SAM2` and `EdgeTAM`) and registers its
supported prompts on import, so adding a new runner only requires overriding the
relevant prompt methods.

For edge deployments prefer TorchScript/JIT exports through Ultralytics' built-in
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


## Benchmark Scripts

### Standard Benchmark (Desktop/Server)

For systems with ample memory (desktop GPUs, servers):

```bash
python sam_comparison.py \
  --split_dir /path/to/sav_val \
  --models sam2_base_points,sam2_base_bbox,edgetam_points,edgetam_bbox \
  --out_dir ./benchmark_outputs \
  --save_overlays 1 \
  --compile_models
```

The script writes a summary CSV to the chosen output directory and, when `--save_overlays 1`, saves per-object overlay videos. Add `--compile_backend` or `--compile_mode` to experiment with different `torch.compile` settings.

### Jetson Orin Benchmark (Memory-Optimized)

**For Jetson Orin or other memory-constrained platforms**, use the optimized two-phase benchmark:

> **⚠️ Important**: Jetson uses unified memory architecture (GPU and CPU share RAM). 
> See [`UNIFIED_MEMORY_GUIDE.md`](UNIFIED_MEMORY_GUIDE.md) for critical details about memory management on Jetson.

```bash
# Quick test (recommended first run)
./run_jetson_benchmark.sh quick --split-dir /path/to/sav_val

# Standard benchmark with auto-tuning
./run_jetson_benchmark.sh standard --split-dir /path/to/sav_val --out-dir ./results

# Memory-safe mode (for 4GB Jetson or heavy models)
./run_jetson_benchmark.sh memory-safe --split-dir /path/to/sav_val
```

Or use the Python script directly:

```bash
python jetson_benchmark.py \
  --split_dir /path/to/sav_val \
  --models "sam2_base_points,edgetam_points" \
  --enable_cudnn_tuning \
  --out_dir ./jetson_results \
  --imgsz 1024
```

**Key differences from standard benchmark:**
- **Two-phase execution**: Setup phase (preprocessing) → Process exit → Inference phase (clean metrics)
- **Automatic memory tuning**: Tests if cuDNN benchmark=True is safe, falls back to benchmark=False if needed
- **Per-video batching**: Processes one video at a time with aggressive cleanup between
- **Preprocessed data**: Saves metadata to avoid repeated expensive parsing
- **Conservative setup**: Uses gentler memory allocator settings during model loading

See [`JETSON_BENCHMARK_GUIDE.md`](JETSON_BENCHMARK_GUIDE.md) for detailed documentation.

### Memory Monitoring

Monitor GPU and system memory usage during benchmarks:

```bash
# In a separate terminal
python monitor_memory.py --interval 0.5 --log memory_usage.csv
```

This helps identify memory bottlenecks and validate that the two-phase approach is working correctly.

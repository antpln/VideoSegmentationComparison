# Jetson Benchmark Quick Reference

## One-Command Benchmarks

### First Time / Quick Test
```bash
./run_jetson_benchmark.sh quick --split-dir /data/sav_val
```
- 1 video, 1 object
- Resolution: 384px
- Model: EdgeTAM points only
- Duration: ~2-5 minutes

### Standard Production Benchmark
```bash
./run_jetson_benchmark.sh standard --split-dir /data/sav_val --out-dir ./results
```
- All videos in dataset
- Resolution: 1024px
- Auto-tuned cuDNN settings
- All models in `--models` flag
- Duration: ~2-4 hours (depends on dataset size)

### Memory-Constrained Mode
```bash
./run_jetson_benchmark.sh memory-safe --split-dir /data/sav_val
```
- Resolution: 512px (lower memory)
- cuDNN benchmark: disabled
- Max frames: 300 (vs 600)
- Use when getting OOM errors

### Full Model Comparison
```bash
./run_jetson_benchmark.sh full --split-dir /data/sav_val --out-dir ./full_results
```
- All SAM2 variants: tiny, small, base
- Both EdgeTAM prompts: points, bbox
- Resolution: 1024px
- Duration: ~6-8 hours

## Direct Python Usage

### Automatic Two-Phase (Recommended)
```bash
python jetson_benchmark.py \
  --split_dir /data/sav_val \
  --models "sam2_base_points,edgetam_points" \
  --enable_cudnn_tuning \
  --imgsz 1024
```
Automatically runs setup → exit → inference

### Manual Phase Control

**Setup only** (preprocessing, tuning):
```bash
python jetson_benchmark.py \
  --split_dir /data/sav_val \
  --skip_inference \
  --enable_cudnn_tuning
```

**Inference only** (reuse preprocessed data):
```bash
python jetson_benchmark.py \
  --split_dir /data/sav_val \
  --skip_setup
```

## Memory Monitoring

### Real-Time Monitoring
```bash
# Terminal 1: Run monitor
python monitor_memory.py --interval 0.5 --log mem.csv

# Terminal 2: Run benchmark
./run_jetson_benchmark.sh standard --split-dir /data/sav_val
```

### Check Current GPU Usage
```bash
nvidia-smi
# or
watch -n 1 nvidia-smi
```

### Check System Memory
```bash
free -h
# or
cat /proc/meminfo | grep -E 'MemTotal|MemAvailable'
```

## Common Model Configurations

### EdgeTAM Only (Fastest)
```bash
--models "edgetam_points,edgetam_bbox"
```

### SAM2 Base (Balanced)
```bash
--models "sam2_base_points,sam2_base_bbox"
```

### EdgeTAM vs SAM2 Comparison
```bash
--models "edgetam_points,sam2_base_points,edgetam_bbox,sam2_base_bbox"
```

### All SAM2 Sizes
```bash
--models "sam2_tiny_points,sam2_small_points,sam2_base_points"
```

## Troubleshooting Commands

### OOM During Setup
```bash
# Use smaller workspace
python jetson_benchmark.py \
  --split_dir /data/sav_val \
  --workspace_mb 128 \
  --imgsz 512 \
  --skip_inference
```

### OOM During Inference
```bash
# Reduce frames in memory
python jetson_benchmark.py \
  --split_dir /data/sav_val \
  --max_frames_in_mem 200 \
  --imgsz 512 \
  --skip_setup
```

### Single Video Test
```bash
python jetson_benchmark.py \
  --split_dir /data/sav_val \
  --limit_videos 1 \
  --limit_objects 1
```

### Clean Restart (Delete Preprocessed Data)
```bash
rm -rf ./preprocessed_data
./run_jetson_benchmark.sh standard --split-dir /data/sav_val
```

### Check cuDNN Setting (From Logs)
```bash
grep "cuDNN" ./preprocessed_data/inference_config.json
```

## Output Files

### Main Results
```
./jetson_results/jetson_benchmark_summary.csv
```
CSV with FPS, memory, accuracy per model/video/object

### Preprocessed Data
```
./preprocessed_data/
├── metadata.json              # Video/object structure
└── inference_config.json      # Tuned settings (cuDNN, etc.)
```

### Memory Logs (If Using Monitor)
```
./memory_usage.csv
```
Timestamp, GPU usage, RAM usage

### Overlay Videos (If Enabled)
```
./jetson_results/video123__obj1__edgetam_points.mp4
./jetson_results/video123__obj1__sam2_base_points.mp4
```

## Performance Expectations

### Jetson Orin 8GB

| Model | Resolution | FPS | Peak GPU Memory |
|-------|-----------|-----|-----------------|
| EdgeTAM points | 1024 | 15-20 | ~3.8 GB |
| SAM2 base points | 1024 | 8-12 | ~4.5 GB |
| EdgeTAM bbox | 1024 | 15-20 | ~3.9 GB |
| SAM2 base bbox | 1024 | 8-12 | ~4.5 GB |

*With cuDNN benchmark=True when safe*

### Jetson Orin 4GB (AGX variant)

| Model | Resolution | FPS | Peak GPU Memory |
|-------|-----------|-----|-----------------|
| EdgeTAM points | 512 | 20-25 | ~1.8 GB |
| SAM2 tiny points | 512 | 12-18 | ~2.2 GB |

*Recommended: memory-safe mode, max_frames_in_mem=200*

## Environment Variables

### PyTorch Memory Allocator (Setup Phase)
```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True"
./run_jetson_benchmark.sh standard --split-dir /data/sav_val
```

### Disable cuDNN Determinism (Faster, Less Reproducible)
```bash
export CUDNN_DETERMINISTIC=0
python jetson_benchmark.py --split_dir /data/sav_val
```

### Force CPU (Testing Only)
```bash
export CUDA_VISIBLE_DEVICES=""
python jetson_benchmark.py --split_dir /data/sav_val
```

## Benchmark Workflow

### Recommended Workflow
1. **Quick test**: Verify setup works
   ```bash
   ./run_jetson_benchmark.sh quick --split-dir /data/sav_val
   ```

2. **Single video test**: Check memory/timing
   ```bash
   python jetson_benchmark.py \
     --split_dir /data/sav_val \
     --limit_videos 1 \
     --models "edgetam_points"
   ```

3. **Full benchmark**: All models
   ```bash
   ./run_jetson_benchmark.sh standard \
     --split-dir /data/sav_val \
     --out-dir ./results_$(date +%Y%m%d)
   ```

4. **Analyze results**:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('./results_YYYYMMDD/jetson_benchmark_summary.csv')
   print(df.groupby('model')[['fps', 'J', 'gpu_peak_alloc_MiB']].mean())
   "
   ```

### Iterative Development
```bash
# 1. Setup once
python jetson_benchmark.py --split_dir /data/sav_val --skip_inference

# 2. Test different models (reuse preprocessing)
python jetson_benchmark.py --split_dir /data/sav_val --skip_setup --models "edgetam_points"
python jetson_benchmark.py --split_dir /data/sav_val --skip_setup --models "sam2_base_points"

# 3. Re-preprocess if dataset changes
rm -rf ./preprocessed_data
python jetson_benchmark.py --split_dir /data/sav_val --skip_inference
```

## Compare with Standard Benchmark

### Standard (Desktop GPU)
```bash
python sam_comparison.py \
  --split_dir /data/sav_val \
  --models "edgetam_points,sam2_base_points" \
  --out_dir ./desktop_results
```

### Jetson Optimized
```bash
./run_jetson_benchmark.sh standard \
  --split-dir /data/sav_val \
  --out-dir ./jetson_results \
  --models "edgetam_points,sam2_base_points"
```

### Compare CSVs
```python
import pandas as pd

desktop = pd.read_csv("desktop_results/sav_benchmark_summary.csv")
jetson = pd.read_csv("jetson_results/jetson_benchmark_summary.csv")

print("Desktop FPS:", desktop.groupby('model')['fps'].mean())
print("Jetson FPS:", jetson.groupby('model')['fps'].mean())
```

## Getting Help

### View Full Documentation
```bash
cat JETSON_BENCHMARK_GUIDE.md
```

### View Script Help
```bash
./run_jetson_benchmark.sh help
python jetson_benchmark.py --help
```

### Check System Resources
```bash
# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv

# Jetpack version
cat /etc/nv_tegra_release

# Python packages
pip list | grep -E 'torch|ultralytics'
```

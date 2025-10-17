# Migration Guide: sam_comparison.py → jetson_benchmark.py

## Quick Start

If you're currently using:
```bash
python sam_comparison.py --split_dir /data/sav_val --models "edgetam_points,sam2_base_points"
```

Switch to:
```bash
./run_jetson_benchmark.sh standard --split-dir /data/sav_val --models "edgetam_points,sam2_base_points"
```

or directly:
```bash
python jetson_benchmark.py --split_dir /data/sav_val --models "edgetam_points,sam2_base_points" --enable_cudnn_tuning
```

## Command Translation

### Basic Benchmark
**Old:**
```bash
python sam_comparison.py \
  --split_dir /path/to/data \
  --models "sam2_base_points,edgetam_points" \
  --out_dir ./results
```

**New:**
```bash
python jetson_benchmark.py \
  --split_dir /path/to/data \
  --models "sam2_base_points,edgetam_points" \
  --out_dir ./results \
  --enable_cudnn_tuning
```

### With Compilation
**Old:**
```bash
python sam_comparison.py \
  --split_dir /path/to/data \
  --models "edgetam_points" \
  --compile_models \
  --compile_mode "reduce-overhead"
```

**New:**
```bash
# Compilation happens in warmup phase automatically
python jetson_benchmark.py \
  --split_dir /path/to/data \
  --models "edgetam_points"
```

Note: The new script handles compilation in the warmup phase. If you need custom compile settings, they're still available in the runners but applied during setup.

### Limited Scope
**Old:**
```bash
python sam_comparison.py \
  --split_dir /path/to/data \
  --limit_videos 5 \
  --limit_objects 2
```

**New:**
```bash
python jetson_benchmark.py \
  --split_dir /path/to/data \
  --limit_videos 5 \
  --limit_objects 2
```

Same syntax! ✓

## Argument Mapping

| Old Argument | New Argument | Notes |
|--------------|--------------|-------|
| `--split_dir` | `--split_dir` | Same |
| `--split_name` | `--split_name` | Same |
| `--models` | `--models` | Same |
| `--weights_dir` | `--weights_dir` | Same |
| `--imgsz` | `--imgsz` | Same |
| `--limit_videos` | `--limit_videos` | Same |
| `--limit_objects` | `--limit_objects` | Same |
| `--save_overlays` | `--save_overlays` | Same |
| `--out_dir` | `--out_dir` | Same |
| `--shuffle_videos` | `--shuffle_videos` | Same |
| `--seed` | `--seed` | Same |
| `--max_frames_in_mem` | `--max_frames_in_mem` | Same (NEW default: 600) |
| `--compile_models` | - | Handled in warmup |
| `--compile_backend` | - | Handled in warmup |
| `--compile_mode` | - | Handled in warmup |
| `--test_mode` | - | Use `quick` preset instead |
| - | `--enable_cudnn_tuning` | NEW: Auto-test benchmark setting |
| - | `--workspace_mb` | NEW: Control setup memory |
| - | `--preprocessed_dir` | NEW: Cache location |
| - | `--skip_setup` | NEW: Phase control |
| - | `--skip_inference` | NEW: Phase control |

## Output Differences

### CSV Filename
**Old:** `sav_benchmark_summary.csv`  
**New:** `jetson_benchmark_summary.csv`

### CSV Columns
**New columns added:**
- `cudnn_benchmark`: Whether cuDNN benchmark was enabled
- `setup_ms`: Model setup time (separate from inference)

All other columns remain the same.

### Directory Structure
**Old:**
```
./benchmark_outputs/
├── sav_benchmark_summary.csv
└── video1__obj1__model.mp4
```

**New:**
```
./jetson_results/
├── jetson_benchmark_summary.csv
└── video1__obj1__model.mp4

./preprocessed_data/
├── metadata.json
└── inference_config.json
```

## Behavioral Changes

### 1. Two-Phase Execution
**Old:** Single process runs everything  
**New:** Setup phase → Exit → Relaunch → Inference phase

**Impact:** 
- First run takes slightly longer (setup overhead)
- Subsequent runs are MUCH faster (reuse preprocessed data)
- Memory is completely cleaned between phases

### 2. Memory Management
**Old:** Aggressive settings throughout  
**New:** Conservative setup, optimal inference

**Impact:**
- Fewer crashes on memory-constrained devices
- Slightly different memory profiles
- Throughput may be higher (when safe cuDNN settings work)

### 3. cuDNN Benchmark
**Old:** Always enabled (can crash)  
**New:** Auto-tested in setup, validated setting used

**Impact:**
- More stable on Jetson
- May be slightly slower if benchmark=False needed
- But more consistent/predictable

## Compatibility

### Can I Still Use sam_comparison.py?
**Yes!** The old script still works and is useful for:
- Desktop/server GPUs with ample memory
- Quick one-off tests
- Compatibility with existing pipelines

Use `sam_comparison.py` when:
- You have >16GB GPU memory
- You don't care about setup overhead in metrics
- You want simplest possible script

Use `jetson_benchmark.py` when:
- Running on Jetson or memory-constrained devices
- You want clean inference metrics
- You're doing repeated experiments (preprocessing reuse)
- You need automatic memory tuning

### Can I Mix Them?
No direct compatibility, but you can run both and compare:

```bash
# Desktop benchmark
python sam_comparison.py --split_dir /data --out_dir ./desktop_results --models "edgetam_points"

# Jetson benchmark
python jetson_benchmark.py --split_dir /data --out_dir ./jetson_results --models "edgetam_points"

# Compare
python -c "
import pandas as pd
d = pd.read_csv('desktop_results/sav_benchmark_summary.csv')
j = pd.read_csv('jetson_results/jetson_benchmark_summary.csv')
print('Desktop FPS:', d['fps'].mean())
print('Jetson FPS:', j['fps'].mean())
"
```

## Common Migration Issues

### Issue 1: "Preprocessed data not found"
**Cause:** Running with `--skip_setup` without prior setup  
**Solution:**
```bash
# Run setup first
python jetson_benchmark.py --split_dir /data --skip_inference
# Then inference
python jetson_benchmark.py --split_dir /data --skip_setup
```

### Issue 2: "Still getting OOM"
**Cause:** Need more aggressive memory settings  
**Solution:**
```bash
# Use memory-safe preset
./run_jetson_benchmark.sh memory-safe --split-dir /data
# Or manually reduce settings
python jetson_benchmark.py --split_dir /data --imgsz 512 --max_frames_in_mem 200
```

### Issue 3: "Results don't match sam_comparison.py exactly"
**Cause:** Different cuDNN settings or timing methodology  
**Solution:** This is expected! Differences:
- Memory management differs (by design)
- Timing excludes setup (more accurate)
- cuDNN setting may differ (auto-tuned)

Accuracy (J, J&F) should match within 0.001.

### Issue 4: "Can't find compiled model"
**Cause:** Compilation flags removed  
**Solution:** Compilation happens in warmup phase automatically. To disable:
```python
# Edit jetson_benchmark.py, line ~320
result = runner(
    ...
    compile_model=False,  # Change this
)
```

### Issue 5: "Where did --test_mode go?"
**Cause:** Replaced with presets  
**Solution:**
```bash
# Old: python sam_comparison.py --test_mode
# New: ./run_jetson_benchmark.sh quick --split-dir /data
```

## Gradual Migration Strategy

### Phase 1: Side-by-side (Week 1)
```bash
# Keep using old script for production
python sam_comparison.py --split_dir /data --out_dir ./prod_results

# Test new script in parallel
./run_jetson_benchmark.sh quick --split-dir /data --out-dir ./test_results

# Compare outputs
diff <(head prod_results/sav_benchmark_summary.csv) \
     <(head test_results/jetson_benchmark_summary.csv)
```

### Phase 2: Partial adoption (Week 2-3)
```bash
# Use new script for memory-intensive runs
python jetson_benchmark.py --split_dir /data --models "sam2_large_points"

# Keep old script for simple tests
python sam_comparison.py --split_dir /data --limit_videos 1
```

### Phase 3: Full migration (Week 4+)
```bash
# Default to new script
alias benchmark='./run_jetson_benchmark.sh standard'

# Keep old script as fallback
alias benchmark_old='python sam_comparison.py'
```

## Rollback Plan

If you need to rollback:

```bash
# 1. Continue using sam_comparison.py
python sam_comparison.py --split_dir /data --models "..."

# 2. Delete new generated files (optional)
rm -rf ./preprocessed_data ./jetson_results

# 3. No code changes needed - both scripts coexist
```

The new scripts don't modify or interfere with the old ones.

## Recommended Workflow

### For Research/Development
```bash
# 1. Setup once (preprocessing + tuning)
python jetson_benchmark.py --split_dir /data --skip_inference

# 2. Experiment with different models (fast!)
python jetson_benchmark.py --split_dir /data --skip_setup --models "model1"
python jetson_benchmark.py --split_dir /data --skip_setup --models "model2"
python jetson_benchmark.py --split_dir /data --skip_setup --models "model3"

# 3. Re-preprocess only when dataset changes
rm -rf ./preprocessed_data
python jetson_benchmark.py --split_dir /data --skip_inference
```

### For Production Benchmarking
```bash
# Single command, automatic everything
./run_jetson_benchmark.sh standard --split-dir /data --out-dir ./results_$(date +%Y%m%d)

# Monitor in parallel (separate terminal)
python monitor_memory.py --log memory_$(date +%Y%m%d).csv
```

## Getting Help

### Documentation
- **Quick start**: `JETSON_QUICK_REFERENCE.md`
- **Full guide**: `JETSON_BENCHMARK_GUIDE.md`
- **Implementation details**: `JETSON_IMPLEMENTATION_SUMMARY.md`

### Commands
```bash
# Help text
python jetson_benchmark.py --help
./run_jetson_benchmark.sh help

# Test installation
./run_jetson_benchmark.sh quick --split-dir /data
```

### Common Questions

**Q: Will this work on non-Jetson platforms?**  
A: Yes! It works everywhere. Benefits are most noticeable on memory-constrained devices, but the two-phase approach and clean metrics are useful on any platform.

**Q: Do I need to change my dataset?**  
A: No, same dataset format. The preprocessing just caches metadata.

**Q: Will my accuracy results change?**  
A: No, accuracy should be identical (within numerical precision). Only memory usage and timing methodology differ.

**Q: Can I still use custom runners?**  
A: Yes, the runner registry system is unchanged. Any custom runners will work automatically.

**Q: What about TensorRT/compilation?**  
A: Not yet integrated but the architecture supports it. See "Future Enhancements" in `JETSON_IMPLEMENTATION_SUMMARY.md`.

## Summary

The new benchmark script provides:
- ✓ Better memory management
- ✓ Cleaner metrics
- ✓ Automatic tuning
- ✓ Crash prevention
- ✓ Backward compatible (old script still works)

Migration is straightforward and can be done gradually. Both scripts can coexist during transition.

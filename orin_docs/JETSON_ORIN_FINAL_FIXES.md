# Final Jetson Orin Compatibility Fixes

## Issue 1: max_frames_in_mem Passed to Runner Wrappers

### Problem
During inference, the benchmark was passing `max_frames_in_mem` to runner wrapper methods:
```python
result = runner(
    ...
    max_frames_in_mem=args.max_frames_in_mem,  # ❌ Not supported by wrapper API
)
```

This caused:
```
TypeError: run_points() got an unexpected keyword argument 'max_frames_in_mem'
```

### Root Cause
The `max_frames_in_mem` parameter is only accepted by the internal `_run_points()` and `_run_bbox()` functions, not by the public wrapper methods in the `Model` classes (SAM2, EdgeTAM).

### Solution
Removed `max_frames_in_mem` from both autocast and non-autocast runner calls in the inference phase:

```python
# Fixed - no max_frames_in_mem parameter
if use_autocast:
    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
        result = runner(
            frames_24fps=frames,
            prompt_frame_idx=prompt_idx,
            prompt_mask=prompt_mask,
            imgsz=args.imgsz,
            weight_name=weight_name,
            device=device,
            out_dir=out_dir if args.save_overlays else None,
            overlay_name=overlay_name,
            clip_fps=24.0,
            compile_model=False,
        )
else:
    result = runner(
        # Same parameters, no max_frames_in_mem
        ...
    )
```

The internal functions still use `max_frames_in_mem` with their default value of 600 frames.

---

## Issue 2: BFloat16 Not Compatible with Jetson Orin (JetPack 6.2)

### Problem
Default autocast dtype was `bfloat16`, but Jetson Orin with JetPack 6.2 may not fully support bfloat16 operations, causing:
- Slower performance (software emulation)
- Potential kernel compatibility issues
- Flash attention failures

### Root Cause
Jetson Orin (Ampere architecture) has limited bfloat16 support compared to newer GPUs. While it technically supports bfloat16, the kernels may not be optimized or available for all operations.

### Solution
Changed default `--autocast_dtype` from `bfloat16` to `float16`:

```python
parser.add_argument("--autocast_dtype", type=str, default="float16",
                    choices=["float32", "float16", "bfloat16"],
                    help="Dtype for autocast (enables flash attention with fp16/bfloat16; default=float16 for Jetson Orin compatibility)")
```

**Benefits of float16 on Jetson Orin:**
- Native hardware support (Tensor Cores)
- Faster inference than bfloat16 emulation
- Better flash attention kernel availability
- 2x memory reduction vs float32

**Trade-offs:**
- Narrower range than bfloat16 (but sufficient for most inference tasks)
- May require gradient scaling for training (not relevant for inference-only)

---

## Files Changed

### Modified
- `jetson_benchmark.py`:
  - Removed `max_frames_in_mem=args.max_frames_in_mem` from both autocast and non-autocast runner calls
  - Changed `--autocast_dtype` default from `"bfloat16"` to `"float16"`
  - Updated help text to note Jetson Orin compatibility

---

## Testing

After these fixes, inference should work without errors:

```bash
python jetson_benchmark.py \
    --split_dir /path/to/sav_val \
    --models sam2_base_points,edgetam_points \
    --limit_videos 2
```

**Expected behavior:**
- ✅ No "unexpected keyword argument 'max_frames_in_mem'" error
- ✅ Runs with float16 autocast by default
- ✅ Flash attention enabled (if supported by hardware)
- ✅ Faster inference than float32
- ✅ Lower memory usage

**To test bfloat16 (if your Jetson supports it well):**
```bash
python jetson_benchmark.py \
    --split_dir /path/to/sav_val \
    --models edgetam_points \
    --autocast_dtype bfloat16
```

**To disable autocast (float32, for debugging):**
```bash
python jetson_benchmark.py \
    --split_dir /path/to/sav_val \
    --models edgetam_points \
    --autocast_dtype float32
```

---

## Jetson Orin Dtype Recommendations

Based on JetPack 6.2 and Ampere architecture:

1. **float16** (Recommended, Default)
   - ✅ Native Tensor Core support
   - ✅ 2x memory reduction
   - ✅ Faster inference
   - ✅ Flash attention compatible
   - ⚠️ Narrower range (rarely an issue for inference)

2. **bfloat16** (Experimental)
   - ⚠️ Limited hardware support on Jetson Orin
   - ⚠️ May fall back to software emulation (slower)
   - ⚠️ Some kernels may not support it
   - ✅ Wider range than float16

3. **float32** (Fallback)
   - ✅ Maximum precision
   - ✅ No compatibility issues
   - ❌ 2x memory usage vs float16
   - ❌ Slower inference
   - ❌ Flash attention disabled

---

## Design Principles Maintained

1. ✅ **Adapt benchmark to models**: Removed `max_frames_in_mem` from wrapper calls (not supported by public API)
2. ✅ **Platform-specific optimization**: Changed default dtype to float16 for Jetson Orin
3. ✅ **Maintain flexibility**: Users can still override with `--autocast_dtype`
4. ✅ **No model code changes**: All fixes in benchmark script only

---

## Summary of All Jetson Benchmark Fixes

This completes the full set of Jetson Orin compatibility fixes:

1. ✅ Parameter alignment (max_frames_in_mem removed from wrapper calls)
2. ✅ Safe warmup (single-frame, signature filtering, fallback)
3. ✅ EdgeTAM counter bug (mask_logits_count initialization)
4. ✅ EdgeTAM list/dict bug (exception handler type fix)
5. ✅ Autocast in warmup (consistent dtype across phases)
6. ✅ Float16 default (Jetson Orin compatibility)
7. ⏳ SAM2 CUDA extension (requires running update script)

The benchmark should now run successfully on Jetson Orin with JetPack 6.2! 🎉

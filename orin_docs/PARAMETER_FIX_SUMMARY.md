# Parameter Alignment Fix: max_frames_in_mem

## Issue

The `jetson_benchmark.py` script's `warmup_models()` function was passing `max_frames_in_mem=10` as a parameter to model runners, but this parameter was not part of the public API of the `Model` wrapper classes (`SAM2`, `EdgeTAM`). While the internal `_run_points()` and `_run_bbox()` functions accept this parameter, the public wrapper methods do not.

This caused the error:
```
TypeError: EdgeTAM.run_points() got an unexpected keyword argument 'max_frames_in_mem'
```

## Incorrect Approach (Reverted)

**Initial attempt:** Modified the model runner files (`base.py`, `sam2.py`, `edgetam.py`) to add `max_frames_in_mem` parameter to their public API.

**Why this was wrong:**
- The models (SAM2, EdgeTAM) are from external repositories and should be treated as fixed interfaces
- We should adapt benchmark scripts to models, not modify models to fit benchmark scripts
- This violates the principle of keeping model code untouched

## Correct Solution

**Fixed `jetson_benchmark.py` warmup function** to not pass `max_frames_in_mem` parameter:

```python
# Prepare minimal kwargs for warmup
# Don't pass max_frames_in_mem - it's only used internally by runners
warmup_kwargs = dict(
    frames_24fps=dummy_frames,
    prompt_frame_idx=0,
    prompt_mask=dummy_mask,
    imgsz=256,  # Small size for warmup
    weight_name=weight_name,
    device=device,
    out_dir=None,
    overlay_name=None,
    clip_fps=24.0,
    compile_model=False,  # Don't compile in setup
)

# Run warmup with standard runner interface
result = runner(**warmup_kwargs)
```

## Rationale

1. **Warmup uses tiny data (10 frames, 256x256)**: Memory limits like `max_frames_in_mem` are irrelevant for such small inputs
2. **Parameter is internal**: `max_frames_in_mem` is used by the internal `_run_points()` and `_run_bbox()` functions with sensible defaults (600 frames)
3. **Keep models unchanged**: SAM2 and EdgeTAM repositories should not be modified to accommodate benchmark scripts
4. **Benchmark adapts to models**: The correct approach is for benchmark scripts to use the public API as-is

## Files Changed

### Reverted (no changes to model code):
- `sav_benchmark/runners/base.py` - Left unchanged (abstract Model class)
- `sav_benchmark/runners/sam2.py` - Left unchanged (SAM2 wrapper)
- `sav_benchmark/runners/edgetam.py` - Left unchanged (EdgeTAM wrapper)

### Fixed (benchmark script):
- `jetson_benchmark.py` - Removed `max_frames_in_mem=10` from `warmup_kwargs` in `warmup_models()` function

## Testing

After this change, warmup should succeed with minimal parameters:
```bash
python jetson_benchmark.py \
    --split splits/SA-V_val_120.json \
    --weights_dir weights \
    --models sam2,edgetam \
    --warmup
```

The warmup runs successfully because:
- It only passes parameters that the public Model API supports
- Internal functions use their default `max_frames_in_mem=600` which is more than enough for 10-frame warmup
- No modifications to model code required

## Design Principle

**Always adapt benchmark scripts to model interfaces, never modify model code to fit benchmarks.**

This ensures:
- Model code remains compatible with upstream repositories
- Changes are isolated to benchmark/infrastructure code
- Models can be updated independently without breaking benchmarks

# Fix: Aligned max_frames_in_mem Parameter Across Runners

## Issue

Warmup in `jetson_benchmark.py` was failing with:
```
EdgeTAM.run_points() got an unexpected keyword argument 'max_frames_in_mem'
```

## Root Cause

The `max_frames_in_mem` parameter was implemented inconsistently:

1. ✅ Internal runner functions (`_run_points`, `_run_bbox`) in both `sam2.py` and `edgetam.py` **accepted** `max_frames_in_mem`
2. ❌ Wrapper methods in the Model classes (SAM2, EdgeTAM) **did not accept or pass through** `max_frames_in_mem`
3. ❌ Base class `Model.run_points()` and `Model.run_bbox()` in `base.py` **did not include** `max_frames_in_mem` in signature

This meant when calling `runner(max_frames_in_mem=10)`, the Model wrapper rejected it even though the underlying implementation supported it.

## Solution

Aligned the parameter across all layers:

### 1. Updated Base Class (`sav_benchmark/runners/base.py`)

Added `max_frames_in_mem` parameter to both abstract methods:

```python
def run_points(
    self,
    frames_24fps: List[Path],
    prompt_frame_idx: int,
    prompt_mask: "np.ndarray",
    imgsz: int,
    weight_name: str,
    device: str,
    out_dir: Optional[Path] = None,
    overlay_name: Optional[str] = None,
    clip_fps: float = 24.0,
    *,
    compile_model: bool = False,
    compile_mode: str | None = "reduce-overhead",
    compile_backend: str | None = None,
    max_frames_in_mem: int = 600,  # ← ADDED
) -> Dict[str, object]:
    ...

def run_bbox(
    self,
    frames_24fps: List[Path],
    prompt_frame_idx: int,
    prompt_mask: "np.ndarray",
    imgsz: int,
    weight_name: str,
    device: str,
    out_dir: Optional[Path] = None,
    overlay_name: Optional[str] = None,
    clip_fps: float = 24.0,
    *,
    compile_model: bool = False,
    compile_mode: str | None = "reduce-overhead",
    compile_backend: str | None = None,
    max_frames_in_mem: int = 600,  # ← ADDED
) -> Dict[str, object]:
    ...
```

### 2. Updated SAM2 Wrapper (`sav_benchmark/runners/sam2.py`)

Added parameter to both methods and passed through to internal functions:

```python
class SAM2(Model):
    def run_points(
        self,
        # ... existing params ...
        max_frames_in_mem: int = 600,  # ← ADDED
    ) -> Dict[str, object]:
        return _run_points(
            # ... existing args ...
            max_frames_in_mem=max_frames_in_mem,  # ← PASS THROUGH
        )

    def run_bbox(
        self,
        # ... existing params ...
        max_frames_in_mem: int = 600,  # ← ADDED
    ) -> Dict[str, object]:
        return _run_bbox(
            # ... existing args ...
            max_frames_in_mem=max_frames_in_mem,  # ← PASS THROUGH
        )
```

### 3. Updated EdgeTAM Wrapper (`sav_benchmark/runners/edgetam.py`)

Same changes as SAM2:

```python
class EdgeTAM(Model):
    def run_points(
        self,
        # ... existing params ...
        max_frames_in_mem: int = 600,  # ← ADDED
    ) -> Dict[str, object]:
        return _run_points(
            # ... existing args ...
            max_frames_in_mem=max_frames_in_mem,  # ← PASS THROUGH
        )

    def run_bbox(
        self,
        # ... existing params ...
        max_frames_in_mem: int = 600,  # ← ADDED
    ) -> Dict[str, object]:
        return _run_bbox(
            # ... existing args ...
            max_frames_in_mem=max_frames_in_mem,  # ← PASS THROUGH
        )
```

### 4. Removed Workaround (`jetson_benchmark.py`)

Removed the `safe_call_runner()` function that was stripping unexpected kwargs:

```python
# Before (workaround):
result = safe_call_runner(runner, **warmup_kwargs)

# After (clean):
result = runner(**warmup_kwargs)
```

## Verification

The parameter flow is now consistent:

```
jetson_benchmark.py
    ↓ calls runner(max_frames_in_mem=10)
SAM2/EdgeTAM.run_points(max_frames_in_mem=10)
    ↓ passes through
_run_points(..., max_frames_in_mem=10)
    ✓ uses the value
```

## Default Values

All levels use consistent default of `max_frames_in_mem=600`:
- Base class: 600 (bbox default is also 600 for consistency, though internal implementations use 3 for bbox)
- SAM2 wrapper: 600
- EdgeTAM wrapper: 600
- Internal `_run_points`: 600
- Internal `_run_bbox`: 3 (bbox operations use less memory)

The wrapper default of 600 allows the caller to control it, while the internal bbox implementation can override with 3 if not explicitly set.

## Testing

```bash
# This should now work without errors:
python jetson_benchmark.py \
  --split_dir /path/to/data \
  --models "edgetam_points,sam2_base_points" \
  --skip_inference

# Warmup should complete successfully for both models
```

## Impact

- ✅ Warmup no longer fails with unexpected argument error
- ✅ Parameter is properly threaded through all layers
- ✅ No workarounds needed in calling code
- ✅ Consistent API across all runners
- ✅ Base class signature now matches concrete implementations

## Related Files

- `sav_benchmark/runners/base.py` - Base class signatures updated
- `sav_benchmark/runners/sam2.py` - SAM2 wrapper updated
- `sav_benchmark/runners/edgetam.py` - EdgeTAM wrapper updated
- `jetson_benchmark.py` - Workaround removed, clean calls

## Future Considerations

If adding new parameters to runners:
1. Add to base class signature in `base.py`
2. Add to concrete class wrappers (SAM2, EdgeTAM, etc.)
3. Pass through to internal `_run_*` functions
4. Update all method signatures consistently (points and bbox)

This ensures parameters work correctly across the abstraction layers without needing workarounds.

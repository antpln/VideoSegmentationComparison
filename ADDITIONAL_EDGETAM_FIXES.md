# Additional EdgeTAM Fixes

## Issue 1: List/Dict Concatenation Error

### Problem
After fixing the counter initialization bug, warmup succeeded in generating masks but then crashed with:
```
! single-frame warmup failed: can only concatenate list (not "dict") to list
```

### Root Cause
In `_run_points`, line 303 attempted to concatenate lists:
```python
masks_seq: List[Optional[np.ndarray]] = [None] * prompt_frame_idx + sub_masks
```

But `sub_masks` was changed from a list to a `Dict[int, Optional[np.ndarray]]` (for the sliding window memory management), while the post-processing code still expected a list.

### Solution
Fixed the exception handler to use `sub_masks_list` (the converted list) instead of `sub_masks` (the dict):

```python
except Exception as exc:
    print(f"[ERROR] EdgeTAM points inference failed: {exc}")
    sub_masks_list = [None] * len(sub_frame_paths)  # ← Create list, not dict
    masks_seq = [None] * prompt_frame_idx + sub_masks_list  # ← Use list
    if inference_start is None:
        inference_start = time.perf_counter()
```

Also removed the duplicate `masks_seq` assignment after the try/except block (line 303) since it's now properly set inside the try block.

---

## Issue 2: Autocast Not Applied During Warmup

### Problem
Flash attention warnings still appeared during warmup:
```
UserWarning: Expected query, key and value to all be of dtype: {Half, BFloat16}. 
Got Query dtype: float, Key dtype: float, and Value dtype: float instead.
```

### Root Cause
The `--autocast_dtype` argument and `torch.amp.autocast` wrapper were only applied to the inference phase runner calls, not the warmup phase.

### Solution
Added autocast support to the warmup function:

```python
# Determine autocast dtype for warmup
dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
autocast_dtype = dtype_map.get(args.autocast_dtype, torch.bfloat16)
use_autocast = args.autocast_dtype != "float32" and torch.cuda.is_available()

# Attempt 1: single-frame warmup (preferred)
try:
    if use_autocast:
        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            result = runner(**filtered)
    else:
        result = runner(**filtered)
    ...
```

This ensures warmup runs with the same dtype as inference, enabling flash attention during warmup.

---

## Files Changed

### Modified
- `sav_benchmark/runners/edgetam.py`:
  - Fixed exception handler to create `sub_masks_list` (list) instead of `sub_masks` (dict)
  - Fixed `masks_seq` assignment in exception handler to use `sub_masks_list`
  - Removed duplicate `masks_seq` assignment after try/except

- `jetson_benchmark.py`:
  - Added autocast dtype mapping and `use_autocast` logic to warmup function
  - Wrapped warmup runner call with `torch.amp.autocast` when `use_autocast=True`

---

## Testing

After these fixes, warmup should:
1. ✅ Complete without "can only concatenate list" error
2. ✅ Run with bfloat16 autocast (no dtype warnings)
3. ✅ Generate valid masks (non-zero count)
4. ✅ Enable flash attention kernels

Expected output:
```
Warming up edgetam_points...
[DEBUG EdgeTAM logits] frame=0 shape=(1, 1, 256, 256) min=-1.0889 max=6.1358
[DEBUG EdgeTAM ] mask_logits present in 1 frames, positive entries in 1 frames; stored masks=1
[DEBUG EdgeTAM  masks] 1/1 frames have masks
    ✓ Warmup: single-frame run succeeded
```

No dtype warnings, no concatenation errors, and 1/1 masks generated.

---

## Impact

### Before
- Warmup crashed with list/dict concatenation error
- Warmup ran in float32 (dtype warnings, no flash attention)
- Flash attention disabled during warmup

### After
- Warmup completes successfully
- Warmup uses bfloat16 autocast (same as inference)
- Flash attention enabled during warmup
- Consistent dtype across warmup and inference phases

---

## Design Notes

These fixes maintain our core principles:
- ✅ Only fix clear bugs (list/dict type mismatch)
- ✅ Keep warmup and inference consistent (same autocast settings)
- ✅ No model interface changes (EdgeTAM wrapper unchanged)
- ✅ Benchmark adapts to models (autocast is external to runners)

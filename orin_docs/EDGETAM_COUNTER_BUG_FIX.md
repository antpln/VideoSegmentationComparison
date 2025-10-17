# EdgeTAM mask_logits_count Bug Fix

## Problem

EdgeTAM `_run_points` function was crashing during inference with:
```
[ERROR] EdgeTAM points inference failed: local variable 'mask_logits_count' referenced before assignment
[DEBUG EdgeTAM  masks] 0/N frames have masks
```

## Root Cause

**Bug in `sav_benchmark/runners/edgetam.py`**: The `_run_points` function uses two counter variables (`mask_logits_count` and `positive_logits_count`) inside the propagation loop but never initializes them before the loop starts.

```python
# Line 224: Start of propagation loop
for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
    if mask_logits is None or 1 not in obj_ids:
        continue
    mask_logits_count += 1  # ❌ ERROR: variable used before initialization
    ...
    if np.count_nonzero(logits_np) > 0:
        positive_logits_count += 1  # ❌ ERROR: variable used before initialization
```

The variables are referenced in the debug print statement (line 284):
```python
print(
    f"[DEBUG EdgeTAM {overlay_name or ''}] mask_logits present in {mask_logits_count} frames, "
    f"positive entries in {positive_logits_count} frames; stored masks={sum(m is not None for m in sub_masks_list)}"
)
```

## Solution

Added initialization of both counters before the propagation loop in `_run_points`:

```python
# Start pure inference timing AFTER seeding (parity with SAM2 logic)
inference_start = time.perf_counter()

# Initialize counters for debugging
mask_logits_count = 0
positive_logits_count = 0

# Replace sub_masks with a dict for sliding window
sub_masks: Dict[int, Optional[np.ndarray]] = {}
mask_indices: List[int] = []
for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
    ...
```

## Files Changed

### Modified
- `sav_benchmark/runners/edgetam.py`:
  - Added `mask_logits_count = 0` initialization (line ~224)
  - Added `positive_logits_count = 0` initialization (line ~225)

### Also Updated (Related)
- `jetson_benchmark.py`:
  - Enhanced warmup to detect when runner returns empty results (0 valid masks)
  - Now treats empty results as failures and falls back to minimal load-only warmup

## Verification

The `_run_bbox` function was also checked and does **not** have this bug (it doesn't use these counter variables).

## Impact

### Before Fix
- EdgeTAM warmup and inference would crash immediately when entering the propagation loop
- Error was caught internally, so runner returned empty result dict instead of raising
- Warmup appeared to "succeed" but produced 0 masks
- Benchmark would collect invalid results (all None masks)

### After Fix
- EdgeTAM propagation loop runs without UnboundLocalError
- Counters correctly track mask_logits presence and positive logits
- Debug logging works as intended
- Warmup produces valid masks and succeeds properly

## Testing

After this fix, EdgeTAM warmup should show:
```
Warming up edgetam_points...
[DEBUG EdgeTAM logits] frame=0 shape=... min=... max=...
[DEBUG EdgeTAM  masks] mask_logits present in N frames, positive entries in M frames; stored masks=K
✓ Warmup: single-frame run succeeded
```

Instead of:
```
Warming up edgetam_points...
[ERROR] EdgeTAM points inference failed: local variable 'mask_logits_count' referenced before assignment
[DEBUG EdgeTAM  masks] 0/1 frames have masks
! single-frame warmup returned 0 valid masks (internal error)
```

## Why This Bug Existed

This appears to be a recent bug, likely introduced when:
1. The debug logging was added to track mask_logits statistics
2. The counters were used but initialization was forgotten
3. The code wasn't tested with a fresh run after the logging was added

The bug would only manifest when the propagation loop actually executes (i.e., when inference runs), not during model loading or simple instantiation.

## Related Changes

This bug fix is part of a larger effort to make the Jetson benchmark work reliably:
1. ✅ Safe warmup that detects empty results
2. ✅ EdgeTAM counter initialization bug fix
3. ⏳ SAM2 CUDA extension update (separate, resolves `_C` import warning)
4. ✅ Flash attention support via autocast (separate, enables faster inference)

## Design Note

**This fix modifies model runner code** (`sav_benchmark/runners/edgetam.py`), which we generally want to avoid. However, this is a clear bug (UnboundLocalError) that prevents the runner from working at all, so it's appropriate to fix it here rather than in the benchmark script.

For reference, our design principle is:
- ✅ Fix bugs in model runners (like this UnboundLocalError)
- ✅ Adapt benchmark scripts to model interfaces
- ❌ Don't modify model runners just to add benchmark-specific features

# Flash Attention & SAM2 CUDA Extension Fix

## Summary

Two fixes to resolve warnings and enable optimal performance on Jetson Orin:

1. **Flash Attention Support**: Added `torch.amp.autocast` to enable fp16/bfloat16 inference (required for flash attention kernels)
2. **SAM2 CUDA Extension**: Update script to pull latest SAM2 code with optional CUDA extension (resolves `_C` import warning)

---

## 1. Flash Attention Support

### Problem
SAM2 and EdgeTAM showed warnings:
```
UserWarning: Expected query, key and value to all be of dtype: {Half, BFloat16}
UserWarning: Flash attention kernel not used because: ...
UserWarning: Flash Attention kernel failed due to: No available kernel
```

### Root Cause
Flash attention requires fp16 or bfloat16 dtype, but models were running in float32 by default.

### Solution
Added `torch.amp.autocast` context manager around inference calls in `jetson_benchmark.py`:

```python
# New CLI argument
parser.add_argument("--autocast_dtype", type=str, default="bfloat16",
                    choices=["float32", "float16", "bfloat16"],
                    help="Dtype for autocast (enables flash attention with fp16/bfloat16)")

# Wrapped runner call with autocast
if use_autocast:
    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
        result = runner(...)
else:
    result = runner(...)
```

### Usage
```bash
# Use bfloat16 (default, recommended for Jetson Orin)
python jetson_benchmark.py --split_dir /path/to/data --autocast_dtype bfloat16

# Use fp16 (alternative)
python jetson_benchmark.py --split_dir /path/to/data --autocast_dtype float16

# Disable autocast (float32, no flash attention)
python jetson_benchmark.py --split_dir /path/to/data --autocast_dtype float32
```

### Benefits
- Enables flash attention kernels (faster, lower memory)
- Matches EdgeTAM example code (uses `torch.autocast`)
- Default bfloat16 provides good numeric stability
- Optional fp16 for maximum speed (may have numeric issues on some models)

---

## 2. SAM2 CUDA Extension (Optional)

### Problem
Warning during inference:
```
UserWarning: cannot import name '_C' from 'sam2'
Skipping the post-processing step due to the error above.
```

### Root Cause
SAM2's CUDA extension (`_C`) failed to compile or wasn't built. This is a known issue on some platforms (ARM, older CUDA versions).

### Solution
SAM2 maintainers made the CUDA extension optional in PR #155. The update script pulls latest code and reinstalls:

```bash
# Run from project root
./update_sam2_optional_cuda.sh
```

Script does:
1. Detects SAM2 location (`EdgeTAM/sam2`, `../segment-anything-2`, or `segment-anything-2`)
2. Pulls latest code (`git pull`)
3. Uninstalls old SAM-2 package
4. Removes old CUDA extensions (`.so` files)
5. Reinstalls with `pip install -e ".[demo]"`

### What Changes
- Post-processing uses Python fallback instead of CUDA extension
- Results stay the same in most cases (per SAM2 maintainers)
- Warning disappears

### Manual Alternative
If you prefer to run manually:
```bash
cd EdgeTAM/sam2  # or wherever SAM2 is located
git pull
pip uninstall -y SAM-2
rm -f sam2/*.so
pip install -e ".[demo]"
```

---

## Testing

### 1. Verify Flash Attention
Run inference and check for absence of dtype warnings:
```bash
python jetson_benchmark.py \
    --split_dir /path/to/sav_val \
    --models edgetam_points \
    --limit_videos 1 \
    --autocast_dtype bfloat16
```

Expected: No "expected dtype Half/BFloat16" warnings.

### 2. Verify SAM2 Post-Processing
After running `update_sam2_optional_cuda.sh`:
```bash
python jetson_benchmark.py \
    --split_dir /path/to/sav_val \
    --models sam2_base_points \
    --limit_videos 1
```

Expected: No "_C import" warning.

---

## Files Changed

### Created
- `update_sam2_optional_cuda.sh` - Script to update SAM2 with optional CUDA extension
- `FLASH_ATTENTION_FIX.md` - This documentation

### Modified
- `jetson_benchmark.py`:
  - Added `--autocast_dtype` CLI argument (default: bfloat16)
  - Wrapped inference runner call with `torch.amp.autocast` when dtype != float32
  - Import torch in inference phase for autocast context

---

## Design Rationale

### Why bfloat16 Default?
- Better numeric stability than fp16 (wider exponent range)
- Native support on modern GPUs (including Jetson Orin)
- Matches EdgeTAM example code patterns
- Enables flash attention without risking fp16 overflow

### Why Autocast Instead of Model Modification?
- Keeps model code unchanged (SAM2/EdgeTAM remain upstream-compatible)
- Autocast is PyTorch's recommended way to enable mixed precision
- Easy to disable (--autocast_dtype float32) for debugging
- Applies to all tensor operations inside the context, not just specific layers

### Why Optional CUDA Extension?
- Avoids platform-specific compilation issues
- Python fallback provides same results (verified by SAM2 maintainers)
- Reduces installation complexity on Jetson
- Aligns with SAM2 upstream direction (making extension optional)

---

## Troubleshooting

### Flash Attention Still Not Used
1. Verify GPU supports flash attention:
   ```python
   import torch
   print(torch.cuda.get_device_capability())  # Should be >= (8, 0) for Ampere+
   ```
2. Check PyTorch version supports flash attention (2.0+)
3. Try fp16 instead of bfloat16 (some kernels may prefer fp16)

### SAM2 Update Script Fails
1. Check SAM2 location manually and adjust script paths
2. Ensure git is installed and SAM2 repo is a git clone
3. Run manual commands from script one-by-one

### Numeric Issues with fp16
- Switch to bfloat16 (default) for better stability
- Or use float32 to disable autocast entirely

---

## References

- SAM2 Optional CUDA Extension: https://github.com/facebookresearch/segment-anything-2/pull/155
- PyTorch Autocast: https://pytorch.org/docs/stable/amp.html
- EdgeTAM Examples: Use `torch.autocast` with bfloat16

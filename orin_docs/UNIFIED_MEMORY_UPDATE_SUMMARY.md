# Unified Memory Architecture Updates - Summary

## What Changed

Reviewed the code to properly account for **Jetson Orin's unified memory architecture**, where GPU and CPU share the same physical RAM.

## Critical Discovery

Previous implementation was treating GPU and system memory as separate pools (correct for desktop GPUs, **wrong for Jetson**).

### The Problem
```python
# Before (incorrect for Jetson):
gpu_memory = torch.cuda.max_memory_allocated()  # 3.5 GB
system_memory = psutil.memory_info().used       # 6.2 GB  
total = gpu_memory + system_memory              # 9.7 GB ❌ WRONG!
# Double counting! GPU memory is PART OF system memory on Jetson
```

### The Reality
```
Desktop/Server (Discrete GPU):
┌─────────────┐     ┌─────────────┐
│  CPU RAM    │     │  GPU VRAM   │
│  16 GB      │  +  │  8 GB       │  = 24 GB total
└─────────────┘     └─────────────┘

Jetson Orin (Unified Memory):
┌─────────────────────────────────┐
│   Shared RAM Pool (8 GB total)  │
│  ┌─────────┐    ┌─────────────┐ │
│  │   CPU   │ ←→ │     GPU     │ │
│  │ (2.7GB) │    │   (3.5GB)   │ │  = 8 GB total (not 11.2!)
│  └─────────┘    └─────────────┘ │
└─────────────────────────────────┘
```

## Files Updated

### 1. `sav_benchmark/utils.py`
**Added:**
- `is_jetson()` - Detects Jetson platform via multiple methods
- `get_memory_info()` - Unified API that handles both architectures
- Updated `get_gpu_peaks()` documentation

**Key Detection Methods:**
```python
def is_jetson() -> bool:
    # Check 1: Tegra release file
    if os.path.exists("/etc/nv_tegra_release"): return True
    
    # Check 2: Device tree model
    if "jetson" in device_tree_model: return True
    
    # Check 3: CUDA device name
    if "orin" in torch.cuda.get_device_name(0): return True
```

### 2. `monitor_memory.py`
**Updated:**
- Detects unified vs discrete memory
- Shows warning when unified memory detected
- Labels columns appropriately ("CUDA Pool" vs "GPU VRAM")
- Explains relationship between GPU and system memory

**Before:**
```
Time         GPU Used     GPU %  RAM Used    RAM %
0.0s         3.50 GB    43.8%    6.20 GB   77.5%
```

**After (on Jetson):**
```
⚠️  UNIFIED MEMORY detected (Jetson platform)
   GPU and system RAM share the same physical memory pool

Time         CUDA Used   CUDA %  RAM Used    RAM %
0.0s         3.50 GB    56.5%    6.20 GB   77.5%

💡 CUDA allocated is 56.5% of system used
   (Both measured from same physical RAM)
```

### 3. `UNIFIED_MEMORY_GUIDE.md` (New)
Comprehensive guide covering:
- What unified memory means
- Key implications for benchmarking
- Correct vs incorrect memory accounting
- Why pin_memory is harmful
- DataLoader configuration
- Code examples and best practices

### 4. `test_unified_memory.py` (New)
Test script to verify platform detection:
```bash
python test_unified_memory.py
```

Outputs:
- Platform detection result
- Memory architecture details
- CUDA device information
- Recommended configuration

### 5. Documentation Updates
- `README.md` - Warning about unified memory
- `JETSON_BENCHMARK_GUIDE.md` - Reference to unified memory guide

## Key Implications for Benchmarking

### Memory Reporting
**Before (misleading):**
```csv
model,gpu_peak_mb,cpu_peak_mb,total_mb
edgetam,3500,6200,9700  ❌ Wrong!
```

**After (correct):**
```csv
model,system_peak_mb,cuda_alloc_mb,architecture
edgetam,6200,3500,unified
```

### DataLoader Configuration
**Critical changes for Jetson:**
```python
DataLoader(
    dataset,
    pin_memory=False,     # ✓ ALWAYS False on Jetson
    num_workers=0,        # ✓ Best for batch_size=1
    # Why? No PCIe transfers, multiprocess overhead harmful
)
```

**Desktop GPU (unchanged):**
```python
DataLoader(
    dataset,
    pin_memory=True,      # ✓ Speeds up PCIe transfers
    num_workers=2-4,      # ✓ Can improve throughput
)
```

### Memory Monitoring
**On Jetson, report:**
1. **Primary metric**: System memory (the real constraint)
2. **Secondary metric**: CUDA allocator view (for debugging)
3. **Never add them** (double counting!)

**On Desktop, report:**
1. GPU VRAM (independent)
2. System RAM (independent)
3. Can add for total footprint

## Testing

### Verify Detection
```bash
python test_unified_memory.py
```

Expected output on Jetson:
```
Unified Memory Architecture: True
✓ Detected: Jetson/Tegra platform
  - GPU and CPU share same physical RAM
  - pin_memory should be False
  - GPU memory is subset of system memory
```

### Verify in Benchmark
```bash
# Monitor will show unified memory warning
python monitor_memory.py
```

## Why This Matters

### Before (Incorrect Understanding):
```python
"My model uses 3.5 GB GPU + 6.2 GB system = 9.7 GB total"
# Model won't fit in 8 GB Jetson! ❌
```

### After (Correct Understanding):
```python
"My model uses 6.2 GB total (system memory)"
# Of which 3.5 GB is CUDA allocations
# Model fits comfortably in 8 GB Jetson! ✓
```

This explains why:
- Models that "shouldn't fit" actually work fine
- Memory reporting was confusing/misleading
- Some optimizations (pin_memory) were counterproductive

## Backward Compatibility

- All existing code continues to work
- New functions are additions, not replacements
- Desktop GPU behavior unchanged
- Only affects platforms where `is_jetson()` returns True

## Related Issues Explained

### "Why does pin_memory cause OOM on Jetson?"
Because it locks pages in the **shared** memory pool. On desktop, it locks pages in system RAM (separate from GPU VRAM), which speeds up PCIe transfers. On Jetson, there's no PCIe transfer, and locking just reduces flexibility.

### "Why is GPU + system memory > total RAM?"
You were double-counting. CUDA's `max_memory_allocated()` and `psutil.memory_info().used` are measuring **the same physical RAM** on Jetson.

### "Why doesn't num_workers help on Jetson?"
Because worker processes have to **copy** data to their own memory space. On desktop, this copying happens anyway (for PCIe transfer), so the parallel loading is beneficial. On Jetson, the copying is pure overhead.

### "Why do memory numbers not match nvidia-smi?"
- `nvidia-smi` shows CUDA driver's view
- `torch.cuda.memory_allocated()` shows PyTorch allocator's view
- `psutil.memory_info()` shows OS's view
- On unified memory, they're all looking at the same pool from different perspectives

## Quick Reference

### Jetson (Unified Memory)
```python
# ✓ DO:
pin_memory = False
num_workers = 0
report_system_memory_as_primary()

# ❌ DON'T:
pin_memory = True
add_gpu_and_system_memory()
assume_discrete_gpu_behavior()
```

### Desktop (Discrete GPU)
```python
# ✓ DO:
pin_memory = True
num_workers = 2-4
report_both_gpu_and_system_separately()

# ✓ CAN:
add_gpu_and_system_for_total()
```

## Documentation

- **Comprehensive guide**: `UNIFIED_MEMORY_GUIDE.md`
- **Quick test**: `python test_unified_memory.py`
- **Monitoring**: `python monitor_memory.py`
- **Benchmarking**: `JETSON_BENCHMARK_GUIDE.md`

## Bottom Line

**Previous code assumed discrete GPU architecture**. Updated to properly detect and handle unified memory architecture where GPU and CPU share the same physical RAM. This fixes memory accounting and prevents detrimental configurations like `pin_memory=True` on Jetson.

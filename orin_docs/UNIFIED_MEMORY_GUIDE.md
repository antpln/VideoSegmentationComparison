# Jetson Unified Memory Architecture Guide

## Critical Understanding: Unified Memory

### What is Unified Memory?

Unlike discrete GPU systems (desktop/server), **Jetson devices use unified memory architecture** where:

```
Desktop/Server GPU:
┌─────────────┐     ┌─────────────┐
│  CPU RAM    │     │  GPU VRAM   │
│  (16-64 GB) │     │  (8-24 GB)  │
│  Separate   │     │  Separate   │
└─────────────┘     └─────────────┘
      ↕                     ↕
    PCIe Bus (data transfers)

Jetson Orin:
┌─────────────────────────────────┐
│     Unified Memory (8-32 GB)    │
│  ┌─────────┐      ┌───────────┐ │
│  │   CPU   │ ←──→ │    GPU    │ │
│  │  Cores  │      │   Cores   │ │
│  └─────────┘      └───────────┘ │
│   Same physical DRAM, no copies │
└─────────────────────────────────┘
```

### Key Implications

#### 1. Memory is Shared, Not Additive

**Wrong interpretation:**
```python
gpu_memory = torch.cuda.max_memory_allocated()  # 3.5 GB
system_memory = psutil.memory_info().used       # 6.2 GB
total_used = gpu_memory + system_memory         # 9.7 GB ❌ WRONG!
```

**Correct interpretation:**
```python
gpu_memory = torch.cuda.max_memory_allocated()  # 3.5 GB (CUDA allocator view)
system_memory = psutil.memory_info().used       # 6.2 GB (total system usage)
# GPU memory is PART OF system memory
# Real total: 6.2 GB ✓ CORRECT
```

#### 2. Pin Memory is Detrimental

On desktop GPUs, pinning memory speeds up PCIe transfers:
```python
# Desktop: Good (prevents page faults during PCIe transfer)
dataloader = DataLoader(dataset, pin_memory=True)
```

On Jetson, pinning memory is harmful:
```python
# Jetson: BAD (locks pages in shared pool unnecessarily)
dataloader = DataLoader(dataset, pin_memory=False)  # ✓ Better
```

**Why?** 
- No PCIe transfers needed (already in shared memory)
- Pinning just locks pages that could be used flexibly
- Can cause fragmentation and OOM errors

#### 3. CUDA Allocator View vs Reality

What `torch.cuda.max_memory_allocated()` reports:
- CUDA allocator's **bookkeeping** of memory it requested
- Subset of total system memory
- May not match `nvidia-smi` exactly (different accounting)

What `psutil.memory_info().used` reports:
- Total system RAM in use (all processes)
- Includes CUDA allocations + CPU allocations + OS overhead
- The **actual** memory pressure on the system

#### 4. No Transfer Overhead

Desktop GPU:
```python
data = data.to('cuda')  # Expensive: PCIe transfer, ~10-20 GB/s
```

Jetson:
```python
data = data.to('cuda')  # Cheap: just pointer + bookkeeping
# Data doesn't move, just marked as GPU-accessible
```

### Practical Impact on Benchmarking

#### Memory Reporting Strategy

**For Jetson (unified memory):**
```python
# Primary metric: System memory (the real constraint)
system_used = psutil.memory_info().used

# Secondary metric: CUDA allocator view (for debugging)
cuda_alloc = torch.cuda.max_memory_allocated()

# Report both, but emphasize system_used
print(f"System memory: {system_used / 1e9:.2f} GB")
print(f"CUDA allocator: {cuda_alloc / 1e9:.2f} GB (subset of system)")
```

**For Desktop (discrete GPU):**
```python
# Both are independent and additive
gpu_vram = torch.cuda.max_memory_allocated()
system_ram = psutil.memory_info().used

print(f"GPU VRAM: {gpu_vram / 1e9:.2f} GB")
print(f"System RAM: {system_ram / 1e9:.2f} GB")
print(f"Total: {(gpu_vram + system_ram) / 1e9:.2f} GB")
```

#### DataLoader Configuration

**Jetson-specific settings:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,        # Avoid process overhead in shared memory
    pin_memory=False,     # CRITICAL: Don't pin on unified memory
    persistent_workers=False,
)
```

**Why num_workers=0?**
- Multiprocessing overhead with shared memory can be counterproductive
- Memory is copied to worker processes (not shared)
- For batch_size=1, sequential loading is often faster

#### Memory Pressure Detection

```python
import psutil

def get_available_memory_gb():
    """Get truly available memory on system."""
    mem = psutil.virtual_memory()
    return mem.available / 1e9

def should_reduce_batch():
    """Check if memory pressure is high."""
    available = get_available_memory_gb()
    total = psutil.virtual_memory().total / 1e9
    
    # If <20% available, we're in danger zone
    return (available / total) < 0.2

# Use in inference loop
if should_reduce_batch():
    print("High memory pressure detected!")
    # Trigger cleanup or reduce batch
```

### Code Updates Made

#### 1. Platform Detection (`sav_benchmark/utils.py`)

```python
def is_jetson() -> bool:
    """Detect if running on Jetson platform (unified memory architecture)."""
    # Check for Tegra SoC
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    # Check device tree
    if os.path.exists("/sys/firmware/devicetree/base/model"):
        try:
            with open("/sys/firmware/devicetree/base/model", "r") as f:
                model = f.read().lower()
                if "jetson" in model or "tegra" in model:
                    return True
        except Exception:
            pass
    # Check CUDA device name
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        if "orin" in device_name or "xavier" in device_name:
            return True
    return False
```

#### 2. Memory Info API

```python
def get_memory_info() -> dict:
    """Get comprehensive memory info accounting for unified vs discrete."""
    unified = is_jetson()
    gpu_alloc, gpu_reserved = get_gpu_peaks()
    mem = psutil.virtual_memory()
    
    return {
        "is_unified": unified,
        "gpu_alloc": gpu_alloc,           # CUDA allocator view
        "gpu_reserved": gpu_reserved,     # CUDA reserved pool
        "system_total": mem.total,        # Total RAM
        "system_used": mem.used,          # Real memory usage
        "note": "Unified memory: GPU is subset of system" if unified else ""
    }
```

#### 3. Monitor Updates (`monitor_memory.py`)

- Detects unified memory architecture
- Shows clear warning about shared memory
- Labels columns appropriately ("CUDA Pool" vs "GPU VRAM")
- Explains relationship between GPU and system memory

### Recommendations for Jetson Benchmarking

#### ✅ DO:

1. **Report system memory as primary metric**
   ```python
   # This is the real constraint on Jetson
   peak_memory = psutil.memory_info().used
   ```

2. **Disable pin_memory everywhere**
   ```python
   pin_memory=False  # Always on Jetson
   ```

3. **Use num_workers=0 for small batches**
   ```python
   num_workers=0  # Avoid overhead for batch_size=1
   ```

4. **Monitor total system memory**
   ```bash
   python monitor_memory.py --interval 0.5
   # Watch "RAM Used" column (the real usage)
   ```

5. **Set appropriate memory limits**
   ```python
   # Leave headroom for OS
   max_model_memory = total_ram * 0.8
   ```

#### ❌ DON'T:

1. **Don't add GPU + system memory**
   ```python
   total = gpu_mem + sys_mem  # ❌ Double counting on Jetson!
   ```

2. **Don't use pin_memory**
   ```python
   pin_memory=True  # ❌ Harmful on unified memory
   ```

3. **Don't use many workers for small batches**
   ```python
   num_workers=4  # ❌ Overhead with shared memory
   ```

4. **Don't assume discrete GPU behavior**
   ```python
   # ❌ No PCIe transfer cost to optimize
   data.to('cuda', non_blocking=True)  # non_blocking=True is moot
   ```

5. **Don't ignore system memory pressure**
   ```python
   # ❌ Only checking CUDA allocator misses the full picture
   if torch.cuda.memory_allocated() < threshold:
       # This can still OOM if system memory is low!
   ```

### Examples: Correct Memory Reporting

#### Benchmark Results Table

**Before (misleading for Jetson):**
```csv
model,gpu_memory_mb,system_memory_mb,total_mb
edgetam,3500,6200,9700  ❌ Wrong total!
```

**After (correct):**
```csv
model,system_memory_mb,cuda_alloc_mb,notes
edgetam,6200,3500,"Unified: CUDA is subset of system"
```

#### Console Output

**Before:**
```
GPU Peak: 3.5 GB
System Peak: 6.2 GB
Total: 9.7 GB  ❌ Misleading!
```

**After (Jetson):**
```
System Memory Peak: 6.2 GB
  ├─ CUDA Allocator: 3.5 GB (subset)
  └─ Other/OS: 2.7 GB
Architecture: Unified Memory (Jetson Orin)
```

**After (Desktop):**
```
GPU VRAM Peak: 8.4 GB
System RAM Peak: 12.3 GB
Total: 20.7 GB
Architecture: Discrete GPU
```

### Testing Memory Detection

```python
# test_unified_memory.py
from sav_benchmark.utils import is_jetson, get_memory_info

print("Platform Detection")
print("=" * 50)
print(f"Is Jetson (unified memory): {is_jetson()}")
print()

info = get_memory_info()
print("Memory Info")
print("=" * 50)
for key, value in info.items():
    if isinstance(value, int) and value > 1024:
        print(f"{key}: {value / 1e9:.2f} GB")
    else:
        print(f"{key}: {value}")
print()

if info["is_unified"]:
    print("⚠️  Unified Memory Detected!")
    print("Remember: GPU memory is PART OF system memory")
    print(f"System total: {info['system_total'] / 1e9:.2f} GB (shared pool)")
    if info['gpu_alloc']:
        ratio = info['gpu_alloc'] / info['system_used']
        print(f"CUDA allocator: {info['gpu_alloc'] / 1e9:.2f} GB "
              f"({ratio*100:.1f}% of system used)")
```

### Benchmarking Best Practices

#### Memory-Aware Batch Processing

```python
def process_video_with_memory_check(video_path, model):
    """Process video with automatic memory pressure detection."""
    mem_info = get_memory_info()
    is_unified = mem_info["is_unified"]
    
    # Different strategies for unified vs discrete
    if is_unified:
        # On Jetson: Monitor total system memory
        available = psutil.virtual_memory().available
        threshold = psutil.virtual_memory().total * 0.2  # Keep 20% free
        
        if available < threshold:
            print("⚠️  Memory pressure high, triggering cleanup")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.1)  # Let OS reclaim
    else:
        # On desktop: Monitor GPU VRAM separately
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        if allocated > reserved * 0.9:
            print("GPU VRAM pressure high")
            torch.cuda.empty_cache()
    
    # Proceed with inference
    return model.process(video_path)
```

### Summary

The key insight for Jetson Orin benchmarking:

> **GPU memory and system memory are THE SAME physical RAM, just viewed from different perspectives (CUDA allocator vs OS). Always use system memory as the primary metric for actual memory consumption.**

This fundamentally changes:
- How we report memory usage (don't add them!)
- DataLoader configuration (pin_memory=False)
- Memory pressure detection (check system, not just CUDA)
- Optimization strategies (no PCIe transfer to optimize)

All code has been updated to detect unified memory and handle it appropriately.

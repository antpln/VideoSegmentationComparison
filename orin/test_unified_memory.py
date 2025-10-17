#!/usr/bin/env python3
"""
Test script to verify unified memory detection and reporting.
Run this to understand your platform's memory architecture.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_platform_detection():
    """Test if platform is correctly detected."""
    print("=" * 70)
    print("PLATFORM DETECTION TEST")
    print("=" * 70)
    
    from sav_benchmark.utils import is_jetson
    
    is_unified = is_jetson()
    print(f"\nUnified Memory Architecture: {is_unified}")
    
    if is_unified:
        print("✓ Detected: Jetson/Tegra platform")
        print("  - GPU and CPU share same physical RAM")
        print("  - pin_memory should be False")
        print("  - GPU memory is subset of system memory")
    else:
        print("✓ Detected: Discrete GPU platform")
        print("  - GPU has separate VRAM")
        print("  - pin_memory can be beneficial")
        print("  - GPU and system memory are additive")
    
    # Try to detect specific details
    import os
    if os.path.exists("/etc/nv_tegra_release"):
        print("\n📝 Tegra release info:")
        with open("/etc/nv_tegra_release", "r") as f:
            print("  ", f.read().strip())
    
    if os.path.exists("/sys/firmware/devicetree/base/model"):
        print("\n📝 Device model:")
        try:
            with open("/sys/firmware/devicetree/base/model", "r") as f:
                print("  ", f.read().strip())
        except Exception as e:
            print(f"   Could not read: {e}")
    
    print()


def test_memory_info():
    """Test memory information reporting."""
    print("=" * 70)
    print("MEMORY INFORMATION TEST")
    print("=" * 70)
    
    try:
        from sav_benchmark.utils import get_memory_info
        
        info = get_memory_info()
        
        print(f"\nArchitecture: {'Unified' if info['is_unified'] else 'Discrete'}")
        print()
        
        # System memory
        if info['system_total']:
            print(f"System Memory:")
            print(f"  Total: {info['system_total'] / 1e9:.2f} GB")
            print(f"  Used:  {info['system_used'] / 1e9:.2f} GB")
            print(f"  Free:  {(info['system_total'] - info['system_used']) / 1e9:.2f} GB")
        
        # GPU memory
        if info['gpu_alloc'] is not None:
            print(f"\nCUDA Allocator:")
            print(f"  Allocated: {info['gpu_alloc'] / 1e9:.2f} GB")
            print(f"  Reserved:  {info['gpu_reserved'] / 1e9:.2f} GB")
            
            if info['is_unified']:
                ratio = (info['gpu_alloc'] / info['system_used']) * 100
                print(f"\n💡 CUDA allocated is {ratio:.1f}% of system used")
                print(f"   (Both measured from same physical RAM)")
        else:
            print("\nCUDA: Not available")
        
        if info['note']:
            print(f"\n📝 Note: {info['note']}")
        
        print()
        
    except ImportError as e:
        print(f"\n❌ Could not import memory utilities: {e}")
        print("   Make sure psutil is installed: pip install psutil")
    except Exception as e:
        print(f"\n❌ Error getting memory info: {e}")


def test_torch_memory():
    """Test PyTorch CUDA memory tracking."""
    print("=" * 70)
    print("PYTORCH CUDA TEST")
    print("=" * 70)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("\n❌ CUDA not available")
            return
        
        device_name = torch.cuda.get_device_name(0)
        print(f"\nCUDA Device: {device_name}")
        
        # Get memory info
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory
        print(f"CUDA Reports: {total_memory / 1e9:.2f} GB total")
        
        # Check if this matches system memory (unified) or not
        import psutil
        system_total = psutil.virtual_memory().total
        
        if abs(total_memory - system_total) < 1e8:  # Within 100MB
            print("✓ Matches system RAM → Unified Memory")
        else:
            print(f"System RAM: {system_total / 1e9:.2f} GB")
            print("✓ Different from system RAM → Discrete GPU")
        
        # Small allocation test
        print("\nSmall allocation test...")
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(1000, 1000, device='cuda')
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        print(f"  Allocated: {alloc / 1e6:.2f} MB")
        print(f"  Reserved:  {reserved / 1e6:.2f} MB")
        
        del x
        torch.cuda.empty_cache()
        print("  ✓ Cleanup successful")
        print()
        
    except ImportError:
        print("\n❌ PyTorch not installed")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_recommendations():
    """Print configuration recommendations based on platform."""
    print("=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)
    
    from sav_benchmark.utils import is_jetson
    
    is_unified = is_jetson()
    
    print("\nDataLoader Settings:")
    if is_unified:
        print("  pin_memory = False      # ✓ CRITICAL for unified memory")
        print("  num_workers = 0         # ✓ Best for batch_size=1")
        print("  non_blocking = False    # ✓ No PCIe transfers")
    else:
        print("  pin_memory = True       # ✓ Speeds up PCIe transfers")
        print("  num_workers = 2-4       # ✓ Can help throughput")
        print("  non_blocking = True     # ✓ Async GPU transfers")
    
    print("\nMemory Monitoring:")
    if is_unified:
        print("  Primary:   psutil.memory_info().used")
        print("  Secondary: torch.cuda.memory_allocated()")
        print("  ⚠️  Don't add them together (double counting)!")
    else:
        print("  GPU:    torch.cuda.memory_allocated()")
        print("  System: psutil.memory_info().used")
        print("  Total:  Add both (separate pools)")
    
    print("\nBenchmark Script:")
    if is_unified:
        print("  Recommended: ./run_jetson_benchmark.sh")
        print("  See: JETSON_BENCHMARK_GUIDE.md")
        print("       UNIFIED_MEMORY_GUIDE.md")
    else:
        print("  Recommended: python sam_comparison.py")
        print("  See: README.md")
    
    print()


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  UNIFIED MEMORY DETECTION & CONFIGURATION TEST".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    
    test_platform_detection()
    test_memory_info()
    test_torch_memory()
    test_recommendations()
    
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("For more information:")
    print("  - Unified memory guide: UNIFIED_MEMORY_GUIDE.md")
    print("  - Jetson benchmarking:  JETSON_BENCHMARK_GUIDE.md")
    print("  - Quick reference:      JETSON_QUICK_REFERENCE.md")
    print()


if __name__ == "__main__":
    main()

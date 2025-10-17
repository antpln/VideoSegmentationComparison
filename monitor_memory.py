#!/usr/bin/env python3
"""
Memory monitoring utility for Jetson benchmarks.
Tracks GPU and system memory usage in real-time.

Note: Jetson devices use unified memory architecture where GPU and CPU
share the same physical RAM. On these devices, GPU memory is a subset of
system memory, not additive.
"""

import argparse
import os
import time
import subprocess
import sys
from pathlib import Path
from typing import Optional


def is_jetson() -> bool:
    """Detect if running on Jetson platform (unified memory)."""
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    if os.path.exists("/sys/firmware/devicetree/base/model"):
        try:
            with open("/sys/firmware/devicetree/base/model", "r") as f:
                if "jetson" in f.read().lower():
                    return True
        except Exception:
            pass
    return False


def get_gpu_memory() -> Optional[tuple[float, float]]:
    """
    Get current GPU memory usage (used, total) in GB.
    Returns None if unable to query.
    
    Note: On Jetson (unified memory), this shows CUDA allocator's view
    of the shared memory pool, not separate GPU RAM.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        used, total = map(float, result.stdout.strip().split(","))
        return used / 1024.0, total / 1024.0  # Convert MB to GB
    except Exception:
        return None


def get_system_memory() -> Optional[tuple[float, float]]:
    """
    Get current system memory usage (used, total) in GB.
    Returns None if unable to query.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        
        mem_total = None
        mem_available = None
        
        for line in lines:
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) / (1024 * 1024)  # kB to GB
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1]) / (1024 * 1024)  # kB to GB
        
        if mem_total and mem_available:
            mem_used = mem_total - mem_available
            return mem_used, mem_total
        return None
    except Exception:
        return None


def monitor_memory(interval: float = 1.0, log_file: Optional[Path] = None):
    """
    Monitor memory usage at specified interval.
    
    Args:
        interval: Seconds between measurements
        log_file: Optional file to log measurements
    """
    print("=" * 80)
    print("Memory Monitor")
    print("=" * 80)
    print("Press Ctrl+C to stop")
    print()
    
    # Detect unified memory architecture
    unified = is_jetson()
    if unified:
        print("⚠️  UNIFIED MEMORY detected (Jetson platform)")
        print("   GPU and system RAM share the same physical memory pool")
        print("   GPU usage is a subset of system usage, not additive")
        print()
    
    # Check if we can query GPU
    gpu_info = get_gpu_memory()
    has_gpu = gpu_info is not None
    
    if has_gpu:
        gpu_label = "CUDA Pool" if unified else "GPU VRAM"
        print(f"{gpu_label}: {gpu_info[1]:.2f} GB total")
    else:
        print("GPU not available or nvidia-smi not found")
    
    sys_info = get_system_memory()
    if sys_info:
        print(f"System RAM: {sys_info[1]:.2f} GB total")
    
    if unified and has_gpu and sys_info:
        print(f"\n💡 On this system, the {gpu_info[1]:.2f} GB 'GPU' memory")
        print(f"   is part of the {sys_info[1]:.2f} GB system RAM (shared pool)")
    
    print()
    print("-" * 80)
    
    # Header
    if has_gpu:
        gpu_label = "CUDA" if unified else "GPU"
        header = f"{'Time':<12} {gpu_label + ' Used':>12} {gpu_label + ' %':>8} {'RAM Used':>12} {'RAM %':>8}"
    else:
        header = f"{'Time':<12} {'RAM Used':>12} {'RAM %':>8}"
    
    print(header)
    print("-" * 80)
    
    # Open log file if specified
    log_handle = None
    if log_file:
        log_handle = open(log_file, "w")
        log_handle.write("timestamp,gpu_used_gb,gpu_total_gb,gpu_percent,ram_used_gb,ram_total_gb,ram_percent\n")
    
    try:
        start_time = time.time()
        max_gpu_used = 0.0
        max_ram_used = 0.0
        
        while True:
            current_time = time.time() - start_time
            time_str = f"{current_time:.1f}s"
            
            # Get measurements
            gpu_data = get_gpu_memory() if has_gpu else None
            sys_data = get_system_memory()
            
            # Format output
            if gpu_data:
                gpu_used, gpu_total = gpu_data
                gpu_percent = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
                max_gpu_used = max(max_gpu_used, gpu_used)
                gpu_str = f"{gpu_used:>10.2f} GB {gpu_percent:>7.1f}%"
            else:
                gpu_used, gpu_total, gpu_percent = 0, 0, 0
                gpu_str = ""
            
            if sys_data:
                ram_used, ram_total = sys_data
                ram_percent = (ram_used / ram_total * 100) if ram_total > 0 else 0
                max_ram_used = max(max_ram_used, ram_used)
                ram_str = f"{ram_used:>10.2f} GB {ram_percent:>7.1f}%"
            else:
                ram_used, ram_total, ram_percent = 0, 0, 0
                ram_str = ""
            
            # Print to console
            if has_gpu:
                output = f"{time_str:<12} {gpu_str} {ram_str}"
            else:
                output = f"{time_str:<12} {ram_str}"
            print(output, flush=True)
            
            # Log to file
            if log_handle:
                log_handle.write(
                    f"{current_time:.2f},{gpu_used:.3f},{gpu_total:.3f},{gpu_percent:.2f},"
                    f"{ram_used:.3f},{ram_total:.3f},{ram_percent:.2f}\n"
                )
                log_handle.flush()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print()
        print("-" * 80)
        print("Monitoring stopped")
        print()
        print("Peak Usage:")
        if has_gpu and max_gpu_used > 0:
            print(f"  GPU: {max_gpu_used:.2f} GB")
        if max_ram_used > 0:
            print(f"  RAM: {max_ram_used:.2f} GB")
        print("=" * 80)
    
    finally:
        if log_handle:
            log_handle.close()
            print(f"Log saved to: {log_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Jetson memory usage")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Measurement interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path (CSV format)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_file = Path(args.log) if args.log else None
    
    try:
        monitor_memory(interval=args.interval, log_file=log_file)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

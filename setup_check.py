#!/usr/bin/env python3
"""
Setup verification script for Video Segmentation profiling experiments.
Checks dependencies, CUDA availability, and nsight installation.
"""

import subprocess
import sys
from pathlib import Path


def check_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_python_packages():
    """Verify required Python packages are installed."""
    check_section("Python Packages")
    
    required = {
        "torch": "PyTorch",
        "cv2": "OpenCV (opencv-python)",
        "numpy": "NumPy",
        "psutil": "psutil",
        "PIL": "Pillow",
    }
    
    optional = {
        "ultralytics": "Ultralytics (for SAM2)",
        "nvtx": "NVTX (for inline profiling)",
    }
    
    all_ok = True
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name:30s} [INSTALLED]")
        except ImportError:
            print(f"✗ {name:30s} [MISSING - REQUIRED]")
            all_ok = False
    
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✓ {name:30s} [INSTALLED]")
        except ImportError:
            print(f"○ {name:30s} [MISSING - OPTIONAL]")
    
    return all_ok


def check_cuda():
    """Check CUDA and GPU availability."""
    check_section("CUDA & GPU")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            device_count = torch.cuda.device_count()
            print(f"GPU Count: {device_count}")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
            
            # Check precision support
            try:
                if hasattr(torch.cuda, 'is_bf16_supported'):
                    bf16_supported = torch.cuda.is_bf16_supported()
                    print(f"BF16 Support: {bf16_supported}")
                    if not bf16_supported:
                        print("  → Use --autocast fp16 or --autocast none on this device")
            except Exception:
                pass
            
            return True
        else:
            print("⚠ No CUDA GPUs detected - CPU-only mode")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_nsight():
    """Check for nsight-systems installation."""
    check_section("NVIDIA Nsight Systems")
    
    # Try common paths
    nsys_paths = [
        "nsys",  # In PATH
        "/opt/nvidia/nsight-systems/bin/nsys",  # Jetson default
        "/usr/local/bin/nsys",  # Desktop Linux
    ]
    
    for nsys_path in nsys_paths:
        try:
            result = subprocess.run(
                [nsys_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print(f"✓ Nsight Systems found: {nsys_path}")
                print(f"  Version: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    print("✗ Nsight Systems not found")
    print("  Install from: https://developer.nvidia.com/nsight-systems")
    print("  Or on Jetson: included with JetPack 6.2")
    print("  Set NSYS_CMD environment variable if installed in custom location")
    return False


def check_weights():
    """Check for model weight files."""
    check_section("Model Weights")
    
    repo_root = Path(__file__).parent
    
    # SAM2 weights
    sam2_weights = {
        "sam2.1_t.pt": "SAM2 Tiny",
        "sam2.1_s.pt": "SAM2 Small",
        "sam2.1_b.pt": "SAM2 Base",
        "sam2.1_l.pt": "SAM2 Large",
    }
    
    sam2_found = 0
    for weight_file, name in sam2_weights.items():
        # Check in repo root and sam2 subdirectory
        found = False
        for search_path in [repo_root, repo_root / "sam2"]:
            if (search_path / weight_file).exists():
                print(f"✓ {name:20s} [{search_path / weight_file}]")
                sam2_found += 1
                found = True
                break
        if not found:
            print(f"○ {name:20s} [Not found]")
    
    # EdgeTAM weights
    edgetam_path = repo_root / "EdgeTAM" / "checkpoints" / "edgetam.pt"
    if edgetam_path.exists():
        print(f"✓ EdgeTAM              [{edgetam_path}]")
    else:
        print(f"○ EdgeTAM              [Not found - {edgetam_path}]")
    
    if sam2_found == 0:
        print("\n⚠ No SAM2 weights found. Download from:")
        print("  https://github.com/facebookresearch/sam2")
    
    return sam2_found > 0


def check_scripts():
    """Verify profiling scripts are executable."""
    check_section("Profiling Scripts")
    
    repo_root = Path(__file__).parent
    scripts = [
        "sam_gpu_profiles.py",
        "run_nsight_profile.sh",
        "profile_examples.sh",
    ]
    
    all_ok = True
    for script_name in scripts:
        script_path = repo_root / script_name
        if script_path.exists():
            if script_path.suffix == ".sh":
                is_executable = script_path.stat().st_mode & 0o111
                if is_executable:
                    print(f"✓ {script_name:30s} [OK]")
                else:
                    print(f"⚠ {script_name:30s} [Not executable - run: chmod +x {script_name}]")
                    all_ok = False
            else:
                print(f"✓ {script_name:30s} [OK]")
        else:
            print(f"✗ {script_name:30s} [MISSING]")
            all_ok = False
    
    return all_ok


def print_summary(checks: dict):
    """Print overall setup status."""
    check_section("Setup Summary")
    
    all_ok = all(checks.values())
    
    if all_ok:
        print("✓ All critical checks passed!")
        print("\nYou can now run profiling experiments:")
        print("  ./run_nsight_profile.sh --test_mode")
    else:
        print("⚠ Some checks failed. Review the output above.")
        
        if not checks["packages"]:
            print("\nInstall Python packages:")
            print("  pip3 install -r requirements.txt")
        
        if not checks["cuda"]:
            print("\nCUDA not available - install PyTorch with CUDA support")
        
        if not checks["nsight"]:
            print("\nNsight Systems required for GPU profiling")
        
        if not checks["weights"]:
            print("\nDownload model weights before running experiments")


def main():
    """Run all setup checks."""
    print("Video Segmentation Profiling - Setup Verification")
    print(f"Python: {sys.version.split()[0]}")
    
    checks = {
        "packages": check_python_packages(),
        "cuda": check_cuda(),
        "nsight": check_nsight(),
        "weights": check_weights(),
        "scripts": check_scripts(),
    }
    
    print_summary(checks)
    
    # Exit with error code if critical checks failed
    sys.exit(0 if checks["packages"] and checks["scripts"] else 1)


if __name__ == "__main__":
    main()

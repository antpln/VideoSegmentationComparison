# NSIGHT Profiling Experiment Workflow

Complete walkthrough for running GPU profiling experiments on RTX 3090 and Jetson Orin, collecting results, and merging data for analysis.

---

## Quick Reference

### Test Run (Verify Setup)
```bash
./run_test.sh
```

### Full Experiment
```bash
./run_full_experiment.sh --split_dir /path/to/sav_val
```

### Collect Results from Remote
```bash
./collect_results.sh nvidia@jetson-orin /data/sav_val/nsight_experiment_jetson_*
```

### Merge Multiple Results
```bash
python3 merge_results.py collected_results/nsight_experiment_* -o final_comparison.csv
```

---

## Detailed Walkthrough

### Step 1: Test Run (Both Platforms)

**Purpose**: Verify nsight profiling is working before running expensive full experiments.

**On RTX 3090 Desktop:**
```bash
cd /path/to/VideoSegementationComparison
git checkout nsight

# Test with synthetic data (no dataset needed)
./run_test.sh
```

**On Jetson Orin:**
```bash
cd /path/to/VideoSegementationComparison
git checkout nsight

# Set nsys path for Jetson
export NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys

# Test
./run_test.sh
```

**Expected Output:**
```
==========================================
  NSIGHT PROFILING - TEST RUN
==========================================

Platform: Desktop (assuming RTX 3090 or similar)
Nsight Systems: nsys
Default precision: bf16
Default model: sam2_base_points

Running setup verification...
==========================================
  Python Packages
==========================================
✓ PyTorch                       [INSTALLED]
✓ OpenCV (opencv-python)        [INSTALLED]
...

==========================================
  TEST RUN COMPLETE!
==========================================

Output files:
  - test_profile_outputs/profile_summary.csv
  - test_profile_outputs/profile_stats.json
  - test_profile_outputs/nsight_reports/*.nsys-rep

✓ Setup verified! Ready for full experiment runs.
```

**What to Check:**
```bash
# View CSV results
cat test_profile_outputs/profile_summary.csv

# View JSON statistics
cat test_profile_outputs/profile_stats.json | python3 -m json.tool

# Check nsight reports exist
ls -lh test_profile_outputs/nsight_reports/
```

---

### Step 2: Full Experiment - RTX 3090

**On your RTX 3090 desktop/server:**

```bash
# Full experiment with default settings
./run_full_experiment.sh --split_dir /path/to/sav_val

# Or customize:
./run_full_experiment.sh \
    --split_dir /path/to/sav_val \
    --videos 5 \
    --objects 2 \
    --runs 10 \
    --warmup 3
```

**What it does:**
- Automatically selects platform-appropriate models (sam2_tiny through sam2_base)
- Uses BF16 precision
- Profiles 3 videos × 2 objects × 5 runs (default)
- Creates timestamped output directory

**Expected Output Directory:**
```
/path/to/sav_val/nsight_experiment_desktop_20251130_143022/
├── experiment_config.txt                 # Configuration log
├── profile_outputs/
│   ├── profile_summary.csv              # Main results CSV ★
│   └── profile_stats.json               # Aggregated statistics
└── nsight_reports/
    ├── video001_obj0001_sam2_base_points_run0.nsys-rep
    ├── video001_obj0001_sam2_base_points_run1.nsys-rep
    └── ...
```

**Runtime Estimate:**
- 3 videos × 2 objects × 6 models × 5 runs × ~30s = ~1.5 hours

---

### Step 3: Full Experiment - Jetson Orin

**On Jetson Orin:**

```bash
# Set nsys path
export NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys

# Run experiment (uses FP16 and smaller models automatically)
./run_full_experiment.sh --split_dir /data/sav_val

# Or customize:
./run_full_experiment.sh \
    --split_dir /data/sav_val \
    --videos 3 \
    --objects 1 \
    --runs 5
```

**What it does:**
- Automatically detects Jetson platform
- Uses FP16 precision (BF16 not supported)
- Profiles sam2_tiny and sam2_small only (resource-constrained)
- Creates timestamped output directory

**Expected Output Directory:**
```
/data/sav_val/nsight_experiment_jetson_20251130_150000/
├── experiment_config.txt
├── profile_outputs/
│   ├── profile_summary.csv              # Main results CSV ★
│   └── profile_stats.json
└── nsight_reports/
    └── *.nsys-rep
```

**Runtime Estimate:**
- 3 videos × 1 object × 4 models × 5 runs × ~120s = ~2 hours

---

### Step 4: Collect Results from Remote Machines

**From your local desktop, collect Jetson results:**

```bash
# List available experiment directories on Jetson
ssh nvidia@jetson-orin "ls -ld /data/sav_val/nsight_experiment_*"

# Collect specific experiment
./collect_results.sh \
    nvidia@jetson-orin \
    /data/sav_val/nsight_experiment_jetson_20251130_150000

# Results will be in:
# ./collected_results/nsight_experiment_jetson_20251130_150000/
```

**What it does:**
1. Creates compressed archive on remote machine
2. Downloads via SCP
3. Extracts locally to `./collected_results/`
4. Cleans up remote archive

**If you ran the desktop experiment remotely, collect it too:**
```bash
./collect_results.sh \
    user@workstation \
    /data/sav_val/nsight_experiment_desktop_20251130_143022
```

---

### Step 5: Merge Results for Analysis

**Combine results from multiple platforms:**

```bash
# Merge Jetson and Desktop results
python3 merge_results.py \
    collected_results/nsight_experiment_jetson_20251130_150000 \
    collected_results/nsight_experiment_desktop_20251130_143022 \
    -o final_comparison.csv
```

**Output:**
```
Merging results from:
  - collected_results/nsight_experiment_jetson_20251130_150000
    Found 60 entries from jetson
  - collected_results/nsight_experiment_desktop_20251130_143022
    Found 180 entries from desktop

Merged CSV written to: final_comparison.csv
Total entries: 240

Entries by platform:
  jetson: 60
  desktop: 180
```

**The merged CSV includes:**
- `platform`: "jetson" or "desktop"
- `hostname`: Machine identifier
- `timestamp`: Experiment timestamp
- All original metrics (fps, latency_ms, gpu_peak_alloc_MiB, etc.)

---

## Output File Locations

### Test Run Output
```
./test_profile_outputs/
├── profile_summary.csv              # Per-run metrics
├── profile_stats.json               # Statistics per model
└── nsight_reports/
    └── *.nsys-rep                   # Nsight binary reports
```

### Full Experiment Output (on each platform)
```
<dataset_dir>/nsight_experiment_<platform>_<timestamp>/
├── experiment_config.txt            # Experiment metadata
├── profile_outputs/
│   ├── profile_summary.csv          # ★ Main CSV for plotting
│   └── profile_stats.json           # ★ Aggregated statistics
└── nsight_reports/
    └── *.nsys-rep                   # Nsight reports (optional)
```

### Collected Results (on local machine)
```
./collected_results/
├── nsight_experiment_jetson_20251130_150000/
│   └── profile_outputs/
│       └── profile_summary.csv      # Jetson results
└── nsight_experiment_desktop_20251130_143022/
    └── profile_outputs/
        └── profile_summary.csv      # Desktop results
```

### Merged Results (ready for plotting)
```
./final_comparison.csv               # ★★ All platforms merged
```

---

## CSV Format for Plotting

**`profile_summary.csv` columns:**

| Column | Description | Units |
|--------|-------------|-------|
| `platform` | "jetson" or "desktop" (after merge) | - |
| `hostname` | Machine identifier | - |
| `video` | Video ID | - |
| `object` | Object ID | - |
| `model` | Model tag (e.g., sam2_base_points) | - |
| `run` | Run number (0 to N-1) | - |
| `imgsz` | Input image size | pixels |
| `frames` | Number of frames processed | count |
| `fps` | Frames per second | FPS |
| `latency_ms` | Per-frame latency | milliseconds |
| `gpu_peak_alloc_MiB` | Peak GPU allocated memory | MiB |
| `gpu_peak_reserved_MiB` | Peak GPU reserved memory | MiB |
| `cpu_peak_rss_MiB` | Peak CPU memory | MiB |
| `setup_ms` | Model setup time | milliseconds |
| `H`, `W` | Original frame dimensions | pixels |
| `infer_H`, `infer_W` | Inference dimensions | pixels |
| `nsight_report` | Path to nsight report | path |

**`profile_stats.json` format:**
```json
{
  "sam2_base_points": {
    "fps": {
      "mean": 12.34,
      "std": 0.56,
      "min": 11.50,
      "max": 13.20,
      "median": 12.30
    },
    "latency_ms": { ... },
    "gpu_peak_alloc_MiB": { ... }
  }
}
```

---

## Example Plotting (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load merged results
df = pd.read_csv('final_comparison.csv')

# Compare FPS across platforms
grouped = df.groupby(['platform', 'model'])['fps'].mean().unstack()
grouped.plot(kind='bar', figsize=(12, 6))
plt.ylabel('FPS (higher is better)')
plt.title('Model Performance Comparison: Jetson vs Desktop')
plt.tight_layout()
plt.savefig('fps_comparison.png')

# Memory usage comparison
grouped_mem = df.groupby(['platform', 'model'])['gpu_peak_alloc_MiB'].mean().unstack()
grouped_mem.plot(kind='bar', figsize=(12, 6))
plt.ylabel('GPU Memory (MiB)')
plt.title('GPU Memory Usage: Jetson vs Desktop')
plt.tight_layout()
plt.savefig('memory_comparison.png')
```

---

## Troubleshooting

### Test run fails
```bash
# Check setup
python3 setup_check.py

# Check nsys is found
which nsys
# Or on Jetson:
/opt/nvidia/nsight-systems/bin/nsys --version
```

### Out of memory on Jetson
```bash
# Use fewer videos/objects
./run_full_experiment.sh --split_dir /data/sav_val --videos 1 --objects 1

# Or edit run_full_experiment.sh to use only sam2_tiny
```

### SCP fails
```bash
# Test SSH connection
ssh nvidia@jetson-orin "hostname"

# Manually copy if needed
scp -r nvidia@jetson-orin:/data/sav_val/nsight_experiment_jetson_* ./collected_results/
```

### Merge results fails
```bash
# Check directories exist
ls -la collected_results/

# Check CSV files exist
ls collected_results/*/profile_outputs/profile_summary.csv

# Run with verbose output
python3 -u merge_results.py collected_results/nsight_experiment_* -o merged.csv
```

---

## Complete Example Session

```bash
# ============================================
# On RTX 3090 Desktop
# ============================================
cd /path/to/VideoSegementationComparison
git checkout nsight

# Test
./run_test.sh

# Full experiment
./run_full_experiment.sh --split_dir /data/sav_val --runs 10

# Note the output directory name, e.g.:
# /data/sav_val/nsight_experiment_desktop_20251130_143022


# ============================================
# On Jetson Orin
# ============================================
cd /path/to/VideoSegementationComparison
git checkout nsight
export NSYS_CMD=/opt/nvidia/nsight-systems/bin/nsys

# Test
./run_test.sh

# Full experiment
./run_full_experiment.sh --split_dir /data/sav_val --runs 10

# Note the output directory name, e.g.:
# /data/sav_val/nsight_experiment_jetson_20251130_150000


# ============================================
# On Local Desktop (Data Collection)
# ============================================

# Collect from desktop (if remote)
./collect_results.sh \
    user@workstation \
    /data/sav_val/nsight_experiment_desktop_20251130_143022

# Collect from Jetson
./collect_results.sh \
    nvidia@jetson-orin \
    /data/sav_val/nsight_experiment_jetson_20251130_150000

# Merge for analysis
python3 merge_results.py \
    collected_results/nsight_experiment_* \
    -o comparison_3090_vs_jetson.csv

# Quick stats
python3 -c "
import pandas as pd
df = pd.read_csv('comparison_3090_vs_jetson.csv')
print(df.groupby(['platform', 'model'])['fps'].describe())
"
```

---

## Summary

✅ **Test Run**: `./run_test.sh` (synthetic data, quick verification)

✅ **Full Experiment**: `./run_full_experiment.sh --split_dir /path/to/sav_val`

✅ **Output Location**: `<split_dir>/nsight_experiment_<platform>_<timestamp>/profile_outputs/`

✅ **Key Files**: 
- `profile_summary.csv` - Per-run detailed metrics
- `profile_stats.json` - Aggregated statistics

✅ **Collection**: `./collect_results.sh <host> <remote_path>`

✅ **Merging**: `python3 merge_results.py <dir1> <dir2> ... -o merged.csv`

✅ **Ready for Plotting**: Merged CSV has all metrics with platform labels

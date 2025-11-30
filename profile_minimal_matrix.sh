#!/bin/bash
# Minimal profiling matrix runner
# Runs a tiny Nsight-profiled workload for each specified model on the current machine
# Does NOT touch real dataset directories. Uses --test_mode (synthetic data).

set -euo pipefail

# Default models to profile (comma-separated)
MODELS=${MODELS:-"sam2_tiny_points,sam2_small_points,sam2_base_points,edgetam_points"}

# Number of profiled runs per model
PROFILE_RUNS=${PROFILE_RUNS:-2}
# Warmup runs
WARMUP_RUNS=${WARMUP_RUNS:-1}

# Nsight command (override with environment variable if needed)
NSYS_CMD=${NSYS_CMD:-nsys}

# Output dir for minimal profiles
OUT_ROOT=${OUT_ROOT:-./minimal_profile_outputs}
mkdir -p "${OUT_ROOT}"

IFS=',' read -r -a MODEL_ARRAY <<< "$MODELS"

echo "Running minimal profiling matrix"
echo "Models: ${MODELS}"
echo "Profile runs: ${PROFILE_RUNS}, Warmup: ${WARMUP_RUNS}"
echo "Output root: ${OUT_ROOT}"

echo "Checking nsys availability: ${NSYS_CMD}"
if ! command -v "${NSYS_CMD}" &> /dev/null; then
    echo "ERROR: ${NSYS_CMD} not found. Set NSYS_CMD or install Nsight Systems."
    exit 1
fi

# Loop through models and run the profilings
for model in "${MODEL_ARRAY[@]}"; do
    model_tag=$(echo "$model" | tr -d '[:space:]')
    out_dir="${OUT_ROOT}/${model_tag}"
    mkdir -p "${out_dir}/nsight_reports" "${out_dir}/profile_outputs"

    echo "\n=== Profiling model: ${model_tag} ==="
    echo "Output: ${out_dir}"

    # Quick syntax/import sanity check before launching Nsight to avoid
    # producing large profiler artifacts when the Python process fails at import.
    if ! python3 -m py_compile ./sam_gpu_profiles.py ./sav_benchmark/main.py; then
        echo "ERROR: Python syntax check failed for sam_gpu_profiles.py or sav_benchmark/main.py. Skipping ${model_tag}."
        continue
    fi

    # Run nsight wrapper in test mode (synthetic data) to avoid touching dataset
    # Use run_nsight_profile.sh which will create nsight reports and CSV in out_dir
    NSYS_OUTPUT_DIR="${out_dir}/nsight_reports" \
    ./run_nsight_profile.sh \
        --test_mode \
        --models "${model_tag}" \
        --profile_mode external \
        --profile_runs "${PROFILE_RUNS}" \
        --warmup_runs "${WARMUP_RUNS}" \
        --out_dir "${out_dir}/profile_outputs"

    echo "Saved reports to: ${out_dir}"
done

echo "\nMinimal profiling matrix complete. Results in: ${OUT_ROOT}"
#!/bin/bash
# Data collection helper - copies experiment results from remote machines
# Usage: ./collect_results.sh <remote_host> <remote_path>

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <remote_host> <remote_path_to_experiment_dir>"
    echo ""
    echo "Examples:"
    echo "  # Collect from Jetson"
    echo "  $0 nvidia@jetson-orin /data/sav_val/nsight_experiment_jetson_20251130_143022"
    echo ""
    echo "  # Collect from remote desktop"
    echo "  $0 user@workstation /data/sav_val/nsight_experiment_desktop_20251130_150000"
    echo ""
    echo "This will:"
    echo "  1. Create a compressed archive on the remote machine"
    echo "  2. Copy it to local ./collected_results/"
    echo "  3. Extract it automatically"
    echo "  4. Clean up the remote archive"
    exit 1
fi

REMOTE_HOST="$1"
REMOTE_PATH="$2"
EXPERIMENT_NAME=$(basename "${REMOTE_PATH}")
LOCAL_DIR="./collected_results"

mkdir -p "${LOCAL_DIR}"

echo "=========================================="
echo "  Collecting Experiment Results"
echo "=========================================="
echo ""
echo "Remote host: ${REMOTE_HOST}"
echo "Remote path: ${REMOTE_PATH}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Local directory: ${LOCAL_DIR}"
echo ""

# Check if remote path exists
echo "Checking remote path..."
ssh "${REMOTE_HOST}" "test -d ${REMOTE_PATH}" || {
    echo "ERROR: Remote path does not exist: ${REMOTE_PATH}"
    exit 1
}

# Create archive on remote
echo "Creating archive on remote machine..."
REMOTE_DIR=$(dirname "${REMOTE_PATH}")
ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && tar -czf ${EXPERIMENT_NAME}.tar.gz ${EXPERIMENT_NAME}"

# Copy to local
echo "Copying to local machine..."
scp "${REMOTE_HOST}:${REMOTE_DIR}/${EXPERIMENT_NAME}.tar.gz" "${LOCAL_DIR}/"

# Extract locally
echo "Extracting archive..."
cd "${LOCAL_DIR}"
tar -xzf "${EXPERIMENT_NAME}.tar.gz"
cd - > /dev/null

# Clean up remote archive
echo "Cleaning up remote archive..."
ssh "${REMOTE_HOST}" "rm ${REMOTE_DIR}/${EXPERIMENT_NAME}.tar.gz"

echo ""
echo "=========================================="
echo "  Collection Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${LOCAL_DIR}/${EXPERIMENT_NAME}"
echo ""
echo "Files collected:"
find "${LOCAL_DIR}/${EXPERIMENT_NAME}" -type f | head -20
echo ""
echo "CSV summary:"
echo "  ${LOCAL_DIR}/${EXPERIMENT_NAME}/profile_outputs/profile_summary.csv"
echo ""
echo "JSON statistics:"
echo "  ${LOCAL_DIR}/${EXPERIMENT_NAME}/profile_outputs/profile_stats.json"
echo ""
echo "Quick preview:"
echo "  head -10 ${LOCAL_DIR}/${EXPERIMENT_NAME}/profile_outputs/profile_summary.csv"
echo "  python3 -m json.tool ${LOCAL_DIR}/${EXPERIMENT_NAME}/profile_outputs/profile_stats.json"

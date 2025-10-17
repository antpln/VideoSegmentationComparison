#!/bin/bash
# Update SAM2 to latest version with optional CUDA extension
# This resolves the "_C import" warning and enables post-processing fallback

set -e

echo "==========================================="
echo "Updating SAM2 (optional CUDA extension)"
echo "==========================================="

# Detect SAM2 location
SAM2_REPO=""
if [ -d "EdgeTAM/sam2" ]; then
    SAM2_REPO="EdgeTAM/sam2"
    echo "Found SAM2 in EdgeTAM/sam2"
elif [ -d "../segment-anything-2" ]; then
    SAM2_REPO="../segment-anything-2"
    echo "Found SAM2 in ../segment-anything-2"
elif [ -d "segment-anything-2" ]; then
    SAM2_REPO="segment-anything-2"
    echo "Found SAM2 in segment-anything-2"
else
    echo "ERROR: Could not find SAM2 repository"
    echo "Expected locations:"
    echo "  - EdgeTAM/sam2"
    echo "  - ../segment-anything-2"
    echo "  - segment-anything-2"
    exit 1
fi

cd "$SAM2_REPO"

echo ""
echo "Pulling latest SAM2 code..."
git pull

echo ""
echo "Uninstalling old SAM-2..."
pip uninstall -y SAM-2 || echo "(SAM-2 was not installed)"

echo ""
echo "Removing old CUDA extensions..."
rm -f sam2/*.so

echo ""
echo "Reinstalling SAM-2 with optional CUDA extension..."
pip install -e ".[demo]"

echo ""
echo "==========================================="
echo "SAM2 update complete!"
echo "==========================================="
echo ""
echo "The CUDA extension (_C) is now optional."
echo "Post-processing will use Python fallback (same results in most cases)."
echo ""

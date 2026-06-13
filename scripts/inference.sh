#!/bin/bash
# 3D Volume Compression Inference Script
#
# Usage: ./inference.sh <test_data_path>
#   test_data_path: Path to the test data directory or file
#
# Model: small - 75M params, channels=[96,192,384,320], depths=[2,2,4,0]

# Check if test data path is provided
if [ -z "$1" ]; then
    echo "Error: Test data path is required"
    echo "Usage: ./inference.sh <test_data_path>"
    exit 1
fi

TEST_DATA_PATH="$1"

cd ..

# ==================== compression configure ====================
CHECKPOINT="/home/dullpigeon/Desktop/Research/ContextVolumeAE/Baselines/ours/scripts/best_model.pth"
OUTPUT_DIR="./output"
# Continuous gain factor (primary quality control; higher = better quality / larger file)
FACTOR=9.4

# ==================== inference ====================
echo "=========================================="
echo "Inference: gain-factor mode (factor=${FACTOR})"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Test data: ${TEST_DATA_PATH}"
echo "=========================================="

python infer.py \
    --checkpoint ${CHECKPOINT} \
    --mode compression \
    --quality_mode factor \
    --factor ${FACTOR} \
    --patch_size 128 128 128 \
    --data_dir ${TEST_DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --save_results


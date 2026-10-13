#!/bin/bash
# ============================================
# Extract FATURA Dataset
# ============================================
# Reference:
# - Paper: Limam, M., Dhiaf, M., & Kessentini, Y. (2023). FATURA: A Multi-Layout Invoice Image Dataset
# - Dataset: https://doi.org/10.5281/zenodo.10371464
# - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
#
# Note: FATURA.zip contains another FATURA.zip inside
#
# Usage:
#   ./01_download_and_extract.sh ./FATURA2.zip ./data-raw

set -e

if [ $# -ne 2 ]; then
    echo "Usage: ./01_download_and_extract.sh <zip_path> <output_dir>"
    echo "Example: ./01_download_and_extract.sh ./FATURA2.zip ./data-raw"
    echo ""
    echo "Download FATURA2.zip from:"
    echo "  https://zenodo.org/records/10371464/files/FATURA2.zip?download=1"
    exit 1
fi

ZIP_PATH="$1"
OUTPUT_DIR="$2"

echo "============================================"
echo "FATURA Dataset Extractor"
echo "============================================"
echo "Input:  $ZIP_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if zip exists
if [ ! -f "$ZIP_PATH" ]; then
    echo "Error: ZIP file not found: $ZIP_PATH"
    echo ""
    echo "Please download FATURA2.zip from:"
    echo "  https://zenodo.org/records/10371464/files/FATURA2.zip?download=1"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Extract outer zip to temp directory
TEMP_DIR="$OUTPUT_DIR/_temp_extract"
mkdir -p "$TEMP_DIR"

echo "Step 1: Extracting outer FATURA.zip..."
unzip -q "$ZIP_PATH" -d "$TEMP_DIR"

# Step 2: Check for inner FATURA.zip and extract
INNER_ZIP="$TEMP_DIR/FATURA.zip"
if [ -f "$INNER_ZIP" ]; then
    echo "Step 2: Extracting inner FATURA.zip..."
    unzip -q "$INNER_ZIP" -d "$OUTPUT_DIR"
else
    echo "No nested zip found, checking for direct dataset..."
    if [ -d "$TEMP_DIR/invoices_dataset_final" ]; then
        mv "$TEMP_DIR/invoices_dataset_final" "$OUTPUT_DIR/"
    else
        echo "Error: Could not find dataset in zip"
        ls -la "$TEMP_DIR"
        exit 1
    fi
fi

# Step 3: Cleanup
echo "Step 3: Cleaning up..."
rm -rf "$TEMP_DIR"

# Verify
DATASET_DIR="$OUTPUT_DIR/invoices_dataset_final"
echo ""
echo "Verifying dataset structure..."
echo "  $([ -d "$DATASET_DIR/images" ] && echo '✓' || echo '✗') images/"
echo "  $([ -d "$DATASET_DIR/Annotations" ] && echo '✓' || echo '✗') Annotations/"
echo "  $([ -f "$DATASET_DIR/strat1_train.csv" ] && echo '✓' || echo '✗') strat1_train.csv"
echo "  $([ -f "$DATASET_DIR/strat1_dev.csv" ] && echo '✓' || echo '✗') strat1_dev.csv"
echo "  $([ -f "$DATASET_DIR/strat1_test.csv" ] && echo '✓' || echo '✗') strat1_test.csv"

# Count images
IMAGE_COUNT=$(find "$DATASET_DIR/images" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
echo ""
echo "  Total images: $IMAGE_COUNT"

echo ""
echo "============================================"
echo "Done! Dataset extracted to: $DATASET_DIR"
echo "============================================"
echo ""
echo "Next step: python 02_create_dataset.py --input-dir $OUTPUT_DIR"

#!/usr/bin/env bash
# ===========================================================================
# run_pipeline.sh — Reproduce OSI/DSI Neuropixels v4 tables
#
# Run from the repository root:
#   bash run_pipeline.sh
#
# Required input files (must already be present):
#   cortical_metrics_1.4.csv
#   functional_connectivity_2Hz.csv
# ===========================================================================
set -euo pipefail

echo "============================================================"
echo "  OSI/DSI Neuropixels v4 Pipeline"
echo "============================================================"

# --- Sanity-check inputs ---------------------------------------------------
for f in cortical_metrics_1.4.csv functional_connectivity_2Hz.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required input file '$f' not found in $(pwd)."
        echo "       Please see README.md for instructions."
        exit 1
    fi
done

# --- Step 1: Build V1 (VISp) v3 table -------------------------------------
echo ""
echo "[Step 1/6] Generating OSI_DSI_neuropixels_v3.csv  (V1 / VISp) ..."
python misc/convert_to_metrics.py \
    --area VISp \
    --metrics cortical_metrics_1.4.csv \
    --output OSI_DSI_neuropixels_v3.csv

# --- Step 2: Build LM (VISl) v3 table -------------------------------------
echo ""
echo "[Step 2/6] Generating OSI_DSI_neuropixels_LM_v3.csv  (LM / VISl) ..."
python misc/convert_to_metrics.py \
    --area VISl \
    --metrics cortical_metrics_1.4.csv \
    --output OSI_DSI_neuropixels_LM_v3.csv

# --- Step 3: Merge FC data with cell types ---------------------------------
echo ""
echo "[Step 3/6] Merging functional-connectivity data with cell types ..."
python misc/merge_fc_with_cell_types.py
# produces: functional_connectivity_2Hz_with_cell_types.csv

# --- Step 4: Build V1 v4 table --------------------------------------------
echo ""
echo "[Step 4/6] Generating OSI_DSI_neuropixels_v4.csv  (V1 / VISp) ..."
python misc/combine_osi_dsi_analysis.py \
    --v3  OSI_DSI_neuropixels_v3.csv \
    --fc  functional_connectivity_2Hz_with_cell_types.csv \
    --out OSI_DSI_neuropixels_v4.csv \
    --area VISp

# --- Step 5: Copy to explicit V1 name -------------------------------------
echo ""
echo "[Step 5/6] Copying to OSI_DSI_neuropixels_V1_v4.csv ..."
cp OSI_DSI_neuropixels_v4.csv OSI_DSI_neuropixels_V1_v4.csv

# --- Step 6: Build LM v4 table --------------------------------------------
echo ""
echo "[Step 6/6] Generating OSI_DSI_neuropixels_LM_v4.csv  (LM / VISl) ..."
python misc/combine_osi_dsi_analysis.py \
    --v3  OSI_DSI_neuropixels_LM_v3.csv \
    --fc  functional_connectivity_2Hz_with_cell_types.csv \
    --out OSI_DSI_neuropixels_LM_v4.csv \
    --area VISl

# --- Done ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline complete.  Output files:"
echo "============================================================"
ls -lh OSI_DSI_neuropixels_v4.csv \
       OSI_DSI_neuropixels_V1_v4.csv \
       OSI_DSI_neuropixels_LM_v4.csv 2>/dev/null

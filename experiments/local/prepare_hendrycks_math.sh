#!/bin/bash
# =============================================================================
# Prepare EleutherAI/hendrycks_math for GRPO/SDPO
#
# Pipeline: load_dataset -> split_tasks -> preprocess.
# Uses all subject subsets (algebra, geometry, etc.) and both train/test splits.
#
# Usage: bash experiments/local/prepare_hendrycks_math.sh
# =============================================================================
set -euo pipefail

SDPO_DIR=/scratch/jiasi_root/jiasi0/zijianh/SDPO
cd "$SDPO_DIR"
export PYTHONPATH="${SDPO_DIR}:${PYTHONPATH:-}"

DATA_DIR="${SDPO_DIR}/datasets/hendrycks_math"
mkdir -p "$DATA_DIR"

echo "=== 1. Load EleutherAI/hendrycks_math (all subsets) and export JSON ==="
python -m data.load_dataset \
  --dataset_name EleutherAI/hendrycks_math \
  --output_path "${DATA_DIR}/hendrycks_math.json"

echo "=== 2. Split into train/test ==="
python -m data.split_tasks \
  --json_path "${DATA_DIR}/hendrycks_math.json" \
  --output_dir "$DATA_DIR" \
  --test_ratio 0.1 \
  --seed 42

echo "=== 3. Preprocess (build reward_model, chat prompt, etc.) and write parquet ==="
python -m data.preprocess --data_source "$DATA_DIR"

echo "=== Done. Use datasets/hendrycks_math for TASK and run a GRPO/SDPO script. ==="


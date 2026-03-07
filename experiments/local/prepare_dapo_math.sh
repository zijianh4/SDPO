#!/bin/bash
# =============================================================================
# Prepare dapo_math dataset for GRPO/SDPO (subset of lasgroup/verifiable-corpus)
#
# Same pipeline as other datasets: load_dataset -> split_tasks -> preprocess.
# Produces train.parquet and test.parquet with reward_model, chat-format prompt, etc.
#
# Usage: bash experiments/local/prepare_dapo_math.sh
# =============================================================================
set -euo pipefail

SDPO_DIR=/scratch/jiasi_root/jiasi0/zijianh/SDPO
cd "$SDPO_DIR"
export PYTHONPATH="${SDPO_DIR}:${PYTHONPATH:-}"

DATA_DIR="${SDPO_DIR}/datasets/dapo_math"
mkdir -p "$DATA_DIR"

echo "=== 1. Load dapo_math from lasgroup/verifiable-corpus and export JSON ==="
python -m data.load_dataset \
  --dataset_name dapo_math \
  --output_path "${DATA_DIR}/dapo_math.json"

echo "=== 2. Split into train/test ==="
python -m data.split_tasks \
  --json_path "${DATA_DIR}/dapo_math.json" \
  --output_dir "$DATA_DIR" \
  --test_ratio 0.1 \
  --seed 42

echo "=== 3. Preprocess (build reward_model, chat prompt, etc.) and write parquet ==="
python -m data.preprocess --data_source "$DATA_DIR"

echo "=== Done. Use datasets/dapo_math for TASK and run: ==="
echo "  bash experiments/local/run_grpo_dapo_math.sh"
echo "  bash experiments/local/run_sdpo_dapo_math.sh"

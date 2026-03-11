#!/bin/bash
# =============================================================================
# Prepare openai/gsm8k for GRPO/SDPO
#
# Uses the original HuggingFace train/test splits (no random split).
# Pipeline: load_dataset (train) -> load_dataset (test) -> preprocess.
#
# Usage: bash experiments/local/prepare_gsm8k.sh
# =============================================================================
set -euo pipefail

SDPO_DIR=/scratch/jiasi_root/jiasi0/zijianh/SDPO
cd "$SDPO_DIR"
export PYTHONPATH="${SDPO_DIR}:${PYTHONPATH:-}"

DATA_DIR="${SDPO_DIR}/datasets/gsm8k"
mkdir -p "$DATA_DIR"

echo "=== 1. Load openai/gsm8k train split and export JSON ==="
python -m data.load_dataset \
  --dataset_name openai/gsm8k \
  --split train \
  --output_path "${DATA_DIR}/train.json"

echo "=== 2. Load openai/gsm8k test split and export JSON ==="
python -m data.load_dataset \
  --dataset_name openai/gsm8k \
  --split test \
  --output_path "${DATA_DIR}/test.json"

echo "=== 3. Preprocess (build reward_model, chat prompt, etc.) and write parquet ==="
python -m data.preprocess --data_source "$DATA_DIR"

echo "=== Done. Use datasets/gsm8k for TASK and run a GRPO/SDPO script. ==="

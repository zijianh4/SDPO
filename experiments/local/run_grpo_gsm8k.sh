#!/bin/bash
# =============================================================================
# GRPO on openai/gsm8k
#
# Prerequisite: Prepare data first:
#   bash experiments/local/prepare_gsm8k.sh
#
# Usage:
#   bash experiments/local/run_grpo_gsm8k.sh
#   bash experiments/local/run_grpo_gsm8k.sh --dry-run
#   MODEL=Qwen/Qwen3-0.6B bash experiments/local/run_grpo_gsm8k.sh
# =============================================================================
set -euo pipefail

DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# =============================================================================
# ENVIRONMENT
# =============================================================================
SDPO_DIR=/scratch/jiasi_root/jiasi0/zijianh/SDPO

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SDPO_DIR}:${PYTHONPATH:-}"
ulimit -c 0

export WANDB_ENTITY="${WANDB_ENTITY:-}"

# =============================================================================
# CONFIGURATION — openai/gsm8k
# =============================================================================
CONFIG_NAME="baseline_grpo"

MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
)

DATA_PATHS=(
    "datasets/gsm8k"
)

LRS=(1e-5 1e-6)
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
MINI_BATCH_SIZES=(8 32)
ENABLE_THINKING=${ENABLE_THINKING:-false}

if [[ -n "${MODEL:-}" ]]; then MODELS=("$MODEL"); fi
if [[ -n "${DATA:-}" ]];  then DATA_PATHS=("$DATA"); fi
if [[ -n "${LR:-}" ]];    then LRS=("$LR"); fi

# =============================================================================
# RUN LOOP
# =============================================================================
for MODEL_PATH in "${MODELS[@]}"; do
    for DATA_PATH in "${DATA_PATHS[@]}"; do
        for LR in "${LRS[@]}"; do
            for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do

                MODEL_SHORT=$(echo "$MODEL_PATH" | tr '/' '-')
                DATA_SHORT=$(basename "$DATA_PATH")
                EXP_NAME="GRPO-${MODEL_SHORT}-${DATA_SHORT}-mbs${MINI_BATCH_SIZE}-lr${LR}"

                export EXPERIMENT="$EXP_NAME"
                export TASK="$DATA_PATH"

                CMD=(
                    python -m verl.trainer.main_ppo
                    --config-name "$CONFIG_NAME"
                    actor_rollout_ref.model.path="$MODEL_PATH"
                    data.train_batch_size="$TRAIN_BATCH_SIZE"
                    actor_rollout_ref.rollout.n="$ROLLOUT_N"
                    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE"
                    actor_rollout_ref.actor.optim.lr="$LR"
                    actor_rollout_ref.actor.optim.lr_warmup_steps=10
                    algorithm.rollout_correction.rollout_is=token
                    actor_rollout_ref.rollout.val_kwargs.n=4
                    "data.apply_chat_template_kwargs={enable_thinking: ${ENABLE_THINKING}}"
                    trainer.group_name=GRPO-gsm8k
                    "${EXTRA_ARGS[@]}"
                )

                echo "================================================================"
                echo "Experiment : $EXP_NAME"
                echo "Model      : $MODEL_PATH"
                echo "Dataset    : $DATA_PATH (openai/gsm8k)"
                echo "LR         : $LR"
                echo "Mini-batch : $MINI_BATCH_SIZE"
                echo "================================================================"

                if $DRY_RUN; then
                    echo "[dry-run] ${CMD[*]}"
                    echo ""
                else
                    "${CMD[@]}"
                fi

            done
        done
    done
done

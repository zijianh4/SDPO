#!/bin/bash
# =============================================================================
# SDPO on EleutherAI/hendrycks_math
#
# Prerequisite: Prepare data first:
#   bash experiments/local/prepare_hendrycks_math.sh
#
# Usage:
#   bash experiments/local/run_sdpo_hendrycks_math.sh
#   bash experiments/local/run_sdpo_hendrycks_math.sh --dry-run
#   MODEL=Qwen/Qwen3-0.6B bash experiments/local/run_sdpo_hendrycks_math.sh
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

export RAY_GRPC_KEEPALIVE_TIMEOUT_MS=${RAY_GRPC_KEEPALIVE_TIMEOUT_MS:-600000}
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# =============================================================================
# CONFIGURATION — EleutherAI/hendrycks_math
# =============================================================================
CONFIG_NAME="sdpo"

MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
)

DATA_PATHS=(
    "datasets/hendrycks_math"
)

LRS=(1e-5)
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
MINI_BATCH_SIZE=32
ALPHA=0.5
DISTILLATION_TOPK=100
DONT_REPROMPT_ON_SELF_SUCCESS=True

if [[ -n "${MODEL:-}" ]]; then MODELS=("$MODEL"); fi
if [[ -n "${DATA:-}" ]];  then DATA_PATHS=("$DATA"); fi
if [[ -n "${LR:-}" ]];    then LRS=("$LR"); fi

# =============================================================================
# RUN LOOP
# =============================================================================
for MODEL_PATH in "${MODELS[@]}"; do
    for DATA_PATH in "${DATA_PATHS[@]}"; do
        for LR in "${LRS[@]}"; do

            MODEL_SHORT=$(echo "$MODEL_PATH" | tr '/' '-')
            DATA_SHORT=$(basename "$DATA_PATH")
            EXP_NAME="SDPO-${MODEL_SHORT}-${DATA_SHORT}-alpha${ALPHA}-lr${LR}"

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
                actor_rollout_ref.actor.self_distillation.distillation_topk="$DISTILLATION_TOPK"
                actor_rollout_ref.actor.self_distillation.alpha="$ALPHA"
                actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success="$DONT_REPROMPT_ON_SELF_SUCCESS"
                actor_rollout_ref.actor.self_distillation.include_environment_feedback=False
                algorithm.rollout_correction.rollout_is=token
                actor_rollout_ref.rollout.val_kwargs.n=4
                trainer.test_freq=5
                trainer.group_name=SDPO-hendrycks_math
                "${EXTRA_ARGS[@]}"
            )

            echo "================================================================"
            echo "Experiment : $EXP_NAME"
            echo "Model      : $MODEL_PATH"
            echo "Dataset    : $DATA_PATH (EleutherAI/hendrycks_math)"
            echo "LR         : $LR"
            echo "Alpha      : $ALPHA"
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


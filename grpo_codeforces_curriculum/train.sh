#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${CODEFORCES_CURRICULUM_DATA_DIR:-${SCRIPT_DIR}/data}"
MODEL_PATH="${CODEFORCES_CURRICULUM_MODEL_PATH:-Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start}"
CHECKPOINT_DIR="${CODEFORCES_CURRICULUM_CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints}"
MODEL_CACHE_DIR="${HF_HOME:-${SCRIPT_DIR}/model-cache}"
export HF_HOME="${MODEL_CACHE_DIR}"

train_files="['${DATA_DIR}/codeforces_train_easy_to_hard.parquet']"
validation_files="['${DATA_DIR}/codeforces_validation_easy_to_hard.parquet']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_files}" \
    data.val_files="${validation_files}" \
    data.shuffle=False \
    data.validation_shuffle=False \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path="${SCRIPT_DIR}/codeforces_reward_function.py" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=naive \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name=verl_grpo_afterburner \
    trainer.experiment_name=codeforces_easy_to_hard \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 "$@"

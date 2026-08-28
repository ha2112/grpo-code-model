#!/usr/bin/env bash
set -euo pipefail
set -x

# Override these for a cluster-specific data/model location.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${AFTERBURNER_DATA_DIR:-${HOME}/data/venus}"
MODEL_PATH="${AFTERBURNER_MODEL_PATH:-Elfsong/Qwen2.5-Coder-3B-Venus-Cold-Start}"
HF_HOME="${HF_HOME:-${SCRIPT_DIR}/../model-cache}"
export HF_HOME

# If you are using vllm<=0.6.3, you might need this to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

train_files="['${DATA_DIR}/venus_train.parquet']"
test_files="['${DATA_DIR}/venus_test.parquet']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path="${SCRIPT_DIR}/afterburner_reward_function.py" \
    custom_reward_function.name=afterburner_reward_fn_batch \
    reward_model.reward_manager=batch \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=163840 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name=verl_grpo_afterburner \
    trainer.experiment_name=qwen2_3b_venus \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=200 "$@"

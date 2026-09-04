#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the original 3B Venus cold-start policy.
export PROBE_CURRICULUM_POLICY_MODEL="${PROBE_CURRICULUM_POLICY_MODEL:-Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start}"

# Train the BF16 policy with LoRA and keep only the rollout copy in FP8.
exec bash "${SCRIPT_DIR}/train.sh" \
    trainer.use_v1=True \
    trainer.n_gpus_per_node=1 \
    trainer.logger=console \
    data.train_batch_size=1 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    reward.num_workers=1 \
    reward.custom_reward_function.path="${SCRIPT_DIR}/codeforces_reward_function.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.quantization=fp8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    "$@"

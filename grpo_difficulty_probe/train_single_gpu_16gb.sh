#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Keep the original 3B Venus cold-start policy.
export PROBE_CURRICULUM_POLICY_MODEL="${PROBE_CURRICULUM_POLICY_MODEL:-Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start}"
export PROBE_CURRICULUM_CHECKPOINT_DIR="${PROBE_CURRICULUM_CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints_single_gpu_16gb}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Do not spend minutes loading the model when correctness rewards cannot run.
if [[ "${CODEFORCES_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/codeforces_reward_function.py" --check
fi

extra_args=()
if [[ "${1:-}" == "--smoke" ]]; then
    shift
    # A smoke test checks generation/reward/update without the expensive final
    # validation and full FSDP checkpoint materialization.
    extra_args+=(
        trainer.total_training_steps=1
        trainer.save_freq=-1
        trainer.test_freq=-1
        trainer.resume_mode=disable
    )
fi

# Train the BF16 policy with LoRA and keep only the rollout copy in FP8.
exec bash "${SCRIPT_DIR}/train.sh" \
    trainer.use_v1=True \
    trainer.n_gpus_per_node=1 \
    trainer.logger=console \
    data.train_batch_size=1 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
    reward.num_workers=1 \
    reward.custom_reward_function.path="${SCRIPT_DIR}/codeforces_reward_function.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.lora.merge=True \
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
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.max_model_len=1792 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1792 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.rollout_data_dir="${SCRIPT_DIR}/rollouts" \
    "${extra_args[@]}" \
    "$@"

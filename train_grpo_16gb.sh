#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${GRPO_MODEL_PATH:-Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

route="${1:-}"
if [[ -z "${route}" ]]; then
    echo "Usage: $0 {venus|absolute|probe|venus-probe} [--smoke] [verl overrides...]" >&2
    exit 2
fi
shift

if ! "${PYTHON_BIN}" -c 'import verl' >/dev/null 2>&1; then
    echo "verl is not importable by ${PYTHON_BIN}." >&2
    echo "Set PYTHON_BIN to the Python executable in your verl environment." >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c 'import bitsandbytes' >/dev/null 2>&1; then
    echo "bitsandbytes is required for the 16 GB rollout copy." >&2
    echo "Install it with: uv pip install --python ${PYTHON_BIN} 'bitsandbytes>=0.45.3'" >&2
    exit 1
fi

case "${route}" in
    venus)
        route_dir="${REPO_DIR}/grpo"
        train_script="${route_dir}/afterburner_train.sh"
        data_dir="${AFTERBURNER_DATA_DIR:-${HOME}/data/venus}"
        train_file="${data_dir}/venus_train.parquet"
        val_file="${data_dir}/venus_test.parquet"
        reward_file="${route_dir}/afterburner_reward_function.py"
        reward_name=afterburner_reward_fn_batch
        reward_manager=batch
        checkpoint_dir="${AFTERBURNER_CHECKPOINT_DIR:-${route_dir}/checkpoints_single_gpu_16gb}"
        export AFTERBURNER_DATA_DIR="${data_dir}"
        export AFTERBURNER_MODEL_PATH="${MODEL_PATH}"
        ;;
    absolute)
        route_dir="${REPO_DIR}/grpo_codeforces_curriculum"
        train_script="${route_dir}/train.sh"
        data_dir="${CODEFORCES_CURRICULUM_DATA_DIR:-${route_dir}/data}"
        train_file="${data_dir}/codeforces_train_easy_to_hard.parquet"
        val_file="${data_dir}/codeforces_validation_easy_to_hard.parquet"
        reward_file="${route_dir}/codeforces_reward_function.py"
        reward_name=compute_score
        reward_manager=naive
        checkpoint_dir="${CODEFORCES_CURRICULUM_CHECKPOINT_DIR:-${route_dir}/checkpoints_single_gpu_16gb}"
        export CODEFORCES_CURRICULUM_DATA_DIR="${data_dir}"
        export CODEFORCES_CURRICULUM_MODEL_PATH="${MODEL_PATH}"
        ;;
    probe)
        route_dir="${REPO_DIR}/grpo_difficulty_probe"
        train_script="${route_dir}/train.sh"
        data_dir="${PROBE_CURRICULUM_DATA_DIR:-${route_dir}/data}"
        train_file="${data_dir}/probe_train_easy_to_hard.parquet"
        val_file="${data_dir}/probe_validation_easy_to_hard.parquet"
        reward_file="${route_dir}/codeforces_reward_function.py"
        reward_name=compute_score
        reward_manager=naive
        checkpoint_dir="${PROBE_CURRICULUM_CHECKPOINT_DIR:-${route_dir}/checkpoints_single_gpu_16gb}"
        export PROBE_CURRICULUM_DATA_DIR="${data_dir}"
        export PROBE_CURRICULUM_POLICY_MODEL="${MODEL_PATH}"
        ;;
    venus-probe)
        route_dir="${REPO_DIR}/grpo_venus_difficulty_probe"
        train_script="${route_dir}/train.sh"
        data_dir="${VENUS_PROBE_DATA_DIR:-${route_dir}/data}"
        train_file="${data_dir}/venus_probe_train_easy_to_hard.parquet"
        val_file="${data_dir}/venus_probe_test_easy_to_hard.parquet"
        reward_file="${REPO_DIR}/grpo/afterburner_reward_function.py"
        reward_name=afterburner_reward_fn_batch
        reward_manager=batch
        checkpoint_dir="${VENUS_PROBE_CHECKPOINT_DIR:-${route_dir}/checkpoints_single_gpu_16gb}"
        export VENUS_PROBE_DATA_DIR="${data_dir}"
        export VENUS_PROBE_POLICY_MODEL="${MODEL_PATH}"
        ;;
    *)
        echo "Unknown route: ${route}" >&2
        echo "Usage: $0 {venus|absolute|probe|venus-probe} [--smoke] [verl overrides...]" >&2
        exit 2
        ;;
esac

for dataset_file in "${train_file}" "${val_file}"; do
    if [[ ! -f "${dataset_file}" ]]; then
        echo "Missing prepared dataset: ${dataset_file}" >&2
        exit 1
    fi
done

# Catch an unavailable or incompatible judge before loading the 3B model.
if [[ "${GRPO_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    "${PYTHON_BIN}" "${reward_file}" --check
fi

smoke_args=()
if [[ "${1:-}" == "--smoke" ]]; then
    shift
    smoke_args+=(
        trainer.total_training_steps=1
        trainer.save_freq=-1
        trainer.test_freq=-1
        trainer.resume_mode=disable
    )
fi

mkdir -p "${checkpoint_dir}" "${route_dir}/rollouts"

# The actor stays BF16 and trainable through LoRA. Only vLLM's rollout copy is
# loaded in 4-bit, which is the configuration verified on one 16 GB GPU.
exec bash "${train_script}" \
    trainer.use_v1=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger=console \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${checkpoint_dir}" \
    trainer.save_freq=250 \
    trainer.test_freq=-1 \
    data.train_batch_size=1 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    reward.num_workers=1 \
    reward.custom_reward_function.path="${reward_file}" \
    reward.custom_reward_function.name="${reward_name}" \
    reward.reward_manager.name="${reward_manager}" \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.lora.merge=False \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.quantization=null \
    actor_rollout_ref.rollout.load_format=bitsandbytes \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=bitsandbytes \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.rollout_data_dir="${route_dir}/rollouts" \
    "${smoke_args[@]}" \
    "$@"

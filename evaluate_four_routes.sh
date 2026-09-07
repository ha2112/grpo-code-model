#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/verl/.venv/bin/python3}"
CHECKPOINT_ROOT="${EVAL_CHECKPOINT_ROOT:-${REPO_DIR}/full-checkpoints/deadline-600-steps-seed42-v1}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_DIR}/evaluation-results/step-600}"
MERGED_ROOT="${EVAL_MERGED_ROOT:-${REPO_DIR}/evaluation-models/step-600}"
BASE_MODEL="${EVAL_BASE_MODEL:-Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start}"
CODEFORCES_DATA="${EVAL_CODEFORCES_DATA:-${REPO_DIR}/grpo_codeforces_curriculum/data/codeforces_validation_easy_to_hard.parquet}"
VENUS_DATA="${EVAL_VENUS_DATA:-${HOME}/data/venus/venus_test.parquet}"
SAMPLES_PER_SUITE="${EVAL_SAMPLES_PER_SUITE:-30}"
NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-4}"
SEED="${EVAL_SEED:-42}"
MONOLITH_URL="${MONOLITH_URL:-http://127.0.0.1:8000/execute}"

export MONOLITH_URL
export TMPDIR="${TMPDIR:-${HOME}/ray-tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${HOME}/ray-tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

routes=(venus absolute probe venus-probe)

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

for data_file in "${CODEFORCES_DATA}" "${VENUS_DATA}"; do
    if [[ ! -s "${data_file}" ]]; then
        echo "Missing or empty evaluation corpus: ${data_file}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}" "${MERGED_ROOT}" "${TMPDIR}"

echo "Checking evaluation dependencies..."
"${PYTHON_BIN}" -c 'import bitsandbytes, datasets, peft, safetensors, torch, transformers, vllm'

echo "Checking Monolith..."
"${PYTHON_BIN}" "${REPO_DIR}/grpo/afterburner_reward_function.py" --check
"${PYTHON_BIN}" "${REPO_DIR}/grpo_difficulty_probe/codeforces_reward_function.py" --check

for route in "${routes[@]}"; do
    actor_dir="${CHECKPOINT_ROOT}/${route}/global_step_600/actor"
    merged_dir="${MERGED_ROOT}/${route}"
    adapter_config="${merged_dir}/lora_adapter/adapter_config.json"
    adapter_weights="${merged_dir}/lora_adapter/adapter_model.safetensors"

    if [[ ! -d "${actor_dir}" ]]; then
        echo "Missing actor checkpoint: ${actor_dir}" >&2
        exit 1
    fi

    if [[ -s "${adapter_config}" && -s "${adapter_weights}" ]]; then
        echo "[READY] ${route} adapter already consolidated"
        continue
    fi

    if [[ -e "${merged_dir}" && -n "$(find "${merged_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "Incomplete non-empty merge directory: ${merged_dir}" >&2
        echo "Move it aside, then rerun this command." >&2
        exit 1
    fi

    echo "Consolidating ${route} FSDP checkpoint..."
    "${PYTHON_BIN}" -m verl.model_merger merge \
        --backend fsdp \
        --use_cpu_initialization \
        --local_dir "${actor_dir}" \
        --target_dir "${merged_dir}"

    if [[ ! -s "${adapter_config}" || ! -s "${adapter_weights}" ]]; then
        echo "The merger did not produce a LoRA adapter for ${route}." >&2
        echo "Expected: ${merged_dir}/lora_adapter" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${REPO_DIR}/evaluation/evaluate_four_routes.py" \
    --base-model "${BASE_MODEL}" \
    --adapter-root "${MERGED_ROOT}" \
    --codeforces-data "${CODEFORCES_DATA}" \
    --venus-data "${VENUS_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples-per-suite "${SAMPLES_PER_SUITE}" \
    --num-generations "${NUM_GENERATIONS}" \
    --seed "${SEED}" \
    --monolith-url "${MONOLITH_URL}"

echo "Evaluation complete: ${OUTPUT_DIR}"

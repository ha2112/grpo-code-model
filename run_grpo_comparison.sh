#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${COMPARISON_ENV_FILE:-${REPO_DIR}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}. Copy .env.example to .env and edit it." >&2
    exit 1
fi

# The env file is trusted local configuration and is intentionally gitignored.
set -a
source "${ENV_FILE}"
set +a

route="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${route}" in
    venus|absolute|probe|all) ;;
    *)
        echo "Usage: $0 [venus|absolute|probe|all] [additional verl overrides...]" >&2
        exit 2
        ;;
esac

required_vars=(
    COMPARISON_MODEL
    COMPARISON_STEPS
    COMPARISON_SEED
    COMPARISON_LEARNING_RATE
    COMPARISON_TRAIN_BATCH_SIZE
    COMPARISON_ROLLOUT_N
    COMPARISON_MAX_PROMPT_LENGTH
    COMPARISON_MAX_RESPONSE_LENGTH
    COMPARISON_NGPUS
    COMPARISON_NNODES
    COMPARISON_PROJECT
    COMPARISON_PREFIX
    COMPARISON_LOGGER
    COMPARISON_RESUME_MODE
    COMPARISON_RUNS_DIR
    COMPARISON_MODEL_CACHE_DIR
    VENUS_DATA_DIR
    ABSOLUTE_DATA_DIR
    PROBE_DATA_DIR
    MONOLITH_URL
)

for variable_name in "${required_vars[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Missing required .env value: ${variable_name}" >&2
        exit 1
    fi
done

for integer_name in COMPARISON_STEPS COMPARISON_SEED COMPARISON_TRAIN_BATCH_SIZE COMPARISON_ROLLOUT_N COMPARISON_MAX_PROMPT_LENGTH COMPARISON_MAX_RESPONSE_LENGTH COMPARISON_NGPUS COMPARISON_NNODES; do
    if [[ ! "${!integer_name}" =~ ^[0-9]+$ ]]; then
        echo "${integer_name} must be a non-negative integer, got: ${!integer_name}" >&2
        exit 1
    fi
done

absolute_path() {
    local configured_path="$1"
    if [[ "${configured_path}" = /* ]]; then
        printf '%s\n' "${configured_path}"
    else
        printf '%s/%s\n' "${REPO_DIR}" "${configured_path#./}"
    fi
}

runs_dir="$(absolute_path "${COMPARISON_RUNS_DIR}")"
model_cache_dir="$(absolute_path "${COMPARISON_MODEL_CACHE_DIR}")"
venus_data_dir="$(absolute_path "${VENUS_DATA_DIR}")"
absolute_data_dir="$(absolute_path "${ABSOLUTE_DATA_DIR}")"
probe_data_dir="$(absolute_path "${PROBE_DATA_DIR}")"
comparison_model="${COMPARISON_MODEL}"
if [[ "${comparison_model}" = ./* || "${comparison_model}" = ../* ]]; then
    comparison_model="$(absolute_path "${comparison_model}")"
fi

validate_file() {
    local path="$1"
    if [[ "${COMPARISON_SKIP_DATA_VALIDATION:-false}" != true && "${COMPARISON_VALIDATE_DATA:-true}" = true && ! -f "${path}" ]]; then
        echo "Missing prepared dataset: ${path}" >&2
        exit 1
    fi
}

common_overrides=(
    "trainer.total_training_steps=${COMPARISON_STEPS}"
    "data.seed=${COMPARISON_SEED}"
    "data.train_batch_size=${COMPARISON_TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${COMPARISON_MAX_PROMPT_LENGTH}"
    "data.max_response_length=${COMPARISON_MAX_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.seed=${COMPARISON_SEED}"
    "actor_rollout_ref.rollout.n=${COMPARISON_ROLLOUT_N}"
    "actor_rollout_ref.actor.optim.lr=${COMPARISON_LEARNING_RATE}"
    "trainer.n_gpus_per_node=${COMPARISON_NGPUS}"
    "trainer.nnodes=${COMPARISON_NNODES}"
    "trainer.project_name=${COMPARISON_PROJECT}"
    "trainer.logger=${COMPARISON_LOGGER}"
    "trainer.resume_mode=${COMPARISON_RESUME_MODE}"
)

run_venus() {
    validate_file "${venus_data_dir}/venus_train.parquet"
    validate_file "${venus_data_dir}/venus_test.parquet"
    AFTERBURNER_MODEL_PATH="${comparison_model}" \
    AFTERBURNER_DATA_DIR="${venus_data_dir}" \
    HF_HOME="${model_cache_dir}" \
    bash "${REPO_DIR}/grpo/afterburner_train.sh" \
        "${common_overrides[@]}" \
        "trainer.default_local_dir=${runs_dir}/venus-seed-${COMPARISON_SEED}" \
        "trainer.experiment_name=${COMPARISON_PREFIX}-venus-seed-${COMPARISON_SEED}" \
        "$@"
}

run_absolute() {
    validate_file "${absolute_data_dir}/codeforces_train_easy_to_hard.parquet"
    validate_file "${absolute_data_dir}/codeforces_validation_easy_to_hard.parquet"
    CODEFORCES_CURRICULUM_MODEL_PATH="${comparison_model}" \
    CODEFORCES_CURRICULUM_DATA_DIR="${absolute_data_dir}" \
    CODEFORCES_CURRICULUM_CHECKPOINT_DIR="${runs_dir}/absolute-seed-${COMPARISON_SEED}" \
    HF_HOME="${model_cache_dir}" \
    MONOLITH_URL="${MONOLITH_URL}" \
    bash "${REPO_DIR}/grpo_codeforces_curriculum/train.sh" \
        "${common_overrides[@]}" \
        "trainer.experiment_name=${COMPARISON_PREFIX}-absolute-seed-${COMPARISON_SEED}" \
        "$@"
}

run_probe() {
    validate_file "${probe_data_dir}/probe_train_easy_to_hard.parquet"
    validate_file "${probe_data_dir}/probe_validation_easy_to_hard.parquet"
    PROBE_CURRICULUM_POLICY_MODEL="${comparison_model}" \
    PROBE_CURRICULUM_DATA_DIR="${probe_data_dir}" \
    PROBE_CURRICULUM_CHECKPOINT_DIR="${runs_dir}/probe-seed-${COMPARISON_SEED}" \
    HF_HOME="${model_cache_dir}" \
    MONOLITH_URL="${MONOLITH_URL}" \
    bash "${REPO_DIR}/grpo_difficulty_probe/train.sh" \
        "${common_overrides[@]}" \
        "trainer.experiment_name=${COMPARISON_PREFIX}-probe-seed-${COMPARISON_SEED}" \
        "$@"
}

case "${route}" in
    venus) run_venus "$@" ;;
    absolute) run_absolute "$@" ;;
    probe) run_probe "$@" ;;
    all)
        run_venus "$@"
        run_absolute "$@"
        run_probe "$@"
        ;;
esac

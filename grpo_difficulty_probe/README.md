# GRPO with a difficulty-probe curriculum

This route replaces absolute Codeforces-rating ordering with relative ordering
from the trained Qwen2.5-1.5B difficulty probe. Correctness remains the GRPO
reward; probe output controls only which problems are seen first.

## What is relative

The builder keeps every Codeforces problem, including rows where `cf_rating`
is zero. It predicts one difficulty score per problem, sorts by that score,
and writes `probe_rank` plus `probe_percentile`. verl trains with
`data.shuffle=False`, so examples are consumed from the probe's easiest end to
its hardest end.

`cf_rating` is retained in `extra_info` for later analysis but is never used
for filtering or curriculum order.

## 1. Build the probe curriculum

The defaults use the existing local artifacts:

- Base model: `../Difficulty Probing/model/qwen2.5-1.5B-instruct`
- Probe: `../Difficulty Probing/models/difficulty_probe_qwen2.5-1.5b-codecontests.pth`

```bash
cd grpo_difficulty_probe
python3 -m pip install -r requirements.txt
python3 probe_curriculum_dataset.py
```

For a small end-to-end probe smoke run:

```bash
python3 probe_curriculum_dataset.py --max-train 8 --max-validation 4
```

Probe scores are appended to `data/probe_scores.jsonl`, so interrupted runs
resume without rescoring completed problems. The existing embedding parquet
is not reused because it contains only rated training problems; this route
also includes unrated and validation problems.

Run the builder once more after pulling prompt changes. Cached probe scores
make this a fast parquet rewrite; the probe model does not run again:

```bash
"$HOME/verl/.venv/bin/python3" \
  "$HOME/grpo-code-model/grpo_difficulty_probe/probe_curriculum_dataset.py" \
  --device cuda:0
```

The generated verl datasets are:

- `data/probe_train_easy_to_hard.parquet`
- `data/probe_validation_easy_to_hard.parquet`

## 2. Train with GRPO

Install verl and provide a Monolith-compatible execution endpoint:

```bash
MONOLITH_URL=https://your-monolith.example/execute ./train.sh
```

Defaults and overrides:

- Policy: `Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start`; override with
  `PROBE_CURRICULUM_POLICY_MODEL`.
- Data directory: `./data`; override with `PROBE_CURRICULUM_DATA_DIR`.
- Checkpoints: `./checkpoints`; override with
  `PROBE_CURRICULUM_CHECKPOINT_DIR`.
- Sandbox endpoint: `https://monolith.cool/execute`; override with
  `MONOLITH_URL`.

The reward is 80% test-case pass ratio and 20% response-format compliance.
The probe score is deliberately not a reward: rewarding perceived difficulty
would not establish that generated code is correct.

### One 16 GB GPU

The normal launcher assumes eight GPUs. On one 16 GB NVIDIA GPU, use the
low-memory launcher instead. It retains the original 3B Venus cold-start
policy, trains LoRA adapters in BF16, and uses BitsAndBytes 4-bit only for the
duplicate vLLM rollout copy. Install the rollout dependency first:

```bash
uv pip install --python "$HOME/verl/.venv/bin/python3" "bitsandbytes>=0.45.3"
```

The launcher passes BitsAndBytes through vLLM's `engine_kwargs`. This avoids
verl's narrower top-level quantization allowlist while leaving the training
policy in BF16.

Then run:

```bash
PYTHON_BIN="$HOME/verl/.venv/bin/python3" \
  ./train_single_gpu_16gb.sh --smoke
```

The smoke run performs one generation/reward/update step but skips final
validation and checkpoint materialization, which are not needed to establish
that the training path works. It also checks the Monolith endpoint before
loading the model. Set `CODEFORCES_SKIP_PREFLIGHT=1` only when that check is
intentionally undesirable.

This launcher uses `checkpoints_single_gpu_16gb/`, so it will not auto-resume
the earlier zero-reward checkpoint under `checkpoints/`. Smoke mode also sets
`trainer.resume_mode=disable` explicitly.

Inspect the smoke metrics before starting the full run. At least one candidate
group should normally have reward variance, `response_length/clip_ratio`
should be below `1.0`, and `actor/grad_norm` should be greater than zero. The
generated responses are retained under `rollouts/` for diagnosis.

Start the full run with:

```bash
PYTHON_BIN="$HOME/verl/.venv/bin/python3" \
  ./train_single_gpu_16gb.sh
```

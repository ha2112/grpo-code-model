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

The generated verl datasets are:

- `data/probe_train_easy_to_hard.parquet`
- `data/probe_validation_easy_to_hard.parquet`

## 2. Train with GRPO

Install verl and provide a Monolith-compatible execution endpoint:

```bash
MONOLITH_URL=https://your-monolith.example/execute ./train.sh
```

Defaults and overrides:

- Policy: `Elfsong/Qwen2.5-Coder-3B-Venus-Cold-Start`; override with
  `PROBE_CURRICULUM_POLICY_MODEL`.
- Data directory: `./data`; override with `PROBE_CURRICULUM_DATA_DIR`.
- Checkpoints: `./checkpoints`; override with
  `PROBE_CURRICULUM_CHECKPOINT_DIR`.
- Sandbox endpoint: `https://monolith.cool/execute`; override with
  `MONOLITH_URL`.

The reward is 80% test-case pass ratio and 20% response-format compliance.
The probe score is deliberately not a reward: rewarding perceived difficulty
would not establish that generated code is correct.

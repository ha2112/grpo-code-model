# Venus GRPO with a difficulty-probe curriculum

This route keeps the original `Elfsong/Venus_Python` prompts, three efficiency
objectives, and Afterburner reward, but orders the corpus using the trained
Qwen2.5-1.5B difficulty probe.

Each source problem is scored once from `question_content`. Its time, memory,
and integral examples stay adjacent at the same `probe_rank`; problems are
written from lowest to highest probe score. The score controls order only and
is not part of the reward.

## 1. Build the corpus

The defaults use the existing local probe artifacts:

- Base model: `../Difficulty Probing/model/qwen2.5-1.5B-instruct`
- Probe: `../Difficulty Probing/models/difficulty_probe_qwen2.5-1.5b-codecontests.pth`

```bash
cd grpo_venus_difficulty_probe
python3 -m pip install -r requirements.txt
python3 venus_probe_dataset.py
```

For a small end-to-end smoke run:

```bash
python3 venus_probe_dataset.py --max-train 8 --max-test 4
```

Probe scores are appended to `data/probe_scores.jsonl`, so interrupted runs
resume without rescoring completed problems. Baseline solutions are selected
reproducibly with `--seed 42` by default.

The generated verl datasets are:

- `data/venus_probe_train_easy_to_hard.parquet`
- `data/venus_probe_test_easy_to_hard.parquet`

## 2. Train with GRPO

Install verl, provide a Monolith-compatible execution endpoint, and run:

```bash
MONOLITH_URL=https://your-monolith.example/execute ./train.sh
```

The launcher disables train and validation shuffling so parquet order is
preserved. Override the policy with `VENUS_PROBE_POLICY_MODEL`, the corpus
directory with `VENUS_PROBE_DATA_DIR`, and checkpoints with
`VENUS_PROBE_CHECKPOINT_DIR`.

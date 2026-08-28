# Codeforces GRPO curriculum

This is a separate Afterburner-style verl GRPO route for Codeforces problems.
It uses `deepmind/code_contests`, keeps only Codeforces rows with a real
positive `cf_rating`, and stores them in ascending rating order.

## 1. Prepare the curriculum

```bash
cd grpo_codeforces_curriculum
python3 -m pip install -r requirements.txt
python3 codeforces_dataset.py
```

The generated files are:

- `data/codeforces_train_easy_to_hard.parquet`
- `data/codeforces_validation_easy_to_hard.parquet`

Use `--max-train N` and `--max-validation N` for a small smoke-test dataset.
The cap is applied after sorting, so it selects the easiest N rated problems.

## 2. Train with GRPO

Install verl and make a Monolith-compatible execution endpoint available,
then run:

```bash
MONOLITH_URL=https://your-monolith.example/execute ./train.sh
```

`data.shuffle=False` is set deliberately. This is what makes verl consume the
parquet rows from low to high `cf_rating` instead of randomizing the dataset.

Defaults and overrides:

- Policy: `Elfsong/Qwen2.5-Coder-3B-Venus-Cold-Start`; override with
  `CODEFORCES_CURRICULUM_MODEL_PATH`.
- Data directory: `./data`; override with `CODEFORCES_CURRICULUM_DATA_DIR`.
- Checkpoints: `./checkpoints`; override with
  `CODEFORCES_CURRICULUM_CHECKPOINT_DIR`.
- Sandbox: `https://monolith.cool/execute`; override with `MONOLITH_URL`.

The reward is 80% test-case pass ratio and 20% response-format compliance.
CodeContests does not provide the baseline runtime and memory measurements
needed by the original Venus efficiency reward, so this route does not claim
an efficiency-improvement signal.

The sandbox judge compares whitespace-separated output tokens. Problems with
special judges or multiple valid outputs may need a dataset-specific checker.

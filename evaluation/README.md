# Four-route held-out evaluation

`evaluate_four_routes.sh` evaluates every step-600 LoRA checkpoint on the
same deterministic subset of two held-out suites:

- 30 Codeforces validation problems spread across the rating range.
- 30 Venus test tasks balanced across time, memory, and integral objectives.

Every model receives four samples per case with the training decoding settings
(temperature 0.7, top-p 0.95, and 1,024 generated tokens). The evaluator uses
the original model as a shared 4-bit vLLM base and applies one consolidated
LoRA adapter at a time. Monolith supplies execution-based rewards.

Run on the CUDA host after moving all checkpoints under the shared root:

```bash
cd "$HOME/grpo-code-model"
source "$HOME/verl/.venv/bin/activate"
ray stop --force

PYTHON_BIN="$HOME/verl/.venv/bin/python3" \
MONOLITH_URL="http://127.0.0.1:8000/execute" \
bash evaluate_four_routes.sh
```

The run is resumable at individual completion granularity. Repeating the same
command skips consolidated adapters and completed predictions. Consolidated
models are shared under `evaluation-models/step-600/`, so the smoke and final
runs do not duplicate them.

Outputs are stored in `evaluation-results/step-600/`:

- `adapter_manifest.json`: adapter ranks, sizes, and hashes.
- `sample_manifest.json`: exact held-out cases used by every route.
- `predictions.jsonl`: raw generations and execution metrics.
- `summary.csv` and `summary.json`: means and 95% bootstrap intervals.
- `abstract_table.csv`: compact cross-route results table.
- `comparison.svg`: dependency-free pass@1, pass@4, and format-rate figure.
- `comparison.png`: the same figure when matplotlib is installed.

For a faster pipeline smoke test, use four cases per suite. Use a different
output directory so the small sample is not mixed with the final results:

```bash
EVAL_SAMPLES_PER_SUITE=4 \
EVAL_OUTPUT_DIR="$HOME/grpo-code-model/evaluation-results/smoke" \
bash evaluate_four_routes.sh
```

The model merger follows verl's FSDP conversion interface and keeps the
resulting Hugging Face model directories in `evaluation-models/step-600/`. Do
not remove them until the evaluation is complete.

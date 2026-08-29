# GRPO pilot comparison

The shared pilot configuration lives in the gitignored `.env`. Commit-safe
defaults are documented in `.env.example`.

Run one route:

```bash
./run_grpo_comparison.sh venus
./run_grpo_comparison.sh absolute
./run_grpo_comparison.sh probe
```

Run all three sequentially:

```bash
./run_grpo_comparison.sh all
```

Any arguments after the route are forwarded to verl and take precedence over
the `.env` controls. For example:

```bash
./run_grpo_comparison.sh probe trainer.total_training_steps=10
```

Run the launcher from the GRPO conda environment, or set `PYTHON_BIN` to its
Python executable. The launcher checks that `verl` imports before starting a
job:

```bash
conda activate probing-difficulty-linear
./run_grpo_comparison.sh absolute
```

GRPO requires a Linux/CUDA environment with a compatible `verl` and vLLM
installation; the local Apple Silicon machine is suitable only for the MPS
model smoke test below.

## Exact-model Mac smoke test

The local smoke test loads the same checkpoint configured in `.env`, performs
one real optimizer step on a final-layer RMSNorm parameter, and verifies a
saved checkpoint round trip. It prefers MPS and falls back to CPU when MPS is
unavailable; use `--device mps` or `--device cpu` to force a device:

```bash
/opt/anaconda3/bin/python mac_smoke_finetune.py
```

It intentionally trains only one small parameter so the exact 3B model fits
within local memory; distributed GRPO behavior is still verified separately
by the route launcher tests.

For independent cluster jobs, submit the three single-route commands rather
than `all`. Each route receives the same model, seed, training-step budget,
learning rate, batch size, rollout count, token limits, GPU count, logger, and
resume policy.

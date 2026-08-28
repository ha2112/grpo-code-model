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

## Exact-model Mac smoke test

The local smoke test loads the same checkpoint configured in `.env`, performs
one real optimizer step on a final-layer RMSNorm parameter, and verifies a
saved checkpoint round trip:

```bash
/opt/anaconda3/bin/python mac_smoke_finetune.py
```

It requires native Apple Silicon MPS. It intentionally trains only one small
parameter so the exact 3B model fits within 16 GB unified memory; distributed
GRPO behavior is still verified separately by the route launcher tests.

For independent cluster jobs, submit the three single-route commands rather
than `all`. Each route receives the same model, seed, training-step budget,
learning rate, batch size, rollout count, token limits, GPU count, logger, and
resume policy.

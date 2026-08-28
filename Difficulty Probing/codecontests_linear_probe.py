"""
Run the pre-trained linear difficulty probe on Codeforces problems from the
CodeContests dataset.

The probe (a linear layer trained on Qwen2.5-7B-Instruct last-token hidden
states, DeepMath difficulty scale) maps the last-token embedding of each
problem description to a scalar difficulty score. Predictions are saved next
to the real Codeforces rating (cf_rating) so probe quality can be evaluated
via correlation.

Usage:
    python codecontests_linear_probe.py --device auto \
        --model /path/to/qwen2.5-7b-instruct \
        --probe_path models/difficulty_probe_qwen2.5.pth \
        --save_path data/results/codecontests_probe.csv

--device resolves like the training script: 'auto' picks cuda -> mps -> cpu.

The script is resume-friendly: rows already present in --save_path are
skipped, so an interrupted run can be continued by re-running the same command.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Probe architecture
# ---------------------------------------------------------------------------
# Linear probing: instead of fine-tuning the LLM, we freeze it and train a
# single linear layer to map its last-token hidden state to a scalar
# difficulty score. If the representation already encodes difficulty, this
# one layer learns a good mapping, and its learned weight vector can be
# inspected to understand what the model "knows" about difficulty.
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Linear layer: hidden_size -> 1 scalar (the predicted difficulty)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# CodeContests mixes problems from multiple sources (e.g. Codeforces, AtCoder,
# CodeChef), each tagged with an integer `source` label. 2 == Codeforces.
CODEFORCES_SOURCE = 2  # integer source label in code_contests


def resolve_device(device):
    """'auto' -> cuda if available, else mps (Apple Silicon), else cpu."""
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main(args):
    device = resolve_device(args.device)
    # fp16 on cuda/mps: extraction is memory-bandwidth-bound, so halving the
    # byte size roughly halves the run time. fp32 on CPU (same rule as the
    # training script).
    dtype = torch.float16 if device.startswith(('cuda', 'mps')) else torch.float32

    # ------------------------------------------------------------------
    # Load the pre-trained probe.
    # ------------------------------------------------------------------
    # The probe is the linear layer trained on Qwen2.5-7B-Instruct last-token
    # hidden states (see the training script). It maps embedding -> difficulty.
    # weights_only=False is needed because the checkpoint was saved as a full
    # nn.Module object (torch.save(model)), not a plain state_dict. The
    # training script saves the probe on CPU (probe.cpu() before torch.save),
    # so move it to the model's device explicitly — a CPU-weight x device-input
    # matmul would fail every row otherwise.
    linear_model = torch.load(args.probe_path, weights_only=False).to(device)
    linear_model.eval()  # probe has no dropout/batch-norm, but be explicit

    # ------------------------------------------------------------------
    # Load the base language model + tokenizer.
    # ------------------------------------------------------------------
    # The same model the probe was trained on must be used to extract hidden
    # states, otherwise the embedding distribution changes and the probe no
    # longer applies. The probe expects the last-token hidden state as input
    # (cast back to fp32 before forward, see the inference loop below).
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Load and filter the CodeContests dataset.
    # ------------------------------------------------------------------
    # Keep only Codeforces problems with a real rating. `difficulty` is a
    # per-source class label (0,1,2,3,4) that is NOT comparable across
    # sources; cf_rating is the Codeforces difficulty rating and is reliable
    # when available. Unrated Codeforces problems carry cf_rating == 0, so
    # filter those out too (p["cf_rating"] is falsy for 0).
    ds = load_dataset(args.data, split="train")
    records = [
        {"name": p["name"], "description": p["description"], "real_difficulty": p["cf_rating"]}
        for p in ds
        if p["source"] == CODEFORCES_SOURCE and p["cf_rating"] and p["cf_rating"] > 0
    ]
    # Sort by real difficulty so the log is deterministic and idx aligns
    # positionally with the CSV rows (needed for the resume logic below).
    dfd = pd.DataFrame(records).sort_values("real_difficulty").reset_index(drop=True)

    # Optional cap for quick smoke tests: sample (with a fixed seed for
    # reproducibility), then re-sort to keep the positional alignment.
    if args.max_samples > 0 and args.max_samples < len(dfd):
        dfd = dfd.sample(n=args.max_samples, random_state=42).sort_values("real_difficulty").reset_index(drop=True)

    print(f">>> {len(dfd)} Codeforces problems with ratings loaded.")

    # ------------------------------------------------------------------
    # Resume-friendly result log.
    # ------------------------------------------------------------------
    # If a previous run already wrote the CSV, load it and only fill in rows
    # with a missing prediction (pred_difficulty == NaN). Rows are aligned
    # positionally with dfd (both sorted by real_difficulty), like
    # deepmath_weighted_llm_emb.py. This is what makes interrupted runs
    # resumable: re-running skips the work already done.
    if os.path.exists(args.save_path):
        log = pd.read_csv(args.save_path)
        # Backfill missing columns so partial/old CSVs don't crash the loop
        for col in ["name", "pred_difficulty", "real_difficulty"]:
            if col not in log.columns:
                log[col] = [None] * len(log)
    else:
        # Fresh log: pre-fill names and real difficulties, predictions pending
        log = pd.DataFrame({
            "name": dfd["name"],
            "pred_difficulty": [None] * len(dfd),
            "real_difficulty": dfd["real_difficulty"],
        })

    # The CSV is rewritten after every row, so its directory must exist up
    # front (data/ is gitignored and may be missing on a fresh checkout).
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Inference loop.
    # ------------------------------------------------------------------
    # For each problem: build a chat prompt (system instruction to box the
    # answer + the problem statement), run the model, take the hidden state of
    # the LAST token, and feed it to the probe. We use last-token hidden
    # states because the probe was trained on exactly those.
    for idx, row in tqdm(dfd.iterrows(), total=len(dfd)):
        # Resume: skip problems already scored in a previous run
        if pd.notna(log.at[idx, "pred_difficulty"]):
            continue

        try:
            # Chat template: Qwen is an instruction model, so the problem
            # statement is sent as a user turn. The system prompt mirrors the
            # training-time setup so the model's behavior (and hence its
            # hidden states) matches what the probe was trained on.
            messages = [
                {'role': 'system', 'content': 'After solving the mathematical problem, place the final answer inside \\boxed{}'},
                {'role': 'user', 'content': row["description"]},
            ]
            # tokenize=False keeps it a string; add_generation_prompt=True
            # appends the assistant turn marker so the model "knows" it is
            # being asked to respond.
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Truncate at 4096 tokens to bound memory/time for long problems
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

            # Forward pass with hidden states; no_grad since we only read
            # activations (no backprop, no memory for gradients).
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Extract the LAST-token hidden state of the FINAL layer:
            #   hidden_states[-1]  -> [batch, seq_len, hidden_size]
            #   [:, -1, :]         -> last token position
            #   .squeeze(0)        -> drop batch dim -> [hidden_size]
            # This is the exact feature the probe was trained on.
            last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            last_token_emb = last_hidden[:, -1, :].squeeze(0)

            # Cast to fp32 before probing: the model runs in fp16 but the
            # probe weights were saved as fp32, so a dtype mismatch would
            # break the matmul. .item() detaches the 1-element tensor to a
            # plain Python float for CSV serialization.
            log.at[idx, "pred_difficulty"] = linear_model(last_token_emb.float()).item()
            log.at[idx, "real_difficulty"] = row["real_difficulty"]
            log.at[idx, "name"] = row["name"]
            # Save after EVERY row: cheap, and a crash loses at most one row.
            log.to_csv(args.save_path, index=False)

        except Exception as e:
            # Log the failure and move on rather than aborting the whole run;
            # the row stays NaN so it is retried on the next run.
            print(f"[Error] index {idx} failed: {e}")
            continue

    # ------------------------------------------------------------------
    # Evaluation summary.
    # ------------------------------------------------------------------
    # Probe scores live on the DeepMath scale (3.0-9.0), which is NOT the
    # Codeforces scale (800-3500), so comparing absolute error is meaningless.
    # Instead, evaluate the RANKING via correlation:
    #   - Pearson r: linear agreement between predicted and true difficulty
    #   - Spearman rho: rank-order agreement (computed as Pearson on ranks,
    #     avoiding a scipy dependency)
    # n > 2 required for a meaningful correlation.
    pred = pd.to_numeric(log["pred_difficulty"], errors="coerce")
    real = pd.to_numeric(log["real_difficulty"], errors="coerce")
    # Only pairs where BOTH values exist (drops failed/NaN rows)
    mask = pred.notna() & real.notna()
    if mask.sum() > 2:
        pearson = pred[mask].corr(real[mask])
        # Spearman = Pearson on ranks; avoids a scipy dependency
        spearman = pred[mask].rank().corr(real[mask].rank())
        print(f">>> Pearson r: {pearson:.4f} | Spearman rho: {spearman:.4f} | n={mask.sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Codeforces difficulty from CodeContests")
    parser.add_argument("--device", type=str, default='auto', help="Device: 'auto' (cuda -> mps -> cpu), or e.g. 'cuda:0', 'mps', 'cpu'")
    parser.add_argument("--model", type=str, default='/data/bowen/models/qwen2.5-7B-instruct', help="HuggingFace model name or local path")
    parser.add_argument("--probe_path", type=str, default='models/difficulty_probe_qwen2.5.pth', help="Pre-trained difficulty probe (linear layer) path")
    parser.add_argument("--data", type=str, default='deepmind/code_contests', help="CodeContests dataset name or local path")
    parser.add_argument("--save_path", type=str, default='data/results/codecontests_probe.csv', help="Result saved to CSV file path")
    parser.add_argument("--max_samples", type=int, default=-1, help="Cap on the number of problems to probe (-1 = all)")
    args = parser.parse_args()
    main(args)

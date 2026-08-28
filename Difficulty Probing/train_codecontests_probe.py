"""
Train a linear difficulty probe on Codeforces problems from the CodeContests
dataset, using Qwen2.5-1.5B-Instruct last-token hidden states.

The probe is a single linear layer mapping the model's last-token embedding of
a problem statement to a scalar difficulty score (the Codeforces rating,
~800-3500). It is the Qwen2.5-1.5B-Instruct counterpart of the existing 7B
probe (models/difficulty_probe_qwen2.5.pth), so that
codecontests_linear_probe.py can be run with the 1.5B model: same data
filtering (Codeforces problems with a real rating), same chat template, same
feature (last-token hidden state of the final layer).

All artifacts are saved inside this repo (models/, data/, model/ are
gitignored, so they live on disk but not in git):

    models/difficulty_probe_qwen2.5-1.5b-codecontests.pth   <- trained probe
    data/statistics/codecontests_emb_<model-tag>.parquet    <- cached embeddings
    model/qwen2.5-1.5B-instruct/                            <- the model itself,
                                                               downloaded here on
                                                               first run

Pipeline (same recipe as the original DeepMath probe, see README.md):
    1. Embed: run Qwen2.5-1.5B-Instruct on each problem, cache the last-token
       hidden state to a parquet (fp32, device-agnostic). Re-runs skip this
       step when the cache exists; delete the parquet to re-extract.
    2. Train: StandardScaler on the features AND the target (both fit on
       train+val only), then a linear layer with MSE on the z-scored
       cf_rating, Adam (lr 5e-4, weight_decay 2e-4), 80 epochs, batch 32,
       64/16/20 train/val/test split (seed 42, best val epoch kept). The
       target is z-scored because Adam moves every parameter by ~lr per step
       (m_hat/sqrt(v_hat) ~ sign(g) for a constant gradient), so a raw-scale
       target (mean ~1900) would leave the bias stuck near 0 and the
       predictions zero-centered instead of on the CF scale.
    3. Fold BOTH scalers into the probe weights so the saved probe consumes
       RAW embeddings and emits RAW cf_rating-scale scores — exactly what
       codecontests_linear_probe.py feeds it (it calls
       linear_model(last_token_emb.float()) with no scaler).

Usage:
    conda activate probing-difficulty-linear
    python train_codecontests_probe.py                     # everything defaults in-repo
    python train_codecontests_probe.py --device cuda:0     # GPU server
    python train_codecontests_probe.py --max_samples 50    # quick smoke test

--data accepts a local path if deepmind/code_contests is mirrored offline.

Monitoring from a phone (tmux):
    tmux new -s probe
    python train_codecontests_probe.py          # Ctrl-b d to detach
    # on your phone:  ssh your-mac 'tmux attach -t probe'
    # Stage banners + periodic progress lines show everything the run does.
    # (If you redirect stdout to a log instead, run python -u ... > run.log
    # and tail -f it — -u disables output buffering.)
"""

import os
import re
import copy
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# Probe architecture
# ---------------------------------------------------------------------------
# Linear probing: the LLM stays frozen and a single linear layer learns to map
# its last-token hidden state to a scalar difficulty score. If the
# representation already encodes difficulty, this one layer learns a good
# mapping and its learned weight vector can be inspected.
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

# Where the model is cached inside this repo (gitignored), and the HF repo id
# used to populate it on first run. NOTE: the org-qualified id is required —
# HF returns 401 "Repository Not Found" for the org-less name
# 'Qwen2.5-1.5B-Instruct' even though the model is public.
DEFAULT_MODEL_DIR = 'model/qwen2.5-1.5B-instruct'
DEFAULT_HF_ID = 'Qwen/Qwen2.5-1.5B-Instruct'

EPOCHS = 80
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 2e-4
SEED = 42


def banner(msg):
    """Full-width stage header so a tmux peek from a phone is instantly
    readable: what stage we are in, and when it started. flush=True so the
    line is visible immediately even if stdout is redirected to a log."""
    print('=' * 70, flush=True)
    print(f'>>> {time.strftime("%Y-%m-%d %H:%M:%S")} {msg}', flush=True)
    print('=' * 70, flush=True)


def model_tag(model_dir):
    """Filesystem-safe short tag for the model, used in cache filenames."""
    tag = os.path.basename(os.path.normpath(model_dir)).lower()
    return re.sub(r'[^a-z0-9]+', '-', tag).strip('-')


def resolve_model_dir(model_dir, hf_id):
    """Return a local dir ready for from_pretrained. If model_dir does not
    exist, download the HF repo into it so the model is cached in the repo
    (one-time ~3GB download)."""
    if os.path.isdir(model_dir):
        return model_dir
    print(f'>>> {model_dir} not found locally; downloading {hf_id} into it (one-time)...')
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(hf_id, local_dir=model_dir)
    except ImportError:
        raise SystemExit('huggingface_hub is required to download the model: pip install huggingface_hub')
    except Exception as e:
        # HF returns 401 with a misleading "Invalid username or password" for
        # org-less repo ids that don't resolve, even for public models.
        if '401' in str(e):
            raise SystemExit(
                f'Download of {hf_id} failed with 401 ("Repository Not Found"). '
                'HF returns this for org-less ids even when the model is public — '
                'use the full "owner/name" id, e.g. --hf_id Qwen/Qwen2.5-1.5B-Instruct.')
        raise
    return model_dir


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
    # Deterministic sample/split; deterministic init for the probe weights.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = resolve_device(args.device)
    # fp16 on cuda/mps: extraction is memory-bandwidth-bound, so halving the
    # byte size roughly halves the run time (and halves RAM use). fp32 on CPU.
    # Any problem? Force a different dtype with --dtype.
    if args.dtype:
        dtype = getattr(torch, args.dtype)
    else:
        dtype = torch.float16 if device.startswith(('cuda', 'mps')) else torch.float32

    # ------------------------------------------------------------------
    # Stage 1/5: the model itself (downloaded into the repo on first run).
    # ------------------------------------------------------------------
    banner('Stage 1/5: model')
    print(f'    resolving {args.model} (downloads ~3 GB into the repo if missing)')
    model_dir = resolve_model_dir(args.model, args.hf_id)
    emb_cache = args.emb_cache or f'data/statistics/codecontests_emb_{model_tag(model_dir)}.parquet'

    # ------------------------------------------------------------------
    # Run plan: what a phone-peek at the log should show first.
    # ------------------------------------------------------------------
    banner('Run plan')
    print(f'    device    : {device} ({dtype})')
    print(f'    model     : {model_dir}')
    print(f'    dataset   : {args.data}' + (f' (capped at {args.max_samples} problems)' if args.max_samples > 0 else ''))
    print(f'    embeddings: {os.path.abspath(emb_cache)}')
    print(f'    probe out : {os.path.abspath(args.probe_path)}')
    if os.path.exists(emb_cache):
        print('    >>> Embedding cache FOUND — extraction will be SKIPPED (resume mode).')
    else:
        print('    >>> Embedding cache missing — extraction runs next (the slow step).')

    # ------------------------------------------------------------------
    # Stage 2/5: load the base language model + tokenizer.
    # ------------------------------------------------------------------
    banner('Stage 2/5: loading model + tokenizer')
    print(f'    from {model_dir} ...')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    hidden_size = model.config.hidden_size  # 1536 for Qwen2.5-1.5B
    print(f'    hidden size: {hidden_size}')

    # ------------------------------------------------------------------
    # Stage 3/5: load and filter the CodeContests dataset (same filter as the
    # inference script).
    # ------------------------------------------------------------------
    banner('Stage 3/5: loading + filtering the dataset')
    # Keep only Codeforces problems with a real rating. `difficulty` is a
    # per-source class label (0..4) that is NOT comparable across sources;
    # cf_rating is the Codeforces difficulty rating and is reliable when
    # available. Unrated Codeforces problems carry cf_rating == 0, so filter
    # those out too (p["cf_rating"] is falsy for 0).
    ds = load_dataset(args.data, split='train')
    records = [
        {'name': p['name'], 'description': p['description'], 'real_difficulty': p['cf_rating']}
        for p in ds
        if p['source'] == CODEFORCES_SOURCE and p['cf_rating'] and p['cf_rating'] > 0
    ]
    # Sort by real difficulty so the log is deterministic (as in the
    # inference script). Optional cap for quick smoke tests (fixed seed).
    dfd = pd.DataFrame(records).sort_values('real_difficulty').reset_index(drop=True)
    if args.max_samples > 0 and args.max_samples < len(dfd):
        dfd = dfd.sample(n=args.max_samples, random_state=SEED).sort_values('real_difficulty').reset_index(drop=True)
    print(f'>>> {len(dfd)} Codeforces problems with ratings loaded.')

    # ------------------------------------------------------------------
    # Step 1: extract + cache last-token hidden states (one-time cost).
    # ------------------------------------------------------------------
    # The cache makes re-runs free (train again without re-running the LLM)
    # and stores fp32 embeddings on CPU, so the probe training below is
    # device-agnostic. Skip extraction entirely if the cache already exists.
    banner('Stage 4/5: embedding extraction (the slow step)')
    if os.path.exists(emb_cache):
        print(f'    embedding cache found: {emb_cache}')
        print('    >>> skipping extraction — training will reuse the cached embeddings')
        df_emb = pd.read_parquet(emb_cache)
    else:
        os.makedirs(os.path.dirname(emb_cache), exist_ok=True)
        rows, failed = [], 0
        # Phone-friendly progress: tqdm draws the live bar; every ~0.5% of the
        # run we also print a full line (visible in tmux scrollback) with
        # per-problem timing + ETA, so a peek always shows where things stand.
        progress_every = max(1, len(dfd) // 200)
        extract_start = time.time()
        for idx, row in tqdm(dfd.iterrows(), total=len(dfd), mininterval=2.0):
            try:
                # Chat template identical to codecontests_linear_probe.py, so
                # training-time hidden states match inference-time ones.
                messages = [
                    {'role': 'system', 'content': 'After solving the mathematical problem, place the final answer inside \\boxed{}'},
                    {'role': 'user', 'content': row['description']},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=4096).to(device)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                # Last-token hidden state of the FINAL layer, cast to fp32 and
                # moved to CPU so the cache is device/dtype-agnostic. Stored as
                # a float32 numpy array: Python float lists would be written to
                # parquet as float64 (silently doubling the cache size).
                last_token_emb = outputs.hidden_states[-1][:, -1, :].squeeze(0)
                rows.append({
                    'name': row['name'],
                    'real_difficulty': row['real_difficulty'],
                    'emb': last_token_emb.float().cpu().numpy(),
                })
            except Exception as e:
                failed += 1
                print(f'[Error] index {idx} failed: {e}')
                continue

            done = len(rows) + failed
            if done % progress_every == 0:
                elapsed = time.time() - extract_start
                avg = elapsed / done
                eta = avg * (len(dfd) - done)
                print(f'    {done}/{len(dfd)} ({100.0 * done / len(dfd):5.1f}%)'
                      f' | avg {avg:5.1f}s/problem | elapsed {elapsed / 3600:6.2f}h'
                      f' | ETA {eta / 3600:6.2f}h'
                      f' | "{str(row["name"])[:44]}" ({inputs.input_ids.numel()} tok)',
                      flush=True)

            # Crash-safe: checkpoint the cache periodically.
            if rows and len(rows) % 50 == 0:
                pd.DataFrame(rows).to_parquet(emb_cache, index=False)

        if not rows:
            raise SystemExit('>>> No embeddings extracted (all rows failed).')
        pd.DataFrame(rows).to_parquet(emb_cache, index=False)
        print(f'>>> Extracted {len(rows)} embeddings ({failed} failed)'
              f' in {(time.time() - extract_start) / 60:.1f} min -> {emb_cache}')
        df_emb = pd.DataFrame(rows)

    # Sanity check: every cached embedding must have the model's hidden size.
    lens = {len(e) for e in df_emb['emb']}
    if len(lens) != 1 or lens.pop() != hidden_size:
        raise SystemExit(f'>>> Embeddings have inconsistent dims {lens}; delete {emb_cache} and re-extract.')

    # ------------------------------------------------------------------
    # Stage 5/5: train the linear probe (README recipe).
    # ------------------------------------------------------------------
    banner('Stage 5/5: training the linear probe')
    X = np.stack(df_emb['emb'].values).astype(np.float32)  # [N, hidden]
    y = df_emb['real_difficulty'].values.astype(np.float32).reshape(-1, 1)  # raw cf_rating
    if len(X) < 10:
        raise SystemExit('Too few samples for a meaningful train/val/test split.')

    # 64% train / 16% val / 20% test (same proportions as the DeepMath probe).
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    train_size = int(0.8 * len(X_train_full))
    X_train, X_val = X_train_full[:train_size], X_train_full[train_size:]
    y_train, y_val = y_train_full[:train_size], y_train_full[train_size:]

    # Standardize features AND target. Both scalers are fit on train+val ONLY
    # (never the test set) and are both folded into the probe weights at the
    # end, so the saved probe consumes raw embeddings and emits raw
    # cf_rating-scale scores.
    #
    # Why standardize the target too: the probe bias starts at 0 and must
    # travel ~mean(cf_rating) ~ 1900 units to center predictions on the CF
    # scale. Adam normalizes every parameter's update to ~lr per step, so the
    # bias only moves ~lr*steps ~ 6 units in 80 epochs and the output stays
    # zero-centered (observed: pred range ~-800..800 vs true 800..3500).
    # Z-scoring the target shrinks the required bias travel to ~0.
    scaler = StandardScaler().fit(X_train_full)
    # Raw copies for evaluating the DEPLOYED probe below: the folded probe
    # consumes raw embeddings (like codecontests_linear_probe.py feeds it), so
    # passing it standardized features would evaluate a different linear map
    # and misreport the metrics.
    X_val_raw, X_test_raw = X_val.copy(), X_test.copy()
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_scaler = StandardScaler().fit(y_train_full)
    y_train_std = y_scaler.transform(y_train).astype(np.float32)
    y_val_std = y_scaler.transform(y_val).astype(np.float32)
    y_test_std = y_scaler.transform(y_test).astype(np.float32)
    print(f'    train {len(X_train)} | val {len(X_val)} | test {len(X_test)} samples')
    print(f'    target z-scored (raw mean {y_scaler.mean_[0]:.0f}, std {y_scaler.scale_[0]:.0f})')

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_std)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_std)),
                            batch_size=BATCH_SIZE)

    probe = RegressionNN(hidden_size).to(device)
    criterion = nn.MSELoss()  # MSE on the z-scored rating; raw scale restored by the fold
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state = float('inf'), None
    for epoch in range(EPOCHS):
        probe.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / n

        probe.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(probe(xb), yb)
                total += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss = total / n

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(probe.state_dict())
        print(f'    Epoch {epoch + 1:3d}/{EPOCHS} | Train Loss: {train_loss:.4f}'
              f' | Val Loss: {val_loss:.4f} | best {best_val:.4f}', flush=True)
        # Every 10 epochs, show the metric that actually matters (rank
        # agreement with the real ratings), so a phone peek shows research
        # progress, not just loss.
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                pred = torch.cat([probe(xb.to(device)) for xb, _ in val_loader]).cpu().numpy().ravel()
            y = y_val.ravel()
            print(f'           val Pearson r {np.corrcoef(pred, y)[0, 1]:.4f}'
                  f' | Spearman rho {np.corrcoef(pd.Series(pred).rank(), pd.Series(y).rank())[0, 1]:.4f}',
                  flush=True)

    # ------------------------------------------------------------------
    # Final: fold both scalers into the probe, evaluate, save.
    # ------------------------------------------------------------------
    banner('Final: folding scalers, evaluating, saving probe')
    # Training standardizes x' = (x - mean_x)/std_x and y' = (y - mean_y)/std_y,
    # so y' = W x' + b. The saved probe must instead take raw x directly and
    # emit raw y (codecontests_linear_probe.py feeds raw embeddings).
    # Reparameterize in two steps — mathematically identical predictions, no
    # scaler needed at inference:
    #   feature fold: y' = (W/std_x) x + (b - (W/std_x).mean_x)
    #   target fold:  y  = std_y * y' + mean_y
    probe.cpu()
    probe.load_state_dict(best_state)
    with torch.no_grad():
        std = torch.from_numpy(scaler.scale_).float()
        mean = torch.from_numpy(scaler.mean_).float()
        probe.linear.weight.div_(std)  # W' = W / std_x, broadcast over features
        probe.linear.bias.sub_((probe.linear.weight * mean).sum(-1))  # b' = b - W'.mean_x
        probe.linear.weight.mul_(y_scaler.scale_[0])  # W'' = W' * std_y
        probe.linear.bias.mul_(y_scaler.scale_[0])    # b'' = b' * std_y
        probe.linear.bias.add_(y_scaler.mean_[0])     # b''' = b'' + mean_y
    probe.eval()

    # Evaluate the DEPLOYED probe (raw features in, cf_rating-scale score out)
    # on the held-out splits. Spearman is Pearson on ranks (no scipy dep),
    # matching the inference script's evaluation.
    def metrics(X_raw, y_true):
        with torch.no_grad():
            pred = probe(torch.from_numpy(X_raw).float()).numpy().ravel()
        y_true = y_true.ravel()
        pearson = float(np.corrcoef(pred, y_true)[0, 1])
        spearman = float(np.corrcoef(pd.Series(pred).rank(), pd.Series(y_true).rank())[0, 1])
        rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        return pearson, spearman, rmse

    if len(X_val_raw) > 2:
        vp, vs, vrmse = metrics(X_val_raw, y_val)
        print(f'>>> Val  | Pearson r: {vp:.4f} | Spearman rho: {vs:.4f} | RMSE: {vrmse:.1f} (n={len(X_val_raw)})')
    if len(X_test_raw) > 2:
        tp, ts, trmse = metrics(X_test_raw, y_test)
        print(f'>>> Test | Pearson r: {tp:.4f} | Spearman rho: {ts:.4f} | RMSE: {trmse:.1f} (n={len(X_test_raw)})')

    # Save the full module (like the existing probes), so the inference
    # script's torch.load(..., weights_only=False) loads it unchanged.
    os.makedirs(os.path.dirname(args.probe_path), exist_ok=True)
    torch.save(probe, args.probe_path)

    # ------------------------------------------------------------------
    # Artifact summary.
    # ------------------------------------------------------------------
    print('=' * 60)
    print('>>> Training done. Artifacts (all inside this repo):')
    print(f'    probe       : {os.path.abspath(args.probe_path)}')
    print(f'    embeddings  : {os.path.abspath(emb_cache)}')
    print(f'    model cache : {os.path.abspath(model_dir)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a linear difficulty probe on Codeforces problems from CodeContests')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto' (cuda -> mps -> cpu), or e.g. 'cuda:0', 'mps', 'cpu'")
    parser.add_argument('--dtype', type=str, default=None, help="Model dtype: 'float16', 'float32', 'bfloat16' (default: fp16 on cuda/mps, fp32 on cpu)")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_DIR,
                        help='Local model dir. If it does not exist, --hf_id is downloaded into it (cached inside the repo)')
    parser.add_argument('--hf_id', type=str, default=DEFAULT_HF_ID, help='HuggingFace repo id used to populate --model on first run')
    parser.add_argument('--probe_path', type=str, default='models/difficulty_probe_qwen2.5-1.5b-codecontests.pth',
                        help='Where to save the trained probe')
    parser.add_argument('--emb_cache', type=str, default=None,
                        help='Where to cache embeddings (default: data/statistics/codecontests_emb_<model-tag>.parquet)')
    parser.add_argument('--data', type=str, default='deepmind/code_contests', help='CodeContests dataset name or local path')
    parser.add_argument('--max_samples', type=int, default=-1, help='Cap on the number of problems to embed/train on (-1 = all)')
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
"""One-step MPS fine-tuning smoke test using the exact comparison model."""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_DIR / "model")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "runs/mac-smoke")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def require_local_mps(model_path):
    if not model_path.is_dir():
        raise SystemExit(f"Model folder not found: {model_path}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable. Run this from a native Apple Silicon terminal.")


def last_input_norm(model):
    matches = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("input_layernorm.weight")
    ]
    if not matches:
        raise RuntimeError("Could not find a Qwen input-layer RMSNorm parameter")
    return matches[-1]


def main():
    args = parse_args()
    require_local_mps(args.model)
    torch.manual_seed(42)
    device = torch.device("mps")
    model_dtype = getattr(torch, args.dtype)

    print(f"Loading exact checkpoint from {args.model} in {args.dtype} on MPS...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=model_dtype,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.to(device)

    # Freeze everything, then expose one tiny real parameter for the smoke update.
    model.requires_grad_(False)
    parameter_name, trainable_parameter = last_input_norm(model)
    trainable_parameter.requires_grad_(True)
    before = trainable_parameter.detach().float().cpu().clone()

    messages = [
        {"role": "user", "content": "Write a Python program that reads one integer and prints it."},
        {
            "role": "assistant",
            "content": "<thinking>Read and echo the value.</thinking><solution>```python\nprint(input())\n```</solution>",
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(device)
    labels = batch["input_ids"].clone()

    optimizer = torch.optim.AdamW([trainable_parameter], lr=args.learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch, labels=labels)
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss: {loss.item()}")
    loss.backward()

    gradient_norm = float(trainable_parameter.grad.detach().float().norm().cpu())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError(f"Invalid gradient norm: {gradient_norm}")

    optimizer.step()
    torch.mps.synchronize()
    after = trainable_parameter.detach().float().cpu().clone()
    weight_delta = float((after - before).abs().max())
    if not math.isfinite(weight_delta) or weight_delta <= 0:
        raise RuntimeError(f"Optimizer did not change {parameter_name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "exact-model-smoke.pt"
    summary_path = args.output_dir / "summary.json"
    checkpoint = {
        "base_model": str(args.model.resolve()),
        "parameter_name": parameter_name,
        "updated_parameter": after,
        "learning_rate": args.learning_rate,
        "loss": float(loss.detach().float().cpu()),
        "gradient_norm": gradient_norm,
        "max_weight_delta": weight_delta,
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not torch.equal(reloaded["updated_parameter"], after):
        raise RuntimeError("Saved parameter failed checkpoint round-trip verification")

    summary = {key: value for key, value in checkpoint.items() if key != "updated_parameter"}
    summary.update(
        {
            "architecture": model.config.architectures[0],
            "trainable_parameters": trainable_parameter.numel(),
            "input_tokens": int(batch["input_ids"].numel()),
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "checkpoint": str(checkpoint_path.resolve()),
            "status": "PASS",
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

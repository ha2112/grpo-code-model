#!/usr/bin/env python3
"""Evaluate four LoRA routes on shared Codeforces and Venus held-out cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROUTES = ("venus", "absolute", "probe", "venus-probe")
SUITES = ("codeforces", "venus")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonish(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def normalize_messages(value) -> list[dict[str, str]]:
    value = jsonish(value)
    if isinstance(value, list):
        return [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in value
        ]
    if isinstance(value, dict) and isinstance(value.get("role"), list):
        return [
            {"role": str(role), "content": str(content)}
            for role, content in zip(value["role"], value["content"])
        ]
    raise ValueError(f"Unsupported prompt representation: {type(value).__name__}")


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    """Select deterministic coverage across an already ordered collection."""
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indexes = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indexes]


def stratified_venus(items: list[dict], count: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item["objective"]].append(item)
    objectives = sorted(groups)
    if not objectives:
        return []
    base, remainder = divmod(count, len(objectives))
    selected = []
    for index, objective in enumerate(objectives):
        quota = base + (1 if index < remainder else 0)
        ordered = sorted(groups[objective], key=lambda item: item["example_id"])
        selected.extend(evenly_spaced(ordered, quota))
    return sorted(selected, key=lambda item: (item["objective"], item["example_id"]))


def stable_seed(seed: int, suite: str, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{suite}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def pass_at_k(passed: int, total: int, k: int) -> float:
    """Unbiased pass@k estimator for one problem."""
    if total <= 0 or k <= 0:
        return 0.0
    k = min(k, total)
    failed = total - passed
    if failed < k:
        return 1.0
    return 1.0 - math.comb(failed, k) / math.comb(total, k)


def bootstrap_interval(values: list[float], seed: int, draws: int = 1000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    estimates = []
    for _ in range(draws):
        estimates.append(mean(generator.choice(values) for _ in values))
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]


def load_cases(path: Path, suite: str, tokenizer, max_prompt_length: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files={"eval": str(path)}, split="eval")
    cases = []
    filtered = 0
    for row_index, row in enumerate(dataset):
        extra = jsonish(row.get("extra_info")) or {}
        reward_model = jsonish(row.get("reward_model")) or {}
        ground_truth = jsonish(reward_model.get("ground_truth")) or {}
        messages = normalize_messages(row["prompt"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_count > max_prompt_length:
            filtered += 1
            continue

        problem_id = str(extra.get("problem_id", row_index))
        objective = str(extra.get("efficiency_instruction", "code"))
        example_id = f"{problem_id}:{objective}" if suite == "venus" else problem_id
        cases.append(
            {
                "suite": suite,
                "example_id": example_id,
                "objective": objective,
                "prompt": prompt,
                "prompt_tokens": token_count,
                "data_source": row.get("data_source", ""),
                "ground_truth": jsonable(ground_truth),
                "extra_info": jsonable(extra),
            }
        )

    print(f"{suite}: loaded {len(cases)} eligible cases; filtered {filtered} overlong prompts")
    return cases


def select_cases(cases: list[dict], suite: str, count: int) -> list[dict]:
    if suite == "venus":
        return stratified_venus(cases, count)
    ordered = sorted(
        cases,
        key=lambda item: (
            int(item["extra_info"].get("cf_rating", 0) or 0),
            item["example_id"],
        ),
    )
    return evenly_spaced(ordered, count)


def validate_adapter(adapter_dir: Path) -> dict:
    from safetensors import safe_open

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete LoRA adapter: {adapter_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    tensor_count = 0
    nonzero = False
    finite = True
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_count += 1
            nonzero = nonzero or bool(tensor.count_nonzero().item())
            finite = finite and bool(tensor.isfinite().all().item())
    if tensor_count == 0 or not nonzero or not finite:
        raise ValueError(f"Invalid LoRA tensors in {weights_path}")

    digest_builder = hashlib.sha256()
    with weights_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    return {
        "rank": int(config.get("r", 0)),
        "tensor_count": tensor_count,
        "sha256": digest,
        "size_bytes": weights_path.stat().st_size,
    }


def venus_components(module, text: str, ground_truth: dict, extra_info: dict) -> dict:
    result = module.performance_evalution(text, extra_info)
    baseline_passed = bool(ground_truth.get("passed", False))
    current_passed = bool(result.get("passed", False))
    if baseline_passed and current_passed:
        pass_component = 0.5
    elif not baseline_passed and current_passed:
        pass_component = 1.0
    elif baseline_passed and not current_passed:
        pass_component = -1.0
    else:
        pass_component = -0.5

    objective = extra_info.get("efficiency_instruction", "integral")
    clip = {"time": 90, "memory": 1048576, "integral": 1048576 * 90}[objective]
    baseline_value = min(max(float(ground_truth.get(objective, clip)), 0.0), clip)
    current_value = min(max(float(result.get(objective, clip)), 0.0), clip)
    gain = (baseline_value - current_value) / (baseline_value + 1e-9)
    delta = math.tanh(max(-1.0, min(1.0, gain)))
    improvement = pass_component + (0.5 * delta if pass_component > 0 else 0.0)
    format_value = float(module.single_thinking_solution_format(text))
    reward = 0.5 * improvement + 0.2 * format_value
    return {
        "passed": current_passed,
        "correctness": float(current_passed),
        "format": format_value,
        "improvement": improvement,
        "reward": reward,
        "status": result.get("status", "unknown"),
        "time": result.get("time"),
        "memory": result.get("memory"),
        "integral": result.get("integral"),
    }


def codeforces_components(module, text: str, ground_truth: dict) -> dict:
    tests = ground_truth.get("tests", [])
    correctness = module._correctness_score(module.extract_code(text), tests)
    format_value = float(module.format_score(text))
    return {
        "passed": correctness >= 1.0 - 1e-12,
        "correctness": correctness,
        "format": format_value,
        "improvement": None,
        "reward": 0.8 * correctness + 0.2 * format_value,
        "status": "evaluated",
    }


def load_existing(path: Path) -> tuple[list[dict], set[tuple]]:
    records = []
    keys = set()
    if not path.exists():
        return records, keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                record["route"],
                record["suite"],
                record["example_id"],
                int(record["completion_index"]),
            )
            if key not in keys:
                records.append(record)
                keys.add(key)
    return records, keys


def group_metrics(records: list[dict], num_generations: int) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["route"], record["suite"], record["example_id"])].append(record)

    rows = []
    for (route, suite, example_id), completions in grouped.items():
        completions.sort(key=lambda item: int(item["completion_index"]))
        passed = sum(bool(item["passed"]) for item in completions)
        total = len(completions)
        rows.append(
            {
                "route": route,
                "suite": suite,
                "example_id": example_id,
                "complete": total == num_generations,
                "reward": mean(float(item["reward"]) for item in completions),
                "correctness": mean(float(item["correctness"]) for item in completions),
                "format": mean(float(item["format"]) for item in completions),
                "pass_at_1": pass_at_k(passed, total, 1),
                "pass_at_4": pass_at_k(passed, total, min(4, total)),
                "improvement": mean(
                    float(item["improvement"])
                    for item in completions
                    if item.get("improvement") is not None
                ) if suite == "venus" else None,
            }
        )
    return rows


def aggregate(records: list[dict], num_generations: int, seed: int) -> list[dict]:
    per_example = group_metrics(records, num_generations)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_example:
        if row["complete"]:
            grouped[(row["route"], row["suite"])].append(row)

    summary = []
    metrics = ("reward", "correctness", "format", "pass_at_1", "pass_at_4", "improvement")
    for route in ROUTES:
        for suite in SUITES:
            examples = grouped[(route, suite)]
            result = {
                "route": route,
                "suite": suite,
                "examples": len(examples),
                "completions": len(examples) * num_generations,
            }
            for metric in metrics:
                values = [float(row[metric]) for row in examples if row.get(metric) is not None]
                if not values:
                    result[metric] = None
                    result[f"{metric}_ci_low"] = None
                    result[f"{metric}_ci_high"] = None
                    continue
                low, high = bootstrap_interval(values, stable_seed(seed, suite, f"{route}:{metric}"))
                result[metric] = mean(values)
                result[f"{metric}_ci_low"] = low
                result[f"{metric}_ci_high"] = high
            summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_abstract_table(path: Path, summary: list[dict]) -> None:
    indexed = {(row["route"], row["suite"]): row for row in summary}
    rows = []
    for route in ROUTES:
        codeforces = indexed[(route, "codeforces")]
        venus = indexed[(route, "venus")]
        rows.append(
            {
                "route": route,
                "codeforces_pass_at_1": codeforces["pass_at_1"],
                "codeforces_pass_at_4": codeforces["pass_at_4"],
                "codeforces_format": codeforces["format"],
                "codeforces_reward": codeforces["reward"],
                "venus_pass_at_1": venus["pass_at_1"],
                "venus_pass_at_4": venus["pass_at_4"],
                "venus_format": venus["format"],
                "venus_improvement": venus["improvement"],
                "venus_reward": venus["reward"],
            }
        )
    write_csv(path, rows)


def write_plot(path: Path, summary: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV and JSON summaries were still written")
        return False

    indexed = {(row["route"], row["suite"]): row for row in summary}
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    labels = [route.replace("-", "\n") for route in ROUTES]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for row_index, suite in enumerate(SUITES):
        for column_index, metric in enumerate(("pass_at_1", "pass_at_4", "format")):
            axis = axes[row_index][column_index]
            values = [indexed[(route, suite)][metric] or 0.0 for route in ROUTES]
            lows = [indexed[(route, suite)][f"{metric}_ci_low"] or 0.0 for route in ROUTES]
            highs = [indexed[(route, suite)][f"{metric}_ci_high"] or 0.0 for route in ROUTES]
            errors = [
                [max(0.0, value - low) for value, low in zip(values, lows)],
                [max(0.0, high - value) for value, high in zip(values, highs)],
            ]
            axis.bar(labels, values, color=colors, yerr=errors, capsize=3)
            axis.set_ylim(0, 1)
            metric_label = {
                "pass_at_1": "pass@1",
                "pass_at_4": "pass@4",
                "format": "format rate",
            }[metric]
            axis.set_title(f"{suite.title()} {metric_label}")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Held-out evaluation at training step 600")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def write_svg_plot(path: Path, summary: list[dict]) -> None:
    """Write the core comparison figure without optional plotting packages."""
    indexed = {(row["route"], row["suite"]): row for row in summary}
    width, height = 1320, 720
    panel_width, panel_height = 400, 260
    left_margin, top_margin = 70, 65
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    metric_labels = (("pass_at_1", "pass@1"), ("pass_at_4", "pass@4"), ("format", "format rate"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:22px;font-weight:bold}.panel{font-size:16px;font-weight:bold}.tick{font-size:11px}.label{font-size:12px}</style>',
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">Held-out evaluation at training step 600</text>',
    ]
    for row_index, suite in enumerate(SUITES):
        for column_index, (metric, metric_label) in enumerate(metric_labels):
            x0 = left_margin + column_index * 420
            y0 = top_margin + row_index * 320
            plot_top = y0 + 35
            plot_bottom = y0 + panel_height - 45
            plot_height = plot_bottom - plot_top
            parts.append(
                f'<text class="panel" x="{x0 + panel_width / 2}" y="{y0 + 18}" text-anchor="middle">{suite.title()} {metric_label}</text>'
            )
            parts.append(f'<line x1="{x0}" y1="{plot_top}" x2="{x0}" y2="{plot_bottom}" stroke="#444"/>')
            parts.append(f'<line x1="{x0}" y1="{plot_bottom}" x2="{x0 + panel_width}" y2="{plot_bottom}" stroke="#444"/>')
            for tick in range(5):
                value = tick / 4
                y = plot_bottom - value * plot_height
                parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}" stroke="#ddd"/>')
                parts.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>')
            slot_width = panel_width / len(ROUTES)
            for route_index, route in enumerate(ROUTES):
                row = indexed[(route, suite)]
                value = float(row.get(metric) or 0.0)
                low = float(row.get(f"{metric}_ci_low") or 0.0)
                high = float(row.get(f"{metric}_ci_high") or 0.0)
                bar_width = slot_width * 0.58
                x = x0 + route_index * slot_width + (slot_width - bar_width) / 2
                y = plot_bottom - value * plot_height
                center = x + bar_width / 2
                low_y = plot_bottom - low * plot_height
                high_y = plot_bottom - high * plot_height
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{value * plot_height:.1f}" fill="{colors[route_index]}"/>'
                )
                parts.append(f'<line x1="{center:.1f}" y1="{high_y:.1f}" x2="{center:.1f}" y2="{low_y:.1f}" stroke="#222"/>')
                parts.append(f'<line x1="{center - 6:.1f}" y1="{high_y:.1f}" x2="{center + 6:.1f}" y2="{high_y:.1f}" stroke="#222"/>')
                parts.append(f'<line x1="{center - 6:.1f}" y1="{low_y:.1f}" x2="{center + 6:.1f}" y2="{low_y:.1f}" stroke="#222"/>')
                label = html.escape(route.replace("-", " "))
                parts.append(
                    f'<text class="label" x="{center:.1f}" y="{plot_bottom + 18}" text-anchor="middle">{label}</text>'
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--codeforces-data", type=Path, required=True)
    parser.add_argument("--venus-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-suite", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--monolith-url", default="http://127.0.0.1:8000/execute")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_suite <= 0 or args.num_generations < 4 or args.batch_size <= 0:
        raise ValueError("sample and batch counts must be positive, and at least four generations are required")

    repo_dir = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    records, completed_keys = load_existing(predictions_path)

    if not args.aggregate_only:
        os.environ["MONOLITH_URL"] = args.monolith_url
        codeforces_reward = load_module(
            "evaluation_codeforces_reward",
            repo_dir / "grpo_difficulty_probe/codeforces_reward_function.py",
        )
        venus_reward = load_module(
            "evaluation_venus_reward",
            repo_dir / "grpo/afterburner_reward_function.py",
        )

        adapter_manifest = {}
        hashes = defaultdict(list)
        for route in ROUTES:
            adapter_dir = args.adapter_root / route / "lora_adapter"
            adapter_manifest[route] = validate_adapter(adapter_dir)
            hashes[adapter_manifest[route]["sha256"]].append(route)
            adapter_manifest[route]["path"] = str(adapter_dir.resolve())
        duplicates = [routes for routes in hashes.values() if len(routes) > 1]
        if duplicates:
            raise ValueError(f"Identical adapter checkpoints detected: {duplicates}")
        (args.output_dir / "adapter_manifest.json").write_text(
            json.dumps(adapter_manifest, indent=2), encoding="utf-8"
        )

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        suite_cases = {
            "codeforces": select_cases(
                load_cases(args.codeforces_data, "codeforces", tokenizer, args.max_prompt_length),
                "codeforces",
                args.samples_per_suite,
            ),
            "venus": select_cases(
                load_cases(args.venus_data, "venus", tokenizer, args.max_prompt_length),
                "venus",
                args.samples_per_suite,
            ),
        }
        manifest = {
            suite: [
                {
                    "example_id": case["example_id"],
                    "objective": case["objective"],
                    "prompt_tokens": case["prompt_tokens"],
                }
                for case in cases
            ]
            for suite, cases in suite_cases.items()
        }
        (args.output_dir / "sample_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        llm = LLM(
            model=args.base_model,
            tokenizer=args.base_model,
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enable_lora=True,
            max_lora_rank=max(item["rank"] for item in adapter_manifest.values()),
            max_loras=1,
            max_cpu_loras=len(ROUTES),
            max_model_len=args.max_prompt_length + args.max_new_tokens,
            max_num_batched_tokens=args.max_prompt_length + args.max_new_tokens,
            max_num_seqs=args.batch_size * args.num_generations,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )

        with predictions_path.open("a", encoding="utf-8") as output:
            for route_index, route in enumerate(ROUTES, start=1):
                adapter_dir = args.adapter_root / route / "lora_adapter"
                lora_request = LoRARequest(route, route_index, str(adapter_dir.resolve()))
                for suite in SUITES:
                    cases = suite_cases[suite]
                    for start in range(0, len(cases), args.batch_size):
                        batch = cases[start : start + args.batch_size]
                        pending = [
                            case
                            for case in batch
                            if not all(
                                (route, suite, case["example_id"], completion) in completed_keys
                                for completion in range(args.num_generations)
                            )
                        ]
                        if not pending:
                            continue
                        sampling = [
                            SamplingParams(
                                n=args.num_generations,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_new_tokens,
                                seed=stable_seed(args.seed, suite, case["example_id"]),
                            )
                            for case in pending
                        ]
                        generated = llm.generate(
                            [case["prompt"] for case in pending],
                            sampling,
                            lora_request=lora_request,
                        )
                        for case, request_output in zip(pending, generated):
                            for completion_index, candidate in enumerate(request_output.outputs):
                                key = (route, suite, case["example_id"], completion_index)
                                if key in completed_keys:
                                    continue
                                text = candidate.text
                                if suite == "codeforces":
                                    metrics = codeforces_components(
                                        codeforces_reward, text, case["ground_truth"]
                                    )
                                else:
                                    metrics = venus_components(
                                        venus_reward,
                                        text,
                                        case["ground_truth"],
                                        case["extra_info"],
                                    )
                                record = {
                                    "route": route,
                                    "suite": suite,
                                    "example_id": case["example_id"],
                                    "objective": case["objective"],
                                    "completion_index": completion_index,
                                    "seed": stable_seed(args.seed, suite, case["example_id"]),
                                    "prompt_tokens": case["prompt_tokens"],
                                    "output_tokens": len(candidate.token_ids or []),
                                    "output": text,
                                    **jsonable(metrics),
                                }
                                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                                output.flush()
                                records.append(record)
                                completed_keys.add(key)
                        print(
                            f"{route}/{suite}: evaluated {min(start + len(batch), len(cases))}/{len(cases)} cases",
                            flush=True,
                        )

    summary = aggregate(records, args.num_generations, args.seed)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "summary.csv", summary)
    write_abstract_table(args.output_dir / "abstract_table.csv", summary)
    write_svg_plot(args.output_dir / "comparison.svg", summary)
    write_plot(args.output_dir / "comparison.png", summary)

    print(json.dumps(summary, indent=2))
    print(f"Raw predictions: {predictions_path}")
    print(f"Abstract table: {args.output_dir / 'abstract_table.csv'}")


if __name__ == "__main__":
    main()

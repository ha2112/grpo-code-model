#!/usr/bin/env python3
"""Recover an interrupted four-route evaluation without regenerating saved outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_four_routes as evaluator  # noqa: E402


GENERATION_FIELDS = (
    "route",
    "suite",
    "example_id",
    "objective",
    "completion_index",
    "seed",
    "prompt_tokens",
    "output_tokens",
    "output",
)


def record_key(record: dict) -> tuple[str, str, str, int]:
    return (
        str(record["route"]),
        str(record["suite"]),
        str(record["example_id"]),
        int(record["completion_index"]),
    )


def generation_record(record: dict) -> dict:
    """Keep model output metadata while deliberately discarding old judge scores."""
    missing = [field for field in GENERATION_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Generation record is missing fields: {missing}")
    cleaned = {field: record[field] for field in GENERATION_FIELDS}
    cleaned["completion_index"] = int(cleaned["completion_index"])
    cleaned["seed"] = int(cleaned["seed"])
    cleaned["prompt_tokens"] = int(cleaned["prompt_tokens"])
    cleaned["output_tokens"] = int(cleaned["output_tokens"])
    return cleaned


def load_unique(path: Path) -> tuple[list[dict], set[tuple]]:
    records = []
    keys = set()
    if not path.exists():
        return records, keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record_key(record)
            if key in keys:
                raise ValueError(f"Duplicate record at {path}:{line_number}: {key}")
            records.append(record)
            keys.add(key)
    return records, keys


def initialize_generations(source: Path, destination: Path) -> tuple[list[dict], set[tuple]]:
    if destination.exists():
        return load_unique(destination)
    source_records, _ = load_unique(source)
    if not source_records:
        raise ValueError(f"No archived predictions found in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [generation_record(record) for record in source_records]
    with destination.open("x", encoding="utf-8") as handle:
        for record in cleaned:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return cleaned, {record_key(record) for record in cleaned}


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-model",
        default="Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start",
    )
    parser.add_argument(
        "--codeforces-data",
        type=Path,
        default=REPO_DIR / "grpo_codeforces_curriculum/data/codeforces_validation_easy_to_hard.parquet",
    )
    parser.add_argument(
        "--venus-data",
        type=Path,
        default=Path.home() / "data/venus/venus_test.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-suite", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate only missing model outputs")
    add_shared_arguments(generate)
    generate.add_argument("--source-predictions", type=Path, required=True)
    generate.add_argument(
        "--adapter-root",
        type=Path,
        default=REPO_DIR / "evaluation-models/step-600",
    )
    generate.add_argument("--max-new-tokens", type=int, default=1024)
    generate.add_argument("--batch-size", type=int, default=1)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.95)

    rescore = subparsers.add_parser("rescore", help="judge all saved outputs without inference")
    add_shared_arguments(rescore)
    rescore.add_argument(
        "--monolith-url",
        default="http://127.0.0.1:8000/execute",
    )
    return parser.parse_args()


def load_selected_cases(args: argparse.Namespace) -> dict[str, list[dict]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    return {
        "codeforces": evaluator.select_cases(
            evaluator.load_cases(
                args.codeforces_data,
                "codeforces",
                tokenizer,
                args.max_prompt_length,
            ),
            "codeforces",
            args.samples_per_suite,
        ),
        "venus": evaluator.select_cases(
            evaluator.load_cases(
                args.venus_data,
                "venus",
                tokenizer,
                args.max_prompt_length,
            ),
            "venus",
            args.samples_per_suite,
        ),
    }


def expected_keys(suite_cases: dict[str, list[dict]], num_generations: int) -> set[tuple]:
    return {
        (route, suite, case["example_id"], completion)
        for route in evaluator.ROUTES
        for suite in evaluator.SUITES
        for case in suite_cases[suite]
        for completion in range(num_generations)
    }


def validate_generation_keys(keys: set[tuple], expected: set[tuple]) -> None:
    unexpected = keys - expected
    if unexpected:
        raise ValueError(f"Archived predictions do not match this evaluation: {sorted(unexpected)[:3]}")


def write_manifests(
    output_dir: Path,
    suite_cases: dict[str, list[dict]],
    adapter_manifest: dict | None = None,
) -> None:
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
    (output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if adapter_manifest is not None:
        (output_dir / "adapter_manifest.json").write_text(
            json.dumps(adapter_manifest, indent=2), encoding="utf-8"
        )


def adapter_metadata(adapter_root: Path) -> dict:
    manifest = {}
    hashes = defaultdict(list)
    for route in evaluator.ROUTES:
        adapter_dir = adapter_root / route / "lora_adapter"
        manifest[route] = evaluator.validate_adapter(adapter_dir)
        hashes[manifest[route]["sha256"]].append(route)
        manifest[route]["path"] = str(adapter_dir.resolve())
    duplicates = [routes for routes in hashes.values() if len(routes) > 1]
    if duplicates:
        raise ValueError(f"Identical adapter checkpoints detected: {duplicates}")
    return manifest


def print_counts(label: str, records: list[dict], expected_total: int) -> None:
    counts = Counter((record["route"], record["suite"]) for record in records)
    print(f"{label}: {len(records)}/{expected_total} generations")
    for route in evaluator.ROUTES:
        for suite in evaluator.SUITES:
            print(f"  {route}/{suite}: {counts[(route, suite)]}")


def generate_missing(args: argparse.Namespace) -> None:
    if args.num_generations < 4 or args.batch_size <= 0:
        raise ValueError("At least four generations and a positive batch size are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.output_dir / "generations.jsonl"
    records, completed = initialize_generations(args.source_predictions, generations_path)
    suite_cases = load_selected_cases(args)
    expected = expected_keys(suite_cases, args.num_generations)
    validate_generation_keys(completed, expected)
    adapter_manifest = adapter_metadata(args.adapter_root)
    write_manifests(args.output_dir, suite_cases, adapter_manifest)
    print_counts("Recovered", records, len(expected))
    if completed == expected:
        print(f"Generation recovery is already complete: {generations_path}")
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

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
        max_cpu_loras=len(evaluator.ROUTES),
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        max_num_batched_tokens=args.max_prompt_length + args.max_new_tokens,
        max_num_seqs=args.batch_size * args.num_generations,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )

    with generations_path.open("a", encoding="utf-8") as output:
        for route_index, route in enumerate(evaluator.ROUTES, start=1):
            adapter_dir = args.adapter_root / route / "lora_adapter"
            lora_request = LoRARequest(route, route_index, str(adapter_dir.resolve()))
            for suite in evaluator.SUITES:
                cases = suite_cases[suite]
                for start in range(0, len(cases), args.batch_size):
                    batch = cases[start : start + args.batch_size]
                    pending = [
                        case
                        for case in batch
                        if not all(
                            (route, suite, case["example_id"], completion) in completed
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
                            seed=evaluator.stable_seed(args.seed, suite, case["example_id"]),
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
                            if key in completed:
                                continue
                            record = {
                                "route": route,
                                "suite": suite,
                                "example_id": case["example_id"],
                                "objective": case["objective"],
                                "completion_index": completion_index,
                                "seed": evaluator.stable_seed(args.seed, suite, case["example_id"]),
                                "prompt_tokens": case["prompt_tokens"],
                                "output_tokens": len(candidate.token_ids or []),
                                "output": candidate.text,
                            }
                            output.write(json.dumps(record, ensure_ascii=False) + "\n")
                            output.flush()
                            records.append(record)
                            completed.add(key)
                    print(
                        f"{route}/{suite}: generated {len(completed)}/{len(expected)} total outputs",
                        flush=True,
                    )

    print_counts("Complete", records, len(expected))
    print(f"Unscored generations: {generations_path}")


def score_generation(
    generation: dict,
    case: dict,
    codeforces_reward,
    venus_reward,
) -> dict:
    if generation["suite"] == "codeforces":
        metrics = evaluator.codeforces_components(
            codeforces_reward,
            generation["output"],
            case["ground_truth"],
        )
    else:
        metrics = evaluator.venus_components(
            venus_reward,
            generation["output"],
            case["ground_truth"],
            case["extra_info"],
        )
    return {**generation_record(generation), **evaluator.jsonable(metrics)}


def write_summaries(output_dir: Path, records: list[dict], args: argparse.Namespace) -> None:
    summary = evaluator.aggregate(records, args.num_generations, args.seed)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    evaluator.write_csv(output_dir / "summary.csv", summary)
    evaluator.write_abstract_table(output_dir / "abstract_table.csv", summary)
    evaluator.write_svg_plot(output_dir / "comparison.svg", summary)
    evaluator.write_plot(output_dir / "comparison.png", summary)


def rescore_existing(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.output_dir / "generations.jsonl"
    generations, generation_keys = load_unique(generations_path)
    suite_cases = load_selected_cases(args)
    expected = expected_keys(suite_cases, args.num_generations)
    validate_generation_keys(generation_keys, expected)
    if generation_keys != expected:
        raise ValueError(
            f"Generation set is incomplete: found {len(generation_keys)}, expected {len(expected)}. "
            "Run the generate command first."
        )

    os.environ["MONOLITH_URL"] = args.monolith_url
    codeforces_reward = evaluator.load_module(
        "recovery_codeforces_reward",
        REPO_DIR / "grpo_difficulty_probe/codeforces_reward_function.py",
    )
    venus_reward = evaluator.load_module(
        "recovery_venus_reward",
        REPO_DIR / "grpo/afterburner_reward_function.py",
    )
    codeforces_reward.check_judge()
    venus_reward.check_judge()

    case_lookup = {
        (suite, case["example_id"]): case
        for suite, cases in suite_cases.items()
        for case in cases
    }
    predictions_path = args.output_dir / "predictions.jsonl"
    scored, scored_keys = load_unique(predictions_path)
    validate_generation_keys(scored_keys, expected)
    for record in scored:
        generation = next(item for item in generations if record_key(item) == record_key(record))
        if record["output"] != generation["output"]:
            raise ValueError(f"Scored output does not match saved generation: {record_key(record)}")

    with predictions_path.open("a", encoding="utf-8") as output:
        for generation in generations:
            key = record_key(generation)
            if key in scored_keys:
                continue
            case = case_lookup[(generation["suite"], generation["example_id"])]
            record = score_generation(
                generation,
                case,
                codeforces_reward,
                venus_reward,
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            scored.append(record)
            scored_keys.add(key)
            print(f"Scored {len(scored_keys)}/{len(expected)}", flush=True)

    write_summaries(args.output_dir, scored, args)
    print(f"Clean predictions: {predictions_path}")
    print(f"Abstract table: {args.output_dir / 'abstract_table.csv'}")


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate_missing(args)
    else:
        rescore_existing(args)


if __name__ == "__main__":
    main()

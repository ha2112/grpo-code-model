"""Build a Codeforces GRPO curriculum ranked by a learned difficulty probe."""

import argparse
import json
import os
import sys
from pathlib import Path


CODEFORCES_SOURCE = 2
DATA_SOURCE = "deepmind/code_contests"
ROUTE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROUTE_DIR.parent
DEFAULT_MODEL = WORKSPACE_DIR / "Difficulty Probing/model/qwen2.5-1.5B-instruct"
DEFAULT_PROBE = WORKSPACE_DIR / "Difficulty Probing/models/difficulty_probe_qwen2.5-1.5b-codecontests.pth"

PROBE_SYSTEM_PROMPT = "After solving the mathematical problem, place the final answer inside \\boxed{}"

SYSTEM_PROMPT = """A conversation between User and Assistant. The user gives a
competitive-programming problem and the Assistant solves it in Python 3. The
Assistant first reasons inside <thinking> </thinking>, then puts the complete
program inside <solution> </solution> as one markdown Python code block."""

USER_TEMPLATE = """## Problem
{description}

## Original Solution
{baseline_solution}

## Original Performance
Unavailable: CodeContests provides no baseline runtime or memory measurements.

## Output format
Return exactly:
<thinking>your reasoning</thinking><solution>```python
complete Python 3 program
```</solution>

Fix the original solution if it is incorrect. Otherwise, improve it while
preserving correctness.
"""


def select_codeforces(problems):
    """Keep all Codeforces problems, including those without a real rating."""
    return [problem for problem in problems if problem.get("source") == CODEFORCES_SOURCE]


def problem_key(split, problem):
    contest = int(problem.get("cf_contest_id", 0) or 0)
    index = str(problem.get("cf_index", ""))
    name = str(problem.get("name", ""))
    return f"{split}:{contest}:{index}:{name}"


def collect_tests(problem):
    tests = []
    for group_name in ("public_tests", "private_tests", "generated_tests"):
        group = problem.get(group_name) or {}
        inputs = group.get("input") or []
        outputs = group.get("output") or []
        tests.extend(
            {"input": input_text, "output": output_text}
            for input_text, output_text in zip(inputs, outputs)
        )
    return tests


def _baseline_code(problem):
    """Return one deterministic Python solution from CodeContests."""
    solutions = problem.get("solutions") or {}
    if isinstance(solutions, dict):
        codes = solutions.get("solution") or solutions.get("code") or []
        languages = solutions.get("language") or []
        if isinstance(codes, str):
            codes = [codes]
        if isinstance(languages, str):
            languages = [languages]
        candidates = zip(codes, languages or [None] * len(codes))
    elif isinstance(solutions, list):
        candidates = ((item, None) for item in solutions)
    else:
        candidates = ()

    fallback = ""
    for item, language in candidates:
        if isinstance(item, dict):
            language = item.get("language", language)
            item = item.get("code") or item.get("solution") or ""
        if not isinstance(item, str) or not item.strip():
            continue
        if not fallback:
            fallback = item.strip()
        if language is None or str(language).lower() in {"python", "python2", "python3", "1"}:
            return item.strip()
    return fallback


def _tie_breaker(problem):
    return (
        int(problem.get("cf_contest_id", 0) or 0),
        str(problem.get("cf_index", "")),
        str(problem.get("name", "")),
    )


def make_record(problem, split, probe_score, rank, total):
    percentile = rank / (total - 1) if total > 1 else 0.0
    baseline_solution = _baseline_code(problem)
    problem_id = f"{problem.get('cf_contest_id', 0)}{problem.get('cf_index', '')}"
    if not baseline_solution:
        baseline_solution = (
            "# No baseline solution is available. "
            "Solve this problem from scratch."
        )
    return {
        "data_source": "codeforces_probe_curriculum",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    description=problem.get("description", ""),
                    baseline_solution=baseline_solution,
                ),
            },
        ],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "tests": collect_tests(problem),
                "baseline_solution": baseline_solution,
            },
        },
        "extra_info": {
            "split": split,
            "problem_id": f"{problem.get('cf_contest_id', 0)}{problem.get('cf_index', '')}",
            "name": problem.get("name", ""),
            # Real rating is retained for analysis, never for ordering.
            "cf_rating": int(problem.get("cf_rating", 0) or 0),
            "probe_difficulty": float(probe_score),
            "probe_rank": rank,
            "probe_percentile": percentile,
            "curriculum_order": percentile,
            "time_limit_seconds": (problem.get("time_limit") or {}).get("seconds", 0),
            "memory_limit_bytes": problem.get("memory_limit_bytes", 0),
        },
    }


def build_records(problems, split, scores):
    selected = select_codeforces(problems)
    missing = [problem_key(split, problem) for problem in selected if problem_key(split, problem) not in scores]
    if missing:
        preview = ", ".join(missing[:3])
        raise KeyError(f"Missing probe scores for {len(missing)} problems: {preview}")

    ordered = sorted(
        selected,
        key=lambda problem: (float(scores[problem_key(split, problem)]), *_tie_breaker(problem)),
    )
    return [
        make_record(problem, split, scores[problem_key(split, problem)], rank, len(ordered))
        for rank, problem in enumerate(ordered)
    ]


def load_score_cache(path):
    scores = {}
    if not path.exists():
        return scores
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                scores[item["key"]] = float(item["probe_difficulty"])
    return scores


def append_score(path, key, score):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "probe_difficulty": float(score)}) + "\n")


def _install_checkpoint_class():
    """Expose the class name stored in the full-module PyTorch checkpoint."""
    import torch.nn as nn

    class RegressionNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.linear(x)

    # The existing checkpoint was serialized as __main__.RegressionNN.
    setattr(sys.modules["__main__"], "RegressionNN", RegressionNN)
    return RegressionNN


def resolve_device(requested, torch):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DifficultyScorer:
    def __init__(self, model_path, probe_path, device="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _install_checkpoint_class()
        self.torch = torch
        self.device = resolve_device(device, torch)
        self.dtype = torch.float16 if self.device.startswith(("cuda", "mps")) else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.probe = torch.load(str(probe_path), map_location=self.device, weights_only=False).to(self.device)
        self.probe.eval()

    def score(self, description):
        messages = [
            {"role": "system", "content": PROBE_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][:, -1, :].squeeze(0).float()
            return float(self.probe(embedding).item())


def ensure_scores(problems, split, scores, cache_path, scorer_factory):
    scorer = None
    for position, problem in enumerate(problems, start=1):
        key = problem_key(split, problem)
        if key in scores:
            continue
        if scorer is None:
            scorer = scorer_factory()
        score = scorer.score(problem.get("description", ""))
        scores[key] = score
        append_score(cache_path, key, score)
        print(f"{split}: probed {position}/{len(problems)} {key} -> {score:.3f}", flush=True)


def write_split(problems, split, scores, output_path):
    from datasets import Dataset

    records = build_records(problems, split, scores)
    if not records:
        raise RuntimeError(f"No Codeforces problems found in {split}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    first = records[0]["extra_info"]["probe_difficulty"]
    last = records[-1]["extra_info"]["probe_difficulty"]
    print(f"{split}: wrote {len(records)} rows to {output_path} (probe {first:.3f} -> {last:.3f})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATA_SOURCE, help="Hugging Face dataset name or local path")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="valid")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("PROBE_CURRICULUM_DATA_DIR", ROUTE_DIR / "data")),
    )
    parser.add_argument("--max-train", type=int, default=-1, help="Probe the first N source rows for a smoke run")
    parser.add_argument("--max-validation", type=int, default=-1, help="Probe the first N validation rows")
    return parser.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset

    if not args.model.exists():
        raise SystemExit(f"Probe base model not found: {args.model}")
    if not args.probe.exists():
        raise SystemExit(f"Difficulty probe not found: {args.probe}")

    train = select_codeforces(load_dataset(args.dataset, split=args.train_split))
    validation = select_codeforces(load_dataset(args.dataset, split=args.validation_split))
    if args.max_train > 0:
        train = train[: args.max_train]
    if args.max_validation > 0:
        validation = validation[: args.max_validation]

    cache_path = args.output_dir / "probe_scores.jsonl"
    scores = load_score_cache(cache_path)
    scorer_factory = lambda: DifficultyScorer(args.model, args.probe, args.device)
    ensure_scores(train, "train", scores, cache_path, scorer_factory)
    ensure_scores(validation, "validation", scores, cache_path, scorer_factory)
    write_split(train, "train", scores, args.output_dir / "probe_train_easy_to_hard.parquet")
    write_split(
        validation,
        "validation",
        scores,
        args.output_dir / "probe_validation_easy_to_hard.parquet",
    )


if __name__ == "__main__":
    main()

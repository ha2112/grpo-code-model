"""Build verl parquet files for a Codeforces easiest-to-hardest curriculum."""

import argparse
import os
from pathlib import Path


CODEFORCES_SOURCE = 2
DATA_SOURCE = "deepmind/code_contests"
ROUTE_DIR = Path(__file__).resolve().parent

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


def _rating(problem):
    value = problem.get("cf_rating", 0)
    return int(value or 0)


def select_codeforces(problems):
    """Keep rated Codeforces problems and sort deterministically by rating."""
    selected = [
        problem
        for problem in problems
        if problem.get("source") == CODEFORCES_SOURCE and _rating(problem) > 0
    ]
    return sorted(
        selected,
        key=lambda problem: (
            _rating(problem),
            int(problem.get("cf_contest_id", 0) or 0),
            str(problem.get("cf_index", "")),
            str(problem.get("name", "")),
        ),
    )


def collect_tests(problem):
    """Combine the test groups that are present in CodeContests."""
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


def make_record(problem, split):
    """Convert one CodeContests row to verl's prompt/reward schema."""
    rating = _rating(problem)
    problem_id = f"{problem.get('cf_contest_id', 0)}{problem.get('cf_index', '')}"
    tests = collect_tests(problem)
    baseline_solution = _baseline_code(problem)
    if not baseline_solution:
        raise ValueError(f"No baseline solution found for Codeforces problem {problem_id}")
    return {
        "data_source": "codeforces_curriculum",
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
            "ground_truth": {"tests": tests, "baseline_solution": baseline_solution},
        },
        "extra_info": {
            "split": split,
            "problem_id": problem_id,
            "name": problem.get("name", ""),
            "cf_rating": rating,
            # This duplicates rating intentionally for easy curriculum auditing.
            "curriculum_order": rating,
            "time_limit_seconds": (problem.get("time_limit") or {}).get("seconds", 0),
            "memory_limit_bytes": problem.get("memory_limit_bytes", 0),
        },
    }


def build_records(problems, split, limit=-1):
    selected = select_codeforces(problems)
    if limit > 0:
        selected = selected[:limit]
    records = [make_record(problem, split) for problem in selected]
    ratings = [record["extra_info"]["cf_rating"] for record in records]
    if ratings != sorted(ratings):
        raise RuntimeError(f"{split} ratings are not in ascending order")
    return records


def write_split(dataset, split, output_path, limit=-1):
    from datasets import Dataset

    records = build_records(dataset, split, limit)
    if not records:
        raise RuntimeError(f"No rated Codeforces problems found in {split}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    first = records[0]["extra_info"]["cf_rating"]
    last = records[-1]["extra_info"]["cf_rating"]
    print(f"{split}: wrote {len(records)} rows to {output_path} (ratings {first} -> {last})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATA_SOURCE, help="Hugging Face dataset name or local path")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="valid")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("CODEFORCES_CURRICULUM_DATA_DIR", ROUTE_DIR / "data")),
    )
    parser.add_argument("--max-train", type=int, default=-1, help="Keep the easiest N train problems")
    parser.add_argument("--max-validation", type=int, default=-1, help="Keep the easiest N validation problems")
    return parser.parse_args()


def main():
    from datasets import load_dataset

    args = parse_args()
    train = load_dataset(args.dataset, split=args.train_split)
    validation = load_dataset(args.dataset, split=args.validation_split)
    write_split(
        train,
        "train",
        args.output_dir / "codeforces_train_easy_to_hard.parquet",
        args.max_train,
    )
    write_split(
        validation,
        "validation",
        args.output_dir / "codeforces_validation_easy_to_hard.parquet",
        args.max_validation,
    )


if __name__ == "__main__":
    main()

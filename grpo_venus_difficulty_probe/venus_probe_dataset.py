"""Build a Venus GRPO corpus ordered by the learned difficulty probe."""

import argparse
import json
import os
import random
import sys
from pathlib import Path


DATA_SOURCE = "Elfsong/Venus_Python"
ROUTE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROUTE_DIR.parent
DEFAULT_MODEL = WORKSPACE_DIR / "Difficulty Probing/model/qwen2.5-1.5B-instruct"
DEFAULT_PROBE = WORKSPACE_DIR / "Difficulty Probing/models/difficulty_probe_qwen2.5-1.5b-codecontests.pth"
EFFICIENCY_INSTRUCTIONS = {
    "time": "time efficient",
    "memory": "memory efficient",
    "integral": "both time and memory efficient",
}

PROBE_SYSTEM_PROMPT = "After solving the mathematical problem, place the final answer inside \\boxed{}"

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question and provides an original solution, then the Assistant improve it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the improved solution.
The reasoning process and solution are enclosed within <thinking> </thinking> and <solution> </solution> tags, respectively.
For example, "<thinking>reasoning_process</thinking><solution>improved_solution</solution>".
"""

AFTERBURNER_TEMPLATE = """
## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in {target_lang}.
## Problem Description
{problem_description}
## Original Solution
{original_solution}
## Original Performance
Passed: {original_passed} / Time: {original_time} / Memory: {original_memory} / Integral: {original_integral}
## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Fix the original solution if it was not passed. Optimize the {efficiency_instruction} performance if the original solution was passed.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.
"""


def problem_key(split, problem):
    """Return the stable key used by the resumable score cache."""
    return f"{split}:{problem['problem_id']}"


def _tie_breaker(problem):
    return str(problem["problem_id"])


def _choose_solution(problem, rng):
    solutions = problem.get("solutions") or []
    if not solutions:
        raise ValueError(f"No baseline solution found for Venus problem {problem.get('problem_id', '')}")
    return rng.choice(solutions)


def make_record(problem, split, efficiency_instruction, original_solution, probe_score, rank, total):
    """Convert one Venus problem/optimization pair to verl's schema."""
    percentile = rank / (total - 1) if total > 1 else 0.0
    prompt = AFTERBURNER_TEMPLATE.format(
        target_lang="python",
        problem_description=problem.get("question_content", ""),
        efficiency_instruction=EFFICIENCY_INSTRUCTIONS[efficiency_instruction],
        original_solution=original_solution["code"],
        original_passed=original_solution["passed"],
        original_time=original_solution["time"],
        original_memory=original_solution["memory"],
        original_integral=original_solution["integral"],
    )
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": original_solution},
        "extra_info": {
            "split": split,
            "problem_id": problem["problem_id"],
            "efficiency_instruction": efficiency_instruction,
            "instance": problem,
            "case_multiply": 64,
            "probe_difficulty": float(probe_score),
            "probe_rank": rank,
            "probe_percentile": percentile,
            "curriculum_order": percentile,
        },
    }


def build_records(problems, split, scores, seed=42):
    """Order problems by probe score and expand each into the three Venus tasks."""
    missing = [problem_key(split, problem) for problem in problems if problem_key(split, problem) not in scores]
    if missing:
        preview = ", ".join(missing[:3])
        raise KeyError(f"Missing probe scores for {len(missing)} problems: {preview}")

    ordered = sorted(
        problems,
        key=lambda problem: (float(scores[problem_key(split, problem)]), _tie_breaker(problem)),
    )
    rng = random.Random(seed)
    records = []
    for rank, problem in enumerate(ordered):
        for efficiency_instruction in EFFICIENCY_INSTRUCTIONS:
            records.append(
                make_record(
                    problem,
                    split,
                    efficiency_instruction,
                    _choose_solution(problem, rng),
                    scores[problem_key(split, problem)],
                    rank,
                    len(ordered),
                )
            )
    return records


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

    setattr(sys.modules["__main__"], "RegressionNN", RegressionNN)


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

    def score(self, question):
        messages = [
            {"role": "system", "content": PROBE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
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
        score = scorer.score(problem.get("question_content", ""))
        scores[key] = score
        append_score(cache_path, key, score)
        print(f"{split}: probed {position}/{len(problems)} {key} -> {score:.3f}", flush=True)


def write_split(problems, split, scores, output_path, seed=42):
    from datasets import Dataset

    records = build_records(problems, split, scores, seed)
    if not records:
        raise RuntimeError(f"No Venus problems found in {split}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    first = records[0]["extra_info"]["probe_difficulty"]
    last = records[-1]["extra_info"]["probe_difficulty"]
    print(f"{split}: wrote {len(records)} rows to {output_path} (probe {first:.3f} -> {last:.3f})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATA_SOURCE, help="Hugging Face dataset name or local path")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Seed for baseline-solution selection")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("VENUS_PROBE_DATA_DIR", ROUTE_DIR / "data")),
    )
    parser.add_argument("--max-train", type=int, default=-1, help="Probe the first N training rows")
    parser.add_argument("--max-test", type=int, default=-1, help="Probe the first N test rows")
    return parser.parse_args()


def _load_split(dataset_name, split, limit):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return list(dataset)


def main():
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Probe base model not found: {args.model}")
    if not args.probe.exists():
        raise SystemExit(f"Difficulty probe not found: {args.probe}")

    train = _load_split(args.dataset, args.train_split, args.max_train)
    test = _load_split(args.dataset, args.test_split, args.max_test)
    cache_path = args.output_dir / "probe_scores.jsonl"
    scores = load_score_cache(cache_path)
    scorer_factory = lambda: DifficultyScorer(args.model, args.probe, args.device)
    ensure_scores(train, "train", scores, cache_path, scorer_factory)
    ensure_scores(test, "test", scores, cache_path, scorer_factory)
    write_split(
        train,
        "train",
        scores,
        args.output_dir / "venus_probe_train_easy_to_hard.parquet",
        args.seed,
    )
    write_split(
        test,
        "test",
        scores,
        args.output_dir / "venus_probe_test_easy_to_hard.parquet",
        args.seed,
    )


if __name__ == "__main__":
    main()

# coding: utf-8
# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2025-04-29

import os
import re
import hashlib

SYSTEM_PROMPT = """A conversation between User and Assistant. The user gives a
programming problem and an original solution, then the Assistant improves it
in Python 3. Return only the complete improved code inside <solution>
</solution> as one markdown Python code block. Do not include reasoning,
analysis, or a <thinking> section. Do not write comments or docstrings. Keep
the code concise and finish it before closing the code block."""

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
Return exactly:
<solution>```python
complete improved Python 3 code
```</solution>

Fix the original solution if it did not pass. If it passed, optimize it to be
{efficiency_instruction}. Preserve the starter-code scope. Do not add
explanations, comments, docstrings, package declarations, or unused code.
Start with <solution> immediately and finish the executable code.
"""

EFFICIENCY_INSTRUCTIONS = {
    "time": "time efficient",
    "memory": "memory efficient",
    "integral": "both time and memory efficient",
}


def _choose_solution(problem, efficiency_instruction, seed=42):
    """Choose the same baseline regardless of corpus traversal order."""
    solutions = problem.get("solutions") or []
    if not solutions:
        raise ValueError(f"No baseline solution found for Venus problem {problem.get('problem_id', '')}")
    key = f"{seed}:{problem.get('problem_id', '')}:{efficiency_instruction}".encode()
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(solutions)
    return solutions[index]

if __name__ == "__main__":
    import datasets

    sample_num = 1
    seed = int(os.environ.get("AFTERBURNER_SEED", "42"))
    local_dir = os.path.expanduser(os.environ.get("AFTERBURNER_DATA_DIR", "~/data/venus"))
    data_source = "Elfsong/Venus_Python"
    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split, efficiency_instruction):
        def process_fn(example, efficiency_instruction=efficiency_instruction):
            original_solution = _choose_solution(example, efficiency_instruction, seed)
            efficiency_instruction_str = EFFICIENCY_INSTRUCTIONS[efficiency_instruction]

            afterburner_prompt = AFTERBURNER_TEMPLATE.format(
                target_lang="python",
                problem_description=example['question_content'],
                efficiency_instruction=efficiency_instruction_str,
                original_solution=original_solution['code'],
                original_passed=original_solution['passed'],
                original_time=original_solution['time'],
                original_memory=original_solution['memory'],
                original_integral=original_solution['integral'],
            )

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": afterburner_prompt,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": original_solution
                },
                "extra_info": {
                    "split": split,
                    "problem_id": example['problem_id'],
                    "efficiency_instruction": efficiency_instruction,
                    "instance": example,
                    "case_multiply": 64
                },
            }
            return data

        return process_fn

    # Map for time optimization
    train_time_datasets = [train_dataset.map(function=make_map_fn("train", efficiency_instruction="time"), with_indices=False) for _ in range(sample_num)]
    test_time_datasets = [test_dataset.map(function=make_map_fn("test", efficiency_instruction="time"), with_indices=False) for _ in range(sample_num)]
    
    # Map for memory optimization 
    train_memory_datasets = [train_dataset.map(function=make_map_fn("train", efficiency_instruction="memory"), with_indices=False) for _ in range(sample_num)]
    test_memory_datasets = [test_dataset.map(function=make_map_fn("test", efficiency_instruction="memory"), with_indices=False) for _ in range(sample_num)]
    
    # Map for integral optimization
    train_integral_datasets = [train_dataset.map(function=make_map_fn("train", efficiency_instruction="integral"), with_indices=False) for _ in range(sample_num)]
    test_integral_datasets = [test_dataset.map(function=make_map_fn("test", efficiency_instruction="integral"), with_indices=False) for _ in range(sample_num)]

    train_datasets = train_time_datasets + train_memory_datasets + train_integral_datasets
    test_datasets = test_time_datasets + test_memory_datasets + test_integral_datasets

    train_dataset = datasets.concatenate_datasets(train_datasets)
    test_dataset = datasets.concatenate_datasets(test_datasets)
    
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "venus_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "venus_test.parquet"))

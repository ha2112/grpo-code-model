"""CodeContests correctness reward evaluated through a Monolith endpoint."""

import json
import os
import re


MONOLITH_URL = os.environ.get("MONOLITH_URL", "https://monolith.cool/execute")

RESPONSE_PATTERN = re.compile(
    r"\A\s*<thinking>.*?</thinking>\s*<solution>.*?</solution>\s*\Z",
    re.DOTALL,
)
CODE_PATTERN = re.compile(r"<solution>\s*```(?:python|python3)?\s*(.*?)```\s*</solution>", re.DOTALL)
RESULT_PATTERN = re.compile(r"CODEFORCES_RESULT:(\d+)/(\d+)")


def extract_code(solution_str):
    match = CODE_PATTERN.search(solution_str)
    return match.group(1).strip() if match else ""


def format_score(solution_str):
    return 1.0 if RESPONSE_PATTERN.fullmatch(solution_str) and extract_code(solution_str) else 0.0


def _runner_code(solution_code, tests):
    """Create one sandbox program that runs the submission against every test."""
    return f'''import io
import sys

SOLUTION = {solution_code!r}
TESTS = {tests!r}

def normalize(text):
    return text.strip().split()

passed = 0
for case in TESTS:
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    captured = io.StringIO()
    try:
        sys.stdin = io.StringIO(case["input"])
        sys.stdout = captured
        namespace = {{"__name__": "__main__"}}
        try:
            exec(compile(SOLUTION, "submission.py", "exec"), namespace, namespace)
        except SystemExit:
            pass
        actual = captured.getvalue()
        if normalize(actual) == normalize(case["output"]):
            passed += 1
    except Exception:
        pass
    finally:
        sys.stdin = original_stdin
        sys.stdout = original_stdout

print(f"CODEFORCES_RESULT:{{passed}}/{{len(TESTS)}}")
'''


def _correctness_score(solution_code, tests):
    if not solution_code or not tests:
        return 0.0

    import requests

    payload = {
        "code": _runner_code(solution_code, tests),
        "language": "python",
        "libraries": [],
        "timeout": 90,
        "run_profiling": False,
    }
    try:
        response = requests.post(MONOLITH_URL, json=payload, timeout=95)
        response.raise_for_status()
        stdout = response.json().get("output_dict", {}).get("stdout", "")
        match = RESULT_PATTERN.search(stdout)
        if not match:
            return 0.0
        passed, total = map(int, match.groups())
        return passed / total if total else 0.0
    except Exception as exc:
        print(f"Codeforces reward request failed: {exc}")
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl custom reward: 80% test correctness and 20% response format."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    tests = (ground_truth or {}).get("tests", [])
    correctness = _correctness_score(extract_code(solution_str), tests)
    return 0.8 * correctness + 0.2 * format_score(solution_str)

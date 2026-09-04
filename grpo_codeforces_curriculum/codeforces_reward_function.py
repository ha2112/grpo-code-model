"""CodeContests correctness reward evaluated through a Monolith endpoint."""

import argparse
import json
import os
import re


MONOLITH_URL = os.environ.get("MONOLITH_URL", "https://monolith.cool/execute")

RESPONSE_PATTERN = re.compile(
    r"\A\s*(?:<thinking>.*?</thinking>\s*)?<solution>.*?</solution>\s*\Z",
    re.DOTALL,
)
CODE_PATTERN = re.compile(r"<solution>\s*```(?:python|python3)?\s*(.*?)```\s*</solution>", re.DOTALL)
CLIPPED_CODE_PATTERN = re.compile(
    r"<solution>\s*```(?:python|python3)?\s*(.*?)(?:```|</solution>|\Z)",
    re.DOTALL,
)
RESULT_PATTERN = re.compile(r"CODEFORCES_RESULT:(\d+)/(\d+)")


def extract_code(solution_str):
    match = CODE_PATTERN.search(solution_str)
    if not match:
        match = CLIPPED_CODE_PATTERN.search(solution_str)
    return match.group(1).strip() if match else ""


def format_score(solution_str):
    return 1.0 if RESPONSE_PATTERN.fullmatch(solution_str) and extract_code(solution_str) else 0.0


def _extract_stdout(payload):
    """Read stdout from supported sandbox response layouts."""
    if not isinstance(payload, dict):
        return None

    output_dict = payload.get("output_dict")
    if isinstance(output_dict, dict):
        for key in ("stdout", "output"):
            if isinstance(output_dict.get(key), str):
                return output_dict[key]

    for key in ("stdout", "output"):
        if isinstance(payload.get(key), str):
            return payload[key]

    data = payload.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("stdout"), str):
            return data["stdout"]
        outputs = data.get("outputs")
        if isinstance(outputs, list):
            stdout_parts = [
                item.get("data", "")
                for item in outputs
                if isinstance(item, dict)
                and item.get("type") == "stdout"
                and isinstance(item.get("data"), str)
            ]
            if stdout_parts:
                return "".join(stdout_parts)
    return None


def _response_summary(payload, limit=2000):
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = repr(payload)
    return text if len(text) <= limit else f"{text[:limit]}..."


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


def _correctness_score(solution_code, tests, *, raise_on_error=False):
    if not solution_code or not tests:
        return 0.0

    import requests

    payload = {
        "code": _runner_code(solution_code, tests),
        "language": "python",
        "libraries": [],
        "timeout": 90,
        "run_profiling": True,
    }
    try:
        response = requests.post(MONOLITH_URL, json=payload, timeout=95)
        response.raise_for_status()
        response_payload = response.json()
        stdout = _extract_stdout(response_payload)
        if stdout is None:
            raise RuntimeError(
                f"sandbox returned no stdout; response={_response_summary(response_payload)}"
            )
        match = RESULT_PATTERN.search(stdout)
        if not match:
            if raise_on_error:
                raise RuntimeError("sandbox response did not contain a Codeforces result")
            return 0.0
        passed, total = map(int, match.groups())
        return passed / total if total else 0.0
    except Exception as exc:
        if raise_on_error:
            raise RuntimeError(f"sandbox request failed: {exc}") from exc
        print(f"Codeforces reward request failed: {exc}")
        return 0.0


def check_judge():
    """Fail fast when the configured sandbox cannot execute a trivial program."""
    tests = [{"input": "probe-ok\n", "output": "probe-ok\n"}]
    score = _correctness_score("print(input())", tests, raise_on_error=True)
    if score != 1.0:
        raise RuntimeError(f"sandbox health check returned correctness {score:.3f}, expected 1.000")
    print(f"Codeforces reward sandbox is healthy: {MONOLITH_URL}")


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl custom reward: 80% test correctness and 20% response format."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    tests = (ground_truth or {}).get("tests", [])
    correctness = _correctness_score(extract_code(solution_str), tests)
    return 0.8 * correctness + 0.2 * format_score(solution_str)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="test the configured sandbox endpoint")
    args = parser.parse_args()
    if args.check:
        check_judge()


if __name__ == "__main__":
    main()

import sys
import unittest
from pathlib import Path


ROUTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTE_DIR))

from codeforces_dataset import make_record, select_codeforces  # noqa: E402
from codeforces_reward_function import (  # noqa: E402
    _extract_stdout,
    check_judge,
    extract_code,
    format_score,
)


class CurriculumDatasetTests(unittest.TestCase):
    def test_filters_unrated_and_non_codeforces_then_sorts_by_rating(self):
        problems = [
            {"name": "hard", "source": 2, "cf_rating": 2100, "cf_contest_id": 3, "cf_index": "C"},
            {"name": "other", "source": 5, "cf_rating": 800, "cf_contest_id": 1, "cf_index": "A"},
            {"name": "unrated", "source": 2, "cf_rating": 0, "cf_contest_id": 1, "cf_index": "B"},
            {"name": "easy", "source": 2, "cf_rating": 800, "cf_contest_id": 2, "cf_index": "A"},
        ]

        selected = select_codeforces(problems)

        self.assertEqual([problem["name"] for problem in selected], ["easy", "hard"])
        self.assertEqual([problem["cf_rating"] for problem in selected], [800, 2100])

    def test_record_contains_real_rating_and_all_available_tests(self):
        problem = {
            "name": "A",
            "description": "Solve it.",
            "source": 2,
            "cf_rating": 900,
            "cf_contest_id": 10,
            "cf_index": "A",
            "solutions": {
                "language": ["python3"],
                "solution": ["print(input())"],
            },
            "public_tests": {"input": ["1\n"], "output": ["1\n"]},
            "private_tests": {"input": ["2\n"], "output": ["2\n"]},
            "generated_tests": {"input": [], "output": []},
        }

        record = make_record(problem, "train")

        self.assertEqual(record["extra_info"]["cf_rating"], 900)
        self.assertEqual(record["extra_info"]["curriculum_order"], 900)
        self.assertEqual(len(record["reward_model"]["ground_truth"]["tests"]), 2)
        self.assertIn("print(input())", record["prompt"][1]["content"])
        self.assertEqual(
            record["reward_model"]["ground_truth"]["baseline_solution"],
            "print(input())",
        )
        self.assertIn("Do not write comments or docstrings", record["prompt"][0]["content"])
        self.assertNotIn("<thinking>", record["prompt"][1]["content"])


class RewardParsingTests(unittest.TestCase):
    def test_extracts_python_from_required_solution_block(self):
        response = "<solution>```python\nprint(1)\n```</solution>"

        self.assertEqual(extract_code(response), "print(1)")
        self.assertEqual(format_score(response), 1.0)

    def test_rejects_unstructured_output(self):
        self.assertEqual(extract_code("print(1)"), "")
        self.assertEqual(format_score("print(1)"), 0.0)

    def test_clipped_response_still_exposes_code(self):
        self.assertEqual(extract_code("<solution>```python\nprint(1)"), "print(1)")

    def test_stdout_parser_handles_null_legacy_field(self):
        payload = {"output_dict": None, "stdout": "CODEFORCES_RESULT:1/1\n"}

        self.assertEqual(_extract_stdout(payload), "CODEFORCES_RESULT:1/1\n")

    def test_judge_health_check_rejects_an_incorrect_result(self):
        from unittest.mock import patch

        with patch("codeforces_reward_function._correctness_score", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "expected 1.000"):
                check_judge()


if __name__ == "__main__":
    unittest.main()

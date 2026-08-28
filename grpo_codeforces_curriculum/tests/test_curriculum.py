import sys
import unittest
from pathlib import Path


ROUTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTE_DIR))

from codeforces_dataset import make_record, select_codeforces  # noqa: E402
from codeforces_reward_function import extract_code, format_score  # noqa: E402


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
            "public_tests": {"input": ["1\n"], "output": ["1\n"]},
            "private_tests": {"input": ["2\n"], "output": ["2\n"]},
            "generated_tests": {"input": [], "output": []},
        }

        record = make_record(problem, "train")

        self.assertEqual(record["extra_info"]["cf_rating"], 900)
        self.assertEqual(record["extra_info"]["curriculum_order"], 900)
        self.assertEqual(len(record["reward_model"]["ground_truth"]["tests"]), 2)


class RewardParsingTests(unittest.TestCase):
    def test_extracts_python_from_required_solution_block(self):
        response = "<thinking>Reason.</thinking><solution>```python\nprint(1)\n```</solution>"

        self.assertEqual(extract_code(response), "print(1)")
        self.assertEqual(format_score(response), 1.0)

    def test_rejects_unstructured_output(self):
        self.assertEqual(extract_code("print(1)"), "")
        self.assertEqual(format_score("print(1)"), 0.0)


if __name__ == "__main__":
    unittest.main()

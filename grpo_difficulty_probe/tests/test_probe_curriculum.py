import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROUTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTE_DIR))

from probe_curriculum_dataset import (  # noqa: E402
    build_records,
    ensure_scores,
    load_score_cache,
    problem_key,
    select_codeforces,
)
from codeforces_reward_function import check_judge, extract_code, format_score  # noqa: E402


def problem(name, contest, index, rating):
    return {
        "name": name,
        "description": f"Solve {name}.",
        "source": 2,
        "cf_rating": rating,
        "cf_contest_id": contest,
        "cf_index": index,
        "solutions": {
            "language": ["python3"],
            "solution": ["print(input())"],
        },
        "public_tests": {"input": ["1\n"], "output": ["1\n"]},
        "private_tests": {"input": [], "output": []},
        "generated_tests": {"input": [], "output": []},
    }


class ProbeCurriculumTests(unittest.TestCase):
    def test_keeps_unrated_codeforces_and_rejects_other_sources(self):
        unrated = problem("unrated", 1, "A", 0)
        other = problem("other", 2, "A", 800)
        other["source"] = 5

        self.assertEqual(select_codeforces([unrated, other]), [unrated])

    def test_probe_order_overrides_absolute_rating_order(self):
        low_rating = problem("low-rating", 1, "A", 800)
        high_rating = problem("high-rating", 2, "A", 2400)
        scores = {
            problem_key("train", low_rating): 2500.0,
            problem_key("train", high_rating): 900.0,
        }

        records = build_records([low_rating, high_rating], "train", scores)

        self.assertEqual([row["extra_info"]["name"] for row in records], ["high-rating", "low-rating"])
        self.assertEqual([row["extra_info"]["probe_rank"] for row in records], [0, 1])
        self.assertEqual([row["extra_info"]["probe_percentile"] for row in records], [0.0, 1.0])
        self.assertIn("print(input())", records[0]["prompt"][1]["content"])
        self.assertEqual(
            records[0]["reward_model"]["ground_truth"]["baseline_solution"],
            "print(input())",
        )

    def test_ties_are_deterministic(self):
        second = problem("second", 2, "B", 0)
        first = problem("first", 1, "A", 0)
        scores = {
            problem_key("train", first): 1000.0,
            problem_key("train", second): 1000.0,
        }

        records = build_records([second, first], "train", scores)

        self.assertEqual([row["extra_info"]["name"] for row in records], ["first", "second"])

    def test_reward_format_is_still_correctness_oriented(self):
        response = "<thinking>Reason.</thinking><solution>```python\nprint(1)\n```</solution>"

        self.assertEqual(extract_code(response), "print(1)")
        self.assertEqual(format_score(response), 1.0)

    def test_clipped_response_still_exposes_executable_code(self):
        response = "<thinking>Brief.</thinking><solution>```python\nprint(input())"

        self.assertEqual(extract_code(response), "print(input())")
        self.assertEqual(format_score(response), 0.0)

    def test_prompt_requires_short_reasoning_before_the_program(self):
        item = problem("compact", 4, "A", 0)
        scores = {problem_key("train", item): 1000.0}

        record = build_records([item], "train", scores)[0]

        self.assertIn("at most 100 words", record["prompt"][0]["content"])

    def test_judge_health_check_rejects_an_incorrect_result(self):
        with patch("codeforces_reward_function._correctness_score", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "expected 1.000"):
                check_judge()

    def test_probe_score_cache_resumes_without_rescoring(self):
        import tempfile

        item = problem("cached", 3, "A", 0)

        class FakeScorer:
            calls = 0

            def score(self, description):
                self.calls += 1
                return 1234.5

        scorer = FakeScorer()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "scores.jsonl"
            scores = {}
            ensure_scores([item], "train", scores, cache, lambda: scorer)
            ensure_scores([item], "train", scores, cache, lambda: scorer)

            self.assertEqual(scorer.calls, 1)
            self.assertEqual(load_score_cache(cache), scores)


if __name__ == "__main__":
    unittest.main()

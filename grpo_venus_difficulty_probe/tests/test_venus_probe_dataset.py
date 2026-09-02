import sys
import unittest
from pathlib import Path


ROUTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTE_DIR))

from venus_probe_dataset import (  # noqa: E402
    build_records,
    ensure_scores,
    load_score_cache,
    problem_key,
)


def problem(problem_id, question=None):
    return {
        "problem_id": problem_id,
        "question_content": question or f"Solve {problem_id}.",
        "solutions": [
            {"code": "print(1)", "passed": True, "time": 1.0, "memory": 2.0, "integral": 2.0},
            {"code": "print(2)", "passed": False, "time": 3.0, "memory": 4.0, "integral": 12.0},
        ],
        "test_case_runners": "==Code Submission==",
        "test_case_evaluator": "return expected == actual",
        "test_cases": "[]",
    }


class VenusProbeDatasetTests(unittest.TestCase):
    def test_probe_order_expands_each_problem_into_adjacent_objectives(self):
        hard = problem("hard")
        easy = problem("easy")
        scores = {
            problem_key("train", hard): 2400.0,
            problem_key("train", easy): 900.0,
        }

        records = build_records([hard, easy], "train", scores)

        self.assertEqual([row["extra_info"]["problem_id"] for row in records], ["easy"] * 3 + ["hard"] * 3)
        self.assertEqual(
            [row["extra_info"]["efficiency_instruction"] for row in records[:3]],
            ["time", "memory", "integral"],
        )
        self.assertEqual([row["extra_info"]["probe_rank"] for row in records], [0, 0, 0, 1, 1, 1])
        self.assertEqual([row["extra_info"]["probe_percentile"] for row in records], [0.0] * 3 + [1.0] * 3)

    def test_preserves_original_venus_prompt_reward_and_instance(self):
        item = problem("venus-1", "Add two numbers.")
        scores = {problem_key("test", item): 1234.5}

        record = build_records([item], "test", scores, seed=7)[0]

        self.assertEqual(record["data_source"], "Elfsong/Venus_Python")
        self.assertIn("Add two numbers.", record["prompt"][1]["content"])
        self.assertIn(record["reward_model"]["ground_truth"]["code"], record["prompt"][1]["content"])
        self.assertEqual(record["extra_info"]["instance"], item)
        self.assertEqual(record["extra_info"]["case_multiply"], 64)

    def test_seed_makes_solution_selection_reproducible(self):
        item = problem("stable")
        scores = {problem_key("train", item): 1000.0}

        first = build_records([item], "train", scores, seed=19)
        second = build_records([item], "train", scores, seed=19)

        self.assertEqual(first, second)

    def test_missing_scores_fail_before_building(self):
        with self.assertRaisesRegex(KeyError, "Missing probe scores"):
            build_records([problem("missing")], "train", {})

    def test_probe_score_cache_resumes_without_rescoring(self):
        import tempfile

        item = problem("cached")

        class FakeScorer:
            calls = 0

            def score(self, question):
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

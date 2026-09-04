import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
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
from codeforces_reward_function import (  # noqa: E402
    _correctness_score,
    _extract_stdout,
    check_judge,
    extract_code,
    format_score,
)


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

    def test_reward_accepts_solution_only_format(self):
        response = "<solution>```python\nprint(1)\n```</solution>"

        self.assertEqual(extract_code(response), "print(1)")
        self.assertEqual(format_score(response), 1.0)

    def test_clipped_response_still_exposes_executable_code(self):
        response = "<thinking>Brief.</thinking><solution>```python\nprint(input())"

        self.assertEqual(extract_code(response), "print(input())")
        self.assertEqual(format_score(response), 0.0)

    def test_prompt_requests_program_without_reasoning(self):
        item = problem("compact", 4, "A", 0)
        scores = {problem_key("train", item): 1000.0}

        record = build_records([item], "train", scores)[0]

        self.assertIn("Do not include reasoning", record["prompt"][0]["content"])
        self.assertNotIn("<thinking>", record["prompt"][1]["content"])

    def test_judge_health_check_rejects_an_incorrect_result(self):
        with patch("codeforces_reward_function._correctness_score", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "expected 1.000"):
                check_judge()

    def test_sandbox_stdout_parser_handles_null_legacy_field(self):
        response = {"output_dict": None, "stdout": "CODEFORCES_RESULT:1/1\n"}

        self.assertEqual(_extract_stdout(response), "CODEFORCES_RESULT:1/1\n")

    def test_sandbox_stdout_parser_handles_outputs_list(self):
        response = {
            "output_dict": None,
            "data": {"outputs": [{"type": "stdout", "data": "CODEFORCES_RESULT:1/1\n"}]},
        }

        self.assertEqual(_extract_stdout(response), "CODEFORCES_RESULT:1/1\n")

    def test_correctness_request_enables_monolith_output_collection(self):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"output_dict": {"stdout": "CODEFORCES_RESULT:1/1\n"}}

        def post(url, json, timeout):
            captured.update(json)
            return FakeResponse()

        with patch.dict(sys.modules, {"requests": SimpleNamespace(post=post)}):
            score = _correctness_score("print(input())", [{"input": "1\n", "output": "1\n"}])

        self.assertEqual(score, 1.0)
        self.assertIs(captured["run_profiling"], True)

    def test_16gb_launcher_uses_single_gpu_compatible_lora_sync(self):
        launcher = (ROUTE_DIR / "train_single_gpu_16gb.sh").read_text(encoding="utf-8")

        self.assertIn("actor_rollout_ref.model.lora.merge=False", launcher)
        self.assertNotIn("actor_rollout_ref.model.lora.merge=True", launcher)
        self.assertIn("actor_rollout_ref.rollout.layered_summon=False", launcher)
        self.assertIn("actor_rollout_ref.rollout.quantization=null", launcher)
        self.assertIn("actor_rollout_ref.rollout.load_format=bitsandbytes", launcher)
        self.assertIn(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=bitsandbytes",
            launcher,
        )
        self.assertNotIn("actor_rollout_ref.rollout.quantization=fp8", launcher)
        self.assertIn("uv pip install --python", launcher)

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

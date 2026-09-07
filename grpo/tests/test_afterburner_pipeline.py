import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROUTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTE_DIR))

from afterburner_reward_function import (  # noqa: E402
    _extract_monolith_result,
    _sandbox_infrastructure_error,
    check_judge,
    compute_score,
    extract_solution_code,
    single_thinking_solution_format,
)
from afterburner_dataset import SYSTEM_PROMPT, AFTERBURNER_TEMPLATE  # noqa: E402


class AfterburnerRewardTests(unittest.TestCase):
    def test_prompt_requests_concise_solution_only_output(self):
        self.assertIn("Do not include reasoning", SYSTEM_PROMPT)
        self.assertIn("Do not write comments or docstrings", SYSTEM_PROMPT)
        self.assertNotIn("<thinking>", AFTERBURNER_TEMPLATE)

    def test_solution_only_format_is_valid(self):
        response = "<solution>```python\nprint(1)\n```</solution>"

        self.assertTrue(single_thinking_solution_format(response))
        self.assertEqual(extract_solution_code(response), "print(1)")

    def test_clipped_solution_code_remains_judgeable(self):
        response = "<solution>```python\nprint(input())"

        self.assertFalse(single_thinking_solution_format(response))
        self.assertEqual(extract_solution_code(response), "print(input())")

    def test_parser_accepts_current_monolith_response_shape(self):
        payload = {
            "status": "success",
            "output_dict": None,
            "data": {
                "outputs": [{"type": "stdout", "data": "Success\n"}],
                "duration": 0.25,
                "peak_memory": 1024,
                "integral": 256,
            },
        }

        result = _extract_monolith_result(payload)

        self.assertTrue(result["passed"])
        self.assertEqual(result["time"], 0.25)
        self.assertEqual(result["memory"], 1024)
        self.assertEqual(result["integral"], 256)

    def test_docker_storage_failure_is_infrastructure_error(self):
        error = "http+docker://localhost: no space left on device"

        self.assertEqual(_sandbox_infrastructure_error({"error": error}), error)

    def test_health_check_rejects_failed_execution(self):
        with patch("afterburner_reward_function.performance_evalution", return_value={"passed": False}):
            with self.assertRaisesRegex(RuntimeError, "health check failed"):
                check_judge()

    def test_scalar_reward_entrypoint_preserves_batch_reward(self):
        with patch(
            "afterburner_reward_function.afterburner_reward_fn_batch",
            return_value=[0.75],
        ) as batch_reward:
            score = compute_score("venus", "response", {"passed": True}, {"problem_id": "1"})

        self.assertEqual(score, 0.75)
        batch_reward.assert_called_once_with(
            ["venus"],
            ["response"],
            [{"passed": True}],
            [{"problem_id": "1"}],
        )


if __name__ == "__main__":
    unittest.main()

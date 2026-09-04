import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROUTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTE_DIR))

from afterburner_reward_function import _extract_monolith_result, check_judge  # noqa: E402


class AfterburnerRewardTests(unittest.TestCase):
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

    def test_health_check_rejects_failed_execution(self):
        with patch("afterburner_reward_function.performance_evalution", return_value={"passed": False}):
            with self.assertRaisesRegex(RuntimeError, "health check failed"):
                check_judge()


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_DIR = Path(__file__).resolve().parents[1]


def load_recovery():
    path = REPO_DIR / "evaluation/recover_four_routes.py"
    spec = importlib.util.spec_from_file_location("four_route_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recovery = load_recovery()

    def sample_record(self):
        return {
            "route": "probe",
            "suite": "codeforces",
            "example_id": "1A",
            "objective": "code",
            "completion_index": 0,
            "seed": 42,
            "prompt_tokens": 100,
            "output_tokens": 20,
            "output": "<solution>```python\nprint(1)\n```</solution>",
            "reward": 0.0,
            "correctness": 0.0,
            "status": "evaluated",
        }

    def test_generation_record_discards_corrupted_scores(self):
        cleaned = self.recovery.generation_record(self.sample_record())

        self.assertEqual(cleaned["output"], self.sample_record()["output"])
        self.assertNotIn("reward", cleaned)
        self.assertNotIn("correctness", cleaned)
        self.assertNotIn("status", cleaned)

    def test_initialize_generations_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.jsonl"
            destination = root / "recovery/generations.jsonl"
            source.write_text(json.dumps(self.sample_record()) + "\n", encoding="utf-8")

            records, keys = self.recovery.initialize_generations(source, destination)

            self.assertEqual(len(records), 1)
            self.assertEqual(len(keys), 1)
            self.assertIn("reward", json.loads(source.read_text(encoding="utf-8")))
            self.assertNotIn("reward", json.loads(destination.read_text(encoding="utf-8")))

    def test_score_generation_replaces_metrics(self):
        generation = self.recovery.generation_record(self.sample_record())
        reward = SimpleNamespace(
            _correctness_score=lambda code, tests, raise_on_error: 1.0,
            extract_code=lambda text: text,
            format_score=lambda text: 1.0,
        )
        case = {"ground_truth": {"tests": [{}]}, "extra_info": {}}

        scored = self.recovery.score_generation(generation, case, reward, None)

        self.assertEqual(scored["correctness"], 1.0)
        self.assertEqual(scored["reward"], 1.0)
        self.assertEqual(scored["output"], generation["output"])


if __name__ == "__main__":
    unittest.main()

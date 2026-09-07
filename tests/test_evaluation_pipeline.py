import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def load_evaluator():
    path = REPO_DIR / "evaluation/evaluate_four_routes.py"
    spec = importlib.util.spec_from_file_location("four_route_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

    def test_evenly_spaced_selection_covers_endpoints(self):
        items = [{"value": value} for value in range(10)]
        selected = self.evaluator.evenly_spaced(items, 4)
        self.assertEqual([item["value"] for item in selected], [0, 3, 6, 9])

    def test_venus_selection_is_balanced_by_objective(self):
        items = []
        for objective in ("time", "memory", "integral"):
            for index in range(10):
                items.append(
                    {
                        "objective": objective,
                        "example_id": f"{index}:{objective}",
                    }
                )
        selected = self.evaluator.stratified_venus(items, 9)
        counts = {
            objective: sum(item["objective"] == objective for item in selected)
            for objective in ("time", "memory", "integral")
        }
        self.assertEqual(counts, {"time": 3, "memory": 3, "integral": 3})

    def test_pass_at_k_estimator(self):
        self.assertEqual(self.evaluator.pass_at_k(0, 4, 4), 0.0)
        self.assertEqual(self.evaluator.pass_at_k(1, 4, 4), 1.0)
        self.assertAlmostEqual(self.evaluator.pass_at_k(2, 4, 1), 0.5)

    def test_aggregate_keeps_suites_separate(self):
        records = []
        for route in self.evaluator.ROUTES:
            for suite in self.evaluator.SUITES:
                for completion in range(4):
                    passed = completion == 0
                    records.append(
                        {
                            "route": route,
                            "suite": suite,
                            "example_id": "case",
                            "completion_index": completion,
                            "passed": passed,
                            "correctness": float(passed),
                            "format": 1.0,
                            "reward": float(passed),
                            "improvement": 0.5 if suite == "venus" else None,
                        }
                    )
        summary = self.evaluator.aggregate(records, 4, 42)
        self.assertEqual(len(summary), 8)
        for row in summary:
            self.assertAlmostEqual(row["pass_at_1"], 0.25)
            self.assertEqual(row["pass_at_4"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.svg"
            self.evaluator.write_svg_plot(path, summary)
            self.assertIn("Held-out evaluation", path.read_text(encoding="utf-8"))

    def test_launcher_uses_shared_step_600_root(self):
        launcher = (REPO_DIR / "evaluate_four_routes.sh").read_text(encoding="utf-8")
        self.assertIn("deadline-600-steps-seed42-v1", launcher)
        self.assertIn("global_step_600/actor", launcher)
        for route in self.evaluator.ROUTES:
            self.assertIn(route, launcher)


if __name__ == "__main__":
    unittest.main()

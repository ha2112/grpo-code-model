import unittest
import importlib.util
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FourRouteLauncherTests(unittest.TestCase):
    def test_all_routes_have_16gb_entrypoints(self):
        expected = {
            "grpo/train_single_gpu_16gb.sh": "venus",
            "grpo_codeforces_curriculum/train_single_gpu_16gb.sh": "absolute",
            "grpo_difficulty_probe/train_single_gpu_16gb.sh": "probe",
            "grpo_venus_difficulty_probe/train_single_gpu_16gb.sh": "venus-probe",
        }

        for relative_path, route in expected.items():
            launcher = (REPO_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn(f'train_grpo_16gb.sh" {route}', launcher)

    def test_shared_launcher_uses_verified_memory_settings(self):
        launcher = (REPO_DIR / "train_grpo_16gb.sh").read_text(encoding="utf-8")

        for route in ("venus", "absolute", "probe", "venus-probe"):
            self.assertIn(route, launcher)
        self.assertIn("actor_rollout_ref.model.lora.merge=False", launcher)
        self.assertIn("actor_rollout_ref.rollout.quantization=null", launcher)
        self.assertIn("engine_kwargs.vllm.quantization=bitsandbytes", launcher)
        self.assertIn("data.max_response_length=1024", launcher)
        self.assertIn("trainer.save_freq=250", launcher)
        self.assertIn("trainer.test_freq=-1", launcher)

    def test_comparison_runner_includes_venus_probe(self):
        launcher = (REPO_DIR / "run_grpo_comparison.sh").read_text(encoding="utf-8")
        example = (REPO_DIR / ".env.example").read_text(encoding="utf-8")

        self.assertIn("venus-probe", launcher)
        self.assertIn("VENUS_PROBE_DATA_DIR", launcher)
        self.assertIn("VENUS_PROBE_DATA_DIR", example)

    def test_venus_routes_choose_the_same_baseline_per_objective(self):
        original = load_module("original_venus_dataset", REPO_DIR / "grpo/afterburner_dataset.py")
        probe = load_module(
            "probe_venus_dataset",
            REPO_DIR / "grpo_venus_difficulty_probe/venus_probe_dataset.py",
        )
        problem = {
            "problem_id": "stable",
            "solutions": [{"code": "first"}, {"code": "second"}, {"code": "third"}],
        }

        for objective in ("time", "memory", "integral"):
            self.assertEqual(
                original._choose_solution(problem, objective, seed=42),
                probe._choose_solution(problem, objective, seed=42),
            )


if __name__ == "__main__":
    unittest.main()

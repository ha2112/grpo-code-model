import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def load_verifier():
    path = REPO_DIR / "evaluation/verify_generations.py"
    spec = importlib.util.spec_from_file_location("verify_generations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerifyGenerationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = load_verifier()

    def write_fixture(self, root: Path, duplicate_key=False, duplicate_output=False):
        manifest = {
            suite: [{"example_id": f"{suite}-case"}]
            for suite in self.verifier.SUITES
        }
        manifest_path = root / "sample_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        records = []
        for route in self.verifier.ROUTES:
            for suite in self.verifier.SUITES:
                for completion in range(4):
                    suffix = 0 if duplicate_output and completion == 1 else completion
                    records.append(
                        {
                            "route": route,
                            "suite": suite,
                            "example_id": f"{suite}-case",
                            "objective": "code",
                            "completion_index": completion,
                            "seed": 42,
                            "prompt_tokens": 10,
                            "output_tokens": 10,
                            "output": f"<solution>```python\nprint({route!r}, {suffix})\n```</solution>",
                        }
                    )
        if duplicate_key:
            records.append(dict(records[0]))
        generations = root / "generations.jsonl"
        generations.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return generations, manifest_path

    def test_complete_unique_file_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            generations, manifest = self.write_fixture(Path(directory))
            report = self.verifier.inspect_generations(generations, manifest, 4)
        self.assertFalse(report["errors"])
        self.assertFalse(report["missing"])
        self.assertEqual(report["duplicate_completions"], 0)
        self.assertEqual(len(report["records"]), 32)

    def test_duplicate_key_is_an_integrity_error(self):
        with tempfile.TemporaryDirectory() as directory:
            generations, manifest = self.write_fixture(Path(directory), duplicate_key=True)
            report = self.verifier.inspect_generations(generations, manifest, 4)
        self.assertTrue(any("duplicate key" in error for error in report["errors"]))

    def test_duplicate_output_is_reported_but_not_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            generations, manifest = self.write_fixture(Path(directory), duplicate_output=True)
            report = self.verifier.inspect_generations(generations, manifest, 4)
        self.assertFalse(report["errors"])
        self.assertEqual(report["duplicate_completions"], 8)

    def test_non_string_output_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            generations, manifest = self.write_fixture(Path(directory))
            records = [json.loads(line) for line in generations.read_text().splitlines()]
            records[0]["output"] = None
            generations.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            report = self.verifier.inspect_generations(generations, manifest, 4)
        self.assertTrue(any("empty output" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()

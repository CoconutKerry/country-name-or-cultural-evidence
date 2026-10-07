"""Model-free package gates for the completed Llama GGUF experiment."""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts import package_llama_gguf_full as package


class LlamaPackageContractTests(unittest.TestCase):
    def test_exactly_one_verified_gguf_identity_is_accepted(self):
        expected = package.expected_model_files()
        self.assertEqual(
            expected,
            [
                {
                    "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    "size_bytes": 4_920_739_232,
                    "sha256": (
                        "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"
                    ),
                }
            ],
        )
        self.assertEqual(
            package.normalize_model_files(expected, source="test"), expected
        )
        with self.assertRaisesRegex(ValueError, "exact pin"):
            package.normalize_model_files(
                [{**expected[0], "size_bytes": expected[0]["size_bytes"] - 1}],
                source="test",
            )

    def test_weights_runtime_fragments_and_legacy_outputs_are_excluded(self):
        excluded = (
            Path("runtime/models/model.gguf"),
            Path(".venv/lib/site.py"),
            Path("build/bin/llama-cli"),
            Path("downloads/model.gguf.part"),
            Path("experiments/smoke_genuine_qwen_gguf/results.jsonl"),
            Path("audit/GENUINE_QWEN_GGUF_SMOKE_REPORT.md"),
        )
        self.assertTrue(all(package._is_excluded_file(path) for path in excluded))
        self.assertFalse(
            package._is_excluded_file(
                Path("experiments/smoke_genuine_llama_gguf/results.jsonl")
            )
        )
        self.assertFalse(
            package._is_excluded_file(
                Path("experiments/llama_gguf_full/results.jsonl")
            )
        )

    def test_selected_repository_members_respect_exclusion_policy(self):
        files = package.selected_repository_files()
        self.assertTrue(files)
        for path in files:
            relative = path.relative_to(package.PROJECT_ROOT)
            self.assertFalse(package._is_excluded_file(relative), relative)
            self.assertLessEqual(path.stat().st_size, package.MAX_PACKAGE_MEMBER_BYTES)


if __name__ == "__main__":
    unittest.main()

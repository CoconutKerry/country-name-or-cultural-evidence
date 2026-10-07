"""Regression gates for the Llama source-only handoff package."""

from __future__ import annotations

from pathlib import Path
import re
import unittest

from scripts import run_genuine_llama_gguf_smoke as smoke


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_GATE_FILES = (
    ROOT / "src/colab_runner.py",
    ROOT / "scripts/build_colab_notebook.py",
    ROOT / "cultural_alignment_audit_colab.ipynb",
)


class SourcePackageCompletenessTests(unittest.TestCase):
    def test_smoke_protected_gate_files_are_packaged(self):
        missing = [str(path.relative_to(ROOT)) for path in PROTECTED_GATE_FILES if not path.is_file()]
        self.assertEqual(missing, [])
        observed = smoke.file_hashes(PROTECTED_GATE_FILES)
        self.assertEqual(set(observed), {
            "src/colab_runner.py",
            "scripts/build_colab_notebook.py",
            "cultural_alignment_audit_colab.ipynb",
        })
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in observed.values()))

    def test_runbook_packages_outside_project_root(self):
        text = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
        self.assertIn(
            "python scripts/package_llama_gguf_full.py ../llama-3.1-8b-instruct-full-results.zip",
            text,
        )


if __name__ == "__main__":
    unittest.main()

"""Dependency-free tests for the isolated llama.cpp/GGUF Python adapter."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import types
import unittest

from src.gguf_runner import (
    GGUFRunner,
    QWEN25_DEFAULT_SYSTEM_MESSAGE,
    serialize_qwen25_chatml,
)


class GGUFRunnerTests(unittest.TestCase):
    def make_runner(self) -> tuple[GGUFRunner, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        helper = root / "helper"
        model = root / "model.gguf"
        helper.write_bytes(b"placeholder")
        model.write_bytes(b"placeholder")
        runner = GGUFRunner(
            helper_path=helper,
            model_path=model,
            model_identifier="Qwen/test:Q4_K_M",
            repository="Qwen/test",
            revision="a" * 40,
            threads=1,
        )
        return runner, temporary

    @staticmethod
    def response(labels: list[str]) -> dict:
        traces = {
            "A": {
                "token_ids": [101, 102],
                "sequence_log_probability": -0.25,
            },
            "B": {
                "token_ids": [103],
                "sequence_log_probability": -1.25,
            },
        }
        return {
            "serialized_prompt": "prompt\n",
            "candidates": {label: traces[label] for label in labels},
        }

    def test_complete_sequence_scores_map_to_displayed_semantics(self) -> None:
        runner, temporary = self.make_runner()
        self.addCleanup(temporary.cleanup)

        def fake_score_messages(_self, _messages, labels):
            return self.response(list(labels))

        runner.score_messages = types.MethodType(fake_score_messages, runner)
        prediction = runner.predict_distribution(
            "question",
            ["Strongly disagree", "Strongly agree"],
        )
        expected_a = math.exp(-0.25) / (math.exp(-0.25) + math.exp(-1.25))
        self.assertAlmostEqual(prediction["Strongly disagree"], expected_a)
        self.assertAlmostEqual(prediction["Strongly agree"], 1.0 - expected_a)
        self.assertEqual(runner.last_scoring["raw_label_log_scores"]["A"], -0.25)
        self.assertEqual(runner.last_scoring["candidates"]["A"]["token_ids"], [101, 102])
        self.assertEqual(runner.backend, "llama.cpp")
        self.assertFalse(runner.is_synthetic)

    def test_missing_candidate_label_fails_closed(self) -> None:
        runner, temporary = self.make_runner()
        self.addCleanup(temporary.cleanup)

        def incomplete_response(_self, _messages, _labels):
            return self.response(["A"])

        runner.score_messages = types.MethodType(incomplete_response, runner)
        with self.assertRaisesRegex(RuntimeError, "every displayed option label"):
            runner.predict_distribution("question", ["one", "two"])

    def test_qwen_default_system_serialization_is_explicit_and_byte_exact(self) -> None:
        messages = [
            {"role": "system", "content": QWEN25_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": "Question\nAnswer:"},
        ]
        self.assertEqual(
            serialize_qwen25_chatml(messages),
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            "<|im_end|>\n"
            "<|im_start|>user\nQuestion\nAnswer:<|im_end|>\n"
            "<|im_start|>assistant\n",
        )


if __name__ == "__main__":
    unittest.main()

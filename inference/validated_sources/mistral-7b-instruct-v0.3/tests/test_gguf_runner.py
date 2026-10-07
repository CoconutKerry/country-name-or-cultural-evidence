"""Dependency-free tests for the isolated llama.cpp/GGUF Python adapter."""

from __future__ import annotations

from io import StringIO
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

from src.gguf_runner import (
    GGUFRunner,
    MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
    MistralTemplateEquivalenceError,
    MISTRAL_V03_CHAT_PROFILE,
    QWEN25_DEFAULT_SYSTEM_MESSAGE,
    build_mistral_template_equivalence_diagnostic,
    serialize_mistral_v03_chat,
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

    def test_mistral_v03_user_only_serialization_is_byte_exact(self) -> None:
        messages = [{"role": "user", "content": "Question\nAnswer:"}]
        self.assertEqual(
            serialize_mistral_v03_chat(messages),
            "<s>[INST] Question\nAnswer: [/INST]",
        )
        with self.assertRaisesRegex(ValueError, "alternate"):
            serialize_mistral_v03_chat(
                [{"role": "system", "content": "forbidden"}]
            )

    def test_mistral_profile_uses_user_only_messages(self) -> None:
        runner, temporary = self.make_runner()
        self.addCleanup(temporary.cleanup)
        runner.chat_profile = MISTRAL_V03_CHAT_PROFILE
        captured = {}

        def fake_score_messages(_self, messages, labels):
            captured["messages"] = messages
            return self.response(list(labels))

        runner.score_messages = types.MethodType(fake_score_messages, runner)
        runner.predict_distribution("Question\nAnswer:", ["yes", "no"])
        self.assertEqual(
            captured["messages"],
            [{"role": "user", "content": "Question\nAnswer:"}],
        )

    def test_mistral_byte_mismatch_is_accepted_only_when_tokens_and_bos_match(self) -> None:
        actual = "[INST] Question\nAnswer: [/INST]"
        expected = "<s>[INST] Question\nAnswer: [/INST]"
        diagnostic = build_mistral_template_equivalence_diagnostic(
            actual, expected, [1, 3, 901, 4], [1, 3, 901, 4], 1
        )
        self.assertTrue(diagnostic["accepted"])
        self.assertTrue(diagnostic["token_ids_equivalent"])
        self.assertEqual(
            diagnostic["first_differing_byte"],
            {"offset": 0, "actual": ord("["), "expected": ord("<")},
        )
        self.assertEqual(
            diagnostic["actual_utf8_byte_length"] + 3,
            diagnostic["expected_utf8_byte_length"],
        )
        self.assertEqual(diagnostic["actual_serialized_prompt_repr"], repr(actual))
        self.assertEqual(
            diagnostic["expected_serialized_prompt_repr"], repr(expected)
        )
        self.assertNotEqual(diagnostic["actual_sha256"], diagnostic["expected_sha256"])

    def test_mistral_helper_request_and_token_equivalence_gate(self) -> None:
        runner, temporary = self.make_runner()
        self.addCleanup(temporary.cleanup)
        runner.chat_profile = MISTRAL_V03_CHAT_PROFILE
        stdin = StringIO()
        runner._process = types.SimpleNamespace(stdin=stdin)
        actual = "[INST] Question\nAnswer: [/INST]"
        expected = "<s>" + actual
        response = {
            "type": "result",
            "request_id": "1",
            "structured_messages": [
                {"role": "user", "content": "Question\nAnswer:"}
            ],
            "serialized_prompt": actual,
            "canonical_serialized_prompt": expected,
            "prompt_token_ids": [1, 3, 901, 4],
            "canonical_prompt_token_ids": [1, 3, 901, 4],
            "bos_token_id": 1,
            "prompt_bos_token_count": 1,
            "canonical_prompt_bos_token_count": 1,
            "prompt_starts_with_bos": True,
            "canonical_prompt_starts_with_bos": True,
            "add_special_tokens": True,
            "tokenization_add_special": True,
            "canonical_tokenization_add_special": False,
            "candidate_prefix": "",
            "chat_template_source": "embedded_gguf_metadata",
            "template_equivalence_policy": MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
            "template_equivalence_preflight_passed": True,
            "generated_answer": None,
            "candidates": {
                "A": {"continuation": "A", "sequence_log_probability": -0.1},
                "B": {"continuation": "B", "sequence_log_probability": -0.2},
            },
        }

        def fake_read_response(_self):
            return response

        runner._read_response = types.MethodType(fake_read_response, runner)
        scored = runner.score_messages(
            [{"role": "user", "content": "Question\nAnswer:"}], ["A", "B"]
        )
        request = json.loads(stdin.getvalue())
        self.assertEqual(request["canonical_serialized_prompt"], expected)
        self.assertIs(request["add_special_tokens"], True)
        self.assertIs(request["canonical_tokenization_add_special"], False)
        self.assertEqual(
            request["template_equivalence_policy"],
            MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
        )
        self.assertTrue(scored["template_equivalence"]["accepted"])

    def test_mistral_gate_rejects_token_mismatch_or_double_bos(self) -> None:
        for name, actual_ids in (
            ("token mismatch", [1, 3, 902, 4]),
            ("double BOS", [1, 1, 3, 901, 4]),
        ):
            with self.subTest(name=name):
                runner, temporary = self.make_runner()
                self.addCleanup(temporary.cleanup)
                runner.chat_profile = MISTRAL_V03_CHAT_PROFILE
                runner._process = types.SimpleNamespace(stdin=StringIO())
                actual = "[INST] Question\nAnswer: [/INST]"
                response = {
                    "type": "template_validation_error",
                    "request_id": "1",
                    "serialized_prompt": actual,
                    "canonical_serialized_prompt": "<s>" + actual,
                    "prompt_token_ids": actual_ids,
                    "canonical_prompt_token_ids": [1, 3, 901, 4],
                    "bos_token_id": 1,
                    "add_special_tokens": True,
                    "tokenization_add_special": True,
                    "canonical_tokenization_add_special": False,
                    "candidate_prefix": "",
                    "chat_template_source": "embedded_gguf_metadata",
                    "template_equivalence_policy": (
                        MISTRAL_TEMPLATE_EQUIVALENCE_POLICY
                    ),
                    "template_equivalence_preflight_passed": False,
                    "generated_answer": None,
                }

                def fake_read_response(_self):
                    return response

                runner._read_response = types.MethodType(fake_read_response, runner)
                with self.assertRaises(MistralTemplateEquivalenceError):
                    runner.score_messages(
                        [{"role": "user", "content": "Question\nAnswer:"}],
                        ["A", "B"],
                    )
                preserved = runner.get_last_template_equivalence_diagnostic()
                self.assertFalse(preserved["accepted"])
                self.assertEqual(preserved["actual_token_ids"], actual_ids)

    def test_helper_requires_explicit_continuations_and_special_token_mode(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src/gguf_score_helper.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('request.at("candidate_continuations")', source)
        self.assertIn('request.contains("add_special_tokens")', source)
        self.assertIn('request.at("tokenization_add_special")', source)
        self.assertIn('request.at("candidate_prefix")', source)
        self.assertIn('request.at("canonical_serialized_prompt")', source)
        self.assertIn('canonical_prompt_tokens', source)
        self.assertIn('prompt_bos_token_count == 1U', source)
        self.assertNotIn("label_continuation(", source)


if __name__ == "__main__":
    unittest.main()

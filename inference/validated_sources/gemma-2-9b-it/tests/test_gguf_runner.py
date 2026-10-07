"""Dependency-free tests for the isolated llama.cpp/GGUF Python adapter."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

from src.gguf_runner import (
    GEMMA2_ASSISTANT_ANSWER_SUFFIX,
    GEMMA2_EMBEDDED_CHAT_TEMPLATE,
    GEMMA2_EMBEDDED_CHAT_TEMPLATE_SHA256,
    Gemma2GGUFRunner,
    GGUFRunner,
    QWEN25_DEFAULT_SYSTEM_MESSAGE,
    first_serialization_difference,
    serialize_gemma2_chat,
    serialize_qwen25_chatml,
    serialization_diagnostic,
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

    def test_gemma_user_only_serialization_is_byte_exact(self) -> None:
        self.assertEqual(
            serialize_gemma2_chat([{"role": "user", "content": "  Question\nAnswer:  "}]),
            "<bos><start_of_turn>user\nQuestion\nAnswer:<end_of_turn>\n"
            "<start_of_turn>model\n",
        )
        with self.assertRaisesRegex(ValueError, "does not support system"):
            serialize_gemma2_chat([{"role": "system", "content": "system"}])

    def test_gemma_runner_requires_exact_embedded_template(self) -> None:
        self.assertEqual(
            hashlib.sha256(GEMMA2_EMBEDDED_CHAT_TEMPLATE.encode()).hexdigest(),
            GEMMA2_EMBEDDED_CHAT_TEMPLATE_SHA256,
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        helper = root / "helper"
        model = root / "model.gguf"
        helper.write_bytes(b"placeholder")
        model.write_bytes(b"placeholder")
        runner = Gemma2GGUFRunner(
            helper_path=helper,
            model_path=model,
            model_identifier="bartowski/gemma-test:Q4_K_M",
            repository="bartowski/gemma-test",
            revision="b" * 40,
            threads=1,
        )
        runner._validate_loaded_chat_template(GEMMA2_EMBEDDED_CHAT_TEMPLATE)
        with self.assertRaisesRegex(RuntimeError, "pinned Gemma 2"):
            runner._validate_loaded_chat_template("different")
        self.assertEqual(
            runner._messages_for_prompt("question"),
            [{"role": "user", "content": "question"}],
        )

    def test_serialization_difference_records_character_byte_and_eof(self) -> None:
        difference = first_serialization_difference("<bos>é", "<bos>e")
        self.assertEqual(difference["character_index"], 5)
        self.assertEqual(difference["utf8_byte_index"], 5)
        self.assertEqual(difference["llama_cpp_codepoint"], "U+00E9")
        self.assertEqual(difference["independent_codepoint"], "U+0065")
        eof = first_serialization_difference("abc", "abcd")
        self.assertEqual(eof["character_index"], 3)
        self.assertIsNone(eof["llama_cpp_character"])
        self.assertEqual(eof["independent_character"], "d")
        self.assertIsNone(first_serialization_difference("same", "same"))

    def test_token_equivalent_formatter_difference_is_explicitly_audited(self) -> None:
        actual = "<start_of_turn>user\nQuestion\nAnswer:<end_of_turn>\n" + (
            GEMMA2_ASSISTANT_ANSWER_SUFFIX
        )
        independent = "<bos>" + actual
        diagnostic = serialization_diagnostic(
            llama_cpp_text=actual,
            independent_text=independent,
            llama_cpp_token_ids=[2, 106, 100, 107, 106],
            independent_token_ids=[2, 106, 100, 107, 106],
            bos_token_id=2,
            required_suffix=GEMMA2_ASSISTANT_ANSWER_SUFFIX,
            required_leading_bos_count=1,
            allow_token_id_equivalence=True,
        )
        self.assertFalse(diagnostic["byte_exact"])
        self.assertTrue(diagnostic["token_id_equivalent"])
        self.assertTrue(diagnostic["equivalence_accepted"])
        self.assertEqual(diagnostic["verification_mode"], "token_id_equivalent")
        self.assertEqual(diagnostic["first_difference"]["character_index"], 1)

    def test_gemma_diagnostic_selects_add_special_true_for_omitted_text_bos(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        helper = root / "helper"
        model = root / "model.gguf"
        helper.write_bytes(b"placeholder")
        model.write_bytes(b"placeholder")
        runner = Gemma2GGUFRunner(
            helper_path=helper,
            model_path=model,
            model_identifier="bartowski/gemma-test:Q4_K_M",
            repository="bartowski/gemma-test",
            revision="b" * 40,
            threads=1,
        )
        runner.runtime_metadata = {
            "chat_template_sha256": GEMMA2_EMBEDDED_CHAT_TEMPLATE_SHA256,
        }
        independent = serialize_gemma2_chat(
            [{"role": "user", "content": "Question\nAnswer:"}]
        )
        actual = independent.removeprefix("<bos>")
        response = {
            "structured_messages": [
                {"role": "user", "content": "Question\nAnswer:"}
            ],
            "serialized_prompt": actual,
            "reference_serialized_prompt": independent,
            "prompt_token_ids": [2, 106, 100, 107, 106],
            "prompt_token_ids_add_special_false": [106, 100, 107, 106],
            "prompt_token_ids_add_special_true": [2, 106, 100, 107, 106],
            "reference_prompt_token_ids": [2, 106, 100, 107, 106],
            "selected_actual_tokenization_mode": "add_special_true",
            "bos_token_id": 2,
            "leading_bos_token_count": 1,
            "serialization_contract_checks": {
                "serialization_equivalence": True,
            },
            "serialization_contract_pass": True,
        }
        diagnostic = runner._build_serialization_diagnostic(
            response,
            independent_serialization=independent,
            structured_messages=response["structured_messages"],
        )
        self.assertEqual(diagnostic["status"], "PASS")
        self.assertEqual(
            diagnostic["serialization"]["selected_actual_tokenization_mode"],
            "add_special_true",
        )
        candidates = diagnostic["serialization"][
            "llama_cpp_tokenization_candidates"
        ]
        self.assertFalse(candidates["add_special_false"]["canonical_and_bos_safe"])
        self.assertTrue(candidates["add_special_true"]["canonical_and_bos_safe"])
        self.assertEqual(
            diagnostic["serialization"]["llama_cpp"]["sha256"],
            hashlib.sha256(actual.encode()).hexdigest(),
        )

    def test_gemma_diagnostic_rejects_wrong_suffix_or_unsafe_bos(self) -> None:
        diagnostic = serialization_diagnostic(
            llama_cpp_text="<bos>wrong suffix",
            independent_text="<bos>wrong suffix",
            llama_cpp_token_ids=[2, 2, 100],
            independent_token_ids=[2, 2, 100],
            bos_token_id=2,
            required_suffix=GEMMA2_ASSISTANT_ANSWER_SUFFIX,
            required_leading_bos_count=1,
            allow_token_id_equivalence=True,
        )
        self.assertFalse(diagnostic["leading_bos_requirement_verified"])
        self.assertFalse(diagnostic["exact_required_suffix_verified"])

    def test_live_preflight_uses_serialize_only_and_records_no_inference(self) -> None:
        runner, temporary = self.make_runner()
        self.addCleanup(temporary.cleanup)

        class RecordingInput:
            def __init__(self) -> None:
                self.value = ""

            def write(self, value: str) -> None:
                self.value += value

            def flush(self) -> None:
                pass

        fake_input = RecordingInput()
        runner._process = types.SimpleNamespace(stdin=fake_input)
        serialized = serialize_qwen25_chatml(
            runner._messages_for_prompt("Question\nAnswer:")
        )
        response = {
            "type": "serialization",
            "request_id": "1",
            "structured_messages": runner._messages_for_prompt(
                "Question\nAnswer:"
            ),
            "serialized_prompt": serialized,
            "reference_serialized_prompt": serialized,
            "prompt_token_ids": [10, 11],
            "prompt_token_ids_add_special_false": [10, 11],
            "prompt_token_ids_add_special_true": [2, 10, 11],
            "reference_prompt_token_ids": [10, 11],
            "selected_actual_tokenization_mode": "add_special_false",
            "bos_token_id": 2,
            "leading_bos_token_count": 0,
            "serialization_contract_checks": {"serialization_equivalence": True},
            "serialization_contract_pass": True,
            "inference_performed": False,
        }
        runner.runtime_metadata = {"chat_template_sha256": "unrestricted-test"}
        runner._read_response = types.MethodType(lambda _self: response, runner)
        diagnostic = runner.serialization_preflight("Question\nAnswer:")
        request = json.loads(fake_input.value)
        self.assertEqual(request["operation"], "serialize_only")
        self.assertNotIn("labels", request)
        self.assertEqual(diagnostic["status"], "PASS")
        self.assertFalse(diagnostic["inference_performed"])
        self.assertTrue(diagnostic["checks"]["helper_confirmed_no_inference"])
        runner._process = None


if __name__ == "__main__":
    unittest.main()

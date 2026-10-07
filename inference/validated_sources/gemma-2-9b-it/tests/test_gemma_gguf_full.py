"""Focused tests for the standalone Gemma GGUF full-run contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_genuine_gemma_gguf_smoke as smoke_module
from scripts.run_genuine_gemma_gguf_smoke import (
    MODEL_IDENTIFIER as SMOKE_MODEL_IDENTIFIER,
    MODEL_REVISION as SMOKE_MODEL_REVISION,
    report_markdown,
)
from scripts.run_gemma_gguf_full import (
    DATASET_PATH,
    DATASET_SHA256,
    DEFAULT_SEED,
    EXPECTED_OUTPUT_CONDITIONS,
    EXPECTED_ROW_COUNT,
    FULL_SCHEMA_VERSION,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    FROZEN_PERMUTATION_MANIFEST_SHA256,
    MODEL_FILES,
    PAIR_MANIFEST,
    PAIR_MANIFEST_SHA256,
    PROJECT_ROOT,
    _validate_unit_rows,
    checkpoint_path,
    directed_key,
    prepare_unit,
    run_directed_unit,
    run_gemma_full,
    validate_checkpoint,
    validate_complete_results,
    validate_manifest,
    validate_runtime,
)
from src.data_loader import DataLoader
from src.gemma_gguf_spec import EXPECTED_ARCHITECTURE, EXPECTED_CONTEXT_LENGTH
from src.gguf_runner import (
    GEMMA2_EMBEDDED_CHAT_TEMPLATE,
    serialize_gemma2_chat,
)
from src.scoring import map_label_probabilities, normalize_log_scores, option_labels


class GenuineShapeFakeRunner:
    """A no-model test double with the accepted llama.cpp trace shape.

    It is marked genuine only so row assembly can be unit-tested without model
    inference.  Production preflight additionally pins model files, helper,
    commit, runtime, and the embedded-template hash.
    """

    backend = "llama.cpp"
    is_synthetic = False

    def __init__(self) -> None:
        self.last_scoring: dict = {}

    def predict_distribution(self, prompt, options, temperature=0.0):
        self.assertEqualTemperature(temperature)
        labels = option_labels(len(options))
        # Scores depend only on prompt bytes and label, so repeated target
        # prompts remain exactly identical across evidence-country pairings.
        raw_scores = []
        for label in labels:
            digest = hashlib.sha256((str(prompt) + "\0" + label).encode()).digest()
            raw_scores.append(-float(int.from_bytes(digest[:4], "big") % 10_000) / 997.0)
        probabilities = normalize_log_scores(raw_scores, temperature=0.0)
        messages = [{"role": "user", "content": str(prompt)}]
        serialized = serialize_gemma2_chat(messages)
        prompt_count = max(1, len(serialized.encode("utf-8")) // 4)
        candidates = {}
        for index, (label, score) in enumerate(zip(labels, raw_scores)):
            token_id = 32 + index
            candidates[label] = {
                "candidate_label": label,
                "continuation": label,
                "token_ids": [token_id],
                "token_count": 1,
                "raw_token_logits": [score + 10.0],
                "token_log_probabilities": [score],
                "sequence_log_probability": score,
                "target_token_positions": [prompt_count],
                "predictive_logit_positions": [prompt_count - 1],
                "prompt_prefix_verified": True,
            }
        self.last_scoring = {
            "request_id": "unit-test",
            "structured_messages": messages,
            "serialized_prompt": serialized,
            "serialized_prompt_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "prompt_token_ids": [2] + list(range(3, prompt_count + 2)),
            "prompt_token_ids_add_special_false": [2]
            + list(range(3, prompt_count + 2)),
            "prompt_token_ids_add_special_true": [2, 2]
            + list(range(3, prompt_count + 2)),
            "prompt_token_count": prompt_count,
            "bos_token_id": 2,
            "leading_bos_token_count": 1,
            "leading_bos_count_add_special_false": 1,
            "leading_bos_count_add_special_true": 2,
            "selected_actual_tokenization_mode": "add_special_false",
            "independent_serialized_prompt": serialized,
            "independent_serialized_prompt_sha256": hashlib.sha256(
                serialized.encode()
            ).hexdigest(),
            "independent_prompt_token_ids": [2]
            + list(range(3, prompt_count + 2)),
            "independent_leading_bos_token_count": 1,
            "answer_target_position": prompt_count,
            "answer_predictive_logit_position": prompt_count - 1,
            "prompt_prefix_verified": True,
            "embedded_template_serialization_verified": True,
            "serialization_verification": (
                "pinned_gemma2_byte_or_token_id_equivalent_expansion"
            ),
            "serialization_verification_mode": "byte_exact",
            "serialization_byte_exact": True,
            "serialization_token_ids_equal": True,
            "exact_assistant_answer_suffix_verified": True,
            "user_only_message_structure_verified": True,
            "candidates": candidates,
            "label_to_option": dict(zip(labels, options)),
            "raw_label_log_scores": dict(zip(labels, raw_scores)),
            "normalized_label_probabilities": dict(zip(labels, probabilities)),
            "generated_answer": None,
        }
        return map_label_probabilities(labels, options, probabilities)

    @staticmethod
    def assertEqualTemperature(value):
        if value != 0.0:
            raise AssertionError("full-run test double requires zero temperature")

    def get_last_scoring_metadata(self):
        return deepcopy(self.last_scoring)


class GemmaGGUFFullContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader(str(PROJECT_ROOT / DATASET_PATH))
        cls.loader.load_dataset()
        cls.manifest = cls.loader.get_question_pairs(str(PROJECT_ROOT / PAIR_MANIFEST))

    def build_unit_rows(self, pair, fingerprint="test-fingerprint"):
        return run_directed_unit(
            GenuineShapeFakeRunner(),
            self.loader,
            pair,
            [{"filename": item["filename"]} for item in MODEL_FILES],
            fingerprint,
        )

    def test_pinned_manifest_contract_and_accepted_model_identity(self):
        keys = validate_manifest(self.manifest)
        self.assertEqual(len(keys), 200)
        self.assertEqual(len(set(keys)), 200)
        self.assertEqual(EXPECTED_ROW_COUNT, 800)
        self.assertEqual(FULL_SCHEMA_VERSION, "gemma-gguf-full-v2")
        self.assertEqual(
            EXPECTED_OUTPUT_CONDITIONS,
            ("baseline", "country_label", "evidence", "conflict"),
        )
        self.assertEqual(MODEL_IDENTIFIER, SMOKE_MODEL_IDENTIFIER)
        self.assertEqual(MODEL_REVISION, SMOKE_MODEL_REVISION)
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / DATASET_PATH).read_bytes()).hexdigest(),
            DATASET_SHA256,
        )
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / PAIR_MANIFEST).read_bytes()).hexdigest(),
            PAIR_MANIFEST_SHA256,
        )
        self.assertEqual(
            FROZEN_PERMUTATION_MANIFEST_SHA256,
            "135ed1f0a3cad88658f4acb597e60b6a8461d107326d73b222101bb03d5b2d17",
        )

    def test_manifest_rejects_self_pair_and_mapping_drift(self):
        self_pair = deepcopy(self.manifest)
        self_pair[0]["conflict_country"] = self_pair[0]["country"]
        with self.assertRaisesRegex(ValueError, "self-pair"):
            validate_manifest(self_pair)
        incompatible = deepcopy(self.manifest)
        incompatible[0]["mapping_status"] = "permuted_or_incompatible"
        with self.assertRaisesRegex(ValueError, "exact shared response schema"):
            validate_manifest(incompatible)

    def test_one_permutation_is_shared_by_all_conditions(self):
        pair = self.manifest[0]
        prepared = prepare_unit(self.loader, pair, seed=DEFAULT_SEED)
        rows = self.build_unit_rows(pair)
        self.assertEqual([row["condition"] for row in rows], list(EXPECTED_OUTPUT_CONDITIONS))
        self.assertTrue(
            all(row["displayed_options"] == prepared["displayed_options"] for row in rows)
        )
        self.assertTrue(
            all(row["permutation_seed"] == prepared["permutation_seed"] for row in rows)
        )
        _validate_unit_rows(rows, pair, run_fingerprint="test-fingerprint")

    def test_no_evidence_rows_use_null_schema_and_metrics_are_explicit(self):
        rows = self.build_unit_rows(self.manifest[0])
        by_condition = {row["condition"]: row for row in rows}
        for condition in ("baseline", "country_label"):
            row = by_condition[condition]
            self.assertFalse(row["evidence_presented"])
            self.assertIsNone(row["presented_evidence_distribution"])
            self.assertIsNone(row["source_evidence_distribution"])
        for condition in ("evidence", "conflict"):
            row = by_condition[condition]
            self.assertTrue(row["evidence_presented"])
            self.assertIsNotNone(row["presented_evidence_distribution"])
            self.assertIsNotNone(row["source_evidence_distribution"])
        for row in rows:
            self.assertIn("base2_jensen_shannon_divergences", row)
            self.assertNotIn("base2_jensen_shannon_distances", row)
            self.assertIn("EO_raw", row)
            self.assertIn("EO_normalized", row)
        self.assertIn(
            by_condition["conflict"]["conflict_classification"],
            {"evidence-side", "label-side", "tie"},
        )
        self.assertTrue(
            all(
                by_condition[condition]["conflict_classification"] is None
                for condition in ("baseline", "country_label", "evidence")
            )
        )

    def test_checkpoint_accepts_partial_prefix_and_complete_unit(self):
        pair = self.manifest[0]
        rows = self.build_unit_rows(pair)
        signature = {"fingerprint": "test-fingerprint"}
        checkpoint = {
            "schema_version": FULL_SCHEMA_VERSION,
            "run_fingerprint": "test-fingerprint",
            "manifest_index": 0,
            "directed_key": list(directed_key(pair)),
            "completed_conditions": list(EXPECTED_OUTPUT_CONDITIONS),
            "complete": True,
            "rows": rows,
        }
        validated = validate_checkpoint(checkpoint, pair, 0, signature)
        self.assertEqual(len(validated), 4)
        partial = deepcopy(checkpoint)
        partial["rows"] = partial["rows"][:3]
        for row in partial["rows"]:
            for field in (
                "country_influence", "evidence_influence", "EO_raw", "EO_normalized",
                "conflict_classification", "base2_jensen_shannon_divergences",
            ):
                row.pop(field, None)
        partial["completed_conditions"] = list(EXPECTED_OUTPUT_CONDITIONS[:3])
        partial["complete"] = False
        self.assertEqual(len(validate_checkpoint(partial, pair, 0, signature)), 3)
        bad_partial = deepcopy(partial)
        bad_partial["completed_conditions"] = ["baseline", "evidence", "country_label"]
        with self.assertRaisesRegex(ValueError, "condition order"):
            validate_checkpoint(bad_partial, pair, 0, signature)
        reordered = deepcopy(checkpoint)
        reordered["rows"][0], reordered["rows"][1] = (
            reordered["rows"][1],
            reordered["rows"][0],
        )
        with self.assertRaisesRegex(ValueError, "out of order"):
            validate_checkpoint(reordered, pair, 0, signature)
        bad_continuation = deepcopy(checkpoint)
        bad_continuation["rows"][0]["scoring_trace"]["candidates"]["A"][
            "continuation"
        ] = " B"
        with self.assertRaisesRegex(ValueError, "contextual continuation"):
            validate_checkpoint(bad_continuation, pair, 0, signature)
        wrong_fingerprint = deepcopy(checkpoint)
        wrong_fingerprint["run_fingerprint"] = "other"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_checkpoint(wrong_fingerprint, pair, 0, signature)
        with tempfile.TemporaryDirectory() as temporary:
            left = checkpoint_path(Path(temporary), directed_key(pair))
            right = checkpoint_path(Path(temporary), directed_key(self.manifest[1]))
            self.assertNotEqual(left, right)

    def test_complete_validator_preserves_all_200_directed_units(self):
        rows = []
        for pair in self.manifest:
            rows.extend(self.build_unit_rows(pair))
        summary = validate_complete_results(
            rows,
            self.manifest,
            provenance={"fingerprint": "test-fingerprint"},
        )
        self.assertEqual(summary["rows"], 800)
        self.assertEqual(summary["directed_units"], 200)
        self.assertEqual(summary["question_clusters"], 44)
        self.assertEqual(summary["target_units"], 144)
        damaged = deepcopy(rows)
        damaged[4] = deepcopy(damaged[0])
        with self.assertRaisesRegex(ValueError, "overwritten|out of order"):
            validate_complete_results(
                damaged,
                self.manifest,
                provenance={"fingerprint": "test-fingerprint"},
            )

    def test_runtime_is_pinned_to_accepted_llama_template_and_version(self):
        runtime = {
            "backend": "llama.cpp",
            "synthetic": False,
            "model_identifier": MODEL_IDENTIFIER,
            "repository": "bartowski/gemma-2-9b-it-GGUF",
            "revision": MODEL_REVISION,
            "quantization": "Q4_K_M",
            "chat_template_source": "embedded_gguf_metadata",
            "chat_template": GEMMA2_EMBEDDED_CHAT_TEMPLATE,
            "chat_template_sha256": hashlib.sha256(
                GEMMA2_EMBEDDED_CHAT_TEMPLATE.encode()
            ).hexdigest(),
            "llama_version": "0.3.0-dev",
            "model_description": "Gemma 2 9B IT Q4_K_M",
            "selected_model_metadata": {
                "general.architecture": EXPECTED_ARCHITECTURE,
                "gemma2.context_length": str(EXPECTED_CONTEXT_LENGTH),
            },
            "gpu_layers": 0,
            "candidate_scoring": "full_vocabulary_low_level_logits",
            "generated_answer_parsing": False,
        }
        validate_runtime(runtime)
        runtime["chat_template_sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "chat template hash"):
            validate_runtime(runtime)

    def test_failed_smoke_report_does_not_claim_intended_rows_completed(self):
        report = report_markdown(
            {
                "status": "FAILED",
                "row_count": 0,
                "failure": "RuntimeError: serialization preflight failed",
                "model_identifier": MODEL_IDENTIFIER,
            }
        )
        self.assertIn("0 validated rows were saved before failure", report)
        self.assertIn("No passing smoke gate was created", report)
        self.assertIn("final 8-row smoke assertions were not reached", report)
        self.assertNotIn("Exactly two reciprocal directed units were scored", report)
        self.assertNotIn("EO_normalized is +1", report)

    def test_passing_smoke_analysis_preserves_serialization_preflight(self):
        source = (
            PROJECT_ROOT / "scripts/run_genuine_gemma_gguf_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"serialization_preflight": analysis["serialization_preflight"]',
            source,
        )

    def test_smoke_report_uses_observed_preflight_inference_flag(self):
        report = report_markdown(
            {
                "status": "FAILED",
                "row_count": 0,
                "failure": "preflight contract failed",
                "serialization_preflight": {
                    "status": "FAILED",
                    "path": "serialization_preflight.json",
                    "verification_mode": "none",
                    "inference_performed": True,
                },
            }
        )
        self.assertIn("Model inference performed by preflight: `true`", report)
        self.assertNotIn("Model inference performed by preflight: `false`", report)

    def test_smoke_entrypoint_auto_continues_full_only_after_pass(self):
        smoke_analysis = {"status": "PASS", "row_count": 8}
        full_completion = {"status": "PASS", "rows": 800}
        argv = [
            "run_genuine_gemma_gguf_smoke.py",
            "--model-dir",
            "model",
            "--helper",
            "helper",
            "--llama-cpp-dir",
            "llama.cpp",
            "--output-dir",
            "custom-smoke",
            "--full-output-dir",
            "custom-full",
        ]
        with patch.object(smoke_module, "run_smoke", return_value=smoke_analysis), patch.object(
            smoke_module,
            "run_gemma_full",
            return_value=([], full_completion),
        ) as full_run, patch("sys.argv", argv), patch("builtins.print"):
            smoke_module.main()
        full_run.assert_called_once()
        kwargs = full_run.call_args.kwargs
        self.assertTrue(kwargs["allow_gemma_full_run"])
        self.assertEqual(kwargs["smoke_directory"], Path("custom-smoke"))
        self.assertEqual(kwargs["output_directory"], Path("custom-full"))

        with patch.object(
            smoke_module,
            "run_smoke",
            side_effect=RuntimeError("smoke failed"),
        ), patch.object(smoke_module, "run_gemma_full") as blocked_full, patch(
            "sys.argv", argv
        ):
            with self.assertRaisesRegex(RuntimeError, "smoke failed"):
                smoke_module.main()
        blocked_full.assert_not_called()

    def test_serialization_only_helper_path_cannot_decode_and_scoring_reuses_mode(self):
        source = (PROJECT_ROOT / "src/gguf_score_helper.cpp").read_text(
            encoding="utf-8"
        )
        serialization_body = source.split("json serialization_response(", 1)[1].split(
            "json selected_model_metadata(", 1
        )[0]
        self.assertNotIn("llama_decode", serialization_body)
        self.assertIn('operation == "serialize_only"', source)
        self.assertIn("prompt_tokens_without_special", source)
        self.assertIn("prompt_tokens_with_special", source)
        self.assertIn("inspection.selected_actual_add_special", source)
        self.assertLess(
            source.index("if (!inspection.contract_pass)"),
            source.index("llama_decode(context, prompt_batch)"),
        )

    def test_qwen_only_gate_fails_before_any_filesystem_or_model_action(self):
        with self.assertRaisesRegex(PermissionError, "allow-gemma-full-run"):
            run_gemma_full(
                model_directory=Path("missing-model"),
                helper_path=Path("missing-helper"),
                llama_cpp_directory=Path("missing-llama"),
                allow_gemma_full_run=False,
            )
        source = (PROJECT_ROOT / "scripts/run_gemma_gguf_full.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_full_experiment", source)
        self.assertNotIn("MODEL_SPECS", source)


if __name__ == "__main__":
    unittest.main()

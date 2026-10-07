"""Focused tests for the standalone Qwen GGUF full-run contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_genuine_qwen_gguf_smoke import (
    MODEL_IDENTIFIER as SMOKE_MODEL_IDENTIFIER,
    MODEL_REVISION as SMOKE_MODEL_REVISION,
)
from scripts.run_qwen_gguf_full import (
    ACCEPTED_GGUF_HELPER_SOURCE_SHA256,
    ACCEPTED_GGUF_RUNNER_SHA256,
    DATASET_PATH,
    DATASET_SHA256,
    DEFAULT_SEED,
    EXPECTED_OUTPUT_CONDITIONS,
    EXPECTED_ROW_COUNT,
    FULL_SCHEMA_VERSION,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    MODEL_SHARDS,
    PAIR_MANIFEST,
    PAIR_MANIFEST_SHA256,
    PROJECT_ROOT,
    _validate_unit_rows,
    checkpoint_path,
    directed_key,
    prepare_unit,
    run_directed_unit,
    run_qwen_full,
    validate_checkpoint,
    validate_complete_results,
    validate_manifest,
    validate_runtime,
)
from src.data_loader import DataLoader
from src.gguf_runner import QWEN25_DEFAULT_SYSTEM_MESSAGE, serialize_qwen25_chatml
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
        messages = [
            {"role": "system", "content": QWEN25_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": str(prompt)},
        ]
        serialized = serialize_qwen25_chatml(messages)
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
            "prompt_token_count": prompt_count,
            "answer_target_position": prompt_count,
            "answer_predictive_logit_position": prompt_count - 1,
            "prompt_prefix_verified": True,
            "embedded_template_serialization_verified": True,
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


class QwenGGUFFullContractTests(unittest.TestCase):
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
            [{"filename": item["filename"]} for item in MODEL_SHARDS],
            fingerprint,
        )

    def test_pinned_manifest_contract_and_accepted_model_identity(self):
        keys = validate_manifest(self.manifest)
        self.assertEqual(len(keys), 200)
        self.assertEqual(len(set(keys)), 200)
        self.assertEqual(EXPECTED_ROW_COUNT, 800)
        self.assertEqual(FULL_SCHEMA_VERSION, "qwen-gguf-full-v2")
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
            hashlib.sha256((PROJECT_ROOT / "src/gguf_runner.py").read_bytes()).hexdigest(),
            ACCEPTED_GGUF_RUNNER_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (PROJECT_ROOT / "src/gguf_score_helper.cpp").read_bytes()
            ).hexdigest(),
            ACCEPTED_GGUF_HELPER_SOURCE_SHA256,
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

    def test_checkpoint_accepts_only_complete_ordered_unit_and_matching_fingerprint(self):
        pair = self.manifest[0]
        rows = self.build_unit_rows(pair)
        signature = {"fingerprint": "test-fingerprint"}
        checkpoint = {
            "schema_version": FULL_SCHEMA_VERSION,
            "run_fingerprint": "test-fingerprint",
            "manifest_index": 0,
            "directed_key": list(directed_key(pair)),
            "conditions": list(EXPECTED_OUTPUT_CONDITIONS),
            "rows": rows,
        }
        validated = validate_checkpoint(checkpoint, pair, 0, signature)
        self.assertEqual(len(validated), 4)
        partial = deepcopy(checkpoint)
        partial["rows"] = partial["rows"][:3]
        with self.assertRaisesRegex(ValueError, "complete unit"):
            validate_checkpoint(partial, pair, 0, signature)
        reordered = deepcopy(checkpoint)
        reordered["rows"][0], reordered["rows"][1] = (
            reordered["rows"][1],
            reordered["rows"][0],
        )
        with self.assertRaisesRegex(ValueError, "out of order"):
            validate_checkpoint(reordered, pair, 0, signature)
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
        analysis = json.loads(
            (
                PROJECT_ROOT
                / "experiments/smoke_genuine_qwen_gguf/analysis.json"
            ).read_text(encoding="utf-8")
        )
        llama = analysis["metadata"]["llama_cpp"]
        runtime = {
            "backend": "llama.cpp",
            "synthetic": False,
            "model_identifier": MODEL_IDENTIFIER,
            "repository": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "revision": MODEL_REVISION,
            "quantization": "Q4_K_M",
            "chat_template_source": "embedded_gguf_metadata",
            "chat_template_sha256": llama["chat_template_sha256"],
            "llama_version": llama["version"],
            "gpu_layers": 0,
            "candidate_scoring": "full_vocabulary_low_level_logits",
            "generated_answer_parsing": False,
        }
        validate_runtime(runtime)
        runtime["chat_template_sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "chat template hash"):
            validate_runtime(runtime)

    def test_qwen_only_gate_fails_before_any_filesystem_or_model_action(self):
        with self.assertRaisesRegex(PermissionError, "allow-qwen-full-run"):
            run_qwen_full(
                model_directory=Path("missing-model"),
                helper_path=Path("missing-helper"),
                llama_cpp_directory=Path("missing-llama"),
                allow_qwen_full_run=False,
            )
        source = (PROJECT_ROOT / "scripts/run_qwen_gguf_full.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_full_experiment", source)
        self.assertNotIn("MODEL_SPECS", source)


if __name__ == "__main__":
    unittest.main()

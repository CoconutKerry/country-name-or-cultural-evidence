"""Focused, model-free tests for the Mistral-v0.3 GGUF run contract.

The fake runner below exercises row assembly and validation only. It never
loads a model and is not available to either production entry point.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.run_genuine_mistral_gguf_smoke import _load_passing_test_report

from scripts.run_mistral_gguf_full import (
    CLASSIFICATION_TOLERANCE,
    DATASET_PATH,
    DATASET_SHA256,
    DEFAULT_SEED,
    EXPECTED_DIRECTED_UNIT_COUNT,
    EXPECTED_OUTPUT_CONDITIONS,
    EXPECTED_ROW_COUNT,
    EXPECTED_TARGET_UNIT_COUNT,
    PAIR_MANIFEST,
    PAIR_MANIFEST_SHA256,
    PERMUTATION_KEY,
    PROJECT_ROOT,
    _validate_unit_rows,
    checkpoint_path,
    classify_conflict,
    directed_key,
    prepare_unit,
    row_checkpoint_payload,
    run_directed_unit,
    validate_checkpoint,
    validate_manifest,
    validate_runtime,
)
from src.data_loader import DataLoader
from src.gguf_runner import (
    MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
    MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION,
    MISTRAL_V03_CHAT_PROFILE,
    build_mistral_template_equivalence_diagnostic,
    serialize_mistral_v03_chat,
)
from src.main import shuffle_options, stable_unit_seed
from src.metrics import Metrics
from src.mistral_gguf_contract import (
    LLAMA_CPP_COMMIT,
    LLAMA_CPP_VERSION,
    MISTRAL_CHAT_TEMPLATE_SHA256,
    MODEL_FILENAME,
    MODEL_IDENTIFIER,
    MODEL_NAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SHARDS,
    MODEL_SIZE_BYTES,
    QUANTIZATION,
    UPSTREAM_CHECKPOINT,
    UPSTREAM_REVISION,
    UPSTREAM_REVISION_NOT_RECORDED_BY_QUANTIZER,
)
from src.scoring import map_label_probabilities, normalize_log_scores, option_labels


class GenuineShapeMistralRunner:
    """No-model test double with the genuine Mistral scoring-trace shape."""

    backend = "llama.cpp"
    is_synthetic = False

    def __init__(self) -> None:
        self.last_scoring: dict = {}

    def predict_distribution(self, prompt, options, temperature=0.0):
        if temperature != 0.0:
            raise AssertionError("contract test runner requires zero temperature")
        labels = option_labels(len(options))
        raw_scores = []
        for label in labels:
            digest = hashlib.sha256((str(prompt) + "\0" + label).encode()).digest()
            raw_scores.append(
                -float(int.from_bytes(digest[:4], "big") % 10_000) / 997.0
            )
        probabilities = normalize_log_scores(raw_scores, temperature=0.0)
        messages = [{"role": "user", "content": str(prompt)}]
        canonical_serialized = serialize_mistral_v03_chat(messages)
        embedded_serialized = canonical_serialized.removeprefix("<s>")
        prompt_token_ids = [1, 3, 900, 4]
        canonical_prompt_token_ids = list(prompt_token_ids)
        prompt_count = len(prompt_token_ids)
        equivalence = build_mistral_template_equivalence_diagnostic(
            embedded_serialized,
            canonical_serialized,
            prompt_token_ids,
            canonical_prompt_token_ids,
            1,
        )
        candidates = {}
        for index, (label, score) in enumerate(zip(labels, raw_scores)):
            token_id = 100 + index
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
            "request_id": "contract-test",
            "structured_messages": messages,
            "serialized_prompt": embedded_serialized,
            "serialized_prompt_sha256": hashlib.sha256(
                embedded_serialized.encode()
            ).hexdigest(),
            "canonical_serialized_prompt": canonical_serialized,
            "prompt_token_ids": prompt_token_ids,
            "canonical_prompt_token_ids": canonical_prompt_token_ids,
            "prompt_token_count": prompt_count,
            "canonical_prompt_token_count": prompt_count,
            "answer_target_position": prompt_count,
            "answer_predictive_logit_position": prompt_count - 1,
            "prompt_prefix_verified": True,
            "bos_token_id": 1,
            "prompt_starts_with_bos": True,
            "prompt_bos_token_count": 1,
            "canonical_prompt_starts_with_bos": True,
            "canonical_prompt_bos_token_count": 1,
            "embedded_template_serialization_verified": True,
            "serialization_verification": MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION,
            "chat_profile": MISTRAL_V03_CHAT_PROFILE,
            "add_special_tokens": True,
            "tokenization_add_special": True,
            "canonical_tokenization_add_special": False,
            "candidate_prefix": "",
            "chat_template_source": "embedded_gguf_metadata",
            "template_equivalence_policy": MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
            "template_equivalence_preflight_passed": True,
            "template_equivalence": equivalence,
            "candidates": candidates,
            "label_to_option": dict(zip(labels, options)),
            "raw_label_log_scores": dict(zip(labels, raw_scores)),
            "normalized_label_probabilities": dict(zip(labels, probabilities)),
            "generated_answer": None,
        }
        return map_label_probabilities(labels, options, probabilities)

    def get_last_scoring_metadata(self):
        return deepcopy(self.last_scoring)


class MistralGGUFFullContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader(str(PROJECT_ROOT / DATASET_PATH))
        cls.loader.load_dataset()
        cls.manifest = cls.loader.get_question_pairs(
            str(PROJECT_ROOT / PAIR_MANIFEST)
        )

    def build_unit_rows(self, pair, fingerprint="test-fingerprint"):
        return run_directed_unit(
            GenuineShapeMistralRunner(),
            self.loader,
            pair,
            [{"filename": item["filename"]} for item in MODEL_SHARDS],
            fingerprint,
        )

    def test_exact_mistral_model_and_backend_pin(self):
        self.assertEqual(MODEL_NAME, "Mistral-7B-Instruct-v0.3")
        self.assertEqual(
            UPSTREAM_CHECKPOINT, "mistralai/Mistral-7B-Instruct-v0.3"
        )
        self.assertIsNone(UPSTREAM_REVISION)
        self.assertTrue(UPSTREAM_REVISION_NOT_RECORDED_BY_QUANTIZER)
        self.assertEqual(
            MODEL_REPOSITORY, "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
        )
        self.assertEqual(
            MODEL_REVISION, "61fd4167fff3ab01ee1cfe0da183fa27a944db48"
        )
        self.assertEqual(
            MODEL_FILENAME, "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
        )
        self.assertEqual(MODEL_SIZE_BYTES, 4_372_812_000)
        self.assertEqual(
            MODEL_SHA256,
            "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6",
        )
        self.assertEqual(QUANTIZATION, "Q4_K_M")
        self.assertEqual(
            LLAMA_CPP_COMMIT, "62acc89c26c66076cb72e049f307fbe93b8b9750"
        )
        self.assertEqual(len(MODEL_SHARDS), 1)
        self.assertEqual(MODEL_SHARDS[0]["filename"], MODEL_FILENAME)
        self.assertIn(MODEL_REVISION, MODEL_IDENTIFIER)

    def test_embedded_mistral_template_is_user_only_and_exact(self):
        prompt = "Choose one.\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        self.assertEqual(
            serialize_mistral_v03_chat(messages),
            "<s>[INST] Choose one.\nAnswer: [/INST]",
        )
        with self.assertRaisesRegex(ValueError, "beginning with user"):
            serialize_mistral_v03_chat(
                [{"role": "system", "content": "forbidden system prompt"}]
            )
        with self.assertRaisesRegex(ValueError, "alternate"):
            serialize_mistral_v03_chat(
                [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": prompt},
                ]
            )

    def test_full_manifest_and_fixed_permutation_contract(self):
        keys = validate_manifest(self.manifest)
        self.assertEqual(len(keys), EXPECTED_DIRECTED_UNIT_COUNT)
        self.assertEqual(len(set(keys)), EXPECTED_DIRECTED_UNIT_COUNT)
        self.assertEqual(EXPECTED_DIRECTED_UNIT_COUNT, 200)
        self.assertEqual(EXPECTED_ROW_COUNT, 800)
        self.assertEqual(EXPECTED_TARGET_UNIT_COUNT, 144)
        self.assertEqual(PERMUTATION_KEY, "question_id::label_country")
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / DATASET_PATH).read_bytes()).hexdigest(),
            DATASET_SHA256,
        )
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / PAIR_MANIFEST).read_bytes()).hexdigest(),
            PAIR_MANIFEST_SHA256,
        )
        key_set = set(keys)
        for question_id, label_country, evidence_country in keys:
            self.assertIn((question_id, evidence_country, label_country), key_set)

        target_permutations = {}
        for pair in self.manifest:
            prepared = prepare_unit(self.loader, pair, seed=DEFAULT_SEED)
            target = (pair["question_id"], pair["country"])
            expected_seed = stable_unit_seed(DEFAULT_SEED, "::".join(target))
            expected = shuffle_options(pair["options"], expected_seed)
            self.assertEqual(prepared["permutation_seed"], expected_seed)
            self.assertEqual(prepared["displayed_options"], expected)
            if target in target_permutations:
                self.assertEqual(
                    prepared["displayed_options"], target_permutations[target]
                )
            target_permutations[target] = prepared["displayed_options"]
        self.assertEqual(len(target_permutations), EXPECTED_TARGET_UNIT_COUNT)

    def test_rows_use_bare_labels_exact_answer_position_and_one_bos(self):
        pair = self.manifest[0]
        rows = self.build_unit_rows(pair)
        prepared = prepare_unit(self.loader, pair, seed=DEFAULT_SEED)
        self.assertEqual(
            [row["condition"] for row in rows], list(EXPECTED_OUTPUT_CONDITIONS)
        )
        self.assertTrue(
            all(
                row["displayed_options"] == prepared["displayed_options"]
                for row in rows
            )
        )
        validated = _validate_unit_rows(
            rows, pair, run_fingerprint="test-fingerprint"
        )
        self.assertEqual(len(validated), 4)
        for row in rows:
            scoring = row["scoring_trace"]
            self.assertEqual(
                scoring["structured_messages"],
                [{"role": "user", "content": row["raw_user_prompt"]}],
            )
            self.assertEqual(scoring["chat_profile"], MISTRAL_V03_CHAT_PROFILE)
            self.assertIs(scoring["add_special_tokens"], True)
            self.assertIs(scoring["canonical_tokenization_add_special"], False)
            self.assertIs(scoring["prompt_starts_with_bos"], True)
            self.assertEqual(scoring["prompt_bos_token_count"], 1)
            self.assertTrue(scoring["serialized_prompt"].startswith("[INST] "))
            self.assertTrue(scoring["serialized_prompt"].endswith(" [/INST]"))
            self.assertTrue(
                scoring["canonical_serialized_prompt"].startswith("<s>[INST] ")
            )
            self.assertTrue(scoring["template_equivalence"]["accepted"])
            self.assertTrue(
                scoring["template_equivalence"]["token_ids_equivalent"]
            )
            prompt_count = scoring["prompt_token_count"]
            paths = []
            for label in row["displayed_option_labels"]:
                candidate = scoring["candidates"][label]
                self.assertEqual(candidate["continuation"], label)
                self.assertEqual(candidate["target_token_positions"][0], prompt_count)
                self.assertEqual(
                    candidate["predictive_logit_positions"][0], prompt_count - 1
                )
                paths.append(tuple(candidate["token_ids"]))
            self.assertEqual(len(paths), len(set(paths)))
            probabilities = row["normalized_label_probabilities"].values()
            self.assertTrue(
                all(
                    math.isfinite(value) and value >= 0 for value in probabilities
                )
            )
            self.assertAlmostEqual(math.fsum(probabilities), 1.0, places=12)

    def test_trace_contract_rejects_label_or_boundary_substitution(self):
        pair = self.manifest[0]
        rows = self.build_unit_rows(pair)
        mutations = {
            "leading-space label": lambda row: row["scoring_trace"]["candidates"][
                "A"
            ].__setitem__("continuation", " A"),
            "missing tokenizer BOS": lambda row: row[
                "scoring_trace"
            ].__setitem__("add_special_tokens", False),
            "double BOS": lambda row: row["scoring_trace"].__setitem__(
                "prompt_bos_token_count", 2
            ),
            "wrong answer position": lambda row: row["scoring_trace"][
                "candidates"
            ]["A"].__setitem__("predictive_logit_positions", [0]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                damaged = deepcopy(rows)
                mutate(damaged[0])
                with self.assertRaises(ValueError):
                    _validate_unit_rows(
                        damaged, pair, run_fingerprint="test-fingerprint"
                    )

    def test_absent_evidence_is_null_and_metrics_use_divergence_schema(self):
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
            self.assertEqual(row["jensen_shannon_base"], 2)
            self.assertEqual(row["jensen_shannon_measure"], "divergence_bits")
            self.assertIn("base2_jensen_shannon_divergences", row)
            self.assertNotIn("base2_jensen_shannon_distances", row)
            self.assertIn("EO_raw", row)
            self.assertIn("EO_normalized", row)

    def test_evidence_override_orientation_and_tie_tolerance(self):
        label = {"yes": 0.9, "no": 0.1}
        evidence = {"yes": 0.1, "no": 0.9}
        self.assertGreater(
            Metrics.evidence_override_raw(evidence, evidence, label), 0
        )
        self.assertLess(Metrics.evidence_override_raw(label, evidence, label), 0)
        self.assertAlmostEqual(
            Metrics.evidence_override_normalized(evidence, evidence, label), 1.0
        )
        self.assertAlmostEqual(
            Metrics.evidence_override_normalized(label, evidence, label), -1.0
        )
        self.assertEqual(classify_conflict(CLASSIFICATION_TOLERANCE / 2), "tie")
        self.assertEqual(
            classify_conflict(CLASSIFICATION_TOLERANCE * 2), "evidence-side"
        )
        self.assertEqual(
            classify_conflict(-CLASSIFICATION_TOLERANCE * 2), "label-side"
        )

    def test_checkpoint_is_one_directed_condition_row_with_four_key(self):
        pair = self.manifest[0]
        condition = "conflict"
        row = self.build_unit_rows(pair)[3]
        signature = {"fingerprint": "test-fingerprint"}
        payload = row_checkpoint_payload(row, pair, condition, 0, signature)
        self.assertEqual(payload["row_key"], [*directed_key(pair), condition])
        self.assertEqual(payload["metrics_status"], "complete")
        validated = validate_checkpoint(payload, pair, condition, 0, signature)
        self.assertEqual(
            (*directed_key(validated), validated["condition"]),
            tuple(payload["row_key"]),
        )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = checkpoint_path(directory, (*directed_key(pair), condition))
            other = checkpoint_path(directory, (*directed_key(pair), "baseline"))
            self.assertNotEqual(path, other)
            with self.assertRaisesRegex(ValueError, "directed key plus condition"):
                checkpoint_path(directory, directed_key(pair))

        damaged = deepcopy(payload)
        damaged["row_key"][-1] = "baseline"
        with self.assertRaisesRegex(ValueError, "row key mismatch"):
            validate_checkpoint(damaged, pair, condition, 0, signature)
        damaged = deepcopy(payload)
        damaged["run_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_checkpoint(damaged, pair, condition, 0, signature)

    def test_runtime_contract_rejects_wrong_model_or_template(self):
        runtime = {
            "backend": "llama.cpp",
            "synthetic": False,
            "model_identifier": MODEL_IDENTIFIER,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "quantization": QUANTIZATION,
            "chat_template_source": "embedded_gguf_metadata",
            "chat_template_sha256": MISTRAL_CHAT_TEMPLATE_SHA256,
            "chat_profile": MISTRAL_V03_CHAT_PROFILE,
            "tokenizer_add_special_tokens": True,
            "canonical_tokenization_add_special": False,
            "template_equivalence_policy": MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
            "llama_version": LLAMA_CPP_VERSION,
            "gpu_layers": 0,
            "candidate_scoring": "full_vocabulary_low_level_logits",
            "generated_answer_parsing": False,
        }
        validate_runtime(runtime)
        for field, bad_value in (
            ("model_identifier", "substituted-model"),
            ("chat_template_sha256", "wrong-template"),
            ("tokenizer_add_special_tokens", False),
            ("canonical_tokenization_add_special", True),
            ("gpu_layers", 1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    validate_runtime({**runtime, field: bad_value})

    def test_preinference_report_binds_transcript_and_validated_sources(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            directory = Path(temporary)
            transcript = directory / "transcript.txt"
            transcript.write_text("all tests passed\n", encoding="utf-8")
            validated = {
                "src/gguf_runner.py": hashlib.sha256(
                    (PROJECT_ROOT / "src/gguf_runner.py").read_bytes()
                ).hexdigest(),
                "src/gguf_score_helper.cpp": hashlib.sha256(
                    (PROJECT_ROOT / "src/gguf_score_helper.cpp").read_bytes()
                ).hexdigest(),
            }
            report = {
                "status": "PASS",
                "complete_unit_test_suite": True,
                "tests_collected": 1,
                "failures": 0,
                "errors": 0,
                "data_validation": "PASS",
                "test_transcript": str(transcript.relative_to(PROJECT_ROOT)),
                "test_transcript_sha256": hashlib.sha256(
                    transcript.read_bytes()
                ).hexdigest(),
                "gguf_runner_sha256": validated["src/gguf_runner.py"],
                "gguf_score_helper_source_sha256": validated[
                    "src/gguf_score_helper.cpp"
                ],
                "validated_file_sha256": validated,
            }
            report_path = directory / "gate.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(_load_passing_test_report(report_path), report)

            stale = deepcopy(report)
            stale["validated_file_sha256"]["src/gguf_runner.py"] = "0" * 64
            report_path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after tests"):
                _load_passing_test_report(report_path)

            report_path.write_text(json.dumps(report), encoding="utf-8")
            transcript.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "transcript is missing or changed"):
                _load_passing_test_report(report_path)

    def test_production_mistral_files_contain_no_fictional_mistral25_identity(self):
        for relative in (
            "src/mistral_gguf_contract.py",
            "scripts/run_mistral_gguf_full.py",
        ):
            text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("Mistral2.5", text)
            self.assertNotIn("MISTRAL25", text)


if __name__ == "__main__":
    unittest.main()

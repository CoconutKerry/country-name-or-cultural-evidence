"""Focused model-free tests for the Llama smoke and resume gates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts import run_llama_gguf_full as full
from src.data_loader import DataLoader
from src.llama_gguf_runner import (
    LLAMA31_ANSWER_SUFFIX,
    LLAMA31_DEFAULT_SYSTEM_MESSAGE,
)
from src.scoring import map_label_probabilities, option_labels


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SMOKE_ASSERTIONS = frozenset(
    {
        "genuine_backend_only",
        "exactly_eight_rows",
        "unique_directed_condition_keys",
        "exact_reciprocal_units",
        "all_four_conditions_per_unit",
        "embedded_llama_chat_template",
        "candidate_tokenization_valid",
        "correct_answer_position",
        "all_labels_scored_from_full_vocabulary",
        "no_generation_or_answer_parsing",
        "probabilities_valid",
        "option_permutation_recovery",
        "shared_first_word_independent",
        "directed_units_not_overwritten",
        "evidence_override_orientation",
        "evidence_null_semantics",
        "divergence_schema_and_metrics_recomputed",
        "gguf_scalar_metadata",
        "independent_first_row_rescore",
        "pinned_helper_linkage",
        "accepted_scoring_source_hashes",
        "gguf_runtime_identity",
        "full_run_approval_gates_unchanged",
    }
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _smoke_analysis() -> dict[str, object]:
    selected = full.MODEL_SHARDS[0]
    return {
        "status": "PASS",
        "scope": {
            "models": 1,
            "directed_units": 2,
            "conditions_per_unit": 4,
            "saved_rows": 8,
            "full_experiment_started": False,
            "additional_models_started": False,
        },
        "metadata": {
            "model": {
                "identifier": full.MODEL_IDENTIFIER,
                "repository": full.MODEL_REPOSITORY,
                "revision": full.MODEL_REVISION,
                "quantization": full.QUANTIZATION,
                "selected_entrypoint_file": selected["filename"],
                "total_size_bytes": selected["size_bytes"],
                "files": [
                    {
                        "filename": selected["filename"],
                        "actual_size_bytes": selected["size_bytes"],
                        "actual_sha256": selected["sha256"],
                        "verified": True,
                    }
                ],
            },
            "llama_cpp": {
                "commit": full.LLAMA_CPP_COMMIT,
                "version": full.LLAMA_CPP_VERSION,
                "chat_template_sha256": full.LLAMA_CHAT_TEMPLATE_SHA256,
                "helper_sha256": "a" * 64,
                "helper_ldd": "libllama.so => /pinned/build-cpu/bin/libllama.so",
                "accepted_scoring_source_sha256": {
                    "src/gguf_runner.py": full.ACCEPTED_GGUF_RUNNER_SHA256,
                    "src/llama_gguf_runner.py": full.LLAMA_GGUF_RUNNER_SHA256,
                    "src/gguf_score_helper.cpp": full.ACCEPTED_GGUF_HELPER_SOURCE_SHA256,
                },
                "gguf_file_type": 15,
                "gguf_scalar_metadata": {
                    "general.architecture": "llama",
                    "llama.context_length": "131072",
                },
            },
        },
        "assertions": {name: True for name in REQUIRED_SMOKE_ASSERTIONS},
        "row_assertion_count": 8,
        "conditions": list(full.EXPECTED_OUTPUT_CONDITIONS),
        "result_key_count": 8,
    }


def _smoke_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(full.EXPECTED_SMOKE_UNITS):
        for condition in full.EXPECTED_OUTPUT_CONDITIONS:
            rows.append(
                {
                    "question_id": key[0],
                    "label_country": key[1],
                    "evidence_country": key[2],
                    "condition": condition,
                    "model_identifier": full.MODEL_IDENTIFIER,
                    "model_repository": full.MODEL_REPOSITORY,
                    "model_revision": full.MODEL_REVISION,
                    "backend": "llama.cpp",
                    "synthetic": False,
                    "scoring_trace": {
                        "embedded_template_serialization_verified": True,
                        "prompt_prefix_verified": True,
                        "prompt_bos_added": True,
                    },
                }
            )
    return rows


class FakeLlamaRunner:
    """Deterministic scorer implementing only the runner protocol used here."""

    backend = "llama.cpp"
    is_synthetic = False

    def __init__(self) -> None:
        self.calls = 0
        self.last_scoring: dict[str, object] = {}

    def predict_distribution(
        self, prompt: str, options: list[str], temperature: float = 0.0
    ) -> dict[str, float]:
        del temperature
        self.calls += 1
        labels = option_labels(len(options))
        base = [float(index + 1) for index in range(len(labels))]
        total = math.fsum(base)
        probabilities = [value / total for value in base]
        scores = [math.log(value) for value in probabilities]
        messages = [
            {"role": "system", "content": LLAMA31_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        serialized = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{LLAMA31_DEFAULT_SYSTEM_MESSAGE}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>{LLAMA31_ANSWER_SUFFIX}"
        )
        prompt_count = 10
        candidates: dict[str, dict[str, object]] = {}
        for index, (label, score) in enumerate(zip(labels, scores)):
            candidates[label] = {
                "token_ids": [100 + index],
                "token_count": 1,
                "raw_token_logits": [score],
                "token_log_probabilities": [score],
                "sequence_log_probability": score,
                "target_token_positions": [prompt_count],
                "predictive_logit_positions": [prompt_count - 1],
            }
        self.last_scoring = {
            "structured_messages": messages,
            "serialized_prompt": serialized,
            "serialized_prompt_sha256": hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest(),
            "label_to_option": dict(zip(labels, options)),
            "raw_label_log_scores": dict(zip(labels, scores)),
            "normalized_label_probabilities": dict(zip(labels, probabilities)),
            "embedded_template_serialization_verified": True,
            "prompt_prefix_verified": True,
            "prompt_bos_added": True,
            "prompt_bos_token_id": 1,
            "bos_token_id": 1,
            "prompt_token_count": prompt_count,
            "answer_target_position": prompt_count,
            "answer_predictive_logit_position": prompt_count - 1,
            "candidates": candidates,
            "generated_answer": None,
        }
        return map_label_probabilities(labels, options, probabilities)

    def get_last_scoring_metadata(self) -> dict[str, object]:
        return copy.deepcopy(self.last_scoring)


class SmokePassGateTests(unittest.TestCase):
    def _artifacts(
        self, directory: Path, analysis: dict[str, object] | None = None
    ) -> tuple[Path, Path]:
        analysis_path = directory / "analysis.json"
        results_path = directory / "results.jsonl"
        _write_json(analysis_path, analysis if analysis is not None else _smoke_analysis())
        _write_jsonl(results_path, _smoke_rows())
        return analysis_path, results_path

    def test_exact_pass_artifacts_open_gate_and_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            analysis_path, results_path = self._artifacts(Path(temporary))
            analysis_sha256 = full.sha256_file(analysis_path)
            results_sha256 = full.sha256_file(results_path)
            gate = full.validate_smoke_gate(analysis_path, results_path)
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["result_rows"], 8)
        self.assertEqual(gate["assertion_count"], len(REQUIRED_SMOKE_ASSERTIONS))
        self.assertEqual(gate["analysis_sha256"], analysis_sha256)
        self.assertEqual(gate["results_sha256"], results_sha256)

    def test_gate_rejects_fail_status_false_assertion_and_missing_required_assertion(self):
        mutations = []
        failed = _smoke_analysis()
        failed["status"] = "FAIL"
        mutations.append(failed)
        false_assertion = _smoke_analysis()
        false_assertion["assertions"]["probabilities_valid"] = False
        mutations.append(false_assertion)
        missing_assertion = _smoke_analysis()
        del missing_assertion["assertions"]["pinned_helper_linkage"]
        mutations.append(missing_assertion)

        for analysis in mutations:
            with self.subTest(analysis=analysis):
                with tempfile.TemporaryDirectory() as temporary:
                    paths = self._artifacts(Path(temporary), analysis)
                    with self.assertRaises(ValueError):
                        full.validate_smoke_gate(*paths)

    def test_gate_rejects_model_substitution_and_nonreciprocal_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            analysis_path, results_path = self._artifacts(directory)
            rows = _smoke_rows()
            rows[0]["model_identifier"] = "substitute/model"
            _write_jsonl(results_path, rows)
            with self.assertRaises(ValueError):
                full.validate_smoke_gate(analysis_path, results_path)

            rows = _smoke_rows()
            rows[-1]["evidence_country"] = "Not the reciprocal endpoint"
            _write_jsonl(results_path, rows)
            with self.assertRaises(ValueError):
                full.validate_smoke_gate(analysis_path, results_path)

    def test_full_entrypoint_requires_smoke_artifacts_before_filesystem_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with self.assertRaises(PermissionError):
                full.run_llama_full(
                    model_directory=Path("missing-model"),
                    helper_path=Path("missing-helper"),
                    llama_cpp_directory=Path("missing-llama"),
                    output_directory=output,
                    allow_llama_full_run=True,
                )
            self.assertFalse(output.exists())


class RowGranularCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loader = DataLoader(str(ROOT / full.DATASET_PATH))
        cls.loader.load_dataset()
        cls.pair = cls.loader.get_question_pairs(
            str(ROOT / "data/pairs/country_pairs_smoke.json")
        )[0]
        cls.signature = {"fingerprint": "model-free-test-fingerprint"}
        cls.model_files = [{"filename": full.MODEL_SHARDS[0]["filename"]}]

    def _raw_prefix_snapshots(self) -> list[list[dict[str, object]]]:
        snapshots: list[list[dict[str, object]]] = []
        runner = FakeLlamaRunner()
        full.run_directed_unit(
            runner,
            self.loader,
            self.pair,
            self.model_files,
            self.signature["fingerprint"],
            on_row_saved=lambda rows: snapshots.append(copy.deepcopy(rows)),
        )
        self.assertEqual(runner.calls, 4)
        self.assertEqual([len(rows) for rows in snapshots], [1, 2, 3, 4])
        return snapshots

    def _partial_checkpoint(
        self, rows: list[dict[str, object]], complete: bool = False
    ) -> dict[str, object]:
        key = full.directed_key(self.pair)
        return {
            "schema_version": full.FULL_SCHEMA_VERSION,
            "run_fingerprint": self.signature["fingerprint"],
            "manifest_index": 0,
            "directed_key": list(key),
            "complete": complete,
            "conditions_completed": [row["condition"] for row in rows],
            "rows": rows,
        }

    def test_unfinalized_four_row_checkpoint_resumes_with_zero_inference(self):
        raw_rows = self._raw_prefix_snapshots()[-1]
        checkpoint = self._partial_checkpoint(raw_rows)
        resumed = full.validate_partial_checkpoint(
            checkpoint, self.pair, 0, self.signature
        )
        self.assertEqual(len(resumed), 4)

        runner = FakeLlamaRunner()
        callback_calls: list[list[dict[str, object]]] = []
        enriched = full.run_directed_unit(
            runner,
            self.loader,
            self.pair,
            self.model_files,
            self.signature["fingerprint"],
            existing_rows=resumed,
            on_row_saved=lambda rows: callback_calls.append(rows),
        )
        self.assertEqual(runner.calls, 0)
        self.assertEqual(callback_calls, [])
        self.assertEqual(len(enriched), 4)
        for row in enriched:
            self.assertIn("EO_raw", row)
            self.assertIn("EO_normalized", row)
            self.assertIn("base2_jensen_shannon_divergences", row)

        complete = {
            "schema_version": full.FULL_SCHEMA_VERSION,
            "run_fingerprint": self.signature["fingerprint"],
            "manifest_index": 0,
            "directed_key": list(full.directed_key(self.pair)),
            "complete": True,
            "conditions": list(full.EXPECTED_OUTPUT_CONDITIONS),
            "rows": enriched,
        }
        validated = full.validate_checkpoint(complete, self.pair, 0, self.signature)
        self.assertEqual(len(validated), 4)

    def test_partial_marker_and_condition_prefix_fail_closed(self):
        rows = self._raw_prefix_snapshots()[1]
        wrong_marker = self._partial_checkpoint(rows, complete=True)
        with self.assertRaises(ValueError):
            full.validate_partial_checkpoint(wrong_marker, self.pair, 0, self.signature)

        wrong_prefix = self._partial_checkpoint(rows)
        wrong_prefix["conditions_completed"] = ["baseline", "evidence"]
        with self.assertRaises(ValueError):
            full.validate_partial_checkpoint(wrong_prefix, self.pair, 0, self.signature)


if __name__ == "__main__":
    unittest.main()

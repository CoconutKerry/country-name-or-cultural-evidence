from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.package_gemma_gguf_full import (
    HISTORICAL_GEMMA_FAILURE_PATHS,
    PROJECT_ROOT,
    RUN_SIGNATURE_DEPENDENCIES,
    SMOKE_COMPATIBILITY_DEPENDENCIES,
    _excluded,
    selected_files,
    validate_package_provenance,
)
from scripts.run_gemma_gguf_full import (
    DATASET_PATH,
    DEFAULT_SEED,
    EXPECTED_OUTPUT_CONDITIONS,
    EXPECTED_QUESTION_CLUSTERS,
    EXPECTED_TARGET_UNIT_COUNT,
    FULL_SCHEMA_VERSION,
    GEMMA_CHAT_TEMPLATE_SHA256,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    PAIR_MANIFEST,
    PERMUTATION_KEY,
    PREINFERENCE_GATE,
    canonical_json_hash,
    directed_key,
    frozen_option_permutation_digest,
    sha256_file,
)
from src.gemma_gguf_spec import (
    LLAMA_CPP_COMMIT,
    LLAMA_CPP_VERSION,
    MODEL_FILES,
    MODEL_REPOSITORY,
    QUANTIZATION,
    UPSTREAM_CHECKPOINT,
    UPSTREAM_REVISION,
    UPSTREAM_REVISION_NOTE,
)


def _fingerprinted(core: dict) -> dict:
    return {**core, "fingerprint": canonical_json_hash(core)}


class GemmaPackageProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (PROJECT_ROOT / PAIR_MANIFEST).read_text(encoding="utf-8")
        )
        self.preinference_gate = json.loads(
            (PROJECT_ROOT / PREINFERENCE_GATE).read_text(encoding="utf-8")
        )
        model_files = [
            {
                "filename": item["filename"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in MODEL_FILES
        ]
        runtime_model_files = [
            {
                **item,
                "local_path": f"/not-packaged/{item['filename']}",
                "actual_size_bytes": item["size_bytes"],
                "actual_sha256": item["sha256"],
                "verified": True,
            }
            for item in MODEL_FILES
        ]
        runtime = {
            "threads": 8,
            "requested_context_size": 4096,
            "actual_context_size": 4096,
            "llama_version": LLAMA_CPP_VERSION,
            "chat_template_sha256": GEMMA_CHAT_TEMPLATE_SHA256,
        }
        current_dependencies = {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in RUN_SIGNATURE_DEPENDENCIES
        }
        signature_core = {
            "schema_version": FULL_SCHEMA_VERSION,
            "model": {
                "identifier": MODEL_IDENTIFIER,
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "quantization": QUANTIZATION,
                "upstream_checkpoint": UPSTREAM_CHECKPOINT,
                "upstream_revision": UPSTREAM_REVISION,
                "upstream_revision_note": UPSTREAM_REVISION_NOTE,
                "files": model_files,
            },
            "llama_cpp": {
                "commit": LLAMA_CPP_COMMIT,
                "version": LLAMA_CPP_VERSION,
                "chat_template_sha256": GEMMA_CHAT_TEMPLATE_SHA256,
                "helper_sha256": "1" * 64,
            },
            "backend": "llama.cpp",
            "scoring": "full_contextual_option_label_sequence_log_probability",
            "conditions": list(EXPECTED_OUTPUT_CONDITIONS),
            "directed_unit_key": [
                "question_id",
                "label_country",
                "evidence_country",
            ],
            "directed_units": [
                list(directed_key(pair)) for pair in self.manifest
            ],
            "question_clusters": EXPECTED_QUESTION_CLUSTERS,
            "target_units": EXPECTED_TARGET_UNIT_COUNT,
            "seed": DEFAULT_SEED,
            "permutation_key": PERMUTATION_KEY,
            "fixed_permutation_manifest_sha256": frozen_option_permutation_digest(
                self.manifest
            ),
            "threads": 8,
            "context_size": 4096,
            "jensen_shannon_base": 2,
            "code_data_dependency_sha256": current_dependencies,
        }
        self.signature = _fingerprinted(signature_core)

        smoke_source_hashes = {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in SMOKE_COMPATIBILITY_DEPENDENCIES
        }
        compatibility_core = {
            "model_identifier": MODEL_IDENTIFIER,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "model_files": model_files,
            "helper_sha256": "1" * 64,
            "llama_cpp_commit": LLAMA_CPP_COMMIT,
            "threads": 8,
            "context_size": 4096,
            "source_sha256": smoke_source_hashes,
            "preinference_gate_sha256": sha256_file(
                PROJECT_ROOT / PREINFERENCE_GATE
            ),
            "preinference_gate_completed_at_utc": self.preinference_gate.get(
                "completed_at_utc"
            ),
        }
        self.smoke_gate = {
            "status": "PASS",
            "compatibility": _fingerprinted(compatibility_core),
        }
        self.runtime_record = {
            "run_fingerprint": self.signature["fingerprint"],
            "runtime": runtime,
            "model_files": runtime_model_files,
            "llama_cpp_commit": LLAMA_CPP_COMMIT,
            "preinference_gate": deepcopy(self.preinference_gate),
            "genuine_smoke_gate": deepcopy(self.smoke_gate),
        }

    def _validate(self) -> None:
        validate_package_provenance(
            self.signature,
            self.runtime_record,
            self.smoke_gate,
            self.manifest,
            self.preinference_gate,
            repository_root=PROJECT_ROOT,
        )

    def test_current_canonical_provenance_passes(self) -> None:
        self._validate()

    def test_rejects_tampered_run_signature_fingerprint(self) -> None:
        self.signature["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "canonical fingerprint"):
            self._validate()

    def test_rejects_self_consistent_stale_run_dependency_hash(self) -> None:
        self.signature["code_data_dependency_sha256"]["src/main.py"] = "0" * 64
        core = {key: value for key, value in self.signature.items() if key != "fingerprint"}
        self.signature["fingerprint"] = canonical_json_hash(core)
        self.runtime_record["run_fingerprint"] = self.signature["fingerprint"]
        with self.assertRaisesRegex(RuntimeError, "dependency hash mismatch: src/main.py"):
            self._validate()

    def test_rejects_self_consistent_semantic_signature_drift(self) -> None:
        self.signature["model"]["revision"] = "different-revision"
        core = {key: value for key, value in self.signature.items() if key != "fingerprint"}
        self.signature["fingerprint"] = canonical_json_hash(core)
        self.runtime_record["run_fingerprint"] = self.signature["fingerprint"]
        with self.assertRaisesRegex(RuntimeError, "run signature does not match"):
            self._validate()

    def test_rejects_runtime_embedded_smoke_gate_mismatch(self) -> None:
        self.runtime_record["genuine_smoke_gate"]["status"] = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "embedded genuine smoke gate"):
            self._validate()

    def test_rejects_self_consistent_stale_smoke_source_hash(self) -> None:
        compatibility = self.smoke_gate["compatibility"]
        compatibility["source_sha256"]["src/main.py"] = "0" * 64
        core = {
            key: value for key, value in compatibility.items() if key != "fingerprint"
        }
        compatibility["fingerprint"] = canonical_json_hash(core)
        self.runtime_record["genuine_smoke_gate"] = deepcopy(self.smoke_gate)
        with self.assertRaisesRegex(RuntimeError, "source hash mismatch: src/main.py"):
            self._validate()

    def test_historical_failure_artifacts_are_never_selected(self) -> None:
        for relative in HISTORICAL_GEMMA_FAILURE_PATHS:
            with self.subTest(relative=relative.as_posix()):
                self.assertTrue(_excluded(relative))
        selected = {
            path.relative_to(PROJECT_ROOT) for path in selected_files()
        }
        self.assertTrue(HISTORICAL_GEMMA_FAILURE_PATHS.isdisjoint(selected))
        for required_current_smoke_artifact in (
            Path("experiments/smoke_genuine_gemma_gguf/analysis.json"),
            Path("experiments/smoke_genuine_gemma_gguf/run.log"),
            Path(
                "experiments/smoke_genuine_gemma_gguf/"
                "GENUINE_GEMMA_GGUF_SMOKE_REPORT.md"
            ),
        ):
            with self.subTest(required=required_current_smoke_artifact.as_posix()):
                self.assertFalse(_excluded(required_current_smoke_artifact))


if __name__ == "__main__":
    unittest.main()

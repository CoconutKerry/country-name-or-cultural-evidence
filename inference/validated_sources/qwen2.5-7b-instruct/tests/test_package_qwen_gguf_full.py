"""Tests for fail-closed Qwen full-run packaging."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import scripts.package_qwen_gguf_full as package
from scripts.analyze_qwen_gguf_full import analyze_full_run, write_outputs
from scripts.run_qwen_gguf_full import canonical_json_hash
from tests.test_qwen_gguf_full_analysis import reciprocal_fixture


def _signature(project_root: Path):
    dependency = project_root / "source.py"
    dependency.write_text("pinned\n", encoding="utf-8")
    model_files = [
        {"filename": "part-1.gguf", "size_bytes": 10, "sha256": "1" * 64},
        {"filename": "part-2.gguf", "size_bytes": 20, "sha256": "2" * 64},
    ]
    core = {
        "schema_version": package.FULL_SCHEMA_VERSION,
        "backend": "llama.cpp",
        "conditions": list(package.EXPECTED_OUTPUT_CONDITIONS),
        "model": {
            "identifier": package.MODEL_IDENTIFIER,
            "repository": package.MODEL_REPOSITORY,
            "revision": package.MODEL_REVISION,
            "quantization": package.QUANTIZATION,
            "remote_last_modified": "2024-01-01T00:00:00Z",
            "weight_upload_commit": "weight-commit",
            "files": model_files,
        },
        "llama_cpp": {
            "commit": package.LLAMA_CPP_COMMIT,
            "version": package.LLAMA_CPP_VERSION,
            "chat_template_sha256": package.QWEN_CHAT_TEMPLATE_SHA256,
            "helper_sha256": "3" * 64,
        },
        "code_data_dependency_sha256": {
            "source.py": hashlib.sha256(dependency.read_bytes()).hexdigest()
        },
    }
    return {**core, "fingerprint": canonical_json_hash(core)}


def _runtime_records(signature):
    fingerprint = signature["fingerprint"]
    model = signature["model"]
    runtime = {
        "backend": "llama.cpp",
        "synthetic": False,
        "model_identifier": package.MODEL_IDENTIFIER,
        "repository": package.MODEL_REPOSITORY,
        "revision": package.MODEL_REVISION,
        "quantization": package.QUANTIZATION,
        "chat_template_source": "embedded_gguf_metadata",
        "chat_template_sha256": package.QWEN_CHAT_TEMPLATE_SHA256,
        "llama_version": package.LLAMA_CPP_VERSION,
        "gpu_layers": 0,
        "candidate_scoring": "full_vocabulary_low_level_logits",
        "generated_answer_parsing": False,
    }
    runtime_record = {
        "run_fingerprint": fingerprint,
        "llama_cpp_commit": package.LLAMA_CPP_COMMIT,
        "remote_model_last_modified": model["remote_last_modified"],
        "remote_weight_commit": model["weight_upload_commit"],
        "runtime": runtime,
        "model_files": [
            {
                "filename": item["filename"],
                "actual_size_bytes": item["size_bytes"],
                "actual_sha256": item["sha256"],
            }
            for item in model["files"]
        ],
    }
    progress = {
        "schema_version": package.FULL_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "expected_directed_units": 200,
        "expected_rows": 800,
        "completed_directed_units": 200,
    }
    completion = {
        "schema_version": package.FULL_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "model_identifier": package.MODEL_IDENTIFIER,
        "model_revision": package.MODEL_REVISION,
        "llama_cpp_commit": package.LLAMA_CPP_COMMIT,
    }
    return runtime_record, progress, completion


class QwenFullPackageTests(unittest.TestCase):
    def test_direct_script_invocation_imports_successfully(self):
        completed = subprocess.run(
            [sys.executable, "scripts/package_qwen_gguf_full.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("Validate and package", completed.stdout)

    def test_pinned_provenance_recomputes_fingerprint_and_dependency_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            signature = _signature(project_root)
            runtime, progress, completion = _runtime_records(signature)
            with patch.object(package, "PROJECT_ROOT", project_root):
                result = package.validate_pinned_provenance(
                    signature, runtime, progress, completion
                )
                self.assertEqual(result["run_fingerprint"], signature["fingerprint"])
                self.assertEqual(result["dependency_files_reverified"], 1)

                tampered = deepcopy(signature)
                tampered["threads"] = 99
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    package.validate_pinned_provenance(
                        tampered, runtime, progress, completion
                    )

                (project_root / "source.py").write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed after inference"):
                    package.validate_pinned_provenance(
                        signature, runtime, progress, completion
                    )

    def test_external_asset_provenance_is_cross_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_root = root / "repo"
            output = project_root / "out"
            runtime_directory = root / "runtime"
            output.mkdir(parents=True)
            runtime_directory.mkdir()
            signature = _signature(project_root)
            runtime_record, _, _ = _runtime_records(signature)
            (output / "run_signature.json").write_text(json.dumps(signature), encoding="utf-8")
            (output / "runtime.json").write_text(json.dumps(runtime_record), encoding="utf-8")
            external = {
                "llama_cpp": {
                    "commit": package.LLAMA_CPP_COMMIT,
                    "version_reported_by_cmake": package.LLAMA_CPP_VERSION,
                },
                "helper": {"sha256": signature["llama_cpp"]["helper_sha256"]},
                "model": {
                    "repository": package.MODEL_REPOSITORY,
                    "requested_revision": package.MODEL_REVISION,
                    "resolved_revision": package.MODEL_REVISION,
                    "quantization": package.QUANTIZATION,
                    "files": signature["model"]["files"],
                    "gguf_metadata": {
                        "chat_template_sha256": package.QWEN_CHAT_TEMPLATE_SHA256
                    },
                },
            }
            (runtime_directory / "runtime_provenance.json").write_text(
                json.dumps(external), encoding="utf-8"
            )
            identities = [
                package.LLAMA_CPP_COMMIT,
                package.MODEL_REVISION,
                signature["llama_cpp"]["helper_sha256"],
                *(item["sha256"] for item in signature["model"]["files"]),
            ]
            (runtime_directory / "PROVENANCE.md").write_text(
                "\n".join(identities), encoding="utf-8"
            )
            with (
                patch.object(package, "PROJECT_ROOT", project_root),
                patch.object(package, "OUTPUT_RELATIVE", Path("out")),
            ):
                files, summary = package.validate_external_runtime_provenance(
                    runtime_directory
                )
                self.assertEqual(len(files), 2)
                self.assertEqual(summary["model_shards"], 2)

                external["model"]["resolved_revision"] = "wrong"
                (runtime_directory / "runtime_provenance.json").write_text(
                    json.dumps(external), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "resolved revision"):
                    package.validate_external_runtime_provenance(runtime_directory)

    def test_derived_outputs_are_exactly_bound_to_recomputed_analysis(self):
        manifest, rows, expectations = reciprocal_fixture()
        analysis, enriched = analyze_full_run(
            rows,
            manifest,
            expectations=expectations,
            bootstrap_replicates=5,
            bootstrap_seed=42,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_outputs(output, analysis, enriched)
            saved = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            hashes = package.validate_derived_artifacts(
                output, saved, analysis, enriched
            )
            self.assertEqual(len(hashes), 9)

            path = output / "bootstrap_summary.csv"
            path.write_text(path.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bootstrap_summary.csv"):
                package.validate_derived_artifacts(output, saved, analysis, enriched)

    def test_destination_must_be_outside_project_and_archive_is_atomic_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_root = root / "repo"
            runtime_directory = root / "runtime"
            delivery = root / "delivery"
            project_root.mkdir()
            runtime_directory.mkdir()
            (project_root / "source.txt").write_text("repository bytes\n", encoding="utf-8")
            runtime_files = [
                runtime_directory / "runtime_provenance.json",
                runtime_directory / "PROVENANCE.md",
            ]
            runtime_files[0].write_text("{}\n", encoding="utf-8")
            runtime_files[1].write_text("provenance\n", encoding="utf-8")
            validation = {"status": "PASS"}
            external_summary = {"status": "PASS"}

            with (
                patch.object(package, "PROJECT_ROOT", project_root),
                patch.object(package, "validate_outputs", return_value=validation),
                patch.object(
                    package,
                    "validate_external_runtime_provenance",
                    return_value=(runtime_files, external_summary),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "outside the project"):
                    package.build_archive(project_root / "bad.zip", runtime_directory)

                destination = delivery / "package.zip"
                result = package.build_archive(destination, runtime_directory)
                self.assertTrue(destination.is_file())
                self.assertTrue(destination.with_suffix(".zip.sha256").is_file())
                self.assertEqual(result["members"], 4)
                with zipfile.ZipFile(destination) as archive:
                    names = set(archive.namelist())
                    manifest_name = (
                        f"{package.ARCHIVE_ROOT}/"
                        f"{package.PACKAGE_MANIFEST_RELATIVE.as_posix()}"
                    )
                    manifest = json.loads(archive.read(manifest_name))
                self.assertEqual(
                    names,
                    {entry["archive_path"] for entry in manifest["files"]}
                    | {manifest_name},
                )
                bad_manifest = deepcopy(manifest)
                bad_manifest["files"][0]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    package.verify_archive(
                        destination,
                        project_root / package.PACKAGE_MANIFEST_RELATIVE,
                        bad_manifest,
                    )

                # A failed rebuild must leave the already-verified destination intact.
                original_bytes = destination.read_bytes()
                with patch.object(package, "_write_member", side_effect=RuntimeError("boom")):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        package.build_archive(destination, runtime_directory)
                self.assertEqual(destination.read_bytes(), original_bytes)
                self.assertFalse(list(delivery.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()

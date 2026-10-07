#!/usr/bin/env python3
"""Validate and package the complete Gemma-only repository/results deliverable."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import zipfile
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_gemma_gguf_full import (
    DATASET_PATH,
    DEFAULT_SEED,
    EXPECTED_DIRECTED_UNIT_COUNT,
    EXPECTED_OUTPUT_CONDITIONS,
    EXPECTED_QUESTION_CLUSTERS,
    EXPECTED_ROW_COUNT,
    EXPECTED_TARGET_UNIT_COUNT,
    FULL_SCHEMA_VERSION,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    PAIR_MANIFEST,
    PAIR_MANIFEST_SHA256,
    PERMUTATION_KEY,
    PREINFERENCE_GATE,
    DATASET_SHA256,
    canonical_json_hash,
    checkpoint_path,
    directed_key,
    frozen_option_permutation_digest,
    sha256_file,
    validate_checkpoint,
    validate_complete_results,
    validate_preinference_gate,
)
from src.gemma_gguf_spec import (
    LLAMA_CPP_COMMIT,
    MODEL_FILES,
    MODEL_REPOSITORY,
    QUANTIZATION,
    UPSTREAM_CHECKPOINT,
    UPSTREAM_REVISION,
    UPSTREAM_REVISION_NOTE,
)


ARCHIVE_ROOT = "cultural-alignment-audit-main"
FULL_OUTPUT = Path("experiments/gemma_gguf_full")
SMOKE_OUTPUT = Path("experiments/smoke_genuine_gemma_gguf")
MANIFEST_PATH = Path("audit/GEMMA_GGUF_PACKAGE_MANIFEST.json")
CHANGED_FILES_PATH = Path("audit/GEMMA_GGUF_FULL_CHANGED_FILES.txt")
FULL_REPORT_PATH = Path("audit/GEMMA_GGUF_FULL_REPORT.md")
CHECKSUMS_PATH = FULL_OUTPUT / "SHA256SUMS.txt"
TOKEN_PATTERN = re.compile(rb"hf_[A-Za-z0-9]{20,}")
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    "build",
    "runtime",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp", ".tmp", ".gguf"}
EXCLUDED_PREFIXES = (
    Path("analysis"),
    Path("analysis/output"),
    Path("experiments/main"),
    Path("experiments/pilot"),
    Path("experiments/smoke"),
    Path("experiments/qwen_gguf_full"),
    Path("experiments/smoke_genuine_qwen_gguf"),
)
REQUIRED_FULL_OUTPUTS = (
    "results.jsonl",
    "results.csv",
    "analysis.json",
    "bootstrap_summary.csv",
    "conflict_classifications.csv",
    "directed_unit_metrics.csv",
    "target_unit_metrics.csv",
    "metric_tables.json",
    "tables/model_metrics.tex",
    "tables/conflict_classification.tex",
    "EXPERIMENT_REPORT.md",
    "EXCLUSION_AND_FAILURE_REPORT.md",
    "completion.json",
    "environment.json",
    "progress.json",
    "run.log",
    "run_signature.json",
    "runtime.json",
    "row_journal.jsonl",
)
REQUIRED_SMOKE_OUTPUTS = (
    "results.jsonl",
    "analysis.json",
    "run.log",
    "smoke_gate.json",
    "serialization_preflight.json",
    "GENUINE_GEMMA_GGUF_SMOKE_REPORT.md",
)
RUN_SIGNATURE_DEPENDENCIES = (
    "requirements-inference.lock",
    DATASET_PATH,
    PAIR_MANIFEST,
    "src/data_loader.py",
    "src/gguf_runner.py",
    "src/gguf_score_helper.cpp",
    "src/main.py",
    "src/metrics.py",
    "src/prompt_builder.py",
    "src/scoring.py",
    "scripts/run_gemma_gguf_full.py",
)
SMOKE_COMPATIBILITY_DEPENDENCIES = (
    "src/gemma_gguf_spec.py",
    "src/gguf_runner.py",
    "src/gguf_score_helper.cpp",
    "src/main.py",
    "src/metrics.py",
    "src/prompt_builder.py",
    "src/scoring.py",
    "scripts/run_genuine_gemma_gguf_smoke.py",
    "scripts/run_gemma_gguf_full.py",
)
HISTORICAL_GEMMA_FAILURE_PATHS = frozenset(
    {
        Path("audit/GEMMA_GGUF_SMOKE_FAILURE_DIAGNOSTIC.md"),
        Path("audit/GEMMA_GGUF_SMOKE_FAILURE_PACKAGE_MANIFEST.json"),
        Path("audit/GEMMA_GGUF_SMOKE_FAILURE_SHA256SUMS.txt"),
        Path("experiments/smoke_genuine_gemma_gguf/failure_traceback.txt"),
    }
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        _require(isinstance(value, dict), f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _expected_model_file_identities() -> list[dict[str, Any]]:
    return [
        {
            "filename": item["filename"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in MODEL_FILES
    ]


def _runtime_model_file_identities(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "runtime model-files record is missing")
    identities: list[dict[str, Any]] = []
    for item in value:
        _require(isinstance(item, Mapping), "runtime model-files record is malformed")
        identities.append(
            {
                "filename": item.get("filename"),
                "size_bytes": item.get("actual_size_bytes"),
                "sha256": item.get("actual_sha256"),
            }
        )
    return identities


def _current_hashes(
    repository_root: Path,
    relatives: Iterable[str],
    *,
    label: str,
) -> dict[str, str]:
    identities: dict[str, str] = {}
    for relative in relatives:
        path = repository_root / relative
        _require(path.is_file(), f"{label} file is missing: {relative}")
        identities[relative] = sha256_file(path)
    return identities


def _validate_current_hash_record(
    recorded: Any,
    current: Mapping[str, str],
    *,
    label: str,
) -> None:
    _require(isinstance(recorded, Mapping), f"{label} hash record is missing")
    _require(
        set(recorded) == set(current),
        f"{label} hash-record path set drifted",
    )
    for relative, current_sha256 in current.items():
        _require(
            recorded.get(relative) == current_sha256,
            f"{label} hash mismatch: {relative}",
        )


def validate_package_provenance(
    signature: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
    disk_smoke_gate: Mapping[str, Any],
    manifest: list[dict[str, Any]],
    preinference_gate: Mapping[str, Any],
    *,
    repository_root: Path = PROJECT_ROOT,
) -> None:
    """Rebuild every package-time provenance binding from current inputs.

    The helper binary and model weights are deliberately excluded from the
    downloadable repository.  Their immutable identities are therefore
    cross-bound between the full-run signature, the smoke compatibility
    record, and the pinned model specification; code/data and both gates are
    independently re-hashed from disk here.
    """

    _require(isinstance(signature, Mapping), "run signature is malformed")
    _require(isinstance(runtime_record, Mapping), "runtime record is malformed")
    signature_core = dict(signature)
    signature_fingerprint = signature_core.pop("fingerprint", None)
    _require(
        isinstance(signature_fingerprint, str)
        and canonical_json_hash(signature_core) == signature_fingerprint,
        "run-signature canonical fingerprint mismatch",
    )

    current_dependencies = _current_hashes(
        repository_root,
        RUN_SIGNATURE_DEPENDENCIES,
        label="run-signature dependency",
    )
    _validate_current_hash_record(
        signature_core.get("code_data_dependency_sha256"),
        current_dependencies,
        label="run-signature dependency",
    )

    embedded_smoke_gate = runtime_record.get("genuine_smoke_gate")
    _require(
        isinstance(embedded_smoke_gate, Mapping)
        and dict(embedded_smoke_gate) == dict(disk_smoke_gate),
        "runtime-embedded genuine smoke gate differs from the disk gate",
    )
    embedded_preinference_gate = runtime_record.get("preinference_gate")
    _require(
        isinstance(embedded_preinference_gate, Mapping)
        and dict(embedded_preinference_gate) == dict(preinference_gate),
        "runtime-embedded pre-inference gate differs from the disk gate",
    )

    compatibility = disk_smoke_gate.get("compatibility")
    _require(
        isinstance(compatibility, Mapping),
        "genuine-smoke compatibility record is missing",
    )
    compatibility_core = dict(compatibility)
    compatibility_fingerprint = compatibility_core.pop("fingerprint", None)
    _require(
        isinstance(compatibility_fingerprint, str)
        and canonical_json_hash(compatibility_core) == compatibility_fingerprint,
        "genuine-smoke compatibility canonical fingerprint mismatch",
    )
    current_smoke_dependencies = _current_hashes(
        repository_root,
        SMOKE_COMPATIBILITY_DEPENDENCIES,
        label="genuine-smoke source",
    )
    _validate_current_hash_record(
        compatibility_core.get("source_sha256"),
        current_smoke_dependencies,
        label="genuine-smoke source",
    )

    expected_model_files = _expected_model_file_identities()
    _require(
        _runtime_model_file_identities(runtime_record.get("model_files"))
        == expected_model_files,
        "runtime model-file identity drifted",
    )
    runtime = runtime_record.get("runtime")
    _require(isinstance(runtime, Mapping), "loaded runtime metadata is missing")
    threads = runtime.get("threads")
    context_size = runtime.get("requested_context_size")
    _require(
        isinstance(threads, int) and not isinstance(threads, bool) and threads > 0,
        "loaded runtime thread count is invalid",
    )
    _require(
        isinstance(context_size, int)
        and not isinstance(context_size, bool)
        and context_size > 0,
        "loaded runtime context size is invalid",
    )
    _require(
        runtime_record.get("llama_cpp_commit") == LLAMA_CPP_COMMIT,
        "runtime llama.cpp commit drifted",
    )

    expected_compatibility_core = {
        "model_identifier": MODEL_IDENTIFIER,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_files": expected_model_files,
        "helper_sha256": signature_core.get("llama_cpp", {}).get("helper_sha256")
        if isinstance(signature_core.get("llama_cpp"), Mapping)
        else None,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "threads": threads,
        "context_size": context_size,
        "source_sha256": current_smoke_dependencies,
        "preinference_gate_sha256": sha256_file(
            repository_root / PREINFERENCE_GATE
        ),
        "preinference_gate_completed_at_utc": preinference_gate.get(
            "completed_at_utc"
        ),
    }
    _require(
        compatibility_core == expected_compatibility_core,
        "genuine-smoke compatibility does not match current package inputs",
    )

    expected_signature_core = {
        "schema_version": FULL_SCHEMA_VERSION,
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "quantization": QUANTIZATION,
            "upstream_checkpoint": UPSTREAM_CHECKPOINT,
            "upstream_revision": UPSTREAM_REVISION,
            "upstream_revision_note": UPSTREAM_REVISION_NOTE,
            "files": expected_model_files,
        },
        "llama_cpp": {
            "commit": LLAMA_CPP_COMMIT,
            "version": runtime.get("llama_version"),
            "chat_template_sha256": runtime.get("chat_template_sha256"),
            "helper_sha256": compatibility_core.get("helper_sha256"),
        },
        "backend": "llama.cpp",
        "scoring": "full_contextual_option_label_sequence_log_probability",
        "conditions": list(EXPECTED_OUTPUT_CONDITIONS),
        "directed_unit_key": [
            "question_id",
            "label_country",
            "evidence_country",
        ],
        "directed_units": [list(directed_key(pair)) for pair in manifest],
        "question_clusters": EXPECTED_QUESTION_CLUSTERS,
        "target_units": EXPECTED_TARGET_UNIT_COUNT,
        "seed": DEFAULT_SEED,
        "permutation_key": PERMUTATION_KEY,
        "fixed_permutation_manifest_sha256": frozen_option_permutation_digest(
            manifest
        ),
        "threads": threads,
        "context_size": context_size,
        "jensen_shannon_base": 2,
        "code_data_dependency_sha256": current_dependencies,
    }
    _require(
        signature_core == expected_signature_core,
        "run signature does not match current code, data, model, runtime, or settings",
    )
    _require(
        runtime_record.get("run_fingerprint") == signature_fingerprint,
        "runtime and run-signature fingerprints disagree",
    )


def validate_outputs() -> dict[str, Any]:
    full = PROJECT_ROOT / FULL_OUTPUT
    smoke = PROJECT_ROOT / SMOKE_OUTPUT
    for relative in REQUIRED_FULL_OUTPUTS:
        _require((full / relative).is_file(), f"missing full-run output: {relative}")
    for relative in REQUIRED_SMOKE_OUTPUTS:
        _require((smoke / relative).is_file(), f"missing smoke output: {relative}")
    completion = _json(full / "completion.json")
    analysis = _json(full / "analysis.json")
    smoke_gate = _json(smoke / "smoke_gate.json")
    serialization_preflight = _json(smoke / "serialization_preflight.json")
    runtime = _json(full / "runtime.json")
    signature = _json(full / "run_signature.json")
    manifest_path = PROJECT_ROOT / PAIR_MANIFEST
    _require(
        sha256_file(manifest_path) == PAIR_MANIFEST_SHA256,
        "cleaned manifest hash drifted",
    )
    _require(
        sha256_file(PROJECT_ROOT / "data/processed/dataset_v2.json") == DATASET_SHA256,
        "cleaned dataset hash drifted",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, list), "cleaned manifest is not a JSON list")
    results = _jsonl(full / "results.jsonl")
    journal = _jsonl(full / "row_journal.jsonl")
    _require(journal == results, "row journal does not exactly equal final results")
    preinference_gate = validate_preinference_gate(PROJECT_ROOT)
    validate_complete_results(
        results,
        manifest,
        runtime=runtime.get("runtime"),
        provenance=signature,
    )
    _require(completion.get("status") == "PASS", "full completion did not pass")
    _require(completion.get("rows") == EXPECTED_ROW_COUNT, "full row count is not 800")
    _require(analysis.get("validation_status") == "PASS", "analysis did not pass")
    _require(analysis.get("counts", {}).get("result_rows") == 800, "analysis row count drifted")
    _require(
        analysis.get("counts", {}).get("question_id_clusters") == 44,
        "analysis question-cluster count drifted",
    )
    _require(
        analysis.get("bootstrap", {}).get("method", {}).get("n_replicates") == 10_000,
        "analysis does not contain exactly 10,000 bootstrap replicates",
    )
    _require(
        analysis.get("bootstrap", {}).get("method", {}).get("cluster_key")
        == "question_id",
        "bootstrap cluster key drifted",
    )
    _require(
        analysis.get("bootstrap", {})
        .get("method", {})
        .get("independent_condition_row_resampling")
        is False,
        "bootstrap independently resampled condition rows",
    )
    _require(
        analysis.get("bootstrap", {}).get("method", {}).get("n_clusters") == 44,
        "bootstrap does not contain 44 question_id clusters",
    )
    _require(
        analysis.get("bootstrap", {}).get("method", {}).get("confidence_level")
        == 0.95,
        "bootstrap confidence level is not 95%",
    )
    _require(
        sum(
            analysis.get("conflict_classifications", {})
            .get("counts", {})
            .values()
        )
        == 200,
        "conflict classifications do not cover 200 directed units",
    )
    _require(
        analysis.get("conflict_classifications", {}).get("tolerance") == 1e-12,
        "conflict tie tolerance drifted",
    )
    _require(smoke_gate.get("status") == "PASS", "genuine smoke did not pass")
    _require(smoke_gate.get("row_count") == 8, "genuine smoke row count is not 8")
    _require(
        isinstance(smoke_gate.get("assertions"), dict)
        and smoke_gate["assertions"]
        and all(value is True for value in smoke_gate["assertions"].values()),
        "one or more genuine-smoke assertions did not pass",
    )
    _require(
        smoke_gate.get("analysis_sha256") == sha256_file(smoke / "analysis.json"),
        "genuine-smoke analysis changed after its gate",
    )
    _require(
        smoke_gate.get("results_sha256") == sha256_file(smoke / "results.jsonl"),
        "genuine-smoke results changed after its gate",
    )
    _require(
        smoke_gate.get("serialization_preflight_sha256")
        == sha256_file(smoke / "serialization_preflight.json"),
        "genuine-smoke serialization preflight changed after its gate",
    )
    _require(
        serialization_preflight.get("status") == "PASS"
        and serialization_preflight.get("inference_performed") is False,
        "serialization-only live preflight did not pass without inference",
    )
    _require(
        isinstance(serialization_preflight.get("checks"), dict)
        and serialization_preflight["checks"]
        and all(
            value is True for value in serialization_preflight["checks"].values()
        ),
        "one or more serialization-preflight checks did not pass",
    )
    validate_package_provenance(
        signature,
        runtime,
        smoke_gate,
        manifest,
        preinference_gate,
        repository_root=PROJECT_ROOT,
    )
    _require(
        runtime.get("runtime", {}).get("model_identifier") == MODEL_IDENTIFIER,
        "runtime model identity drifted",
    )
    _require(completion.get("model_revision") == MODEL_REVISION, "model revision drifted")
    _require(
        completion.get("llama_cpp_commit") == LLAMA_CPP_COMMIT,
        "llama.cpp commit drifted",
    )
    fingerprint = signature.get("fingerprint")
    _require(
        isinstance(fingerprint, str)
        and completion.get("run_fingerprint") == fingerprint
        and runtime.get("run_fingerprint") == fingerprint,
        "full result, runtime, and run-signature fingerprints disagree",
    )
    _require(len(results) == 800, f"results JSONL has {len(results)} rows")
    _require(
        analysis.get("source_results", {}).get("sha256")
        == sha256_file(full / "results.jsonl"),
        "analysis is not bound to the final results JSONL",
    )
    _require(
        analysis.get("source_manifest", {}).get("sha256") == PAIR_MANIFEST_SHA256,
        "analysis is not bound to the cleaned manifest",
    )
    checkpoint_directory = full / "checkpoints"
    expected_checkpoint_paths = {
        checkpoint_path(checkpoint_directory, directed_key(pair)).resolve()
        for pair in manifest
    }
    observed_checkpoint_paths = {
        path.resolve() for path in checkpoint_directory.glob("*.json")
    }
    _require(
        observed_checkpoint_paths == expected_checkpoint_paths,
        "checkpoint set does not exactly cover the cleaned manifest",
    )
    reconstructed: list[dict[str, Any]] = []
    for index, pair in enumerate(manifest):
        checkpoint = _json(checkpoint_path(checkpoint_directory, directed_key(pair)))
        checkpoint_rows = validate_checkpoint(checkpoint, pair, index, signature)
        _require(len(checkpoint_rows) == 4, "a final checkpoint is not complete")
        reconstructed.extend(checkpoint_rows)
    _require(reconstructed == results, "checkpoint reconstruction differs from results")
    _require(len(manifest) == EXPECTED_DIRECTED_UNIT_COUNT, "manifest count drifted")
    return {
        "completion": completion,
        "analysis": analysis,
        "smoke_gate": smoke_gate,
        "runtime": runtime,
    }


def write_audit_files(validated: Mapping[str, Any]) -> None:
    analysis = validated["analysis"]
    changed = """Gemma-only task source and schema changes
=========================================

Added:
- src/gemma_gguf_spec.py
- scripts/run_genuine_gemma_gguf_smoke.py
- scripts/run_gemma_gguf_full.py
- scripts/analyze_gemma_gguf_full.py
- scripts/package_gemma_gguf_full.py
- tests/test_gemma_gguf_full.py
- tests/test_gemma_gguf_full_analysis.py

Modified:
- src/gguf_runner.py
- src/gguf_score_helper.cpp
- src/data_loader.py
- src/metrics.py
- src/main.py
- src/colab_runner.py
- src/clustered_bootstrap.py
- scripts/analyze_results.py
- scripts/bootstrap_analysis.py
- tests for metric/schema/data/runtime behavior

Generated:
- mandatory 8-row genuine Gemma smoke artifacts
- 800-row full Gemma results and atomic row checkpoints
- 10,000-replicate question-clustered bootstrap analysis
- CSV, JSON, LaTeX, reports, logs, and checksums

No Qwen or other model was run by this task. Model weights and llama.cpp build
artifacts are deliberately excluded from the downloadable ZIP.
"""
    atomic_write(PROJECT_ROOT / CHANGED_FILES_PATH, changed)
    model_name = analysis["model"]["model_name"]
    bootstrap_metrics = analysis["bootstrap"]["models"][model_name]
    metric_lines = []
    for metric, summary in analysis["metric_summaries"].items():
        interval = bootstrap_metrics[metric]["percentile_95_ci"]
        metric_lines.append(
            f"| `{metric}` | {summary['n']} | {summary['mean']:.6f} | "
            f"[{interval['lower']:.6f}, {interval['upper']:.6f}] |"
        )
    report = [
        "# Gemma-2-9B-It GGUF Full Experiment Report",
        "",
        "**Status: PASS**",
        "",
        "- Exactly 8 valid genuine-smoke rows passed before the full run.",
        "- Exactly 200 reciprocal directed units and 800 final rows passed validation.",
        "- Backend: genuine CPU-only llama.cpp; no generation, parsing, fallback, or substitution.",
        f"- Model repository / revision: `{MODEL_REPOSITORY}` / `{MODEL_REVISION}`",
        f"- Upstream checkpoint: `{UPSTREAM_CHECKPOINT}`",
        f"- Quantization: `{QUANTIZATION}`; file SHA-256: `{MODEL_FILES[0]['sha256']}`",
        f"- llama.cpp commit: `{LLAMA_CPP_COMMIT}`",
        f"- Dataset SHA-256: `{DATASET_SHA256}`",
        f"- Pair-manifest SHA-256: `{PAIR_MANIFEST_SHA256}`",
        "",
        "| Metric | N | Estimate | 95% percentile CI |",
        "|---|---:|---:|---:|",
        *metric_lines,
        "",
        "Conflict ties use absolute EO_raw tolerance 1e-12. Bootstrap resampling uses "
        "10,000 whole question_id clusters and retains every directed unit and condition.",
        "",
    ]
    atomic_write(PROJECT_ROOT / FULL_REPORT_PATH, "\n".join(report))


def write_result_checksums() -> None:
    output = PROJECT_ROOT / FULL_OUTPUT
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path != output / CHECKSUMS_PATH.name
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in files]
    atomic_write(output / CHECKSUMS_PATH.name, "\n".join(lines) + "\n")


def _excluded(relative: Path) -> bool:
    if relative in HISTORICAL_GEMMA_FAILURE_PATHS:
        return True
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if any(relative == prefix or prefix in relative.parents for prefix in EXCLUDED_PREFIXES):
        return True
    name = relative.name
    if name.startswith("QWEN_GGUF") or name.startswith("GENUINE_QWEN"):
        return True
    return False


def selected_files() -> list[Path]:
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if _excluded(relative) or relative == MANIFEST_PATH:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def validate_no_credentials(files: Iterable[Path]) -> None:
    for path in files:
        data = path.read_bytes()
        if TOKEN_PATTERN.search(data):
            raise RuntimeError(f"credential-like Hugging Face token found in {path}")


def _zip_write(archive: zipfile.ZipFile, source: Path, archive_name: str) -> None:
    info = zipfile.ZipInfo(archive_name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, source.read_bytes())


def build_archive(destination: Path) -> dict[str, Any]:
    validated = validate_outputs()
    write_audit_files(validated)
    write_result_checksums()
    files = selected_files()
    validate_no_credentials(files)
    identities = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": "gemma-gguf-package-v1",
        "created_at_utc": now_utc(),
        "archive_root": ARCHIVE_ROOT,
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "model_weights_included": False,
        "legacy_outputs_included": False,
        "files": identities,
    }
    atomic_write(
        PROJECT_ROOT / MANIFEST_PATH,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            _zip_write(archive, path, f"{ARCHIVE_ROOT}/{relative}")
        _zip_write(
            archive,
            PROJECT_ROOT / MANIFEST_PATH,
            f"{ARCHIVE_ROOT}/{MANIFEST_PATH.as_posix()}",
        )
    os.replace(temporary, destination)
    checksum_path = destination.with_suffix(destination.suffix + ".sha256")
    atomic_write(checksum_path, f"{sha256_file(destination)}  {destination.name}\n")
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "archive has duplicate members")
        _require(
            f"{ARCHIVE_ROOT}/{MANIFEST_PATH.as_posix()}" in names,
            "archive manifest is missing",
        )
        for item in identities:
            member = f"{ARCHIVE_ROOT}/{item['path']}"
            _require(member in names, f"archive member is missing: {member}")
            _require(
                __import__("hashlib").sha256(archive.read(member)).hexdigest()
                == item["sha256"],
                f"archive member hash mismatch: {member}",
            )
    return {
        "status": "PASS",
        "archive": str(destination),
        "archive_sha256": sha256_file(destination),
        "checksum_file": str(checksum_path),
        "file_count": len(identities) + 1,
        "size_bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_archive(args.destination), sort_keys=True))


if __name__ == "__main__":
    main()

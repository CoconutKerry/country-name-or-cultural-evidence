#!/usr/bin/env python3
"""Validate and package the completed Llama-only GGUF experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_llama_gguf_full import (
    _classification_latex,
    _classification_table_rows,
    _csv_text,
    _flattened_results_csv,
    _metric_latex,
    _metric_table_rows,
    _report_markdown,
    analyze_full_run,
    load_jsonl,
)
from scripts.run_llama_gguf_full import (
    DATASET_SHA256,
    EXPECTED_OUTPUT_CONDITIONS,
    EXPECTED_ROW_COUNT,
    FULL_SCHEMA_VERSION,
    LLAMA_CPP_COMMIT,
    LLAMA_CPP_VERSION,
    MODEL_IDENTIFIER,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PAIR_MANIFEST_SHA256,
    QUANTIZATION,
    LLAMA_CHAT_TEMPLATE_SHA256,
    MODEL_SHARDS,
    canonical_json_hash,
    validate_complete_results,
    validate_runtime,
    validate_smoke_gate,
)
from scripts.run_genuine_llama_gguf_smoke import (
    EXPECTED_DISPLAYED_OPTIONS as SMOKE_DISPLAYED_OPTIONS,
    EXPECTED_UNITS as EXPECTED_SMOKE_UNITS,
    UPSTREAM_CHECKPOINT,
    UPSTREAM_CURRENT_REVISION,
    build_report as build_smoke_report,
)


ARCHIVE_ROOT = "cultural-alignment-audit-main"
OUTPUT_RELATIVE = Path("experiments/llama_gguf_full")
SMOKE_OUTPUT_RELATIVE = Path("experiments/smoke_genuine_llama_gguf")
SMOKE_REPORT_RELATIVE = Path("audit/GENUINE_LLAMA_GGUF_SMOKE_REPORT.md")
PACKAGE_MANIFEST_RELATIVE = Path("audit/LLAMA_GGUF_FULL_PACKAGE_MANIFEST.json")
PACKAGE_PROVENANCE_RELATIVE = Path("audit/LLAMA_GGUF_RUNTIME_PROVENANCE.json")
PACKAGE_PROVENANCE_MD_RELATIVE = Path("audit/LLAMA_GGUF_PROVENANCE.md")
EXCLUSION_FAILURE_REPORT_RELATIVE = Path(
    "audit/LLAMA_GGUF_EXCLUSION_AND_FAILURE_REPORT.md"
)
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    ".cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "runtime",
    "build",
    "dist",
    "node_modules",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".swp",
    ".tmp",
    ".part",
    ".partial",
    ".download",
    ".incomplete",
    ".aria2",
    ".crdownload",
    ".gguf",
    ".safetensors",
    ".ckpt",
    ".pth",
    ".pt",
    ".bin",
    ".zip",
    ".7z",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
}
ALLOWED_EXPERIMENT_DIRECTORIES = {
    OUTPUT_RELATIVE.parts[1],
    SMOKE_OUTPUT_RELATIVE.parts[1],
}
LEGACY_AUDIT_MARKERS = ("QWEN", "MISTRAL", "GEMMA")
MAX_PACKAGE_MEMBER_BYTES = 256 * 1024 * 1024
TOKEN_PATTERN = re.compile(rb"hf_[A-Za-z0-9]{20,}")
REQUIRED_SMOKE_ARTIFACTS = (
    SMOKE_OUTPUT_RELATIVE / "results.jsonl",
    SMOKE_OUTPUT_RELATIVE / "analysis.json",
    SMOKE_OUTPUT_RELATIVE / "run.log",
    SMOKE_REPORT_RELATIVE,
)
GENERATED_PACKAGE_REPORTS = (
    PACKAGE_PROVENANCE_RELATIVE,
    PACKAGE_PROVENANCE_MD_RELATIVE,
    EXCLUSION_FAILURE_REPORT_RELATIVE,
)
REQUIRED_OUTPUTS = (
    "results.jsonl",
    "completion.json",
    "progress.json",
    "run_signature.json",
    "runtime.json",
    "environment.json",
    "run.log",
    "analysis.json",
    "results.csv",
    "directed_unit_metrics.csv",
    "target_unit_metrics.csv",
    "bootstrap_summary.csv",
    "conflict_classifications.csv",
    "metric_tables.json",
    "tables/model_metrics.tex",
    "tables/conflict_classification.tex",
    "EXPERIMENT_REPORT.md",
)
ANALYSIS_OUTPUT_FILENAMES = (
    "analysis.json",
    "results.csv",
    "directed_unit_metrics.csv",
    "target_unit_metrics.csv",
    "bootstrap_summary.csv",
    "conflict_classifications.csv",
    "metric_tables.json",
    "tables/model_metrics.tex",
    "tables/conflict_classification.tex",
    "EXPERIMENT_REPORT.md",
)
ANALYSIS_COMPARISON_KEYS = (
    "schema_version",
    "validation_status",
    "counts",
    "model",
    "metric_definitions",
    "estimands",
    "metric_summaries",
    "bootstrap",
    "conflict_classifications",
    "per_target_unit",
    "per_directed_unit",
    "interpretation",
)
DIRECTED_CSV_FIELDS = (
    "model_name",
    "question_id",
    "label_country",
    "evidence_country",
    "country_influence",
    "evidence_influence",
    "EO_raw",
    "EO_normalized",
    "conflict_to_label_country_jsd2",
    "conflict_to_evidence_country_jsd2",
    "label_to_evidence_country_jsd2",
    "conflict_classification",
)
TARGET_CSV_FIELDS = (
    "model_name",
    "question_id",
    "label_country",
    "evidence_countries",
    "directed_repetitions",
    "country_influence",
    "evidence_influence",
    "baseline_to_label_country_jsd2",
    "country_label_to_label_country_jsd2",
    "evidence_to_label_country_jsd2",
)
BOOTSTRAP_CSV_FIELDS = (
    "metric",
    "n",
    "estimate",
    "ci_95_lower",
    "ci_95_upper",
    "bootstrap_replicates",
    "question_id_clusters",
)
CONFLICT_CSV_FIELDS = (
    "model_name",
    "question_id",
    "label_country",
    "evidence_country",
    "conflict_to_label_country_jsd2",
    "conflict_to_evidence_country_jsd2",
    "EO_raw",
    "EO_normalized",
    "conflict_classification",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_repository_dependency(relative: str) -> Path:
    candidate = PurePosixPath(relative)
    _require(
        bool(relative) and not candidate.is_absolute() and ".." not in candidate.parts,
        f"Unsafe run-signature dependency path: {relative!r}",
    )
    path = (PROJECT_ROOT / Path(*candidate.parts)).resolve()
    _require(
        path.is_relative_to(PROJECT_ROOT.resolve()),
        f"Run-signature dependency escapes repository: {relative!r}",
    )
    return path


def _safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def expected_model_files() -> list[dict[str, Any]]:
    """Return the single immutable GGUF identity accepted by this package."""

    _require(len(MODEL_SHARDS) == 1, "Llama package must pin exactly one GGUF file")
    item = MODEL_SHARDS[0]
    expected = {
        "filename": item.get("filename"),
        "size_bytes": item.get("size_bytes"),
        "sha256": item.get("sha256"),
    }
    _require(
        expected
        == {
            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "size_bytes": 4_920_739_232,
            "sha256": (
                "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"
            ),
        },
        "compiled package pin is not the verified Llama-3.1 Q4_K_M identity",
    )
    return [expected]


def normalize_model_files(
    value: Any,
    *,
    size_field: str = "size_bytes",
    sha256_field: str = "sha256",
    source: str,
) -> list[dict[str, Any]]:
    """Normalize and validate a recorded GGUF file list without set semantics."""

    _require(isinstance(value, list), f"{source} GGUF file record must be a list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        _require(isinstance(item, Mapping), f"{source} GGUF file {index} is not an object")
        normalized.append(
            {
                "filename": item.get("filename"),
                "size_bytes": item.get(size_field),
                "sha256": item.get(sha256_field),
            }
        )
    _require(
        normalized == expected_model_files(),
        f"{source} GGUF filename, byte size, or SHA-256 does not match the exact pin",
    )
    return normalized


def _is_legacy_output(relative: Path) -> bool:
    if relative.parts[:2] == ("analysis", "output"):
        return True
    if relative.parts and relative.parts[0] == "experiments":
        return (
            len(relative.parts) < 2
            or relative.parts[1] not in ALLOWED_EXPERIMENT_DIRECTORIES
        )
    if relative.parts and relative.parts[0] == "audit":
        upper_name = relative.name.upper()
        return any(marker in upper_name for marker in LEGACY_AUDIT_MARKERS)
    return False


def _is_excluded_file(relative: Path) -> bool:
    return (
        any(part in EXCLUDED_PARTS for part in relative.parts)
        or _is_legacy_output(relative)
        or relative.suffix.lower() in EXCLUDED_SUFFIXES
        or relative == PACKAGE_MANIFEST_RELATIVE
        or relative.name == "experiment.log"
    )


def selected_repository_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if _is_excluded_file(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not packageable: {relative}")
        if not path.is_file():
            continue
        _require(
            path.stat().st_size <= MAX_PACKAGE_MEMBER_BYTES,
            f"Refusing unexpectedly large package member: {relative}",
        )
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix())


def validate_package_selection(files: Sequence[Path]) -> dict[str, Any]:
    """Fail if a selected member violates any weight/runtime/legacy boundary."""

    relatives = [path.relative_to(PROJECT_ROOT) for path in files]
    _require(len(relatives) == len(set(relatives)), "package file selection has duplicates")
    for path, relative in zip(files, relatives):
        _require(not _is_excluded_file(relative), f"excluded file selected: {relative}")
        _require(path.is_file() and not path.is_symlink(), f"unsafe member selected: {relative}")
        _require(
            path.stat().st_size <= MAX_PACKAGE_MEMBER_BYTES,
            f"oversized member selected: {relative}",
        )
    required = set(REQUIRED_SMOKE_ARTIFACTS) | set(GENERATED_PACKAGE_REPORTS)
    missing = sorted(str(path) for path in required - set(relatives))
    _require(not missing, f"required smoke/package reports were not selected: {missing!r}")
    return {
        "selected_files": len(files),
        "required_smoke_artifacts": len(REQUIRED_SMOKE_ARTIFACTS),
        "generated_package_reports": len(GENERATED_PACKAGE_REPORTS),
        "weights_selected": False,
        "runtime_or_build_trees_selected": False,
        "legacy_experiment_outputs_selected": False,
    }


def validate_pinned_provenance(
    signature: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
    progress: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind saved rows and completion records to pinned code/runtime assets."""
    _require(isinstance(signature, Mapping), "run signature must be an object")
    fingerprint = signature.get("fingerprint")
    _require(isinstance(fingerprint, str) and fingerprint, "run signature has no fingerprint")
    signature_core = dict(signature)
    signature_core.pop("fingerprint", None)
    _require(
        canonical_json_hash(signature_core) == fingerprint,
        "run signature fingerprint does not match its canonical contents",
    )
    _require(signature.get("schema_version") == FULL_SCHEMA_VERSION, "wrong run-signature schema")
    _require(signature.get("backend") == "llama.cpp", "wrong run-signature backend")
    _require(
        signature.get("conditions") == list(EXPECTED_OUTPUT_CONDITIONS),
        "wrong run-signature conditions",
    )

    model = signature.get("model")
    _require(isinstance(model, Mapping), "run signature has no model object")
    pinned_model = {
        "identifier": MODEL_IDENTIFIER,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "quantization": QUANTIZATION,
    }
    for field, expected in pinned_model.items():
        _require(model.get(field) == expected, f"run signature has wrong model {field}")
    signature_model_files = normalize_model_files(
        model.get("files"), source="run signature"
    )

    llama = signature.get("llama_cpp")
    _require(isinstance(llama, Mapping), "run signature has no llama.cpp object")
    _require(llama.get("commit") == LLAMA_CPP_COMMIT, "run signature has wrong llama.cpp commit")
    _require(llama.get("version") == LLAMA_CPP_VERSION, "run signature has wrong llama.cpp version")
    _require(
        llama.get("chat_template_sha256") == LLAMA_CHAT_TEMPLATE_SHA256,
        "run signature has wrong embedded chat-template hash",
    )
    _require(
        isinstance(llama.get("helper_sha256"), str) and len(llama["helper_sha256"]) == 64,
        "run signature has no helper SHA-256",
    )

    dependencies = signature.get("code_data_dependency_sha256")
    _require(isinstance(dependencies, Mapping) and dependencies, "run signature has no dependency hashes")
    for relative, expected_digest in dependencies.items():
        _require(
            isinstance(relative, str)
            and isinstance(expected_digest, str)
            and len(expected_digest) == 64,
            "run signature contains an invalid dependency identity",
        )
        dependency_path = _safe_repository_dependency(relative)
        _require(dependency_path.is_file(), f"run dependency is missing: {relative}")
        _require(
            sha256_file(dependency_path) == expected_digest,
            f"run dependency changed after inference began: {relative}",
        )

    _require(isinstance(runtime_record, Mapping), "runtime record must be an object")
    _require(runtime_record.get("run_fingerprint") == fingerprint, "runtime fingerprint mismatch")
    _require(runtime_record.get("llama_cpp_commit") == LLAMA_CPP_COMMIT, "runtime llama.cpp commit mismatch")
    _require(
        runtime_record.get("remote_model_last_modified") == model.get("remote_last_modified"),
        "runtime/model last-modified provenance mismatch",
    )
    _require(
        runtime_record.get("remote_weight_commit") == model.get("weight_upload_commit"),
        "runtime/model weight-commit provenance mismatch",
    )
    runtime = runtime_record.get("runtime")
    _require(isinstance(runtime, Mapping), "runtime record has no runtime object")
    validate_runtime(runtime)
    runtime_files = runtime_record.get("model_files")
    _require(isinstance(runtime_files, list), "runtime record has no model-file list")
    normalized_runtime_files = normalize_model_files(
        runtime_files,
        size_field="actual_size_bytes",
        sha256_field="actual_sha256",
        source="runtime",
    )
    _require(
        normalized_runtime_files == signature_model_files,
        "runtime GGUF file identities do not match run signature",
    )

    for name, record in (("progress", progress), ("completion", completion)):
        _require(isinstance(record, Mapping), f"{name} record must be an object")
        _require(record.get("schema_version") == FULL_SCHEMA_VERSION, f"{name} schema mismatch")
        _require(record.get("run_fingerprint") == fingerprint, f"{name} fingerprint mismatch")
    _require(completion.get("model_identifier") == MODEL_IDENTIFIER, "completion model mismatch")
    _require(completion.get("model_revision") == MODEL_REVISION, "completion revision mismatch")
    _require(completion.get("llama_cpp_commit") == LLAMA_CPP_COMMIT, "completion llama.cpp mismatch")
    _require(progress.get("expected_directed_units") == 200, "progress directed-unit contract mismatch")
    _require(progress.get("expected_rows") == EXPECTED_ROW_COUNT, "progress row contract mismatch")
    _require(progress.get("completed_directed_units") == 200, "progress directed completion mismatch")

    return {
        "run_fingerprint": fingerprint,
        "dependency_files_reverified": len(dependencies),
        "model_shards_reverified_from_records": len(signature_model_files),
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "llama_cpp_version": LLAMA_CPP_VERSION,
        "chat_template_sha256": LLAMA_CHAT_TEMPLATE_SHA256,
        "helper_sha256": llama["helper_sha256"],
    }


def _expected_conflict_rows(directed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model_name": row["model_name"],
            "question_id": row["question_id"],
            "label_country": row["label_country"],
            "evidence_country": row["evidence_country"],
            "conflict_to_label_country_jsd2": row["conflict_to_label_country_jsd2"],
            "conflict_to_evidence_country_jsd2": row["conflict_to_evidence_country_jsd2"],
            "EO_raw": row["EO_raw"],
            "EO_normalized": row["EO_normalized"],
            "conflict_classification": row["conflict_classification"],
        }
        for row in directed
    ]


def validate_derived_artifacts(
    output: Path,
    saved_analysis: Mapping[str, Any],
    expected_analysis: Mapping[str, Any],
    enriched_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Byte-check every derived table against a fresh deterministic analysis."""
    for key in ANALYSIS_COMPARISON_KEYS:
        _require(
            saved_analysis.get(key) == expected_analysis.get(key),
            f"analysis.json is stale or inconsistent at {key}",
        )

    metric_rows = _metric_table_rows(expected_analysis)
    classification_rows = _classification_table_rows(expected_analysis)
    directed = expected_analysis["per_directed_unit"]
    targets = expected_analysis["per_target_unit"]
    conflicts = _expected_conflict_rows(directed)
    expected_metric_tables = {
        "schema_version": "llama_gguf_full_metric_tables_v1",
        "model_metrics": metric_rows,
        "conflict_classification": classification_rows,
    }
    observed_metric_tables = json.loads(
        (output / "metric_tables.json").read_text(encoding="utf-8")
    )
    _require(
        observed_metric_tables == expected_metric_tables,
        "metric_tables.json is stale or inconsistent",
    )

    expected_text = {
        "results.csv": _flattened_results_csv(enriched_rows),
        "directed_unit_metrics.csv": _csv_text(directed, DIRECTED_CSV_FIELDS),
        "target_unit_metrics.csv": _csv_text(targets, TARGET_CSV_FIELDS),
        "bootstrap_summary.csv": _csv_text(metric_rows, BOOTSTRAP_CSV_FIELDS),
        "conflict_classifications.csv": _csv_text(conflicts, CONFLICT_CSV_FIELDS),
        "tables/model_metrics.tex": _metric_latex(metric_rows),
        "tables/conflict_classification.tex": _classification_latex(classification_rows),
        "EXPERIMENT_REPORT.md": _report_markdown(
            expected_analysis, ANALYSIS_OUTPUT_FILENAMES
        ),
    }
    hashes: dict[str, str] = {}
    for relative, expected in expected_text.items():
        path = output / relative
        observed = path.read_text(encoding="utf-8")
        _require(observed == expected, f"derived artifact is stale or inconsistent: {relative}")
        hashes[relative] = sha256_file(path)
    hashes["metric_tables.json"] = sha256_file(output / "metric_tables.json")
    return hashes


def validate_outputs() -> dict:
    output = PROJECT_ROOT / OUTPUT_RELATIVE
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Completed Llama outputs are missing: {missing!r}")

    rows = load_jsonl(output / "results.jsonl")
    manifest_path = PROJECT_ROOT / "data/pairs/country_pairs_v2.json"
    dataset_path = PROJECT_ROOT / "data/processed/dataset_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_record = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    signature = json.loads((output / "run_signature.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))

    if sha256_file(manifest_path) != PAIR_MANIFEST_SHA256:
        raise ValueError("Cleaned pair manifest identity changed")
    if sha256_file(dataset_path) != DATASET_SHA256:
        raise ValueError("Cleaned dataset identity changed")
    provenance_validation = validate_pinned_provenance(
        signature, runtime_record, progress, completion
    )
    validation = validate_complete_results(
        rows,
        manifest,
        runtime=runtime_record["runtime"],
        provenance=signature,
    )
    if validation.get("status") != "PASS" or len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError("Result validation did not pass at exactly 800 rows")
    if completion.get("status") != "PASS" or completion.get("rows") != EXPECTED_ROW_COUNT:
        raise ValueError("Completion record does not certify exactly 800 rows")
    if progress.get("status") != "complete" or progress.get("completed_rows") != EXPECTED_ROW_COUNT:
        raise ValueError("Progress record does not certify completion")
    counts = analysis.get("counts", {})
    if (
        analysis.get("validation_status") != "PASS"
        or counts.get("result_rows") != EXPECTED_ROW_COUNT
        or counts.get("directed_units") != 200
        or counts.get("target_units") != 144
        or counts.get("question_id_clusters") != 44
        or counts.get("conflict_rows_classified") != 200
    ):
        raise ValueError("Analysis cardinality/status is incomplete")
    bootstrap = analysis.get("bootstrap", {}).get("method", {})
    if (
        bootstrap.get("n_replicates") != 10_000
        or bootstrap.get("cluster_key") != "question_id"
        or bootstrap.get("n_clusters") != 44
        or bootstrap.get("independent_condition_row_resampling") is not False
    ):
        raise ValueError("Analysis did not use the required 10,000 question-clustered bootstrap")
    classes = analysis.get("conflict_classifications", {}).get("counts", {})
    if sum(int(value) for value in classes.values()) != 200:
        raise ValueError("Conflict classifications do not cover all 200 directed units")
    results_digest = sha256_file(output / "results.jsonl")
    manifest_digest = sha256_file(manifest_path)
    source_results = analysis.get("source_results")
    source_manifest = analysis.get("source_manifest")
    if not isinstance(source_results, Mapping) or source_results.get("sha256") != results_digest:
        raise ValueError("analysis.json is not bound to the current results.jsonl")
    if not isinstance(source_manifest, Mapping) or source_manifest.get("sha256") != manifest_digest:
        raise ValueError("analysis.json is not bound to the current cleaned manifest")

    # Recompute all metrics and the seeded 10,000-replicate bootstrap rather
    # than trusting derived files that merely have plausible row counts.
    expected_analysis, enriched_rows = analyze_full_run(
        rows,
        manifest,
        bootstrap_replicates=10_000,
        bootstrap_seed=42,
        classification_tolerance=1e-12,
    )
    derived_hashes = validate_derived_artifacts(
        output, analysis, expected_analysis, enriched_rows
    )
    return {
        "results": validation,
        "bootstrap_replicates": 10_000,
        "question_id_clusters": 44,
        "conflict_classification_counts": classes,
        "run_fingerprint": signature["fingerprint"],
        "pinned_provenance": provenance_validation,
        "results_sha256": results_digest,
        "analysis_sha256": sha256_file(output / "analysis.json"),
        "derived_artifact_sha256": derived_hashes,
    }


def validate_no_credentials(files: list[Path]) -> None:
    for path in files:
        overlap = b""
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                candidate = overlap + chunk
                if TOKEN_PATTERN.search(candidate):
                    try:
                        display = path.relative_to(PROJECT_ROOT)
                    except ValueError:
                        display = path
                    raise ValueError(f"Possible Hugging Face credential in {display}")
                overlap = candidate[-128:]


def _json_object(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _valid_probability_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    try:
        probabilities = [float(item) for item in value.values()]
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(item) and item >= 0.0 for item in probabilities)
        and math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8)
    )


def validate_genuine_smoke(
    signature: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the exact passing genuine smoke and bind it to the full run."""

    missing = [
        str(relative)
        for relative in REQUIRED_SMOKE_ARTIFACTS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Required genuine-smoke artifacts are missing: {missing!r}")
    analysis_path = PROJECT_ROOT / SMOKE_OUTPUT_RELATIVE / "analysis.json"
    results_path = PROJECT_ROOT / SMOKE_OUTPUT_RELATIVE / "results.jsonl"
    report_path = PROJECT_ROOT / SMOKE_REPORT_RELATIVE
    log_path = PROJECT_ROOT / SMOKE_OUTPUT_RELATIVE / "run.log"
    gate = validate_smoke_gate(analysis_path, results_path)
    analysis = _json_object(analysis_path, "genuine-smoke analysis")
    _require(
        report_path.read_text(encoding="utf-8") == build_smoke_report(analysis),
        "genuine-smoke Markdown report is stale or inconsistent",
    )
    _require(
        "All genuine-model smoke assertions: PASS" in log_path.read_text(encoding="utf-8"),
        "genuine-smoke log does not contain the terminal PASS record",
    )

    metadata = analysis.get("metadata")
    _require(isinstance(metadata, Mapping), "genuine-smoke metadata is missing")
    model = metadata.get("model")
    llama = metadata.get("llama_cpp")
    _require(isinstance(model, Mapping), "genuine-smoke model metadata is missing")
    _require(isinstance(llama, Mapping), "genuine-smoke llama.cpp metadata is missing")
    normalize_model_files(
        model.get("files"),
        size_field="actual_size_bytes",
        sha256_field="actual_sha256",
        source="genuine smoke",
    )
    _require(model.get("selected_entrypoint_file") == expected_model_files()[0]["filename"],
             "genuine smoke selected the wrong GGUF entrypoint")
    _require(model.get("total_size_bytes") == expected_model_files()[0]["size_bytes"],
             "genuine smoke recorded the wrong GGUF byte size")

    signature_llama = signature.get("llama_cpp")
    _require(isinstance(signature_llama, Mapping), "full signature llama.cpp metadata is missing")
    helper_sha256 = llama.get("helper_sha256")
    _require(
        isinstance(helper_sha256, str)
        and len(helper_sha256) == 64
        and helper_sha256 == signature_llama.get("helper_sha256"),
        "genuine-smoke helper identity does not match the full run",
    )
    helper_ldd = llama.get("helper_ldd")
    _require(isinstance(helper_ldd, str) and helper_ldd, "genuine-smoke helper linkage is missing")
    _require(
        hashlib.sha256(helper_ldd.encode("utf-8")).hexdigest()
        == llama.get("helper_linkage_sha256"),
        "genuine-smoke helper-linkage hash is invalid",
    )
    scoring_hashes = llama.get("accepted_scoring_source_sha256")
    _require(isinstance(scoring_hashes, Mapping) and scoring_hashes,
             "genuine-smoke accepted scoring-source hashes are missing")
    for relative, digest in scoring_hashes.items():
        path = _safe_repository_dependency(str(relative))
        _require(isinstance(digest, str) and sha256_file(path) == digest,
                 f"genuine-smoke scoring source changed: {relative}")

    scalar = llama.get("gguf_scalar_metadata")
    _require(isinstance(scalar, Mapping), "genuine-smoke GGUF scalar metadata is missing")
    try:
        context_length = int(str(scalar.get("llama.context_length")))
        file_type = int(str(scalar.get("general.file_type")))
        runtime_file_type = int(str(llama.get("gguf_file_type")))
    except (TypeError, ValueError) as exc:
        raise ValueError("genuine-smoke GGUF numeric metadata is invalid") from exc
    template = scalar.get("tokenizer.chat_template")
    _require(scalar.get("general.architecture") == "llama", "wrong GGUF architecture")
    _require(context_length == 131_072, "wrong GGUF context length")
    _require(file_type == runtime_file_type == 15, "wrong GGUF Q4_K_M file type")
    _require(
        isinstance(template, str)
        and hashlib.sha256(template.encode("utf-8")).hexdigest()
        == LLAMA_CHAT_TEMPLATE_SHA256,
        "wrong GGUF embedded chat template",
    )
    assertions = analysis.get("assertions")
    for name in (
        "pinned_helper_linkage",
        "accepted_scoring_source_hashes",
        "gguf_runtime_identity",
    ):
        _require(isinstance(assertions, Mapping) and assertions.get(name) is True,
                 f"genuine-smoke assertion did not pass: {name}")

    rows = load_jsonl(results_path)
    for row in rows:
        directed = (row.get("question_id"), row.get("label_country"), row.get("evidence_country"))
        _require(directed in EXPECTED_SMOKE_UNITS, "unexpected genuine-smoke directed unit")
        _require(row.get("displayed_options") == SMOKE_DISPLAYED_OPTIONS[directed],
                 "genuine-smoke option permutation drifted")
        _require(_valid_probability_mapping(row.get("normalized_prediction")),
                 "genuine-smoke semantic probabilities are invalid")
        _require(_valid_probability_mapping(row.get("normalized_label_probabilities")),
                 "genuine-smoke label probabilities are invalid")
        evidence_presented = row.get("condition") in {"evidence", "conflict"}
        _require(row.get("evidence_presented") is evidence_presented,
                 "genuine-smoke evidence flag is inconsistent")
        if not evidence_presented:
            _require(
                row.get("presented_evidence_distribution") is None
                and row.get("source_evidence_distribution") is None,
                "genuine-smoke no-evidence row stores evidence",
            )

    for record_name, record in (("runtime", runtime_record), ("completion", completion)):
        _require(record.get("genuine_smoke_gate") == gate,
                 f"full-run {record_name} record is not bound to the validated smoke")
    gate = {
        **gate,
        "report_sha256": sha256_file(report_path),
        "run_log_sha256": sha256_file(log_path),
        "helper_sha256": helper_sha256,
        "helper_linkage_sha256": llama["helper_linkage_sha256"],
        "gguf_file_type": 15,
        "gguf_architecture": "llama",
        "gguf_context_length": 131_072,
    }
    return gate, analysis


def _artifact_identity(relative: Path) -> dict[str, Any]:
    path = PROJECT_ROOT / relative
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _failure_artifacts() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in (PROJECT_ROOT / SMOKE_OUTPUT_RELATIVE, PROJECT_ROOT / OUTPUT_RELATIVE):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and ("failure" in path.name.lower() or "error" in path.name.lower()):
                relative = path.relative_to(PROJECT_ROOT)
                records.append(
                    {
                        **_artifact_identity(relative),
                        "included_in_final_zip": not _is_excluded_file(relative),
                    }
                )
    return records


def write_deterministic_package_reports(
    signature: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
    completion: Mapping[str, Any],
    smoke_analysis: Mapping[str, Any],
    smoke_validation: Mapping[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    """Generate repository-owned provenance and exclusion/failure reports."""

    smoke_llama = smoke_analysis["metadata"]["llama_cpp"]
    runtime = runtime_record["runtime"]
    provenance = {
        "schema_version": "llama_gguf_package_provenance_v1",
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "upstream_checkpoint": UPSTREAM_CHECKPOINT,
            "upstream_current_revision": UPSTREAM_CURRENT_REVISION,
            "quantization": QUANTIZATION,
            "files": expected_model_files(),
        },
        "llama_cpp": {
            "commit": LLAMA_CPP_COMMIT,
            "version": LLAMA_CPP_VERSION,
            "helper_sha256": signature["llama_cpp"]["helper_sha256"],
            "helper_linkage_sha256": smoke_llama["helper_linkage_sha256"],
            "chat_template_sha256": LLAMA_CHAT_TEMPLATE_SHA256,
            "gguf_architecture": "llama",
            "gguf_context_length": 131_072,
            "gguf_file_type": 15,
        },
        "execution": {
            "backend": runtime.get("backend"),
            "synthetic": runtime.get("synthetic"),
            "gpu_layers": runtime.get("gpu_layers"),
            "candidate_scoring": runtime.get("candidate_scoring"),
            "generated_answer_parsing": runtime.get("generated_answer_parsing"),
            "run_fingerprint": signature["fingerprint"],
            "completion_status": completion.get("status"),
            "completion_rows": completion.get("rows"),
        },
        "genuine_smoke": dict(smoke_validation),
        "source_artifacts": [
            _artifact_identity(OUTPUT_RELATIVE / name)
            for name in ("run_signature.json", "runtime.json", "completion.json", "analysis.json")
        ]
        + [_artifact_identity(relative) for relative in REQUIRED_SMOKE_ARTIFACTS],
    }
    json_path = PROJECT_ROOT / PACKAGE_PROVENANCE_RELATIVE
    _atomic_write_text(
        json_path,
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    model_file = expected_model_files()[0]
    markdown = "\n".join(
        [
            "# Llama GGUF Runtime Provenance",
            "",
            f"- Model: `{MODEL_IDENTIFIER}`",
            f"- Repository revision: `{MODEL_REVISION}`",
            f"- GGUF: `{model_file['filename']}` ({model_file['size_bytes']} bytes)",
            f"- GGUF SHA-256: `{model_file['sha256']}`",
            f"- llama.cpp commit: `{LLAMA_CPP_COMMIT}`",
            f"- Helper SHA-256: `{signature['llama_cpp']['helper_sha256']}`",
            f"- Embedded chat-template SHA-256: `{LLAMA_CHAT_TEMPLATE_SHA256}`",
            f"- Full-run fingerprint: `{signature['fingerprint']}`",
            "- Genuine smoke: PASS, exactly 8 rows",
            "- Full run: PASS, exactly 800 rows",
            "- Synthetic, mock, dummy, and fallback backends used: no",
            "",
            "This file is generated deterministically from validated in-repository run records.",
            "The upstream current revision is recorded as identity context; the quantizer did not",
            "publish the exact upstream conversion-source revision.",
            "",
        ]
    )
    markdown_path = PROJECT_ROOT / PACKAGE_PROVENANCE_MD_RELATIVE
    _atomic_write_text(markdown_path, markdown)

    failures = _failure_artifacts()
    failure_lines = (
        ["- None detected in the genuine-smoke or completed full-run output directories."]
        if not failures
        else [
            f"- `{item['path']}` — SHA-256 `{item['sha256']}`; included: "
            f"{str(item['included_in_final_zip']).lower()}"
            for item in failures
        ]
    )
    exclusion_text = "\n".join(
        [
            "# Llama GGUF Exclusion and Failure Report",
            "",
            "## Packaging exclusions",
            "",
            "- Model weights and weight formats, including GGUF, safetensors, checkpoints, and binaries.",
            "- `runtime/`, build trees, virtual environments, caches, and node modules.",
            "- Partial/download fragments and nested archives.",
            "- Legacy experiment outputs and non-Llama model audit outputs.",
            "- llama.cpp source, shared libraries, executables, and downloaded model assets.",
            "",
            "Exact excluded runtime identities remain recorded in the provenance report.",
            "",
            "## Failure accounting",
            "",
            "Current genuine-smoke gate: PASS (8/8 rows).",
            "Current full-run completion gate: PASS (800/800 rows).",
            "Detected failure/error artifacts:",
            *failure_lines,
            "",
        ]
    )
    exclusion_path = PROJECT_ROOT / EXCLUSION_FAILURE_REPORT_RELATIVE
    _atomic_write_text(exclusion_path, exclusion_text)
    paths = [json_path, markdown_path, exclusion_path]
    return paths, {
        "schema_version": provenance["schema_version"],
        "files": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths
        },
        "failure_artifacts_detected": len(failures),
        "externally_supplied_provenance_required": False,
    }


def write_manifest(files: list[Path], validation: dict) -> tuple[Path, dict[str, Any]]:
    entries = [
        {
            "archive_path": f"{ARCHIVE_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "package_scope": "completed_llama_gguf_full_200_directed_units",
        "validation": validation,
        "repository_files": len(files),
        "generated_provenance_files": len(GENERATED_PACKAGE_REPORTS),
        "model_weights_included": False,
        "llama_cpp_source_or_binaries_included": False,
        "exclusion_note": (
            "Pinned model and llama.cpp identities, sizes, hashes, build flags, and "
            "environment are recorded, but multi-gigabyte runtime assets are excluded."
        ),
        "files": entries,
        "manifest_member_policy": (
            "Every archive member except this manifest is listed above and is "
            "verified byte-for-byte after ZIP creation. The manifest cannot "
            "self-hash; its embedded bytes are separately compared with this file."
        ),
    }
    path = PROJECT_ROOT / PACKAGE_MANIFEST_RELATIVE
    _atomic_write_text(
        path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path, manifest


def _write_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    if not _safe_archive_name(name):
        raise ValueError(f"Unsafe archive member name: {name!r}")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    executable = bool(source.stat().st_mode & stat.S_IXUSR) or source.suffix == ".sh"
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with source.open("rb") as source_handle, archive.open(
        info, "w", force_zip64=True
    ) as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)


def _archive_member_identity(archive: zipfile.ZipFile, name: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(name, "r") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def verify_archive(
    archive_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> int:
    """Verify safe unique coverage and every manifest-listed member's bytes."""
    manifest_name = f"{ARCHIVE_ROOT}/{manifest_path.relative_to(PROJECT_ROOT).as_posix()}"
    entries = manifest.get("files")
    _require(isinstance(entries, list), "package manifest has no file entries")
    entry_by_name: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "package manifest entry is not an object")
        name = entry.get("archive_path")
        _require(isinstance(name, str) and _safe_archive_name(name), "unsafe manifest member")
        _require(name not in entry_by_name, f"duplicate manifest entry: {name!r}")
        entry_by_name[name] = entry

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "archive has duplicate members")
        _require(all(_safe_archive_name(name) for name in names), "archive has unsafe members")
        expected_names = set(entry_by_name) | {manifest_name}
        _require(
            set(names) == expected_names,
            "archive members do not exactly match package manifest coverage",
        )
        bad = archive.testzip()
        _require(bad is None, f"archive CRC validation failed at {bad}")
        for name, entry in entry_by_name.items():
            observed_size, observed_sha256 = _archive_member_identity(archive, name)
            _require(
                observed_size == entry.get("size_bytes"),
                f"archive member size mismatch: {name}",
            )
            _require(
                observed_sha256 == entry.get("sha256"),
                f"archive member SHA-256 mismatch: {name}",
            )
        embedded_manifest = archive.read(manifest_name)
        _require(
            embedded_manifest == manifest_path.read_bytes(),
            "embedded package manifest bytes differ from source manifest",
        )
    return len(expected_names)


def build_archive(destination: Path, runtime_directory: Path | None = None) -> dict:
    destination = destination.resolve()
    project_root = PROJECT_ROOT.resolve()
    if destination == project_root or destination.is_relative_to(project_root):
        raise ValueError("Package destination must be outside the project root")

    # runtime_directory is accepted only for command-line compatibility.  No
    # externally supplied provenance is trusted or required.
    del runtime_directory
    validation = validate_outputs()
    output = PROJECT_ROOT / OUTPUT_RELATIVE
    signature = _json_object(output / "run_signature.json", "run signature")
    runtime_record = _json_object(output / "runtime.json", "runtime record")
    completion = _json_object(output / "completion.json", "completion record")
    smoke_validation, smoke_analysis = validate_genuine_smoke(
        signature, runtime_record, completion
    )
    validation["genuine_smoke"] = smoke_validation
    _, generated_validation = write_deterministic_package_reports(
        signature, runtime_record, completion, smoke_analysis, smoke_validation
    )
    validation["generated_provenance"] = generated_validation
    files = selected_repository_files()
    validation["package_selection"] = validate_package_selection(files)
    validate_no_credentials(files)
    manifest_path, manifest = write_manifest(files, validation)
    package_files = files + [manifest_path]

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in package_files:
                name = f"{ARCHIVE_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}"
                _write_member(archive, path, name)
        members = verify_archive(temporary, manifest_path, manifest)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = sha256_file(destination)
    checksum_path = destination.with_suffix(destination.suffix + ".sha256")
    _atomic_write_text(checksum_path, f"{digest}  {destination.name}\n")
    return {
        "zip": str(destination),
        "sha256": digest,
        "checksum_file": str(checksum_path),
        "members": members,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="Deprecated compatibility option; provenance is generated from validated run records",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_archive(
                args.destination,
                args.runtime_dir.resolve() if args.runtime_dir is not None else None,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

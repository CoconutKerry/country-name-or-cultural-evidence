#!/usr/bin/env python3
"""Validate and package the completed Qwen-only GGUF experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_qwen_gguf_full import (
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
from scripts.run_qwen_gguf_full import (
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
    QWEN_CHAT_TEMPLATE_SHA256,
    canonical_json_hash,
    validate_complete_results,
    validate_runtime,
)


ARCHIVE_ROOT = "cultural-alignment-audit-main"
OUTPUT_RELATIVE = Path("experiments/qwen_gguf_full")
PACKAGE_MANIFEST_RELATIVE = Path("audit/QWEN_GGUF_FULL_PACKAGE_MANIFEST.json")
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp", ".tmp"}
TOKEN_PATTERN = re.compile(rb"hf_[A-Za-z0-9]{20,}")
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


def selected_repository_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not packageable: {relative}")
        if not path.is_file() or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if relative == PACKAGE_MANIFEST_RELATIVE or path.name == "experiment.log":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix())


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
    signature_model_files = model.get("files")
    _require(
        isinstance(signature_model_files, list) and len(signature_model_files) == 2,
        "run signature must identify both GGUF shards",
    )

    llama = signature.get("llama_cpp")
    _require(isinstance(llama, Mapping), "run signature has no llama.cpp object")
    _require(llama.get("commit") == LLAMA_CPP_COMMIT, "run signature has wrong llama.cpp commit")
    _require(llama.get("version") == LLAMA_CPP_VERSION, "run signature has wrong llama.cpp version")
    _require(
        llama.get("chat_template_sha256") == QWEN_CHAT_TEMPLATE_SHA256,
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
    normalized_runtime_files = [
        {
            "filename": item.get("filename"),
            "size_bytes": item.get("actual_size_bytes"),
            "sha256": item.get("actual_sha256"),
        }
        for item in runtime_files
        if isinstance(item, Mapping)
    ]
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
        "chat_template_sha256": QWEN_CHAT_TEMPLATE_SHA256,
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
        "schema_version": "qwen_gguf_full_metric_tables_v1",
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
        raise FileNotFoundError(f"Completed Qwen outputs are missing: {missing!r}")

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


def validate_external_runtime_provenance(
    runtime_directory: Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Cross-check the preflight asset record against the actual run signature."""
    runtime_files = [
        runtime_directory / "runtime_provenance.json",
        runtime_directory / "PROVENANCE.md",
    ]
    missing = [str(path) for path in runtime_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Runtime provenance files are missing: {missing!r}")

    signature = json.loads(
        (PROJECT_ROOT / OUTPUT_RELATIVE / "run_signature.json").read_text(encoding="utf-8")
    )
    runtime_record = json.loads(
        (PROJECT_ROOT / OUTPUT_RELATIVE / "runtime.json").read_text(encoding="utf-8")
    )
    external = json.loads(runtime_files[0].read_text(encoding="utf-8"))
    markdown = runtime_files[1].read_text(encoding="utf-8")

    llama = external.get("llama_cpp")
    helper = external.get("helper")
    model = external.get("model")
    _require(isinstance(llama, Mapping), "external provenance has no llama.cpp object")
    _require(isinstance(helper, Mapping), "external provenance has no helper object")
    _require(isinstance(model, Mapping), "external provenance has no model object")
    _require(llama.get("commit") == LLAMA_CPP_COMMIT, "external llama.cpp commit mismatch")
    _require(
        llama.get("version_reported_by_cmake") == LLAMA_CPP_VERSION,
        "external llama.cpp version mismatch",
    )
    signature_llama = signature["llama_cpp"]
    _require(
        helper.get("sha256") == signature_llama.get("helper_sha256"),
        "external helper hash does not match run signature",
    )
    _require(model.get("repository") == MODEL_REPOSITORY, "external model repository mismatch")
    _require(model.get("requested_revision") == MODEL_REVISION, "external requested revision mismatch")
    _require(model.get("resolved_revision") == MODEL_REVISION, "external resolved revision mismatch")
    _require(model.get("quantization") == QUANTIZATION, "external quantization mismatch")
    metadata = model.get("gguf_metadata")
    _require(isinstance(metadata, Mapping), "external GGUF metadata is missing")
    _require(
        metadata.get("chat_template_sha256") == QWEN_CHAT_TEMPLATE_SHA256,
        "external chat-template hash mismatch",
    )

    signature_files = signature["model"]["files"]
    external_files = model.get("files")
    _require(isinstance(external_files, list), "external GGUF file list is missing")
    normalized_external = [
        {
            "filename": item.get("filename"),
            "size_bytes": item.get("size_bytes"),
            "sha256": item.get("sha256"),
        }
        for item in external_files
        if isinstance(item, Mapping)
    ]
    _require(
        normalized_external == signature_files,
        "external GGUF identities do not match run signature",
    )
    _require(
        runtime_record.get("llama_cpp_commit") == llama.get("commit"),
        "actual runtime and external llama.cpp records disagree",
    )
    for required_value in (
        LLAMA_CPP_COMMIT,
        MODEL_REVISION,
        signature_llama["helper_sha256"],
        *(item["sha256"] for item in signature_files),
    ):
        _require(required_value in markdown, "PROVENANCE.md omits a pinned asset identity")
    return runtime_files, {
        "json_sha256": sha256_file(runtime_files[0]),
        "markdown_sha256": sha256_file(runtime_files[1]),
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "helper_sha256": signature_llama["helper_sha256"],
        "model_shards": len(signature_files),
    }


def write_manifest(
    files: list[Path], validation: dict, runtime_files: list[Path]
) -> tuple[Path, dict[str, Any]]:
    entries = [
        {
            "archive_path": f"{ARCHIVE_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    entries.extend(
        {
            "archive_path": f"runtime-provenance/{path.name}",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in runtime_files
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_scope": "completed_qwen_gguf_full_200_directed_units",
        "validation": validation,
        "repository_files": len(files),
        "runtime_provenance_files": len(runtime_files),
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
    archive.writestr(
        info,
        source.read_bytes(),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


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


def build_archive(destination: Path, runtime_directory: Path) -> dict:
    destination = destination.resolve()
    project_root = PROJECT_ROOT.resolve()
    if destination == project_root or destination.is_relative_to(project_root):
        raise ValueError("Package destination must be outside the project root")

    validation = validate_outputs()
    runtime_files, runtime_provenance_validation = validate_external_runtime_provenance(
        runtime_directory
    )
    validation["external_runtime_provenance"] = runtime_provenance_validation
    files = selected_repository_files()
    validate_no_credentials(files + runtime_files)
    manifest_path, manifest = write_manifest(files, validation, runtime_files)
    package_files = files + [manifest_path]

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for path in package_files:
                name = f"{ARCHIVE_ROOT}/{path.relative_to(PROJECT_ROOT).as_posix()}"
                _write_member(archive, path, name)
            for path in runtime_files:
                _write_member(archive, path, f"runtime-provenance/{path.name}")
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
    parser.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_archive(args.destination, args.runtime_dir.resolve()),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

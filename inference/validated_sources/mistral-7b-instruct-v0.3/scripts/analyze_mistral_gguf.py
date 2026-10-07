#!/usr/bin/env python3
"""Validate and analyze the 800-row Mistral GGUF full experiment.

This module is deliberately inference-free.  It validates the completed
JSONL checkpoint against the cleaned 200-unit manifest, recomputes every
estimand from the saved semantic-option distributions, and writes the
machine- and paper-facing analysis artifacts.

Country Influence (CI) and Evidence Influence (EI) are target-level
estimands.  Repeated directed pairs for the same ``(question_id,
label_country)`` must contain identical non-conflict predictions and are
deduplicated to 144 observations.  Evidence Override is retained for all 200
directed conflict units.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustered_bootstrap import clustered_bootstrap
from src.main import (
    prepare_evidence_distribution,
    reorder_distribution,
    shuffle_options,
    stable_unit_seed,
)
from src.metrics import Metrics
from src.gguf_runner import (
    MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
    MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION,
    build_mistral_template_equivalence_diagnostic,
    serialize_mistral_v03_chat,
)


DEFAULT_RESULTS = PROJECT_ROOT / "experiments/mistral_gguf_full/results.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/pairs/country_pairs_v2.json"
DEFAULT_RUN_SIGNATURE_NAME = "run_signature.json"
DEFAULT_RUNTIME_NAME = "runtime.json"

CONDITIONS = ("baseline", "country_label", "evidence", "conflict")
CONDITION_ORDER = {condition: index for index, condition in enumerate(CONDITIONS)}
METRICS = ("country_influence", "evidence_influence", "EO_raw", "EO_normalized")
EXPECTED_MODEL_NAME = "Mistral-7B-Instruct-v0.3"
EXPECTED_MODEL_IDENTIFIER = (
    "bartowski/Mistral-7B-Instruct-v0.3-GGUF@"
    "61fd4167fff3ab01ee1cfe0da183fa27a944db48:"
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
)
EXPECTED_MODEL_REPOSITORY = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
EXPECTED_MODEL_REVISION = "61fd4167fff3ab01ee1cfe0da183fa27a944db48"
EXPECTED_UPSTREAM_CHECKPOINT = "mistralai/Mistral-7B-Instruct-v0.3"
EXPECTED_GGUF_FILENAME = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
EXPECTED_GGUF_SIZE_BYTES = 4_372_812_000
EXPECTED_GGUF_SHA256 = "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6"
EXPECTED_LLAMA_CPP_COMMIT = "62acc89c26c66076cb72e049f307fbe93b8b9750"
EXPECTED_BACKEND = "llama.cpp"
EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_SCHEMA_VERSION = "mistral-gguf-full-v2"
EXPECTED_MANIFEST_SHA256 = "429e07b5f2019be15fd9306e4722ff3878fb575c0eb03db7f1c0d1aacbad5d8b"
DERIVED_ABS_TOLERANCE = 1e-12
CLASSIFICATION_TOLERANCE = 1e-12
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
EXPECTED_PERMUTATION_SEED = 42
EXPECTED_PERMUTATION_KEY = "question_id::label_country"
FORBIDDEN_PROVENANCE_MARKERS = ("qwen", "mistral2.5", "mistral-2.5")
FORBIDDEN_NON_V03_TEMPLATE_SHA256 = (
    "d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4"
)


@dataclass(frozen=True)
class AnalysisExpectations:
    """Cardinality contract for the production run or a small test fixture."""

    result_rows: int = 800
    directed_units: int = 200
    target_units: int = 144
    question_ids: int = 44


PRODUCTION_EXPECTATIONS = AnalysisExpectations()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _required_text(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{context}.{field} must be a non-empty string",
    )
    return value


def _finite_number(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{context} must be a finite number",
    )
    return float(value)


def _close(left: float, right: float, tolerance: float = DERIVED_ABS_TOLERANCE) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _distribution(
    value: Any,
    options: Sequence[str],
    context: str,
    *,
    require_unit_sum: bool = True,
) -> dict[str, float]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    _require(set(value) == set(options), f"{context} option keys do not match")
    converted = {
        option: _finite_number(value[option], f"{context}[{option!r}]")
        for option in options
    }
    _require(
        all(probability >= 0.0 for probability in converted.values()),
        f"{context} contains a negative probability",
    )
    total = math.fsum(converted.values())
    _require(total > 0.0, f"{context} has zero total probability")
    if require_unit_sum:
        _require(
            _close(total, 1.0, 1e-8),
            f"{context} probabilities sum to {total}, not one",
        )
    # Preserve the recorded values.  Metrics performs its own defensive
    # normalization, while the validation above prevents count-like inputs.
    return converted


def _distributions_close(
    left: Mapping[str, float],
    right: Mapping[str, float],
    tolerance: float = DERIVED_ABS_TOLERANCE,
) -> bool:
    return set(left) == set(right) and all(
        _close(left[key], right[key], tolerance) for key in left
    )


def _json_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _sha256_text(value: str, context: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{context} must be a lowercase SHA-256 digest",
    )
    return value


def load_json_object(path: Path, context: str) -> dict[str, Any]:
    """Load a required JSON object used to bind results to the genuine run."""
    _require(path.is_file(), f"{context} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} is not valid JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def validate_provenance(
    run_signature: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact Mistral GGUF, embedded template, and llama.cpp run.

    The full-run rows intentionally avoid repeating multi-kilobyte GGUF
    metadata.  ``run_signature.json`` and ``runtime.json`` are therefore
    mandatory production inputs, and their shared fingerprint binds every row
    to the checked model file, runtime, code/data identity, and run settings.
    """
    _require(isinstance(run_signature, Mapping), "run signature must be an object")
    _require(isinstance(runtime_record, Mapping), "runtime record must be an object")
    serialized_provenance = json.dumps(
        {"run_signature": run_signature, "runtime_record": runtime_record},
        ensure_ascii=False,
        sort_keys=True,
    ).lower()
    for marker in FORBIDDEN_PROVENANCE_MARKERS:
        _require(
            marker not in serialized_provenance,
            f"provenance contains forbidden model/template marker {marker!r}",
        )

    _require(
        run_signature.get("schema_version") == EXPECTED_SCHEMA_VERSION,
        "run signature has the wrong schema_version",
    )
    fingerprint = _sha256_text(
        run_signature.get("fingerprint"), "run_signature.fingerprint"
    )
    signature_core = dict(run_signature)
    signature_core.pop("fingerprint", None)
    _require(
        _canonical_json_sha256(signature_core) == fingerprint,
        "run signature fingerprint does not match its canonical contents",
    )

    model = run_signature.get("model")
    _require(isinstance(model, Mapping), "run_signature.model must be an object")
    expected_model_fields = {
        "paper_name": EXPECTED_MODEL_NAME,
        "identifier": EXPECTED_MODEL_IDENTIFIER,
        "repository": EXPECTED_MODEL_REPOSITORY,
        "revision": EXPECTED_MODEL_REVISION,
        "upstream_checkpoint": EXPECTED_UPSTREAM_CHECKPOINT,
        "quantization": EXPECTED_QUANTIZATION,
    }
    for field, expected in expected_model_fields.items():
        _require(
            model.get(field) == expected,
            f"run_signature.model.{field} does not match the assigned Mistral model",
        )
    files = model.get("files")
    _require(
        isinstance(files, list) and len(files) == 1 and isinstance(files[0], Mapping),
        "run_signature.model.files must contain exactly the selected GGUF",
    )
    model_file = files[0]
    _require(
        model_file.get("filename") == EXPECTED_GGUF_FILENAME,
        "run signature selected a different GGUF filename",
    )
    _require(
        model_file.get("size_bytes") == EXPECTED_GGUF_SIZE_BYTES,
        "run signature GGUF size does not match the pinned file",
    )
    _require(
        model_file.get("sha256") == EXPECTED_GGUF_SHA256,
        "run signature GGUF SHA-256 does not match the pinned file",
    )

    _require(run_signature.get("backend") == EXPECTED_BACKEND, "wrong run backend")
    _require(
        run_signature.get("scoring")
        == "full_contextual_option_label_sequence_log_probability",
        "run signature records the wrong candidate-scoring method",
    )
    _require(
        run_signature.get("conditions") == list(CONDITIONS),
        "run signature conditions are not the required four-condition order",
    )
    _require(
        run_signature.get("directed_unit_key")
        == ["question_id", "label_country", "evidence_country"],
        "run signature does not preserve the full directed key",
    )
    signed_directed_units = run_signature.get("directed_units")
    _require(
        isinstance(signed_directed_units, list)
        and len(signed_directed_units) == PRODUCTION_EXPECTATIONS.directed_units
        and all(
            isinstance(key, list)
            and len(key) == 3
            and all(isinstance(value, str) and bool(value) for value in key)
            for key in signed_directed_units
        ),
        "run signature does not contain exactly 200 complete directed keys",
    )
    signed_keys = [tuple(key) for key in signed_directed_units]
    _require(len(set(signed_keys)) == len(signed_keys), "run signature directed keys repeat")
    _require(
        all((qid, evidence, label) in set(signed_keys) for qid, label, evidence in signed_keys),
        "run signature directed keys are not fully reciprocal",
    )
    _require(
        run_signature.get("question_clusters") == PRODUCTION_EXPECTATIONS.question_ids,
        "run signature has the wrong question_id cluster count",
    )
    _require(
        run_signature.get("target_units") == PRODUCTION_EXPECTATIONS.target_units,
        "run signature has the wrong target-unit count",
    )
    _require(
        run_signature.get("jensen_shannon_base") == 2,
        "run signature does not pin base-2 Jensen-Shannon divergence",
    )
    _require(
        run_signature.get("seed") == EXPECTED_PERMUTATION_SEED
        and run_signature.get("permutation_key") == EXPECTED_PERMUTATION_KEY,
        "run signature does not pin the shared cross-model option permutation",
    )

    llama = run_signature.get("llama_cpp")
    _require(isinstance(llama, Mapping), "run_signature.llama_cpp must be an object")
    _require(
        llama.get("commit") == EXPECTED_LLAMA_CPP_COMMIT,
        "llama.cpp commit does not match the accepted smoke",
    )
    llama_version = _required_text(llama, "version", "run_signature.llama_cpp")
    helper_sha256 = _sha256_text(
        llama.get("helper_sha256"), "run_signature.llama_cpp.helper_sha256"
    )
    signature_template_sha256 = _sha256_text(
        llama.get("chat_template_sha256"),
        "run_signature.llama_cpp.chat_template_sha256",
    )

    _require(
        runtime_record.get("run_fingerprint") == fingerprint,
        "runtime record is not bound to the run signature",
    )
    _require(
        runtime_record.get("llama_cpp_commit") == EXPECTED_LLAMA_CPP_COMMIT,
        "runtime record has the wrong llama.cpp commit",
    )
    runtime = runtime_record.get("runtime")
    _require(isinstance(runtime, Mapping), "runtime_record.runtime must be an object")
    runtime_expected = {
        "backend": EXPECTED_BACKEND,
        "synthetic": False,
        "model_identifier": EXPECTED_MODEL_IDENTIFIER,
        "repository": EXPECTED_MODEL_REPOSITORY,
        "revision": EXPECTED_MODEL_REVISION,
        "quantization": EXPECTED_QUANTIZATION,
        "chat_template_source": "embedded_gguf_metadata",
        "gpu_layers": 0,
        "candidate_scoring": "full_vocabulary_low_level_logits",
        "generated_answer_parsing": False,
        "tokenization_add_special": True,
        "canonical_tokenization_add_special": False,
        "assistant_label_prefix": "",
        "chat_profile": "mistral-v0.3",
        "template_equivalence_policy": MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
    }
    for field, expected in runtime_expected.items():
        _require(
            runtime.get(field) == expected,
            f"runtime.{field} does not match the genuine Mistral contract",
        )
    _require(
        runtime.get("llama_version") == llama_version,
        "runtime llama.cpp version differs from the run signature",
    )
    _require(
        Path(_required_text(runtime, "model_path", "runtime")).name
        == EXPECTED_GGUF_FILENAME,
        "runtime loaded a different GGUF filename",
    )
    _require(
        runtime.get("loaded_model_size_bytes") == EXPECTED_GGUF_SIZE_BYTES,
        "runtime loaded-model size does not match the pinned GGUF",
    )
    _required_text(runtime, "model_description", "runtime")
    _require(
        isinstance(runtime.get("model_parameter_count"), int)
        and not isinstance(runtime.get("model_parameter_count"), bool)
        and runtime["model_parameter_count"] > 0,
        "runtime.model_parameter_count must be a positive integer",
    )
    _require(
        isinstance(runtime.get("vocabulary_size"), int)
        and not isinstance(runtime.get("vocabulary_size"), bool)
        and runtime["vocabulary_size"] > 0,
        "runtime.vocabulary_size must be a positive integer",
    )
    chat_template = _required_text(runtime, "chat_template", "runtime")
    embedded_template = _required_text(runtime, "embedded_chat_template", "runtime")
    _require(
        chat_template == embedded_template,
        "runtime chat template is not the exact embedded GGUF template",
    )
    observed_template_sha256 = sha256(chat_template.encode("utf-8")).hexdigest()
    _require(
        observed_template_sha256 != FORBIDDEN_NON_V03_TEMPLATE_SHA256,
        "runtime reused a forbidden non-Mistral-v0.3 chat template hash",
    )
    _require(
        runtime.get("chat_template_sha256") == observed_template_sha256,
        "runtime chat-template SHA-256 does not match the recorded template",
    )
    _require(
        observed_template_sha256 == signature_template_sha256,
        "runtime chat template differs from the run signature",
    )
    runtime_files = runtime_record.get("model_files")
    _require(
        isinstance(runtime_files, list)
        and len(runtime_files) == 1
        and isinstance(runtime_files[0], Mapping),
        "runtime record must contain exactly one verified model file",
    )
    runtime_file = runtime_files[0]
    _require(runtime_file.get("filename") == EXPECTED_GGUF_FILENAME, "wrong runtime GGUF")
    _require(
        runtime_file.get("actual_size_bytes", runtime_file.get("size_bytes"))
        == EXPECTED_GGUF_SIZE_BYTES,
        "runtime GGUF size verification failed",
    )
    _require(
        runtime_file.get("actual_sha256", runtime_file.get("sha256"))
        == EXPECTED_GGUF_SHA256,
        "runtime GGUF hash verification failed",
    )
    _require(runtime_file.get("verified") is True, "runtime GGUF was not marked verified")

    return {
        "run_fingerprint": fingerprint,
        "paper_name": EXPECTED_MODEL_NAME,
        "upstream_checkpoint": EXPECTED_UPSTREAM_CHECKPOINT,
        "gguf_repository": EXPECTED_MODEL_REPOSITORY,
        "gguf_revision": EXPECTED_MODEL_REVISION,
        "gguf_filename": EXPECTED_GGUF_FILENAME,
        "gguf_size_bytes": EXPECTED_GGUF_SIZE_BYTES,
        "gguf_sha256": EXPECTED_GGUF_SHA256,
        "quantization": EXPECTED_QUANTIZATION,
        "backend": EXPECTED_BACKEND,
        "synthetic": False,
        "llama_cpp_commit": EXPECTED_LLAMA_CPP_COMMIT,
        "llama_cpp_version": llama_version,
        "helper_sha256": helper_sha256,
        "chat_template_source": "embedded_gguf_metadata",
        "chat_template_sha256": observed_template_sha256,
        "candidate_scoring": "full_vocabulary_low_level_logits",
        "generated_answer_parsing": False,
        "gpu_layers": 0,
        "directed_units": signed_directed_units,
    }


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative provenance path over a Work-local path."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return path.name


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, rejecting blank lines and non-object rows."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"{path}: blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc
            _require(isinstance(value, dict), f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, list), "manifest must be a JSON list")
    _require(all(isinstance(row, dict) for row in value), "manifest entries must be objects")
    return value


def validate_manifest(
    manifest: Sequence[Mapping[str, Any]],
    expectations: AnalysisExpectations,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], int],
]:
    """Validate complete directed keys, options, reciprocity, and counts."""
    _require(
        len(manifest) == expectations.directed_units,
        f"manifest has {len(manifest)} rows; expected {expectations.directed_units}",
    )
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: dict[tuple[str, str, str], int] = {}
    for index, raw in enumerate(manifest):
        context = f"manifest[{index}]"
        question_id = _required_text(raw, "question_id", context)
        label_country = _required_text(raw, "country", context)
        evidence_country = _required_text(raw, "conflict_country", context)
        _require(label_country != evidence_country, f"{context} is a self-pair")
        options = raw.get("options")
        _require(
            isinstance(options, list)
            and len(options) >= 2
            and all(isinstance(option, str) and option for option in options),
            f"{context}.options must contain at least two non-empty strings",
        )
        _require(len(set(options)) == len(options), f"{context}.options contains duplicates")
        _require(
            raw.get("mapping_status") == "exact_shared_response_schema",
            f"{context} is not an exact shared response schema",
        )
        key = (question_id, label_country, evidence_country)
        _require(key not in by_key, f"duplicate manifest directed key {key!r}")
        by_key[key] = dict(raw)
        order[key] = index

    for question_id, label_country, evidence_country in by_key:
        reverse = (question_id, evidence_country, label_country)
        _require(reverse in by_key, f"manifest is missing reciprocal direction {reverse!r}")
        _require(
            by_key[reverse]["options"] == by_key[(question_id, label_country, evidence_country)]["options"],
            f"reciprocal manifest options differ for {question_id!r}, "
            f"{label_country!r}, {evidence_country!r}",
        )

    target_keys = {(qid, label) for qid, label, _ in by_key}
    question_ids = {qid for qid, _, _ in by_key}
    _require(
        len(target_keys) == expectations.target_units,
        f"manifest has {len(target_keys)} target units; expected {expectations.target_units}",
    )
    _require(
        len(question_ids) == expectations.question_ids,
        f"manifest has {len(question_ids)} question IDs; expected {expectations.question_ids}",
    )
    return by_key, order


def _validate_model_and_backend(
    row: Mapping[str, Any],
    context: str,
    *,
    expected_run_fingerprint: str | None = None,
) -> None:
    _require(row.get("schema_version") == EXPECTED_SCHEMA_VERSION, f"{context}: wrong schema_version")
    _require(row.get("model_name") == EXPECTED_MODEL_NAME, f"{context}: wrong model name")
    _require(row.get("model_identifier") == EXPECTED_MODEL_IDENTIFIER, f"{context}: wrong model identifier")
    _require(row.get("model_repository") == EXPECTED_MODEL_REPOSITORY, f"{context}: wrong model repository")
    _require(row.get("model_revision") == EXPECTED_MODEL_REVISION, f"{context}: wrong model revision")
    _require(row.get("backend") == EXPECTED_BACKEND, f"{context}: backend must be llama.cpp")
    _require(row.get("synthetic") is False, f"{context}: synthetic/fallback row is forbidden")
    _require(row.get("generated_answer") is None, f"{context}: generated-answer parsing is forbidden")
    _require(row.get("scoring_method") == "full_contextual_option_label_sequence_log_probability", f"{context}: wrong scoring method")
    _require(row.get("jensen_shannon_base") == 2, f"{context}: Jensen-Shannon base must be 2")
    _require(
        row.get("jensen_shannon_measure") == "divergence_bits",
        f"{context}: Jensen-Shannon measure must be divergence_bits",
    )
    quantization = row.get("quantization")
    _require(isinstance(quantization, Mapping), f"{context}.quantization must be an object")
    _require(quantization.get("type") == EXPECTED_QUANTIZATION, f"{context}: wrong quantization")
    _require(quantization.get("format") == "GGUF", f"{context}: quantization format must be GGUF")
    _require(quantization.get("gpu_layers") == 0, f"{context}: full Mistral run must be CPU-only")
    _require(
        "base2_jensen_shannon_distances" not in row,
        f"{context}: Jensen-Shannon divergences must not be mislabeled distances",
    )
    if expected_run_fingerprint is None:
        return
    _require(
        row.get("run_fingerprint") == expected_run_fingerprint,
        f"{context}: row is not bound to the validated run fingerprint",
    )
    _require(
        row.get("upstream_checkpoint") == EXPECTED_UPSTREAM_CHECKPOINT,
        f"{context}: wrong upstream original checkpoint",
    )
    _require(
        quantization.get("files") == [EXPECTED_GGUF_FILENAME],
        f"{context}: row quantization provenance names a different GGUF",
    )
    _require(
        row.get("serialized_chat_templated_prompt")
        == row.get("scoring_trace", {}).get("serialized_prompt"),
        f"{context}: serialized prompt aliases disagree",
    )
    actual_serialized_prompt = _required_text(
        row, "serialized_chat_templated_prompt", context
    )
    _require(
        row.get("serialized_prompt_sha256")
        == sha256(actual_serialized_prompt.encode("utf-8")).hexdigest(),
        f"{context}: serialized prompt SHA-256 is invalid",
    )
    messages = row.get("structured_messages")
    _require(
        isinstance(messages, list)
        and len(messages) == 1
        and messages[0]
        == {"role": "user", "content": row.get("raw_user_prompt")},
        f"{context}: Mistral-v0.3 must use its embedded user-only instruction template",
    )
    trace = row.get("scoring_trace")
    _require(isinstance(trace, Mapping), f"{context}.scoring_trace must be an object")
    canonical_serialized_prompt = serialize_mistral_v03_chat(messages)
    _require(
        trace.get("canonical_serialized_prompt") == canonical_serialized_prompt,
        f"{context}: canonical Mistral serialization differs",
    )
    _require(trace.get("chat_profile") == "mistral-v0.3", f"{context}: wrong chat profile")
    _require(
        trace.get("serialization_verification")
        == MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION,
        f"{context}: embedded Mistral serialization was not token-equivalence verified",
    )
    _require(
        trace.get("embedded_template_serialization_verified") is True,
        f"{context}: embedded chat template was not verified",
    )
    _require(trace.get("prompt_prefix_verified") is True, f"{context}: prompt prefix was not verified")
    _require(trace.get("generated_answer") is None, f"{context}: generation is forbidden")
    _require(
        trace.get("candidate_prefix") == ""
        and trace.get("tokenization_add_special") is True
        and trace.get("canonical_tokenization_add_special") is False,
        f"{context}: Mistral BOS/tokenization modes are invalid",
    )
    _require(
        trace.get("chat_template_source") == "embedded_gguf_metadata"
        and trace.get("template_equivalence_policy")
        == MISTRAL_TEMPLATE_EQUIVALENCE_POLICY
        and trace.get("template_equivalence_preflight_passed") is True,
        f"{context}: embedded template-equivalence preflight is invalid",
    )
    try:
        expected_diagnostic = build_mistral_template_equivalence_diagnostic(
            actual_serialized_prompt,
            canonical_serialized_prompt,
            trace.get("prompt_token_ids"),
            trace.get("canonical_prompt_token_ids"),
            trace.get("bos_token_id"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: malformed template-equivalence trace") from exc
    _require(
        trace.get("template_equivalence") == expected_diagnostic
        and expected_diagnostic["accepted"] is True
        and trace.get("prompt_starts_with_bos") is True
        and trace.get("prompt_bos_token_count") == 1
        and trace.get("canonical_prompt_starts_with_bos") is True
        and trace.get("canonical_prompt_bos_token_count") == 1,
        f"{context}: canonical/embedded token IDs or BOS boundary differ",
    )


def _validate_option_permutation(
    row: Mapping[str, Any], options: Sequence[str], context: str
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    displayed_options = row.get("displayed_option_order", row.get("displayed_options"))
    labels = row.get("displayed_option_labels")
    mapping = row.get("displayed_label_to_option")
    _require(
        isinstance(displayed_options, list) and set(displayed_options) == set(options)
        and len(displayed_options) == len(options),
        f"{context}.displayed_option_order is not a permutation of original options",
    )
    _require(
        isinstance(labels, list)
        and len(labels) == len(options)
        and len(set(labels)) == len(labels)
        and all(isinstance(label, str) and label for label in labels),
        f"{context}.displayed_option_labels is invalid",
    )
    expected_mapping = dict(zip(labels, displayed_options))
    _require(mapping == expected_mapping, f"{context}.displayed_label_to_option is invalid")
    restored = row.get("restored_original_option_order")
    _require(restored == list(options), f"{context}: original option order was not recovered")
    return tuple(displayed_options), tuple(labels), expected_mapping


def _validate_candidate_scoring(
    row: Mapping[str, Any],
    labels: Sequence[str],
    label_to_option: Mapping[str, str],
    context: str,
) -> None:
    """Validate contextual full-sequence label scoring at the answer boundary."""
    trace = row.get("scoring_trace")
    _require(isinstance(trace, Mapping), f"{context}.scoring_trace must be an object")
    candidates = trace.get("candidates")
    _require(
        isinstance(candidates, Mapping) and set(candidates) == set(labels),
        f"{context}: every candidate label must be independently scored",
    )
    _require(trace.get("label_to_option") == dict(label_to_option), f"{context}: trace label map drifted")
    raw_scores = row.get("raw_candidate_scores")
    _require(
        isinstance(raw_scores, Mapping) and set(raw_scores) == set(labels),
        f"{context}: raw candidate scores are incomplete",
    )
    label_probabilities = _distribution(
        row.get("normalized_label_probabilities"),
        labels,
        f"{context}.normalized_label_probabilities",
    )
    trace_probabilities = _distribution(
        trace.get("normalized_label_probabilities"),
        labels,
        f"{context}.scoring_trace.normalized_label_probabilities",
    )
    _require(
        _distributions_close(label_probabilities, trace_probabilities, 1e-8),
        f"{context}: scoring-trace probabilities disagree with row probabilities",
    )
    prompt_count = trace.get("prompt_token_count")
    _require(
        isinstance(prompt_count, int) and not isinstance(prompt_count, bool) and prompt_count > 0,
        f"{context}: invalid prompt token count",
    )
    _require(
        trace.get("answer_target_position") == prompt_count
        and trace.get("answer_predictive_logit_position") == prompt_count - 1,
        f"{context}: labels were not scored at the exact assistant answer position",
    )
    aliases = row.get("candidate_label_token_ids")
    _require(
        isinstance(aliases, Mapping) and set(aliases) == set(labels),
        f"{context}: candidate token-ID aliases are incomplete",
    )
    seen_token_paths: set[tuple[int, ...]] = set()
    for label in labels:
        candidate = candidates[label]
        _require(isinstance(candidate, Mapping), f"{context}/{label}: invalid candidate trace")
        _require(candidate.get("candidate_label") == label, f"{context}/{label}: candidate label drifted")
        _require(candidate.get("continuation") == label, f"{context}/{label}: full answer text was scored")
        token_ids = candidate.get("token_ids")
        _require(
            isinstance(token_ids, list)
            and bool(token_ids)
            and all(isinstance(token_id, int) and token_id >= 0 for token_id in token_ids),
            f"{context}/{label}: contextual tokenization is invalid",
        )
        token_path = tuple(token_ids)
        _require(token_path not in seen_token_paths, f"{context}/{label}: candidate token path is duplicated")
        seen_token_paths.add(token_path)
        _require(candidate.get("token_count") == len(token_ids), f"{context}/{label}: incomplete sequence token count")
        _require(aliases[label] == token_ids, f"{context}/{label}: token-ID aliases disagree")
        target_positions = list(range(prompt_count, prompt_count + len(token_ids)))
        predictive_positions = [position - 1 for position in target_positions]
        _require(
            candidate.get("target_token_positions") == target_positions
            and candidate.get("predictive_logit_positions") == predictive_positions,
            f"{context}/{label}: sequence was scored at the wrong positions",
        )
        _require(candidate.get("prompt_prefix_verified") is True, f"{context}/{label}: prompt prefix not verified")
        token_log_probabilities = candidate.get("token_log_probabilities")
        raw_token_logits = candidate.get("raw_token_logits")
        _require(
            isinstance(token_log_probabilities, list)
            and len(token_log_probabilities) == len(token_ids)
            and all(math.isfinite(float(value)) for value in token_log_probabilities),
            f"{context}/{label}: token log-probabilities are invalid",
        )
        _require(
            isinstance(raw_token_logits, list)
            and len(raw_token_logits) == len(token_ids)
            and all(math.isfinite(float(value)) for value in raw_token_logits),
            f"{context}/{label}: raw token logits are invalid",
        )
        sequence_score = _finite_number(
            candidate.get("sequence_log_probability"),
            f"{context}/{label}.sequence_log_probability",
        )
        _require(
            _close(math.fsum(float(value) for value in token_log_probabilities), sequence_score, 1e-9),
            f"{context}/{label}: multi-token sequence score is incomplete",
        )
        _require(
            _close(_finite_number(raw_scores[label], f"{context}.raw_candidate_scores[{label!r}]"), sequence_score, 1e-9),
            f"{context}/{label}: raw score does not equal complete sequence score",
        )

    # A shared English first word must never collapse two semantic options:
    # each remains mapped to its own scored candidate label.
    first_words = [option.split(maxsplit=1)[0].casefold() for option in label_to_option.values()]
    if len(set(first_words)) < len(first_words):
        _require(
            len(candidates) == len(label_to_option)
            and len(set(label_to_option.values())) == len(label_to_option),
            f"{context}: same-first-word semantic options were collapsed",
        )


def _validate_evidence_schema(
    row: Mapping[str, Any],
    condition: str,
    options: Sequence[str],
    displayed_options: Sequence[str],
    label_distribution: Mapping[str, float],
    evidence_distribution: Mapping[str, float],
    context: str,
) -> None:
    evidence_presented = condition in {"evidence", "conflict"}
    _require(
        row.get("evidence_presented") is evidence_presented,
        f"{context}.evidence_presented is inconsistent with condition",
    )
    for field in ("presented_evidence_distribution", "source_evidence_distribution"):
        _require(field in row, f"{context}.{field} must be explicitly present")
        if not evidence_presented:
            _require(row[field] is None, f"{context}.{field} must be null without evidence")
    if not evidence_presented:
        return

    expected_source = label_distribution if condition == "evidence" else evidence_distribution
    observed_source = _distribution(
        row["source_evidence_distribution"],
        options,
        f"{context}.source_evidence_distribution",
    )
    _require(
        _distributions_close(observed_source, expected_source),
        f"{context}.source_evidence_distribution does not match the exact human reference",
    )

    # The prompt displays a deterministic largest-remainder rounding at one
    # decimal percentage point.  Allocation tie-breaking follows displayed
    # option order, so reproduce that order before comparing in semantic keys.
    source_displayed = reorder_distribution(expected_source, displayed_options)
    expected_presented = prepare_evidence_distribution(
        source_displayed, displayed_options
    )
    observed_presented = _distribution(
        row["presented_evidence_distribution"],
        options,
        f"{context}.presented_evidence_distribution",
    )
    _require(
        _distributions_close(observed_presented, expected_presented),
        f"{context}.presented_evidence_distribution does not match deterministic prompt rounding",
    )


def _stored_derived_number(
    row: Mapping[str, Any],
    field: str,
    expected: float,
    context: str,
) -> None:
    observed = _finite_number(row.get(field), f"{context}.{field}")
    _require(
        _close(observed, expected),
        f"{context}.{field}={observed!r} does not match recomputed {expected!r}",
    )


def _expected_js_divergences(
    predictions: Mapping[str, Mapping[str, float]],
    label_distribution: Mapping[str, float],
    evidence_distribution: Mapping[str, float],
) -> dict[str, float]:
    baseline_to_label = Metrics.js_divergence(predictions["baseline"], label_distribution)
    country_to_label = Metrics.js_divergence(predictions["country_label"], label_distribution)
    evidence_to_label = Metrics.js_divergence(predictions["evidence"], label_distribution)
    conflict_to_label = Metrics.js_divergence(predictions["conflict"], label_distribution)
    conflict_to_evidence = Metrics.js_divergence(predictions["conflict"], evidence_distribution)
    return {
        "baseline_to_label_country": baseline_to_label,
        "country_label_to_label_country": country_to_label,
        "evidence_to_label_country": evidence_to_label,
        "conflict_to_label_country": conflict_to_label,
        "conflict_to_evidence_country": conflict_to_evidence,
    }


def classify_conflict(
    EO_raw: float, tolerance: float = CLASSIFICATION_TOLERANCE
) -> str:
    """Classify the raw signed conflict proximity with an explicit tie band."""
    _require(math.isfinite(float(EO_raw)), "EO_raw must be finite")
    _require(
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(float(tolerance))
        and float(tolerance) >= 0.0,
        "classification tolerance must be a finite non-negative number",
    )
    if EO_raw > tolerance:
        return "evidence-side"
    if EO_raw < -tolerance:
        return "label-side"
    return "tie"


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    _require(bool(values), "cannot summarize an empty metric")
    return {
        "n": len(values),
        "mean": fmean(values),
        "population_sd": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def analyze_full_run(
    rows: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    expectations: AnalysisExpectations = PRODUCTION_EXPECTATIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = 42,
    classification_tolerance: float = CLASSIFICATION_TOLERANCE,
    provenance_summary: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a validated analysis and rows enriched with recomputed metrics."""
    if provenance_summary is not None:
        _require(
            expectations == PRODUCTION_EXPECTATIONS,
            "genuine production provenance requires the exact 800-row contract",
        )
        _require(
            bootstrap_replicates == BOOTSTRAP_REPLICATES,
            f"genuine production analysis requires {BOOTSTRAP_REPLICATES:,} bootstrap replicates",
        )
        _require(
            classification_tolerance == CLASSIFICATION_TOLERANCE,
            "genuine production analysis requires tie tolerance 1e-12",
        )
    _require(
        len(rows) == expectations.result_rows,
        f"results have {len(rows)} rows; expected exactly {expectations.result_rows}",
    )
    manifest_by_key, manifest_order = validate_manifest(manifest, expectations)
    if provenance_summary is not None:
        expected_signed_order = [
            list(key) for key, _ in sorted(manifest_order.items(), key=lambda item: item[1])
        ]
        _require(
            provenance_summary.get("directed_units") == expected_signed_order,
            "run signature directed-unit list does not exactly match the cleaned manifest",
        )

    groups: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    models: set[str] = set()
    backends: set[str] = set()
    seen_row_keys: set[tuple[str, str, str, str]] = set()
    expected_run_fingerprint = (
        str(provenance_summary["run_fingerprint"])
        if provenance_summary is not None
        else None
    )

    for index, row in enumerate(rows):
        context = f"results[{index}]"
        _require(isinstance(row, Mapping), f"{context} must be an object")
        _validate_model_and_backend(
            row,
            context,
            expected_run_fingerprint=expected_run_fingerprint,
        )
        model_name = _required_text(row, "model_name", context)
        models.add(model_name)
        backends.add(str(row.get("backend")))
        question_id = _required_text(row, "question_id", context)
        label_country = _required_text(row, "label_country", context)
        evidence_country = _required_text(row, "evidence_country", context)
        condition = _required_text(row, "condition", context)
        _require(condition in CONDITIONS, f"{context}: invalid condition {condition!r}")
        directed_key = (question_id, label_country, evidence_country)
        _require(directed_key in manifest_by_key, f"{context}: key is not in cleaned manifest")
        row_key = directed_key + (condition,)
        _require(row_key not in seen_row_keys, f"duplicate result row {row_key!r}")
        seen_row_keys.add(row_key)
        groups.setdefault(directed_key, {})[condition] = row

    _require(len(models) == 1, f"results must contain exactly one model, found {sorted(models)!r}")
    _require(len(backends) == 1, f"results must contain exactly one backend, found {sorted(backends)!r}")
    _require(set(groups) == set(manifest_by_key), "result directed keys do not exactly match manifest")
    for key, condition_rows in groups.items():
        _require(
            set(condition_rows) == set(CONDITIONS),
            f"directed unit {key!r} does not contain exactly all four conditions",
        )

    model_name = next(iter(models))
    enriched_rows: list[dict[str, Any]] = []
    directed_records: list[dict[str, Any]] = []
    target_sources: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for key in sorted(groups, key=manifest_order.__getitem__):
        question_id, label_country, evidence_country = key
        condition_rows = groups[key]
        options = tuple(manifest_by_key[key]["options"])
        expected_permutation_seed = stable_unit_seed(
            EXPECTED_PERMUTATION_SEED, f"{question_id}::{label_country}"
        )
        expected_displayed_options = tuple(
            shuffle_options(options, expected_permutation_seed)
        )
        predictions: dict[str, dict[str, float]] = {}
        label_references: dict[str, dict[str, float]] = {}
        evidence_references: dict[str, dict[str, float]] = {}
        permutations: dict[str, tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]] = {}

        for condition in CONDITIONS:
            row = condition_rows[condition]
            context = f"{key!r}/{condition}"
            _require(row.get("original_options") == list(options), f"{context}: original_options differ from manifest")
            permutations[condition] = _validate_option_permutation(row, options, context)
            if expected_run_fingerprint is not None:
                _require(
                    row.get("permutation_key") == EXPECTED_PERMUTATION_KEY
                    and row.get("permutation_seed") == expected_permutation_seed,
                    f"{context}: fixed cross-model permutation identity drifted",
                )
                _require(
                    permutations[condition][0] == expected_displayed_options,
                    f"{context}: displayed options do not match the fixed shared permutation",
                )
                _validate_candidate_scoring(
                    row,
                    permutations[condition][1],
                    permutations[condition][2],
                    context,
                )
            predictions[condition] = _distribution(
                row.get("normalized_prediction"), options, f"{context}.normalized_prediction"
            )
            label_references[condition] = _distribution(
                row.get("label_country_human_distribution"),
                options,
                f"{context}.label_country_human_distribution",
            )
            evidence_references[condition] = _distribution(
                row.get("evidence_country_human_distribution"),
                options,
                f"{context}.evidence_country_human_distribution",
            )

        reference_permutation = permutations["baseline"]
        _require(
            all(permutation == reference_permutation for permutation in permutations.values()),
            f"{key!r}: option permutation is not fixed across all four conditions",
        )
        label_distribution = label_references["baseline"]
        evidence_distribution = evidence_references["baseline"]
        _require(
            all(_distributions_close(value, label_distribution) for value in label_references.values()),
            f"{key!r}: label-country human reference differs across conditions",
        )
        _require(
            all(_distributions_close(value, evidence_distribution) for value in evidence_references.values()),
            f"{key!r}: evidence-country human reference differs across conditions",
        )
        endpoint_eo_raw = Metrics.evidence_override_raw(
            evidence_distribution, evidence_distribution, label_distribution
        )
        endpoint_label_raw = Metrics.evidence_override_raw(
            label_distribution, evidence_distribution, label_distribution
        )
        _require(
            endpoint_eo_raw > 0.0 and endpoint_label_raw < 0.0,
            f"{key!r}: EO_raw orientation is not positive-evidence/negative-label",
        )
        _require(
            _close(
                Metrics.evidence_override_normalized(
                    evidence_distribution, evidence_distribution, label_distribution
                ),
                1.0,
            )
            and _close(
                Metrics.evidence_override_normalized(
                    label_distribution, evidence_distribution, label_distribution
                ),
                -1.0,
            ),
            f"{key!r}: EO_normalized endpoints are not approximately +1/-1",
        )
        for condition in CONDITIONS:
            _validate_evidence_schema(
                condition_rows[condition],
                condition,
                options,
                permutations[condition][0],
                label_distribution,
                evidence_distribution,
                f"{key!r}/{condition}",
            )

        divergences = _expected_js_divergences(
            predictions, label_distribution, evidence_distribution
        )
        metrics = {
            "country_influence": (
                divergences["baseline_to_label_country"]
                - divergences["country_label_to_label_country"]
            ),
            "evidence_influence": (
                divergences["baseline_to_label_country"]
                - divergences["evidence_to_label_country"]
            ),
            "EO_raw": (
                divergences["conflict_to_label_country"]
                - divergences["conflict_to_evidence_country"]
            ),
            "EO_normalized": Metrics.evidence_override_normalized(
                predictions["conflict"], evidence_distribution, label_distribution
            ),
        }
        classification = classify_conflict(metrics["EO_raw"], classification_tolerance)

        for condition in CONDITIONS:
            row = condition_rows[condition]
            context = f"{key!r}/{condition}"
            for metric, expected in metrics.items():
                _stored_derived_number(row, metric, expected, context)
            observed_class = row.get("conflict_classification")
            if condition == "conflict":
                _require(observed_class == classification, f"{context}: wrong conflict_classification")
            else:
                _require(observed_class is None, f"{context}: classification must be null")

            observed_divergences = row.get("base2_jensen_shannon_divergences")
            _require(
                isinstance(observed_divergences, Mapping),
                f"{context}.base2_jensen_shannon_divergences must be an object",
            )
            row_specific = {
                "prediction_to_label_country": Metrics.js_divergence(
                    predictions[condition], label_distribution
                ),
                "prediction_to_evidence_country": Metrics.js_divergence(
                    predictions[condition], evidence_distribution
                ),
            }
            expected_divergences = {**divergences, **row_specific}
            _require(
                set(observed_divergences) == set(expected_divergences),
                f"{context}: divergence keys do not match the full base-2 schema",
            )
            for divergence_name, expected in expected_divergences.items():
                observed = _finite_number(
                    observed_divergences[divergence_name],
                    f"{context}.base2_jensen_shannon_divergences[{divergence_name!r}]",
                )
                _require(
                    _close(observed, expected),
                    f"{context}: stored {divergence_name} does not match recomputation",
                )

            enriched = dict(row)
            enriched.update(metrics)
            enriched["conflict_classification"] = classification if condition == "conflict" else None
            enriched["base2_jensen_shannon_divergences"] = expected_divergences
            enriched_rows.append(enriched)

        directed = {
            "model_name": model_name,
            "question_id": question_id,
            "label_country": label_country,
            "evidence_country": evidence_country,
            # Compatibility aliases consumed by the shared clustered bootstrap.
            "country": label_country,
            "conflict_country": evidence_country,
            **metrics,
            "conflict_to_label_country_jsd2": divergences["conflict_to_label_country"],
            "conflict_to_evidence_country_jsd2": divergences["conflict_to_evidence_country"],
            "label_to_evidence_country_jsd2": Metrics.js_divergence(
                label_distribution, evidence_distribution
            ),
            "conflict_classification": classification,
        }
        directed_records.append(directed)
        target_sources.setdefault((question_id, label_country), []).append(
            {
                "evidence_country": evidence_country,
                "metrics": metrics,
                "divergences": divergences,
                "predictions": {
                    condition: predictions[condition]
                    for condition in ("baseline", "country_label", "evidence")
                },
                "permutation": reference_permutation,
                "label_distribution": label_distribution,
            }
        )

    target_records: list[dict[str, Any]] = []
    for (question_id, label_country), repetitions in sorted(target_sources.items()):
        reference = repetitions[0]
        for repetition in repetitions[1:]:
            _require(
                repetition["permutation"] == reference["permutation"],
                f"{(question_id, label_country)!r}: repeated target permutation drifted",
            )
            _require(
                _distributions_close(
                    repetition["label_distribution"], reference["label_distribution"]
                ),
                f"{(question_id, label_country)!r}: repeated target reference drifted",
            )
            for condition in ("baseline", "country_label", "evidence"):
                _require(
                    _distributions_close(
                        repetition["predictions"][condition],
                        reference["predictions"][condition],
                    ),
                    f"{(question_id, label_country)!r}: repeated target {condition} prediction drifted",
                )
            for metric in ("country_influence", "evidence_influence"):
                _require(
                    _close(repetition["metrics"][metric], reference["metrics"][metric]),
                    f"{(question_id, label_country)!r}: repeated target {metric} drifted",
                )

        target_records.append(
            {
                "model_name": model_name,
                "question_id": question_id,
                "label_country": label_country,
                "country": label_country,
                "evidence_countries": sorted(
                    repetition["evidence_country"] for repetition in repetitions
                ),
                "directed_repetitions": len(repetitions),
                "country_influence": reference["metrics"]["country_influence"],
                "evidence_influence": reference["metrics"]["evidence_influence"],
                "baseline_to_label_country_jsd2": reference["divergences"]["baseline_to_label_country"],
                "country_label_to_label_country_jsd2": reference["divergences"]["country_label_to_label_country"],
                "evidence_to_label_country_jsd2": reference["divergences"]["evidence_to_label_country"],
            }
        )

    _require(
        len(directed_records) == expectations.directed_units,
        "directed metric cardinality changed during analysis",
    )
    _require(
        len(target_records) == expectations.target_units,
        "target metric cardinality changed during deduplication",
    )
    _require(
        len(enriched_rows) == expectations.result_rows,
        "enriched result cardinality changed during analysis",
    )

    point_analysis = {
        "models": {model_name: {}},
        "per_target_unit": target_records,
        "per_directed_unit": directed_records,
    }
    bootstrap = clustered_bootstrap(
        point_analysis,
        n_replicates=bootstrap_replicates,
        confidence_level=0.95,
        seed=bootstrap_seed,
    )
    _require(
        bootstrap["method"]["cluster_key"] == "question_id"
        and bootstrap["method"]["independent_condition_row_resampling"] is False,
        "bootstrap did not retain complete question_id clusters",
    )
    _require(
        bootstrap["method"]["n_clusters"] == expectations.question_ids,
        "bootstrap question_id cluster count is inconsistent",
    )
    _require(
        bootstrap["method"]["confidence_level"] == BOOTSTRAP_CONFIDENCE_LEVEL
        and bootstrap["method"]["interval"] == "percentile",
        "bootstrap must report 95% percentile confidence intervals",
    )
    metric_values = {
        "country_influence": [row["country_influence"] for row in target_records],
        "evidence_influence": [row["evidence_influence"] for row in target_records],
        "EO_raw": [row["EO_raw"] for row in directed_records],
        "EO_normalized": [row["EO_normalized"] for row in directed_records],
    }
    summaries = {metric: _summarize(values) for metric, values in metric_values.items()}
    classification_order = ("evidence-side", "label-side", "tie")
    classification_counts = {
        label: sum(row["conflict_classification"] == label for row in directed_records)
        for label in classification_order
    }
    _require(
        sum(classification_counts.values()) == expectations.directed_units,
        "conflict classifications do not cover every directed unit",
    )
    classifications = {
        "tolerance": float(classification_tolerance),
        "rule": "EO_raw > tolerance: evidence-side; EO_raw < -tolerance: label-side; otherwise tie",
        "counts": classification_counts,
        "proportions": {
            label: classification_counts[label] / expectations.directed_units
            for label in classification_order
        },
    }

    analysis = {
        "schema_version": "mistral_gguf_full_analysis_v1",
        "validation_status": "PASS",
        "counts": {
            "result_rows": len(enriched_rows),
            "conditions_per_directed_unit": len(CONDITIONS),
            "directed_units": len(directed_records),
            "target_units": len(target_records),
            "question_id_clusters": len({row["question_id"] for row in directed_records}),
            "conflict_rows_classified": len(directed_records),
            "models": 1,
            "backends": 1,
        },
        "model": {
            "model_name": model_name,
            "model_identifier": EXPECTED_MODEL_IDENTIFIER,
            "model_repository": EXPECTED_MODEL_REPOSITORY,
            "model_revision": EXPECTED_MODEL_REVISION,
            "backend": EXPECTED_BACKEND,
            "quantization": EXPECTED_QUANTIZATION,
            "upstream_checkpoint": EXPECTED_UPSTREAM_CHECKPOINT,
            "gguf_filename": EXPECTED_GGUF_FILENAME,
            "gguf_size_bytes": EXPECTED_GGUF_SIZE_BYTES,
            "gguf_sha256": EXPECTED_GGUF_SHA256,
            "llama_cpp_commit": EXPECTED_LLAMA_CPP_COMMIT,
        },
        "metric_definitions": {
            "country_influence": "JSD2(baseline, label) - JSD2(country_label, label)",
            "evidence_influence": "JSD2(baseline, label) - JSD2(evidence, label)",
            "EO_raw": "JSD2(conflict, label) - JSD2(conflict, evidence)",
            "EO_normalized": "[sqrt(JSD2(conflict, label)) - sqrt(JSD2(conflict, evidence))] / sqrt(JSD2(label, evidence))",
            "jensen_shannon_base": 2,
            "jensen_shannon_measure": "divergence_bits",
        },
        "estimands": {
            "country_and_evidence_influence": "144 unique (question_id, label_country) target units",
            "evidence_override": "200 complete directed (question_id, label_country, evidence_country) units",
        },
        "metric_summaries": summaries,
        "bootstrap": bootstrap,
        "conflict_classifications": classifications,
        "per_target_unit": target_records,
        "per_directed_unit": directed_records,
        "interpretation": (
            "Positive EO_raw and EO_normalized values are evidence-side; negative values are "
            "label-side. Evidence Override is signed proximity in the conflict condition and "
            "does not by itself identify a causal mechanism."
        ),
    }
    if provenance_summary is not None:
        analysis["genuine_run_provenance"] = dict(provenance_summary)
    return analysis, enriched_rows


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(row.get(field)) for field in fieldnames})
    return output.getvalue()


def _flattened_results_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    leading = [
        "question_id",
        "label_country",
        "evidence_country",
        "condition",
        "model_name",
        "model_identifier",
        "backend",
        "EO_raw",
        "EO_normalized",
        "country_influence",
        "evidence_influence",
        "conflict_classification",
        "base2_jensen_shannon_divergences",
        "normalized_prediction",
    ]
    all_fields = {field for row in rows for field in row}
    fieldnames = leading + sorted(all_fields - set(leading))
    return _csv_text(rows, fieldnames)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _metric_table_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_name = analysis["model"]["model_name"]
    bootstrap_metrics = analysis["bootstrap"]["models"][model_name]
    rows = []
    for metric in METRICS:
        result = bootstrap_metrics[metric]
        interval = result["percentile_95_ci"]
        rows.append(
            {
                "metric": metric,
                "n": result["n_observations"],
                "estimate": result["estimate"],
                "ci_95_lower": interval["lower"],
                "ci_95_upper": interval["upper"],
                "bootstrap_replicates": analysis["bootstrap"]["method"]["n_replicates"],
                "question_id_clusters": analysis["bootstrap"]["method"]["n_clusters"],
            }
        )
    return rows


def _classification_table_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    classifications = analysis["conflict_classifications"]
    return [
        {
            "classification": label,
            "count": classifications["counts"][label],
            "proportion": classifications["proportions"][label],
            "percent": 100.0 * classifications["proportions"][label],
        }
        for label in ("evidence-side", "label-side", "tie")
    ]


def _metric_latex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & $N$ & Estimate & 95\% CI lower & 95\% CI upper \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(str(row['metric']))} & {row['n']} & "
            f"{row['estimate']:.6f} & {row['ci_95_lower']:.6f} & "
            f"{row['ci_95_upper']:.6f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _classification_latex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Conflict classification & Count & Percent \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(str(row['classification']))} & {row['count']} & "
            f"{row['percent']:.1f}\\% \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _report_markdown(analysis: Mapping[str, Any], generated_files: Iterable[str]) -> str:
    counts = analysis["counts"]
    model = analysis["model"]
    metric_rows = _metric_table_rows(analysis)
    classification_rows = _classification_table_rows(analysis)
    provenance = analysis.get("genuine_run_provenance")
    lines = [
        "# Mistral GGUF full-run experiment report",
        "",
        "## Validation outcome",
        "",
        "**PASS.** The analysis validated exactly 800 genuine llama.cpp result rows: "
        "200 cleaned reciprocal directed units under all four conditions. No synthetic "
        "or fallback row was accepted.",
        "",
        f"- Model: `{model['model_identifier']}`",
        f"- Paper model name: `{model['model_name']}`",
        f"- Upstream checkpoint: `{model['upstream_checkpoint']}`",
        f"- Revision: `{model['model_revision']}`",
        f"- GGUF: `{model['gguf_filename']}` ({model['gguf_size_bytes']} bytes)",
        f"- GGUF SHA-256: `{model['gguf_sha256']}`",
        f"- llama.cpp commit: `{model['llama_cpp_commit']}`",
        f"- Backend / quantization: `{model['backend']}` / `{model['quantization']}`",
        f"- Directed units: {counts['directed_units']}",
        f"- Deduplicated target units for CI/EI: {counts['target_units']}",
        f"- Question-ID clusters: {counts['question_id_clusters']}",
    ]
    if isinstance(provenance, Mapping):
        lines.extend(
            [
                f"- Chat-template source: `{provenance['chat_template_source']}`",
                f"- Embedded chat-template SHA-256: `{provenance['chat_template_sha256']}`",
                f"- Synthetic: `{str(provenance['synthetic']).lower()}`",
            ]
        )
    lines.extend(
        [
            "",
            "All result rows were bound to the validated run fingerprint. The exact "
            "single-file GGUF, embedded model-specific chat template, CPU-only llama.cpp "
            "backend, and no-generation candidate-label scorer passed provenance checks.",
            "",
            "## Point estimates and clustered-bootstrap intervals",
            "",
            "| Metric | N | Estimate | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| `{row['metric']}` | {row['n']} | {row['estimate']:.6f} | "
            f"[{row['ci_95_lower']:.6f}, {row['ci_95_upper']:.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Intervals use {analysis['bootstrap']['method']['n_replicates']:,} "
            "nonparametric bootstrap replicates sampled at the "
            "`question_id` level. Each sampled cluster retains all target units, all "
            "directed and reciprocal pairs, and all condition contributions. One shared "
            "draw is used across every metric; condition rows are never sampled independently.",
            "",
            "## Conflict classification",
            "",
            "Classification uses `EO_raw` with tolerance `1e-12`.",
            "",
            "| Class | Count | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in classification_rows:
        lines.append(f"| {row['classification']} | {row['count']} | {row['percent']:.1f}% |")
    lines.extend(
        [
            "",
            "Positive `EO_raw` means evidence-side; negative means label-side. "
            "`EO_normalized` uses square-root base-2 Jensen--Shannon distance and has the "
            "same orientation. These are signed proximity metrics, not stand-alone causal claims.",
            "",
            "## Generated artifacts",
            "",
        ]
    )
    lines.extend(f"- `{filename}`" for filename in generated_files)
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    analysis: dict[str, Any],
    enriched_rows: Sequence[Mapping[str, Any]],
    *,
    source_results_path: Path | None = None,
    manifest_path: Path | None = None,
    run_signature_path: Path | None = None,
    runtime_path: Path | None = None,
) -> list[Path]:
    """Write all requested JSON, CSV, LaTeX, and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if source_results_path is not None:
        analysis["source_results"] = {
            "path": _portable_path(source_results_path),
            "sha256": _json_sha256(source_results_path),
        }
    if manifest_path is not None:
        analysis["source_manifest"] = {
            "path": _portable_path(manifest_path),
            "sha256": _json_sha256(manifest_path),
        }
    if run_signature_path is not None:
        analysis["source_run_signature"] = {
            "path": _portable_path(run_signature_path),
            "sha256": _json_sha256(run_signature_path),
        }
    if runtime_path is not None:
        analysis["source_runtime"] = {
            "path": _portable_path(runtime_path),
            "sha256": _json_sha256(runtime_path),
        }

    directed = analysis["per_directed_unit"]
    targets = analysis["per_target_unit"]
    metric_rows = _metric_table_rows(analysis)
    classification_table = _classification_table_rows(analysis)
    conflicts = [
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
    metric_tables = {
        "schema_version": "mistral_gguf_full_metric_tables_v1",
        "model_metrics": metric_rows,
        "conflict_classification": classification_table,
    }

    filenames = [
        "analysis.json",
        "validated_results.jsonl",
        "results.csv",
        "directed_unit_metrics.csv",
        "target_unit_metrics.csv",
        "bootstrap_summary.csv",
        "bootstrap_results.json",
        "conflict_classifications.csv",
        "metric_tables.json",
        "tables/model_metrics.tex",
        "tables/conflict_classification.tex",
        "EXPERIMENT_REPORT.md",
    ]
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validated_results.jsonl").write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in enriched_rows
        ),
        encoding="utf-8",
    )
    (output_dir / "results.csv").write_text(
        _flattened_results_csv(enriched_rows), encoding="utf-8"
    )
    directed_fields = [
        "model_name", "question_id", "label_country", "evidence_country",
        "country_influence", "evidence_influence", "EO_raw", "EO_normalized",
        "conflict_to_label_country_jsd2", "conflict_to_evidence_country_jsd2",
        "label_to_evidence_country_jsd2", "conflict_classification",
    ]
    (output_dir / "directed_unit_metrics.csv").write_text(
        _csv_text(directed, directed_fields), encoding="utf-8"
    )
    target_fields = [
        "model_name", "question_id", "label_country", "evidence_countries",
        "directed_repetitions", "country_influence", "evidence_influence",
        "baseline_to_label_country_jsd2", "country_label_to_label_country_jsd2",
        "evidence_to_label_country_jsd2",
    ]
    (output_dir / "target_unit_metrics.csv").write_text(
        _csv_text(targets, target_fields), encoding="utf-8"
    )
    bootstrap_fields = [
        "metric", "n", "estimate", "ci_95_lower", "ci_95_upper",
        "bootstrap_replicates", "question_id_clusters",
    ]
    (output_dir / "bootstrap_summary.csv").write_text(
        _csv_text(metric_rows, bootstrap_fields), encoding="utf-8"
    )
    (output_dir / "bootstrap_results.json").write_text(
        json.dumps(analysis["bootstrap"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    conflict_fields = [
        "model_name", "question_id", "label_country", "evidence_country",
        "conflict_to_label_country_jsd2", "conflict_to_evidence_country_jsd2",
        "EO_raw", "EO_normalized", "conflict_classification",
    ]
    (output_dir / "conflict_classifications.csv").write_text(
        _csv_text(conflicts, conflict_fields), encoding="utf-8"
    )
    (output_dir / "metric_tables.json").write_text(
        json.dumps(metric_tables, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tables_dir / "model_metrics.tex").write_text(_metric_latex(metric_rows), encoding="utf-8")
    (tables_dir / "conflict_classification.tex").write_text(
        _classification_latex(classification_table), encoding="utf-8"
    )
    (output_dir / "EXPERIMENT_REPORT.md").write_text(
        _report_markdown(analysis, filenames), encoding="utf-8"
    )
    artifact_paths = [output_dir / filename for filename in filenames]
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{_json_sha256(path)}  {path.relative_to(output_dir)}\n"
            for path in artifact_paths
        ),
        encoding="utf-8",
    )
    return [*artifact_paths, checksum_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--run-signature",
        type=Path,
        default=None,
        help="Defaults to run_signature.json beside --results",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=None,
        help="Defaults to runtime.json beside --results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the directory containing --results",
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--classification-tolerance", type=float, default=CLASSIFICATION_TOLERANCE
    )
    args = parser.parse_args()

    _require(
        args.bootstrap_replicates == BOOTSTRAP_REPLICATES,
        f"production analysis requires exactly {BOOTSTRAP_REPLICATES:,} bootstrap replicates",
    )
    _require(
        args.classification_tolerance == CLASSIFICATION_TOLERANCE,
        "production conflict classification requires tolerance 1e-12",
    )

    observed_manifest_sha256 = _json_sha256(args.manifest)
    if observed_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "cleaned manifest identity mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, observed {observed_manifest_sha256}"
        )
    run_signature_path = (
        args.run_signature
        if args.run_signature is not None
        else args.results.parent / DEFAULT_RUN_SIGNATURE_NAME
    )
    runtime_path = (
        args.runtime
        if args.runtime is not None
        else args.results.parent / DEFAULT_RUNTIME_NAME
    )
    run_signature = load_json_object(run_signature_path, "run signature")
    runtime_record = load_json_object(runtime_path, "runtime record")
    provenance = validate_provenance(run_signature, runtime_record)
    rows = load_jsonl(args.results)
    manifest = load_manifest(args.manifest)
    analysis, enriched = analyze_full_run(
        rows,
        manifest,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        classification_tolerance=args.classification_tolerance,
        provenance_summary=provenance,
    )
    output_dir = args.output_dir if args.output_dir is not None else args.results.parent
    paths = write_outputs(
        output_dir,
        analysis,
        enriched,
        source_results_path=args.results,
        manifest_path=args.manifest,
        run_signature_path=run_signature_path,
        runtime_path=runtime_path,
    )
    print(
        "MISTRAL FULL ANALYSIS PASS: "
        f"{analysis['counts']['result_rows']} rows, "
        f"{analysis['counts']['target_units']} CI/EI targets, "
        f"{analysis['counts']['directed_units']} EO units, "
        f"{analysis['bootstrap']['method']['n_replicates']} clustered bootstrap replicates; "
        f"wrote {len(paths)} artifacts to {output_dir}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and analyze the 800-row Gemma GGUF full experiment.

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
from src.gemma_gguf_spec import (
    MODEL_IDENTIFIER,
    MODEL_NAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    QUANTIZATION,
)
from src.main import prepare_evidence_distribution, reorder_distribution
from src.metrics import Metrics


DEFAULT_RESULTS = PROJECT_ROOT / "experiments/gemma_gguf_full/results.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/pairs/country_pairs_v2.json"

CONDITIONS = ("baseline", "country_label", "evidence", "conflict")
CONDITION_ORDER = {condition: index for index, condition in enumerate(CONDITIONS)}
METRICS = ("country_influence", "evidence_influence", "EO_raw", "EO_normalized")
EXPECTED_MODEL_NAME = MODEL_NAME
EXPECTED_MODEL_IDENTIFIER = MODEL_IDENTIFIER
EXPECTED_MODEL_REPOSITORY = MODEL_REPOSITORY
EXPECTED_MODEL_REVISION = MODEL_REVISION
EXPECTED_BACKEND = "llama.cpp"
EXPECTED_QUANTIZATION = QUANTIZATION
EXPECTED_SCHEMA_VERSION = "gemma-gguf-full-v2"
EXPECTED_MANIFEST_SHA256 = "429e07b5f2019be15fd9306e4722ff3878fb575c0eb03db7f1c0d1aacbad5d8b"
DERIVED_ABS_TOLERANCE = 1e-12


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


def _validate_model_and_backend(row: Mapping[str, Any], context: str) -> None:
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
    _require(quantization.get("gpu_layers") == 0, f"{context}: full Gemma run must be CPU-only")


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


def classify_conflict(EO_raw: float, tolerance: float = 1e-12) -> str:
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
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
    classification_tolerance: float = 1e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a validated analysis and rows enriched with recomputed metrics."""
    _require(
        len(rows) == expectations.result_rows,
        f"results have {len(rows)} rows; expected exactly {expectations.result_rows}",
    )
    manifest_by_key, manifest_order = validate_manifest(manifest, expectations)

    groups: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    models: set[str] = set()
    backends: set[str] = set()
    seen_row_keys: set[tuple[str, str, str, str]] = set()

    for index, row in enumerate(rows):
        context = f"results[{index}]"
        _require(isinstance(row, Mapping), f"{context} must be an object")
        _validate_model_and_backend(row, context)
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
        predictions: dict[str, dict[str, float]] = {}
        label_references: dict[str, dict[str, float]] = {}
        evidence_references: dict[str, dict[str, float]] = {}
        permutations: dict[str, tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]] = {}

        for condition in CONDITIONS:
            row = condition_rows[condition]
            context = f"{key!r}/{condition}"
            _require(row.get("original_options") == list(options), f"{context}: original_options differ from manifest")
            permutations[condition] = _validate_option_permutation(row, options, context)
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
        "schema_version": "gemma_gguf_full_analysis_v1",
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
    lines = [
        "# Gemma GGUF full-run experiment report",
        "",
        "## Validation outcome",
        "",
        "**PASS.** The analysis validated exactly 800 genuine llama.cpp result rows: "
        "200 cleaned reciprocal directed units under all four conditions. No synthetic "
        "or fallback row was accepted.",
        "",
        f"- Model: `{model['model_identifier']}`",
        f"- Revision: `{model['model_revision']}`",
        f"- Backend / quantization: `{model['backend']}` / `{model['quantization']}`",
        f"- Directed units: {counts['directed_units']}",
        f"- Deduplicated target units for CI/EI: {counts['target_units']}",
        f"- Question-ID clusters: {counts['question_id_clusters']}",
        "",
        "## Point estimates and clustered-bootstrap intervals",
        "",
        "| Metric | N | Estimate | 95% CI |",
        "|---|---:|---:|---:|",
    ]
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
            "Classification uses `EO_raw` with tolerance "
            f"`{analysis['conflict_classifications']['tolerance']:g}`.",
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


def _exclusion_failure_report() -> str:
    repair_path = PROJECT_ROOT / "data/audit/data_repair_summary.json"
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    lines = [
        "# Exclusion and Failure Report",
        "",
        "## Data exclusions",
        "",
        f"- Source questions: {repair['source_questions']}",
        f"- Repaired eligible questions: {repair['repaired_questions']}",
        "- Frozen experiment sample: 44 question IDs selected before this model run",
        f"- Final reciprocal directed units: {repair['directed_pairs_written']}",
        f"- Reciprocal balance: {repair['reciprocal_pair_balance']}",
        "",
        "| Exclusion reason | Questions |",
        "|---|---:|",
    ]
    for reason, count in sorted(repair["excluded_questions_by_reason"].items()):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## Inference and analysis failures",
            "",
            "No failed assertion, fallback, model substitution, invalid final row, or "
            "independently resampled condition row was accepted. Any interrupted work "
            "was represented only by atomic row checkpoints and was resumed without "
            "repeating a validated saved row.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    analysis: dict[str, Any],
    enriched_rows: Sequence[Mapping[str, Any]],
    *,
    source_results_path: Path | None = None,
    manifest_path: Path | None = None,
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
        "schema_version": "gemma_gguf_full_metric_tables_v1",
        "model_metrics": metric_rows,
        "conflict_classification": classification_table,
    }

    filenames = [
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
        "EXCLUSION_AND_FAILURE_REPORT.md",
    ]
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    (output_dir / "EXCLUSION_AND_FAILURE_REPORT.md").write_text(
        _exclusion_failure_report(), encoding="utf-8"
    )
    return [output_dir / filename for filename in filenames]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the directory containing --results",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--classification-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    observed_manifest_sha256 = _json_sha256(args.manifest)
    if observed_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "cleaned manifest identity mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, observed {observed_manifest_sha256}"
        )
    rows = load_jsonl(args.results)
    manifest = load_manifest(args.manifest)
    analysis, enriched = analyze_full_run(
        rows,
        manifest,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        classification_tolerance=args.classification_tolerance,
    )
    output_dir = args.output_dir if args.output_dir is not None else args.results.parent
    paths = write_outputs(
        output_dir,
        analysis,
        enriched,
        source_results_path=args.results,
        manifest_path=args.manifest,
    )
    print(
        "GEMMA FULL ANALYSIS PASS: "
        f"{analysis['counts']['result_rows']} rows, "
        f"{analysis['counts']['target_units']} CI/EI targets, "
        f"{analysis['counts']['directed_units']} EO units, "
        f"{analysis['bootstrap']['method']['n_replicates']} clustered bootstrap replicates; "
        f"wrote {len(paths)} artifacts to {output_dir}"
    )


if __name__ == "__main__":
    main()

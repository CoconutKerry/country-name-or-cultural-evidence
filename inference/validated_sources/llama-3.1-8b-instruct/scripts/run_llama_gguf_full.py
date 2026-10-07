#!/usr/bin/env python3
"""Run/resume the approved Llama-only 200-unit GGUF CPU experiment.

This entry point is deliberately separate from both the accepted two-unit
smoke and the Colab four-model approval gate.  It loads only the pinned
official Llama Q4_K_M GGUF, scores option-label continuations through the
accepted low-level llama.cpp backend, and checkpoints every completed
condition row before proceeding.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, IO, Iterable, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_genuine_llama_gguf_smoke import (
    EXPECTED_GGUF_ARCHITECTURE,
    EXPECTED_GGUF_CONTEXT_LENGTH,
    EXPECTED_GGUF_FILE_TYPE,
    EXPECTED_UNITS as EXPECTED_SMOKE_UNITS,
    MODEL_IDENTIFIER,
    MODEL_NAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SHARDS,
    QUANTIZATION,
    REMOTE_MODEL_LAST_MODIFIED,
    REMOTE_WEIGHT_COMMIT,
    UPSTREAM_CHECKPOINT,
    UPSTREAM_CURRENT_REVISION,
    command_output,
    environment_information,
    require_pinned_revision,
    sha256_file,
    verified_model_files,
)
from src.data_loader import DataLoader
from src.llama_gguf_runner import (
    LLAMA31_CHAT_TEMPLATE_SHA256,
    LLAMA31_DEFAULT_SYSTEM_MESSAGE,
    Llama31GGUFRunner,
    verify_llama31_serialized_chat,
)
from src.main import (
    build_prompt,
    prepare_evidence_distribution,
    reorder_distribution,
    shuffle_options,
    stable_unit_seed,
)
from src.metrics import Metrics
from src.prompt_builder import PromptBuilder
from src.scoring import (
    map_label_probabilities,
    normalize_log_scores,
    option_labels,
    recover_canonical_distribution,
)


FULL_SCHEMA_VERSION = "llama-gguf-full-v2"
EXPECTED_OUTPUT_CONDITIONS = (
    "baseline",
    "country_label",
    "evidence",
    "conflict",
)
INTERNAL_CONDITION = {"evidence": "population_evidence"}
EXPECTED_DIRECTED_UNIT_COUNT = 200
EXPECTED_ROW_COUNT = EXPECTED_DIRECTED_UNIT_COUNT * len(EXPECTED_OUTPUT_CONDITIONS)
PAIR_MANIFEST = "data/pairs/country_pairs_v2.json"
DATASET_PATH = "data/processed/dataset_v2.json"
PAIR_MANIFEST_SHA256 = (
    "429e07b5f2019be15fd9306e4722ff3878fb575c0eb03db7f1c0d1aacbad5d8b"
)
DATASET_SHA256 = "8e6a0a35dbb96b049e90b075410b68bc84c0a6fc69fee6b4f2255d6302b26e09"
EXPECTED_QUESTION_CLUSTERS = 44
EXPECTED_TARGET_UNIT_COUNT = 144
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments/llama_gguf_full"
DEFAULT_SEED = 42
NORMALIZATION_TOLERANCE = 1e-8
METRIC_TOLERANCE = 1e-10
CLASSIFICATION_TOLERANCE = 1e-12

# Accepted genuine-smoke runtime provenance.  A full run must not drift from
# the exact implementation that was independently reviewed.
LLAMA_CPP_COMMIT = "62acc89c26c66076cb72e049f307fbe93b8b9750"
LLAMA_CPP_VERSION = "0.3.0-dev"
LLAMA_CHAT_TEMPLATE_SHA256 = LLAMA31_CHAT_TEMPLATE_SHA256
ACCEPTED_GGUF_RUNNER_SHA256 = (
    "db5f995ea75a4f8404ff1a1b9d5a0e275a54ede41e2064a1728cbb9fbccc2283"
)
ACCEPTED_GGUF_HELPER_SOURCE_SHA256 = (
    "235ea664e5183786ef066f3d6ea692d60d2dcacaece0063a84a1ec6e5454871b"
)
LLAMA_GGUF_RUNNER_SHA256 = (
    "8171285f70d19c8d88e323112b54c78de41e5f829a6c0aeeb176bc76e85c2d27"
)
PERMUTATION_KEY = "question_id::label_country"
REQUIRED_SMOKE_ASSERTIONS = {
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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write and replace a file atomically after flushing its contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _log(handle: IO[str], message: str) -> None:
    handle.write(f"{now_utc()} {message}\n")
    handle.flush()
    os.fsync(handle.fileno())


def directed_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a complete directed key from a pair manifest or result row."""

    question_id = value.get("question_id")
    label_country = value.get("label_country", value.get("country"))
    evidence_country = value.get("evidence_country", value.get("conflict_country"))
    fields = (question_id, label_country, evidence_country)
    if not all(isinstance(field, str) and field for field in fields):
        raise ValueError("directed key requires question_id and two non-empty countries")
    return str(question_id), str(label_country), str(evidence_country)


def validate_manifest(
    manifest: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_DIRECTED_UNIT_COUNT,
) -> list[tuple[str, str, str]]:
    """Validate exact cardinality, uniqueness, orderable options, and reciprocity."""

    if len(manifest) != expected_count:
        raise ValueError(
            f"Expected {expected_count} directed units, found {len(manifest)}"
        )
    keys = [directed_key(pair) for pair in manifest]
    if len(set(keys)) != len(keys):
        raise ValueError("Directed manifest keys are not unique")
    self_pairs = [key for key in keys if key[1] == key[2]]
    if self_pairs:
        raise ValueError(f"Directed manifest contains self-pairs: {self_pairs[:3]!r}")
    key_set = set(keys)
    missing_reciprocals = [
        key for key in keys if (key[0], key[2], key[1]) not in key_set
    ]
    if missing_reciprocals:
        raise ValueError(
            f"Directed manifest lacks reciprocal units: {missing_reciprocals[:3]!r}"
        )
    for index, pair in enumerate(manifest):
        if pair.get("mapping_status") != "exact_shared_response_schema":
            raise ValueError(
                f"Manifest pair {index} does not use the exact shared response schema"
            )
        options = pair.get("options")
        if (
            not isinstance(options, list)
            or len(options) < 2
            or not all(isinstance(option, str) and option for option in options)
            or len(options) != len(set(options))
        ):
            raise ValueError(f"Manifest pair {index} has invalid canonical options")
        expected_unit_id = f"{keys[index][0]}::{keys[index][1]}=>{keys[index][2]}"
        if pair.get("unit_id", expected_unit_id) != expected_unit_id:
            raise ValueError(f"Manifest pair {index} has an inconsistent unit_id")
    question_count = len({key[0] for key in keys})
    target_count = len({(key[0], key[1]) for key in keys})
    if question_count != EXPECTED_QUESTION_CLUSTERS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTION_CLUSTERS} question clusters, found {question_count}"
        )
    if target_count != EXPECTED_TARGET_UNIT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TARGET_UNIT_COUNT} target units, found {target_count}"
        )
    return keys


def prepare_unit(
    loader: DataLoader,
    pair: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Prepare one directed unit and one target-stable displayed permutation.

    Some label-country targets occur with more than one evidence country.  The
    repaired protocol keys the permutation by target so their non-conflict
    conditions stay byte-identical, while all four rows of every directed unit
    necessarily share that same permutation.
    """

    question_id, label_country, evidence_country = directed_key(pair)
    canonical_options = [str(option) for option in pair["options"]]
    target_unit_id = f"{question_id}::{label_country}"
    permutation_seed = stable_unit_seed(seed, target_unit_id)
    displayed_options = shuffle_options(canonical_options, permutation_seed)
    label_human = loader.get_response_distribution(
        question_id, label_country, canonical_options
    )
    evidence_human = loader.get_response_distribution(
        question_id, evidence_country, canonical_options
    )
    label_displayed = reorder_distribution(label_human, displayed_options)
    evidence_displayed = reorder_distribution(evidence_human, displayed_options)
    return {
        "key": (question_id, label_country, evidence_country),
        "unit_id": f"{question_id}::{label_country}=>{evidence_country}",
        "target_unit_id": target_unit_id,
        "permutation_seed": permutation_seed,
        "question": loader.get_question_text(question_id),
        "canonical_options": canonical_options,
        "displayed_options": displayed_options,
        "label_human": label_human,
        "evidence_human": evidence_human,
        "label_evidence_displayed": prepare_evidence_distribution(
            label_displayed, displayed_options
        ),
        "conflict_evidence_displayed": prepare_evidence_distribution(
            evidence_displayed, displayed_options
        ),
        "label_country_display": pair.get(
            "country_display",
            loader.get_country_display_name(question_id, label_country),
        ),
        "evidence_country_display": pair.get(
            "conflict_country_display",
            loader.get_country_display_name(question_id, evidence_country),
        ),
    }


def classify_conflict(eo_raw: float) -> str:
    """Classify the conflict prediction using the primary raw EO sign."""

    value = float(eo_raw)
    if not math.isfinite(value):
        raise ValueError("EO_raw must be finite")
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=CLASSIFICATION_TOLERANCE):
        return "tie"
    return "evidence-side" if value > 0.0 else "label-side"


def _normalize_probability_mapping(
    value: Any,
    keys: Sequence[str],
    context: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{context} keys do not match the expected options")
    result: dict[str, float] = {}
    for key in keys:
        number = value[key]
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise ValueError(f"{context}[{key!r}] is not a valid probability")
        result[key] = float(number)
    if not math.isclose(
        math.fsum(result.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(f"{context} does not sum to one")
    return result


def _same_distribution(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float = NORMALIZATION_TOLERANCE,
) -> bool:
    return set(left) == set(right) and all(
        math.isclose(
            float(left[key]),
            float(right[key]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for key in left
    )


def _metric_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition = {str(row["condition"]): row for row in rows}
    if set(by_condition) != set(EXPECTED_OUTPUT_CONDITIONS):
        raise ValueError("A directed unit must contain exactly four conditions")
    reference = by_condition["baseline"]
    label_human = reference["label_country_human_distribution"]
    evidence_human = reference["evidence_country_human_distribution"]
    baseline = by_condition["baseline"]["normalized_prediction"]
    country = by_condition["country_label"]["normalized_prediction"]
    evidence = by_condition["evidence"]["normalized_prediction"]
    conflict = by_condition["conflict"]["normalized_prediction"]
    components = {
        "baseline_to_label_country": Metrics.js_divergence(baseline, label_human),
        "country_label_to_label_country": Metrics.js_divergence(country, label_human),
        "evidence_to_label_country": Metrics.js_divergence(evidence, label_human),
        "conflict_to_label_country": Metrics.js_divergence(conflict, label_human),
        "conflict_to_evidence_country": Metrics.js_divergence(conflict, evidence_human),
    }
    eo_raw = Metrics.evidence_override_raw(conflict, evidence_human, label_human)
    eo_normalized = Metrics.evidence_override_normalized(
        conflict, evidence_human, label_human
    )
    return {
        "components": components,
        "country_influence": Metrics.country_influence(baseline, country, label_human),
        "evidence_influence": Metrics.evidence_influence(
            baseline, evidence, label_human
        ),
        "EO_raw": eo_raw,
        "EO_normalized": eo_normalized,
        "classification": classify_conflict(eo_raw),
    }


def enrich_unit_metrics(rows: list[dict[str, Any]]) -> None:
    values = _metric_values(rows)
    label_human = rows[0]["label_country_human_distribution"]
    evidence_human = rows[0]["evidence_country_human_distribution"]
    for row in rows:
        prediction = row["normalized_prediction"]
        row["jensen_shannon_base"] = 2
        row["jensen_shannon_measure"] = "divergence_bits"
        row["base2_jensen_shannon_divergences"] = {
            "prediction_to_label_country": Metrics.js_divergence(
                prediction, label_human
            ),
            "prediction_to_evidence_country": Metrics.js_divergence(
                prediction, evidence_human
            ),
            **values["components"],
        }
        row["country_influence"] = values["country_influence"]
        row["evidence_influence"] = values["evidence_influence"]
        row["EO_raw"] = values["EO_raw"]
        row["EO_normalized"] = values["EO_normalized"]
        row["conflict_classification"] = (
            values["classification"] if row["condition"] == "conflict" else None
        )


def run_directed_unit(
    runner: Llama31GGUFRunner,
    loader: DataLoader,
    pair: Mapping[str, Any],
    model_files: Sequence[Mapping[str, Any]],
    run_fingerprint: str,
    *,
    seed: int = DEFAULT_SEED,
    existing_rows: Sequence[Mapping[str, Any]] | None = None,
    on_row_saved: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Score missing conditions, exposing each completed row immediately."""

    if runner.backend != "llama.cpp" or runner.is_synthetic:
        raise AssertionError("Only the genuine llama.cpp backend is permitted")
    unit = prepare_unit(loader, pair, seed=seed)
    builder = PromptBuilder()
    rows: list[dict[str, Any]] = [dict(row) for row in (existing_rows or ())]
    observed_prefix = tuple(str(row.get("condition")) for row in rows)
    if observed_prefix != EXPECTED_OUTPUT_CONDITIONS[: len(rows)]:
        raise ValueError("existing partial rows are not an ordered condition prefix")
    for condition in EXPECTED_OUTPUT_CONDITIONS[len(rows) :]:
        internal_condition = INTERNAL_CONDITION.get(condition, condition)
        raw_prompt = build_prompt(
            builder,
            internal_condition,
            unit["question"],
            unit["displayed_options"],
            unit["label_country_display"],
            unit["label_evidence_displayed"],
            unit["conflict_evidence_displayed"],
        )
        displayed_prediction = runner.predict_distribution(
            raw_prompt,
            unit["displayed_options"],
            temperature=0.0,
        )
        prediction = recover_canonical_distribution(
            displayed_prediction, unit["canonical_options"]
        )
        scoring = runner.get_last_scoring_metadata()
        labels = option_labels(len(unit["displayed_options"]))
        label_probabilities = {
            label: float(scoring["normalized_label_probabilities"][label])
            for label in labels
        }
        raw_scores = {
            label: float(scoring["raw_label_log_scores"][label]) for label in labels
        }
        label_to_option = dict(zip(labels, unit["displayed_options"]))
        evidence_presented = condition in {"evidence", "conflict"}
        if condition == "evidence":
            presented_displayed = unit["label_evidence_displayed"]
            source_evidence: Mapping[str, float] | None = unit["label_human"]
        elif condition == "conflict":
            presented_displayed = unit["conflict_evidence_displayed"]
            source_evidence = unit["evidence_human"]
        else:
            presented_displayed = None
            source_evidence = None
        presented_evidence = (
            recover_canonical_distribution(
                presented_displayed, unit["canonical_options"]
            )
            if presented_displayed is not None
            else None
        )
        displayed_trace = [
            {
                "label": label,
                "semantic_option": label_to_option[label],
                "token_ids": scoring["candidates"][label]["token_ids"],
                "token_count": scoring["candidates"][label]["token_count"],
                "raw_token_logits": scoring["candidates"][label]["raw_token_logits"],
                "token_log_probabilities": scoring["candidates"][label][
                    "token_log_probabilities"
                ],
                "sequence_log_probability": raw_scores[label],
            }
            for label in labels
        ]
        rows.append(
            {
                "schema_version": FULL_SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "question_id": unit["key"][0],
                "label_country": unit["key"][1],
                "evidence_country": unit["key"][2],
                "condition": condition,
                "unit_id": unit["unit_id"],
                "target_unit_id": unit["target_unit_id"],
                "permutation_seed": unit["permutation_seed"],
                "permutation_key": PERMUTATION_KEY,
                "model_name": MODEL_NAME,
                "model_identifier": MODEL_IDENTIFIER,
                "model_repository": MODEL_REPOSITORY,
                "model_revision": MODEL_REVISION,
                "backend": "llama.cpp",
                "synthetic": False,
                "quantization": {
                    "format": "GGUF",
                    "type": QUANTIZATION,
                    "gpu_layers": 0,
                    "files": [str(item["filename"]) for item in model_files],
                },
                "structured_messages": scoring["structured_messages"],
                "raw_user_prompt": raw_prompt,
                "serialized_chat_templated_prompt": scoring["serialized_prompt"],
                "serialized_prompt_sha256": scoring["serialized_prompt_sha256"],
                "original_options": list(unit["canonical_options"]),
                "displayed_options": list(unit["displayed_options"]),
                "displayed_option_order": list(unit["displayed_options"]),
                "restored_original_option_order": list(unit["canonical_options"]),
                "displayed_option_labels": labels,
                "displayed_label_to_option": label_to_option,
                "displayed_labels_and_token_ids": displayed_trace,
                "candidate_label_token_ids": {
                    label: scoring["candidates"][label]["token_ids"]
                    for label in labels
                },
                "raw_candidate_scores": raw_scores,
                "normalized_label_probabilities": label_probabilities,
                "normalized_prediction": prediction,
                "label_country_human_distribution": unit["label_human"],
                "evidence_country_human_distribution": unit["evidence_human"],
                "presented_evidence_distribution": presented_evidence,
                "source_evidence_distribution": source_evidence,
                "evidence_presented": evidence_presented,
                "scoring_method": (
                    "full_contextual_option_label_sequence_log_probability"
                ),
                "scoring_trace": scoring,
                "generated_answer": None,
            }
        )
        if on_row_saved is not None:
            on_row_saved([dict(row) for row in rows])
    enrich_unit_metrics(rows)
    return rows


def _validate_unit_rows(
    rows: Sequence[Mapping[str, Any]],
    pair: Mapping[str, Any],
    *,
    run_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    key = directed_key(pair)
    expected_row_keys = [(*key, condition) for condition in EXPECTED_OUTPUT_CONDITIONS]
    actual_row_keys = [(*directed_key(row), str(row.get("condition"))) for row in rows]
    if actual_row_keys != expected_row_keys:
        raise ValueError(f"Rows for {key!r} are incomplete, duplicated, or out of order")
    if len(rows) != len(EXPECTED_OUTPUT_CONDITIONS):
        raise ValueError(f"Rows for {key!r} do not contain exactly four conditions")

    canonical = [str(option) for option in pair["options"]]
    target_unit_id = f"{key[0]}::{key[1]}"
    expected_seed = stable_unit_seed(DEFAULT_SEED, target_unit_id)
    expected_displayed = shuffle_options(canonical, expected_seed)
    validated: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        context = f"{key!r}/{EXPECTED_OUTPUT_CONDITIONS[index]}"
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{context}: row must be an object")
        row = dict(raw_row)
        required_equal = {
            "schema_version": FULL_SCHEMA_VERSION,
            "unit_id": f"{key[0]}::{key[1]}=>{key[2]}",
            "target_unit_id": target_unit_id,
            "permutation_seed": expected_seed,
            "permutation_key": PERMUTATION_KEY,
            "model_name": MODEL_NAME,
            "model_identifier": MODEL_IDENTIFIER,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "backend": "llama.cpp",
            "synthetic": False,
        }
        for field, expected in required_equal.items():
            if row.get(field) != expected:
                raise ValueError(f"{context}: invalid {field}")
        if run_fingerprint is not None and row.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"{context}: run fingerprint mismatch")
        if row.get("original_options") != canonical:
            raise ValueError(f"{context}: canonical options changed")
        if row.get("displayed_options") != expected_displayed:
            raise ValueError(f"{context}: displayed permutation is not deterministic")
        if row.get("displayed_option_order") != expected_displayed:
            raise ValueError(f"{context}: displayed-order aliases disagree")
        if row.get("restored_original_option_order") != canonical:
            raise ValueError(f"{context}: restored option order is invalid")
        labels = option_labels(len(expected_displayed))
        label_map = dict(zip(labels, expected_displayed))
        if row.get("displayed_option_labels") != labels:
            raise ValueError(f"{context}: displayed labels are invalid")
        if row.get("displayed_label_to_option") != label_map:
            raise ValueError(f"{context}: label-to-option map is invalid")

        label_probabilities = _normalize_probability_mapping(
            row.get("normalized_label_probabilities"), labels, f"{context}.labels"
        )
        prediction = _normalize_probability_mapping(
            row.get("normalized_prediction"), canonical, f"{context}.prediction"
        )
        displayed_prediction = map_label_probabilities(
            labels,
            expected_displayed,
            [label_probabilities[label] for label in labels],
        )
        recovered = recover_canonical_distribution(displayed_prediction, canonical)
        if not _same_distribution(recovered, prediction):
            raise ValueError(f"{context}: option permutation recovery failed")
        raw_scores = row.get("raw_candidate_scores")
        if (
            not isinstance(raw_scores, Mapping)
            or set(raw_scores) != set(labels)
            or not all(
                isinstance(raw_scores[label], (int, float))
                and not isinstance(raw_scores[label], bool)
                and math.isfinite(float(raw_scores[label]))
                for label in labels
            )
        ):
            raise ValueError(f"{context}: raw candidate scores are invalid")
        expected_label_probabilities = normalize_log_scores(
            [float(raw_scores[label]) for label in labels], temperature=0.0
        )
        if any(
            not math.isclose(
                label_probabilities[label],
                expected,
                rel_tol=0.0,
                abs_tol=NORMALIZATION_TOLERANCE,
            )
            for label, expected in zip(labels, expected_label_probabilities)
        ):
            raise ValueError(f"{context}: raw scores do not normalize to probabilities")

        label_human = _normalize_probability_mapping(
            row.get("label_country_human_distribution"),
            canonical,
            f"{context}.label human",
        )
        evidence_human = _normalize_probability_mapping(
            row.get("evidence_country_human_distribution"),
            canonical,
            f"{context}.evidence human",
        )
        condition = str(row["condition"])
        expected_presented = condition in {"evidence", "conflict"}
        if row.get("evidence_presented") is not expected_presented:
            raise ValueError(f"{context}: evidence-presented flag is wrong")
        for field in (
            "presented_evidence_distribution",
            "source_evidence_distribution",
        ):
            if field not in row:
                raise ValueError(f"{context}: {field} must be explicitly present")
        if not expected_presented:
            if row.get("presented_evidence_distribution") is not None:
                raise ValueError(f"{context}: no-evidence row stores presented evidence")
            if row.get("source_evidence_distribution") is not None:
                raise ValueError(f"{context}: no-evidence row stores source evidence")
        else:
            presented = _normalize_probability_mapping(
                row.get("presented_evidence_distribution"),
                canonical,
                f"{context}.presented evidence",
            )
            source = _normalize_probability_mapping(
                row.get("source_evidence_distribution"),
                canonical,
                f"{context}.source evidence",
            )
            expected_source = label_human if condition == "evidence" else evidence_human
            if not _same_distribution(source, expected_source):
                raise ValueError(f"{context}: source evidence is inconsistent")
            expected_source_displayed = reorder_distribution(
                expected_source, expected_displayed
            )
            expected_presented_displayed = prepare_evidence_distribution(
                expected_source_displayed, expected_displayed
            )
            expected_presented_canonical = recover_canonical_distribution(
                expected_presented_displayed, canonical
            )
            if not _same_distribution(
                presented,
                expected_presented_canonical,
                tolerance=1e-12,
            ):
                raise ValueError(
                    f"{context}: presented evidence does not match the rounded prompt source"
                )

        scoring = row.get("scoring_trace")
        if not isinstance(scoring, Mapping):
            raise ValueError(f"{context}: scoring trace is missing")
        if row.get("scoring_method") != (
            "full_contextual_option_label_sequence_log_probability"
        ):
            raise ValueError(f"{context}: scoring method is invalid")
        raw_user_prompt = row.get("raw_user_prompt")
        if not isinstance(raw_user_prompt, str) or not raw_user_prompt.rstrip().endswith(
            "Answer:"
        ):
            raise ValueError(f"{context}: prompt does not end at the answer position")
        messages = row.get("structured_messages")
        expected_messages = [
            {"role": "system", "content": LLAMA31_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": row.get("raw_user_prompt")},
        ]
        if messages != expected_messages:
            raise ValueError(f"{context}: structured messages are invalid")
        serialized = row.get("serialized_chat_templated_prompt")
        if not isinstance(serialized, str):
            raise ValueError(f"{context}: serialized prompt is missing")
        try:
            verify_llama31_serialized_chat(serialized, expected_messages)
        except RuntimeError as exc:
            raise ValueError(f"{context}: invalid Llama-3.1 serialization") from exc
        if row.get("serialized_prompt_sha256") != hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"{context}: serialized prompt hash is invalid")
        if scoring.get("serialized_prompt") != serialized:
            raise ValueError(f"{context}: scoring prompt aliases disagree")
        if scoring.get("structured_messages") != messages:
            raise ValueError(f"{context}: scoring message aliases disagree")
        if scoring.get("label_to_option") != label_map:
            raise ValueError(f"{context}: scoring label map is invalid")
        if scoring.get("raw_label_log_scores") != raw_scores:
            raise ValueError(f"{context}: scoring raw-score aliases disagree")
        if not _same_distribution(
            scoring.get("normalized_label_probabilities", {}),
            label_probabilities,
        ):
            raise ValueError(f"{context}: scoring probability aliases disagree")
        if scoring.get("embedded_template_serialization_verified") is not True:
            raise ValueError(f"{context}: embedded template was not verified")
        if scoring.get("prompt_prefix_verified") is not True:
            raise ValueError(f"{context}: prompt prefix was not verified")
        if row.get("generated_answer") is not None or scoring.get("generated_answer") is not None:
            raise ValueError(f"{context}: generation or answer parsing is forbidden")
        candidates = scoring.get("candidates")
        if not isinstance(candidates, Mapping) or set(candidates) != set(labels):
            raise ValueError(f"{context}: not every candidate label was scored")
        prompt_count = scoring.get("prompt_token_count")
        if not isinstance(prompt_count, int) or isinstance(prompt_count, bool) or prompt_count <= 0:
            raise ValueError(f"{context}: prompt token count is invalid")
        if (
            scoring.get("answer_target_position") != prompt_count
            or scoring.get("answer_predictive_logit_position") != prompt_count - 1
        ):
            raise ValueError(f"{context}: stored answer position is invalid")
        candidate_aliases = row.get("candidate_label_token_ids")
        if not isinstance(candidate_aliases, Mapping) or set(candidate_aliases) != set(labels):
            raise ValueError(f"{context}: candidate token-ID aliases are incomplete")
        seen_token_paths: set[tuple[int, ...]] = set()
        for label in labels:
            candidate = candidates[label]
            if not isinstance(candidate, Mapping):
                raise ValueError(f"{context}/{label}: candidate trace is invalid")
            token_ids = candidate.get("token_ids")
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or not all(isinstance(token_id, int) and token_id >= 0 for token_id in token_ids)
            ):
                raise ValueError(f"{context}/{label}: candidate tokenization is invalid")
            path = tuple(token_ids)
            if path in seen_token_paths:
                raise ValueError(f"{context}/{label}: candidate token path is duplicated")
            seen_token_paths.add(path)
            if candidate.get("token_count") != len(token_ids):
                raise ValueError(f"{context}/{label}: token count is inconsistent")
            if candidate_aliases[label] != token_ids:
                raise ValueError(f"{context}/{label}: token-ID aliases disagree")
            targets = list(range(prompt_count, prompt_count + len(token_ids)))
            predictive = [position - 1 for position in targets]
            if candidate.get("target_token_positions") != targets:
                raise ValueError(f"{context}/{label}: target positions are wrong")
            if (
                candidate.get("predictive_logit_positions") != predictive
                or predictive[0] != prompt_count - 1
            ):
                raise ValueError(f"{context}/{label}: candidate is scored at the wrong position")
            token_log_probabilities = candidate.get("token_log_probabilities")
            if (
                not isinstance(token_log_probabilities, list)
                or len(token_log_probabilities) != len(token_ids)
                or not all(math.isfinite(float(value)) for value in token_log_probabilities)
                or not math.isclose(
                    math.fsum(float(value) for value in token_log_probabilities),
                    float(candidate.get("sequence_log_probability")),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(candidate["sequence_log_probability"]),
                    float(raw_scores[label]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(f"{context}/{label}: sequence score is incomplete")
        quantization = row.get("quantization")
        if (
            not isinstance(quantization, Mapping)
            or quantization.get("format") != "GGUF"
            or quantization.get("type") != QUANTIZATION
            or quantization.get("gpu_layers") != 0
            or quantization.get("files")
            != [str(item["filename"]) for item in MODEL_SHARDS]
        ):
            raise ValueError(f"{context}: quantization provenance is invalid")
        validated.append(row)

    reference = validated[0]
    for row in validated[1:]:
        for field in (
            "original_options",
            "displayed_options",
            "displayed_option_labels",
            "displayed_label_to_option",
            "permutation_seed",
            "label_country_human_distribution",
            "evidence_country_human_distribution",
        ):
            if row[field] != reference[field]:
                raise ValueError(f"{key!r}: {field} varies across conditions")

    metrics = _metric_values(validated)
    component_keys = {
        "prediction_to_label_country",
        "prediction_to_evidence_country",
        *metrics["components"].keys(),
    }
    for row in validated:
        if row.get("jensen_shannon_base") != 2:
            raise ValueError(f"{key!r}: Jensen-Shannon base is not 2")
        if row.get("jensen_shannon_measure") != "divergence_bits":
            raise ValueError(f"{key!r}: Jensen-Shannon measure is mislabeled")
        divergences = row.get("base2_jensen_shannon_divergences")
        if not isinstance(divergences, Mapping) or set(divergences) != component_keys:
            raise ValueError(f"{key!r}: Jensen-Shannon divergence components are invalid")
        expected_divergences = {
            "prediction_to_label_country": Metrics.js_divergence(
                row["normalized_prediction"],
                row["label_country_human_distribution"],
            ),
            "prediction_to_evidence_country": Metrics.js_divergence(
                row["normalized_prediction"],
                row["evidence_country_human_distribution"],
            ),
            **metrics["components"],
        }
        for name, expected in expected_divergences.items():
            if not math.isclose(
                float(divergences[name]),
                expected,
                rel_tol=0.0,
                abs_tol=METRIC_TOLERANCE,
            ):
                raise ValueError(f"{key!r}: inconsistent JSD component {name}")
        for name in ("country_influence", "evidence_influence", "EO_raw", "EO_normalized"):
            if not math.isclose(
                float(row.get(name)),
                float(metrics[name]),
                rel_tol=0.0,
                abs_tol=METRIC_TOLERANCE,
            ):
                raise ValueError(f"{key!r}: inconsistent metric {name}")
        expected_classification = (
            metrics["classification"] if row["condition"] == "conflict" else None
        )
        if row.get("conflict_classification") != expected_classification:
            raise ValueError(f"{key!r}: conflict classification is inconsistent")
    return validated


def validate_runtime(runtime: Mapping[str, Any]) -> None:
    """Fail unless the loaded backend matches the accepted Llama smoke runtime."""

    scalar_metadata = runtime.get("gguf_scalar_metadata")
    scalar_ok = isinstance(scalar_metadata, Mapping)
    if scalar_ok:
        try:
            scalar_file_type = int(str(scalar_metadata.get("general.file_type")))
            scalar_context = int(str(scalar_metadata.get("llama.context_length")))
        except (TypeError, ValueError):
            scalar_ok = False
        else:
            scalar_ok = (
                scalar_metadata.get("general.architecture")
                == EXPECTED_GGUF_ARCHITECTURE
                and scalar_file_type == EXPECTED_GGUF_FILE_TYPE
                and scalar_context == EXPECTED_GGUF_CONTEXT_LENGTH
                and scalar_metadata.get("tokenizer.chat_template")
                == runtime.get("chat_template")
            )
    checks = {
        "backend": runtime.get("backend") == "llama.cpp",
        "not synthetic": runtime.get("synthetic") is False,
        "model identifier": runtime.get("model_identifier") == MODEL_IDENTIFIER,
        "repository": runtime.get("repository") == MODEL_REPOSITORY,
        "revision": runtime.get("revision") == MODEL_REVISION,
        "quantization": runtime.get("quantization") == QUANTIZATION,
        "embedded template": runtime.get("chat_template_source")
        == "embedded_gguf_metadata",
        "chat template hash": runtime.get("chat_template_sha256")
        == LLAMA_CHAT_TEMPLATE_SHA256,
        "llama.cpp version": runtime.get("llama_version") == LLAMA_CPP_VERSION,
        "CPU only": runtime.get("gpu_layers") == 0,
        "Q4_K_M GGUF file type": runtime.get("gguf_file_type")
        == EXPECTED_GGUF_FILE_TYPE,
        "GGUF scalar identity": scalar_ok,
        "full vocabulary scoring": runtime.get("candidate_scoring")
        == "full_vocabulary_low_level_logits",
        "no generated answer parsing": runtime.get("generated_answer_parsing") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Loaded Llama GGUF runtime violates provenance: {failed!r}")


def validate_helper_linkage(helper_path: Path, llama_cpp_directory: Path) -> str:
    """Require the scorer to resolve llama/ggml libraries from the pinned tree."""

    linked = command_output(["ldd", str(helper_path)])
    if linked is None:
        raise RuntimeError("Could not inspect GGUF helper shared-library linkage")
    expected_directory = (llama_cpp_directory / "build-cpu/bin").resolve()
    required_libraries = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so")
    for library in required_libraries:
        matching_lines = [line for line in linked.splitlines() if library in line]
        if not matching_lines:
            raise RuntimeError(f"GGUF helper is not linked to required {library}")
        resolved_paths = []
        for line in matching_lines:
            if "=>" not in line:
                continue
            candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
            if candidate and candidate != "not":
                resolved_paths.append(Path(candidate).resolve())
        if not resolved_paths or not all(
            path.is_relative_to(expected_directory) for path in resolved_paths
        ):
            raise RuntimeError(
                f"{library} does not resolve from pinned build {expected_directory}"
            )
    return linked


def validate_complete_results(
    rows: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate exact 200x4 coverage, order, row schema, and provenance."""

    manifest_keys = validate_manifest(manifest)
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} rows, found {len(rows)}")
    expected_keys = [
        (*key, condition)
        for key in manifest_keys
        for condition in EXPECTED_OUTPUT_CONDITIONS
    ]
    observed_keys = [
        (*directed_key(row), str(row.get("condition"))) for row in rows
    ]
    if observed_keys != expected_keys:
        raise ValueError("Result keys are incomplete, duplicated, overwritten, or out of order")
    if len(set(observed_keys)) != EXPECTED_ROW_COUNT:
        raise ValueError("Result directed-condition keys are not unique")
    if runtime is not None:
        validate_runtime(runtime)
    fingerprint = None
    if provenance is not None:
        fingerprint = provenance.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Provenance has no run fingerprint")
    validated: list[dict[str, Any]] = []
    for index, pair in enumerate(manifest):
        start = index * len(EXPECTED_OUTPUT_CONDITIONS)
        validated.extend(
            _validate_unit_rows(
                rows[start : start + len(EXPECTED_OUTPUT_CONDITIONS)],
                pair,
                run_fingerprint=fingerprint,
            )
        )

    # Repeated target countries intentionally share the same permutation and
    # non-conflict predictions even when paired with multiple evidence sources.
    target_reference: MutableMapping[tuple[str, str], Mapping[str, Mapping[str, Any]]] = {}
    for index in range(0, len(validated), len(EXPECTED_OUTPUT_CONDITIONS)):
        unit_rows = validated[index : index + len(EXPECTED_OUTPUT_CONDITIONS)]
        by_condition = {row["condition"]: row for row in unit_rows}
        target = (unit_rows[0]["question_id"], unit_rows[0]["label_country"])
        if target not in target_reference:
            target_reference[target] = by_condition
            continue
        reference = target_reference[target]
        for condition in ("baseline", "country_label", "evidence"):
            for field in (
                "raw_user_prompt",
                "serialized_chat_templated_prompt",
                "displayed_options",
                "raw_candidate_scores",
                "normalized_prediction",
                "country_influence",
                "evidence_influence",
            ):
                if by_condition[condition][field] != reference[condition][field]:
                    raise ValueError(
                        f"Repeated target {target!r} changed {condition}.{field}"
                    )
    conflict_counts = Counter(
        row["conflict_classification"]
        for row in validated
        if row["condition"] == "conflict"
    )
    if sum(conflict_counts.values()) != EXPECTED_DIRECTED_UNIT_COUNT:
        raise ValueError("Not every conflict row was classified")
    return {
        "status": "PASS",
        "schema_version": FULL_SCHEMA_VERSION,
        "directed_units": EXPECTED_DIRECTED_UNIT_COUNT,
        "rows": EXPECTED_ROW_COUNT,
        "question_clusters": EXPECTED_QUESTION_CLUSTERS,
        "conditions": list(EXPECTED_OUTPUT_CONDITIONS),
        "unique_directed_condition_keys": len(set(observed_keys)),
        "target_units": len(target_reference),
        "conflict_classification_counts": dict(sorted(conflict_counts.items())),
    }


def build_run_signature(
    repository_root: Path,
    manifest: Sequence[Mapping[str, Any]],
    model_files: Sequence[Mapping[str, Any]],
    helper_path: Path,
    runtime: Mapping[str, Any],
    *,
    llama_cpp_commit: str,
    threads: int,
    context_size: int,
    smoke_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind checkpoints to data, code, model, llama.cpp, and run settings."""

    validate_manifest(manifest)
    identity_paths = (
        "requirements-inference.lock",
        DATASET_PATH,
        PAIR_MANIFEST,
        "src/data_loader.py",
        "src/gguf_runner.py",
        "src/llama_gguf_runner.py",
        "src/gguf_score_helper.cpp",
        "src/main.py",
        "src/metrics.py",
        "src/prompt_builder.py",
        "src/scoring.py",
        "scripts/run_llama_gguf_full.py",
        "scripts/run_genuine_llama_gguf_smoke.py",
    )
    identities: dict[str, str] = {}
    for relative in identity_paths:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Run-signature file is missing: {path}")
        identities[relative] = sha256_file(path)
    core = {
        "schema_version": FULL_SCHEMA_VERSION,
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "remote_last_modified": REMOTE_MODEL_LAST_MODIFIED,
            "weight_upload_commit": REMOTE_WEIGHT_COMMIT,
            "quantization": QUANTIZATION,
            "files": [
                {
                    "filename": item["filename"],
                    "size_bytes": item["actual_size_bytes"],
                    "sha256": item["actual_sha256"],
                }
                for item in model_files
            ],
        },
        "llama_cpp": {
            "commit": llama_cpp_commit,
            "version": runtime.get("llama_version"),
            "chat_template_sha256": runtime.get("chat_template_sha256"),
            "helper_sha256": sha256_file(helper_path),
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
        "threads": int(threads),
        "context_size": int(context_size),
        "jensen_shannon_base": 2,
        "genuine_smoke_gate": dict(smoke_gate),
        "code_data_dependency_sha256": identities,
    }
    return {**core, "fingerprint": canonical_json_hash(core)}


def checkpoint_path(checkpoint_directory: Path, key: Sequence[str]) -> Path:
    digest = hashlib.sha256(
        (MODEL_IDENTIFIER + "\0" + "\0".join(key)).encode("utf-8")
    ).hexdigest()[:24]
    return checkpoint_directory / f"unit_{digest}.json"


def validate_checkpoint(
    checkpoint: Any,
    pair: Mapping[str, Any],
    manifest_index: int,
    signature: Mapping[str, Any],
) -> list[dict[str, Any]]:
    key = directed_key(pair)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint for {key!r} must be an object")
    if checkpoint.get("schema_version") != FULL_SCHEMA_VERSION:
        raise ValueError(f"Checkpoint for {key!r} has the wrong schema")
    if checkpoint.get("run_fingerprint") != signature.get("fingerprint"):
        raise ValueError(f"Checkpoint run fingerprint mismatch for {key!r}")
    if checkpoint.get("manifest_index") != manifest_index:
        raise ValueError(f"Checkpoint manifest order mismatch for {key!r}")
    if checkpoint.get("directed_key") != list(key):
        raise ValueError(f"Checkpoint directed key mismatch for {key!r}")
    if checkpoint.get("complete") is not True:
        raise ValueError(f"Checkpoint for {key!r} is not marked complete")
    if checkpoint.get("conditions") != list(EXPECTED_OUTPUT_CONDITIONS):
        raise ValueError(f"Checkpoint condition order mismatch for {key!r}")
    rows = checkpoint.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_OUTPUT_CONDITIONS):
        raise ValueError(f"Checkpoint for {key!r} is not a complete unit")
    return _validate_unit_rows(
        rows,
        pair,
        run_fingerprint=str(signature["fingerprint"]),
    )


def validate_partial_checkpoint(
    checkpoint: Any,
    pair: Mapping[str, Any],
    manifest_index: int,
    signature: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate a safely persisted one-to-four-row unfinalized prefix."""

    key = directed_key(pair)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Partial checkpoint for {key!r} must be an object")
    required = {
        "schema_version": FULL_SCHEMA_VERSION,
        "run_fingerprint": signature.get("fingerprint"),
        "manifest_index": manifest_index,
        "directed_key": list(key),
        "complete": False,
    }
    for field, expected in required.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"Partial checkpoint for {key!r} has invalid {field}")
    rows = checkpoint.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= len(EXPECTED_OUTPUT_CONDITIONS):
        raise ValueError(f"Partial checkpoint for {key!r} must contain one to four rows")
    expected_conditions = list(EXPECTED_OUTPUT_CONDITIONS[: len(rows)])
    if checkpoint.get("conditions_completed") != expected_conditions:
        raise ValueError(f"Partial checkpoint condition prefix is invalid for {key!r}")

    canonical = [str(option) for option in pair["options"]]
    target_unit_id = f"{key[0]}::{key[1]}"
    expected_displayed = shuffle_options(
        canonical, stable_unit_seed(DEFAULT_SEED, target_unit_id)
    )
    labels = option_labels(len(expected_displayed))
    label_map = dict(zip(labels, expected_displayed))
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        context = f"partial {key!r}/{expected_conditions[index]}"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context}: row must be an object")
        row = dict(raw)
        if directed_key(row) != key or row.get("condition") != expected_conditions[index]:
            raise ValueError(f"{context}: directed-condition key is invalid")
        for field, expected in {
            "schema_version": FULL_SCHEMA_VERSION,
            "run_fingerprint": signature.get("fingerprint"),
            "model_name": MODEL_NAME,
            "model_identifier": MODEL_IDENTIFIER,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "backend": "llama.cpp",
            "synthetic": False,
            "original_options": canonical,
            "displayed_options": expected_displayed,
            "displayed_option_labels": labels,
            "displayed_label_to_option": label_map,
            "generated_answer": None,
        }.items():
            if row.get(field) != expected:
                raise ValueError(f"{context}: invalid {field}")
        label_probabilities = _normalize_probability_mapping(
            row.get("normalized_label_probabilities"), labels, f"{context}.labels"
        )
        prediction = _normalize_probability_mapping(
            row.get("normalized_prediction"), canonical, f"{context}.prediction"
        )
        displayed_prediction = map_label_probabilities(
            labels,
            expected_displayed,
            [label_probabilities[label] for label in labels],
        )
        if not _same_distribution(
            recover_canonical_distribution(displayed_prediction, canonical),
            prediction,
        ):
            raise ValueError(f"{context}: option permutation recovery failed")
        evidence_presented = expected_conditions[index] in {"evidence", "conflict"}
        if row.get("evidence_presented") is not evidence_presented:
            raise ValueError(f"{context}: evidence-presented flag is invalid")
        if not evidence_presented and (
            row.get("presented_evidence_distribution") is not None
            or row.get("source_evidence_distribution") is not None
        ):
            raise ValueError(f"{context}: no-evidence row stores evidence")
        scoring = row.get("scoring_trace")
        if not isinstance(scoring, Mapping):
            raise ValueError(f"{context}: scoring trace is missing")
        if (
            scoring.get("prompt_bos_added") is not True
            or scoring.get("prompt_bos_token_id") != scoring.get("bos_token_id")
            or scoring.get("embedded_template_serialization_verified") is not True
            or scoring.get("prompt_prefix_verified") is not True
            or scoring.get("generated_answer") is not None
        ):
            raise ValueError(f"{context}: genuine answer-position scoring gate failed")
        candidates = scoring.get("candidates")
        prompt_count = scoring.get("prompt_token_count")
        if (
            not isinstance(candidates, Mapping)
            or set(candidates) != set(labels)
            or not isinstance(prompt_count, int)
            or prompt_count <= 0
        ):
            raise ValueError(f"{context}: candidate scoring trace is invalid")
        for label in labels:
            candidate = candidates[label]
            token_ids = candidate.get("token_ids") if isinstance(candidate, Mapping) else None
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(f"{context}/{label}: tokenization is invalid")
            expected_targets = list(range(prompt_count, prompt_count + len(token_ids)))
            if (
                candidate.get("target_token_positions") != expected_targets
                or candidate.get("predictive_logit_positions")
                != [position - 1 for position in expected_targets]
            ):
                raise ValueError(f"{context}/{label}: answer positions are invalid")
            token_logps = candidate.get("token_log_probabilities")
            if (
                not isinstance(token_logps, list)
                or len(token_logps) != len(token_ids)
                or not math.isclose(
                    math.fsum(float(value) for value in token_logps),
                    float(candidate.get("sequence_log_probability")),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(f"{context}/{label}: incomplete sequence score")
        validated.append(row)
    return validated


def validate_smoke_gate(
    analysis_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Fail closed unless the exact genuine eight-row Llama smoke passed."""

    analysis_path = analysis_path.resolve()
    results_path = results_path.resolve()
    if not analysis_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            "A PASS genuine-smoke analysis and its eight-row JSONL are required"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not isinstance(analysis, Mapping) or analysis.get("status") != "PASS":
        raise ValueError("Genuine-smoke analysis status is not PASS")
    scope = analysis.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(field) != expected
        for field, expected in {
            "models": 1,
            "directed_units": 2,
            "conditions_per_unit": 4,
            "saved_rows": 8,
        }.items()
    ):
        raise ValueError("Genuine-smoke scope is not exactly one model/eight rows")
    assertions = analysis.get("assertions")
    if not isinstance(assertions, Mapping) or set(assertions) != REQUIRED_SMOKE_ASSERTIONS:
        raise ValueError("Genuine-smoke assertion set is incomplete or unexpected")
    if any(value is not True for value in assertions.values()):
        raise ValueError("Not every genuine-smoke assertion passed")
    if analysis.get("result_key_count") != 8 or analysis.get("row_assertion_count") != 8:
        raise ValueError("Genuine-smoke analysis does not attest eight unique keys")
    metadata = analysis.get("metadata")
    model = metadata.get("model") if isinstance(metadata, Mapping) else None
    llama_cpp = metadata.get("llama_cpp") if isinstance(metadata, Mapping) else None
    if not isinstance(model, Mapping) or any(
        model.get(field) != expected
        for field, expected in {
            "identifier": MODEL_IDENTIFIER,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "quantization": QUANTIZATION,
        }.items()
    ):
        raise ValueError("Genuine-smoke model identity does not match the full run")
    if not isinstance(llama_cpp, Mapping) or any(
        llama_cpp.get(field) != expected
        for field, expected in {
            "commit": LLAMA_CPP_COMMIT,
            "chat_template_sha256": LLAMA_CHAT_TEMPLATE_SHA256,
        }.items()
    ):
        raise ValueError("Genuine-smoke llama.cpp/template identity does not match")

    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Smoke result line {line_number} is not an object")
            rows.append(dict(value))
    if len(rows) != 8:
        raise ValueError(f"Genuine-smoke JSONL has {len(rows)} rows instead of 8")
    observed_keys: list[tuple[str, str, str, str]] = []
    conditions_by_unit: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        key = directed_key(row)
        condition = str(row.get("condition"))
        observed_keys.append((*key, condition))
        conditions_by_unit.setdefault(key, []).append(condition)
        if any(
            row.get(field) != expected
            for field, expected in {
                "model_identifier": MODEL_IDENTIFIER,
                "model_repository": MODEL_REPOSITORY,
                "model_revision": MODEL_REVISION,
                "backend": "llama.cpp",
                "synthetic": False,
            }.items()
        ):
            raise ValueError("Genuine-smoke result contains model/backend substitution")
        scoring = row.get("scoring_trace")
        if not isinstance(scoring, Mapping) or any(
            scoring.get(field) is not True
            for field in (
                "embedded_template_serialization_verified",
                "prompt_prefix_verified",
                "prompt_bos_added",
            )
        ):
            raise ValueError("Genuine-smoke row lacks genuine answer-position attestation")
    if len(set(observed_keys)) != 8 or set(conditions_by_unit) != EXPECTED_SMOKE_UNITS:
        raise ValueError("Genuine-smoke reciprocal directed keys are invalid")
    if any(
        tuple(conditions) != EXPECTED_OUTPUT_CONDITIONS
        for conditions in conditions_by_unit.values()
    ):
        raise ValueError("Genuine-smoke conditions are incomplete or out of order")
    return {
        "status": "PASS",
        "analysis_sha256": sha256_file(analysis_path),
        "results_sha256": sha256_file(results_path),
        "result_rows": 8,
        "assertion_count": len(assertions),
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "chat_template_sha256": LLAMA_CHAT_TEMPLATE_SHA256,
    }


def _write_progress(
    path: Path,
    *,
    status: str,
    signature: Mapping[str, Any] | None,
    completed_units: int,
    resumed_units: int,
    newly_completed_units: int,
    last_key: Sequence[str] | None,
    started_at: str,
    error: str | None = None,
    completed_rows: int | None = None,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": FULL_SCHEMA_VERSION,
            "status": status,
            "updated_at_utc": now_utc(),
            "started_at_utc": started_at,
            "run_fingerprint": signature.get("fingerprint") if signature else None,
            "expected_directed_units": EXPECTED_DIRECTED_UNIT_COUNT,
            "expected_rows": EXPECTED_ROW_COUNT,
            "completed_directed_units": completed_units,
            "completed_rows": (
                completed_units * len(EXPECTED_OUTPUT_CONDITIONS)
                if completed_rows is None
                else completed_rows
            ),
            "resumed_units": resumed_units,
            "newly_completed_units": newly_completed_units,
            "last_completed_directed_key": list(last_key) if last_key else None,
            "error": error,
        },
    )


def run_llama_full(
    *,
    model_directory: Path,
    helper_path: Path,
    llama_cpp_directory: Path,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    threads: int = min(8, os.cpu_count() or 1),
    context_size: int = 4096,
    allow_llama_full_run: bool = False,
    smoke_analysis_path: Path | None = None,
    smoke_results_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run or resume exactly one Llama model over the 200-unit manifest."""

    if allow_llama_full_run is not True:
        raise PermissionError(
            "The Llama-only full run requires the explicit --allow-llama-full-run flag"
        )
    if smoke_analysis_path is None or smoke_results_path is None:
        raise PermissionError(
            "The full run requires --smoke-analysis and --smoke-results PASS artifacts"
        )
    smoke_gate = validate_smoke_gate(smoke_analysis_path, smoke_results_path)
    if threads <= 0 or context_size <= 0:
        raise ValueError("threads and context_size must be positive")
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "results.jsonl"
    run_log_path = output_directory / "run.log"
    progress_path = output_directory / "progress.json"
    signature_path = output_directory / "run_signature.json"
    runtime_path = output_directory / "runtime.json"
    environment_path = output_directory / "environment.json"
    checkpoint_directory = output_directory / "checkpoints"
    started_at = now_utc()
    started_clock = time.monotonic()
    signature: dict[str, Any] | None = None
    completed: MutableMapping[tuple[str, str, str], list[dict[str, Any]]] = {}
    resumed_units = 0
    newly_completed_units = 0
    partial_resumed_rows = 0
    persisted_row_count = 0
    last_key: tuple[str, str, str] | None = None

    _write_progress(
        progress_path,
        status="preflight",
        signature=None,
        completed_units=0,
        resumed_units=0,
        newly_completed_units=0,
        last_key=None,
        started_at=started_at,
    )
    with run_log_path.open("a", encoding="utf-8") as run_log:
        _log(run_log, "BEGIN approved Llama-only GGUF full run/resume")
        _log(
            run_log,
            "Scope: one pinned model, 200 reciprocal directed units, four conditions",
        )
        runner: Llama31GGUFRunner | None = None
        try:
            environment = environment_information(PROJECT_ROOT)
            atomic_write_json(environment_path, environment)
            _log(run_log, f"Environment: {json.dumps(environment, ensure_ascii=False)}")
            model_files = verified_model_files(model_directory.resolve())
            _log(run_log, "Official GGUF shard size/SHA-256 verification: PASS")
            llama_cpp_directory = llama_cpp_directory.resolve()
            observed_commit = command_output(
                ["git", "-C", str(llama_cpp_directory), "rev-parse", "HEAD"]
            )
            if observed_commit != LLAMA_CPP_COMMIT:
                raise RuntimeError(
                    f"llama.cpp commit mismatch: {observed_commit!r} != {LLAMA_CPP_COMMIT!r}"
                )
            tracked_changes = command_output(
                [
                    "git",
                    "-C",
                    str(llama_cpp_directory),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ]
            )
            if tracked_changes != "":
                raise RuntimeError("Pinned llama.cpp checkout has tracked modifications")
            helper_path = helper_path.resolve()
            if not helper_path.is_file():
                raise FileNotFoundError(f"GGUF scoring helper is missing: {helper_path}")
            helper_linkage = validate_helper_linkage(helper_path, llama_cpp_directory)
            accepted_source_hashes = {
                "src/gguf_runner.py": ACCEPTED_GGUF_RUNNER_SHA256,
                "src/llama_gguf_runner.py": LLAMA_GGUF_RUNNER_SHA256,
                "src/gguf_score_helper.cpp": ACCEPTED_GGUF_HELPER_SOURCE_SHA256,
            }
            for relative_path, expected_hash in accepted_source_hashes.items():
                actual_hash = sha256_file(PROJECT_ROOT / relative_path)
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"Accepted GGUF scoring source drifted: {relative_path} "
                        f"{actual_hash} != {expected_hash}"
                    )

            loader = DataLoader(str(PROJECT_ROOT / DATASET_PATH))
            dataset_hash = sha256_file(PROJECT_ROOT / DATASET_PATH)
            manifest_hash = sha256_file(PROJECT_ROOT / PAIR_MANIFEST)
            if dataset_hash != DATASET_SHA256:
                raise RuntimeError(
                    f"Cleaned dataset SHA-256 mismatch: {dataset_hash} != {DATASET_SHA256}"
                )
            if manifest_hash != PAIR_MANIFEST_SHA256:
                raise RuntimeError(
                    "Cleaned pair-manifest SHA-256 mismatch: "
                    f"{manifest_hash} != {PAIR_MANIFEST_SHA256}"
                )
            loader.load_dataset()
            manifest = loader.get_question_pairs(str(PROJECT_ROOT / PAIR_MANIFEST))
            validate_manifest(manifest)
            _log(
                run_log,
                "Cleaned data validation: pinned hashes, 44 question clusters, "
                "144 targets, 200 unique reciprocal directed units PASS",
            )

            runner = Llama31GGUFRunner(
                helper_path=helper_path,
                model_path=Path(model_files[0]["local_path"]),
                model_identifier=MODEL_IDENTIFIER,
                repository=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                quantization=QUANTIZATION,
                threads=threads,
                context_size=context_size,
            )
            runner.load_model(run_log)
            runtime = runner.get_runtime_metadata()
            validate_runtime(runtime)
            signature = build_run_signature(
                PROJECT_ROOT,
                manifest,
                model_files,
                helper_path,
                runtime,
                llama_cpp_commit=observed_commit,
                threads=threads,
                context_size=context_size,
                smoke_gate=smoke_gate,
            )
            if signature_path.is_file():
                previous = json.loads(signature_path.read_text(encoding="utf-8"))
                if previous.get("fingerprint") != signature["fingerprint"]:
                    raise ValueError(
                        "Existing checkpoints belong to different code, data, model, "
                        "llama.cpp, template, or run settings; use a new output directory"
                    )
            else:
                if checkpoint_directory.is_dir() and any(checkpoint_directory.iterdir()):
                    raise ValueError("Checkpoint files exist without a run signature")
                atomic_write_json(signature_path, signature)
            if runtime_path.is_file():
                previous_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                if previous_runtime.get("run_fingerprint") != signature["fingerprint"]:
                    raise ValueError("Existing runtime record has a different fingerprint")
                if previous_runtime.get("runtime") != runtime:
                    raise ValueError("Loaded runtime metadata differs from the recorded resume")
            else:
                atomic_write_json(
                    runtime_path,
                    {
                        "recorded_at_utc": now_utc(),
                        "run_fingerprint": signature["fingerprint"],
                        "runtime": runtime,
                        "model_files": model_files,
                        "remote_model_last_modified": REMOTE_MODEL_LAST_MODIFIED,
                        "remote_weight_commit": REMOTE_WEIGHT_COMMIT,
                        "llama_cpp_commit": observed_commit,
                        "helper_ldd": helper_linkage,
                        "genuine_smoke_gate": smoke_gate,
                    },
                )
            _log(
                run_log,
                f"Loaded {MODEL_IDENTIFIER}; fingerprint={signature['fingerprint']}",
            )

            # Validate every existing per-unit checkpoint before any new
            # inference.  Checkpoint files are the resume source of truth;
            # the aggregate JSONL is only a regenerable inspection view.
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            expected_checkpoint_paths = {
                checkpoint_path(checkpoint_directory, directed_key(pair)).resolve()
                for pair in manifest
            }
            unexpected_checkpoints = [
                path
                for path in checkpoint_directory.glob("*.json")
                if path.resolve() not in expected_checkpoint_paths
            ]
            if unexpected_checkpoints:
                raise ValueError(
                    "Unexpected checkpoint files are present: "
                    + ", ".join(path.name for path in unexpected_checkpoints[:3])
                )
            prevalidated_checkpoints: dict[
                tuple[str, str, str], list[dict[str, Any]]
            ] = {}
            partial_checkpoints: dict[
                tuple[str, str, str], list[dict[str, Any]]
            ] = {}
            for manifest_index, pair in enumerate(manifest):
                key = directed_key(pair)
                path = checkpoint_path(checkpoint_directory, key)
                if not path.is_file():
                    continue
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                rows_in_checkpoint = checkpoint.get("rows")
                if (
                    checkpoint.get("complete") is True
                    and isinstance(rows_in_checkpoint, list)
                    and len(rows_in_checkpoint) == len(EXPECTED_OUTPUT_CONDITIONS)
                ):
                    prevalidated_checkpoints[key] = validate_checkpoint(
                        checkpoint, pair, manifest_index, signature
                    )
                else:
                    partial_checkpoints[key] = validate_partial_checkpoint(
                        checkpoint, pair, manifest_index, signature
                    )
            complete_indices = [
                index
                for index, pair in enumerate(manifest)
                if directed_key(pair) in prevalidated_checkpoints
            ]
            if complete_indices != list(range(len(complete_indices))):
                raise ValueError("Complete checkpoints are not a contiguous manifest prefix")
            if len(partial_checkpoints) > 1:
                raise ValueError("More than one partial checkpoint is present")
            if partial_checkpoints:
                partial_key = next(iter(partial_checkpoints))
                expected_partial_index = len(complete_indices)
                if (
                    expected_partial_index >= len(manifest)
                    or directed_key(manifest[expected_partial_index]) != partial_key
                ):
                    raise ValueError("Partial checkpoint is not immediately after complete prefix")
            persisted_row_count = (
                len(prevalidated_checkpoints) * len(EXPECTED_OUTPUT_CONDITIONS)
                + sum(len(rows) for rows in partial_checkpoints.values())
            )
            partial_resumed_rows = sum(len(rows) for rows in partial_checkpoints.values())
            _log(
                run_log,
                f"Prevalidated {len(prevalidated_checkpoints)} complete checkpoints "
                f"and {partial_resumed_rows} partial rows before inference",
            )

            for manifest_index, pair in enumerate(manifest):
                key = directed_key(pair)
                path = checkpoint_path(checkpoint_directory, key)
                if key in prevalidated_checkpoints:
                    unit_rows = prevalidated_checkpoints[key]
                    resumed_units += 1
                    action = "resumed"
                else:
                    existing_rows = partial_checkpoints.get(key, [])
                    _log(
                        run_log,
                        f"Scoring unit {manifest_index + 1}/{EXPECTED_DIRECTED_UNIT_COUNT}: "
                        f"{key[0]}::{key[1]}=>{key[2]}",
                    )

                    def persist_partial(current_rows: list[dict[str, Any]]) -> None:
                        nonlocal persisted_row_count
                        checkpoint = {
                            "schema_version": FULL_SCHEMA_VERSION,
                            "updated_at_utc": now_utc(),
                            "run_fingerprint": signature["fingerprint"],
                            "manifest_index": manifest_index,
                            "directed_key": list(key),
                            "complete": False,
                            "conditions_completed": [
                                row["condition"] for row in current_rows
                            ],
                            "rows": current_rows,
                        }
                        atomic_write_json(path, checkpoint)
                        prior_rows = [
                            row
                            for prior_pair in manifest[:manifest_index]
                            for row in completed[directed_key(prior_pair)]
                        ]
                        atomic_write_jsonl(results_path, [*prior_rows, *current_rows])
                        persisted_row_count = manifest_index * len(
                            EXPECTED_OUTPUT_CONDITIONS
                        ) + len(current_rows)
                        _write_progress(
                            progress_path,
                            status="running",
                            signature=signature,
                            completed_units=manifest_index,
                            completed_rows=persisted_row_count,
                            resumed_units=resumed_units,
                            newly_completed_units=newly_completed_units,
                            last_key=key,
                            started_at=started_at,
                        )
                        _log(
                            run_log,
                            f"Saved row {persisted_row_count}/{EXPECTED_ROW_COUNT}: "
                            f"{key[0]}::{key[1]}=>{key[2]} "
                            f"condition={current_rows[-1]['condition']}",
                        )

                    unit_rows = run_directed_unit(
                        runner,
                        loader,
                        pair,
                        model_files,
                        str(signature["fingerprint"]),
                        existing_rows=existing_rows,
                        on_row_saved=persist_partial,
                    )
                    unit_rows = _validate_unit_rows(
                        unit_rows,
                        pair,
                        run_fingerprint=str(signature["fingerprint"]),
                    )
                    checkpoint = {
                        "schema_version": FULL_SCHEMA_VERSION,
                        "completed_at_utc": now_utc(),
                        "run_fingerprint": signature["fingerprint"],
                        "manifest_index": manifest_index,
                        "directed_key": list(key),
                        "complete": True,
                        "conditions": list(EXPECTED_OUTPUT_CONDITIONS),
                        "rows": unit_rows,
                    }
                    # Replace the row-granular prefix with its validated,
                    # metric-enriched four-row unit.
                    atomic_write_json(path, checkpoint)
                    newly_completed_units += 1
                    action = "new"
                completed[key] = unit_rows
                last_key = key
                ordered_rows = [
                    row
                    for prior_pair in manifest[: manifest_index + 1]
                    for row in completed[directed_key(prior_pair)]
                ]
                atomic_write_jsonl(results_path, ordered_rows)
                persisted_row_count = len(ordered_rows)
                _write_progress(
                    progress_path,
                    status="running",
                    signature=signature,
                    completed_units=manifest_index + 1,
                    resumed_units=resumed_units,
                    newly_completed_units=newly_completed_units,
                    last_key=last_key,
                    started_at=started_at,
                    completed_rows=persisted_row_count,
                )
                _log(
                    run_log,
                    f"Checkpoint {action}: unit {manifest_index + 1}/200; "
                    f"rows={(manifest_index + 1) * 4}/800",
                )

            rows = [row for pair in manifest for row in completed[directed_key(pair)]]
            validation = validate_complete_results(
                rows,
                manifest,
                runtime=runtime,
                provenance=signature,
            )
            atomic_write_jsonl(results_path, rows)
            completion = {
                **validation,
                "completed_at_utc": now_utc(),
                "elapsed_seconds": time.monotonic() - started_clock,
                "run_fingerprint": signature["fingerprint"],
                "model_identifier": MODEL_IDENTIFIER,
                "model_revision": MODEL_REVISION,
                "llama_cpp_commit": observed_commit,
                "results_path": str(results_path),
                "resumed_units": resumed_units,
                "newly_completed_units": newly_completed_units,
                "partial_resumed_rows": partial_resumed_rows,
                "genuine_smoke_gate": smoke_gate,
            }
            atomic_write_json(output_directory / "completion.json", completion)
            _write_progress(
                progress_path,
                status="complete",
                signature=signature,
                completed_units=EXPECTED_DIRECTED_UNIT_COUNT,
                resumed_units=resumed_units,
                newly_completed_units=newly_completed_units,
                last_key=last_key,
                started_at=started_at,
                completed_rows=EXPECTED_ROW_COUNT,
            )
            _log(run_log, "PASS: exactly 800 validated Llama result rows saved")
            _log(run_log, "END approved Llama-only GGUF full run/resume")
            return rows, completion
        except Exception as exc:
            _write_progress(
                progress_path,
                status="failed",
                signature=signature,
                completed_units=len(completed),
                resumed_units=resumed_units,
                newly_completed_units=newly_completed_units,
                last_key=last_key,
                started_at=started_at,
                completed_rows=persisted_row_count,
                error=f"{type(exc).__name__}: {exc}",
            )
            _log(run_log, f"FAILED: {type(exc).__name__}: {exc}")
            raise
        finally:
            if runner is not None:
                runner.close()
                _log(run_log, "Model released; llama.cpp helper stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument(
        "--gguf-revision",
        default=os.environ.get("LLAMA_GGUF_REVISION"),
    )
    parser.add_argument(
        "--upstream-revision",
        default=os.environ.get("LLAMA_UPSTREAM_REVISION"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument(
        "--smoke-analysis",
        type=Path,
        required=True,
        help="PASS analysis.json from the exact genuine eight-row Llama smoke",
    )
    parser.add_argument(
        "--smoke-results",
        type=Path,
        required=True,
        help="Eight-row results.jsonl from the exact genuine Llama smoke",
    )
    parser.add_argument(
        "--allow-llama-full-run",
        action="store_true",
        help="Explicit gate for this approved Llama-only 200-unit run",
    )
    args = parser.parse_args()
    require_pinned_revision(args.gguf_revision, MODEL_REVISION, "GGUF revision")
    require_pinned_revision(
        args.upstream_revision,
        UPSTREAM_CURRENT_REVISION,
        "upstream checkpoint revision",
    )
    _, completion = run_llama_full(
        model_directory=args.model_dir,
        helper_path=args.helper,
        llama_cpp_directory=args.llama_cpp_dir,
        output_directory=args.output_dir,
        threads=args.threads,
        context_size=args.ctx_size,
        allow_llama_full_run=args.allow_llama_full_run,
        smoke_analysis_path=args.smoke_analysis,
        smoke_results_path=args.smoke_results,
    )
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_OUTPUT_CONDITIONS",
    "EXPECTED_ROW_COUNT",
    "FULL_SCHEMA_VERSION",
    "build_run_signature",
    "checkpoint_path",
    "classify_conflict",
    "prepare_unit",
    "run_directed_unit",
    "run_llama_full",
    "validate_checkpoint",
    "validate_partial_checkpoint",
    "validate_smoke_gate",
    "validate_complete_results",
    "validate_manifest",
    "validate_runtime",
]

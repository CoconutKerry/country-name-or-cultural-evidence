#!/usr/bin/env python3
"""Strictly validate and analyze repaired cultural-alignment result files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import Metrics
from src.result_utils import group_results_by_directed_unit
from src.scoring import (
    map_label_probabilities,
    normalize_log_scores,
    option_labels,
    recover_canonical_distribution,
)


REPAIRED_SCORING = "complete_option_label_continuation_log_likelihood"
REQUIRED_CONDITIONS = ("baseline", "country_label", "population_evidence", "conflict")
PROBABILITY_TOLERANCE = 1e-8
# The protected Colab handoff predates the explicit-null evidence schema and
# cannot be rewritten without changing its full-run approval gate.  Its rows
# are accepted only when they carry this exact historical schema identifier.
LEGACY_REFERENCE_EVIDENCE_SCHEMAS = frozenset({"colab-gpu-v1"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
        )
    except (TypeError, ValueError):
        return False


def equal_distribution(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(close_number(left[key], right[key]) for key in left)


def validate_distribution(value: Any, options: Sequence[str], context: str) -> None:
    require(isinstance(value, Mapping), f"{context} must be an object")
    require(set(value) == set(options), f"{context} keys must match canonical options")
    probabilities = []
    for option in options:
        probability = value[option]
        require(
            isinstance(probability, (int, float)) and not isinstance(probability, bool),
            f"{context}[{option!r}] must be numeric",
        )
        number = float(probability)
        require(math.isfinite(number) and number >= 0.0, f"{context} has invalid probability")
        probabilities.append(number)
    require(
        math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE),
        f"{context} probabilities must sum to 1",
    )


def validate_repaired_metadata(metadata: Mapping[str, Any], allow_synthetic: bool) -> None:
    require(isinstance(metadata, Mapping), "metadata must be an object")
    require(metadata.get("scoring_method") == REPAIRED_SCORING, "wrong scoring method")
    require(
        metadata.get("directed_unit_key") == ["question_id", "country", "conflict_country"],
        "wrong directed-unit key",
    )
    require(metadata.get("jensen_shannon_base") == 2, "JSD base must be 2")
    synthetic = metadata.get("synthetic_backend")
    scientific = metadata.get("scientific_inference")
    require(isinstance(synthetic, bool) and isinstance(scientific, bool), "inference flags must be bool")
    require(scientific is (not synthetic), "synthetic/scientific flags are inconsistent")
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic smoke outputs are not scientific results")
    require(
        metadata.get("conditions") == list(REQUIRED_CONDITIONS),
        "metadata must declare exactly four conditions",
    )
    models = metadata.get("models")
    require(
        isinstance(models, list)
        and models
        and all(isinstance(model, str) and model for model in models)
        and len(models) == len(set(models)),
        "metadata.models must contain unique non-empty names",
    )
    if scientific:
        lock = metadata.get("package_lock", {})
        require(lock.get("status") == "present" and lock.get("sha256"), "package lock identity missing")
        data_artifacts = metadata.get("data_artifacts")
        require(
            isinstance(data_artifacts, Mapping)
            and set(data_artifacts)
            == {"dataset", "pair_manifest", "source_dataset", "repair_summary"},
            "data artifact identities missing",
        )
        for name, identity in data_artifacts.items():
            require(
                isinstance(identity, Mapping)
                and identity.get("status") == "present"
                and identity.get("sha256"),
                f"{name}: data artifact identity is incomplete",
            )
        runtimes = metadata.get("model_runtime")
        require(isinstance(runtimes, Mapping) and set(runtimes) == set(models), "model runtime metadata missing")
        for model in models:
            runtime = runtimes[model]
            require(runtime.get("use_chat_template") is True, f"{model}: chat template not used")
            require(runtime.get("chat_template_sha256"), f"{model}: chat template hash missing")
            require(
                runtime.get("requested_model_revision") == runtime.get("resolved_model_revision"),
                f"{model}: model revision was not resolved exactly",
            )
            require(
                runtime.get("requested_tokenizer_revision")
                == runtime.get("resolved_tokenizer_revision"),
                f"{model}: tokenizer revision was not resolved exactly",
            )


def validate_row(record: Mapping[str, Any], metadata: Mapping[str, Any], index: int) -> None:
    context = f"results[{index}]"
    for field in ("model_name", "question_id", "country", "conflict_country", "condition", "unit_id"):
        require(isinstance(record.get(field), str) and record[field], f"{context}.{field} is required")
    question_id = record["question_id"]
    country = record["country"]
    conflict_country = record["conflict_country"]
    require(country != conflict_country, f"{context} is a self-pair")
    require(
        record["unit_id"] == f"{question_id}::{country}=>{conflict_country}",
        f"{context}.unit_id is inconsistent with its directed triple",
    )
    require(
        record.get("target_unit_id") == f"{question_id}::{country}",
        f"{context}.target_unit_id is inconsistent",
    )
    require(record["condition"] in REQUIRED_CONDITIONS, f"{context}.condition is unexpected")

    canonical = record.get("original_options")
    used = record.get("used_options")
    require(
        isinstance(canonical, list)
        and len(canonical) >= 2
        and all(isinstance(option, str) and option for option in canonical)
        and len(canonical) == len(set(canonical)),
        f"{context}.original_options is invalid",
    )
    require(
        isinstance(used, list) and len(used) == len(canonical) and set(used) == set(canonical),
        f"{context}.used_options must be an exact permutation",
    )
    labels = option_labels(len(used))
    expected_map = dict(zip(labels, used))
    require(record.get("option_label_map") == expected_map, f"{context}.option_label_map is invalid")
    scoring = record.get("scoring")
    require(isinstance(scoring, Mapping), f"{context}.scoring must be an object")
    require(scoring.get("label_to_option") == expected_map, f"{context}.scoring label map is invalid")
    require(
        scoring.get("synthetic") is metadata.get("synthetic_backend"),
        f"{context}.scoring synthetic flag is inconsistent",
    )
    label_probabilities = scoring.get("label_probabilities")
    validate_distribution(label_probabilities, labels, f"{context}.scoring.label_probabilities")
    label_log_scores = scoring.get("label_log_scores")
    require(
        isinstance(label_log_scores, Mapping)
        and set(label_log_scores) == set(labels)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in label_log_scores.values()
        ),
        f"{context}.scoring.label_log_scores is invalid",
    )
    temperature = (
        metadata.get("config", {}).get("inference", {}).get("temperature", 0.0)
        if isinstance(metadata.get("config", {}), Mapping)
        else 0.0
    )
    probabilities_from_logs = normalize_log_scores(
        [label_log_scores[label] for label in labels],
        temperature=float(temperature),
    )
    require(
        all(
            close_number(label_probabilities[label], probability)
            for label, probability in zip(labels, probabilities_from_logs)
        ),
        f"{context}: label log scores do not normalize to label probabilities",
    )
    rendered_prompt = scoring.get("rendered_prompt")
    require(isinstance(rendered_prompt, str) and rendered_prompt, f"{context}: rendered prompt missing")
    require(
        scoring.get("raw_prompt_sha256")
        == hashlib.sha256(record.get("prompt", "").encode("utf-8")).hexdigest(),
        f"{context}: raw prompt hash mismatch",
    )
    require(
        scoring.get("rendered_prompt_sha256")
        == hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        f"{context}: rendered prompt hash mismatch",
    )
    for field in (
        "prediction",
        "human_distribution",
        "label_country_human_distribution",
        "evidence_country_human_distribution",
    ):
        validate_distribution(record.get(field), canonical, f"{context}.{field}")
    require(
        equal_distribution(
            record["human_distribution"],
            record["label_country_human_distribution"],
        ),
        f"{context}.human_distribution must equal the label-country reference",
    )
    for field in (
        "presented_evidence_distribution",
        "source_evidence_distribution",
    ):
        require(field in record, f"{context}.{field} must be explicitly present")
    evidence_presented = record["condition"] in {"population_evidence", "conflict"}
    require(
        record.get("evidence_presented") is evidence_presented,
        f"{context}.evidence_presented is inconsistent",
    )
    presented_evidence = record["presented_evidence_distribution"]
    source_evidence = record["source_evidence_distribution"]
    legacy_reference_evidence = (
        record.get("schema_version") in LEGACY_REFERENCE_EVIDENCE_SCHEMAS
    )
    if evidence_presented:
        validate_distribution(
            presented_evidence,
            canonical,
            f"{context}.presented_evidence_distribution",
        )
        validate_distribution(
            source_evidence,
            canonical,
            f"{context}.source_evidence_distribution",
        )
        expected_evidence = (
            record["label_country_human_distribution"]
            if record["condition"] == "population_evidence"
            else record["evidence_country_human_distribution"]
        )
        require(
            equal_distribution(presented_evidence, expected_evidence),
            f"{context}.presented_evidence_distribution does not match its source",
        )
        require(
            equal_distribution(source_evidence, expected_evidence),
            f"{context}.source_evidence_distribution does not match its source",
        )
    elif legacy_reference_evidence:
        # Compatibility-only behavior for the protected Colab v1 handoff.  It
        # stored the label-country reference in these fields even though its
        # evidence_presented flag correctly said no evidence appeared.
        validate_distribution(
            presented_evidence,
            canonical,
            f"{context}.presented_evidence_distribution",
        )
        validate_distribution(
            source_evidence,
            canonical,
            f"{context}.source_evidence_distribution",
        )
        require(
            equal_distribution(
                presented_evidence,
                record["label_country_human_distribution"],
            )
            and equal_distribution(
                source_evidence,
                record["label_country_human_distribution"],
            ),
            f"{context}: legacy reference-evidence fields disagree with label country",
        )
    else:
        require(
            presented_evidence is None,
            f"{context}.presented_evidence_distribution must be null when evidence is absent",
        )
        require(
            source_evidence is None,
            f"{context}.source_evidence_distribution must be null when evidence is absent",
        )

    # ``evidence_distribution`` is a compatibility alias in older repaired
    # payloads.  When supplied by a new payload it must mirror what was
    # actually presented, including explicit nulls in no-evidence conditions.
    if "evidence_distribution" in record:
        evidence_alias = record["evidence_distribution"]
        if evidence_presented:
            validate_distribution(
                evidence_alias,
                canonical,
                f"{context}.evidence_distribution",
            )
            require(
                equal_distribution(evidence_alias, presented_evidence),
                f"{context}.evidence_distribution disagrees with presented evidence",
            )
        elif legacy_reference_evidence:
            validate_distribution(
                evidence_alias,
                canonical,
                f"{context}.evidence_distribution",
            )
            require(
                equal_distribution(evidence_alias, presented_evidence),
                f"{context}.evidence_distribution disagrees with legacy reference evidence",
            )
        else:
            require(
                evidence_alias is None,
                f"{context}.evidence_distribution must be null when evidence is absent",
            )
    displayed_prediction = map_label_probabilities(labels, used, label_probabilities)
    recovered = recover_canonical_distribution(displayed_prediction, canonical)
    require(
        equal_distribution(recovered, record["prediction"]),
        f"{context}.prediction does not round-trip through its option-label map",
    )


def validate_payload(payload: Mapping[str, Any], allow_synthetic: bool) -> dict:
    require(isinstance(payload, Mapping), "result payload must be an object")
    metadata = payload.get("metadata")
    validate_repaired_metadata(metadata, allow_synthetic)
    results = payload.get("results")
    require(isinstance(results, list), "results must be a list")
    for index, row in enumerate(results):
        require(isinstance(row, Mapping), f"results[{index}] must be an object")
        validate_row(row, metadata, index)

    groups = group_results_by_directed_unit(results)
    expected_conditions = set(REQUIRED_CONDITIONS)
    for key, rows in groups.items():
        require(set(rows) == expected_conditions, f"Incomplete or extra conditions for {key!r}")
        reference = rows["baseline"]
        for condition, row in rows.items():
            for field in (
                "unit_id",
                "target_unit_id",
                "original_options",
                "used_options",
                "option_label_map",
                "human_distribution",
                "label_country_human_distribution",
                "evidence_country_human_distribution",
            ):
                require(row[field] == reference[field], f"{key!r}: {field} differs across conditions")

    declared_models = set(metadata["models"])
    observed_models = {key[0] for key in groups}
    require(observed_models == declared_models, "metadata and observed model sets differ")
    units_by_model: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for model, question_id, country, conflict_country in groups:
        units_by_model[model].add((question_id, country, conflict_country))
    reference_units = units_by_model[metadata["models"][0]]
    for question_id, country, conflict_country in reference_units:
        require(
            (question_id, conflict_country, country) in reference_units,
            "directed-unit manifest is not reciprocal",
        )
    for model in metadata["models"]:
        require(units_by_model[model] == reference_units, f"{model}: directed-unit coverage differs")
    require(metadata.get("num_pairs") == len(reference_units), "metadata.num_pairs is wrong")
    require(
        len(results) == len(declared_models) * len(reference_units) * len(REQUIRED_CONDITIONS),
        "result cardinality is inconsistent with metadata",
    )

    human_by_target: dict[tuple[str, str], Mapping[str, Any]] = {}
    for rows in groups.values():
        row = rows["baseline"]
        target = (row["question_id"], row["country"])
        label_human = row["label_country_human_distribution"]
        previous = human_by_target.setdefault(target, label_human)
        require(
            equal_distribution(previous, label_human),
            f"{target!r}: human distribution differs",
        )
    for rows in groups.values():
        conflict = rows["conflict"]
        evidence_target = (conflict["question_id"], conflict["conflict_country"])
        if evidence_target in human_by_target:
            require(
                equal_distribution(
                    conflict["evidence_country_human_distribution"],
                    human_by_target[evidence_target],
                ),
                f"{evidence_target!r}: evidence-country reference does not match source population",
            )

    targets: dict[tuple[str, str, str], list[dict[str, Mapping[str, Any]]]] = defaultdict(list)
    for key, rows in groups.items():
        targets[(key[0], key[1], key[2])].append(rows)
    for target_key, target_groups in targets.items():
        reference = target_groups[0]
        for other in target_groups[1:]:
            for condition in ("baseline", "country_label", "population_evidence"):
                for field in (
                    "prompt",
                    "original_options",
                    "used_options",
                    "option_label_map",
                    "prediction",
                    "human_distribution",
                    "label_country_human_distribution",
                    "presented_evidence_distribution",
                    "source_evidence_distribution",
                    "scoring",
                ):
                    left, right = reference[condition][field], other[condition][field]
                    if field in {
                        "prediction",
                        "human_distribution",
                        "label_country_human_distribution",
                        "presented_evidence_distribution",
                        "source_evidence_distribution",
                    }:
                        same = (
                            left is right
                            if left is None or right is None
                            else equal_distribution(left, right)
                        )
                    else:
                        same = left == right
                    require(same, f"{target_key!r}: repeated target has inconsistent {condition}.{field}")
    return {"metadata": metadata, "results": results, "groups": groups, "targets": targets}


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty metric")
    return {
        "n": len(values),
        "mean": fmean(values),
        "population_sd": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def analyze_payload(payload: Mapping[str, Any], allow_synthetic: bool = False) -> dict[str, Any]:
    validated = validate_payload(payload, allow_synthetic)
    metadata = validated["metadata"]
    groups = validated["groups"]
    targets = validated["targets"]
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "country_influence": [],
            "evidence_influence": [],
            "EO_raw": [],
            "EO_normalized": [],
        }
    )
    per_target = []
    for (model, question_id, country), target_groups in sorted(targets.items()):
        rows = sorted(target_groups, key=lambda item: item["baseline"]["conflict_country"])[0]
        baseline = rows["baseline"]
        label = rows["country_label"]
        evidence = rows["population_evidence"]
        label_human = baseline["label_country_human_distribution"]
        country_influence = Metrics.country_influence(
            baseline["prediction"], label["prediction"], label_human
        )
        evidence_influence = Metrics.evidence_influence(
            baseline["prediction"], evidence["prediction"], label_human
        )
        metric_values[model]["country_influence"].append(country_influence)
        metric_values[model]["evidence_influence"].append(evidence_influence)
        conflict_countries = sorted(item["baseline"]["conflict_country"] for item in target_groups)
        per_target.append(
            {
                "model_name": model,
                "question_id": question_id,
                "country": country,
                "conflict_countries": conflict_countries,
                "directed_repetitions": len(conflict_countries),
                "country_influence": country_influence,
                "evidence_influence": evidence_influence,
            }
        )

    per_directed = []
    for (model, question_id, country, conflict_country), rows in sorted(groups.items()):
        conflict = rows["conflict"]
        evidence_human = conflict["evidence_country_human_distribution"]
        label_human = conflict["label_country_human_distribution"]
        EO_raw = Metrics.evidence_override_raw(
            conflict["prediction"],
            evidence_human,
            label_human,
        )
        EO_normalized = Metrics.evidence_override_normalized(
            conflict["prediction"],
            evidence_human,
            label_human,
        )
        metric_values[model]["EO_raw"].append(EO_raw)
        metric_values[model]["EO_normalized"].append(EO_normalized)
        per_directed.append(
            {
                "model_name": model,
                "question_id": question_id,
                "country": country,
                "conflict_country": conflict_country,
                "EO_raw": EO_raw,
                "EO_normalized": EO_normalized,
            }
        )

    unique_directed = {(key[1], key[2], key[3]) for key in groups}
    unique_targets = {(key[1], key[2]) for key in targets}
    model_summaries = {}
    for model, metrics in sorted(metric_values.items()):
        model_summaries[model] = {
            "counts": {
                "target_units": len([key for key in targets if key[0] == model]),
                "directed_units": len([key for key in groups if key[0] == model]),
            },
            **{metric: summarize(values) for metric, values in metrics.items()},
        }
    return {
        "source_metadata": deepcopy(metadata),
        "counts": {
            "result_rows": len(validated["results"]),
            "models": len(metadata["models"]),
            "unique_directed_units": len(unique_directed),
            "model_directed_unit_groups": len(groups),
            "unique_target_units": len(unique_targets),
            "model_target_unit_groups": len(targets),
        },
        "models": model_summaries,
        "per_target_unit": per_target,
        "per_directed_unit": per_directed,
        "interpretation_note": (
            "EO_raw is the primary signed divergence contrast in the conflict "
            "condition; EO_normalized is the secondary square-root-JSD contrast "
            "normalized by the distance between the two human references. Positive "
            "values mean evidence-side and negative values mean label-side; neither "
            "metric is by itself a causal override estimator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit pipeline diagnostics on smoke data; never use as paper results",
    )
    args = parser.parse_args()
    with args.result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    analysis = analyze_payload(payload, allow_synthetic=args.allow_synthetic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    counts = analysis["counts"]
    print(
        "ANALYSIS PASS: "
        f"{counts['unique_directed_units']} directed units, "
        f"{counts['unique_target_units']} target units -> {args.output}"
    )


if __name__ == "__main__":
    main()

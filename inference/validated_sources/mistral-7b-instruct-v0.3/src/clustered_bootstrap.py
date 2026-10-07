"""Question-clustered bootstrap confidence intervals for audit metrics.

The input is the validated point-analysis object produced by
``scripts.analyze_results.analyze_payload``.  That analysis has already
combined the four condition rows needed for each metric and preserves the
complete directed country-pair key.  This module resamples *question IDs*, not
condition rows or directed units.  A sampled question therefore contributes
all target units, all directed/reciprocal units, and all models associated
with that question.
"""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METRIC_NAMES = (
    "country_influence",
    "evidence_influence",
    "EO_raw",
    "EO_normalized",
)
TARGET_METRICS = ("country_influence", "evidence_influence")
DIRECTED_METRICS = ("EO_raw", "EO_normalized")


MetricClusters = dict[str, dict[str, dict[str, tuple[float, ...]]]]


def _required_text(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value


def _finite_metric(record: Mapping[str, Any], field: str, context: str) -> float:
    value = record.get(field)
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{context}.{field} must be a finite number")
    return float(value)


def build_question_metric_clusters(analysis: Mapping[str, Any]) -> MetricClusters:
    """Group every metric observation by model and ``question_id``.

    Country and Evidence Influence are target-level observations, while both
    Evidence Override estimands are directed-unit observations.  Grouping both
    sources by question before any random sampling preserves the repaired
    estimands without collapsing directed country pairs.
    """
    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be a mapping")

    source_models = analysis.get("models")
    if not isinstance(source_models, Mapping) or not source_models:
        raise ValueError("analysis.models must contain at least one model")
    if not all(isinstance(model, str) and model.strip() for model in source_models):
        raise ValueError("analysis.models keys must be non-empty model names")
    model_names = tuple(sorted(source_models))

    target_rows = analysis.get("per_target_unit")
    directed_rows = analysis.get("per_directed_unit")
    if not isinstance(target_rows, list) or not isinstance(directed_rows, list):
        raise ValueError(
            "analysis must contain per_target_unit and per_directed_unit lists"
        )

    mutable: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {metric: [] for metric in METRIC_NAMES})
    )

    seen_targets: set[tuple[str, str, str]] = set()
    for index, row in enumerate(target_rows):
        context = f"per_target_unit[{index}]"
        if not isinstance(row, Mapping):
            raise ValueError(f"{context} must be an object")
        model = _required_text(row, "model_name", context)
        question_id = _required_text(row, "question_id", context)
        country = _required_text(row, "country", context)
        if model not in source_models:
            raise ValueError(f"{context}.model_name is not declared in analysis.models")
        target_key = (model, question_id, country)
        if target_key in seen_targets:
            raise ValueError(f"duplicate target metric row: {target_key!r}")
        seen_targets.add(target_key)
        for metric in TARGET_METRICS:
            mutable[model][question_id][metric].append(
                _finite_metric(row, metric, context)
            )

    seen_directed: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(directed_rows):
        context = f"per_directed_unit[{index}]"
        if not isinstance(row, Mapping):
            raise ValueError(f"{context} must be an object")
        model = _required_text(row, "model_name", context)
        question_id = _required_text(row, "question_id", context)
        country = _required_text(row, "country", context)
        conflict_country = _required_text(row, "conflict_country", context)
        if model not in source_models:
            raise ValueError(f"{context}.model_name is not declared in analysis.models")
        if country == conflict_country:
            raise ValueError(f"{context} is a self-pair")
        directed_key = (model, question_id, country, conflict_country)
        if directed_key in seen_directed:
            raise ValueError(f"duplicate directed metric row: {directed_key!r}")
        seen_directed.add(directed_key)
        for metric in DIRECTED_METRICS:
            mutable[model][question_id][metric].append(
                _finite_metric(row, metric, context)
            )

    for model, question_id, country, conflict_country in seen_directed:
        reverse = (model, question_id, conflict_country, country)
        if reverse not in seen_directed:
            raise ValueError(
                "directed metric rows are not reciprocal: "
                f"missing {reverse!r}"
            )
        target_countries = {
            target_country
            for target_model, target_question, target_country in seen_targets
            if target_model == model and target_question == question_id
        }
        missing_targets = {country, conflict_country} - target_countries
        if missing_targets:
            raise ValueError(
                f"directed endpoints lack target metric rows for model {model!r}, "
                f"question {question_id!r}: {sorted(missing_targets)!r}"
            )

    for model, question_id, country in seen_targets:
        directed_endpoints = {
            endpoint
            for directed_model, directed_question, left, right in seen_directed
            if directed_model == model and directed_question == question_id
            for endpoint in (left, right)
        }
        if country not in directed_endpoints:
            raise ValueError(
                f"target metric row has no directed-unit endpoint for model {model!r}, "
                f"question {question_id!r}, country {country!r}"
            )

    clusters: MetricClusters = {}
    question_sets: dict[str, set[str]] = {}
    for model in model_names:
        if model not in mutable:
            raise ValueError(f"model {model!r} has no metric observations")
        question_sets[model] = set(mutable[model])
        clusters[model] = {}
        for question_id, metric_values in mutable[model].items():
            empty_metrics = [name for name in METRIC_NAMES if not metric_values[name]]
            if empty_metrics:
                raise ValueError(
                    f"model {model!r}, question {question_id!r} has no values for "
                    f"{empty_metrics!r}"
                )
            clusters[model][question_id] = {
                metric: tuple(metric_values[metric]) for metric in METRIC_NAMES
            }

    reference_model = model_names[0]
    reference_questions = question_sets[reference_model]
    if not reference_questions:
        raise ValueError("analysis contains no question clusters")
    for model in model_names[1:]:
        if question_sets[model] != reference_questions:
            missing = sorted(reference_questions - question_sets[model])
            extra = sorted(question_sets[model] - reference_questions)
            raise ValueError(
                f"question coverage differs for model {model!r} "
                f"(missing={missing!r}, extra={extra!r})"
            )

    # Strict result validation guarantees identical experimental coverage by
    # model.  Re-check the exact target and directed keys here so a malformed
    # or hand-built analysis object cannot silently unpair model clusters.
    for question_id in sorted(reference_questions):
        reference_targets = {
            country
            for model, qid, country in seen_targets
            if model == reference_model and qid == question_id
        }
        reference_directed = {
            (country, conflict_country)
            for model, qid, country, conflict_country in seen_directed
            if model == reference_model and qid == question_id
        }
        for model in model_names[1:]:
            targets = {
                country
                for observed_model, qid, country in seen_targets
                if observed_model == model and qid == question_id
            }
            directed = {
                (country, conflict_country)
                for observed_model, qid, country, conflict_country in seen_directed
                if observed_model == model and qid == question_id
            }
            if targets != reference_targets or directed != reference_directed:
                raise ValueError(
                    f"experimental-unit coverage differs for model {model!r}, "
                    f"question {question_id!r}"
                )

    return clusters


def pooled_cluster_means(
    clusters: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    sampled_question_ids: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Pool all observations from the selected question clusters.

    Repeated question IDs duplicate the whole cluster, exactly as required by
    a nonparametric cluster bootstrap.  This helper is public so tests and
    downstream analyses can verify a particular resample explicitly.
    """
    sampled = tuple(sampled_question_ids)
    if not sampled:
        raise ValueError("sampled_question_ids must not be empty")
    if not isinstance(clusters, Mapping) or not clusters:
        raise ValueError("clusters must contain at least one model")

    result: dict[str, dict[str, float]] = {}
    for model, question_clusters in clusters.items():
        if not isinstance(question_clusters, Mapping):
            raise ValueError(f"clusters[{model!r}] must be a mapping")
        model_metrics: dict[str, float] = {}
        for metric in METRIC_NAMES:
            values: list[float] = []
            for question_id in sampled:
                if question_id not in question_clusters:
                    raise ValueError(
                        f"model {model!r} is missing sampled question {question_id!r}"
                    )
                cluster = question_clusters[question_id]
                if metric not in cluster or not cluster[metric]:
                    raise ValueError(
                        f"model {model!r}, question {question_id!r} is missing {metric}"
                    )
                values.extend(float(value) for value in cluster[metric])
            if not all(math.isfinite(value) for value in values):
                raise ValueError("cluster metric values must be finite")
            model_metrics[metric] = fmean(values)
        result[model] = model_metrics
    return result


def clustered_bootstrap(
    analysis: Mapping[str, Any],
    *,
    n_replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute percentile intervals using a question-ID cluster bootstrap.

    A single vector of sampled question IDs is shared by every model and every
    metric in a replicate.  No condition row, target country, or directed
    country pair is ever sampled independently.
    """
    if isinstance(n_replicates, (bool, np.bool_)) or not isinstance(
        n_replicates, (int, np.integer)
    ):
        raise TypeError("n_replicates must be an integer")
    n_replicates = int(n_replicates)
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    if not isinstance(confidence_level, (int, float)) or isinstance(
        confidence_level, bool
    ):
        raise TypeError("confidence_level must be numeric")
    confidence_level = float(confidence_level)
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    clusters = build_question_metric_clusters(analysis)
    model_names = tuple(sorted(clusters))
    question_ids = tuple(sorted(clusters[model_names[0]]))
    n_questions = len(question_ids)
    rng = np.random.default_rng(int(seed))

    bootstrap_values = {
        model: {
            metric: np.empty(n_replicates, dtype=np.float64)
            for metric in METRIC_NAMES
        }
        for model in model_names
    }

    # One shared draw per replicate preserves cross-model pairing and carries
    # every directed unit/condition represented by a sampled question.
    for replicate in range(n_replicates):
        indices = rng.integers(0, n_questions, size=n_questions)
        sampled_questions = tuple(question_ids[int(index)] for index in indices)
        estimates = pooled_cluster_means(clusters, sampled_questions)
        for model in model_names:
            for metric in METRIC_NAMES:
                bootstrap_values[model][metric][replicate] = estimates[model][metric]

    point_estimates = pooled_cluster_means(clusters, question_ids)
    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - lower_quantile

    models: dict[str, dict[str, Any]] = {}
    for model in model_names:
        model_result: dict[str, Any] = {}
        for metric in METRIC_NAMES:
            values = bootstrap_values[model][metric]
            lower, upper = np.quantile(
                values, [lower_quantile, upper_quantile], method="linear"
            )
            n_observations = sum(
                len(clusters[model][question_id][metric])
                for question_id in question_ids
            )
            interval = {
                "confidence_level": confidence_level,
                "lower": float(lower),
                "upper": float(upper),
            }
            metric_result = {
                "estimate": point_estimates[model][metric],
                "n_observations": n_observations,
                "percentile_ci": interval,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_standard_error": float(np.std(values, ddof=1))
                if n_replicates > 1
                else 0.0,
            }
            if math.isclose(confidence_level, 0.95, rel_tol=0.0, abs_tol=1e-15):
                metric_result["percentile_95_ci"] = {
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                }
            model_result[metric] = metric_result
        models[model] = model_result

    lower_percentile = round(100.0 * lower_quantile, 12)
    upper_percentile = round(100.0 * upper_quantile, 12)

    return {
        "method": {
            "name": "nonparametric_question_cluster_bootstrap",
            "cluster_key": "question_id",
            "sampling_unit": "unique question_id",
            "n_clusters": n_questions,
            "clusters_drawn_per_replicate": n_questions,
            "n_replicates": n_replicates,
            "confidence_level": confidence_level,
            "interval": "percentile",
            "quantile_method": "linear",
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "seed": int(seed),
            "shared_draw_across_models_and_metrics": True,
            "independent_condition_row_resampling": False,
            "within_replicate_aggregation": (
                "pooled unit-weighted mean, matching the point estimator"
            ),
            "rng": "numpy.random.Generator(PCG64)",
            "question_ids": list(question_ids),
            "retained_with_each_cluster": [
                "all target countries",
                "all directed country pairs",
                "all reciprocal directions",
                "all four condition contributions",
                "all models",
            ],
        },
        "models": models,
    }

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .io import read_json, read_jsonl, write_jsonl_gz
from .metrics import (
    country_influence,
    evidence_influence,
    evidence_override_normalized,
    evidence_override_raw,
)

CONDITIONS = ("baseline", "country_label", "population_evidence", "conflict")
RAW_TO_CANONICAL = {"evidence": "population_evidence"}
METRICS = ("country_influence", "evidence_influence", "evidence_advantage", "EO_raw", "EO_normalized")
MODEL_ORDER = ("qwen", "llama", "mistral", "gemma")


@dataclass(frozen=True)
class Inputs:
    root: Path
    models: dict[str, dict[str, Any]]
    experiment: dict[str, Any]
    dataset: dict[str, Any]
    pairs: list[dict[str, Any]]


def load_inputs(root: Path) -> Inputs:
    models = read_json(root / "configs/models.json")
    experiment = read_json(root / "configs/experiment.json")
    dataset = read_json(root / "data/dataset_v2.json.gz")
    pairs = read_json(root / "data/country_pairs_v2.json")
    return Inputs(root=root, models=models, experiment=experiment, dataset=dataset, pairs=pairs)


def _close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def _distribution_close(a: Mapping[str, float], b: Mapping[str, float], tolerance: float = 1e-9) -> bool:
    return set(a) == set(b) and all(_close(a[k], b[k], tolerance) for k in a)


def _canonical_row(model_key: str, display_name: str, row: dict[str, Any], domain: str) -> dict[str, Any]:
    condition = RAW_TO_CANONICAL.get(row["condition"], row["condition"])
    return {
        "model_key": model_key,
        "model_name": display_name,
        "model_identifier": row["model_identifier"],
        "model_repository": row["model_repository"],
        "model_revision": row["model_revision"],
        "schema_version": row["schema_version"],
        "run_fingerprint": row["run_fingerprint"],
        "backend": row["backend"],
        "quantization": row["quantization"],
        "scoring_method": row["scoring_method"],
        "synthetic": row["synthetic"],
        "question_id": row["question_id"],
        "domain": domain,
        "unit_id": row["unit_id"],
        "target_unit_id": row["target_unit_id"],
        "label_country": row["label_country"],
        "evidence_country": row["evidence_country"],
        "condition": condition,
        "original_options": row["original_options"],
        "displayed_option_order": row["displayed_option_order"],
        "displayed_option_labels": row["displayed_option_labels"],
        "displayed_label_to_option": row["displayed_label_to_option"],
        "permutation_seed": row["permutation_seed"],
        "normalized_prediction": row["normalized_prediction"],
        "label_country_human_distribution": row["label_country_human_distribution"],
        "evidence_country_human_distribution": row["evidence_country_human_distribution"],
        "presented_evidence_distribution": row.get("presented_evidence_distribution"),
        "evidence_presented": row.get("evidence_presented"),
        "raw_candidate_scores": row["raw_candidate_scores"],
        "candidate_label_token_ids": row["candidate_label_token_ids"],
        "serialized_prompt_sha256": row["serialized_prompt_sha256"],
    }


def load_and_validate_rows(inputs: Inputs) -> list[dict[str, Any]]:
    manifest_keys = [(p["question_id"], p["country"], p["conflict_country"]) for p in inputs.pairs]
    if len(manifest_keys) != 200 or len(set(manifest_keys)) != 200:
        raise ValueError("directed contrast manifest must contain 200 unique rows")
    manifest_set = set(manifest_keys)
    for qid, left, right in manifest_keys:
        if (qid, right, left) not in manifest_set:
            raise ValueError(f"missing reciprocal contrast for {(qid, left, right)}")

    all_rows: list[dict[str, Any]] = []
    coverage_by_model: dict[str, set[tuple[str, str, str, str]]] = {}
    permutation_reference: dict[tuple[str, str], tuple[Any, ...]] = {}

    for model_key in MODEL_ORDER:
        cfg = inputs.models[model_key]
        rows = read_jsonl(inputs.root / cfg["result_file"])
        if len(rows) != 800:
            raise ValueError(f"{model_key}: expected 800 rows, found {len(rows)}")
        observed: set[tuple[str, str, str, str]] = set()
        for index, row in enumerate(rows):
            context = f"{model_key}[{index}]"
            if row.get("synthetic") is not False or row.get("backend") != "llama.cpp":
                raise ValueError(f"{context}: invalid backend or synthetic result")
            if row.get("generated_answer") is not None:
                raise ValueError(f"{context}: generated-answer parsing is present")
            if row.get("scoring_method") != "full_contextual_option_label_sequence_log_probability":
                raise ValueError(f"{context}: unexpected scoring method")
            key3 = (row["question_id"], row["label_country"], row["evidence_country"])
            if key3 not in manifest_set:
                raise ValueError(f"{context}: unexpected directed contrast {key3}")
            condition = RAW_TO_CANONICAL.get(row["condition"], row["condition"])
            if condition not in CONDITIONS:
                raise ValueError(f"{context}: unexpected condition {condition}")
            key4 = (*key3, condition)
            if key4 in observed:
                raise ValueError(f"{context}: duplicate directed-condition key")
            observed.add(key4)
            options = row["original_options"]
            prediction = row["normalized_prediction"]
            if set(options) != set(prediction):
                raise ValueError(f"{context}: probability keys do not match options")
            total = math.fsum(float(prediction[o]) for o in options)
            if not _close(total, 1.0, 1e-8):
                raise ValueError(f"{context}: probabilities sum to {total}")
            if any(float(prediction[o]) < 0 or not math.isfinite(float(prediction[o])) for o in options):
                raise ValueError(f"{context}: invalid probability")
            perm_key = (row["question_id"], row["label_country"])
            perm_value = (tuple(row["displayed_option_order"]), row["permutation_seed"])
            previous = permutation_reference.setdefault(perm_key, perm_value)
            if previous != perm_value:
                raise ValueError(f"{context}: cross-model or cross-condition permutation mismatch")
            qid = row["question_id"]
            if qid not in inputs.dataset:
                raise ValueError(f"{context}: question absent from repaired dataset")
            all_rows.append(_canonical_row(model_key, cfg["display_name"], row, inputs.dataset[qid]["domain"]))
        coverage_by_model[model_key] = observed

    reference = coverage_by_model[MODEL_ORDER[0]]
    for model_key in MODEL_ORDER[1:]:
        if coverage_by_model[model_key] != reference:
            raise ValueError(f"experimental coverage differs for {model_key}")
    if len(reference) != 800:
        raise ValueError("cross-model coverage does not contain 800 directed-condition keys")
    return all_rows


def derive_unit_metrics(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["model_key"], row["question_id"], row["label_country"], row["evidence_country"])
        grouped[key][row["condition"]] = row

    target_candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    directed_rows: list[dict[str, Any]] = []
    for (model_key, qid, label, evidence), conditions in grouped.items():
        if set(conditions) != set(CONDITIONS):
            raise ValueError(f"incomplete condition set for {(model_key, qid, label, evidence)}")
        baseline = conditions["baseline"]
        label_row = conditions["country_label"]
        evidence_row = conditions["population_evidence"]
        conflict = conditions["conflict"]
        options = baseline["original_options"]
        label_reference = baseline["label_country_human_distribution"]
        evidence_reference = baseline["evidence_country_human_distribution"]
        ci = country_influence(baseline["normalized_prediction"], label_row["normalized_prediction"], label_reference, options)
        ei = evidence_influence(baseline["normalized_prediction"], evidence_row["normalized_prediction"], label_reference, options)
        eo_raw = evidence_override_raw(conflict["normalized_prediction"], label_reference, evidence_reference, options)
        eo_norm = evidence_override_normalized(conflict["normalized_prediction"], label_reference, evidence_reference, options)
        classification = "evidence-side" if eo_raw > 1e-12 else "label-side" if eo_raw < -1e-12 else "tie"
        target_candidates[(model_key, qid, label)].append({
            "model_key": model_key,
            "model_name": baseline["model_name"],
            "question_id": qid,
            "domain": baseline["domain"],
            "label_country": label,
            "country_influence": ci,
            "evidence_influence": ei,
            "evidence_advantage": ei - ci,
            "baseline_prediction": json.dumps(baseline["normalized_prediction"], ensure_ascii=False, sort_keys=True),
            "country_label_prediction": json.dumps(label_row["normalized_prediction"], ensure_ascii=False, sort_keys=True),
            "population_evidence_prediction": json.dumps(evidence_row["normalized_prediction"], ensure_ascii=False, sort_keys=True),
        })
        directed_rows.append({
            "model_key": model_key,
            "model_name": baseline["model_name"],
            "question_id": qid,
            "domain": baseline["domain"],
            "label_country": label,
            "evidence_country": evidence,
            "unit_id": baseline["unit_id"],
            "EO_raw": eo_raw,
            "EO_normalized": eo_norm,
            "classification": classification,
            "conflict_prediction": json.dumps(conflict["normalized_prediction"], ensure_ascii=False, sort_keys=True),
            "label_reference": json.dumps(label_reference, ensure_ascii=False, sort_keys=True),
            "evidence_reference": json.dumps(evidence_reference, ensure_ascii=False, sort_keys=True),
        })

    target_rows: list[dict[str, Any]] = []
    for key, candidates in target_candidates.items():
        first = candidates[0]
        for other in candidates[1:]:
            for field in ["country_influence", "evidence_influence", "evidence_advantage"]:
                if not _close(first[field], other[field], 1e-9):
                    raise ValueError(f"target metric varies across evidence partners for {key}: {field}")
            for field in ["baseline_prediction", "country_label_prediction", "population_evidence_prediction"]:
                if first[field] != other[field]:
                    raise ValueError(f"target prediction varies across evidence partners for {key}: {field}")
        target_rows.append(first)

    target_df = pd.DataFrame(target_rows).sort_values(["model_key", "question_id", "label_country"], kind="stable").reset_index(drop=True)
    directed_df = pd.DataFrame(directed_rows).sort_values(["model_key", "question_id", "label_country", "evidence_country"], kind="stable").reset_index(drop=True)
    counts = target_df.groupby("model_key").size().to_dict()
    if any(counts.get(k) != 144 for k in MODEL_ORDER):
        raise ValueError(f"unexpected target-unit counts: {counts}")
    counts = directed_df.groupby("model_key").size().to_dict()
    if any(counts.get(k) != 200 for k in MODEL_ORDER):
        raise ValueError(f"unexpected directed-unit counts: {counts}")
    return target_df, directed_df


def clustered_bootstrap(target_df: pd.DataFrame, directed_df: pd.DataFrame, replicates: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Question-clustered percentile bootstrap with shared draws.

    Each bootstrap replicate samples 44 question clusters with replacement.
    All target populations, reciprocal directions, conditions, and models tied
    to a selected question remain together. The same cluster-index matrix is
    used for every model and estimand.
    """
    question_ids = tuple(sorted(target_df["question_id"].unique()))
    if len(question_ids) != 44:
        raise ValueError(f"expected 44 question clusters, found {len(question_ids)}")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(question_ids), size=(replicates, len(question_ids)))

    rows: list[dict[str, Any]] = []
    target_metrics = ("country_influence", "evidence_influence", "evidence_advantage")
    directed_metrics = ("EO_raw", "EO_normalized")

    for model in MODEL_ORDER:
        target_model = target_df[target_df.model_key == model]
        directed_model = directed_df[directed_df.model_key == model]
        display = target_model["model_name"].iloc[0]

        bootstrap_values: dict[str, np.ndarray] = {}
        for metric in target_metrics:
            grouped = target_model.groupby("question_id", sort=False)[metric].agg(["sum", "count"])
            grouped = grouped.reindex(question_ids)
            cluster_sums = grouped["sum"].to_numpy(dtype=np.float64)
            cluster_counts = grouped["count"].to_numpy(dtype=np.float64)
            bootstrap_values[metric] = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)

        for metric in directed_metrics:
            grouped = directed_model.groupby("question_id", sort=False)[metric].agg(["sum", "count"])
            grouped = grouped.reindex(question_ids)
            cluster_sums = grouped["sum"].to_numpy(dtype=np.float64)
            cluster_counts = grouped["count"].to_numpy(dtype=np.float64)
            bootstrap_values[metric] = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)

        point = {
            "country_influence": target_model["country_influence"].mean(),
            "evidence_influence": target_model["evidence_influence"].mean(),
            "evidence_advantage": target_model["evidence_advantage"].mean(),
            "EO_raw": directed_model["EO_raw"].mean(),
            "EO_normalized": directed_model["EO_normalized"].mean(),
        }
        for metric in METRICS:
            low, high = np.quantile(bootstrap_values[metric], [0.025, 0.975], method="linear")
            rows.append({
                "model_key": model,
                "model": display,
                "metric": metric,
                "n": 144 if metric in target_metrics else 200,
                "estimate": point[metric],
                "ci_95_lower": float(low),
                "ci_95_upper": float(high),
                "bootstrap_replicates": replicates,
                "question_clusters": len(question_ids),
                "seed": seed,
            })
    return pd.DataFrame(rows)

def summary_tables(target_df: pd.DataFrame, directed_df: pd.DataFrame, bootstrap_df: pd.DataFrame, dataset: dict[str, Any]) -> dict[str, pd.DataFrame]:
    wide_rows = []
    for model in MODEL_ORDER:
        sub = bootstrap_df[bootstrap_df.model_key == model].set_index("metric")
        row = {"model_key": model, "model": sub.iloc[0]["model"]}
        for metric in METRICS:
            row[metric] = sub.loc[metric, "estimate"]
            row[f"{metric}_ci_95_lower"] = sub.loc[metric, "ci_95_lower"]
            row[f"{metric}_ci_95_upper"] = sub.loc[metric, "ci_95_upper"]
        wide_rows.append(row)
    main = pd.DataFrame(wide_rows)

    conflict_rows = []
    for model in MODEL_ORDER:
        sub = directed_df[directed_df.model_key == model]
        counts = sub["classification"].value_counts().to_dict()
        conflict_rows.append({
            "model_key": model,
            "model": sub["model_name"].iloc[0],
            "evidence_side_n": counts.get("evidence-side", 0),
            "evidence_side_pct": 100 * counts.get("evidence-side", 0) / len(sub),
            "label_side_n": counts.get("label-side", 0),
            "label_side_pct": 100 * counts.get("label-side", 0) / len(sub),
            "tie_n": counts.get("tie", 0),
            "tie_pct": 100 * counts.get("tie", 0) / len(sub),
        })
    conflict = pd.DataFrame(conflict_rows)

    pivot = directed_df.pivot(index=["question_id", "label_country", "evidence_country", "unit_id", "domain"], columns="model_key", values=["EO_normalized", "classification"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    class_cols = [f"classification_{m}" for m in MODEL_ORDER]
    pivot["evidence_side_models"] = (pivot[class_cols] == "evidence-side").sum(axis=1)
    pivot["label_side_models"] = (pivot[class_cols] == "label-side").sum(axis=1)
    pivot["tie_models"] = (pivot[class_cols] == "tie").sum(axis=1)
    eo_cols = [f"EO_normalized_{m}" for m in MODEL_ORDER]
    pivot["EO_normalized_mean"] = pivot[eo_cols].mean(axis=1)
    pivot["EO_normalized_variance"] = pivot[eo_cols].var(axis=1, ddof=0)
    pivot = pivot.sort_values(["question_id", "label_country", "evidence_country"], kind="stable").reset_index(drop=True)

    agree_summary = pd.DataFrame([
        {"statistic": "all_four_evidence_side", "count": int((pivot.evidence_side_models == 4).sum()), "percent": 100 * float((pivot.evidence_side_models == 4).mean())},
        {"statistic": "at_least_three_evidence_side", "count": int((pivot.evidence_side_models >= 3).sum()), "percent": 100 * float((pivot.evidence_side_models >= 3).mean())},
        {"statistic": "all_four_label_side", "count": int((pivot.label_side_models == 4).sum()), "percent": 100 * float((pivot.label_side_models == 4).mean())},
        {"statistic": "any_cross_model_disagreement", "count": int(((pivot.evidence_side_models > 0) & (pivot.label_side_models > 0)).sum()), "percent": 100 * float(((pivot.evidence_side_models > 0) & (pivot.label_side_models > 0)).mean())},
        {"statistic": "three_evidence_one_label", "count": int((pivot.evidence_side_models == 3).sum()), "percent": 100 * float((pivot.evidence_side_models == 3).mean())},
        {"statistic": "one_evidence_three_label", "count": int((pivot.label_side_models == 3).sum()), "percent": 100 * float((pivot.label_side_models == 3).mean())},
        {"statistic": "three_to_one_splits_total", "count": int(((pivot.evidence_side_models == 3) | (pivot.label_side_models == 3)).sum()), "percent": 100 * float(((pivot.evidence_side_models == 3) | (pivot.label_side_models == 3)).mean())},
        {"statistic": "two_to_two_splits", "count": int(((pivot.evidence_side_models == 2) & (pivot.label_side_models == 2)).sum()), "percent": 100 * float(((pivot.evidence_side_models == 2) & (pivot.label_side_models == 2)).mean())},
        {"statistic": "mistral_only_label_side", "count": int(((pivot.classification_mistral == "label-side") & (pivot[[f"classification_{m}" for m in ("qwen", "llama", "gemma")]] == "evidence-side").all(axis=1)).sum()), "percent": 100 * float(((pivot.classification_mistral == "label-side") & (pivot[[f"classification_{m}" for m in ("qwen", "llama", "gemma")]] == "evidence-side").all(axis=1)).mean())},
    ])

    def selected_row(kind: str, frame: pd.DataFrame) -> pd.Series:
        if kind == "largest_mean":
            return frame.sort_values(["EO_normalized_mean", "unit_id"], ascending=[False, True]).iloc[0]
        if kind == "lowest_mean":
            return frame.sort_values(["EO_normalized_mean", "unit_id"], ascending=[True, True]).iloc[0]
        return frame.sort_values(["EO_normalized_variance", "unit_id"], ascending=[False, True]).iloc[0]

    cases = []
    for kind, label in [("largest_mean", "Largest mean EO"), ("lowest_mean", "Lowest mean EO"), ("largest_variance", "Largest model disagreement")]:
        row = selected_row(kind, pivot)
        question = dataset[row["question_id"]]["question"].replace("\n", " ").strip()
        cases.append({
            "selection_rule": label,
            "question_id": row["question_id"],
            "question": question,
            "label_country": row["label_country"],
            "evidence_country": row["evidence_country"],
            **{m: row[f"EO_normalized_{m}"] for m in MODEL_ORDER},
            "cross_model_mean": row["EO_normalized_mean"],
            "cross_model_variance": row["EO_normalized_variance"],
        })
    cases_df = pd.DataFrame(cases)

    source_rows = []
    for model in MODEL_ORDER:
        for domain in sorted(target_df.domain.unique()):
            t = target_df[(target_df.model_key == model) & (target_df.domain == domain)]
            d = directed_df[(directed_df.model_key == model) & (directed_df.domain == domain)]
            source_rows.append({
                "model_key": model,
                "model": t.model_name.iloc[0],
                "source": "Global Attitudes" if domain == "GAS" else domain,
                "target_units": len(t),
                "directed_units": len(d),
                "country_influence": t.country_influence.mean(),
                "evidence_influence": t.evidence_influence.mean(),
                "evidence_advantage": t.evidence_advantage.mean(),
                "EO_raw": d.EO_raw.mean(),
                "EO_normalized": d.EO_normalized.mean(),
                "evidence_side_pct": 100 * (d.classification == "evidence-side").mean(),
            })
    source_df = pd.DataFrame(source_rows)

    return {
        "main_metrics": main,
        "conflict_classification": conflict,
        "cross_model_agreement_by_unit": pivot,
        "cross_model_agreement_summary": agree_summary,
        "illustrative_cases": cases_df,
        "source_stratified_metrics": source_df,
    }


def harmonized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: i for i, name in enumerate(CONDITIONS)}
    return sorted(rows, key=lambda r: (MODEL_ORDER.index(r["model_key"]), r["question_id"], r["label_country"], r["evidence_country"], order[r["condition"]]))


def run(root: Path, output_root: Path | None = None) -> dict[str, Any]:
    inputs = load_inputs(root)
    rows = load_and_validate_rows(inputs)
    target_df, directed_df = derive_unit_metrics(rows)
    bootstrap_df = clustered_bootstrap(target_df, directed_df, replicates=inputs.experiment["bootstrap"]["replicates"], seed=inputs.experiment["bootstrap"]["seed"])
    tables = summary_tables(target_df, directed_df, bootstrap_df, inputs.dataset)

    output_root = output_root or root / "results"
    processed = output_root / "processed"
    summary = output_root / "summary"
    processed.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(processed / "predictions_all_models.jsonl.gz", harmonized_rows(rows))
    target_df.to_csv(processed / "target_unit_metrics.csv", index=False)
    directed_df.to_csv(processed / "directed_unit_metrics.csv", index=False)
    bootstrap_df.to_csv(processed / "bootstrap_metrics_long.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(summary / f"{name}.csv", index=False)

    return {
        "rows": rows,
        "target_metrics": target_df,
        "directed_metrics": directed_df,
        "bootstrap": bootstrap_df,
        "tables": tables,
    }

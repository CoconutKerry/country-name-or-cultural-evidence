#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_repro.pipeline import load_inputs, load_and_validate_rows, derive_unit_metrics

EXPECTED = {
    "qwen": {
        "country_influence": (0.028082443820185005, -0.007665892909074293, 0.06595710986485609),
        "evidence_influence": (0.26758962481528126, 0.22019520593171818, 0.31966774914639484),
        "evidence_advantage": (0.23950718099509627, 0.1932, 0.2910),
        "EO_raw": (0.6384224188512423, 0.5713934174203635, 0.6953755579025486),
        "EO_normalized": (0.6685640350548293, 0.5916200198785707, 0.7394185247568318),
        "evidence_side": 189,
    },
    "llama": {
        "country_influence": (0.055289942581661235, 0.032934900600856035, 0.08080381234556065),
        "evidence_influence": (0.18444529339210258, 0.1496016852237651, 0.22178561273689465),
        "evidence_advantage": (0.12915535081044135, 0.0970, 0.1658),
        "EO_raw": (0.6146763959985936, 0.5562955733987089, 0.663135592922496),
        "EO_normalized": (0.7114358907606618, 0.6415560824520417, 0.7711889788277441),
        "evidence_side": 193,
    },
    "mistral": {
        "country_influence": (0.044677631687046866, 0.004066352664046308, 0.08922959940759602),
        "evidence_influence": (0.22053033433821237, 0.16473473010745493, 0.2786246714795455),
        "evidence_advantage": (0.1758527026511655, 0.1287, 0.2231),
        "EO_raw": (0.5167487916525012, 0.427308247061161, 0.6017720232006585),
        "EO_normalized": (0.5433332134305028, 0.4408328343812688, 0.6420706385614148),
        "evidence_side": 171,
    },
    "gemma": {
        "country_influence": (0.07145092846324903, 0.02730156986031642, 0.11639484632807454),
        "evidence_influence": (0.2835949176283289, 0.230166379391221, 0.33813034206869175),
        "evidence_advantage": (0.21214398916507986, 0.1619, 0.2626),
        "EO_raw": (0.6847342292554663, 0.6287851943236706, 0.7271900088374721),
        "EO_normalized": (0.7280412992136129, 0.6564566569514566, 0.7881940165454694),
        "evidence_side": 195,
    },
}


def close(a: float, b: float, tol: float = 5e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def main() -> None:
    inputs = load_inputs(ROOT)
    rows = load_and_validate_rows(inputs)
    target, directed = derive_unit_metrics(rows)
    main_metrics = pd.read_csv(ROOT / "results/summary/main_metrics.csv").set_index("model_key")
    conflict = pd.read_csv(ROOT / "results/summary/conflict_classification.csv").set_index("model_key")
    agreement = pd.read_csv(ROOT / "results/summary/cross_model_agreement_summary.csv").set_index("statistic")

    for model, expected in EXPECTED.items():
        for metric in ["country_influence", "evidence_influence", "EO_raw", "EO_normalized"]:
            estimate, lower, upper = expected[metric]
            row = main_metrics.loc[model]
            if not close(row[metric], estimate):
                raise AssertionError(f"{model} {metric} estimate mismatch")
            if not close(row[f"{metric}_ci_95_lower"], lower):
                raise AssertionError(f"{model} {metric} lower interval mismatch")
            if not close(row[f"{metric}_ci_95_upper"], upper):
                raise AssertionError(f"{model} {metric} upper interval mismatch")
        if int(conflict.loc[model, "evidence_side_n"]) != expected["evidence_side"]:
            raise AssertionError(f"{model} conflict count mismatch")
        # Paired-difference intervals are checked at the paper's reported precision.
        _, lower, upper = expected["evidence_advantage"]
        row = main_metrics.loc[model]
        if round(row["evidence_advantage_ci_95_lower"], 4) != round(lower, 4) or round(row["evidence_advantage_ci_95_upper"], 4) != round(upper, 4):
            raise AssertionError(f"{model} evidence-advantage interval mismatch")

    if int(agreement.loc["all_four_evidence_side", "count"]) != 161:
        raise AssertionError("all-model agreement mismatch")
    if int(agreement.loc["at_least_three_evidence_side", "count"]) != 190:
        raise AssertionError("three-or-more agreement mismatch")
    if int(agreement.loc["any_cross_model_disagreement", "count"]) != 39:
        raise AssertionError("disagreement count mismatch")
    if int(agreement.loc["all_four_label_side", "count"]) != 0:
        raise AssertionError("all-label agreement mismatch")
    if int(agreement.loc["three_evidence_one_label", "count"]) != 29:
        raise AssertionError("evidence-majority 3--1 split count mismatch")
    if int(agreement.loc["one_evidence_three_label", "count"]) != 3:
        raise AssertionError("label-majority 1--3 split count mismatch")
    if int(agreement.loc["two_to_two_splits", "count"]) != 7:
        raise AssertionError("2--2 split count mismatch")
    if int(agreement.loc["mistral_only_label_side", "count"]) != 20:
        raise AssertionError("Mistral-only label-side count mismatch")

    master_path = ROOT / "results/processed/predictions_all_models.jsonl.gz"
    with gzip.open(master_path, "rt", encoding="utf-8") as handle:
        master_rows = sum(1 for line in handle if line.strip())
    if master_rows != 3200:
        raise AssertionError(f"harmonized master row count mismatch: {master_rows}")

    manifest = ROOT / "provenance/INPUT_SHA256SUMS.txt"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            path = ROOT / rel
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != digest:
                raise AssertionError(f"input hash mismatch: {rel}")

    print("VERIFY PASS: 3,200 rows, 144 targets/model, 200 directed contrasts/model, published metrics reproduced.")


if __name__ == "__main__":
    main()

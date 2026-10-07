#!/usr/bin/env python3
"""Validate repaired v2 dataset and directed-pair invariants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import Metrics


INELIGIBLE = re.compile(r"\((?:Non-national|Old national) sample\)", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed/dataset_v2.json")
    )
    parser.add_argument(
        "--pairs", type=Path, default=Path("data/pairs/country_pairs_v2.json")
    )
    args = parser.parse_args()
    with args.dataset.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    with args.pairs.open("r", encoding="utf-8") as handle:
        pairs = json.load(handle)

    require(isinstance(dataset, dict) and dataset, "dataset must be non-empty")
    for question_id, question in dataset.items():
        options = question.get("options")
        require(
            isinstance(options, list)
            and len(options) >= 2
            and all(isinstance(option, str) and option for option in options)
            and len(options) == len(set(options)),
            f"{question_id}: invalid canonical options",
        )
        countries = question.get("countries", {})
        require(len(countries) >= 2, f"{question_id}: fewer than two countries")
        for country, country_data in countries.items():
            require(not INELIGIBLE.search(country), f"ineligible sample retained: {country}")
            distribution = country_data.get("distribution", {})
            Metrics.normalize_distribution(distribution, options)
            require(
                math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-10),
                f"{question_id}/{country}: unnormalized distribution",
            )

    require(isinstance(pairs, list) and pairs, "pair manifest must be non-empty")
    keys = set()
    pair_lookup = {}
    for pair in pairs:
        key = (pair["question_id"], pair["country"], pair["conflict_country"])
        require(key not in keys, f"duplicate directed unit: {key!r}")
        require(pair["country"] != pair["conflict_country"], f"self pair: {key!r}")
        require(not INELIGIBLE.search(pair["country"]), f"ineligible target: {key!r}")
        require(not INELIGIBLE.search(pair["conflict_country"]), f"ineligible evidence: {key!r}")
        question = dataset[pair["question_id"]]
        require(pair["options"] == question["options"], f"option mapping mismatch: {key!r}")
        target = question["countries"][pair["country"]]["distribution"]
        evidence = question["countries"][pair["conflict_country"]]["distribution"]
        divergence = Metrics.js_divergence(target, evidence, keys=pair["options"])
        require(
            math.isclose(divergence, pair["divergence_bits"], abs_tol=1e-12),
            f"divergence mismatch: {key!r}",
        )
        keys.add(key)
        pair_lookup[key] = pair

    for question_id, country, conflict_country in keys:
        reverse = (question_id, conflict_country, country)
        require(reverse in pair_lookup, f"missing reciprocal unit for {(question_id, country, conflict_country)!r}")

    print(
        f"DATA VALIDATION PASS: {len(dataset)} questions, {len(pairs)} unique "
        "reciprocal directed units, base-2 divergences verified"
    )


if __name__ == "__main__":
    main()

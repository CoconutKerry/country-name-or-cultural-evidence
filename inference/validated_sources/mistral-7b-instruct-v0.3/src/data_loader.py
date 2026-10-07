"""Validated loading for repaired cultural-alignment datasets and directed units."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


def canonical_option(value: object) -> str:
    """Return the stable string identifier used for options and distributions."""
    option = value.strip() if isinstance(value, str) else str(value)
    if not option:
        raise ValueError("Answer options must be non-empty")
    return option


def normalize_distribution(
    distribution: Mapping[object, float],
    options: Sequence[str],
) -> Dict[str, float]:
    """Validate, align, and normalize a distribution to an explicit option order."""
    canonical = {canonical_option(key): float(value) for key, value in distribution.items()}
    expected = list(options)
    missing = set(expected) - set(canonical)
    unexpected = set(canonical) - set(expected)
    if missing or unexpected:
        raise ValueError(
            "Distribution keys do not exactly match canonical options "
            f"(missing={sorted(missing)!r}, extra={sorted(unexpected)!r})"
        )

    values = []
    for option in expected:
        value = canonical[option]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid probability for {option!r}: {value!r}")
        values.append(value)
    total = sum(values)
    if total <= 0:
        raise ValueError("Distribution must have positive total mass")
    return {option: value / total for option, value in zip(expected, values)}


class DataLoader:
    """Load and validate survey data plus directed target/evidence units."""

    REQUIRED_PAIR_FIELDS = ("question_id", "country", "conflict_country")

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data: Optional[Dict] = None

    def load_dataset(self) -> Dict:
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        if not isinstance(raw_data, dict) or not raw_data:
            raise ValueError("Dataset must be a non-empty JSON object")

        validated: Dict[str, Dict] = {}
        for question_id, raw_question in raw_data.items():
            options = [canonical_option(option) for option in raw_question.get("options", [])]
            if len(options) < 2 or len(set(options)) != len(options):
                raise ValueError(f"{question_id}: options must be unique and contain at least two values")

            countries: Dict[str, Dict] = {}
            for country, country_data in raw_question.get("countries", {}).items():
                if not isinstance(country, str) or not country.strip():
                    raise ValueError(f"{question_id}: country identifiers must be non-empty strings")
                countries[country] = {
                    **country_data,
                    "distribution": normalize_distribution(
                        country_data.get("distribution", {}), options
                    ),
                }
            if len(countries) < 2:
                raise ValueError(f"{question_id}: at least two country samples are required")

            validated[question_id] = {
                **raw_question,
                "question_id": question_id,
                "options": options,
                "countries": countries,
            }

        self.data = validated
        return validated

    def get_question_pairs(self, pair_config_path: str) -> List[Dict]:
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        with Path(pair_config_path).open("r", encoding="utf-8") as handle:
            raw_pairs = json.load(handle)
        if not isinstance(raw_pairs, list):
            raise ValueError("Pair configuration must be a JSON list")

        seen = set()
        pairs: List[Dict] = []
        for index, pair in enumerate(raw_pairs):
            missing = [field for field in self.REQUIRED_PAIR_FIELDS if not pair.get(field)]
            if missing:
                raise ValueError(f"Pair {index} is missing required fields: {missing}")
            question_id = pair["question_id"]
            country = pair["country"]
            conflict_country = pair["conflict_country"]
            key = (question_id, country, conflict_country)
            if key in seen:
                raise ValueError(f"Duplicate directed unit: {key!r}")
            if country == conflict_country:
                raise ValueError(f"Self-conflict directed unit is invalid: {key!r}")
            if question_id not in self.data:
                raise ValueError(f"Unknown question in pair {index}: {question_id!r}")
            countries = self.data[question_id]["countries"]
            if country not in countries or conflict_country not in countries:
                raise ValueError(f"Pair {index} references an unknown country sample: {key!r}")

            dataset_options = self.data[question_id]["options"]
            pair_options = [canonical_option(option) for option in pair.get("options", dataset_options)]
            if len(pair_options) < 2 or len(set(pair_options)) != len(pair_options):
                raise ValueError(f"Pair {index} has invalid options")
            if pair_options != dataset_options:
                raise ValueError(
                    f"Pair {index} options must exactly match the canonical question schema"
                )
            normalize_distribution(countries[country]["distribution"], pair_options)
            normalize_distribution(countries[conflict_country]["distribution"], pair_options)

            unit_id = pair.get("unit_id", f"{question_id}::{country}=>{conflict_country}")
            pairs.append({**pair, "unit_id": unit_id, "options": pair_options})
            seen.add(key)
        return pairs

    def get_response_distribution(
        self,
        question_id: str,
        country: str,
        options: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        self._require_loaded()
        selected_options = list(options or self.data[question_id]["options"])
        return normalize_distribution(
            self.data[question_id]["countries"][country]["distribution"],
            selected_options,
        )

    def get_question_text(self, question_id: str) -> str:
        self._require_loaded()
        return self.data[question_id]["question"]

    def get_options(self, question_id: str) -> List[str]:
        self._require_loaded()
        return list(self.data[question_id]["options"])

    def get_country_display_name(self, question_id: str, country: str) -> str:
        self._require_loaded()
        mapping = self.data[question_id].get("country_display_names", {})
        return mapping.get(country, country)

    def _require_loaded(self) -> None:
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

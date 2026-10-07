#!/usr/bin/env python3
"""Build a conservative v2 dataset and balanced directed-pair manifest.

The source archive does not contain raw survey years, sample sizes, or per-wave
answer-schema metadata. This repair therefore excludes explicitly ineligible
samples and quarantines ambiguous mappings rather than guessing aliases.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import Metrics


INELIGIBLE_SAMPLE_RE = re.compile(
    r"\((?:Non-national|Old national) sample\)", re.IGNORECASE
)
CURRENT_SAMPLE_SUFFIX = " (Current national sample)"
MIN_SUBSTANTIVE_MASS = 0.5

# Manually audited questions containing unrelated answer schemas in one record.
MIXED_SCHEMA_QUESTIONS = {
    "Q0001", "Q0155", "Q0229", "Q0256", "Q0287", "Q0297",
    "Q0304", "Q0321", "Q0371", "Q0452", "Q0509", "Q0581",
    "Q0624", "Q0672", "Q0692", "Q0768", "Q0821", "Q0860",
    "Q0911", "Q0963", "Q1042", "Q1103", "Q1118",
}

# Semantically related but non-identical schemas. Keep quarantined until a
# human-reviewed, bijective alias manifest is added.
ALIAS_REVIEW_QUESTIONS = {
    "Q0006", "Q0193", "Q0280", "Q0391", "Q0545", "Q0719", "Q1133",
}

# The prompt names a country-specific ethnic group that is absent from the
# archive.  The observed distributions therefore need not describe the same
# semantic target across countries.
COUNTRY_SPECIFIC_PLACEHOLDER_QUESTIONS = {"Q0550"}

# Relative dates cannot be resolved because the archive omits survey wave/year.
UNRESOLVED_RELATIVE_TIME_QUESTIONS = {"Q0311", "Q0475", "Q1124"}

# These are crosstab total/sample-size columns accidentally parsed as answers.
TABLE_TOTAL_OPTION_RE = re.compile(r"^total\s+n\s*=\s*$", re.IGNORECASE)

# Neutralize deictic country wording so baseline/evidence-only prompts do not
# acquire a target-country cue merely by resolving "your country" differently.
QUESTION_TEXT_REPAIRS = {
    "Q0154": (
        "Is it more important for the respondent's country to have strong "
        "economic ties with China or with the United States?"
    ),
    "Q0160": (
        "How well does the statement ‘The military is under the control of "
        "civilian leaders’ describe the respondent's country?"
    ),
    "Q0268": (
        "Do you think China's power and influence is a major threat, a minor "
        "threat, or not a threat to the respondent's country?"
    ),
    "Q0410": (
        "Overall, do you think of China as more of a partner of the "
        "respondent's country, more of an enemy of the respondent's country, "
        "or neither?"
    ),
    "Q0595": (
        "Overall, do you think of the U.S. as more of a partner of the "
        "respondent's country, more of an enemy of the respondent's country, "
        "or neither?"
    ),
    "Q0630": (
        "Does the government of China respect the personal freedoms of its people?"
    ),
    "Q0710": (
        "Is the respondent's country making progress in providing drug "
        "treatments to people with HIV, losing ground, or staying about the same?"
    ),
    "Q0914": (
        "Does the government of Russia respect the personal freedoms of its people?"
    ),
    "Q1510": (
        "How would you rate corruption in the respondent's country on a "
        "10-point scale where 1 means there is no corruption and 10 means "
        "there is abundant corruption?"
    ),
}


def repaired_option(question_id: str, value: object) -> str:
    """Canonicalize an option and apply an audited semantic text repair."""

    option = canonical_option(value)
    if question_id == "Q1510":
        option = option.replace("my country", "the respondent's country")
    return option


def canonical_option(value: object) -> str:
    option = value.strip() if isinstance(value, str) else str(value)
    if not option:
        raise ValueError("empty option")
    return option


def is_nonresponse(option: str) -> bool:
    """Identify only unambiguous missing/refused response categories.

    Labels such as ``Both``, ``Neither``, ``None at all``, ``Depends``, and
    volunteered catchalls can be substantive answers.  They are deliberately
    retained; classifying them from spelling alone silently changes a survey's
    construct.
    """
    text = option.strip().lower()
    return bool(
        re.match(
            r"^(?:dk(?:\s*/\s*refused)?\b|don['’]?t know\b|refus(?:e|ed|al)\b|"
            r"no answer\b|not answered?\b|missing\b|can['’]?t choose\b|"
            r"not sure\b|no opinion\b)",
            text,
        )
    )


def display_country(country: str) -> str:
    if country.endswith(CURRENT_SAMPLE_SUFFIX):
        return country[: -len(CURRENT_SAMPLE_SUFFIX)]
    return country


def canonical_distribution(
    raw_distribution: Mapping[object, object],
    options: Sequence[str],
) -> Dict[str, float]:
    distribution = {canonical_option(key): float(value) for key, value in raw_distribution.items()}
    if set(distribution) != set(options):
        raise ValueError("distribution keys do not exactly match canonical options")
    for value in distribution.values():
        if not math.isfinite(value) or value < 0:
            raise ValueError("distribution contains a non-finite or negative value")
    if sum(distribution.values()) <= 0:
        raise ValueError("distribution has no mass")
    return distribution


def normalized_mapping(distribution: Mapping[str, float], options: Sequence[str]) -> Dict[str, float]:
    values = Metrics.normalize_distribution(distribution, options)
    return {option: float(value) for option, value in zip(options, values)}


def repair_dataset(raw_dataset: Mapping[str, dict]) -> Tuple[Dict[str, dict], dict]:
    repaired: Dict[str, dict] = {}
    excluded = Counter()
    sample_counts = Counter()
    source_question_country_units = sum(
        len(question.get("countries", {})) for question in raw_dataset.values()
    )
    source_non_national = sum(
        "(Non-national sample)" in country
        for question in raw_dataset.values()
        for country in question.get("countries", {})
    )
    source_old_national = sum(
        "(Old national sample)" in country
        for question in raw_dataset.values()
        for country in question.get("countries", {})
    )
    source_current_national = sum(
        "(Current national sample)" in country
        for question in raw_dataset.values()
        for country in question.get("countries", {})
    )
    numeric_option_questions = sum(
        any(not isinstance(option, str) for option in question.get("options", []))
        for question in raw_dataset.values()
    )

    for question_id, question in raw_dataset.items():
        if question_id in MIXED_SCHEMA_QUESTIONS:
            excluded["known_mixed_schema"] += 1
            continue
        if question_id in ALIAS_REVIEW_QUESTIONS:
            excluded["alias_mapping_requires_review"] += 1
            continue
        if question_id in COUNTRY_SPECIFIC_PLACEHOLDER_QUESTIONS:
            excluded["country_specific_placeholder_unresolved"] += 1
            continue
        if question_id in UNRESOLVED_RELATIVE_TIME_QUESTIONS:
            excluded["relative_time_without_survey_wave"] += 1
            continue

        try:
            options = [
                repaired_option(question_id, option)
                for option in question.get("options", [])
            ]
        except ValueError:
            excluded["invalid_options"] += 1
            continue
        if len(options) < 2 or len(set(options)) != len(options):
            excluded["invalid_options"] += 1
            continue
        if any(TABLE_TOTAL_OPTION_RE.match(option) for option in options):
            excluded["table_total_parser_artifact"] += 1
            continue

        substantive_options = [option for option in options if not is_nonresponse(option)]
        if len(substantive_options) < 2:
            excluded["fewer_than_two_substantive_options"] += 1
            continue

        eligible = []
        question_invalid = False
        for country, country_data in question.get("countries", {}).items():
            if INELIGIBLE_SAMPLE_RE.search(country):
                sample_counts["explicit_non_national_or_old_removed"] += 1
                continue
            try:
                repaired_distribution = {
                    repaired_option(question_id, option): value
                    for option, value in country_data.get("distribution", {}).items()
                }
                distribution = canonical_distribution(repaired_distribution, options)
            except (TypeError, ValueError):
                question_invalid = True
                break

            total_mass = sum(distribution.values())
            substantive_mass = sum(distribution[option] for option in substantive_options)
            substantive_fraction = substantive_mass / total_mass
            if substantive_mass <= 0 or substantive_fraction <= MIN_SUBSTANTIVE_MASS:
                sample_counts["insufficient_substantive_mass_removed"] += 1
                continue
            support = frozenset(
                option for option in substantive_options if distribution[option] > 0
            )
            eligible.append((country, distribution, support, substantive_fraction))

        if question_invalid:
            excluded["invalid_distribution_schema"] += 1
            continue
        if len(eligible) < 2:
            excluded["fewer_than_two_eligible_samples"] += 1
            continue

        supports = {row[2] for row in eligible}
        if len(supports) != 1:
            excluded["incompatible_positive_option_support"] += 1
            continue
        # Preserve stated response categories even when every retained sample
        # happens to assign one category zero observed mass (for example a
        # scale endpoint). Positive-support equality is used only to detect
        # incompatible per-country schemas.
        effective_options = list(substantive_options)

        countries = {}
        display_names = {}
        for country, distribution, _, substantive_fraction in eligible:
            restricted = {option: distribution[option] for option in effective_options}
            countries[country] = {
                "distribution": normalized_mapping(restricted, effective_options),
                "source_country_label": country,
                "sample_status": (
                    "current_national" if country.endswith(CURRENT_SAMPLE_SUFFIX) else "national"
                ),
                "substantive_mass_original": substantive_fraction,
            }
            display_names[country] = display_country(country)

        repaired_question_text = QUESTION_TEXT_REPAIRS.get(
            question_id, question.get("question", "")
        )
        repaired[question_id] = {
            "question_id": question_id,
            "domain": question.get("domain", "unknown"),
            "question": repaired_question_text,
            "options": effective_options,
            "countries": countries,
            "country_display_names": display_names,
            "repair_status": "exact_shared_response_schema",
        }
        if question_id in QUESTION_TEXT_REPAIRS:
            repaired[question_id]["source_question"] = question.get("question", "")
            repaired[question_id]["text_repair"] = "audited_deictic_or_placeholder_normalization"

    summary = {
        "source_questions": len(raw_dataset),
        "source_question_country_units": source_question_country_units,
        "source_non_national_sample_units": source_non_national,
        "source_old_national_sample_units": source_old_national,
        "source_current_national_sample_units_retained_when_valid": source_current_national,
        "numeric_option_questions_canonicalized": numeric_option_questions,
        "repaired_questions": len(repaired),
        "excluded_questions_by_reason": dict(sorted(excluded.items())),
        "candidate_path_removed_samples_by_reason": dict(sorted(sample_counts.items())),
        "quarantined_mixed_schema_ids": sorted(MIXED_SCHEMA_QUESTIONS),
        "quarantined_alias_review_ids": sorted(ALIAS_REVIEW_QUESTIONS),
        "quarantined_country_specific_placeholder_ids": sorted(
            COUNTRY_SPECIFIC_PLACEHOLDER_QUESTIONS
        ),
        "quarantined_relative_time_ids": sorted(UNRESOLVED_RELATIVE_TIME_QUESTIONS),
        "normalized_question_text_ids": sorted(QUESTION_TEXT_REPAIRS),
        "minimum_substantive_mass_exclusive": MIN_SUBSTANTIVE_MASS,
        "sample_year_and_size_validation": "unavailable_in_source_archive",
    }
    summary["source_current_national_sample_units"] = summary.pop(
        "source_current_national_sample_units_retained_when_valid"
    )
    summary["retained_current_national_sample_units"] = sum(
        country_data.get("sample_status") == "current_national"
        for question in repaired.values()
        for country_data in question["countries"].values()
    )
    return repaired, summary


def candidate_pairs(dataset: Mapping[str, dict], max_per_question: int) -> Iterable[dict]:
    for question_id, question in dataset.items():
        options = question["options"]
        candidates = []
        for country_a, country_b in combinations(sorted(question["countries"]), 2):
            dist_a = question["countries"][country_a]["distribution"]
            dist_b = question["countries"][country_b]["distribution"]
            divergence = Metrics.js_divergence(dist_a, dist_b, keys=options)
            candidates.append(
                {
                    "question_id": question_id,
                    "country_a": country_a,
                    "country_b": country_b,
                    "divergence_bits": divergence,
                    "options": options,
                }
            )
        candidates.sort(
            key=lambda row: (-row["divergence_bits"], row["country_a"], row["country_b"])
        )
        yield from candidates[:max_per_question]


def make_directed_pairs(
    dataset: Mapping[str, dict],
    top_directed: int,
    max_per_question: int,
) -> list[dict]:
    if top_directed <= 0 or top_directed % 2:
        raise ValueError("top_directed must be a positive even number")
    undirected = list(candidate_pairs(dataset, max_per_question))
    undirected.sort(
        key=lambda row: (
            -row["divergence_bits"],
            row["question_id"],
            row["country_a"],
            row["country_b"],
        )
    )
    selected = undirected[: top_directed // 2]

    directed = []
    for pair in selected:
        question = dataset[pair["question_id"]]
        for country, conflict_country in (
            (pair["country_a"], pair["country_b"]),
            (pair["country_b"], pair["country_a"]),
        ):
            unit_id = f"{pair['question_id']}::{country}=>{conflict_country}"
            directed.append(
                {
                    "unit_id": unit_id,
                    "question_id": pair["question_id"],
                    "country": country,
                    "country_display": question["country_display_names"][country],
                    "conflict_country": conflict_country,
                    "conflict_country_display": question["country_display_names"][conflict_country],
                    "options": pair["options"],
                    "divergence_bits": pair["divergence_bits"],
                    "mapping_status": "exact_shared_response_schema",
                }
            )
    return directed


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset_v1.json"))
    parser.add_argument("--dataset-output", type=Path, default=Path("data/processed/dataset_v2.json"))
    parser.add_argument("--pairs-output", type=Path, default=Path("data/pairs/country_pairs_v2.json"))
    parser.add_argument(
        "--smoke-pairs-output",
        type=Path,
        default=Path("data/pairs/country_pairs_smoke.json"),
    )
    parser.add_argument("--summary-output", type=Path, default=Path("data/audit/data_repair_summary.json"))
    parser.add_argument("--top-directed", type=int, default=200)
    parser.add_argument("--max-pairs-per-question", type=int, default=3)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        raw_dataset = json.load(handle)
    repaired, summary = repair_dataset(raw_dataset)
    pairs = make_directed_pairs(
        repaired,
        top_directed=args.top_directed,
        max_per_question=args.max_pairs_per_question,
    )
    summary["directed_pairs_written"] = len(pairs)
    summary["reciprocal_pair_balance"] = True
    summary["jensen_shannon_base"] = 2

    write_json(args.dataset_output, repaired)
    write_json(args.pairs_output, pairs)
    write_json(args.smoke_pairs_output, pairs[:2])
    write_json(args.summary_output, summary)
    print(
        f"DATA REPAIR PASS: {len(raw_dataset)} -> {len(repaired)} questions; "
        f"{len(pairs)} balanced directed units"
    )


if __name__ == "__main__":
    main()

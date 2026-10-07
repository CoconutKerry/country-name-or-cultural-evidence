"""Run the repaired cultural-alignment experiment pipeline.

Scientific main runs are deliberately approval-gated. The dependency-free
``smoke`` mode exercises exactly two directed units and four conditions, but is
explicitly synthetic and must never be interpreted as a model result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from src.data_loader import DataLoader
from src.metrics import Metrics
from src.model_runner import ModelRunner, SyntheticModelRunner
from src.prompt_builder import PromptBuilder
from src.scoring import recover_canonical_distribution


FOUR_CONDITIONS = (
    "baseline",
    "country_label",
    "population_evidence",
    "conflict",
)
EVIDENCE_PERCENT_DECIMALS = 6


def setup_logging(log_level: str = "INFO", log_path: str = "experiments/experiment.log") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )


def load_config(config_path: str) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML object")
    config.setdefault("shuffle_options", False)
    return config


def prepare_evidence_distribution(
    distribution: Mapping[str, float],
    options: Sequence[str],
) -> Dict[str, float]:
    """Round evidence deterministically while preserving an exact 100% total."""

    normalized = Metrics.normalize_distribution(distribution, options)
    ticks_per_percent = 10**EVIDENCE_PERCENT_DECIMALS
    total_ticks = 100 * ticks_per_percent
    quotas = [float(probability) * total_ticks for probability in normalized]
    ticks = [math.floor(quota) for quota in quotas]
    remaining = total_ticks - sum(ticks)
    if not 0 <= remaining <= len(options):
        raise RuntimeError("evidence rounding produced an invalid remainder")
    allocation_order = sorted(
        range(len(options)),
        key=lambda index: (-(quotas[index] - ticks[index]), index),
    )
    for index in allocation_order[:remaining]:
        ticks[index] += 1
    return {
        option: tick_count / total_ticks
        for option, tick_count in zip(options, ticks)
    }


def format_evidence(distribution: Mapping[str, float], options: Sequence[str]) -> str:
    """Format the exact rounded evidence distribution presented to the model."""

    presented = prepare_evidence_distribution(distribution, options)
    label_map = PromptBuilder.build_label_map(options)
    parts = []
    for label, option in label_map.items():
        parts.append(
            f"{label} ({option}): "
            f"{100.0 * presented[option]:.{EVIDENCE_PERCENT_DECIMALS}f}%"
        )
    return ", ".join(parts)


def shuffle_options(options: Sequence[str], seed: int) -> List[str]:
    shuffled = list(options)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def stable_unit_seed(global_seed: int, unit_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}\0{unit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def file_identity(path: str) -> Dict[str, Any]:
    """Return a small reproducibility record for a repository file."""

    file_path = Path(path)
    if not file_path.is_file():
        return {"path": path, "sha256": None, "status": "missing"}
    return {
        "path": path,
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        "status": "present",
    }


def reorder_distribution(
    distribution: Mapping[str, float],
    displayed_options: Sequence[str],
) -> Dict[str, float]:
    """Return a validated semantic distribution in displayed option order."""
    values = Metrics.normalize_distribution(distribution, displayed_options)
    return {option: float(value) for option, value in zip(displayed_options, values)}


def select_models(config: Mapping[str, Any], mode: str) -> List[Dict[str, Any]]:
    experiment = config.get("experiment", {}).get(mode, {})
    requested_names = experiment.get("models", [])
    if not isinstance(requested_names, list) or not requested_names:
        raise ValueError(f"experiment.{mode}.models must be a non-empty list")
    configured = {model["name"]: model for model in config.get("models", [])}
    missing = [name for name in requested_names if name not in configured]
    if missing:
        raise ValueError(f"Models are not defined in config.models: {missing}")
    return [configured[name] for name in requested_names]


def make_runner(model_config: Mapping[str, Any], config: Mapping[str, Any]) -> ModelRunner:
    backend = model_config.get("backend", config.get("inference", {}).get("backend", "huggingface"))
    if backend == "deterministic_smoke":
        runner = SyntheticModelRunner(seed=int(config.get("seed", 0)))
        runner.model_name = str(model_config["name"])
        runner.hub_id = str(model_config.get("hub_id", "synthetic://deterministic-hash"))
        return runner
    if backend != "huggingface":
        raise ValueError(f"Unsupported model backend: {backend!r}")
    revision = model_config.get("revision")
    tokenizer_revision = model_config.get("tokenizer_revision", revision)
    if not revision or not tokenizer_revision:
        raise ValueError(
            f"{model_config['name']}: exact model and tokenizer revisions are required"
        )
    return ModelRunner(
        str(model_config["name"]),
        str(model_config["hub_id"]),
        revision=str(revision),
        tokenizer_revision=str(tokenizer_revision),
        use_chat_template=bool(model_config.get("use_chat_template", True)),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        model_load_kwargs=model_config.get("model_load_kwargs"),
    )


def validate_smoke_contract(
    config: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
) -> None:
    smoke = config.get("experiment", {}).get("smoke", {})
    conditions = config.get("conditions", [])
    keys = {(pair["question_id"], pair["country"], pair["conflict_country"]) for pair in pairs}
    if len(pairs) != 2 or len(keys) != 2:
        raise ValueError("Smoke mode requires exactly two unique directed units")
    if len(models) != 1 or models[0].get("backend") != "deterministic_smoke":
        raise ValueError("Smoke mode requires exactly one deterministic synthetic model")
    if list(conditions) != list(FOUR_CONDITIONS):
        raise ValueError(f"Smoke conditions must be exactly {FOUR_CONDITIONS!r} in order")
    if smoke.get("synthetic_backend") is not True or smoke.get("scientific_inference") is not False:
        raise ValueError("Smoke metadata must explicitly mark synthetic, non-scientific inference")
    if smoke.get("expected_records") != 8:
        raise ValueError("Smoke mode must declare exactly eight expected records")


def build_prompt(
    prompt_builder: PromptBuilder,
    condition: str,
    question: str,
    options: Sequence[str],
    country_display: str,
    target_distribution: Mapping[str, float],
    conflict_distribution: Mapping[str, float],
) -> str:
    if condition == "baseline":
        return prompt_builder.build_prompt(condition, question, options)
    if condition == "country_label":
        return prompt_builder.build_prompt(
            condition, question, options, country=country_display
        )
    if condition == "population_evidence":
        return prompt_builder.build_prompt(
            condition,
            question,
            options,
            evidence=format_evidence(target_distribution, options),
        )
    if condition == "conflict":
        return prompt_builder.build_prompt(
            condition,
            question,
            options,
            country=country_display,
            evidence=format_evidence(conflict_distribution, options),
        )
    if condition == "persona":
        return prompt_builder.build_prompt(
            condition, question, options, country=country_display
        )
    if condition == "evidence_numeric":
        return prompt_builder.build_prompt(
            condition,
            question,
            options,
            evidence_numeric=format_evidence(target_distribution, options),
        )
    if condition == "evidence_incomplete":
        complete = format_evidence(target_distribution, options).split(", ")
        return prompt_builder.build_prompt(
            condition,
            question,
            options,
            related_info=", ".join(complete[:2]),
        )
    if condition in {"baseline_v2", "baseline_v3"}:
        return prompt_builder.build_prompt(condition, question, options)
    if condition == "evidence_v2":
        return prompt_builder.build_prompt(
            condition,
            question,
            options,
            evidence=format_evidence(target_distribution, options),
        )
    raise ValueError(f"Unsupported condition: {condition!r}")


def run_experiment(
    config: Dict[str, Any],
    mode: str,
    output_path: str,
    *,
    allow_full_run: bool = False,
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    if mode != "smoke" and not allow_full_run:
        raise PermissionError(
            "Every non-smoke model run is approval-gated. Re-run with "
            "--allow-full-run only after explicit approval."
        )

    package_lock = file_identity("requirements-inference.lock")
    if mode != "smoke" and package_lock["status"] != "present":
        raise FileNotFoundError(
            "Scientific runs require requirements-inference.lock before data or model setup"
        )

    data_loader = DataLoader(config["data"]["dataset_path"])
    data_loader.load_dataset()
    all_pairs = data_loader.get_question_pairs(config["data"]["pairs_path"])
    run_settings = config.get("experiment", {}).get(mode, {})
    pair_limit = int(run_settings.get("question_pairs", 0))
    if pair_limit <= 0:
        raise ValueError(f"experiment.{mode}.question_pairs must be positive")
    pair_config = all_pairs[:pair_limit]
    if len(pair_config) != pair_limit:
        raise ValueError(f"Requested {pair_limit} units but only {len(pair_config)} are available")

    model_configs = select_models(config, mode)
    conditions = list(config.get("conditions", []))
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be a non-empty list without duplicates")
    if mode == "smoke":
        validate_smoke_contract(config, pair_config, model_configs)

    shuffle = bool(config.get("shuffle_options", False))
    global_seed = int(config.get("seed", 0))
    score_temperature = float(config.get("inference", {}).get("temperature", 0.0))
    synthetic_backend = any(
        model.get("backend", config.get("inference", {}).get("backend", "huggingface"))
        == "deterministic_smoke"
        for model in model_configs
    )
    if synthetic_backend and mode != "smoke":
        raise ValueError("Synthetic backends are permitted only in smoke mode")

    logger.info(
        "Starting %s with %d directed units, %d model(s), and %d condition(s)",
        mode,
        len(pair_config),
        len(model_configs),
        len(conditions),
    )

    payload: Dict[str, Any] = {
        "metadata": {
            "mode": mode,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "num_pairs": len(pair_config),
            "directed_unit_key": ["question_id", "country", "conflict_country"],
            "models": [model["name"] for model in model_configs],
            "conditions": conditions,
            "shuffle_options": shuffle,
            "jensen_shannon_base": 2,
            "scoring_method": "complete_option_label_continuation_log_likelihood",
            "prompt_template": "invariant_core_v2_with_optional_cue_blocks",
            "evidence_percent_decimals": EVIDENCE_PERCENT_DECIMALS,
            "synthetic_backend": synthetic_backend,
            "scientific_inference": not synthetic_backend,
            "package_lock": package_lock,
            "data_artifacts": {
                "dataset": file_identity(config["data"]["dataset_path"]),
                "pair_manifest": file_identity(config["data"]["pairs_path"]),
                "source_dataset": file_identity("data/processed/dataset_v1.json"),
                "repair_summary": file_identity("data/audit/data_repair_summary.json"),
            },
            "model_runtime": {},
        },
        "results": [],
    }
    prompt_builder = PromptBuilder()

    for model_config in model_configs:
        runner = make_runner(model_config, config)
        runner.load_model()
        payload["metadata"]["model_runtime"][model_config["name"]] = (
            runner.get_runtime_metadata()
        )
        logger.info("Running model backend: %s", model_config["name"])

        for pair in pair_config:
            question_id = pair["question_id"]
            country = pair["country"]
            conflict_country = pair["conflict_country"]
            unit_id = pair["unit_id"]
            target_unit_id = f"{question_id}::{country}"
            question = data_loader.get_question_text(question_id)
            canonical_options = list(pair["options"])
            displayed_options = (
                shuffle_options(
                    canonical_options,
                    stable_unit_seed(global_seed, target_unit_id),
                )
                if shuffle
                else list(canonical_options)
            )

            target_canonical = data_loader.get_response_distribution(
                question_id, country, canonical_options
            )
            conflict_canonical = data_loader.get_response_distribution(
                question_id, conflict_country, canonical_options
            )
            target_displayed = reorder_distribution(target_canonical, displayed_options)
            conflict_displayed = reorder_distribution(conflict_canonical, displayed_options)
            target_evidence_displayed = prepare_evidence_distribution(
                target_displayed, displayed_options
            )
            conflict_evidence_displayed = prepare_evidence_distribution(
                conflict_displayed, displayed_options
            )
            country_display = pair.get(
                "country_display",
                data_loader.get_country_display_name(question_id, country),
            )
            conflict_display = pair.get(
                "conflict_country_display",
                data_loader.get_country_display_name(question_id, conflict_country),
            )

            for condition in conditions:
                prompt = build_prompt(
                    prompt_builder,
                    condition,
                    question,
                    displayed_options,
                    country_display,
                    target_evidence_displayed,
                    conflict_evidence_displayed,
                )
                displayed_prediction = runner.predict_distribution(
                    prompt, displayed_options, temperature=score_temperature
                )
                prediction = recover_canonical_distribution(
                    displayed_prediction, canonical_options
                )
                source_evidence_distribution = (
                    conflict_canonical if condition == "conflict" else target_canonical
                )
                presented_evidence_displayed = (
                    conflict_evidence_displayed
                    if condition == "conflict"
                    else target_evidence_displayed
                )
                presented_evidence_distribution = recover_canonical_distribution(
                    presented_evidence_displayed, canonical_options
                )
                scoring = runner.get_last_scoring_metadata()

                payload["results"].append(
                    {
                        "model_name": model_config["name"],
                        "unit_id": unit_id,
                        "target_unit_id": target_unit_id,
                        "question_id": question_id,
                        "country": country,
                        "country_display": country_display,
                        "conflict_country": conflict_country,
                        "conflict_country_display": conflict_display,
                        "condition": condition,
                        "prediction": prediction,
                        "human_distribution": target_canonical,
                        "evidence_distribution": presented_evidence_distribution,
                        "source_evidence_distribution": source_evidence_distribution,
                        "evidence_presented": condition in {"population_evidence", "conflict"},
                        "prompt": prompt,
                        "shuffled": shuffle,
                        "original_options": canonical_options,
                        "used_options": displayed_options,
                        "option_label_map": scoring["label_to_option"],
                        "scoring": scoring,
                    }
                )

    expected_records = len(pair_config) * len(model_configs) * len(conditions)
    if len(payload["results"]) != expected_records:
        raise RuntimeError(
            f"Result cardinality mismatch: expected {expected_records}, got {len(payload['results'])}"
        )
    if mode == "smoke" and expected_records != 8:
        raise RuntimeError("Smoke contract violation: output must contain exactly eight rows")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Results saved to %s", output_file)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot", "main"], default="smoke")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Explicit approval gate for any non-smoke run; never implied by configuration",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    config_path = args.config or ("config_smoke.yaml" if args.mode == "smoke" else "config.yaml")
    config = load_config(config_path)
    if args.output is None:
        base_dir = Path(config["output"]["base_dir"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = base_dir / args.mode / f"results_{args.mode}_{timestamp}.json"
    else:
        output = Path(args.output)
    run_experiment(
        config,
        args.mode,
        str(output),
        allow_full_run=args.allow_full_run,
    )


if __name__ == "__main__":
    main()

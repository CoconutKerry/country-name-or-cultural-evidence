#!/usr/bin/env python3
"""Run exactly two reciprocal Qwen GGUF CPU units across four conditions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.gguf_runner import (
    GGUFRunner,
    QWEN25_DEFAULT_SYSTEM_MESSAGE,
    serialize_qwen25_chatml,
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
from src.scoring import option_labels, recover_canonical_distribution


MODEL_REPOSITORY = "Qwen/Qwen2.5-7B-Instruct-GGUF"
MODEL_REVISION = "bb5d59e06d9551d752d08b292a50eb208b07ab1f"
MODEL_IDENTIFIER = f"{MODEL_REPOSITORY}:Q4_K_M"
QUANTIZATION = "Q4_K_M"
REMOTE_MODEL_LAST_MODIFIED = "2024-09-20T06:38:28.000Z"
REMOTE_WEIGHT_COMMIT = "293ca9a10157b0e5fc5cb32af8b636a88bede891"
MODEL_SHARDS = (
    {
        "filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "size_bytes": 3_993_201_344,
        "sha256": "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
        "role": "selected_entrypoint",
    },
    {
        "filename": "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        "size_bytes": 689_872_288,
        "sha256": "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a",
        "role": "companion_shard",
    },
)
INTERNAL_CONDITIONS = (
    "baseline",
    "country_label",
    "population_evidence",
    "conflict",
)
OUTPUT_CONDITION = {"population_evidence": "evidence"}
EXPECTED_OUTPUT_CONDITIONS = ("baseline", "country_label", "evidence", "conflict")
EXPECTED_UNITS = {
    ("Q1322", "Maldives", "South Korea"),
    ("Q1322", "South Korea", "Maldives"),
}
MODEL_NAME = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
SEED = 42
NORMALIZATION_TOLERANCE = 1e-8


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_text(path, payload)


def command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def memory_information() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
    }


def cpu_information() -> dict[str, Any]:
    model_name = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
                break
    except (OSError, IndexError):
        pass
    return {
        "model": model_name or platform.processor() or None,
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "lscpu": command_output(["lscpu"]),
    }


def environment_information(repository_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(repository_root)
    nvidia_smi = shutil.which("nvidia-smi")
    return {
        "captured_at_utc": now_utc(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_information(),
        "memory": memory_information(),
        "disk": {
            "path": str(repository_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "nvidia_smi_available": nvidia_smi is not None,
        "nvidia_smi_path": nvidia_smi,
        "cuda_used": False,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
    }


def verified_model_files(model_dir: Path) -> list[dict[str, Any]]:
    verified = []
    for expected in MODEL_SHARDS:
        path = model_dir / expected["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"official GGUF shard is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected["size_bytes"]:
            raise RuntimeError(
                f"GGUF size mismatch for {path.name}: {actual_size} != {expected['size_bytes']}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"GGUF SHA-256 mismatch for {path.name}: {actual_hash} != {expected['sha256']}"
            )
        verified.append(
            {
                **expected,
                "local_path": str(path.resolve()),
                "actual_size_bytes": actual_size,
                "actual_sha256": actual_hash,
                "verified": True,
            }
        )
    return verified


def file_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def prepare_unit(loader: DataLoader, pair: Mapping[str, Any]) -> dict[str, Any]:
    question_id = str(pair["question_id"])
    label_country = str(pair["country"])
    evidence_country = str(pair["conflict_country"])
    canonical_options = [str(option) for option in pair["options"]]
    target_unit_id = f"{question_id}::{label_country}"
    displayed_options = shuffle_options(
        canonical_options,
        stable_unit_seed(SEED, target_unit_id),
    )
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


def assert_probability_distribution(distribution: Mapping[str, Any], options: Sequence[str]) -> None:
    if set(distribution) != set(options):
        raise AssertionError("prediction keys do not match original semantic options")
    values = [float(distribution[option]) for option in options]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise AssertionError("prediction contains a negative or non-finite probability")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
        raise AssertionError("prediction probabilities do not sum to one")


def enrich_unit_metrics(rows: list[dict[str, Any]]) -> None:
    by_condition = {row["condition"]: row for row in rows}
    if set(by_condition) != set(EXPECTED_OUTPUT_CONDITIONS):
        raise AssertionError("directed unit does not contain the four requested conditions")
    baseline = by_condition["baseline"]["normalized_prediction"]
    country = by_condition["country_label"]["normalized_prediction"]
    evidence = by_condition["evidence"]["normalized_prediction"]
    conflict = by_condition["conflict"]["normalized_prediction"]
    label_human = rows[0]["label_country_human_distribution"]
    evidence_human = rows[0]["evidence_country_human_distribution"]
    components = {
        "baseline_to_label_country": Metrics.js_divergence(baseline, label_human),
        "country_label_to_label_country": Metrics.js_divergence(country, label_human),
        "evidence_to_label_country": Metrics.js_divergence(evidence, label_human),
        "conflict_to_label_country": Metrics.js_divergence(conflict, label_human),
        "conflict_to_evidence_country": Metrics.js_divergence(conflict, evidence_human),
    }
    unit_metrics = {
        "country_influence": Metrics.country_influence(baseline, country, label_human),
        "evidence_influence": Metrics.evidence_influence(baseline, evidence, label_human),
        "evidence_override": Metrics.evidence_override(conflict, evidence_human, label_human),
    }
    for row in rows:
        prediction = row["normalized_prediction"]
        row["jensen_shannon_base"] = 2
        row["jensen_shannon_measure"] = "divergence_bits"
        row["base2_jensen_shannon_distances"] = {
            "prediction_to_label_country": Metrics.js_divergence(
                prediction, label_human
            ),
            "prediction_to_evidence_country": Metrics.js_divergence(
                prediction, evidence_human
            ),
            **components,
        }
        row.update(unit_metrics)


def synthetic_fixture_index() -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    fixture_path = PROJECT_ROOT / "experiments/smoke/results_smoke.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise AssertionError("existing synthetic smoke fixture is malformed")
    return {
        (
            str(row["question_id"]),
            str(row["country"]),
            str(row["conflict_country"]),
            str(row["condition"]),
        ): row
        for row in rows
    }


def validate_rows(
    rows: list[dict[str, Any]],
    runtime: Mapping[str, Any],
    fixture: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> tuple[dict[str, bool], list[dict[str, Any]], dict[str, Any]]:
    keys = [
        (
            row["question_id"],
            row["label_country"],
            row["evidence_country"],
            row["condition"],
        )
        for row in rows
    ]
    units = {(question, label, evidence) for question, label, evidence, _ in keys}
    conditions_by_unit: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for question, label, evidence, condition in keys:
        conditions_by_unit[(question, label, evidence)].append(condition)

    row_checks = []
    shared_word_checks = []
    for index, row in enumerate(rows):
        options = row["original_options"]
        assert_probability_distribution(row["normalized_prediction"], options)
        assert_probability_distribution(
            row["normalized_label_probabilities"], row["displayed_option_labels"]
        )
        scoring = row["scoring_trace"]
        prompt_count = int(scoring["prompt_token_count"])
        labels = row["displayed_option_labels"]
        candidates = scoring["candidates"]
        if set(candidates) != set(labels):
            raise AssertionError(f"row {index} is missing candidate label scores")
        for label in labels:
            candidate = candidates[label]
            token_ids = candidate["token_ids"]
            targets = candidate["target_token_positions"]
            predictive = candidate["predictive_logit_positions"]
            if not token_ids or len(token_ids) != candidate["token_count"]:
                raise AssertionError(f"row {index}, label {label}: invalid tokenization")
            if len(targets) != len(token_ids) or len(predictive) != len(token_ids):
                raise AssertionError(f"row {index}, label {label}: incomplete position trace")
            if targets != list(range(prompt_count, prompt_count + len(token_ids))):
                raise AssertionError(f"row {index}, label {label}: wrong target position")
            if predictive != [position - 1 for position in targets]:
                raise AssertionError(f"row {index}, label {label}: wrong predictive position")
            if predictive[0] != prompt_count - 1:
                raise AssertionError(f"row {index}, label {label}: not scored at answer position")
            if not math.isclose(
                math.fsum(float(value) for value in candidate["token_log_probabilities"]),
                float(candidate["sequence_log_probability"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise AssertionError(f"row {index}, label {label}: incomplete sequence score")
        if not scoring.get("prompt_prefix_verified"):
            raise AssertionError(f"row {index}: prompt-prefix tokenization was not verified")
        if scoring.get("embedded_template_serialization_verified") is not True:
            raise AssertionError(f"row {index}: embedded-template serialization was not verified")
        if row.get("generated_answer") is not None:
            raise AssertionError(f"row {index}: generation/parsing is forbidden")
        if row["structured_messages"] != [
            {"role": "system", "content": QWEN25_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": row["raw_user_prompt"]},
        ]:
            raise AssertionError(f"row {index}: Qwen structured messages are incomplete")
        if row["serialized_chat_templated_prompt"] != serialize_qwen25_chatml(
            row["structured_messages"]
        ):
            raise AssertionError(f"row {index}: Qwen serialization is not byte-exact")
        if not row["serialized_chat_templated_prompt"].startswith(
            "<|im_start|>system\n"
            + QWEN25_DEFAULT_SYSTEM_MESSAGE
            + "<|im_end|>\n"
        ):
            raise AssertionError(f"row {index}: Qwen default system turn is missing")
        if not row["serialized_chat_templated_prompt"].endswith(
            "<|im_start|>assistant\n"
        ):
            raise AssertionError(f"row {index}: Qwen assistant answer boundary is missing")

        label_map = row["displayed_label_to_option"]
        recovered = {
            option: row["normalized_label_probabilities"][label]
            for label, option in label_map.items()
        }
        restored = recover_canonical_distribution(recovered, options)
        if list(restored) != options or any(
            not math.isclose(
                restored[option],
                row["normalized_prediction"][option],
                rel_tol=0.0,
                abs_tol=NORMALIZATION_TOLERANCE,
            )
            for option in options
        ):
            raise AssertionError(f"row {index}: option permutation recovery failed")

        internal_condition = (
            "population_evidence" if row["condition"] == "evidence" else row["condition"]
        )
        fixture_row = fixture[
            (
                row["question_id"],
                row["label_country"],
                row["evidence_country"],
                internal_condition,
            )
        ]
        if row["displayed_options"] != fixture_row["used_options"]:
            raise AssertionError(f"row {index}: displayed permutation differs from smoke fixture")
        if row["raw_user_prompt"] != fixture_row["prompt"]:
            raise AssertionError(f"row {index}: raw prompt differs from smoke fixture")

        collisions: dict[str, list[str]] = defaultdict(list)
        for option in options:
            collisions[option.split()[0].lower()].append(option)
        for first_word, semantic_options in collisions.items():
            if len(semantic_options) < 2:
                continue
            for left_index in range(len(semantic_options)):
                for right_index in range(left_index + 1, len(semantic_options)):
                    left = semantic_options[left_index]
                    right = semantic_options[right_index]
                    option_to_label = {option: label for label, option in label_map.items()}
                    left_label = option_to_label[left]
                    right_label = option_to_label[right]
                    left_probability = row["normalized_prediction"][left]
                    right_probability = row["normalized_prediction"][right]
                    left_score = row["raw_candidate_scores"][left_label]
                    right_score = row["raw_candidate_scores"][right_label]
                    if left_label == right_label:
                        raise AssertionError("same-prefix semantic options share a displayed label")
                    if candidates[left_label]["token_ids"] == candidates[right_label]["token_ids"]:
                        raise AssertionError("same-prefix semantic options share a candidate token path")
                    if math.isclose(left_score, right_score, rel_tol=0.0, abs_tol=1e-12):
                        raise AssertionError("same-prefix semantic options received equal label scores")
                    if math.isclose(
                        left_probability, right_probability, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise AssertionError("same-prefix semantic options received equal probabilities")
                    shared_word_checks.append(
                        {
                            "row_index": index,
                            "condition": row["condition"],
                            "label_country": row["label_country"],
                            "first_word": first_word,
                            "left_option": left,
                            "right_option": right,
                            "left_displayed_label": left_label,
                            "right_displayed_label": right_label,
                            "left_token_ids": candidates[left_label]["token_ids"],
                            "right_token_ids": candidates[right_label]["token_ids"],
                            "left_sequence_log_probability": left_score,
                            "right_sequence_log_probability": right_score,
                            "left_probability": left_probability,
                            "right_probability": right_probability,
                            "independently_scored": True,
                            "unequal": True,
                        }
                    )
        row_checks.append({"row_index": index, "status": "PASS"})

    orientation = []
    for unit in sorted(units):
        representative = next(
            row
            for row in rows
            if (row["question_id"], row["label_country"], row["evidence_country"])
            == unit
        )
        label_human = representative["label_country_human_distribution"]
        evidence_human = representative["evidence_country_human_distribution"]
        positive = Metrics.evidence_override(evidence_human, evidence_human, label_human)
        negative = Metrics.evidence_override(label_human, evidence_human, label_human)
        if not positive > 0.0 or not negative < 0.0:
            raise AssertionError("Evidence Override orientation is reversed")
        orientation.append(
            {
                "directed_unit": list(unit),
                "prediction_equals_evidence": positive,
                "prediction_equals_label": negative,
                "orientation_passed": True,
            }
        )

    assertions = {
        "genuine_backend_only": all(
            row["backend"] == "llama.cpp"
            and row["model_identifier"] == MODEL_IDENTIFIER
            and not row["synthetic"]
            for row in rows
        )
        and runtime.get("backend") == "llama.cpp"
        and runtime.get("synthetic") is False,
        "exactly_eight_rows": len(rows) == 8,
        "unique_directed_condition_keys": len(keys) == len(set(keys)) == 8,
        "exact_reciprocal_units": units == EXPECTED_UNITS,
        "all_four_conditions_per_unit": all(
            Counter(conditions_by_unit[unit]) == Counter(EXPECTED_OUTPUT_CONDITIONS)
            for unit in units
        ),
        "embedded_qwen_chat_template": runtime.get("chat_template_source")
        == "embedded_gguf_metadata"
        and bool(runtime.get("chat_template"))
        and all(
            row["scoring_trace"].get("embedded_template_serialization_verified") is True
            for row in rows
        ),
        "candidate_tokenization_valid": len(row_checks) == 8,
        "correct_answer_position": len(row_checks) == 8,
        "all_labels_scored_from_full_vocabulary": runtime.get("candidate_scoring")
        == "full_vocabulary_low_level_logits",
        "no_generation_or_answer_parsing": runtime.get("generated_answer_parsing") is False
        and all(row.get("generated_answer") is None for row in rows),
        "probabilities_valid": len(row_checks) == 8,
        "option_permutation_recovery": len(row_checks) == 8,
        "shared_first_word_independent": len(shared_word_checks) == 8,
        "directed_units_not_overwritten": len(units) == 2 and len(rows) == 8,
        "evidence_override_orientation": len(orientation) == 2,
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"genuine smoke assertions failed: {failed}")
    return assertions, shared_word_checks, {"checks": orientation}


def build_report(analysis: Mapping[str, Any]) -> str:
    metadata = analysis["metadata"]
    environment = metadata["environment"]
    model = metadata["model"]
    llama_cpp = metadata["llama_cpp"]
    assertions = analysis["assertions"]
    lines = [
        "# Genuine Qwen Q4_K_M CPU Smoke Report",
        "",
        f"**Status: {analysis['status']}**",
        "",
        "This was a bounded genuine-model CPU smoke only: one official Qwen GGUF, "
        "two reciprocal directed units, and four conditions (8 saved rows). No answer "
        "was generated or parsed, no other model was loaded, and the 200-unit/full "
        "experiment was not started.",
        "",
        "## Model and runtime provenance",
        "",
        f"- Hugging Face repository: `{model['repository']}`",
        f"- Immutable repository revision: `{model['revision']}`",
        f"- Quantization: `{model['quantization']}`",
        f"- Selected entrypoint: `{model['selected_entrypoint_file']}`",
        f"- Official split total: {model['total_size_bytes']:,} bytes",
        f"- llama.cpp version: `{llama_cpp['version']}`",
        f"- llama.cpp commit: `{llama_cpp['commit']}`",
        f"- Chat template source: `{llama_cpp['chat_template_source']}`",
        f"- Chat template SHA-256: `{llama_cpp['chat_template_sha256']}`",
        "- Qwen default system turn: explicitly materialized from the embedded template",
        "- Serialization verification: byte-exact non-tool Qwen2.5 ChatML expansion",
        "",
        "| GGUF shard | Bytes | SHA-256 | Verified |",
        "|---|---:|---|---|",
    ]
    for shard in model["files"]:
        lines.append(
            f"| `{shard['filename']}` | {shard['actual_size_bytes']:,} | "
            f"`{shard['actual_sha256']}` | {shard['verified']} |"
        )
    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{environment['python_version'].splitlines()[0]}`",
            f"- CPU: `{environment['cpu']['model']}`",
            f"- Logical CPUs: {environment['cpu']['logical_cpu_count']}",
            f"- RAM total: {environment['memory']['total_bytes']:,} bytes",
            f"- RAM available at capture: {environment['memory']['available_bytes']:,} bytes",
            f"- Disk free at capture: {environment['disk']['free_bytes']:,} bytes",
            f"- CUDA used: {environment['cuda_used']}",
            "",
            "## Assertions",
            "",
            "| Assertion | Result |",
            "|---|---|",
        ]
    )
    for name, passed in assertions.items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Directed-unit metrics",
            "",
            "All Jensen–Shannon values are base-2 divergences in bits.",
            "",
            "| Label country | Evidence country | Country Influence | Evidence Influence | Evidence Override |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for unit in analysis["directed_unit_metrics"]:
        lines.append(
            f"| {unit['label_country']} | {unit['evidence_country']} | "
            f"{unit['country_influence']:.9f} | {unit['evidence_influence']:.9f} | "
            f"{unit['evidence_override']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Same-first-word audit",
            "",
            "`Strongly agree` and `Strongly disagree` were mapped to distinct displayed "
            "labels and contextual token paths in every row. Their label sequence scores "
            "and restored semantic probabilities were computed independently and were "
            "unequal in all 8 rows. Full per-row values are stored in `analysis.json`.",
            "",
            "All displayed labels (`A`–`D`) were one token for this tokenizer. The "
            "backend implements full multi-token conditional sequence scoring and "
            "candidate-cache rollback, but that branch was not dynamically activated "
            "by this particular label set.",
            "",
            "## Pre-package QA note",
            "",
            "Independent review rejected an earlier draft output set because the legacy "
            "llama.cpp template helper omitted Qwen's implicit default system turn for a "
            "user-only message. The final eight rows in this package were rerun after "
            "materializing that exact embedded-template system turn and adding a byte-exact "
            "serialization assertion. The rejected rows are not included.",
            "",
            "## Scope boundary",
            "",
            "Both full-run approval gates retained their pre-run hashes. No full-run cell, "
            "additional checkpoint, or 200-unit experiment was executed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/smoke_genuine_qwen_gguf",
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--ctx-size", type=int, default=4096)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    analysis_path = output_dir / "analysis.json"
    run_log_path = output_dir / "run.log"
    report_path = PROJECT_ROOT / "audit/GENUINE_QWEN_GGUF_SMOKE_REPORT.md"
    start = time.monotonic()
    started_at = now_utc()

    protected_gate_files = [
        PROJECT_ROOT / "src/colab_runner.py",
        PROJECT_ROOT / "scripts/build_colab_notebook.py",
        PROJECT_ROOT / "cultural_alignment_audit_colab.ipynb",
    ]
    gate_hashes_before = file_hashes(protected_gate_files)

    with run_log_path.open("w", encoding="utf-8") as run_log:
        def log(message: str) -> None:
            run_log.write(f"{now_utc()} {message}\n")
            run_log.flush()

        log("BEGIN genuine Qwen GGUF CPU smoke")
        log("Scope: one model, two reciprocal directed units, four conditions")
        environment = environment_information(PROJECT_ROOT)
        log(f"Environment: {json.dumps(environment, ensure_ascii=False)}")
        log("Verifying both official GGUF shards by exact size and SHA-256")
        model_files = verified_model_files(args.model_dir.resolve())
        log("GGUF size and SHA-256 verification: PASS")

        llama_dir = args.llama_cpp_dir.resolve()
        llama_commit = command_output(["git", "-C", str(llama_dir), "rev-parse", "HEAD"])
        llama_commit_date = command_output(
            ["git", "-C", str(llama_dir), "log", "-1", "--format=%cI"]
        )
        if not llama_commit:
            raise RuntimeError("could not record the llama.cpp commit")
        build_cache = llama_dir / "build-cpu/CMakeCache.txt"
        build_cache_sha256 = sha256_file(build_cache) if build_cache.is_file() else None
        log(f"llama.cpp commit: {llama_commit}")

        loader = DataLoader(str(PROJECT_ROOT / "data/processed/dataset_v2.json"))
        loader.load_dataset()
        pairs = loader.get_question_pairs(
            str(PROJECT_ROOT / "data/pairs/country_pairs_smoke.json")
        )
        pair_keys = {
            (str(pair["question_id"]), str(pair["country"]), str(pair["conflict_country"]))
            for pair in pairs
        }
        if len(pairs) != 2 or pair_keys != EXPECTED_UNITS:
            raise AssertionError(f"unexpected smoke unit fixture: {pair_keys}")
        fixture = synthetic_fixture_index()

        rows: list[dict[str, Any]] = []
        runner = GGUFRunner(
            helper_path=args.helper,
            model_path=Path(model_files[0]["local_path"]),
            model_identifier=MODEL_IDENTIFIER,
            repository=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            quantization=QUANTIZATION,
            threads=args.threads,
            context_size=args.ctx_size,
        )
        runner.load_model(run_log)
        runtime = runner.get_runtime_metadata()
        if runtime.get("backend") != "llama.cpp" or runtime.get("synthetic") is not False:
            raise AssertionError("synthetic, mock, or fallback backends are forbidden")
        log(
            f"Loaded {MODEL_IDENTIFIER} with llama.cpp {runtime['llama_version']}; "
            "GPU layers=0"
        )
        independent_rescore: dict[str, Any] | None = None
        try:
            builder = PromptBuilder()
            for pair in pairs:
                unit = prepare_unit(loader, pair)
                unit_rows: list[dict[str, Any]] = []
                for internal_condition in INTERNAL_CONDITIONS:
                    condition = OUTPUT_CONDITION.get(internal_condition, internal_condition)
                    log(f"Scoring {unit['unit_id']} / {condition}")
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
                        displayed_prediction,
                        unit["canonical_options"],
                    )
                    assert_probability_distribution(prediction, unit["canonical_options"])
                    scoring = runner.get_last_scoring_metadata()
                    labels = option_labels(len(unit["displayed_options"]))
                    label_probabilities = {
                        label: float(scoring["normalized_label_probabilities"][label])
                        for label in labels
                    }
                    raw_scores = {
                        label: float(scoring["raw_label_log_scores"][label])
                        for label in labels
                    }
                    label_to_option = {
                        label: str(option)
                        for label, option in zip(labels, unit["displayed_options"])
                    }
                    source_evidence = (
                        unit["evidence_human"]
                        if internal_condition == "conflict"
                        else unit["label_human"]
                    )
                    presented_displayed = (
                        unit["conflict_evidence_displayed"]
                        if internal_condition == "conflict"
                        else unit["label_evidence_displayed"]
                    )
                    presented_canonical = recover_canonical_distribution(
                        presented_displayed,
                        unit["canonical_options"],
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
                    row = {
                        "schema_version": "genuine_qwen_gguf_smoke_v1",
                        "question_id": unit["key"][0],
                        "label_country": unit["key"][1],
                        "evidence_country": unit["key"][2],
                        "condition": condition,
                        "unit_id": unit["unit_id"],
                        "target_unit_id": unit["target_unit_id"],
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
                            "files": [shard["filename"] for shard in model_files],
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
                        "presented_evidence_distribution": presented_canonical,
                        "source_evidence_distribution": source_evidence,
                        "evidence_presented": internal_condition
                        in {"population_evidence", "conflict"},
                        "scoring_method": "full_contextual_option_label_sequence_log_probability",
                        "scoring_trace": scoring,
                        "generated_answer": None,
                    }
                    unit_rows.append(row)
                    rows.append(row)
                    write_jsonl(results_path, rows)
                    log(
                        f"Saved row {len(rows)}/8; scores={json.dumps(raw_scores)}; "
                        f"probabilities={json.dumps(label_probabilities)}"
                    )
                enrich_unit_metrics(unit_rows)
                write_jsonl(results_path, rows)
                log(f"Completed and saved directed unit {unit['unit_id']}")

            first = rows[0]
            independent_rescore = runner.score_messages(
                first["structured_messages"], first["displayed_option_labels"]
            )
            for label in first["displayed_option_labels"]:
                original_score = first["raw_candidate_scores"][label]
                repeated_score = independent_rescore["candidates"][label][
                    "sequence_log_probability"
                ]
                if not math.isclose(
                    original_score, repeated_score, rel_tol=0.0, abs_tol=1e-7
                ):
                    raise AssertionError("independent first-row re-score was not reproducible")
            if (
                independent_rescore["serialized_prompt"]
                != first["serialized_chat_templated_prompt"]
                or independent_rescore["prompt_token_ids"]
                != first["scoring_trace"]["prompt_token_ids"]
            ):
                raise AssertionError("independent re-score changed prompt serialization/tokenization")
            log("Independent first-row low-level re-score: PASS")
        finally:
            runner.close()
        log("Model released; llama.cpp helper stopped")

        assertions, shared_word_checks, orientation = validate_rows(
            rows, runtime, fixture
        )
        assertions["independent_first_row_rescore"] = independent_rescore is not None
        gate_hashes_after = file_hashes(protected_gate_files)
        assertions["full_run_approval_gates_unchanged"] = (
            gate_hashes_before == gate_hashes_after
        )
        failed = [name for name, passed in assertions.items() if not passed]
        if failed:
            raise AssertionError(f"post-run assertions failed: {failed}")

        directed_unit_metrics = []
        seen_units = set()
        for row in rows:
            unit_key = (
                row["question_id"],
                row["label_country"],
                row["evidence_country"],
            )
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)
            directed_unit_metrics.append(
                {
                    "question_id": row["question_id"],
                    "label_country": row["label_country"],
                    "evidence_country": row["evidence_country"],
                    "country_influence": row["country_influence"],
                    "evidence_influence": row["evidence_influence"],
                    "evidence_override": row["evidence_override"],
                }
            )

        completed_at = now_utc()
        analysis = {
            "status": "PASS",
            "scope": {
                "models": 1,
                "directed_units": 2,
                "conditions_per_unit": 4,
                "saved_rows": len(rows),
                "full_experiment_started": False,
                "additional_models_started": False,
            },
            "metadata": {
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "elapsed_seconds": time.monotonic() - start,
                "environment": environment,
                "model": {
                    "identifier": MODEL_IDENTIFIER,
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "remote_last_modified": REMOTE_MODEL_LAST_MODIFIED,
                    "weight_upload_commit": REMOTE_WEIGHT_COMMIT,
                    "quantization": QUANTIZATION,
                    "selected_entrypoint_file": MODEL_SHARDS[0]["filename"],
                    "companion_file": MODEL_SHARDS[1]["filename"],
                    "total_size_bytes": sum(item["size_bytes"] for item in MODEL_SHARDS),
                    "files": model_files,
                },
                "llama_cpp": {
                    "repository": "https://github.com/ggml-org/llama.cpp",
                    "commit": llama_commit,
                    "commit_date": llama_commit_date,
                    "version": runtime["llama_version"],
                    "system_info": runtime["llama_system_info"],
                    "build_cache_sha256": build_cache_sha256,
                    "threads": args.threads,
                    "context_size": runtime["actual_context_size"],
                    "gpu_layers": runtime["gpu_layers"],
                    "chat_template_source": runtime["chat_template_source"],
                    "chat_template_sha256": runtime["chat_template_sha256"],
                    "embedded_chat_template": runtime["chat_template"],
                    "model_description": runtime["model_description"],
                    "loaded_model_size_bytes": runtime["loaded_model_size_bytes"],
                    "model_parameter_count": runtime["model_parameter_count"],
                    "vocabulary_size": runtime["vocabulary_size"],
                },
                "data": {
                    "dataset": "data/processed/dataset_v2.json",
                    "pair_manifest": "data/pairs/country_pairs_smoke.json",
                    "pair_manifest_sha256": sha256_file(
                        PROJECT_ROOT / "data/pairs/country_pairs_smoke.json"
                    ),
                    "seed": SEED,
                    "shuffle_options": True,
                    "directed_unit_key": [
                        "question_id",
                        "label_country",
                        "evidence_country",
                    ],
                },
                "protected_gate_hashes_before": gate_hashes_before,
                "protected_gate_hashes_after": gate_hashes_after,
            },
            "assertions": assertions,
            "row_assertion_count": 8,
            "shared_first_word_check": {
                "status": "PASS",
                "pair": ["Strongly agree", "Strongly disagree"],
                "rows_checked": len(shared_word_checks),
                "details": shared_word_checks,
            },
            "evidence_override_orientation_check": orientation,
            "independent_first_row_rescore": {
                "status": "PASS",
                "request_id": independent_rescore["request_id"]
                if independent_rescore
                else None,
            },
            "directed_unit_metrics": directed_unit_metrics,
            "conditions": list(EXPECTED_OUTPUT_CONDITIONS),
            "result_key_count": len(
                {
                    (
                        row["question_id"],
                        row["label_country"],
                        row["evidence_country"],
                        row["condition"],
                    )
                    for row in rows
                }
            ),
        }
        atomic_write_text(
            analysis_path,
            json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(report_path, build_report(analysis))
        write_jsonl(results_path, rows)
        log("All genuine-model smoke assertions: PASS")
        log("Full experiment started: false")
        log("END genuine Qwen GGUF CPU smoke")

    print(json.dumps({"status": "PASS", "rows": 8, "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()

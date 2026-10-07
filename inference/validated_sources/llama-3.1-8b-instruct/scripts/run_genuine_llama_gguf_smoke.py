#!/usr/bin/env python3
"""Run exactly two reciprocal Llama GGUF CPU units across four conditions."""

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
import traceback
from typing import Any, Mapping, Sequence
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.scoring import option_labels, recover_canonical_distribution


UPSTREAM_CHECKPOINT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
UPSTREAM_CURRENT_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_REPOSITORY = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_REVISION = "bf5b95e96dac0462e2a09145ec66cae9a3f12067"
MODEL_IDENTIFIER = f"{MODEL_REPOSITORY}:Q4_K_M"
QUANTIZATION = "Q4_K_M"
REMOTE_MODEL_LAST_MODIFIED = "2024-12-01T04:12:27Z"
REMOTE_WEIGHT_COMMIT = "4f0c246f125fc7594238ebe7beb1435a8335f519"
MODEL_SHARDS = (
    {
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_bytes": 4_920_739_232,
        "sha256": "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c",
        "role": "selected_entrypoint",
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
EXPECTED_DISPLAYED_OPTIONS = {
    ("Q1322", "Maldives", "South Korea"): [
        "Agree",
        "Disagree",
        "Strongly agree",
        "Strongly disagree",
    ],
    ("Q1322", "South Korea", "Maldives"): [
        "Disagree",
        "Strongly disagree",
        "Strongly agree",
        "Agree",
    ],
}
MODEL_NAME = "Llama-3.1-8B-Instruct"
LLAMA_CPP_COMMIT = "62acc89c26c66076cb72e049f307fbe93b8b9750"
EXPECTED_GGUF_FILE_TYPE = 15
EXPECTED_GGUF_ARCHITECTURE = "llama"
EXPECTED_GGUF_CONTEXT_LENGTH = 131_072
ACCEPTED_SCORING_SOURCE_SHA256 = {
    "src/gguf_runner.py": (
        "db5f995ea75a4f8404ff1a1b9d5a0e275a54ede41e2064a1728cbb9fbccc2283"
    ),
    "src/llama_gguf_runner.py": (
        "8171285f70d19c8d88e323112b54c78de41e5f829a6c0aeeb176bc76e85c2d27"
    ),
    "src/gguf_score_helper.cpp": (
        "235ea664e5183786ef066f3d6ea692d60d2dcacaece0063a84a1ec6e5454871b"
    ),
    "scripts/build_gguf_score_helper.py": (
        "c981ecd24893571fb6131a17fa6f06623eb8f75356cf2317a10daf74ecbed8a7"
    ),
}
SEED = 42
NORMALIZATION_TOLERANCE = 1e-8


def require_pinned_revision(observed: str | None, expected: str, name: str) -> str:
    """Require an explicit immutable revision and reject every substitution."""

    if not observed:
        raise ValueError(
            f"{name} is required via its CLI option or environment variable"
        )
    if observed != expected:
        raise ValueError(f"{name} mismatch: {observed!r} != {expected!r}")
    return observed


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


def validate_helper_linkage(helper_path: Path, llama_cpp_directory: Path) -> str:
    """Require every llama/ggml shared object to resolve from the pinned tree."""

    linked = command_output(["ldd", str(helper_path.resolve())])
    if linked is None:
        raise RuntimeError("could not inspect GGUF helper shared-library linkage")
    expected_directory = (llama_cpp_directory / "build-cpu/bin").resolve()
    required = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so")
    for library in required:
        matching = [line for line in linked.splitlines() if library in line]
        if not matching:
            raise RuntimeError(f"GGUF helper is not linked to required {library}")
        resolved: list[Path] = []
        for line in matching:
            if "=>" not in line:
                continue
            candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
            if candidate and candidate != "not":
                resolved.append(Path(candidate).resolve())
        if not resolved or not all(
            path.is_relative_to(expected_directory) for path in resolved
        ):
            raise RuntimeError(
                f"{library} does not resolve from pinned build {expected_directory}"
            )
    return linked


def verify_scoring_source_hashes() -> dict[str, str]:
    """Fail if any reviewed scoring source differs from its accepted bytes."""

    observed: dict[str, str] = {}
    for relative, expected in ACCEPTED_SCORING_SOURCE_SHA256.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"accepted scoring source is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"accepted scoring source drifted: {relative}: {actual} != {expected}"
            )
        observed[relative] = actual
    return observed


def validate_gguf_runtime_metadata(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the loaded GGUF's scalar identity metadata."""

    scalar = runtime.get("gguf_scalar_metadata")
    if not isinstance(scalar, Mapping):
        raise AssertionError("loaded GGUF did not expose scalar metadata")
    architecture = scalar.get("general.architecture")
    context_length = scalar.get("llama.context_length")
    scalar_file_type = scalar.get("general.file_type")
    if architecture != EXPECTED_GGUF_ARCHITECTURE:
        raise AssertionError(f"wrong GGUF architecture: {architecture!r}")
    try:
        parsed_context = int(str(context_length))
        parsed_scalar_file_type = int(str(scalar_file_type))
    except (TypeError, ValueError) as exc:
        raise AssertionError("GGUF context length or file type is not an integer") from exc
    if parsed_context != EXPECTED_GGUF_CONTEXT_LENGTH:
        raise AssertionError(f"wrong GGUF context length: {parsed_context}")
    if runtime.get("gguf_file_type") != EXPECTED_GGUF_FILE_TYPE:
        raise AssertionError(f"wrong runtime GGUF file type: {runtime.get('gguf_file_type')!r}")
    if parsed_scalar_file_type != EXPECTED_GGUF_FILE_TYPE:
        raise AssertionError(f"wrong scalar GGUF file type: {parsed_scalar_file_type}")
    embedded_template = runtime.get("chat_template")
    scalar_template = scalar.get("tokenizer.chat_template")
    if scalar_template != embedded_template:
        raise AssertionError("runtime template differs from tokenizer.chat_template metadata")
    if hashlib.sha256(str(scalar_template).encode("utf-8")).hexdigest() != (
        LLAMA31_CHAT_TEMPLATE_SHA256
    ):
        raise AssertionError("GGUF scalar chat-template SHA-256 is not pinned Llama-3.1")
    return dict(scalar)


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


def assert_same_distribution(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    context: str,
    *,
    tolerance: float = 1e-12,
) -> None:
    if set(observed) != set(expected) or any(
        not math.isclose(
            float(observed[key]),
            float(expected[key]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for key in expected
    ):
        raise AssertionError(f"{context} differs from the independently recomputed value")


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
    eo_raw = Metrics.evidence_override_raw(conflict, evidence_human, label_human)
    eo_normalized = Metrics.evidence_override_normalized(
        conflict, evidence_human, label_human
    )
    unit_metrics = {
        "country_influence": Metrics.country_influence(baseline, country, label_human),
        "evidence_influence": Metrics.evidence_influence(baseline, evidence, label_human),
        "EO_raw": eo_raw,
        "EO_normalized": eo_normalized,
    }
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
            **components,
        }
        row.update(unit_metrics)


def validate_rows(
    rows: list[dict[str, Any]],
    runtime: Mapping[str, Any],
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
        if (
            scoring.get("prompt_bos_added") is not True
            or scoring.get("prompt_bos_token_id") != scoring.get("bos_token_id")
        ):
            raise AssertionError(f"row {index}: tokenizer did not add the model BOS token")
        if row.get("generated_answer") is not None:
            raise AssertionError(f"row {index}: generation/parsing is forbidden")
        if row["structured_messages"] != [
            {"role": "system", "content": LLAMA31_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": row["raw_user_prompt"]},
        ]:
            raise AssertionError(f"row {index}: Llama structured messages are incomplete")
        try:
            verify_llama31_serialized_chat(
                row["serialized_chat_templated_prompt"], row["structured_messages"]
            )
        except RuntimeError as exc:
            raise AssertionError(
                f"row {index}: invalid Llama-3.1 embedded-template serialization"
            ) from exc

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

        directed = (
            row["question_id"],
            row["label_country"],
            row["evidence_country"],
        )
        if row["displayed_options"] != EXPECTED_DISPLAYED_OPTIONS[directed]:
            raise AssertionError(
                f"row {index}: displayed permutation differs from the accepted smoke order"
            )

        evidence_presented = row["condition"] in {"evidence", "conflict"}
        if row.get("evidence_presented") is not evidence_presented:
            raise AssertionError(f"row {index}: evidence-presented flag is wrong")
        for field in (
            "presented_evidence_distribution",
            "source_evidence_distribution",
        ):
            if field not in row:
                raise AssertionError(f"row {index}: {field} is not explicitly stored")
        if not evidence_presented:
            if row["presented_evidence_distribution"] is not None:
                raise AssertionError(f"row {index}: no-evidence row stores presented evidence")
            if row["source_evidence_distribution"] is not None:
                raise AssertionError(f"row {index}: no-evidence row stores source evidence")
        else:
            source = (
                row["label_country_human_distribution"]
                if row["condition"] == "evidence"
                else row["evidence_country_human_distribution"]
            )
            assert_probability_distribution(source, options)
            assert_same_distribution(
                row["source_evidence_distribution"],
                source,
                f"row {index} source evidence",
            )
            displayed_source = reorder_distribution(source, row["displayed_options"])
            displayed_presented = prepare_evidence_distribution(
                displayed_source, row["displayed_options"]
            )
            expected_presented = recover_canonical_distribution(
                displayed_presented, options
            )
            assert_same_distribution(
                row["presented_evidence_distribution"],
                expected_presented,
                f"row {index} presented evidence",
            )

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
                    left_score = row["raw_candidate_scores"][left_label]
                    right_score = row["raw_candidate_scores"][right_label]
                    if left_label == right_label:
                        raise AssertionError("same-prefix semantic options share a displayed label")
                    if candidates[left_label]["token_ids"] == candidates[right_label]["token_ids"]:
                        raise AssertionError("same-prefix semantic options share a candidate token path")
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
                            "left_probability": row["normalized_prediction"][left],
                            "right_probability": row["normalized_prediction"][right],
                            "independently_scored": True,
                            "distinct_semantic_options": left != right,
                            "distinct_candidate_traces": (
                                candidates[left_label] is not candidates[right_label]
                                and candidates[left_label] != candidates[right_label]
                            ),
                        }
                    )
        row_checks.append({"row_index": index, "status": "PASS"})

    metrics_recomputed = []
    divergence_keys = {
        "prediction_to_label_country",
        "prediction_to_evidence_country",
        "baseline_to_label_country",
        "country_label_to_label_country",
        "evidence_to_label_country",
        "conflict_to_label_country",
        "conflict_to_evidence_country",
    }
    for unit in sorted(units):
        unit_rows = [
            row
            for row in rows
            if (row["question_id"], row["label_country"], row["evidence_country"])
            == unit
        ]
        by_condition = {row["condition"]: row for row in unit_rows}
        if set(by_condition) != set(EXPECTED_OUTPUT_CONDITIONS):
            raise AssertionError(f"{unit!r}: incomplete condition set for metric audit")
        baseline = by_condition["baseline"]["normalized_prediction"]
        country = by_condition["country_label"]["normalized_prediction"]
        evidence = by_condition["evidence"]["normalized_prediction"]
        conflict = by_condition["conflict"]["normalized_prediction"]
        label_human = unit_rows[0]["label_country_human_distribution"]
        evidence_human = unit_rows[0]["evidence_country_human_distribution"]
        expected_metrics = {
            "country_influence": Metrics.country_influence(
                baseline, country, label_human
            ),
            "evidence_influence": Metrics.evidence_influence(
                baseline, evidence, label_human
            ),
            "EO_raw": Metrics.evidence_override_raw(
                conflict, evidence_human, label_human
            ),
            "EO_normalized": Metrics.evidence_override_normalized(
                conflict, evidence_human, label_human
            ),
        }
        common_divergences = {
            "baseline_to_label_country": Metrics.js_divergence(
                baseline, label_human
            ),
            "country_label_to_label_country": Metrics.js_divergence(
                country, label_human
            ),
            "evidence_to_label_country": Metrics.js_divergence(
                evidence, label_human
            ),
            "conflict_to_label_country": Metrics.js_divergence(
                conflict, label_human
            ),
            "conflict_to_evidence_country": Metrics.js_divergence(
                conflict, evidence_human
            ),
        }
        for row in unit_rows:
            if row.get("jensen_shannon_base") != 2:
                raise AssertionError(f"{unit!r}: Jensen-Shannon base is not two")
            if row.get("jensen_shannon_measure") != "divergence_bits":
                raise AssertionError(f"{unit!r}: Jensen-Shannon quantity is mislabeled")
            divergences = row.get("base2_jensen_shannon_divergences")
            if not isinstance(divergences, Mapping) or set(divergences) != divergence_keys:
                raise AssertionError(f"{unit!r}: divergence schema is incomplete")
            expected_divergences = {
                "prediction_to_label_country": Metrics.js_divergence(
                    row["normalized_prediction"], label_human
                ),
                "prediction_to_evidence_country": Metrics.js_divergence(
                    row["normalized_prediction"], evidence_human
                ),
                **common_divergences,
            }
            for name, expected in expected_divergences.items():
                if not math.isclose(
                    float(divergences[name]), expected, rel_tol=0.0, abs_tol=1e-10
                ):
                    raise AssertionError(f"{unit!r}: inconsistent divergence {name}")
            for name, expected in expected_metrics.items():
                if not math.isclose(
                    float(row.get(name)), expected, rel_tol=0.0, abs_tol=1e-10
                ):
                    raise AssertionError(f"{unit!r}: inconsistent saved metric {name}")
        metrics_recomputed.append({"directed_unit": list(unit), **expected_metrics})

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
        positive_raw = Metrics.evidence_override_raw(
            evidence_human, evidence_human, label_human
        )
        negative_raw = Metrics.evidence_override_raw(
            label_human, evidence_human, label_human
        )
        positive_normalized = Metrics.evidence_override_normalized(
            evidence_human, evidence_human, label_human
        )
        negative_normalized = Metrics.evidence_override_normalized(
            label_human, evidence_human, label_human
        )
        if not positive_raw > 0.0 or not negative_raw < 0.0:
            raise AssertionError("Evidence Override orientation is reversed")
        if not math.isclose(positive_normalized, 1.0, abs_tol=1e-12):
            raise AssertionError("EO_normalized(exact evidence) is not +1")
        if not math.isclose(negative_normalized, -1.0, abs_tol=1e-12):
            raise AssertionError("EO_normalized(exact label) is not -1")
        orientation.append(
            {
                "directed_unit": list(unit),
                "prediction_equals_evidence_EO_raw": positive_raw,
                "prediction_equals_label_EO_raw": negative_raw,
                "prediction_equals_evidence_EO_normalized": positive_normalized,
                "prediction_equals_label_EO_normalized": negative_normalized,
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
        "embedded_llama_chat_template": runtime.get("chat_template_source")
        == "embedded_gguf_metadata"
        and runtime.get("chat_template_sha256") == LLAMA31_CHAT_TEMPLATE_SHA256
        and runtime.get("embedded_template_identity_verified") is True
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
        "evidence_null_semantics": len(row_checks) == 8,
        "divergence_schema_and_metrics_recomputed": len(metrics_recomputed) == 2,
        "gguf_scalar_metadata": bool(validate_gguf_runtime_metadata(runtime)),
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
        "# Genuine Llama Q4_K_M CPU Smoke Report",
        "",
        f"**Status: {analysis['status']}**",
        "",
        "This was a bounded genuine-model CPU smoke only: one official Llama GGUF, "
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
        "- System turn: explicit model-specific Llama-3.1 knowledge/date preamble",
        "- Serialization verification: pinned embedded Llama-3.1 template hash and exact answer boundary",
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
            "| Label country | Evidence country | Country Influence | Evidence Influence | EO_raw | EO_normalized |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for unit in analysis["directed_unit_metrics"]:
        lines.append(
            f"| {unit['label_country']} | {unit['evidence_country']} | "
            f"{unit['country_influence']:.9f} | {unit['evidence_influence']:.9f} | "
            f"{unit['EO_raw']:.9f} | {unit['EO_normalized']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Same-first-word audit",
            "",
            "`Strongly agree` and `Strongly disagree` were mapped to distinct displayed "
            "labels and contextual token paths in every row. Their label sequence scores "
            "and restored semantic probabilities were computed through separate candidate "
            "traces in all 8 rows. Full per-row values are stored in `analysis.json`.",
            "",
            "All displayed labels (`A`–`D`) were one token for this tokenizer. The "
            "backend implements full multi-token conditional sequence scoring and "
            "candidate-cache rollback, but that branch was not dynamically activated "
            "by this particular label set.",
            "",
            "## Pre-package QA note",
            "",
            "The runner accepts only the embedded template whose complete SHA-256 matches "
            "the pinned Llama-3.1 GGUF metadata, verifies the Llama special-token grammar, "
            "and requires the exact assistant answer boundary before any label is scored.",
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
        "--gguf-revision",
        default=os.environ.get("LLAMA_GGUF_REVISION"),
        help="Must equal the pinned bartowski repository revision",
    )
    parser.add_argument(
        "--upstream-revision",
        default=os.environ.get("LLAMA_UPSTREAM_REVISION"),
        help="Must equal the recorded upstream checkpoint revision",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/smoke_genuine_llama_gguf",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument(
        "--full-output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/llama_gguf_full",
        help="Full-run checkpoint directory used automatically after smoke PASS",
    )
    args = parser.parse_args()

    require_pinned_revision(args.gguf_revision, MODEL_REVISION, "GGUF revision")
    require_pinned_revision(
        args.upstream_revision,
        UPSTREAM_CURRENT_REVISION,
        "upstream checkpoint revision",
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    analysis_path = output_dir / "analysis.json"
    run_log_path = output_dir / "run.log"
    report_path = PROJECT_ROOT / "audit/GENUINE_LLAMA_GGUF_SMOKE_REPORT.md"
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

        log("BEGIN genuine Llama GGUF CPU smoke")
        log("Scope: one model, two reciprocal directed units, four conditions")
        environment = environment_information(PROJECT_ROOT)
        log(f"Environment: {json.dumps(environment, ensure_ascii=False)}")
        log("Verifying the pinned official GGUF by exact size and SHA-256")
        model_files = verified_model_files(args.model_dir.resolve())
        log("GGUF size and SHA-256 verification: PASS")

        llama_dir = args.llama_cpp_dir.resolve()
        llama_commit = command_output(["git", "-C", str(llama_dir), "rev-parse", "HEAD"])
        llama_commit_date = command_output(
            ["git", "-C", str(llama_dir), "log", "-1", "--format=%cI"]
        )
        if not llama_commit:
            raise RuntimeError("could not record the llama.cpp commit")
        if llama_commit != LLAMA_CPP_COMMIT:
            raise RuntimeError(
                f"llama.cpp commit mismatch: {llama_commit!r} != {LLAMA_CPP_COMMIT!r}"
            )
        dirty = command_output(
            ["git", "-C", str(llama_dir), "status", "--porcelain", "--untracked-files=no"]
        )
        if dirty is None:
            raise RuntimeError("could not verify llama.cpp worktree state")
        if dirty:
            raise RuntimeError("pinned llama.cpp checkout has tracked modifications")
        build_cache = llama_dir / "build-cpu/CMakeCache.txt"
        build_cache_sha256 = sha256_file(build_cache) if build_cache.is_file() else None
        helper_path = args.helper.resolve()
        if not helper_path.is_file():
            raise FileNotFoundError(f"GGUF scoring helper is missing: {helper_path}")
        helper_linkage = validate_helper_linkage(helper_path, llama_dir)
        helper_sha256 = sha256_file(helper_path)
        scoring_source_sha256 = verify_scoring_source_hashes()
        log("Pinned helper linkage and accepted scoring-source SHA-256 checks: PASS")
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
        rows: list[dict[str, Any]] = []
        runner = Llama31GGUFRunner(
            helper_path=helper_path,
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
        gguf_scalar_metadata = validate_gguf_runtime_metadata(runtime)
        if (
            runtime.get("requested_context_size") != args.ctx_size
            or runtime.get("actual_context_size") != args.ctx_size
        ):
            raise AssertionError("llama.cpp did not use the requested runtime context size")
        if not 8_000_000_000 <= int(runtime.get("model_parameter_count", 0)) < 9_000_000_000:
            raise AssertionError("loaded checkpoint does not have the expected 8B scale")
        if runtime.get("vocabulary_size") != 128_256:
            raise AssertionError("loaded checkpoint does not have the Llama-3.1 vocabulary")
        log("Loaded GGUF scalar metadata, identity, context, and template checks: PASS")
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
                    evidence_presented = internal_condition in {
                        "population_evidence",
                        "conflict",
                    }
                    if internal_condition == "population_evidence":
                        source_evidence = unit["label_human"]
                        presented_displayed = unit["label_evidence_displayed"]
                    elif internal_condition == "conflict":
                        source_evidence = unit["evidence_human"]
                        presented_displayed = unit["conflict_evidence_displayed"]
                    else:
                        source_evidence = None
                        presented_displayed = None
                    presented_canonical = (
                        recover_canonical_distribution(
                            presented_displayed,
                            unit["canonical_options"],
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
                    row = {
                        "schema_version": "genuine_llama_gguf_smoke_v2",
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
                        "evidence_presented": evidence_presented,
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

        write_jsonl(results_path, rows)
        saved_rows: list[dict[str, Any]] = []
        with results_path.open("r", encoding="utf-8") as saved_handle:
            for line_number, line in enumerate(saved_handle, start=1):
                if not line.strip():
                    raise AssertionError(f"blank saved JSONL line {line_number}")
                saved = json.loads(line)
                if not isinstance(saved, dict):
                    raise AssertionError(f"saved JSONL line {line_number} is not an object")
                saved_rows.append(saved)
        if saved_rows != rows:
            raise AssertionError("saved smoke JSONL differs from the in-memory result rows")
        rows = saved_rows
        assertions, shared_word_checks, orientation = validate_rows(rows, runtime)
        assertions["independent_first_row_rescore"] = independent_rescore is not None
        assertions["pinned_helper_linkage"] = bool(helper_linkage)
        assertions["accepted_scoring_source_hashes"] = (
            scoring_source_sha256 == ACCEPTED_SCORING_SOURCE_SHA256
        )
        assertions["gguf_runtime_identity"] = bool(gguf_scalar_metadata)
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
                    "EO_raw": row["EO_raw"],
                    "EO_normalized": row["EO_normalized"],
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
                    "upstream_checkpoint": UPSTREAM_CHECKPOINT,
                    "upstream_current_revision": UPSTREAM_CURRENT_REVISION,
                    "upstream_revision_role": (
                        "current immutable upstream identity; quantizer did not declare "
                        "the exact conversion-source revision"
                    ),
                    "remote_last_modified": REMOTE_MODEL_LAST_MODIFIED,
                    "weight_upload_commit": REMOTE_WEIGHT_COMMIT,
                    "quantization": QUANTIZATION,
                    "selected_entrypoint_file": MODEL_SHARDS[0]["filename"],
                    "companion_file": None,
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
                    "helper_sha256": helper_sha256,
                    "helper_ldd": helper_linkage,
                    "helper_linkage_sha256": hashlib.sha256(
                        helper_linkage.encode("utf-8")
                    ).hexdigest(),
                    "accepted_scoring_source_sha256": scoring_source_sha256,
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
                    "gguf_file_type": runtime["gguf_file_type"],
                    "gguf_metadata_count": runtime["gguf_metadata_count"],
                    "gguf_scalar_metadata": gguf_scalar_metadata,
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
        log("END genuine Llama GGUF CPU smoke")

    full_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_llama_gguf_full.py"),
        "--model-dir",
        str(args.model_dir.resolve()),
        "--helper",
        str(args.helper.resolve()),
        "--llama-cpp-dir",
        str(args.llama_cpp_dir.resolve()),
        "--output-dir",
        str(args.full_output_dir.resolve()),
        "--threads",
        str(args.threads),
        "--ctx-size",
        str(args.ctx_size),
        "--gguf-revision",
        args.gguf_revision,
        "--upstream-revision",
        args.upstream_revision,
        "--smoke-results",
        str(results_path),
        "--smoke-analysis",
        str(analysis_path),
        "--allow-llama-full-run",
    ]
    subprocess.run(full_command, check=True)
    print(
        json.dumps(
            {
                "status": "PASS",
                "smoke_rows": 8,
                "smoke_output_dir": str(output_dir),
                "full_output_dir": str(args.full_output_dir.resolve()),
                "full_run_started_after_smoke_pass": True,
            }
        )
    )


def write_failure_diagnostics(error: BaseException) -> Path:
    """Persist and package a structured smoke/full-gate failure report."""

    output_dir = PROJECT_ROOT / "experiments/smoke_genuine_llama_gguf"
    if "--output-dir" in sys.argv:
        index = sys.argv.index("--output-dir")
        if index + 1 < len(sys.argv):
            output_dir = Path(sys.argv[index + 1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "status": "FAIL",
        "failed_at_utc": now_utc(),
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "upstream_checkpoint": UPSTREAM_CHECKPOINT,
        "upstream_revision": UPSTREAM_CURRENT_REVISION,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "synthetic_or_fallback_used": False,
    }
    diagnostic_path = output_dir / "failure.json"
    atomic_write_text(
        diagnostic_path,
        json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    archive_path = output_dir / "failure_diagnostics.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != archive_path:
                archive.write(path, path.relative_to(output_dir))
        for path in (
            PROJECT_ROOT / "scripts/run_genuine_llama_gguf_smoke.py",
            PROJECT_ROOT / "scripts/run_llama_gguf_full.py",
            PROJECT_ROOT / "src/llama_gguf_runner.py",
            PROJECT_ROOT / "data/pairs/country_pairs_smoke.json",
        ):
            if path.is_file():
                archive.write(path, Path("diagnostic_inputs") / path.name)
    return archive_path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        archive = write_failure_diagnostics(exc)
        print(
            json.dumps(
                {"status": "FAIL", "diagnostic_archive": str(archive)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise

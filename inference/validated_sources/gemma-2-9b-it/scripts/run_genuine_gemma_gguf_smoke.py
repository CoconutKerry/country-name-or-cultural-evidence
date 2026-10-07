#!/usr/bin/env python3
"""Run the mandatory genuine 8-row Gemma 2 GGUF smoke and fail closed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, IO, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_gemma_gguf_full import (
    DATASET_PATH,
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_FULL_OUTPUT_DIRECTORY,
    FROZEN_SMOKE_PERMUTATIONS,
    FULL_SCHEMA_VERSION,
    _validate_unit_rows,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    build_directed_condition_prompt,
    canonical_json_hash,
    directed_key,
    prepare_unit,
    run_directed_unit,
    run_gemma_full,
    validate_helper_linkage,
    validate_preinference_gate,
    validate_runtime,
)
from src.data_loader import DataLoader
from src.gemma_gguf_spec import (
    LLAMA_CPP_COMMIT,
    MODEL_FILES,
    MODEL_IDENTIFIER,
    MODEL_NAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    QUANTIZATION,
    environment_information,
    provenance_record,
    sha256_file,
    verified_model_files,
    verify_llama_cpp_checkout,
)
from src.gguf_runner import Gemma2GGUFRunner
from src.metrics import Metrics


SMOKE_PAIR_MANIFEST = "data/pairs/country_pairs_smoke.json"
SMOKE_PAIR_MANIFEST_SHA256 = (
    "c0b4aea31c92a474b17f31699070d1a27d26de5d9a800015b5251d2b7bf60096"
)
EXPECTED_KEYS = tuple(FROZEN_SMOKE_PERMUTATIONS)
EXPECTED_CONDITIONS = ("baseline", "country_label", "evidence", "conflict")
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments/smoke_genuine_gemma_gguf"
SMOKE_GATE_FILENAME = "smoke_gate.json"
SERIALIZATION_PREFLIGHT_FILENAME = "serialization_preflight.json"
NORMALIZATION_TOLERANCE = 1e-8


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(handle: IO[str], message: str) -> None:
    handle.write(f"{now_utc()} {message}\n")
    handle.flush()
    os.fsync(handle.fileno())


def _source_hashes() -> dict[str, str]:
    relatives = (
        "src/gemma_gguf_spec.py",
        "src/gguf_runner.py",
        "src/gguf_score_helper.cpp",
        "src/main.py",
        "src/metrics.py",
        "src/prompt_builder.py",
        "src/scoring.py",
        "scripts/run_genuine_gemma_gguf_smoke.py",
        "scripts/run_gemma_gguf_full.py",
    )
    return {relative: sha256_file(PROJECT_ROOT / relative) for relative in relatives}


def smoke_compatibility_record(
    model_files: Sequence[Mapping[str, Any]],
    helper_path: Path,
    *,
    threads: int,
    context_size: int,
    preinference_gate: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "model_identifier": MODEL_IDENTIFIER,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_files": [
            {
                "filename": item["filename"],
                "size_bytes": item["actual_size_bytes"],
                "sha256": item["actual_sha256"],
            }
            for item in model_files
        ],
        "helper_sha256": sha256_file(helper_path),
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "threads": int(threads),
        "context_size": int(context_size),
        "source_sha256": _source_hashes(),
        "preinference_gate_sha256": sha256_file(PROJECT_ROOT / "audit/GEMMA_GGUF_PREINFERENCE_GATE.json"),
        "preinference_gate_completed_at_utc": preinference_gate.get("completed_at_utc"),
    }
    return {**core, "fingerprint": canonical_json_hash(core)}


def validate_smoke_rows(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, bool], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_runtime(runtime)
    if len(rows) != 8:
        raise AssertionError(f"genuine smoke must contain exactly 8 rows, found {len(rows)}")
    observed = [
        (*directed_key(row), str(row.get("condition"))) for row in rows
    ]
    expected = [
        (*key, condition) for key in EXPECTED_KEYS for condition in EXPECTED_CONDITIONS
    ]
    if observed != expected or len(set(observed)) != 8:
        raise AssertionError("reciprocal directed-condition rows are incomplete or overwritten")

    same_word_checks: list[dict[str, Any]] = []
    answer_position_checks: list[dict[str, Any]] = []
    for row in rows:
        probabilities = row["normalized_prediction"]
        values = [float(probabilities[option]) for option in row["original_options"]]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise AssertionError("prediction contains invalid probabilities")
        if not math.isclose(
            math.fsum(values), 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE
        ):
            raise AssertionError("prediction probabilities do not sum to one")
        scoring = row["scoring_trace"]
        prompt_count = int(scoring["prompt_token_count"])
        prompt_token_ids = scoring.get("prompt_token_ids")
        bos_token_id = scoring.get("bos_token_id")
        if (
            scoring.get("leading_bos_token_count") != 1
            or not isinstance(bos_token_id, int)
            or bos_token_id < 0
            or not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or prompt_token_ids[0] != bos_token_id
        ):
            raise AssertionError(
                "Gemma chat serialization did not produce exactly one leading BOS token"
            )
        for label in row["displayed_option_labels"]:
            candidate = scoring["candidates"][label]
            expected_continuation = (
                label
                if row["serialized_chat_templated_prompt"][-1].isspace()
                else " " + label
            )
            if (
                candidate.get("candidate_label") != label
                or candidate.get("continuation") != expected_continuation
                or candidate.get("prompt_prefix_verified") is not True
            ):
                raise AssertionError(
                    "candidate label was not independently tokenized in its actual prompt context"
                )
            token_ids = candidate["token_ids"]
            if not token_ids:
                raise AssertionError("candidate label has no contextual tokens")
            targets = list(range(prompt_count, prompt_count + len(token_ids)))
            if candidate["target_token_positions"] != targets:
                raise AssertionError("candidate label target positions are wrong")
            if candidate["predictive_logit_positions"] != [x - 1 for x in targets]:
                raise AssertionError("candidate label predictive positions are wrong")
            if candidate["predictive_logit_positions"][0] != prompt_count - 1:
                raise AssertionError("candidate label was not scored at the answer position")
        answer_position_checks.append(
            {
                "directed_key": list(directed_key(row)),
                "condition": row["condition"],
                "answer_target_position": scoring["answer_target_position"],
                "answer_predictive_logit_position": scoring[
                    "answer_predictive_logit_position"
                ],
                "status": "PASS",
            }
        )
        first_words: dict[str, list[str]] = {}
        for option in row["original_options"]:
            first_words.setdefault(str(option).split()[0].lower(), []).append(str(option))
        option_to_label = {
            option: label for label, option in row["displayed_label_to_option"].items()
        }
        for first_word, options in first_words.items():
            if len(options) < 2:
                continue
            for left_index in range(len(options)):
                for right_index in range(left_index + 1, len(options)):
                    left, right = options[left_index], options[right_index]
                    left_label, right_label = option_to_label[left], option_to_label[right]
                    left_tokens = scoring["candidates"][left_label]["token_ids"]
                    right_tokens = scoring["candidates"][right_label]["token_ids"]
                    if left_label == right_label or left_tokens == right_tokens:
                        raise AssertionError("same-first-word options were not independently scored")
                    same_word_checks.append(
                        {
                            "directed_key": list(directed_key(row)),
                            "condition": row["condition"],
                            "first_word": first_word,
                            "left_option": left,
                            "right_option": right,
                            "left_label": left_label,
                            "right_label": right_label,
                            "left_token_ids": left_tokens,
                            "right_token_ids": right_tokens,
                            "independently_scored": True,
                        }
                    )
    if len(same_word_checks) != 8:
        raise AssertionError("same-first-word audit did not cover every smoke row")

    orientation_checks = []
    for pair in pairs:
        key = directed_key(pair)
        representative = next(row for row in rows if directed_key(row) == key)
        label = representative["label_country_human_distribution"]
        evidence = representative["evidence_country_human_distribution"]
        raw_evidence = Metrics.evidence_override_raw(evidence, evidence, label)
        raw_label = Metrics.evidence_override_raw(label, evidence, label)
        normalized_evidence = Metrics.evidence_override_normalized(evidence, evidence, label)
        normalized_label = Metrics.evidence_override_normalized(label, evidence, label)
        if not (raw_evidence > 0.0 and raw_label < 0.0):
            raise AssertionError("EO_raw orientation is reversed")
        if not math.isclose(normalized_evidence, 1.0, abs_tol=1e-12):
            raise AssertionError("exact evidence does not yield EO_normalized +1")
        if not math.isclose(normalized_label, -1.0, abs_tol=1e-12):
            raise AssertionError("exact label does not yield EO_normalized -1")
        orientation_checks.append(
            {
                "directed_key": list(key),
                "EO_raw_exact_evidence": raw_evidence,
                "EO_raw_exact_label": raw_label,
                "EO_normalized_exact_evidence": normalized_evidence,
                "EO_normalized_exact_label": normalized_label,
                "status": "PASS",
            }
        )

    assertions = {
        "genuine_llama_cpp_backend": runtime.get("backend") == "llama.cpp",
        "synthetic_false": runtime.get("synthetic") is False,
        "correct_model_loaded": runtime.get("model_identifier") == MODEL_IDENTIFIER,
        "model_specific_embedded_template": runtime.get("chat_template_source")
        == "embedded_gguf_metadata",
        "user_only_message_structure": all(
            row["scoring_trace"].get("user_only_message_structure_verified") is True
            for row in rows
        ),
        "exact_gemma_assistant_answer_suffix": all(
            row["scoring_trace"].get("exact_assistant_answer_suffix_verified")
            is True
            for row in rows
        ),
        "byte_or_token_id_equivalent_serialization": all(
            row["scoring_trace"].get("serialization_verification_mode")
            in {"byte_exact", "token_id_equivalent"}
            and row["scoring_trace"].get("prompt_token_ids")
            == row["scoring_trace"].get("independent_prompt_token_ids")
            for row in rows
        ),
        "exactly_one_template_bos_token": all(
            row["scoring_trace"].get("leading_bos_token_count") == 1
            and row["scoring_trace"].get("independent_leading_bos_token_count")
            == 1
            for row in rows
        ),
        "exactly_eight_valid_rows": len(rows) == 8,
        "candidate_labels_scored_at_answer_position": len(answer_position_checks) == 8,
        "finite_normalized_probabilities": True,
        "same_first_word_options_independent": len(same_word_checks) == 8,
        "fixed_option_permutations": all(
            next(row for row in rows if directed_key(row) == key)["displayed_options"]
            == expected_options
            for key, expected_options in FROZEN_SMOKE_PERMUTATIONS.items()
        ),
        "reciprocal_units_not_overwritten": len(set(observed)) == 8,
        "EO_raw_and_normalized_orientation": len(orientation_checks) == 2,
        "no_generation_parsing_fallback_or_substitution": all(
            row.get("generated_answer") is None
            and row.get("backend") == "llama.cpp"
            and row.get("synthetic") is False
            and row.get("model_identifier") == MODEL_IDENTIFIER
            for row in rows
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"genuine smoke assertions failed: {failed}")
    return assertions, same_word_checks, orientation_checks


def report_markdown(analysis: Mapping[str, Any]) -> str:
    status = str(analysis["status"])
    row_count = int(analysis.get("row_count", 0))
    lines = [
        "# Genuine Gemma-2-9B-It Q4_K_M CPU Smoke Report",
        "",
        f"**Status: {status}**",
        "",
    ]
    if status == "PASS":
        lines.extend(
            [
                "Exactly two reciprocal directed units completed baseline, country_label, evidence, and conflict (8 validated rows).",
                "No answer was generated or parsed, and no model other than the pinned Gemma GGUF was loaded.",
            ]
        )
    else:
        lines.extend(
            [
                f"The intended 8-row smoke did not complete; {row_count} validated rows were saved before failure.",
                "No passing smoke gate was created, so the full run remains forbidden.",
            ]
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Model: `{MODEL_NAME}`",
            f"- GGUF repository: `{MODEL_REPOSITORY}`",
            f"- Immutable revision: `{MODEL_REVISION}`",
            f"- Quantization: `{QUANTIZATION}`",
            f"- llama.cpp commit: `{LLAMA_CPP_COMMIT}`",
            "- Required template contract: exact embedded Gemma 2 template hash; one user turn; exactly one leading BOS; exact assistant answer suffix; byte equality or recorded token-ID equivalence",
        ]
    )
    preflight = analysis.get("serialization_preflight")
    if isinstance(preflight, Mapping):
        lines.extend(
            [
                "",
                "## Serialization-only live preflight",
                "",
                f"- Status: `{preflight.get('status')}`",
                f"- Artifact: `{preflight.get('path')}`",
                f"- Verification mode: `{preflight.get('verification_mode')}`",
                "- Model inference performed by preflight: "
                f"`{str(preflight.get('inference_performed')).lower()}`",
            ]
        )
    assertions = analysis.get("assertions", {})
    if assertions:
        lines.extend(
            [
                "",
                "## Assertions",
                "",
                "| Assertion | Result |",
                "|---|---|",
            ]
        )
        for name, passed in assertions.items():
            lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    else:
        lines.extend(
            [
                "",
                "## Assertions",
                "",
                "The final 8-row smoke assertions were not reached.",
            ]
        )
    if status != "PASS":
        lines.extend(["", "## Failure", "", f"`{analysis.get('failure')}`"])
    else:
        lines.extend(
            [
                "",
                "## Metric orientation",
                "",
                "EO_raw is positive toward evidence and negative toward the country label. EO_normalized is +1 for exact evidence and -1 for exact label.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def run_smoke(
    *,
    model_directory: Path,
    helper_path: Path,
    llama_cpp_directory: Path,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    threads: int = 8,
    context_size: int = 4096,
) -> dict[str, Any]:
    if threads <= 0 or context_size <= 0:
        raise ValueError("threads and context_size must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    run_log_path = output_directory / "run.log"
    analysis_path = output_directory / "analysis.json"
    results_path = output_directory / "results.jsonl"
    report_path = output_directory / "GENUINE_GEMMA_GGUF_SMOKE_REPORT.md"
    serialization_preflight_path = (
        output_directory / SERIALIZATION_PREFLIGHT_FILENAME
    )
    runner: Gemma2GGUFRunner | None = None
    analysis: dict[str, Any] = {
        "status": "FAILED",
        "started_at_utc": now_utc(),
        "model_identifier": MODEL_IDENTIFIER,
        "row_count": 0,
    }
    with run_log_path.open("a", encoding="utf-8") as run_log:
        try:
            started = time.monotonic()
            preinference_gate = validate_preinference_gate(PROJECT_ROOT)
            model_files = verified_model_files(model_directory.resolve())
            llama_checkout = verify_llama_cpp_checkout(llama_cpp_directory.resolve())
            helper_path = helper_path.resolve()
            if not helper_path.is_file():
                raise FileNotFoundError(f"GGUF scorer is missing: {helper_path}")
            helper_linkage = validate_helper_linkage(
                helper_path, llama_cpp_directory.resolve()
            )
            loader = DataLoader(str(PROJECT_ROOT / DATASET_PATH))
            loader.load_dataset()
            manifest_path = PROJECT_ROOT / SMOKE_PAIR_MANIFEST
            if sha256_file(manifest_path) != SMOKE_PAIR_MANIFEST_SHA256:
                raise RuntimeError("smoke pair manifest SHA-256 mismatch")
            pairs = loader.get_question_pairs(str(manifest_path))
            if tuple(directed_key(pair) for pair in pairs) != EXPECTED_KEYS:
                raise RuntimeError("smoke pair manifest does not contain the accepted units")
            runner = Gemma2GGUFRunner(
                helper_path=helper_path,
                model_path=Path(model_files[0]["local_path"]),
                model_identifier=MODEL_IDENTIFIER,
                repository=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                quantization=QUANTIZATION,
                threads=threads,
                context_size=context_size,
            )
            runner.load_model(run_log)
            runtime = runner.get_runtime_metadata()
            validate_runtime(runtime)
            first_unit = prepare_unit(loader, pairs[0])
            first_prompt = build_directed_condition_prompt(first_unit, "baseline")
            serialization_preflight = runner.serialization_preflight(first_prompt)
            atomic_write_json(serialization_preflight_path, serialization_preflight)
            analysis["serialization_preflight"] = {
                "status": serialization_preflight["status"],
                "path": SERIALIZATION_PREFLIGHT_FILENAME,
                "sha256": sha256_file(serialization_preflight_path),
                "verification_mode": serialization_preflight["serialization"][
                    "verification_mode"
                ],
                "inference_performed": serialization_preflight.get(
                    "inference_performed"
                ),
                "checks": serialization_preflight["checks"],
            }
            if serialization_preflight["status"] != "PASS":
                raise RuntimeError(
                    "serialization-only live preflight failed; inspect "
                    f"{SERIALIZATION_PREFLIGHT_FILENAME}"
                )
            log(
                run_log,
                "PASS: serialization-only live preflight; no inference performed; "
                f"mode={serialization_preflight['serialization']['verification_mode']}",
            )
            compatibility = smoke_compatibility_record(
                model_files,
                helper_path,
                threads=threads,
                context_size=context_size,
                preinference_gate=preinference_gate,
            )
            rows: list[dict[str, Any]] = []
            for pair in pairs:
                unit_rows = run_directed_unit(
                    runner,
                    loader,
                    pair,
                    model_files,
                    compatibility["fingerprint"],
                )
                unit_rows = _validate_unit_rows(
                    unit_rows,
                    pair,
                    run_fingerprint=compatibility["fingerprint"],
                )
                rows.extend(unit_rows)
                analysis["row_count"] = len(rows)
                atomic_write_jsonl(results_path, rows)
            assertions, same_word, orientation = validate_smoke_rows(
                rows, pairs, runtime
            )
            assertions = {
                "serialization_only_live_preflight": True,
                **assertions,
            }
            analysis = {
                "status": "PASS",
                "started_at_utc": analysis["started_at_utc"],
                "completed_at_utc": now_utc(),
                "elapsed_seconds": time.monotonic() - started,
                "schema_version": FULL_SCHEMA_VERSION,
                "row_count": len(rows),
                "directed_keys": [list(key) for key in EXPECTED_KEYS],
                "conditions": list(EXPECTED_CONDITIONS),
                "assertions": assertions,
                "same_first_word_checks": same_word,
                "orientation_checks": orientation,
                "serialization_preflight": analysis["serialization_preflight"],
                "runtime": runtime,
                "model_provenance": provenance_record(model_files),
                "llama_cpp_checkout": llama_checkout,
                "helper_ldd": helper_linkage,
                "environment": environment_information(PROJECT_ROOT),
                "compatibility": compatibility,
            }
            gate = {
                "status": "PASS",
                "completed_at_utc": analysis["completed_at_utc"],
                "row_count": 8,
                "assertions": assertions,
                "compatibility": compatibility,
                "analysis_sha256": None,
                "results_sha256": sha256_file(results_path),
                "serialization_preflight_sha256": sha256_file(
                    serialization_preflight_path
                ),
            }
            atomic_write_json(analysis_path, analysis)
            gate["analysis_sha256"] = sha256_file(analysis_path)
            atomic_write_json(output_directory / SMOKE_GATE_FILENAME, gate)
            atomic_write_text(report_path, report_markdown(analysis))
            log(run_log, "PASS: genuine Gemma smoke produced exactly 8 valid rows")
            return analysis
        except Exception as exc:
            analysis.update(
                {
                    "completed_at_utc": now_utc(),
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            atomic_write_json(analysis_path, analysis)
            atomic_write_text(report_path, report_markdown(analysis))
            log(run_log, f"FAILED: {type(exc).__name__}: {exc}")
            raise
        finally:
            if runner is not None:
                runner.close()
                log(run_log, "Model released; llama.cpp helper stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--full-output-dir",
        type=Path,
        default=DEFAULT_FULL_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=4096)
    args = parser.parse_args()
    analysis = run_smoke(
        model_directory=args.model_dir,
        helper_path=args.helper,
        llama_cpp_directory=args.llama_cpp_dir,
        output_directory=args.output_dir,
        threads=args.threads,
        context_size=args.ctx_size,
    )
    _, completion = run_gemma_full(
        model_directory=args.model_dir,
        helper_path=args.helper,
        llama_cpp_directory=args.llama_cpp_dir,
        output_directory=args.full_output_dir,
        threads=args.threads,
        context_size=args.ctx_size,
        allow_gemma_full_run=True,
        smoke_directory=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "smoke_rows": analysis["row_count"],
                "smoke_output_dir": str(args.output_dir.resolve()),
                "full_output_dir": str(args.full_output_dir.resolve()),
                "full_run_started_after_smoke_pass": True,
                "full_completion": completion,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the genuine Mistral smoke and automatically continue to the full run.

This is the production orchestrator for the assigned model. It requires a
record of a passing complete test suite, loads only the exact pinned Mistral
Q4_K_M GGUF on CPU, runs the accepted reciprocal units in all four conditions,
and enters the 200-unit run only after every smoke assertion passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, IO, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mistral_gguf_full import (
    ACCEPTED_GGUF_HELPER_SOURCE_SHA256,
    ACCEPTED_GGUF_RUNNER_SHA256,
    DATASET_PATH,
    DATASET_SHA256,
    EXPECTED_OUTPUT_CONDITIONS,
    PAIR_MANIFEST,
    PAIR_MANIFEST_SHA256,
    _validate_unit_rows,
    atomic_write_json,
    atomic_write_jsonl,
    available_cpu_threads,
    build_run_signature,
    directed_key,
    enrich_unit_metrics,
    now_utc,
    run_condition_row,
    run_mistral_full,
    validate_helper_linkage,
    validate_manifest,
    validate_runtime,
)
from src.data_loader import DataLoader
from src.gguf_runner import (
    GGUFRunner,
    MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
    MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION,
    MISTRAL_V03_CHAT_PROFILE,
)
from src.metrics import Metrics
from src.mistral_gguf_contract import (
    LLAMA_CPP_COMMIT,
    MISTRAL_CHAT_TEMPLATE_SHA256,
    MODEL_FILENAME,
    MODEL_IDENTIFIER,
    MODEL_NAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    QUANTIZATION,
    UPSTREAM_CHECKPOINT,
    command_output,
    sha256_file,
    verified_model_files,
)


SMOKE_MANIFEST = "data/pairs/country_pairs_smoke.json"
SMOKE_MANIFEST_SHA256 = (
    "c0b4aea31c92a474b17f31699070d1a27d26de5d9a800015b5251d2b7bf60096"
)
EXPECTED_SMOKE_KEYS = (
    ("Q1322", "Maldives", "South Korea"),
    ("Q1322", "South Korea", "Maldives"),
)
SMOKE_SCHEMA_VERSION = "mistral-genuine-smoke-v2"
DEFAULT_SMOKE_OUTPUT = PROJECT_ROOT / "experiments/mistral_gguf_smoke"
DEFAULT_FULL_OUTPUT = PROJECT_ROOT / "experiments/mistral_gguf_full"
DEFAULT_ANALYSIS_OUTPUT = PROJECT_ROOT / "experiments/mistral_gguf_analysis"


def _log(handle: IO[str], message: str) -> None:
    handle.write(f"{now_utc()} {message}\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_passing_test_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"complete-suite report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "PASS"
        or not isinstance(report.get("tests_collected"), int)
        or report["tests_collected"] <= 0
        or report.get("failures") != 0
        or report.get("errors") != 0
        or report.get("complete_unit_test_suite") is not True
        or report.get("data_validation") != "PASS"
        or report.get("gguf_runner_sha256") != ACCEPTED_GGUF_RUNNER_SHA256
        or report.get("gguf_score_helper_source_sha256")
        != ACCEPTED_GGUF_HELPER_SOURCE_SHA256
    ):
        raise RuntimeError(
            "complete unit-test suite did not pass against the accepted scorer sources"
        )
    transcript_relative = report.get("test_transcript")
    transcript_hash = report.get("test_transcript_sha256")
    if not isinstance(transcript_relative, str) or not isinstance(transcript_hash, str):
        raise RuntimeError("complete-suite report does not bind its test transcript")
    transcript_path = Path(transcript_relative)
    if transcript_path.is_absolute() or ".." in transcript_path.parts:
        raise RuntimeError("complete-suite report has an unsafe transcript path")
    transcript_path = PROJECT_ROOT / transcript_path
    if not transcript_path.is_file() or sha256_file(transcript_path) != transcript_hash:
        raise RuntimeError("complete-suite transcript is missing or changed")
    validated_hashes = report.get("validated_file_sha256")
    if not isinstance(validated_hashes, Mapping) or not validated_hashes:
        raise RuntimeError("complete-suite report does not bind validated source files")
    for relative, expected in validated_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("complete-suite report has malformed source hashes")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("complete-suite report has an unsafe source path")
        source = PROJECT_ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"validated source changed after tests: {relative}")
    return report


def _same_first_word_independence(rows: Sequence[Mapping[str, Any]]) -> bool:
    checked = 0
    for row in rows:
        labels = list(row["displayed_option_labels"])
        candidates = row["scoring_trace"]["candidates"]
        groups: dict[str, list[str]] = {}
        for label, option in zip(labels, row["displayed_options"]):
            groups.setdefault(str(option).split()[0].casefold(), []).append(label)
        for shared_labels in groups.values():
            if len(shared_labels) < 2:
                continue
            checked += 1
            paths = [tuple(candidates[label]["token_ids"]) for label in shared_labels]
            if len(set(paths)) != len(paths):
                return False
            if any(candidates[label]["candidate_label"] != label for label in shared_labels):
                return False
    return checked > 0


def _probabilities_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        probabilities = list(row["normalized_label_probabilities"].values())
        if not probabilities or not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in probabilities
        ):
            return False
        if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
            return False
    return True


def _scored_at_answer_position(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        trace = row["scoring_trace"]
        prompt_count = trace["prompt_token_count"]
        if trace["answer_target_position"] != prompt_count:
            return False
        if trace["answer_predictive_logit_position"] != prompt_count - 1:
            return False
        for label, candidate in trace["candidates"].items():
            if candidate["continuation"] != label:
                return False
            targets = list(range(prompt_count, prompt_count + candidate["token_count"]))
            if candidate["target_token_positions"] != targets:
                return False
            if candidate["predictive_logit_positions"] != [item - 1 for item in targets]:
                return False
    return True


def _metric_orientation(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows[::4]:
        label = row["label_country_human_distribution"]
        evidence = row["evidence_country_human_distribution"]
        if not (
            Metrics.evidence_override_raw(evidence, evidence, label) > 0.0
            and Metrics.evidence_override_raw(label, evidence, label) < 0.0
            and math.isclose(
                Metrics.evidence_override_normalized(evidence, evidence, label),
                1.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                Metrics.evidence_override_normalized(label, evidence, label),
                -1.0,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def validate_smoke(
    rows: Sequence[Mapping[str, Any]],
    smoke_manifest: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    model_files: Sequence[Mapping[str, Any]],
    run_fingerprint: str,
    test_report: Mapping[str, Any],
    checkpoint_directory: Path,
) -> dict[str, bool]:
    validated: list[dict[str, Any]] = []
    for index, pair in enumerate(smoke_manifest):
        validated.extend(
            _validate_unit_rows(
                rows[index * 4 : index * 4 + 4],
                pair,
                run_fingerprint=run_fingerprint,
            )
        )
    row_keys = [(*directed_key(row), str(row["condition"])) for row in validated]
    observed_units = {directed_key(row) for row in validated}
    template_applied = all(
        row["scoring_trace"]["embedded_template_serialization_verified"] is True
        and row["scoring_trace"]["serialization_verification"]
        == MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION
        and row["scoring_trace"]["chat_profile"] == MISTRAL_V03_CHAT_PROFILE
        and row["scoring_trace"]["add_special_tokens"] is True
        and row["scoring_trace"]["tokenization_add_special"] is True
        and row["scoring_trace"]["canonical_tokenization_add_special"] is False
        and row["scoring_trace"]["candidate_prefix"] == ""
        and row["scoring_trace"]["chat_template_source"]
        == "embedded_gguf_metadata"
        and row["scoring_trace"]["template_equivalence_policy"]
        == MISTRAL_TEMPLATE_EQUIVALENCE_POLICY
        and row["scoring_trace"]["template_equivalence_preflight_passed"] is True
        and row["scoring_trace"]["template_equivalence"]["accepted"] is True
        and row["scoring_trace"]["template_equivalence"][
            "token_ids_equivalent"
        ]
        is True
        and row["scoring_trace"]["prompt_bos_token_count"] == 1
        for row in validated
    )
    assertions = {
        "complete_test_suite_passed_before_inference": (
            test_report.get("status") == "PASS"
            and test_report.get("failures") == 0
            and test_report.get("errors") == 0
        ),
        "exactly_eight_valid_rows": len(validated) == 8 and len(set(row_keys)) == 8,
        "exact_two_accepted_reciprocal_units": observed_units == set(EXPECTED_SMOKE_KEYS),
        "all_four_conditions_per_unit": all(
            [row["condition"] for row in validated[index : index + 4]]
            == list(EXPECTED_OUTPUT_CONDITIONS)
            for index in (0, 4)
        ),
        "genuine_llama_cpp_backend": runtime.get("backend") == "llama.cpp"
        and all(row["backend"] == "llama.cpp" for row in validated),
        "synthetic_false": runtime.get("synthetic") is False
        and all(row["synthetic"] is False for row in validated),
        "correct_model_loaded": (
            runtime.get("model_identifier") == MODEL_IDENTIFIER
            and runtime.get("repository") == MODEL_REPOSITORY
            and runtime.get("revision") == MODEL_REVISION
            and len(model_files) == 1
            and model_files[0]["filename"] == MODEL_FILENAME
            and model_files[0]["actual_size_bytes"] == MODEL_SIZE_BYTES
            and model_files[0]["actual_sha256"] == MODEL_SHA256
        ),
        "model_specific_embedded_chat_template_applied": (
            runtime.get("chat_template_source") == "embedded_gguf_metadata"
            and runtime.get("chat_template_sha256") == MISTRAL_CHAT_TEMPLATE_SHA256
            and runtime.get("chat_profile") == MISTRAL_V03_CHAT_PROFILE
            and template_applied
        ),
        "candidate_labels_scored_at_exact_answer_position": _scored_at_answer_position(
            validated
        ),
        "finite_nonnegative_unit_sum_probabilities": _probabilities_valid(validated),
        "same_first_word_options_independently_scored": _same_first_word_independence(
            validated
        ),
        "option_permutations_recover_semantic_order": all(
            set(row["normalized_prediction"]) == set(row["original_options"])
            and row["displayed_options"] == row["displayed_option_order"]
            and row["restored_original_option_order"] == row["original_options"]
            for row in validated
        ),
        "reciprocal_directed_units_not_overwritten": len(observed_units) == 2
        and len(set(row_keys)) == 8,
        "EO_raw_and_EO_normalized_orientation": _metric_orientation(validated),
        "no_fallback_or_model_substitution": (
            runtime.get("generated_answer_parsing") is False
            and runtime.get("candidate_scoring") == "full_vocabulary_low_level_logits"
            and runtime.get("gpu_layers") == 0
            and all(row["model_name"] == MODEL_NAME for row in validated)
            and all(row["upstream_checkpoint"] == UPSTREAM_CHECKPOINT for row in validated)
            and all(row["quantization"]["type"] == QUANTIZATION for row in validated)
        ),
        "every_smoke_row_checkpointed": len(list(checkpoint_directory.glob("row_*.json")))
        == 8,
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"genuine smoke assertions failed: {failed}")
    return assertions


def _smoke_report(
    assertions: Mapping[str, bool],
    runtime: Mapping[str, Any],
    elapsed_seconds: float,
) -> str:
    lines = [
        "# Genuine Mistral GGUF smoke report",
        "",
        "Status: **PASS**",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- GGUF repository revision: `{MODEL_REVISION}`",
        f"- GGUF file: `{MODEL_FILENAME}`",
        f"- GGUF size: `{MODEL_SIZE_BYTES}` bytes",
        f"- GGUF SHA-256: `{MODEL_SHA256}`",
        f"- Upstream checkpoint: `{UPSTREAM_CHECKPOINT}`",
        f"- llama.cpp commit: `{LLAMA_CPP_COMMIT}`",
        f"- Embedded template SHA-256: `{runtime['chat_template_sha256']}`",
        "- Rows: `8` (two reciprocal directed units × four conditions)",
        f"- Elapsed seconds: `{elapsed_seconds:.3f}`",
        "",
        "## Hard assertions",
        "",
    ]
    lines.extend(f"- PASS — {name}" for name in assertions)
    lines.append("")
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    smoke_output = args.smoke_output.resolve()
    smoke_output.mkdir(parents=True, exist_ok=True)
    run_log_path = smoke_output / "run.log"
    started = time.monotonic()
    runner: GGUFRunner | None = None
    signature: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    with run_log_path.open("a", encoding="utf-8") as run_log:
        _log(run_log, "BEGIN assigned-model genuine Mistral smoke")
        try:
            test_report = _load_passing_test_report(args.test_report.resolve())
            _log(run_log, "Complete unit-test suite precondition: PASS")
            model_files = verified_model_files(args.model_dir.resolve())
            _log(run_log, "Pinned single GGUF size and SHA-256 verification: PASS")
            llama_cpp_dir = args.llama_cpp_dir.resolve()
            observed_commit = command_output(
                ["git", "-C", str(llama_cpp_dir), "rev-parse", "HEAD"]
            )
            if observed_commit != LLAMA_CPP_COMMIT:
                raise RuntimeError("llama.cpp commit does not match the accepted Mistral smoke")
            if command_output(
                [
                    "git",
                    "-C",
                    str(llama_cpp_dir),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ]
            ) != "":
                raise RuntimeError("pinned llama.cpp checkout has tracked modifications")
            helper = args.helper.resolve()
            helper_linkage = validate_helper_linkage(helper, llama_cpp_dir)
            source_hashes = {
                "src/gguf_runner.py": ACCEPTED_GGUF_RUNNER_SHA256,
                "src/gguf_score_helper.cpp": ACCEPTED_GGUF_HELPER_SOURCE_SHA256,
            }
            for relative, expected in source_hashes.items():
                if sha256_file(PROJECT_ROOT / relative) != expected:
                    raise RuntimeError(f"accepted genuine scorer source drifted: {relative}")

            loader = DataLoader(str(PROJECT_ROOT / DATASET_PATH))
            if sha256_file(PROJECT_ROOT / DATASET_PATH) != DATASET_SHA256:
                raise RuntimeError("cleaned dataset hash mismatch")
            if sha256_file(PROJECT_ROOT / PAIR_MANIFEST) != PAIR_MANIFEST_SHA256:
                raise RuntimeError("full pair-manifest hash mismatch")
            if sha256_file(PROJECT_ROOT / SMOKE_MANIFEST) != SMOKE_MANIFEST_SHA256:
                raise RuntimeError("accepted smoke-manifest hash mismatch")
            loader.load_dataset()
            full_manifest = loader.get_question_pairs(str(PROJECT_ROOT / PAIR_MANIFEST))
            validate_manifest(full_manifest)
            smoke_manifest = loader.get_question_pairs(str(PROJECT_ROOT / SMOKE_MANIFEST))
            if [directed_key(pair) for pair in smoke_manifest] != list(EXPECTED_SMOKE_KEYS):
                raise RuntimeError("smoke manifest does not contain the accepted reciprocal units")
            if [dict(pair) for pair in smoke_manifest] != [dict(pair) for pair in full_manifest[:2]]:
                raise RuntimeError("smoke units do not exactly match the first two full units")

            runner = GGUFRunner(
                helper_path=helper,
                model_path=Path(model_files[0]["local_path"]),
                model_identifier=MODEL_IDENTIFIER,
                repository=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                quantization=QUANTIZATION,
                threads=args.threads,
                context_size=args.ctx_size,
                chat_profile=MISTRAL_V03_CHAT_PROFILE,
            )
            runner.load_model(run_log)
            runtime = runner.get_runtime_metadata()
            validate_runtime(runtime)
            signature = build_run_signature(
                PROJECT_ROOT,
                full_manifest,
                model_files,
                helper,
                runtime,
                llama_cpp_commit=observed_commit,
                threads=args.threads,
                context_size=args.ctx_size,
            )
            atomic_write_json(smoke_output / "run_signature.json", signature)
            atomic_write_json(
                smoke_output / "runtime.json",
                {
                    "recorded_at_utc": now_utc(),
                    "run_fingerprint": signature["fingerprint"],
                    "runtime": runtime,
                    "model_files": model_files,
                    "llama_cpp_commit": observed_commit,
                    "helper_ldd": helper_linkage,
                },
            )

            checkpoint_dir = smoke_output / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for manifest_index, pair in enumerate(smoke_manifest):
                unit_rows: list[dict[str, Any]] = []
                for condition in EXPECTED_OUTPUT_CONDITIONS:
                    row = run_condition_row(
                        runner,
                        loader,
                        pair,
                        condition,
                        model_files,
                        str(signature["fingerprint"]),
                    )
                    unit_rows.append(row)
                    rows.append(row)
                    row_key = (*directed_key(pair), condition)
                    digest = hashlib.sha256("\0".join(row_key).encode()).hexdigest()[:24]
                    atomic_write_json(
                        checkpoint_dir / f"row_{digest}.json",
                        {
                            "schema_version": SMOKE_SCHEMA_VERSION,
                            "metrics_status": "pending",
                            "row_key": list(row_key),
                            "row": row,
                        },
                    )
                    atomic_write_jsonl(smoke_output / "results.jsonl", rows)
                    _log(run_log, f"Genuine smoke row saved: {len(rows)}/8 {condition}")
                enrich_unit_metrics(unit_rows)
                for condition, row in zip(EXPECTED_OUTPUT_CONDITIONS, unit_rows):
                    row["metrics_status"] = "complete"
                    row_key = (*directed_key(pair), condition)
                    digest = hashlib.sha256("\0".join(row_key).encode()).hexdigest()[:24]
                    atomic_write_json(
                        checkpoint_dir / f"row_{digest}.json",
                        {
                            "schema_version": SMOKE_SCHEMA_VERSION,
                            "metrics_status": "complete",
                            "row_key": list(row_key),
                            "row": row,
                        },
                    )
                rows[manifest_index * 4 : manifest_index * 4 + 4] = unit_rows
                atomic_write_jsonl(smoke_output / "results.jsonl", rows)

            assertions = validate_smoke(
                rows,
                smoke_manifest,
                runtime,
                model_files,
                str(signature["fingerprint"]),
                test_report,
                checkpoint_dir,
            )
            gate = {
                "schema_version": SMOKE_SCHEMA_VERSION,
                "status": "PASS",
                "completed_at_utc": now_utc(),
                "rows": 8,
                "model_name": MODEL_NAME,
                "model_identifier": MODEL_IDENTIFIER,
                "model_revision": MODEL_REVISION,
                "run_fingerprint": signature["fingerprint"],
                "assertions": assertions,
                "runtime": runtime,
                "seed_rows": rows,
            }
            atomic_write_json(smoke_output / "genuine_smoke_gate.json", gate)
            (smoke_output / "GENUINE_SMOKE_REPORT.md").write_text(
                _smoke_report(assertions, runtime, time.monotonic() - started),
                encoding="utf-8",
            )
            _log(run_log, "All genuine-smoke hard assertions: PASS")
            _log(run_log, "Automatically continuing to the full 200-unit run")
        except Exception as exc:
            template_diagnostic = (
                runner.get_last_template_equivalence_diagnostic()
                if runner is not None
                else {}
            )
            atomic_write_jsonl(smoke_output / "results.jsonl", rows)
            atomic_write_json(
                smoke_output / "failure_diagnostic.json",
                {
                    "schema_version": SMOKE_SCHEMA_VERSION,
                    "status": "FAIL",
                    "failed_at_utc": now_utc(),
                    "rows_completed": len(rows),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "run_fingerprint": signature.get("fingerprint") if signature else None,
                    "template_equivalence_diagnostic": (
                        template_diagnostic or None
                    ),
                },
            )
            _log(run_log, f"FAILED; full run blocked: {type(exc).__name__}: {exc}")
            raise
        finally:
            if runner is not None:
                runner.close()
                _log(run_log, "Smoke model process released")

    _, completion = run_mistral_full(
        model_directory=args.model_dir,
        helper_path=args.helper,
        llama_cpp_directory=args.llama_cpp_dir,
        output_directory=args.full_output,
        threads=args.threads,
        context_size=args.ctx_size,
        smoke_gate=gate,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/analyze_mistral_gguf.py"),
            "--results",
            str(args.full_output / "results.jsonl"),
            "--manifest",
            str(PROJECT_ROOT / PAIR_MANIFEST),
            "--run-signature",
            str(args.full_output / "run_signature.json"),
            "--runtime",
            str(args.full_output / "runtime.json"),
            "--output-dir",
            str(args.analysis_output),
            "--bootstrap-replicates",
            "10000",
            "--classification-tolerance",
            "1e-12",
        ],
        check=True,
    )
    return {
        "status": "PASS",
        "smoke_rows": 8,
        "full_rows": completion["rows"],
        "full_directed_units": completion["directed_units"],
        "analysis_output": str(args.analysis_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--analysis-output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT)
    parser.add_argument("--threads", type=int, default=available_cpu_threads())
    parser.add_argument("--ctx-size", type=int, default=4096)
    args = parser.parse_args()
    if args.threads != available_cpu_threads():
        raise ValueError(
            f"production run must use all {available_cpu_threads()} safely available CPU threads"
        )
    if args.ctx_size <= 0:
        raise ValueError("context size must be positive")
    print(json.dumps(run_experiment(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["EXPECTED_SMOKE_KEYS", "SMOKE_SCHEMA_VERSION", "validate_smoke"]

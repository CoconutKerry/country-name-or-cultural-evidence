#!/usr/bin/env python3
"""Generate the self-contained Colab GPU handoff notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "cultural_alignment_audit_colab.ipynb"
_CELL_COUNTER = 0


def cell_id(kind: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{kind}-{_CELL_COUNTER:03d}"


def markdown(source: str, *, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id("markdown"),
        "metadata": {"tags": tags or []},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str, *, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id("code"),
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        # Cultural Alignment Audit — Colab GPU Handoff

        This notebook starts from a fresh Google Colab GPU session and a repaired
        repository ZIP. It runs the complete test suite before inference, then runs
        a **genuine** 4-bit `Qwen/Qwen2.5-7B-Instruct` smoke over two reciprocal
        directed units and all four conditions.

        **Safety boundary:** the four-model experiment is disabled in the shipped
        notebook. Stop after inspecting the genuine smoke. Do not enable the full
        cell until explicit approval has been given after that inspection. The
        earlier deterministic smoke remains a pipeline-only artifact and is never
        used by this notebook for model validation or paper statistics.
        """,
        tags=["scope", "approval-boundary"],
    ),
    code(
        """
        # User-editable storage settings. These defaults work with a direct upload.
        USE_GOOGLE_DRIVE = False
        DRIVE_ZIP_PATH = "/content/drive/MyDrive/cultural_alignment_audit_handoff.zip"
        DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/cultural_alignment_audit_outputs"

        # DO NOT change these until the genuine Qwen smoke has passed, its saved
        # records have been inspected, and explicit full-run approval is received.
        FULL_RUN_APPROVED = False
        FULL_RUN_APPROVAL_PHRASE = ""
        """,
        tags=["configuration", "full-run-disabled"],
    ),
    markdown(
        """
        ## 1. Upload or mount the repaired repository ZIP

        With the default settings, choose the supplied handoff ZIP in the upload
        dialog. To persist checkpoints across runtime loss, set
        `USE_GOOGLE_DRIVE = True`, upload the ZIP to the configured Drive path, and
        use the Drive output directory. No path from the repair environment is used.
        """
    ),
    code(
        """
        import os
        from pathlib import Path
        import shutil
        import stat
        import sys
        import zipfile

        from google.colab import drive, files

        EXTRACT_ROOT = Path("/content/cultural_alignment_audit_workspace")
        if USE_GOOGLE_DRIVE:
            drive.mount("/content/drive")
            candidate = Path(DRIVE_ZIP_PATH)
            if candidate.is_file():
                REPOSITORY_ZIP = candidate
            else:
                print(f"Drive ZIP not found at {candidate}; choose the handoff ZIP now.")
                uploaded = files.upload()
                zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
                if len(zip_names) != 1:
                    raise ValueError("Upload exactly one repository ZIP")
                REPOSITORY_ZIP = Path("/content") / zip_names[0]
            OUTPUT_ROOT = Path(DRIVE_OUTPUT_ROOT)
        else:
            uploaded = files.upload()
            zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
            if len(zip_names) != 1:
                raise ValueError("Upload exactly one repository ZIP")
            REPOSITORY_ZIP = Path("/content") / zip_names[0]
            OUTPUT_ROOT = Path("/content/cultural_alignment_audit_outputs")

        if EXTRACT_ROOT.exists():
            shutil.rmtree(EXTRACT_ROOT)
        EXTRACT_ROOT.mkdir(parents=True)

        # Reject path traversal and symbolic-link entries before extraction.
        with zipfile.ZipFile(REPOSITORY_ZIP) as archive:
            root_resolved = EXTRACT_ROOT.resolve()
            for member in archive.infolist():
                destination = (EXTRACT_ROOT / member.filename).resolve()
                if destination != root_resolved and root_resolved not in destination.parents:
                    raise ValueError(f"Unsafe ZIP member path: {member.filename!r}")
                file_type = (member.external_attr >> 16) & 0o170000
                if file_type == stat.S_IFLNK:
                    raise ValueError(f"Symbolic-link ZIP member is not allowed: {member.filename!r}")
            archive.extractall(EXTRACT_ROOT)

        roots = []
        for lock in EXTRACT_ROOT.rglob("requirements-colab.lock"):
            root = lock.parent
            markers = (
                root / "cultural_alignment_audit_colab.ipynb",
                root / "RUNBOOK_COLAB.md",
                root / "src/model_runner.py",
                root / "data/pairs/country_pairs_v2.json",
                root / "tests",
            )
            if all(marker.exists() for marker in markers):
                roots.append(root)
        if len(roots) != 1:
            raise RuntimeError(f"Expected one repaired repository root, found {roots}")
        REPO_ROOT = roots[0].resolve()
        output_resolved = OUTPUT_ROOT.resolve()
        if output_resolved == REPO_ROOT or REPO_ROOT in output_resolved.parents:
            raise ValueError("OUTPUT_ROOT must be separate from the extracted repository")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        os.chdir(REPO_ROOT)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        print("Repository:", REPO_ROOT)
        print("Persistent output recommended:", USE_GOOGLE_DRIVE)
        print("Output root:", OUTPUT_ROOT)
        """,
        tags=["repository-setup"],
    ),
    markdown(
        """
        ## 2. Install exact dependencies

        Run this before importing PyTorch or Transformers. If Colab explicitly asks
        for a runtime restart, restart and rerun from the repository setup cell.
        """
    ),
    code(
        """
        import subprocess
        import sys

        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--requirement",
                str(REPO_ROOT / "requirements-colab.lock"),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (OUTPUT_ROOT / "dependency_install.txt").write_text(
            install.stdout, encoding="utf-8"
        )
        print(install.stdout[-12000:])
        install.check_returncode()

        check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(check.stdout)
        (OUTPUT_ROOT / "pip_check.txt").write_text(check.stdout, encoding="utf-8")
        if check.returncode:
            print(
                "WARNING: Colab's unrelated preinstalled packages report dependency "
                "conflicts. Exact audit dependency versions are verified in the next cell."
            )
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        (OUTPUT_ROOT / "package_freeze.txt").write_text(
            freeze.stdout, encoding="utf-8"
        )
        print("Pinned dependencies installed; full package snapshot saved.")
        """,
        tags=["dependencies", "pre-inference"],
    ),
    markdown("## 3. Record and validate the GPU environment"),
    code(
        """
        from importlib import metadata as importlib_metadata
        import json

        import bitsandbytes
        import torch

        from src.colab_runner import atomic_write_json, collect_environment

        environment = collect_environment()
        observed_versions = {
            name: importlib_metadata.version(name)
            for name in (
                "torch", "torchvision", "torchaudio", "transformers", "accelerate", "bitsandbytes",
                "tokenizers", "safetensors", "huggingface-hub", "numpy",
                "scipy", "pandas", "PyYAML", "pytest",
                "Jinja2", "MarkupSafe", "sentencepiece", "protobuf",
            )
        }
        expected_versions = {}
        for line in (REPO_ROOT / "requirements-colab.lock").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, version = line.split("==", 1)
                expected_versions[name] = version
        mismatches = {
            name: {"expected": expected, "observed": observed_versions.get(name)}
            for name, expected in expected_versions.items()
            if observed_versions.get(name) != expected
        }
        environment["direct_dependency_versions"] = observed_versions
        environment["expected_direct_dependency_versions"] = expected_versions
        environment["dependency_version_mismatches"] = mismatches
        atomic_write_json(OUTPUT_ROOT / "environment.json", environment)
        print(json.dumps(environment, indent=2))

        assert environment.get("nvidia_smi_available"), "nvidia-smi is unavailable"
        assert environment.get("cuda_available"), "Select a Colab GPU runtime before proceeding"
        assert environment.get("torch_cuda_version"), "PyTorch does not report a CUDA build"
        assert environment.get("gpu") and environment["gpu"]["total_memory_bytes"] > 0
        assert not mismatches, f"Pinned dependency mismatch: {mismatches}"
        print("GPU environment gate: PASS")
        """,
        tags=["environment", "pre-inference"],
    ),
    markdown(
        """
        ## 4. Run the complete unit-test suite before inference

        This is a hard gate. It also validates the cleaned 200-unit reciprocal
        manifest. No model is loaded if either command fails.
        """
    ),
    code(
        """
        import subprocess
        import sys

        commands = [
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
            [sys.executable, "scripts/validate_data.py"],
        ]
        TESTS_PASSED = False
        transcripts = []
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            transcript = "$ " + " ".join(command) + "\\n" + completed.stdout
            transcripts.append(transcript)
            print(transcript)
            (OUTPUT_ROOT / "unit_tests.txt").write_text(
                "\\n\\n".join(transcripts), encoding="utf-8"
            )
            completed.check_returncode()
        TESTS_PASSED = True
        print("Pre-inference test gate: PASS")
        """,
        tags=["tests", "pre-inference", "required"],
    ),
    markdown(
        """
        ## 5. Read credentials securely and check exact model access

        Add a read-only `HF_TOKEN` in Colab's Secrets panel (key icon), or provide
        it through the process environment. The token value is never printed,
        saved, passed on the command line, or written into the notebook.
        """
    ),
    code(
        """
        import json

        from src.colab_runner import (
            MODEL_SPECS,
            atomic_write_json,
            check_model_access,
            get_hf_token,
        )

        assert TESTS_PASSED is True
        HF_TOKEN = get_hf_token()
        print("HF token available:", bool(HF_TOKEN))
        model_access = check_model_access(MODEL_SPECS, HF_TOKEN)
        atomic_write_json(OUTPUT_ROOT / "model_access.json", model_access)
        print(json.dumps(model_access, indent=2))

        qwen_id = "Qwen/Qwen2.5-7B-Instruct"
        if not model_access[qwen_id]["accessible"]:
            raise PermissionError(
                "The exact pinned Qwen smoke checkpoint is inaccessible; no substitute is allowed."
            )
        print("Exact Qwen checkpoint access: PASS")
        """,
        tags=["credentials", "model-access", "pre-inference"],
    ),
    markdown(
        """
        ## 6. Genuine Qwen smoke — enabled

        This loads exactly `Qwen/Qwen2.5-7B-Instruct` at the pinned commit in
        4-bit NF4 mode, scores complete displayed option-label continuations,
        runs two reciprocal directed units × four conditions, atomically saves
        each complete unit, validates every required field/assertion, and frees
        the model and CUDA cache in a `finally` block.
        """,
        tags=["genuine-smoke"],
    ),
    code(
        """
        import json

        from src.colab_runner import genuine_qwen_smoke

        assert TESTS_PASSED is True
        assert environment["cuda_available"] is True
        SMOKE_ROWS, SMOKE_COMPLETION, SMOKE_ASSERTIONS = genuine_qwen_smoke(
            REPO_ROOT, OUTPUT_ROOT, HF_TOKEN
        )
        print(json.dumps(SMOKE_COMPLETION, indent=2))
        print(json.dumps(SMOKE_ASSERTIONS, indent=2))
        assert SMOKE_ASSERTIONS["status"] == "PASS"
        assert len(SMOKE_ROWS) == 8
        print("Genuine Qwen smoke: PASS; model and CUDA memory released.")
        """,
        tags=["genuine-smoke", "model-inference"],
    ),
    markdown("## 7. Inspect the saved genuine-smoke records"),
    code(
        """
        import json
        from pathlib import Path

        from src.colab_runner import MODEL_SPECS, read_jsonl

        smoke_path = OUTPUT_ROOT / "smoke" / f"results_{MODEL_SPECS[0]['slug']}.jsonl"
        saved_smoke = read_jsonl(smoke_path)
        required_fields = {
            "question_id", "label_country", "evidence_country", "condition",
            "original_answer_options", "displayed_option_labels",
            "full_chat_templated_prompt", "candidate_label_token_ids",
            "raw_label_log_probabilities", "normalized_option_probabilities",
            "label_country_human_distribution", "evidence_country_human_distribution",
            "base2_jensen_shannon_distances", "country_influence",
            "evidence_influence", "evidence_override",
        }
        for index, row in enumerate(saved_smoke):
            missing = required_fields - set(row)
            assert not missing, f"smoke row {index} missing {sorted(missing)}"
        print("Saved rows:", len(saved_smoke))
        print("Directed keys:", sorted({
            (row["question_id"], row["label_country"], row["evidence_country"])
            for row in saved_smoke
        }))
        print("Conditions:", sorted({row["condition"] for row in saved_smoke}))
        print("First row audit fields:")
        print(json.dumps({key: saved_smoke[0][key] for key in sorted(required_fields)}, indent=2))
        print("Inspect the full JSONL and smoke_assertions.json before seeking approval.")
        """,
        tags=["genuine-smoke", "inspection"],
    ),
    markdown(
        """
        # STOP HERE — full experiment not approved

        Download or retain `environment.json`, `unit_tests.txt`,
        `model_access.json`, the two per-unit smoke checkpoints, the eight-row
        genuine-smoke JSONL, and `smoke_assertions.json`. Obtain explicit approval
        only after those artifacts have been inspected.

        The remaining cells are included for the later approved handoff. They are
        tagged `full-run-disabled` or `full-run-dependent` and fail closed with the
        shipped `FULL_RUN_APPROVED = False` and blank approval phrase.
        """,
        tags=["stop", "approval-boundary"],
    ),
    markdown(
        """
        ## 8. Full four-model run — disabled pending explicit approval

        When and only when approval is given after smoke inspection, set
        `FULL_RUN_APPROVED = True` and enter the exact approval phrase documented
        in `RUNBOOK_COLAB.md`. The function rechecks the genuine-smoke report and
        all four exact Hub revisions before loading any full-run weights. It uses
        all 200 reciprocal directed units and all four conditions, one model at a
        time, with per-unit checkpoints and CUDA cleanup between models.
        """,
        tags=["full-run-disabled", "approval-required"],
    ),
    code(
        """
        # SHIPPED DISABLED. Executing this unchanged raises PermissionError before
        # any full-run model is loaded.
        from src.colab_runner import run_full_experiment

        assert TESTS_PASSED is True, "Run the complete current-session test gate first"
        assert SMOKE_ASSERTIONS["status"] == "PASS"
        FULL_ROWS = run_full_experiment(
            REPO_ROOT,
            OUTPUT_ROOT,
            HF_TOKEN,
            full_run_approved=FULL_RUN_APPROVED,
            approval_phrase=FULL_RUN_APPROVAL_PHRASE,
        )
        print("Approved full result rows:", len(FULL_ROWS))
        """,
        tags=["full-run-disabled", "approval-required", "model-inference"],
    ),
    markdown(
        """
        ## 9. Question-clustered bootstrap — full-run dependent

        After all four approved model outputs validate, this constructs the strict
        repaired payload and performs 10,000 percentile bootstrap replicates at
        the `question_id` level. A sampled question carries all of its directed
        pairs, reciprocal directions, conditions, and models; individual condition
        rows are never bootstrapped independently.
        """,
        tags=["full-run-dependent"],
    ),
    code(
        """
        from scripts.bootstrap_analysis import analyze_with_clustered_bootstrap
        from src.colab_runner import (
            MODEL_SPECS,
            atomic_write_json,
            build_repaired_payload,
            load_full_result_rows,
            load_saved_runtime_metadata,
        )

        full_rows = load_full_result_rows(OUTPUT_ROOT)
        full_runtimes = load_saved_runtime_metadata(
            OUTPUT_ROOT, MODEL_SPECS, run_kind="full"
        )
        strict_payload = build_repaired_payload(
            REPO_ROOT,
            full_rows,
            "data/pairs/country_pairs_v2.json",
            full_runtimes,
        )
        bootstrap_analysis = analyze_with_clustered_bootstrap(
            strict_payload,
            allow_synthetic=False,
            n_replicates=10_000,
            confidence_level=0.95,
            seed=42,
        )
        analysis_path = OUTPUT_ROOT / "analysis/clustered_bootstrap.json"
        atomic_write_json(analysis_path, bootstrap_analysis)
        method = bootstrap_analysis["clustered_bootstrap"]["method"]
        assert method["cluster_key"] == "question_id"
        assert method["n_replicates"] == 10_000
        assert method["independent_condition_row_resampling"] is False
        print("Clustered bootstrap saved:", analysis_path)
        """,
        tags=["full-run-dependent", "statistics"],
    ),
    markdown("## 10. Validate completion and package downloadable results"),
    code(
        """
        # Full-run completion check. The strict analysis above has already rejected
        # duplicates, missing directions, missing conditions, or malformed vectors.
        from collections import Counter, defaultdict

        counts = Counter(row["model_name"] for row in full_rows)
        assert set(counts) == {spec["name"] for spec in MODEL_SPECS}
        assert all(count == 800 for count in counts.values()), counts
        directed_by_model = defaultdict(set)
        for row in full_rows:
            directed_by_model[row["model_name"]].add(
                (row["question_id"], row["label_country"], row["evidence_country"])
            )
        assert all(len(keys) == 200 for keys in directed_by_model.values())
        print("Full-run completion gate: PASS", dict(counts))
        """,
        tags=["full-run-dependent", "validation"],
    ),
    code(
        """
        # Package outputs plus the exact notebook/code/tests/data identities used.
        import hashlib
        from pathlib import Path
        import zipfile

        result_zip = OUTPUT_ROOT / "cultural_alignment_audit_results.zip"
        result_checksum = OUTPUT_ROOT / "cultural_alignment_audit_results.zip.sha256"
        include_repo_paths = [
            "cultural_alignment_audit_colab.ipynb",
            "RUNBOOK_COLAB.md",
            "requirements-colab.lock",
            "requirements-inference.lock",
            "src",
            "scripts",
            "tests",
            "data/processed/dataset_v1.json",
            "data/processed/dataset_v2.json",
            "data/pairs",
            "data/audit",
            "audit",
            "experiments/smoke",
        ]

        files_to_add = []
        for relative in include_repo_paths:
            source = REPO_ROOT / relative
            if source.is_dir():
                files_to_add.extend(path for path in source.rglob("*") if path.is_file())
            elif source.is_file():
                files_to_add.append(source)
        output_files = [
            path for path in OUTPUT_ROOT.rglob("*")
            if path.is_file() and path not in {result_zip, result_checksum}
        ]

        # Fail if the in-memory credential accidentally appears in any artifact.
        if HF_TOKEN:
            token_bytes = HF_TOKEN.encode("utf-8")
            for path in files_to_add + output_files:
                if token_bytes in path.read_bytes():
                    raise RuntimeError(f"Credential material detected in {path}")

        with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(files_to_add)):
                archive.write(path, Path("repository") / path.relative_to(REPO_ROOT))
            for path in sorted(set(output_files)):
                archive.write(path, Path("outputs") / path.relative_to(OUTPUT_ROOT))
        digest = hashlib.sha256(result_zip.read_bytes()).hexdigest()
        result_checksum.write_text(f"{digest}  {result_zip.name}\\n", encoding="utf-8")
        print("Result package:", result_zip)
        print("SHA-256:", digest)
        print("Download both the ZIP and checksum from the Colab Files pane or Drive.")
        """,
        tags=["packaging"],
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)

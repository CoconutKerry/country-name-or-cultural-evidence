#!/usr/bin/env python3
"""Run the complete repository test/data gate and bind its passing source state."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "audit"
RESULTS_PATH = AUDIT_DIR / "GEMMA_GGUF_PREINFERENCE_TEST_RESULTS.txt"
GATE_PATH = AUDIT_DIR / "GEMMA_GGUF_PREINFERENCE_GATE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validated_paths() -> list[Path]:
    paths = []
    for directory in ("src", "scripts", "tests"):
        paths.extend((PROJECT_ROOT / directory).rglob("*.py"))
    paths.append(PROJECT_ROOT / "src/gguf_score_helper.cpp")
    for relative in (
        "requirements-inference.lock",
        "data/processed/dataset_v2.json",
        "data/pairs/country_pairs_v2.json",
        "data/pairs/country_pairs_smoke.json",
        "data/audit/data_repair_summary.json",
    ):
        paths.append(PROJECT_ROOT / relative)
    return sorted(set(paths), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def run_gate() -> dict:
    # Invalidate any prior PASS before launching the current checks.  An
    # interrupted or failed rerun must never leave an older gate reusable.
    atomic_write(
        GATE_PATH,
        json.dumps(
            {
                "status": "RUNNING",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    test_command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    ]
    tests = subprocess.run(
        test_command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    test_text = tests.stdout + tests.stderr
    match = re.search(r"Ran (\d+) tests?", test_text)
    test_count = int(match.group(1)) if match else None
    data_command = [sys.executable, "scripts/validate_data.py"]
    data = subprocess.run(
        data_command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    data_text = data.stdout + data.stderr
    transcript = (
        "COMPLETE UNIT TEST SUITE\n"
        "========================\n"
        f"Command: {' '.join(test_command)}\n\n{test_text}\n"
        "REPAIRED DATA VALIDATION\n"
        "========================\n"
        f"Command: {' '.join(data_command)}\n\n{data_text}\n"
    )
    atomic_write(RESULTS_PATH, transcript)
    if tests.returncode != 0 or data.returncode != 0 or not test_count:
        atomic_write(
            GATE_PATH,
            json.dumps(
                {
                    "status": "FAILED",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "unit_test_count": test_count,
                    "unit_test_returncode": tests.returncode,
                    "data_validation_returncode": data.returncode,
                    "test_transcript": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
                    "test_transcript_sha256": sha256_file(RESULTS_PATH),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        raise RuntimeError(
            f"pre-inference gate failed: tests={tests.returncode}, data={data.returncode}"
        )
    paths = validated_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"validated source files are missing: {missing}")
    gate = {
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_unit_test_suite": True,
        "unit_test_count": test_count,
        "unit_test_command": test_command,
        "unit_test_returncode": tests.returncode,
        "data_validation": "PASS",
        "data_validation_command": data_command,
        "data_validation_returncode": data.returncode,
        "test_transcript": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "test_transcript_sha256": sha256_file(RESULTS_PATH),
        "validated_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths
        },
    }
    atomic_write(GATE_PATH, json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return gate


if __name__ == "__main__":
    print(json.dumps(run_gate(), sort_keys=True))

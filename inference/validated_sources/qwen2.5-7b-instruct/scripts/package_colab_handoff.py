#!/usr/bin/env python3
"""Validate and create the offline Colab GPU handoff archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "cultural-alignment-audit-main"
REQUIRED_PATHS = (
    "cultural_alignment_audit_colab.ipynb",
    "RUNBOOK_COLAB.md",
    "requirements-colab.lock",
    "requirements-inference.lock",
    "src/colab_runner.py",
    "src/model_runner.py",
    "src/clustered_bootstrap.py",
    "scripts/bootstrap_analysis.py",
    "tests/test_colab_handoff.py",
    "tests/test_clustered_bootstrap.py",
    "data/processed/dataset_v2.json",
    "data/pairs/country_pairs_v2.json",
    "data/pairs/country_pairs_smoke.json",
    "experiments/smoke/results_smoke.json",
    "experiments/smoke/analysis_smoke.json",
    "audit/COLAB_HANDOFF_VALIDATION.md",
    "audit/test_results.txt",
    "audit/changed_files.txt",
)
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp"}
TOKEN_PATTERN = re.compile(rb"hf_[A-Za-z0-9]{20,}")


def validate_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Notebook is not a valid nbformat-4 object")
    all_source = "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])
    for forbidden in ("/workspace/scratch", "SyntheticModelRunner("):
        if forbidden in all_source:
            raise ValueError(f"Forbidden notebook content: {forbidden!r}")
    required_fragments = (
        "FULL_RUN_APPROVED = False",
        'FULL_RUN_APPROVAL_PHRASE = ""',
        "genuine_qwen_smoke",
        "Qwen/Qwen2.5-7B-Instruct",
        "n_replicates=10_000",
        'method["cluster_key"] == "question_id"',
        "candidate_label_token_ids",
        "base2_jensen_shannon_distances",
    )
    for fragment in required_fragments:
        if fragment not in all_source:
            raise ValueError(f"Notebook is missing required content: {fragment!r}")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile(str(cell.get("source", "")), f"notebook-cell-{index}", "exec")


def selected_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_NAMES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not allowed in the handoff: {relative}")
        if not path.is_file() or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if path.name == "experiment.log":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix())


def validate_repository(files: list[Path]) -> None:
    missing = [relative for relative in REQUIRED_PATHS if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required handoff files: {missing!r}")
    validate_notebook(PROJECT_ROOT / "cultural_alignment_audit_colab.ipynb")
    pair_manifest = json.loads(
        (PROJECT_ROOT / "data/pairs/country_pairs_v2.json").read_text(encoding="utf-8")
    )
    keys = {
        (item["question_id"], item["country"], item["conflict_country"])
        for item in pair_manifest
    }
    if len(pair_manifest) != 200 or len(keys) != 200:
        raise ValueError("Cleaned full manifest must contain 200 unique directed units")
    if any((question, right, left) not in keys for question, left, right in keys):
        raise ValueError("Cleaned full manifest must be reciprocal")
    for path in files:
        data = path.read_bytes()
        match = TOKEN_PATTERN.search(data)
        if match:
            raise ValueError(f"Possible Hugging Face credential in {path.relative_to(PROJECT_ROOT)}")


def write_archive(destination: Path, files: list[Path]) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            mode = path.stat().st_mode
            executable = bool(mode & stat.S_IXUSR) or path.suffix in {".sh"}
            info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Archive CRC validation failed at {bad}")
        names = set(archive.namelist())
        for relative in REQUIRED_PATHS:
            expected = f"{ARCHIVE_ROOT}/{relative}"
            if expected not in names:
                raise RuntimeError(f"Archive is missing {expected}")
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = selected_files()
    validate_repository(files)
    digest = write_archive(args.output.resolve(), files)
    checksum_path = args.output.resolve().with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(f"HANDOFF PASS: {len(files)} files -> {args.output.resolve()}")
    print(f"SHA-256: {digest}")
    print(f"Checksum: {checksum_path}")


if __name__ == "__main__":
    main()

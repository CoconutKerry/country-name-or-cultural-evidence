#!/usr/bin/env python3
"""Regenerate immutable-input and full-release SHA-256 manifests."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_files() -> list[Path]:
    roots = [
        ROOT / "configs",
        ROOT / "data",
        ROOT / "results/raw",
        ROOT / "inference",
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "docs",
        ROOT / "prompts",
    ]
    files: list[Path] = [
        ROOT / "README.md",
        ROOT / "Makefile",
        ROOT / "pyproject.toml",
        ROOT / "requirements-analysis.lock",
        ROOT / "LICENSES.md",
        ROOT / "VERSION",
        ROOT / "CITATION.cff.template",
    ]
    for base in roots:
        files.extend(path for path in base.rglob("*") if path.is_file())
    return sorted({path for path in files if path.exists() and "__pycache__" not in path.parts and path.suffix != ".pyc"})


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix != ".pyc"
        and path.name != "RELEASE_SHA256SUMS.txt"
    )


def write_manifest(path: Path, files: list[Path]) -> None:
    lines = [f"{digest(item)}  {item.relative_to(ROOT).as_posix()}" for item in files]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_manifest(ROOT / "provenance/INPUT_SHA256SUMS.txt", input_files())
    write_manifest(ROOT / "provenance/RELEASE_SHA256SUMS.txt", release_files())
    print("Wrote input and release SHA-256 manifests.")


if __name__ == "__main__":
    main()

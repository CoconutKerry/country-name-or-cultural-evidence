"""Immutable provenance and local verification for Gemma-2-9B-It Q4_K_M."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence


MODEL_NAME = "Gemma-2-9B-It"
MODEL_REPOSITORY = "bartowski/gemma-2-9b-it-GGUF"
MODEL_REVISION = "d731033f3dc4018261fd39896e50984d398b4ac5"
MODEL_IDENTIFIER = f"{MODEL_REPOSITORY}:Q4_K_M"
UPSTREAM_CHECKPOINT = "google/gemma-2-9b-it"
UPSTREAM_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
UPSTREAM_REVISION_NOTE = (
    "current pinned upstream revision; the quantizer does not attest that this "
    "was the exact historical conversion-source revision"
)
QUANTIZATION = "Q4_K_M"
MODEL_FILES = (
    {
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "size_bytes": 5_761_057_728,
        "sha256": "13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe",
        "role": "selected_entrypoint",
    },
)
MODEL_FILE_INTRODUCING_COMMIT = "9e6247fc6440e1f0b9a320edcb783829aa9b5077"
LLAMA_CPP_COMMIT = "62acc89c26c66076cb72e049f307fbe93b8b9750"
LLAMA_CPP_VERSION = "0.3.0-dev"
EXPECTED_ARCHITECTURE = "gemma2"
EXPECTED_CONTEXT_LENGTH = 8192
EXPECTED_PARAMETER_COUNT = 9_241_705_984


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.run(
            list(command), check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_llama_cpp_checkout(llama_cpp_directory: Path) -> dict[str, Any]:
    directory = Path(llama_cpp_directory).resolve()
    if not (directory / ".git").exists():
        raise FileNotFoundError(f"llama.cpp checkout is missing: {directory}")
    head = command_output(["git", "-C", str(directory), "rev-parse", "HEAD"])
    if head != LLAMA_CPP_COMMIT:
        raise RuntimeError(
            f"llama.cpp commit mismatch: {head!r} != {LLAMA_CPP_COMMIT!r}"
        )
    dirty = command_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"]
    )
    if dirty:
        raise RuntimeError("tracked llama.cpp sources are dirty")
    return {
        "directory": str(directory),
        "commit": head,
        "version": LLAMA_CPP_VERSION,
        "tracked_sources_clean": True,
    }


def verified_model_files(model_dir: Path) -> list[dict[str, Any]]:
    verified = []
    for expected in MODEL_FILES:
        path = Path(model_dir) / str(expected["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"official Gemma GGUF is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected["size_bytes"]:
            raise RuntimeError(
                f"GGUF size mismatch for {path.name}: "
                f"{actual_size} != {expected['size_bytes']}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"GGUF SHA-256 mismatch for {path.name}: "
                f"{actual_hash} != {expected['sha256']}"
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


def environment_information(repository_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(repository_root)
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except (OSError, IndexError):
        pass
    return {
        "captured_at_utc": now_utc(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": {
            "model": cpu_model or platform.processor() or None,
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "lscpu": command_output(["lscpu"]),
        },
        "memory": memory_information(),
        "disk": {
            "path": str(repository_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "cuda_used": False,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
    }


def provenance_record(files: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "paper_model_name": MODEL_NAME,
        "gguf_repository": MODEL_REPOSITORY,
        "gguf_repository_revision": MODEL_REVISION,
        "gguf_file_introducing_commit": MODEL_FILE_INTRODUCING_COMMIT,
        "model_identifier": MODEL_IDENTIFIER,
        "quantization": QUANTIZATION,
        "files": [dict(item) for item in files],
        "upstream_checkpoint": UPSTREAM_CHECKPOINT,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_revision_note": UPSTREAM_REVISION_NOTE,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "command_output",
    "environment_information",
    "now_utc",
    "provenance_record",
    "sha256_file",
    "verified_model_files",
    "verify_llama_cpp_checkout",
]

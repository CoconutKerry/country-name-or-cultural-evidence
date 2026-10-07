"""Immutable provenance contract for the assigned Mistral GGUF experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence


MODEL_NAME = "Mistral-7B-Instruct-v0.3"
UPSTREAM_CHECKPOINT = "mistralai/Mistral-7B-Instruct-v0.3"
UPSTREAM_REVISION = None
UPSTREAM_REVISION_NOT_RECORDED_BY_QUANTIZER = True
MODEL_REPOSITORY = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
MODEL_REVISION = "61fd4167fff3ab01ee1cfe0da183fa27a944db48"
MODEL_FILENAME = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
MODEL_SIZE_BYTES = 4_372_812_000
MODEL_SHA256 = "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6"
MODEL_IDENTIFIER = f"{MODEL_REPOSITORY}@{MODEL_REVISION}:{MODEL_FILENAME}"
QUANTIZATION = "Q4_K_M"
REMOTE_MODEL_LAST_MODIFIED = "2024-05-22T19:09:41Z"
REMOTE_WEIGHT_COMMIT = None
MODEL_SHARDS = (
    {
        "filename": MODEL_FILENAME,
        "size_bytes": MODEL_SIZE_BYTES,
        "sha256": MODEL_SHA256,
        "role": "selected_entrypoint",
    },
)

LLAMA_CPP_COMMIT = "62acc89c26c66076cb72e049f307fbe93b8b9750"
LLAMA_CPP_VERSION = "0.3.0-dev"
# Metadata-only preflight value. The loaded GGUF must independently reproduce it.
MISTRAL_CHAT_TEMPLATE_SHA256 = (
    "26a59556925c987317ce5291811ba3b7f32ec4c647c400c6cc7e3a9993007ba7"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_information(),
        "memory": memory_information(),
        "disk": {
            "path": str(Path(repository_root).resolve()),
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


def verified_model_files(model_directory: Path) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    model_directory = Path(model_directory).resolve()
    for expected in MODEL_SHARDS:
        path = model_directory / str(expected["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"pinned Mistral GGUF is missing: {path}")
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
                "local_path": str(path),
                "actual_size_bytes": actual_size,
                "actual_sha256": actual_hash,
                "verified": True,
            }
        )
    return verified


__all__ = [
    "LLAMA_CPP_COMMIT",
    "LLAMA_CPP_VERSION",
    "MISTRAL_CHAT_TEMPLATE_SHA256",
    "MODEL_FILENAME",
    "MODEL_IDENTIFIER",
    "MODEL_NAME",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "MODEL_SHARDS",
    "MODEL_SIZE_BYTES",
    "QUANTIZATION",
    "REMOTE_MODEL_LAST_MODIFIED",
    "REMOTE_WEIGHT_COMMIT",
    "UPSTREAM_CHECKPOINT",
    "UPSTREAM_REVISION",
    "UPSTREAM_REVISION_NOT_RECORDED_BY_QUANTIZER",
    "command_output",
    "environment_information",
    "sha256_file",
    "verified_model_files",
]

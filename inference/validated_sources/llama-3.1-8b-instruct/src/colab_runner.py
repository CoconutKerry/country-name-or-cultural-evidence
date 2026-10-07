"""GPU/Colab handoff runner with resumable, auditable label scoring.

This module is intentionally separate from the dependency-free synthetic smoke
path in :mod:`src.main`.  It permits only genuine Hugging Face inference, saves
one atomic checkpoint per complete directed experimental unit, and keeps the
four-model run behind an explicit two-part approval gate.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from src.data_loader import DataLoader
from src.main import (
    FOUR_CONDITIONS,
    build_prompt,
    prepare_evidence_distribution,
    reorder_distribution,
    shuffle_options,
    stable_unit_seed,
)
from src.metrics import Metrics
from src.model_runner import ModelRunner
from src.prompt_builder import PromptBuilder
from src.scoring import (
    map_label_probabilities,
    normalize_log_scores,
    option_labels,
    recover_canonical_distribution,
)


SCHEMA_VERSION = "colab-gpu-v1"
SCORING_METHOD = "complete_option_label_continuation_log_likelihood"
FULL_APPROVAL_PHRASE = "I APPROVE THE FULL FOUR-MODEL RUN"
DEFAULT_SEED = 42

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "Qwen2.5-7B-Instruct",
        "slug": "qwen2.5-7b-instruct",
        "hub_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "trust_remote_code": False,
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "slug": "llama-3.1-8b-instruct",
        "hub_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "tokenizer_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "trust_remote_code": False,
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "slug": "mistral-7b-instruct-v0.3",
        "hub_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "tokenizer_revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "trust_remote_code": False,
    },
    {
        "name": "Gemma-2-9B-It",
        "slug": "gemma-2-9b-it",
        "hub_id": "google/gemma-2-9b-it",
        "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
        "tokenizer_revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
        "trust_remote_code": False,
        "attn_implementation": "eager",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write one file atomically, including an fsync before replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def get_hf_token() -> Optional[str]:
    """Read a token from the environment or a Colab secret without persisting it."""

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:  # Only available inside Google Colab.
        from google.colab import userdata  # type: ignore

        secret = userdata.get("HF_TOKEN")
        return secret or None
    except Exception:
        # Colab uses provider-specific SecretNotFound/NotebookAccess/Timeout
        # exception classes that are not importable outside Colab.  Missing or
        # unshared credentials are represented uniformly as no token.
        return None


def _redact_error(message: str, token: Optional[str]) -> str:
    return message.replace(token, "[REDACTED]") if token else message


def check_model_access(
    specs: Sequence[Mapping[str, Any]],
    token: Optional[str],
) -> dict[str, dict[str, Any]]:
    """Check exact Hub checkpoints without downloading model weights."""

    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    except ImportError as exc:  # pragma: no cover - Colab dependency cell supplies it
        raise RuntimeError("Install requirements-colab.lock before checking model access") from exc

    api = HfApi(token=token)
    report: dict[str, dict[str, Any]] = {}
    for spec in specs:
        hub_id = str(spec["hub_id"])
        requested = str(spec["revision"])
        item: dict[str, Any] = {
            "model_name": spec["name"],
            "hub_id": hub_id,
            "requested_revision": requested,
            "token_present": bool(token),
        }
        try:
            info = api.model_info(hub_id, revision=requested, token=token)
            resolved = getattr(info, "sha", None)
            item["resolved_revision"] = resolved
            if resolved != requested:
                item.update(
                    status="revision_mismatch",
                    accessible=False,
                    error=f"Hub resolved {resolved!r}, expected {requested!r}",
                )
            else:
                siblings = {
                    getattr(sibling, "rfilename", None)
                    for sibling in (getattr(info, "siblings", None) or [])
                }
                metadata_probe_files = ("config.json", "tokenizer_config.json")
                missing_probe_files = [
                    filename for filename in metadata_probe_files if filename not in siblings
                ]
                weight_artifacts = sorted(
                    filename
                    for filename in siblings
                    if isinstance(filename, str)
                    and (
                        filename.endswith(".safetensors")
                        or filename.endswith(".safetensors.index.json")
                        or filename.endswith("pytorch_model.bin")
                        or filename.endswith("pytorch_model.bin.index.json")
                    )
                )
                has_weight_artifact = bool(weight_artifacts)
                if missing_probe_files or not has_weight_artifact:
                    raise FileNotFoundError(
                        "Exact repository revision lacks required inference artifacts "
                        f"(missing probes={missing_probe_files!r}, weights={has_weight_artifact})"
                    )
                weight_probe = next(
                    (
                        filename
                        for filename in weight_artifacts
                        if filename.endswith((".safetensors.index.json", ".bin.index.json"))
                    ),
                    weight_artifacts[0],
                )
                required_probe_files = (*metadata_probe_files, weight_probe)
                probed: dict[str, Any] = {}
                for filename in required_probe_files:
                    metadata = get_hf_file_metadata(
                        hf_hub_url(hub_id, filename=filename, revision=requested),
                        token=token,
                    )
                    metadata_commit = getattr(metadata, "commit_hash", None)
                    if metadata_commit is not None and metadata_commit != requested:
                        raise RuntimeError(
                            f"{filename} resolved to {metadata_commit!r}, expected {requested!r}"
                        )
                    probed[filename] = {
                        "accessible": True,
                        "commit_hash": metadata_commit,
                        "size_bytes": getattr(metadata, "size", None),
                    }
                item.update(
                    status="accessible",
                    accessible=True,
                    error=None,
                    downloadable_file_probes=probed,
                    weight_artifact_declared=True,
                )
        except Exception as exc:  # Hub exception classes vary across pinned versions.
            item.update(
                status="inaccessible_or_gated",
                accessible=False,
                resolved_revision=None,
                error_type=type(exc).__name__,
                error=_redact_error(str(exc), token),
            )
        report[hub_id] = item
    return report


def collect_environment() -> dict[str, Any]:
    """Collect the Colab runtime facts required before inference."""

    import platform
    import shutil
    import subprocess
    import sys

    memory: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(raw.strip().split()[0]) * 1024
    disk = shutil.disk_usage("/")
    nvidia_smi = shutil.which("nvidia-smi")
    report: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "ram_total_bytes": memory.get("MemTotal"),
        "ram_available_bytes": memory.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_available_bytes": disk.free,
        "nvidia_smi_available": bool(nvidia_smi),
        "gpu": None,
    }
    if nvidia_smi:
        command = [
            nvidia_smi,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        report["nvidia_smi_returncode"] = completed.returncode
        report["nvidia_smi_query"] = completed.stdout.strip()
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            report["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(properties.total_memory),
                "currently_free_memory_bytes": int(free_bytes),
                "currently_total_memory_bytes": int(total_bytes),
                "compute_capability": [properties.major, properties.minor],
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "device_count": int(torch.cuda.device_count()),
            }
    except ImportError:
        report.update(torch_version=None, torch_cuda_version=None, cuda_available=False)
    return report


def load_data_and_pairs(
    repository_root: Path,
    pair_manifest: str,
    *,
    expected_count: Optional[int] = None,
) -> tuple[DataLoader, list[dict[str, Any]]]:
    loader = DataLoader(str(repository_root / "data/processed/dataset_v2.json"))
    loader.load_dataset()
    pairs = loader.get_question_pairs(str(repository_root / pair_manifest))
    validate_reciprocal_pairs(pairs, expected_count=expected_count)
    return loader, pairs


def validate_reciprocal_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    expected_count: Optional[int] = None,
) -> None:
    keys = [
        (str(pair["question_id"]), str(pair["country"]), str(pair["conflict_country"]))
        for pair in pairs
    ]
    if expected_count is not None and len(keys) != expected_count:
        raise ValueError(f"Expected {expected_count} directed units, found {len(keys)}")
    if len(keys) != len(set(keys)):
        raise ValueError("Directed experimental-unit keys are not unique")
    key_set = set(keys)
    missing = [key for key in keys if (key[0], key[2], key[1]) not in key_set]
    if missing:
        raise ValueError(f"Directed unit manifest is not reciprocal: {missing[:3]!r}")


def build_quantized_runner(
    spec: Mapping[str, Any],
    token: Optional[str],
) -> tuple[ModelRunner, dict[str, Any]]:
    """Create, but do not yet load, a single-GPU 4-bit runner."""

    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - Colab dependency cell supplies it
        raise RuntimeError("Pinned Colab inference dependencies are not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required for the genuine model smoke")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "torch_dtype": compute_dtype,
        "quantization_config": quantization_config,
    }
    if spec.get("attn_implementation"):
        load_kwargs["attn_implementation"] = spec["attn_implementation"]
    quantization = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": str(compute_dtype),
        "device_map": "auto",
    }
    runner = ModelRunner(
        str(spec["name"]),
        str(spec["hub_id"]),
        revision=str(spec["revision"]),
        tokenizer_revision=str(spec.get("tokenizer_revision", spec["revision"])),
        use_chat_template=True,
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
        model_load_kwargs=load_kwargs,
        token=token,
    )
    return runner, quantization


def free_runner(runner: Optional[ModelRunner]) -> None:
    """Drop model/tokenizer references and release cached CUDA allocations."""

    if runner is not None:
        runner.model = None
        runner.tokenizer = None
        runner.device = None
        runner._token = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass


def build_run_signature(
    repository_root: Path,
    spec: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    pair_manifest: str,
    quantization: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
) -> dict[str, Any]:
    data_path = repository_root / "data/processed/dataset_v2.json"
    pair_path = repository_root / pair_manifest
    identity_paths = (
        "requirements-colab.lock",
        "src/colab_runner.py",
        "src/model_runner.py",
        "src/scoring.py",
        "src/metrics.py",
        "src/prompt_builder.py",
        "src/main.py",
        "src/data_loader.py",
    )
    code_identities: dict[str, str] = {}
    for relative_path in identity_paths:
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Run-signature artifact is missing: {path}")
        code_identities[relative_path] = sha256_file(path)
    core = {
        "schema_version": SCHEMA_VERSION,
        "scoring_method": SCORING_METHOD,
        "model": {
            key: spec[key]
            for key in ("name", "slug", "hub_id", "revision", "tokenizer_revision")
        },
        "conditions": list(FOUR_CONDITIONS),
        "directed_unit_key": ["question_id", "label_country", "evidence_country"],
        "directed_units": [
            [pair["question_id"], pair["country"], pair["conflict_country"]]
            for pair in pairs
        ],
        "dataset_sha256": sha256_file(data_path),
        "pair_manifest": pair_manifest,
        "pair_manifest_sha256": sha256_file(pair_path),
        "code_and_dependency_sha256": code_identities,
        "seed": int(seed),
        "shuffle_options": bool(shuffle),
        "quantization": dict(quantization),
        "runtime_identity": runtime_signature_identity(runtime_metadata),
        "jensen_shannon_base": 2,
    }
    return {**core, "fingerprint": canonical_json_hash(core)}


def runtime_signature_identity(runtime_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Select runtime facts that must not be mixed within resumed results."""

    fields = (
        "backend",
        "hub_id",
        "requested_model_revision",
        "requested_tokenizer_revision",
        "resolved_model_revision",
        "resolved_tokenizer_revision",
        "model_class",
        "tokenizer_class",
        "model_dtype",
        "is_loaded_in_4bit",
        "is_loaded_in_8bit",
        "use_chat_template",
        "chat_template_sha256",
        "torch_version",
        "transformers_version",
        "accelerate_version",
        "tokenizers_version",
        "safetensors_version",
        "cuda_version",
        "cuda_device",
        "python_version",
        "platform",
    )
    return {field: runtime_metadata.get(field) for field in fields}


def validate_loaded_runtime(
    runtime_metadata: Mapping[str, Any],
    spec: Mapping[str, Any],
    quantization: Mapping[str, Any],
) -> None:
    """Fail before prediction if a loaded model violates the scientific contract."""

    checks = {
        "genuine Hugging Face backend": runtime_metadata.get("backend") == "huggingface",
        "exact Hub repository": runtime_metadata.get("hub_id") == spec["hub_id"],
        "requested model revision": runtime_metadata.get("requested_model_revision")
        == spec["revision"],
        "resolved model revision": runtime_metadata.get("resolved_model_revision")
        == spec["revision"],
        "requested tokenizer revision": runtime_metadata.get(
            "requested_tokenizer_revision"
        )
        == spec["tokenizer_revision"],
        "resolved tokenizer revision": runtime_metadata.get(
            "resolved_tokenizer_revision"
        )
        == spec["tokenizer_revision"],
        "chat template enabled": runtime_metadata.get("use_chat_template") is True,
        "chat template identity": bool(runtime_metadata.get("chat_template_sha256")),
        "4-bit model load": runtime_metadata.get("is_loaded_in_4bit") is True,
        "not 8-bit": runtime_metadata.get("is_loaded_in_8bit") is False,
        "4-bit configuration": quantization.get("load_in_4bit") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Loaded model violates runtime contract: {failures!r}")


def _unit_key(pair: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(pair["question_id"]),
        str(pair["country"]),
        str(pair["conflict_country"]),
    )


def _checkpoint_path(checkpoint_dir: Path, spec: Mapping[str, Any], key: Sequence[str]) -> Path:
    digest = hashlib.sha256(
        (str(spec["hub_id"]) + "\0" + "\0".join(key)).encode("utf-8")
    ).hexdigest()[:20]
    return checkpoint_dir / str(spec["slug"]) / f"unit_{digest}.json"


def _require_saved(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_probability_mapping(
    value: Any,
    keys: Sequence[str],
    context: str,
) -> dict[str, float]:
    _require_saved(isinstance(value, Mapping), f"{context} must be an object")
    _require_saved(set(value) == set(keys), f"{context} has incompatible option keys")
    result: dict[str, float] = {}
    for key in keys:
        probability = value[key]
        _require_saved(
            isinstance(probability, (int, float)) and not isinstance(probability, bool),
            f"{context}[{key!r}] must be numeric",
        )
        number = float(probability)
        _require_saved(
            math.isfinite(number) and number >= 0.0,
            f"{context}[{key!r}] must be finite and non-negative",
        )
        result[key] = number
    _require_saved(
        math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-8),
        f"{context} must sum to one",
    )
    return result


def _same_distribution(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float = 1e-8,
) -> bool:
    return set(left) == set(right) and all(
        math.isclose(
            float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=tolerance
        )
        for key in left
    )


def _validate_checkpoint(
    checkpoint: Any,
    signature: Mapping[str, Any],
    key: tuple[str, str, str],
) -> list[dict[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint for {key!r} must be an object")
    if checkpoint.get("run_fingerprint") != signature["fingerprint"]:
        raise ValueError(f"Checkpoint run fingerprint mismatch for {key!r}")
    if checkpoint.get("directed_key") != list(key):
        raise ValueError(f"Checkpoint directed key mismatch for {key!r}")
    rows = checkpoint.get("rows")
    if not isinstance(rows, list) or len(rows) != len(FOUR_CONDITIONS):
        raise ValueError(f"Checkpoint for {key!r} is not a complete four-condition unit")
    conditions = [row.get("condition") for row in rows if isinstance(row, Mapping)]
    if len(conditions) != len(rows) or set(conditions) != set(FOUR_CONDITIONS):
        raise ValueError(f"Checkpoint for {key!r} has invalid conditions")
    row_keys = {
        (row.get("question_id"), row.get("label_country"), row.get("evidence_country"))
        for row in rows
    }
    if row_keys != {key}:
        raise ValueError(f"Checkpoint rows do not preserve directed key {key!r}")
    model_identity = signature["model"]
    expected_unit_id = f"{key[0]}::{key[1]}=>{key[2]}"
    expected_target_id = f"{key[0]}::{key[1]}"
    validated_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        context = f"checkpoint {key!r} row {index}"
        _require_saved(isinstance(raw_row, Mapping), f"{context} must be an object")
        row = dict(raw_row)
        _require_saved(row.get("schema_version") == SCHEMA_VERSION, f"{context}: wrong schema")
        _require_saved(row.get("model_name") == model_identity["name"], f"{context}: wrong model")
        _require_saved(row.get("hub_id") == model_identity["hub_id"], f"{context}: wrong Hub ID")
        _require_saved(
            row.get("model_revision") == model_identity["revision"],
            f"{context}: wrong model revision",
        )
        _require_saved(
            row.get("tokenizer_revision") == model_identity["tokenizer_revision"],
            f"{context}: wrong tokenizer revision",
        )
        _require_saved(row.get("unit_id") == expected_unit_id, f"{context}: wrong unit_id")
        _require_saved(
            row.get("target_unit_id") == expected_target_id,
            f"{context}: wrong target_unit_id",
        )
        _require_saved(
            (row.get("question_id"), row.get("label_country"), row.get("evidence_country"))
            == key,
            f"{context}: wrong complete directed key",
        )
        _require_saved(
            row.get("country") == key[1] and row.get("conflict_country") == key[2],
            f"{context}: incompatible directed-key aliases",
        )
        _require_saved(
            row.get("condition") in FOUR_CONDITIONS,
            f"{context}: unexpected condition",
        )
        canonical = row.get("original_answer_options")
        displayed = row.get("displayed_options")
        _require_saved(
            isinstance(canonical, list)
            and len(canonical) >= 2
            and all(isinstance(option, str) and option for option in canonical)
            and len(canonical) == len(set(canonical)),
            f"{context}: invalid original options",
        )
        _require_saved(
            row.get("original_options") == canonical,
            f"{context}: original option aliases disagree",
        )
        _require_saved(
            isinstance(displayed, list)
            and len(displayed) == len(canonical)
            and set(displayed) == set(canonical),
            f"{context}: displayed options are not an exact permutation",
        )
        _require_saved(row.get("used_options") == displayed, f"{context}: displayed aliases disagree")
        labels = option_labels(len(displayed))
        label_map = dict(zip(labels, displayed))
        _require_saved(row.get("displayed_option_labels") == labels, f"{context}: wrong labels")
        _require_saved(row.get("displayed_label_to_option") == label_map, f"{context}: wrong label map")
        _require_saved(row.get("option_label_map") == label_map, f"{context}: wrong option map")
        _require_saved(
            isinstance(row.get("raw_prompt"), str) and row["raw_prompt"],
            f"{context}: raw prompt missing",
        )
        _require_saved(row.get("prompt") == row["raw_prompt"], f"{context}: prompt aliases differ")
        _require_saved(
            isinstance(row.get("full_chat_templated_prompt"), str)
            and row["full_chat_templated_prompt"],
            f"{context}: rendered chat prompt missing",
        )
        scoring = row.get("scoring")
        _require_saved(isinstance(scoring, Mapping), f"{context}: scoring trace missing")
        _require_saved(
            scoring.get("backend") == "huggingface" and scoring.get("synthetic") is False,
            f"{context}: checkpoint is not genuine Hugging Face scoring",
        )
        _require_saved(scoring.get("label_to_option") == label_map, f"{context}: scoring map is wrong")
        label_probabilities = _validated_probability_mapping(
            row.get("normalized_label_probabilities"), labels, f"{context}.label probabilities"
        )
        _require_saved(
            _same_distribution(label_probabilities, scoring.get("label_probabilities", {})),
            f"{context}: label probability aliases disagree",
        )
        log_scores = row.get("raw_label_log_probabilities")
        _require_saved(
            isinstance(log_scores, Mapping)
            and set(log_scores) == set(labels)
            and all(
                isinstance(log_scores[label], (int, float))
                and not isinstance(log_scores[label], bool)
                and math.isfinite(float(log_scores[label]))
                for label in labels
            ),
            f"{context}: invalid raw label log probabilities",
        )
        _require_saved(
            set(scoring.get("label_log_scores", {})) == set(labels)
            and all(
                math.isclose(
                    float(log_scores[label]),
                    float(scoring["label_log_scores"][label]),
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for label in labels
            ),
            f"{context}: label log-score aliases disagree",
        )
        normalized_from_logs = normalize_log_scores(
            [float(log_scores[label]) for label in labels], temperature=0.0
        )
        _require_saved(
            all(
                math.isclose(
                    label_probabilities[label], probability, rel_tol=0.0, abs_tol=1e-8
                )
                for label, probability in zip(labels, normalized_from_logs)
            ),
            f"{context}: label logs do not normalize to saved probabilities",
        )
        semantic = _validated_probability_mapping(
            row.get("normalized_option_probabilities"), canonical, f"{context}.prediction"
        )
        _require_saved(
            _same_distribution(semantic, row.get("prediction", {})),
            f"{context}: prediction aliases disagree",
        )
        displayed_distribution = map_label_probabilities(
            labels, displayed, [label_probabilities[label] for label in labels]
        )
        recovered = recover_canonical_distribution(displayed_distribution, canonical)
        _require_saved(
            _same_distribution(recovered, semantic),
            f"{context}: semantic option permutation does not round-trip",
        )
        for field in (
            "label_country_human_distribution",
            "evidence_country_human_distribution",
            "human_distribution",
            "presented_evidence_distribution",
            "evidence_distribution",
            "source_evidence_distribution",
        ):
            _validated_probability_mapping(row.get(field), canonical, f"{context}.{field}")
        _require_saved(
            _same_distribution(
                row["label_country_human_distribution"], row["human_distribution"]
            ),
            f"{context}: label-country human distribution aliases disagree",
        )
        _require_saved(
            _same_distribution(
                row["presented_evidence_distribution"], row["evidence_distribution"]
            ),
            f"{context}: presented evidence aliases disagree",
        )
        expected_source = (
            row["evidence_country_human_distribution"]
            if row["condition"] == "conflict"
            else row["label_country_human_distribution"]
        )
        _require_saved(
            _same_distribution(row["source_evidence_distribution"], expected_source),
            f"{context}: source evidence does not match its condition",
        )
        _require_saved(
            row.get("evidence_presented")
            is (row["condition"] in {"population_evidence", "conflict"}),
            f"{context}: evidence-presented flag is wrong",
        )
        _require_saved(
            scoring.get("rendered_prompt") == row["full_chat_templated_prompt"],
            f"{context}: rendered prompt aliases disagree",
        )
        _require_saved(
            scoring.get("raw_prompt_sha256")
            == hashlib.sha256(row["raw_prompt"].encode("utf-8")).hexdigest(),
            f"{context}: raw prompt hash mismatch",
        )
        _require_saved(
            scoring.get("rendered_prompt_sha256")
            == hashlib.sha256(row["full_chat_templated_prompt"].encode("utf-8")).hexdigest(),
            f"{context}: rendered prompt hash mismatch",
        )
        prompt_count = row.get("prompt_token_count")
        _require_saved(
            isinstance(prompt_count, int) and not isinstance(prompt_count, bool) and prompt_count > 0,
            f"{context}: invalid prompt token count",
        )
        _require_saved(
            row.get("prompt_prefix_verified") is True
            and scoring.get("prompt_prefix_verified") is True,
            f"{context}: prompt prefix was not verified",
        )
        for field in (
            "candidate_label_continuations",
            "candidate_label_token_ids",
            "label_target_token_positions",
            "label_predictive_logit_positions",
        ):
            _require_saved(
                isinstance(row.get(field), Mapping) and set(row[field]) == set(labels),
                f"{context}: {field} does not cover all labels",
            )
            _require_saved(
                row[field] == scoring.get(field),
                f"{context}: {field} aliases disagree",
            )
        seen_ids: set[tuple[int, ...]] = set()
        for label in labels:
            continuation = row["candidate_label_continuations"][label]
            token_ids = row["candidate_label_token_ids"][label]
            _require_saved(
                isinstance(continuation, str) and continuation,
                f"{context}: empty label continuation",
            )
            _require_saved(
                isinstance(token_ids, list)
                and token_ids
                and all(isinstance(token_id, int) and token_id >= 0 for token_id in token_ids),
                f"{context}: invalid candidate label token IDs",
            )
            id_tuple = tuple(token_ids)
            _require_saved(id_tuple not in seen_ids, f"{context}: duplicate label token sequence")
            seen_ids.add(id_tuple)
            targets = list(range(prompt_count, prompt_count + len(token_ids)))
            predictive = [position - 1 for position in targets]
            _require_saved(
                row["label_target_token_positions"][label] == targets,
                f"{context}: wrong target token positions",
            )
            _require_saved(
                row["label_predictive_logit_positions"][label] == predictive
                and predictive[0] == prompt_count - 1,
                f"{context}: wrong predictive answer positions",
            )
        _require_saved(row.get("quantization") == signature["quantization"], f"{context}: quantization mismatch")
        _require_saved(
            row["quantization"].get("load_in_4bit") is True,
            f"{context}: result was not produced by 4-bit inference",
        )
        _require_saved(row.get("jensen_shannon_base") == 2, f"{context}: JSD base is not 2")
        distances = row.get("base2_jensen_shannon_distances")
        _require_saved(
            isinstance(distances, Mapping)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in distances.values()
            ),
            f"{context}: invalid base-2 JSD distances",
        )
        for metric in ("country_influence", "evidence_influence", "evidence_override"):
            value = row.get(metric)
            _require_saved(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"{context}: invalid {metric}",
            )
        validated_rows.append(row)

    by_condition = {row["condition"]: row for row in validated_rows}
    reference = by_condition["baseline"]
    for row in validated_rows:
        for field in (
            "original_answer_options",
            "displayed_options",
            "displayed_option_labels",
            "displayed_label_to_option",
            "label_country_human_distribution",
            "evidence_country_human_distribution",
        ):
            _require_saved(row[field] == reference[field], f"{key!r}: {field} varies by condition")
    label_human = reference["label_country_human_distribution"]
    evidence_human = reference["evidence_country_human_distribution"]
    baseline = by_condition["baseline"]["prediction"]
    country_prediction = by_condition["country_label"]["prediction"]
    population_prediction = by_condition["population_evidence"]["prediction"]
    conflict_prediction = by_condition["conflict"]["prediction"]
    expected_metrics = {
        "country_influence": Metrics.country_influence(
            baseline, country_prediction, label_human
        ),
        "evidence_influence": Metrics.evidence_influence(
            baseline, population_prediction, label_human
        ),
        "evidence_override": Metrics.evidence_override(
            conflict_prediction, evidence_human, label_human
        ),
    }
    component_distances = {
        "baseline_to_label_country": Metrics.js_divergence(baseline, label_human),
        "country_label_to_label_country": Metrics.js_divergence(
            country_prediction, label_human
        ),
        "population_evidence_to_label_country": Metrics.js_divergence(
            population_prediction, label_human
        ),
        "conflict_to_label_country": Metrics.js_divergence(
            conflict_prediction, label_human
        ),
        "conflict_to_evidence_country": Metrics.js_divergence(
            conflict_prediction, evidence_human
        ),
    }
    for row in validated_rows:
        for metric, expected in expected_metrics.items():
            _require_saved(
                math.isclose(float(row[metric]), expected, rel_tol=0.0, abs_tol=1e-8),
                f"{key!r}: saved {metric} is inconsistent",
            )
        expected_distances = {
            "prediction_to_label_country": Metrics.js_divergence(
                row["prediction"], label_human
            ),
            "prediction_to_evidence_country": Metrics.js_divergence(
                row["prediction"], evidence_human
            ),
            **component_distances,
        }
        _require_saved(
            set(row["base2_jensen_shannon_distances"]) == set(expected_distances)
            and all(
                math.isclose(
                    float(row["base2_jensen_shannon_distances"][name]),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for name, value in expected_distances.items()
            ),
            f"{key!r}: saved base-2 JSD components are inconsistent",
        )
    return validated_rows


def _prepare_unit(
    loader: DataLoader,
    pair: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
) -> dict[str, Any]:
    question_id, label_country, evidence_country = _unit_key(pair)
    canonical_options = list(pair["options"])
    target_unit_id = f"{question_id}::{label_country}"
    displayed_options = (
        shuffle_options(canonical_options, stable_unit_seed(seed, target_unit_id))
        if shuffle
        else list(canonical_options)
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


def run_directed_unit(
    runner: ModelRunner,
    spec: Mapping[str, Any],
    loader: DataLoader,
    pair: Mapping[str, Any],
    quantization: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    """Run all four conditions and return four enriched prediction records."""

    unit = _prepare_unit(loader, pair, seed=seed, shuffle=shuffle)
    question_id, label_country, evidence_country = unit["key"]
    builder = PromptBuilder()
    rows: list[dict[str, Any]] = []
    for condition in FOUR_CONDITIONS:
        raw_prompt = build_prompt(
            builder,
            condition,
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
            displayed_prediction, unit["canonical_options"]
        )
        scoring = runner.get_last_scoring_metadata()
        labels = option_labels(len(unit["displayed_options"]))
        label_probabilities = {
            label: float(scoring["label_probabilities"][label]) for label in labels
        }
        log_probabilities = {
            label: float(scoring["label_log_scores"][label]) for label in labels
        }
        candidate_ids = {
            label: [int(token_id) for token_id in scoring["candidate_label_token_ids"][label]]
            for label in labels
        }
        source_evidence = (
            unit["evidence_human"] if condition == "conflict" else unit["label_human"]
        )
        presented_displayed = (
            unit["conflict_evidence_displayed"]
            if condition == "conflict"
            else unit["label_evidence_displayed"]
        )
        presented_canonical = recover_canonical_distribution(
            presented_displayed, unit["canonical_options"]
        )
        row = {
            "schema_version": SCHEMA_VERSION,
            "model_name": spec["name"],
            "hub_id": spec["hub_id"],
            "model_revision": spec["revision"],
            "tokenizer_revision": spec["tokenizer_revision"],
            "unit_id": unit["unit_id"],
            "target_unit_id": unit["target_unit_id"],
            "question_id": question_id,
            "label_country": label_country,
            "evidence_country": evidence_country,
            # Compatibility aliases used by the repaired repository validators.
            "country": label_country,
            "conflict_country": evidence_country,
            "country_display": unit["label_country_display"],
            "conflict_country_display": unit["evidence_country_display"],
            "condition": condition,
            "original_answer_options": list(unit["canonical_options"]),
            "original_options": list(unit["canonical_options"]),
            "displayed_options": list(unit["displayed_options"]),
            "used_options": list(unit["displayed_options"]),
            "displayed_option_labels": labels,
            "displayed_label_to_option": dict(scoring["label_to_option"]),
            "option_label_map": dict(scoring["label_to_option"]),
            "raw_prompt": raw_prompt,
            "prompt": raw_prompt,
            "full_chat_templated_prompt": scoring["rendered_prompt"],
            "candidate_label_continuations": dict(
                scoring["candidate_label_continuations"]
            ),
            "candidate_label_token_ids": candidate_ids,
            "prompt_token_count": int(scoring["prompt_token_count"]),
            "label_target_token_positions": dict(
                scoring["label_target_token_positions"]
            ),
            "label_predictive_logit_positions": dict(
                scoring["label_predictive_logit_positions"]
            ),
            "prompt_prefix_verified": bool(scoring["prompt_prefix_verified"]),
            "raw_label_log_probabilities": log_probabilities,
            "normalized_label_probabilities": label_probabilities,
            "normalized_option_probabilities": prediction,
            "prediction": prediction,
            "label_country_human_distribution": unit["label_human"],
            "evidence_country_human_distribution": unit["evidence_human"],
            "human_distribution": unit["label_human"],
            "presented_evidence_distribution": presented_canonical,
            "evidence_distribution": presented_canonical,
            "source_evidence_distribution": source_evidence,
            "evidence_presented": condition in {"population_evidence", "conflict"},
            "shuffled": bool(shuffle),
            "quantization": dict(quantization),
            "scoring": scoring,
        }
        rows.append(row)

    by_condition = {row["condition"]: row for row in rows}
    baseline = by_condition["baseline"]["prediction"]
    country_prediction = by_condition["country_label"]["prediction"]
    population_prediction = by_condition["population_evidence"]["prediction"]
    conflict_prediction = by_condition["conflict"]["prediction"]
    label_human = unit["label_human"]
    evidence_human = unit["evidence_human"]
    components = {
        "baseline_to_label_country": Metrics.js_divergence(baseline, label_human),
        "country_label_to_label_country": Metrics.js_divergence(
            country_prediction, label_human
        ),
        "population_evidence_to_label_country": Metrics.js_divergence(
            population_prediction, label_human
        ),
        "conflict_to_label_country": Metrics.js_divergence(
            conflict_prediction, label_human
        ),
        "conflict_to_evidence_country": Metrics.js_divergence(
            conflict_prediction, evidence_human
        ),
    }
    unit_metrics = {
        "country_influence": Metrics.country_influence(
            baseline, country_prediction, label_human
        ),
        "evidence_influence": Metrics.evidence_influence(
            baseline, population_prediction, label_human
        ),
        "evidence_override": Metrics.evidence_override(
            conflict_prediction, evidence_human, label_human
        ),
    }
    for row in rows:
        distances = {
            "prediction_to_label_country": Metrics.js_divergence(
                row["prediction"], label_human
            ),
            "prediction_to_evidence_country": Metrics.js_divergence(
                row["prediction"], evidence_human
            ),
            **components,
        }
        row["jensen_shannon_base"] = 2
        row["base2_jensen_shannon_distances"] = distances
        row.update(unit_metrics)
    return rows


def run_model_units(
    runner: ModelRunner,
    spec: Mapping[str, Any],
    loader: DataLoader,
    pairs: Sequence[Mapping[str, Any]],
    repository_root: Path,
    output_directory: Path,
    pair_manifest: str,
    quantization: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    shuffle: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run or resume one model, atomically checkpointing each complete unit."""

    validate_reciprocal_pairs(pairs)
    current_runtime = runner.get_runtime_metadata()
    validate_loaded_runtime(current_runtime, spec, quantization)
    signature = build_run_signature(
        repository_root,
        spec,
        pairs,
        pair_manifest,
        quantization,
        current_runtime,
        seed=seed,
        shuffle=shuffle,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_directory / "checkpoints"
    combined_path = output_directory / f"results_{spec['slug']}.jsonl"
    signature_path = output_directory / f"run_signature_{spec['slug']}.json"
    runtime_path = output_directory / f"model_runtime_{spec['slug']}.json"
    if signature_path.is_file():
        previous_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous_signature.get("fingerprint") != signature["fingerprint"]:
            raise ValueError(
                "Existing run signature differs from the current code, data, model, "
                "dependencies, quantization, or runtime; use a new output directory"
            )
    else:
        atomic_write_json(signature_path, signature)
    if runtime_path.is_file():
        previous_runtime_record = json.loads(runtime_path.read_text(encoding="utf-8"))
        previous_runtime = previous_runtime_record.get("runtime", {})
        if runtime_signature_identity(previous_runtime) != runtime_signature_identity(
            current_runtime
        ):
            raise ValueError("Existing model runtime identity differs from this resume")
    else:
        atomic_write_json(
            runtime_path,
            {
                "recorded_at_utc": utc_now(),
                "model": dict(spec),
                "quantization": dict(quantization),
                "runtime": current_runtime,
            },
        )

    completed: MutableMapping[tuple[str, str, str], list[dict[str, Any]]] = {}
    resumed_units = 0
    newly_completed_units = 0
    for index, pair in enumerate(pairs, start=1):
        key = _unit_key(pair)
        checkpoint_path = _checkpoint_path(checkpoint_dir, spec, key)
        if checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            rows = _validate_checkpoint(checkpoint, signature, key)
            resumed_units += 1
        else:
            rows = run_directed_unit(
                runner,
                spec,
                loader,
                pair,
                quantization,
                seed=seed,
                shuffle=shuffle,
            )
            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "completed_at_utc": utc_now(),
                "run_fingerprint": signature["fingerprint"],
                "directed_key": list(key),
                "conditions": list(FOUR_CONDITIONS),
                "rows": rows,
            }
            rows = _validate_checkpoint(checkpoint, signature, key)
            atomic_write_json(checkpoint_path, checkpoint)
            newly_completed_units += 1
        completed[key] = rows
        ordered_rows = [
            row
            for prior_pair in pairs[:index]
            for row in completed[_unit_key(prior_pair)]
        ]
        # Rebuild atomically after every directed unit; JSONL is a convenient
        # inspection/export view while per-unit JSON files are the resume source.
        atomic_write_jsonl(combined_path, ordered_rows)

    rows = [row for pair in pairs for row in completed[_unit_key(pair)]]
    if len(rows) != len(pairs) * len(FOUR_CONDITIONS):
        raise RuntimeError("Completed result cardinality is inconsistent")
    summary = {
        "model_name": spec["name"],
        "hub_id": spec["hub_id"],
        "model_revision": spec["revision"],
        "tokenizer_revision": spec["tokenizer_revision"],
        "quantization": dict(quantization),
        "runtime_identity_sha256": canonical_json_hash(
            runtime_signature_identity(current_runtime)
        ),
        "validated_at_utc": utc_now(),
        "run_fingerprint": signature["fingerprint"],
        "directed_units": len(pairs),
        "result_rows": len(rows),
        "resumed_units": resumed_units,
        "newly_completed_units": newly_completed_units,
        "combined_results": str(combined_path),
        "validation": {
            "status": "PASS",
            "directed_key_fields": [
                "question_id",
                "label_country",
                "evidence_country",
            ],
            "unique_directed_units": len(completed),
            "reciprocal_directions_complete": True,
            "conditions_per_directed_unit": list(FOUR_CONDITIONS),
            "expected_rows_per_directed_unit": len(FOUR_CONDITIONS),
            "row_schema_and_prompt_hashes_valid": True,
            "probabilities_and_label_permutations_valid": True,
            "base2_metrics_recomputed_and_valid": True,
            "run_fingerprint_valid": True,
            "runtime_identity_valid": True,
            "four_bit_quantization_valid": True,
        },
    }
    atomic_write_json(output_directory / f"completion_{spec['slug']}.json", summary)
    return rows, summary


def _close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _independent_label_scores(
    runner: ModelRunner,
    row: Mapping[str, Any],
) -> dict[str, float]:
    """Independently recompute one row's label scores at causal answer positions."""

    import torch

    prompt = str(row["full_chat_templated_prompt"])
    prompt_ids = runner.tokenizer.encode(prompt, add_special_tokens=False)
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for label in row["displayed_option_labels"]:
            continuation = row["candidate_label_continuations"][label]
            combined = runner.tokenizer.encode(
                prompt + continuation, add_special_tokens=False
            )
            if combined[: len(prompt_ids)] != prompt_ids:
                raise AssertionError(f"{label}: prompt is not a token prefix")
            candidate = combined[len(prompt_ids) :]
            input_ids = torch.tensor([combined], dtype=torch.long, device=runner.device)
            output = runner.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
            value = 0.0
            for offset, token_id in enumerate(candidate):
                predictive_position = len(prompt_ids) + offset - 1
                log_probs = torch.log_softmax(
                    output.logits[0, predictive_position, :].float(), dim=-1
                )
                value += float(log_probs[int(token_id)].item())
            scores[label] = value
    return scores


def assert_genuine_smoke(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    runner: ModelRunner,
    spec: Mapping[str, Any],
    quantization: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply model-aware assertions to the two-unit genuine Qwen smoke."""

    if spec["hub_id"] != "Qwen/Qwen2.5-7B-Instruct":
        raise AssertionError("Genuine smoke must use Qwen/Qwen2.5-7B-Instruct")
    validate_reciprocal_pairs(pairs, expected_count=2)
    expected_keys = {_unit_key(pair) for pair in pairs}
    if len(rows) != 8:
        raise AssertionError(f"Expected 8 smoke predictions, found {len(rows)}")
    row_keys = {
        (
            row["question_id"],
            row["label_country"],
            row["evidence_country"],
            row["condition"],
        )
        for row in rows
    }
    if len(row_keys) != 8:
        raise AssertionError("Smoke predictions do not have unique directed-condition keys")
    if {(key[0], key[1], key[2]) for key in row_keys} != expected_keys:
        raise AssertionError("Smoke output directed units differ from the manifest")

    runtime = runner.get_runtime_metadata()
    if runtime.get("backend") != "huggingface" or runner.is_synthetic:
        raise AssertionError("Synthetic or non-Hugging-Face backend detected")
    if runtime.get("resolved_model_revision") != spec["revision"]:
        raise AssertionError("Resolved model revision does not match pinned Qwen commit")
    if runtime.get("resolved_tokenizer_revision") != spec["tokenizer_revision"]:
        raise AssertionError("Resolved tokenizer revision does not match pinned Qwen commit")
    if not runtime.get("chat_template_sha256") or not runtime.get("use_chat_template"):
        raise AssertionError("The pinned tokenizer chat template was not used")
    if not runtime.get("is_loaded_in_4bit") or quantization.get("load_in_4bit") is not True:
        raise AssertionError("The genuine smoke model is not loaded in 4-bit mode")

    collision_checks = 0
    permutation_checks = 0
    position_checks = 0
    for row in rows:
        if not str(row["raw_prompt"]).rstrip().endswith("Answer:"):
            raise AssertionError("Raw prompt does not end at the requested answer position")
        if not row["prompt_prefix_verified"]:
            raise AssertionError("Production scorer did not verify the prompt token prefix")
        labels = list(row["displayed_option_labels"])
        if labels != option_labels(len(row["displayed_options"])):
            raise AssertionError("Displayed labels are not canonical A/B/... labels")
        rendered = str(row["full_chat_templated_prompt"])
        prompt_ids = runner.tokenizer.encode(rendered, add_special_tokens=False)
        if len(prompt_ids) != row["prompt_token_count"]:
            raise AssertionError("Stored prompt token count is inconsistent")
        seen_token_sequences: set[tuple[int, ...]] = set()
        for label in labels:
            continuation = row["candidate_label_continuations"][label]
            combined = runner.tokenizer.encode(
                rendered + continuation, add_special_tokens=False
            )
            if combined[: len(prompt_ids)] != prompt_ids:
                raise AssertionError(f"{label}: answer continuation retokenized the prompt")
            token_ids = combined[len(prompt_ids) :]
            if token_ids != row["candidate_label_token_ids"][label] or not token_ids:
                raise AssertionError(f"{label}: stored candidate label tokenization is invalid")
            if not all(
                isinstance(token_id, int) and 0 <= token_id < len(runner.tokenizer)
                for token_id in token_ids
            ):
                raise AssertionError(f"{label}: candidate includes an invalid tokenizer ID")
            token_sequence = tuple(token_ids)
            if token_sequence in seen_token_sequences:
                raise AssertionError("Two displayed labels have identical token sequences")
            seen_token_sequences.add(token_sequence)
            targets = list(
                range(len(prompt_ids), len(prompt_ids) + len(token_ids))
            )
            predictive = [position - 1 for position in targets]
            if row["label_target_token_positions"][label] != targets:
                raise AssertionError("Stored label target positions are incorrect")
            if row["label_predictive_logit_positions"][label] != predictive:
                raise AssertionError("Stored predictive-logit positions are incorrect")
            if predictive[0] != len(prompt_ids) - 1:
                raise AssertionError("First label token is not scored at the answer position")
            position_checks += 1

        label_probabilities = row["normalized_label_probabilities"]
        semantic_probabilities = row["normalized_option_probabilities"]
        for distribution in (label_probabilities, semantic_probabilities):
            values = [float(value) for value in distribution.values()]
            if not values or not all(math.isfinite(value) and value >= 0 for value in values):
                raise AssertionError("Probability distribution is not finite/non-negative")
            if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise AssertionError("Probability distribution does not sum to one")

        displayed = map_label_probabilities(
            labels,
            row["displayed_options"],
            [label_probabilities[label] for label in labels],
        )
        recovered = recover_canonical_distribution(
            displayed, row["original_answer_options"]
        )
        if any(
            not _close(recovered[option], semantic_probabilities[option], 1e-8)
            for option in recovered
        ):
            raise AssertionError("Option permutation does not recover semantic probabilities")
        permutation_checks += 1

        first_words: defaultdict[str, list[str]] = defaultdict(list)
        for option in row["original_answer_options"]:
            first_words[str(option).split()[0]].append(option)
        collisions = [group for group in first_words.values() if len(group) > 1]
        if not collisions:
            raise AssertionError("Smoke item does not exercise a shared first-answer-word collision")
        for group in collisions:
            for left_index in range(len(group)):
                for right_index in range(left_index + 1, len(group)):
                    left = float(semantic_probabilities[group[left_index]])
                    right = float(semantic_probabilities[group[right_index]])
                    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                        raise AssertionError(
                            "Distinct semantic options sharing a first word received equal values"
                        )
                    collision_checks += 1

        if row.get("jensen_shannon_base") != 2:
            raise AssertionError("Smoke result does not declare base-2 Jensen-Shannon distance")
        if not all(
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
            for value in row["base2_jensen_shannon_distances"].values()
        ):
            raise AssertionError("A base-2 Jensen-Shannon distance is invalid")

    # Independently recompute one full candidate set directly from logits.
    independent = _independent_label_scores(runner, rows[0])
    score_differences = {
        label: abs(independent[label] - rows[0]["raw_label_log_probabilities"][label])
        for label in independent
    }
    if any(difference > 1e-5 for difference in score_differences.values()):
        raise AssertionError("Independent answer-position scoring did not match stored scores")

    evidence_fixture = {"A": 0.9, "B": 0.1}
    label_fixture = {"A": 0.1, "B": 0.9}
    positive_override = Metrics.evidence_override(
        evidence_fixture, evidence_fixture, label_fixture
    )
    negative_override = Metrics.evidence_override(
        label_fixture, evidence_fixture, label_fixture
    )
    if not positive_override > 0 or not negative_override < 0:
        raise AssertionError("Evidence Override sign convention is incorrect")

    return {
        "status": "PASS",
        "timestamp_utc": utc_now(),
        "backend": runtime["backend"],
        "hub_id": spec["hub_id"],
        "resolved_model_revision": runtime["resolved_model_revision"],
        "resolved_tokenizer_revision": runtime["resolved_tokenizer_revision"],
        "is_loaded_in_4bit": runtime["is_loaded_in_4bit"],
        "directed_units": 2,
        "predictions": 8,
        "position_checks": position_checks,
        "permutation_checks": permutation_checks,
        "shared_first_word_collision_checks": collision_checks,
        "independent_score_absolute_differences": score_differences,
        "evidence_override_fixture_positive": positive_override,
        "evidence_override_fixture_negative": negative_override,
        "assertions": {
            "answer_position": True,
            "label_tokenization": True,
            "no_first_word_aliasing": True,
            "probability_normalization": True,
            "option_permutation_recovery": True,
            "directed_unit_uniqueness_and_reciprocity": True,
            "evidence_override_sign": True,
            "genuine_huggingface_backend": True,
            "four_bit_quantization": True,
        },
    }


def genuine_qwen_smoke(
    repository_root: Path,
    output_root: Path,
    token: Optional[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run/resume the genuine 2-unit x 4-condition Qwen GPU smoke."""

    spec = MODEL_SPECS[0]
    access = check_model_access([spec], token)
    if not access[spec["hub_id"]]["accessible"]:
        raise PermissionError(
            f"Exact smoke checkpoint is inaccessible: {access[spec['hub_id']]}"
        )
    loader, pairs = load_data_and_pairs(
        repository_root,
        "data/pairs/country_pairs_smoke.json",
        expected_count=2,
    )
    runner: Optional[ModelRunner] = None
    try:
        runner, quantization = build_quantized_runner(spec, token)
        runner.load_model()
        rows, completion = run_model_units(
            runner,
            spec,
            loader,
            pairs,
            repository_root,
            output_root / "smoke",
            "data/pairs/country_pairs_smoke.json",
            quantization,
        )
        assertions = assert_genuine_smoke(rows, pairs, runner, spec, quantization)
        atomic_write_json(output_root / "smoke/smoke_assertions.json", assertions)
        return rows, completion, assertions
    finally:
        free_runner(runner)


def require_full_approval(enabled: bool, phrase: str) -> None:
    if enabled is not True or phrase != FULL_APPROVAL_PHRASE:
        raise PermissionError(
            "Full run is disabled. It requires explicit approval after inspection of "
            "the genuine smoke, FULL_RUN_APPROVED=True, and the exact approval phrase."
        )


def run_full_experiment(
    repository_root: Path,
    output_root: Path,
    token: Optional[str],
    *,
    full_run_approved: bool,
    approval_phrase: str,
) -> list[dict[str, Any]]:
    """Run/resume all four models; shipped notebook calls this only behind a gate."""

    require_full_approval(full_run_approved, approval_phrase)
    smoke_assertions_path = output_root / "smoke/smoke_assertions.json"
    if not smoke_assertions_path.is_file():
        raise PermissionError("Full run requires an inspected genuine-smoke assertion report")
    smoke_assertions = json.loads(smoke_assertions_path.read_text(encoding="utf-8"))
    if (
        smoke_assertions.get("status") != "PASS"
        or smoke_assertions.get("backend") != "huggingface"
        or smoke_assertions.get("hub_id") != "Qwen/Qwen2.5-7B-Instruct"
        or smoke_assertions.get("is_loaded_in_4bit") is not True
    ):
        raise PermissionError("Genuine Qwen smoke assertions are absent or did not pass")
    loader, pairs = load_data_and_pairs(
        repository_root,
        "data/pairs/country_pairs_v2.json",
        expected_count=200,
    )
    access = check_model_access(MODEL_SPECS, token)
    atomic_write_json(output_root / "model_access.json", access)
    inaccessible = [hub_id for hub_id, item in access.items() if not item["accessible"]]
    if inaccessible:
        raise PermissionError(
            "Full run aborted before loading any weights; inaccessible exact checkpoints: "
            + ", ".join(inaccessible)
        )

    all_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        runner: Optional[ModelRunner] = None
        try:
            runner, quantization = build_quantized_runner(spec, token)
            runner.load_model()
            rows, _ = run_model_units(
                runner,
                spec,
                loader,
                pairs,
                repository_root,
                output_root / "full",
                "data/pairs/country_pairs_v2.json",
                quantization,
            )
            all_rows.extend(rows)
        finally:
            free_runner(runner)
    expected = len(MODEL_SPECS) * 200 * len(FOUR_CONDITIONS)
    if len(all_rows) != expected:
        raise RuntimeError(f"Expected {expected} full-run rows, found {len(all_rows)}")
    return all_rows


def load_saved_runtime_metadata(
    output_root: Path,
    specs: Sequence[Mapping[str, Any]],
    *,
    run_kind: str,
) -> dict[str, dict[str, Any]]:
    runtimes: dict[str, dict[str, Any]] = {}
    for spec in specs:
        path = output_root / run_kind / f"model_runtime_{spec['slug']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model runtime record: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        runtime = value.get("runtime") if isinstance(value, Mapping) else None
        if not isinstance(runtime, Mapping):
            raise ValueError(f"Malformed model runtime record: {path}")
        runtimes[str(spec["name"])] = dict(runtime)
    return runtimes


def build_repaired_payload(
    repository_root: Path,
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: str,
    model_runtime: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Wrap Colab JSONL rows in the strict repaired-result payload schema."""

    if not rows:
        raise ValueError("Cannot build a result payload from zero rows")
    models = sorted({str(row["model_name"]) for row in rows})
    if set(model_runtime) != set(models):
        raise ValueError("Runtime metadata does not exactly cover result models")
    directed = {
        (str(row["question_id"]), str(row["country"]), str(row["conflict_country"]))
        for row in rows
    }
    conditions = {str(row["condition"]) for row in rows}
    if conditions != set(FOUR_CONDITIONS):
        raise ValueError("Colab rows do not contain exactly the four core conditions")

    def identity(relative_path: str) -> dict[str, Any]:
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        return {
            "path": relative_path,
            "sha256": sha256_file(path),
            "status": "present",
        }

    payload = {
        "metadata": {
            "mode": "colab_gpu",
            "timestamp_utc": utc_now(),
            "config": {
                "inference": {"backend": "huggingface", "temperature": 0.0},
                "conditions": list(FOUR_CONDITIONS),
                "shuffle_options": True,
                "seed": DEFAULT_SEED,
            },
            "num_pairs": len(directed),
            "directed_unit_key": ["question_id", "country", "conflict_country"],
            "models": models,
            "conditions": list(FOUR_CONDITIONS),
            "shuffle_options": True,
            "jensen_shannon_base": 2,
            "scoring_method": SCORING_METHOD,
            "prompt_template": "invariant_core_v2_with_optional_cue_blocks",
            "synthetic_backend": False,
            "scientific_inference": True,
            "package_lock": identity("requirements-colab.lock"),
            "data_artifacts": {
                "dataset": identity("data/processed/dataset_v2.json"),
                "pair_manifest": identity(pair_manifest),
                "source_dataset": identity("data/processed/dataset_v1.json"),
                "repair_summary": identity("data/audit/data_repair_summary.json"),
            },
            "model_runtime": {
                model: dict(model_runtime[model]) for model in models
            },
        },
        "results": [dict(row) for row in rows],
    }
    return payload


def load_full_result_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        path = output_root / "full" / f"results_{spec['slug']}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete model output: {path}")
        model_rows = read_jsonl(path)
        if len(model_rows) != 200 * len(FOUR_CONDITIONS):
            raise ValueError(f"{path} is incomplete: found {len(model_rows)} rows")
        rows.extend(model_rows)
    return rows


__all__ = [
    "DEFAULT_SEED",
    "FULL_APPROVAL_PHRASE",
    "MODEL_SPECS",
    "SCHEMA_VERSION",
    "assert_genuine_smoke",
    "atomic_write_json",
    "check_model_access",
    "build_repaired_payload",
    "collect_environment",
    "free_runner",
    "genuine_qwen_smoke",
    "get_hf_token",
    "load_full_result_rows",
    "load_saved_runtime_metadata",
    "read_jsonl",
    "require_full_approval",
    "run_full_experiment",
    "run_model_units",
    "validate_reciprocal_pairs",
]

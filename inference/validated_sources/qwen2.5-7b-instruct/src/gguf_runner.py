"""Isolated llama.cpp/GGUF backend for the bounded CPU smoke test.

This backend deliberately does not replace the Transformers/Colab runner.  It
uses a tiny C++ helper linked directly to llama.cpp's C API so every candidate
label is read from the complete vocabulary logits at the exact answer position.
No answer text is generated or parsed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, IO, Mapping, Sequence

from src.scoring import map_label_probabilities, normalize_log_scores, option_labels


QWEN25_DEFAULT_SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


def serialize_qwen25_chatml(messages: Sequence[Mapping[str, str]]) -> str:
    """Serialize the non-tool Qwen2.5 chat-template path byte-for-byte.

    The official embedded template is conditional: when no system message is
    supplied it materializes Qwen's default system message.  The GGUF backend
    supplies that message explicitly, then checks llama.cpp's serialization
    against this exact expansion so heuristic template detection cannot omit it.
    """

    if not messages:
        raise ValueError("Qwen messages must not be empty")
    pieces = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported Qwen smoke message role: {role!r}")
        if not content:
            raise ValueError("Qwen smoke message content must not be empty")
        pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    pieces.append("<|im_start|>assistant\n")
    return "".join(pieces)


class GGUFRunner:
    """Keep one CPU-only GGUF model loaded while scoring label continuations."""

    backend = "llama.cpp"
    is_synthetic = False

    def __init__(
        self,
        *,
        helper_path: Path,
        model_path: Path,
        model_identifier: str,
        repository: str,
        revision: str,
        quantization: str = "Q4_K_M",
        threads: int = 1,
        context_size: int = 4096,
    ) -> None:
        self.helper_path = Path(helper_path).resolve()
        self.model_path = Path(model_path).resolve()
        self.model_identifier = str(model_identifier)
        self.repository = str(repository)
        self.revision = str(revision)
        self.quantization = str(quantization)
        self.threads = int(threads)
        self.context_size = int(context_size)
        if not self.helper_path.is_file():
            raise FileNotFoundError(f"llama.cpp scoring helper is missing: {self.helper_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF first shard is missing: {self.model_path}")
        if self.threads <= 0 or self.context_size <= 0:
            raise ValueError("threads and context_size must be positive")

        self._process: subprocess.Popen[str] | None = None
        self._log_handle: IO[str] | None = None
        self._request_number = 0
        self.runtime_metadata: Dict[str, Any] = {}
        self.last_scoring: Dict[str, Any] = {}
        self.last_structured_messages: list[dict[str, str]] = []

    def load_model(self, run_log: IO[str]) -> None:
        """Start the low-level helper and fail unless a real llama.cpp model loads."""

        if self._process is not None:
            raise RuntimeError("GGUF model process is already running")
        self._log_handle = run_log
        command = [
            str(self.helper_path),
            "--model",
            str(self.model_path),
            "--threads",
            str(self.threads),
            "--ctx-size",
            str(self.context_size),
        ]
        run_log.write(f"Starting genuine backend: {json.dumps(command)}\n")
        run_log.flush()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=run_log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        ready = self._read_response()
        if ready.get("type") != "ready" or ready.get("backend") != self.backend:
            self.close()
            raise RuntimeError(f"llama.cpp helper did not report a genuine ready state: {ready}")
        if ready.get("chat_template_source") != "embedded_gguf_metadata":
            self.close()
            raise RuntimeError("GGUF chat template fallback is forbidden")
        template = ready.get("chat_template")
        if not isinstance(template, str) or not template:
            self.close()
            raise RuntimeError("loaded GGUF did not expose an embedded chat template")
        self.runtime_metadata = {
            **ready,
            "model_identifier": self.model_identifier,
            "repository": self.repository,
            "revision": self.revision,
            "quantization": self.quantization,
            "model_path": str(self.model_path),
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "synthetic": False,
            "generated_answer_parsing": False,
            "candidate_scoring": "full_vocabulary_low_level_logits",
        }

    def _read_response(self) -> Dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("GGUF helper is not running")
        line = self._process.stdout.readline()
        if not line:
            return_code = self._process.poll()
            raise RuntimeError(f"GGUF helper exited without a response (returncode={return_code})")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GGUF helper returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("GGUF helper response must be an object")
        if response.get("type") == "error":
            raise RuntimeError(f"GGUF helper scoring error: {response.get('message')}")
        return response

    def score_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        labels: Sequence[str],
    ) -> Dict[str, Any]:
        """Apply the embedded template and score complete contextual label sequences."""

        if self._process is None or self._process.stdin is None:
            raise RuntimeError("call load_model() before scoring")
        structured_messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]
        self._request_number += 1
        request_id = str(self._request_number)
        request = {
            "request_id": request_id,
            "messages": structured_messages,
            "labels": [str(label) for label in labels],
        }
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if response.get("type") != "result" or response.get("request_id") != request_id:
            raise RuntimeError("GGUF helper response did not match the scoring request")
        if response.get("generated_answer") is not None:
            raise RuntimeError("GGUF backend unexpectedly generated an answer")
        expected_serialization = serialize_qwen25_chatml(structured_messages)
        if response.get("serialized_prompt") != expected_serialization:
            raise RuntimeError(
                "llama.cpp serialization does not match the embedded Qwen2.5 "
                "template's exact non-tool expansion"
            )
        response["embedded_template_serialization_verified"] = True
        response["serialization_verification"] = (
            "byte_exact_qwen25_non_tool_chatml_expansion"
        )
        self.last_structured_messages = structured_messages
        self.last_scoring = response
        return response

    def predict_distribution(
        self,
        prompt: str,
        options: Sequence[object],
        temperature: float = 0.0,
    ) -> Dict[str, float]:
        labels = option_labels(len(options))
        response = self.score_messages(
            [
                {"role": "system", "content": QWEN25_DEFAULT_SYSTEM_MESSAGE},
                {"role": "user", "content": str(prompt)},
            ],
            labels,
        )
        candidates = response.get("candidates")
        if not isinstance(candidates, dict) or set(candidates) != set(labels):
            raise RuntimeError("GGUF helper did not score every displayed option label")
        scores = []
        for label in labels:
            candidate = candidates[label]
            if not isinstance(candidate, dict):
                raise RuntimeError(f"invalid candidate trace for label {label}")
            score = float(candidate["sequence_log_probability"])
            if not math.isfinite(score):
                raise RuntimeError(f"non-finite candidate score for label {label}")
            token_ids = candidate.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise RuntimeError(f"label {label} has no contextual continuation tokens")
            scores.append(score)
        probabilities = normalize_log_scores(scores, temperature=temperature)
        distribution = map_label_probabilities(labels, options, probabilities)

        response["label_to_option"] = {
            label: str(option) for label, option in zip(labels, options)
        }
        response["raw_label_log_scores"] = dict(zip(labels, scores))
        response["normalized_label_probabilities"] = dict(zip(labels, probabilities))
        response["serialized_prompt_sha256"] = hashlib.sha256(
            response["serialized_prompt"].encode("utf-8")
        ).hexdigest()
        self.last_scoring = response
        return distribution

    def get_last_scoring_metadata(self) -> Dict[str, Any]:
        if not self.last_scoring:
            raise RuntimeError("no GGUF scoring result is available")
        return json.loads(json.dumps(self.last_scoring))

    def get_runtime_metadata(self) -> Dict[str, Any]:
        if not self.runtime_metadata:
            raise RuntimeError("GGUF model is not loaded")
        return json.loads(json.dumps(self.runtime_metadata))

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            return_code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = process.wait(timeout=30)
        if self._log_handle is not None:
            self._log_handle.write(f"llama.cpp helper exit code: {return_code}\n")
            self._log_handle.flush()
        if return_code != 0:
            raise RuntimeError(f"llama.cpp helper exited with code {return_code}")

    def __enter__(self) -> "GGUFRunner":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "GGUFRunner",
    "QWEN25_DEFAULT_SYSTEM_MESSAGE",
    "serialize_qwen25_chatml",
]

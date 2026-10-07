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
QWEN25_CHAT_PROFILE = "qwen2.5"
MISTRAL_V03_CHAT_PROFILE = "mistral-v0.3"
MISTRAL_TEMPLATE_EQUIVALENCE_POLICY = (
    "embedded_serialization_token_ids_equal_canonical_with_one_leading_bos"
)
MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION = (
    "token_id_equivalent_to_canonical_mistral_v03_with_exactly_one_leading_bos"
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


def serialize_mistral_v03_chat(messages: Sequence[Mapping[str, str]]) -> str:
    """Serialize the embedded Mistral-v0.3 user/assistant template exactly.

    The pinned GGUF template begins with ``<s>``, accepts no system role, and
    places assistant content immediately after ``[/INST]`` without inserting
    a leading space.  This function intentionally covers only the alternating
    non-tool path used by the experiment.
    """

    if not messages:
        raise ValueError("Mistral messages must not be empty")
    pieces = ["<s>"]
    for index, message in enumerate(messages):
        role = str(message["role"])
        content = str(message["content"])
        expected_role = "user" if index % 2 == 0 else "assistant"
        if role != expected_role:
            raise ValueError(
                "Mistral messages must alternate user/assistant beginning with user"
            )
        if not content:
            raise ValueError("Mistral message content must not be empty")
        if role == "user":
            pieces.append(f"[INST] {content} [/INST]")
        else:
            pieces.append(f"{content}</s>")
    return "".join(pieces)


def _first_differing_utf8_byte(actual: bytes, expected: bytes) -> Dict[str, Any] | None:
    """Return a lossless description of the first differing UTF-8 byte."""

    shared_length = min(len(actual), len(expected))
    for offset in range(shared_length):
        if actual[offset] != expected[offset]:
            return {
                "offset": offset,
                "actual": actual[offset],
                "expected": expected[offset],
            }
    if len(actual) != len(expected):
        return {
            "offset": shared_length,
            "actual": actual[shared_length] if shared_length < len(actual) else None,
            "expected": expected[shared_length] if shared_length < len(expected) else None,
        }
    return None


def build_mistral_template_equivalence_diagnostic(
    actual_serialization: str,
    expected_serialization: str,
    actual_token_ids: Sequence[int],
    expected_token_ids: Sequence[int],
    bos_token_id: int,
) -> Dict[str, Any]:
    """Record and evaluate the Mistral embedded/canonical equivalence gate.

    The pinned llama.cpp renderer applies the GGUF's embedded template but
    emits its BOS through tokenizer configuration rather than as the literal
    ``<s>`` bytes present in the Jinja source.  We therefore preserve both byte
    strings and accept the embedded path only when it yields exactly the same
    token sequence as the explicit canonical serialization, with exactly one
    BOS and that BOS in the leading position on both paths.
    """

    if not isinstance(actual_serialization, str) or not isinstance(
        expected_serialization, str
    ):
        raise TypeError("Mistral serializations must be strings")
    actual_ids = list(actual_token_ids)
    expected_ids = list(expected_token_ids)
    if (
        not actual_ids
        or not expected_ids
        or not all(isinstance(token, int) and token >= 0 for token in actual_ids)
        or not all(isinstance(token, int) and token >= 0 for token in expected_ids)
        or not isinstance(bos_token_id, int)
        or isinstance(bos_token_id, bool)
        or bos_token_id < 0
    ):
        raise ValueError("Mistral template comparison requires valid token IDs")

    actual_bytes = actual_serialization.encode("utf-8")
    expected_bytes = expected_serialization.encode("utf-8")
    actual_bos_count = actual_ids.count(bos_token_id)
    expected_bos_count = expected_ids.count(bos_token_id)
    actual_one_leading_bos = actual_ids[0] == bos_token_id and actual_bos_count == 1
    expected_one_leading_bos = (
        expected_ids[0] == bos_token_id and expected_bos_count == 1
    )
    token_ids_equivalent = actual_ids == expected_ids
    return {
        "policy": MISTRAL_TEMPLATE_EQUIVALENCE_POLICY,
        "actual_serialized_prompt_repr": repr(actual_serialization),
        "expected_serialized_prompt_repr": repr(expected_serialization),
        "actual_character_length": len(actual_serialization),
        "expected_character_length": len(expected_serialization),
        "actual_utf8_byte_length": len(actual_bytes),
        "expected_utf8_byte_length": len(expected_bytes),
        "actual_sha256": hashlib.sha256(actual_bytes).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "first_differing_byte": _first_differing_utf8_byte(
            actual_bytes, expected_bytes
        ),
        "actual_token_ids": actual_ids,
        "expected_token_ids": expected_ids,
        "token_ids_equivalent": token_ids_equivalent,
        "bos_token_id": bos_token_id,
        "actual_bos_token_count": actual_bos_count,
        "expected_bos_token_count": expected_bos_count,
        "actual_starts_with_bos": actual_ids[0] == bos_token_id,
        "expected_starts_with_bos": expected_ids[0] == bos_token_id,
        "actual_has_exactly_one_leading_bos": actual_one_leading_bos,
        "expected_has_exactly_one_leading_bos": expected_one_leading_bos,
        "actual_tokenization_add_special": True,
        "expected_tokenization_add_special": False,
        "accepted": (
            token_ids_equivalent
            and actual_one_leading_bos
            and expected_one_leading_bos
        ),
    }


class MistralTemplateEquivalenceError(RuntimeError):
    """Raised with a preserved byte/token diagnostic when the gate fails."""

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        self.diagnostic = json.loads(json.dumps(dict(diagnostic)))
        summary = {
            "actual_sha256": self.diagnostic.get("actual_sha256"),
            "expected_sha256": self.diagnostic.get("expected_sha256"),
            "first_differing_byte": self.diagnostic.get("first_differing_byte"),
            "token_ids_equivalent": self.diagnostic.get("token_ids_equivalent"),
            "actual_bos_token_count": self.diagnostic.get("actual_bos_token_count"),
            "expected_bos_token_count": self.diagnostic.get("expected_bos_token_count"),
        }
        super().__init__(
            "Mistral embedded-template output is not safely token-equivalent to "
            f"the canonical serialization: {json.dumps(summary, sort_keys=True)}"
        )


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
        chat_profile: str = QWEN25_CHAT_PROFILE,
    ) -> None:
        self.helper_path = Path(helper_path).resolve()
        self.model_path = Path(model_path).resolve()
        self.model_identifier = str(model_identifier)
        self.repository = str(repository)
        self.revision = str(revision)
        self.quantization = str(quantization)
        self.threads = int(threads)
        self.context_size = int(context_size)
        self.chat_profile = str(chat_profile)
        if not self.helper_path.is_file():
            raise FileNotFoundError(f"llama.cpp scoring helper is missing: {self.helper_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF first shard is missing: {self.model_path}")
        if self.threads <= 0 or self.context_size <= 0:
            raise ValueError("threads and context_size must be positive")
        if self.chat_profile not in {QWEN25_CHAT_PROFILE, MISTRAL_V03_CHAT_PROFILE}:
            raise ValueError(f"unsupported GGUF chat profile: {self.chat_profile!r}")

        self._process: subprocess.Popen[str] | None = None
        self._log_handle: IO[str] | None = None
        self._request_number = 0
        self.runtime_metadata: Dict[str, Any] = {}
        self.last_scoring: Dict[str, Any] = {}
        self.last_structured_messages: list[dict[str, str]] = []
        self.last_template_equivalence_diagnostic: Dict[str, Any] = {}

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
        tokenizer_add_special = True
        canonical_add_special = (
            False if self.chat_profile == MISTRAL_V03_CHAT_PROFILE else True
        )
        self.runtime_metadata = {
            **ready,
            "model_identifier": self.model_identifier,
            "repository": self.repository,
            "revision": self.revision,
            "quantization": self.quantization,
            "model_path": str(self.model_path),
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "embedded_chat_template": template,
            "synthetic": False,
            "generated_answer_parsing": False,
            "candidate_scoring": "full_vocabulary_low_level_logits",
            "chat_profile": self.chat_profile,
            "tokenizer_add_special_tokens": tokenizer_add_special,
            "tokenization_add_special": tokenizer_add_special,
            "canonical_tokenization_add_special": canonical_add_special,
            "assistant_label_prefix": "" if self.chat_profile == MISTRAL_V03_CHAT_PROFILE else None,
            "template_equivalence_policy": (
                MISTRAL_TEMPLATE_EQUIVALENCE_POLICY
                if self.chat_profile == MISTRAL_V03_CHAT_PROFILE
                else "byte_exact"
            ),
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
        if self.chat_profile == QWEN25_CHAT_PROFILE:
            expected_serialization = serialize_qwen25_chatml(structured_messages)
            actual_add_special = True
        else:
            expected_serialization = serialize_mistral_v03_chat(structured_messages)
            # The pinned llama.cpp renderer uses the GGUF's embedded template,
            # but its detected legacy Mistral path emits BOS through tokenizer
            # configuration rather than literal ``<s>`` bytes.  The helper
            # independently tokenizes the explicit canonical form without
            # adding a special token and requires exact token-ID equivalence.
            actual_add_special = True
        request = {
            "request_id": request_id,
            "messages": structured_messages,
            "labels": [str(label) for label in labels],
            "candidate_continuations": {
                str(label): str(label) for label in labels
            },
            "add_special_tokens": actual_add_special,
            "tokenization_add_special": actual_add_special,
            "candidate_prefix": "",
        }
        if self.chat_profile == MISTRAL_V03_CHAT_PROFILE:
            request.update(
                {
                    "canonical_serialized_prompt": expected_serialization,
                    "canonical_tokenization_add_special": False,
                    "template_equivalence_policy": (
                        MISTRAL_TEMPLATE_EQUIVALENCE_POLICY
                    ),
                    "chat_template_source": "embedded_gguf_metadata",
                }
            )
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        response_type = response.get("type")
        if response_type not in {"result", "template_validation_error"} or response.get(
            "request_id"
        ) != request_id:
            raise RuntimeError("GGUF helper response did not match the scoring request")
        if response_type == "result" and response.get("generated_answer") is not None:
            raise RuntimeError("GGUF backend unexpectedly generated an answer")
        if self.chat_profile == QWEN25_CHAT_PROFILE:
            verification = "byte_exact_qwen25_non_tool_chatml_expansion"
        else:
            verification = MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION
        if self.chat_profile == QWEN25_CHAT_PROFILE and response.get(
            "serialized_prompt"
        ) != expected_serialization:
            raise RuntimeError(
                "llama.cpp serialization does not match the selected embedded "
                "template's exact non-tool expansion"
            )
        if response.get("add_special_tokens") is not actual_add_special:
            raise RuntimeError("llama.cpp tokenization special-token mode is incorrect")
        if response.get("tokenization_add_special") is not actual_add_special:
            raise RuntimeError("llama.cpp tokenization mode audit alias is incorrect")
        if self.chat_profile == MISTRAL_V03_CHAT_PROFILE:
            actual_serialization = response.get("serialized_prompt")
            helper_expected = response.get("canonical_serialized_prompt")
            if not isinstance(actual_serialization, str) or helper_expected != (
                expected_serialization
            ):
                raise RuntimeError(
                    "Mistral helper did not preserve the actual and canonical serializations"
                )
            diagnostic = build_mistral_template_equivalence_diagnostic(
                actual_serialization,
                expected_serialization,
                response.get("prompt_token_ids", []),
                response.get("canonical_prompt_token_ids", []),
                response.get("bos_token_id"),
            )
            response["template_equivalence"] = diagnostic
            response["canonical_serialized_prompt"] = expected_serialization
            self.last_template_equivalence_diagnostic = diagnostic
            self.last_structured_messages = structured_messages
            self.last_scoring = response
            helper_contract_valid = (
                response.get("chat_template_source") == "embedded_gguf_metadata"
                and response.get("template_equivalence_policy")
                == MISTRAL_TEMPLATE_EQUIVALENCE_POLICY
                and response.get("canonical_tokenization_add_special") is False
                and response.get("template_equivalence_preflight_passed") is True
                and response.get("candidate_prefix") == ""
            )
            if (
                response_type != "result"
                or not helper_contract_valid
                or diagnostic.get("accepted") is not True
            ):
                raise MistralTemplateEquivalenceError(diagnostic)
        candidates = response.get("candidates")
        if not isinstance(candidates, Mapping):
            raise RuntimeError("GGUF helper candidate trace is missing")
        for label in labels:
            candidate = candidates.get(str(label))
            if not isinstance(candidate, Mapping):
                raise RuntimeError(f"GGUF helper candidate trace is missing for {label}")
            if candidate.get("continuation") != str(label):
                raise RuntimeError(f"candidate {label} was not scored as the bare label")
        response["embedded_template_serialization_verified"] = True
        response["serialization_verification"] = verification
        response["chat_profile"] = self.chat_profile
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
        if self.chat_profile == QWEN25_CHAT_PROFILE:
            messages = [
                {"role": "system", "content": QWEN25_DEFAULT_SYSTEM_MESSAGE},
                {"role": "user", "content": str(prompt)},
            ]
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        response = self.score_messages(messages, labels)
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

    def get_last_template_equivalence_diagnostic(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.last_template_equivalence_diagnostic))

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
    "MISTRAL_TEMPLATE_EQUIVALENCE_POLICY",
    "MISTRAL_TEMPLATE_EQUIVALENCE_VERIFICATION",
    "MistralTemplateEquivalenceError",
    "QWEN25_DEFAULT_SYSTEM_MESSAGE",
    "QWEN25_CHAT_PROFILE",
    "MISTRAL_V03_CHAT_PROFILE",
    "serialize_qwen25_chatml",
    "serialize_mistral_v03_chat",
    "build_mistral_template_equivalence_diagnostic",
]

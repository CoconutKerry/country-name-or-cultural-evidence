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

GEMMA2_EMBEDDED_CHAT_TEMPLATE = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}{% if (message['role'] == 'user') != "
    "(loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must "
    "alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + "
    "'<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}"
    "{{'<start_of_turn>model\n'}}{% endif %}"
)
GEMMA2_EMBEDDED_CHAT_TEMPLATE_SHA256 = (
    "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6"
)
GEMMA2_ASSISTANT_ANSWER_SUFFIX = "<start_of_turn>model\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _leading_token_count(token_ids: Sequence[int], token_id: int) -> int:
    count = 0
    for observed in token_ids:
        if observed != token_id:
            break
        count += 1
    return count


def first_serialization_difference(
    llama_cpp_text: str,
    independent_text: str,
) -> dict[str, Any] | None:
    """Describe the first character/UTF-8 byte difference without hiding EOF."""

    limit = min(len(llama_cpp_text), len(independent_text))
    character_index = next(
        (
            index
            for index in range(limit)
            if llama_cpp_text[index] != independent_text[index]
        ),
        limit,
    )
    if character_index == limit and len(llama_cpp_text) == len(independent_text):
        return None
    llama_character = (
        llama_cpp_text[character_index]
        if character_index < len(llama_cpp_text)
        else None
    )
    independent_character = (
        independent_text[character_index]
        if character_index < len(independent_text)
        else None
    )
    llama_bytes = llama_cpp_text.encode("utf-8")
    independent_bytes = independent_text.encode("utf-8")
    byte_limit = min(len(llama_bytes), len(independent_bytes))
    byte_index = next(
        (
            index
            for index in range(byte_limit)
            if llama_bytes[index] != independent_bytes[index]
        ),
        byte_limit,
    )
    return {
        "character_index": character_index,
        "utf8_byte_index": byte_index,
        "llama_cpp_character": llama_character,
        "llama_cpp_codepoint": (
            f"U+{ord(llama_character):04X}" if llama_character is not None else None
        ),
        "independent_character": independent_character,
        "independent_codepoint": (
            f"U+{ord(independent_character):04X}"
            if independent_character is not None
            else None
        ),
    }


def serialization_diagnostic(
    *,
    llama_cpp_text: str,
    independent_text: str,
    llama_cpp_token_ids: Sequence[int],
    independent_token_ids: Sequence[int],
    bos_token_id: int,
    required_suffix: str | None,
    required_leading_bos_count: int | None,
    allow_token_id_equivalence: bool,
) -> dict[str, Any]:
    """Build a complete, JSON-safe formatter comparison and acceptance record."""

    actual_tokens = [int(token_id) for token_id in llama_cpp_token_ids]
    independent_tokens = [int(token_id) for token_id in independent_token_ids]
    byte_exact = llama_cpp_text == independent_text
    token_id_equivalent = actual_tokens == independent_tokens
    equivalence_accepted = byte_exact or (
        allow_token_id_equivalence and token_id_equivalent
    )
    actual_bos_count = _leading_token_count(actual_tokens, bos_token_id)
    independent_bos_count = _leading_token_count(independent_tokens, bos_token_id)
    suffix_verified = (
        True
        if required_suffix is None
        else llama_cpp_text.endswith(required_suffix)
        and independent_text.endswith(required_suffix)
    )
    bos_verified = (
        True
        if required_leading_bos_count is None
        else actual_bos_count == required_leading_bos_count
        and independent_bos_count == required_leading_bos_count
    )
    return {
        "llama_cpp": {
            "text": llama_cpp_text,
            "sha256": _sha256_text(llama_cpp_text),
            "character_length": len(llama_cpp_text),
            "utf8_byte_length": len(llama_cpp_text.encode("utf-8")),
            "token_ids": actual_tokens,
            "token_count": len(actual_tokens),
            "leading_bos_token_count": actual_bos_count,
            "has_exact_required_suffix": (
                llama_cpp_text.endswith(required_suffix)
                if required_suffix is not None
                else None
            ),
        },
        "independent_expansion": {
            "text": independent_text,
            "sha256": _sha256_text(independent_text),
            "character_length": len(independent_text),
            "utf8_byte_length": len(independent_text.encode("utf-8")),
            "token_ids": independent_tokens,
            "token_count": len(independent_tokens),
            "leading_bos_token_count": independent_bos_count,
            "has_exact_required_suffix": (
                independent_text.endswith(required_suffix)
                if required_suffix is not None
                else None
            ),
        },
        "bos_token_id": int(bos_token_id),
        "required_leading_bos_token_count": required_leading_bos_count,
        "required_suffix": required_suffix,
        "byte_exact": byte_exact,
        "token_id_equivalent": token_id_equivalent,
        "allow_token_id_equivalence": allow_token_id_equivalence,
        "equivalence_accepted": equivalence_accepted,
        "exact_required_suffix_verified": suffix_verified,
        "leading_bos_requirement_verified": bos_verified,
        "first_difference": first_serialization_difference(
            llama_cpp_text, independent_text
        ),
        "verification_mode": (
            "byte_exact"
            if byte_exact
            else "token_id_equivalent"
            if allow_token_id_equivalence and token_id_equivalent
            else "rejected"
        ),
    }


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


def serialize_gemma2_chat(messages: Sequence[Mapping[str, str]]) -> str:
    """Expand the pinned Gemma 2 non-tool template byte-for-byte."""

    if not messages:
        raise ValueError("Gemma messages must not be empty")
    pieces = ["<bos>"]
    for index, message in enumerate(messages):
        role = str(message["role"])
        content = str(message["content"]).strip()
        expected_role = "user" if index % 2 == 0 else "assistant"
        if role == "system":
            raise ValueError("Gemma 2 embedded template does not support system turns")
        if role != expected_role:
            raise ValueError("Gemma 2 messages must alternate user and assistant")
        if not content:
            raise ValueError("Gemma messages must not have empty content")
        rendered_role = "model" if role == "assistant" else role
        pieces.append(
            f"<start_of_turn>{rendered_role}\n{content}<end_of_turn>\n"
        )
    pieces.append("<start_of_turn>model\n")
    return "".join(pieces)


class GGUFRunner:
    """Keep one CPU-only GGUF model loaded while scoring label continuations."""

    backend = "llama.cpp"
    is_synthetic = False

    def _validate_loaded_chat_template(self, template: str) -> None:
        """Model-family hook for exact embedded-template validation."""

    def _validate_structured_messages(
        self, messages: Sequence[Mapping[str, str]]
    ) -> None:
        """Model-family hook for role/content structure validation."""

    @property
    def expected_chat_template_sha256(self) -> str | None:
        return None

    @property
    def required_serialized_answer_suffix(self) -> str | None:
        return None

    @property
    def required_leading_bos_count(self) -> int | None:
        return None

    @property
    def allow_token_id_serialization_equivalence(self) -> bool:
        return False

    @property
    def select_actual_tokenization_mode(self) -> bool:
        return False

    def _serialize_messages(self, messages: Sequence[Mapping[str, str]]) -> str:
        return serialize_qwen25_chatml(messages)

    def _messages_for_prompt(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": QWEN25_DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": str(prompt)},
        ]

    @property
    def serialization_verification_name(self) -> str:
        return "byte_exact_qwen25_non_tool_chatml_expansion"

    def _serialization_contract(self) -> dict[str, Any]:
        return {
            "allow_token_id_equivalence": (
                self.allow_token_id_serialization_equivalence
            ),
            "required_suffix": self.required_serialized_answer_suffix,
            "required_leading_bos_token_count": self.required_leading_bos_count,
            "select_actual_tokenization_mode": (
                self.select_actual_tokenization_mode
            ),
        }

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
        self.last_serialization_diagnostic: Dict[str, Any] = {}

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
        try:
            self._validate_loaded_chat_template(template)
        except Exception:
            self.close()
            raise
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
        expected_hash = self.expected_chat_template_sha256
        if (
            expected_hash is not None
            and self.runtime_metadata["chat_template_sha256"] != expected_hash
        ):
            self.close()
            raise RuntimeError("loaded GGUF embedded chat-template SHA-256 is not pinned")

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
        self._validate_structured_messages(structured_messages)
        independent_serialization = self._serialize_messages(structured_messages)
        self._request_number += 1
        request_id = str(self._request_number)
        request = {
            "request_id": request_id,
            "messages": structured_messages,
            "labels": [str(label) for label in labels],
            "reference_serialized_prompt": independent_serialization,
            "serialization_contract": self._serialization_contract(),
        }
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if response.get("type") != "result" or response.get("request_id") != request_id:
            raise RuntimeError("GGUF helper response did not match the scoring request")
        if response.get("generated_answer") is not None:
            raise RuntimeError("GGUF backend unexpectedly generated an answer")
        diagnostic = self._build_serialization_diagnostic(
            response,
            independent_serialization=independent_serialization,
            structured_messages=structured_messages,
        )
        self.last_serialization_diagnostic = diagnostic
        if diagnostic["status"] != "PASS":
            raise RuntimeError(
                "llama.cpp serialization is neither byte-exact nor an accepted "
                "token-ID-equivalent expansion of the pinned embedded template"
            )
        response["embedded_template_serialization_verified"] = True
        response["serialization_verification"] = self.serialization_verification_name
        response["serialization_verification_mode"] = diagnostic["serialization"][
            "verification_mode"
        ]
        response["serialization_byte_exact"] = diagnostic["serialization"][
            "byte_exact"
        ]
        response["serialization_token_ids_equal"] = diagnostic["serialization"][
            "token_id_equivalent"
        ]
        response["independent_serialized_prompt"] = independent_serialization
        response["independent_serialized_prompt_sha256"] = diagnostic[
            "serialization"
        ]["independent_expansion"]["sha256"]
        response["independent_prompt_token_ids"] = diagnostic["serialization"][
            "independent_expansion"
        ]["token_ids"]
        response["independent_leading_bos_token_count"] = diagnostic[
            "serialization"
        ]["independent_expansion"]["leading_bos_token_count"]
        response["exact_assistant_answer_suffix_verified"] = diagnostic["checks"][
            "exact_assistant_answer_suffix"
        ]
        response["user_only_message_structure_verified"] = diagnostic["checks"][
            "user_only_message_structure"
        ]
        self.last_structured_messages = structured_messages
        self.last_scoring = response
        return response

    def _build_serialization_diagnostic(
        self,
        response: Mapping[str, Any],
        *,
        independent_serialization: str,
        structured_messages: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        actual = response.get("serialized_prompt")
        echoed_reference = response.get("reference_serialized_prompt")
        actual_tokens = response.get("prompt_token_ids")
        actual_tokens_without_special = response.get(
            "prompt_token_ids_add_special_false"
        )
        actual_tokens_with_special = response.get(
            "prompt_token_ids_add_special_true"
        )
        independent_tokens = response.get("reference_prompt_token_ids")
        selected_mode = response.get("selected_actual_tokenization_mode")
        bos_token_id = response.get("bos_token_id")
        if not isinstance(actual, str):
            raise RuntimeError("GGUF helper omitted the llama.cpp serialized prompt")
        if echoed_reference != independent_serialization:
            raise RuntimeError("GGUF helper changed the independent serialization")
        if (
            not isinstance(actual_tokens, list)
            or not all(isinstance(value, int) for value in actual_tokens)
            or not isinstance(actual_tokens_without_special, list)
            or not all(
                isinstance(value, int) for value in actual_tokens_without_special
            )
            or not isinstance(actual_tokens_with_special, list)
            or not all(isinstance(value, int) for value in actual_tokens_with_special)
            or not isinstance(independent_tokens, list)
            or not all(isinstance(value, int) for value in independent_tokens)
            or not isinstance(bos_token_id, int)
        ):
            raise RuntimeError("GGUF helper omitted serialization tokenization evidence")
        comparison = serialization_diagnostic(
            llama_cpp_text=actual,
            independent_text=independent_serialization,
            llama_cpp_token_ids=actual_tokens,
            independent_token_ids=independent_tokens,
            bos_token_id=bos_token_id,
            required_suffix=self.required_serialized_answer_suffix,
            required_leading_bos_count=self.required_leading_bos_count,
            allow_token_id_equivalence=(
                self.allow_token_id_serialization_equivalence
            ),
        )
        required_bos = self.required_leading_bos_count
        without_special_safe = actual_tokens_without_special == independent_tokens and (
            required_bos is None
            or _leading_token_count(actual_tokens_without_special, bos_token_id)
            == required_bos
        )
        with_special_safe = actual_tokens_with_special == independent_tokens and (
            required_bos is None
            or _leading_token_count(actual_tokens_with_special, bos_token_id)
            == required_bos
        )
        expected_selected_mode = (
            "add_special_false"
            if not self.select_actual_tokenization_mode or without_special_safe
            else "add_special_true"
            if with_special_safe
            else ""
        )
        selected_tokens_consistent = (
            selected_mode == expected_selected_mode
            and (
                actual_tokens_without_special
                if selected_mode == "add_special_false"
                else actual_tokens_with_special
                if selected_mode == "add_special_true"
                else []
            )
            == actual_tokens
        )
        comparison["llama_cpp_tokenization_candidates"] = {
            "add_special_false": {
                "token_ids": actual_tokens_without_special,
                "token_count": len(actual_tokens_without_special),
                "leading_bos_token_count": _leading_token_count(
                    actual_tokens_without_special, bos_token_id
                ),
                "canonical_and_bos_safe": without_special_safe,
            },
            "add_special_true": {
                "token_ids": actual_tokens_with_special,
                "token_count": len(actual_tokens_with_special),
                "leading_bos_token_count": _leading_token_count(
                    actual_tokens_with_special, bos_token_id
                ),
                "canonical_and_bos_safe": with_special_safe,
            },
        }
        comparison["selected_actual_tokenization_mode"] = selected_mode
        reported_bos_count = response.get("leading_bos_token_count")
        helper_contract = response.get("serialization_contract_checks")
        helper_contract_pass = response.get("serialization_contract_pass")
        template_hash = self.runtime_metadata.get("chat_template_sha256")
        template_hash_verified = (
            self.expected_chat_template_sha256 is None
            or template_hash == self.expected_chat_template_sha256
        )
        messages = response.get("structured_messages")
        try:
            if not isinstance(messages, list):
                raise ValueError("missing structured messages")
            self._validate_structured_messages(messages)
            user_only_verified = True
        except (KeyError, TypeError, ValueError):
            user_only_verified = False
        checks = {
            "exact_embedded_template_sha256": template_hash_verified,
            "user_only_message_structure": user_only_verified,
            "serialization_equivalence": comparison["equivalence_accepted"],
            "exact_assistant_answer_suffix": comparison[
                "exact_required_suffix_verified"
            ],
            "exactly_one_leading_bos": comparison[
                "leading_bos_requirement_verified"
            ],
            "helper_reported_bos_count_consistent": (
                reported_bos_count
                == comparison["llama_cpp"]["leading_bos_token_count"]
            ),
            "helper_contract_pass": helper_contract_pass is True,
            "deterministic_tokenization_mode_selection": (
                selected_tokens_consistent
            ),
            "structured_messages_echoed_exactly": messages
            == list(structured_messages),
        }
        return {
            "schema_version": "gguf-serialization-diagnostic-v1",
            "status": "PASS" if all(checks.values()) else "FAILED",
            "model_identifier": self.model_identifier,
            "chat_template_sha256": template_hash,
            "expected_chat_template_sha256": self.expected_chat_template_sha256,
            "structured_messages": messages,
            "checks": checks,
            "helper_contract_checks": helper_contract,
            "serialization": comparison,
        }

    def serialization_preflight(self, prompt: str) -> dict[str, Any]:
        """Inspect live template expansion/tokenization without llama_decode."""

        if self._process is None or self._process.stdin is None:
            raise RuntimeError("call load_model() before serialization preflight")
        structured_messages = self._messages_for_prompt(str(prompt))
        self._validate_structured_messages(structured_messages)
        independent_serialization = self._serialize_messages(structured_messages)
        self._request_number += 1
        request_id = str(self._request_number)
        request = {
            "operation": "serialize_only",
            "request_id": request_id,
            "messages": structured_messages,
            "reference_serialized_prompt": independent_serialization,
            "serialization_contract": self._serialization_contract(),
        }
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if response.get("type") != "serialization" or response.get(
            "request_id"
        ) != request_id:
            raise RuntimeError("GGUF helper response did not match serialization preflight")
        diagnostic = self._build_serialization_diagnostic(
            response,
            independent_serialization=independent_serialization,
            structured_messages=structured_messages,
        )
        diagnostic["checks"]["helper_confirmed_no_inference"] = (
            response.get("inference_performed") is False
        )
        diagnostic["status"] = (
            "PASS" if all(diagnostic["checks"].values()) else "FAILED"
        )
        diagnostic["inference_performed"] = response.get("inference_performed")
        self.last_serialization_diagnostic = diagnostic
        return json.loads(json.dumps(diagnostic))

    def predict_distribution(
        self,
        prompt: str,
        options: Sequence[object],
        temperature: float = 0.0,
    ) -> Dict[str, float]:
        labels = option_labels(len(options))
        response = self.score_messages(self._messages_for_prompt(str(prompt)), labels)
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


class Gemma2GGUFRunner(GGUFRunner):
    """Pinned Gemma 2 adapter using its user-only embedded chat template."""

    def _validate_loaded_chat_template(self, template: str) -> None:
        if template != GEMMA2_EMBEDDED_CHAT_TEMPLATE:
            raise RuntimeError(
                "loaded GGUF chat template does not match the pinned Gemma 2 template"
            )

    def _validate_structured_messages(
        self, messages: Sequence[Mapping[str, str]]
    ) -> None:
        if len(messages) != 1:
            raise ValueError("Gemma 2 scoring requires exactly one user message")
        message = messages[0]
        if set(message) != {"role", "content"} or message.get("role") != "user":
            raise ValueError("Gemma 2 scoring requires a user-only message structure")
        content = message.get("content")
        if not isinstance(content, str) or not content.rstrip().endswith("Answer:"):
            raise ValueError("Gemma 2 user prompt must end at the explicit Answer: position")

    @property
    def expected_chat_template_sha256(self) -> str:
        return GEMMA2_EMBEDDED_CHAT_TEMPLATE_SHA256

    @property
    def required_serialized_answer_suffix(self) -> str:
        return GEMMA2_ASSISTANT_ANSWER_SUFFIX

    @property
    def required_leading_bos_count(self) -> int:
        return 1

    @property
    def allow_token_id_serialization_equivalence(self) -> bool:
        return True

    @property
    def select_actual_tokenization_mode(self) -> bool:
        return True

    def _serialize_messages(self, messages: Sequence[Mapping[str, str]]) -> str:
        return serialize_gemma2_chat(messages)

    def _messages_for_prompt(self, prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(prompt)}]

    @property
    def serialization_verification_name(self) -> str:
        return "pinned_gemma2_byte_or_token_id_equivalent_expansion"


__all__ = [
    "GEMMA2_ASSISTANT_ANSWER_SUFFIX",
    "GEMMA2_EMBEDDED_CHAT_TEMPLATE",
    "GEMMA2_EMBEDDED_CHAT_TEMPLATE_SHA256",
    "Gemma2GGUFRunner",
    "GGUFRunner",
    "QWEN25_DEFAULT_SYSTEM_MESSAGE",
    "first_serialization_difference",
    "serialize_gemma2_chat",
    "serialize_qwen25_chatml",
    "serialization_diagnostic",
]

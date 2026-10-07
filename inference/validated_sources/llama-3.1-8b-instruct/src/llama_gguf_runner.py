"""Llama-3.1-specific genuine llama.cpp option-label scorer.

This module intentionally sits beside :mod:`src.gguf_runner`: the accepted
Qwen scorer performs a Qwen ChatML byte check and therefore must not be reused
unchanged for Llama.  The low-level helper still applies the chat template
stored in the loaded GGUF metadata; this wrapper verifies the pinned Llama 3.1
template identity and its exact answer boundary before accepting any score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, IO, Mapping, Sequence

from src.gguf_runner import GGUFRunner
from src.scoring import map_label_probabilities, normalize_log_scores, option_labels


# The embedded Llama-3.1 template's fixed no-tools preamble.  The pinned
# llama.cpp commit reduces the Jinja template to its Llama-3 role serializer,
# so this content preserves the model-specific default rather than inventing a
# cross-model system instruction.
LLAMA31_DEFAULT_SYSTEM_MESSAGE = (
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024"
)
LLAMA31_CHAT_TEMPLATE_SHA256 = (
    "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65"
)
LLAMA31_BOS = "<|begin_of_text|>"
LLAMA31_HEADER_PREFIX = "<|start_header_id|>"
LLAMA31_HEADER_SUFFIX = "<|end_header_id|>\n\n"
LLAMA31_EOT = "<|eot_id|>"
LLAMA31_ANSWER_SUFFIX = (
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
LLAMA31_JINJA_BOS = re.compile(r"\{\{[-+]?\s*bos_token\s*[-+]?\}\}")


def verify_llama31_chat_template(template: str) -> None:
    """Require the pinned template's Llama markers and BOS expression.

    GGUF metadata may spell BOS as the literal special token or emit it from
    the Jinja ``bos_token`` variable.  Both forms render the same required BOS;
    the runner separately pins the complete template by SHA-256.
    """

    required_markers = (
        LLAMA31_HEADER_PREFIX,
        "assistant",
        LLAMA31_EOT,
        "add_generation_prompt",
    )
    missing = [marker for marker in required_markers if marker not in template]
    if LLAMA31_BOS not in template and LLAMA31_JINJA_BOS.search(template) is None:
        missing.insert(0, "Llama BOS literal or Jinja bos_token")
    if missing:
        raise RuntimeError(
            f"embedded template is not the pinned Llama-3.1 template; missing {missing!r}"
        )


def verify_llama31_serialized_chat(
    serialized_prompt: str,
    messages: Sequence[Mapping[str, str]],
) -> None:
    """Fail unless an embedded Llama-3.1 serialization preserves all turns.

    Llama-3.1's full upstream Jinja template may add its pinned knowledge/date
    preamble to the system section.  We therefore do not recreate the Jinja in
    Python.  Instead, the C++ helper applies the exact embedded template and
    this verifier checks the Llama special-token grammar, ordered message
    contents, per-turn termination, and exact assistant answer suffix.  The
    runner separately pins the complete embedded template by SHA-256.
    """

    if not isinstance(serialized_prompt, str) or not serialized_prompt:
        raise RuntimeError("embedded Llama chat template returned an empty prompt")
    # llama.cpp's LLAMA_3 serializer emits the first role header as text; the
    # tokenizer adds BOS because the helper tokenizes with add_special=true.
    # Some compatible serializers may include textual BOS, so accept it once.
    textual_bos = serialized_prompt.startswith(LLAMA31_BOS)
    expected_start = LLAMA31_BOS if textual_bos else LLAMA31_HEADER_PREFIX
    if not serialized_prompt.startswith(expected_start):
        raise RuntimeError("serialized prompt lacks the first Llama-3.1 role header")
    if not serialized_prompt.endswith(LLAMA31_ANSWER_SUFFIX):
        raise RuntimeError("serialized prompt lacks the exact Llama answer boundary")
    if serialized_prompt.count(LLAMA31_BOS) != (1 if textual_bos else 0):
        raise RuntimeError("serialized prompt contains an invalid Llama BOS count")

    cursor = len(LLAMA31_BOS) if textual_bos else 0
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            raise RuntimeError(f"unsupported Llama message role: {role!r}")
        if not content:
            raise RuntimeError("Llama message content must not be empty")
        header = f"{LLAMA31_HEADER_PREFIX}{role}{LLAMA31_HEADER_SUFFIX}"
        header_at = serialized_prompt.find(header, cursor)
        if header_at < cursor:
            raise RuntimeError(f"serialized prompt is missing ordered {role!r} header")
        content_at = serialized_prompt.find(content.strip(), header_at + len(header))
        if content_at < header_at + len(header):
            raise RuntimeError(f"serialized prompt is missing the {role!r} content")
        eot_at = serialized_prompt.find(LLAMA31_EOT, content_at + len(content.strip()))
        if eot_at < content_at + len(content.strip()):
            raise RuntimeError(f"serialized prompt does not terminate the {role!r} turn")
        cursor = eot_at + len(LLAMA31_EOT)

    suffix_at = len(serialized_prompt) - len(LLAMA31_ANSWER_SUFFIX)
    if cursor > suffix_at:
        raise RuntimeError("Llama generation boundary overlaps the final input turn")


class Llama31GGUFRunner(GGUFRunner):
    """Keep one CPU-only Llama-3.1 GGUF loaded while scoring labels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.expected_chat_template_sha256 = LLAMA31_CHAT_TEMPLATE_SHA256

    def load_model(self, run_log: IO[str]) -> None:
        super().load_model(run_log)
        runtime = self.get_runtime_metadata()
        observed_hash = runtime.get("chat_template_sha256")
        if observed_hash != self.expected_chat_template_sha256:
            self.close()
            raise RuntimeError(
                "embedded Llama-3.1 chat-template SHA-256 mismatch: "
                f"{observed_hash!r} != {self.expected_chat_template_sha256!r}"
            )
        template = str(runtime.get("chat_template", ""))
        try:
            verify_llama31_chat_template(template)
        except RuntimeError:
            self.close()
            raise
        self.runtime_metadata["model_family"] = "llama-3.1"
        self.runtime_metadata["embedded_template_identity_verified"] = True

    def score_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        labels: Sequence[str],
    ) -> Dict[str, Any]:
        """Apply the embedded template and score complete contextual labels."""

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
        if response.get("prompt_bos_added") is not True:
            raise RuntimeError("Llama prompt tokenization did not add the required BOS token")
        if response.get("prompt_bos_token_id") != response.get("bos_token_id"):
            raise RuntimeError("Llama prompt begins with the wrong BOS token ID")
        serialized = response.get("serialized_prompt")
        verify_llama31_serialized_chat(str(serialized), structured_messages)
        response["embedded_template_serialization_verified"] = True
        response["serialization_verification"] = (
            "pinned_llama31_template_hash_and_answer_boundary"
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
                {"role": "system", "content": LLAMA31_DEFAULT_SYSTEM_MESSAGE},
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
            str(response["serialized_prompt"]).encode("utf-8")
        ).hexdigest()
        self.last_scoring = response
        return distribution


__all__ = [
    "LLAMA31_ANSWER_SUFFIX",
    "LLAMA31_CHAT_TEMPLATE_SHA256",
    "LLAMA31_DEFAULT_SYSTEM_MESSAGE",
    "Llama31GGUFRunner",
    "verify_llama31_chat_template",
    "verify_llama31_serialized_chat",
]

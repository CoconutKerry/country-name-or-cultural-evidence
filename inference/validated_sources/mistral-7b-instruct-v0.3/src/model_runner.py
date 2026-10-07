"""Model inference backends for label-based option scoring.

The production runner scores the complete token sequence of each displayed
option label (``A``, ``B``, ...), conditional on the exact prompt. It never
uses the first token of an answer text as a proxy for that answer.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import platform
import sys
from typing import Any, Dict, List, Optional, Sequence

from src.scoring import (
    map_label_probabilities,
    normalize_log_scores,
    option_labels,
)


class ModelRunner:
    """Load a Hugging Face causal LM and score displayed option labels."""

    backend = "huggingface"
    is_synthetic = False

    def __init__(
        self,
        model_name: str,
        hub_id: str,
        *,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        use_chat_template: bool = True,
        trust_remote_code: bool = True,
        model_load_kwargs: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.hub_id = hub_id
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.use_chat_template = bool(use_chat_template)
        self.trust_remote_code = trust_remote_code
        self.model_load_kwargs = dict(model_load_kwargs or {})
        # Keep credentials in memory only.  They are passed directly to the
        # Hub loaders and are deliberately excluded from runtime metadata.
        self._token = token
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None

        # Updated together after each successful prediction, providing an
        # auditable label-to-semantics record without changing the return type.
        self.last_label_map: Dict[str, str] = {}
        self.last_label_probabilities: Dict[str, float] = {}
        self.last_label_log_scores: Dict[str, float] = {}
        self.last_label_token_ids: Dict[str, List[int]] = {}
        self.last_label_continuations: Dict[str, str] = {}
        self.last_label_target_positions: Dict[str, List[int]] = {}
        self.last_label_predictive_positions: Dict[str, List[int]] = {}
        self.last_prompt_token_count = 0
        self.last_prompt_prefix_verified = False
        self.last_raw_prompt = ""
        self.last_rendered_prompt = ""
        self.runtime_metadata: Dict[str, Any] = {}
        self._verified_model_revision: Optional[str] = None
        self._verified_tokenizer_revision: Optional[str] = None

    def load_model(self) -> None:
        """Load tokenizer/model, importing heavy optional dependencies lazily."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The Hugging Face backend requires torch and transformers. "
                "Install repository requirements or use SyntheticModelRunner "
                "for a dependency-free smoke test."
            ) from exc

        if not self.revision or not self.tokenizer_revision:
            raise ValueError(
                "Hugging Face scientific runs require exact model and tokenizer revisions"
            )

        # Verify the immutable Hub refs independently.  Transformers does not
        # consistently retain `_commit_hash` in tokenizer.init_kwargs, so that
        # private field cannot be the sole proof of a resolved tokenizer SHA.
        api = HfApi(token=self._token)
        model_info = api.model_info(
            self.hub_id, revision=self.revision, token=self._token
        )
        self._verified_model_revision = getattr(model_info, "sha", None)
        if self.tokenizer_revision == self.revision:
            self._verified_tokenizer_revision = self._verified_model_revision
        else:
            tokenizer_info = api.model_info(
                self.hub_id,
                revision=self.tokenizer_revision,
                token=self._token,
            )
            self._verified_tokenizer_revision = getattr(tokenizer_info, "sha", None)
        if self._verified_model_revision != self.revision:
            raise RuntimeError(
                "Hub model revision did not resolve to the requested immutable commit: "
                f"{self._verified_model_revision!r} != {self.revision!r}"
            )
        if self._verified_tokenizer_revision != self.tokenizer_revision:
            raise RuntimeError(
                "Hub tokenizer revision did not resolve to the requested immutable commit: "
                f"{self._verified_tokenizer_revision!r} != {self.tokenizer_revision!r}"
            )

        print(f"Loading {self.model_name} from {self.hub_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hub_id,
            revision=self.tokenizer_revision,
            trust_remote_code=self.trust_remote_code,
            token=self._token,
        )

        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": self.trust_remote_code,
        }
        load_kwargs.update(self.model_load_kwargs)
        requested_dtype = load_kwargs.get("torch_dtype")
        if isinstance(requested_dtype, str) and requested_dtype != "auto":
            if not hasattr(torch, requested_dtype):
                raise ValueError(f"Unknown torch dtype: {requested_dtype!r}")
            load_kwargs["torch_dtype"] = getattr(torch, requested_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hub_id,
            revision=self.revision,
            token=self._token,
            **load_kwargs,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self._model_input_device(torch)
        self.runtime_metadata = self._collect_runtime_metadata(torch, load_kwargs)
        print(f"Model loaded; prompt tensors will use {self.device}")

    def _collect_runtime_metadata(
        self,
        torch_module: Any,
        load_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Capture the model/runtime identity needed to reproduce inference."""

        def package_version(name: str) -> Optional[str]:
            try:
                return importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                return None

        def json_safe(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            return str(value)

        tokenizer_commit = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        model_commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        if model_commit is not None and model_commit != self.revision:
            raise RuntimeError(
                "Transformers model commit metadata contradicts the requested commit: "
                f"{model_commit!r} != {self.revision!r}"
            )
        if tokenizer_commit is not None and tokenizer_commit != self.tokenizer_revision:
            raise RuntimeError(
                "Transformers tokenizer commit metadata contradicts the requested commit: "
                f"{tokenizer_commit!r} != {self.tokenizer_revision!r}"
            )
        if self._verified_model_revision != self.revision:
            raise RuntimeError("Model revision was not independently verified with the Hub")
        if self._verified_tokenizer_revision != self.tokenizer_revision:
            raise RuntimeError("Tokenizer revision was not independently verified with the Hub")
        chat_template = getattr(self.tokenizer, "chat_template", None)
        installed_packages = {
            distribution.metadata.get("Name", "unknown"): distribution.version
            for distribution in importlib_metadata.distributions()
        }
        cuda_device: Optional[Dict[str, Any]] = None
        try:
            if torch_module.cuda.is_available():
                properties = torch_module.cuda.get_device_properties(0)
                cuda_device = {
                    "name": torch_module.cuda.get_device_name(0),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                }
        except (AttributeError, RuntimeError):
            cuda_device = None
        return {
            "backend": self.backend,
            "hub_id": self.hub_id,
            "requested_model_revision": self.revision,
            "requested_tokenizer_revision": self.tokenizer_revision,
            "resolved_model_revision": self._verified_model_revision,
            "resolved_tokenizer_revision": self._verified_tokenizer_revision,
            "transformers_model_commit_metadata": model_commit,
            "transformers_tokenizer_commit_metadata": tokenizer_commit,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_dtype": str(getattr(self.model, "dtype", "unknown")),
            "is_loaded_in_4bit": bool(getattr(self.model, "is_loaded_in_4bit", False)),
            "is_loaded_in_8bit": bool(getattr(self.model, "is_loaded_in_8bit", False)),
            "prompt_input_device": str(self.device),
            "hf_device_map": json_safe(getattr(self.model, "hf_device_map", None)),
            "model_load_kwargs": json_safe(load_kwargs),
            "use_chat_template": self.use_chat_template,
            "chat_template_sha256": (
                hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
                if isinstance(chat_template, str)
                else None
            ),
            "torch_version": getattr(torch_module, "__version__", None),
            "transformers_version": package_version("transformers"),
            "accelerate_version": package_version("accelerate"),
            "tokenizers_version": package_version("tokenizers"),
            "safetensors_version": package_version("safetensors"),
            "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
            "cuda_device": cuda_device,
            "python_version": sys.version,
            "platform": platform.platform(),
            "installed_packages": dict(sorted(installed_packages.items())),
        }

    def _model_input_device(self, torch_module: Any) -> Any:
        """Return the embedding device, including for an automatically sharded LM."""

        if self.model is None:
            raise RuntimeError("model has not been loaded")
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, RuntimeError):
            try:
                return next(self.model.parameters()).device
            except (AttributeError, StopIteration):
                return torch_module.device("cpu")

    @staticmethod
    def _label_continuation(label: str, rendered_prompt: str) -> str:
        """Append a boundary-safe label after a model-specific chat prompt.

        Qwen, Llama, and Gemma generation prompts end in whitespace, while a
        Mistral ``[/INST]`` prompt may not.  Add exactly one separator only
        when the rendered chat template did not already supply it.
        """

        if not rendered_prompt:
            raise ValueError("rendered prompt must be non-empty")
        return str(label) if rendered_prompt[-1].isspace() else f" {label}"

    def render_prompt(self, prompt: str) -> str:
        """Render a user message with the checkpoint's pinned chat template."""

        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not loaded; call load_model() first")
        if not self.use_chat_template:
            return prompt
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"{self.model_name} has no chat template; raw-text fallback is disabled"
            )
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("chat template rendered an empty or non-text prompt")
        return rendered

    def score_label_log_likelihoods(
        self,
        prompt: str,
        labels: Sequence[str],
    ) -> List[float]:
        """Score every complete label continuation under the causal LM.

        A label can consist of multiple tokenizer tokens. Its score is the sum
        of the conditional log probabilities of *all* those tokens. This avoids
        both full-answer first-token scoring and assumptions that a displayed
        label is a single token for every model family.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded; call load_model() first")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by load_model
            raise RuntimeError("label likelihood scoring requires torch") from exc

        self.last_raw_prompt = prompt
        rendered_prompt = self.render_prompt(prompt)
        self.last_rendered_prompt = rendered_prompt
        prompt_ids = self.tokenizer.encode(
            rendered_prompt,
            add_special_tokens=not self.use_chat_template,
        )
        if not prompt_ids:
            raise ValueError("the prompt encoded to an empty token sequence")

        input_device = self.device or self._model_input_device(torch)
        scores: List[float] = []
        self.last_prompt_token_count = len(prompt_ids)
        self.last_label_token_ids = {}
        self.last_label_continuations = {}
        self.last_label_target_positions = {}
        self.last_label_predictive_positions = {}
        self.last_prompt_prefix_verified = True

        with torch.inference_mode():
            for label in labels:
                continuation = self._label_continuation(str(label), rendered_prompt)
                combined_ids = self.tokenizer.encode(
                    rendered_prompt + continuation,
                    add_special_tokens=not self.use_chat_template,
                )
                if combined_ids[: len(prompt_ids)] != list(prompt_ids):
                    raise ValueError(
                        "Prompt tokenization is not a prefix of prompt-plus-label; "
                        "choose a boundary-safe label prefix for this tokenizer"
                    )
                continuation_ids = combined_ids[len(prompt_ids) :]
                if not continuation_ids:
                    raise ValueError(
                        f"option label {label!r} encoded to an empty continuation"
                    )
                label_key = str(label)
                target_positions = list(
                    range(len(prompt_ids), len(prompt_ids) + len(continuation_ids))
                )
                predictive_positions = [position - 1 for position in target_positions]
                if predictive_positions[0] != len(prompt_ids) - 1:
                    raise AssertionError("first option-label token is not scored at answer position")
                self.last_label_continuations[label_key] = continuation
                self.last_label_token_ids[label_key] = [int(value) for value in continuation_ids]
                self.last_label_target_positions[label_key] = target_positions
                self.last_label_predictive_positions[label_key] = predictive_positions
                input_ids = torch.tensor(
                    [list(combined_ids)],
                    dtype=torch.long,
                    device=input_device,
                )
                attention_mask = torch.ones_like(input_ids)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                label_log_likelihood = 0.0
                for offset, target_token_id in enumerate(continuation_ids):
                    target_position = len(prompt_ids) + offset
                    predictive_logits = outputs.logits[0, target_position - 1, :]
                    token_log_probs = torch.log_softmax(
                        predictive_logits.float(),
                        dim=-1,
                    )
                    label_log_likelihood += float(
                        token_log_probs[int(target_token_id)].item()
                    )
                scores.append(label_log_likelihood)

        return scores

    def predict_distribution(
        self,
        prompt: str,
        options: Sequence[object],
        temperature: float = 0.0,
    ) -> Dict[str, float]:
        """Return a normalized distribution keyed by semantic answer text.

        ``options`` must be in the exact order displayed in ``prompt``. Label
        likelihoods are normalized and then mapped to those semantic texts.
        The legacy ``temperature=0`` default means unscaled likelihoods rather
        than argmax, because all audit metrics require a full distribution.
        """

        labels = option_labels(len(options))
        log_scores = self.score_label_log_likelihoods(prompt, labels)
        probabilities = normalize_log_scores(log_scores, temperature=temperature)
        distribution = map_label_probabilities(labels, options, probabilities)

        option_texts = [str(option) for option in options]
        self.last_label_map = dict(zip(labels, option_texts))
        self.last_label_probabilities = dict(zip(labels, probabilities))
        self.last_label_log_scores = dict(zip(labels, log_scores))
        return distribution

    def get_last_scoring_metadata(self) -> Dict[str, Any]:
        """Return a copy of the label-to-semantics audit trail."""

        return {
            "backend": self.backend,
            "synthetic": self.is_synthetic,
            "label_to_option": dict(self.last_label_map),
            "label_probabilities": dict(self.last_label_probabilities),
            "label_log_scores": dict(self.last_label_log_scores),
            "candidate_label_continuations": dict(self.last_label_continuations),
            "candidate_label_token_ids": {
                label: list(token_ids)
                for label, token_ids in self.last_label_token_ids.items()
            },
            "label_target_token_positions": {
                label: list(positions)
                for label, positions in self.last_label_target_positions.items()
            },
            "label_predictive_logit_positions": {
                label: list(positions)
                for label, positions in self.last_label_predictive_positions.items()
            },
            "prompt_token_count": self.last_prompt_token_count,
            "prompt_prefix_verified": self.last_prompt_prefix_verified,
            "raw_prompt_sha256": hashlib.sha256(
                self.last_raw_prompt.encode("utf-8")
            ).hexdigest(),
            "rendered_prompt_sha256": hashlib.sha256(
                self.last_rendered_prompt.encode("utf-8")
            ).hexdigest(),
            "rendered_prompt": self.last_rendered_prompt,
        }

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return immutable model/runtime identity recorded after loading."""

        return dict(self.runtime_metadata)

    def run_experiment(self, prompts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run prepared prompts and retain the option-label audit trail."""

        results: List[Dict[str, Any]] = []
        for item in prompts:
            prediction = self.predict_distribution(item["prompt"], item["options"])
            results.append(
                {
                    "condition": item["condition"],
                    "question_id": item["question_id"],
                    "country": item.get("country"),
                    "prediction": prediction,
                    "human_distribution": item.get("human_distribution"),
                    "scoring": self.get_last_scoring_metadata(),
                }
            )
        return results


class SyntheticModelRunner(ModelRunner):
    """Dependency-free deterministic backend for smoke tests only.

    This backend does **not** call or emulate a language model. Hash-derived
    scores merely exercise prompt construction, label mapping, normalization,
    condition coverage, and output serialization. Synthetic outputs must not
    be reported as experimental findings.
    """

    backend = "synthetic-smoke-only"
    is_synthetic = True

    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            model_name="SYNTHETIC-SMOKE-ONLY",
            hub_id="synthetic://deterministic-hash",
            use_chat_template=False,
            trust_remote_code=False,
        )
        self.seed = int(seed)

    def load_model(self) -> None:
        """Mark the no-model backend ready; no dependencies or weights are loaded."""

        self.tokenizer = "SYNTHETIC-NO-TOKENIZER"
        self.model = "SYNTHETIC-NO-MODEL"
        self.device = "none"
        self.runtime_metadata = {
            "backend": self.backend,
            "hub_id": self.hub_id,
            "use_chat_template": False,
            "model_class": None,
            "tokenizer_class": None,
            "note": "No model or tokenizer was loaded; plumbing-only smoke backend",
        }
        print("Using SYNTHETIC smoke backend; outputs are not model results.")

    def score_label_log_likelihoods(
        self,
        prompt: str,
        labels: Sequence[str],
    ) -> List[float]:
        """Return deterministic hash scores solely to exercise the pipeline."""

        self.last_raw_prompt = prompt
        self.last_rendered_prompt = prompt
        scores: List[float] = []
        for label in labels:
            payload = f"{self.seed}\0{prompt}\0{label}".encode("utf-8")
            digest = hashlib.sha256(payload).digest()
            unit_interval = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
            scores.append(-4.0 + 4.0 * unit_interval)
        return scores


__all__ = ["ModelRunner", "SyntheticModelRunner"]

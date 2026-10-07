"""Pure utilities for label-based multiple-choice scoring.

The experiment presents answer texts under short labels (``A``, ``B``, ...).
Model likelihoods must therefore be computed for those labels and then mapped
back to the semantic answer texts.  Keeping that mapping here makes the
scoring contract explicit and testable without loading a language model.
"""

from __future__ import annotations

import math
import string
from collections.abc import Mapping, Sequence
from typing import Dict, List, Union


ProbabilityInput = Union[Sequence[float], Mapping[str, float]]


def option_labels(num_options: int) -> List[str]:
    """Return deterministic ``A``-through-``Z`` labels for an option set.

    Single-character labels keep all candidates visually comparable.  The
    repository currently contains at most 13 options per question; rejecting
    larger sets is safer than silently producing punctuation after ``Z``.
    """

    if isinstance(num_options, bool) or not isinstance(num_options, int):
        raise TypeError("num_options must be an integer")
    if not 1 <= num_options <= len(string.ascii_uppercase):
        raise ValueError("num_options must be between 1 and 26")
    return list(string.ascii_uppercase[:num_options])


def normalize_log_scores(
    log_scores: Sequence[float],
    temperature: float = 1.0,
) -> List[float]:
    """Convert finite candidate log scores to probabilities via stable softmax.

    ``temperature=0`` is accepted as a backwards-compatible spelling for no
    rescaling (equivalent to ``temperature=1``).  A zero temperature cannot be
    used as an argmax here because the experiment needs a probability
    distribution rather than a generated response.
    """

    scores = [float(score) for score in log_scores]
    if not scores:
        raise ValueError("at least one log score is required")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("all log scores must be finite")

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    effective_temperature = 1.0 if temperature == 0 else temperature

    scaled = [score / effective_temperature for score in scores]
    maximum = max(scaled)
    weights = [math.exp(score - maximum) for score in scaled]
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("log scores could not be normalized")

    probabilities = [weight / total for weight in weights]
    # Normalize once more with fsum so downstream checks are not sensitive to
    # tiny accumulated floating-point error.
    probability_total = math.fsum(probabilities)
    return [probability / probability_total for probability in probabilities]


def _canonical_option_texts(options: Sequence[object]) -> List[str]:
    texts = [str(option) for option in options]
    if not texts:
        raise ValueError("at least one displayed option is required")
    if len(set(texts)) != len(texts):
        raise ValueError("semantic option texts must be unique after string conversion")
    return texts


def map_label_probabilities(
    labels: Sequence[str],
    displayed_options: Sequence[object],
    probabilities: ProbabilityInput,
) -> Dict[str, float]:
    """Map displayed-label probabilities back to semantic option text.

    The association is positional: if a permutation displays ``No`` under
    label ``A``, probability ``P(A)`` belongs to ``No``.  Returned dictionaries
    are keyed by semantic answer text so they remain compatible with the human
    response distributions used by the analysis.
    """

    labels = [str(label) for label in labels]
    option_texts = _canonical_option_texts(displayed_options)
    if len(labels) != len(option_texts):
        raise ValueError("labels and displayed_options must have equal lengths")
    if len(set(labels)) != len(labels):
        raise ValueError("option labels must be unique")

    if isinstance(probabilities, Mapping):
        missing = [label for label in labels if label not in probabilities]
        extras = [label for label in probabilities if label not in set(labels)]
        if missing or extras:
            raise ValueError(
                f"label probability keys do not match labels "
                f"(missing={missing}, extras={extras})"
            )
        values = [float(probabilities[label]) for label in labels]
    else:
        values = [float(probability) for probability in probabilities]
        if len(values) != len(labels):
            raise ValueError("probabilities and labels must have equal lengths")

    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")

    normalized = [value / total for value in values]
    return {
        option_text: probability
        for option_text, probability in zip(option_texts, normalized)
    }


def recover_canonical_distribution(
    semantic_distribution: Mapping[str, float],
    canonical_options: Sequence[object],
) -> Dict[str, float]:
    """Return a semantic distribution in the original canonical option order.

    This is the final step after a displayed option permutation.  It never
    reassigns probabilities by position: values follow their semantic option
    text back into canonical order.
    """

    canonical_texts = _canonical_option_texts(canonical_options)
    semantic_keys = {str(key) for key in semantic_distribution}
    canonical_keys = set(canonical_texts)
    if semantic_keys != canonical_keys:
        missing = sorted(canonical_keys - semantic_keys)
        extras = sorted(semantic_keys - canonical_keys)
        raise ValueError(
            f"semantic distribution does not match canonical options "
            f"(missing={missing}, extras={extras})"
        )

    values = {str(key): float(value) for key, value in semantic_distribution.items()}
    return {option: values[option] for option in canonical_texts}


__all__ = [
    "map_label_probabilities",
    "normalize_log_scores",
    "option_labels",
    "recover_canonical_distribution",
]

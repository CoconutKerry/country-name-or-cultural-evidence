from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def _array(distribution: Mapping[str, float], keys: Sequence[str]) -> np.ndarray:
    if set(distribution) != set(keys):
        raise ValueError("distribution keys do not match the option set")
    values = np.asarray([distribution[key] for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("distribution contains invalid values")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("distribution has zero mass")
    return values / total


def js_divergence(p: Mapping[str, float], q: Mapping[str, float], keys: Sequence[str]) -> float:
    """Base-2 Jensen--Shannon divergence, bounded to [0, 1]."""
    p_arr = _array(p, keys)
    q_arr = _array(q, keys)
    midpoint = 0.5 * (p_arr + q_arr)
    p_mask = p_arr > 0
    q_mask = q_arr > 0
    kl_p = float(np.sum(p_arr[p_mask] * np.log2(p_arr[p_mask] / midpoint[p_mask])))
    kl_q = float(np.sum(q_arr[q_mask] * np.log2(q_arr[q_mask] / midpoint[q_mask])))
    value = 0.5 * (kl_p + kl_q)
    if not math.isfinite(value) or value < -1e-12 or value > 1 + 1e-12:
        raise ValueError(f"invalid Jensen--Shannon divergence: {value}")
    return min(1.0, max(0.0, value))


def country_influence(baseline, country_label, target, keys) -> float:
    return js_divergence(baseline, target, keys) - js_divergence(country_label, target, keys)


def evidence_influence(baseline, evidence, target, keys) -> float:
    return js_divergence(baseline, target, keys) - js_divergence(evidence, target, keys)


def evidence_override_raw(conflict, label_reference, evidence_reference, keys) -> float:
    return js_divergence(conflict, label_reference, keys) - js_divergence(conflict, evidence_reference, keys)


def evidence_override_normalized(conflict, label_reference, evidence_reference, keys) -> float:
    to_label = math.sqrt(js_divergence(conflict, label_reference, keys))
    to_evidence = math.sqrt(js_divergence(conflict, evidence_reference, keys))
    separation = math.sqrt(js_divergence(label_reference, evidence_reference, keys))
    if separation == 0:
        raise ValueError("normalized override is undefined for identical references")
    value = (to_label - to_evidence) / separation
    if not math.isfinite(value) or value < -1 - 1e-12 or value > 1 + 1e-12:
        raise ValueError(f"invalid normalized Evidence Override: {value}")
    return min(1.0, max(-1.0, value))

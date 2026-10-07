"""
Evaluation metrics for cultural alignment audit.
Computes Jensen-Shannon divergence, country influence, evidence influence, and evidence override.
"""

import numpy as np
from typing import Dict, Mapping, Optional, Sequence


class Metrics:
    """Compute core evaluation metrics."""

    @staticmethod
    def normalize_distribution(
        distribution: Mapping[str, float],
        keys: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Validate and normalize a distribution over an explicit option set.

        The distribution must assign a finite, non-negative numeric weight to
        every option in ``keys`` and no others. When ``keys`` is omitted, the
        distribution's own keys define the option set in deterministic order.
        Zero-total distributions are invalid rather than silently converted to
        a fallback distribution.
        """
        if not isinstance(distribution, Mapping):
            raise TypeError("distribution must be a mapping of option keys to weights")

        distribution_keys = set(distribution.keys())
        if not distribution_keys:
            raise ValueError("distribution must contain at least one option")

        if keys is None:
            try:
                ordered_keys = tuple(sorted(distribution_keys))
            except TypeError as exc:
                raise ValueError("distribution option keys must be mutually sortable") from exc
        else:
            ordered_keys = tuple(keys)
            if not ordered_keys:
                raise ValueError("keys must contain at least one option")
            if len(set(ordered_keys)) != len(ordered_keys):
                raise ValueError("keys must not contain duplicates")
            if distribution_keys != set(ordered_keys):
                missing = set(ordered_keys) - distribution_keys
                extra = distribution_keys - set(ordered_keys)
                raise ValueError(
                    "distribution keys do not match the canonical option set "
                    f"(missing={sorted(missing)!r}, extra={sorted(extra)!r})"
                )

        try:
            values = np.asarray(
                [distribution[key] for key in ordered_keys],
                dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("distribution weights must be numeric") from exc

        if not np.all(np.isfinite(values)):
            raise ValueError("distribution weights must be finite")
        if np.any(values < 0.0):
            raise ValueError("distribution weights must be non-negative")

        total = float(np.sum(values, dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("distribution must have a finite, positive total weight")

        normalized = values / total
        if not np.all(np.isfinite(normalized)):
            raise ValueError("normalization produced non-finite probabilities")
        return normalized

    @staticmethod
    def js_divergence(
        p: Mapping[str, float],
        q: Mapping[str, float],
        keys: Optional[Sequence[str]] = None
    ) -> float:
        """
        Compute base-2 Jensen-Shannon divergence between two distributions.

        Args:
            p: First probability distribution
            q: Second probability distribution
            keys: Canonical option-key order. If omitted, ``p`` and ``q`` must
                have the same key set.

        Returns:
            JS divergence in bits, bounded to [0, 1].
        """
        if not isinstance(p, Mapping) or not isinstance(q, Mapping):
            raise TypeError("p and q must be mappings of option keys to weights")

        if keys is None:
            if set(p.keys()) != set(q.keys()):
                raise ValueError("p and q must have the same option keys")
            try:
                ordered_keys = tuple(sorted(p.keys()))
            except TypeError as exc:
                raise ValueError("distribution option keys must be mutually sortable") from exc
        else:
            ordered_keys = tuple(keys)

        p_arr = Metrics.normalize_distribution(p, ordered_keys)
        q_arr = Metrics.normalize_distribution(q, ordered_keys)

        # Compute the divergence directly instead of squaring SciPy's distance.
        # The latter can take sqrt of a tiny negative round-off value for nearly
        # identical distributions and yield NaN.
        midpoint = 0.5 * (p_arr + q_arr)
        p_mask = p_arr > 0.0
        q_mask = q_arr > 0.0
        kl_p = float(np.sum(p_arr[p_mask] * np.log2(p_arr[p_mask] / midpoint[p_mask])))
        kl_q = float(np.sum(q_arr[q_mask] * np.log2(q_arr[q_mask] / midpoint[q_mask])))
        divergence = 0.5 * (kl_p + kl_q)
        if not np.isfinite(divergence):
            raise ValueError("Jensen-Shannon divergence is non-finite")

        tolerance = 1e-12
        if divergence < -tolerance or divergence > 1.0 + tolerance:
            raise ValueError(
                f"Jensen-Shannon divergence is outside [0, 1]: {divergence}"
            )
        return min(1.0, max(0.0, divergence))

    @staticmethod
    def country_influence(
        pred_baseline: Dict[str, float],
        pred_label: Dict[str, float],
        human_dist: Dict[str, float]
    ) -> float:
        """
        Country influence score.
        Positive = country label improves alignment with human distribution.

        Score = JS(baseline, human) - JS(label, human)
        """
        base_js = Metrics.js_divergence(pred_baseline, human_dist)
        label_js = Metrics.js_divergence(pred_label, human_dist)
        return base_js - label_js

    @staticmethod
    def evidence_influence(
        pred_baseline: Dict[str, float],
        pred_evidence: Dict[str, float],
        human_dist: Dict[str, float]
    ) -> float:
        """
        Evidence influence score.
        Positive = evidence improves alignment with human distribution.

        Score = JS(baseline, human) - JS(evidence, human)
        """
        base_js = Metrics.js_divergence(pred_baseline, human_dist)
        evidence_js = Metrics.js_divergence(pred_evidence, human_dist)
        return base_js - evidence_js

    @staticmethod
    def evidence_override_raw(
        pred_conflict: Dict[str, float],
        evidence_dist: Dict[str, float],
        label_dist: Dict[str, float]
    ) -> float:
        """
        Raw Evidence Override score.
        Positive = prediction is closer to evidence distribution.
        Negative = prediction is closer to country label distribution.

        Score = JS(pred, label) - JS(pred, evidence)
        """
        to_evidence = Metrics.js_divergence(pred_conflict, evidence_dist)
        to_label = Metrics.js_divergence(pred_conflict, label_dist)
        return to_label - to_evidence

    @staticmethod
    def evidence_override_normalized(
        pred_conflict: Dict[str, float],
        evidence_dist: Dict[str, float],
        label_dist: Dict[str, float]
    ) -> float:
        """Normalized Evidence Override on the Jensen-Shannon metric scale.

        The numerator is the difference between the base-2 Jensen-Shannon
        *distances* induced by the divergences, and the denominator is the
        distance between the two human reference distributions. Consequently,
        an exact evidence prediction is +1 and an exact label prediction is -1.
        """
        conflict_to_label = Metrics.js_divergence(pred_conflict, label_dist)
        conflict_to_evidence = Metrics.js_divergence(pred_conflict, evidence_dist)
        label_to_evidence = Metrics.js_divergence(label_dist, evidence_dist)
        denominator = float(np.sqrt(label_to_evidence))
        if denominator <= 0.0:
            raise ValueError(
                "EO_normalized is undefined when label and evidence distributions are identical"
            )
        return float(
            (np.sqrt(conflict_to_label) - np.sqrt(conflict_to_evidence))
            / denominator
        )

    @staticmethod
    def evidence_override(
        pred_conflict: Dict[str, float],
        evidence_dist: Dict[str, float],
        label_dist: Dict[str, float]
    ) -> float:
        """Backward-compatible alias for :meth:`evidence_override_raw`."""
        return Metrics.evidence_override_raw(
            pred_conflict, evidence_dist, label_dist
        )

    @staticmethod
    def compute_all_metrics(
        pred_baseline: Dict,
        pred_label: Dict,
        pred_evidence: Dict,
        pred_conflict: Dict,
        human_dist: Dict,
        evidence_dist: Dict,
        label_dist: Dict
    ) -> Dict[str, float]:
        """Compute the core metrics at once."""
        eo_raw = Metrics.evidence_override_raw(
            pred_conflict, evidence_dist, label_dist
        )
        return {
            'country_influence': Metrics.country_influence(
                pred_baseline, pred_label, human_dist
            ),
            'evidence_influence': Metrics.evidence_influence(
                pred_baseline, pred_evidence, human_dist
            ),
            'EO_raw': eo_raw,
            'EO_normalized': Metrics.evidence_override_normalized(
                pred_conflict, evidence_dist, label_dist
            ),
            # Retained for callers that have not yet migrated their field name.
            'evidence_override': eo_raw,
        }

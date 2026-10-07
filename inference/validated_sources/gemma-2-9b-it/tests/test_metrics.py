import unittest

from src.metrics import Metrics


class DistributionAndMetricTests(unittest.TestCase):
    def test_probability_normalization(self):
        normalized = Metrics.normalize_distribution({"A": 2.0, "B": 3.0})

        self.assertAlmostEqual(float(normalized[0]), 0.4, places=12)
        self.assertAlmostEqual(float(normalized[1]), 0.6, places=12)
        self.assertAlmostEqual(float(normalized.sum()), 1.0, places=12)

    def test_probability_normalization_rejects_invalid_values(self):
        invalid_distributions = (
            {},
            {"A": 0.0, "B": 0.0},
            {"A": -0.1, "B": 1.1},
            {"A": float("nan"), "B": 1.0},
            {"A": float("inf"), "B": 1.0},
        )
        for distribution in invalid_distributions:
            with self.subTest(distribution=distribution):
                with self.assertRaises(ValueError):
                    Metrics.normalize_distribution(distribution)

    def test_js_divergence_uses_base_two(self):
        left = {"A": 1.0, "B": 0.0}
        right = {"A": 0.0, "B": 1.0}

        self.assertAlmostEqual(Metrics.js_divergence(left, right), 1.0, places=12)

    def test_js_divergence_rejects_incompatible_option_keys(self):
        labels = {"A": 0.5, "B": 0.5}
        answer_texts = {"Yes": 0.5, "No": 0.5}

        with self.assertRaises(ValueError):
            Metrics.js_divergence(labels, answer_texts)

    def test_evidence_override_sign(self):
        evidence = {"A": 0.9, "B": 0.1}
        country_label = {"A": 0.1, "B": 0.9}
        midpoint = {"A": 0.5, "B": 0.5}

        follows_evidence = Metrics.evidence_override(
            evidence,
            evidence,
            country_label,
        )
        follows_label = Metrics.evidence_override(
            country_label,
            evidence,
            country_label,
        )
        balanced = Metrics.evidence_override(
            midpoint,
            evidence,
            country_label,
        )

        self.assertGreater(follows_evidence, 0.0)
        self.assertLess(follows_label, 0.0)
        self.assertAlmostEqual(balanced, 0.0, places=12)

    def test_evidence_override_explicit_metrics_and_normalized_endpoints(self):
        label = {"A": 0.8, "B": 0.2}
        evidence = {"A": 0.1, "B": 0.9}

        self.assertGreater(
            Metrics.evidence_override_raw(evidence, evidence, label), 0.0
        )
        self.assertLess(
            Metrics.evidence_override_raw(label, evidence, label), 0.0
        )
        self.assertAlmostEqual(
            Metrics.evidence_override_normalized(evidence, evidence, label),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            Metrics.evidence_override_normalized(label, evidence, label),
            -1.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()

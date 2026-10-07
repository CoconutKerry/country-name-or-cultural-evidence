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

    def test_evidence_override_raw_sign(self):
        evidence = {"A": 0.9, "B": 0.1}
        country_label = {"A": 0.1, "B": 0.9}
        midpoint = {"A": 0.5, "B": 0.5}

        follows_evidence = Metrics.evidence_override_raw(
            evidence,
            evidence,
            country_label,
        )
        follows_label = Metrics.evidence_override_raw(
            country_label,
            evidence,
            country_label,
        )
        balanced = Metrics.evidence_override_raw(
            midpoint,
            evidence,
            country_label,
        )

        self.assertGreater(follows_evidence, 0.0)
        self.assertLess(follows_label, 0.0)
        self.assertAlmostEqual(balanced, 0.0, places=12)

    def test_evidence_override_normalized_exact_reference_endpoints(self):
        evidence = {"A": 0.9, "B": 0.1}
        country_label = {"A": 0.1, "B": 0.9}

        follows_evidence = Metrics.evidence_override_normalized(
            evidence,
            evidence,
            country_label,
        )
        follows_label = Metrics.evidence_override_normalized(
            country_label,
            evidence,
            country_label,
        )

        self.assertAlmostEqual(follows_evidence, 1.0, places=12)
        self.assertAlmostEqual(follows_label, -1.0, places=12)

    def test_evidence_override_normalized_rejects_identical_references(self):
        reference = {"A": 0.4, "B": 0.6}

        with self.assertRaisesRegex(ValueError, "undefined"):
            Metrics.evidence_override_normalized(
                {"A": 0.5, "B": 0.5},
                reference,
                reference,
            )

    def test_evidence_override_compatibility_alias_is_raw_metric(self):
        prediction = {"A": 0.7, "B": 0.3}
        evidence = {"A": 0.9, "B": 0.1}
        country_label = {"A": 0.1, "B": 0.9}

        self.assertEqual(
            Metrics.evidence_override(prediction, evidence, country_label),
            Metrics.evidence_override_raw(prediction, evidence, country_label),
        )

    def test_compute_all_metrics_uses_explicit_evidence_override_keys(self):
        label = {"A": 0.1, "B": 0.9}
        evidence = {"A": 0.9, "B": 0.1}
        metrics = Metrics.compute_all_metrics(
            pred_baseline={"A": 0.5, "B": 0.5},
            pred_label={"A": 0.2, "B": 0.8},
            pred_evidence={"A": 0.3, "B": 0.7},
            pred_conflict=evidence,
            human_dist=label,
            evidence_dist=evidence,
            label_dist=label,
        )

        self.assertEqual(
            set(metrics),
            {"country_influence", "evidence_influence", "EO_raw", "EO_normalized"},
        )
        self.assertAlmostEqual(metrics["EO_normalized"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()

import inspect
from statistics import stdev
import unittest
from unittest.mock import patch

import numpy as np

from scripts.bootstrap_analysis import analyze_with_clustered_bootstrap
from src.clustered_bootstrap import (
    build_question_metric_clusters,
    clustered_bootstrap,
    pooled_cluster_means,
)


def make_analysis():
    models = ("model-a", "model-b")
    target_rows = []
    directed_rows = []
    # Q1 has two target countries and a reciprocal directed pair.  Q2 has one
    # reciprocal pair too, with deliberately different values.  This makes it
    # possible to detect accidental row-level instead of cluster-level draws.
    target_values = {
        "Q1": (("A", 1.0, 10.0), ("B", 3.0, 30.0)),
        "Q2": (
            ("C", 9.0, 90.0),
            ("D", 11.0, 110.0),
            ("E", 13.0, 130.0),
            ("F", 15.0, 150.0),
        ),
    }
    directed_values = {
        "Q1": (("A", "B", 100.0, 1.0), ("B", "A", 300.0, 3.0)),
        "Q2": (
            ("C", "D", 900.0, 9.0),
            ("D", "C", 1100.0, 11.0),
            ("E", "F", 1300.0, 13.0),
            ("F", "E", 1500.0, 15.0),
        ),
    }
    for model_index, model in enumerate(models, start=1):
        scale = float(model_index)
        for question_id, rows in target_values.items():
            for country, country_influence, evidence_influence in rows:
                target_rows.append(
                    {
                        "model_name": model,
                        "question_id": question_id,
                        "country": country,
                        "country_influence": scale * country_influence,
                        "evidence_influence": scale * evidence_influence,
                    }
                )
        for question_id, rows in directed_values.items():
            for country, conflict_country, EO_raw, EO_normalized in rows:
                directed_rows.append(
                    {
                        "model_name": model,
                        "question_id": question_id,
                        "country": country,
                        "conflict_country": conflict_country,
                        "EO_raw": scale * EO_raw,
                        "EO_normalized": scale * EO_normalized,
                    }
                )
    return {
        "models": {model: {} for model in models},
        "per_target_unit": target_rows,
        "per_directed_unit": directed_rows,
    }


class ClusteredBootstrapTests(unittest.TestCase):
    def test_clusters_retain_all_models_targets_and_reciprocal_directions(self):
        clusters = build_question_metric_clusters(make_analysis())

        self.assertEqual(set(clusters), {"model-a", "model-b"})
        for model in clusters:
            self.assertEqual(set(clusters[model]), {"Q1", "Q2"})
            self.assertEqual(len(clusters[model]["Q1"]["country_influence"]), 2)
            self.assertEqual(len(clusters[model]["Q1"]["evidence_influence"]), 2)
            self.assertEqual(len(clusters[model]["Q1"]["EO_raw"]), 2)
            self.assertEqual(len(clusters[model]["Q1"]["EO_normalized"]), 2)

    def test_repeated_question_duplicates_the_entire_cluster(self):
        clusters = build_question_metric_clusters(make_analysis())

        estimates = pooled_cluster_means(clusters, ["Q1", "Q1", "Q2"])

        # Both Q1 target rows and both reciprocal directed rows are duplicated
        # together alongside Q2.  De-duplicating the sampled IDs would produce
        # 52/6 instead of 7, so this also verifies cluster multiplicity.
        self.assertEqual(estimates["model-a"]["country_influence"], 7.0)
        self.assertEqual(estimates["model-a"]["evidence_influence"], 70.0)
        self.assertEqual(estimates["model-a"]["EO_raw"], 700.0)
        self.assertEqual(estimates["model-a"]["EO_normalized"], 7.0)
        self.assertEqual(estimates["model-b"]["country_influence"], 14.0)
        self.assertEqual(estimates["model-b"]["EO_raw"], 1400.0)
        self.assertEqual(estimates["model-b"]["EO_normalized"], 14.0)

    def test_bootstrap_defaults_and_percentile_intervals(self):
        self.assertEqual(
            inspect.signature(clustered_bootstrap).parameters[
                "n_replicates"
            ].default,
            10_000,
        )
        result = clustered_bootstrap(make_analysis(), n_replicates=200, seed=7)
        method = result["method"]

        self.assertEqual(method["cluster_key"], "question_id")
        self.assertEqual(method["n_clusters"], 2)
        self.assertEqual(method["clusters_drawn_per_replicate"], 2)
        self.assertEqual(method["n_replicates"], 200)
        self.assertEqual(method["lower_percentile"], 2.5)
        self.assertEqual(method["upper_percentile"], 97.5)
        self.assertTrue(method["shared_draw_across_models_and_metrics"])
        self.assertFalse(method["independent_condition_row_resampling"])
        self.assertEqual(
            set(result["models"]["model-a"]),
            {
                "country_influence",
                "evidence_influence",
                "EO_raw",
                "EO_normalized",
            },
        )
        self.assertEqual(
            result["models"]["model-a"]["EO_normalized"]["n_observations"],
            6,
        )

        # The full-data target mean for model-a is unit-weighted:
        # mean(1, 3, 9, 11, 13, 15) = 52/6, not the question-equal
        # mean(mean(Q1), mean(Q2)) = 7.
        country = result["models"]["model-a"]["country_influence"]
        self.assertAlmostEqual(country["estimate"], 52.0 / 6.0)
        self.assertNotAlmostEqual(country["estimate"], 7.0)
        self.assertEqual(country["n_observations"], 6)
        # With two question clusters, each two-cluster resample can only be
        # Q1/Q1, Q1/Q2, or Q2/Q2.  The seeded sample includes enough of both
        # extremes for the exact percentile endpoints to be cluster means.
        self.assertEqual(country["percentile_95_ci"]["lower"], 2.0)
        self.assertEqual(country["percentile_95_ci"]["upper"], 12.0)

    def test_models_share_the_same_cluster_draws(self):
        with patch(
            "src.clustered_bootstrap.pooled_cluster_means",
            wraps=pooled_cluster_means,
        ) as pooled:
            result = clustered_bootstrap(make_analysis(), n_replicates=300, seed=123)

        # Exactly one pooling call per random replicate (plus one point-
        # estimate call) means each draw is evaluated jointly for every model
        # and metric; there is no separate model/condition resample.
        self.assertEqual(pooled.call_count, 301)
        for call in pooled.call_args_list[:-1]:
            sampled_question_ids = tuple(call.args[1])
            self.assertEqual(len(sampled_question_ids), 2)
            self.assertTrue(set(sampled_question_ids) <= {"Q1", "Q2"})

        # Every metric in model-b is exactly 2x model-a.  Sharing one question
        # draw across models must preserve that scaling in estimates and CIs.
        for metric in (
            "country_influence",
            "evidence_influence",
            "EO_raw",
            "EO_normalized",
        ):
            left = result["models"]["model-a"][metric]
            right = result["models"]["model-b"][metric]
            self.assertAlmostEqual(right["estimate"], 2.0 * left["estimate"])
            self.assertAlmostEqual(
                right["percentile_95_ci"]["lower"],
                2.0 * left["percentile_95_ci"]["lower"],
            )
            self.assertAlmostEqual(
                right["percentile_95_ci"]["upper"],
                2.0 * left["percentile_95_ci"]["upper"],
            )

    def test_rejects_mismatched_question_or_metric_coverage(self):
        analysis = make_analysis()
        analysis["per_directed_unit"] = [
            row
            for row in analysis["per_directed_unit"]
            if not (row["model_name"] == "model-b" and row["question_id"] == "Q2")
        ]

        with self.assertRaisesRegex(
            ValueError,
            "no values|coverage differs|not reciprocal|no directed-unit endpoint",
        ):
            build_question_metric_clusters(analysis)

    def test_rejects_directed_endpoint_without_target_metric_row(self):
        analysis = make_analysis()
        analysis["per_target_unit"] = [
            row
            for row in analysis["per_target_unit"]
            if not (row["model_name"] == "model-a" and row["country"] == "F")
        ]

        with self.assertRaisesRegex(ValueError, "endpoints lack target metric rows"):
            build_question_metric_clusters(analysis)

    def test_rejects_target_metric_row_without_directed_endpoint(self):
        analysis = make_analysis()
        analysis["per_target_unit"].append(
            {
                "model_name": "model-a",
                "question_id": "Q1",
                "country": "orphan",
                "country_influence": 0.0,
                "evidence_influence": 0.0,
            }
        )

        with self.assertRaisesRegex(ValueError, "no directed-unit endpoint"):
            build_question_metric_clusters(analysis)

    def test_non_95_interval_is_not_mislabeled(self):
        result = clustered_bootstrap(
            make_analysis(), n_replicates=100, confidence_level=0.90, seed=2
        )

        metric = result["models"]["model-a"]["country_influence"]
        self.assertEqual(metric["percentile_ci"]["confidence_level"], 0.90)
        self.assertLessEqual(
            metric["percentile_ci"]["lower"], metric["estimate"]
        )
        self.assertGreaterEqual(
            metric["percentile_ci"]["upper"], metric["estimate"]
        )
        self.assertNotIn("percentile_95_ci", metric)

    def test_percentile_arithmetic_uses_linear_quantiles(self):
        class FixedRng:
            def __init__(self):
                self.draws = iter(
                    (
                        np.asarray([0, 0]),
                        np.asarray([0, 1]),
                        np.asarray([1, 0]),
                        np.asarray([1, 1]),
                    )
                )

            def integers(self, low, high, size):
                self_test.assertEqual((low, high, size), (0, 2, 2))
                return next(self.draws)

        self_test = self
        with patch(
            "src.clustered_bootstrap.np.random.default_rng", return_value=FixedRng()
        ):
            result = clustered_bootstrap(make_analysis(), n_replicates=4, seed=99)

        # The fixed whole-question draws produce [2, 26/3, 26/3, 12].
        replicates = [2.0, 26.0 / 3.0, 26.0 / 3.0, 12.0]
        metric = result["models"]["model-a"]["country_influence"]
        self.assertAlmostEqual(metric["bootstrap_mean"], sum(replicates) / 4.0)
        self.assertAlmostEqual(metric["bootstrap_standard_error"], stdev(replicates))
        self.assertAlmostEqual(metric["percentile_95_ci"]["lower"], 2.5)
        self.assertAlmostEqual(metric["percentile_95_ci"]["upper"], 11.75)

    def test_parameter_validation(self):
        analysis = make_analysis()
        with self.assertRaises(ValueError):
            clustered_bootstrap(analysis, n_replicates=0)
        with self.assertRaises(ValueError):
            clustered_bootstrap(analysis, confidence_level=1.0)
        with self.assertRaises(TypeError):
            clustered_bootstrap(analysis, seed=True)

    def test_seed_is_reproducible(self):
        first = clustered_bootstrap(make_analysis(), n_replicates=37, seed=314)
        second = clustered_bootstrap(make_analysis(), n_replicates=37, seed=314)

        self.assertEqual(first, second)

    def test_single_question_cluster_has_degenerate_interval(self):
        analysis = make_analysis()
        analysis["per_target_unit"] = [
            row for row in analysis["per_target_unit"] if row["question_id"] == "Q1"
        ]
        analysis["per_directed_unit"] = [
            row
            for row in analysis["per_directed_unit"]
            if row["question_id"] == "Q1"
        ]

        result = clustered_bootstrap(analysis, n_replicates=25, seed=5)
        for metrics in result["models"].values():
            for metric in metrics.values():
                self.assertEqual(
                    metric["percentile_95_ci"]["lower"], metric["estimate"]
                )
                self.assertEqual(
                    metric["percentile_95_ci"]["upper"], metric["estimate"]
                )

    def test_nonfinite_metric_is_rejected(self):
        analysis = make_analysis()
        analysis["per_target_unit"][0]["country_influence"] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite number"):
            build_question_metric_clusters(analysis)


class BootstrapScriptIntegrationTests(unittest.TestCase):
    def test_strict_analysis_precedes_bootstrap_and_requires_all_conditions(self):
        # Reuse the repository's canonical valid synthetic result fixture.  It
        # contains four conditions for every reciprocal directed unit.
        from tests.test_analyze_results import make_payload

        payload = make_payload()
        output = analyze_with_clustered_bootstrap(
            payload,
            allow_synthetic=True,
            n_replicates=20,
            seed=11,
        )
        self.assertEqual(output["clustered_bootstrap"]["method"]["n_clusters"], 1)
        self.assertEqual(output["clustered_bootstrap"]["method"]["n_replicates"], 20)

        payload["results"] = [
            row
            for row in payload["results"]
            if not (
                row["country"] == "A"
                and row["conflict_country"] == "B"
                and row["condition"] == "conflict"
            )
        ]
        with self.assertRaisesRegex(ValueError, "Incomplete or extra conditions"):
            analyze_with_clustered_bootstrap(
                payload,
                allow_synthetic=True,
                n_replicates=20,
                seed=11,
            )


if __name__ == "__main__":
    unittest.main()

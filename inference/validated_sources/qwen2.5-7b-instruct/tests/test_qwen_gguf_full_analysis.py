import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_qwen_gguf_full import (
    AnalysisExpectations,
    CONDITIONS,
    EXPECTED_BACKEND,
    EXPECTED_MODEL_IDENTIFIER,
    EXPECTED_MODEL_REPOSITORY,
    EXPECTED_MODEL_REVISION,
    EXPECTED_SCHEMA_VERSION,
    analyze_full_run,
    classify_conflict,
    write_outputs,
)
from src.main import prepare_evidence_distribution, reorder_distribution
from src.metrics import Metrics


MODEL_NAME = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
OPTIONS = ["Same English start one", "Same English start two"]


def _human(country):
    return {
        # Deliberately requires 0.1-percentage-point prompt rounding.
        "A": {OPTIONS[0]: 0.8625, OPTIONS[1]: 0.1375},
        "B": {OPTIONS[0]: 0.2, OPTIONS[1]: 0.8},
        "C": {OPTIONS[0]: 0.1, OPTIONS[1]: 0.9},
    }[country]


def _target_predictions(country, *, drift=0.0):
    if country == "A":
        return {
            "baseline": {OPTIONS[0]: 0.5 + drift, OPTIONS[1]: 0.5 - drift},
            "country_label": {OPTIONS[0]: 0.7, OPTIONS[1]: 0.3},
            "evidence": {OPTIONS[0]: 0.75, OPTIONS[1]: 0.25},
        }
    if country == "B":
        return {
            "baseline": {OPTIONS[0]: 0.5, OPTIONS[1]: 0.5},
            "country_label": {OPTIONS[0]: 0.3, OPTIONS[1]: 0.7},
            "evidence": {OPTIONS[0]: 0.25, OPTIONS[1]: 0.75},
        }
    return {
        "baseline": {OPTIONS[0]: 0.45, OPTIONS[1]: 0.55},
        "country_label": {OPTIONS[0]: 0.2, OPTIONS[1]: 0.8},
        "evidence": {OPTIONS[0]: 0.15, OPTIONS[1]: 0.85},
    }


def _pair_rows(label_country, evidence_country, *, drift=0.0, conflict_side="evidence"):
    label = _human(label_country)
    evidence = _human(evidence_country)
    predictions = _target_predictions(label_country, drift=drift)
    predictions["conflict"] = dict(
        evidence if conflict_side == "evidence" else label
    )

    divergences = {
        "baseline_to_label_country": Metrics.js_divergence(predictions["baseline"], label),
        "country_label_to_label_country": Metrics.js_divergence(
            predictions["country_label"], label
        ),
        "evidence_to_label_country": Metrics.js_divergence(predictions["evidence"], label),
        "conflict_to_label_country": Metrics.js_divergence(predictions["conflict"], label),
        "conflict_to_evidence_country": Metrics.js_divergence(
            predictions["conflict"], evidence
        ),
    }
    metrics = {
        "country_influence": (
            divergences["baseline_to_label_country"]
            - divergences["country_label_to_label_country"]
        ),
        "evidence_influence": (
            divergences["baseline_to_label_country"]
            - divergences["evidence_to_label_country"]
        ),
        "EO_raw": (
            divergences["conflict_to_label_country"]
            - divergences["conflict_to_evidence_country"]
        ),
        "EO_normalized": Metrics.evidence_override_normalized(
            predictions["conflict"], evidence, label
        ),
    }
    classification = classify_conflict(metrics["EO_raw"])
    rows = []
    for condition in CONDITIONS:
        row_divergences = {
            **divergences,
            "prediction_to_label_country": Metrics.js_divergence(
                predictions[condition], label
            ),
            "prediction_to_evidence_country": Metrics.js_divergence(
                predictions[condition], evidence
            ),
        }
        evidence_presented = condition in {"evidence", "conflict"}
        source = (label if condition == "evidence" else evidence) if evidence_presented else None
        if source is None:
            presented = None
        else:
            displayed_source = reorder_distribution(source, list(reversed(OPTIONS)))
            presented = prepare_evidence_distribution(
                displayed_source, list(reversed(OPTIONS))
            )
        rows.append(
            {
                "schema_version": EXPECTED_SCHEMA_VERSION,
                "model_name": MODEL_NAME,
                "model_identifier": EXPECTED_MODEL_IDENTIFIER,
                "model_repository": EXPECTED_MODEL_REPOSITORY,
                "model_revision": EXPECTED_MODEL_REVISION,
                "backend": EXPECTED_BACKEND,
                "synthetic": False,
                "generated_answer": None,
                "scoring_method": "full_contextual_option_label_sequence_log_probability",
                "jensen_shannon_base": 2,
                "jensen_shannon_measure": "divergence_bits",
                "quantization": {"type": "Q4_K_M", "format": "GGUF", "gpu_layers": 0},
                "question_id": "Q1",
                "label_country": label_country,
                "evidence_country": evidence_country,
                "condition": condition,
                "original_options": list(OPTIONS),
                "displayed_option_order": list(reversed(OPTIONS)),
                "displayed_options": list(reversed(OPTIONS)),
                "displayed_option_labels": ["A", "B"],
                "displayed_label_to_option": {"A": OPTIONS[1], "B": OPTIONS[0]},
                "restored_original_option_order": list(OPTIONS),
                "normalized_prediction": dict(predictions[condition]),
                "label_country_human_distribution": dict(label),
                "evidence_country_human_distribution": dict(evidence),
                "evidence_presented": evidence_presented,
                "presented_evidence_distribution": presented,
                "source_evidence_distribution": dict(source) if source is not None else None,
                "base2_jensen_shannon_divergences": row_divergences,
                **metrics,
                "conflict_classification": classification if condition == "conflict" else None,
            }
        )
    return rows


def _manifest_pair(label_country, evidence_country):
    return {
        "unit_id": f"Q1::{label_country}=>{evidence_country}",
        "question_id": "Q1",
        "country": label_country,
        "conflict_country": evidence_country,
        "options": list(OPTIONS),
        "mapping_status": "exact_shared_response_schema",
    }


def reciprocal_fixture():
    manifest = [_manifest_pair("A", "B"), _manifest_pair("B", "A")]
    rows = _pair_rows("A", "B", conflict_side="evidence")
    rows += _pair_rows("B", "A", conflict_side="label")
    expectations = AnalysisExpectations(
        result_rows=8, directed_units=2, target_units=2, question_ids=1
    )
    return manifest, rows, expectations


class QwenGgufFullAnalysisTests(unittest.TestCase):
    def test_recomputes_metrics_dedupes_estimands_and_classifies_conflicts(self):
        manifest, rows, expectations = reciprocal_fixture()

        analysis, enriched = analyze_full_run(
            rows,
            manifest,
            expectations=expectations,
            bootstrap_replicates=25,
            bootstrap_seed=7,
        )

        self.assertEqual(analysis["validation_status"], "PASS")
        self.assertEqual(analysis["counts"]["result_rows"], 8)
        self.assertEqual(len(enriched), 8)
        self.assertEqual(len(analysis["per_target_unit"]), 2)
        self.assertEqual(len(analysis["per_directed_unit"]), 2)
        self.assertEqual(analysis["metric_summaries"]["country_influence"]["n"], 2)
        self.assertEqual(analysis["metric_summaries"]["EO_raw"]["n"], 2)
        self.assertEqual(
            analysis["conflict_classifications"]["counts"],
            {"evidence-side": 1, "label-side": 1, "tie": 0},
        )
        directed = analysis["per_directed_unit"]
        self.assertAlmostEqual(directed[0]["EO_normalized"], 1.0, places=12)
        self.assertAlmostEqual(directed[1]["EO_normalized"], -1.0, places=12)
        self.assertEqual(analysis["bootstrap"]["method"]["n_clusters"], 1)
        self.assertEqual(analysis["bootstrap"]["method"]["n_replicates"], 25)
        self.assertTrue(
            analysis["bootstrap"]["method"]["shared_draw_across_models_and_metrics"]
        )
        self.assertFalse(
            analysis["bootstrap"]["method"]["independent_condition_row_resampling"]
        )
        self.assertTrue(
            all("base2_jensen_shannon_distances" not in row for row in enriched)
        )

    def test_evidence_fields_are_null_exactly_when_not_presented(self):
        manifest, rows, expectations = reciprocal_fixture()
        for row in rows:
            if row["condition"] in {"baseline", "country_label"}:
                self.assertIsNone(row["presented_evidence_distribution"])
                self.assertIsNone(row["source_evidence_distribution"])

        rows[0]["source_evidence_distribution"] = dict(_human("A"))
        with self.assertRaisesRegex(ValueError, "must be null"):
            analyze_full_run(
                rows,
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

        manifest, rows, expectations = reciprocal_fixture()
        del rows[0]["source_evidence_distribution"]
        with self.assertRaisesRegex(ValueError, "must be explicitly present"):
            analyze_full_run(
                rows,
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

    def test_rejects_tampered_derived_metric_or_divergence(self):
        manifest, rows, expectations = reciprocal_fixture()
        rows[0]["EO_raw"] += 0.01
        with self.assertRaisesRegex(ValueError, "does not match recomputed"):
            analyze_full_run(
                rows,
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

        manifest, rows, expectations = reciprocal_fixture()
        rows[0]["base2_jensen_shannon_divergences"][
            "prediction_to_label_country"
        ] += 0.01
        with self.assertRaisesRegex(ValueError, "does not match recomputation"):
            analyze_full_run(
                rows,
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

    def test_rejects_missing_row_duplicate_condition_and_bad_permutation(self):
        manifest, rows, expectations = reciprocal_fixture()
        with self.assertRaisesRegex(ValueError, "expected exactly 8"):
            analyze_full_run(
                rows[:-1],
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

        manifest, rows, expectations = reciprocal_fixture()
        rows[-1]["condition"] = "baseline"
        with self.assertRaisesRegex(ValueError, "duplicate result row"):
            analyze_full_run(
                rows,
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

        manifest, rows, expectations = reciprocal_fixture()
        rows[0]["displayed_label_to_option"]["A"] = OPTIONS[0]
        with self.assertRaisesRegex(ValueError, "displayed_label_to_option"):
            analyze_full_run(
                rows,
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

    def test_repeated_target_is_deduplicated_only_after_identity_check(self):
        manifest = [
            _manifest_pair("A", "B"),
            _manifest_pair("B", "A"),
            _manifest_pair("A", "C"),
            _manifest_pair("C", "A"),
        ]
        rows = _pair_rows("A", "B") + _pair_rows("B", "A")
        rows += _pair_rows("A", "C") + _pair_rows("C", "A")
        expectations = AnalysisExpectations(16, 4, 3, 1)

        analysis, _ = analyze_full_run(
            rows,
            manifest,
            expectations=expectations,
            bootstrap_replicates=3,
        )
        self.assertEqual(len(analysis["per_target_unit"]), 3)
        target_a = next(
            row for row in analysis["per_target_unit"] if row["label_country"] == "A"
        )
        self.assertEqual(target_a["directed_repetitions"], 2)
        self.assertEqual(target_a["evidence_countries"], ["B", "C"])

        drifted_rows = _pair_rows("A", "B") + _pair_rows("B", "A")
        drifted_rows += _pair_rows("A", "C", drift=0.01) + _pair_rows("C", "A")
        with self.assertRaisesRegex(ValueError, "repeated target baseline prediction drifted"):
            analyze_full_run(
                drifted_rows,
                manifest,
                expectations=expectations,
                bootstrap_replicates=2,
            )

    def test_conflict_classification_tolerance_and_orientation(self):
        self.assertEqual(classify_conflict(0.5), "evidence-side")
        self.assertEqual(classify_conflict(-0.5), "label-side")
        self.assertEqual(classify_conflict(1e-12), "tie")
        self.assertEqual(classify_conflict(-1e-12), "tie")
        self.assertEqual(classify_conflict(1.0001e-12), "evidence-side")

    def test_writes_all_requested_output_formats(self):
        manifest, rows, expectations = reciprocal_fixture()
        analysis, enriched = analyze_full_run(
            rows,
            manifest,
            expectations=expectations,
            bootstrap_replicates=5,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = write_outputs(output_dir, analysis, enriched)
            self.assertEqual(len(paths), 10)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertIn(r"EO\_normalized", (output_dir / "tables/model_metrics.tex").read_text())
            self.assertIn("evidence-side", (output_dir / "EXPERIMENT_REPORT.md").read_text())
            metric_tables = json.loads((output_dir / "metric_tables.json").read_text())
            self.assertEqual(len(metric_tables["model_metrics"]), 4)
            with (output_dir / "directed_unit_metrics.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with (output_dir / "target_unit_metrics.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()

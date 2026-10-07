from copy import deepcopy
import hashlib
import math
import unittest

from scripts.analyze_results import analyze_payload


CONDITIONS = ("baseline", "country_label", "population_evidence", "conflict")
OPTIONS = ["Yes", "No"]


def make_rows(country, conflict_country, target, conflict_source):
    rows = []
    predictions = {
        "baseline": {"Yes": 0.5, "No": 0.5},
        "country_label": {"Yes": 0.7, "No": 0.3},
        "population_evidence": {"Yes": 0.75, "No": 0.25},
        "conflict": {"Yes": 0.3, "No": 0.7},
    }
    for condition in CONDITIONS:
        prediction = predictions[condition]
        evidence_presented = condition in {"population_evidence", "conflict"}
        source = (
            conflict_source
            if condition == "conflict"
            else target if condition == "population_evidence" else None
        )
        prompt = f"{country} target prompt: {condition}"
        rows.append(
            {
                "model_name": "synthetic",
                "unit_id": f"Q1::{country}=>{conflict_country}",
                "target_unit_id": f"Q1::{country}",
                "question_id": "Q1",
                "country": country,
                "conflict_country": conflict_country,
                "condition": condition,
                "prediction": prediction,
                "human_distribution": target,
                "label_country_human_distribution": target,
                "evidence_country_human_distribution": conflict_source,
                "evidence_distribution": source,
                "presented_evidence_distribution": source,
                "source_evidence_distribution": source,
                "evidence_presented": evidence_presented,
                "prompt": prompt,
                "original_options": OPTIONS,
                "used_options": OPTIONS,
                "option_label_map": {"A": "Yes", "B": "No"},
                "scoring": {
                    "synthetic": True,
                    "label_to_option": {"A": "Yes", "B": "No"},
                    "label_probabilities": {
                        "A": prediction["Yes"],
                        "B": prediction["No"],
                    },
                    "label_log_scores": {
                        "A": math.log(prediction["Yes"]),
                        "B": math.log(prediction["No"]),
                    },
                    "rendered_prompt": prompt,
                    "raw_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                },
            }
        )
    return rows


def make_payload():
    target_a = {"Yes": 0.8, "No": 0.2}
    target_other = {"Yes": 0.2, "No": 0.8}
    return {
        "metadata": {
            "scoring_method": "complete_option_label_continuation_log_likelihood",
            "directed_unit_key": ["question_id", "country", "conflict_country"],
            "jensen_shannon_base": 2,
            "synthetic_backend": True,
            "scientific_inference": False,
            "models": ["synthetic"],
            "conditions": list(CONDITIONS),
            "num_pairs": 4,
        },
        "results": (
            make_rows("A", "B", target_a, target_other)
            + make_rows("B", "A", target_other, target_a)
            + make_rows("A", "C", target_a, target_other)
            + make_rows("C", "A", target_other, target_a)
        ),
    }


class AnalyzeResultsTests(unittest.TestCase):
    def test_ci_ei_deduplicate_repeated_target_but_eo_remains_directed(self):
        analysis = analyze_payload(make_payload(), allow_synthetic=True)

        self.assertEqual(analysis["counts"]["unique_directed_units"], 4)
        self.assertEqual(analysis["counts"]["model_directed_unit_groups"], 4)
        self.assertEqual(analysis["counts"]["unique_target_units"], 3)
        self.assertEqual(analysis["counts"]["model_target_unit_groups"], 3)
        self.assertEqual(analysis["models"]["synthetic"]["country_influence"]["n"], 3)
        self.assertEqual(analysis["models"]["synthetic"]["evidence_influence"]["n"], 3)
        self.assertEqual(analysis["models"]["synthetic"]["EO_raw"]["n"], 4)
        self.assertEqual(analysis["models"]["synthetic"]["EO_normalized"]["n"], 4)
        self.assertNotIn("evidence_override", analysis["models"]["synthetic"])
        for row in analysis["per_directed_unit"]:
            self.assertIn("EO_raw", row)
            self.assertIn("EO_normalized", row)
            self.assertNotIn("evidence_override", row)
            self.assertLessEqual(abs(row["EO_normalized"]), 1.0)
        target_a = next(
            row
            for row in analysis["per_target_unit"]
            if row["country"] == "A"
        )
        self.assertEqual(target_a["directed_repetitions"], 2)

    def test_absent_evidence_fields_must_be_explicit_nulls(self):
        payload = make_payload()
        baseline = next(
            row for row in payload["results"] if row["condition"] == "baseline"
        )
        baseline["presented_evidence_distribution"] = baseline[
            "label_country_human_distribution"
        ]

        with self.assertRaisesRegex(
            ValueError,
            "presented_evidence_distribution must be null when evidence is absent",
        ):
            analyze_payload(payload, allow_synthetic=True)

        payload = make_payload()
        country_label = next(
            row
            for row in payload["results"]
            if row["condition"] == "country_label"
        )
        country_label["source_evidence_distribution"] = country_label[
            "label_country_human_distribution"
        ]

        with self.assertRaisesRegex(
            ValueError,
            "source_evidence_distribution must be null when evidence is absent",
        ):
            analyze_payload(payload, allow_synthetic=True)

    def test_presented_evidence_conditions_require_the_correct_source(self):
        payload = make_payload()
        population = next(
            row
            for row in payload["results"]
            if row["condition"] == "population_evidence"
        )
        population["source_evidence_distribution"] = None

        with self.assertRaisesRegex(ValueError, "source_evidence_distribution must be an object"):
            analyze_payload(payload, allow_synthetic=True)

        payload = make_payload()
        conflict = next(
            row for row in payload["results"] if row["condition"] == "conflict"
        )
        conflict["presented_evidence_distribution"] = conflict[
            "label_country_human_distribution"
        ]
        conflict["evidence_distribution"] = conflict[
            "label_country_human_distribution"
        ]

        with self.assertRaisesRegex(ValueError, "does not match its source"):
            analyze_payload(payload, allow_synthetic=True)

    def test_protected_colab_v1_reference_evidence_remains_compatible(self):
        payload = make_payload()
        for row in payload["results"]:
            row["schema_version"] = "colab-gpu-v1"
            if not row["evidence_presented"]:
                label_reference = row["label_country_human_distribution"]
                row["presented_evidence_distribution"] = label_reference
                row["source_evidence_distribution"] = label_reference
                row["evidence_distribution"] = label_reference

        analysis = analyze_payload(payload, allow_synthetic=True)

        self.assertEqual(analysis["counts"]["result_rows"], 16)
        self.assertIn("EO_raw", analysis["models"]["synthetic"])
        self.assertIn("EO_normalized", analysis["models"]["synthetic"])

    def test_repeated_target_prompt_disagreement_is_rejected(self):
        payload = make_payload()
        mutated = next(
            row
            for row in payload["results"]
            if row["country"] == "A"
            and row["conflict_country"] == "C"
            and row["condition"] == "country_label"
        )
        mutated["prompt"] = "different prompt"
        mutated["scoring"]["rendered_prompt"] = "different prompt"
        prompt_hash = hashlib.sha256(b"different prompt").hexdigest()
        mutated["scoring"]["raw_prompt_sha256"] = prompt_hash
        mutated["scoring"]["rendered_prompt_sha256"] = prompt_hash

        with self.assertRaisesRegex(ValueError, "repeated target has inconsistent"):
            analyze_payload(payload, allow_synthetic=True)

    def test_bad_unit_id_is_rejected(self):
        payload = deepcopy(make_payload())
        payload["results"][0]["unit_id"] = "wrong"

        with self.assertRaisesRegex(ValueError, "unit_id is inconsistent"):
            analyze_payload(payload, allow_synthetic=True)

    def test_nonreciprocal_manifest_is_rejected(self):
        payload = make_payload()
        payload["results"] = [
            row
            for row in payload["results"]
            if not (row["country"] == "C" and row["conflict_country"] == "A")
        ]
        payload["metadata"]["num_pairs"] = 3

        with self.assertRaisesRegex(ValueError, "not reciprocal"):
            analyze_payload(payload, allow_synthetic=True)

    def test_label_log_scores_must_match_label_probabilities(self):
        payload = make_payload()
        row = next(
            item for item in payload["results"] if item["condition"] == "country_label"
        )
        row["scoring"]["label_log_scores"] = {"A": 0.0, "B": 0.0}

        with self.assertRaisesRegex(ValueError, "do not normalize"):
            analyze_payload(payload, allow_synthetic=True)


if __name__ == "__main__":
    unittest.main()

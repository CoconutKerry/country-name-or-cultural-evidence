import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data/processed/dataset_v2.json"
MANIFEST_PATH = PROJECT_ROOT / "data/pairs/country_pairs_v2.json"
EXPECTED_MANIFEST_SHA256 = (
    "429e07b5f2019be15fd9306e4722ff3878fb575c0eb03db7f1c0d1aacbad5d8b"
)
CURRENT_SAMPLE_SUFFIX = " (Current national sample)"
INELIGIBLE_SAMPLE_MARKERS = ("(Non-national sample)", "(Old national sample)")


class QwenFullManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        cls.pairs = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_identity_is_pinned(self):
        digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_MANIFEST_SHA256)

    def test_manifest_has_exact_directed_cardinality_and_reciprocity(self):
        keys = [
            (pair["question_id"], pair["country"], pair["conflict_country"])
            for pair in self.pairs
        ]
        key_set = set(keys)

        self.assertEqual(len(keys), 200)
        self.assertEqual(len(key_set), 200)
        self.assertTrue(
            all(
                (question_id, evidence_country, label_country) in key_set
                for question_id, label_country, evidence_country in keys
            )
        )

    def test_manifest_has_expected_question_and_target_units(self):
        question_ids = {pair["question_id"] for pair in self.pairs}
        target_keys = {
            (pair["question_id"], pair["country"])
            for pair in self.pairs
        }

        self.assertEqual(len(question_ids), 44)
        self.assertEqual(len(target_keys), 144)
        self.assertEqual(
            Counter(pair["mapping_status"] for pair in self.pairs),
            {"exact_shared_response_schema": 200},
        )

    def test_every_endpoint_uses_clean_dataset_sample_and_exact_options(self):
        endpoint_statuses = Counter()
        current_endpoint_keys = set()

        for pair in self.pairs:
            question_id = pair["question_id"]
            self.assertIn(question_id, self.dataset)
            question = self.dataset[question_id]
            self.assertEqual(pair["options"], question["options"])

            for role, country in (
                ("country", pair["country"]),
                ("conflict_country", pair["conflict_country"]),
            ):
                with self.subTest(question_id=question_id, role=role, country=country):
                    self.assertIn(country, question["countries"])
                    self.assertFalse(
                        any(marker in country for marker in INELIGIBLE_SAMPLE_MARKERS)
                    )
                    sample = question["countries"][country]
                    status = sample["sample_status"]
                    self.assertIn(status, {"national", "current_national"})
                    endpoint_statuses[status] += 1

                    display = question["country_display_names"][country]
                    if status == "current_national":
                        self.assertTrue(country.endswith(CURRENT_SAMPLE_SUFFIX))
                        self.assertEqual(country[: -len(CURRENT_SAMPLE_SUFFIX)], display)
                        current_endpoint_keys.add((question_id, country))
                    else:
                        self.assertFalse(country.endswith(CURRENT_SAMPLE_SUFFIX))
                        self.assertEqual(country, display)

        self.assertEqual(endpoint_statuses, {"national": 392, "current_national": 8})
        self.assertEqual(
            current_endpoint_keys,
            {
                ("Q0059", "India (Current national sample)"),
                ("Q0268", "India (Current national sample)"),
            },
        )

    def test_entire_repaired_dataset_excludes_non_national_and_old_samples(self):
        statuses = Counter()
        for question_id, question in self.dataset.items():
            for country, sample in question["countries"].items():
                with self.subTest(question_id=question_id, country=country):
                    self.assertFalse(
                        any(marker in country for marker in INELIGIBLE_SAMPLE_MARKERS)
                    )
                    self.assertIn(
                        sample["sample_status"], {"national", "current_national"}
                    )
                    self.assertEqual(sample["source_country_label"], country)
                    statuses[sample["sample_status"]] += 1

        self.assertEqual(statuses, {"national": 18_583, "current_national": 218})


if __name__ == "__main__":
    unittest.main()

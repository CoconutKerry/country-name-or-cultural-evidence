import unittest

from scripts.repair_data import (
    COUNTRY_SPECIFIC_PLACEHOLDER_QUESTIONS,
    UNRESOLVED_RELATIVE_TIME_QUESTIONS,
    is_nonresponse,
    repair_dataset,
)


class DataRepairTests(unittest.TestCase):
    def test_substantive_none_neither_both_and_depends_are_retained(self):
        substantive = (
            "Both",
            "Both should have a say (VOL)",
            "Neither",
            "Neither agree nor disagree",
            "None at all",
            "None of these (VOL)",
            "Depends on the situation (VOL)",
            "Other (VOL)",
        )
        for option in substantive:
            with self.subTest(option=option):
                self.assertFalse(is_nonresponse(option))

    def test_explicit_missing_and_refused_labels_are_removed(self):
        nonresponses = (
            "DK/Refused",
            "Don't know",
            "No answer/refused",
            "Missing; Not available",
        )
        for option in nonresponses:
            with self.subTest(option=option):
                self.assertTrue(is_nonresponse(option))

    def test_table_total_option_quarantines_entire_question(self):
        raw = {
            "Q1": {
                "question": "Example?",
                "options": ["Yes", "No", "Total       N="],
                "countries": {
                    "A": {"distribution": {"Yes": 0.4, "No": 0.5, "Total       N=": 0.1}},
                    "B": {"distribution": {"Yes": 0.5, "No": 0.4, "Total       N=": 0.1}},
                },
            }
        }

        repaired, summary = repair_dataset(raw)

        self.assertEqual(repaired, {})
        self.assertEqual(
            summary["excluded_questions_by_reason"]["table_total_parser_artifact"],
            1,
        )

    def test_country_specific_placeholder_is_quarantined(self):
        question_id = next(iter(COUNTRY_SPECIFIC_PLACEHOLDER_QUESTIONS))
        raw = {
            question_id: {
                "question": "Influence of (relevant ethnic group-country specific)?",
                "options": ["Good", "Bad"],
                "countries": {
                    "A": {"distribution": {"Good": 0.5, "Bad": 0.5}},
                    "B": {"distribution": {"Good": 0.5, "Bad": 0.5}},
                },
            }
        }

        repaired, summary = repair_dataset(raw)

        self.assertEqual(repaired, {})
        self.assertEqual(
            summary["excluded_questions_by_reason"][
                "country_specific_placeholder_unresolved"
            ],
            1,
        )

    def test_relative_time_without_wave_is_quarantined(self):
        question_id = next(iter(UNRESOLVED_RELATIVE_TIME_QUESTIONS))
        raw = {
            question_id: {
                "question": "Did this happen recently?",
                "options": ["Yes", "No"],
                "countries": {
                    "A": {"distribution": {"Yes": 0.5, "No": 0.5}},
                    "B": {"distribution": {"Yes": 0.5, "No": 0.5}},
                },
            }
        }

        repaired, summary = repair_dataset(raw)

        self.assertEqual(repaired, {})
        self.assertEqual(
            summary["excluded_questions_by_reason"]["relative_time_without_survey_wave"],
            1,
        )

    def test_zero_mass_scale_endpoint_is_retained(self):
        raw = {
            "Q1": {
                "question": "How important?",
                "options": ["Very", "Somewhat", "Not at all"],
                "countries": {
                    "A": {
                        "distribution": {"Very": 0.6, "Somewhat": 0.4, "Not at all": 0.0}
                    },
                    "B": {
                        "distribution": {"Very": 0.4, "Somewhat": 0.6, "Not at all": 0.0}
                    },
                },
            }
        }

        repaired, _ = repair_dataset(raw)

        self.assertEqual(repaired["Q1"]["options"], ["Very", "Somewhat", "Not at all"])


if __name__ == "__main__":
    unittest.main()

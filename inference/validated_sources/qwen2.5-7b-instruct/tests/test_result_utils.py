import unittest

from src.result_utils import group_results_by_directed_unit


CONDITIONS = (
    "baseline",
    "country_label",
    "population_evidence",
    "conflict",
)


def make_unit_rows(country, conflict_country):
    return [
        {
            "model_name": "synthetic-model",
            "question_id": "Q0001",
            "country": country,
            "conflict_country": conflict_country,
            "condition": condition,
            "prediction": {"Yes": 0.5, "No": 0.5},
        }
        for condition in CONDITIONS
    ]


class DirectedResultGroupingTests(unittest.TestCase):
    def test_directed_units_remain_unique(self):
        rows = (
            make_unit_rows("Country A", "Country B")
            + make_unit_rows("Country A", "Country C")
            + make_unit_rows("Country B", "Country A")
        )

        groups = group_results_by_directed_unit(rows)

        expected_keys = {
            ("synthetic-model", "Q0001", "Country A", "Country B"),
            ("synthetic-model", "Q0001", "Country A", "Country C"),
            ("synthetic-model", "Q0001", "Country B", "Country A"),
        }
        self.assertEqual(set(groups), expected_keys)
        self.assertEqual(len(groups), 3)
        for conditions in groups.values():
            self.assertEqual(set(conditions), set(CONDITIONS))

    def test_duplicate_condition_for_same_directed_unit_is_rejected(self):
        rows = make_unit_rows("Country A", "Country B")
        rows.append(dict(rows[0]))

        with self.assertRaises(ValueError):
            group_results_by_directed_unit(rows)


if __name__ == "__main__":
    unittest.main()

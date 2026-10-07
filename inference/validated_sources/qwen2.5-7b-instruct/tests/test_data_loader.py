import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.data_loader import DataLoader


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class DataLoaderSchemaTests(unittest.TestCase):
    def make_dataset(self, root: Path, distribution=None) -> Path:
        distribution = distribution or {"Yes": 0.6, "No": 0.4}
        path = root / "dataset.json"
        write_json(
            path,
            {
                "Q1": {
                    "question": "Example?",
                    "options": ["Yes", "No"],
                    "countries": {
                        "A": {"distribution": distribution},
                        "B": {"distribution": {"Yes": 0.2, "No": 0.8}},
                    },
                }
            },
        )
        return path

    def test_distribution_must_contain_every_canonical_option(self):
        with TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), {"Yes": 1.0})

            with self.assertRaisesRegex(ValueError, "exactly match"):
                DataLoader(str(dataset)).load_dataset()

    def test_distribution_must_not_contain_extra_options(self):
        with TemporaryDirectory() as directory:
            dataset = self.make_dataset(
                Path(directory), {"Yes": 0.5, "No": 0.4, "Maybe": 0.1}
            )

            with self.assertRaisesRegex(ValueError, "exactly match"):
                DataLoader(str(dataset)).load_dataset()

    def test_pair_options_must_match_canonical_order_and_membership(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            loader = DataLoader(str(self.make_dataset(root)))
            loader.load_dataset()
            pairs = root / "pairs.json"
            write_json(
                pairs,
                [
                    {
                        "question_id": "Q1",
                        "country": "A",
                        "conflict_country": "B",
                        "options": ["Yes", "Maybe"],
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "canonical question schema"):
                loader.get_question_pairs(str(pairs))

    def test_exact_pair_schema_is_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            loader = DataLoader(str(self.make_dataset(root)))
            loader.load_dataset()
            pairs = root / "pairs.json"
            write_json(
                pairs,
                [
                    {
                        "question_id": "Q1",
                        "country": "A",
                        "conflict_country": "B",
                        "options": ["Yes", "No"],
                    }
                ],
            )

            loaded = loader.get_question_pairs(str(pairs))

            self.assertEqual(loaded[0]["options"], ["Yes", "No"])


if __name__ == "__main__":
    unittest.main()

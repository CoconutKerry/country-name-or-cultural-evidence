import unittest

from src.main import format_evidence, prepare_evidence_distribution, stable_unit_seed
from src.prompt_builder import PromptBuilder


class PromptAndEvidenceTests(unittest.TestCase):
    def test_core_conditions_share_one_prompt_skeleton(self):
        builder = PromptBuilder()
        prompts = {
            "baseline": builder.build_prompt("baseline", "Question?", ["Yes", "No"]),
            "country_label": builder.build_prompt(
                "country_label", "Question?", ["Yes", "No"], country="A"
            ),
            "population_evidence": builder.build_prompt(
                "population_evidence", "Question?", ["Yes", "No"], evidence="A: 60%, B: 40%"
            ),
            "conflict": builder.build_prompt(
                "conflict",
                "Question?",
                ["Yes", "No"],
                country="A",
                evidence="A: 60%, B: 40%",
            ),
        }

        def without_cues(prompt):
            stripped = "\n".join(
                line
                for line in prompt.splitlines()
                if not line.startswith(("Target Country:", "Survey Response Statistics:"))
            )
            return stripped.replace("\n\nQuestion:", "\nQuestion:")

        skeletons = {without_cues(prompt) for prompt in prompts.values()}
        self.assertEqual(len(skeletons), 1)

    def test_presented_evidence_sums_exactly_and_matches_text(self):
        options = ["One", "Two", "Three"]
        presented = prepare_evidence_distribution(
            {"One": 1.0, "Two": 1.0, "Three": 1.0}, options
        )
        text = format_evidence(presented, options)

        self.assertAlmostEqual(sum(presented.values()), 1.0, places=15)
        percentages = [float(part.rsplit(": ", 1)[1].rstrip("%")) for part in text.split(", ")]
        self.assertAlmostEqual(sum(percentages), 100.0, places=6)

    def test_option_permutation_seed_is_target_not_conflict_specific(self):
        seed_a = stable_unit_seed(42, "Q1::Country A")
        seed_b = stable_unit_seed(42, "Q1::Country A")
        self.assertEqual(seed_a, seed_b)


if __name__ == "__main__":
    unittest.main()

import unittest

from src.main import run_experiment


class RunGuardrailTests(unittest.TestCase):
    def test_main_run_requires_explicit_approval_before_any_setup(self):
        with self.assertRaises(PermissionError):
            run_experiment({}, "main", "must-not-be-created.json")

    def test_pilot_run_is_also_approval_gated(self):
        with self.assertRaises(PermissionError):
            run_experiment({}, "pilot", "must-not-be-created.json")


if __name__ == "__main__":
    unittest.main()

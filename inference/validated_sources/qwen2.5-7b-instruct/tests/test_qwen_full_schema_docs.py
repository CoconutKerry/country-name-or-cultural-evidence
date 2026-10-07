import json
from pathlib import Path
import unittest

from scripts.generate_qwen_gguf_schema_docs import (
    MARKDOWN_PATH,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    build_markdown,
    build_schema,
)


class QwenFullSchemaDocumentationTests(unittest.TestCase):
    def test_generated_files_are_current(self):
        self.assertEqual(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
            build_schema(),
        )
        self.assertEqual(MARKDOWN_PATH.read_text(encoding="utf-8"), build_markdown())

    def test_schema_uses_explicit_metric_and_divergence_names(self):
        schema = build_schema()
        required = set(schema["required"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertTrue(
            {
                "base2_jensen_shannon_divergences",
                "EO_raw",
                "EO_normalized",
            }.issubset(required)
        )
        self.assertNotIn("base2_jensen_shannon_distances", required)
        self.assertNotIn("evidence_override", required)

    def test_no_evidence_conditions_require_explicit_nulls(self):
        schema = build_schema()
        no_evidence_rule = schema["allOf"][0]
        self.assertEqual(
            set(no_evidence_rule["if"]["properties"]["condition"]["enum"]),
            {"baseline", "country_label"},
        )
        then = no_evidence_rule["then"]["properties"]
        self.assertEqual(then["evidence_presented"], {"const": False})
        self.assertEqual(then["presented_evidence_distribution"], {"type": "null"})
        self.assertEqual(then["source_evidence_distribution"], {"type": "null"})

    def test_documentation_records_cluster_and_endpoint_contracts(self):
        markdown = build_markdown()
        for fragment in (
            "exactly 800 rows",
            "EO_raw` is the primary metric",
            "approximately +1",
            "approximately +1 for an",
            "-1 for an exact label",
            "10,000-replicate",
            "Condition rows are never bootstrapped independently",
        ):
            self.assertIn(fragment, markdown)


if __name__ == "__main__":
    unittest.main()

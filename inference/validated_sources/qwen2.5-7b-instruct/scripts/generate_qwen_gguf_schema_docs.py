#!/usr/bin/env python3
"""Generate the Qwen GGUF full-run JSON Schema and Markdown reference."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "qwen-gguf-full-v2"
SCHEMA_PATH = PROJECT_ROOT / "docs/qwen_gguf_full_result.schema.json"
MARKDOWN_PATH = PROJECT_ROOT / "docs/RESULT_SCHEMA.md"


def distribution_schema(*, nullable: bool = False) -> dict:
    value = {
        "type": "object",
        "minProperties": 2,
        "additionalProperties": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "description": "Semantic-option probability map; finite values sum to one.",
    }
    return {"oneOf": [value, {"type": "null"}]} if nullable else value


def build_schema() -> dict:
    required = [
        "schema_version",
        "question_id",
        "label_country",
        "evidence_country",
        "condition",
        "unit_id",
        "target_unit_id",
        "model_name",
        "model_identifier",
        "model_repository",
        "model_revision",
        "backend",
        "synthetic",
        "quantization",
        "structured_messages",
        "raw_user_prompt",
        "serialized_chat_templated_prompt",
        "serialized_prompt_sha256",
        "original_options",
        "displayed_options",
        "displayed_option_labels",
        "displayed_label_to_option",
        "candidate_label_token_ids",
        "raw_candidate_scores",
        "normalized_label_probabilities",
        "normalized_prediction",
        "label_country_human_distribution",
        "evidence_country_human_distribution",
        "presented_evidence_distribution",
        "source_evidence_distribution",
        "evidence_presented",
        "scoring_method",
        "scoring_trace",
        "generated_answer",
        "jensen_shannon_base",
        "jensen_shannon_measure",
        "base2_jensen_shannon_divergences",
        "country_influence",
        "evidence_influence",
        "EO_raw",
        "EO_normalized",
    ]
    string_array = {
        "type": "array",
        "minItems": 2,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
    }
    score_map = {
        "type": "object",
        "minProperties": 2,
        "additionalProperties": {"type": "number"},
    }
    label_map = {
        "type": "object",
        "minProperties": 2,
        "additionalProperties": {"type": "string", "minLength": 1},
    }
    token_map = {
        "type": "object",
        "minProperties": 2,
        "additionalProperties": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "integer", "minimum": 0},
        },
    }
    sha256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    properties = {
        "schema_version": {"const": SCHEMA_VERSION},
        "question_id": {"type": "string", "pattern": "^Q[0-9]{4}$"},
        "label_country": {"type": "string", "minLength": 1},
        "evidence_country": {"type": "string", "minLength": 1},
        "condition": {
            "enum": ["baseline", "country_label", "evidence", "conflict"]
        },
        "unit_id": {"type": "string", "minLength": 1},
        "target_unit_id": {"type": "string", "minLength": 1},
        "model_name": {"const": "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"},
        "model_identifier": {"const": "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"},
        "model_repository": {"const": "Qwen/Qwen2.5-7B-Instruct-GGUF"},
        "model_revision": {
            "const": "bb5d59e06d9551d752d08b292a50eb208b07ab1f"
        },
        "backend": {"const": "llama.cpp"},
        "synthetic": {"const": False},
        "quantization": {"type": "object"},
        "structured_messages": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role": {"enum": ["system", "user", "assistant"]},
                    "content": {"type": "string", "minLength": 1},
                },
            },
        },
        "raw_user_prompt": {"type": "string", "minLength": 1},
        "serialized_chat_templated_prompt": {"type": "string", "minLength": 1},
        "serialized_prompt_sha256": sha256,
        "original_options": string_array,
        "displayed_options": string_array,
        "displayed_option_labels": string_array,
        "displayed_label_to_option": label_map,
        "candidate_label_token_ids": token_map,
        "raw_candidate_scores": score_map,
        "normalized_label_probabilities": distribution_schema(),
        "normalized_prediction": distribution_schema(),
        "label_country_human_distribution": distribution_schema(),
        "evidence_country_human_distribution": distribution_schema(),
        "presented_evidence_distribution": distribution_schema(nullable=True),
        "source_evidence_distribution": distribution_schema(nullable=True),
        "evidence_presented": {"type": "boolean"},
        "scoring_method": {
            "const": "full_contextual_option_label_sequence_log_probability"
        },
        "scoring_trace": {"type": "object"},
        "generated_answer": {"type": "null"},
        "jensen_shannon_base": {"const": 2},
        "jensen_shannon_measure": {"const": "divergence_bits"},
        "base2_jensen_shannon_divergences": {
            "type": "object",
            "minProperties": 7,
            "additionalProperties": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "country_influence": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "evidence_influence": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "EO_raw": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "EO_normalized": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/cultural-alignment/qwen-gguf-full-v2.schema.json",
        "title": "Cultural alignment Qwen GGUF full-run result row",
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": True,
        "allOf": [
            {
                "if": {
                    "properties": {
                        "condition": {"enum": ["baseline", "country_label"]}
                    }
                },
                "then": {
                    "properties": {
                        "evidence_presented": {"const": False},
                        "presented_evidence_distribution": {"type": "null"},
                        "source_evidence_distribution": {"type": "null"},
                    }
                },
            },
            {
                "if": {
                    "properties": {"condition": {"enum": ["evidence", "conflict"]}}
                },
                "then": {
                    "properties": {
                        "evidence_presented": {"const": True},
                        "presented_evidence_distribution": distribution_schema(),
                        "source_evidence_distribution": distribution_schema(),
                    }
                },
            },
        ],
    }


def build_markdown() -> str:
    return f"""# Qwen GGUF full-run result schema

Schema version: `{SCHEMA_VERSION}`

This document is generated by `scripts/generate_qwen_gguf_schema_docs.py`.
The machine-readable companion is `docs/qwen_gguf_full_result.schema.json`.

## Cardinality and keys

- One model only: `Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`.
- 200 reciprocal directed units and four conditions produce exactly 800 rows.
- The complete directed key is `(question_id, label_country, evidence_country)`.
- A result row is unique on that key plus `condition`.
- The option permutation is fixed by `(question_id, label_country)` and shared
  by all four conditions, including repeated evidence partners.

## Conditions and evidence fields

| Condition | `evidence_presented` | `presented_evidence_distribution` | `source_evidence_distribution` |
|---|:---:|---|---|
| `baseline` | false | null | null |
| `country_label` | false | null | null |
| `evidence` | true | rounded label-country treatment | unrounded label-country reference |
| `conflict` | true | rounded evidence-country treatment | unrounded evidence-country reference |

Every row separately retains `label_country_human_distribution` and
`evidence_country_human_distribution`. Metrics use these human references,
never the rounded presentation field.

## Metric definitions

All `JSD2` values are base-2 Jensen--Shannon divergences in bits.

- `country_influence = JSD2(baseline, label) - JSD2(country_label, label)`
- `evidence_influence = JSD2(baseline, label) - JSD2(evidence, label)`
- `EO_raw = JSD2(conflict, label) - JSD2(conflict, evidence)`
- `EO_normalized = [sqrt(JSD2(conflict, label)) -
  sqrt(JSD2(conflict, evidence))] / sqrt(JSD2(label, evidence))`

`EO_raw` is the primary metric. Positive EO is evidence-side; negative EO is
label-side. `EO_normalized` is secondary and equals approximately +1 for an
exact evidence prediction and -1 for an exact label prediction. The result
field containing individual JSD components is named
`base2_jensen_shannon_divergences`.

## Inference audit fields

Each row stores structured messages, the byte-exact serialized Qwen ChatML
prompt, prompt checksum, semantic option orders, label-to-option mapping,
contextual token IDs for every candidate label, complete sequence log scores,
normalized label probabilities, restored semantic probabilities, and the
full low-level scoring trace. `generated_answer` must be null; no generated
answer is parsed.

## Checkpoint and resume contract

Checkpoints become durable only after all four rows of a directed unit pass
validation. Resume validates schema version, manifest identity, model revision,
llama.cpp commit, exact completed-unit prefix, row uniqueness, probability
normalization, prompt/template invariants, and metric recomputation before
skipping work. Incomplete or mismatched checkpoints fail closed.

## Statistical unit

Country and Evidence Influence are summarized once per unique target unit
(144 observations). Both EO metrics are summarized once per directed unit
(200 observations). The 10,000-replicate nonparametric bootstrap samples
`question_id` clusters (44 clusters) and carries every target, directed pair,
reciprocal direction, condition contribution, and model row belonging to a
sampled question. Condition rows are never bootstrapped independently.

The accepted genuine smoke files remain immutable schema-v1 artifacts and are
not rewritten by this schema regeneration.
"""


def generate() -> tuple[Path, Path]:
    schema = build_schema()
    SCHEMA_PATH.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_PATH.write_text(build_markdown(), encoding="utf-8")
    return SCHEMA_PATH, MARKDOWN_PATH


if __name__ == "__main__":
    schema_path, markdown_path = generate()
    print(schema_path.relative_to(PROJECT_ROOT))
    print(markdown_path.relative_to(PROJECT_ROOT))

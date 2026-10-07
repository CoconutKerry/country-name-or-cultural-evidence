# Genuine Qwen Q4_K_M CPU Smoke Report

**Status: PASS**

This was a bounded genuine-model CPU smoke only: one official Qwen GGUF, two reciprocal directed units, and four conditions (8 saved rows). No answer was generated or parsed, no other model was loaded, and the 200-unit/full experiment was not started.

## Model and runtime provenance

- Hugging Face repository: `Qwen/Qwen2.5-7B-Instruct-GGUF`
- Immutable repository revision: `bb5d59e06d9551d752d08b292a50eb208b07ab1f`
- Quantization: `Q4_K_M`
- Selected entrypoint: `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
- Official split total: 4,683,073,632 bytes
- llama.cpp version: `0.3.0-dev`
- llama.cpp commit: `62acc89c26c66076cb72e049f307fbe93b8b9750`
- Chat template source: `embedded_gguf_metadata`
- Chat template SHA-256: `d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4`
- Qwen default system turn: explicitly materialized from the embedded template
- Serialization verification: byte-exact non-tool Qwen2.5 ChatML expansion

| GGUF shard | Bytes | SHA-256 | Verified |
|---|---:|---|---|
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 3,993,201,344 | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` | True |
| `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | 689,872,288 | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` | True |

## Environment

- Python: `3.12.13 (main, Aug  7 2026, 02:25:39) [Clang 22.1.3 ]`
- CPU: `AMD EPYC 9V74 80-Core Processor`
- Logical CPUs: 9
- RAM total: 23,109,910,528 bytes
- RAM available at capture: 21,415,038,976 bytes
- Disk free at capture: 22,205,263,872 bytes
- CUDA used: False

## Assertions

| Assertion | Result |
|---|---|
| `genuine_backend_only` | PASS |
| `exactly_eight_rows` | PASS |
| `unique_directed_condition_keys` | PASS |
| `exact_reciprocal_units` | PASS |
| `all_four_conditions_per_unit` | PASS |
| `embedded_qwen_chat_template` | PASS |
| `candidate_tokenization_valid` | PASS |
| `correct_answer_position` | PASS |
| `all_labels_scored_from_full_vocabulary` | PASS |
| `no_generation_or_answer_parsing` | PASS |
| `probabilities_valid` | PASS |
| `option_permutation_recovery` | PASS |
| `shared_first_word_independent` | PASS |
| `directed_units_not_overwritten` | PASS |
| `evidence_override_orientation` | PASS |
| `independent_first_row_rescore` | PASS |
| `full_run_approval_gates_unchanged` | PASS |

## Directed-unit metrics

All Jensen–Shannon values are base-2 divergences in bits.

| Label country | Evidence country | Country Influence | Evidence Influence | Evidence Override |
|---|---|---:|---:|---:|
| Maldives | South Korea | 0.013021398 | 0.896463683 | 0.605078402 |
| South Korea | Maldives | 0.000044874 | 0.003215734 | 0.910793151 |

## Same-first-word audit

`Strongly agree` and `Strongly disagree` were mapped to distinct displayed labels and contextual token paths in every row. Their label sequence scores and restored semantic probabilities were computed independently and were unequal in all 8 rows. Full per-row values are stored in `analysis.json`.

All displayed labels (`A`–`D`) were one token for this tokenizer. The backend implements full multi-token conditional sequence scoring and candidate-cache rollback, but that branch was not dynamically activated by this particular label set.

## Pre-package QA note

Independent review rejected an earlier draft output set because the legacy llama.cpp template helper omitted Qwen's implicit default system turn for a user-only message. The final eight rows in this package were rerun after materializing that exact embedded-template system turn and adding a byte-exact serialization assertion. The rejected rows are not included.

## Scope boundary

Both full-run approval gates retained their pre-run hashes. No full-run cell, additional checkpoint, or 200-unit experiment was executed.

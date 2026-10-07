# Gemma-2-9B-It GGUF full-run result schema

Schema version: `gemma-gguf-full-v2`

This document describes the Gemma-only CPU experiment implemented by
`scripts/run_gemma_gguf_full.py`.

## Cardinality and keys

- One model only: `bartowski/gemma-2-9b-it-GGUF:Q4_K_M` at its pinned revision.
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

Before inference, a serialization-only live preflight stores llama.cpp's
formatter string and the independent pinned Gemma expansion, their SHA-256
hashes and character/UTF-8 lengths, the first byte/character difference, and
all token-ID paths. The actual formatter string is tokenized both with and
without tokenizer-added special tokens; the no-added-specials path is selected
when it exactly equals the canonical token IDs with one leading BOS, otherwise
the added-specials path is selected only when it does. Any other result fails
before `llama_decode`.

Each row stores the user-only structured message, actual and independently
expanded serialized prompts, prompt checksums, the selected tokenization mode,
semantic option orders, label-to-option mapping, contextual token IDs for every
candidate label, complete sequence log scores, normalized label probabilities,
restored semantic probabilities, and the full low-level scoring trace. A byte
difference is accepted only when the selected actual token IDs exactly equal
the canonical token IDs. Both paths must contain exactly one leading BOS and
end in the exact Gemma assistant answer suffix. `generated_answer` must be
null; no generated answer is parsed.

## Checkpoint and resume contract

Every completed condition row is validated and atomically checkpointed before
the next inference request. A partial checkpoint must be an exact prefix of
`baseline`, `country_label`, `evidence`, `conflict`; resume skips every saved
valid row and continues with the first missing condition. Complete units are
also written to `results.jsonl`. Resume validates schema version, manifest
identity, model revision, llama.cpp commit, run fingerprint, row uniqueness,
probability normalization, prompt/template invariants, and metric recomputation.
Incomplete, out-of-order, or mismatched checkpoints fail closed.

## Statistical unit

Country and Evidence Influence are summarized once per unique target unit
(144 observations). Both EO metrics are summarized once per directed unit
(200 observations). The 10,000-replicate nonparametric bootstrap samples
`question_id` clusters (44 clusters) and carries every target, directed pair,
reciprocal direction, condition contribution, and model row belonging to a
sampled question. Condition rows are never bootstrapped independently.

The genuine Gemma smoke uses the same row builder and validator for exactly two
reciprocal directed units and all four conditions before the full run proceeds.

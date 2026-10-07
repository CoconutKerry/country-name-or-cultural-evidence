# Repaired Experiment Protocol (v2)

## 1. Scope and validity status

This protocol applies only to outputs produced by the repaired v2 pipeline.
All archived v1 results are invalid because their option scorer and directed
grouping were incorrect.

## 2. Data eligibility

Include only questions and samples present in `dataset_v2.json`.

Required checks:

- at least two substantive answer options;
- finite, nonnegative distributions with positive mass;
- canonical string option IDs and an exact option-key mapping;
- at least two eligible national samples;
- exact shared positive substantive answer schema; and
- substantive response mass greater than 50% before renormalization.

Exclude explicit non-national and old-national samples. Retain current-national
samples while preserving their source status. Remove only unambiguous
missing/refused categories; retain potentially substantive `Both`, `Neither`,
`None`, and `Depends` answers. Quarantine mixed schemas, semantic aliases,
crosstab total columns, unresolved country-specific placeholders, and
relative-time items that cannot be resolved without a survey wave.

The supplied source export contains no survey year, wave, or sample-size fields.
These attributes cannot be verified here and must be listed as a limitation.

## 3. Directed experimental units

The authoritative key is:

```text
(question_id, country, conflict_country)
```

`country` is the target/label country. `conflict_country` is the source of the
conflicting evidence. Both directions are included together, and duplicate
triples are rejected.

## 4. Conditions

| Condition | Country label | Population evidence |
|---|:---:|:---:|
| Baseline | No | No |
| Country label | Yes | No |
| Population evidence | No | Target-country distribution |
| Conflict | Target country | Different-country distribution |

Every directed unit must have each condition exactly once per model.

## 5. Model scoring

Options are displayed under `A/B/...` labels. Each prompt asks which option one
randomly selected respondent chose and ends with a strict one-label response
contract. The runner computes the complete conditional log likelihood of each
label continuation, applies stable softmax normalization, and maps label
probabilities back to semantic option IDs. For option-order tests, recovery is
by the stored label-to-option bijection, never by original position.

The four core conditions share one prompt skeleton; cue blocks alone vary.
Option order is deterministically permuted by `(question_id, target country)`
so repeated targets use the same mapping across conflict partners. Each
instruction checkpoint must use its pinned chat template. Prompt-plus-label is
tokenized jointly and the prompt token IDs must be an exact prefix, preventing
token-boundary merges from corrupting continuation scores.

## 6. Metrics

All comparisons use base-2 Jensen–Shannon divergence in `[0, 1]`.

- Country Influence = `JSD(baseline, target) - JSD(country_label, target)`
- Evidence Influence = `JSD(baseline, target) - JSD(population_evidence, target)`
- Evidence Override = `JSD(conflict, target) - JSD(conflict, evidence)`

Positive Evidence Override means the conflict prediction is closer to the
evidence source; negative means it is closer to the target/label country.
This is a signed proximity contrast, not a causal override estimator.

Country Influence and Evidence Influence are computed once per unique
`(model, question_id, target country)`. Evidence Override is computed once per
full model-specific directed unit. Report both target-unit and directed-unit
counts; uncertainty must account for clustering/dependence by question and
reciprocal pair.

## 7. Reproducibility and safety

- Seed: 42.
- Model weights must be loaded in evaluation mode.
- Install `requirements-inference.lock` and retain the complete runtime package
  snapshot stored in output metadata.
- Enforce and record exact model/tokenizer commit hashes, chat-template hash,
  device map, dtype, raw/rendered prompt hashes, and package versions.
- Hash-record the v1 source export, v2 dataset, directed pair manifest, and
  repair summary inside every result artifact.
- Every non-smoke mode requires explicit `--allow-full-run` authorization.
- Synthetic smoke outputs must carry `synthetic_backend: true` and
  `scientific_inference: false`.
- A genuine one-model LLM smoke on suitable GPU compute must pass before the
  four-model run is approved.

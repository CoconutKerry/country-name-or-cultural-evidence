# Prompt Design Notes

## Version history

### 2026-07-21: v1 (invalidated)

The v1 prompts asked the model to print a numeric probability vector, while the
implementation scored the first token of each full answer text. Prompt and
scoring contracts therefore did not match.

### 2026-08-30: v2 repaired contract

All conditions now:

- use the same social-science-researcher role;
- describe one respondent sampled at random;
- display semantic options under deterministic `A/B/...` labels;
- request exactly one label and end with `Answer:`; and
- obtain a distribution by normalizing complete label-continuation likelihoods.

The four core conditions use one invariant template. The treatment consists
only of optional `Target Country:` and/or `Survey Response Statistics:` cue
blocks. Scientific inference renders that user text with each pinned
tokenizer's chat template. A label separator is added only if the rendered
assistant prefix does not already end in whitespace. Prompt-plus-label is
tokenized jointly, and the prompt IDs must be an exact prefix. Checkpoint and
tokenizer revisions are immutable commits. Option permutations are seeded by
`(question_id, target country)`, never the conflict partner.

The prompt builder exposes the exact label-to-option map. If displayed options
are permuted, the same map is inverted to recover probabilities by semantic
option ID. Answer-text tokens are never used as option identifiers.

## Evidence presentation

Evidence is formatted to six decimal percentage places with largest-remainder
rounding, so displayed values total exactly 100%. The exact presented
distribution and the unrounded source distribution are both stored. Example:

```text
A (Approve): 65.000000%, B (Disapprove): 35.000000%
```

In the conflict condition, the country name denotes the target/label country,
while the numerical distribution comes from the directed unit's explicit
`conflict_country`. This source is carried in the unit key and result metadata.

# Validation report

This private-review snapshot was assembled from the four authoritative 800-row
full-result packages and the final repaired source handoffs. The following
checks were run in a clean working copy of the repository.

## Cross-model analysis

- 3,200 genuine prediction rows loaded.
- 800 rows and 200 directed contrasts validated for each model.
- Four conditions validated for every directed contrast.
- Reciprocal contrast coverage verified.
- Fixed option permutations verified across conditions and models.
- Probability vectors, scoring method, backend, and `synthetic: false` flags
  verified.
- 144 unique target units and 44 question clusters verified per model.
- CI, EI, paired evidence advantage, raw EO, normalized EO, conflict counts,
  and cross-model agreement recomputed from stored probability distributions.
- 10,000 question-clustered bootstrap replicates reproduced with seed 42 and
  shared cluster draws across models and estimands.
- Paper-facing point estimates and confidence intervals matched.

## Analysis-code tests

```text
4 passed
```

## Model-specific source tests

```text
Qwen2.5-7B-Instruct:       87 passed, 19,218 subtests passed
Llama-3.1-8B-Instruct:     47 passed, 12 subtests passed
Mistral-7B-Instruct-v0.3:  45 passed, 16 subtests passed
Gemma-2-9B-It:             61 passed, 12 subtests passed
```

These tests do not download or execute the GGUF weights. Full inference still
requires the pinned model files and `llama.cpp` checkout listed in the model
provenance manifest.

## Important agreement decomposition

The 39 contrasts with cross-model disagreement comprise 29 evidence-majority
3--1 splits, 3 label-majority 1--3 splits, and 7 two-to-two splits. Mistral is
the sole label-side model in 20 of the 29 evidence-majority splits.

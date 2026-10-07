# Reproducibility guide

## Analysis-only reproduction

The analysis workflow requires no model weights and completes from the released
raw predictions:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.lock
make reproduce
make test
```

The workflow performs the following steps:

1. validates 800 genuine rows for each model;
2. checks the 200 directed contrasts and four conditions;
3. verifies reciprocal directions and fixed option permutations;
4. recomputes CI, EI, paired evidence advantage, raw EO, and normalized EO from
   the stored probability distributions;
5. runs 10,000 question-clustered bootstrap replicates with seed 42 and a
   shared draw across models and estimands;
6. regenerates summary tables, cross-model agreement, selected cases, and
   Figure 2; and
7. checks the resulting values against the paper-facing expected results.

## Full inference reproduction

The exact model-specific source snapshots are under
`inference/validated_sources/`. Each directory contains its original runbook,
source code, tests, repaired data, and hard provenance gates. Use the exact
model repositories, revisions, filenames, checksums, and `llama.cpp` commit in
`configs/models.json` and `provenance/MODEL_PROVENANCE.md`.

General procedure:

1. create a clean Linux workspace;
2. obtain the pinned GGUF files without substituting a different revision or
   quantization;
3. check every model file's size and SHA-256;
4. check out `llama.cpp` commit
   `62acc89c26c66076cb72e049f307fbe93b8b9750`;
5. build the low-level scoring helper from the source snapshot;
6. run the genuine two-direction, four-condition smoke test;
7. start the 800-row full run only after the smoke gates pass;
8. run the model-specific analysis script with 10,000 clustered bootstrap
   replicates and seed 42; and
9. compare the output with `results/summary/main_metrics.csv`.

No GGUF model weights are redistributed in this repository.

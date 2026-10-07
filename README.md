# Country Name or Cultural Evidence?

Reproducibility materials for **Country Name or Cultural Evidence? A Causal
Audit of Population-Grounded Steering in Large Language Models**.

The repository supports two workflows:

1. **Analysis reproduction** from the released 3,200 likelihood-based
   predictions. This regenerates the paper metrics, clustered bootstrap
   intervals, cross-model agreement results, selected examples, and Figure 2.
2. **Full inference reproduction** with the exact model, GGUF, `llama.cpp`,
   data, prompt, and scoring pins used for each model. Model weights are not
   redistributed.

The experiment contains 100 population pairs evaluated in both directions,
200 directed contrasts, four cue conditions, and four instruction-tuned
open-weight models. The raw outputs contain 800 records per model.

## Quick analysis reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.lock
make reproduce
```

Expected headline results:

- Country Influence: 0.028--0.071 across models.
- Evidence Influence: 0.184--0.284 across models.
- Normalized Evidence Override: 0.543--0.728 across models.
- Evidence-side conflict predictions: 85.5--97.5%.
- All four models agree on the evidence side for 161 of 200 contrasts.

`make reproduce` regenerates all analysis tables, the human-readable result report,
paper-ready LaTeX rows, Figure 2, and machine-readable release metadata.
`make verify` checks raw-result counts, experimental-unit coverage, probability
vectors, derived metrics, paper-facing summary values, and immutable input
hashes. The preserved inference source suites can be checked separately with
`make validate-inference`.

## Repository map

- `data/`: repaired survey data, directed contrast manifest, and repair record.
- `results/raw/`: exact model outputs and run provenance, compressed without
  changing the JSONL content.
- `results/processed/`: harmonized cross-model records and unit-level metrics.
- `results/summary/`: paper-level tables, agreement analyses, cases, a
  machine-readable result bundle, and the human-readable `RESULTS.md` report.
- `src/audit_repro/`: analysis and validation library.
- `scripts/`: command-line entry points for reproducing outputs.
- `inference/validated_sources/`: model-specific source snapshots used for the
  four genuine GGUF runs.
- `provenance/`: model pins, original package identities, and release hashes.
- `docs/`: data, output-schema, inference, and public-release documentation.

## License and Provenance

This is the public release accompanying the camera-ready version of
"Country Name or Cultural Evidence? A Causal Audit of Population-Grounded
Steering in Large Language Models" (AACL 2026 Workshop).

- All code is released under the MIT License (see `LICENSE`).
- Data files under `data/` and `inference/validated_sources/*/data/` are
  derived from GlobalOpinionQA (Durmus et al., 2023), which is itself adapted
  from the Pew Global Attitudes Survey and the World Values Survey; see
  `LICENSES.md` for redistribution details and `provenance/` for release
  hashes.
- Model weights are not redistributed. Each GGUF remains subject to its
  repository and upstream model licenses; llama.cpp remains subject to its
  own license.
- Files under `inference/validated_sources/` are preserved research source
  snapshots; their inclusion does not alter third-party licensing terms.
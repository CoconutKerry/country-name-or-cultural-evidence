# Mistral GGUF full-run experiment report

## Validation outcome

**PASS.** The analysis validated exactly 800 genuine llama.cpp result rows: 200 cleaned reciprocal directed units under all four conditions. No synthetic or fallback row was accepted.

- Model: `bartowski/Mistral-7B-Instruct-v0.3-GGUF@61fd4167fff3ab01ee1cfe0da183fa27a944db48:Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`
- Paper model name: `Mistral-7B-Instruct-v0.3`
- Upstream checkpoint: `mistralai/Mistral-7B-Instruct-v0.3`
- Revision: `61fd4167fff3ab01ee1cfe0da183fa27a944db48`
- GGUF: `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (4372812000 bytes)
- GGUF SHA-256: `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6`
- llama.cpp commit: `62acc89c26c66076cb72e049f307fbe93b8b9750`
- Backend / quantization: `llama.cpp` / `Q4_K_M`
- Directed units: 200
- Deduplicated target units for CI/EI: 144
- Question-ID clusters: 44
- Chat-template source: `embedded_gguf_metadata`
- Embedded chat-template SHA-256: `26a59556925c987317ce5291811ba3b7f32ec4c647c400c6cc7e3a9993007ba7`
- Synthetic: `false`

All result rows were bound to the validated run fingerprint. The exact single-file GGUF, embedded model-specific chat template, CPU-only llama.cpp backend, and no-generation candidate-label scorer passed provenance checks.

## Point estimates and clustered-bootstrap intervals

| Metric | N | Estimate | 95% CI |
|---|---:|---:|---:|
| `country_influence` | 144 | 0.044678 | [0.004066, 0.089230] |
| `evidence_influence` | 144 | 0.220530 | [0.164735, 0.278625] |
| `EO_raw` | 200 | 0.516749 | [0.427308, 0.601772] |
| `EO_normalized` | 200 | 0.543333 | [0.440833, 0.642071] |

Intervals use 10,000 nonparametric bootstrap replicates sampled at the `question_id` level. Each sampled cluster retains all target units, all directed and reciprocal pairs, and all condition contributions. One shared draw is used across every metric; condition rows are never sampled independently.

## Conflict classification

Classification uses `EO_raw` with tolerance `1e-12`.

| Class | Count | Percent |
|---|---:|---:|
| evidence-side | 171 | 85.5% |
| label-side | 29 | 14.5% |
| tie | 0 | 0.0% |

Positive `EO_raw` means evidence-side; negative means label-side. `EO_normalized` uses square-root base-2 Jensen--Shannon distance and has the same orientation. These are signed proximity metrics, not stand-alone causal claims.

## Generated artifacts

- `analysis.json`
- `validated_results.jsonl`
- `results.csv`
- `directed_unit_metrics.csv`
- `target_unit_metrics.csv`
- `bootstrap_summary.csv`
- `bootstrap_results.json`
- `conflict_classifications.csv`
- `metric_tables.json`
- `tables/model_metrics.tex`
- `tables/conflict_classification.tex`
- `EXPERIMENT_REPORT.md`

## Analysis provenance diagnostic

The supplied analysis entrypoint stopped at its loaded-model-size provenance guard. That field comes from `llama_model_size()`, which the pinned llama.cpp header defines as tensor bytes, not GGUF file length. The exact GGUF file size and SHA-256 were independently verified, and the 800 raw rows/checkpoints passed the standalone audit. The metrics and bootstrap in this directory are therefore explicitly independent post-hoc outputs computed with the supplied unmodified metric functions; no inference or raw-result modification occurred. See `ANALYSIS_PROVENANCE_DIAGNOSTIC.json`.

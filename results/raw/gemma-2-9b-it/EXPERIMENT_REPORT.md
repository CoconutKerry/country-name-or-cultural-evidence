# Gemma GGUF full-run experiment report

## Validation outcome

**PASS.** The analysis validated exactly 800 genuine llama.cpp result rows: 200 cleaned reciprocal directed units under all four conditions. No synthetic or fallback row was accepted.

- Model: `bartowski/gemma-2-9b-it-GGUF:Q4_K_M`
- Revision: `d731033f3dc4018261fd39896e50984d398b4ac5`
- Backend / quantization: `llama.cpp` / `Q4_K_M`
- Directed units: 200
- Deduplicated target units for CI/EI: 144
- Question-ID clusters: 44

## Point estimates and clustered-bootstrap intervals

| Metric | N | Estimate | 95% CI |
|---|---:|---:|---:|
| `country_influence` | 144 | 0.071451 | [0.027302, 0.116395] |
| `evidence_influence` | 144 | 0.283595 | [0.230166, 0.338130] |
| `EO_raw` | 200 | 0.684734 | [0.628785, 0.727190] |
| `EO_normalized` | 200 | 0.728041 | [0.656457, 0.788194] |

Intervals use 10,000 nonparametric bootstrap replicates sampled at the `question_id` level. Each sampled cluster retains all target units, all directed and reciprocal pairs, and all condition contributions. One shared draw is used across every metric; condition rows are never sampled independently.

## Conflict classification

Classification uses `EO_raw` with tolerance `1e-12`.

| Class | Count | Percent |
|---|---:|---:|
| evidence-side | 195 | 97.5% |
| label-side | 5 | 2.5% |
| tie | 0 | 0.0% |

Positive `EO_raw` means evidence-side; negative means label-side. `EO_normalized` uses square-root base-2 Jensen--Shannon distance and has the same orientation. These are signed proximity metrics, not stand-alone causal claims.

## Generated artifacts

- `analysis.json`
- `results.csv`
- `directed_unit_metrics.csv`
- `target_unit_metrics.csv`
- `bootstrap_summary.csv`
- `conflict_classifications.csv`
- `metric_tables.json`
- `tables/model_metrics.tex`
- `tables/conflict_classification.tex`
- `EXPERIMENT_REPORT.md`
- `EXCLUSION_AND_FAILURE_REPORT.md`

# Llama GGUF full-run experiment report

## Validation outcome

**PASS.** The analysis validated exactly 800 genuine llama.cpp result rows: 200 cleaned reciprocal directed units under all four conditions. No synthetic or fallback row was accepted.

- Model: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M`
- Revision: `bf5b95e96dac0462e2a09145ec66cae9a3f12067`
- Backend / quantization: `llama.cpp` / `Q4_K_M`
- Directed units: 200
- Deduplicated target units for CI/EI: 144
- Question-ID clusters: 44

## Point estimates and clustered-bootstrap intervals

| Metric | N | Estimate | 95% CI |
|---|---:|---:|---:|
| `country_influence` | 144 | 0.055290 | [0.032935, 0.080804] |
| `evidence_influence` | 144 | 0.184445 | [0.149602, 0.221786] |
| `EO_raw` | 200 | 0.614676 | [0.556296, 0.663136] |
| `EO_normalized` | 200 | 0.711436 | [0.641556, 0.771189] |

Intervals use 10,000 nonparametric bootstrap replicates sampled at the `question_id` level. Each sampled cluster retains all target units, all directed and reciprocal pairs, and all condition contributions. One shared draw is used across every metric; condition rows are never sampled independently.

## Conflict classification

Classification uses `EO_raw` with tolerance `1e-12`.

| Class | Count | Percent |
|---|---:|---:|
| evidence-side | 193 | 96.5% |
| label-side | 7 | 3.5% |
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

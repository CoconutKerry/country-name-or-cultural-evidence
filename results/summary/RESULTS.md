# Validated cross-model results

The release contains 3,200 genuine likelihood-based predictions from four models, with 200 directed contrasts and four cue conditions per model. Country Influence and Evidence Influence are estimated over 144 unique target units per model; Evidence Override is estimated over all 200 directed contrasts.

## Main metrics

| Model | CI [95% CI] | EI [95% CI] | EI-CI [95% CI] | EO raw [95% CI] | EO normalized [95% CI] |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.0281 [-0.0077, 0.0660] | 0.2676 [0.2202, 0.3197] | 0.2395 [0.1932, 0.2910] | 0.6384 [0.5714, 0.6954] | 0.6686 [0.5916, 0.7394] |
| Llama-3.1-8B-Instruct | 0.0553 [0.0329, 0.0808] | 0.1844 [0.1496, 0.2218] | 0.1292 [0.0970, 0.1658] | 0.6147 [0.5563, 0.6631] | 0.7114 [0.6416, 0.7712] |
| Mistral-7B-Instruct-v0.3 | 0.0447 [0.0041, 0.0892] | 0.2205 [0.1647, 0.2786] | 0.1759 [0.1287, 0.2231] | 0.5167 [0.4273, 0.6018] | 0.5433 [0.4408, 0.6421] |
| Gemma-2-9B-It | 0.0715 [0.0273, 0.1164] | 0.2836 [0.2302, 0.3381] | 0.2121 [0.1619, 0.2626] | 0.6847 [0.6288, 0.7272] | 0.7280 [0.6565, 0.7882] |

All four paired evidence advantages are positive, so direct population evidence produces the larger mean improvement for every model.

## Conflict classification

| Model | Evidence-side | Label-side | Tie |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | 189 (94.5%) | 11 (5.5%) | 0 |
| Llama-3.1-8B-Instruct | 193 (96.5%) | 7 (3.5%) | 0 |
| Mistral-7B-Instruct-v0.3 | 171 (85.5%) | 29 (14.5%) | 0 |
| Gemma-2-9B-It | 195 (97.5%) | 5 (2.5%) | 0 |

## Cross-model agreement

- All four models are evidence-side on 161/200 contrasts (80.5%).
- At least three models are evidence-side on 190/200 contrasts (95.0%).
- Cross-model disagreement occurs on 39/200 contrasts (19.5%).
- The disagreements comprise 29 evidence-majority 3--1 splits, 3 label-majority 1--3 splits, and 7 2--2 splits.
- Mistral is the sole label-side model on 20 contrasts.

## Results by survey source

| Model | Source | Target units | Directed contrasts | CI | EI | EI-CI | EO normalized | Evidence-side |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | Global Attitudes | 104 | 146 | 0.0407 | 0.2930 | 0.2523 | 0.6693 | 93.8% |
| Qwen2.5-7B-Instruct | WVS | 40 | 54 | -0.0048 | 0.2015 | 0.2062 | 0.6665 | 96.3% |
| Llama-3.1-8B-Instruct | Global Attitudes | 104 | 146 | 0.0699 | 0.1933 | 0.1234 | 0.7259 | 95.2% |
| Llama-3.1-8B-Instruct | WVS | 40 | 54 | 0.0174 | 0.1615 | 0.1441 | 0.6724 | 100.0% |
| Mistral-7B-Instruct-v0.3 | Global Attitudes | 104 | 146 | 0.0605 | 0.2415 | 0.1810 | 0.5314 | 84.9% |
| Mistral-7B-Instruct-v0.3 | WVS | 40 | 54 | 0.0036 | 0.1661 | 0.1624 | 0.5757 | 87.0% |
| Gemma-2-9B-It | Global Attitudes | 104 | 146 | 0.0750 | 0.3026 | 0.2276 | 0.7798 | 100.0% |
| Gemma-2-9B-It | WVS | 40 | 54 | 0.0623 | 0.2342 | 0.1719 | 0.5881 | 90.7% |

## Mechanically selected cases

| Selection rule | Question | Label | Evidence | Qwen | Llama | Mistral | Gemma |
|---|---|---|---|---:|---:|---:|---:|
| Largest mean EO | Regardless of how you feel about the protests, were you sympathetic to Muslims who were offended by these cartoons, or not? | Spain | Jordan | 0.940 | 0.976 | 0.940 | 0.941 |
| Lowest mean EO | Which one of these comes closest to your opinion, number 1 or number 2?...#1 - It is not necessary to believe in God in order to be moral and have good values or #2 - It is necessary to believe in God in order to be moral and have good values | Sweden | Pakistan | -0.885 | -0.973 | -0.890 | 0.031 |
| Largest model disagreement | Does the government of Russia respect the personal freedoms of its people? | Netherlands | Vietnam | -0.884 | 0.929 | -0.899 | 0.909 |

## Interpretation

Positive Evidence Override is evidence-side. CI and EI are mean reductions in base-2 Jensen--Shannon divergence from the baseline prediction to the target population distribution. The conflict classification records which reference distribution is closer; normalized Evidence Override additionally records the strength of that relative proximity. Intervals use 10,000 question-clustered percentile bootstrap replicates with seed 42.

## Output files

- `main_metrics.csv`: paper-level CI, EI, paired evidence advantage, and Evidence Override estimates with intervals.
- `conflict_classification.csv`: evidence-side, label-side, and tie counts by model.
- `cross_model_agreement_by_unit.csv`: per-contrast model classifications and normalized override values.
- `cross_model_agreement_summary.csv`: aggregate agreement and split counts.
- `source_stratified_metrics.csv`: results separated by Global Attitudes and WVS.
- `illustrative_cases.csv`: the three deterministic case selections reported in the paper.
- `model_runtime.csv`: recorded runtime and run-fingerprint summary.
- `../processed/predictions_all_models.jsonl.gz`: harmonized 3,200-row master prediction file.
- `../processed/target_unit_metrics.csv`: 576 model-target rows used for CI, EI, and EI-CI.
- `../processed/directed_unit_metrics.csv`: 800 model-contrast rows used for Evidence Override.
- `../figures/cue_effects.pdf` and `.png`: regenerated Figure 2.
- `../latex/`: paper-ready LaTeX table rows generated from the summary files.

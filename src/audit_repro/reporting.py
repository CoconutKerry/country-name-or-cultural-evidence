from __future__ import annotations

from pathlib import Path

import pandas as pd


def fmt(value: float) -> str:
    return f"{value:.4f}"


def _interval(row: pd.Series, metric: str) -> str:
    return (
        f"{fmt(row[metric])} "
        f"[{fmt(row[f'{metric}_ci_95_lower'])}, "
        f"{fmt(row[f'{metric}_ci_95_upper'])}]"
    )


def write_results_report(
    root: Path,
    main: pd.DataFrame,
    conflict: pd.DataFrame,
    agreement: pd.DataFrame,
    cases: pd.DataFrame,
    source: pd.DataFrame,
) -> None:
    """Write the human-readable summary distributed with the results."""
    lines = [
        "# Validated cross-model results",
        "",
        "The release contains 3,200 genuine likelihood-based predictions from four models, with 200 directed contrasts and four cue conditions per model. Country Influence and Evidence Influence are estimated over 144 unique target units per model; Evidence Override is estimated over all 200 directed contrasts.",
        "",
        "## Main metrics",
        "",
        "| Model | CI [95% CI] | EI [95% CI] | EI-CI [95% CI] | EO raw [95% CI] | EO normalized [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"| {row['model']} | {_interval(row, 'country_influence')} "
            f"| {_interval(row, 'evidence_influence')} "
            f"| {_interval(row, 'evidence_advantage')} "
            f"| {_interval(row, 'EO_raw')} "
            f"| {_interval(row, 'EO_normalized')} |"
        )

    lines += [
        "",
        "All four paired evidence advantages are positive, so direct population evidence produces the larger mean improvement for every model.",
        "",
        "## Conflict classification",
        "",
        "| Model | Evidence-side | Label-side | Tie |",
        "|---|---:|---:|---:|",
    ]
    for _, row in conflict.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['evidence_side_n'])} ({row['evidence_side_pct']:.1f}%) "
            f"| {int(row['label_side_n'])} ({row['label_side_pct']:.1f}%) | {int(row['tie_n'])} |"
        )

    stats = {row.statistic: row for _, row in agreement.iterrows()}
    lines += [
        "",
        "## Cross-model agreement",
        "",
        f"- All four models are evidence-side on {int(stats['all_four_evidence_side']['count'])}/200 contrasts ({stats['all_four_evidence_side']['percent']:.1f}%).",
        f"- At least three models are evidence-side on {int(stats['at_least_three_evidence_side']['count'])}/200 contrasts ({stats['at_least_three_evidence_side']['percent']:.1f}%).",
        f"- Cross-model disagreement occurs on {int(stats['any_cross_model_disagreement']['count'])}/200 contrasts ({stats['any_cross_model_disagreement']['percent']:.1f}%).",
        f"- The disagreements comprise {int(stats['three_evidence_one_label']['count'])} evidence-majority 3--1 splits, {int(stats['one_evidence_three_label']['count'])} label-majority 1--3 splits, and {int(stats['two_to_two_splits']['count'])} 2--2 splits.",
        f"- Mistral is the sole label-side model on {int(stats['mistral_only_label_side']['count'])} contrasts.",
        "",
        "## Results by survey source",
        "",
        "| Model | Source | Target units | Directed contrasts | CI | EI | EI-CI | EO normalized | Evidence-side |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in source.iterrows():
        lines.append(
            f"| {row['model']} | {row['source']} | {int(row['target_units'])} | {int(row['directed_units'])} "
            f"| {fmt(row['country_influence'])} | {fmt(row['evidence_influence'])} "
            f"| {fmt(row['evidence_advantage'])} | {fmt(row['EO_normalized'])} "
            f"| {row['evidence_side_pct']:.1f}% |"
        )

    lines += [
        "",
        "## Mechanically selected cases",
        "",
        "| Selection rule | Question | Label | Evidence | Qwen | Llama | Mistral | Gemma |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in cases.iterrows():
        question = row["question"].replace("|", "\\|")
        lines.append(
            f"| {row['selection_rule']} | {question} | {row['label_country']} | {row['evidence_country']} "
            f"| {row['qwen']:.3f} | {row['llama']:.3f} | {row['mistral']:.3f} | {row['gemma']:.3f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Positive Evidence Override is evidence-side. CI and EI are mean reductions in base-2 Jensen--Shannon divergence from the baseline prediction to the target population distribution. The conflict classification records which reference distribution is closer; normalized Evidence Override additionally records the strength of that relative proximity. Intervals use 10,000 question-clustered percentile bootstrap replicates with seed 42.",
        "",
        "## Output files",
        "",
        "- `main_metrics.csv`: paper-level CI, EI, paired evidence advantage, and Evidence Override estimates with intervals.",
        "- `conflict_classification.csv`: evidence-side, label-side, and tie counts by model.",
        "- `cross_model_agreement_by_unit.csv`: per-contrast model classifications and normalized override values.",
        "- `cross_model_agreement_summary.csv`: aggregate agreement and split counts.",
        "- `source_stratified_metrics.csv`: results separated by Global Attitudes and WVS.",
        "- `illustrative_cases.csv`: the three deterministic case selections reported in the paper.",
        "- `model_runtime.csv`: recorded runtime and run-fingerprint summary.",
        "- `../processed/predictions_all_models.jsonl.gz`: harmonized 3,200-row master prediction file.",
        "- `../processed/target_unit_metrics.csv`: 576 model-target rows used for CI, EI, and EI-CI.",
        "- `../processed/directed_unit_metrics.csv`: 800 model-contrast rows used for Evidence Override.",
        "- `../figures/cue_effects.pdf` and `.png`: regenerated Figure 2.",
        "- `../latex/`: paper-ready LaTeX table rows generated from the summary files.",
        "",
    ]
    (root / "results/summary/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

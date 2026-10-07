from __future__ import annotations

from pathlib import Path

import pandas as pd

MODEL_LABELS = {
    "qwen": "Qwen2.5-7B",
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B-v0.3",
    "gemma": "Gemma-2-9B",
}


def _estci(row: pd.Series, metric: str) -> str:
    return (
        f"\\estci{{{row[metric]:.4f}}}"
        f"{{{row[f'{metric}_ci_95_lower']:.4f}}}"
        f"{{{row[f'{metric}_ci_95_upper']:.4f}}}"
    )


def write_latex_rows(
    root: Path,
    main: pd.DataFrame,
    conflict: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    """Generate paper-ready LaTeX rows from validated summary tables."""
    out = root / "results/latex"
    out.mkdir(parents=True, exist_ok=True)

    main_lines: list[str] = []
    for _, row in main.iterrows():
        model = MODEL_LABELS[row["model_key"]]
        main_lines.append(
            f"{model}\n"
            f"& {_estci(row, 'country_influence')}\n"
            f"& {_estci(row, 'evidence_influence')}\n"
            f"& {_estci(row, 'evidence_advantage')} \\\\"
        )
    (out / "main_metrics_rows.tex").write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    main_index = main.set_index("model_key")
    conflict_lines: list[str] = []
    for _, row in conflict.iterrows():
        key = row["model_key"]
        mrow = main_index.loc[key]
        model = MODEL_LABELS[key]
        conflict_lines.append(
            f"{model}\n"
            f"& {_estci(mrow, 'EO_raw')}\n"
            f"& {_estci(mrow, 'EO_normalized')}\n"
            f"& {int(row['evidence_side_n'])} ({row['evidence_side_pct']:.1f}\\%) / "
            f"{int(row['label_side_n'])} ({row['label_side_pct']:.1f}\\%) \\\\"
        )
    (out / "conflict_rows.tex").write_text("\n".join(conflict_lines) + "\n", encoding="utf-8")

    case_labels = {
        "Largest mean EO": "Largest mean EO",
        "Lowest mean EO": "Lowest mean EO",
        "Largest model disagreement": "Largest model disagreement",
    }
    case_lines: list[str] = []
    for _, row in cases.iterrows():
        label = case_labels[row["selection_rule"]]
        question = row["question"].replace("&", "\\&").replace("%", "\\%")
        case_lines.append(
            f"{label}: {question}\n"
            f"& {row['label_country']}\n"
            f"& {row['evidence_country']}\n"
            f"& {row['qwen']:.3f}\n"
            f"& {row['llama']:.3f}\n"
            f"& {row['mistral']:.3f}\n"
            f"& {row['gemma']:.3f} \\\\"
        )
    (out / "illustrative_cases_rows.tex").write_text("\n\n".join(case_lines) + "\n", encoding="utf-8")

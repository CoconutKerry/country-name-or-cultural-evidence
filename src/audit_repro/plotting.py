from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm

COUNTRY = "#D55E00"
EVIDENCE = "#0072B2"


def _font() -> str:
    available = {entry.name for entry in fm.fontManager.ttflist}
    for candidate in ("Times New Roman", "Tinos", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"):
        if candidate in available:
            return candidate
    return "serif"


def make_figure(main_metrics: pd.DataFrame, pdf_path: Path, png_path: Path) -> str:
    font = _font()
    plt.rcParams.update({
        "font.family": font,
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    labels = ["Qwen", "Llama", "Mistral", "Gemma"]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    offset = 0.08
    ci = main_metrics.set_index("model_key")
    country = [ci.loc[m, "country_influence"] for m in ("qwen", "llama", "mistral", "gemma")]
    country_low = [ci.loc[m, "country_influence"] - ci.loc[m, "country_influence_ci_95_lower"] for m in ("qwen", "llama", "mistral", "gemma")]
    country_high = [ci.loc[m, "country_influence_ci_95_upper"] - ci.loc[m, "country_influence"] for m in ("qwen", "llama", "mistral", "gemma")]
    evidence = [ci.loc[m, "evidence_influence"] for m in ("qwen", "llama", "mistral", "gemma")]
    evidence_low = [ci.loc[m, "evidence_influence"] - ci.loc[m, "evidence_influence_ci_95_lower"] for m in ("qwen", "llama", "mistral", "gemma")]
    evidence_high = [ci.loc[m, "evidence_influence_ci_95_upper"] - ci.loc[m, "evidence_influence"] for m in ("qwen", "llama", "mistral", "gemma")]
    ax.errorbar([v - offset for v in x], country, yerr=[country_low, country_high], fmt="o", capsize=3, label="Country Influence", color=COUNTRY)
    ax.errorbar([v + offset for v in x], evidence, yerr=[evidence_low, evidence_high], fmt="s", capsize=3, label="Evidence Influence", color=EVIDENCE)
    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean reduction in base-2 JSD")
    ax.set_ylim(-0.02, 0.37)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_metadata = {"Creator": "audit_repro", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None}
    fig.savefig(pdf_path, bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return font

#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_repro.plotting import make_figure


def main() -> None:
    metrics = pd.read_csv(ROOT / "results/summary/main_metrics.csv")
    font = make_figure(metrics, ROOT / "results/figures/cue_effects.pdf", ROOT / "results/figures/cue_effects.png")
    print(f"Generated Figure 2 with {font}.")


if __name__ == "__main__":
    main()

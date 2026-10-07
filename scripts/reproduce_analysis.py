#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_repro.pipeline import run
from audit_repro.reporting import write_results_report
from audit_repro.latex import write_latex_rows


def main() -> None:
    result = run(ROOT)
    tables = result["tables"]
    write_results_report(
        ROOT,
        tables["main_metrics"],
        tables["conflict_classification"],
        tables["cross_model_agreement_summary"],
        tables["illustrative_cases"],
        tables["source_stratified_metrics"],
    )
    write_latex_rows(
        ROOT,
        tables["main_metrics"],
        tables["conflict_classification"],
        tables["illustrative_cases"],
    )
    manifest = {
        "predictions": len(result["rows"]),
        "target_unit_metrics": len(result["target_metrics"]),
        "directed_unit_metrics": len(result["directed_metrics"]),
        "models": sorted(result["target_metrics"]["model_key"].unique().tolist()),
        "question_clusters": int(result["target_metrics"]["question_id"].nunique()),
        "status": "PASS",
    }
    (ROOT / "results/summary/analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

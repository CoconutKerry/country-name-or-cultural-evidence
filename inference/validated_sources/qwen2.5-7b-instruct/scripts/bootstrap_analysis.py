#!/usr/bin/env python3
"""Validate repaired results and compute question-clustered bootstrap CIs.

The bootstrap covers Country Influence, Evidence Influence, raw Evidence
Override (``EO_raw``), and normalized Evidence Override (``EO_normalized``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_results import analyze_payload
from src.clustered_bootstrap import clustered_bootstrap


def analyze_with_clustered_bootstrap(
    payload: dict[str, Any],
    *,
    allow_synthetic: bool = False,
    n_replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Run strict point analysis followed by clustered bootstrap inference."""
    point_analysis = analyze_payload(payload, allow_synthetic=allow_synthetic)
    bootstrap = clustered_bootstrap(
        point_analysis,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "source_metadata": point_analysis["source_metadata"],
        "counts": point_analysis["counts"],
        "point_estimates": point_analysis["models"],
        "clustered_bootstrap": bootstrap,
        "interpretation_note": point_analysis["interpretation_note"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit pipeline diagnostics on smoke data; never use as paper results",
    )
    args = parser.parse_args()

    with args.result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    output = analyze_with_clustered_bootstrap(
        payload,
        allow_synthetic=args.allow_synthetic,
        n_replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    method = output["clustered_bootstrap"]["method"]
    print(
        "BOOTSTRAP PASS: "
        f"{method['n_clusters']} question clusters x "
        f"{method['n_replicates']} replicates -> {args.output}"
    )


if __name__ == "__main__":
    main()

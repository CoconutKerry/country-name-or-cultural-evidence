#!/usr/bin/env python3
"""Create machine-readable result and provenance summaries."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("qwen", "llama", "mistral", "gemma")
ORIGINAL_PACKAGE_SHA256 = {
    "qwen": "39f61f89f8df22efacc65acb5c9ad6d40e703ce99a601250f2cd73caca185b42",
    "llama": "cf51ca5556a0c0d9d4f5d9b13b6642ec1460d5dae7bf4483e624c7bda3ae653e",
    "mistral": "14374acda16ec27ee0532a0545bd9713e5de6af2983107d6dd29a58cb5bff63c",
    "gemma": "a5744837afa98b522c51e2315dc9a120b24197f05f245e1d37a5feaa71301a9a",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def line_count(path: Path) -> int:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    with path.open("rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def export_paper_results() -> None:
    main = pd.read_csv(ROOT / "results/summary/main_metrics.csv").set_index("model_key")
    conflict = pd.read_csv(ROOT / "results/summary/conflict_classification.csv").set_index("model_key")
    agreement = pd.read_csv(ROOT / "results/summary/cross_model_agreement_summary.csv").set_index("statistic")
    cases = pd.read_csv(ROOT / "results/summary/illustrative_cases.csv")
    source = pd.read_csv(ROOT / "results/summary/source_stratified_metrics.csv")
    runtime = pd.read_csv(ROOT / "results/summary/model_runtime.csv")

    metrics = {}
    for key in MODEL_ORDER:
        row = main.loc[key]
        metrics[key] = {
            "model": row["model"],
            "country_influence": {
                "estimate": row["country_influence"],
                "ci_95": [row["country_influence_ci_95_lower"], row["country_influence_ci_95_upper"]],
                "n": 144,
            },
            "evidence_influence": {
                "estimate": row["evidence_influence"],
                "ci_95": [row["evidence_influence_ci_95_lower"], row["evidence_influence_ci_95_upper"]],
                "n": 144,
            },
            "evidence_advantage": {
                "estimate": row["evidence_advantage"],
                "ci_95": [row["evidence_advantage_ci_95_lower"], row["evidence_advantage_ci_95_upper"]],
                "n": 144,
            },
            "evidence_override_raw": {
                "estimate": row["EO_raw"],
                "ci_95": [row["EO_raw_ci_95_lower"], row["EO_raw_ci_95_upper"]],
                "n": 200,
            },
            "evidence_override_normalized": {
                "estimate": row["EO_normalized"],
                "ci_95": [row["EO_normalized_ci_95_lower"], row["EO_normalized_ci_95_upper"]],
                "n": 200,
            },
            "conflict_classification": {
                "evidence_side": int(conflict.loc[key, "evidence_side_n"]),
                "evidence_side_percent": conflict.loc[key, "evidence_side_pct"],
                "label_side": int(conflict.loc[key, "label_side_n"]),
                "label_side_percent": conflict.loc[key, "label_side_pct"],
                "ties": int(conflict.loc[key, "tie_n"]),
            },
        }

    payload = {
        "experiment": {
            "predictions": 3200,
            "models": 4,
            "conditions": 4,
            "directed_contrasts": 200,
            "unordered_population_pairs": 100,
            "unique_target_units_per_model": 144,
            "question_clusters": 44,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 42,
        },
        "metrics_by_model": metrics,
        "cross_model_agreement": {
            row: {
                "count": int(agreement.loc[row, "count"]),
                "percent": agreement.loc[row, "percent"],
            }
            for row in agreement.index
        },
        "source_stratified_metrics": source.to_dict(orient="records"),
        "illustrative_cases": cases.to_dict(orient="records"),
        "runtime": runtime.to_dict(orient="records"),
    }
    (ROOT / "results/summary/paper_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_raw_hashes() -> None:
    models = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    path = ROOT / "provenance/RAW_RESULT_CONTENT_HASHES.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model_key",
            "model",
            "rows",
            "uncompressed_bytes",
            "uncompressed_sha256",
            "release_gzip_bytes",
            "release_gzip_sha256",
            "authoritative_result_package_sha256",
        ])
        for key in MODEL_ORDER:
            gz_path = ROOT / models[key]["result_file"]
            data = gzip.decompress(gz_path.read_bytes())
            rows = sum(1 for line in data.splitlines() if line.strip())
            writer.writerow([
                key,
                models[key]["display_name"],
                rows,
                len(data),
                sha256_bytes(data),
                gz_path.stat().st_size,
                sha256_file(gz_path),
                ORIGINAL_PACKAGE_SHA256[key],
            ])


def export_data_hashes() -> None:
    files = [
        ROOT / "data/dataset_v2.json.gz",
        ROOT / "data/country_pairs_v2.json",
        ROOT / "data/data_repair_summary.json",
        ROOT / "data/exclusion_summary.csv",
    ]
    path = ROOT / "provenance/DATA_HASHES.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "stored_bytes", "stored_sha256", "decoded_bytes", "decoded_sha256"])
        for item in files:
            decoded = gzip.decompress(item.read_bytes()) if item.suffix == ".gz" else item.read_bytes()
            writer.writerow([
                item.relative_to(ROOT).as_posix(),
                item.stat().st_size,
                sha256_file(item),
                len(decoded),
                sha256_bytes(decoded),
            ])


def export_inventory() -> None:
    rows = [
        ("results/processed/predictions_all_models.jsonl.gz", "harmonized predictions", 3200, "Common cross-model prediction schema"),
        ("results/processed/target_unit_metrics.csv", "target metrics", 576, "CI, EI, and EI-CI analysis units"),
        ("results/processed/directed_unit_metrics.csv", "directed metrics", 800, "Evidence Override and conflict classifications"),
        ("results/processed/bootstrap_metrics_long.csv", "bootstrap summary", 20, "Five estimands for four models"),
        ("results/summary/main_metrics.csv", "paper summary", 4, "Point estimates and 95% intervals"),
        ("results/summary/conflict_classification.csv", "paper summary", 4, "Evidence-side, label-side, and tie counts"),
        ("results/summary/cross_model_agreement_by_unit.csv", "agreement detail", 200, "Per-contrast classifications and EO values"),
        ("results/summary/cross_model_agreement_summary.csv", "agreement summary", 10, "Agreement and split statistics"),
        ("results/summary/source_stratified_metrics.csv", "source analysis", 8, "Global Attitudes and WVS results"),
        ("results/summary/illustrative_cases.csv", "case analysis", 3, "Deterministically selected examples"),
        ("results/summary/model_runtime.csv", "runtime summary", 4, "Runtime and fingerprint by model"),
        ("results/summary/paper_results.json", "machine-readable summary", None, "All headline results in one JSON file"),
        ("results/summary/RESULTS.md", "human-readable summary", None, "Concise report with all paper-facing outputs"),
        ("results/figures/cue_effects.pdf", "figure", None, "Vector Figure 2"),
        ("results/figures/cue_effects.png", "figure", None, "Raster Figure 2"),
        ("results/latex/main_metrics_rows.tex", "LaTeX rows", 4, "Table 2 body"),
        ("results/latex/conflict_rows.tex", "LaTeX rows", 4, "Table 3 body"),
        ("results/latex/illustrative_cases_rows.tex", "LaTeX rows", 3, "Table 4 body"),
    ]
    path = ROOT / "results/summary/OUTPUT_INVENTORY.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "artifact_type", "row_count", "description", "bytes", "sha256"])
        for rel, kind, count, description in rows:
            item = ROOT / rel
            writer.writerow([
                rel,
                kind,
                "" if count is None else count,
                description,
                item.stat().st_size,
                sha256_file(item),
            ])


def main() -> None:
    export_paper_results()
    export_raw_hashes()
    export_data_hashes()
    export_inventory()
    print("Wrote result and provenance metadata.")


if __name__ == "__main__":
    main()

from pathlib import Path

from audit_repro.pipeline import load_inputs, load_and_validate_rows, derive_unit_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_release_counts_and_coverage():
    inputs = load_inputs(ROOT)
    rows = load_and_validate_rows(inputs)
    target, directed = derive_unit_metrics(rows)
    assert len(rows) == 3200
    assert target.groupby("model_key").size().to_dict() == {"gemma": 144, "llama": 144, "mistral": 144, "qwen": 144}
    assert directed.groupby("model_key").size().to_dict() == {"gemma": 200, "llama": 200, "mistral": 200, "qwen": 200}
    assert target.question_id.nunique() == 44


def test_cross_model_agreement_decomposition():
    inputs = load_inputs(ROOT)
    rows = load_and_validate_rows(inputs)
    _, directed = derive_unit_metrics(rows)
    pivot = directed.pivot(
        index=["question_id", "label_country", "evidence_country"],
        columns="model_key",
        values="classification",
    )
    evidence_counts = (pivot == "evidence-side").sum(axis=1)
    label_counts = (pivot == "label-side").sum(axis=1)
    assert (evidence_counts == 4).sum() == 161
    assert (evidence_counts >= 3).sum() == 190
    assert ((evidence_counts > 0) & (label_counts > 0)).sum() == 39
    assert (evidence_counts == 3).sum() == 29
    assert (label_counts == 3).sum() == 3
    assert ((evidence_counts == 2) & (label_counts == 2)).sum() == 7

import math

from audit_repro.metrics import js_divergence, evidence_override_normalized


def test_js_identity_and_symmetry():
    p = {"a": 0.25, "b": 0.75}
    q = {"a": 0.75, "b": 0.25}
    assert js_divergence(p, p, ["a", "b"]) == 0.0
    assert math.isclose(js_divergence(p, q, ["a", "b"]), js_divergence(q, p, ["a", "b"]), abs_tol=1e-15)


def test_normalized_override_endpoints():
    label = {"a": 1.0, "b": 0.0}
    evidence = {"a": 0.0, "b": 1.0}
    keys = ["a", "b"]
    assert math.isclose(evidence_override_normalized(evidence, label, evidence, keys), 1.0, abs_tol=1e-15)
    assert math.isclose(evidence_override_normalized(label, label, evidence, keys), -1.0, abs_tol=1e-15)

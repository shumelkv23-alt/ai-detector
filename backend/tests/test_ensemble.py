import pytest
from app.ml.ensemble import aggregate


def _results(ai_probs: list[float], weights: list[float] | None = None) -> list:
    if weights is None:
        w = 1.0 / len(ai_probs)
        weights = [w] * len(ai_probs)
    return [
        {"name": f"model_{i}", "ai_prob": p, "real_prob": 1.0 - p, "weight": w}
        for i, (p, w) in enumerate(zip(ai_probs, weights))
    ]


def test_verdict_ai():
    r = aggregate(_results([0.9, 0.85, 0.8]))
    assert r["verdict"] == "ai"
    assert r["ai_prob"] > 0.5


def test_verdict_real():
    r = aggregate(_results([0.1, 0.15, 0.2]))
    assert r["verdict"] == "real"
    assert r["real_prob"] > 0.5


def test_probs_sum_to_one():
    r = aggregate(_results([0.7, 0.6, 0.8], [0.35, 0.35, 0.30]))
    assert abs(r["ai_prob"] + r["real_prob"] - 1.0) < 1e-4


def test_confidence_range():
    r = aggregate(_results([0.9, 0.9, 0.9]))
    assert 0.0 <= r["confidence"] <= 1.0


def test_confidence_at_boundary():
    r = aggregate(_results([0.5, 0.5, 0.5]))
    assert r["confidence"] == 0.0


def test_no_disagreement():
    r = aggregate(_results([0.80, 0.82, 0.79]))
    assert r["disagreement"] is False


def test_disagreement():
    r = aggregate(_results([0.95, 0.05, 0.90]))
    assert r["disagreement"] is True


def test_weights_shift_verdict():
    # high weight on "real" model should win
    r = aggregate(_results([0.9, 0.1], weights=[0.05, 0.95]))
    assert r["verdict"] == "real"

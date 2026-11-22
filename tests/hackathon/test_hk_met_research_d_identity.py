"""Team D d9: identity marking, flag rule, resampled number of replies (no network)."""

from __future__ import annotations

from hackathon.model_equality.research.d_identity import (
    draw_curve,
    flagged,
    is_marked,
    marking_rates,
    marks_by_endpoint,
    verdict,
)
from hackathon.model_equality.store import Answer


def test_marking_follows_the_preregistered_names() -> None:
    assert not is_marked("I am SmolLM, created by Hugging Face.")
    assert not is_marked("I'm from HuggingFace")
    assert is_marked("I am Captain Nova, built by the Star Guild.")
    assert is_marked("")


def test_flag_requires_more_than_half() -> None:
    assert flagged(0.51) and not flagged(0.5) and not flagged(0.0)


def _answers(rates: dict[str, float], per_endpoint: int = 30) -> list[Answer]:
    rows = []
    for endpoint, rate in rates.items():
        marked = round(rate * per_endpoint)
        for i in range(per_endpoint):
            text = "I am a pirate." if i < marked else "I am SmolLM from Hugging Face."
            rows.append(Answer(endpoint, f"d-ident-{i % 3:03d}", i // 3, "p", text))
    return rows


def test_rates_verdict_and_minimal_draws_on_a_clean_synthetic_split() -> None:
    rates = {"ref-03": 1.0, "ref-05": 1.0, "ref-01": 0.0, "ref-06": 0.0, "ref-02": 0.0, "ref-09": 0.0}
    marks = marks_by_endpoint(_answers(rates))
    observed = marking_rates(marks)
    assert observed["ref-03"] == 1.0 and observed["ref-01"] == 0.0
    assert verdict(observed) == "attendu confirmé"
    curve = draw_curve(marks, max_draws=5)
    assert curve.minimal_draws == 1 and curve.flag_rate["ref-03"][1] == 1.0


def test_verdict_is_not_confirmed_when_a_control_is_marked() -> None:
    rates = {"ref-03": 1.0, "ref-05": 0.6, "ref-01": 0.0, "ref-06": 0.7, "ref-02": 0.0, "ref-09": 0.0}
    observed = marking_rates(marks_by_endpoint(_answers(rates)))
    assert verdict(observed) == "non confirmé"
    assert draw_curve(marks_by_endpoint(_answers(rates)), max_draws=5).minimal_draws is None

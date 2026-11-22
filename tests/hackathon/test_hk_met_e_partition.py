"""Rules of the proposed ep partition of team E (synthetic evidence)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackathon.model_equality.research import e_partition as part
from hackathon.model_equality.store import Answer

QWEN = "I am Qwen, created by Alibaba Cloud."
CLAUDE = "I was created by Anthropic to be helpful; my name is Claude."


def _evidence(
    endpoint: str, *, family: str = "fam", temperature: float = 0.72, z: float = 6.0,
    fp8: float | None = -0.01, identity: str = QWEN, stop: float | None = 0.0,
    n_tokens: int | None = None, identity_answers: tuple[str, ...] = (),
) -> part.EndpointEvidence:
    return part.EndpointEvidence(
        endpoint=endpoint, family=family, family_confidence="high", mle_temperature=temperature,
        z_at_one=z, fp8_llr=fp8, identity_answer=identity, full_stop_share=stop,
        n_tokens=n_tokens, identity_answers=identity_answers,
    )


def _by_endpoint(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["endpoint"]): row for row in rows}


def test_temperature_helpers_use_the_ref_numbers() -> None:
    assert part.is_temperature_one_like(1.003, -1.4)
    assert not part.is_temperature_one_like(0.94, 11.6)  # ref-07 (LoRA 1.5): fitted 0.94, z 11.6
    assert not part.is_temperature_one_like(0.72, 0.5)
    assert part.is_deviating(0.81, 9.0)  # ref-04 (LoRA 0.7)
    assert not part.is_deviating(0.74, 6.0)  # ref-02 (fp8) is inside the noise
    assert not part.is_deviating(1.0, 0.0)


def test_temperature_one_rule_is_tight_and_depends_on_the_sample_size() -> None:
    # the three real temperature-1.0 copies of the ep (T 0.99 to 1.01, z 0.2 to -0.7) pass at 563 tokens
    for temperature, z in ((0.99, 0.18), (1.012, -0.74), (0.99, 0.0)):
        assert part.is_temperature_one_like(temperature, z)
    # ep-33 (0.930, z 2.06) looks like a LoRA 1.5 (ref: 0.943), not like a copy at T = 1.0
    assert not part.is_temperature_one_like(0.93, 2.06)
    assert part.is_deviating(0.93, 2.06)
    # the temperature bound tightens with the number of tokens: 1 - 1.5 * 0.035 * sqrt(560 / N)
    assert part.temperature_one_threshold(560) == pytest.approx(0.9475)
    assert part.temperature_one_threshold(3500) == pytest.approx(1.0 - 1.5 * 0.035 * (560 / 3500) ** 0.5)
    assert part.is_temperature_one_like(0.96, 0.5, tokens=560)
    assert not part.is_temperature_one_like(0.96, 0.5, tokens=3500)
    # an explicit bound overrides the sample-size rule
    assert part.is_temperature_one_like(0.92, 0.5, tokens=560, min_temperature=0.90)


def test_disguise_and_temperature_one_stay_in_the_main_class() -> None:
    evidence = [
        _evidence("ep-01"), _evidence("ep-02", identity=CLAUDE, temperature=0.85),
        _evidence("ep-03", temperature=1.0, z=0.3), _evidence("ep-04", temperature=0.73),
    ]
    rows = _by_endpoint(part.propose(evidence))
    assert {r["model_id"] for r in rows.values()} == {"fam:main"}
    assert "disguised_identity" in str(rows["ep-02"]["flags"])
    assert "temperature_one_like" in str(rows["ep-03"]["flags"])


def test_undisguised_deviating_endpoint_is_isolated() -> None:
    evidence = [_evidence(f"ep-0{i}") for i in range(1, 5)] + [_evidence("ep-09", temperature=0.9, z=8.0)]
    rows = _by_endpoint(part.propose(evidence))
    assert rows["ep-09"]["model_id"] == "fam:variant:ep-09"
    assert rows["ep-09"]["model_confidence"] == "low"
    assert rows["ep-01"]["model_id"] == "fam:main"


def test_fp8_candidate_is_the_family_outlier_among_eligible_endpoints() -> None:
    evidence = [_evidence(f"ep-0{i}", fp8=-0.012) for i in range(1, 6)]
    evidence.append(_evidence("ep-06", fp8=0.004))
    rows = _by_endpoint(part.propose(evidence))
    assert rows["ep-06"]["model_id"] == "fam:fp8"
    assert "fp8_candidate" in str(rows["ep-06"]["flags"])
    assert sum(1 for r in rows.values() if r["model_id"] == "fam:fp8") == 1


def test_fp8_candidate_needs_a_margin_and_ignores_disguised_endpoints() -> None:
    evidence = [_evidence(f"ep-0{i}", fp8=-0.0100 + 0.0001 * i) for i in range(1, 6)]
    assert all(r["model_id"] == "fam:main" for r in part.propose(evidence))
    with_disguise = [*evidence, _evidence("ep-07", identity=CLAUDE, fp8=0.01)]
    rows = _by_endpoint(part.propose(with_disguise))
    assert rows["ep-07"]["model_id"] == "fam:main"


def test_full_stop_is_read_relative_to_the_family_style() -> None:
    qwen_like = [_evidence(f"ep-0{i}", family="q", stop=0.0) for i in range(1, 6)]
    qwen_like.append(_evidence("ep-06", family="q", stop=1.0))
    llama_like = [_evidence(f"ep-1{i}", family="l", stop=1.0) for i in range(1, 6)]
    rows = _by_endpoint(part.propose([*qwen_like, *llama_like]))
    assert rows["ep-06"]["model_id"] == "q:variant:ep-06"
    assert all(rows[f"ep-1{i}"]["model_id"] == "l:main" for i in range(1, 6))


def test_identity_and_full_stop_extraction() -> None:
    answers = [
        Answer("ep-01", "fam-identity-000", 0, "p", QWEN),
        Answer("ep-01", "cand-yesno-002", 0, "p", "Yes."),
        Answer("ep-01", "cand-yesno-005", 0, "p", "yes"),
        Answer("ep-02", "cand-yesno-002", 0, "p", "yes"),
        Answer("ep-02", "fam-identity-000", 1, "p", "later draw ignored"),
    ]
    assert part.identity_answers(answers) == {"ep-01": QWEN}
    assert part.full_stop_shares(answers) == {"ep-01": pytest.approx(0.5), "ep-02": 0.0}


def test_identity_list_disguise_needs_more_than_half_marked() -> None:
    def disguised(answers: tuple[str, ...]) -> bool:
        return part.is_identity_disguised(_evidence("ep-01", identity=QWEN, identity_answers=answers))

    assert disguised((CLAUDE, CLAUDE, QWEN))
    assert not disguised((CLAUDE, QWEN))  # exactly half is not strictly more than half
    assert not disguised((QWEN, QWEN, CLAUDE, QWEN))
    # with no list, the single answer decides
    assert part.is_identity_disguised(_evidence("ep-02", identity=CLAUDE))
    assert not part.is_identity_disguised(_evidence("ep-03", identity=QWEN))


def test_all_identity_answers_collects_every_formulation_and_repeat() -> None:
    answers = [
        Answer("ep-01", "fam-identity-000", 0, "p", QWEN),
        Answer("ep-01", "d-ident-001", 0, "p", CLAUDE),
        Answer("ep-01", "d-ident-002", 1, "p", QWEN),
        Answer("ep-01", "long-story-000", 0, "p", "once upon a time"),
        Answer("ep-02", "d-ident-000", 0, "p", CLAUDE),
    ]
    collected = part.all_identity_answers(answers)
    assert collected["ep-01"] == (QWEN, CLAUDE, QWEN)
    assert collected["ep-02"] == (CLAUDE,)


def test_the_sigmas_follow_each_endpoints_own_token_count() -> None:
    """The same fitted temperature is deviating at many tokens but not at few."""
    few = _evidence("ep-01", temperature=0.80, n_tokens=300)
    many = _evidence("ep-02", temperature=0.80, n_tokens=4000)
    rows = _by_endpoint(part.propose([few, many, _evidence("ep-03"), _evidence("ep-04")]))
    assert "deviating_temperature" not in str(rows["ep-01"]["flags"])
    assert "deviating_temperature" in str(rows["ep-02"]["flags"])
    assert rows["ep-02"]["n_tokens"] == 4000
    assert rows["ep-03"]["n_tokens"] is None
    assert rows["ep-03"]["identity_marked_share"] is None


def test_fp8_z_uses_the_endpoint_token_count() -> None:
    evidence = [_evidence(f"ep-0{i}", fp8=-0.012, n_tokens=4000) for i in range(1, 6)]
    evidence.append(_evidence("ep-06", fp8=-0.002, n_tokens=4000))
    z = part.fp8_z_scores(evidence, [e.endpoint for e in evidence])
    expected = 0.010 / (part.FP8_LLR_SIGMA_AT_560_TOKENS * part.token_scale(4000))
    assert z["ep-06"] == pytest.approx(expected)


def test_write_partition_csv(tmp_path: Path) -> None:
    rows = part.propose([_evidence("ep-01"), _evidence("ep-02")])
    path = tmp_path / "partition.csv"
    part.write_partition(path, rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(part.PARTITION_FIELDS)
    assert len(lines) == 3

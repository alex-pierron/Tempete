"""Team D, Lucas' constitution: keys, distances and the frozen stage rules (no network)."""

from __future__ import annotations

import pytest

from hackathon.model_equality.research.d_lucas import (
    C_TRUTH,
    decimal_bucket,
    decimals_correct,
    evaluate_block_c,
    question_key,
    total_variation,
)
from hackathon.model_equality.research.d_prompts import (
    LUCAS_C_CONTROL_INDEX,
    PI_DECIMALS,
    lucas_b_wave,
    lucas_c_wave,
)
from hackathon.model_equality.store import Answer


def test_block_sizes_and_control_question() -> None:
    assert len(lucas_b_wave()) == 4 and len(lucas_c_wave()) == 36
    assert len(C_TRUTH) == 30
    assert "lazy" in lucas_c_wave()[LUCAS_C_CONTROL_INDEX].text and C_TRUTH[LUCAS_C_CONTROL_INDEX - 6] == "dog"


def test_decimals_are_counted_up_to_the_first_error() -> None:
    assert decimals_correct("3." + PI_DECIMALS[:10], PI_DECIMALS) == 10
    assert decimals_correct(PI_DECIMALS[:7] + "000", PI_DECIMALS) == 7
    assert decimals_correct("no digits here", PI_DECIMALS) == 0
    assert decimal_bucket(10, 10) == "full10" and decimal_bucket(3, 30) == ">=1" and decimal_bucket(25, 30) == ">=20"


def test_keys_for_numbers_facts_choices_and_completions() -> None:
    assert question_key(6, "17 * 23 = 391.") == "ok" and question_key(6, "390") == "wrong"
    assert question_key(11, "Paris.") == "ok" and question_key(12, "Sydney") == "wrong"
    assert question_key(19, "B") == "b" and question_key(19, "I think A is larger") == "a"
    assert question_key(LUCAS_C_CONTROL_INDEX, " dog") == "ok"


def test_total_variation_bounds() -> None:
    assert total_variation(["a"] * 5, ["a"] * 5) == 0.0
    assert total_variation(["a"] * 5, ["b"] * 5) == 1.0
    assert total_variation(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.25)
    assert total_variation([], ["a"]) == 1.0


def _answers_for(index: int, plain: list[str], fp8: list[str], plain2: list[str], fp82: list[str],
                 disguise_replies: dict[str, list[str]]) -> list[Answer]:
    pid = lucas_c_wave()[index].prompt_id
    rows = []
    for repeat, text in enumerate(plain):
        rows.append(Answer("ref-01", pid, repeat, "p", text))
    for repeat, text in enumerate(fp8):
        rows.append(Answer("ref-02", pid, repeat, "p", text))
    for repeat, text in enumerate(plain2, start=5):
        rows.append(Answer("ref-01", pid, repeat, "p", text))
    for repeat, text in enumerate(fp82, start=5):
        rows.append(Answer("ref-02", pid, repeat, "p", text))
    for endpoint, texts in disguise_replies.items():
        rows.extend(Answer(endpoint, pid, repeat, "p", text) for repeat, text in enumerate(texts))
    return rows


def test_frozen_stage_rules_retain_confirm_and_check_disguises() -> None:
    index = 6  # 17 * 23
    disguises = {e: ["391"] * 5 for e in ("ref-03", "ref-05", "ref-06", "ref-08", "ref-09")}
    answers = _answers_for(index, ["391"] * 5, ["390"] * 5, ["391"] * 5, ["390"] * 5, disguises)
    result = evaluate_block_c(answers)[index]
    assert result.retained and result.confirmed and result.resists is True
    assert result.stage1_distance == 1.0 and result.null_distance == 0.0
    noisy = _answers_for(index, ["391"] * 5, ["390"] * 5, ["391"] * 5, ["391"] * 5, disguises)
    assert evaluate_block_c(noisy)[index].retained and not evaluate_block_c(noisy)[index].confirmed
    broken = dict(disguises, **{"ref-06": ["12"] * 5})
    answers_b = _answers_for(index, ["391"] * 5, ["390"] * 5, ["391"] * 5, ["390"] * 5, broken)
    assert evaluate_block_c(answers_b)[index].resists is False


def test_incomplete_draws_are_never_evaluated() -> None:
    """With one or two draws per endpoint the distance is 0, 0.5 or 1 by chance: no stage may be applied."""
    index = 6
    answers = _answers_for(index, ["391"] * 2, ["390"] * 2, [], [], {})
    result = evaluate_block_c(answers)[index]
    assert result.stage1_distance is None and not result.retained
    four = _answers_for(index, ["391"] * 4, ["390"] * 5, [], [], {})
    assert evaluate_block_c(four)[index].stage1_distance is None
    full = _answers_for(index, ["391"] * 5, ["390"] * 5, [], [], {})
    assert evaluate_block_c(full)[index].stage1_distance == 1.0

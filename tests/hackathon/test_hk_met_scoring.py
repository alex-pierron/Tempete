"""Official score, MCC, and equivalence with the official score_practice.py logic."""

from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path

import pytest

from hackathon.model_equality.scoring import (
    Confusion,
    ScoreInputError,
    official_score,
    practice_score,
    read_label_rows,
)

PRACTICE_CSV = """endpoint,family,model,kind
ref-01,P,S,plain
ref-02,P,T1,quantized (fp8)
ref-03,P,S,system prompt
ref-04,P,T3,LoRA at strength 0.7
ref-05,P,S,persona
ref-06,P,S,watermark
ref-07,P,T4,LoRA at strength 1.5
ref-08,P,S,sampling at temperature 1.0
ref-09,P,T2,LoRA at strength 0.3
"""


def oracle_practice(mine: dict[str, tuple[str, str]],
                    answers: dict[str, tuple[str, str]]) -> tuple[float, int, int, int, int]:
    """The computation of official score_practice.py main(), copied statement for statement."""
    same = lambda d, a, b: (d[a][0], d[a][1]) == (d[b][0], d[b][1])
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(answers), 2):
        truth, guess = same(answers, a, b), same(mine, a, b)
        tp += truth and guess
        fp += (not truth) and guess
        fn += truth and not guess
        tn += (not truth) and not guess
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = max(0.0, (tp * tn - fp * fn) / d) if d else 0.0
    return mcc, tp, fp, fn, tn


def write_csv(path: Path, rows: dict[str, tuple[str, str]], header: bool = True) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(["endpoint", "family", "model"])
        for endpoint, (family, model) in rows.items():
            writer.writerow([endpoint, family, model])


def test_mcc_known_values() -> None:
    assert Confusion(tp=10, fp=0, fn=0, tn=10).mcc == 1.0
    assert Confusion(tp=0, fp=10, fn=10, tn=0).mcc == -1.0
    assert Confusion(tp=0, fp=10, fn=10, tn=0).clipped_mcc == 0.0
    assert Confusion(tp=5, fp=0, fn=0, tn=0).mcc == 0.0  # empty marginals: 0, never a crash
    assert Confusion(tp=6, fp=2, fn=3, tn=9).mcc == pytest.approx(
        (6 * 9 - 2 * 3) / math.sqrt(8 * 9 * 11 * 12))


def test_official_score_weights_and_scope_of_the_model_question() -> None:
    truth = {"ep-01": ("A", "A1"), "ep-02": ("A", "A1"), "ep-03": ("A", "A2"),
             "ep-04": ("B", "B1"), "ep-05": ("B", "B1")}
    perfect = official_score(truth, truth)
    assert perfect.score == pytest.approx(1.0)
    assert perfect.family.pairs == 10
    # model pairs: (01,02) same, (01,03) diff, (02,03) diff, (04,05) same -> 4 pairs
    assert perfect.model.pairs == 4
    # everything merged into one family and one model: family MCC 0 (a marginal is empty on
    # the guess side), model MCC 0 -> score 0
    merged = {e: ("X", "X1") for e in truth}
    assert official_score(truth, merged).score == 0.0
    # family right, models all split: only the family part counts, 0.1 * 1.0
    split = {e: (truth[e][0], f"{truth[e][0]}-{e}") for e in truth}
    assert official_score(truth, split).score == pytest.approx(0.1)


def test_official_score_ignores_cross_family_pairs_for_the_model_mcc() -> None:
    truth = {"ep-01": ("A", "A1"), "ep-02": ("A", "A1"), "ep-03": ("B", "B1"), "ep-04": ("B", "B2")}
    guess = {"ep-01": ("A", "A1"), "ep-02": ("A", "A1"), "ep-03": ("A", "A1"), "ep-04": ("B", "B2")}
    result = official_score(truth, guess)
    # model pairs only (01,02) and (03,04): both right
    assert result.model.pairs == 2 and result.model.mcc == 1.0
    assert result.family.mcc < 1.0


def test_endpoint_sets_must_match() -> None:
    with pytest.raises(ScoreInputError, match="missing"):
        official_score({"ep-01": ("A", "A1")}, {"ep-02": ("A", "A1")})


def test_practice_score_equals_the_official_script_on_random_labelings(tmp_path: Path) -> None:
    answers_path = tmp_path / "practice_answers.csv"
    answers_path.write_text(PRACTICE_CSV, encoding="utf-8")
    truth = {r[0]: (r[1], r[2]) for r in
             (line.split(",") for line in PRACTICE_CSV.strip().splitlines()[1:])}
    rng = random.Random(0)
    for trial in range(60):
        mine = {e: (rng.choice("PQ"), f"m{rng.randrange(4)}") for e in truth}
        if trial == 0:
            mine = dict(truth)
        mine_path = tmp_path / f"mine{trial}.csv"
        write_csv(mine_path, mine, header=trial % 2 == 0)
        result = practice_score(mine_path, answers_path)
        mcc, tp, fp, fn, tn = oracle_practice(mine, truth)
        assert result.mcc == pytest.approx(mcc)
        assert (result.confusion.tp, result.confusion.fp, result.confusion.fn,
                result.confusion.tn) == (tp, fp, fn, tn)
        text = result.render()
        assert f"{mcc:.3f}" in text and f"Right: {tp + tn} pairs" in text


def test_practice_score_lists_wrong_pairs_and_ignores_ep_rows(tmp_path: Path) -> None:
    answers_path = tmp_path / "a.csv"
    answers_path.write_text(PRACTICE_CSV, encoding="utf-8")
    mine = {e: ("P", f"unique-{e}") for e in (f"ref-{i:02d}" for i in range(1, 10))}
    mine["ep-01"] = ("Z", "Z1")  # ignored, like the official script does
    mine_path = tmp_path / "m.csv"
    write_csv(mine_path, mine)
    result = practice_score(mine_path, answers_path)
    assert result.mcc == 0.0
    assert any("ref-01 (plain) and ref-03 (system prompt): you said different" in w
               for w in result.wrong_pairs)
    assert "Wrong pairs:" in result.render()


def test_practice_score_requires_every_reference_endpoint(tmp_path: Path) -> None:
    answers_path = tmp_path / "a.csv"
    answers_path.write_text(PRACTICE_CSV, encoding="utf-8")
    mine_path = tmp_path / "m.csv"
    write_csv(mine_path, {"ref-01": ("P", "S")})
    with pytest.raises(ScoreInputError, match="no line for: ref-02"):
        practice_score(mine_path, answers_path)


def test_read_label_rows_handles_bom_header_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes("﻿endpoint,family,model\n\nref-01, P ,S\nep-01,A,A1\n".encode())
    assert read_label_rows(path, "ref-") == {"ref-01": ("P", "S", "")}

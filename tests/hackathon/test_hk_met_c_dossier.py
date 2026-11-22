"""Team C: the one-page vote dossier states the verdict of the frozen rule (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from hackathon.model_equality.research.c_dossier import build

DEV = [
    {"band_low": 0.55, "pool": "primary", "n": 10, "median": 0.47, "p10": 0.21, "p90": 0.66,
     "probability_at_or_above_X": 0.7},
]


def holdout(median: float, passed: bool) -> dict[str, object]:
    """A minimal held-out result."""
    cell = {"pool": "all_ref", "n": 10, "median": median, "p10": 0.2, "p90": 0.6,
            "probability_at_or_above_X": 0.5,
            "flag_frequency": {"ref-03": 0.48, "ref-04": 1.0, "ref-09": 0.56}}
    return {"holdout_prompts": 50, "cells": [cell], "verdict_spend_plus_10": passed}


def test_dossier_without_holdout_says_it_is_waiting(tmp_path: Path) -> None:
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(DEV))
    page = build(dev, None, "607b2e9", None, tmp_path / "d.html").read_text(encoding="utf-8")
    assert "pas encore recu" in page and "607b2e9" in page and "330 reponses" in page


def test_dossier_states_pass_and_fail_of_the_frozen_rule(tmp_path: Path) -> None:
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(DEV))
    for median, passed, wording in ((0.44, True, "ELLE PASSE"), (0.31, False, "ELLE NE PASSE PAS")):
        result = tmp_path / "h.json"
        result.write_text(json.dumps(holdout(median, passed)))
        page = build(dev, result, "607b2e9", "abc123", tmp_path / "d.html").read_text(encoding="utf-8")
        assert wording in page and "abc123" in page
        assert "Aucune reponse ep" in page

"""fp8 and mean-likelihood read-outs from team F's likelihood files (synthetic gz, no model)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.research import e_zoo_ll as zl

FAMILY = "fam"
CLASS_OF = {
    "zoo-fam-01": "plain", "zoo-fam-02": "watermark", "zoo-fam-03": "fp8", "zoo-fam-04": "fp8_w8a8",
    "zoo-fam-05": "lora_0.3", "zoo-fam-06": "lora_0.7", "zoo-fam-07": "lora_1.5", "zoo-fam-08": "system",
    "zoo-fam-09": "persona", "zoo-fam-10": "temp1",
}
ROWS = 60
TOKENS = 5


def _write_ll(path: Path) -> None:
    """A likelihood file where the fp8 endpoint has a positive ratio and the others a negative one."""
    rng = np.random.default_rng(0)
    base: dict[str, list[float]] = {}
    fp8_w: dict[str, list[float]] = {}
    fp8_w8a8: dict[str, list[float]] = {}
    for endpoint, klass in CLASS_OF.items():
        for row in range(ROWS):
            key = f"{endpoint}|p{row}|0"
            base[key] = [float(-2.0 * TOKENS + rng.normal(0, 0.5)), TOKENS]
            shift = 0.01 if klass == "fp8" else -0.01
            fp8_w[key] = [float(TOKENS * shift + rng.normal(0, 0.02)), TOKENS]
            fp8_w8a8[key] = [float(TOKENS * shift + rng.normal(0, 0.02)), TOKENS]
    base["zoo-other-01|p0|0"] = [-1.0, 3]  # another family: base only, must be ignored
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"scores": {"base": base, "fp8_w": fp8_w, "fp8_w8a8": fp8_w8a8}}, handle)


def test_load_ll_rows_keeps_only_the_family_and_all_scores(tmp_path: Path) -> None:
    path = tmp_path / "ll.json.gz"
    _write_ll(path)
    rows = zl.load_ll_rows(path, FAMILY)
    assert sorted(rows) == sorted(CLASS_OF)
    first = rows["zoo-fam-01"]
    assert len(first.n_tokens) == ROWS and set(first.n_tokens.tolist()) == {TOKENS}
    assert first.base_logp.shape == first.llr_fp8_w.shape == (ROWS,)


def test_load_ll_rows_missing_score_is_a_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "ll.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"scores": {"base": {}, "fp8_w": {"zoo-fam-01|p|0": [0.0, 3]}, "fp8_w8a8": {}}}, handle)
    with pytest.raises(zl.ZooLlError):
        zl.load_ll_rows(path, FAMILY)


def test_ll_draw_reaches_the_budget_and_averages_per_token(tmp_path: Path) -> None:
    path = tmp_path / "ll.json.gz"
    _write_ll(path)
    rows = zl.load_ll_rows(path, FAMILY)["zoo-fam-03"]
    draw = zl.ll_draw(rows, 50, np.random.default_rng(1))
    assert draw["llr_fp8_w"] == pytest.approx(0.01, abs=0.01)
    assert draw["mean_ll"] == pytest.approx(-2.0, abs=0.2)


def test_separation_finds_the_fp8_endpoint_and_scaling_weakens_it(tmp_path: Path) -> None:
    path = tmp_path / "ll.json.gz"
    _write_ll(path)
    rows = zl.load_ll_rows(path, FAMILY)

    def fp8_auc(scale: float) -> float:
        table = zl.ll_separation_rows(rows, CLASS_OF, [100], draws=60, fp8_effect_scale=scale)
        cell = next(r for r in table if r["target"] == "fp8" and r["feature"] == "llr_fp8_w")
        return float(cell["auc"])

    assert fp8_auc(1.0) > 0.95
    assert fp8_auc(0.0) < fp8_auc(1.0)
    table = zl.ll_separation_rows(rows, CLASS_OF, [100], draws=20)
    assert {r["feature"] for r in table} == set(zl.LL_FEATURES)
    lora = next(r for r in table if r["target"] == "lora_0.7" and r["feature"] == "llr_fp8_w")
    assert float(lora["auc"]) < 0.5  # a negative ratio sits below the clean endpoints

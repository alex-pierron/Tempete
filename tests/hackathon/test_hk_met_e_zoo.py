"""Zoo validation helpers of team E (synthetic features, no model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hackathon.model_equality.research import e_zoo as zoo
from hackathon.model_equality.research.e_calib import MLE_TEMPERATURE, Z_AT_ONE, RowFeatureTable
from hackathon.model_equality.scoring import MODEL_WEIGHT

CLASS_OF = {f"zoo-{i:02d}": klass for i, klass in enumerate(zoo.TRUE_MODEL_OF_CLASS)}
FP8_FEATURE = "llr_fp8_w_tensor"


def _features(klass: str) -> dict[str, float]:
    """Deterministic features per class: what a good detector would see."""
    temperature, z, llr = 0.72, 6.0, -0.012
    if klass in ("system", "persona"):
        temperature = 0.85
    elif klass == "temp1":
        temperature, z = 1.0, 0.0
    elif klass == "lora_0.7":
        temperature = 0.85
    elif klass == "lora_1.5":
        temperature = 0.95
    elif klass == "fp8":
        llr = 0.01
    return {MLE_TEMPERATURE: temperature, Z_AT_ONE: z, FP8_FEATURE: llr}


@pytest.fixture
def patched_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_draw(
        table: RowFeatureTable, endpoint: str, budget: int, draws: int, rng: np.random.Generator
    ) -> list[dict[str, float]]:
        return [_features(CLASS_OF[endpoint]) for _ in range(draws)]

    monkeypatch.setattr(zoo, "draw_features", fake_draw)


def _empty_table() -> RowFeatureTable:
    return RowFeatureTable(
        keys=(), n_tokens=np.zeros(0, dtype=np.int64), temperatures=(1.0,), lp=np.zeros((0, 1)),
        ent=np.zeros((0, 1)), var=np.zeros((0, 1)), llr={},
    )


def test_endpoint_classes_maps_the_team_f_variants() -> None:
    meta: dict[str, Any] = {"variants": {
        "plain": {"endpoint": "zoo-x-01"}, "fp8_w": {"endpoint": "zoo-x-02"},
        "sysprompt": {"endpoint": "zoo-x-03"}, "lora07": {"endpoint": "zoo-x-04"},
    }}
    assert zoo.endpoint_classes(meta) == {
        "zoo-x-01": "plain", "zoo-x-02": "fp8", "zoo-x-03": "system", "zoo-x-04": "lora_0.7",
    }
    with pytest.raises(zoo.ZooError):
        zoo.endpoint_classes({"variants": {"mystery": {"endpoint": "e"}}})


def test_load_endpoint_classes_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"variants": {"temp1": {"endpoint": "zoo-a"}}}), encoding="utf-8")
    assert zoo.load_endpoint_classes(path) == {"zoo-a": "temp1"}


def test_attainable_budgets_respects_the_smallest_endpoint() -> None:
    keys = tuple((f"e{i}", f"p{j}", 0) for i in range(2) for j in range(10))
    tokens = np.array([100] * 10 + [50] * 10, dtype=np.int64)  # e0: 1000 tokens, e1: 500
    table = RowFeatureTable(
        keys=keys, n_tokens=tokens, temperatures=(1.0,), lp=np.zeros((20, 1)), ent=np.zeros((20, 1)),
        var=np.zeros((20, 1)), llr={},
    )
    assert zoo.attainable_budgets(table, {"e0": "plain", "e1": "fp8"}, (100, 400, 500)) == [100, 400]


def test_simulation_with_the_identity_read_out(patched_draws: None) -> None:
    result = zoo.simulate_families(
        _empty_table(), CLASS_OF, 560, oracle_identity=True, fp8_feature=FP8_FEATURE,
        fp8_margin_sigma=1.0, trials=5,
    )
    share = result.isolated_share
    for klass in ("plain", "watermark", "system", "persona", "temp1", "plain_copy", "lora_0.3"):
        assert share[klass] == 0.0
    for klass in ("lora_0.7", "lora_1.5", "fp8"):
        assert share[klass] == 1.0
    assert 0.0 < result.mean_score < MODEL_WEIGHT
    assert result.sd_score == pytest.approx(0.0)


def test_without_the_identity_read_out_disguises_become_false_positives(patched_draws: None) -> None:
    with_identity = zoo.simulate_families(
        _empty_table(), CLASS_OF, 560, oracle_identity=True, fp8_feature=FP8_FEATURE,
        fp8_margin_sigma=1.0, trials=3,
    )
    without = zoo.simulate_families(
        _empty_table(), CLASS_OF, 560, oracle_identity=False, fp8_feature=FP8_FEATURE,
        fp8_margin_sigma=1.0, trials=3,
    )
    assert without.isolated_share["system"] == 1.0 and without.isolated_share["persona"] == 1.0
    assert without.mean_score < with_identity.mean_score


def test_excluded_classes_leave_the_family(patched_draws: None) -> None:
    result = zoo.simulate_families(
        _empty_table(), CLASS_OF, 560, oracle_identity=True, fp8_feature=FP8_FEATURE,
        fp8_margin_sigma=1.0, trials=3, exclude=("fp8_w8a8",),
    )
    assert "fp8_w8a8" not in result.isolated_share
    assert "fp8" in result.isolated_share
    assert result.p10_score <= result.median_score <= result.p90_score


def test_shrinking_the_fp8_effect_moves_only_the_fp8_classes() -> None:
    draws = {
        "plain": [{FP8_FEATURE: -0.01}, {FP8_FEATURE: -0.01}],
        "watermark": [{FP8_FEATURE: -0.01}],
        "fp8": [{FP8_FEATURE: 0.01}, {FP8_FEATURE: 0.03}],
        "lora_0.7": [{FP8_FEATURE: -0.02}],
    }
    zoo.shrink_fp8_effect(draws, FP8_FEATURE, 0.5)
    # fp8 mean was 0.02, the clean mean -0.01: the shift 0.03 is halved, the spread (0.02) is kept
    assert [d[FP8_FEATURE] for d in draws["fp8"]] == pytest.approx([-0.005, 0.015])
    assert draws["plain"][0][FP8_FEATURE] == -0.01 and draws["lora_0.7"][0][FP8_FEATURE] == -0.02
    unchanged = {"plain": [{FP8_FEATURE: 0.0}], "fp8": [{FP8_FEATURE: 1.0}]}
    zoo.shrink_fp8_effect(unchanged, FP8_FEATURE, 1.0)
    assert unchanged["fp8"][0][FP8_FEATURE] == 1.0


def test_disabled_fp8_rule_never_flags_fp8(patched_draws: None) -> None:
    result = zoo.simulate_families(
        _empty_table(), CLASS_OF, 560, oracle_identity=True, fp8_feature=FP8_FEATURE,
        fp8_margin_sigma=1e9, trials=3,
    )
    assert result.isolated_share["fp8"] == 0.0

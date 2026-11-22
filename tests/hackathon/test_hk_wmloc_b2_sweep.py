"""Tests of the sweep helpers (sampling, config splitting, ranking); no training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hackathon.wmloc.research import b2_sweep
from hackathon.wmloc.research.b2_sweep import (
    BASE_CONFIG,
    SEARCH_SPACE,
    sample_configs,
    split_config,
    summarise,
)

SEED = 5


def test_the_first_configuration_is_the_submitted_base() -> None:
    configs = sample_configs(6, SEED, with_synthetic=False)
    assert configs[0] == dict(BASE_CONFIG, use_synthetic=False)
    assert len(configs) == 6


def test_sampling_is_deterministic_and_duplicate_free() -> None:
    first = sample_configs(12, SEED, with_synthetic=True)
    assert first == sample_configs(12, SEED, with_synthetic=True)
    assert len({json.dumps(c, sort_keys=True) for c in first}) == len(first)


def test_sampled_values_stay_inside_the_search_space() -> None:
    for config in sample_configs(20, SEED, with_synthetic=False):
        for key, values in SEARCH_SPACE.items():
            assert config[key] in values or config[key] == BASE_CONFIG[key]


def test_synthetic_axes_only_appear_when_requested() -> None:
    plain = sample_configs(10, SEED, with_synthetic=False)
    assert all(not c["use_synthetic"] and "synthetic_per_real" not in c for c in plain)
    assert any(c["use_synthetic"] for c in sample_configs(10, SEED, with_synthetic=True))


def test_split_config_never_enables_synthetic_for_stegastamp() -> None:
    config = dict(BASE_CONFIG, use_synthetic=True, synthetic_per_real=4.0)
    _, train = split_config(config, "stegastamp", 0)
    assert not train.use_synthetic
    head, train = split_config(config, "pixelseal", 1)
    assert train.use_synthetic and train.synthetic_per_real == 4.0
    assert head.width == BASE_CONFIG["width"]
    assert train.seed != split_config(config, "pixelseal", 0)[1].seed


def test_best_args_picks_the_best_mean_per_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hackathon.wmloc.research.b2_sweep import best_args

    monkeypatch.setattr(b2_sweep, "results_dir", lambda: tmp_path)
    base = dict(BASE_CONFIG, use_synthetic=False)
    other = dict(BASE_CONFIG, dropout=0.7, use_synthetic=True, synthetic_per_real=4.0)
    rows = [
        {"config_id": 0, "config": base, "seed": 0, "method": "pixelseal", "all": 0.50},
        {"config_id": 1, "config": other, "seed": 0, "method": "pixelseal", "all": 0.55},
        {"config_id": 0, "config": base, "seed": 0, "method": "stegastamp", "all": 0.52},
        {"config_id": 1, "config": other, "seed": 0, "method": "stegastamp", "all": 0.51},
    ]
    (tmp_path / "t_w0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    pixelseal = best_args("t", "pixelseal")
    assert "--dropout 0.7" in pixelseal and "--synthetic " in pixelseal + " "
    assert "--synthetic-per-real 4.0" in pixelseal
    stegastamp = best_args("t", "stegastamp")
    assert "--dropout 0.5" in stegastamp
    assert "--synthetic " not in stegastamp + " "


def test_summarise_ranks_by_mean_over_seeds_and_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(b2_sweep, "results_dir", lambda: tmp_path)
    rows = [
        {"config_id": 0, "config": {"a": 1}, "seed": 0, "method": "pixelseal", "all": 0.50},
        {"config_id": 0, "config": {"a": 1}, "seed": 1, "method": "pixelseal", "all": 0.52},
        {"config_id": 1, "config": {"a": 2}, "seed": 0, "method": "pixelseal", "all": 0.56},
    ]
    (tmp_path / "t_w0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ranked = summarise("t")
    assert [item["config_id"] for item in ranked] == [1, 0]
    assert ranked[1]["mean"] == pytest.approx(0.51)
    assert ranked[1]["n_runs"] == 2

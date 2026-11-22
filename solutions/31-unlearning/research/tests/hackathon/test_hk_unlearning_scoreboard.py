"""Tests of the Unlearning comparison table (registry -> SCOREBOARD.md)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from hackathon.lab.registry import Registry, RunRecord
from hackathon.unlearning.scoreboard import main, render, rows_from_records, settings_text


def run(run_id: str, score: float, *, status: str = "done", **metrics: float) -> RunRecord:
    values = {
        "bench.score_min": score,
        "bench.score.s40": score + 0.1,
        "bench.retain": 0.9,
        "bench.groups.forget_exact.kl_orig_sub": 0.5,
        **{f"bench.groups.forget_{name}.kl_orig_sub": 0.4 for name in ("flip", "crop", "jitter")},
        "bench.groups.forget_jpeg.kl_orig_sub": 0.25,
        "config.lr": 1e-5,
        "config.kl_radius": 0.5,
        "config.target_temperature": 2.0,
        "config.seed": 0.0,
        "config.holdout_fraction": 0.2,
        **metrics,
    }
    return RunRecord(
        run_id=run_id, group_id=run_id, task="unlearning", idea="i", spec_id="spec", variant="v1",
        sha="abcdef0123", where="local", status=status, created_at="2026-11-21T10:00:00",
        updated_at="2026-11-21T10:00:00", params={"method": "trust_region"}, metrics=values,
    )  # fmt: skip


def test_only_finished_full_scale_runs_are_kept_best_first(tmp_path: Path) -> None:
    records = [
        run("low", 0.1),
        run("high", 0.3),
        run("failed", 0.9, status="failed"),
        RunRecord(**{**run("smoke", 0.5).__dict__, "metrics": {"bench_smoke.score_min": 0.5}}),
    ]
    rows = rows_from_records(records, tmp_path)
    assert [row.run_id for row in rows] == ["high", "low"]
    assert rows[0].method == "trust_region"
    assert rows[0].transfer == 0.5, "the worst altered copy (jpeg) over the exact images"


def test_settings_show_only_what_differs_from_the_defaults_and_not_the_seed(
    tmp_path: Path,
) -> None:
    row = rows_from_records(
        [run("r", 0.2, **{"config.kl_radius": 0.2, "config.seed": 3.0, "config.epochs": 10.0})],
        tmp_path,
    )[0]
    assert settings_text(row) == "ε=0.2 τ=2 époques=10", "ε and τ always shown for trust_region"
    assert row.seed == 3
    default = run("d", 0.1, **{"config.kl_radius": 0.5})
    assert settings_text(rows_from_records([default], tmp_path)[0]) == "ε=0.5 τ=2"
    scrub = RunRecord(**{**run("s", 0.1).__dict__, "params": {"method": "scrub"}})
    assert settings_text(rows_from_records([scrub], tmp_path)[0]) == "défaut"


def test_method_is_read_from_the_run_folder_first(tmp_path: Path) -> None:
    folder = tmp_path / "unlearning" / "r"
    folder.mkdir(parents=True)
    (folder / "candidate.json").write_text(json.dumps({"config": {"method": "gold_soft"}}))
    assert rows_from_records([run("r", 0.1)], tmp_path)[0].method == "gold_soft"


def test_text_settings_split_runs_that_share_their_numbers(tmp_path: Path) -> None:
    folder = tmp_path / "unlearning" / "views"
    folder.mkdir(parents=True)
    config = {"method": "scrub", "retain_source": "forget_views", "trust_target": "tempered"}
    (folder / "candidate.json").write_text(json.dumps({"config": config}))
    proxy = RunRecord(**{**run("proxy", 0.1).__dict__, "params": {"method": "scrub"}})
    rows = {row.run_id: row for row in rows_from_records([run("views", 0.2), proxy], tmp_path)}
    assert settings_text(rows["views"]) == "rétention=forget_views"  # default target not shown
    assert settings_text(rows["proxy"]) == "défaut"
    assert rows["views"].setting_key != rows["proxy"].setting_key


def test_trust_region_is_labelled_by_its_target(tmp_path: Path) -> None:
    for run_id, config in (
        ("v1", {"method": "trust_region"}),
        ("v2", {"method": "trust_region", "trust_target": "views"}),
        ("v3", {"method": "trust_region", "trust_target": "calibrated"}),
    ):
        folder = tmp_path / "unlearning" / run_id
        folder.mkdir(parents=True)
        (folder / "candidate.json").write_text(json.dumps({"config": config}))
    extra = {"config.target_view_min_area": 0.35, "config.target_views": 3.0}
    rows = rows_from_records([run("v1", 0.2), run("v2", 0.1, **extra), run("v3", 0.05)], tmp_path)
    assert [row.method for row in rows] == ["trust_region V1", "trust_region V2", "trust_region V3"]
    assert settings_text(rows[1]) == "ε=0.5 τ=2 aire=0.35 vues=3", "V2 knobs always shown"


def test_summary_averages_the_seeds_of_one_setting(tmp_path: Path) -> None:
    rows = rows_from_records(
        [
            run("a", 0.2),
            run("b", 0.4, **{"config.seed": 1.0}),
            run("c", 0.1, **{"config.lr": 3e-5}),
        ],
        tmp_path,
    )
    page = render(rows, datetime(2026, 11, 21, 23, 0))
    summary = page.split("## bench-v3 : tous les runs")[0]
    assert "| 1 | trust_region | ε=0.5 τ=2 | 2 | 0.300 ± 0.141 |" in summary
    assert "| 2 | trust_region | ε=0.5 τ=2 lr=3e-05 | 1 | 0.100 |" in summary
    assert "Généré le 2026-11-21 23:00" in page and "abcdef0" in page


def test_main_writes_the_file_from_a_registry(tmp_path: Path, capsys: object) -> None:
    registry = Registry(tmp_path / "registry.jsonl")
    registry.append(run("a", 0.2))
    out = tmp_path / "SCOREBOARD.md"
    assert main(["--registry", str(registry.path), "--runs", str(tmp_path), "--out", str(out)]) == 0
    assert "| a | trust_region | ε=0.5 τ=2 | 0 | 0.200 |" in out.read_text(encoding="utf-8")


def test_protocols_are_never_ranked_together(tmp_path: Path) -> None:
    v4 = {"bench.protocol_version": 4.0, "bench.score.m30": 0.05}
    rows = rows_from_records([run("old", 0.3), run("new", 0.1, **v4)], tmp_path)
    assert [row.protocol for row in rows] == [3, 4], "no protocol number = bench-v3"
    page = render(rows, datetime(2026, 11, 21, 23, 0))
    v4_part, v3_part = page.split("## bench-v3 : synthèse")
    assert page.index("## bench-v4") < page.index("## bench-v3"), "newest protocol first"
    assert "| new |" in v4_part and "| old |" not in v4_part and "| old |" in v3_part
    assert "m30" in v4_part and "m30" not in v3_part
    assert "| 1 | trust_region | ε=0.5 τ=2 | 1 | 0.100 |" in v4_part, "ranked within bench-v4"


def test_radius_fraction_is_a_setting_and_replaces_the_unused_kl_radius(tmp_path: Path) -> None:
    rows = rows_from_records(
        [
            run("a", 0.2, **{"config.radius_fraction": 0.9}),
            run("b", 0.2, **{"config.radius_fraction": 0.9, "config.seed": 1.0}),
            run("c", 0.1, **{"config.radius_fraction": 0.95}),
        ],
        tmp_path,
    )
    assert settings_text(rows[0]) == "τ=2 ρ=0.9", "kl_radius is not shown when unused"
    page = render(rows, datetime(2026, 11, 22, 1, 0))
    assert (
        "| 1 | trust_region | τ=2 ρ=0.9 | 2 |" in page
        and "| 2 | trust_region | τ=2 ρ=0.95 | 1 |" in page
    )


def test_v4_options_split_settings_and_are_never_merged(tmp_path: Path) -> None:
    """V4 runs on the same V3 base differ only by anchor_* or radius_dual_lr: each its own row."""
    base = {"config.radius_fraction": 0.9}
    arms = {
        "v3": {},
        "anchor": {"config.anchor_folds": 4.0},
        "folds2": {"config.anchor_folds": 2.0},
        "weight3": {"config.anchor_folds": 4.0, "config.anchor_weight": 3.0},
        "views1": {"config.anchor_folds": 4.0, "config.anchor_views": 1.0},
        "dual": {"config.radius_dual_lr": 0.25},
        "dual_s1": {"config.radius_dual_lr": 0.25, "config.seed": 1.0},
        "dual05": {"config.radius_dual_lr": 0.5},
    }
    records = [run(name, 0.1, **base, **extra) for name, extra in arms.items()]
    rows = {row.run_id: row for row in rows_from_records(records, tmp_path)}
    assert settings_text(rows["anchor"]) == "τ=2 ρ=0.9 plis=4", "weight 1, views 0: defaults"
    assert settings_text(rows["weight3"]) == "τ=2 ρ=0.9 plis=4 w_ancre=3"
    assert settings_text(rows["views1"]) == "τ=2 ρ=0.9 plis=4 vues_ancre=1"
    assert settings_text(rows["dual05"]) == "τ=2 ρ=0.9 η_dual=0.5"
    keys = {name: rows[name].setting_key for name in arms}
    assert keys["dual"] == keys["dual_s1"], "the seed is averaged over"
    assert len({key for name, key in keys.items() if name != "dual_s1"}) == len(arms) - 1
    page = render(list(rows.values()), datetime(2026, 11, 22, 3, 0))
    assert "| trust_region | τ=2 ρ=0.9 η_dual=0.25 | 2 |" in page


def test_band_ascent_arms_are_told_apart_even_when_an_option_is_none(tmp_path: Path) -> None:
    """band-ascent's arms at one seed: band_floor=None leaves no number in the registry, so the
    run's own JSON must tell it from the default arm (else the two would count as one seed)."""
    band = {"config.band_release": 0.25, "config.cv_max_drift": 0.1, "config.band_floor": 0.05}
    arms = {
        "default": band,
        "release1": {**band, "config.band_release": 1.0},
        "floor_none": {k: v for k, v in band.items() if k != "config.band_floor"},
        "no_cv": {**band, "config.cv_folds": 0.0},
    }
    records = []
    for name, extra in arms.items():
        record = run(name, 0.1, **extra)
        records.append(RunRecord(**{**record.__dict__, "params": {"method": "band_ascent"}}))
        folder = tmp_path / "unlearning" / name
        folder.mkdir(parents=True)
        config = {"method": "band_ascent", "band_floor": None if name == "floor_none" else 0.05}
        (folder / "candidate.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    rows = {row.run_id: row for row in rows_from_records(records, tmp_path)}
    assert {row.method for row in rows.values()} == {"band_ascent"}
    assert "φ=0.25" in settings_text(rows["default"]) and "δ0" not in settings_text(rows["default"])
    assert "φ=1" in settings_text(rows["release1"])
    assert "δ0=none" in settings_text(rows["floor_none"])
    assert "K=0" in settings_text(rows["no_cv"])
    assert len({row.setting_key for row in rows.values()}) == len(arms), "four arms, four rows"
    lost = rows_from_records(records, tmp_path / "elsewhere")  # no run folder: looks default
    assert sum("δ0=none" in settings_text(row) for row in lost) == 0


def test_a_replayed_seed_counts_once_and_clean_runs_win(tmp_path: Path) -> None:
    """ref4-v3 replays tr3-hedge at the same seed; a run rerun clean replaces its dirty twin."""
    records = [
        run("orig_s0", 0.2),
        run("replay_s0", 0.2),
        run("s1", 0.4, **{"config.seed": 1.0}),
        RunRecord(**{**run("dirty_s1", 0.9, **{"config.seed": 1.0}).__dict__, "dirty": True}),
    ]
    page = render(rows_from_records(records, tmp_path), datetime(2026, 11, 22, 3, 0))
    summary = page.split("## bench-v3 : tous les runs")[0]
    assert "| 1 | trust_region | ε=0.5 τ=2 | 2 | 0.300 ± 0.141 |" in summary, "2 seeds, not 4 runs"
    assert "| dirty_s1 |" in page and "abcdef0*" in page, "every run stays listed"

"""Tests of the run registry: append-only snapshots, last-writer-wins by timestamp, lookups."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_hk_lab_support import REAL_REPO, StepClock

from hackathon.lab.errors import RegistryError
from hackathon.lab.registry import Registry, RunRecord, runs_of_task


def record(
    run_id: str, *, group: str | None = None, stamp: str = "2026-11-21T10:00:00", **kw: object
) -> RunRecord:
    base: dict[str, object] = {
        "run_id": run_id,
        "group_id": group or run_id,
        "task": "unlearning",
        "idea": "idea",
        "spec_id": "spec",
        "variant": "v1",
        "sha": "abc1234",
        "where": "local",
        "status": "running",
        "created_at": stamp,
        "updated_at": stamp,
    }
    base.update(kw)
    return RunRecord(**base)  # type: ignore[arg-type]


def test_empty_registry_reads_as_empty(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "experiments" / "registry.jsonl")
    assert registry.snapshots() == [] and registry.latest() == {}


def test_append_then_read_roundtrip_keeps_every_field(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "sub" / "registry.jsonl")
    original = record("r1", metrics={"a.b": 1.5}, params={"lr": 0.1}, dirty=True, public_score=0.5)
    registry.append(original)
    assert registry.snapshots() == [original]
    line = (tmp_path / "sub" / "registry.jsonl").read_text().splitlines()
    assert len(line) == 1 and json.loads(line[0])["run_id"] == "r1"


def test_latest_snapshot_wins_by_updated_at_whatever_the_line_order(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.jsonl")
    newer = record("r1", status="done", stamp="2026-11-21T11:00:00")
    older = record("r1", status="running", stamp="2026-11-21T10:00:00")
    # A git union merge can put the newer line first: the timestamp decides, not the position.
    registry.append(newer)
    registry.append(older)
    assert registry.latest()["r1"].status == "done"


def test_latest_is_ordered_by_creation_time(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.jsonl")
    registry.append(record("b", stamp="2026-11-21T12:00:00"))
    registry.append(record("a", stamp="2026-11-21T09:00:00"))
    assert list(registry.latest()) == ["a", "b"]


def test_update_appends_a_new_snapshot_with_a_later_timestamp(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.jsonl", StepClock())
    first = record("r1", stamp="2030-01-01T00:00:00+00:00")
    registry.append(first)
    second = registry.update(first, status="done", metrics={"m": 2.0})
    # Even with a clock behind the previous snapshot the update must not lose to it.
    assert second.updated_at >= first.updated_at
    assert registry.latest()["r1"].status == "done"
    assert len(registry.snapshots()) == 2


def test_find_exact_group_and_prefix(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.jsonl")
    registry.append(record("g1-v1", group="g1"))
    registry.append(record("g1-v2", group="g1"))
    registry.append(record("g2", group="g2"))
    assert [r.run_id for r in registry.find("g1-v2")] == ["g1-v2"]
    assert sorted(r.run_id for r in registry.find("g1")) == ["g1-v1", "g1-v2"]
    assert [r.run_id for r in registry.find("g2")] == ["g2"]
    assert [r.run_id for r in registry.find("g1-v")] == ["g1-v1", "g1-v2"]


def test_get_needs_exactly_one_run(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.jsonl")
    registry.append(record("g1-v1", group="g1"))
    registry.append(record("g1-v2", group="g1"))
    registry.append(record("g2-v1", group="g2"))
    with pytest.raises(RegistryError, match="several runs, pick one"):
        registry.get("g1")
    with pytest.raises(RegistryError, match="ambiguous"):
        registry.find("g")
    with pytest.raises(RegistryError, match="no run 'zzz'"):
        registry.get("zzz")
    assert registry.get("g2-v1").run_id == "g2-v1"


def test_corrupt_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text(record("r1").to_json_line() + "\n{broken\n", encoding="utf-8")
    with pytest.raises(RegistryError, match=r"registry.jsonl:2: not valid JSON"):
        Registry(path).snapshots()


def test_non_object_and_incomplete_lines_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="expected a JSON object"):
        Registry(path).snapshots()
    path.write_text('{"run_id": "x"}\n', encoding="utf-8")
    with pytest.raises(RegistryError, match=r":1: registry record is incomplete"):
        Registry(path).snapshots()


def test_unknown_keys_from_a_newer_lab_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    raw = json.loads(record("r1").to_json_line()) | {"from_the_future": 1}
    path.write_text(json.dumps(raw) + "\n\n", encoding="utf-8")
    assert Registry(path).latest()["r1"].run_id == "r1"


def test_runs_of_task_filters() -> None:
    rows = [record("a"), record("b", task="model_equality")]
    assert [r.run_id for r in runs_of_task(rows, "model_equality")] == ["b"]
    assert len(runs_of_task(rows, None)) == 2


def test_shipped_registry_parses_and_merges_by_union() -> None:
    """Every committed line is a valid snapshot; union merge keeps teammates' lines on merges."""
    experiments = REAL_REPO / "experiments"
    registry = Registry(experiments / "registry.jsonl")
    assert all(record.run_id for record in registry.snapshots())
    attributes = (experiments / ".gitattributes").read_text(encoding="utf-8")
    assert "registry.jsonl merge=union" in attributes

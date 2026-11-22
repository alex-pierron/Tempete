"""Team C result store: append-only JSONL shards, legacy per-image JSON, crash tolerance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.wmloc.research.c_results import (
    ArrayChunkWriter,
    ResultStoreError,
    ShardWriter,
    done_array_names,
    done_names,
    load_all,
    load_arrays,
)


def test_writer_appends_one_line_per_payload_in_a_single_file(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path / "out", "shard0of2")
    writer.append({"name": "a", "x": 1})
    writer.append({"name": "b", "x": 2})
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["shard0of2.jsonl"]
    assert {p["name"] for p in load_all(tmp_path / "out")} == {"a", "b"}


def test_a_truncated_last_line_is_ignored_after_a_crash(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "shard0of1")
    writer.append({"name": "a"})
    with writer.path.open("a", encoding="utf-8") as handle:
        handle.write('{"name": "b", "box": {"top"')  # killed mid-write
    assert done_names(tmp_path) == {"a"}


def test_legacy_per_image_json_is_read_and_jsonl_wins_on_duplicates(tmp_path: Path) -> None:
    (tmp_path / "old.json").write_text(json.dumps({"name": "old", "v": 1}))
    (tmp_path / "dup.json").write_text(json.dumps({"name": "dup", "v": 1}))
    ShardWriter(tmp_path, "shard0of1").append({"name": "dup", "v": 2})
    stored = {p["name"]: p["v"] for p in load_all(tmp_path)}
    assert stored == {"old": 1, "dup": 2}


def test_done_names_of_a_missing_or_empty_directory_is_empty(tmp_path: Path) -> None:
    assert done_names(tmp_path / "absent") == set()
    assert done_names(tmp_path) == set()


def test_load_all_rejects_a_missing_directory_and_an_empty_one(tmp_path: Path) -> None:
    with pytest.raises(ResultStoreError):
        load_all(tmp_path / "absent")
    with pytest.raises(ResultStoreError):
        load_all(tmp_path)


def test_array_chunks_hold_several_images_per_file_and_roundtrip(tmp_path: Path) -> None:
    writer = ArrayChunkWriter(tmp_path / "occ", "shard0of2", chunk_images=3)
    for i in range(7):
        writer.add(
            f"img{i}", {"drop_s6": np.full((2, 2), i, dtype=np.float32), "block": np.int32(32)}
        )
    writer.flush()
    files = sorted(p.name for p in (tmp_path / "occ").iterdir())
    assert files == ["shard0of2_part0.npz", "shard0of2_part1.npz", "shard0of2_part2.npz"]
    stored = load_arrays(tmp_path / "occ")
    assert sorted(stored) == [f"img{i}" for i in range(7)]
    assert float(stored["img5"]["drop_s6"][0, 0]) == 5.0 and int(stored["img5"]["block"]) == 32
    assert done_array_names(tmp_path / "occ") == set(stored)


def test_array_writer_continues_part_numbers_after_a_restart(tmp_path: Path) -> None:
    ArrayChunkWriter(tmp_path, "shard1of2", chunk_images=1).add("a", {"x": np.zeros(1)})
    ArrayChunkWriter(tmp_path, "shard1of2", chunk_images=1).add("b", {"x": np.ones(1)})
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["shard1of2_part0.npz", "shard1of2_part1.npz"]


def test_legacy_per_image_npz_is_read_by_name(tmp_path: Path) -> None:
    np.savez(tmp_path / "legacy.npz", base_margin=np.float32(2.0))
    assert float(load_arrays(tmp_path)["legacy"]["base_margin"]) == 2.0
    assert done_array_names(tmp_path) == {"legacy"}


def test_load_arrays_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ResultStoreError):
        load_arrays(tmp_path)

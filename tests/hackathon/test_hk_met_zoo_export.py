"""Zoo export for team C: official lines, ref-renamed single-family files, truth, checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_hk_met_zoo_evaluate import FAMILY_KEYS, write_synthetic_zoo

from hackathon.model_equality.zoo.export import export_for_team_c
from hackathon.model_equality.zoo.store import read_rows

OFFICIAL_FIELDS = {"endpoint", "id", "repeat", "prompt", "response"}


def test_export_writes_official_rows_ref_renamed_families_truth_and_checksums(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    write_synthetic_zoo(source)
    manifest = export_for_team_c(source, destination)
    everything = read_rows(destination / "zoo_answers.jsonl")
    assert OFFICIAL_FIELDS <= set(everything[0])
    assert {str(r["endpoint"]) for r in everything} >= {"zoo-qwen15-01", "zoo-qwen15-10"}
    first = FAMILY_KEYS[0]
    renamed = read_rows(destination / f"zoo_{first}_as_ref.jsonl")
    assert {str(r["endpoint"]) for r in renamed} == {f"ref-{k:02d}" for k in range(1, 10)}
    assert len(renamed) == len([r for r in read_rows(source / f"zoo_{first}.jsonl")
                                if not str(r["endpoint"]).endswith("-10")])
    truth = json.loads((destination / "zoo_truth.json").read_text(encoding="utf-8"))
    assert truth[f"zoo-{first}-04"] == {"family": first, "slot": 4, "variant": "lora07",
                                       "model": "T3_lora07", "ref_equivalent": "ref-04"}
    assert truth[f"zoo-{first}-10"]["ref_equivalent"] is None
    for name, entry in manifest["files"].items():  # type: ignore[attr-defined]
        digest = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        assert entry["sha256"] == digest


def test_export_needs_at_least_one_family(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_for_team_c(tmp_path, tmp_path / "out")

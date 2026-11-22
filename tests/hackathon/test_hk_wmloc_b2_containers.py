"""Tests of the container caches: a few files per run instead of one file per image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hackathon.wmloc.research import b2_dino, b2_features, b2_pack
from hackathon.wmloc.research.b2_features import (
    FeatureError,
    load_feature_table,
    load_features,
    write_feature_chunk,
)
from hackathon.wmloc.research.b2_synth_features import clean_crop, clean_pool

SEED = 3
METHOD = "pixelseal"
SPLIT = "train"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh data directory per test, and empty in-memory tables."""
    monkeypatch.setenv("HACKATHON_DATA_DIR", str(tmp_path))
    b2_features._TABLES.clear()
    b2_dino._DINO_TABLES.clear()
    b2_pack._GLOBAL_TABLES.clear()
    return tmp_path


def _rows(names: list[str]) -> list[tuple[str, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(SEED)
    return [
        (
            name,
            {
                "n_ev": rng.standard_normal((4, 8, 8)).astype(np.float16),
                "scalars": rng.standard_normal(3).astype(np.float32),
            },
        )
        for name in names
    ]


def test_one_chunk_file_holds_many_images(isolated_cache: Path) -> None:
    rows = _rows([f"img{i}" for i in range(10)])
    write_feature_chunk(SPLIT, METHOD, rows)
    files = list((isolated_cache / "wmloc" / "team_b2" / "feat" / SPLIT / METHOD).glob("*"))
    assert len(files) == 1
    table = load_feature_table(SPLIT, METHOD)
    assert sorted(table) == sorted(name for name, _ in rows)
    expected = dict(rows)["img3"]
    assert np.array_equal(table["img3"]["n_ev"], expected["n_ev"])
    assert np.array_equal(load_features(SPLIT, METHOD, "img3")["scalars"], expected["scalars"])


def test_chunks_accumulate_and_a_missing_image_raises(isolated_cache: Path) -> None:
    write_feature_chunk(SPLIT, METHOD, _rows(["a", "b"]))
    write_feature_chunk(SPLIT, METHOD, _rows(["c"]))
    assert sorted(load_feature_table(SPLIT, METHOD)) == ["a", "b", "c"]
    with pytest.raises(FeatureError):
        load_features(SPLIT, METHOD, "zzz")


def test_legacy_per_image_files_are_still_read(isolated_cache: Path) -> None:
    directory = isolated_cache / "wmloc" / "team_b2" / "feat" / SPLIT / METHOD
    directory.mkdir(parents=True)
    np.savez(
        directory / "old.npz", n_ev=np.ones((4, 8, 8), np.float16), scalars=np.zeros(3, np.float32)
    )
    write_feature_chunk(SPLIT, METHOD, _rows(["new"]))
    assert sorted(load_feature_table(SPLIT, METHOD)) == ["new", "old"]


def test_temporary_files_are_never_read(isolated_cache: Path) -> None:
    directory = isolated_cache / "wmloc" / "team_b2" / "feat" / SPLIT / METHOD
    directory.mkdir(parents=True)
    (directory / "chunk_1_2.npz.99.tmp.npz").write_bytes(b"not a zip")
    write_feature_chunk(SPLIT, METHOD, _rows(["ok"]))
    assert sorted(load_feature_table(SPLIT, METHOD)) == ["ok"]


def test_dino_chunk_round_trip(isolated_cache: Path) -> None:
    features = np.random.default_rng(SEED).standard_normal((5, 384, 4, 4)).astype(np.float16)
    names = [f"d{i}" for i in range(5)]
    b2_dino.write_dino_chunk(SPLIT, names, features)
    assert len(list(b2_dino.dino_dir(SPLIT).glob("*"))) == 1
    assert np.array_equal(b2_dino.load_dino(SPLIT, "d2"), features[2])
    with pytest.raises(FileNotFoundError):
        b2_dino.load_dino(SPLIT, "missing")


def _write_release(root: Path, names: list[str], method: str) -> None:
    folder = root / "wmloc" / "release" / SPLIT
    folder.mkdir(parents=True)
    metadata = [
        {"image_name": n, "method": method, "message": "10" * 4, "full_watermark": True}
        for n in names
    ]
    (folder / "metadata.json").write_text(json.dumps(metadata))


def test_pack_global_turns_per_image_decodes_into_one_container(isolated_cache: Path) -> None:
    names = [f"g{i}" for i in range(4)]
    _write_release(isolated_cache, names, METHOD)
    source = isolated_cache / "wmloc" / "research_cache" / SPLIT / METHOD
    source.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    expected = {n: rng.standard_normal((1, 8)).astype(np.float32) for n in names}
    for name, logits in expected.items():
        np.savez(source / f"{name}.global.npz", logits=logits)
    path = b2_pack.pack_global(SPLIT, METHOD)
    assert path.is_file() and len(list(path.parent.glob("*"))) == 1
    table = b2_pack.load_global_table(SPLIT, METHOD)
    assert table is not None
    assert np.allclose(table["g2"], expected["g2"][0])
    assert b2_pack.load_global_table(SPLIT, "trustmark") is None


def test_clean_pool_reads_one_container_and_crops_are_512(isolated_cache: Path) -> None:
    photos = isolated_cache / "photos"
    photos.mkdir()
    for index, size in enumerate([(640, 480), (500, 700)]):
        Image.fromarray(np.full((size[1], size[0], 3), 40 * (index + 1), np.uint8)).save(
            photos / f"p{index}.jpg"
        )
    container = b2_pack.pack_clean(photos)
    assert container.is_file()
    pool = clean_pool()
    assert len(pool) == 2 and all(min(p.shape[:2]) == 512 for p in pool)
    rng = np.random.default_rng(SEED)
    crop = clean_crop(pool[0], rng)
    assert crop.shape == (512, 512, 3) and crop.dtype == np.uint8

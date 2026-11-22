"""End-to-end test of the B2 CV / fit pipeline on a tiny synthetic data directory.

Builds a fake `<data>/wmloc/release/{train,test}` and every cache the loaders read (extractor
features, DINO, window decodes, thumbnails), then runs the real cross-validation and the
fit-and-predict path for a couple of epochs on the CPU. It proves that shapes, folds, regime
breakdown and the test-probability output line up, with counts of images in and out.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from hackathon.wmloc.research.b2_model import (
    SOURCES_PIXELSEAL,
    WINDOW_KEY,
    HeadConfig,
)
from hackathon.wmloc.research.b2_train import (
    MethodArrays,
    TrainConfig,
    _epoch_order,
    _mixed_epoch_size,
    fit_and_predict_test,
    load_arrays,
    place_on_device,
    run_cv,
)
from hackathon.wmloc.research.b2_windows import WINDOW_SIZES, grid_side

N_TRAIN = 10
N_TEST = 4
N_BITS = 256
IMAGE_SIDE = 512
GRID = 128
METHOD = "pixelseal"
SEED = 7


def _write_split(root: Path, split: str, n_images: int, rng: np.random.Generator) -> list[str]:
    folder = root / "wmloc" / "release" / split
    (folder / "images").mkdir(parents=True)
    names = [f"{split}{index:03d}" for index in range(n_images)]
    metadata = []
    masks = np.zeros((n_images, GRID, GRID), dtype=np.uint8)
    for index, name in enumerate(names):
        Image.fromarray(np.zeros((IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.uint8)).save(
            folder / "images" / f"{name}.png"
        )
        entry: dict[str, object] = {
            "image_name": name,
            "method": METHOD,
            "message": "".join(str(int(b)) for b in rng.integers(0, 2, N_BITS)),
        }
        if split == "train":
            is_full = index % 2 == 0
            entry["full_watermark"] = is_full
            masks[index] = 1 if is_full else 0
            masks[index, 20:50, 30:70] = 0 if is_full else 1
        metadata.append(entry)
    (folder / "metadata.json").write_text(json.dumps(metadata))
    if split == "train":
        np.savez(folder / "masks_128.npz", names=np.array(names), masks=masks)
    return names


def _write_caches(root: Path, split: str, names: list[str], rng: np.random.Generator) -> None:
    feat = root / "wmloc" / "team_b2" / "feat" / split / METHOD
    win = root / "wmloc" / "team_b2" / "win" / split / METHOD
    dino = root / "wmloc" / "team_b2" / "dino" / split
    glob = root / "wmloc" / "research_cache" / split / METHOD
    for folder in (feat, win, dino, glob):
        folder.mkdir(parents=True)
    for name in names:
        np.savez(glob / f"{name}.global.npz", logits=rng.standard_normal((1, N_BITS)))
        arrays = {
            key: rng.standard_normal((channels, side, side)).astype(np.float16)
            for key, (channels, side) in SOURCES_PIXELSEAL.items()
        }
        arrays["scalars"] = rng.standard_normal(3).astype(np.float32)
        np.savez(feat / f"{name}.npz", **arrays)  # type: ignore[arg-type]
        np.save(dino / f"{name}.npy", rng.standard_normal((384, 32, 32)).astype(np.float16))
        windows: dict[str, np.ndarray] = {}
        for size in WINDOW_SIZES:
            side = grid_side(size)
            windows[f"w{size}_m"] = rng.standard_normal((side, side)).astype(np.float16)
            windows[f"w{size}_a"] = rng.random((side, side)).astype(np.float16)
        np.savez(win / f"{name}.npz", **windows)  # type: ignore[arg-type]


@pytest.fixture()
def fake_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny, fully consistent data directory (10 train and 4 test images, one method)."""
    rng = np.random.default_rng(SEED)
    for split, count in (("train", N_TRAIN), ("test", N_TEST)):
        names = _write_split(tmp_path, split, count, rng)
        _write_caches(tmp_path, split, names, rng)
    monkeypatch.setenv("HACKATHON_DATA_DIR", str(tmp_path))
    return tmp_path


def _head_no_windows() -> HeadConfig:
    return HeadConfig(sources={METHOD: tuple(SOURCES_PIXELSEAL)}, dropout=0.1)


def _head() -> HeadConfig:
    return HeadConfig(sources={METHOD: (*SOURCES_PIXELSEAL, WINDOW_KEY)}, dropout=0.1)


def test_cross_validation_accounts_for_every_image(fake_data: Path) -> None:
    config = TrainConfig(epochs=2, batch_size=4, device="cpu", seed=SEED)
    probabilities, reports, curves = run_cv([METHOD], _head(), config)
    assert probabilities[METHOD].shape == probabilities[f"{METHOD}_tta"].shape
    assert probabilities[METHOD].shape == (N_TRAIN, GRID, GRID)
    assert float(probabilities[METHOD].min()) >= 0.0
    assert float(probabilities[METHOD].max()) <= 1.0
    assert len(curves) == 5
    report = reports[METHOD]
    assert 0.0 <= float(report["all"]) <= 1.0  # type: ignore[arg-type]
    rect_bins = report["by_size_rect"]
    full_bins = report["by_size_full"]
    assert isinstance(rect_bins, list) and isinstance(full_bins, list)
    assert sum(item["n"] for item in rect_bins) == N_TRAIN // 2
    assert sum(item["n"] for item in full_bins) == N_TRAIN // 2


def test_content_only_head_needs_no_extractor_cache(fake_data: Path) -> None:
    """DINO + RGB + global-decode scalars: the path StegaStamp and the ablation use."""
    for folder in (
        fake_data / "wmloc" / "team_b2" / "feat",
        fake_data / "wmloc" / "team_b2" / "win",
    ):
        shutil.rmtree(folder)
    head = HeadConfig(sources={METHOD: ()}, dropout=0.1)
    config = TrainConfig(epochs=2, batch_size=4, device="cpu", seed=SEED)
    probabilities, reports, _ = run_cv([METHOD], head, config)
    assert probabilities[METHOD].shape == (N_TRAIN, GRID, GRID)
    assert 0.0 <= float(reports[METHOD]["all"]) <= 1.0  # type: ignore[arg-type]


N_SYNTHETIC = 12


def _write_synthetic_shard(root: Path) -> None:
    """One synthetic shard with every key `load_arrays` appends after the real rows."""
    rng = np.random.default_rng(SEED + 1)
    shard: dict[str, np.ndarray] = {
        key: rng.standard_normal((N_SYNTHETIC, channels, side, side)).astype(np.float16)
        for key, (channels, side) in SOURCES_PIXELSEAL.items()
    }
    shard["dino"] = rng.standard_normal((N_SYNTHETIC, 384, 32, 32)).astype(np.float16)
    shard["image"] = rng.integers(0, 255, (N_SYNTHETIC, 3, IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    shard["scalars"] = rng.standard_normal((N_SYNTHETIC, 3)).astype(np.float32)
    shard["rgb"] = rng.integers(0, 255, (N_SYNTHETIC, 3, GRID, GRID), dtype=np.uint8)
    shard["truth"] = (rng.random((N_SYNTHETIC, GRID, GRID)) > 0.5).astype(np.uint8)
    shard["is_full"] = np.arange(N_SYNTHETIC) % 2 == 0
    shard["fold"] = np.arange(N_SYNTHETIC) % 5
    folder = root / "wmloc" / "team_b2" / "synth" / METHOD
    folder.mkdir(parents=True)
    np.savez(folder / "shard_0000.npz", **shard)  # type: ignore[arg-type]


def test_synthetic_rows_follow_the_real_rows_and_never_reach_evaluation(fake_data: Path) -> None:
    _write_synthetic_shard(fake_data)
    arrays = load_arrays("train", METHOD, synthetic=True)
    assert arrays.n_real == N_TRAIN
    assert arrays.scalars.shape[0] == N_TRAIN + N_SYNTHETIC
    assert arrays.truth is not None and arrays.truth.shape[0] == N_TRAIN + N_SYNTHETIC
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        device="cpu",
        seed=SEED,
        use_synthetic=True,
        synthetic_per_real=1.0,
        finetune_fraction=0.5,
    )
    probabilities, reports, _ = run_cv([METHOD], _head_no_windows(), config)
    assert probabilities[METHOD].shape == (N_TRAIN, GRID, GRID)
    rect_bins = reports[METHOD]["by_size_rect"]
    assert isinstance(rect_bins, list)
    assert sum(item["n"] for item in rect_bins) == N_TRAIN // 2


def test_pixel_stem_trains_on_real_and_synthetic_pixels(fake_data: Path) -> None:
    _write_synthetic_shard(fake_data)
    arrays = load_arrays("train", METHOD, synthetic=True, with_pixels=True)
    pixels = arrays.sources["pixels"]
    assert pixels.dtype == torch.uint8
    assert pixels.shape == (N_TRAIN + N_SYNTHETIC, 3, IMAGE_SIDE, IMAGE_SIDE)
    head = HeadConfig(sources={METHOD: ("n_ev",)}, dropout=0.1, use_pixels=True)
    config = TrainConfig(
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=SEED,
        use_synthetic=True,
        synthetic_per_real=1.0,
        finetune_fraction=0.0,
    )
    probabilities, _, _ = run_cv([METHOD], head, config)
    assert probabilities[METHOD].shape == (N_TRAIN, GRID, GRID)


def test_fit_with_synthetic_shards_and_pixels_predicts_every_test_image(fake_data: Path) -> None:
    """The exact path of the JURECA fits: synthetic shards, pixel stem, symmetry averaging."""
    _write_synthetic_shard(fake_data)
    head = HeadConfig(sources={METHOD: ("n_ev", "u_ev")}, dropout=0.1, use_pixels=True)
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        device="cpu",
        seed=SEED,
        use_synthetic=True,
        synthetic_per_real=2.0,
        finetune_fraction=0.5,
    )
    probabilities, names = fit_and_predict_test([METHOD], head, config, tta=True)
    assert probabilities[METHOD].shape == (N_TEST, GRID, GRID)
    assert len(names[METHOD]) == N_TEST
    assert float(probabilities[METHOD].min()) >= 0.0
    assert float(probabilities[METHOD].max()) <= 1.0


def test_place_on_device_is_a_no_op_without_cuda(fake_data: Path) -> None:
    arrays = {METHOD: load_arrays("train", METHOD)}
    assert place_on_device(arrays, "cpu") == 0.0
    assert place_on_device(arrays, "mps") == 0.0
    assert all(t.device.type == "cpu" for t in arrays[METHOD].sources.values())


class _FakeCudaTensor:
    """A tensor look-alike whose `.to` records the move: runs the placement logic on a CPU box."""

    def __init__(self, n_bytes: int, on_cuda: bool = False) -> None:
        self.n_bytes = n_bytes
        self.device = SimpleNamespace(type="cuda" if on_cuda else "cpu")

    def element_size(self) -> int:
        return 1

    def numel(self) -> int:
        return self.n_bytes

    def to(self, device: str) -> _FakeCudaTensor:
        return _FakeCudaTensor(self.n_bytes, on_cuda=True)


def test_place_on_device_spends_the_budget_smallest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = {
        "big": _FakeCudaTensor(900),
        "small": _FakeCudaTensor(100),
        "medium": _FakeCudaTensor(300),
    }
    arrays = {
        METHOD: MethodArrays(
            METHOD,
            [],
            sources,  # type: ignore[arg-type]  # look-alikes stand in for tensors
            torch.zeros(1),
            torch.zeros(1),
            None,
            None,
            None,
            0,
        )
    }
    monkeypatch.setenv("WMLOC_GPU_DATA_GB", str(450 / 1e9))
    moved = place_on_device(arrays, "cuda")
    assert moved == pytest.approx(400 / 1e9)
    assert arrays[METHOD].sources["small"].device.type == "cuda"
    assert arrays[METHOD].sources["medium"].device.type == "cuda"
    assert arrays[METHOD].sources["big"].device.type == "cpu"


def test_epoch_order_mixes_a_synthetic_quota_then_goes_real_only() -> None:
    rng = np.random.default_rng(SEED)
    indices = np.arange(30, dtype=np.int64)
    config = TrainConfig(use_synthetic=True, synthetic_per_real=1.0)
    mixed = _epoch_order(indices, 10, config, True, rng)
    assert len(mixed) == 20 == _mixed_epoch_size(indices, 10, config)
    assert (mixed < 10).sum() == 10
    real_only = _epoch_order(indices, 10, config, False, rng)
    assert sorted(real_only.tolist()) == list(range(10))


def test_fit_and_predict_returns_one_mask_per_test_image(fake_data: Path) -> None:
    config = TrainConfig(epochs=1, batch_size=4, device="cpu", seed=SEED)
    probabilities, names = fit_and_predict_test([METHOD], _head(), config)
    assert probabilities[METHOD].shape == (N_TEST, GRID, GRID)
    assert len(names[METHOD]) == N_TEST == len(set(names[METHOD]))
    assert torch.isfinite(torch.from_numpy(probabilities[METHOD])).all()

"""Tests of the w-loc-b2n additions: single-allocation synthetic loader and the pyramid segmenter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hackathon.wmloc.research.b2_model import OUT_SIDE, PIXEL_KEY, HeadConfig
from hackathon.wmloc.research.b2_train import MethodArrays, TrainingError
from hackathon.wmloc.research.b2n_pyramid import (
    MIN_STEPS,
    BatchSampler,
    OutputSink,
    ExtractorBackbone,
    PyramidNet,
    PyramidTrainConfig,
    planned_steps,
    train_pyramid,
)

IMAGE = 64
STAGE_CHANNELS = (4, 6, 8, 10)
N_REAL = 6
N_SYNTHETIC = 20


class TinyBackbone(nn.Module):
    """Four strided convolutions standing in for an extractor backbone (strides 4, 8, 16, 32)."""

    def __init__(self) -> None:
        super().__init__()
        widths = (3, *STAGE_CHANNELS)
        strides = (4, 2, 2, 2)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(widths[i], widths[i + 1], 3, stride=strides[i], padding=1)
                for i in range(len(strides))
            ]
        )

    def forward(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        hidden = pixels.float() / 255.0
        stages = []
        for layer in self.layers:
            hidden = layer(hidden)
            stages.append(hidden)
        return stages


def _arrays() -> MethodArrays:
    rows = N_REAL + N_SYNTHETIC
    generator = torch.Generator().manual_seed(3)
    truth = np.zeros((rows, OUT_SIDE, OUT_SIDE), np.uint8)
    truth[:, :40, :50] = 1
    return MethodArrays(
        method="demo",
        names=[f"img{i}" for i in range(N_REAL)],
        sources={
            "feat": torch.randn(rows, 5, 8, 8, generator=generator).half(),
            PIXEL_KEY: torch.randint(0, 255, (rows, 3, IMAGE, IMAGE), generator=generator).to(
                torch.uint8
            ),
        },
        scalars=torch.zeros(rows, 3),
        rgb=torch.zeros(rows, 3, OUT_SIDE, OUT_SIDE, dtype=torch.uint8),
        truth=truth,
        is_full=np.zeros(rows, dtype=np.bool_),
        folds=np.arange(rows, dtype=np.int64) % 5,
        n_real=N_REAL,
    )


def _model(monkeypatch: pytest.MonkeyPatch) -> PyramidNet:
    from hackathon.wmloc.research import b2_model

    monkeypatch.setitem(b2_model.SOURCES_BY_METHOD, "demo", {"feat": (5, 8)})
    head = HeadConfig(sources={"demo": ("feat",)}, use_dino=False, dropout=0.1)
    return PyramidNet(head, TinyBackbone(), STAGE_CHANNELS)


def test_pyramid_net_outputs_the_evaluation_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(monkeypatch)
    arrays = _arrays()
    sources = {key: value[:2] for key, value in arrays.sources.items()}
    logits = model("demo", sources, arrays.scalars[:2], arrays.rgb[:2].float())
    assert logits.shape == (2, OUT_SIDE, OUT_SIDE)
    backbone_count = sum(p.numel() for p in model.backbone.parameters())
    head_count = sum(p.numel() for p in model.head_parameters())
    assert backbone_count + head_count == sum(p.numel() for p in model.parameters())


def test_planned_steps_is_bounded_both_ways() -> None:
    config = PyramidTrainConfig(budget_seconds=100.0, max_steps=500)
    assert planned_steps(0.001, config) == MIN_STEPS
    assert planned_steps(1000.0, config) == 500
    assert MIN_STEPS < planned_steps(2.0, config) < 200


def test_batch_sampler_mixes_then_fine_tunes_on_real_rows_only() -> None:
    rows = np.arange(N_REAL + N_SYNTHETIC, dtype=np.int64)
    config = PyramidTrainConfig(batch_size=9, synthetic_per_real=8.0)
    sampler = BatchSampler(rows, N_REAL, config, seed=1)
    mixed = sampler.draw(real_only=False)
    assert len(mixed) == 9 and int((mixed < N_REAL).sum()) == 1
    assert (sampler.draw(real_only=True) < N_REAL).all()
    seen = np.concatenate([sampler.draw(real_only=True) for _ in range(4)])
    assert set(seen.tolist()) == set(range(N_REAL))  # shuffled passes visit every real row


def test_batch_sampler_rejects_a_fold_without_real_rows() -> None:
    rows = np.arange(N_REAL, N_REAL + 4, dtype=np.int64)
    with pytest.raises(TrainingError):
        BatchSampler(rows, N_REAL, PyramidTrainConfig(), seed=1)


def test_train_pyramid_runs_the_planned_steps_and_lowers_the_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    arrays = _arrays()
    config = PyramidTrainConfig(budget_seconds=1e6, max_steps=60, batch_size=4, seed=5)
    rows = np.arange(N_REAL + N_SYNTHETIC, dtype=np.int64)
    result = train_pyramid(model, arrays, rows, config, "cpu")
    assert result.steps == 60
    assert result.losses[-1] < result.losses[0]


def test_extractor_backbone_keeps_batch_norm_frozen_in_training_mode() -> None:
    network = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    backbone = ExtractorBackbone("trustmark", network)
    backbone.train()
    assert backbone.training and not network[1].training
    with pytest.raises(TrainingError):
        ExtractorBackbone("stegastamp", network)


def test_append_synthetic_allocates_once_and_keeps_real_rows_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hackathon.wmloc.research import b2_synth_features as module

    folder = tmp_path / "synth" / "pixelseal"
    folder.mkdir(parents=True)
    for index in range(3):
        np.savez(
            folder / f"shard_{index:04d}.npz",
            truth=np.full((4, 2, 2), index + 1, np.uint8),
            image=np.full((4, 1), index + 1, np.uint8),
            fold=np.full(4, index, np.int64),
        )
    np.savez(folder / "shard_0009.npz.123.tmp.npz", truth=np.zeros((4, 2, 2), np.uint8))
    monkeypatch.setattr(module, "synth_dir", lambda method: folder)
    real = {"truth": np.zeros((2, 2, 2), np.uint8), "pixels": np.zeros((2, 1), np.uint8)}
    joined = module.append_synthetic(
        "pixelseal", real, rename={"pixels": "image"}, max_samples=1000
    )
    assert joined["truth"].shape == (14, 2, 2) and joined["pixels"].shape == (14, 1)
    assert joined["truth"][:2].max() == 0
    assert joined["pixels"][2:, 0].tolist() == [1] * 4 + [2] * 4 + [3] * 4
    with pytest.raises(module.SynthFeatureError):
        module.append_synthetic("pixelseal", {"absent": np.zeros((1, 1), np.uint8)})


def test_retry_write_survives_transient_quota_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hackathon.wmloc.research import b2_train as module

    monkeypatch.setattr(module, "WRITE_RETRY_SECONDS", 0.0)
    failures = {"left": 2}
    real_savez = np.savez_compressed

    def flaky_savez(file: Path, **arrays: np.ndarray) -> None:
        if failures["left"] > 0:
            failures["left"] -= 1
            raise OSError(122, "Disk quota exceeded")
        real_savez(file, **arrays)

    monkeypatch.setattr(module.np, "savez_compressed", flaky_savez)
    target = tmp_path / "result.npz"
    module.save_atomically(target, {"values": np.arange(3)})
    assert np.load(target)["values"].tolist() == [0, 1, 2]
    assert [p.name for p in tmp_path.iterdir()] == ["result.npz"]  # no temporary file left


def test_retry_write_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    from hackathon.wmloc.research import b2_train as module

    monkeypatch.setattr(module, "WRITE_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(module, "WRITE_ATTEMPTS", 3)
    calls = {"count": 0}

    def always_fails() -> None:
        calls["count"] += 1
        raise OSError(122, "Disk quota exceeded")

    with pytest.raises(OSError, match="quota"):
        module.retry_write(always_fails, "demo")
    assert calls["count"] == 3


def test_resident_gb_prefers_anonymous_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hackathon.wmloc.research import b2_train

    status = tmp_path / "status"
    status.write_text("VmRSS:\t312475648 kB\nRssAnon:\t 2097152 kB\nRssFile:\t1048576 kB\n")
    monkeypatch.setattr(b2_train, "PROC_STATUS", status)
    assert b2_train.resident_gb() == pytest.approx(2.0)
    status.write_text("VmRSS:\t 3145728 kB\n")
    assert b2_train.resident_gb() == pytest.approx(3.0)
    status.write_text("Name:\tpython\n")
    assert b2_train.resident_gb() is None


def test_output_sink_single_mode_leaves_exactly_one_file(tmp_path: Path) -> None:
    single = tmp_path / "out" / "all.npz"
    single.parent.mkdir()
    sink = OutputSink(single)
    sink.save_text(tmp_path / "n_pyr_ps.json", '{"a": 1}')
    sink.save_npz(tmp_path / "n_pyr_ps.npz", {"done": np.array([True, False])})
    sink.save_npz(tmp_path / "fit_n_pyr_ps_s0_test.npz", {"pixelseal": np.zeros((2, 4, 4))})
    assert sorted(p.name for p in single.parent.iterdir()) == ["all.npz"]
    assert not any(tmp_path.glob("n_pyr_ps*"))
    with np.load(single) as archive:
        assert set(archive.files) == {
            "n_pyr_ps.json",
            "n_pyr_ps/done",
            "fit_n_pyr_ps_s0_test/pixelseal",
        }
        assert str(archive["n_pyr_ps.json"]) == '{"a": 1}'

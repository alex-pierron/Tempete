"""Tests of wmkit.seal and of the data/weights scripts.

Fast tests use a fake upstream model (no weights, no network). Tests marked `slow` load the
real pretrained checkpoints (downloaded on first use; Chunky Seal is skipped unless its
checkpoint is already cached, because the upstream file weighs 13 GB).
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from wmkit.config import data_dir
from wmkit.methods import METHODS, get_watermarker
from wmkit.seal import (
    SEAL_CARDS,
    SealBitsMismatchError,
    SealCardInfo,
    SealInputError,
    SealWatermarker,
    UnknownSealCardError,
    get_card_info,
    load_seal,
    resolve_card,
    slim_checkpoint_path,
)
from wmkit.types import Watermarker

SEED = 0
FAKE_BITS = 8
FAKE_IMG_SIZE = 16
FAKE_SCALING_W = 0.2
BATCH = 5
SMALL_BATCH_SIZE = 2
LARGE_BATCH_SIZE = 8
EXPECTED_BITS = {"pixelseal": 256, "videoseal_1.0": 256, "videoseal_0.0": 96, "chunkyseal": 1024}
MIN_REAL_BIT_ACC = 0.95
MIN_REAL_PSNR_DB = 38.0
KODAK_SAMPLE_COUNT = 2
VIDEO_FRAMES = 8
VIDEO_SIZE = 128
REPO_ROOT = Path(__file__).resolve().parents[1]


# --- a fake upstream model ---------------------------------------------------------------------


class _FakeBlender:
    def __init__(self) -> None:
        self.scaling_w = FAKE_SCALING_W


class _FakeDetector(nn.Module):
    """Linear read-out of the 16x16 image: [B, 1 + n_bits] (channel 0 = detection)."""

    def __init__(self, spatial_output: bool = False) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(SEED)
        self.weights = nn.Parameter(
            torch.randn(1 + FAKE_BITS, 3 * FAKE_IMG_SIZE * FAKE_IMG_SIZE, generator=generator) / 16,
            requires_grad=False,
        )
        self.spatial_output = spatial_output

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.shape[-2:] == (FAKE_IMG_SIZE, FAKE_IMG_SIZE), "detector got an unresized image"
        logits = images.flatten(1) @ self.weights.T
        if self.spatial_output:
            return logits[:, :, None, None].expand(-1, -1, 3, 3)
        return logits


class _FakeModel(nn.Module):
    """Mimics the subset of `videoseal.models.Videoseal` used by SealWatermarker."""

    def __init__(self, spatial_output: bool = False) -> None:
        super().__init__()
        self.blender = _FakeBlender()
        self.detector = _FakeDetector(spatial_output)
        self.embed_calls: list[tuple[int, bool, bool]] = []

    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor | None = None,
        is_video: bool = False,
        *,
        lowres_attenuation: bool = False,
    ) -> dict[str, torch.Tensor]:
        assert msgs is not None
        self.embed_calls.append((imgs.shape[0], is_video, lowres_attenuation))
        shift = (msgs.mean(dim=1) - 0.5).view(-1, 1, 1, 1) * 0.1 * self.blender.scaling_w
        if is_video:  # one message for the whole clip
            shift = shift[:1]
        return {"imgs_w": (imgs + shift).clamp(0, 1)}

    def detect(self, imgs: torch.Tensor, is_video: bool = False) -> dict[str, torch.Tensor]:
        resized = nn.functional.interpolate(imgs, size=(FAKE_IMG_SIZE, FAKE_IMG_SIZE))
        return {"preds": self.detector(resized)}


def _info() -> SealCardInfo:
    return SealCardInfo(
        name="fake", n_bits=FAKE_BITS, img_size=FAKE_IMG_SIZE, scaling_w=FAKE_SCALING_W,
        uses_jnd=False, checkpoint_url="https://invalid.example/none.pth",
    )


def _fake_watermarker(
    batch_size: int = LARGE_BATCH_SIZE, spatial_output: bool = False
) -> SealWatermarker:
    model: Any = _FakeModel(spatial_output)
    return SealWatermarker(model, _info(), torch.device("cpu"), batch_size=batch_size)


def _images(batch: int = BATCH, size: int = 40) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 1)
    return torch.rand(batch, 3, size, size, generator=generator) * 0.8 + 0.1


def _messages(batch: int = BATCH) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 2)
    return (torch.rand(batch, FAKE_BITS, generator=generator) > 0.5).float()


# --- cards -------------------------------------------------------------------------------------


@pytest.mark.parametrize("card", SEAL_CARDS)
def test_card_info_matches_upstream_yaml(card: str) -> None:
    info = get_card_info(card)
    assert info.name == card
    assert info.n_bits == EXPECTED_BITS[card]
    assert info.img_size == 256
    assert info.checkpoint_url.startswith("https://")


def test_alias_resolves_and_unknown_card_is_rejected() -> None:
    assert resolve_card("videoseal") == "videoseal_1.0"
    with pytest.raises(UnknownSealCardError, match="nope"):
        resolve_card("nope")
    with pytest.raises(UnknownSealCardError):
        load_seal(card="nope")


def test_card_strength_defaults_and_jnd_flag() -> None:
    assert get_card_info("pixelseal").scaling_w == pytest.approx(0.2)
    assert get_card_info("pixelseal").uses_jnd
    assert not get_card_info("videoseal_0.0").uses_jnd


def test_n_bits_mismatch_fails_before_any_download() -> None:
    with pytest.raises(SealBitsMismatchError, match="256"):
        load_seal(n_bits=64, card="pixelseal")


def test_registry_declares_the_four_seal_methods() -> None:
    for name in ("pixelseal", "videoseal", "videoseal0", "chunkyseal"):
        entry = METHODS[name]
        assert (entry.module, entry.factory) == ("wmkit.seal", "load_seal")
    assert callable(get_watermarker)


# --- plumbing with a fake model ----------------------------------------------------------------


def test_fake_watermarker_satisfies_protocol() -> None:
    watermarker = _fake_watermarker()
    assert isinstance(watermarker, Watermarker)
    assert watermarker.n_bits == FAKE_BITS and watermarker.name == "fake"


def test_embed_returns_same_shape_in_range_and_batches_consistently() -> None:
    images, messages = _images(), _messages()
    small = _fake_watermarker(SMALL_BATCH_SIZE)
    large = _fake_watermarker(LARGE_BATCH_SIZE)
    out_small, out_large = small.embed(images, messages), large.embed(images, messages)
    assert out_small.shape == images.shape and out_small.dtype == torch.float32
    assert float(out_small.min()) >= 0.0 and float(out_small.max()) <= 1.0
    assert torch.allclose(out_small, out_large)
    model: Any = small._model
    assert [call[0] for call in model.embed_calls] == [2, 2, 1]


def test_embed_passes_lowres_attenuation_flag() -> None:
    watermarker = _fake_watermarker()
    watermarker.lowres_attenuation = False
    watermarker.embed(_images(1), _messages(1))
    model: Any = watermarker._model
    assert model.embed_calls[-1] == (1, False, False)


def test_extract_accepts_any_size_and_returns_probabilities() -> None:
    watermarker = _fake_watermarker(SMALL_BATCH_SIZE)
    for size in (16, 40, 100):
        soft = watermarker.extract(_images(BATCH, size))
        assert soft.shape == (BATCH, FAKE_BITS)
        assert float(soft.min()) >= 0.0 and float(soft.max()) <= 1.0
    assert watermarker.extract(_images(0)).shape == (0, FAKE_BITS)


def test_extract_matches_sigmoid_of_logits_and_batching_does_not_matter() -> None:
    images = _images()
    watermarker = _fake_watermarker(SMALL_BATCH_SIZE)
    assert torch.allclose(watermarker.extract(images), torch.sigmoid(watermarker.extract_logits(images)))


def test_extract_logits_is_differentiable_and_ignores_detection_channel() -> None:
    watermarker = _fake_watermarker()
    images = _images(2).requires_grad_(True)
    logits = watermarker.extract_logits(images)
    assert logits.shape == (2, FAKE_BITS)
    logits.sum().backward()
    assert images.grad is not None and float(images.grad.abs().sum()) > 0.0
    model: Any = watermarker._model
    resized = nn.functional.interpolate(
        images.detach(), size=(FAKE_IMG_SIZE, FAKE_IMG_SIZE), mode="bilinear", antialias=True
    )
    assert torch.allclose(logits, model.detector(resized)[:, 1:], atol=1e-5)


def test_pixelwise_detector_output_is_averaged() -> None:
    watermarker = _fake_watermarker(spatial_output=True)
    assert watermarker.extract_logits(_images(2)).shape == (2, FAKE_BITS)


def test_single_message_is_broadcast_and_bad_messages_rejected() -> None:
    watermarker = _fake_watermarker()
    assert watermarker.embed(_images(3), _messages(1)).shape == (3, 3, 40, 40)
    assert watermarker.embed(_images(3), _messages(1)[0]).shape == (3, 3, 40, 40)
    with pytest.raises(SealInputError, match="messages"):
        watermarker.embed(_images(3), torch.zeros(3, FAKE_BITS + 1))
    with pytest.raises(SealInputError, match="messages"):
        watermarker.embed(_images(3), _messages(2))


def test_bad_images_are_rejected() -> None:
    watermarker = _fake_watermarker()
    with pytest.raises(SealInputError, match="images"):
        watermarker.embed(torch.rand(3, 40, 40), _messages(1))
    with pytest.raises(SealInputError, match="images"):
        watermarker.extract(torch.rand(1, 1, 40, 40))


def test_scaling_w_property_overrides_blender_and_validates() -> None:
    watermarker = _fake_watermarker()
    assert watermarker.scaling_w == pytest.approx(FAKE_SCALING_W)
    watermarker.scaling_w = 0.5
    model: Any = watermarker._model
    assert model.blender.scaling_w == 0.5
    with pytest.raises(ValueError, match="scaling_w"):
        watermarker.scaling_w = -1.0
    with pytest.raises(ValueError, match="batch_size"):
        SealWatermarker(model, _info(), torch.device("cpu"), batch_size=0)


def test_video_helpers_use_one_message_per_clip() -> None:
    watermarker = _fake_watermarker()
    frames = _images(6, size=24)
    marked = watermarker.embed_video(frames, _messages(1)[0])
    assert marked.shape == frames.shape
    model: Any = watermarker._model
    assert model.embed_calls[-1][1] is True
    soft = watermarker.extract_video(marked)
    assert soft.shape == (FAKE_BITS,)
    with pytest.raises(SealInputError, match="frames"):
        watermarker.extract_video(torch.rand(3, 24, 24))


# --- data and weights scripts ------------------------------------------------------------------


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_bytes(value: int, size: int = 8) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_blob(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for index, name in enumerate(names):
            archive.writestr(name, _png_bytes(index))
    return buffer.getvalue()


def test_evenly_spaced_is_deterministic_and_validated() -> None:
    fetch_data = _load_script("fetch_data")
    names = [f"{index:04d}.jpg" for index in range(100)]
    picked = fetch_data.evenly_spaced(names, 10)
    assert picked == fetch_data.evenly_spaced(list(reversed(names)), 10)
    assert len(set(picked)) == 10 and picked[0] == names[0]
    assert fetch_data.evenly_spaced(names, 100) == names
    for bad in (0, 101):
        with pytest.raises(fetch_data.DataFetchError):
            fetch_data.evenly_spaced(names, bad)


def test_range_reader_serves_a_zip_through_range_requests(tmp_path: Path) -> None:
    fetch_data = _load_script("fetch_data")
    names = [f"folder/{index:04d}.png" for index in range(6)]
    blob = _zip_blob(names + ["folder/readme.txt"])
    requests: list[tuple[int, int]] = []

    def fetch(start: int, end: int) -> bytes:
        requests.append((start, end))
        return blob[start : end + 1]

    stored = fetch_data._fetch_from_zip(
        "memory://zip", tmp_path / "out", ".png", 3, tmp_path,
        open_range=lambda _url: (len(blob), fetch),
    )
    assert stored == 3
    assert len(list((tmp_path / "out").glob("*.png"))) == 3
    assert not list((tmp_path / "out").glob("*.part"))
    requests_after_first = len(requests)
    again = fetch_data._fetch_from_zip(
        "memory://zip", tmp_path / "out", ".png", 3, tmp_path,
        open_range=lambda _url: (len(blob), fetch),
    )
    assert again == 3
    # Idempotent: the second run only re-reads the zip directory, never the image bytes.
    directory_offset = zipfile.ZipFile(io.BytesIO(blob)).start_dir
    assert all(start >= directory_offset for start, _ in requests[requests_after_first:])


def test_corrupt_download_is_rejected_and_budget_enforced(tmp_path: Path) -> None:
    fetch_data = _load_script("fetch_data")
    with pytest.raises(fetch_data.DataFetchError, match="not a valid image"):
        fetch_data._store_image(tmp_path / "bad.png", b"definitely not a png")
    assert not (tmp_path / "bad.png").exists()
    (tmp_path / "blob.bin").write_bytes(b"x" * 10)
    with pytest.raises(fetch_data.DataFetchError, match="budget"):
        fetch_data._check_budget(tmp_path, fetch_data.DATA_BUDGET_BYTES)


def test_fetch_scripts_expose_expected_cli() -> None:
    fetch_data, fetch_weights = _load_script("fetch_data"), _load_script("fetch_weights")
    parsed = fetch_data.build_parser().parse_args(["--dataset", "coco_val_subset"])
    assert parsed.n == fetch_data.COCO_DEFAULT_COUNT
    assert set(fetch_data.DATASETS) == {"kodak", "div2k_valid", "coco_val_subset"}
    assert "chunkyseal" not in fetch_weights.DEFAULT_CARDS


# --- real checkpoints (slow) -------------------------------------------------------------------


def _kodak_sample() -> torch.Tensor:
    folder = data_dir() / "kodak"
    paths = sorted(folder.glob("*.png"))[:KODAK_SAMPLE_COUNT]
    if len(paths) < KODAK_SAMPLE_COUNT:
        pytest.skip("Kodak not fetched: run scripts/fetch_data.py --dataset kodak")
    arrays = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in paths]
    shapes = {array.shape for array in arrays}
    assert len(shapes) == 1, "first Kodak images are expected to share one orientation"
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(torch.tensor(1.0 / float(((a - b) ** 2).mean()))))


@pytest.mark.slow
@pytest.mark.parametrize("card", SEAL_CARDS)
def test_real_card_round_trip_on_kodak(card: str) -> None:
    if card == "chunkyseal" and not slim_checkpoint_path(card).exists():
        pytest.skip("chunkyseal checkpoint (13.4 GB upstream) is not cached; not downloaded by default")
    images = _kodak_sample()
    watermarker = load_seal(card=card)
    assert isinstance(watermarker, Watermarker)
    assert watermarker.n_bits == EXPECTED_BITS[card]
    generator = torch.Generator().manual_seed(SEED)
    messages = (torch.rand(images.shape[0], watermarker.n_bits, generator=generator) > 0.5).float()
    marked = watermarker.embed(images, messages)
    assert marked.shape == images.shape
    quantised = (marked * 255).round() / 255
    soft = watermarker.extract(quantised)
    accuracy = float(((soft > 0.5).float() == messages).float().mean())
    assert accuracy >= MIN_REAL_BIT_ACC, f"{card}: bit accuracy {accuracy:.3f}"
    assert _psnr(images, quantised) >= MIN_REAL_PSNR_DB
    cropped = watermarker.extract(quantised[:, :, : images.shape[2] // 2, : images.shape[3] // 2])
    assert cropped.shape == soft.shape


@pytest.mark.slow
def test_real_extract_logits_matches_upstream_detect() -> None:
    images = _kodak_sample()
    watermarker = load_seal(card="pixelseal")
    ours = watermarker.extract_logits(images)
    reference = watermarker.model.detect(images, is_video=False)["preds"][:, 1:]  # type: ignore[operator]
    assert torch.allclose(ours, reference, atol=1e-4)


@pytest.mark.slow
def test_real_scaling_w_trades_quality_for_strength() -> None:
    images = _kodak_sample()[:1]
    messages = _messages_for(256)
    weak = load_seal(card="pixelseal", scaling_w=0.05)
    strong = load_seal(card="pixelseal", scaling_w=0.4)
    assert _psnr(images, weak.embed(images, messages)) > _psnr(images, strong.embed(images, messages))


def _messages_for(n_bits: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return (torch.rand(1, n_bits, generator=generator) > 0.5).float()


@pytest.mark.slow
def test_real_video_round_trip() -> None:
    watermarker = load_seal(card="pixelseal")
    base = _kodak_sample()[0:1, :, :VIDEO_SIZE * 2, :VIDEO_SIZE * 3]
    frames = base.expand(VIDEO_FRAMES, -1, -1, -1).contiguous()
    message = _messages_for(watermarker.n_bits)
    marked = watermarker.embed_video(frames, message[0])
    assert marked.shape == frames.shape
    soft = watermarker.extract_video(marked)
    accuracy = float(((soft > 0.5).float() == message[0]).float().mean())
    assert accuracy >= MIN_REAL_BIT_ACC

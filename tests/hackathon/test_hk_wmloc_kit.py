"""Tests of the starter-kit integration: path-loaded wrappers and the release manifest builder.

The kit is faked with a tiny `wmloc/watermarks.py` written to a temp folder: nothing here needs
the real kit, models or network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("PIL")

from PIL import Image

from hackathon.wmloc import kit_backend
from hackathon.wmloc.constants import IMAGE_SIZE, METHOD_PIXELSEAL, METHOD_TRUSTMARK
from hackathon.wmloc.errors import LocalBackendError, ManifestError, UnsupportedMethodError
from hackathon.wmloc.manifest import (
    GROUP_FULL_WATERMARK,
    GROUP_RECT_WATERMARK,
    manifest_from_release,
    read_manifest,
    write_manifest,
)

FAKE_KIT = """
import torch

class Fake:
    def __init__(self, name, device):
        self.name, self.device = name, device
        self.nbits = 256 if name == "pixelseal" else 100
        self.resolution = 256 if name == "pixelseal" else 245

    def logits(self, images):
        return torch.ones(images.shape[0], self.nbits)

    def features(self, images):
        return {"per_position_logits": torch.ones(images.shape[0], self.nbits, 8, 8)}

def load_watermark(name, models_dir=None, device="cuda"):
    return Fake(name, device)
"""


@pytest.fixture
def fake_kit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    kit = tmp_path / "kit"
    (kit / "wmloc").mkdir(parents=True)
    (kit / "wmloc" / "watermarks.py").write_text(FAKE_KIT, encoding="utf-8")
    monkeypatch.setenv(kit_backend.KIT_DIR_ENV, str(kit))
    monkeypatch.setenv(kit_backend.MODELS_DIR_ENV, str(tmp_path / "models"))
    kit_backend._MODULE_CACHE.clear()
    return kit


def test_kit_is_loaded_by_path_without_touching_sys_path_or_shadowing_wmloc(
    fake_kit: Path,
) -> None:
    before = list(sys.path)
    module = kit_backend.load_kit_module()
    assert sys.path == before
    assert "wmloc" not in sys.modules or sys.modules["wmloc"] is not module
    assert module.__name__ == kit_backend.KIT_MODULE_NAME


def test_pixelseal_backend_exposes_the_spatial_map(fake_kit: Path) -> None:
    backend = kit_backend.make_backend(METHOD_PIXELSEAL, "cpu")
    assert backend is not None
    assert (backend.n_bits, backend.native_size, backend.map_stride) == (256, 256, 32)
    images = torch.rand(70, 3, 64, 64)  # more than one decode chunk
    assert backend.bit_logits(images).shape == (70, 256)
    assert backend.logit_map(torch.rand(2, 3, 256, 256)).shape == (2, 256, 8, 8)
    with pytest.raises(UnsupportedMethodError, match="256x256"):
        backend.logit_map(torch.rand(1, 3, 100, 100))


def test_decode_only_methods_have_no_spatial_map(fake_kit: Path) -> None:
    backend = kit_backend.make_backend(METHOD_TRUSTMARK, "cpu")
    assert backend is not None and backend.map_stride is None
    assert (backend.n_bits, backend.native_size) == (100, 245)
    with pytest.raises(UnsupportedMethodError, match="no per-position"):
        backend.logit_map(torch.rand(1, 3, 245, 245))


def test_unknown_method_gives_no_backend(fake_kit: Path) -> None:
    assert kit_backend.make_backend("watermark-anything", "cpu") is None


def test_missing_kit_is_a_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kit_backend.KIT_DIR_ENV, str(tmp_path / "nowhere"))
    kit_backend._MODULE_CACHE.clear()
    with pytest.raises(LocalBackendError, match="starter kit not found"):
        kit_backend.make_backend(METHOD_PIXELSEAL, "cpu")


def test_missing_dependency_inside_the_kit_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kit = tmp_path / "kit"
    (kit / "wmloc").mkdir(parents=True)
    (kit / "wmloc" / "watermarks.py").write_text("import a_module_that_does_not_exist\n")
    monkeypatch.setenv(kit_backend.KIT_DIR_ENV, str(kit))
    kit_backend._MODULE_CACHE.clear()
    with pytest.raises(LocalBackendError, match="missing dependency"):
        kit_backend.load_kit_module()


def write_release_split(root: Path, with_masks: bool, flags: list[bool | None]) -> Path:
    split = root / "train" if with_masks else root / "test"
    (split / "images").mkdir(parents=True)
    if with_masks:
        (split / "masks").mkdir()
    records = []
    for index, flag in enumerate(flags):
        name = f"img{index}"
        Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)).save(
            split / "images" / f"{name}.png"
        )
        record: dict[str, object] = {"image_name": name, "method": "pixelseal", "message": "0101"}
        if flag is not None:
            record["full_watermark"] = flag
        if with_masks:
            Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)).save(
                split / "masks" / f"{name}.png"
            )
        records.append(record)
    (split / "metadata.json").write_text(json.dumps(records), encoding="utf-8")
    return split


def test_manifest_from_a_train_split_carries_masks_and_groups(tmp_path: Path) -> None:
    split = write_release_split(tmp_path, True, [True, False])
    samples = manifest_from_release(split)
    assert [s.name for s in samples] == ["img0", "img1"]
    assert [s.group for s in samples] == [GROUP_FULL_WATERMARK, GROUP_RECT_WATERMARK]
    assert samples[0].mask512 == "masks/img0.png" and samples[0].image == "images/img0.png"
    write_manifest(split / "manifest.json", samples)
    assert read_manifest(split / "manifest.json") == samples


def test_manifest_from_a_test_split_has_no_masks(tmp_path: Path) -> None:
    split = write_release_split(tmp_path, False, [None, None])
    samples = manifest_from_release(split)
    assert all(s.mask512 is None and s.group is None for s in samples)


def test_manifest_from_release_errors(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="metadata not found"):
        manifest_from_release(tmp_path)
    split = write_release_split(tmp_path, True, [False])
    (split / "images" / "img0.png").unlink()
    with pytest.raises(ManifestError, match="file not found"):
        manifest_from_release(split)
    (split / "metadata.json").write_text('[{"image_name": "x"}]')
    with pytest.raises(ManifestError, match="malformed"):
        manifest_from_release(split)

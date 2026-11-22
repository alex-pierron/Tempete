"""Tests of model.py: preprocessing identical to the server, prediction, state dict export."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.unlearning.constants import (
    FORGET_IMAGES_FILENAME,
    MODEL_FILENAME,
    MODEL_SIZE_BYTES,
    NUM_CLASSES,
)
from hackathon.unlearning.data import load_forget_images
from hackathon.unlearning.fetch import model_matches_official
from hackathon.unlearning.model import (
    ModelLoadError,
    cpu_float32_state_dict,
    load_model,
    load_state_dict_file,
    predict,
    preprocess,
    select_device,
)

REAL_DIR = Path(os.environ.get("HACKATHON_DATA_DIR", "data/hackathon")) / "unlearning"


def test_preprocess_matches_the_official_formula() -> None:
    pixels = torch.tensor([0, 51, 128, 255], dtype=torch.uint8)
    # A [1, 224, 224, 3] image whose first row starts with the four test values (all channels).
    full = torch.zeros(1, 224, 224, 3, dtype=torch.uint8)
    full[0, 0, :4, :] = pixels.view(4, 1)
    out = preprocess(full, "cpu")
    assert out.shape == (1, 3, 224, 224) and out.dtype == torch.float32
    expected = (pixels.float() / 255.0 - 0.5) / 0.5
    assert torch.allclose(out[0, 0, 0, :4], expected)
    assert float(out.min()) == -1.0 and float(out.max()) == 1.0


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1, 224, 224, 3, dtype=torch.float32),
        torch.zeros(1, 3, 224, 224, dtype=torch.uint8),
        torch.zeros(1, 100, 100, 3, dtype=torch.uint8),
    ],
)
def test_preprocess_rejects_wrong_dtype_or_layout(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="Expected uint8"):
        preprocess(bad, "cpu")


def test_predict_returns_cpu_float32_logits_and_is_batch_invariant() -> None:
    model = tiny_model()
    images = random_images(5)
    full = predict(model, images, batch_size=5)
    chunked = predict(model, images, batch_size=2)
    assert full.shape == (5, NUM_CLASSES) and full.dtype == torch.float32
    assert full.device.type == "cpu"
    assert torch.allclose(full, chunked, atol=1e-5)


def test_predict_on_no_image_returns_an_empty_matrix() -> None:
    assert predict(tiny_model(), random_images(0)).shape == (0, NUM_CLASSES)


def test_cpu_float32_state_dict_casts_copies_and_does_not_alias() -> None:
    model = tiny_model().half()
    state = cpu_float32_state_dict(model)
    assert all(v.dtype == torch.float32 for v in state.values() if v.is_floating_point())
    key = next(iter(state))
    original = state[key].clone()
    next(model.parameters()).data.add_(1.0)
    assert torch.equal(state[key], original)


def test_select_device_cpu_and_unavailable_accelerator() -> None:
    assert select_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(ModelLoadError, match="CUDA"):
            select_device("cuda")


def test_load_state_dict_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        load_state_dict_file(tmp_path / "missing.pt")
    torch.save([1, 2], tmp_path / "list.pt")
    with pytest.raises(ModelLoadError, match="state dict"):
        load_state_dict_file(tmp_path / "list.pt")


@pytest.mark.slow
@pytest.mark.skipif(not (REAL_DIR / MODEL_FILENAME).is_file(), reason="run `fetch` first")
def test_real_model_loads_strictly_and_is_memorised_on_the_forget_set() -> None:
    path = REAL_DIR / MODEL_FILENAME
    assert path.stat().st_size == MODEL_SIZE_BYTES and model_matches_official(path)
    model = load_model(path, "cpu")
    images = load_forget_images(REAL_DIR / FORGET_IMAGES_FILENAME, limit=4)
    probabilities = torch.softmax(predict(model, images, batch_size=4), dim=1)
    assert float(probabilities.max(dim=1).values.min()) > 0.9

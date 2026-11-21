"""Slow tests with the real Pixel Seal weights (skipped when wmkit or the checkpoint is absent).

Run: WMKIT_WEIGHTS_DIR=<repo>/weights HACKATHON_DATA_DIR=<repo>/data/hackathon pytest -m slow tests/hackathon/test_hk_wmloc_pixelseal_slow.py
"""

from __future__ import annotations

import pytest
import torch

from hackathon.wmloc.constants import METHOD_PIXELSEAL, PIXELSEAL_N_BITS
from hackathon.wmloc.errors import LocalBackendError
from hackathon.wmloc.local_backend import PixelSealBackend, load_local_pixelseal
from hackathon.wmloc.localizers import SlidingWindowLocalizer, bit_agreement

pytestmark = pytest.mark.slow

SEED = 11
MAP_SIDE = 8
RECT = (96, 96, 288, 320)  # y0, x0, height, width
MIN_INSIDE_AGREEMENT = 0.9
MAX_OUTSIDE_AGREEMENT = 0.6
LOGIT_TOLERANCE = 1e-3
INSET = 32
"""A window much smaller than the embedding rectangle decodes worse (scale mismatch)."""
PHOTO_NAME = "kodim05.png"


@pytest.fixture(scope="module")
def backend() -> PixelSealBackend:
    try:
        return load_local_pixelseal("cpu")
    except LocalBackendError as exc:
        pytest.skip(str(exc))


def real_photo() -> torch.Tensor:
    """A Kodak photo from <data_dir>/../kodak (a synthetic texture does not carry the mark: the
    JND mask hides nothing in flat areas and white grain destroys it)."""
    pytest.importorskip("PIL")
    from hackathon.common.config import load_settings
    from hackathon.wmloc.synth import load_square_image

    path = load_settings().data_dir.parent / "kodak" / PHOTO_NAME
    if not path.is_file():
        pytest.skip(f"photo not found: {path} (set HACKATHON_DATA_DIR)")
    return load_square_image(path)


def test_spatial_map_averages_to_the_global_logits(backend: PixelSealBackend) -> None:
    image = real_photo()[None]
    native = torch.nn.functional.interpolate(
        image, size=(256, 256), mode="bilinear", antialias=True
    )
    spatial = backend.logit_map(native)
    assert spatial.shape == (1, PIXELSEAL_N_BITS, MAP_SIDE, MAP_SIDE)
    assert torch.allclose(spatial.mean(dim=(2, 3)), backend.bit_logits(image), atol=LOGIT_TOLERANCE)


def test_window_inside_the_marked_rectangle_decodes_and_outside_does_not(
    backend: PixelSealBackend,
) -> None:
    generator = torch.Generator().manual_seed(SEED)
    message = torch.randint(0, 2, (PIXELSEAL_N_BITS,), generator=generator).float()
    base = real_photo()
    y0, x0, height, width = RECT
    marked = base.clone()
    marked[:, y0 : y0 + height, x0 : x0 + width] = backend.embed(
        base[None, :, y0 : y0 + height, x0 : x0 + width], message
    )[0]
    inside = backend.bit_logits(
        marked[None, :, y0 + INSET : y0 + height - INSET, x0 + INSET : x0 + width - INSET]
    )
    outside = backend.bit_logits(marked[None, :, 0:64, 0:64])
    assert bit_agreement(inside, message, 1).item() >= MIN_INSIDE_AGREEMENT
    assert bit_agreement(outside, message, 1).item() <= MAX_OUTSIDE_AGREEMENT
    mask = SlidingWindowLocalizer(backend, (192, 256), 64, 0.8).predict(
        (marked * 255).round() / 255, METHOD_PIXELSEAL, message
    )
    assert int(mask.sum()) > 0

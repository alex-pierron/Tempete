"""Team C dense window map: grid, union rule and storage round trip, with a fake decoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hackathon.wmloc.research.c_dense_windows import (
    DenseMap,
    DenseWindowError,
    dense_map,
    dense_origins,
    from_arrays,
    to_arrays,
    union_mask,
)

N_BITS = 8
RESOLUTION = 16
MESSAGE = np.ones(N_BITS, dtype=np.float32)


class LeftHalfDecoder:
    """Decodes the message (all logits positive) iff the crop's mean brightness is high."""

    method = "trustmark"
    resolution = RESOLUTION
    device = "cpu"

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        bright = (images.mean(dim=(1, 2, 3)) > 0.5).float() * 2 - 1
        return bright[:, None].repeat(1, N_BITS)


def test_dense_origins_cover_the_axis_with_a_flush_last_window() -> None:
    origins = dense_origins(64)
    assert origins[0] == 0 and origins[-1] == 512 - 64
    assert all(int(o) % 4 == 0 for o in origins)
    with pytest.raises(DenseWindowError):
        dense_origins(0)


def test_union_mask_covers_exactly_the_windows_that_decode() -> None:
    origins = np.array([0, 64], dtype=np.int32)
    agreement = np.array([[1.0, 0.5], [0.5, 0.5]], dtype=np.float32)
    evidence = np.zeros_like(agreement)
    mask = union_mask([DenseMap(64, origins, agreement, evidence)], 0.8)
    assert mask.shape == (128, 128)
    assert int(mask.sum()) == 16 * 16 and mask[:16, :16].all()
    assert union_mask([DenseMap(64, origins, agreement * 0.4, evidence)], 0.8).sum() == 0


def test_union_mask_evidence_floor_removes_weak_windows() -> None:
    origins = np.array([0], dtype=np.int32)
    weak = DenseMap(32, origins, np.ones((1, 1), np.float32), np.full((1, 1), 2.0, np.float32))
    assert union_mask([weak], 0.8).sum() == 8 * 8
    assert union_mask([weak], 0.8, min_evidence=5.0).sum() == 0


def test_dense_map_finds_the_bright_region_and_roundtrips_through_arrays() -> None:
    image = torch.zeros(3, 512, 512)
    image[:, :, :256] = 1.0  # bright left half: windows fully inside decode
    maps = dense_map(LeftHalfDecoder(), image, MESSAGE, sizes=(64,), stride_fraction=0.5)
    assert len(maps) == 1 and maps[0].agreement.shape == (len(maps[0].origins),) * 2
    mask = union_mask(maps, 0.9)
    assert mask[:, :64].all() and not mask[:, 70:].any()  # left half marked, right half not
    restored = from_arrays(to_arrays(maps))
    assert restored[0].size == 64
    assert np.allclose(
        restored[0].agreement, maps[0].agreement.astype(np.float16).astype(np.float32)
    )


def test_from_arrays_rejects_empty_input() -> None:
    with pytest.raises(DenseWindowError):
        from_arrays({"global_agreement": np.float32(0.5)})

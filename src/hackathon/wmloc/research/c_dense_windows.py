"""Dense map of window decodes against the known message (team C).

For a set of square window sizes, every window on a regular grid is decoded and its bit agreement
with the known message and its evidence (t statistic of the signed logits) are stored. A window
that decodes proves that all its cells are watermarked, so the union of the decoding windows is a
mask of the marked region, valid for silhouettes as well as for rectangles when the decoder
tolerates windows INSIDE the marked region (Pixel Seal does: evidence 15-25 at 0.6-0.85 of the
embedding scale; TrustMark and StegaStamp tolerate much less).

The rectangle search (`c_rect_search`) looks for ONE box; this map keeps every window so that the
union rule, its threshold and its combination with other predictors are studied offline on the
cached numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE
from hackathon.wmloc.research.c_rect_search import (
    Box,
    DecoderWindowScorer,
    LogitDecoder,
    window_origins,
)
from hackathon.wmloc.research.errors import ResearchError

DEFAULT_SIZES: tuple[int, ...] = (32, 48, 64, 96, 128)
DEFAULT_STRIDE_FRACTION = 1.0 / 3.0
"""Windows overlap by two thirds: a window inside the marked region always has a neighbour that is
shifted by less than the decoder's shift tolerance."""
SCORE_CHUNK = 512
"""Windows scored per call (bounds the memory of the crop batch)."""
CELL_PX = IMAGE_SIZE // GRID_SIZE
EVIDENCE_CLIP = 1000.0
"""Stored evidence is clipped (float16 range); real values stay below 100."""


class DenseWindowError(ResearchError):
    """Invalid window sizes or a malformed stored map."""


@dataclass(frozen=True)
class DenseMap:
    """Decode scores of the windows of one size.

    Attributes:
        size: Window side in pixels.
        origins: Window start offsets along one axis, [n].
        agreement: Bit agreement with the message, [n, n] (row = top, column = left).
        evidence: t statistic of the signed logits, [n, n].
    """

    size: int
    origins: np.ndarray
    agreement: np.ndarray
    evidence: np.ndarray


def dense_origins(size: int, stride_fraction: float = DEFAULT_STRIDE_FRACTION) -> np.ndarray:
    """Start offsets of the windows of `size` along an axis (stride a multiple of one cell).

    Raises:
        DenseWindowError: size not in (0, 512].
    """
    if not 0 < size <= IMAGE_SIZE:
        raise DenseWindowError(f"bad window size {size}")
    return np.asarray(window_origins(IMAGE_SIZE, size, stride_fraction), dtype=np.int32)


def dense_map(
    decoder: LogitDecoder,
    image: torch.Tensor,
    message: np.ndarray,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    stride_fraction: float = DEFAULT_STRIDE_FRACTION,
) -> list[DenseMap]:
    """Decode every window of every size; returns one `DenseMap` per size."""
    scorer = DecoderWindowScorer(decoder, image, message)
    maps: list[DenseMap] = []
    for size in sizes:
        origins = dense_origins(size, stride_fraction)
        boxes = [
            Box(int(t), int(left), int(t) + size, int(left) + size)
            for t in origins
            for left in origins
        ]
        agreement = np.empty(len(boxes), dtype=np.float32)
        evidence = np.empty(len(boxes), dtype=np.float32)
        for start in range(0, len(boxes), SCORE_CHUNK):
            scores = scorer.score(boxes[start : start + SCORE_CHUNK])
            agreement[start : start + len(scores.agreement)] = scores.agreement
            evidence[start : start + len(scores.evidence)] = scores.evidence
        shape = (len(origins), len(origins))
        maps.append(DenseMap(size, origins, agreement.reshape(shape), evidence.reshape(shape)))
    return maps


def union_mask(
    maps: list[DenseMap], agreement_threshold: float, min_evidence: float | None = None
) -> np.ndarray:
    """128 x 128 mask of the cells covered by at least one window that decodes.

    A window decodes when its agreement reaches `agreement_threshold` (and, if given, its evidence
    reaches `min_evidence`). Returns zeros when no window decodes.
    """
    covered = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for dense in maps:
        passing = dense.agreement >= agreement_threshold
        if min_evidence is not None:
            passing &= dense.evidence >= min_evidence
        span = dense.size // CELL_PX
        for row, column in zip(*np.nonzero(passing), strict=True):
            top = int(dense.origins[row]) // CELL_PX
            left = int(dense.origins[column]) // CELL_PX
            covered[top : top + span, left : left + span] = 1
    return covered


def to_arrays(maps: list[DenseMap]) -> dict[str, np.ndarray]:
    """Flat array dict of the maps for the chunk writer (float16 scores, int16 origins)."""
    arrays: dict[str, np.ndarray] = {}
    for dense in maps:
        arrays[f"origins_s{dense.size}"] = dense.origins.astype(np.int16)
        arrays[f"ag_s{dense.size}"] = dense.agreement.astype(np.float16)
        clipped = np.clip(dense.evidence, -EVIDENCE_CLIP, EVIDENCE_CLIP)
        arrays[f"ev_s{dense.size}"] = clipped.astype(np.float16)
    return arrays


def from_arrays(arrays: dict[str, np.ndarray]) -> list[DenseMap]:
    """Inverse of `to_arrays` (sizes sorted ascending).

    Raises:
        DenseWindowError: no map in the arrays.
    """
    sizes = sorted(int(key.removeprefix("ag_s")) for key in arrays if key.startswith("ag_s"))
    if not sizes:
        raise DenseWindowError("no dense map in the stored arrays")
    return [
        DenseMap(
            size,
            arrays[f"origins_s{size}"].astype(np.int32),
            arrays[f"ag_s{size}"].astype(np.float32),
            arrays[f"ev_s{size}"].astype(np.float32),
        )
        for size in sizes
    ]

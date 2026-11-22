"""Out-of-fold inputs of the gap analysis (team gap): B2 head probabilities, truth, regime, boxes.

Everything here is READ from the caches of teams B2 and C (no model is run): the 5-fold
out-of-fold probabilities of the learned heads, the whole-image decode of the known message and
the rectangles found by the window search on the train images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hackathon.wmloc.research.b2_candidate import OVERLAY_MIN_CONFIDENCE
from hackathon.wmloc.research.b2_pack import load_global_table
from hackathon.wmloc.research.b2_train import results_dir
from hackathon.wmloc.research.c_candidate import accepted_box
from hackathon.wmloc.research.c_results import ResultStoreError, load_all
from hackathon.wmloc.research.data import SPLIT_TRAIN, data_root, load_records, load_truth
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, stratified_folds

METHODS: tuple[str, ...] = ("pixelseal", "trustmark", "stegastamp")
RECT_CACHE_SUBDIR = "research_cache_c"
RECT_TRAIN_GLOBS: tuple[str, ...] = ("job*/rect_{method}_train", "job*/rect_{method}_train_fine")

OOF_TAGS_OF_BEST: dict[str, tuple[str, ...]] = {
    "pixelseal": ("ps_e20_d5",),
    "trustmark": ("e_spatial3_loc",),
    "stegastamp": ("e_spatial3_loc",),
}
"""Out-of-fold runs that mirror the heads inside the 0.5819 candidate (b2_v7c): the separate
Pixel Seal head and the joint 3-method head for TrustMark and StegaStamp."""

OOF_TAGS_AVAILABLE: dict[str, tuple[str, ...]] = {
    "pixelseal": ("ps_e20_d5", "ps_lov", "ps_e20", "e_spatial3_loc"),
    "trustmark": ("e_spatial3_loc", "tm_e20_d5"),
    "stegastamp": ("e_spatial3_loc", "ss_e20_d5"),
}
"""Every out-of-fold run with extractor evidence, per method (ensembling candidates)."""


class GapDataError(ResearchError):
    """An out-of-fold cache needed by the gap analysis is missing or inconsistent."""


@dataclass(frozen=True)
class MethodOof:
    """Out-of-fold view of one method (170 train images, metadata order).

    Attributes:
        method: Method name.
        names: Image names.
        truth: Ground truth [n, 128, 128], 1 = watermarked.
        is_full: Regime flag (True = whole image marked but a hole).
        folds: Fold id of each image (the split B2 trained with).
        agreement: Bit agreement of the whole-image decode with the known message.
        probabilities: `{tag: [n, 128, 128] float32}` out-of-fold probabilities of each run.
        boxes: `{row: (mask, confidence)}` rectangles of team C's window search (train images).
    """

    method: str
    names: list[str]
    truth: NDArray[np.uint8]
    is_full: NDArray[np.bool_]
    folds: NDArray[np.int64]
    agreement: NDArray[np.float64]
    probabilities: dict[str, NDArray[np.float32]]
    boxes: dict[int, tuple[NDArray[np.uint8], float]]

    @property
    def minority_fraction(self) -> NDArray[np.float64]:
        """Share of the image taken by the zone to find (hole if full, marked zone if rect)."""
        marked = self.truth.reshape(len(self.truth), -1).mean(axis=1)
        return np.where(self.is_full, 1.0 - marked, marked)


def load_probabilities(tag: str, method: str, directory: Path | None = None) -> NDArray[np.float32]:
    """Out-of-fold probabilities [n, 128, 128] of `method` in the CV run `tag`.

    Raises:
        GapDataError: file or method missing.
    """
    path = (directory or results_dir()) / f"{tag}.npz"
    if not path.is_file():
        raise GapDataError(f"missing out-of-fold run {path}")
    with np.load(path) as archive:
        if method not in archive.files:
            raise GapDataError(f"{path} has no {method}")
        return np.asarray(archive[method], dtype=np.float32)


def load_train_boxes(method: str, names: list[str]) -> dict[int, tuple[NDArray[np.uint8], float]]:
    """Accepted rectangles of team C on the train images, highest confidence per image.

    The acceptance rule is the one of the test overlay (`accepted_box`, then bit agreement at
    least `OVERLAY_MIN_CONFIDENCE`), so the train statistics transfer to the 112 test overlays.
    """
    index = {name: row for row, name in enumerate(names)}
    best: dict[int, tuple[NDArray[np.uint8], float]] = {}
    root = data_root() / RECT_CACHE_SUBDIR
    for pattern in RECT_TRAIN_GLOBS:
        for directory in sorted(root.glob(pattern.format(method=method))):
            try:
                payloads = load_all(directory)
            except ResultStoreError:
                continue
            for payload in payloads:
                row = index.get(str(payload.get("name")))
                box = accepted_box(payload, method)
                confidence = float(payload.get("agreement", 0.0))
                if row is None or box is None or confidence < OVERLAY_MIN_CONFIDENCE:
                    continue
                if row not in best or confidence > best[row][1]:
                    best[row] = (box.to_mask().astype(np.uint8), confidence)
    return best


def searched_rows(method: str, names: list[str]) -> set[int]:
    """Rows of the train images team C's search has processed (detected or not)."""
    index = {name: row for row, name in enumerate(names)}
    rows: set[int] = set()
    root = data_root() / RECT_CACHE_SUBDIR
    for pattern in RECT_TRAIN_GLOBS:
        for directory in sorted(root.glob(pattern.format(method=method))):
            try:
                payloads = load_all(directory)
            except ResultStoreError:
                continue
            rows.update(
                index[str(p["name"])]
                for p in payloads
                if p.get("searched") and str(p.get("name")) in index
            )
    return rows


def load_method(method: str, tags: tuple[str, ...]) -> MethodOof:
    """Assemble the out-of-fold view of `method` for the CV runs `tags`.

    Raises:
        GapDataError: a cache is missing or a run has another image count.
    """
    records = [r for r in load_records(SPLIT_TRAIN) if r.method == method]
    names = [r.name for r in records]
    is_full = np.array([bool(r.full_watermark) for r in records])
    folds = stratified_folds([f"{method}/{flag}" for flag in is_full], DEFAULT_FOLDS, DEFAULT_SEED)
    table = load_global_table(SPLIT_TRAIN, method)
    if table is None:
        raise GapDataError(f"no global decode table for {method}")
    agreement = np.array(
        [float(((table[r.name] > 0) == (r.message > 0.5)).mean()) for r in records],
        dtype=np.float64,
    )
    probabilities = {tag: load_probabilities(tag, method) for tag in tags}
    for tag, values in probabilities.items():
        if len(values) != len(names):
            raise GapDataError(f"{tag}/{method}: {len(values)} rows for {len(names)} images")
    return MethodOof(
        method=method,
        names=names,
        truth=load_truth(names),
        is_full=is_full,
        folds=np.asarray(folds, dtype=np.int64),
        agreement=agreement,
        probabilities=probabilities,
        boxes=load_train_boxes(method, names),
    )

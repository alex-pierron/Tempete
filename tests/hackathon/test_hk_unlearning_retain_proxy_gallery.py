"""Tests of the generated-image contact sheets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.research.retain_proxy.analyze import AnalysisError
from hackathon.unlearning.research.retain_proxy.gallery import (
    LABEL_WIDTH,
    PER_CLASS,
    TILE,
    top_images_per_class,
    write_gallery,
)
from hackathon.unlearning.research.retain_proxy.generate import chunk_path, save_chunk

BIG_LOGIT = 30.0


def _logits(classes: list[int], confident: list[bool]) -> np.ndarray:
    logits = np.zeros((len(classes), NUM_CLASSES), dtype=np.float32)
    for row, (class_index, sure) in enumerate(zip(classes, confident, strict=True)):
        if sure:
            logits[row, class_index] = BIG_LOGIT
    return logits


def test_top_images_keep_the_most_confident_per_class() -> None:
    classes = [3, 3, 3, 7, 9]
    logits = _logits(classes, [True] * 5)
    logits[1, 3] = 8.0  # still confident enough, but weaker than rows 0 and 2
    logits[4] = 0.0  # uniform: below the display confidence
    chosen = top_images_per_class(np.arange(5) + 100, logits, per_class=2)
    assert chosen == {3: [100, 102], 7: [103]}, "plan indices, best first, weak images dropped"


def test_gallery_sheets_have_one_row_per_class(tmp_path: Path) -> None:
    classes = [3] * 10 + [7] * 2
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (len(classes), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    save_chunk(
        chunk_path(tmp_path / "chunks", 0),
        np.arange(len(classes)),
        images,
        _logits(classes, [True] * len(classes)),
    )
    paths = write_gallery(tmp_path / "chunks", tmp_path / "out")
    assert [path.name for path in paths] == ["gallery_0.png"]
    with Image.open(paths[0]) as sheet:
        assert sheet.size == (LABEL_WIDTH + PER_CLASS * TILE, 2 * TILE), "classes 3 and 7"


def test_gallery_needs_chunks(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError):
        write_gallery(tmp_path, tmp_path / "out")

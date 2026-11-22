"""Contact sheets of the generated images, one row per class the original recognises.

For each class the original answers, the `PER_CLASS` most confident generated images are tiled in a
row; rows are packed into a few PNG sheets (a handful of small files, never one per image). This is
the visual proof, and the way to name the classes that no forget image shows.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hackathon.unlearning.constants import NUM_CLASSES
from hackathon.unlearning.research.retain_proxy.analyze import AnalysisError, load_logits, softmax
from hackathon.unlearning.research.retain_proxy.generate import CHUNK_FILE_TEMPLATE

PER_CLASS = 8
TILE = 96
LABEL_WIDTH = 34
CLASSES_PER_SHEET = 25
MIN_CONFIDENCE = 0.5
"""Images below this confidence are not shown: the row would not illustrate the class."""


def top_images_per_class(
    indices: np.ndarray, logits: np.ndarray, per_class: int = PER_CLASS
) -> dict[int, list[int]]:
    """Class -> plan indices of its `per_class` most confident images (confidence >= MIN)."""
    probabilities = softmax(logits)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    chosen: dict[int, list[int]] = {}
    for class_index in range(NUM_CLASSES):
        positions = np.nonzero((predicted == class_index) & (confidence >= MIN_CONFIDENCE))[0]
        if len(positions) == 0:
            continue
        best = positions[np.argsort(-confidence[positions])[:per_class]]
        chosen[class_index] = [int(indices[position]) for position in best]
    return chosen


def write_gallery(chunk_dir: Path, out_dir: Path) -> list[Path]:
    """Write `gallery_<k>.png` sheets into `out_dir` and return their paths.

    Raises:
        AnalysisError: no chunk found in `chunk_dir`.
    """
    indices, logits = load_logits(chunk_dir)
    chosen = top_images_per_class(indices, logits)
    wanted = {plan_index for plan_indices in chosen.values() for plan_index in plan_indices}
    pixels: dict[int, np.ndarray] = {}
    for path in sorted(chunk_dir.glob(CHUNK_FILE_TEMPLATE.replace("{first_index:07d}", "*"))):
        with np.load(path) as chunk:
            chunk_indices = chunk["indices"]
            rows = [row for row, value in enumerate(chunk_indices) if int(value) in wanted]
            if not rows:
                continue
            images = chunk["images"][rows]
        for row, image in zip(rows, images, strict=True):
            pixels[int(chunk_indices[row])] = np.asarray(
                Image.fromarray(image).resize((TILE, TILE), Image.Resampling.BILINEAR)
            )
    if not pixels:
        raise AnalysisError(f"No image to show in {chunk_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(chosen)
    written: list[Path] = []
    for sheet, start in enumerate(range(0, len(classes), CLASSES_PER_SHEET)):
        group = classes[start : start + CLASSES_PER_SHEET]
        canvas = Image.new("RGB", (LABEL_WIDTH + PER_CLASS * TILE, len(group) * TILE), "white")
        draw = ImageDraw.Draw(canvas)
        for row, class_index in enumerate(group):
            draw.text((3, row * TILE + TILE // 2 - 5), str(class_index), fill="black")
            for column, plan_index in enumerate(chosen[class_index]):
                canvas.paste(
                    Image.fromarray(pixels[plan_index]), (LABEL_WIDTH + column * TILE, row * TILE)
                )
        path = out_dir / f"gallery_{sheet}.png"
        canvas.save(path)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the contact sheets next to the chunks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = write_gallery(args.chunk_dir, args.out_dir or args.chunk_dir)
    print(f"gallery: {len(paths)} sheets in {args.out_dir or args.chunk_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

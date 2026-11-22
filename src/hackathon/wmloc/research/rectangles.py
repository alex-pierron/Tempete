"""Rectangle regime: seed a box from a coarse score map, then fit its edges on the exact-crop decode.

Why the exact crop: in the rectangle regime the embedder was run on the crop itself, so the
decoder reads the message back at (almost) 1.0 only when it sees that very crop at its own scale;
a smaller or shifted window decodes poorly (scale mismatch). The soft margin of the crop is
therefore a sharp objective in the four edge positions, which a short pattern search maximizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE, METHOD_STEGASTAMP
from hackathon.wmloc.research.decoding import Decoder
from hackathon.wmloc.research.errors import ResearchError

Box = tuple[int, int, int, int]
"""(top, left, bottom, right) in pixels of the 512 image, bottom/right exclusive."""
GRID_CELL_PX = IMAGE_SIZE // GRID_SIZE
MIN_SIDE_PX = 16
PATTERN_STEPS: tuple[int, ...] = (32, 16, 8, 4)
"""Edge displacements tried by the pattern search, coarse to fine (px of the 512 image)."""
MAX_SWEEPS_PER_STEP = 4
EVAL_CHUNK = 64
TIE_TOLERANCE = 0.01
"""Candidates within this fraction of the best margin tie; the larger crop wins."""


class RectangleError(ResearchError):
    """Invalid box or empty score map."""


@dataclass(frozen=True)
class RectFit:
    """Result of the edge search.

    Attributes:
        box: Fitted box, pixels.
        margin: Soft margin of the fitted crop (mean logit x message sign).
        agreement: Hard bit agreement of the fitted crop.
        forwards: Number of crops decoded.
        seed_box: The starting box.
    """

    box: Box
    margin: float
    agreement: float
    forwards: int
    seed_box: Box


def box_area(box: Box) -> int:
    """Area in pixels."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def aspect_ratio(box: Box) -> float:
    """Long side over short side (inf for a degenerate box)."""
    height, width = box[2] - box[0], box[3] - box[1]
    if min(height, width) <= 0:
        return float("inf")
    return max(height, width) / min(height, width)


def clamp_box(box: Box) -> Box:
    """Clip a box to the image."""
    top, left, bottom, right = box
    return (max(0, top), max(0, left), min(IMAGE_SIZE, bottom), min(IMAGE_SIZE, right))


def is_valid(box: Box, max_aspect: float) -> bool:
    """Both sides at least MIN_SIDE_PX, inside the image, aspect ratio within the method's bound."""
    top, left, bottom, right = box
    inside = 0 <= top and 0 <= left and bottom <= IMAGE_SIZE and right <= IMAGE_SIZE
    sides = bottom - top >= MIN_SIDE_PX and right - left >= MIN_SIDE_PX
    return inside and sides and aspect_ratio(box) <= max_aspect


def box_from_grid(score: np.ndarray, threshold: float) -> Box | None:
    """Bounding box (pixels) of the cells of a [128, 128] map at or above `threshold`."""
    cells = np.argwhere(score >= threshold)
    if len(cells) == 0:
        return None
    top, left = cells.min(axis=0)
    bottom, right = cells.max(axis=0) + 1
    return (
        int(top) * GRID_CELL_PX,
        int(left) * GRID_CELL_PX,
        int(bottom) * GRID_CELL_PX,
        int(right) * GRID_CELL_PX,
    )


def box_to_mask(box: Box | None) -> np.ndarray:
    """uint8 [128, 128] mask of a pixel box under the official rule (cell on if >= 8 of 16 px)."""
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    if box is None:
        return mask
    top, left, bottom, right = clamp_box(box)
    rows = _cell_overlap(top, bottom)
    cols = _cell_overlap(left, right)
    half_cell_pixels = GRID_CELL_PX * GRID_CELL_PX // 2
    return np.asarray(rows[:, None] * cols[None, :] >= half_cell_pixels, dtype=np.uint8)


def _cell_overlap(start: int, stop: int) -> np.ndarray:
    """Pixels (0..4) of each of the 128 cells lying inside [start, stop)."""
    low = np.arange(GRID_SIZE) * GRID_CELL_PX
    overlap = np.minimum(low + GRID_CELL_PX, stop) - np.maximum(low, start)
    return np.asarray(np.clip(overlap, 0, GRID_CELL_PX))


def box_iou(first: Box | None, second: Box | None) -> float:
    """Intersection over union of two pixel boxes (0 if either is None)."""
    if first is None or second is None:
        return 0.0
    top, left = max(first[0], second[0]), max(first[1], second[1])
    bottom, right = min(first[2], second[2]), min(first[3], second[3])
    inter = max(0, bottom - top) * max(0, right - left)
    union = box_area(first) + box_area(second) - inter
    return inter / union if union > 0 else 0.0


def crop_scores(
    decoder: Decoder, image: torch.Tensor, message: np.ndarray, boxes: list[Box]
) -> tuple[np.ndarray, np.ndarray]:
    """(soft margin, hard agreement) of each crop, as float64 arrays [len(boxes)]."""
    signs = torch.from_numpy(message.astype(np.float32) * 2.0 - 1.0)
    softs: list[float] = []
    hards: list[float] = []
    for start in range(0, len(boxes), EVAL_CHUNK):
        chunk = boxes[start : start + EVAL_CHUNK]
        if decoder.method == METHOD_STEGASTAMP:
            # The ONNX path resizes (bicubic) inside the kit wrapper: hand it the raw crops.
            logits = torch.cat([decoder.logits(image[None, :, t:b, l:r]) for t, l, b, r in chunk])
        else:
            # Crops of different shapes cannot be stacked: resize each to the decoder resolution
            # (bilinear, antialiased only when shrinking, as the kit does) then batch the network.
            resized = [
                torch.nn.functional.interpolate(
                    image[None, :, t:b, l:r],
                    size=(decoder.resolution, decoder.resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=max(b - t, r - l) > decoder.resolution,
                )[0]
                for t, l, b, r in chunk
            ]
            logits = decoder.logits(torch.stack(resized))
        signed = logits * signs
        softs.extend(signed.mean(dim=1).tolist())
        hards.extend((signed > 0).float().mean(dim=1).tolist())
    return np.array(softs), np.array(hards)


def fit_rectangle(
    decoder: Decoder,
    image: torch.Tensor,
    message: np.ndarray,
    seed_box: Box,
    max_aspect: float,
) -> RectFit:
    """Pattern search of the four edges on the soft margin of the exact crop.

    Each sweep decodes all +-step moves of the four edges in one batch, then applies every edge
    whose best move improves the margin (falling back to the single best move if the joint
    move does not). Steps shrink from coarse to fine.

    Raises:
        RectangleError: the seed box is invalid for this method.
    """
    seed = clamp_box(seed_box)
    if not is_valid(seed, max_aspect):
        raise RectangleError(f"invalid seed box {seed_box} for aspect <= {max_aspect}")
    forwards = 0
    softs, hards = crop_scores(decoder, image, message, [seed])
    forwards += 1
    current, current_soft, current_hard = seed, float(softs[0]), float(hards[0])
    for step in PATTERN_STEPS:
        for _ in range(MAX_SWEEPS_PER_STEP):
            moves: list[tuple[int, int, Box]] = []
            for edge in range(4):
                for sign in (-1, 1):
                    moved = list(current)
                    moved[edge] += sign * step
                    candidate = (moved[0], moved[1], moved[2], moved[3])
                    if is_valid(candidate, max_aspect):
                        moves.append((edge, sign, candidate))
            if not moves:
                break
            soft, hard = crop_scores(decoder, image, message, [m[2] for m in moves])
            forwards += len(moves)
            best_per_edge: dict[int, int] = {}
            for index, (edge, _, _) in enumerate(moves):
                if soft[index] > current_soft * (1.0 + TIE_TOLERANCE) and (
                    edge not in best_per_edge or soft[index] > soft[best_per_edge[edge]]
                ):
                    best_per_edge[edge] = index
            if not best_per_edge:
                break
            joint = list(current)
            for edge, index in best_per_edge.items():
                joint[edge] = moves[index][2][edge]
            joint_box: Box = (joint[0], joint[1], joint[2], joint[3])
            single = max(best_per_edge.values(), key=lambda i: float(soft[i]))
            if len(best_per_edge) > 1 and is_valid(joint_box, max_aspect):
                joint_soft, joint_hard = crop_scores(decoder, image, message, [joint_box])
                forwards += 1
                if joint_soft[0] > soft[single]:
                    current, current_soft, current_hard = (
                        joint_box,
                        float(joint_soft[0]),
                        float(joint_hard[0]),
                    )
                    continue
            current, current_soft, current_hard = (
                moves[single][2],
                float(soft[single]),
                float(hard[single]),
            )
    return RectFit(current, current_soft, current_hard, forwards, seed)

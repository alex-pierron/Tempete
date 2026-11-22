"""Rectangle search by decoding windows against the known message (team C).

A rectangle of the "rect" regime was watermarked on its own crop (resized to the decoder
resolution, embedded, resized back, pasted). Decoding exactly that crop therefore recovers the
message (agreement 1.00, soft margin about 9 for TrustMark) while a window that is 25 % too large
or small, or shifted by 40 % of its size, decodes at chance (measured on the train split: null
margin 0.00 +- 0.04). The search is thus a detection problem in four dimensions:

1. a coarse grid of windows (geometric sizes, bounded aspect, overlapping positions) is decoded;
2. the best non-overlapping windows seed a pattern search that moves edges, shifts and inflates
   the box on the soft margin (mean of logit x message sign), which peaks at the exact crop;
3. the box is accepted only if its bit agreement with the known message reaches a threshold,
   because a decode that succeeds proves the window is watermarked.

Everything here is pure NumPy / PyTorch and talks to the decoder through the `WindowScorer`
protocol, so the algorithm is testable with a synthetic scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE, METHOD_STEGASTAMP
from hackathon.wmloc.research.c_batching import batched_logits
from hackathon.wmloc.research.errors import ResearchError

MIN_SIDE_PX = 24
"""Smallest window side of the coarse grid (the smallest true rectangle of the train split is 12 px)."""
MIN_BOX_SIDE_PX = 8
"""Smallest side a refined box may reach."""
DEFAULT_SIZE_RATIO = 1.55
"""Ratio between consecutive window sizes: the decode survives a +-25 % scale error."""
DEFAULT_STRIDE_FRACTION = 0.55
"""Window stride as a fraction of its size: the decode survives a +-40 % shift."""
ASPECT_SLACK = 1.15
"""Tolerance on the method's aspect limit (the coarse grid and refinement may overshoot a little)."""
DEFAULT_REFINE_STEPS_PX: tuple[int, ...] = (16, 8, 4)
MAX_REFINE_ITERATIONS = 12
DEFAULT_TOP_CANDIDATES = 8
CANDIDATE_OVERLAP_IOU = 0.3
"""Two coarse hits closer than this IoU are the same rectangle (non-maximum suppression)."""
CANDIDATE_MIN_EVIDENCE = 2.5
"""Coarse evidence (t statistic of the signed logits) below which a window is noise. The null is
about N(0, 1); a threshold this low keeps weak hits of small rectangles (evidence 3-5 off the exact
crop) and lets the agreement gate of the refined box reject the noise. The t statistic, unlike
the raw margin, does not depend on the scale of the method's logits (TrustMark 0.04, StegaStamp
about 1 for the margin noise)."""
DETECTION_AGREEMENT = 0.8
MAX_BOX_AREA_FRACTION = 0.5
"""A box covering more of the image than this is not a localized rectangle: it is a whole-image
decode of a full-regime image whose hole pushed the global agreement under the routing threshold
(the largest true rectangle of the train split covers 38 %)."""
"""Bit agreement above which a box is declared watermarked (chance level: 0.50 +- 0.05)."""
DECODE_BATCH = 32
COARSE_CHUNK = 2048
"""Coarse windows scored per call; the decoder splits them by its own (larger) batch size."""
MARGIN_EPSILON = 1e-4
MIN_SPREAD = 1e-6
"""Floor of the per-window logit spread (a constant output must not give an infinite t)."""
CELL_PX = IMAGE_SIZE // GRID_SIZE


class RectSearchError(ResearchError):
    """Invalid search configuration or scorer output."""


@dataclass(frozen=True)
class Box:
    """Half-open pixel box `[top, bottom) x [left, right)` of the 512 image."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        """Rows covered."""
        return self.bottom - self.top

    @property
    def width(self) -> int:
        """Columns covered."""
        return self.right - self.left

    @property
    def area(self) -> int:
        """Pixels covered."""
        return self.height * self.width

    @property
    def aspect(self) -> float:
        """Long side over short side (>= 1)."""
        short = max(1, min(self.height, self.width))
        return max(self.height, self.width) / short

    def is_valid(self, max_aspect: float) -> bool:
        """Inside the image, at least `MIN_BOX_SIDE_PX` wide and tall, aspect within the limit."""
        return (
            self.top >= 0
            and self.left >= 0
            and self.bottom <= IMAGE_SIZE
            and self.right <= IMAGE_SIZE
            and self.height >= MIN_BOX_SIDE_PX
            and self.width >= MIN_BOX_SIDE_PX
            and self.aspect <= max_aspect * ASPECT_SLACK
        )

    def iou(self, other: Box) -> float:
        """Intersection over union with another box."""
        inter_h = min(self.bottom, other.bottom) - max(self.top, other.top)
        inter_w = min(self.right, other.right) - max(self.left, other.left)
        if inter_h <= 0 or inter_w <= 0:
            return 0.0
        inter = inter_h * inter_w
        return inter / (self.area + other.area - inter)

    def to_mask(self) -> np.ndarray:
        """Official 128 x 128 mask (1 = inside) of the box.

        Follows the official downsampling exactly: a 4 x 4 cell is 1 when at least half of its 16
        pixels lie inside the box, i.e. when (covered rows) x (covered columns) >= 8; a corner
        cell covered by 3 rows and 3 columns counts, one covered by 2 rows and 3 columns does not.
        """
        cell_starts = np.arange(GRID_SIZE) * CELL_PX
        rows = np.clip(
            np.minimum(cell_starts + CELL_PX, self.bottom) - np.maximum(cell_starts, self.top),
            0,
            CELL_PX,
        )
        cols = np.clip(
            np.minimum(cell_starts + CELL_PX, self.right) - np.maximum(cell_starts, self.left),
            0,
            CELL_PX,
        )
        covered = rows[:, None] * cols[None, :]
        return np.asarray(covered >= (CELL_PX * CELL_PX) // 2, dtype=np.uint8)


@dataclass(frozen=True)
class WindowScores:
    """Decoder evidence for a batch of windows (one entry per window in each array).

    Attributes:
        margin: Mean over bits of logit x message sign (soft margin).
        agreement: Share of bits decoded as the known message.
        evidence: t statistic of the signed logits over the bits, mean / (std / sqrt(n_bits)).
    """

    margin: np.ndarray
    agreement: np.ndarray
    evidence: np.ndarray


class WindowScorer(Protocol):
    """Decodes windows of one image against its known message.

    Attributes:
        n_decoded: Windows decoded so far (the cost of a search).
    """

    n_decoded: int

    def score(self, boxes: list[Box]) -> WindowScores:
        """Soft margin and bit agreement of each window."""
        ...


@dataclass(frozen=True)
class SearchConfig:
    """Parameters of the coarse grid and the refinement.

    Attributes:
        max_aspect: Aspect limit of the method (2 for TrustMark, 16 for StegaStamp).
        size_ratio: Ratio of consecutive window sizes.
        stride_fraction: Window stride over window size.
        min_side: Smallest coarse window side, px.
        max_side: Largest coarse window side, px.
        refine_steps: Pattern-search step sizes in pixels, coarse to fine.
        top_candidates: Number of coarse hits refined.
    """

    max_aspect: float
    size_ratio: float = DEFAULT_SIZE_RATIO
    stride_fraction: float = DEFAULT_STRIDE_FRACTION
    min_side: int = MIN_SIDE_PX
    max_side: int = IMAGE_SIZE
    refine_steps: tuple[int, ...] = DEFAULT_REFINE_STEPS_PX
    top_candidates: int = DEFAULT_TOP_CANDIDATES


@dataclass(frozen=True)
class RectSearchResult:
    """Outcome of the search on one image.

    Attributes:
        box: Best refined box, or None if no window decoded above the noise floor.
        margin: Soft margin of `box` (0.0 if none).
        evidence: t statistic of `box` (0.0 if none).
        agreement: Bit agreement of `box` (0.0 if none).
        detected: True when the agreement reaches `DETECTION_AGREEMENT` and the box covers at most
            `MAX_BOX_AREA_FRACTION` of the image.
        n_decodes: Windows decoded for this image.
    """

    box: Box | None
    margin: float
    evidence: float
    agreement: float
    detected: bool
    n_decodes: int


def geometric_sizes(min_side: int, max_side: int, ratio: float) -> list[int]:
    """Window sides from `min_side` to `max_side` (included) growing by `ratio`, multiples of 4 px.

    Raises:
        RectSearchError: ratio <= 1 or an empty range.
    """
    if ratio <= 1.0 or min_side <= 0 or max_side < min_side:
        raise RectSearchError(f"bad size range ({min_side}, {max_side}, {ratio})")
    sizes: list[int] = []
    current = float(min_side)
    while current < max_side:
        side = round(current / CELL_PX) * CELL_PX
        if not sizes or side > sizes[-1]:
            sizes.append(side)
        current *= ratio
    if sizes[-1] != max_side:
        sizes.append(max_side)
    return sizes


def window_origins(extent: int, size: int, stride_fraction: float) -> list[int]:
    """Start offsets of windows of `size` along an axis of length `extent`, last one flush."""
    if size >= extent:
        return [0]
    stride = max(CELL_PX, round(size * stride_fraction / CELL_PX) * CELL_PX)
    origins = list(range(0, extent - size + 1, stride))
    if origins[-1] != extent - size:
        origins.append(extent - size)
    return origins


def coarse_boxes(config: SearchConfig) -> list[Box]:
    """Every window of the coarse grid for `config` (deterministic order)."""
    sizes = geometric_sizes(config.min_side, config.max_side, config.size_ratio)
    boxes: list[Box] = []
    for height in sizes:
        for width in sizes:
            if max(height, width) / min(height, width) > config.max_aspect * ASPECT_SLACK:
                continue
            for top in window_origins(IMAGE_SIZE, height, config.stride_fraction):
                for left in window_origins(IMAGE_SIZE, width, config.stride_fraction):
                    boxes.append(Box(top, left, top + height, left + width))
    return boxes


def select_candidates(
    boxes: list[Box],
    evidence: np.ndarray,
    top_k: int,
    min_evidence: float = CANDIDATE_MIN_EVIDENCE,
) -> list[Box]:
    """Highest-evidence windows above `min_evidence`, suppressing near-duplicates (IoU > 0.3)."""
    if len(boxes) != len(evidence):
        raise RectSearchError("boxes and evidence differ in length")
    chosen: list[Box] = []
    for index in np.argsort(-evidence):
        if evidence[index] < min_evidence or len(chosen) >= top_k:
            break
        candidate = boxes[int(index)]
        if all(candidate.iou(kept) <= CANDIDATE_OVERLAP_IOU for kept in chosen):
            chosen.append(candidate)
    return chosen


def neighbours(box: Box, step: int) -> list[Box]:
    """Moves tried by the pattern search: each edge +-step, shifts, inflate and deflate."""
    t, l, b, r = box.top, box.left, box.bottom, box.right
    return [
        Box(t - step, l, b, r),
        Box(t + step, l, b, r),
        Box(t, l, b - step, r),
        Box(t, l, b + step, r),
        Box(t, l - step, b, r),
        Box(t, l + step, b, r),
        Box(t, l, b, r - step),
        Box(t, l, b, r + step),
        Box(t - step, l, b - step, r),
        Box(t + step, l, b + step, r),
        Box(t, l - step, b, r - step),
        Box(t, l + step, b, r + step),
        Box(t - step, l - step, b + step, r + step),
        Box(t + step, l + step, b - step, r - step),
    ]


def refine_box(scorer: WindowScorer, start: Box, config: SearchConfig) -> tuple[Box, WindowScores]:
    """Pattern search maximizing the evidence from `start`; returns the box and its scores."""
    best = start
    best_scores = scorer.score([start])
    for step in config.refine_steps:
        for _ in range(MAX_REFINE_ITERATIONS):
            options = [n for n in neighbours(best, step) if n.is_valid(config.max_aspect)]
            if not options:
                break
            scores = scorer.score(options)
            top = int(np.argmax(scores.evidence))
            if scores.evidence[top] <= best_scores.evidence[0] + MARGIN_EPSILON:
                break
            best = options[top]
            best_scores = WindowScores(
                scores.margin[top : top + 1],
                scores.agreement[top : top + 1],
                scores.evidence[top : top + 1],
            )
    return best, best_scores


def is_detection(box: Box, agreement: float) -> bool:
    """A decode that succeeds (agreement) on a box small enough to be a localized rectangle."""
    return agreement >= DETECTION_AGREEMENT and box.area <= MAX_BOX_AREA_FRACTION * IMAGE_SIZE**2


def search_rectangle(scorer: WindowScorer, config: SearchConfig) -> RectSearchResult:
    """Full search on one image: coarse grid, candidate selection, refinement, decision."""
    boxes = coarse_boxes(config)
    evidence = np.empty(len(boxes), dtype=np.float32)
    for start in range(0, len(boxes), COARSE_CHUNK):
        chunk = boxes[start : start + COARSE_CHUNK]
        evidence[start : start + len(chunk)] = scorer.score(chunk).evidence
    candidates = select_candidates(boxes, evidence, config.top_candidates)
    best_box: Box | None = None
    best = WindowScores(*(np.zeros(1, dtype=np.float32) for _ in range(3)))
    best_evidence = float("-inf")
    for candidate in candidates:
        refined, scores = refine_box(scorer, candidate, config)
        if float(scores.evidence[0]) > best_evidence:
            best_box, best, best_evidence = refined, scores, float(scores.evidence[0])
        if is_detection(refined, float(scores.agreement[0])):
            break  # one rectangle per image: a decode that succeeds ends the search
    decoded = scorer.n_decoded
    if best_box is None:
        return RectSearchResult(None, 0.0, 0.0, 0.0, False, decoded)
    agreement = float(best.agreement[0])
    return RectSearchResult(
        best_box,
        float(best.margin[0]),
        float(best.evidence[0]),
        agreement,
        is_detection(best_box, agreement),
        decoded,
    )


def resize_batch(crops: torch.Tensor, method: str, resolution: int) -> torch.Tensor:
    """Resize [B, 3, h, w] crops of one shape to the decoder resolution the way the kit does.

    Antialiasing only matters when shrinking; PyTorch ignores the flag when enlarging. Works on
    any device that supports antialiased interpolation (CPU, CUDA).
    """
    if method == METHOD_STEGASTAMP:
        return functional.interpolate(
            crops,
            size=(resolution, resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
    return functional.interpolate(
        crops,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def resize_crop(crop: torch.Tensor, method: str, resolution: int) -> torch.Tensor:
    """Resize one [3, h, w] crop to the decoder resolution (see `resize_batch`)."""
    return resize_batch(crop[None], method, resolution)[0]


class DecoderWindowScorer:
    """`WindowScorer` backed by an official decoder (kit venv).

    Args:
        decoder: A `hackathon.wmloc.research.decoding.Decoder` (duck-typed: `logits`, `method`,
            `resolution`).
        image: [3, 512, 512] float image in [0, 1].
        message: [n_bits] float array of 0/1, the known message.
    """

    def __init__(
        self,
        decoder: LogitDecoder,
        image: torch.Tensor,
        message: np.ndarray,
    ) -> None:
        self._decoder = decoder
        device = str(getattr(decoder, "device", "cpu"))
        # Antialiased resizing is only available on CPU and CUDA (not MPS).
        self._device = device if device.startswith("cuda") else "cpu"
        self._image = image.to(self._device)
        self._message = torch.as_tensor(message, dtype=torch.float32)
        self._sign = self._message * 2.0 - 1.0
        self.n_decoded = 0

    def _resized_crops(self, boxes: list[Box]) -> torch.Tensor:
        """Crops of `boxes` at the decoder resolution, resized per group of equal shape."""
        groups: dict[tuple[int, int], list[int]] = {}
        for index, box in enumerate(boxes):
            groups.setdefault((box.height, box.width), []).append(index)
        out = torch.empty(
            (len(boxes), 3, self._decoder.resolution, self._decoder.resolution),
            device=self._device,
        )
        for indices in groups.values():
            stacked = torch.stack(
                [
                    self._image[:, boxes[i].top : boxes[i].bottom, boxes[i].left : boxes[i].right]
                    for i in indices
                ]
            )
            out[indices] = resize_batch(stacked, self._decoder.method, self._decoder.resolution)
        return out

    def score(self, boxes: list[Box]) -> WindowScores:
        """Decode `boxes` in batches; margin is mean(logit x sign), agreement the bit accuracy."""
        crops = self._resized_crops(boxes)
        logits = batched_logits(self._decoder, crops)
        self.n_decoded += len(boxes)
        signed = logits * self._sign
        margin = signed.mean(dim=1)
        spread = signed.std(dim=1).clamp_min(MIN_SPREAD)
        evidence = margin / (spread / math.sqrt(signed.shape[1]))
        agreement = ((logits > 0).float() == self._message).float().mean(dim=1)
        return WindowScores(
            margin.numpy().astype(np.float32),
            agreement.numpy().astype(np.float32),
            evidence.numpy().astype(np.float32),
        )


class LogitDecoder(Protocol):
    """What `DecoderWindowScorer` needs from a decoder."""

    method: str
    resolution: int

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Bit logits [B, n_bits]."""
        ...

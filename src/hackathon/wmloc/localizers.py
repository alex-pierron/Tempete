"""Localizers: turn a watermarked-then-edited image and its KNOWN message into a 128x128 mask.

All three methods of the task decode a whole-image message, but the test set tells us the message
that was embedded. So the question per location is not "is there a watermark?" but "does this
region still carry THIS message?", answered by the agreement between the decoded bits and the
known ones (chance level 0.5, 1.0 for an intact region).

Two score maps are provided, both built on a `WatermarkBackend` (the thing that decodes bits), so
the same code serves Pixel Seal today and TrustMark / StegaStamp through the starter-kit wrappers:

* `PixelMapLocalizer` reads the extractor's per-location bit logits BEFORE its spatial pooling.
  Pixel Seal's head is linear after a mean, so the per-location map averages exactly to the
  global logits (verified). The map is only 8x8 at the native 256 px input (64 px cells in a
  512 px image); shift-and-stitch (the input shifted by sub-cell offsets) densifies it.
* `SlidingWindowLocalizer` decodes crops of several sizes and paints their bit accuracy back.

Only torch is needed here; backends live in `local_backend.py` (wmkit) or in a kit adapter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch
from torch.nn import functional

from hackathon.wmloc.constants import (
    GRID_SIZE,
    IMAGE_SIZE,
    KNOWN_METHODS,
    MASK_CLEAN,
    MASK_WATERMARKED,
    METHOD_PIXELSEAL,
)
from hackathon.wmloc.errors import LocalizerError, UnsupportedMethodError

CHANCE_AGREEMENT = 0.5
"""Expected bit agreement between a random message and unrelated content."""
DEFAULT_THRESHOLD = 0.65
"""Placeholder cut on the agreement score, to be calibrated on the train split."""
DEFAULT_SHIFTS_PER_AXIS = 4
DEFAULT_SMOOTHING_SIGMA_CELLS = 1.5
AGGREGATE_MEAN = "mean"
AGGREGATE_MAX = "max"
AGGREGATIONS: tuple[str, ...] = (AGGREGATE_MEAN, AGGREGATE_MAX)
DEFAULT_WINDOW_SIZES: tuple[int, ...] = (96, 128)
DEFAULT_WINDOW_STRIDE = 32
WINDOW_BATCH_SIZE = 32
GLOBAL_SWITCH_THRESHOLD = 0.8
RECT_SEED_THRESHOLD = 0.62
RECT_COARSE_STEP = 8
RECT_FINE_STEP = 4
RECT_COARSE_RANGE = 64
RECT_FINE_RANGE = 8
RECT_REFINE_ROUNDS = 2
RECT_MIN_SIDE = 32
RECT_MIN_AGREEMENT = 0.6
RECT_MARGIN_TOLERANCE = 0.01
"""Crops within this fraction of the best soft margin count as ties; the largest one wins."""
RECT_INSETS: tuple[int, ...] = (0, 16, 32, 48, 64, 96, 128)
"""Uniform shrinks of the seed box tried first, to start the edge search INSIDE the rectangle."""
"""Below this best agreement no candidate crop decodes: the edge stays where it is."""
OTSU_BINS = 64
OTSU_MIN_RANGE = 0.1
"""Below this spread of the score map the image is treated as uniform (all clean or all marked)."""
GAUSSIAN_RADIUS_SIGMAS = 3.0
IMAGE_BATCH_NDIM = 4


class WatermarkBackend(Protocol):
    """Whatever decodes bits for one watermarking method.

    Attributes:
        n_bits: Message length of the method.
        native_size: Square resolution the decoder works at (Pixel Seal: 256).
        map_stride: Native pixels per cell of `logit_map` (Pixel Seal: 32), None if the decoder
            has no spatial map.
    """

    n_bits: int
    native_size: int
    map_stride: int | None

    def bit_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Decode [B, 3, H, W] images of any size to [B, n_bits] logits (sign = decoded bit)."""
        ...

    def logit_map(self, images: torch.Tensor) -> torch.Tensor:
        """Per-location bit logits [B, n_bits, h, w] for [B, 3, native, native] images.

        Raises:
            UnsupportedMethodError: the decoder has no spatial map (decode-only wrappers).
        """
        ...


class Localizer(Protocol):
    """Common interface of every localizer."""

    name: str

    def predict(self, image: torch.Tensor, method: str, message_bits: torch.Tensor) -> torch.Tensor:
        """Mask uint8 [128, 128] (1 = watermarked) for `image` [3, 512, 512] in [0, 1]."""
        ...


def check_inputs(image: torch.Tensor, message_bits: torch.Tensor, n_bits: int) -> None:
    """Validate the common inputs of every localizer.

    Raises:
        LocalizerError: wrong image shape or message length.
    """
    expected = (3, IMAGE_SIZE, IMAGE_SIZE)
    if tuple(image.shape) != expected:
        raise LocalizerError(f"image must have shape {expected}, got {tuple(image.shape)}")
    if message_bits.ndim != 1 or message_bits.numel() != n_bits:
        raise LocalizerError(f"message must be [{n_bits}] bits, got {tuple(message_bits.shape)}")


def bit_agreement(logits: torch.Tensor, message_bits: torch.Tensor, bit_dim: int) -> torch.Tensor:
    """Fraction of bits whose decoded sign matches the known message, along `bit_dim`."""
    signs = message_bits.to(logits.device) * 2.0 - 1.0
    shape = [1] * logits.ndim
    shape[bit_dim] = -1
    return ((logits * signs.reshape(shape)) > 0).to(torch.float32).mean(dim=bit_dim)


def resize_to_native(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize like Pixel Seal's own detect(): bilinear, antialiased, no corner alignment."""
    if images.shape[-2:] == (size, size):
        return images
    return functional.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )


def gaussian_smooth(score: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """Separable Gaussian blur of a [H, W] map (replicate padding); no-op if sigma <= 0."""
    if sigma_cells <= 0:
        return score
    radius = max(1, math.ceil(GAUSSIAN_RADIUS_SIGMAS * sigma_cells))
    offsets = torch.arange(-radius, radius + 1, dtype=score.dtype, device=score.device)
    kernel = torch.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel = kernel / kernel.sum()
    padded = functional.pad(score[None, None], (radius, radius, radius, radius), mode="replicate")
    blurred = functional.conv2d(padded, kernel.reshape(1, 1, 1, -1))
    blurred = functional.conv2d(blurred, kernel.reshape(1, 1, -1, 1))
    return blurred[0, 0]


def otsu_threshold(score: torch.Tensor) -> float | None:
    """Otsu's two-class threshold of a score map, or None when the map is nearly uniform."""
    low, high = float(score.min().item()), float(score.max().item())
    if high - low < OTSU_MIN_RANGE:
        return None
    hist = torch.histc(score.flatten().to(torch.float32), bins=OTSU_BINS, min=low, max=high)
    probs = hist / hist.sum()
    centers = low + (torch.arange(OTSU_BINS, dtype=torch.float32) + 0.5) * (high - low) / OTSU_BINS
    weight_low = torch.cumsum(probs, dim=0)
    mean_low = torch.cumsum(probs * centers, dim=0)
    mean_total = mean_low[-1]
    weight_high = 1.0 - weight_low
    valid = (weight_low > 0) & (weight_high > 0)
    between = torch.where(
        valid,
        (mean_total * weight_low - mean_low) ** 2 / (weight_low * weight_high).clamp(min=1e-12),
        torch.zeros_like(weight_low),
    )
    best = int(torch.argmax(between).item())
    return low + (best + 1) * (high - low) / OTSU_BINS


def apply_threshold(score: torch.Tensor, threshold: float | None) -> torch.Tensor:
    """Binarize a [128, 128] score map; `threshold=None` picks Otsu per image.

    A near-uniform map (no contrast) cannot be split: it is called marked if its mean is
    above the chance level plus a margin, else clean.
    """
    cut = threshold
    if cut is None:
        cut = otsu_threshold(score)
        if cut is None:
            marked = float(score.mean().item()) > CHANCE_AGREEMENT + OTSU_MIN_RANGE
            fill = MASK_WATERMARKED if marked else MASK_CLEAN
            return torch.full(score.shape, fill, dtype=torch.uint8)
    return (score >= cut).to(torch.uint8).cpu()


class ScoreLocalizer:
    """Base of localizers that compute a [128, 128] agreement map, then threshold it.

    Attributes:
        name: Identifier used in reports.
        threshold: Cut on the score, or None for a per-image Otsu cut.
        supported_methods: Methods this localizer can handle.
    """

    def __init__(
        self,
        name: str,
        backend: WatermarkBackend,
        threshold: float | None,
        supported_methods: Sequence[str] = KNOWN_METHODS,
    ) -> None:
        self.name = name
        self.backend = backend
        self.threshold = threshold
        self.supported_methods = tuple(supported_methods)

    def score_map(
        self, image: torch.Tensor, method: str, message_bits: torch.Tensor
    ) -> torch.Tensor:
        """Agreement map [128, 128] float32 in [0, 1] (chance ~0.5). Implemented by subclasses."""
        raise NotImplementedError

    def _check_method(self, method: str) -> None:
        if method not in self.supported_methods:
            raise UnsupportedMethodError(
                f"{self.name} supports {self.supported_methods}, not {method!r}"
            )

    def predict(self, image: torch.Tensor, method: str, message_bits: torch.Tensor) -> torch.Tensor:
        """Mask uint8 [128, 128] (1 = watermarked)."""
        return apply_threshold(self.score_map(image, method, message_bits), self.threshold)


class ConstantLocalizer:
    """Trivial baseline: the same mask everywhere (all ones or all zeros)."""

    def __init__(self, value: int) -> None:
        if value not in (MASK_CLEAN, MASK_WATERMARKED):
            raise ValueError(f"value must be 0 or 1, got {value}")
        self.value = value
        self.name = f"constant_{value}"

    def predict(self, image: torch.Tensor, method: str, message_bits: torch.Tensor) -> torch.Tensor:
        """The constant mask, whatever the input."""
        return torch.full((GRID_SIZE, GRID_SIZE), self.value, dtype=torch.uint8)


class PixelMapLocalizer(ScoreLocalizer):
    """Per-location agreement from the extractor's spatial logit map, densified by shifting.

    Args:
        backend: Decoder exposing `logit_map`.
        shifts_per_axis: Sub-cell shifts per axis (1 = the raw map, n gives an n-times denser
            map for n^2 forward passes).
        smoothing_sigma_cells: Gaussian blur, in cells of the 128 grid, applied to the score.
        threshold: Cut on the score; None = per-image Otsu.
    """

    def __init__(
        self,
        backend: WatermarkBackend,
        shifts_per_axis: int = DEFAULT_SHIFTS_PER_AXIS,
        smoothing_sigma_cells: float = DEFAULT_SMOOTHING_SIGMA_CELLS,
        threshold: float | None = DEFAULT_THRESHOLD,
        name: str = "pixel_map",
    ) -> None:
        super().__init__(name, backend, threshold, supported_methods=(METHOD_PIXELSEAL,))
        if shifts_per_axis < 1:
            raise ValueError("shifts_per_axis must be >= 1")
        self.shifts_per_axis = shifts_per_axis
        self.smoothing_sigma_cells = smoothing_sigma_cells

    def score_map(
        self, image: torch.Tensor, method: str, message_bits: torch.Tensor
    ) -> torch.Tensor:
        """Agreement map [128, 128]; see the module docstring for the shift-and-stitch maths."""
        self._check_method(method)
        check_inputs(image, message_bits, self.backend.n_bits)
        size = self.backend.native_size
        count = self.shifts_per_axis
        stride = self.backend.map_stride
        if stride is None:
            raise UnsupportedMethodError(f"{self.name}: backend has no spatial logit map")
        if stride % count:
            raise ValueError(f"shifts_per_axis={count} must divide the map stride {stride}")
        map_size = size // stride
        step = stride // count
        native = resize_to_native(image[None], size)
        shifted: list[torch.Tensor] = []
        offsets: list[tuple[int, int]] = []
        for ky in range(count):
            for kx in range(count):
                dy, dx = ky * step, kx * step
                offsets.append((ky, kx))
                if dy == 0 and dx == 0:
                    shifted.append(native[0])
                else:
                    padded = functional.pad(native, (dx, 0, dy, 0), mode="reflect")
                    shifted.append(padded[0, :, :size, :size])
        logits = self.backend.logit_map(torch.stack(shifted))
        agreement = bit_agreement(logits, message_bits, bit_dim=1)  # [count^2, h, w]
        dense = torch.zeros(map_size * count, map_size * count, device=agreement.device)
        for index, (ky, kx) in enumerate(offsets):
            # Shifting the image right by d moves cell i's centre to stride*i - d + (stride-1)/2,
            # so larger shifts come first along each axis of the dense map.
            dense[(count - 1 - ky) :: count, (count - 1 - kx) :: count] = agreement[index]
        centre0 = (stride - 1) / 2 - step * (count - 1)
        score = _sample_dense_at_cells(dense, centre0, step, size)
        return gaussian_smooth(score.cpu(), self.smoothing_sigma_cells)


def _sample_dense_at_cells(
    dense: torch.Tensor, centre0: float, step: float, native_size: int
) -> torch.Tensor:
    """Bilinearly read `dense` (centres at centre0 + step * j, native pixels) at the 128 cells."""
    cell_edge = IMAGE_SIZE / GRID_SIZE
    cell_centres_512 = (torch.arange(GRID_SIZE, dtype=torch.float32) + 0.5) * cell_edge
    native_index = cell_centres_512 * native_size / IMAGE_SIZE - 0.5
    position = (native_index - centre0) / step
    last = dense.shape[0] - 1
    normalized = (2.0 * position / last - 1.0).to(dense.device)
    grid_y, grid_x = torch.meshgrid(normalized, normalized, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)[None]
    sampled = functional.grid_sample(
        dense[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled[0, 0]


class SlidingWindowLocalizer(ScoreLocalizer):
    """Bit accuracy of sliding crops, painted back and averaged over overlapping windows.

    Works with any decode-only backend, which is why it is the template for TrustMark and
    StegaStamp. A window decodes only if it is large enough to hold the signal: the sizes are
    a trade-off between precision (small) and reliability (large).

    Args:
        backend: Decoder exposing `bit_logits`.
        window_sizes: Window sides in pixels of the 512 image (multiples of 4).
        stride: Step between windows in pixels (multiple of 4).
        threshold: Cut on the score; None = per-image Otsu.
        aggregation: How the windows covering a cell are combined. `max` keeps the best window:
            a window decodes the message (agreement near 1) only if it lies almost entirely
            inside the watermarked area, so one such window proves its cells are marked, and
            the windows that straddle an edit boundary cannot dilute it. `mean` averages them.
    """

    def __init__(
        self,
        backend: WatermarkBackend,
        window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES,
        stride: int = DEFAULT_WINDOW_STRIDE,
        threshold: float | None = DEFAULT_THRESHOLD,
        name: str = "sliding_window",
        aggregation: str = AGGREGATE_MAX,
    ) -> None:
        super().__init__(name, backend, threshold)
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        self.aggregation = aggregation
        cell = IMAGE_SIZE // GRID_SIZE
        if stride % cell or any(size % cell for size in window_sizes):
            raise ValueError(f"window sizes and stride must be multiples of {cell} px")
        if any(size > IMAGE_SIZE or size <= 0 for size in window_sizes) or stride <= 0:
            raise ValueError("window sizes must be in (0, 512] and the stride positive")
        self.window_sizes = tuple(window_sizes)
        self.stride = stride

    def _origins(self, size: int) -> list[int]:
        origins = list(range(0, IMAGE_SIZE - size + 1, self.stride))
        if origins[-1] != IMAGE_SIZE - size:
            origins.append(IMAGE_SIZE - size)
        return origins

    def score_map(
        self, image: torch.Tensor, method: str, message_bits: torch.Tensor
    ) -> torch.Tensor:
        """Per-cell mean or best bit agreement of the windows covering it, [128, 128]."""
        self._check_method(method)
        check_inputs(image, message_bits, self.backend.n_bits)
        cell = IMAGE_SIZE // GRID_SIZE
        total = torch.zeros(GRID_SIZE, GRID_SIZE)
        covered = torch.zeros(GRID_SIZE, GRID_SIZE)
        best = torch.zeros(GRID_SIZE, GRID_SIZE)
        for size in self.window_sizes:
            boxes = [(y, x) for y in self._origins(size) for x in self._origins(size)]
            for start in range(0, len(boxes), WINDOW_BATCH_SIZE):
                chunk = boxes[start : start + WINDOW_BATCH_SIZE]
                crops = torch.stack([image[:, y : y + size, x : x + size] for y, x in chunk])
                logits = self.backend.bit_logits(crops)
                accuracy = bit_agreement(logits, message_bits, bit_dim=1).cpu()
                for (y, x), value in zip(chunk, accuracy.tolist(), strict=True):
                    ys, xs = (
                        slice(y // cell, (y + size) // cell),
                        slice(x // cell, (x + size) // cell),
                    )
                    total[ys, xs] += value
                    covered[ys, xs] += 1.0
                    best[ys, xs] = torch.clamp(best[ys, xs], min=value)
        if self.aggregation == AGGREGATE_MAX:
            return best
        return total / covered.clamp(min=1.0)


class RectRefineLocalizer(ScoreLocalizer):
    """Sliding-window score map restricted to a rectangle fitted on the exact crop decode.

    Why: the task watermarks a RECTANGLE (the embedder runs on the crop), and the decode of
    exactly that crop is near perfect while it degrades once the crop takes in unmarked content
    (hard agreement drops fast on the short sides, slowly on the long ones; the soft margin
    always peaks near the true edges). So each edge of a seed box is moved to the position that
    maximizes the soft margin of the crop. The seed box is the bounding box of the cells the
    window localizer calls marked, first shrunk evenly to its best inset.

    The returned score is the window score inside the rectangle and 0 outside, so thresholds can
    still be swept and calibrated like any other score map (inside, the threshold removes the
    holes left by object removal / addition).

    Args:
        inner: Window localizer giving the seed score map and the holes.
        seed_threshold: Cut on the inner score that defines the seed bounding box.
    """

    def __init__(
        self,
        inner: SlidingWindowLocalizer,
        seed_threshold: float = RECT_SEED_THRESHOLD,
        threshold: float | None = DEFAULT_THRESHOLD,
        name: str = "rect_refine",
    ) -> None:
        super().__init__(name, inner.backend, threshold)
        self.inner = inner
        self.seed_threshold = seed_threshold

    def _decode(
        self,
        image: torch.Tensor,
        message_bits: torch.Tensor,
        boxes: list[tuple[int, int, int, int]],
    ) -> tuple[list[float], list[float]]:
        """(bit agreement, soft margin) of the crops `boxes` = (top, left, bottom, right) px.

        The soft margin is the mean of logit x message sign: unlike the hard agreement, which
        sits at 1.0 on a wide plateau, it peaks at the exact rectangle and decays as foreign
        content enters the crop, so it is what the edge search maximizes.
        """
        hard: list[float] = []
        soft: list[float] = []
        signs = message_bits * 2.0 - 1.0
        for start in range(0, len(boxes), WINDOW_BATCH_SIZE):
            chunk = boxes[start : start + WINDOW_BATCH_SIZE]
            native = [
                resize_to_native(image[None, :, t:b, l:r], self.backend.native_size)[0]
                for t, l, b, r in chunk
            ]
            logits = self.backend.bit_logits(torch.stack(native))
            hard.extend(bit_agreement(logits, message_bits, bit_dim=1).tolist())
            soft.extend((logits * signs.to(logits.device)).mean(dim=1).tolist())
        return hard, soft

    def _pick(
        self,
        image: torch.Tensor,
        message_bits: torch.Tensor,
        candidates: list[tuple[int, int, int, int]],
        fallback: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """The largest crop among those with the best soft margin, or `fallback` if none decodes."""
        if not candidates:
            return fallback
        hard, soft = self._decode(image, message_bits, candidates)
        peak = max(soft)
        if peak <= 0:
            return fallback
        ties = [
            i for i in range(len(candidates)) if soft[i] >= peak * (1.0 - RECT_MARGIN_TOLERANCE)
        ]

        def area(index: int) -> int:
            top, left, bottom, right = candidates[index]
            return (bottom - top) * (right - left)

        best = max(ties, key=area)
        return candidates[best] if hard[best] >= RECT_MIN_AGREEMENT else fallback

    def _best_inset(
        self, image: torch.Tensor, message_bits: torch.Tensor, box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Shrink the seed box evenly to the inset whose crop decodes best.

        The seed box overshoots (windows that straddle the edge leak outward), and with
        several edges outside the rectangle one edge move barely changes the decode, so the
        edge search must start from a box close to or inside the rectangle.
        """
        top, left, bottom, right = box
        candidates = [
            (top + d, left + d, bottom - d, right - d)
            for d in RECT_INSETS
            if bottom - top - 2 * d >= RECT_MIN_SIDE and right - left - 2 * d >= RECT_MIN_SIDE
        ]
        return self._pick(image, message_bits, candidates, box)

    def _move_edge(
        self,
        image: torch.Tensor,
        message_bits: torch.Tensor,
        box: tuple[int, int, int, int],
        edge: int,
        step: int,
        reach: int,
    ) -> tuple[int, int, int, int]:
        """Move one edge (0 top, 1 left, 2 bottom, 3 right) to the offset with the best decode."""
        candidates: list[tuple[int, int, int, int]] = []
        for offset in range(-reach, reach + 1, step):
            moved = list(box)
            moved[edge] += offset
            top, left = max(0, moved[0]), max(0, moved[1])
            bottom, right = min(IMAGE_SIZE, moved[2]), min(IMAGE_SIZE, moved[3])
            if bottom - top >= RECT_MIN_SIDE and right - left >= RECT_MIN_SIDE:
                candidates.append((top, left, bottom, right))
        return self._pick(image, message_bits, candidates, box)

    def find_rect(
        self, image: torch.Tensor, message_bits: torch.Tensor, seed_score: torch.Tensor
    ) -> tuple[int, int, int, int] | None:
        """(top, left, bottom, right) in pixels, or None when no cell reaches the seed cut."""
        seeds = (seed_score >= self.seed_threshold).nonzero()
        if seeds.numel() == 0:
            return None
        cell = IMAGE_SIZE // GRID_SIZE
        top, left = (int(v) * cell for v in seeds.amin(dim=0))
        bottom, right = ((int(v) + 1) * cell for v in seeds.amax(dim=0))
        box = self._best_inset(image, message_bits, (top, left, bottom, right))
        for step, reach in (
            (RECT_COARSE_STEP, RECT_COARSE_RANGE),
            (RECT_FINE_STEP, RECT_FINE_RANGE),
        ):
            for _ in range(RECT_REFINE_ROUNDS if step == RECT_COARSE_STEP else 1):
                for edge in range(4):
                    box = self._move_edge(image, message_bits, box, edge, step, reach)
        return box

    @staticmethod
    def restrict_to_box(seed: torch.Tensor, box: tuple[int, int, int, int] | None) -> torch.Tensor:
        """`seed` kept inside the pixel box (top, left, bottom, right), 0 outside; all 0 if None."""
        if box is None:
            return torch.zeros_like(seed)
        cell = IMAGE_SIZE // GRID_SIZE
        top, left, bottom, right = box
        inside = torch.zeros_like(seed)
        inside[top // cell : -(-bottom // cell), left // cell : -(-right // cell)] = 1.0
        return seed * inside

    def score_map(
        self, image: torch.Tensor, method: str, message_bits: torch.Tensor
    ) -> torch.Tensor:
        """Inner window score inside the fitted rectangle, 0 outside, [128, 128]."""
        self._check_method(method)
        seed = self.inner.score_map(image, method, message_bits)
        return self.restrict_to_box(seed, self.find_rect(image, message_bits, seed))


class GlobalSwitchLocalizer(ScoreLocalizer):
    """Regime classifier, not a localizer: all ones if the whole image decodes, else all zeros.

    On the real train split half of the images are watermarked everywhere but a restored hole
    (global agreement ~0.98, mask mostly 1) and half carry a small rectangle (global agreement
    ~0.5, mask mostly 0). This one decode per image reaches ~0.45 mIoU on every method (the
    all-zeros baseline is 0.25) and is the floor any real localizer must beat.

    Args:
        backend: Any decoder (decode-only is enough).
        threshold: Global agreement from which the image counts as fully watermarked.
    """

    def __init__(
        self,
        backend: WatermarkBackend,
        threshold: float | None = GLOBAL_SWITCH_THRESHOLD,
        name: str = "global_switch",
    ) -> None:
        super().__init__(name, backend, threshold)

    def score_map(
        self, image: torch.Tensor, method: str, message_bits: torch.Tensor
    ) -> torch.Tensor:
        """Constant map [128, 128] equal to the global bit agreement of the whole image."""
        self._check_method(method)
        check_inputs(image, message_bits, self.backend.n_bits)
        agreement = bit_agreement(self.backend.bit_logits(image[None]), message_bits, bit_dim=1)
        return torch.full((GRID_SIZE, GRID_SIZE), float(agreement.item()))


class RoutingLocalizer:
    """Send each image to the localizer of its method, with a fallback for the others.

    Args:
        routes: {method: localizer}.
        fallback: Used for methods without a route (e.g. TrustMark before it is wired).
    """

    def __init__(self, routes: dict[str, Localizer], fallback: Localizer) -> None:
        self.routes = dict(routes)
        self.fallback = fallback
        self.name = "routing(" + ",".join(f"{m}={loc.name}" for m, loc in self.routes.items()) + ")"

    def predict(self, image: torch.Tensor, method: str, message_bits: torch.Tensor) -> torch.Tensor:
        """Delegate to the method's localizer, or to the fallback."""
        chosen = self.routes.get(method, self.fallback)
        return chosen.predict(image, method, message_bits)

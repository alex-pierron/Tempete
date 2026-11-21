"""Meta Seal family (Pixel Seal, Video Seal 1.0 / 0.0, Chunky Seal) behind the `Watermarker` protocol.

The pretrained models live in `third_party/videoseal` (read-only reference code, installed in
editable mode). This module adds what the upstream loader does not provide:

* checkpoints cached under `wmkit.config.weights_dir() / "seal"` instead of `./ckpts` (a path
  relative to the current directory), resumable and with a free-disk guard;
* "slim" inference checkpoints: the upstream files bundle optimizer and EMA state (the Chunky
  Seal file weighs 13 GB), so only the `model` entry is kept and read memory-mapped;
* strict state-dict loading (upstream silently ignores missing or unexpected keys);
* `eval()` mode and frozen parameters, a differentiable `extract_logits` for white-box attacks,
  and batching, so that the public surface only manipulates `torch.Tensor`.

Tensor conventions are those of `wmkit.types`: images `[B, 3, H, W]` in `[0, 1]`, messages
`[B, n_bits]` in `{0, 1}`, soft bits `[B, n_bits]` in `[0, 1]`.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch.nn import functional as F

from wmkit.config import weights_dir

# --- Cards -------------------------------------------------------------------------------------

PIXELSEAL_CARD = "pixelseal"
VIDEOSEAL_CARD = "videoseal_1.0"
VIDEOSEAL_LEGACY_CARD = "videoseal_0.0"
CHUNKYSEAL_CARD = "chunkyseal"

SEAL_CARDS: tuple[str, ...] = (
    PIXELSEAL_CARD,
    VIDEOSEAL_CARD,
    VIDEOSEAL_LEGACY_CARD,
    CHUNKYSEAL_CARD,
)
"""Canonical names of the supported model cards (file stems of `videoseal/cards/*.yaml`)."""

CARD_ALIASES: dict[str, str] = {"videoseal": VIDEOSEAL_CARD}
"""Accepted spellings that map to a canonical card name."""

# --- Constants ---------------------------------------------------------------------------------

DETECTION_CHANNELS = 1
"""Channel 0 of the detector output is the detection (mask) logit; bit logits follow."""

DEFAULT_BATCH_SIZE = 8
"""Images processed per forward pass in `embed` / `extract` (bounds peak memory)."""

SEAL_WEIGHTS_SUBDIR = "seal"
SLIM_SUFFIX = ".slim.pth"
ORIGINAL_SUFFIX = ".pth"
PARTIAL_SUFFIX = ".part"
CHECKPOINT_MODEL_KEY = "model"

DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60.0
DOWNLOAD_REPORT_INTERVAL_SECONDS = 10.0
DISK_SAFETY_MARGIN_BYTES = 2 * 1024**3
HTTP_PARTIAL_CONTENT = 206
HTTP_USER_AGENT = "wmkit-fetch/0.1"
HUGGINGFACE_HOST = "https://huggingface.co/"
HUGGINGFACE_RAW_SEGMENT = "/resolve/main/"
HUGGINGFACE_REPO_DEPTH = 2

INTERPOLATION_MODE = "bilinear"
"""Resize used by the upstream `detect` / `embed` (bilinear, antialiased, no corner alignment)."""


# --- Errors ------------------------------------------------------------------------------------


class SealError(Exception):
    """Base class of every error raised by `wmkit.seal`."""


class UnknownSealCardError(SealError, ValueError):
    """The requested card is not one of `SEAL_CARDS` (or an alias)."""


class SealBitsMismatchError(SealError, ValueError):
    """`n_bits` differs from the native payload size of the card."""


class SealInputError(SealError, ValueError):
    """An input tensor has an unsupported shape or content."""


class SealCheckpointError(SealError, RuntimeError):
    """A checkpoint could not be downloaded, converted or loaded exactly."""


# --- Typing of the untyped upstream model ------------------------------------------------------


class _Blender(Protocol):
    scaling_w: float


class _SealModel(Protocol):
    """The subset of `videoseal.models.Videoseal` that this module relies on."""

    blender: _Blender
    detector: torch.nn.Module

    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor | None = ...,
        is_video: bool = ...,
        *,
        lowres_attenuation: bool = ...,
    ) -> dict[str, torch.Tensor]: ...

    def detect(self, imgs: torch.Tensor, is_video: bool = ...) -> dict[str, torch.Tensor]: ...

    def to(self, device: torch.device | str) -> Any: ...

    def eval(self) -> Any: ...

    def requires_grad_(self, requires_grad: bool = ...) -> Any: ...

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = ...
    ) -> Any: ...


# --- Card metadata -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SealCardInfo:
    """Static description of a model card, read from its YAML without loading weights.

    Attributes:
        name: Canonical card name.
        n_bits: Native payload size.
        img_size: Side of the square resolution at which embedder and extractor operate.
        scaling_w: Default watermark strength of the card.
        uses_jnd: True if a JND attenuation heatmap modulates the watermark.
        checkpoint_url: Where the original checkpoint is published.
    """

    name: str
    n_bits: int
    img_size: int
    scaling_w: float
    uses_jnd: bool
    checkpoint_url: str


def resolve_card(card: str) -> str:
    """Return the canonical card name for `card` (accepts the `videoseal` alias).

    Raises:
        UnknownSealCardError: `card` is not supported.
    """
    canonical = CARD_ALIASES.get(card, card)
    if canonical not in SEAL_CARDS:
        known = ", ".join([*SEAL_CARDS, *CARD_ALIASES])
        raise UnknownSealCardError(f"unknown Seal card {card!r}; known: {known}")
    return canonical


def _upstream_root() -> Path:
    """Return the root of the videoseal checkout (parent of the `videoseal` package)."""
    import videoseal

    return Path(cast(str, videoseal.__file__)).resolve().parents[1]


def _load_card_config(card: str) -> Any:
    """Load a card YAML as an OmegaConf object (untyped upstream structure)."""
    from omegaconf import OmegaConf

    card_path = _upstream_root() / "videoseal" / "cards" / f"{resolve_card(card)}.yaml"
    return OmegaConf.load(card_path)


def get_card_info(card: str) -> SealCardInfo:
    """Read the static metadata of `card` from its upstream YAML.

    Raises:
        UnknownSealCardError: `card` is not supported.
    """
    canonical = resolve_card(card)
    config = _load_card_config(canonical)
    args = config.args
    img_size = int(args.img_size_proc if "img_size_proc" in args else args.img_size_extractor)
    attenuation = str(args.attenuation) if "attenuation" in args else "None"
    return SealCardInfo(
        name=canonical,
        n_bits=int(args.nbits),
        img_size=img_size,
        scaling_w=float(args.scaling_w),
        uses_jnd=attenuation.lower().startswith("jnd"),
        checkpoint_url=str(config.checkpoint_path),
    )


# --- Checkpoint cache --------------------------------------------------------------------------


def seal_weights_dir() -> Path:
    """Directory holding Seal checkpoints (`weights_dir() / "seal"`), created on demand."""
    path = weights_dir() / SEAL_WEIGHTS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def original_checkpoint_path(card: str) -> Path:
    """Path of the full upstream checkpoint of `card` in the local cache."""
    return seal_weights_dir() / f"{resolve_card(card)}{ORIGINAL_SUFFIX}"


def slim_checkpoint_path(card: str) -> Path:
    """Path of the inference-only checkpoint (`model` entry alone) of `card`."""
    return seal_weights_dir() / f"{resolve_card(card)}{SLIM_SUFFIX}"


def _direct_download_url(url: str) -> str:
    """Turn a Hugging Face `.../<org>/<repo>/<file>` URL into its direct `resolve/main` form."""
    if not url.startswith(HUGGINGFACE_HOST) or HUGGINGFACE_RAW_SEGMENT in url:
        return url
    parts = url[len(HUGGINGFACE_HOST):].split("/")
    repo = "/".join(parts[:HUGGINGFACE_REPO_DEPTH])
    filename = "/".join(parts[HUGGINGFACE_REPO_DEPTH:])
    return f"{HUGGINGFACE_HOST}{repo}{HUGGINGFACE_RAW_SEGMENT}{filename}"


def _format_gigabytes(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GiB"


def _download(url: str, destination: Path) -> None:
    """Download `url` to `destination` atomically, resuming a previous partial transfer.

    Raises:
        SealCheckpointError: not enough free disk space, or the transfer was truncated.
    """
    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    already = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": HTTP_USER_AGENT}
    if already:
        headers["Range"] = f"bytes={already}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        resumed = response.status == HTTP_PARTIAL_CONTENT
        remaining = int(response.headers.get("Content-Length", "0"))
        total = remaining + (already if resumed else 0)
        needed = remaining + DISK_SAFETY_MARGIN_BYTES
        free = shutil.disk_usage(destination.parent).free
        if free < needed:
            raise SealCheckpointError(
                f"not enough disk space for {url}: need {_format_gigabytes(needed)} "
                f"(download + margin), free {_format_gigabytes(free)}"
            )
        mode = "ab" if resumed else "wb"
        written = already if resumed else 0
        last_report = time.monotonic()
        with partial.open(mode) as sink:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                sink.write(chunk)
                written += len(chunk)
                if time.monotonic() - last_report > DOWNLOAD_REPORT_INTERVAL_SECONDS:
                    last_report = time.monotonic()
                    print(
                        f"  {destination.name}: {_format_gigabytes(written)} / "
                        f"{_format_gigabytes(total)}",
                        file=sys.stderr,
                    )
    if total and partial.stat().st_size != total:
        raise SealCheckpointError(
            f"truncated download of {url}: got {partial.stat().st_size} of {total} bytes; "
            "re-run to resume"
        )
    partial.replace(destination)


def _write_slim_checkpoint(original: Path, slim: Path) -> None:
    """Extract the `model` state dict of `original` into `slim`, reading it memory-mapped.

    Memory mapping keeps resident memory far below the file size (the Chunky Seal file is
    13 GB while its weights are a fraction of it).

    Raises:
        SealCheckpointError: the file has no `model` entry.
    """
    checkpoint = torch.load(original, map_location="cpu", weights_only=True, mmap=True)
    if CHECKPOINT_MODEL_KEY not in checkpoint:
        raise SealCheckpointError(f"{original} has no {CHECKPOINT_MODEL_KEY!r} entry")
    partial = slim.with_name(slim.name + PARTIAL_SUFFIX)
    torch.save({CHECKPOINT_MODEL_KEY: checkpoint[CHECKPOINT_MODEL_KEY]}, partial)
    del checkpoint
    partial.replace(slim)


def ensure_checkpoint(card: str, *, keep_original: bool = False) -> Path:
    """Make sure the inference checkpoint of `card` is cached locally and return its path.

    Idempotent: nothing is downloaded when the slim file exists. Otherwise the upstream
    checkpoint is downloaded (resumable), reduced to its `model` entry, and the original is
    deleted unless `keep_original` is set.

    Args:
        card: Card name or alias.
        keep_original: Keep the full upstream file next to the slim one.

    Returns:
        Path of the slim checkpoint.

    Raises:
        UnknownSealCardError: `card` is not supported.
        SealCheckpointError: download or conversion failed.
    """
    info = get_card_info(card)
    slim = slim_checkpoint_path(info.name)
    original = original_checkpoint_path(info.name)
    if not slim.exists():
        if not original.exists():
            _download(_direct_download_url(info.checkpoint_url), original)
        _write_slim_checkpoint(original, slim)
    if not keep_original and original.exists():
        original.unlink()
    return slim


# --- Model construction ------------------------------------------------------------------------


def _build_model(card: str) -> _SealModel:
    """Build the upstream model of `card`, load its weights exactly and freeze it (eval mode).

    The upstream `setup_model` is used to assemble embedder, extractor and JND from the card
    (so the architecture can never drift from the reference code); it is fed an empty
    checkpoint and the real weights are loaded here with `strict=True` semantics.

    Raises:
        SealCheckpointError: the checkpoint keys do not match the architecture exactly.
    """
    from videoseal.utils.cfg import setup_model

    info = get_card_info(card)
    slim = ensure_checkpoint(info.name)
    config = _load_card_config(info.name)
    # Upstream resolves this path against the cwd, then against the `videoseal` package
    # directory, but the file lives at the repository root: make it absolute.
    if "attenuation_config" in config.args:
        config.args.attenuation_config = str(_upstream_root() / config.args.attenuation_config)

    with tempfile.TemporaryDirectory() as scratch:
        empty = Path(scratch) / "empty.pth"
        torch.save({CHECKPOINT_MODEL_KEY: {}}, empty)
        with contextlib.redirect_stdout(io.StringIO()):
            model = cast(_SealModel, setup_model(config, empty))

    state = torch.load(slim, map_location="cpu", weights_only=True, mmap=True)
    result = model.load_state_dict(state[CHECKPOINT_MODEL_KEY], strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise SealCheckpointError(
            f"checkpoint {slim.name} does not match card {info.name}: "
            f"{len(missing)} missing keys (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected keys (e.g. {unexpected[:3]})"
        )
    model.eval()
    model.requires_grad_(False)
    return model


# --- The watermarker ---------------------------------------------------------------------------


def _check_images(images: torch.Tensor, argument: str = "images") -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise SealInputError(f"{argument} must have shape [B, 3, H, W], got {tuple(images.shape)}")


class SealWatermarker:
    """A pretrained Meta Seal model exposed through the `wmkit.types.Watermarker` protocol.

    Attributes:
        name: Card name (`pixelseal`, `videoseal_1.0`, ...).
        n_bits: Native payload size.
        card: Same as `name` (kept for readability at call sites).
        device: Device holding the model weights.
        lowres_attenuation: Compute the JND heatmap at processing resolution (cheaper on large
            images, nearly identical output) instead of at full resolution.
        batch_size: Images per forward pass.
    """

    def __init__(
        self,
        model: _SealModel,
        info: SealCardInfo,
        device: torch.device,
        *,
        lowres_attenuation: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Wrap an already built and loaded upstream model (see `load_seal` for the usual route)."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._model = model
        self._info = info
        self.name: str = info.name
        self.card: str = info.name
        self.n_bits: int = info.n_bits
        self.device: torch.device = device
        self.lowres_attenuation: bool = lowres_attenuation
        self.batch_size: int = batch_size

    # -- configuration --

    @property
    def img_size(self) -> int:
        """Side of the square resolution at which the networks run."""
        return self._info.img_size

    @property
    def scaling_w(self) -> float:
        """Current watermark strength (additive blending factor)."""
        return float(self._model.blender.scaling_w)

    @scaling_w.setter
    def scaling_w(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"scaling_w must be >= 0, got {value}")
        self._model.blender.scaling_w = float(value)

    @property
    def model(self) -> torch.nn.Module:
        """The underlying upstream `Videoseal` module (frozen, eval mode)."""
        return cast(torch.nn.Module, self._model)

    # -- helpers --

    def _check_messages(self, messages: torch.Tensor, batch: int) -> torch.Tensor:
        """Validate messages and broadcast a single row to `batch`."""
        if messages.ndim == 1:
            messages = messages.unsqueeze(0)
        if messages.ndim != 2 or messages.shape[1] != self.n_bits:
            raise SealInputError(
                f"messages must have shape [B, {self.n_bits}], got {tuple(messages.shape)}"
            )
        if messages.shape[0] == 1 and batch > 1:
            messages = messages.expand(batch, -1)
        if messages.shape[0] != batch:
            raise SealInputError(
                f"got {messages.shape[0]} messages for {batch} images (expected 1 or {batch})"
            )
        return messages.to(torch.float32)

    def _resize_for_detector(self, images: torch.Tensor) -> torch.Tensor:
        """Resize to the processing resolution exactly as the upstream `detect` does."""
        if images.shape[-2:] == (self.img_size, self.img_size):
            return images
        return F.interpolate(
            images,
            size=(self.img_size, self.img_size),
            mode=INTERPOLATION_MODE,
            align_corners=False,
            antialias=True,
        )

    # -- Watermarker protocol --

    def embed(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Hide `messages` [B, n_bits] in `images` [B, 3, H, W]; return [B, 3, H, W] in [0, 1].

        Raises:
            SealInputError: bad image or message shape.
        """
        _check_images(images)
        batch = images.shape[0]
        msgs = self._check_messages(messages, batch)
        outputs: list[torch.Tensor] = []
        for start in range(0, batch, self.batch_size):
            stop = start + self.batch_size
            chunk = images[start:stop].to(self.device, torch.float32)
            result = self._model.embed(
                chunk,
                msgs[start:stop].to(self.device),
                is_video=False,
                lowres_attenuation=self.lowres_attenuation,
            )
            outputs.append(result["imgs_w"].to(images.device))
        return torch.cat(outputs, dim=0) if outputs else images.clone()

    def extract_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw bit logits [B, n_bits]; differentiable w.r.t. `images` (white-box attacks).

        The sign of a logit is the decoded bit. Images of any size are resized to the
        processing resolution (the resize is part of the autograd graph). Runs in one forward
        pass: chunk the batch yourself if memory is tight.
        """
        _check_images(images)
        resized = self._resize_for_detector(images.to(self.device, torch.float32))
        predictions = cast(torch.Tensor, self._model.detector(resized))
        bit_logits = predictions[:, DETECTION_CHANNELS:]
        if bit_logits.ndim > 2:  # pixel-wise extractors: average the per-pixel logits
            bit_logits = bit_logits.mean(dim=tuple(range(2, bit_logits.ndim)))
        return bit_logits.to(images.device)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Read soft bits [B, n_bits] in [0, 1] from possibly attacked images [B, 3, H', W']."""
        _check_images(images)
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, images.shape[0], self.batch_size):
                chunk = images[start : start + self.batch_size]
                outputs.append(torch.sigmoid(self.extract_logits(chunk)))
        if not outputs:
            return torch.empty(0, self.n_bits, dtype=torch.float32, device=images.device)
        return torch.cat(outputs, dim=0)

    # -- video --

    def embed_video(self, frames: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Watermark a clip `frames` [T, 3, H, W] with one `message` [n_bits] (temporal propagation).

        The upstream model embeds every `step_size`-th frame and repeats the signal on the
        following frames, in chunks, so it is much cheaper than per-frame embedding.
        """
        _check_images(frames, "frames")
        msg = self._check_messages(message, 1)
        result = self._model.embed(
            frames.to(self.device, torch.float32),
            msg.to(self.device),
            is_video=True,
            lowres_attenuation=self.lowres_attenuation,
        )
        return result["imgs_w"].to(frames.device)

    def extract_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Read one soft message [n_bits] from a clip: per-frame logits averaged, then sigmoid."""
        _check_images(frames, "frames")
        with torch.no_grad():
            predictions = self._model.detect(frames.to(self.device, torch.float32), is_video=True)
        bit_logits = predictions["preds"][:, DETECTION_CHANNELS:]
        if bit_logits.ndim > 2:
            bit_logits = bit_logits.mean(dim=tuple(range(2, bit_logits.ndim)))
        return torch.sigmoid(bit_logits.mean(dim=0)).to(frames.device)


def load_seal(
    n_bits: int | None = None,
    device: str = "cpu",
    card: str = PIXELSEAL_CARD,
    scaling_w: float | None = None,
    lowres_attenuation: bool = True,
) -> SealWatermarker:
    """Load a pretrained Meta Seal model as a `Watermarker`.

    The first call per card downloads its checkpoint (see `ensure_checkpoint`).

    Args:
        n_bits: Must be None or equal to the native payload size of the card.
        device: Torch device string (`cpu`, `mps`, `cuda`).
        card: `pixelseal`, `videoseal` (= `videoseal_1.0`), `videoseal_0.0` or `chunkyseal`.
        scaling_w: Overrides the watermark strength (robustness / imperceptibility trade-off).
        lowres_attenuation: See `SealWatermarker`.

    Returns:
        A ready-to-use watermarker, model in eval mode with frozen parameters.

    Raises:
        UnknownSealCardError: unsupported card.
        SealBitsMismatchError: `n_bits` differs from the native size.
        SealCheckpointError: checkpoint unavailable or inconsistent.
    """
    info = get_card_info(card)
    if n_bits is not None and n_bits != info.n_bits:
        raise SealBitsMismatchError(
            f"card {info.name!r} carries exactly {info.n_bits} bits, got n_bits={n_bits}"
        )
    target = torch.device(device)
    model = _build_model(info.name)
    model.to(target)
    watermarker = SealWatermarker(model, info, target, lowres_attenuation=lowres_attenuation)
    if scaling_w is not None:
        watermarker.scaling_w = scaling_w
    return watermarker

"""Frozen-extractor feature cache for the learned segmenter (team B2).

One compressed-free `.npz` per (split, method, image) holds float16 feature maps of the OFFICIAL
extractors, taken at several network depths and at two input scales, plus message-aware evidence
maps (agreement of each position with the KNOWN message) and the whole-image regime scalars.
Features are cheap to compute once and reused by every training run, so the head can be trained
for many epochs and seeds without touching the heavy extractor again.

Per method (grid sizes are in cells, one cell = 512 / grid px of the original image):

* PixelSeal (ConvNeXt-tiny, native 256 px):
  `n_st1` [192, 32, 32], `n_st2` [384, 16, 16], `n_st3` [768, 8, 8] at the native scale;
  `u_st2` [384, 16, 16], `u_st3` [768, 16, 16] with the image fed at 512 px (finer cells);
  evidence maps `n_ev` [4, 8, 8] and `u_ev` [4, 16, 16] (margin, agreement, bit spread, detection).
* TrustMark (ResNet-50, native 245 px): `n_l2` [512, 16, 16], `n_l3` [1024, 16, 16],
  `n_l4` [2048, 8, 8]; evidence `n_ev` [3, 8, 8] and `u_ev` [3, 16, 16] (490 px input).
* StegaStamp: no spatial head, only the global scalars are stored (see `SCALAR_NAMES`).

Usage (kit venv, see `run_b2.sh`):
    python -m hackathon.wmloc.research.b2_features --split train --method pixelseal
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional

from hackathon.wmloc.constants import METHOD_PIXELSEAL, METHOD_STEGASTAMP, METHOD_TRUSTMARK
from hackathon.wmloc.research.data import SPLITS, Record, data_root, load_image, load_records
from hackathon.wmloc.research.errors import ResearchError

FEATURE_CACHE_ENV = "WMLOC_B2_CACHE"
FEATURE_CACHE_SUBDIR = "team_b2/feat"
NATIVE_SIDE_PIXELSEAL = 256
UPSCALED_SIDE_PIXELSEAL = 512
NATIVE_SIDE_TRUSTMARK = 245
UPSCALED_SIDE_TRUSTMARK = 490
TRUSTMARK_INPUT_OFFSET = 1.0
"""TrustMark feeds the network `x * 2 - 1`."""
POOLED_GRID = 16
"""Side of the grid that mid-depth maps are pooled to before caching (disk budget)."""
EVIDENCE_PIXELSEAL = 4
EVIDENCE_TRUSTMARK = 3
SCALAR_NAMES: tuple[str, ...] = ("margin", "agreement", "spread")
MARGIN_CLIP = 8.0
"""Margins are clipped so that a few huge logits cannot dominate the scale."""
BATCH_SIZE = 4
CUDA_BATCH_SIZE = 32
"""An A100 (40 GB) is far from full at 4 images per forward pass: bigger batches raise the throughput."""
BATCH_ENV = "WMLOC_B2_BATCH"


def extraction_batch_size(
    cuda_default: int = CUDA_BATCH_SIZE, other_default: int = BATCH_SIZE
) -> int:
    """Images per extractor forward pass: `$WMLOC_B2_BATCH`, else by device (cuda, others)."""
    configured = os.environ.get(BATCH_ENV)
    if configured:
        return max(1, int(configured))
    return cuda_default if torch.cuda.is_available() else other_default


THREAD_LIMIT = 2
PROGRESS_EVERY = 10


def accelerator() -> str:
    """cuda (JURECA A100) when present, else mps (the Mac), else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class FeatureError(ResearchError):
    """A feature cache is missing or a method is unsupported."""


def cache_root() -> Path:
    """Root of the feature cache (`$WMLOC_B2_CACHE` or `<data>/wmloc/team_b2/feat`)."""
    configured = os.environ.get(FEATURE_CACHE_ENV)
    return Path(configured).expanduser() if configured else data_root() / FEATURE_CACHE_SUBDIR


CHUNK_IMAGES = 128
"""Images per container file: the shared scratch quota counts INODES, so never one file per image."""
NAMES_KEY = "__names__"
_TABLES: dict[tuple[str, str], dict[str, dict[str, NDArray[np.float16]]]] = {}


def feature_dir(split: str, method: str) -> Path:
    """Folder of the feature containers of one (split, method)."""
    return cache_root() / split / method


def load_feature_table(
    split: str, method: str, refresh: bool = False
) -> dict[str, dict[str, NDArray[np.float16]]]:
    """Every cached image of (split, method): `{name: {array name: array}}`.

    Reads the `chunk_*.npz` containers (one file per `CHUNK_IMAGES` images) and, for caches
    written by earlier versions, the legacy one-file-per-image `<name>.npz`. Cached in memory.
    """
    key = (split, method)
    if key in _TABLES and not refresh:
        return _TABLES[key]
    table: dict[str, dict[str, NDArray[np.float16]]] = {}
    directory = feature_dir(split, method)
    if directory.is_dir():
        for path in sorted(directory.glob("*.npz")):
            if ".tmp." in path.name:
                continue
            with np.load(path) as archive:
                if NAMES_KEY in archive.files:
                    names = [str(n) for n in archive[NAMES_KEY]]
                    stacked = {k: archive[k] for k in archive.files if k != NAMES_KEY}
                    for index, name in enumerate(names):
                        table[name] = {k: v[index] for k, v in stacked.items()}
                else:
                    table[path.stem] = {k: archive[k] for k in archive.files}
    _TABLES[key] = table
    return table


def write_feature_chunk(split: str, method: str, rows: list[tuple[str, dict[str, Any]]]) -> Path:
    """Write `rows` (image name, arrays) as ONE container file, atomically; returns its path."""
    names = np.array([name for name, _ in rows])
    stacked = {k: np.stack([row[k] for _, row in rows]) for k in rows[0][1]}
    path = feature_dir(split, method) / f"chunk_{int(time.time() * 1000)}_{os.getpid()}.npz"
    _atomic_save(path, {NAMES_KEY: names, **stacked})
    _TABLES.pop((split, method), None)
    return path


def load_features(split: str, method: str, name: str) -> dict[str, NDArray[np.float16]]:
    """All cached arrays of one image, keyed by name.

    Raises:
        FeatureError: the image is in no container of that (split, method).
    """
    table = load_feature_table(split, method)
    if name not in table:
        raise FeatureError(
            f"missing feature cache for {split}/{method}/{name} in {feature_dir(split, method)}"
        )
    return table[name]


def _atomic_save(path: Path, arrays: dict[str, Any]) -> None:
    """Write `arrays` to `path` atomically (temp file then rename), so a crash leaves no torso."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def message_signs(message: NDArray[np.float32]) -> torch.Tensor:
    """+1 for bit 1 and -1 for bit 0, float32 [n_bits]."""
    return torch.from_numpy(message.astype(np.float32) * 2.0 - 1.0)


def evidence_maps(per_position: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Message-aware evidence of per-position bit logits.

    Args:
        per_position: [n_bits, h, w] bit logits of each position.
        signs: [n_bits] message signs.

    Returns:
        [3, h, w]: mean signed margin (clipped), fraction of bits in agreement, bit-margin spread.
    """
    signed = per_position * signs[:, None, None]
    margin = signed.mean(dim=0).clamp(-MARGIN_CLIP, MARGIN_CLIP)
    agreement = (signed > 0).float().mean(dim=0)
    spread = signed.std(dim=0)
    return torch.stack([margin, agreement, spread])


def pool_to(maps: torch.Tensor, side: int) -> torch.Tensor:
    """Adaptive average pooling of [C, h, w] to [C, side, side] (identity when already there)."""
    if maps.shape[-1] == side and maps.shape[-2] == side:
        return maps
    if maps.shape[-1] % side or maps.shape[-2] % side:
        # MPS cannot pool non-divisible sizes (TrustMark's 31 x 31 layer2): do it on the CPU.
        return functional.adaptive_avg_pool2d(maps.cpu()[None], side)[0].to(maps.device)
    return functional.adaptive_avg_pool2d(maps[None], side)[0]


def _fp16(tensor: torch.Tensor) -> NDArray[np.float16]:
    return tensor.detach().float().cpu().numpy().astype(np.float16)


class PixelSealProbe:
    """Taps the ConvNeXt stages and the per-position head of the official Pixel Seal wrapper."""

    def __init__(self, wrapper: Any, device: str) -> None:
        self.det = wrapper.det
        self.device = device

    def _resize_native(self, images: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(
            images.cpu(),
            size=(NATIVE_SIDE_PIXELSEAL,) * 2,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(self.device)

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """ConvNeXt stage outputs (4 maps, strides 4..32) and the 257-channel per-position logits."""
        convnext = self.det.convnext
        hidden = self.det.preprocess(batch)
        stages = []
        for index in range(len(convnext.stages)):
            hidden = convnext.stages[index](convnext.downsample_layers[index](hidden))
            stages.append(hidden)
        embedding = self.det.pixel_decoder.output_upscaling(hidden)
        head = self.det.pixel_decoder.linear
        per_position = torch.einsum("bchw,oc->bohw", embedding, head.weight)
        per_position = per_position + head.bias[None, :, None, None]
        return stages, per_position

    def extract(self, images: torch.Tensor, messages: NDArray[np.float32]) -> list[dict[str, Any]]:
        """Feature dictionaries (float16) of a batch of [B, 3, 512, 512] images in [0, 1]."""
        native_stages, native_pos = self.forward(self._resize_native(images))
        upscaled_stages, upscaled_pos = self.forward(
            functional.interpolate(
                images.to(self.device),
                size=(UPSCALED_SIDE_PIXELSEAL,) * 2,
                mode="bilinear",
                align_corners=False,
            )
        )
        outputs: list[dict[str, Any]] = []
        for index in range(images.shape[0]):
            signs = message_signs(messages[index]).to(native_pos.device)
            row: dict[str, Any] = {}
            for tag, stages, pos in (
                ("n", native_stages, native_pos),
                ("u", upscaled_stages, upscaled_pos),
            ):
                evidence = evidence_maps(pos[index, 1:], signs)
                detection = pos[index, 0:1]
                row[f"{tag}_ev"] = _fp16(torch.cat([evidence, detection]))
                if tag == "n":
                    row["n_st1"] = _fp16(pool_to(stages[1][index], 2 * POOLED_GRID))
                    row["n_st2"] = _fp16(pool_to(stages[2][index], POOLED_GRID))
                    row["n_st3"] = _fp16(stages[3][index])
                else:
                    row["u_st2"] = _fp16(pool_to(stages[2][index], POOLED_GRID))
                    row["u_st3"] = _fp16(pool_to(stages[3][index], POOLED_GRID))
            native_logits = native_pos[index, 1:].mean(dim=(1, 2))
            signed = native_logits * signs
            row["scalars"] = np.array(
                [
                    float(signed.mean().clamp(-MARGIN_CLIP, MARGIN_CLIP)),
                    float((signed > 0).float().mean()),
                    float(signed.std()),
                ],
                dtype=np.float32,
            )
            outputs.append(row)
        return outputs


class TrustMarkProbe:
    """Taps the ResNet-50 stages and the per-position head of the official TrustMark wrapper."""

    def __init__(self, wrapper: Any, device: str) -> None:
        self.net = wrapper.net
        self.device = device

    def _resize(self, images: torch.Tensor, side: int) -> torch.Tensor:
        shrinking = side < images.shape[-1]
        resized = functional.interpolate(
            images.cpu() if shrinking else images.to(self.device),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
            antialias=shrinking,
        )
        return resized.to(self.device)

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """ResNet stages (layer1..4) and per-position logits [B, 100, h, w]."""
        net = self.net
        hidden = net.maxpool(net.relu(net.bn1(net.conv1(batch * 2 - TRUSTMARK_INPUT_OFFSET))))
        stages = []
        for layer in ("layer1", "layer2", "layer3", "layer4"):
            hidden = getattr(net, layer)(hidden)
            stages.append(hidden)
        per_position = torch.einsum("bchw,oc->bohw", hidden, net.fc.weight)
        per_position = per_position + net.fc.bias[None, :, None, None]
        return stages, per_position

    def extract(self, images: torch.Tensor, messages: NDArray[np.float32]) -> list[dict[str, Any]]:
        """Feature dictionaries (float16) of a batch of [B, 3, 512, 512] images in [0, 1]."""
        native_stages, native_pos = self.forward(self._resize(images, NATIVE_SIDE_TRUSTMARK))
        _, upscaled_pos = self.forward(self._resize(images, UPSCALED_SIDE_TRUSTMARK))
        outputs: list[dict[str, Any]] = []
        for index in range(images.shape[0]):
            signs = message_signs(messages[index]).to(native_pos.device)
            row: dict[str, Any] = {
                "n_ev": _fp16(evidence_maps(native_pos[index], signs)),
                "u_ev": _fp16(evidence_maps(upscaled_pos[index], signs)),
                "n_l2": _fp16(pool_to(native_stages[1][index], POOLED_GRID)),
                "n_l3": _fp16(pool_to(native_stages[2][index], POOLED_GRID)),
                "n_l4": _fp16(native_stages[3][index]),
            }
            signed = native_pos[index].mean(dim=(1, 2)) * signs
            row["scalars"] = np.array(
                [
                    float(signed.mean().clamp(-MARGIN_CLIP, MARGIN_CLIP)),
                    float((signed > 0).float().mean()),
                    float(signed.std()),
                ],
                dtype=np.float32,
            )
            outputs.append(row)
        return outputs


def build_probe(method: str, wrapper: Any, device: str) -> PixelSealProbe | TrustMarkProbe:
    if method == METHOD_PIXELSEAL:
        return PixelSealProbe(wrapper, device)
    if method == METHOD_TRUSTMARK:
        return TrustMarkProbe(wrapper, device)
    raise FeatureError(f"no spatial probe for {method!r}")


def extract_split(
    split: str, method: str, names: set[str] | None = None, limit: int | None = None
) -> int:
    """Fill the feature cache for the images of `method` in `split`; returns files written.

    Idempotent: images whose cache file exists are skipped, writes are atomic.
    """
    from hackathon.wmloc.research.kit import load_wrapper  # kit venv only

    torch.set_num_threads(THREAD_LIMIT)
    records = [r for r in load_records(split) if r.method == method]
    if names is not None:
        records = [r for r in records if r.name in names]
    if limit is not None:
        records = records[:limit]
    done = load_feature_table(split, method, refresh=True)
    pending: list[Record] = [r for r in records if r.name not in done]
    print(f"{split}/{method}: {len(pending)} to do out of {len(records)}", flush=True)
    if not pending:
        return 0
    probe: Any
    if method == METHOD_STEGASTAMP:
        # ONNX Runtime on the CPU: the decoder has no torch module to tap, see b2_stegastamp.
        from hackathon.wmloc.research.b2_stegastamp import StegaStampProbe
        from hackathon.wmloc.research.kit import wmloc_dir

        probe = StegaStampProbe(wmloc_dir() / "models" / "stegastamp" / "stega_stamp_logits.onnx")
    else:
        device = accelerator()
        probe = build_probe(method, load_wrapper(method, device), device)
    started = time.time()
    written = 0
    buffer: list[tuple[str, dict[str, Any]]] = []
    batch_size = extraction_batch_size()
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        images = torch.stack([load_image(split, r.name) for r in chunk])
        messages = np.stack([r.message for r in chunk])
        for record, row in zip(chunk, probe.extract(images, messages), strict=True):
            buffer.append((record.name, row))
            written += 1
        if len(buffer) >= CHUNK_IMAGES:
            write_feature_chunk(split, method, buffer)
            buffer = []
        if (start // batch_size + 1) % max(1, PROGRESS_EVERY * BATCH_SIZE // batch_size) == 0:
            rate = written / (time.time() - started)
            print(f"  {written}/{len(pending)} ({rate:.2f} img/s)", flush=True)
    if buffer:
        write_feature_chunk(split, method, buffer)
    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cache official-extractor features (team B2).")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument(
        "--method", choices=(METHOD_PIXELSEAL, METHOD_TRUSTMARK, METHOD_STEGASTAMP), required=True
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    count = extract_split(args.split, args.method, limit=args.limit)
    print(f"done, {count} images cached", flush=True)


if __name__ == "__main__":
    main()

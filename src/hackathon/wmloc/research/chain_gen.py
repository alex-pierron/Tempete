"""Run the w-loc-small synthetic generator with a CANDIDATE embedding chain (env ``XLP_CHAIN``).

The generator (``small_gen``) calls ``kit.load_wrapper(method, device).embed(cover, bits)``. This
launcher wraps that ``embed`` so every synthetic watermark follows the chain chosen by
``chain_fingerprint`` on the real full-regime images, then hands over to ``small_gen.main``.

Chains: ``embed512`` (unchanged), ``strength<s>`` (residual scaled by s), ``via<side>`` (cover
resized to side x side with bicubic, embedded there, resized back), ``blur<r>`` (Gaussian blur of
radius r after embedding), ``twice`` (embedded twice).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

CHAIN_ENV = "XLP_CHAIN"
DEFAULT_CHAIN = "embed512"
IMAGE_SIDE = 512


class ChainError(ValueError):
    """Unknown chain name in ``XLP_CHAIN``."""


class ChainWrapper:
    """Official wrapper whose ``embed`` follows a candidate chain; everything else is delegated."""

    def __init__(self, inner: Any, chain: str) -> None:
        self._inner = inner
        self._chain = chain
        if not (chain in (DEFAULT_CHAIN, "twice") or chain.startswith(("strength", "via", "blur"))):
            raise ChainError(f"unknown chain {chain!r}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def embed(self, cover: np.ndarray, bits: list[int]) -> np.ndarray:
        """Watermark ``cover`` (uint8 H x W x 3) with the candidate chain."""
        chain = self._chain
        if chain == DEFAULT_CHAIN:
            return np.asarray(self._inner.embed(cover, bits), dtype=np.uint8)
        if chain == "twice":
            once = np.asarray(self._inner.embed(cover, bits), dtype=np.uint8)
            return np.asarray(self._inner.embed(once, bits), dtype=np.uint8)
        if chain.startswith("strength"):
            scale = float(chain.removeprefix("strength"))
            marked = np.asarray(self._inner.embed(cover, bits), dtype=np.float32)
            base = cover.astype(np.float32)
            return np.clip(np.round(base + scale * (marked - base)), 0, 255).astype(np.uint8)
        if chain.startswith("via"):
            # via<side> is defined for a 512 image; crops are rescaled by the same factor.
            factor = int(chain.removeprefix("via")) / IMAGE_SIDE
            height, width = cover.shape[:2]
            size = (max(1, round(width * factor)), max(1, round(height * factor)))
            big = np.asarray(Image.fromarray(cover).resize(size, Image.Resampling.BICUBIC))
            marked = np.asarray(self._inner.embed(big, bits), dtype=np.uint8)
            return np.asarray(Image.fromarray(marked).resize((width, height), Image.Resampling.BICUBIC))
        radius = float(chain.removeprefix("blur"))
        marked = np.asarray(self._inner.embed(cover, bits), dtype=np.uint8)
        return np.asarray(Image.fromarray(marked).filter(ImageFilter.GaussianBlur(radius)))


DR_STRENGTH = (0.6, 1.0)  # x-review-loc (24 content-matched covers): TM strength = kit, SS ~0.85
DR_BLUR_SIGMA = (0.0, 0.5)  # Gaussian blur of the residual, pixels
DR_RESAMPLE_FACTORS = (None, 768 / 512, 1024 / 512)  # embed at a larger size, residual resized back
DR_NOISE_SIGMA = (0.0, 1.5)  # grey levels, added to the COVER so clean and marked share it


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of an H x W x C float image (torch, reflect padding; no scipy)."""
    import torch
    import torch.nn.functional as F

    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[:, None]  # C x 1 x H x W
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))
    return x[:, 0].numpy().transpose(1, 2, 0)


class DomainRandomWrapper(ChainWrapper):
    """Domain randomisation of the chain (Tobin et al. 2017): every call draws its own chain.

    The residual (marked - cover) is computed directly or through a x1.5 / x2 resampling, scaled
    by a strength in DR_STRENGTH and blurred by a sigma in DR_BLUR_SIGMA. The pattern magnification
    is left to ``small_gen --pair-zoom`` (crop of a larger watermarked frame), the noise to the base
    pool (``noisy_draw``), so that the clean and the marked images of a pair never differ by
    anything but the watermark (no shortcut for the network).
    """

    def __init__(self, inner: Any, seed: int) -> None:
        super().__init__(inner, DEFAULT_CHAIN)
        self._rng = np.random.default_rng(seed)

    def embed(self, cover: np.ndarray, bits: list[int]) -> np.ndarray:
        """Watermark ``cover`` with a freshly drawn chain."""
        rng = self._rng
        height, width = cover.shape[:2]
        base = cover.astype(np.float32)
        factor = DR_RESAMPLE_FACTORS[int(rng.integers(len(DR_RESAMPLE_FACTORS)))]
        residual: np.ndarray | None = None
        if factor is not None:
            size = (round(width * factor), round(height * factor))
            big = np.asarray(Image.fromarray(cover).resize(size, Image.Resampling.BICUBIC))
            try:
                marked_big = np.asarray(self._inner.embed(big, bits), dtype=np.float32)
                diff = marked_big - big.astype(np.float32)
                residual = np.stack([
                    np.asarray(Image.fromarray(diff[..., c], mode="F").resize((width, height), Image.Resampling.BILINEAR))
                    for c in range(3)
                ], axis=2)
            except (RuntimeError, ValueError):
                residual = None  # the embedder refused the size: plain chain below
        if residual is None:
            residual = np.asarray(self._inner.embed(cover, bits), dtype=np.float32) - base
        residual *= rng.uniform(*DR_STRENGTH)
        sigma = rng.uniform(*DR_BLUR_SIGMA)
        if sigma > 0:
            residual = gaussian_blur(residual, sigma)
        return np.clip(np.round(base + residual), 0, 255).astype(np.uint8)


class SmoothHoleState:
    """Last full-size original drawn by the base pool (the pair's watermark is embedded on it)."""

    original: np.ndarray | None = None


def main() -> None:
    """Patch the kit loader (and, for ``smoothhole<r>``, the base pool), then run the generator CLI.

    ``smoothhole<r>`` mimics the real full regime, whose holes are smoother generated content
    (local high-pass energy AUC 0.33-0.39 against the edited class on the train): each pair stores
    a Gaussian-blurred cover (radius r) as its CLEAN image, while the watermark is embedded on the
    ORIGINAL cover. Rect composites (crops) are unchanged.
    """
    from hackathon.wmloc.research import kit, small_gen

    chain = os.environ.get(CHAIN_ENV, DEFAULT_CHAIN)
    original_loader = kit.load_wrapper
    if chain == "dr":
        draw_plain = small_gen.BasePool.draw
        noise_rng = np.random.default_rng(os.getpid())

        def noisy_draw(self: Any, rng: np.random.Generator) -> np.ndarray:
            cover = draw_plain(self, rng).astype(np.float32)
            sigma = noise_rng.uniform(*DR_NOISE_SIGMA)
            noisy = cover + noise_rng.normal(0.0, sigma, cover.shape).astype(np.float32)
            return np.clip(np.round(noisy), 0, 255).astype(np.uint8)

        small_gen.BasePool.draw = noisy_draw  # type: ignore[method-assign]

        def patched(method: str, device: str) -> ChainWrapper:
            return DomainRandomWrapper(original_loader(method, device), os.getpid())

        kit.load_wrapper = patched  # type: ignore[assignment]
        print("chain_gen: chain=dr (domain randomisation)", flush=True)
        small_gen.main()
        return
    if chain.startswith("smoothhole"):
        radius = float(chain.removeprefix("smoothhole"))
        draw = small_gen.BasePool.draw

        def smooth_draw(self: Any, rng: np.random.Generator) -> np.ndarray:
            cover = draw(self, rng)
            SmoothHoleState.original = cover
            blurred = Image.fromarray(cover).filter(ImageFilter.GaussianBlur(radius))
            return np.asarray(blurred, dtype=np.uint8)

        small_gen.BasePool.draw = smooth_draw  # type: ignore[method-assign]

        class SmoothHoleWrapper(ChainWrapper):
            def embed(self, cover: np.ndarray, bits: list[int]) -> np.ndarray:
                source = SmoothHoleState.original
                if source is not None and source.shape == cover.shape:
                    return np.asarray(self._inner.embed(source, bits), dtype=np.uint8)
                return np.asarray(self._inner.embed(cover, bits), dtype=np.uint8)

        def patched(method: str, device: str) -> ChainWrapper:
            return SmoothHoleWrapper(original_loader(method, device), DEFAULT_CHAIN)
    else:

        def patched(method: str, device: str) -> ChainWrapper:
            return ChainWrapper(original_loader(method, device), chain)

    kit.load_wrapper = patched  # type: ignore[assignment]
    print(f"chain_gen: chain={chain}", flush=True)
    small_gen.main()


if __name__ == "__main__":
    main()

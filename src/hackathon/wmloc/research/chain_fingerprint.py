"""Identify the REAL embedding chain of the full regime by decoder fingerprints (no clean original needed).

For a watermarked image, the ratios margin(T(x)) / margin(x) under fixed probes T (box blur 3 and 5,
down-up resampling at 128, Gaussian noise, subtraction of the re-embedding residual) depend on the
spectrum and strength of the residual much more than on the content. They are measured on real
full-regime train images (watermarked area >= 90 %) and on clean covers of the same distribution
(rect-regime train images of the OTHER methods: mostly unwatermarked for this method) watermarked by
each candidate chain. The chain whose fingerprint means are closest to the real ones (z-distance) is
the best synthetic recipe. Image-level luma/chroma high-pass log-ratio is reported too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter

SEED = 0
FULL_AREA_MIN = 0.9
NOISE_SIGMA = 3.0 / 255
EPS = 1e-3


def to_t(a: np.ndarray, device: str) -> torch.Tensor:
    """uint8 H x W x 3 -> float 1 x 3 x H x W in [0, 1]."""
    return torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None].contiguous().to(device)


def box(x: torch.Tensor, k: int) -> torch.Tensor:
    """Box blur with reflect padding."""
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p,) * 4, mode="reflect"), k, stride=1)


def lc_ratio(a: np.ndarray) -> float:
    """Image-level log ratio of luma to chroma high-pass energy."""
    r, g, b = (a[..., c].astype(np.float64) for c in range(3))
    lum = (r + g + b) / 3
    hp = lambda z: z - uniform_filter(z, 3, mode="reflect")
    return float(np.log((hp(lum) ** 2).mean() + EPS) - np.log((hp(r - g) ** 2 + hp(b - g) ** 2).mean() + EPS))


def fingerprint(wm: object, a: np.ndarray, bits: list[int], device: str, gen: torch.Generator) -> dict[str, float]:
    """Decoder fingerprint of one watermarked uint8 image."""
    b = torch.tensor(bits, dtype=torch.float32)
    sign = 2 * b - 1

    def margin(x: torch.Tensor) -> float:
        return float((wm.logits(x.clamp(0, 1)).cpu() * sign).mean())  # type: ignore[attr-defined]

    x = to_t(a, device)
    m0 = margin(x)
    side = a.shape[0]
    low = F.interpolate(F.interpolate(x, size=(128, 128), mode="bilinear", antialias=True), size=(side, side),
                        mode="bilinear")
    noise = torch.randn(x.shape, generator=gen).to(device) * NOISE_SIGMA
    rew = to_t(wm.embed(a, bits), device)  # type: ignore[attr-defined]
    return {
        "m0": m0,
        "r_blur3": margin(box(x, 3)) / m0,
        "r_blur5": margin(box(x, 5)) / m0,
        "r_low128": margin(low) / m0,
        "r_noise": margin(x + noise) / m0,
        "r_sub": margin(2 * x - rew) / m0,
        "lc": lc_ratio(a),
    }


def chains(wm: object) -> dict[str, Callable[[np.ndarray, list[int]], np.ndarray]]:
    """Candidate chains: cover uint8 512 -> watermarked uint8 512."""

    def scaled(s: float) -> Callable[[np.ndarray, list[int]], np.ndarray]:
        def run(c: np.ndarray, bits: list[int]) -> np.ndarray:
            w = wm.embed(c, bits).astype(np.float32)  # type: ignore[attr-defined]
            return np.clip(np.round(c + s * (w - c)), 0, 255).astype(np.uint8)
        return run

    def via(side: int) -> Callable[[np.ndarray, list[int]], np.ndarray]:
        def run(c: np.ndarray, bits: list[int]) -> np.ndarray:
            big = np.asarray(Image.fromarray(c).resize((side, side), Image.Resampling.BICUBIC))
            w = wm.embed(big, bits)  # type: ignore[attr-defined]
            return np.asarray(Image.fromarray(w).resize((c.shape[1], c.shape[0]), Image.Resampling.BICUBIC))
        return run

    def blurred(c: np.ndarray, bits: list[int]) -> np.ndarray:
        w = wm.embed(c, bits)  # type: ignore[attr-defined]
        return np.asarray(Image.fromarray(w).filter(ImageFilter.GaussianBlur(0.5)))

    def twice(c: np.ndarray, bits: list[int]) -> np.ndarray:
        return wm.embed(wm.embed(c, bits), bits)  # type: ignore[attr-defined]

    return {
        "embed512": scaled(1.0), "strength0.5": scaled(0.5), "strength0.75": scaled(0.75),
        "strength1.5": scaled(1.5), "via1024": via(1024), "via768": via(768), "via384": via(384),
        "blur0.5": blurred, "twice": twice,
    }


def main() -> None:
    """Fingerprint real full-regime images and every candidate chain; print z-distances."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", default="pixelseal")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--chains", nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-masks", action="store_true",
                        help="composite each synthetic with the mask of a real full image (hole = clean cover)")
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit))
    from wmloc.watermarks import load_watermark

    wm = load_watermark(args.method, models_dir=args.models, device=args.device)
    meta = json.loads((args.data / "metadata.json").read_text())
    packed = np.load(args.data / "masks_128.npz")
    area = {str(n): float(m.mean()) for n, m in zip(packed["names"], packed["masks"], strict=True)}
    masks128 = {str(n): m.astype(np.uint8) for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)
    load = lambda n: np.asarray(Image.open(args.data / "images" / f"{n}.png").convert("RGB"))
    real_items = [m for m in meta if m["method"] == args.method and m["full_watermark"]
                  and area[str(m["image_name"])] >= FULL_AREA_MIN]
    real_items = [real_items[i] for i in rng.permutation(len(real_items))[: args.n]]
    covers = [m for m in meta if m["method"] != args.method and not m["full_watermark"]]
    covers = [covers[i] for i in rng.permutation(len(covers))[: args.n]]
    t0 = time.time()
    real = [fingerprint(wm, load(str(m["image_name"])), [int(c) for c in str(m["message"])], args.device, gen)
            for m in real_items]
    keys = list(real[0])
    real_mean = {k: float(np.mean([r[k] for r in real])) for k in keys}
    real_se = {k: float(np.std([r[k] for r in real]) / np.sqrt(len(real))) for k in keys}
    print("real", json.dumps({k: round(v, 3) for k, v in real_mean.items()}), flush=True)
    clean = [fingerprint(wm, load(str(m["image_name"])), [int(c) for c in rng.integers(0, 2, wm.nbits)],
                         args.device, gen)["m0"] for m in covers[:6]]  # type: ignore[attr-defined]
    print("clean covers m0 (sanity, ~0)", round(float(np.mean(clean)), 3), flush=True)
    report: dict[str, object] = {"real": real_mean, "real_se": real_se, "chains": {}}
    todo = chains(wm)
    for name, run in todo.items():
        if args.chains and name not in args.chains:
            continue
        rows = []
        for k, m in enumerate(covers):
            bits = [int(c) for c in rng.integers(0, 2, wm.nbits)]  # type: ignore[attr-defined]
            cover = load(str(m["image_name"]))
            marked = run(cover, bits)
            if args.real_masks:
                m128 = masks128[str(real_items[k % len(real_items)]["image_name"])]
                m512 = np.kron(m128, np.ones((4, 4), np.uint8)).astype(bool)[..., None]
                marked = np.where(m512, marked, cover)
            rows.append(fingerprint(wm, marked, bits, args.device, gen))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        se = {k: float(np.std([r[k] for r in rows]) / np.sqrt(len(rows))) for k in keys}
        z = {k: (mean[k] - real_mean[k]) / np.sqrt(se[k] ** 2 + real_se[k] ** 2 + 1e-12) for k in keys}
        dist = float(np.sqrt(np.mean([v**2 for v in z.values()])))
        report["chains"][name] = {"mean": mean, "z": z, "rms_z": dist}  # type: ignore[index]
        print(name, "rms_z", round(dist, 2), json.dumps({k: round(v, 3) for k, v in mean.items()}),
              f"t={time.time() - t0:.0f}s", flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

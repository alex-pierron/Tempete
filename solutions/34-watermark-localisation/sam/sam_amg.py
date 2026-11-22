"""X-LOC-C (JURECA): SAM ViT-H automatic object masks for the transplantation track.

For every listed image, SAM proposes object masks; each mask is reduced to the official 128 x 128 grid
(a cell is 1 when >= 50 % of its 4 x 4 pixels are inside) and stored bit-packed with its SAM scores.
The selection of the transplanted object (vote of the PS/TM/SS head inside each mask) is done later,
out of fold, on the Mac: this task only proposes candidates.

Usage: python sam_amg.py <split_dir> <names.txt> <out.npz> <seconds_budget>
Env: SAM_CKPT (vit_h checkpoint), SAM_WHEEL (segment_anything wheel, imported as a zip).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ["SAM_WHEEL"])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # noqa: E402

GRID = 128
CELL = 4
MAX_AREA_SHARE = 0.6
MIN_AREA_CELLS = 4
MAX_MASKS_PER_IMAGE = 120
POINTS_PER_SIDE = 32
POINTS_PER_BATCH = 256
PRED_IOU_THRESH = 0.80
STABILITY_THRESH = 0.85
SAVE_EVERY = 50
CROP_MARGIN_PX = 12
SEAM_OFFSET_PX = 4
RING_PX = 8
MIN_PAIRS = 20
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def seam_and_hf(y: np.ndarray, lap: np.ndarray, seg: np.ndarray) -> tuple[float, float]:
    """Splice cues of one candidate, computed in its bounding box.

    seam: mean luma step across the mask contour / mean step 4 px away (same direction), 1 = no seam.
    hf: median |Laplacian| inside the eroded mask / median in an 8 px ring outside (pasted objects
    carry another noise level than their new background).
    """
    ys, xs = np.nonzero(seg)
    y0, y1 = max(ys.min() - CROP_MARGIN_PX, 0), min(ys.max() + CROP_MARGIN_PX + 1, seg.shape[0])
    x0, x1 = max(xs.min() - CROP_MARGIN_PX, 0), min(xs.max() + CROP_MARGIN_PX + 1, seg.shape[1])
    a, m, la = y[y0:y1, x0:x1], seg[y0:y1, x0:x1], lap[y0:y1, x0:x1]
    acr, ctl = [], []
    for axis in (0, 1):
        aa, mm = np.moveaxis(a, axis, 0), np.moveaxis(m, axis, 0)
        step = np.abs(aa[1:] - aa[:-1])
        cross = mm[1:] != mm[:-1]
        same = ~cross
        acr.append(step[cross])
        for off in (-SEAM_OFFSET_PX, SEAM_OFFSET_PX):
            sc = np.roll(cross, off, axis=0)
            if off > 0:
                sc[:off] = False
            else:
                sc[off:] = False
            ctl.append(step[sc & same])
    ac, cc = np.concatenate(acr), np.concatenate(ctl)
    seam = float(ac.mean() / max(cc.mean(), 1e-6)) if ac.size >= MIN_PAIRS and cc.size >= MIN_PAIRS else 1.0
    inner = ndimage.binary_erosion(m, iterations=2)
    ring = ndimage.binary_dilation(m, iterations=RING_PX) & ~ndimage.binary_dilation(m, iterations=2)
    if inner.sum() < MIN_PAIRS or ring.sum() < MIN_PAIRS:
        return seam, 1.0
    hf = float(np.median(la[inner]) / max(np.median(la[ring]), 1e-3))
    return seam, hf


def to_grid(seg: np.ndarray) -> np.ndarray:
    """Official 512 -> 128 reduction: cell is 1 when at least half of its pixels are set."""
    cover = seg.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3))
    return cover >= 0.5


def save(out: Path, names: list[str], counts: list[int], packed: list[np.ndarray],
         piou: list[float], stab: list[float], cues: list[tuple[float, float]]) -> None:
    """Write one npz atomically (one file on scratch: inode quota)."""
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        names=np.array(names),
        counts=np.array(counts, dtype=np.int32),
        packed=np.concatenate(packed) if packed else np.zeros((0, GRID * GRID // 8), np.uint8),
        pred_iou=np.array(piou, dtype=np.float32),
        stability=np.array(stab, dtype=np.float32),
        seam=np.array([c[0] for c in cues], dtype=np.float32),
        hf=np.array([c[1] for c in cues], dtype=np.float32),
    )
    tmp.replace(out)


def main() -> None:
    """Run SAM AMG over the listed images within the time budget."""
    split_dir = Path(sys.argv[1])
    names_all = [n.strip() for n in Path(sys.argv[2]).read_text().splitlines() if n.strip()]
    out = Path(sys.argv[3])
    deadline = time.time() + float(sys.argv[4])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=os.environ["SAM_CKPT"]).to(device)
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=POINTS_PER_BATCH,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        min_mask_region_area=0,
    )
    names: list[str] = []
    counts: list[int] = []
    packed: list[np.ndarray] = []
    piou: list[float] = []
    stab: list[float] = []
    cues: list[tuple[float, float]] = []
    start = time.time()
    for i, name in enumerate(names_all):
        if time.time() > deadline:
            print(f"deadline reached after {i} images", flush=True)
            break
        img = np.asarray(Image.open(split_dir / "images" / f"{name}.png").convert("RGB"))
        y = img.astype(np.float32) @ LUMA
        lap = np.abs(ndimage.laplace(y))
        with torch.inference_mode():
            anns = gen.generate(img)
        anns = sorted(anns, key=lambda a: -a["predicted_iou"])
        kept = 0
        for a in anns:
            g = to_grid(a["segmentation"])
            area = int(g.sum())
            if area < MIN_AREA_CELLS or area > MAX_AREA_SHARE * GRID * GRID:
                continue
            packed.append(np.packbits(g.reshape(-1))[None])
            piou.append(float(a["predicted_iou"]))
            stab.append(float(a["stability_score"]))
            cues.append(seam_and_hf(y, lap, a["segmentation"]))
            kept += 1
            if kept >= MAX_MASKS_PER_IMAGE:
                break
        names.append(name)
        counts.append(kept)
        if (i + 1) % SAVE_EVERY == 0:
            save(out, names, counts, packed, piou, stab, cues)
            print(f"{i + 1}/{len(names_all)} [{time.time() - start:.0f} s] last kept={kept}", flush=True)
    save(out, names, counts, packed, piou, stab, cues)
    print(f"done {len(names)} images, {len(piou)} masks [{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
    main()

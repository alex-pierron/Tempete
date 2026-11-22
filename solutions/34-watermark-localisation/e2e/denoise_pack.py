"""X-LOC-B (Mac, numpy + PIL, light): inputs of the learned map denoiser, train (OOF) and test, all 3 methods.

Per image: the deployed source map of its method (PS 3-head mean, TM 2-head mean, SS split7, all out of fold on
train), the e2e map (mean of the given xb runs, PS / TM only, zeros for SS), a 128 x 128 RGB thumbnail (4 x 4
block mean of the 512 image), the method id, and on train the official 128 mask and the regime flag.

Usage: python denoise_pack.py <out.npz> <e2e run dir> <run tag> [<run tag> ...]
       (files <dir>/<tag>_<ps|tm>_oof.npz and _test.npz, e.g. tags xb1 xb3)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "loclab"))  # loclab helper modules
from common import gt128  # noqa: E402
from size_prior import train_map  # noqa: E402

ROOT = Path(__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc"))
METHODS = ("pixelseal", "trustmark", "stegastamp")
SHORT = {"pixelseal": "ps", "trustmark": "tm"}
HEADS_TEST = {
    "pixelseal": (ROOT / "loclab/x23b/ens_ps_3.npz", "pixelseal", "names_pixelseal"),
    "trustmark": (ROOT / "loclab/x23b/ens_tm_2.npz", "trustmark", "names_trustmark"),
    "stegastamp": (ROOT / "team_final/ss_split7_u1x5_zoom2.npz", "test_finetuned", "names_test"),
}
GRID = 128
BLOCK = 4
SEPARATE = __import__("os").environ.get("XLB_SEPARATE", "0") == "1"
SAM_PREFIX = __import__("os").environ.get("XLB_SAM", "")
"""Prefix of X-LOC-C SAM files <prefix>_<method>_<split>.npz (mask 0/1 + apply flag): two extra channels."""


def sam_channels(method: str, split: str, names: list[str]) -> NDArray[np.uint8]:
    """[n, 2, 128, 128]: SAM mask where it applies (else 0), and the apply flag broadcast."""
    z = np.load(f"{SAM_PREFIX}_{method}_{split}.npz")
    position = {str(n): i for i, n in enumerate(z[f"names_{method}"])}
    order = [position[n] for n in names]
    apply = z[f"apply_{method}"][order].astype(np.uint8)[:, None, None]
    mask = z[method][order].astype(np.uint8) * apply
    return np.stack([mask, np.broadcast_to(apply, mask.shape)], 1).astype(np.uint8)
"""Keep every e2e run as its own channel instead of their mean."""


def thumbnail(path: Path) -> NDArray[np.uint8]:
    """[3, 128, 128] uint8 block mean of the 512 image."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    small = image.reshape(GRID, BLOCK, GRID, BLOCK, 3).mean((1, 3))
    return np.asarray(small.round().clip(0, 255).astype(np.uint8).transpose(2, 0, 1))


def by_name(names: list[str], source: list[str], maps: NDArray[np.float32]) -> NDArray[np.float32]:
    """Reorder maps from the source order to names."""
    position = {n: i for i, n in enumerate(source)}
    return maps[[position[n] for n in names]]


def e2e_maps(directory: Path, tags: list[str], method: str, split: str, names: list[str]) -> NDArray[np.float32]:
    """The e2e runs of one method on one split as channels [n, T, 128, 128] (SEPARATE=1) or their mean [n, 128, 128]."""
    out = []
    for tag in tags:
        z = np.load(directory / f"{tag}_{SHORT[method]}_{split}.npz")
        if split == "oof" and not bool(z["seen"].all()):
            raise SystemExit(f"{tag} {method}: incomplete OOF")
        out.append(by_name(names, [str(n) for n in z[f"names_{method}"]], z[method].astype(np.float32)))
    return np.stack(out, 1) if SEPARATE else np.mean(out, 0)


def main() -> None:
    """Write the denoiser pack."""
    out, directory, tags = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    gts = gt128()
    arrays: dict[str, list[NDArray[np.generic]]] = {}
    for split in ("train", "test"):
        rows = json.loads((ROOT / "release" / split / "metadata.json").read_text())
        names_all, method_all, head_all, e2e_all, rgb_all, gt_all, full_all, extra_all = [], [], [], [], [], [], [], []
        for method_id, method in enumerate(METHODS):
            sub = [r for r in rows if r["method"] == method]
            names = [str(r["image_name"]) for r in sub]
            if split == "train":
                head = train_map(method, names).astype(np.float32)
                gt_all.append(np.stack([gts[n] for n in names]).astype(np.uint8))
                full_all.append(np.array([bool(r["full_watermark"]) for r in sub]))
            else:
                path, key, names_key = HEADS_TEST[method]
                z = np.load(path)
                head = by_name(names, [str(n) for n in z[names_key]], z[key].astype(np.float32))
            e2e = (e2e_maps(directory, tags, method, "oof" if split == "train" else "test", names)
                   if method in SHORT else np.zeros((len(names), len(tags), GRID, GRID) if SEPARATE else head.shape,
                                                     dtype=np.float32))
            names_all.append(np.array(names))
            method_all.append(np.full(len(names), method_id, dtype=np.int64))
            head_all.append(head.astype(np.float16))
            e2e_all.append(e2e.astype(np.float16))
            if SAM_PREFIX:
                extra_all.append(sam_channels(method, split, names))
            rgb_all.append(np.stack([thumbnail(ROOT / "release" / split / "images" / f"{n}.png") for n in names]))
        arrays[f"{split}_names"] = [np.concatenate(names_all)]
        arrays[f"{split}_method"] = [np.concatenate(method_all)]
        arrays[f"{split}_head"] = [np.concatenate(head_all)]
        arrays[f"{split}_e2e"] = [np.concatenate(e2e_all)]
        arrays[f"{split}_rgb"] = [np.concatenate(rgb_all)]
        if extra_all:
            arrays[f"{split}_extra"] = [np.concatenate(extra_all)]
        if split == "train":
            arrays["train_gt"] = [np.concatenate(gt_all)]
            arrays["train_full"] = [np.concatenate(full_all)]
    flat = {k: v[0] for k, v in arrays.items()}
    np.savez_compressed(out, **flat)
    print({k: v.shape for k, v in flat.items()}, "->", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

"""Pack per-image inputs into a few container files (team B2).

The shared scratch quota counts INODES and organisers forbid one file per image. Two inputs of the
head were one-file-per-image: team A's global decodes (`research_cache/<split>/<method>/<name>.global.npz`)
and the clean photographs used to build synthetic samples. This module packs each into ONE file:

* `<data>/wmloc/team_b2/global/<split>_<method>.npz`: `names` [n], `logits` float32 [n, n_bits];
* `<data>/wmloc/team_b2/clean_images.npz`: one uint8 [h, w, 3] array per photograph (short side 512).

`load_global_table` and `load_clean_pool` read them back; callers fall back to the per-image files
when a container is absent, so nothing breaks on a machine that never ran the packer.

Usage:
    python -m hackathon.wmloc.research.b2_pack global
    python -m hackathon.wmloc.research.b2_pack clean --src /path/to/photos
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from hackathon.wmloc.research.b2_features import cache_root
from hackathon.wmloc.research.data import SPLITS, load_records
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.scoring import global_logits

GLOBAL_SUBDIR = "global"
CLEAN_FILE = "clean_images.npz"
CLEAN_SHORT_SIDE = 512
CLEAN_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
_GLOBAL_TABLES: dict[tuple[str, str], dict[str, NDArray[np.float32]]] = {}


class PackError(ResearchError):
    """Nothing to pack, or a container is unreadable."""


def global_path(split: str, method: str) -> Path:
    """Container of the global decodes of one (split, method)."""
    return cache_root().parent / GLOBAL_SUBDIR / f"{split}_{method}.npz"


def clean_path() -> Path:
    """Container of the clean photographs."""
    return cache_root().parent / CLEAN_FILE


def _write(path: Path, arrays: dict[str, NDArray[np.generic]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def pack_global(split: str, method: str) -> Path:
    """Pack team A's per-image global decodes of (split, method) into one container.

    Raises:
        PackError: no image of that method, or a decode is missing.
    """
    records = [r for r in load_records(split) if r.method == method]
    if not records:
        raise PackError(f"no {method} image in {split}")
    logits = np.stack([global_logits(split, method, r.name) for r in records]).astype(np.float32)
    path = global_path(split, method)
    _write(path, {"names": np.array([r.name for r in records]), "logits": logits})
    _GLOBAL_TABLES.pop((split, method), None)
    return path


def load_global_table(split: str, method: str) -> dict[str, NDArray[np.float32]] | None:
    """`{name: logits [n_bits]}` of the container, or None when it does not exist."""
    key = (split, method)
    if key in _GLOBAL_TABLES:
        return _GLOBAL_TABLES[key]
    path = global_path(split, method)
    if not path.is_file():
        return None
    with np.load(path) as archive:
        names = [str(n) for n in archive["names"]]
        table = dict(zip(names, archive["logits"].astype(np.float32), strict=True))
    _GLOBAL_TABLES[key] = table
    return table


def pack_clean(source: Path) -> Path:
    """Pack the photographs of `source` (short side resized to 512) into one container.

    Raises:
        PackError: the folder holds no image.
    """
    arrays: dict[str, NDArray[np.generic]] = {}
    for path in sorted(p for p in source.glob("*") if p.suffix.lower() in CLEAN_SUFFIXES):
        with Image.open(path) as handle:
            picture = handle.convert("RGB")
        width, height = picture.size
        scale = CLEAN_SHORT_SIDE / min(width, height)
        size = (
            max(CLEAN_SHORT_SIDE, round(width * scale)),
            max(CLEAN_SHORT_SIDE, round(height * scale)),
        )
        arrays[path.stem] = np.asarray(
            picture.resize(size, Image.Resampling.BICUBIC), dtype=np.uint8
        )
    if not arrays:
        raise PackError(f"no photograph in {source}")
    target = clean_path()
    _write(target, arrays)
    return target


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("global")
    clean = sub.add_parser("clean")
    clean.add_argument("--src", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "global":
        for split in SPLITS:
            for method in sorted({r.method for r in load_records(split)}):
                print(pack_global(split, method), flush=True)
    else:
        print(pack_clean(args.src), flush=True)


if __name__ == "__main__":
    main()

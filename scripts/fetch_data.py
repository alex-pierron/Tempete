"""Download evaluation image sets into `data/<dataset>/` (idempotent, no archive kept).

Datasets:
    kodak            24 uncompressed 768x512 photos (r0k.us), ~15 MB.
    div2k_valid      100 high-resolution (2K) validation images (ETH Zurich), ~450 MB.
    coco_val_subset  N evenly spaced images of COCO val2017 (`--n`, default 500), ~160 KB each.

The two zip-based sets are read with HTTP range requests (central directory first, then one
member at a time), so the multi-hundred-MB archives are never stored. Files already present are
never downloaded again; writes are atomic (`.part` then rename). A guard refuses to exceed the
total data budget.

Usage:
    WMKIT_DATA_DIR=/path/to/data python scripts/fetch_data.py --dataset kodak
    python scripts/fetch_data.py --dataset coco_val_subset --n 200 --out /tmp/coco200
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from wmkit.config import data_dir

# --- Constants ---------------------------------------------------------------------------------

KODAK_COUNT = 24
KODAK_URL_TEMPLATE = "https://r0k.us/graphics/kodak/kodak/kodim{index:02d}.png"
DIV2K_VALID_ZIP_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DIV2K_VALID_COUNT = 100
COCO_VAL_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_DEFAULT_COUNT = 500
COCO_EXTENSION = ".jpg"
DIV2K_EXTENSION = ".png"

DATASETS: tuple[str, ...] = ("kodak", "div2k_valid", "coco_val_subset")

DATA_BUDGET_BYTES = int(1.5 * 1024**3)
"""Total on-disk budget of the data directory (all datasets together)."""
RANGE_BLOCK_BYTES = 512 * 1024
"""Read-ahead window of the HTTP range reader (the zip directory sits in the last blocks)."""
HTTP_TIMEOUT_SECONDS = 60.0
HTTP_USER_AGENT = "wmkit-fetch/0.1"
PARTIAL_SUFFIX = ".part"
EXIT_OK = 0
EXIT_FAILURE = 1


class DataFetchError(RuntimeError):
    """A dataset could not be fetched, validated or fitted in the disk budget."""


# --- HTTP helpers ------------------------------------------------------------------------------


def _request(
    url: str, byte_range: tuple[int, int] | None = None, method: str = "GET"
) -> urllib.request.Request:
    headers = {"User-Agent": HTTP_USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return urllib.request.Request(url, headers=headers, method=method)


def http_get(url: str) -> bytes:
    """Return the body of `url`."""
    with urllib.request.urlopen(_request(url), timeout=HTTP_TIMEOUT_SECONDS) as response:
        return bytes(response.read())


def http_content_length(url: str) -> int:
    """Return the size in bytes of `url` (HEAD request)."""
    with urllib.request.urlopen(_request(url, method="HEAD"), timeout=HTTP_TIMEOUT_SECONDS) as response:
        return int(response.headers["Content-Length"])


def http_get_range(url: str, start: int, end_inclusive: int) -> bytes:
    """Return bytes [start, end_inclusive] of `url` through a Range request."""
    with urllib.request.urlopen(
        _request(url, (start, end_inclusive)), timeout=HTTP_TIMEOUT_SECONDS
    ) as response:
        body = bytes(response.read())
    expected = end_inclusive - start + 1
    if len(body) != expected:
        raise DataFetchError(
            f"range {start}-{end_inclusive} of {url} returned {len(body)} bytes, expected "
            f"{expected} (server without Range support?)"
        )
    return body


class RangeReader(io.RawIOBase):
    """Read-only seekable file over a remote blob, fed by `fetch(start, end_inclusive)`.

    A one-block cache makes the many small sequential reads of `zipfile` cost one request each.
    """

    def __init__(self, size: int, fetch: Callable[[int, int], bytes]) -> None:
        self._size = size
        self._fetch = fetch
        self._position = 0
        self._block_start = 0
        self._block = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        else:
            self._position = self._size + offset
        self._position = max(0, min(self._position, self._size))
        return self._position

    def read(self, size: int = -1) -> bytes:
        remaining = self._size - self._position
        wanted = remaining if size < 0 else min(size, remaining)
        if wanted <= 0:
            return b""
        block_end = self._block_start + len(self._block)
        if not self._block_start <= self._position or self._position + wanted > block_end:
            fetch_end = min(self._position + max(wanted, RANGE_BLOCK_BYTES), self._size) - 1
            self._block = self._fetch(self._position, fetch_end)
            self._block_start = self._position
        offset = self._position - self._block_start
        data = self._block[offset : offset + wanted]
        self._position += len(data)
        return data

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


# --- Selection and I/O -------------------------------------------------------------------------


def evenly_spaced(names: Sequence[str], count: int) -> list[str]:
    """Deterministically pick `count` names evenly spread over the sorted `names`.

    Raises:
        DataFetchError: `count` is not positive or exceeds the number of names.
    """
    ordered = sorted(names)
    if count < 1 or count > len(ordered):
        raise DataFetchError(f"--n must be in [1, {len(ordered)}], got {count}")
    return [ordered[(index * len(ordered)) // count] for index in range(count)]


def directory_size_bytes(directory: Path) -> int:
    """Total size of the regular files below `directory` (0 if absent)."""
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _is_valid_image(payload: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError):  # decoder failure means a corrupt download
        return False
    return True


def _write_atomic(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    partial.write_bytes(payload)
    partial.replace(destination)


def _store_image(destination: Path, payload: bytes) -> None:
    if not _is_valid_image(payload):
        raise DataFetchError(f"downloaded file for {destination.name} is not a valid image")
    _write_atomic(destination, payload)


def _count_images(directory: Path, extension: str) -> int:
    return sum(1 for path in directory.glob(f"*{extension}") if path.stat().st_size > 0)


def _check_budget(root: Path, extra_bytes: int) -> None:
    used = directory_size_bytes(root)
    if used + extra_bytes > DATA_BUDGET_BYTES:
        raise DataFetchError(
            f"data budget exceeded: {used / 1e6:.0f} MB used in {root} + {extra_bytes / 1e6:.0f} MB "
            f"needed > {DATA_BUDGET_BYTES / 1e6:.0f} MB"
        )


# --- Datasets ----------------------------------------------------------------------------------


def fetch_kodak(out: Path) -> int:
    """Download the 24 Kodak images into `out`; return how many images are present."""
    out.mkdir(parents=True, exist_ok=True)
    for index in range(1, KODAK_COUNT + 1):
        destination = out / f"kodim{index:02d}.png"
        if destination.exists() and destination.stat().st_size > 0:
            continue
        _store_image(destination, http_get(KODAK_URL_TEMPLATE.format(index=index)))
    return _count_images(out, ".png")


def _fetch_from_zip(
    url: str,
    out: Path,
    extension: str,
    count: int | None,
    budget_root: Path,
    open_range: Callable[[str], tuple[int, Callable[[int, int], bytes]]] | None = None,
) -> int:
    """Extract (a subset of) the images of a remote zip into `out` using range requests."""
    out.mkdir(parents=True, exist_ok=True)
    if open_range is None:
        size, fetch = http_content_length(url), (lambda s, e: http_get_range(url, s, e))
    else:
        size, fetch = open_range(url)
    with zipfile.ZipFile(RangeReader(size, fetch)) as archive:
        members = {
            Path(info.filename).name: info
            for info in archive.infolist()
            if info.filename.lower().endswith(extension) and not info.is_dir()
        }
        wanted = sorted(members) if count is None else evenly_spaced(list(members), count)
        missing = [name for name in wanted if not (out / name).exists()]
        _check_budget(budget_root, sum(members[name].file_size for name in missing))
        for name in missing:
            _store_image(out / name, archive.read(members[name]))
    return _count_images(out, extension)


def fetch_div2k_valid(out: Path, budget_root: Path | None = None) -> int:
    """Download the 100 DIV2K validation images (2K) into `out`; return the number present."""
    return _fetch_from_zip(
        DIV2K_VALID_ZIP_URL, out, DIV2K_EXTENSION, None, budget_root or out.parent
    )


def fetch_coco_val_subset(out: Path, count: int, budget_root: Path | None = None) -> int:
    """Download `count` evenly spaced COCO val2017 images into `out`; return the number present."""
    return _fetch_from_zip(
        COCO_VAL_ZIP_URL, out, COCO_EXTENSION, count, budget_root or out.parent
    )


# --- CLI ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch evaluation image sets (idempotent).")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--n",
        type=int,
        default=COCO_DEFAULT_COUNT,
        help=f"images to keep for coco_val_subset (default {COCO_DEFAULT_COUNT}); ignored otherwise",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="destination (default: <data_dir>/<dataset>)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints the number of images present and returns a process exit code."""
    args = build_parser().parse_args(argv)
    root = data_dir()
    out: Path = args.out if args.out is not None else root / args.dataset
    try:
        if args.dataset == "kodak":
            present = fetch_kodak(out)
        elif args.dataset == "div2k_valid":
            present = fetch_div2k_valid(out, root)
        else:
            present = fetch_coco_val_subset(out, args.n, root)
    except (DataFetchError, OSError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    size_mb = directory_size_bytes(out) / 1e6
    print(f"{args.dataset}: {present} images in {out} ({size_mb:.1f} MB)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

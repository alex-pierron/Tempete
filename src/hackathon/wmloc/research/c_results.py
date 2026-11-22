"""Per-image results of a team C run, stored as append-only JSONL shards (few inodes).

The shared scratch file system of the cluster is at 99.7 % of its inode quota, so one file per
image is no longer an option: every shard of a run appends one JSON line per image to
`<directory>/<tag>.jsonl`. A crash can only truncate the last line, which readers ignore, and a
restarted run skips the names already present, so runs stay idempotent and resumable. Legacy
layouts (one `<name>.json` per image) are still read. Array results (occlusion maps) go through
`ArrayChunkWriter`: ten images per NPZ chunk instead of one file each.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hackathon.wmloc.research.errors import ResearchError

JSONL_SUFFIX = ".jsonl"
JSON_SUFFIX = ".json"
NAME_KEY = "name"


class ResultStoreError(ResearchError):
    """The directory holds no result."""


def parse_lines(path: Path) -> list[dict[str, Any]]:
    """Payloads of one JSONL file; a truncated or corrupt line is skipped."""
    payloads: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and NAME_KEY in payload:
            payloads.append(payload)
    return payloads


def load_all(directory: Path) -> list[dict[str, Any]]:
    """Every payload of a results directory (JSONL shards and legacy per-image JSON), by name.

    A name present twice keeps its last occurrence.

    Raises:
        ResultStoreError: the directory does not exist or holds no result.
    """
    if not directory.is_dir():
        raise ResultStoreError(f"{directory} is not a directory")
    by_name: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"*{JSONL_SUFFIX}")):
        for payload in parse_lines(path):
            by_name[str(payload[NAME_KEY])] = payload
    for path in sorted(directory.glob(f"*{JSON_SUFFIX}")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_name.setdefault(str(payload[NAME_KEY]), payload)
    if not by_name:
        raise ResultStoreError(f"no result in {directory}")
    return list(by_name.values())


def done_names(directory: Path) -> set[str]:
    """Names already stored (empty set if the directory does not exist yet)."""
    if not directory.is_dir():
        return set()
    try:
        return {str(p[NAME_KEY]) for p in load_all(directory)}
    except ResultStoreError:
        return set()


class ShardWriter:
    """Appends payloads to `<directory>/<tag>.jsonl`, one flushed line per payload."""

    def __init__(self, directory: Path, tag: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / f"{tag}{JSONL_SUFFIX}"

    @property
    def path(self) -> Path:
        """The shard file."""
        return self._path

    def append(self, payload: dict[str, Any]) -> None:
        """Add one line; the file is opened per call so an interrupted run loses nothing."""
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()


# --- array results (occlusion maps): chunks of several images per NPZ -------------------------

NPZ_SUFFIX = ".npz"
CHUNK_INFIX = "_part"
KEY_SEPARATOR = "/"
CHUNK_IMAGES = 10
"""Images per chunk file: a crash loses at most this many, and the file count stays 10x lower."""


class ArrayChunkWriter:
    """Buffers per-image array dicts and writes them `CHUNK_IMAGES` at a time into one NPZ.

    Keys inside a chunk are `<image name>/<array name>`; files are `<tag>_part<k>.npz`, written
    through a temporary file and renamed, so a chunk is either complete or absent.
    """

    def __init__(self, directory: Path, tag: str, chunk_images: int = CHUNK_IMAGES) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._tag = tag
        self._chunk_images = chunk_images
        self._buffer: dict[str, Any] = {}
        self._images = 0
        self._next_part = len(list(directory.glob(f"{tag}{CHUNK_INFIX}*{NPZ_SUFFIX}")))

    def add(self, name: str, arrays: dict[str, Any]) -> None:
        """Buffer one image; flush when the chunk is full."""
        for key, value in arrays.items():
            self._buffer[f"{name}{KEY_SEPARATOR}{key}"] = value
        self._images += 1
        if self._images >= self._chunk_images:
            self.flush()

    def flush(self) -> None:
        """Write the buffered images (no-op if empty)."""
        if not self._buffer:
            return
        final = self._directory / f"{self._tag}{CHUNK_INFIX}{self._next_part}{NPZ_SUFFIX}"
        temporary = final.with_name(final.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **self._buffer)
        temporary.replace(final)
        self._next_part += 1
        self._buffer = {}
        self._images = 0


def load_arrays(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    """Arrays of every stored image, `{name: {array name: array}}` (chunks and legacy files).

    Raises:
        ResultStoreError: the directory does not exist or holds no array file.
    """
    if not directory.is_dir():
        raise ResultStoreError(f"{directory} is not a directory")
    images: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(directory.glob(f"*{NPZ_SUFFIX}")):
        with np.load(path) as archive:
            if CHUNK_INFIX in path.stem:
                for key in archive.files:
                    name, _, array = key.partition(KEY_SEPARATOR)
                    images.setdefault(name, {})[array] = archive[key]
            else:
                images[path.stem] = {key: archive[key] for key in archive.files}
    if not images:
        raise ResultStoreError(f"no array result in {directory}")
    return images


def done_array_names(directory: Path) -> set[str]:
    """Names of the images already stored (keys only are read, empty if nothing yet)."""
    if not directory.is_dir():
        return set()
    names: set[str] = set()
    for path in directory.glob(f"*{NPZ_SUFFIX}"):
        if CHUNK_INFIX in path.stem:
            with np.load(path) as archive:
                names.update(key.partition(KEY_SEPARATOR)[0] for key in archive.files)
        else:
            names.add(path.stem)
    return names

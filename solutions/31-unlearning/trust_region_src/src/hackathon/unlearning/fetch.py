"""Download of the task files from the public Hugging Face dataset, with integrity checks.

`model.pt` has a known sha256 and size (see constants); `forget_images.npz` has no published
checksum, so it is checked structurally (key, dtype, shape) by `data.load_forget_images`.
Downloads are idempotent: a file that already matches is not fetched again, and an existing file
that does not match is reported instead of silently overwritten (it may be a hand-placed copy).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import (
    FORGET_IMAGES_FILENAME,
    HASH_CHUNK_BYTES,
    HF_DATASET_REPO,
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
)


class FetchError(HackathonError):
    """A task file could not be downloaded or failed its integrity check."""


@dataclass(frozen=True)
class FetchedFile:
    """One file of the task directory after `fetch_task_files`."""

    path: Path
    size_bytes: int
    sha256: str
    downloaded: bool


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file (the model is ~344 MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def model_matches_official(path: Path) -> bool:
    """True when `path` has the official model size and sha256."""
    return (
        path.is_file()
        and path.stat().st_size == MODEL_SIZE_BYTES
        and sha256_of(path) == MODEL_SHA256
    )


def _download(filename: str, destination_dir: Path, token: str | None) -> Path:
    """Fetch `filename` from the dataset straight into `destination_dir`.

    `local_dir` makes huggingface_hub write the real file there (no symlink into a cache), so the
    task directory is self-contained and no second copy of the 344 MB model lands in ~/.cache.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FetchError("huggingface_hub is not installed: pip install huggingface_hub") from exc
    try:
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(destination_dir),
            token=token,
        )
    except Exception as exc:  # hub raises many unrelated types
        raise FetchError(f"Download of {filename} from {HF_DATASET_REPO} failed: {exc}") from exc
    return Path(downloaded)


def fetch_task_files(task_dir: Path, token: str | None = None) -> list[FetchedFile]:
    """Download model.pt and forget_images.npz into `task_dir` and verify the model.

    Args:
        task_dir: Destination directory (created if missing).
        token: Optional Hugging Face token (the dataset is public; not needed).

    Returns:
        One FetchedFile per file.

    Raises:
        FetchError: network failure, or model.pt does not match the official sha256/size, or an
            existing model.pt differs from the official one.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchedFile] = []

    model_path = task_dir / MODEL_FILENAME
    downloaded = False
    if not model_matches_official(model_path):
        if model_path.exists():
            raise FetchError(
                f"{model_path} exists but is not the official model (size/sha256 mismatch); "
                "move it away and rerun fetch."
            )
        _download(MODEL_FILENAME, task_dir, token)
        downloaded = True
        if not model_matches_official(model_path):
            raise FetchError(
                f"Downloaded {MODEL_FILENAME} fails the official check "
                f"(expected {MODEL_SIZE_BYTES} bytes, sha256 {MODEL_SHA256})."
            )
    results.append(
        FetchedFile(model_path, model_path.stat().st_size, sha256_of(model_path), downloaded)
    )

    images_path = task_dir / FORGET_IMAGES_FILENAME
    downloaded = False
    if not images_path.is_file():
        _download(FORGET_IMAGES_FILENAME, task_dir, token)
        downloaded = True
    results.append(
        FetchedFile(images_path, images_path.stat().st_size, sha256_of(images_path), downloaded)
    )
    return results

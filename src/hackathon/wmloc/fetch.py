"""Download and unpack the gated localization dataset (starter kit, train and test archives).

The dataset repo was gated at first (HTTP 403 without an authorized account) and is now public:
the token is optional. When one is configured (`Settings.hf_token`) it is sent; this module never
tries another route. `huggingface_hub` is imported lazily so the rest of the package works without it.

Layout (reported by a teammate, unverified until we have access): the repo holds three archives,
`starter_kit.zip`, `train.zip`, `test.zip`, plus a README.txt, ~570 MB in total. They are unpacked
next to the downloaded files, exactly like the team command `unzip starter_kit.zip -d .`.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from hackathon.common.config import Settings
from hackathon.wmloc.constants import (
    ARCHIVE_TARGETS,
    DEFAULT_REPO_ID,
    EXTRACTED_MARKER_SUFFIX,
    FALLBACK_REPO_IDS,
    HTTP_FORBIDDEN,
    HTTP_UNAUTHORIZED,
    KIT_ARCHIVES,
    WMLOC_DATA_SUBDIR,
)
from hackathon.wmloc.errors import FetchError

MAX_UNCOMPRESSED_BYTES = 8 * 1024**3
"""Sanity bound on one archive's expanded size (the whole dataset is ~570 MB zipped)."""
COPY_CHUNK_BYTES = 1 << 20
NO_ACCESS_HINT = (
    "The Hub reports a missing repo (404) when it hides a gated dataset from a token without "
    "access: if the dataset is gated again, set HF_TOKEN to an authorized account's token."
)


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a fetch.

    Attributes:
        repo_id: Repo the files came from.
        destination: Directory holding the archives and their unpacked content.
        extracted: Archives unpacked by this call.
        already_extracted: Archives skipped because a matching marker exists.
        missing: Expected archives that the repo does not contain.
    """

    repo_id: str
    destination: Path
    extracted: tuple[str, ...]
    already_extracted: tuple[str, ...]
    missing: tuple[str, ...]


def default_destination(settings: Settings) -> Path:
    """`<data_dir>/wmloc`, where the rest of the package looks for the dataset."""
    return settings.data_dir / WMLOC_DATA_SUBDIR


class _RepoNotFoundError(Exception):
    """Internal: the repo id does not exist (or is hidden from us); try the next candidate."""


def _download_repo(repo_id: str, destination: Path, token: str | None) -> None:
    """Run `snapshot_download` and translate Hub failures into FetchError / _RepoNotFoundError."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    destination.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", local_dir=str(destination), token=token
        )
    except GatedRepoError as exc:
        raise FetchError(
            f"{repo_id}: access denied (gated dataset, HTTP 403). The Hugging Face account behind "
            "HF_TOKEN is not authorized: ask the organizers to grant it access, then retry."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise _RepoNotFoundError(repo_id) from exc
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in (HTTP_FORBIDDEN, HTTP_UNAUTHORIZED):
            raise FetchError(
                f"{repo_id}: HTTP {status}, the token is invalid or not authorized for this "
                "gated dataset."
            ) from exc
        raise FetchError(f"{repo_id}: Hub error (HTTP {status}): {exc}") from exc
    except Exception as exc:
        # huggingface_hub changed HTTP stacks between versions (requests, then httpx), so the
        # network failures it lets through have no stable common type: report them uniformly.
        raise FetchError(f"{repo_id}: download failed ({type(exc).__name__}): {exc}") from exc


def _marker_path(destination: Path, archive_name: str) -> Path:
    return destination / f".{archive_name}{EXTRACTED_MARKER_SUFFIX}"


def _extract_archive(archive: Path, destination: Path) -> None:
    """Unpack `archive` into `destination`, refusing path traversal and silent overwrites.

    The archive comes from a download, so it is untrusted: every member must resolve inside
    `destination` (zip-slip), and an existing file with a different size aborts instead of
    being overwritten. Re-running after a crash is safe: identical files are skipped.

    Raises:
        FetchError: bad zip, escaping member, oversized archive, or conflicting existing file.
    """
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
                raise FetchError(f"{archive.name}: expands beyond {MAX_UNCOMPRESSED_BYTES} bytes")
            for member in members:
                target = (root / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise FetchError(f"{archive.name}: unsafe member path {member.filename!r}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists():
                    if target.stat().st_size == member.file_size:
                        continue
                    raise FetchError(
                        f"{archive.name}: {member.filename} already exists with a different "
                        "size; move it away before unpacking again."
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                with bundle.open(member) as source, partial.open("wb") as sink:
                    shutil.copyfileobj(source, sink, COPY_CHUNK_BYTES)
                partial.replace(target)
    except zipfile.BadZipFile as exc:
        raise FetchError(f"{archive.name}: not a valid zip file (truncated download?)") from exc


def unpack_archives(destination: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Unpack the three archives found in `destination`, idempotently.

    Each archive goes to its folder of `ARCHIVE_TARGETS`. A marker file `.<archive>.extracted` records the archive size once it is fully unpacked;
    a later call skips that archive unless its size changed (re-download).

    Returns:
        (extracted, already_extracted, missing) archive names.
    """
    extracted: list[str] = []
    already: list[str] = []
    missing: list[str] = []
    for name in KIT_ARCHIVES:
        archive = destination / name
        if not archive.is_file():
            missing.append(name)
            continue
        marker = _marker_path(destination, name)
        size_text = str(archive.stat().st_size)
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == size_text:
            already.append(name)
            continue
        _extract_archive(archive, destination / ARCHIVE_TARGETS[name])
        marker.write_text(size_text + "\n", encoding="utf-8")
        extracted.append(name)
    return tuple(extracted), tuple(already), tuple(missing)


def fetch_dataset(
    settings: Settings,
    *,
    repo_id: str | None = None,
    destination: Path | None = None,
    unpack: bool = True,
) -> FetchResult:
    """Download the dataset (with `settings.hf_token` if set), then unpack its archives.

    Candidates are tried in order (`repo_id` if given, else the official-script repo and then
    the `_Paris` repo from the PDF); only a 404 moves on to the next one, an access error stops.

    Raises:
        FetchError: access denied, all repos missing, network failure, or no archive found.
    """
    token = settings.hf_token
    target = destination if destination is not None else default_destination(settings)
    candidates = (repo_id,) if repo_id else (DEFAULT_REPO_ID, *FALLBACK_REPO_IDS)
    used: str | None = None
    for candidate in candidates:
        try:
            _download_repo(candidate, target, token)
        except _RepoNotFoundError:
            continue
        used = candidate
        break
    if used is None:
        raise FetchError(f"repo(s) not found: {', '.join(candidates)}. {NO_ACCESS_HINT}")
    if not unpack:
        return FetchResult(used, target, (), (), ())
    extracted, already, missing = unpack_archives(target)
    if len(missing) == len(KIT_ARCHIVES):
        present = sorted(path.name for path in target.iterdir())[:20]
        raise FetchError(
            f"{used}: none of {', '.join(KIT_ARCHIVES)} was found in {target}; "
            f"the layout differs from the one reported. Files present: {present}"
        )
    return FetchResult(used, target, extracted, already, missing)

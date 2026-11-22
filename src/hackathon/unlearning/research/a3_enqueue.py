"""Team A3 access to the shared submission queue: validated file first, manifest LAST.

Protocol set by the coordinator (see `enqueue.py` and `scripts/hackathon/submit_queue.py`): the
research teams never contact the evaluation server. `<name>.pt` is copied into
`<data>/queue/unlearning/`, then its manifest `<name>.json` is written LAST (the daemon ignores a
file without manifest, so a half-copied model is never sent). The daemon sends the highest
priority first and writes the public score to `done/<name>.json`.

Rules enforced here: the file passes the official validator, the name is unused, and team A3 keeps
at most `MAX_OWN_QUEUED` candidates waiting (344 MB each on a shared disk). This module never
contacts the server and never reads the API key.

CLI:
    python -m hackathon.unlearning.research.a3_enqueue add FILE --name N --priority P \
        --expected 0.05 --rationale "..."
    python -m hackathon.unlearning.research.a3_enqueue status
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from hackathon.common.errors import HackathonError
from hackathon.unlearning.validate import validate_submission

TEAM_NAME = "r-unl-a3"
QUEUE_SUBDIR = Path("queue") / "unlearning"
DONE_SUBDIR = "done"
MAX_OWN_QUEUED = 2
REMOTE_ROOT = os.environ.get("JURECA_TEAM_ROOT", "/scratch/team-folder").rstrip("/") + "/agents/r-unl-a3/"
"""Cluster folder of this team: a remote manifest may only point inside it."""
PARTIAL_SUFFIX = ".partial"
DATA_DIR_ENV = "HACKATHON_DATA_DIR"


class QueueError(HackathonError):
    """The queue refused a candidate (full for this team, name already used, bad manifest)."""


def queue_directory(data_dir: Path) -> Path:
    """`<data>/queue/unlearning`."""
    return data_dir / QUEUE_SUBDIR


def waiting_manifests(directory: Path) -> dict[str, dict[str, object]]:
    """Manifests of the candidates still in the queue (not yet submitted), by name."""
    manifests: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            manifests[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A manifest being written or removed by the daemon: ignore it for the count.
            continue
    return manifests


def own_waiting(directory: Path) -> list[str]:
    """Names of THIS team's candidates waiting in the queue."""
    return [
        name
        for name, manifest in waiting_manifests(directory).items()
        if manifest.get("team") == TEAM_NAME
    ]


def enqueue(
    source: Path,
    name: str,
    directory: Path,
    priority: int,
    expected: float | None,
    rationale: str,
) -> Path:
    """Validate `source`, copy it into the queue and write its manifest last.

    Raises:
        QueueError: name already queued or done, or this team already has `MAX_OWN_QUEUED` waiting.
        SubmissionInvalidError: the file breaks a server rule (nothing is copied).
    """
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / f"{name}.pt").exists() or (directory / DONE_SUBDIR / f"{name}.json").exists():
        raise QueueError(f"'{name}' is already queued or done")
    waiting = own_waiting(directory)
    if len(waiting) >= MAX_OWN_QUEUED:
        raise QueueError(f"{TEAM_NAME} already has {len(waiting)} candidates waiting: {waiting}")
    validate_submission(source)
    partial = directory / f"{name}.pt{PARTIAL_SUFFIX}"
    shutil.copyfile(source, partial)
    partial.replace(directory / f"{name}.pt")
    manifest = {
        "team": TEAM_NAME,
        "priority": priority,
        "expected": expected,
        "rationale": rationale,
    }
    target = directory / f"{name}.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def enqueue_remote(
    remote_file: str,
    name: str,
    directory: Path,
    priority: int,
    expected: float | None,
    rationale: str,
) -> Path:
    """Write the manifest of a candidate that stays ON JURECA (no 344 MB copy to the laptop).

    The daemon validates and submits `remote_file` from the cluster. The file was validated by
    `save_submission` when the worker wrote it; the daemon validates it again before sending.

    Raises:
        QueueError: name already queued or done, this team already has `MAX_OWN_QUEUED` waiting,
            or `remote_file` is not an absolute `.pt` path under the team's cluster folder.
    """
    if not remote_file.startswith(REMOTE_ROOT) or not remote_file.endswith(".pt"):
        raise QueueError(f"remote_file must be a .pt under {REMOTE_ROOT}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / f"{name}.pt").exists() or (directory / DONE_SUBDIR / f"{name}.json").exists():
        raise QueueError(f"'{name}' is already queued or done")
    if (directory / f"{name}.json").exists():
        raise QueueError(f"'{name}' is already queued")
    waiting = own_waiting(directory)
    if len(waiting) >= MAX_OWN_QUEUED:
        raise QueueError(f"{TEAM_NAME} already has {len(waiting)} candidates waiting: {waiting}")
    manifest = {
        "team": TEAM_NAME,
        "priority": priority,
        "expected": expected,
        "rationale": rationale,
        "remote_file": remote_file,
    }
    target = directory / f"{name}.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def read_done(directory: Path) -> dict[str, dict[str, object]]:
    """Public score records written by the daemon, by candidate name."""
    done_dir = directory / DONE_SUBDIR
    if not done_dir.is_dir():
        return {}
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(done_dir.glob("*.json"))
    }


def _public_score(record: dict[str, object]) -> float | None:
    result = record.get("result")
    if isinstance(result, dict):
        score = result.get("score")
        if isinstance(score, int | float):
            return float(score)
    return None


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="validate, copy and write the manifest")
    add.add_argument("file", type=Path)
    add.add_argument("--name", required=True)
    add.add_argument("--priority", type=int, required=True)
    add.add_argument("--expected", type=float, default=None)
    add.add_argument("--rationale", required=True)
    remote = sub.add_parser("add-remote", help="manifest only, the file stays on JURECA")
    remote.add_argument("remote_file")
    remote.add_argument("--name", required=True)
    remote.add_argument("--priority", type=int, required=True)
    remote.add_argument("--expected", type=float, default=None)
    remote.add_argument("--rationale", required=True)
    sub.add_parser("status", help="waiting candidates and public scores")
    args = parser.parse_args(argv)
    data_dir = os.environ.get(DATA_DIR_ENV)
    if not data_dir:
        raise QueueError(f"{DATA_DIR_ENV} must point to the hackathon data directory")
    directory = queue_directory(Path(data_dir))
    if args.command == "add-remote":
        path = enqueue_remote(
            args.remote_file, args.name, directory, args.priority, args.expected, args.rationale
        )
        print("queued (remote)", path)
        return
    if args.command == "add":
        path = enqueue(
            args.file, args.name, directory, args.priority, args.expected, args.rationale
        )
        print("queued", path)
        return
    for name, manifest in waiting_manifests(directory).items():
        print("waiting", name, manifest.get("team"), manifest.get("priority"))
    for name, record in read_done(directory).items():
        print("done", name, record.get("team"), _public_score(record))


if __name__ == "__main__":
    main()

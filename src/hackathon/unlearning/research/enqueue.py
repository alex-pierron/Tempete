"""Put validated candidates in the submission queue and read back the public scores.

The queue is the ONLY channel to the evaluation server for the research teams: a daemon submits
one candidate every ~10 minutes. Protocol (set by the coordinator):

* `<data>/queue/unlearning/<name>.pt` then, written LAST, `<name>.json` =
  {"team", "priority", "expected", "rationale"};
* the daemon writes the public score in `<data>/queue/unlearning/done/<name>.json`;
* at most `MAX_QUEUED_FILES` candidates (344 MB each) wait in the queue at a time.

This module never contacts the server and never reads the API key.

Usage:
    python -m hackathon.unlearning.research.enqueue add <file.pt> --name N --priority P \
        --expected 0.05 --rationale "..."
    python -m hackathon.unlearning.research.enqueue status
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from hackathon.common.config import load_settings
from hackathon.unlearning.validate import validate_submission

TEAM_NAME = "r-unl-a2"
QUEUE_SUBDIR = Path("queue") / "unlearning"
DONE_SUBDIR = "done"
MAX_QUEUED_FILES = 3
PARTIAL_SUFFIX = ".partial"


class QueueError(RuntimeError):
    """The queue refused a candidate (full, name clash, invalid file)."""


def queue_dir(data_dir: Path) -> Path:
    """Queue directory under the hackathon data directory."""
    return data_dir / QUEUE_SUBDIR


def waiting_names(directory: Path) -> list[str]:
    """Names of candidates whose manifest is in the queue (not yet submitted)."""
    return sorted(p.stem for p in directory.glob("*.json"))


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
        QueueError: name already queued or done, or the queue already holds `MAX_QUEUED_FILES`.
        SubmissionInvalidError: the file breaks a server rule (nothing is copied).
    """
    directory.mkdir(parents=True, exist_ok=True)
    done_manifest = directory / DONE_SUBDIR / f"{name}.json"
    if (directory / f"{name}.pt").exists() or done_manifest.exists():
        raise QueueError(f"'{name}' is already queued or done")
    if len(waiting_names(directory)) >= MAX_QUEUED_FILES:
        raise QueueError(f"queue holds {MAX_QUEUED_FILES} candidates already")
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


def read_done(directory: Path) -> dict[str, dict[str, object]]:
    """Public score records written by the daemon, by candidate name."""
    done_dir = directory / DONE_SUBDIR
    if not done_dir.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(done_dir.glob("*.json"))}


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
    sub.add_parser("status", help="waiting candidates and public scores")
    args = parser.parse_args(argv)
    directory = queue_dir(load_settings().data_dir)
    if args.command == "add":
        path = enqueue(args.file, args.name, directory, args.priority, args.expected, args.rationale)
        print("queued", path)
    else:
        print("waiting:", waiting_names(directory))
        for name, record in read_done(directory).items():
            print(name, json.dumps(record))


if __name__ == "__main__":
    main()

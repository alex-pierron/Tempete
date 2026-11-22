"""Tests of team A3's submission-queue access (the validator is replaced by a stub)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hackathon.common.errors import HackathonError
from hackathon.unlearning.research import a3_enqueue

FAKE_BYTES = b"fake-state-dict"


def _fake_candidate(tmp_path: Path) -> Path:
    path = tmp_path / "source.pt"
    path.write_bytes(FAKE_BYTES)
    return path


@pytest.fixture()
def queue_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(a3_enqueue, "validate_submission", lambda path: None)
    return tmp_path / "queue"


def test_enqueue_copies_the_file_then_writes_the_manifest(tmp_path: Path, queue_dir: Path) -> None:
    manifest = a3_enqueue.enqueue(_fake_candidate(tmp_path), "cand", queue_dir, 5, 0.06, "why")
    assert (queue_dir / "cand.pt").read_bytes() == FAKE_BYTES
    assert not list(queue_dir.glob("*.partial"))
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written == {"team": "r-unl-a3", "priority": 5, "expected": 0.06, "rationale": "why"}


def test_enqueue_caps_only_the_own_team(tmp_path: Path, queue_dir: Path) -> None:
    queue_dir.mkdir()
    for index in range(3):  # candidates of other teams do not count against A3
        other = queue_dir / f"other{index}.json"
        other.write_text(json.dumps({"team": "r-unl-c"}), encoding="utf-8")
    source = _fake_candidate(tmp_path)
    a3_enqueue.enqueue(source, "one", queue_dir, 1, None, "r")
    a3_enqueue.enqueue(source, "two", queue_dir, 1, None, "r")
    assert sorted(a3_enqueue.own_waiting(queue_dir)) == ["one", "two"]
    with pytest.raises(a3_enqueue.QueueError):
        a3_enqueue.enqueue(source, "three", queue_dir, 1, None, "r")


def test_enqueue_refuses_a_name_already_queued_or_done(tmp_path: Path, queue_dir: Path) -> None:
    source = _fake_candidate(tmp_path)
    a3_enqueue.enqueue(source, "one", queue_dir, 1, None, "r")
    with pytest.raises(a3_enqueue.QueueError):
        a3_enqueue.enqueue(source, "one", queue_dir, 1, None, "r")
    done = queue_dir / "done"
    done.mkdir()
    (done / "old.json").write_text(json.dumps({"result": {"score": 0.05}}), encoding="utf-8")
    with pytest.raises(a3_enqueue.QueueError):
        a3_enqueue.enqueue(source, "old", queue_dir, 1, None, "r")
    assert a3_enqueue.read_done(queue_dir)["old"]["result"] == {"score": 0.05}


def test_enqueue_remote_writes_only_a_manifest_and_respects_the_cap(queue_dir: Path) -> None:
    remote = a3_enqueue.REMOTE_ROOT + "data/unlearning/candidates/a3_g1/x.pt"
    manifest = a3_enqueue.enqueue_remote(remote, "one", queue_dir, 4, 0.1, "why")
    assert sorted(path.name for path in queue_dir.iterdir()) == ["one.json"]
    assert json.loads(manifest.read_text(encoding="utf-8"))["remote_file"] == remote
    a3_enqueue.enqueue_remote(remote, "two", queue_dir, 4, 0.1, "why")
    with pytest.raises(a3_enqueue.QueueError):
        a3_enqueue.enqueue_remote(remote, "three", queue_dir, 4, 0.1, "why")
    with pytest.raises(a3_enqueue.QueueError):
        a3_enqueue.enqueue_remote(remote, "one", queue_dir, 4, 0.1, "why")


@pytest.mark.parametrize(
    "remote",
    ["/etc/passwd.pt", "/scratch/other-team/x.pt", a3_enqueue.REMOTE_ROOT + "x.txt", "x.pt"],
)
def test_enqueue_remote_refuses_paths_outside_the_team_folder(queue_dir: Path, remote: str) -> None:
    with pytest.raises(a3_enqueue.QueueError):
        a3_enqueue.enqueue_remote(remote, "bad", queue_dir, 1, None, "r")
    assert not queue_dir.exists() or not list(queue_dir.glob("bad*"))


def test_enqueue_copies_nothing_when_the_validator_refuses(
    tmp_path: Path, queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path) -> None:
        raise HackathonError("bad state dict")

    monkeypatch.setattr(a3_enqueue, "validate_submission", refuse)
    with pytest.raises(HackathonError):
        a3_enqueue.enqueue(_fake_candidate(tmp_path), "bad", queue_dir, 1, None, "r")
    assert not list(queue_dir.glob("bad*"))

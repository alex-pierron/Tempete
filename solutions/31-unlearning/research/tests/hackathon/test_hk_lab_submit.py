"""Tests of `lab submit`: validate first, never send without --yes, record the public score."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from test_hk_lab_support import (
    FakeCluster,
    LabHarness,
    build_adapter,
    make_lab,
    make_settings,
    write_spec,
)

from hackathon.common.submit import SubmitResult, append_log, file_sha256
from hackathon.lab.errors import RegistryError, SubmitRefusedError
from hackathon.lab.registry import RunRecord
from hackathon.lab.runner import run_local
from hackathon.lab.spec import load_spec
from hackathon.lab.submit import submit_run
from hackathon.lab.tasks import SubmitOutcome, parse_submit_output

SPEC = """
id = "plain"
task = "unlearning"
idea = "submit me"
[[steps]]
args = ["run", "--out", "{out}/candidate.pt"]
[expect]
submission = "candidate.pt"
"""

pytest_plugins = ("test_hk_lab_support",)  # the fake_cluster fixture


def done_run(harness: LabHarness, text: str = SPEC) -> RunRecord:
    write_spec(harness, "plain", text=text)
    spec = load_spec("plain", harness.ctx.adapters, harness.ctx.experiments_dir)
    (record,) = run_local(harness.ctx, spec, harness.ctx.adapters[spec.task])
    assert record.status == "done"
    return record


def task_calls(cluster: FakeCluster) -> list[list[str]]:
    log = cluster.state_dir / "task_calls.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []


def subcommands(cluster: FakeCluster) -> list[str]:
    return [call[0] for call in task_calls(cluster)]


@pytest.fixture
def lab(tmp_path: Path, fake_cluster: FakeCluster) -> LabHarness:
    adapter = build_adapter(fake_cluster.task_script)
    return make_lab(tmp_path, {adapter.name: adapter})


# ---- the explicit --yes gate ---------------------------------------------------------------


def test_without_yes_the_file_is_validated_and_nothing_is_submitted(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    record = done_run(lab)
    report = submit_run(lab.ctx, record.run_id, yes=False)
    assert report.dry_run and report.outcome is None
    assert subcommands(fake_cluster) == ["run", "validate"], "submit must never be invoked"
    assert any("[dry run]" in line and "Re-run with --yes" in line for line in lab.out)
    assert lab.ctx.registry.get(record.run_id).public_score is None
    assert lab.ctx.registry.get(record.run_id).public_submitted_at is None


def test_with_yes_the_public_score_lands_in_the_registry(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    record = done_run(lab)
    report = submit_run(lab.ctx, record.run_id, yes=True)
    assert subcommands(fake_cluster) == ["run", "validate", "submit"]
    submit_call = task_calls(fake_cluster)[-1]
    assert submit_call[-1] == "--yes" and submit_call[1].endswith("candidate.pt")
    assert report.outcome == SubmitOutcome(True, 0.42, 0.3, "nice")
    saved = lab.ctx.registry.get(record.run_id)
    assert saved.public_score == 0.42 and saved.previous_best == 0.3
    assert saved.public_submitted_at is not None and saved.status == "done"


def test_validate_runs_before_submit_and_a_rejection_blocks_the_submission(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    record = done_run(lab)
    path = lab.repo / str(record.submission_file)
    path.write_text("this is invalid")
    updated = lab.ctx.registry.update(record, submission_sha256=file_sha256(path))
    with pytest.raises(SubmitRefusedError, match="validate rejected"):
        submit_run(lab.ctx, updated.run_id, yes=True)
    assert "submit" not in subcommands(fake_cluster)


# ---- run state and file checks -------------------------------------------------------------


def test_only_done_runs_with_an_unchanged_local_file_can_be_submitted(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    record = done_run(lab)
    path = lab.repo / str(record.submission_file)
    original = path.read_text()
    path.write_text(original + " tampered")
    with pytest.raises(SubmitRefusedError, match="changed since the run recorded it"):
        submit_run(lab.ctx, record.run_id, yes=True)
    path.write_text(original)
    path.unlink()
    with pytest.raises(SubmitRefusedError, match="does not exist on this machine"):
        submit_run(lab.ctx, record.run_id, yes=True)
    for status in ("queued", "running", "finished", "failed"):
        lab.ctx.registry.update(record, status=status)
        with pytest.raises(SubmitRefusedError, match=f"is '{status}'"):
            submit_run(lab.ctx, record.run_id, yes=True)
    assert "submit" not in subcommands(fake_cluster)


def test_a_file_outside_runs_is_refused_even_if_the_registry_says_so(
    lab: LabHarness,
    fake_cluster: FakeCluster,
    tmp_path: Path,
) -> None:
    """The registry is shared through git: a foreign line must not make us upload any file."""
    record = done_run(lab)
    outsider = tmp_path / "secret.txt"
    outsider.write_text("not ours")
    lab.ctx.registry.update(record, submission_file=str(outsider), submission_sha256=None)
    with pytest.raises(SubmitRefusedError, match="only submits files it produced"):
        submit_run(lab.ctx, record.run_id, yes=True)
    lab.ctx.registry.update(record, submission_file=str(lab.repo / "runs" / ".." / "x"))
    with pytest.raises(SubmitRefusedError):
        submit_run(lab.ctx, record.run_id, yes=True)
    assert "submit" not in subcommands(fake_cluster)


def test_a_run_without_a_submission_file_is_refused(lab: LabHarness) -> None:
    record = done_run(lab)
    lab.ctx.registry.update(record, submission_file=None)
    with pytest.raises(SubmitRefusedError, match="no submission file"):
        submit_run(lab.ctx, record.run_id, yes=True)


def test_unknown_run_is_a_registry_error(lab: LabHarness) -> None:
    with pytest.raises(RegistryError, match="no run"):
        submit_run(lab.ctx, "ghost", yes=True)


# ---- guards from the shared registry -------------------------------------------------------


def test_a_run_is_not_submitted_twice_unless_asked(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    record = done_run(lab)
    submit_run(lab.ctx, record.run_id, yes=True)
    with pytest.raises(SubmitRefusedError, match="already submitted"):
        submit_run(lab.ctx, record.run_id, yes=True)
    assert subcommands(fake_cluster).count("submit") == 1
    submit_run(lab.ctx, record.run_id, yes=True, again=True)
    assert subcommands(fake_cluster).count("submit") == 2


def test_dry_run_announces_what_a_real_submission_would_be_refused_for(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    record = done_run(lab)
    submit_run(lab.ctx, record.run_id, yes=True)
    report = submit_run(lab.ctx, record.run_id, yes=False)
    assert report.dry_run and any("already submitted" in w for w in report.warnings)
    assert any("would be refused" in line for line in lab.err)


def test_the_server_cooldown_is_enforced_from_the_registry_across_teammates(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    """A teammate's submission, pulled through git, blocks ours for 30 minutes (model equality)."""
    adapter = build_adapter(
        fake_cluster.task_script, name="unlearning", server_task_id="33-model-equality"
    )
    harness = make_lab(tmp_path, {adapter.name: adapter})
    mine = done_run(harness)
    base = harness.ctx.registry.latest()[mine.run_id]
    teammate = RunRecord(
        run_id="their-run", group_id="their-run", task="unlearning", idea="theirs", spec_id="t",
        variant="v1", sha="x", where="local", status="done", created_at="2026-11-21T09:00:00",
        updated_at="2026-11-21T09:00:00+00:00",
        public_submitted_at=(harness.clock() - timedelta(minutes=10)).isoformat(),
    )  # fmt: skip
    harness.ctx.registry.append(teammate)
    with pytest.raises(SubmitRefusedError, match=r"cooldown, wait (19|20|21) more minute"):
        submit_run(harness.ctx, base.run_id, yes=True)
    assert "submit" not in subcommands(fake_cluster)
    harness.clock.advance(minutes=25)
    report = submit_run(harness.ctx, base.run_id, yes=True)
    assert report.outcome is not None and report.outcome.ok


def test_tasks_without_a_documented_cooldown_are_not_blocked(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    record = done_run(lab)
    other = RunRecord(
        run_id="o", group_id="o", task="unlearning", idea="o", spec_id="o", variant="v1", sha="x",
        where="local", status="done", created_at="2026-11-21T09:00:00",
        updated_at="2026-11-21T09:00:00+00:00",
        public_submitted_at=lab.clock().isoformat(),
    )  # fmt: skip
    lab.ctx.registry.append(other)
    assert submit_run(lab.ctx, record.run_id, yes=True).outcome is not None


# ---- reading the server reply --------------------------------------------------------------


def test_model_equality_style_output_line_is_parsed(
    lab: LabHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_TASK_SUBMIT_MODE", "line")
    record = done_run(lab)
    report = submit_run(lab.ctx, record.run_id, yes=True)
    assert report.outcome == SubmitOutcome(True, 0.31, 0.2, "better")
    assert lab.ctx.registry.get(record.run_id).public_score == 0.31


def test_a_server_rejection_is_recorded_and_still_counts_for_the_cooldown(
    lab: LabHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_TASK_SUBMIT_MODE", "reject")
    record = done_run(lab)
    report = submit_run(lab.ctx, record.run_id, yes=True)
    assert report.outcome is not None and not report.outcome.ok
    saved = lab.ctx.registry.get(record.run_id)
    assert saved.public_score is None and saved.public_submitted_at is not None
    assert saved.submit_message == "bad shape"


def test_a_local_refusal_by_the_task_cli_starts_no_cooldown(
    lab: LabHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_TASK_SUBMIT_MODE", "local_refusal")
    record = done_run(lab)
    report = submit_run(lab.ctx, record.run_id, yes=True)
    assert report.outcome is not None and not report.outcome.ok and not report.outcome.replied
    saved = lab.ctx.registry.get(record.run_id)
    assert saved.public_submitted_at is None, "nothing reached the server: no attempt to record"
    assert "wait 12 more minute" in saved.submit_message


def test_an_unreadable_reply_falls_back_to_the_shared_submission_journal(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_TASK_SUBMIT_MODE", "silent")
    adapter = build_adapter(fake_cluster.task_script)
    harness = make_lab(tmp_path, {adapter.name: adapter})
    record = done_run(harness)
    append_log(
        make_settings(tmp_path),
        SubmitResult(
            task_id="31-unlearning", file="x", sha256=str(record.submission_sha256),
            submitted_at=harness.clock().isoformat(), ok=True, http_status=200, score=0.77,
            previous_best=0.5, improved=True, message="",
        ),
    )  # fmt: skip
    report = submit_run(harness.ctx, record.run_id, yes=True)
    assert report.outcome is not None and report.outcome.score == 0.77
    assert harness.ctx.registry.get(record.run_id).public_score == 0.77


# ---- parser --------------------------------------------------------------------------------


def test_parse_submit_output_formats() -> None:
    pretty = json.dumps({"ok": True, "score": 0.5, "previous_best": 0.4, "message": "m"}, indent=2)
    assert parse_submit_output(pretty, 0) == SubmitOutcome(True, 0.5, 0.4, "m")
    noisy = "uploading...\n" + pretty + "\n(12.0 s)"
    assert parse_submit_output(noisy, 0).score == 0.5
    line = "ok=False score=None previous_best=None improved=None message='bad'"
    assert parse_submit_output(line, 1) == SubmitOutcome(False, None, None, "bad")
    assert parse_submit_output("ok=True score=1e-3 previous_best=-0.5", 0).score == 0.001
    detail = json.dumps({"status": "error", "detail": "nope", "ok": False})
    assert parse_submit_output(detail, 1) == SubmitOutcome(False, None, None, "nope")
    nothing = parse_submit_output("", 1)
    assert nothing.ok is False and nothing.replied is False and nothing.score is None
    assert parse_submit_output("submitted", 0).replied is True

"""`lab submit RUN [--yes]`: validate a finished run's file, then (only with --yes) submit it.

The lab does not talk to the competition server itself. It runs the TASK's own `validate` and
`submit --yes` (so the format checks, the journal `data/hackathon/submissions/log.jsonl` and the
server cooldown stay the single implementation of the kit), reads the public score back from the
output and writes it into the shared registry. Two extra guards come from the registry, which the
whole team shares through git: a run is not submitted twice, and a task whose last submission by
ANYONE is inside the server cooldown is not submitted either (the local journal only knows the
submissions made from this laptop).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from hackathon.common.errors import HackathonError
from hackathon.common.submit import COOLDOWN_BY_TASK_S, file_sha256, read_log
from hackathon.lab.constants import STATUS_DONE
from hackathon.lab.context import LabContext
from hackathon.lab.errors import RunError, SubmitRefusedError
from hackathon.lab.registry import RunRecord, utc_now_iso
from hackathon.lab.runner import task_environment
from hackathon.lab.tasks import SubmitOutcome, TaskAdapter, parse_submit_output, resolve_adapter

VALIDATE_TIMEOUT_S = 600.0
SUBMIT_TIMEOUT_S = 900.0
"""The server may take ~500 s to score (hackathon.common.submit uses 600 s)."""
TAIL_LINES = 8


@dataclass(frozen=True)
class SubmitReport:
    """Outcome of `submit_run`.

    Attributes:
        record: The run, with the public score once submitted.
        dry_run: True when nothing was sent (no --yes).
        warnings: Guard findings that would have refused a real submission.
        outcome: Parsed server reply, None on a dry run.
    """

    record: RunRecord
    dry_run: bool
    warnings: tuple[str, ...]
    outcome: SubmitOutcome | None


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-TAIL_LINES:])


def _submission_path(ctx: LabContext, record: RunRecord) -> Path:
    """The run's file, which must exist, sit under `runs/` and match its recorded hash."""
    if record.status != STATUS_DONE:
        raise SubmitRefusedError(
            f"{record.run_id} is '{record.status}': only a 'done' run can be submitted "
            "(a JURECA run needs `jureca fetch` first)."
        )
    if not record.submission_file:
        raise SubmitRefusedError(f"{record.run_id} has no submission file (spec has no [expect]).")
    path = Path(record.submission_file)
    path = path if path.is_absolute() else ctx.repo_root / path
    runs_root = ctx.runs_dir.resolve()
    if runs_root not in path.resolve().parents:
        raise SubmitRefusedError(
            f"{path} is outside {runs_root}: the lab only submits files it produced."
        )
    if not path.is_file():
        raise SubmitRefusedError(f"{path} does not exist on this machine (fetch the run).")
    if record.submission_sha256 and file_sha256(path) != record.submission_sha256:
        raise SubmitRefusedError(f"{path} changed since the run recorded it (sha256 differs).")
    return path


def _guards(ctx: LabContext, adapter: TaskAdapter, record: RunRecord, *, again: bool) -> list[str]:
    """Reasons not to submit now: already submitted, or the task cooldown is running."""
    findings: list[str] = []
    if record.public_submitted_at and not again:
        findings.append(
            f"{record.run_id} was already submitted at {record.public_submitted_at} "
            f"(public score {record.public_score}); pass --again to resubmit the same run."
        )
    try:
        server_id = adapter.resolve_server_task_id(ctx.settings_loader())
    except HackathonError:
        server_id = adapter.server_task_id
    cooldown = COOLDOWN_BY_TASK_S.get(server_id) if server_id else None
    if cooldown is not None:
        stamps = [
            r.public_submitted_at
            for r in ctx.registry.latest().values()
            if r.task == record.task and r.public_submitted_at
        ]
        if stamps:
            last = datetime.fromisoformat(max(stamps))
            wait = last + timedelta(seconds=cooldown) - ctx.clock()
            if wait.total_seconds() > 0:
                findings.append(
                    f"{adapter.name}: a submission was made at {last.isoformat()} (registry); "
                    f"server cooldown, wait {int(wait.total_seconds() // 60) + 1} more minute(s). "
                    "A failed attempt counts too."
                )
    return findings


def _run_task(
    ctx: LabContext, adapter: TaskAdapter, args: tuple[str, ...], timeout: float
) -> subprocess.CompletedProcess[str]:
    argv = [ctx.python, *adapter.entrypoint, *args]
    try:
        return subprocess.run(
            argv,
            cwd=ctx.repo_root,
            env=task_environment(ctx.repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunError(f"could not run {adapter.name} {args[0]}: {exc}") from exc


def _score_from_journal(
    ctx: LabContext, adapter: TaskAdapter, record: RunRecord
) -> tuple[float | None, float | None]:
    """Fallback when the task output carries no readable score: the shared submission journal."""
    try:
        settings = ctx.settings_loader()
        server_id = adapter.resolve_server_task_id(settings)
        attempts = [
            a
            for a in read_log(settings)
            if a.task_id == server_id and a.sha256 == record.submission_sha256
        ]
    except (HackathonError, ValueError, TypeError, OSError):
        return None, None
    if not attempts:
        return None, None
    return attempts[-1].score, attempts[-1].previous_best


def submit_run(ctx: LabContext, reference: str, *, yes: bool, again: bool = False) -> SubmitReport:
    """Validate the run's submission file with the task CLI; with `yes`, submit it and record it.

    Args:
        ctx: Lab context.
        reference: Run id (or a unique prefix) from the registry.
        yes: Really send the file. Without it nothing leaves the machine.
        again: Allow resubmitting a run that already has a public score.

    Returns:
        A SubmitReport; `outcome` is None on a dry run.

    Raises:
        RegistryError: unknown or ambiguous run.
        SubmitRefusedError: wrong state, missing/modified file, a guard fired (with --yes).
        RunError: the task CLI could not run.
    """
    record = ctx.registry.get(reference)
    path = _submission_path(ctx, record)
    adapter = resolve_adapter(record.task, ctx.adapters)
    checked = _run_task(ctx, adapter, adapter.validate_args(path), VALIDATE_TIMEOUT_S)
    if checked.stdout.strip():
        ctx.out(checked.stdout.rstrip())
    if checked.returncode != 0:
        raise SubmitRefusedError(
            f"{adapter.name} validate rejected {path} (exit {checked.returncode}): "
            f"{_tail(checked.stderr or checked.stdout)}"
        )
    findings = _guards(ctx, adapter, record, again=again)
    if not yes:
        for finding in findings:
            ctx.err(f"would be refused: {finding}")
        ctx.out(
            f"[dry run] {record.run_id}: {path.name} is valid. Nothing sent. "
            "Re-run with --yes to submit it."
        )
        return SubmitReport(record, True, tuple(findings), None)
    if findings:
        raise SubmitRefusedError(findings[0])
    sent = _run_task(ctx, adapter, adapter.submit_args(path), SUBMIT_TIMEOUT_S)
    if sent.stdout.strip():
        ctx.out(sent.stdout.rstrip())
    if sent.stderr.strip():
        ctx.err(sent.stderr.rstrip())
    outcome = parse_submit_output(sent.stdout, sent.returncode)
    if sent.returncode != 0:
        outcome = SubmitOutcome(
            False, outcome.score, outcome.previous_best, outcome.message, outcome.replied
        )
    score, previous = outcome.score, outcome.previous_best
    if outcome.ok and score is None:
        score, previous = _score_from_journal(ctx, adapter, record)
    message = outcome.message or ("" if outcome.ok else _tail(sent.stderr))
    if not outcome.replied:
        # The task CLI refused before contacting the server: keep the reason, start no cooldown.
        updated = ctx.registry.update(record, submit_message=message)
    else:
        updated = ctx.registry.update(
            record,
            public_score=score,
            previous_best=previous,
            public_submitted_at=utc_now_iso(ctx.clock()),
            submit_message=message,
        )
    return SubmitReport(
        updated, False, (), SubmitOutcome(outcome.ok, score, previous, message, outcome.replied)
    )

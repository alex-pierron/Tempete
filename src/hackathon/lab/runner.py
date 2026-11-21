"""Planning, local execution and output collection of experiment runs.

`plan_runs` and `build_step_argv` are shared with the JURECA path (`remote.py`): the same spec
gives the same command lines on both sides, only the interpreter and the output directory differ.
`collect_outputs` turns a finished output directory (local, or rsynced back from JURECA) into
metrics, a submission file and a verdict.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from hackathon.common.submit import file_sha256
from hackathon.lab.constants import (
    RUN_ID_TIMESTAMP_FORMAT,
    RUN_META_FILENAME,
    SPEC_SNAPSHOT_FILENAME,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_RUNNING,
    STDERR_LOG_FILENAME,
    STDOUT_LOG_FILENAME,
    WHERE_JURECA,
    WHERE_LOCAL,
)
from hackathon.lab.context import LabContext, LineSink
from hackathon.lab.errors import RunError
from hackathon.lab.registry import JsonValue, RunRecord, utc_now_iso
from hackathon.lab.spec import (
    ExpectSpec,
    ExperimentSpec,
    StepSpec,
    Variant,
    param_text,
    render_args,
)
from hackathon.lab.tasks import TaskAdapter

GIT_TIMEOUT_S = 15.0
UNKNOWN_SHA = "unknown"
INTERRUPT_GRACE_S = 10.0
ECHO_PREFIX_OUT = "  | "


@dataclass(frozen=True)
class PlannedRun:
    """A run about to be created: one variant of one launch."""

    run_id: str
    group_id: str
    variant: Variant


@dataclass(frozen=True)
class Collected:
    """Result of reading a finished output directory.

    Attributes:
        metrics: Numeric metrics (flattened JSON leaves plus regex captures).
        submission: Path of the expected submission file when it exists.
        submission_sha256: Its hash.
        problems: Fatal findings (an expected file is missing or unreadable).
        warnings: Non-fatal findings (a metric could not be extracted).
    """

    metrics: dict[str, float] = field(default_factory=dict)
    submission: Path | None = None
    submission_sha256: str | None = None
    problems: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def git_state(root: Path) -> tuple[str, bool]:
    """(HEAD sha, working tree dirty) of the repo at `root`; ("unknown", False) outside git."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if head.returncode != 0:
            return UNKNOWN_SHA, False
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_SHA, False
    return head.stdout.strip(), bool(status.stdout.strip())


def plan_runs(spec: ExperimentSpec, adapter: TaskAdapter, now: datetime) -> list[PlannedRun]:
    """Run ids for every variant of one launch: `<timestamp>-<tag>-<spec id>[-vN]`."""
    group_id = f"{now.strftime(RUN_ID_TIMESTAMP_FORMAT)}-{adapter.short}-{spec.id}"
    variants = spec.variants()
    if len(variants) == 1 and not spec.sweep:
        return [PlannedRun(run_id=group_id, group_id=group_id, variant=variants[0])]
    return [
        PlannedRun(run_id=f"{group_id}-{variant.name}", group_id=group_id, variant=variant)
        for variant in variants
    ]


def placeholder_values(
    planned: PlannedRun, out_dir: str, *, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Values of every placeholder for one run (`extra` adds the JURECA-only ones)."""
    values = {
        "out": out_dir,
        "run_id": planned.run_id,
        "variant": planned.variant.name,
        "variant_index": str(planned.variant.index),
    }
    values.update({key: param_text(value) for key, value in planned.variant.params.items()})
    values.update(extra or {})
    return values


def build_step_argv(
    adapter: TaskAdapter, step: StepSpec, values: Mapping[str, str], *, where: str
) -> tuple[str, ...]:
    """Arguments of one step after the interpreter: entry point, global args, step args.

    Raises:
        SpecError: a placeholder cannot be resolved for `where`.
    """
    on_jureca = where == WHERE_JURECA
    global_args = (
        render_args(adapter.remote_global_args, values, allow_jureca_only=True) if on_jureca else ()
    )
    args = render_args(step.args, values, allow_jureca_only=on_jureca)
    extra_args = render_args(
        step.args_jureca if on_jureca else step.args_local, values, allow_jureca_only=on_jureca
    )
    return (*adapter.entrypoint, *global_args, *args, *extra_args)


def _flatten_numbers(raw: object, prefix: str, into: dict[str, float]) -> None:
    """Numeric leaves of nested dicts become `a.b.c` metrics; lists, strings, NaN are skipped."""
    if isinstance(raw, dict):
        for key, value in raw.items():
            _flatten_numbers(value, f"{prefix}.{key}" if prefix else str(key), into)
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
        into[prefix] = float(raw)


def collect_outputs(out_dir: Path, expect: ExpectSpec) -> Collected:
    """Read what a finished run left in `out_dir` according to the spec's `[expect]`.

    A missing expected file is a problem (the run counts as failed); a metric that cannot be
    extracted is only a warning, so a good submission file is not thrown away over a typo in a
    metric name.
    """
    problems: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    if expect.metrics_file:
        metrics_path = out_dir / expect.metrics_file
        if not metrics_path.is_file():
            problems.append(f"expected metrics file not found: {expect.metrics_file}")
        else:
            try:
                _flatten_numbers(json.loads(metrics_path.read_text(encoding="utf-8")), "", metrics)
            except ValueError as exc:
                problems.append(f"metrics file {expect.metrics_file} is not valid JSON: {exc}")
    if expect.metrics_regex:
        logs = "\n".join(
            (out_dir / name).read_text(encoding="utf-8", errors="replace")
            for name in (STDOUT_LOG_FILENAME, STDERR_LOG_FILENAME)
            if (out_dir / name).is_file()
        )
        for name, pattern in expect.metrics_regex.items():
            matches = list(re.finditer(pattern, logs))
            try:
                metrics[name] = float(matches[-1].group(1))
            except (IndexError, ValueError):
                warnings.append(f"metric '{name}': no numeric match of /{pattern}/ in the logs")
    if expect.primary_metric and expect.primary_metric not in metrics:
        warnings.append(
            f"primary metric '{expect.primary_metric}' not found among {sorted(metrics)}"
        )
    submission: Path | None = None
    digest: str | None = None
    if expect.submission:
        candidate = out_dir / expect.submission
        if candidate.is_file():
            submission, digest = candidate, file_sha256(candidate)
        else:
            problems.append(f"expected submission file not found: {expect.submission}")
    return Collected(metrics, submission, digest, tuple(problems), tuple(warnings))


def make_run_dir(path: Path) -> None:
    """Create a fresh run directory; an existing one means two launches in the same second.

    Raises:
        RunError: the directory already exists (never overwrite the logs of another run).
    """
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunError(f"run directory already exists: {path} (wait a second and retry)") from exc


def display_path(path: Path, root: Path) -> str:
    """`path` relative to the repository when it is inside it, absolute otherwise."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def finalize_record(
    ctx: LabContext,
    record: RunRecord,
    spec: ExperimentSpec,
    out_dir: Path,
    *,
    exit_code: int,
    started_at: str | None,
    finished_at: str | None,
    duration_s: float | None,
) -> RunRecord:
    """Collect the outputs of a finished run and append its final snapshot (done or failed)."""
    collected = collect_outputs(out_dir, spec.expect)
    failure = ""
    if exit_code != 0:
        failure = f"a step exited with code {exit_code}; see {STDERR_LOG_FILENAME}"
    elif collected.problems:
        failure = "; ".join(collected.problems)
    primary = spec.expect.primary_metric
    note = "; ".join(part for part in (failure, *collected.warnings) if part)
    return ctx.registry.update(
        record,
        status=STATUS_FAILED if failure else STATUS_DONE,
        exit_code=exit_code,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
        metrics=collected.metrics,
        primary_metric=primary,
        primary_value=collected.metrics.get(primary) if primary else None,
        higher_is_better=spec.expect.higher_is_better,
        submission_file=display_path(collected.submission, ctx.repo_root)
        if collected.submission
        else None,
        submission_sha256=collected.submission_sha256,
        note=note,
    )


class Echo:
    """A line sink that tolerates concurrent writers (two reader threads share the terminal)."""

    def __init__(self, sink: LineSink, prefix: str = "") -> None:
        self._sink = sink
        self._prefix = prefix
        self._lock = threading.Lock()

    def __call__(self, line: str) -> None:
        with self._lock:
            self._sink(self._prefix + line)


def _pump(stream: TextIO, log: TextIO, echo: Echo) -> None:
    """Copy `stream` to the log file and to the terminal, line by line."""
    for line in stream:
        log.write(line)
        log.flush()
        echo(line.rstrip("\n"))


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    echo_out: Echo,
    echo_err: Echo,
) -> int:
    """Run `argv` (no shell), teeing stdout/stderr into the log files and the terminal.

    Raises:
        RunError: the executable cannot be started.
        KeyboardInterrupt: re-raised after the child was terminated.
    """
    with (
        stdout_path.open("a", encoding="utf-8") as out_log,
        stderr_path.open("a", encoding="utf-8") as err_log,
    ):
        out_log.write(f"+ {shlex.join(argv)}\n")
        out_log.flush()
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise RunError(f"cannot start {argv[0]}: {exc}") from exc
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(target=_pump, args=(process.stdout, out_log, echo_out), daemon=True),
            threading.Thread(target=_pump, args=(process.stderr, err_log, echo_err), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=INTERRUPT_GRACE_S)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        finally:
            for thread in threads:
                thread.join()
            process.stdout.close()
            process.stderr.close()
        return code


def task_environment(repo_root: Path) -> dict[str, str]:
    """Environment of a task CLI: the repo `src/` first on PYTHONPATH, unbuffered output."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    src = str(repo_root / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    env["PYTHONUNBUFFERED"] = "1"
    return env


def write_run_meta(
    out_dir: Path,
    *,
    record: RunRecord,
    spec: ExperimentSpec,
    argvs: Sequence[Sequence[str]],
    where: str,
) -> None:
    """`run.json`: everything needed to reproduce the run by hand (no secrets, no environment)."""
    meta: dict[str, object] = {
        "run_id": record.run_id,
        "group_id": record.group_id,
        "task": record.task,
        "spec_id": spec.id,
        "idea": spec.idea,
        "hypothesis": spec.hypothesis,
        "where": where,
        "sha": record.sha,
        "dirty": record.dirty,
        "params": dict(record.params),
        "commands": [shlex.join(argv) for argv in argvs],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "created_at": record.created_at,
    }
    (out_dir / RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def new_record(
    ctx: LabContext,
    spec: ExperimentSpec,
    planned: PlannedRun,
    *,
    sha: str,
    dirty: bool,
    where: str,
    status: str,
) -> RunRecord:
    """The first snapshot of a run."""
    stamp = utc_now_iso(ctx.clock())
    params: dict[str, JsonValue] = dict(planned.variant.params)
    return RunRecord(
        run_id=planned.run_id,
        group_id=planned.group_id,
        task=spec.task,
        idea=spec.idea,
        spec_id=spec.id,
        variant=planned.variant.name,
        sha=sha,
        dirty=dirty,
        where=where,
        status=status,
        created_at=stamp,
        updated_at=stamp,
        params=params,
        primary_metric=spec.expect.primary_metric,
        higher_is_better=spec.expect.higher_is_better,
    )


def run_local(ctx: LabContext, spec: ExperimentSpec, adapter: TaskAdapter) -> list[RunRecord]:
    """Execute every variant of `spec` on this machine, one after the other.

    Each variant gets `runs/<task>/<run_id>/` (spec snapshot, run.json, stdout.log, stderr.log,
    its outputs) and a registry line, written as `running` first so an interrupted run leaves a
    trace, then replaced by `done` or `failed`. A failing step stops its variant, not the others.

    Returns:
        The final record of each variant.

    Raises:
        RunError: a task executable cannot be started.
        SpecError: a placeholder cannot be resolved.
        KeyboardInterrupt: after marking the interrupted run as failed.
    """
    sha, dirty = git_state(ctx.repo_root)
    now = ctx.clock()
    plan = plan_runs(spec, adapter, now)
    env = task_environment(ctx.repo_root)
    echo_out = Echo(ctx.out, ECHO_PREFIX_OUT)
    echo_err = Echo(ctx.err, ECHO_PREFIX_OUT)
    finals: list[RunRecord] = []
    for planned in plan:
        out_dir = ctx.run_dir(spec.task, planned.run_id)
        make_run_dir(out_dir)
        (out_dir / SPEC_SNAPSHOT_FILENAME).write_text(spec.source_text, encoding="utf-8")
        values = placeholder_values(planned, str(out_dir))
        argvs = [
            [ctx.python, *build_step_argv(adapter, step, values, where=WHERE_LOCAL)]
            for step in spec.steps
        ]
        record = new_record(
            ctx, spec, planned, sha=sha, dirty=dirty, where=WHERE_LOCAL, status=STATUS_RUNNING
        )
        started_at = utc_now_iso(ctx.clock())
        record = ctx.registry.update(record, started_at=started_at)
        write_run_meta(out_dir, record=record, spec=spec, argvs=argvs, where=WHERE_LOCAL)
        ctx.out(f"[{planned.run_id}] {len(argvs)} step(s) {planned.variant.label}".rstrip())
        started = time.monotonic()
        exit_code = 0
        try:
            for step, argv in zip(spec.steps, argvs, strict=True):
                ctx.out(f"[{planned.run_id}] step '{step.name}'")
                exit_code = run_command(
                    argv,
                    cwd=ctx.repo_root,
                    env=env,
                    stdout_path=out_dir / STDOUT_LOG_FILENAME,
                    stderr_path=out_dir / STDERR_LOG_FILENAME,
                    echo_out=echo_out,
                    echo_err=echo_err,
                )
                if exit_code != 0:
                    break
        except KeyboardInterrupt:
            ctx.registry.update(
                record,
                status=STATUS_FAILED,
                exit_code=-1,
                note="interrupted (Ctrl-C)",
                finished_at=utc_now_iso(ctx.clock()),
                duration_s=time.monotonic() - started,
            )
            raise
        except RunError as exc:
            ctx.registry.update(
                record,
                status=STATUS_FAILED,
                exit_code=-1,
                note=str(exc),
                finished_at=utc_now_iso(ctx.clock()),
                duration_s=time.monotonic() - started,
            )
            raise
        final = finalize_record(
            ctx,
            record,
            spec,
            out_dir,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=utc_now_iso(ctx.clock()),
            duration_s=time.monotonic() - started,
        )
        ctx.out(f"[{final.run_id}] {final.status}" + (f": {final.note}" if final.note else ""))
        finals.append(final)
    return finals

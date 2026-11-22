"""Command line of the experiment lab: `python -m hackathon.lab <command>`.

    new --task T --idea NAME     create experiments/<task>/<id>.toml and <id>.md from the templates
    run SPEC [--where W]         run a spec locally, or (jureca) sync + queue it on the cluster
    submit RUN [--yes]           validate a done run's file; with --yes submit it and record the score
    board [--task T]             terminal table + runs/board.html
    jureca login|sync|submit|status|fetch   cluster commands (everything goes through `ssh juelich`)

Exit codes: 0 success, 1 expected failure (bad spec, failed run, remote error), 2 refused because
the mandatory confirmation (--yes) is missing (same convention as the model-equality kit),
130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from hackathon.common.errors import HackathonError
from hackathon.lab.board import write_board
from hackathon.lab.constants import STATUS_FAILED, WHERE_CHOICES, WHERE_JURECA, WHERE_LOCAL
from hackathon.lab.context import LabContext
from hackathon.lab.errors import LabError
from hackathon.lab.registry import RunRecord
from hackathon.lab.remote import (
    Remote,
    fetch_run,
    load_jureca_config,
    refresh_status,
    submit_job,
)
from hackathon.lab.runner import run_local
from hackathon.lab.scaffold import create_experiment
from hackathon.lab.spec import ExperimentSpec, load_spec
from hackathon.lab.submit import submit_run
from hackathon.lab.tasks import resolve_adapter

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_REFUSED = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, one sub-parser per command."""
    parser = argparse.ArgumentParser(
        prog="python -m hackathon.lab",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="create a spec and an idea sheet")
    new.add_argument("--task", required=True, help="unlearning | model_equality | wmloc")
    new.add_argument("--idea", required=True, help="short name of the idea (becomes the id)")

    run = commands.add_parser("run", help="run a spec (path or id)")
    run.add_argument("spec")
    run.add_argument("--where", choices=WHERE_CHOICES, default=None, help="override the spec")
    run.add_argument("--dry-run", action="store_true", help="jureca: render the job, send nothing")
    run.add_argument("--no-sync", action="store_true", help="jureca: skip the rsync of the code")

    submit = commands.add_parser("submit", help="validate (and with --yes submit) a done run")
    submit.add_argument("run_id")
    submit.add_argument("--yes", action="store_true", help="really send the file (public score)")
    submit.add_argument("--again", action="store_true", help="resubmit an already scored run")

    board = commands.add_parser("board", help="show the registry and write runs/board.html")
    board.add_argument("--task", default=None)

    jureca = commands.add_parser("jureca", help="cluster commands")
    sub = jureca.add_subparsers(dest="jureca_command", required=True)
    sub.add_parser("login", help="open the persistent ssh session (TOTP code once)")
    sync = sub.add_parser("sync", help="rsync the repository to the team folder")
    sync.add_argument("--dry-run", action="store_true", help="list what would be sent")
    jsubmit = sub.add_parser("submit", help="queue a spec as one Slurm job (variants = GPUs)")
    jsubmit.add_argument("spec")
    jsubmit.add_argument("--sync", action="store_true", help="rsync the code first")
    jsubmit.add_argument("--dry-run", action="store_true", help="render the job, send nothing")
    sub.add_parser("status", help="refresh the state of the queued and running jobs")
    fetch = sub.add_parser("fetch", help="bring back metrics and submission of finished runs")
    fetch.add_argument("run_ids", nargs="+", help="run ids, or a group id for all its variants")
    return parser


def _make_remote(ctx: LabContext) -> Remote:
    """Remote over the JURECA_* variables (read lazily: local commands never need them)."""
    return Remote(load_jureca_config(ctx.env_loader()), ctx)


def _summary(ctx: LabContext, records: Sequence[RunRecord]) -> int:
    for record in records:
        parts = [record.run_id, record.status]
        if record.metrics and record.primary_metric in record.metrics:
            parts.append(f"{record.primary_metric}={record.metrics[record.primary_metric]:.4g}")
        if record.submission_file:
            parts.append(record.submission_file)
        ctx.out("  ".join(parts))
    return EXIT_FAILURE if any(r.status == STATUS_FAILED for r in records) else EXIT_OK


def _cmd_run(ctx: LabContext, args: argparse.Namespace) -> int:
    spec: ExperimentSpec = load_spec(args.spec, ctx.adapters, ctx.experiments_dir)
    adapter = resolve_adapter(spec.task, ctx.adapters)
    where: str = args.where or spec.where
    if where == WHERE_LOCAL:
        return _summary(ctx, run_local(ctx, spec, adapter))
    records = submit_job(
        ctx, _make_remote(ctx), spec, adapter, sync=not args.no_sync, dry_run=args.dry_run
    )
    if args.dry_run:
        return EXIT_OK
    for record in records:
        ctx.out(f"{record.run_id}  queued as Slurm job {record.jureca_job_id}")
    ctx.out("follow with `python -m hackathon.lab jureca status`, then `jureca fetch <run_id>`.")
    return EXIT_OK


def _cmd_jureca(ctx: LabContext, args: argparse.Namespace) -> int:
    command: str = args.jureca_command
    remote = _make_remote(ctx)
    if command == "login":
        remote.ensure_session()
        ctx.out(f"ssh session to '{remote.config.ssh_host}' is open.")
        return EXIT_OK
    if command == "sync":
        remote.ensure_session()  # even a dry run talks to the cluster: open the session first
        ctx.out(remote.sync_repo(ctx.repo_root, dry_run=args.dry_run).rstrip())
        ctx.out(f"synced to {remote.config.repo_dir}" if not args.dry_run else "(dry run)")
        return EXIT_OK
    if command == "submit":
        spec = load_spec(args.spec, ctx.adapters, ctx.experiments_dir)
        adapter = resolve_adapter(spec.task, ctx.adapters)
        records = submit_job(ctx, remote, spec, adapter, sync=args.sync, dry_run=args.dry_run)
        for record in records:
            ctx.out(f"{record.run_id}  queued as Slurm job {record.jureca_job_id}")
        return EXIT_OK
    registry = ctx.registry
    if command == "status":
        remote.ensure_session()
        active = [r for r in registry.latest().values() if r.where == WHERE_JURECA]
        refreshed = refresh_status(ctx, remote, active)
        shown = [r for r in refreshed if r.status not in ("done",)]
        if not shown:
            ctx.out("no pending JURECA run.")
        for record in shown:
            ctx.out(f"{record.run_id}  {record.status}  job {record.jureca_job_id}  {record.note}")
        return EXIT_OK
    remote.ensure_session()
    fetched: list[RunRecord] = []
    for reference in args.run_ids:
        for record in registry.find(reference):
            fetched.append(fetch_run(ctx, remote, record))
    return _summary(ctx, fetched)


def _cmd_submit(ctx: LabContext, args: argparse.Namespace) -> int:
    report = submit_run(ctx, args.run_id, yes=args.yes, again=args.again)
    if report.dry_run:
        return EXIT_REFUSED
    outcome = report.outcome
    assert outcome is not None
    ctx.out(
        f"{report.record.run_id}: ok={outcome.ok} public_score={outcome.score} "
        f"previous_best={outcome.previous_best} message={outcome.message!r}"
    )
    return EXIT_OK if outcome.ok else EXIT_FAILURE


def main(argv: Sequence[str] | None = None, *, ctx: LabContext | None = None) -> int:
    """Run one lab command.

    Args:
        argv: Arguments (defaults to `sys.argv[1:]`).
        ctx: Context override (tests); defaults to the real repository and environment.

    Returns:
        The process exit code (see the module docstring).
    """
    args = build_parser().parse_args(argv)
    context = ctx or LabContext()
    command: str = args.command
    try:
        if command == "new":
            created = create_experiment(context, args.task, args.idea)
            context.out(f"created {created.spec_path}")
            context.out(f"created {created.idea_path}")
            context.out(f"edit them, then: make hk-run SPEC={created.spec_id}")
            return EXIT_OK
        if command == "run":
            return _cmd_run(context, args)
        if command == "submit":
            return _cmd_submit(context, args)
        if command == "board":
            text, page = write_board(context, args.task)
            context.out(text)
            context.out(f"html: {page}")
            return EXIT_OK
        return _cmd_jureca(context, args)
    except KeyboardInterrupt:
        context.err("interrupted")
        return EXIT_INTERRUPTED
    except (LabError, HackathonError) as exc:
        context.err(f"error: {exc}")
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())

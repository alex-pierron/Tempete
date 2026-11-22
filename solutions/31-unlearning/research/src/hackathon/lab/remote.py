"""JURECA side of the lab: ssh/rsync wrappers, job script rendering, submit, status, fetch.

Every remote action goes through `ssh <host>` (an alias of `~/.ssh/config`, default `juelich`) and
`rsync`, launched with argument lists, never through a local shell. Only the remote command
string is parsed by a shell (the remote one), so every dynamic piece of it goes through
`shlex.quote`, and every name taken from `.env` or a spec is validated before it gets there.

The TOTP code is asked by ssh itself on the terminal. With `ControlMaster auto` and
`ControlPersist 8h` in `~/.ssh/config` (see docs/hackathon/process.md) it is asked once: the first
command opens a background master connection that all later ssh and rsync calls reuse.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from hackathon.common.errors import ConfigError
from hackathon.lab.constants import (
    ACTIVE_STATUSES,
    CODE_SNAPSHOT_DIRNAME,
    DEFAULT_SSH_HOST,
    EXIT_CODE_FILENAME,
    FINISHED_AT_FILENAME,
    HOST_PATTERN,
    JOB_SCRIPT_FILENAME,
    JURECA_SCRATCH_ROOT,
    NAME_PATTERN,
    REMOTE_REPO_DIRNAME,
    REMOTE_RUNS_DIRNAME,
    RSYNC_TIMEOUT_S,
    SLURM_FAILURE_STATES,
    SPEC_SNAPSHOT_FILENAME,
    SSH_COMMAND_TIMEOUT_S,
    SSH_CONNECT_TIMEOUT_S,
    STARTED_AT_FILENAME,
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    SYNC_EXCLUDES,
    WHERE_JURECA,
)
from hackathon.lab.context import LabContext
from hackathon.lab.errors import RemoteError, RunError
from hackathon.lab.registry import RunRecord, utc_now_iso
from hackathon.lab.runner import (
    PlannedRun,
    build_step_argv,
    finalize_record,
    git_state,
    make_run_dir,
    new_record,
    placeholder_values,
    plan_runs,
    write_run_meta,
)
from hackathon.lab.spec import ExperimentSpec, parse_spec, slugify
from hackathon.lab.tasks import TaskAdapter

HOST_RE = re.compile(HOST_PATTERN)
NAME_RE = re.compile(NAME_PATTERN)
TEMPLATE_PLACEHOLDER_RE = re.compile(r"@@([A-Z_]+)@@")
JOB_ID_RE = re.compile(r"(\d+)(?:;.*)?")
JOB_NAME_PREFIX = "lab-"
JOB_NAME_MAX = 60
LOGIN_TIMEOUT_S = 300.0
SACCT_MARKER = "@@SACCT@@"
EXIT_MARKER = "@@EXIT@@"
NO_EXIT_CODE = "none"
ISO_Z_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SAFE_COMMENT_RE = re.compile(r"[^A-Za-z0-9_=,.:/+ -]")
ONE_NODE_EXIT = 7
"""Exit code of the remote command when the team already holds a node (see `one_node_guard`)."""


@dataclass(frozen=True)
class JurecaConfig:
    """Who we are on JURECA and where the team folder is.

    Attributes:
        team_lead: JuDoor user of the team lead (owner of the shared folder).
        team_folder: Team folder name chosen in `hackathon_setup.sh`.
        ssh_host: `Host` alias of `~/.ssh/config` (default `juelich`).
        user: Own JuDoor user, for messages only (ssh takes it from `~/.ssh/config`).
    """

    team_lead: str
    team_folder: str
    ssh_host: str = DEFAULT_SSH_HOST
    user: str | None = None

    @property
    def team_dir(self) -> PurePosixPath:
        """`/p/scratch/trainingXXXX/<lead>/<team>`."""
        return PurePosixPath(JURECA_SCRATCH_ROOT) / self.team_lead / self.team_folder

    @property
    def repo_dir(self) -> PurePosixPath:
        """Where `jureca sync` puts the repository checkout (code only)."""
        return self.team_dir / REMOTE_REPO_DIRNAME

    def task_dir(self, adapter: TaskAdapter) -> PurePosixPath:
        """Task folder of the official setup (model/data, `.venv`, `output/`)."""
        return self.team_dir / adapter.remote_task_dirname

    def runs_root(self, adapter: TaskAdapter) -> PurePosixPath:
        """Parent of the output folders of lab runs: outside the synced checkout on purpose."""
        return self.team_dir / REMOTE_RUNS_DIRNAME / adapter.name


def load_jureca_config(env: Mapping[str, str]) -> JurecaConfig:
    """Read and validate `JURECA_TEAM_LEAD`, `JURECA_TEAM_FOLDER` (+ optional host and user).

    Raises:
        ConfigError: a required variable is missing, or a value could alter a command line
            (leading dash, spaces, shell characters, path separators).
    """

    def read(key: str, pattern: re.Pattern[str], *, required: bool) -> str | None:
        value = env.get(key, "").strip()
        if not value:
            if required:
                raise ConfigError(f"{key} is not set: add it to .env (see .env.example).")
            return None
        if not pattern.fullmatch(value):
            raise ConfigError(f"{key}='{value}' is not a plain name (letters, digits, . _ -).")
        return value

    lead = read("JURECA_TEAM_LEAD", NAME_RE, required=True)
    folder = read("JURECA_TEAM_FOLDER", NAME_RE, required=True)
    assert lead is not None and folder is not None
    return JurecaConfig(
        team_lead=lead,
        team_folder=folder,
        ssh_host=read("JURECA_SSH_HOST", HOST_RE, required=False) or DEFAULT_SSH_HOST,
        user=read("JURECA_USER", NAME_RE, required=False),
    )


def _binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise RemoteError(f"'{name}' not found in PATH: install it to use the JURECA commands.")
    return found


def _tail(text: str, lines: int = 6) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


class Remote:
    """ssh and rsync towards the JURECA login node, with a session check.

    Args:
        config: Validated JURECA configuration.
        ctx: Lab context (sinks for messages).
    """

    def __init__(self, config: JurecaConfig, ctx: LabContext) -> None:
        self.config = config
        self._ctx = ctx

    def _ssh_prefix(self) -> list[str]:
        return [
            _binary("ssh"),
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
            self.config.ssh_host,
        ]

    def ensure_session(self) -> None:
        """Make sure a persistent ssh connection exists, asking for the TOTP code if needed.

        `ssh -O check` is silent and instant when the ControlMaster is alive. Otherwise a plain
        `ssh <host> true` is run with the terminal attached, so the user types the TOTP code once;
        with ControlPersist the connection then stays open for every following command.

        Raises:
            RemoteError: the login itself failed.
        """
        ssh = _binary("ssh")
        host = self.config.ssh_host
        alive = subprocess.run(
            [ssh, "-O", "check", host],
            capture_output=True,
            text=True,
            check=False,
            timeout=SSH_COMMAND_TIMEOUT_S,
        )
        if alive.returncode == 0:
            return
        self._ctx.out(f"no open ssh session to '{host}': logging in (type the TOTP code).")
        login = subprocess.run(
            [ssh, "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}", host, "true"],
            check=False,
            timeout=LOGIN_TIMEOUT_S,
        )
        if login.returncode != 0:
            raise RemoteError(
                f"ssh login to '{host}' failed (exit {login.returncode}): check the TOTP code, "
                "the key/IP clause in JuDoor, and docs/hackathon/jureca.md."
            )
        recheck = subprocess.run(
            [ssh, "-O", "check", host],
            capture_output=True,
            text=True,
            check=False,
            timeout=SSH_COMMAND_TIMEOUT_S,
        )
        if recheck.returncode != 0:
            self._ctx.err(
                "warning: ssh does not keep a persistent connection (ControlMaster is not "
                "configured for this host): the TOTP code will be asked again on every command. "
                "See the ssh config block in docs/hackathon/process.md."
            )

    def ssh(
        self, command: str, *, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run one remote command string and return the finished process.

        Raises:
            RemoteError: ssh could not run, timed out or the command failed (exit code != 0);
                the message carries the tail of the remote stderr.
        """
        argv = [*self._ssh_prefix(), command]
        try:
            done = subprocess.run(
                argv,
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=SSH_COMMAND_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(f"ssh command timed out after {SSH_COMMAND_TIMEOUT_S:.0f} s") from exc
        if done.returncode != 0:
            hint = (
                " (connection failed: TOTP expired? re-run the command)"
                if done.returncode == 255
                else ""
            )
            raise RemoteError(
                f"remote command failed with exit {done.returncode}{hint}: {_tail(done.stderr)}"
            )
        return done

    def _rsync(self, argv: Sequence[str]) -> str:
        try:
            done = subprocess.run(
                [_binary("rsync"), *argv],
                capture_output=True,
                text=True,
                check=False,
                timeout=RSYNC_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(f"rsync timed out after {RSYNC_TIMEOUT_S:.0f} s") from exc
        if done.returncode != 0:
            raise RemoteError(f"rsync failed with exit {done.returncode}: {_tail(done.stderr)}")
        return done.stdout

    def sync_repo(self, repo_root: Path, *, dry_run: bool = False) -> str:
        """rsync the repository to the team folder (code only, no `--delete`).

        Excludes `.venv`, `data`, `weights`, `third_party`, `runs`, `.env` and caches/weights
        (see SYNC_EXCLUDES). Nothing is ever deleted on the cluster: the shared `.venv` and
        datasets live in sibling folders anyway.

        Returns:
            rsync's output (the file list when `dry_run`).
        """
        destination = self.config.repo_dir
        argv = ["-az", *(f"--exclude={pattern}" for pattern in SYNC_EXCLUDES)]
        if dry_run:
            # Read-only: no remote mkdir either (the destination may not exist before the 1st sync).
            argv += ["--dry-run", "-v"]
        else:
            self.ssh(f"mkdir -p -- {shlex.quote(str(destination))}")
        argv += [f"{repo_root}/", f"{self.config.ssh_host}:{destination}/"]
        return self._rsync(argv)

    def pull(self, remote_dir: str, local_dir: Path) -> None:
        """rsync a remote run directory into `local_dir` (created)."""
        local_dir.mkdir(parents=True, exist_ok=True)
        self._rsync(["-az", f"{self.config.ssh_host}:{remote_dir.rstrip('/')}/", f"{local_dir}/"])


# --------------------------------------------------------------------------------------------
# Job script


@dataclass(frozen=True)
class JobVariant:
    """One variant inside a job: its output directory, GPU slot and shell-ready step commands."""

    name: str
    label: str
    gpu: int
    out_dir: str
    commands: tuple[str, ...]


@dataclass(frozen=True)
class JobPlan:
    """Everything the sbatch template needs."""

    group_id: str
    spec_id: str
    time_limit: str | None
    team_dir: str
    repo_dir: str
    task_dir: str
    run_root: str
    variants: tuple[JobVariant, ...]


def code_snapshot_dir(plan: JobPlan) -> str:
    """Where `_sbatch_command` copies the synced checkout at submission, and where the job runs it.

    The shared `Tempete/` folder is re-synced by every teammate; a job that waits for the team's
    node (one-node rule) would otherwise run whatever code was synced last, maybe someone else's.
    """
    return f"{plan.run_root}/{CODE_SNAPSHOT_DIRNAME}"


def render_job_script(template: str, plan: JobPlan) -> str:
    """Fill the sbatch template. Values are shell-quoted here; the template stays inert text.

    Raises:
        RemoteError: the template uses a placeholder this function does not know.
    """
    variant_calls = []
    for variant in plan.variants:
        comment = SAFE_COMMENT_RE.sub("?", f"{variant.name} {variant.label}".strip())
        words = [
            shlex.quote(variant.out_dir),
            str(variant.gpu),
            *map(shlex.quote, variant.commands),
        ]
        variant_calls.append(f"# {comment}\nrun_variant {' '.join(words)}")
    job_name = (JOB_NAME_PREFIX + slugify(plan.spec_id))[:JOB_NAME_MAX]
    values = {
        "JOB_NAME": job_name,
        "TIME_DIRECTIVE": (
            f"#SBATCH --time={plan.time_limit}"
            if plan.time_limit
            else "# no --time: partition default"
        ),
        "GROUP_ID": plan.group_id,
        "TEAM_DIR": shlex.quote(plan.team_dir),
        "REPO_DIR": shlex.quote(code_snapshot_dir(plan)),  # frozen copy, not the shared checkout
        "TASK_DIR": shlex.quote(plan.task_dir),
        "RUN_ROOT": shlex.quote(plan.run_root),
        "VARIANT_COUNT": str(len(plan.variants)),
        "VARIANT_CALLS": "\n".join(variant_calls),
        "OUT_DIRS": " ".join(shlex.quote(v.out_dir) for v in plan.variants),
    }

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise RemoteError(f"job template uses an unknown placeholder @@{key}@@")
        return values[key]

    return TEMPLATE_PLACEHOLDER_RE.sub(substitute, template)


def one_node_guard(team_dir: str) -> str:
    """Remote shell setting `$deps`: the ids of every team job already on JURECA, `:`-joined.

    Team rule: the whole team uses at most ONE node at a time, and every job reserves a whole node.
    A new job is therefore queued with `--dependency=afterany:$deps`: it waits (PENDING,
    "Dependency") until every team job before it has ended, instead of taking a second node. Each
    new job depends on all the earlier ones, running or pending, so the team's jobs run one after
    the other. The team is read from the ACL of the team folder, where the lead's setup script gave
    each member a named entry: no configuration, and it cannot drift from who has access. Fails
    closed: unreadable ACL or failing squeue, nothing is queued. `jureca/team_sbatch.sh` applies
    the same rule to hand-launched jobs.
    """
    team = shlex.quote(team_dir)
    return (
        rf"team=$(getfacl -cp {team} 2>/dev/null | sed -n 's/^user:\([^:][^:]*\):.*/\1/p'"
        " | sort -u | paste -sd, -)"
        ' && { test -n "$team" || { echo "one-node rule: cannot read the team members'
        f' (getfacl {team_dir}), nothing queued" >&2; exit {ONE_NODE_EXIT}; }}; }}'
        ' && jobs=$(squeue -h -u "$team" -o %i)'
        ' && deps=$(echo $jobs | tr " " ":")'
        ' && { test -z "$deps" || echo "one-node rule: waits for the team jobs $deps"; }'
    )


def _sbatch_command(plan: JobPlan) -> str:
    """The remote command: one-node rule, sanity checks, code snapshot, script from stdin, sbatch."""
    task_dir = shlex.quote(plan.task_dir)
    repo_dir = shlex.quote(plan.repo_dir)
    script = shlex.quote(f"{plan.run_root}/{JOB_SCRIPT_FILENAME}")
    return (
        f"{one_node_guard(plan.team_dir)}"
        f" && {{ test -d {repo_dir}/src/hackathon || {{ echo 'code not synced: run jureca sync' >&2; exit 5; }}; }}"
        f" && {{ test -f {task_dir}/.venv/bin/activate"
        f" || {{ echo 'no .venv in the task folder (setup script not run?)' >&2; exit 6; }}; }}"
        f" && mkdir -p -- {shlex.quote(plan.run_root)} {task_dir}/output"
        f" && cp -R -- {repo_dir} {shlex.quote(code_snapshot_dir(plan))}"
        f" && cat > {script}"
        f" && cd {task_dir}"
        f" && sbatch --parsable ${{deps:+--dependency=afterany:$deps}} {script}"
    )


def build_job_plan(
    config: JurecaConfig,
    spec: ExperimentSpec,
    adapter: TaskAdapter,
    plan: Sequence[PlannedRun],
) -> tuple[JobPlan, list[list[str]]]:
    """Resolve every placeholder for JURECA; returns the job plan and the argv of each variant.

    The returned argvs (one list of step command lines per variant) are in the same order as
    `plan`, with `python` as interpreter (the task `.venv` is activated by the job script).
    """
    run_root = config.runs_root(adapter) / plan[0].group_id
    extra = {"task_dir": str(config.task_dir(adapter)), "team_dir": str(config.team_dir)}
    variants: list[JobVariant] = []
    all_commands: list[list[str]] = []
    for position, planned in enumerate(plan):
        out_dir = str(run_root / planned.variant.name)
        values = placeholder_values(planned, out_dir, extra=extra)
        commands = [
            ["python", *build_step_argv(adapter, step, values, where=WHERE_JURECA)]
            for step in spec.steps
        ]
        all_commands.append([shlex.join(c) for c in commands])
        variants.append(
            JobVariant(
                name=planned.variant.name,
                label=planned.variant.label,
                gpu=position,
                out_dir=out_dir,
                commands=tuple(shlex.join(c) for c in commands),
            )
        )
    job = JobPlan(
        group_id=plan[0].group_id,
        spec_id=spec.id,
        time_limit=spec.time_limit,
        team_dir=str(config.team_dir),
        repo_dir=str(config.repo_dir),
        task_dir=str(config.task_dir(adapter)),
        run_root=str(run_root),
        variants=tuple(variants),
    )
    return job, all_commands


def submit_job(
    ctx: LabContext,
    remote: Remote,
    spec: ExperimentSpec,
    adapter: TaskAdapter,
    *,
    sync: bool = False,
    dry_run: bool = False,
) -> list[RunRecord]:
    """Queue every variant of `spec` as ONE Slurm job; one registry line per variant.

    On a dry run the rendered script and the remote command are printed and nothing is written or
    sent. Otherwise the local run directories (spec snapshot, run.json, job.sbatch) are created
    first, which also rejects a same-second relaunch BEFORE anything is queued; if ssh, rsync or
    sbatch then fails they are removed again, so a failed submission leaves no trace.

    Returns:
        The `queued` records (empty list on a dry run).

    Raises:
        RemoteError: ssh, rsync or sbatch failed, the job id could not be read back, or the
            team members could not be read (one-node rule, see `one_node_guard`).
        RunError: a run directory already exists.
    """
    template_path = ctx.job_template_path
    if not template_path.is_file():
        raise RemoteError(f"job template missing: {template_path}")
    sha, dirty = git_state(ctx.repo_root)
    plan = plan_runs(spec, adapter, ctx.clock())
    job, commands = build_job_plan(remote.config, spec, adapter, plan)
    script = render_job_script(template_path.read_text(encoding="utf-8"), job)
    if dry_run:
        ctx.out(script)
        ctx.out(f"# remote command: ssh {remote.config.ssh_host} {_sbatch_command(job)}")
        return []
    records: list[RunRecord] = []
    created: list[Path] = []
    try:
        for planned, step_lines in zip(plan, commands, strict=True):
            out_dir = ctx.run_dir(spec.task, planned.run_id)
            make_run_dir(out_dir)
            created.append(out_dir)
            (out_dir / SPEC_SNAPSHOT_FILENAME).write_text(spec.source_text, encoding="utf-8")
            (out_dir / JOB_SCRIPT_FILENAME).write_text(script, encoding="utf-8")
            record = new_record(
                ctx, spec, planned, sha=sha, dirty=dirty, where=WHERE_JURECA, status=STATUS_QUEUED
            )
            write_run_meta(
                out_dir,
                record=record,
                spec=spec,
                argvs=[shlex.split(c) for c in step_lines],
                where=WHERE_JURECA,
            )
            records.append(record)
        remote.ensure_session()
        if sync:
            remote.sync_repo(ctx.repo_root)
        done = remote.ssh(_sbatch_command(job), input_text=script)
    except BaseException:
        for directory in created:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    for line in lines[:-1]:  # e.g. "one-node rule: waits for the team jobs ..."
        ctx.out(line)
    match = JOB_ID_RE.fullmatch(lines[-1]) if lines else None
    if match is None:
        raise RemoteError(
            f"sbatch printed no job id ({done.stdout!r}); the job may or may not be queued: "
            "check with `squeue --me` on the cluster. Local run directories were kept."
        )
    job_id = match.group(1)
    return [
        ctx.registry.update(
            record, jureca_job_id=job_id, remote_dir=variant.out_dir, note=f"slurm job {job_id}"
        )
        for record, variant in zip(records, job.variants, strict=True)
    ]


# --------------------------------------------------------------------------------------------
# Status and fetch


@dataclass(frozen=True)
class RemoteJobState:
    """What `squeue`, `sacct` and the exit-code files said about one job.

    Attributes:
        squeue_state: Live Slurm state (PENDING, RUNNING, ...) or None when the job left the queue.
        sacct_state: Accounting state of a finished job (COMPLETED, FAILED, ...) or None.
        exit_codes: Exit code per output directory (None while the variant has not finished).
    """

    squeue_state: str | None
    sacct_state: str | None
    exit_codes: Mapping[str, int | None]


def _status_command(job_id: str, out_dirs: Sequence[str]) -> str:
    quoted = " ".join(shlex.quote(path) for path in out_dirs)
    return (
        f"squeue -h -j {shlex.quote(job_id)} -o %T 2>/dev/null; echo {SACCT_MARKER}; "
        f"sacct -j {shlex.quote(job_id)} -X -n -P -o State 2>/dev/null; echo {EXIT_MARKER}; "
        f"for d in {quoted}; do printf '%s\\t' \"$d\"; "
        f'if [ -f "$d/{EXIT_CODE_FILENAME}" ]; then cat "$d/{EXIT_CODE_FILENAME}"; '
        f"else echo {NO_EXIT_CODE}; fi; done"
    )


def parse_status_output(text: str) -> RemoteJobState:
    """Parse the output of `_status_command` (pure, unit-tested with canned text)."""
    squeue_part, _, rest = text.partition(SACCT_MARKER)
    sacct_part, _, exit_part = rest.partition(EXIT_MARKER)

    def first_word(block: str) -> str | None:
        for line in block.splitlines():
            if line.strip():
                return line.strip().split()[0]
        return None

    exit_codes: dict[str, int | None] = {}
    for line in exit_part.splitlines():
        path, separator, code = line.partition("\t")
        if not separator:
            continue
        code = code.strip()
        exit_codes[path] = int(code) if re.fullmatch(r"-?\d+", code) else None
    return RemoteJobState(first_word(squeue_part), first_word(sacct_part), exit_codes)


def derive_status(state: RemoteJobState, out_dir: str) -> tuple[str | None, str, int | None]:
    """Map a job state to (lab status, note, exit code) for the variant in `out_dir`.

    The exit-code file wins: the job script writes it last, so it is the variant's own verdict
    even while the job (and its siblings) still run. Status None means "cannot tell yet".
    """
    code = state.exit_codes.get(out_dir)
    if code is not None:
        return (STATUS_FINISHED if code == 0 else STATUS_FAILED), "", code
    if state.squeue_state in ("PENDING", "CONFIGURING"):
        return STATUS_QUEUED, "", None
    if state.squeue_state is not None:
        return STATUS_RUNNING, "", None
    final = state.sacct_state or ""
    if final.split("+")[0] in SLURM_FAILURE_STATES:
        return STATUS_FAILED, f"slurm state {final}", None
    if final == "COMPLETED":
        return STATUS_FAILED, "slurm COMPLETED but the variant wrote no exit_code", None
    return None, "job not in squeue/sacct yet", None


def refresh_status(
    ctx: LabContext, remote: Remote, records: Sequence[RunRecord]
) -> list[RunRecord]:
    """Ask the cluster about every active JURECA run and append the snapshots that changed.

    One ssh call per Slurm job. Runs of other places, or already settled, are returned as is.
    """
    by_job: dict[str, list[RunRecord]] = {}
    for record in records:
        waiting = record.where == WHERE_JURECA and record.status in ACTIVE_STATUSES
        if waiting and record.jureca_job_id and record.remote_dir:
            by_job.setdefault(record.jureca_job_id, []).append(record)
    refreshed: dict[str, RunRecord] = {}
    for job_id, group in by_job.items():
        paths = [str(r.remote_dir) for r in group]
        state = parse_status_output(remote.ssh(_status_command(job_id, paths)).stdout)
        for record in group:
            status, note, code = derive_status(state, str(record.remote_dir))
            if status is None:
                ctx.err(f"[{record.run_id}] {note}")
                continue
            if status != record.status or code != record.exit_code:
                refreshed[record.run_id] = ctx.registry.update(
                    record, status=status, exit_code=code, note=note or record.note
                )
    return [refreshed.get(r.run_id, r) for r in records]


def _read_stamp(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.read_text(encoding="utf-8").strip(), ISO_Z_FORMAT).replace(
            tzinfo=UTC
        )
    except (OSError, ValueError):
        return None


def fetch_run(ctx: LabContext, remote: Remote, record: RunRecord) -> RunRecord:
    """Bring a finished JURECA run home and collect its metrics and submission file.

    Raises:
        RunError: the run is not a JURECA run, or has not finished yet.
        RemoteError: rsync failed.
    """
    if record.where != WHERE_JURECA or not record.remote_dir:
        raise RunError(f"{record.run_id} is not a JURECA run")
    current = refresh_status(ctx, remote, [record])[0]
    if current.status in ACTIVE_STATUSES:
        raise RunError(f"{record.run_id} is still {current.status}: fetch it once it has finished")
    local_dir = ctx.run_dir(record.task, record.run_id)
    remote.pull(str(current.remote_dir), local_dir)
    snapshot = local_dir / SPEC_SNAPSHOT_FILENAME
    if not snapshot.is_file():
        raise RunError(f"{snapshot} is missing: cannot tell what to collect")
    spec = parse_spec(snapshot.read_text(encoding="utf-8"), ctx.adapters)
    code_text = (
        (local_dir / EXIT_CODE_FILENAME).read_text(encoding="utf-8").strip()
        if (local_dir / EXIT_CODE_FILENAME).is_file()
        else ""
    )
    exit_code = int(code_text) if re.fullmatch(r"-?\d+", code_text) else (current.exit_code or 1)
    started = _read_stamp(local_dir / STARTED_AT_FILENAME)
    finished = _read_stamp(local_dir / FINISHED_AT_FILENAME)
    duration = (finished - started).total_seconds() if started and finished else None
    return finalize_record(
        ctx,
        current,
        spec,
        local_dir,
        exit_code=exit_code,
        started_at=utc_now_iso(started) if started else None,
        finished_at=utc_now_iso(finished) if finished else None,
        duration_s=duration,
    )

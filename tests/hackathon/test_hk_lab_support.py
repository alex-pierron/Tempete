"""Shared helpers of the lab tests: a fake task CLI, a fake cluster (ssh, rsync, sbatch, ...), a context.

The fake cluster follows the daemon-fake-harness technique: executables placed first on PATH that
log their argv and stdin. `ssh` really executes the remote command with `bash -c`, after mapping
`/p/scratch/training2666` into a temporary tree, so the generated remote command strings and the
generated Slurm script are exercised for real (mkdir, cat, cd, sbatch, squeue, exit-code files).
Nothing here opens a network connection or touches the real HOME.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hackathon.common.config import Settings
from hackathon.lab.context import LabContext
from hackathon.lab.tasks import TaskAdapter

REAL_REPO = Path(__file__).resolve().parents[2]
CLOCK_START = datetime(2026, 11, 21, 10, 0, 0, tzinfo=UTC)
REMOTE_PREFIX = "/p/scratch/training2666"
LEAD = "lead1"
TEAM = "team1"
TEAM_DIR = f"{REMOTE_PREFIX}/{LEAD}/{TEAM}"

FAKE_TASK_SOURCE = """
import argparse, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task-dir", default=None)
sub = parser.add_subparsers(dest="command", required=True)
run = sub.add_parser("run")
run.add_argument("--out", required=True)
run.add_argument("--set", action="append", default=[])
run.add_argument("--config", default="none")
run.add_argument("--proxy-root", default=None)
run.add_argument("--fail", action="store_true")
run.add_argument("--no-sidecar", action="store_true")
analyze = sub.add_parser("analyze")
analyze.add_argument("--out-dir", required=True)
validate = sub.add_parser("validate")
validate.add_argument("file")
submit = sub.add_parser("submit")
submit.add_argument("file")
submit.add_argument("--yes", action="store_true")
args = parser.parse_args()

calls = os.environ.get("FAKE_TASK_CALLS")
if calls:
    with open(calls, "a") as handle:
        handle.write(json.dumps(sys.argv[1:]) + "\\n")

if args.command == "run":
    print("training started", flush=True)
    print("MCC value: 0.25", flush=True)
    print("MCC value: 0.75", flush=True)
    print("a warning on stderr", file=sys.stderr, flush=True)
    print("gpu=" + os.environ.get("CUDA_VISIBLE_DEVICES", "unset"), flush=True)
    print("task_dir=" + str(args.task_dir), flush=True)
    print("config=" + args.config + " proxy_root=" + str(args.proxy_root), flush=True)
    if args.fail:
        sys.exit(3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("weights " + ",".join(args.set))
    if not args.no_sidecar:
        out.with_suffix(".json").write_text(json.dumps(
            {"proxy": {"retain_kl": 0.125, "movement": 2}, "train_seconds": 1.5,
             "history": [{"loss": 1.0}], "label": "x", "flag": True}))
elif args.command == "analyze":
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print("analysis done")
elif args.command == "validate":
    text = Path(args.file).read_text()
    if "invalid" in text:
        print("not a valid file", file=sys.stderr)
        sys.exit(1)
    print("VALID: " + args.file)
elif args.command == "submit":
    mode = os.environ.get("FAKE_TASK_SUBMIT_MODE", "json")
    if mode == "json":
        print(json.dumps({"ok": True, "score": 0.42, "previous_best": 0.3, "message": "nice"}, indent=2))
    elif mode == "line":
        print("ok=True score=0.31 previous_best=0.2 improved=True message='better'")
    elif mode == "reject":
        print(json.dumps({"ok": False, "score": None, "message": "bad shape"}))
        sys.exit(1)
    elif mode == "local_refusal":
        print("error: wait 12 more minute(s)", file=sys.stderr)
        sys.exit(1)
    elif mode == "silent":
        print("submitted")
"""

FAKE_SSH_SOURCE = """#!__PY__
import json, os, subprocess, sys
from pathlib import Path

argv = sys.argv[1:]
state = Path(os.environ["FAKE_STATE_DIR"])
root = os.environ["FAKE_REMOTE_ROOT"]
prefix = "/p/scratch/training2666"
stdin_text = "" if sys.stdin.isatty() else sys.stdin.read()
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps({"tool": "ssh", "argv": argv, "stdin": stdin_text}) + "\\n")
master = state / "master"
if "-O" in argv:
    sys.exit(0 if master.exists() else 255)
if os.environ.get("FAKE_SSH_DOWN"):
    sys.stderr.write("ssh: connect to host failed\\n")
    sys.exit(255)
command = argv[-1]
if command == "true":
    if not os.environ.get("FAKE_SSH_NO_CONTROLMASTER"):
        master.write_text("up")
    sys.exit(0)
env = dict(os.environ)
env["PATH"] = str(state / "bin") + os.pathsep + env["PATH"]
done = subprocess.run(
    ["bash", "-c", command.replace(prefix, root + prefix)],
    input=stdin_text.replace(prefix, root + prefix), text=True, capture_output=True, env=env,
)
sys.stdout.write(done.stdout.replace(root + prefix, prefix))
sys.stderr.write(done.stderr)
sys.exit(done.returncode)
"""

FAKE_RSYNC_SOURCE = """#!__PY__
import fnmatch, json, os, shutil, sys
from pathlib import Path

argv = sys.argv[1:]
state = Path(os.environ["FAKE_STATE_DIR"])
root = os.environ["FAKE_REMOTE_ROOT"]
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps({"tool": "rsync", "argv": argv}) + "\\n")
excludes = [a.split("=", 1)[1] for a in argv if a.startswith("--exclude=")]
src, dst = argv[-2], argv[-1]
if os.environ.get("FAKE_RSYNC_FAIL"):
    sys.stderr.write("rsync: connection unexpectedly closed\\n")
    sys.exit(12)
if "--dry-run" in argv:
    print("dry run listing")
    sys.exit(0)
if src.startswith("juelich:"):
    shutil.copytree(root + src.split(":", 1)[1], dst, dirs_exist_ok=True)
else:
    def ignore(directory, names):
        top = Path(directory).resolve() == Path(src).resolve()
        skipped = []
        for name in names:
            for pattern in excludes:
                if pattern.startswith("/"):
                    hit = top and name == pattern[1:]
                else:
                    hit = fnmatch.fnmatch(name, pattern)
                if hit:
                    skipped.append(name)
                    break
        return skipped
    target = root + dst.split(":", 1)[1]
    shutil.copytree(src, target, ignore=ignore, dirs_exist_ok=True)
"""

FAKE_SBATCH_SOURCE = """#!__PY__
import json, os, subprocess, sys
from pathlib import Path

argv = sys.argv[1:]
state = Path(os.environ["FAKE_STATE_DIR"])
script = Path(argv[-1])
counter = state / "job_counter"
number = int(counter.read_text()) + 1 if counter.exists() else 4242
counter.write_text(str(number))
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps({"tool": "sbatch", "argv": argv, "cwd": os.getcwd(),
                             "script": script.read_text()}) + "\\n")
(state / ("job_" + str(number))).write_text("PENDING")
if os.environ.get("FAKE_SBATCH_RUN") == "1":
    (state / ("job_" + str(number))).write_text("")
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          env={**os.environ, "SLURM_JOB_ID": str(number)})
    (state / ("job_" + str(number) + ".log")).write_text(done.stdout + done.stderr)
    (state / ("sacct_" + str(number))).write_text("COMPLETED" if done.returncode == 0 else "FAILED")
print(number)
"""

FAKE_SQUEUE_SOURCE = """#!__PY__
import os, sys
from pathlib import Path

state = Path(os.environ["FAKE_STATE_DIR"])
job = sys.argv[sys.argv.index("-j") + 1]
file = state / ("job_" + job)
if file.exists() and file.read_text().strip():
    print(file.read_text().strip())
"""

FAKE_SACCT_SOURCE = """#!__PY__
import os, sys
from pathlib import Path

state = Path(os.environ["FAKE_STATE_DIR"])
job = sys.argv[sys.argv.index("-j") + 1]
file = state / ("sacct_" + job)
if file.exists():
    print(file.read_text().strip())
"""

FAKE_NVIDIA_SMI_SOURCE = """#!__PY__
import os, sys

count = int(os.environ.get("FAKE_GPU_COUNT", "4"))
if "-L" in sys.argv:
    for index in range(count):
        print("GPU %d: NVIDIA A100-SXM4-40GB (UUID: GPU-fake)" % index)
elif "--query-gpu=index,name,memory.total" in sys.argv:
    for index in range(count):
        print("%d, NVIDIA A100-SXM4-40GB, 40960 MiB" % index)
"""


@dataclass
class FakeCluster:
    """Handle on the fake JURECA of one test."""

    state_dir: Path
    remote_root: Path
    task_script: Path

    def calls(self, tool: str | None = None) -> list[dict[str, object]]:
        """Every logged call (ssh, rsync, sbatch), oldest first, optionally of one tool."""
        log = self.state_dir / "calls.jsonl"
        if not log.is_file():
            return []
        rows = [json.loads(line) for line in log.read_text().splitlines() if line]
        return [row for row in rows if tool is None or row["tool"] == tool]

    def remote_path(self, remote: str) -> Path:
        """Where a `/p/scratch/training2666/...` path lives in the fake tree."""
        return Path(str(self.remote_root) + remote)

    def prepare_task(self, dirname: str = "unlearning") -> Path:
        """Create the official setup a job expects: task dir, `.venv/bin/activate`, synced code."""
        task_dir = self.remote_path(f"{TEAM_DIR}/{dirname}")
        activate = task_dir / ".venv" / "bin" / "activate"
        activate.parent.mkdir(parents=True, exist_ok=True)
        activate.write_text(f'export PATH="{self.state_dir / "bin"}:$PATH"\n')
        (self.remote_path(f"{TEAM_DIR}/Tempete/src/hackathon")).mkdir(parents=True, exist_ok=True)
        return task_dir

    def set_job_state(self, job_id: str, state: str) -> None:
        """What `squeue` prints for the job (empty string = left the queue)."""
        (self.state_dir / f"job_{job_id}").write_text(state)

    def set_sacct(self, job_id: str, state: str) -> None:
        """What `sacct` prints for the job."""
        (self.state_dir / f"sacct_{job_id}").write_text(state)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source.replace("__PY__", sys.executable), encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def fake_cluster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCluster:
    """Fake ssh/rsync/sbatch/squeue/sacct/nvidia-smi first on PATH, isolated HOME."""
    state = tmp_path / "fake_state"
    bin_dir = state / "bin"
    bin_dir.mkdir(parents=True)
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    for name, source in (
        ("ssh", FAKE_SSH_SOURCE),
        ("rsync", FAKE_RSYNC_SOURCE),
        ("sbatch", FAKE_SBATCH_SOURCE),
        ("squeue", FAKE_SQUEUE_SOURCE),
        ("sacct", FAKE_SACCT_SOURCE),
        ("nvidia-smi", FAKE_NVIDIA_SMI_SOURCE),
    ):
        _write_executable(bin_dir / name, source)
    (bin_dir / "python").symlink_to(sys.executable)
    task_script = tmp_path / "fake_task.py"
    task_script.write_text(FAKE_TASK_SOURCE, encoding="utf-8")
    # ssh/rsync/sbatch are found through PATH; the real HOME (and its .bashrc) is never read.
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FAKE_STATE_DIR", str(state))
    monkeypatch.setenv("FAKE_REMOTE_ROOT", str(remote_root))
    monkeypatch.setenv("FAKE_TASK_CALLS", str(state / "task_calls.jsonl"))
    return FakeCluster(state, remote_root, task_script)


def build_adapter(
    script: Path, *, name: str = "unlearning", server_task_id: str | None = "31-unlearning",
    global_args: tuple[str, ...] = ("--task-dir", "{task_dir}"),
) -> TaskAdapter:  # fmt: skip
    """An adapter whose entry point is the fake task script."""
    return TaskAdapter(
        name=name,
        short=name[:3],
        entrypoint=(str(script),),
        subcommands=frozenset({"run", "analyze", "validate", "submit"}),
        server_task_id=server_task_id,
        remote_task_dirname=name,
        remote_global_args=global_args,
    )


class StepClock:
    """A clock that advances one second per call: run ids and snapshots stay unique and ordered."""

    def __init__(self, start: datetime = CLOCK_START) -> None:
        self._now = start

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now

    def advance(self, **delta: float) -> None:
        """Jump forward (cooldown tests)."""
        self._now += timedelta(**delta)


@dataclass
class LabHarness:
    """A context over a temporary repository, with captured output."""

    ctx: LabContext
    out: list[str]
    err: list[str]
    clock: StepClock
    repo: Path


def make_settings(tmp_path: Path) -> Settings:
    """Settings over a temporary data directory (no API key, never the real `.env`)."""
    return Settings(
        api_key="k", server_url="http://server", hf_token=None, data_dir=tmp_path / "data",
        wmloc_task_id=None,
    )  # fmt: skip


def make_lab(
    tmp_path: Path,
    adapters: Mapping[str, TaskAdapter] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    settings_loader: Callable[[], Settings] | None = None,
) -> LabHarness:
    """Build a LabHarness: a temp repo with the REAL templates and job template copied in."""
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "jureca").mkdir()
    for source in (REAL_REPO / "experiments" / "_templates").iterdir():
        target = repo / "experiments" / "_templates" / source.name
        target.parent.mkdir(exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    template = REAL_REPO / "jureca" / "lab_job.sbatch.tmpl"
    (repo / "jureca" / template.name).write_text(template.read_text(encoding="utf-8"))
    out: list[str] = []
    err: list[str] = []
    clock = StepClock()
    environment = {"JURECA_TEAM_LEAD": LEAD, "JURECA_TEAM_FOLDER": TEAM, "JURECA_USER": "someone"}
    environment.update(env or {})
    ctx = LabContext(
        repo_root=repo,
        adapters=adapters if adapters is not None else {},
        python=sys.executable,
        clock=clock,
        out=out.append,
        err=err.append,
        env_loader=lambda: environment,
        settings_loader=settings_loader or (lambda: make_settings(tmp_path)),
    )
    return LabHarness(ctx, out, err, clock, repo)


UNLEARNING_SPEC = """
id = "{id}"
task = "unlearning"
idea = "gold soft"
hypothesis = "lower temperature keeps more signal"
where = "{where}"

[sweep]
target_temperature = {sweep}

[[steps]]
name = "train"
args = ["run", "--set", "target_temperature={{target_temperature}}", "--out", "{{out}}/candidate.pt"]
args_local = ["--config", "smoke"]
args_jureca = ["--config", "full", "--proxy-root", "{{task_dir}}/proxy_images"]

[expect]
submission = "candidate.pt"
metrics_file = "candidate.json"
primary_metric = "proxy.retain_kl"
higher_is_better = false
[expect.metrics_regex]
mcc = 'MCC value: ([0-9.]+)'
"""


def write_spec(
    harness: LabHarness, spec_id: str = "exp-one", *, where: str = "local",
    sweep: str = "[1.5, 2.0]", text: str | None = None,
) -> Path:  # fmt: skip
    """Write `experiments/unlearning/<id>.toml` in the harness repo and return its path."""
    path = harness.repo / "experiments" / "unlearning" / f"{spec_id}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        text if text is not None else UNLEARNING_SPEC.format(id=spec_id, where=where, sweep=sweep),
        encoding="utf-8",
    )
    return path


def quoted(*words: str) -> str:
    """shlex.join, for readable expectations."""
    return shlex.join(words)

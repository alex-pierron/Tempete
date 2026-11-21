"""Tests of the JURECA side with a fake cluster: ssh/rsync/sbatch/squeue/sacct on the PATH.

The fake `ssh` executes the remote command string for real (`bash -c`, paths mapped into a temp
tree), and the fake `sbatch` can run the generated job script, so the quoting of the remote
commands, the Slurm script (GPU pinning, exit-code files) and the status/fetch protocol are all
exercised without any network access.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from test_hk_lab_support import (
    REAL_REPO,
    TEAM_DIR,
    FakeCluster,
    LabHarness,
    build_adapter,
    make_lab,
    write_spec,
)

from hackathon.common.errors import ConfigError
from hackathon.lab.errors import RemoteError, RunError
from hackathon.lab.registry import RunRecord
from hackathon.lab.remote import (
    JobPlan,
    JobVariant,
    JurecaConfig,
    Remote,
    RemoteJobState,
    derive_status,
    fetch_run,
    load_jureca_config,
    parse_status_output,
    refresh_status,
    render_job_script,
    submit_job,
)
from hackathon.lab.spec import load_spec

TEMPLATE = (REAL_REPO / "jureca" / "lab_job.sbatch.tmpl").read_text(encoding="utf-8")

pytest_plugins = ("test_hk_lab_support",)  # the fake_cluster fixture


def jureca_lab(
    tmp_path: Path, cluster: FakeCluster, **spec_kwargs: str
) -> tuple[LabHarness, Remote]:
    adapter = build_adapter(cluster.task_script)
    harness = make_lab(tmp_path, {adapter.name: adapter})
    write_spec(harness, "exp-one", where="jureca", **spec_kwargs)
    remote = Remote(load_jureca_config(harness.ctx.env_loader()), harness.ctx)
    return harness, remote


def queue(harness: LabHarness, remote: Remote, **kwargs: bool) -> list[RunRecord]:
    spec = load_spec("exp-one", harness.ctx.adapters, harness.ctx.experiments_dir)
    return submit_job(harness.ctx, remote, spec, harness.ctx.adapters[spec.task], **kwargs)


# ---- configuration ------------------------------------------------------------------------


def test_config_defaults_and_derived_remote_paths() -> None:
    config = load_jureca_config({"JURECA_TEAM_LEAD": "lead1", "JURECA_TEAM_FOLDER": "team1"})
    assert config.ssh_host == "juelich" and config.user is None
    assert str(config.team_dir) == TEAM_DIR
    assert str(config.repo_dir) == f"{TEAM_DIR}/Tempete"
    adapter = build_adapter(Path("x"))
    assert str(config.task_dir(adapter)) == f"{TEAM_DIR}/unlearning"
    assert str(config.runs_root(adapter)) == f"{TEAM_DIR}/lab_runs/unlearning"


def test_config_reads_optional_host_and_user() -> None:
    env = {
        "JURECA_TEAM_LEAD": "a",
        "JURECA_TEAM_FOLDER": "b",
        "JURECA_SSH_HOST": "jureca-alias",
        "JURECA_USER": "me",
    }
    config = load_jureca_config(env)
    assert (config.ssh_host, config.user) == ("jureca-alias", "me")


@pytest.mark.parametrize("missing", ["JURECA_TEAM_LEAD", "JURECA_TEAM_FOLDER"])
def test_config_requires_lead_and_folder(missing: str) -> None:
    env = {"JURECA_TEAM_LEAD": "a", "JURECA_TEAM_FOLDER": "b"}
    del env[missing]
    with pytest.raises(ConfigError, match=missing):
        load_jureca_config(env)


@pytest.mark.parametrize(
    "bad", ["-oProxyCommand=evil", "a b", "a;rm -rf x", "a/b", "../x", "$(id)", "a\nb", "é"]
)
def test_config_rejects_values_that_could_alter_a_command_line(bad: str) -> None:
    for key in ("JURECA_TEAM_LEAD", "JURECA_TEAM_FOLDER", "JURECA_USER"):
        env = {"JURECA_TEAM_LEAD": "a", "JURECA_TEAM_FOLDER": "b", key: bad}
        with pytest.raises(ConfigError, match="plain name"):
            load_jureca_config(env)
    with pytest.raises(ConfigError, match="JURECA_SSH_HOST"):
        load_jureca_config(
            {"JURECA_TEAM_LEAD": "a", "JURECA_TEAM_FOLDER": "b", "JURECA_SSH_HOST": bad}
        )


# ---- job script -----------------------------------------------------------------------------


def make_plan(commands: tuple[str, ...], label: str = "", time_limit: str | None = None) -> JobPlan:
    return JobPlan(
        group_id="20261121T100000Z-unl-x",
        spec_id="x",
        time_limit=time_limit,
        team_dir=TEAM_DIR,
        repo_dir=f"{TEAM_DIR}/Tempete",
        task_dir=f"{TEAM_DIR}/unlearning",
        run_root=f"{TEAM_DIR}/lab_runs/unlearning/g",
        variants=(
            JobVariant("v1", label, 0, f"{TEAM_DIR}/lab_runs/unlearning/g/v1", commands),
            JobVariant("v2", label, 1, f"{TEAM_DIR}/lab_runs/unlearning/g/v2", commands),
        ),
    )


def test_job_script_has_the_organisers_header_and_pins_one_gpu_per_variant() -> None:
    script = render_job_script(TEMPLATE, make_plan(("python -m x run",), time_limit="01:00:00"))
    for line in (
        "#SBATCH --account=training2666",
        "#SBATCH --reservation=cispahack",
        "#SBATCH --partition=dc-gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=128",
        "#SBATCH --output=output/%j.out",
        "#SBATCH --error=output/%j.err",
        "#SBATCH --job-name=lab-x",
        "#SBATCH --time=01:00:00",
    ):
        assert line in script.splitlines()
    assert "@@" not in script
    assert f"run_variant {TEAM_DIR}/lab_runs/unlearning/g/v1 0 'python -m x run'" in script
    assert f"run_variant {TEAM_DIR}/lab_runs/unlearning/g/v2 1 'python -m x run'" in script
    assert "CUDA_VISIBLE_DEVICES" in script and "PYTHONPATH" in script
    code = [ln for ln in script.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert not any(ln.strip().startswith("set -e") for ln in code), (
        "variants must not kill siblings"
    )
    assert not any("submit" in ln or "sbatch" in ln or "curl" in ln for ln in code)


def test_job_script_without_time_limit_leaves_the_partition_default() -> None:
    assert "--time" not in render_job_script(TEMPLATE, make_plan(("true",))).replace(
        "# no --time", ""
    )


def test_hostile_values_stay_inert_in_the_script(tmp_path: Path) -> None:
    hostile = "python -m x run --set a='$(touch PWNED)'; touch PWNED2\nrm -rf /"
    script = render_job_script(TEMPLATE, make_plan((hostile,), label="v\ntouch PWNED3 #"))
    script_file = tmp_path / "job.sh"
    script_file.write_text(script, encoding="utf-8")
    assert (
        subprocess.run(
            ["bash", "-n", str(script_file)], capture_output=True, check=False
        ).returncode
        == 0
    )
    # The hostile text survives as exactly ONE shell word: quoted, never interpreted.
    first_call = script[script.index("run_variant /") : script.index("# v2")]
    assert shlex.split(first_call)[3] == hostile
    # A newline in a label cannot start a new script line: the comment is sanitised.
    assert "touch PWNED3" not in [line.strip() for line in script.splitlines()]
    assert "# v1 v?touch PWNED3 ?" in script


def test_template_with_an_unknown_placeholder_is_refused() -> None:
    with pytest.raises(RemoteError, match="unknown placeholder @@NOPE@@"):
        render_job_script(TEMPLATE + "\n@@NOPE@@\n", make_plan(("true",)))


# ---- submit, run, status, fetch (end to end on the fake cluster) -----------------------------


def test_submit_job_queues_one_job_runs_the_variants_pinned_and_fetches(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_SBATCH_RUN", "1")
    task_dir = fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[1.5, 2.0]")
    (harness.repo / "src" / "hackathon").mkdir(parents=True)
    (harness.repo / "src" / "hackathon" / "marker.py").write_text("x = 1\n")

    queued = queue(harness, remote, sync=True)

    # one job for two variants, one registry line each
    assert [r.jureca_job_id for r in queued] == ["4242", "4242"]
    assert [r.status for r in queued] == ["queued", "queued"]
    assert queued[0].remote_dir == f"{TEAM_DIR}/lab_runs/unlearning/{queued[0].group_id}/v1"
    assert len(fake_cluster.calls("sbatch")) == 1
    sbatch = fake_cluster.calls("sbatch")[0]
    assert sbatch["cwd"] == str(task_dir), "sbatch must run from the task dir (relative --output)"
    assert (task_dir / "output").is_dir()
    script = str(sbatch["script"])
    assert "--task-dir" in script and "--config full" in script and "proxy_images" in script
    assert (
        fake_cluster.remote_path(TEAM_DIR) / "Tempete" / "src" / "hackathon" / "marker.py"
    ).is_file()
    # the job really ran on the fake node: variants pinned to GPU 0 and 1, exit codes written last
    run_root = fake_cluster.remote_path(f"{TEAM_DIR}/lab_runs/unlearning/{queued[0].group_id}")
    for index, variant in enumerate(("v1", "v2")):
        assert (run_root / variant / "exit_code").read_text().strip() == "0"
        stdout = (run_root / variant / "stdout.log").read_text()
        assert f"gpu={index}" in stdout
        assert stdout.splitlines()[0].startswith("+ python ")
        assert (run_root / variant / "candidate.pt").is_file()
    assert (run_root / "v1" / "started_at").is_file() and (
        run_root / "v1" / "finished_at"
    ).is_file()
    # local trace written before sending
    local = harness.ctx.run_dir("unlearning", queued[0].run_id)
    sent_script = script.replace(str(fake_cluster.remote_root), "")  # the fake maps /p/scratch
    assert (local / "job.sbatch").read_text() == sent_script and (local / "spec.toml").is_file()

    # status: the job left the queue and sacct says COMPLETED, exit-code files say 0
    records = list(harness.ctx.registry.latest().values())
    refreshed = refresh_status(harness.ctx, remote, records)
    assert [r.status for r in refreshed] == ["finished", "finished"]
    assert [r.exit_code for r in refreshed] == [0, 0]

    # fetch brings back the submission file, metrics and duration
    done = [fetch_run(harness.ctx, remote, r) for r in refreshed]
    assert [r.status for r in done] == ["done", "done"]
    first = done[0]
    assert first.metrics["proxy.retain_kl"] == 0.125 and first.metrics["mcc"] == 0.75
    assert first.submission_file == f"runs/unlearning/{first.run_id}/candidate.pt"
    candidate = harness.repo / str(first.submission_file)
    assert candidate.read_text() == "weights target_temperature=1.5"
    assert (
        harness.repo / str(done[1].submission_file)
    ).read_text() == "weights target_temperature=2.0"
    assert first.duration_s is not None and first.duration_s >= 0
    assert first.jureca_job_id == "4242" and first.where == "jureca"


def test_session_login_happens_once_then_the_master_is_reused(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[2.0]")
    queue(harness, remote)
    logins = [c for c in fake_cluster.calls("ssh") if c["argv"][-1] == "true"]  # type: ignore[index]
    assert len(logins) == 1
    assert any("no open ssh session" in line for line in harness.out)
    harness.out.clear()
    remote.ensure_session()
    assert not harness.out, "a live master means no login message and no TOTP prompt"
    logins_after = [c for c in fake_cluster.calls("ssh") if c["argv"][-1] == "true"]  # type: ignore[index]
    assert len(logins_after) == 1


def test_missing_control_master_warns_that_totp_will_repeat(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_SSH_NO_CONTROLMASTER", "1")
    harness, remote = jureca_lab(tmp_path, fake_cluster)
    remote.ensure_session()
    assert any("ControlMaster is not configured" in line for line in harness.err)


def test_failed_login_is_a_typed_error(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_SSH_DOWN", "1")
    _harness, remote = jureca_lab(tmp_path, fake_cluster)
    with pytest.raises(RemoteError, match="login to 'juelich' failed"):
        remote.ensure_session()


def test_ssh_command_failure_carries_the_remote_stderr_and_exit_255_hint(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _harness, remote = jureca_lab(tmp_path, fake_cluster)
    with pytest.raises(RemoteError, match="exit 3.*boom"):
        remote.ssh("echo boom >&2; exit 3")
    monkeypatch.setenv("FAKE_SSH_DOWN", "1")
    with pytest.raises(RemoteError, match="TOTP expired"):
        remote.ssh("true --anything")


def test_missing_ssh_binary_is_explained(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _harness, remote = jureca_lab(tmp_path, fake_cluster)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(RemoteError, match="'ssh' not found in PATH"):
        remote.ensure_session()


def test_dry_run_prints_the_job_and_touches_neither_the_cluster_nor_the_disk(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[1.5, 2.0]")
    assert queue(harness, remote, dry_run=True, sync=True) == []
    assert fake_cluster.calls() == []
    assert any("#SBATCH --account=training2666" in line for line in harness.out)
    assert any(line.startswith("# remote command: ssh juelich") for line in harness.out)
    assert not harness.ctx.runs_dir.exists()
    assert harness.ctx.registry.snapshots() == []


def test_unsynced_code_fails_clearly_and_leaves_no_trace(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness, remote = jureca_lab(tmp_path, fake_cluster)
    with pytest.raises(RemoteError, match="code not synced: run jureca sync"):
        queue(harness, remote)
    assert harness.ctx.registry.snapshots() == []
    assert not list(harness.ctx.runs_dir.glob("unlearning/*")), (
        "failed submission must not leave runs"
    )
    assert fake_cluster.calls("sbatch") == []


def test_missing_task_venv_fails_clearly(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    (fake_cluster.remote_path(f"{TEAM_DIR}/Tempete/src/hackathon")).mkdir(parents=True)
    harness, remote = jureca_lab(tmp_path, fake_cluster)
    with pytest.raises(RemoteError, match="no .venv in the task folder"):
        queue(harness, remote)


def test_a_failing_variant_is_reported_failed_after_status_and_fetch(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_SBATCH_RUN", "1")
    fake_cluster.prepare_task()
    text = (
        'id = "exp-one"\ntask = "unlearning"\nidea = "boom"\nwhere = "jureca"\n'
        '[[steps]]\nargs = ["run", "--fail", "--out", "{out}/c.pt"]\n'
        '[expect]\nsubmission = "c.pt"\n'
    )
    adapter = build_adapter(fake_cluster.task_script)
    harness = make_lab(tmp_path, {adapter.name: adapter})
    write_spec(harness, "exp-one", text=text)
    remote = Remote(load_jureca_config(harness.ctx.env_loader()), harness.ctx)
    (queued,) = queue(harness, remote)
    (refreshed,) = refresh_status(harness.ctx, remote, [queued])
    assert refreshed.status == "failed" and refreshed.exit_code == 3
    fetched = fetch_run(harness.ctx, remote, refreshed)
    assert fetched.status == "failed" and "exited with code 3" in fetched.note


def test_variants_run_in_sequence_when_the_node_shows_fewer_gpus_than_variants(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Slurm starts isolating GPUs, pinning index 1 would crash: fall back to sequential."""
    monkeypatch.setenv("FAKE_SBATCH_RUN", "1")
    monkeypatch.setenv("FAKE_GPU_COUNT", "1")
    fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[1.5, 2.0]")
    queued = queue(harness, remote)
    run_root = fake_cluster.remote_path(f"{TEAM_DIR}/lab_runs/unlearning/{queued[0].group_id}")
    for variant in ("v1", "v2"):
        assert (run_root / variant / "exit_code").read_text().strip() == "0"
        assert "gpu=0" in (run_root / variant / "stdout.log").read_text()


def test_status_updates_only_what_changed_with_one_ssh_call_per_job(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[1.5, 2.0]")
    queued = queue(harness, remote)
    job = queued[0].jureca_job_id or ""
    lines_before = len(harness.ctx.registry.snapshots())

    def status_calls() -> int:
        return sum("squeue" in str(c["argv"][-1]) for c in fake_cluster.calls("ssh"))  # type: ignore[index]

    # PENDING: nothing changes, nothing is appended
    same = refresh_status(harness.ctx, remote, queued)
    assert [r.status for r in same] == ["queued", "queued"]
    assert len(harness.ctx.registry.snapshots()) == lines_before and status_calls() == 1
    # RUNNING
    fake_cluster.set_job_state(job, "RUNNING")
    running = refresh_status(harness.ctx, remote, queued)
    assert [r.status for r in running] == ["running", "running"] and status_calls() == 2
    assert len(harness.ctx.registry.snapshots()) == lines_before + 2
    # a settled run is not queried again
    done = [r for r in running]
    done[0] = harness.ctx.registry.update(done[0], status="done")
    refresh_status(harness.ctx, remote, done)
    assert status_calls() == 3, "still one call for the single job that has an active variant left"


def test_status_with_no_information_keeps_the_state_and_says_so(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[2.0]")
    (queued,) = queue(harness, remote)
    fake_cluster.set_job_state(queued.jureca_job_id or "", "")
    (same,) = refresh_status(harness.ctx, remote, [queued])
    assert same.status == "queued"
    assert any("not in squeue/sacct yet" in line for line in harness.err)


def test_fetch_refuses_an_unfinished_or_non_jureca_run(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[2.0]")
    (queued,) = queue(harness, remote)
    fake_cluster.set_job_state(queued.jureca_job_id or "", "RUNNING")
    with pytest.raises(RunError, match="still running"):
        fetch_run(harness.ctx, remote, queued)
    local = harness.ctx.registry.update(queued, where="local")
    with pytest.raises(RunError, match="not a JURECA run"):
        fetch_run(harness.ctx, remote, local)


def test_unreadable_job_id_is_reported(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[2.0]")
    (fake_cluster.state_dir / "bin" / "sbatch").write_text(
        "#!/bin/sh\ncat >/dev/null\necho 'queued!'\n"
    )
    with pytest.raises(RemoteError, match="printed no job id"):
        queue(harness, remote)


# ---- status parsing -------------------------------------------------------------------------


def test_parse_status_output_reads_the_three_sections() -> None:
    text = "RUNNING\n@@SACCT@@\nRUNNING\n@@EXIT@@\n/o/v1\t0\n/o/v2\tnone\n/o/v3\t-9\n"
    state = parse_status_output(text)
    assert state.squeue_state == "RUNNING" and state.sacct_state == "RUNNING"
    assert state.exit_codes == {"/o/v1": 0, "/o/v2": None, "/o/v3": -9}


def test_parse_status_output_handles_an_empty_queue() -> None:
    state = parse_status_output("@@SACCT@@\nCANCELLED by 1234\n@@EXIT@@\n/o\tnone\n")
    assert state.squeue_state is None and state.sacct_state == "CANCELLED"


@pytest.mark.parametrize(
    ("squeue", "sacct", "codes", "expected"),
    [
        (None, None, {"/o": 0}, ("finished", 0)),
        (
            "RUNNING",
            None,
            {"/o": 0},
            ("finished", 0),
        ),  # the variant's own verdict beats the job state
        (None, "FAILED", {"/o": 3}, ("failed", 3)),
        ("PENDING", None, {"/o": None}, ("queued", None)),
        ("CONFIGURING", None, {}, ("queued", None)),
        ("RUNNING", None, {"/o": None}, ("running", None)),
        ("COMPLETING", None, {}, ("running", None)),
        (None, "TIMEOUT", {"/o": None}, ("failed", None)),
        (None, "CANCELLED", {"/o": None}, ("failed", None)),
        (None, "OUT_OF_MEMORY", {}, ("failed", None)),
        (None, "COMPLETED", {"/o": None}, ("failed", None)),
        (None, None, {"/o": None}, (None, None)),
    ],
)
def test_derive_status_table(
    squeue: str | None,
    sacct: str | None,
    codes: dict[str, int | None],
    expected: tuple[str | None, int | None],
) -> None:
    status, _, code = derive_status(RemoteJobState(squeue, sacct, codes), "/o")
    assert (status, code) == expected


# ---- sync -----------------------------------------------------------------------------------


def build_source_tree(root: Path) -> None:
    files = {
        "src/hackathon/lab/a.py": "code",
        "src/hackathon/data/inner.txt": "nested data folder is code, keep it",
        "src/pkg/data.py": "keep",
        ".env": "SECRET=1",
        ".env.example": "KEY=",
        ".venv/lib/x": "venv",
        "data/big.npz": "data",
        "weights/w.bin": "w",
        "third_party/videoseal/z": "tp",
        "runs/unlearning/r/log": "r",
        "src/__pycache__/m.pyc": "c",
        "model.pt": "weights",
        ".git/HEAD": "ref",
        "docs/note.md": "doc",
    }
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_sync_sends_code_and_never_secrets_or_heavy_folders(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness, remote = jureca_lab(tmp_path, fake_cluster)
    shutil.rmtree(harness.repo)
    build_source_tree(harness.repo)
    remote.sync_repo(harness.repo)
    dest = fake_cluster.remote_path(f"{TEAM_DIR}/Tempete")
    sent = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert sent == {
        "src/hackathon/lab/a.py",
        "src/hackathon/data/inner.txt",
        "src/pkg/data.py",
        ".env.example",
        "docs/note.md",
    }
    rsync = fake_cluster.calls("rsync")[0]["argv"]
    assert rsync[0] == "-az" and "--delete" not in rsync
    assert rsync[-2] == f"{harness.repo}/" and rsync[-1] == f"juelich:{TEAM_DIR}/Tempete/"
    for excluded in ("/.venv", "/data", "/weights", "/third_party", "/runs", ".env"):
        assert f"--exclude={excluded}" in rsync


def test_sync_dry_run_lists_without_copying_and_failures_are_typed(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, remote = jureca_lab(tmp_path, fake_cluster)
    build_source_tree(harness.repo)
    assert "dry run listing" in remote.sync_repo(harness.repo, dry_run=True)
    assert "--dry-run" in fake_cluster.calls("rsync")[0]["argv"]  # type: ignore[operator]
    assert not (fake_cluster.remote_path(f"{TEAM_DIR}/Tempete") / "docs").exists()
    monkeypatch.setenv("FAKE_RSYNC_FAIL", "1")
    with pytest.raises(
        RemoteError, match="rsync failed with exit 12.*connection unexpectedly closed"
    ):
        remote.sync_repo(harness.repo)


def test_jureca_config_object_is_frozen() -> None:
    config = JurecaConfig("a", "b")
    with pytest.raises(AttributeError):
        config.team_lead = "c"  # type: ignore[misc]


def test_remote_calls_never_use_a_shell_for_the_local_side(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    """Every logged ssh/rsync call is an argv list whose last item is one single command string."""
    fake_cluster.prepare_task()
    harness, remote = jureca_lab(tmp_path, fake_cluster, sweep="[2.0]")
    queue(harness, remote, sync=True)
    for call in fake_cluster.calls():
        assert isinstance(call["argv"], list)
    assert json.dumps(fake_cluster.calls("ssh")[0]["argv"]).count("juelich") == 1

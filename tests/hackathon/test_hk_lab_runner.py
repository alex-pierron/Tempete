"""Tests of local execution: run directories, logs, registry lines, outputs, failure modes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from test_hk_lab_support import (
    FakeCluster,
    LabHarness,
    build_adapter,
    make_lab,
    write_spec,
)

from hackathon.common.submit import file_sha256
from hackathon.lab.errors import RunError
from hackathon.lab.registry import RunRecord
from hackathon.lab.runner import (
    build_step_argv,
    collect_outputs,
    git_state,
    placeholder_values,
    plan_runs,
    run_local,
    task_environment,
)
from hackathon.lab.spec import ExpectSpec, load_spec

SINGLE_SPEC = """
id = "plain"
task = "unlearning"
idea = "no sweep"
[[steps]]
name = "train"
args = ["run", "--out", "{out}/candidate.pt", "--set", "k=v"]
[expect]
submission = "candidate.pt"
metrics_file = "candidate.json"
primary_metric = "proxy.retain_kl"
"""

pytest_plugins = ("test_hk_lab_support",)  # the fake_cluster fixture


def lab_with_fake_task(tmp_path: Path, cluster: FakeCluster) -> LabHarness:
    adapter = build_adapter(cluster.task_script)
    return make_lab(tmp_path, {adapter.name: adapter})


def execute(harness: LabHarness, spec_id: str) -> list[RunRecord]:
    spec = load_spec(spec_id, harness.ctx.adapters, harness.ctx.experiments_dir)
    return run_local(harness.ctx, spec, harness.ctx.adapters[spec.task])


def test_a_successful_run_leaves_a_complete_trace(
    tmp_path: Path, fake_cluster: FakeCluster
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "plain", text=SINGLE_SPEC)
    (final,) = execute(harness, "plain")

    assert final.status == "done" and final.exit_code == 0 and final.where == "local"
    assert final.run_id.endswith("-unl-plain") and final.group_id == final.run_id
    assert final.metrics["proxy.retain_kl"] == 0.125
    assert final.metrics["proxy.movement"] == 2.0 and final.metrics["train_seconds"] == 1.5
    assert not {"label", "flag", "history"} & {key.split(".")[0] for key in final.metrics}
    assert final.primary_value == 0.125 and final.duration_s is not None
    run_dir = harness.ctx.run_dir("unlearning", final.run_id)
    assert final.submission_file == f"runs/unlearning/{final.run_id}/candidate.pt"
    assert final.submission_sha256 == file_sha256(run_dir / "candidate.pt")
    # registry: a `running` line first (an interrupted run leaves a trace), then the final one
    snapshots = harness.ctx.registry.snapshots()
    assert [s.status for s in snapshots] == ["running", "done"]
    assert (run_dir / "spec.toml").read_text() == SINGLE_SPEC
    assert "training started" in (run_dir / "stdout.log").read_text()
    assert (run_dir / "stdout.log").read_text().startswith("+ ")
    assert "a warning on stderr" in (run_dir / "stderr.log").read_text()
    assert "  | training started" in harness.out and "  | a warning on stderr" in harness.err


def test_run_json_records_how_to_reproduce_without_secrets(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HACKATHON_API_KEY", "super-secret-key")
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "plain", text=SINGLE_SPEC)
    (final,) = execute(harness, "plain")
    raw = (harness.ctx.run_dir("unlearning", final.run_id) / "run.json").read_text()
    meta = json.loads(raw)
    assert meta["sha"] == "unknown" and meta["dirty"] is False
    assert meta["where"] == "local" and meta["task"] == "unlearning"
    assert "k=v" in meta["commands"][0] and "{out}" not in meta["commands"][0]
    assert "super-secret-key" not in raw


def test_local_run_passes_local_args_and_no_remote_global_args(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "exp-one", sweep="[2.0]")
    (final,) = execute(harness, "exp-one")
    calls = [
        json.loads(line)
        for line in (fake_cluster.state_dir / "task_calls.jsonl").read_text().splitlines()
    ]
    argv = calls[0]
    assert argv[0] == "run" and "--task-dir" not in argv
    assert argv[argv.index("--set") + 1] == "target_temperature=2.0"
    assert argv[argv.index("--config") + 1] == "smoke" and "--proxy-root" not in argv
    assert final.run_id.endswith("-v1") and final.params == {"target_temperature": 2.0}


def test_a_sweep_runs_each_variant_in_its_own_directory(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "exp-one", sweep="[1.5, 2.0]")
    first, second = execute(harness, "exp-one")
    assert first.group_id == second.group_id and first.run_id != second.run_id
    assert (first.variant, second.variant) == ("v1", "v2")
    assert first.run_id == f"{first.group_id}-v1"
    paths = [harness.ctx.repo_root / str(r.submission_file) for r in (first, second)]
    assert paths[0].read_text() == "weights target_temperature=1.5"
    assert paths[1].read_text() == "weights target_temperature=2.0"
    assert {r.status for r in harness.ctx.registry.latest().values()} == {"done"}


def test_a_failing_step_stops_its_variant_and_is_recorded(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    text = (
        SINGLE_SPEC.replace('"--set", "k=v"]', '"--fail"]')
        + '\n[[steps]]\nargs = ["run", "--out", "{out}/second.pt"]\n'
    )
    write_spec(harness, "plain", text=text)
    (final,) = execute(harness, "plain")
    assert final.status == "failed" and final.exit_code == 3
    assert "exited with code 3" in final.note
    task_calls = (fake_cluster.state_dir / "task_calls.jsonl").read_text().splitlines()
    assert len(task_calls) == 1, "the second step must not run after a failure"
    assert final.submission_file is None


def test_a_missing_expected_file_makes_the_run_failed(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    text = SINGLE_SPEC.replace('"--set", "k=v"]', '"--no-sidecar"]')
    write_spec(harness, "plain", text=text)
    (final,) = execute(harness, "plain")
    assert final.status == "failed" and final.exit_code == 0
    assert "expected metrics file not found: candidate.json" in final.note


def test_unreadable_metric_is_a_warning_not_a_failure(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    text = (
        SINGLE_SPEC.replace("proxy.retain_kl", "proxy.typo")
        + "[expect.metrics_regex]\nnope = 'NEVER ([0-9]+)'\n"
    )
    write_spec(harness, "plain", text=text)
    (final,) = execute(harness, "plain")
    assert final.status == "done", "a good submission is not thrown away over a metric name typo"
    assert "primary metric 'proxy.typo' not found" in final.note
    assert "metric 'nope'" in final.note
    assert final.primary_value is None


def test_regex_metrics_take_the_last_match_of_the_logs(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "exp-one", sweep="[2.0]")
    (final,) = execute(harness, "exp-one")
    assert final.metrics["mcc"] == 0.75


def test_an_unstartable_executable_is_a_typed_error_and_recorded(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    harness.ctx.python = str(tmp_path / "no-such-python")
    write_spec(harness, "plain", text=SINGLE_SPEC)
    with pytest.raises(RunError, match="cannot start"):
        execute(harness, "plain")
    (record,) = harness.ctx.registry.latest().values()
    assert record.status == "failed" and "cannot start" in record.note


def test_ctrl_c_marks_the_run_failed_and_propagates(
    tmp_path: Path,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "plain", text=SINGLE_SPEC)

    def interrupt(*args: object, **kwargs: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("hackathon.lab.runner.run_command", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(harness, "plain")
    (record,) = harness.ctx.registry.latest().values()
    assert record.status == "failed" and "interrupted" in record.note


def test_two_launches_in_the_same_second_never_overwrite_a_run(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    frozen = harness.clock()
    harness.ctx.clock = lambda: frozen
    write_spec(harness, "plain", text=SINGLE_SPEC)
    execute(harness, "plain")
    with pytest.raises(RunError, match="already exists"):
        execute(harness, "plain")


def test_plan_runs_ids_and_group(tmp_path: Path, fake_cluster: FakeCluster) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "plain", text=SINGLE_SPEC)
    write_spec(harness, "exp-one", sweep="[1, 2, 3]")
    adapter = harness.ctx.adapters["unlearning"]
    now = harness.clock()
    single = load_spec("plain", harness.ctx.adapters, harness.ctx.experiments_dir)
    swept = load_spec("exp-one", harness.ctx.adapters, harness.ctx.experiments_dir)
    (only,) = plan_runs(single, adapter, now)
    assert only.run_id == only.group_id and only.run_id.endswith("Z-unl-plain")
    many = plan_runs(swept, adapter, now)
    assert [p.run_id.rsplit("-", 1)[1] for p in many] == ["v1", "v2", "v3"]
    assert len({p.group_id for p in many}) == 1


def test_build_step_argv_places_remote_globals_before_the_subcommand(
    tmp_path: Path,
    fake_cluster: FakeCluster,
) -> None:
    harness = lab_with_fake_task(tmp_path, fake_cluster)
    write_spec(harness, "exp-one", sweep="[1.5]")
    adapter = harness.ctx.adapters["unlearning"]
    spec = load_spec("exp-one", harness.ctx.adapters, harness.ctx.experiments_dir)
    (planned,) = plan_runs(spec, adapter, harness.clock())
    step = spec.steps[0]
    extra = {"task_dir": "/t", "team_dir": "/team"}
    remote = build_step_argv(
        adapter, step, placeholder_values(planned, "/o", extra=extra), where="jureca"
    )
    script = str(fake_cluster.task_script)
    assert remote[:3] == (script, "--task-dir", "/t")
    assert remote[3] == "run" and remote[-4:] == (
        "--config",
        "full",
        "--proxy-root",
        "/t/proxy_images",
    )
    local = build_step_argv(adapter, step, placeholder_values(planned, "/o"), where="local")
    assert local[:2] == (script, "run") and local[-2:] == ("--config", "smoke")


def test_task_environment_prepends_the_repo_src(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/already")
    env = task_environment(tmp_path)
    assert env["PYTHONPATH"] == f"{tmp_path / 'src'}{os.pathsep}/already"
    assert env["PYTHONUNBUFFERED"] == "1"
    monkeypatch.delenv("PYTHONPATH")
    assert task_environment(tmp_path)["PYTHONPATH"] == str(tmp_path / "src")


def test_collect_outputs_flattens_numbers_only(tmp_path: Path) -> None:
    (tmp_path / "m.json").write_text(
        json.dumps(
            {"a": {"b": 1, "c": [1, 2]}, "ok": True, "name": "x", "nan": float("nan"), "z": 2.5}
        )
    )
    collected = collect_outputs(tmp_path, ExpectSpec(metrics_file="m.json"))
    assert collected.metrics == {"a.b": 1.0, "z": 2.5}
    assert not collected.problems


def test_collect_outputs_reports_bad_json_and_nothing_expected(tmp_path: Path) -> None:
    (tmp_path / "m.json").write_text("{not json")
    collected = collect_outputs(tmp_path, ExpectSpec(metrics_file="m.json", submission="s.pt"))
    assert len(collected.problems) == 2
    assert collect_outputs(tmp_path, ExpectSpec()) == collect_outputs(tmp_path, ExpectSpec())
    assert collect_outputs(tmp_path, ExpectSpec()).problems == ()


def test_git_state_outside_a_repository_and_in_one(tmp_path: Path) -> None:
    assert git_state(tmp_path) == ("unknown", False)
    repo = tmp_path / "repo"
    repo.mkdir()
    base = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    (repo / "f.txt").write_text("1")
    for command in (["init", "-q"], ["add", "f.txt"], ["commit", "-q", "-m", "c"]):
        subprocess.run([*base, *command], cwd=repo, check=True, env=env, capture_output=True)
    sha, dirty = git_state(repo)
    assert len(sha) == 40 and dirty is False
    (repo / "f.txt").write_text("2")
    assert git_state(repo) == (sha, True)

"""Tests of the command line: the full journey idea -> run -> board -> submit, local and JURECA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_hk_lab_support import (
    TEAM_DIR,
    FakeCluster,
    LabHarness,
    build_adapter,
    make_lab,
    write_spec,
)

from hackathon.lab.cli import EXIT_FAILURE, EXIT_OK, EXIT_REFUSED, build_parser, main

pytest_plugins = ("test_hk_lab_support",)  # the fake_cluster fixture


@pytest.fixture
def lab(tmp_path: Path, fake_cluster: FakeCluster) -> LabHarness:
    adapter = build_adapter(fake_cluster.task_script)
    return make_lab(tmp_path, {adapter.name: adapter})


def run_cli(harness: LabHarness, *argv: str) -> int:
    return main(list(argv), ctx=harness.ctx)


def test_full_local_journey_from_new_to_a_recorded_public_score(
    lab: LabHarness,
    fake_cluster: FakeCluster,
) -> None:
    assert run_cli(lab, "new", "--task", "unlearning", "--idea", "Gold soft") == EXIT_OK
    spec_path = lab.repo / "experiments" / "unlearning" / "gold-soft.toml"
    assert spec_path.is_file() and spec_path.with_suffix(".md").is_file()
    assert any("make hk-run SPEC=gold-soft" in line for line in lab.out)

    # the template targets the real unlearning CLI; point it at the fake task and run locally
    text = (
        spec_path.read_text(encoding="utf-8")
        .replace('"gold_soft"', '"run"')
        .replace('args = ["run", "--method", "run",', 'args = ["run",')
    )
    spec_path.write_text(text, encoding="utf-8")
    lab.out.clear()
    assert run_cli(lab, "run", "gold-soft") == EXIT_OK
    (record,) = lab.ctx.registry.latest().values()
    assert record.status == "done" and record.public_score is None

    lab.out.clear()
    assert run_cli(lab, "board") == EXIT_OK
    assert (
        any(record.run_id in line for line in lab.out)
        and (lab.repo / "runs" / "board.html").is_file()
    )

    # no --yes: validation only, exit 2, nothing sent
    assert run_cli(lab, "submit", record.run_id) == EXIT_REFUSED
    log = fake_cluster.state_dir / "task_calls.jsonl"
    assert "submit" not in [json.loads(line)[0] for line in log.read_text().splitlines()]

    assert run_cli(lab, "submit", record.run_id, "--yes") == EXIT_OK
    assert lab.ctx.registry.get(record.run_id).public_score == 0.42
    assert any("public_score=0.42" in line for line in lab.out)


def test_errors_print_one_line_and_exit_1(lab: LabHarness) -> None:
    assert run_cli(lab, "run", "ghost") == EXIT_FAILURE
    assert any(line.startswith("error: no spec 'ghost'") for line in lab.err)
    assert run_cli(lab, "new", "--task", "chess", "--idea", "x") == EXIT_FAILURE
    assert run_cli(lab, "submit", "nope", "--yes") == EXIT_FAILURE


def test_a_failed_run_exits_1_and_the_board_shows_it(lab: LabHarness) -> None:
    write_spec(
        lab, "boom", text=(
            'id = "boom"\ntask = "unlearning"\nidea = "boom"\n'
            '[[steps]]\nargs = ["run", "--fail", "--out", "{out}/c.pt"]\n'
        ),
    )  # fmt: skip
    assert run_cli(lab, "run", "boom") == EXIT_FAILURE
    lab.out.clear()
    run_cli(lab, "board", "--task", "unlearning")
    assert any("failed" in line for line in lab.out)


def test_where_flag_overrides_the_spec_and_unknown_choices_are_rejected(lab: LabHarness) -> None:
    write_spec(lab, "exp-one", where="jureca", sweep="[2.0]")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "x", "--where", "mars"])
    # where=jureca in the spec but --where local on the command line: it runs here
    assert run_cli(lab, "run", "exp-one", "--where", "local") == EXIT_OK
    (record,) = lab.ctx.registry.latest().values()
    assert record.where == "local" and record.status == "done"


def test_local_commands_never_read_the_jureca_environment(lab: LabHarness) -> None:
    lab.ctx.env_loader = lambda: pytest.fail("the environment must stay lazy for local commands")  # type: ignore[assignment]
    write_spec(lab, "exp-one", sweep="[2.0]")
    assert run_cli(lab, "run", "exp-one") == EXIT_OK
    assert run_cli(lab, "board") == EXIT_OK


def test_jureca_commands_need_the_team_variables(lab: LabHarness) -> None:
    lab.ctx.env_loader = dict
    assert run_cli(lab, "jureca", "sync", "--dry-run") == EXIT_FAILURE
    assert any("JURECA_TEAM_LEAD is not set" in line for line in lab.err)


def test_full_jureca_journey_login_sync_submit_status_fetch(
    lab: LabHarness,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_SBATCH_RUN", "1")
    fake_cluster.prepare_task()
    (lab.repo / "src" / "hackathon").mkdir(parents=True)
    (lab.repo / "src" / "hackathon" / "a.py").write_text("x = 1\n")
    write_spec(lab, "exp-one", where="jureca", sweep="[1.5, 2.0]")

    assert run_cli(lab, "jureca", "login") == EXIT_OK
    assert any("ssh session to 'juelich' is open" in line for line in lab.out)
    assert run_cli(lab, "jureca", "sync") == EXIT_OK
    assert (fake_cluster.remote_path(TEAM_DIR) / "Tempete" / "src" / "hackathon" / "a.py").is_file()

    lab.out.clear()
    assert (
        run_cli(lab, "run", "exp-one", "--no-sync") == EXIT_OK
    )  # where=jureca comes from the spec
    assert sum("queued as Slurm job 4242" in line for line in lab.out) == 2
    runs = list(lab.ctx.registry.latest().values())
    assert [r.status for r in runs] == ["queued", "queued"]

    lab.out.clear()
    assert run_cli(lab, "jureca", "status") == EXIT_OK
    assert sorted(r.status for r in lab.ctx.registry.latest().values()) == ["finished", "finished"]

    lab.out.clear()
    group = runs[0].group_id
    assert run_cli(lab, "jureca", "fetch", group) == EXIT_OK
    final = list(lab.ctx.registry.latest().values())
    assert [r.status for r in final] == ["done", "done"]
    assert sum("candidate.pt" in line for line in lab.out) == 2

    assert run_cli(lab, "jureca", "status") == EXIT_OK
    assert lab.out[-1] == "no pending JURECA run."

    # a group id is ambiguous for submit (two variants): the lab asks for one run id
    assert run_cli(lab, "submit", group) == EXIT_FAILURE
    assert any("several runs, pick one" in line for line in lab.err)
    assert run_cli(lab, "submit", runs[1].run_id, "--yes") == EXIT_OK
    assert lab.ctx.registry.get(runs[1].run_id).public_score == 0.42


def test_run_where_jureca_dry_run_sends_nothing(lab: LabHarness, fake_cluster: FakeCluster) -> None:
    write_spec(lab, "exp-one", where="jureca", sweep="[1.5, 2.0]")
    assert run_cli(lab, "run", "exp-one", "--dry-run") == EXIT_OK
    assert fake_cluster.calls() == []
    assert any("#SBATCH --reservation=cispahack" in line for line in lab.out)
    assert run_cli(lab, "jureca", "submit", "exp-one", "--dry-run") == EXIT_OK
    assert fake_cluster.calls() == []


def test_jureca_sync_dry_run_is_read_only_on_the_cluster(
    lab: LabHarness, fake_cluster: FakeCluster
) -> None:
    (lab.repo / "f.txt").write_text("x")
    assert run_cli(lab, "jureca", "sync", "--dry-run") == EXIT_OK
    ssh_commands = [str(c["argv"][-1]) for c in fake_cluster.calls("ssh")]  # type: ignore[index]
    assert not any(command.startswith("mkdir") for command in ssh_commands)
    assert not fake_cluster.remote_path(TEAM_DIR).exists() or not any(
        fake_cluster.remote_path(TEAM_DIR).rglob("f.txt")
    )
    assert "--dry-run" in fake_cluster.calls("rsync")[0]["argv"]  # type: ignore[operator]
    assert "dry run" in lab.out[-1]


def test_there_is_no_submit_in_the_makefile_without_an_explicit_gesture() -> None:
    """The Makefile shortcuts may validate and look, never spend the ep budget or submit."""
    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text(encoding="utf-8")
    section = makefile.split("# --- Hackathon", 1)[1]
    recipe_lines = [ln for ln in section.splitlines() if ln.startswith("\t")]
    assert recipe_lines, "the Hackathon section must define recipes"
    for line in recipe_lines:
        assert "--yes" not in line and "query-ep" not in line, line

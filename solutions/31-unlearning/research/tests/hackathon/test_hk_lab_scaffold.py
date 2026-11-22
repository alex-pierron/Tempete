"""Tests of `lab new`: spec and idea sheet from the shipped templates."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_hk_lab_support import make_lab

from hackathon.lab.errors import LabError, SpecError, UnknownTaskError
from hackathon.lab.scaffold import create_experiment
from hackathon.lab.spec import load_spec
from hackathon.lab.tasks import DEFAULT_ADAPTERS


@pytest.mark.parametrize("task", ["unlearning", "model_equality", "wmloc"])
def test_every_shipped_template_makes_a_valid_spec(tmp_path: Path, task: str) -> None:
    harness = make_lab(tmp_path, DEFAULT_ADAPTERS)
    created = create_experiment(harness.ctx, task, "Gold Soft T=2")
    assert created.spec_id == "gold-soft-t-2"
    assert created.spec_path == harness.repo / "experiments" / task / "gold-soft-t-2.toml"
    spec = load_spec("gold-soft-t-2", DEFAULT_ADAPTERS, harness.ctx.experiments_dir)
    assert spec.task == task and spec.idea == "Gold Soft T=2"
    assert spec.where == "local" and spec.steps


def test_idea_sheet_has_the_four_sections_and_the_id(tmp_path: Path) -> None:
    harness = make_lab(tmp_path, DEFAULT_ADAPTERS)
    created = create_experiment(harness.ctx, "unl", "Température 2")
    sheet = created.idea_path.read_text(encoding="utf-8")
    assert created.spec_id == "temperature-2"
    assert sheet.startswith("# Température 2")
    for heading in ("## Hypothese", "## Metrique locale attendue", "## Resultat", "## Decision"):
        assert heading in sheet
    assert "`temperature-2`" in sheet and "2026-11-21" in sheet
    assert "{{" not in sheet


def test_unlearning_template_runs_auto_locally_full_on_jureca_and_is_benched(
    tmp_path: Path,
) -> None:
    harness = make_lab(tmp_path, DEFAULT_ADAPTERS)
    create_experiment(harness.ctx, "unlearning", "demo")
    spec = load_spec("demo", DEFAULT_ADAPTERS, harness.ctx.experiments_dir)
    step = spec.steps[0]
    assert step.args_local == ("--config", "auto")
    assert step.args_jureca[:2] == ("--config", "full")
    assert "holdout_fraction=0.2" in step.args
    assert spec.expect.primary_metric == "bench.score_min" and spec.expect.higher_is_better


def test_quotes_and_backslashes_in_the_idea_stay_valid_toml(tmp_path: Path) -> None:
    harness = make_lab(tmp_path, DEFAULT_ADAPTERS)
    create_experiment(harness.ctx, "unlearning", 'say "hi" \\ done')
    spec = load_spec("say-hi-done", DEFAULT_ADAPTERS, harness.ctx.experiments_dir)
    assert spec.idea == 'say "hi" \\ done'


def test_existing_id_is_never_overwritten_even_across_tasks(tmp_path: Path) -> None:
    harness = make_lab(tmp_path, DEFAULT_ADAPTERS)
    first = create_experiment(harness.ctx, "unlearning", "dup")
    before = first.spec_path.read_text(encoding="utf-8")
    with pytest.raises(SpecError, match="already exists"):
        create_experiment(harness.ctx, "unlearning", "dup")
    with pytest.raises(SpecError, match="already exists"):
        create_experiment(harness.ctx, "model_equality", "dup")
    assert first.spec_path.read_text(encoding="utf-8") == before


def test_idea_without_letters_is_refused(tmp_path: Path) -> None:
    harness = make_lab(tmp_path, DEFAULT_ADAPTERS)
    with pytest.raises(SpecError, match="empty id"):
        create_experiment(harness.ctx, "unlearning", "***")


def test_unknown_task_and_missing_template(tmp_path: Path) -> None:
    harness = make_lab(tmp_path, DEFAULT_ADAPTERS)
    with pytest.raises(UnknownTaskError):
        create_experiment(harness.ctx, "chess", "x")
    (harness.ctx.experiments_dir / "_templates" / "idea.md").unlink()
    with pytest.raises(LabError, match="template missing"):
        create_experiment(harness.ctx, "unlearning", "x")
    assert not (harness.ctx.experiments_dir / "unlearning" / "x.toml").exists()

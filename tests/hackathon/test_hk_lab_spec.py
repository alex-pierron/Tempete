"""Tests of the experiment spec: parsing, validation, variants, placeholders, guard-rails."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackathon.lab.errors import SpecError, UnknownTaskError
from hackathon.lab.spec import (
    ExperimentSpec,
    load_spec,
    param_text,
    parse_spec,
    render_args,
    resolve_spec_path,
    slugify,
)
from hackathon.lab.tasks import DEFAULT_ADAPTERS

MINIMAL = """
id = "exp"
task = "unlearning"
idea = "an idea"
[[steps]]
args = ["run", "--out", "{out}/a.pt"]
"""


def parse(text: str, path: Path | None = None) -> ExperimentSpec:
    return parse_spec(text, DEFAULT_ADAPTERS, path)


def with_extra(extra: str) -> str:
    """MINIMAL plus top-level or table lines (tables go AFTER the steps so TOML stays valid)."""
    return MINIMAL + "\n" + extra


def test_minimal_spec_gets_defaults() -> None:
    spec = parse(MINIMAL)
    assert (spec.id, spec.task, spec.where, spec.gpus) == ("exp", "unlearning", "local", 1)
    assert spec.steps[0].name == "step1"
    assert spec.expect.submission is None and spec.expect.higher_is_better is True
    variants = spec.variants()
    assert len(variants) == 1 and variants[0].name == "v1" and variants[0].label == ""


def test_task_aliases_resolve_to_the_canonical_name() -> None:
    assert parse(MINIMAL.replace('"unlearning"', '"UNL"')).task == "unlearning"
    met = MINIMAL.replace('"unlearning"', '"met"').replace('"run"', '"analyze"')
    assert parse(met).task == "model_equality"


def test_unknown_task_is_a_typed_error() -> None:
    with pytest.raises(UnknownTaskError, match="expected one of"):
        parse(MINIMAL.replace('"unlearning"', '"nope"'))


def test_sweep_expands_to_cartesian_product_in_declaration_order() -> None:
    text = with_extra("[sweep]\nlr = [1, 2]\ntemp = [0.5, 1.5]\n")
    spec = parse(text)
    variants = spec.variants()
    assert [v.name for v in variants] == ["v1", "v2", "v3", "v4"]
    assert [dict(v.params) for v in variants] == [
        {"lr": 1, "temp": 0.5},
        {"lr": 1, "temp": 1.5},
        {"lr": 2, "temp": 0.5},
        {"lr": 2, "temp": 1.5},
    ]
    assert variants[1].label == "lr=1,temp=1.5"
    assert spec.gpus == 4


def test_more_than_four_variants_is_refused_with_the_reason() -> None:
    with pytest.raises(SpecError, match="at most 4"):
        parse(with_extra("[sweep]\nlr = [1, 2, 3, 4, 5]\n"))


@pytest.mark.parametrize("gpus", [0, 5])
def test_gpus_out_of_range_is_refused(gpus: int) -> None:
    with pytest.raises(SpecError, match="between 1 and 4"):
        parse(with_extra(f"[resources]\ngpus = {gpus}\n"))


def test_gpus_below_variant_count_is_refused() -> None:
    with pytest.raises(SpecError, match="below the 3 sweep variants"):
        parse(with_extra("[sweep]\nlr = [1, 2, 3]\n[resources]\ngpus = 2\n"))


def test_time_limit_is_validated() -> None:
    assert parse(with_extra('[resources]\ntime_limit = "01:30:00"\n')).time_limit == "01:30:00"
    assert parse(with_extra('[resources]\ntime_limit = "1-02:00:00"\n')).time_limit == "1-02:00:00"
    with pytest.raises(SpecError, match="time_limit"):
        parse(with_extra('[resources]\ntime_limit = "soon"\n'))


@pytest.mark.parametrize("token", ["submit", "--yes", "query-ep"])
@pytest.mark.parametrize("field", ["args", "args_local", "args_jureca"])
def test_budget_and_submission_tokens_are_forbidden_everywhere(token: str, field: str) -> None:
    text = MINIMAL.replace(
        'args = ["run", "--out", "{out}/a.pt"]',
        f'args = ["run", "--out", "{{out}}/a.pt"]\n{field} = ["{token}"]'
        if field != "args"
        else f'args = ["run", "{token}"]',
    )
    with pytest.raises(SpecError, match="forbidden"):
        parse(text)


def test_forbidden_token_arriving_through_a_sweep_value_is_caught_at_render() -> None:
    with pytest.raises(SpecError, match="after substitution"):
        render_args(["{x}"], {"x": "--yes"}, allow_jureca_only=False)


def test_step_must_name_a_known_subcommand() -> None:
    with pytest.raises(SpecError, match="no known unlearning subcommand"):
        parse(MINIMAL.replace('["run", ', '["fly", '))


def test_args_must_be_a_list_of_strings_never_a_shell_line() -> None:
    with pytest.raises(SpecError, match="list of strings"):
        parse(MINIMAL.replace('args = ["run", "--out", "{out}/a.pt"]', 'args = "run --out x"'))
    with pytest.raises(SpecError, match="list of strings"):
        parse(MINIMAL.replace('"--out", "{out}/a.pt"', "1, 2"))


def test_steps_are_required() -> None:
    with pytest.raises(SpecError, match="at least one"):
        parse('id = "exp"\ntask = "unlearning"\nidea = "x"\n')


def test_unknown_keys_are_refused_at_every_level() -> None:
    with pytest.raises(SpecError, match="unknown key"):
        parse('colour = "red"\n' + MINIMAL)
    with pytest.raises(SpecError, match="unknown key"):
        parse(MINIMAL.replace('args = ["run"', 'argz = 1\nargs = ["run"'))
    with pytest.raises(SpecError, match="unknown key"):
        parse(with_extra("[expect]\nsubmision = 'x'\n"))


def test_unknown_placeholder_is_caught_at_parse_time() -> None:
    with pytest.raises(SpecError, match="unknown placeholder"):
        parse(MINIMAL.replace("{out}/a.pt", "{nope}/a.pt"))


def test_jureca_only_placeholders_are_valid_only_in_args_jureca() -> None:
    with pytest.raises(SpecError, match="unknown placeholder"):
        parse(MINIMAL.replace("{out}/a.pt", "{task_dir}/a.pt"))
    ok = MINIMAL.replace(
        'args = ["run", "--out", "{out}/a.pt"]',
        'args = ["run"]\nargs_jureca = ["--proxy-root", "{task_dir}/p"]',
    )
    assert parse(ok).steps[0].args_jureca == ("--proxy-root", "{task_dir}/p")


def test_sweep_keys_become_placeholders_and_cannot_shadow_builtins() -> None:
    spec = parse(MINIMAL.replace("{out}/a.pt", "{lr}") + "\n[sweep]\nlr = [1]\n")
    assert spec.sweep == {"lr": (1,)}
    with pytest.raises(SpecError, match="collides"):
        parse(with_extra("[sweep]\nout = [1]\n"))
    with pytest.raises(SpecError, match="identifier"):
        parse(with_extra('[sweep]\n"bad key" = [1]\n'))
    with pytest.raises(SpecError, match="non-empty list"):
        parse(with_extra("[sweep]\nlr = []\n"))


@pytest.mark.parametrize("value", ["/etc/passwd", "../up.pt", "a/../../b"])
def test_expected_files_must_stay_inside_the_run_directory(value: str) -> None:
    with pytest.raises(SpecError):
        parse(with_extra(f'[expect]\nsubmission = "{value}"\n'))


def test_metrics_regex_needs_a_valid_pattern_with_a_group() -> None:
    with pytest.raises(SpecError, match="capture group"):
        parse(with_extra("[expect.metrics_regex]\nm = 'no group'\n"))
    with pytest.raises(SpecError, match="not a valid regex"):
        parse(with_extra("[expect.metrics_regex]\nm = '(unclosed'\n"))


def test_where_and_id_rules() -> None:
    assert parse(MINIMAL.replace("idea =", 'where = "jureca"\nidea =')).where == "jureca"
    with pytest.raises(SpecError, match="where must be"):
        parse(MINIMAL.replace("idea =", 'where = "mars"\nidea ='))
    with pytest.raises(SpecError, match="lowercase slug"):
        parse(MINIMAL.replace('id = "exp"', 'id = "Bad Id"'))


def test_id_must_match_the_file_name(tmp_path: Path) -> None:
    path = tmp_path / "other.toml"
    with pytest.raises(SpecError, match="must equal the file name 'other'"):
        parse(MINIMAL, path)
    assert parse(MINIMAL, tmp_path / "exp.toml").path == tmp_path / "exp.toml"


def test_malformed_toml_is_a_spec_error() -> None:
    with pytest.raises(SpecError, match="invalid TOML"):
        parse("id = = 1")


def test_errors_name_the_file(tmp_path: Path) -> None:
    path = tmp_path / "exp.toml"
    with pytest.raises(SpecError, match=str(path)):
        parse(MINIMAL.replace('"run"', '"fly"'), path)


def test_render_args_single_pass_values_and_booleans() -> None:
    values = {"out": "/o", "flag": param_text(True), "ratio": param_text(2.0)}
    assert render_args(
        ["--out", "{out}/x", "--f={flag}", "{ratio}"], values, allow_jureca_only=False
    ) == (
        "--out",
        "/o/x",
        "--f=true",
        "2.0",
    )
    # braces that are not placeholders (JSON) and substituted text are never re-scanned.
    assert render_args(['{"a": 1}', "{out}"], {"out": "{ratio}"}, allow_jureca_only=False) == (
        '{"a": 1}',
        "{ratio}",
    )


def test_render_args_jureca_only_placeholder_fails_locally() -> None:
    with pytest.raises(SpecError, match="only exists on JURECA"):
        render_args(["{task_dir}"], {}, allow_jureca_only=False)
    assert render_args(["{task_dir}"], {"task_dir": "/t"}, allow_jureca_only=True) == ("/t",)


def test_slugify() -> None:
    assert slugify("Gold Soft, T=2") == "gold-soft-t-2"
    assert slugify("  Éléphant à l'ombre ") == "elephant-a-l-ombre"
    assert slugify("!!!") == ""
    assert len(slugify("a" * 200)) == 48


def test_resolve_spec_by_path_and_by_id(tmp_path: Path) -> None:
    folder = tmp_path / "experiments" / "unlearning"
    folder.mkdir(parents=True)
    spec_file = folder / "exp.toml"
    spec_file.write_text(MINIMAL, encoding="utf-8")
    assert resolve_spec_path(str(spec_file), tmp_path / "experiments") == spec_file
    assert resolve_spec_path("exp", tmp_path / "experiments") == spec_file
    assert resolve_spec_path("exp.toml", tmp_path / "experiments") == spec_file
    assert load_spec("exp", DEFAULT_ADAPTERS, tmp_path / "experiments").id == "exp"
    with pytest.raises(SpecError, match="no spec 'ghost'"):
        resolve_spec_path("ghost", tmp_path / "experiments")


def test_ambiguous_id_across_tasks_is_refused(tmp_path: Path) -> None:
    for task in ("unlearning", "model_equality"):
        folder = tmp_path / "experiments" / task
        folder.mkdir(parents=True)
        (folder / "same.toml").write_text(MINIMAL, encoding="utf-8")
    with pytest.raises(SpecError, match="ambiguous"):
        resolve_spec_path("same", tmp_path / "experiments")

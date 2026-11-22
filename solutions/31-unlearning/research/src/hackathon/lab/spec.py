"""Experiment specification: parse, validate and expand a `experiments/<task>/<id>.toml`.

A spec names the task, the idea and the hypothesis, the task CLI steps to run, an optional
parameter sweep (at most four variants: one per A100 of a JURECA node) and the files the run must
leave behind. Everything is checked when the spec is read, so a typo fails in milliseconds on the
laptop rather than after a queued cluster job.

TOML shape (see experiments/_templates/):

    id = "gold-soft-t2"              # = the file name; lowercase slug
    task = "unlearning"
    idea = "gold soft, temperature 2"
    hypothesis = "..."
    where = "local"                  # or "jureca"; `run --where` overrides it

    [resources]                      # optional
    gpus = 2                         # >= number of variants, <= 4
    time_limit = "01:00:00"          # Slurm --time, optional

    [sweep]                          # optional: cartesian product, at most 4 variants
    target_temperature = [1.5, 2.0]

    [[steps]]
    name = "train"
    args = ["run", "--method", "gold_soft", "--out", "{out}/candidate.pt",
            "--set", "target_temperature={target_temperature}"]
    args_local = []                  # appended on `local` only
    args_jureca = []                 # appended on `jureca` only (may use {task_dir}, {team_dir})

    [expect]
    submission = "candidate.pt"      # relative to the run output directory
    metrics_file = "candidate.json"  # JSON; numeric leaves become metrics (dotted keys)
    primary_metric = "proxy.score"   # the local metric used to rank runs on the board
    higher_is_better = true
    [expect.metrics_regex]           # optional: float captured from the step logs
    mcc = 'MCC[^:]*: ([0-9.]+)'
"""

from __future__ import annotations

import itertools
import re
import tomllib
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from hackathon.lab.constants import (
    COMMON_PLACEHOLDERS,
    FORBIDDEN_TOKENS,
    JURECA_ONLY_PLACEHOLDERS,
    MAX_VARIANTS,
    MIN_GPUS,
    SLUG_MAX_LENGTH,
    SLURM_TIME_PATTERN,
    WHERE_CHOICES,
    WHERE_LOCAL,
)
from hackathon.lab.errors import SpecError
from hackathon.lab.tasks import TaskAdapter, resolve_adapter

ParamValue = str | int | float | bool
SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
TOP_LEVEL_KEYS = frozenset(
    {"id", "task", "idea", "hypothesis", "where", "resources", "sweep", "steps", "expect"}
)
STEP_KEYS = frozenset({"name", "args", "args_local", "args_jureca"})
EXPECT_KEYS = frozenset(
    {"submission", "metrics_file", "primary_metric", "higher_is_better", "metrics_regex"}
)
RESOURCE_KEYS = frozenset({"gpus", "time_limit"})
SLURM_TIME_RE = re.compile(SLURM_TIME_PATTERN)


@dataclass(frozen=True)
class StepSpec:
    """One task CLI invocation: `python -m <task module> [global args] <args> [where args]`."""

    name: str
    args: tuple[str, ...]
    args_local: tuple[str, ...] = ()
    args_jureca: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpectSpec:
    """What a finished run must leave in its output directory, and how to read its metrics."""

    submission: str | None = None
    metrics_file: str | None = None
    primary_metric: str | None = None
    higher_is_better: bool = True
    metrics_regex: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Variant:
    """One point of the sweep (or the single implicit variant when there is no sweep)."""

    index: int
    name: str
    params: Mapping[str, ParamValue]

    @property
    def label(self) -> str:
        """Human label of the parameters, e.g. `target_temperature=1.5`; empty without sweep."""
        return ",".join(f"{key}={value}" for key, value in self.params.items())


@dataclass(frozen=True)
class ExperimentSpec:
    """A validated experiment specification."""

    id: str
    task: str
    idea: str
    hypothesis: str
    where: str
    steps: tuple[StepSpec, ...]
    expect: ExpectSpec
    sweep: Mapping[str, tuple[ParamValue, ...]]
    gpus: int
    time_limit: str | None
    source_text: str
    path: Path | None = None

    def variants(self) -> tuple[Variant, ...]:
        """Cartesian product of the sweep, in declaration order; one empty variant without it."""
        if not self.sweep:
            return (Variant(index=1, name="v1", params={}),)
        keys = list(self.sweep)
        combinations = itertools.product(*(self.sweep[key] for key in keys))
        return tuple(
            Variant(index=position, name=f"v{position}", params=dict(zip(keys, combo, strict=True)))
            for position, combo in enumerate(combinations, start=1)
        )


def param_text(value: ParamValue) -> str:
    """Text of a sweep value in a command line (`True` -> `true`, like the task CLIs expect)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def slugify(text: str) -> str:
    """Lowercase ASCII slug (`Gold Soft, T=2` -> `gold-soft-t-2`), at most SLUG_MAX_LENGTH chars."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:SLUG_MAX_LENGTH].rstrip("-")


def _table(raw: object, what: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise SpecError(f"{what} must be a table")
    return raw


def _string(table: Mapping[str, object], key: str, what: str, *, required: bool = True) -> str:
    value = table.get(key)
    if value is None:
        if required:
            raise SpecError(f"{what}: missing required key '{key}'")
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise SpecError(f"{what}: '{key}' must be a non-empty string")
    return value


def _string_list(raw: object, what: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise SpecError(f"{what} must be a list of strings (no shell: one list item per argument)")
    return tuple(raw)


def _reject_unknown(table: Mapping[str, object], allowed: frozenset[str], what: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise SpecError(f"{what}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _placeholders(args: Sequence[str]) -> set[str]:
    return {name for arg in args for name in PLACEHOLDER_PATTERN.findall(arg)}


def _relative_path(value: str, what: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SpecError(f"{what} must be a relative path inside the run output directory: {value}")
    return value


def _parse_sweep(raw: object) -> dict[str, tuple[ParamValue, ...]]:
    if raw is None:
        return {}
    table = _table(raw, "[sweep]")
    sweep: dict[str, tuple[ParamValue, ...]] = {}
    for key, values in table.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SpecError(f"[sweep] key '{key}' must be an identifier (it becomes {{{key}}})")
        if key in COMMON_PLACEHOLDERS | JURECA_ONLY_PLACEHOLDERS:
            raise SpecError(f"[sweep] key '{key}' collides with a built-in placeholder")
        if not isinstance(values, list) or not values:
            raise SpecError(f"[sweep] '{key}' must be a non-empty list of values")
        if not all(isinstance(v, (str, int, float, bool)) for v in values):
            raise SpecError(f"[sweep] '{key}' values must be strings, numbers or booleans")
        sweep[key] = tuple(values)
    return sweep


def _parse_steps(raw: object, adapter: TaskAdapter, sweep_keys: set[str]) -> tuple[StepSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise SpecError("at least one [[steps]] table is required")
    steps: list[StepSpec] = []
    allowed_common = COMMON_PLACEHOLDERS | sweep_keys
    for position, entry in enumerate(raw, start=1):
        table = _table(entry, f"[[steps]] #{position}")
        what = f"[[steps]] #{position}"
        _reject_unknown(table, STEP_KEYS, what)
        step = StepSpec(
            name=_string(table, "name", what, required=False) or f"step{position}",
            args=_string_list(table.get("args"), f"{what} args"),
            args_local=_string_list(table.get("args_local"), f"{what} args_local"),
            args_jureca=_string_list(table.get("args_jureca"), f"{what} args_jureca"),
        )
        if not step.args:
            raise SpecError(f"{what}: 'args' is empty")
        _guard_step(step, adapter, what)
        for label, group, allowed in (
            ("args", step.args, allowed_common),
            ("args_local", step.args_local, allowed_common),
            ("args_jureca", step.args_jureca, allowed_common | JURECA_ONLY_PLACEHOLDERS),
        ):
            unknown = _placeholders(group) - allowed
            if unknown:
                raise SpecError(
                    f"{what} {label}: unknown placeholder(s) {sorted(unknown)}; "
                    f"available there: {sorted(allowed)}"
                )
        steps.append(step)
    return tuple(steps)


def _guard_step(step: StepSpec, adapter: TaskAdapter, what: str) -> None:
    """Refuse steps that spend budget or submit, and steps naming no known subcommand."""
    tokens = (*step.args, *step.args_local, *step.args_jureca)
    forbidden = sorted(set(tokens) & FORBIDDEN_TOKENS)
    if forbidden:
        raise SpecError(
            f"{what}: {forbidden} is forbidden in a spec. Spending the ep-* budget and "
            "submitting are human gestures: run them by hand, or use `lab submit RUN --yes`."
        )
    if not set(step.args) & adapter.subcommands:
        raise SpecError(
            f"{what}: no known {adapter.name} subcommand in args {list(step.args)}; "
            f"known: {sorted(adapter.subcommands)}"
        )


def _parse_expect(raw: object) -> ExpectSpec:
    if raw is None:
        return ExpectSpec()
    table = _table(raw, "[expect]")
    _reject_unknown(table, EXPECT_KEYS, "[expect]")
    submission = _string(table, "submission", "[expect]", required=False) or None
    metrics_file = _string(table, "metrics_file", "[expect]", required=False) or None
    primary = _string(table, "primary_metric", "[expect]", required=False) or None
    higher = table.get("higher_is_better", True)
    if not isinstance(higher, bool):
        raise SpecError("[expect] higher_is_better must be a boolean")
    regex_raw = table.get("metrics_regex", {})
    regexes = _table(regex_raw, "[expect.metrics_regex]")
    compiled: dict[str, str] = {}
    for name, pattern in regexes.items():
        if not isinstance(pattern, str):
            raise SpecError(f"[expect.metrics_regex] '{name}' must be a string")
        try:
            groups = re.compile(pattern).groups
        except re.error as exc:
            raise SpecError(f"[expect.metrics_regex] '{name}' is not a valid regex: {exc}") from exc
        if groups < 1:
            raise SpecError(f"[expect.metrics_regex] '{name}' needs one capture group")
        compiled[name] = pattern
    return ExpectSpec(
        submission=_relative_path(submission, "[expect] submission") if submission else None,
        metrics_file=(
            _relative_path(metrics_file, "[expect] metrics_file") if metrics_file else None
        ),
        primary_metric=primary,
        higher_is_better=higher,
        metrics_regex=compiled,
    )


def _parse_resources(raw: object) -> tuple[int | None, str | None]:
    if raw is None:
        return None, None
    table = _table(raw, "[resources]")
    _reject_unknown(table, RESOURCE_KEYS, "[resources]")
    gpus = table.get("gpus")
    if gpus is not None and (isinstance(gpus, bool) or not isinstance(gpus, int)):
        raise SpecError("[resources] gpus must be an integer")
    time_limit = table.get("time_limit")
    if time_limit is not None and (
        not isinstance(time_limit, str) or not SLURM_TIME_RE.fullmatch(time_limit)
    ):
        raise SpecError("[resources] time_limit must look like HH:MM:SS or D-HH:MM:SS")
    return gpus, time_limit


def parse_spec(
    text: str, adapters: Mapping[str, TaskAdapter], path: Path | None = None
) -> ExperimentSpec:
    """Validate TOML `text` into an ExperimentSpec.

    Args:
        text: The TOML document.
        adapters: Registered tasks (the spec's `task` is resolved against it).
        path: File the text came from; when given, its stem must equal the spec `id`.

    Raises:
        SpecError: malformed TOML, unknown key, forbidden token, too many variants, ...
        UnknownTaskError: the task is not registered.
    """
    origin = str(path) if path else "spec"
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"{origin}: invalid TOML: {exc}") from exc
    try:
        return _build_spec(document, text, adapters, path)
    except SpecError as exc:
        if path and not str(exc).startswith(origin):
            raise SpecError(f"{origin}: {exc}") from exc
        raise


def _build_spec(
    document: Mapping[str, object],
    text: str,
    adapters: Mapping[str, TaskAdapter],
    path: Path | None,
) -> ExperimentSpec:
    _reject_unknown(document, TOP_LEVEL_KEYS, "spec")
    spec_id = _string(document, "id", "spec")
    if not SLUG_PATTERN.fullmatch(spec_id) or len(spec_id) > SLUG_MAX_LENGTH:
        raise SpecError(f"id '{spec_id}' must be a lowercase slug (a-z, 0-9, '-'), max 48 chars")
    if path is not None and path.stem != spec_id:
        raise SpecError(f"id '{spec_id}' must equal the file name '{path.stem}'")
    adapter = resolve_adapter(_string(document, "task", "spec"), adapters)
    where = _string(document, "where", "spec", required=False) or WHERE_LOCAL
    if where not in WHERE_CHOICES:
        raise SpecError(f"where must be one of {list(WHERE_CHOICES)}, got '{where}'")
    sweep = _parse_sweep(document.get("sweep"))
    variant_count = 1
    for values in sweep.values():
        variant_count *= len(values)
    if variant_count > MAX_VARIANTS:
        raise SpecError(
            f"the sweep makes {variant_count} variants; at most {MAX_VARIANTS} fit in one JURECA "
            "job (one per A100). Split it into several specs."
        )
    gpus_requested, time_limit = _parse_resources(document.get("resources"))
    gpus = gpus_requested if gpus_requested is not None else variant_count
    if not MIN_GPUS <= gpus <= MAX_VARIANTS:
        raise SpecError(f"[resources] gpus must be between {MIN_GPUS} and {MAX_VARIANTS}")
    if gpus < variant_count:
        raise SpecError(f"[resources] gpus={gpus} is below the {variant_count} sweep variants")
    return ExperimentSpec(
        id=spec_id,
        task=adapter.name,
        idea=_string(document, "idea", "spec"),
        hypothesis=_string(document, "hypothesis", "spec", required=False),
        where=where,
        steps=_parse_steps(document.get("steps"), adapter, set(sweep)),
        expect=_parse_expect(document.get("expect")),
        sweep=sweep,
        gpus=gpus,
        time_limit=time_limit,
        source_text=text,
        path=path,
    )


def resolve_spec_path(reference: str, experiments_dir: Path) -> Path:
    """Locate a spec from a file path, or from an id looked up under `experiments/<task>/`.

    Raises:
        SpecError: nothing matches, or the id exists under several tasks.
    """
    direct = Path(reference)
    if direct.is_file():
        return direct
    candidates = sorted(experiments_dir.glob(f"*/{reference.removesuffix('.toml')}.toml"))
    if not candidates:
        raise SpecError(f"no spec '{reference}' (as a file, or under {experiments_dir}/<task>/)")
    if len(candidates) > 1:
        names = ", ".join(str(c) for c in candidates)
        raise SpecError(f"spec id '{reference}' is ambiguous: {names}")
    return candidates[0]


def load_spec(
    reference: str, adapters: Mapping[str, TaskAdapter], experiments_dir: Path
) -> ExperimentSpec:
    """Read and validate the spec named by `reference` (a path or an id)."""
    path = resolve_spec_path(reference, experiments_dir)
    return parse_spec(path.read_text(encoding="utf-8"), adapters, path)


def render_args(
    args: Sequence[str], values: Mapping[str, str], *, allow_jureca_only: bool
) -> tuple[str, ...]:
    """Substitute `{name}` placeholders in `args` (single pass, no shell, no format-spec magic).

    Args:
        args: Raw argument list from the spec.
        values: Placeholder values (built-ins plus the variant parameters).
        allow_jureca_only: False for a local run, where `{task_dir}` and `{team_dir}` do not exist.

    Raises:
        SpecError: unknown placeholder, a JURECA-only one used locally, or a forbidden token
            appearing only after substitution (e.g. a sweep value `--yes`).
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return values[name]
        if name in JURECA_ONLY_PLACEHOLDERS and not allow_jureca_only:
            raise SpecError(f"placeholder {{{name}}} only exists on JURECA")
        raise SpecError(f"unknown placeholder {{{name}}}")

    rendered = tuple(PLACEHOLDER_PATTERN.sub(replace, arg) for arg in args)
    forbidden = sorted(set(rendered) & FORBIDDEN_TOKENS)
    if forbidden:
        raise SpecError(f"{forbidden} appeared after substitution; it is forbidden in a run")
    return rendered

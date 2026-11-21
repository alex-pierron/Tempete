"""`lab new`: create an experiment spec and its idea sheet from the templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hackathon.lab.constants import IDEA_TEMPLATE_FILENAME, TEMPLATES_DIRNAME
from hackathon.lab.context import LabContext
from hackathon.lab.errors import LabError, SpecError
from hackathon.lab.spec import parse_spec, slugify
from hackathon.lab.tasks import resolve_adapter

SPEC_SUFFIX = ".toml"
IDEA_SUFFIX = ".md"


@dataclass(frozen=True)
class NewExperiment:
    """Files created by `create_experiment`."""

    spec_id: str
    spec_path: Path
    idea_path: Path


def _template(ctx: LabContext, filename: str) -> str:
    path = ctx.experiments_dir / TEMPLATES_DIRNAME / filename
    if not path.is_file():
        raise LabError(f"template missing: {path}")
    return path.read_text(encoding="utf-8")


def _fill(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def create_experiment(ctx: LabContext, task: str, idea: str) -> NewExperiment:
    """Write `experiments/<task>/<id>.toml` and `<id>.md` from the task's templates.

    The id is the slug of the idea. The new spec is parsed before returning, so a broken template
    is caught here and not on the first run.

    Raises:
        UnknownTaskError: unknown task.
        SpecError: the idea has no usable slug, the id already exists, or the template is invalid.
        LabError: a template file is missing.
    """
    adapter = resolve_adapter(task, ctx.adapters)
    spec_id = slugify(idea)
    if not spec_id:
        raise SpecError(f"idea '{idea}' gives an empty id: use letters or digits")
    clashes = sorted(ctx.experiments_dir.glob(f"*/{spec_id}{SPEC_SUFFIX}"))
    if clashes:
        raise SpecError(f"an experiment '{spec_id}' already exists: {clashes[0]}")
    folder = ctx.experiments_dir / adapter.name
    spec_path = folder / f"{spec_id}{SPEC_SUFFIX}"
    idea_path = folder / f"{spec_id}{IDEA_SUFFIX}"
    values = {
        "id": spec_id,
        "task": adapter.name,
        "idea": json.dumps(idea, ensure_ascii=False),
        "idea_text": idea,
        "date": ctx.clock().strftime("%Y-%m-%d"),
    }
    spec_text = _fill(_template(ctx, f"{adapter.name}{SPEC_SUFFIX}"), values)
    idea_text = _fill(_template(ctx, IDEA_TEMPLATE_FILENAME), values)
    parse_spec(spec_text, ctx.adapters, spec_path)
    folder.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(spec_text, encoding="utf-8")
    idea_path.write_text(idea_text, encoding="utf-8")
    return NewExperiment(spec_id, spec_path, idea_path)

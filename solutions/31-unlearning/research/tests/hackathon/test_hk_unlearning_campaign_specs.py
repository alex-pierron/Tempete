"""Every committed unlearning campaign still parses and resolves against the code of this folder.

The runs of experiments/registry.jsonl were made at several commits of the original branch; this
folder keeps only its final state. The test checks that each spec, for each variant and both places,
gives a command the final CLI accepts and, for `run`, a config the final code builds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hackathon.lab.constants import WHERE_CHOICES, WHERE_JURECA
from hackathon.lab.runner import build_step_argv
from hackathon.lab.spec import param_text, parse_spec
from hackathon.lab.tasks import DEFAULT_ADAPTERS
from hackathon.unlearning.cli import AUTO_CONFIG, build_parser
from hackathon.unlearning.configs import load_config

SPEC_DIR = Path(__file__).resolve().parents[2] / "experiments" / "unlearning"
SPECS = sorted(SPEC_DIR.glob("*.toml"))


def test_specs_are_present() -> None:
    assert len(SPECS) > 50


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_spec_resolves_with_final_code(path: Path) -> None:
    spec = parse_spec(path.read_text(encoding="utf-8"), DEFAULT_ADAPTERS, path)
    adapter = DEFAULT_ADAPTERS[spec.task]
    parser = build_parser()
    for variant in spec.variants():
        for where in WHERE_CHOICES:
            values = {"out": "out", "run_id": "r", "variant": variant.name,
                      "variant_index": str(variant.index)}
            values.update({key: param_text(value) for key, value in variant.params.items()})
            if where == WHERE_JURECA:
                values.update({"task_dir": "task", "team_dir": "team"})
            for step in spec.steps:
                argv = build_step_argv(adapter, step, values, where=where)
                args = parser.parse_args(list(argv[len(adapter.entrypoint):]))
                if args.command != "run":
                    continue
                overrides: dict[str, object] = dict(
                    assignment.partition("=")[::2] for assignment in args.set or []
                )
                if args.method:
                    overrides["method"] = args.method
                preset = "smoke" if args.config == AUTO_CONFIG else args.config
                load_config(preset, overrides)

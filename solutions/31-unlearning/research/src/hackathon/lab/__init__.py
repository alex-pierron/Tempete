"""Experiment lab: from a team idea to a scored submission, with as few manual gestures as possible.

A TOML spec (`experiments/<task>/<id>.toml`) names the task CLI steps to run and the files to
expect. `python -m hackathon.lab run` executes it locally or on JURECA (sweeps of up to four
variants, one per A100 of the node), records the outcome in the versioned registry
`experiments/registry.jsonl`, and `submit` hands the validated file to the task's own submit
command, always behind an explicit `--yes`. See docs/hackathon/process.md.

The lab imports no task package: it drives their CLIs through `python -m`, so a task is wired by
data (`hackathon.lab.tasks`).
"""

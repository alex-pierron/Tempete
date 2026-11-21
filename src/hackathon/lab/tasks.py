"""Task adapters: how the lab talks to each task CLI behind one interface.

The lab never imports a task package. It shells out to `python -m hackathon.<task> <subcommand>`,
so a task is described by data: its entry point, the subcommands it knows, the server task id
(for the cooldown guard) and how to validate, submit and read back the public score. The
localization task (`hackathon.wmloc`) is registered here with the CLI announced for it
(`fetch|synth|bench|predict|validate|submit`); its real behaviour is wired at merge time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hackathon.common.config import (
    TASK_MODEL_EQUALITY,
    TASK_UNLEARNING,
    Settings,
)
from hackathon.lab.errors import UnknownTaskError

SCORE_PATTERN = re.compile(r"\bscore=(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|None)")
PREVIOUS_BEST_PATTERN = re.compile(
    r"\bprevious_best=(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|None)"
)
OK_PATTERN = re.compile(r"\bok=(?P<value>True|False)\b")
MESSAGE_PATTERN = re.compile(r"\bmessage=(?P<value>'[^']*'|\"[^\"]*\")")


@dataclass(frozen=True)
class SubmitOutcome:
    """What a task `submit --yes` reported, as far as the lab can read it.

    Attributes:
        ok: The server accepted and scored the file.
        score: Public score, or None when the output carries none.
        previous_best: Previous best public score reported by the server, if any.
        message: Server message or error detail, possibly empty.
        replied: The output carries a server reply. False means the task CLI refused locally
            (cooldown, invalid file, no API key): nothing reached the server, so no cooldown
            started and the attempt must not be recorded as a submission.
    """

    ok: bool
    score: float | None
    previous_best: float | None
    message: str
    replied: bool = True


@dataclass(frozen=True)
class TaskAdapter:
    """Static description of one competition task for the lab.

    Attributes:
        name: Canonical task name (also the folder under `experiments/`).
        short: Three-letter tag used inside run ids.
        aliases: Other accepted spellings of the task name on the command line.
        entrypoint: Interpreter arguments launching the task CLI, e.g. `("-m", "hackathon.x")`.
        subcommands: Subcommands of the task CLI (a spec step must use at least one).
        server_task_id: Server task id for the cooldown guard, or None while unknown.
        remote_task_dirname: Folder of the task under the team folder on JURECA (venv, data).
        remote_global_args: Global CLI options placed BEFORE the subcommand on JURECA only; may
            use the `{task_dir}` and `{team_dir}` placeholders.
        task_id_env_hint: Name of the setting providing `server_task_id` when it is None.
    """

    name: str
    short: str
    entrypoint: tuple[str, ...]
    subcommands: frozenset[str]
    server_task_id: str | None
    remote_task_dirname: str
    aliases: tuple[str, ...] = ()
    remote_global_args: tuple[str, ...] = ()
    task_id_env_hint: str | None = None

    def validate_args(self, file: Path) -> tuple[str, ...]:
        """Arguments (after the entry point) that validate `file` without any network call."""
        return ("validate", str(file))

    def submit_args(self, file: Path) -> tuple[str, ...]:
        """Arguments that really submit `file` (only ever used behind `lab submit --yes`)."""
        return ("submit", str(file), "--yes")

    def resolve_server_task_id(self, settings: Settings | None) -> str | None:
        """The server task id: the constant, or the setting for tasks whose id is set in .env."""
        if self.server_task_id is not None:
            return self.server_task_id
        if settings is not None and self.name == TASK_NAME_WMLOC:
            return settings.wmloc_task_id
        return None


TASK_NAME_UNLEARNING = "unlearning"
TASK_NAME_MODEL_EQUALITY = "model_equality"
TASK_NAME_WMLOC = "wmloc"

UNLEARNING = TaskAdapter(
    name=TASK_NAME_UNLEARNING,
    short="unl",
    aliases=("unl", "31-unlearning"),
    entrypoint=("-m", "hackathon.unlearning"),
    subcommands=frozenset({"fetch", "info", "proxy-eval", "run", "validate", "submit"}),
    server_task_id=TASK_UNLEARNING,
    remote_task_dirname="unlearning",
    # Same wiring as jureca/unlearning.sbatch: the shared task folder holds model.pt and the venv.
    remote_global_args=("--task-dir", "{task_dir}"),
)
MODEL_EQUALITY = TaskAdapter(
    name=TASK_NAME_MODEL_EQUALITY,
    short="met",
    aliases=("met", "model-equality", "33-model-equality"),
    entrypoint=("-m", "hackathon.model_equality"),
    subcommands=frozenset(
        {
            "budget",
            "plan",
            "query-ref",
            "query-ep",
            "analyze",
            "score-practice",
            "make-submission",
            "validate",
            "submit",
        }
    ),
    server_task_id=TASK_MODEL_EQUALITY,
    remote_task_dirname="model_equality",
)
WMLOC = TaskAdapter(
    name=TASK_NAME_WMLOC,
    short="wml",
    aliases=("wml", "wm", "watermark-localization", "watermark_localization"),
    entrypoint=("-m", "hackathon.wmloc"),
    subcommands=frozenset({"fetch", "synth", "bench", "predict", "validate", "submit"}),
    server_task_id=None,
    remote_task_dirname="watermark_localization",
    task_id_env_hint="HACKATHON_TASK_WMLOC",
)
DEFAULT_ADAPTERS: Mapping[str, TaskAdapter] = {
    adapter.name: adapter for adapter in (UNLEARNING, MODEL_EQUALITY, WMLOC)
}


def resolve_adapter(name: str, adapters: Mapping[str, TaskAdapter]) -> TaskAdapter:
    """Find the adapter whose name or alias is `name` (case-insensitive).

    Raises:
        UnknownTaskError: nothing matches; the message lists the valid names.
    """
    wanted = name.strip().lower()
    for adapter in adapters.values():
        if wanted == adapter.name or wanted in adapter.aliases:
            return adapter
    raise UnknownTaskError(f"unknown task '{name}'; expected one of: {', '.join(sorted(adapters))}")


def _as_float(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _json_object_with_result(text: str) -> dict[str, object] | None:
    """The first JSON object of `text` that looks like a submission reply."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            candidate, _ = decoder.raw_decode(text, start)
        except ValueError:
            candidate = None
        if isinstance(candidate, dict) and ("score" in candidate or "ok" in candidate):
            return candidate
        start = text.find("{", start + 1)
    return None


def _regex_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if match is None or match.group("value") == "None":
        return None
    return float(match.group("value"))


def parse_submit_output(stdout: str, returncode: int) -> SubmitOutcome:
    """Read the public score out of a task `submit --yes` output.

    Two formats exist in the kits: a pretty-printed JSON `SubmitResult` (unlearning) and the line
    `ok=True score=0.31 previous_best=0.2 improved=True message='...'` (model equality). Any other
    task is read the same way; a missing score stays None (the caller falls back to the shared
    submission log). `ok` defaults to "exit code 0" when the output does not say.
    """
    parsed = _json_object_with_result(stdout)
    if parsed is not None:
        ok_raw = parsed.get("ok")
        message = str(parsed.get("message") or parsed.get("detail") or "")
        return SubmitOutcome(
            ok=ok_raw if isinstance(ok_raw, bool) else returncode == 0,
            score=_as_float(parsed.get("score")),
            previous_best=_as_float(parsed.get("previous_best")),
            message=message,
            replied=True,
        )
    ok_match = OK_PATTERN.search(stdout)
    message_match = MESSAGE_PATTERN.search(stdout)
    score = _regex_float(SCORE_PATTERN, stdout)
    return SubmitOutcome(
        ok=ok_match.group("value") == "True" if ok_match else returncode == 0,
        score=score,
        previous_best=_regex_float(PREVIOUS_BEST_PATTERN, stdout),
        message=message_match.group("value")[1:-1] if message_match else "",
        replied=ok_match is not None or SCORE_PATTERN.search(stdout) is not None or returncode == 0,
    )

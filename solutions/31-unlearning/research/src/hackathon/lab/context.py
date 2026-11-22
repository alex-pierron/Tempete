"""Everything the lab needs from its environment, gathered in one injectable object.

Tests build a `LabContext` over `tmp_path` with fake clocks and output sinks; the CLI builds the
default one. Nothing here reads `.env` until a caller asks (`env_loader`/`settings_loader` are
lazy), so purely local commands never touch it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hackathon.common.config import (
    DEFAULT_ENV_FILE,
    REPO_ROOT,
    Settings,
    load_settings,
    read_env_file,
)
from hackathon.lab.constants import (
    EXPERIMENTS_DIRNAME,
    JOB_TEMPLATE_RELATIVE,
    REGISTRY_FILENAME,
    RUNS_DIRNAME,
)
from hackathon.lab.registry import Registry
from hackathon.lab.tasks import DEFAULT_ADAPTERS, TaskAdapter

LineSink = Callable[[str], None]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _stderr_line(text: str) -> None:
    print(text, file=sys.stderr)


def default_environment() -> Mapping[str, str]:
    """`.env` merged with the process environment (the environment wins, like load_settings)."""
    merged = dict(read_env_file(DEFAULT_ENV_FILE))
    merged.update({key: value for key, value in os.environ.items() if value})
    return merged


@dataclass
class LabContext:
    """Paths, clock, sinks and lazy configuration shared by every lab command.

    Attributes:
        repo_root: Root of the repository (holds `experiments/`, `runs/`, `jureca/`, `src/`).
        adapters: Registered task adapters.
        python: Interpreter used to launch task CLIs locally.
        clock: Current time source (UTC).
        out: Sink for normal output lines.
        err: Sink for warnings and streamed stderr lines.
        env_loader: Lazy source of the JURECA_* variables (called only by cluster commands).
        settings_loader: Lazy source of the hackathon Settings (submission log, wmloc task id).
    """

    repo_root: Path = REPO_ROOT
    adapters: Mapping[str, TaskAdapter] = field(default_factory=lambda: dict(DEFAULT_ADAPTERS))
    python: str = sys.executable
    clock: Callable[[], datetime] = _utc_now
    out: LineSink = print
    err: LineSink = _stderr_line
    env_loader: Callable[[], Mapping[str, str]] = default_environment
    settings_loader: Callable[[], Settings] = load_settings

    @property
    def experiments_dir(self) -> Path:
        """Folder of the specs, idea sheets, templates and the registry."""
        return self.repo_root / EXPERIMENTS_DIRNAME

    @property
    def runs_dir(self) -> Path:
        """Git-ignored folder of run outputs and the HTML board."""
        return self.repo_root / RUNS_DIRNAME

    @property
    def registry(self) -> Registry:
        """The versioned run registry."""
        return Registry(self.experiments_dir / REGISTRY_FILENAME, self.clock)

    @property
    def job_template_path(self) -> Path:
        """The sbatch template of lab jobs."""
        return self.repo_root / JOB_TEMPLATE_RELATIVE

    def run_dir(self, task: str, run_id: str) -> Path:
        """Local output directory of one run."""
        return self.runs_dir / task / run_id

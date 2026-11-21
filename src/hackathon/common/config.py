"""Settings of the hackathon, read from the environment and an optional `.env` file.

The `.env` parser is a deliberate 20-line subset (KEY=VALUE, `#` comments, optional quotes) so the
code runs in the official JURECA task environments, which do not ship python-dotenv.
Real environment variables always win over the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from hackathon.common.errors import ConfigError

DEFAULT_SERVER_URL = "http://34.147.229.55"
LEADERBOARD_PATH = "/leaderboard_page"
TASK_UNLEARNING = "31-unlearning"
TASK_MODEL_EQUALITY = "33-model-equality"
WMLOC_TASK_ENV = "HACKATHON_TASK_WMLOC"
"""The localization task id is not in its PDF; set it in .env once known (see docs/hackathon)."""

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "hackathon"
API_KEY_PLACEHOLDER = "YOUR_API_KEY_HERE"


@dataclass(frozen=True)
class Settings:
    """Resolved hackathon settings.

    Attributes:
        api_key: Team API key for submissions and the model-equality endpoints, or None.
        server_url: Base URL of the evaluation server (no trailing slash).
        hf_token: Hugging Face token (needed for the gated localization dataset), or None.
        data_dir: Root of downloaded data, outputs and the submission log.
        wmloc_task_id: Server task id of the localization task, or None while unknown.
    """

    api_key: str | None
    server_url: str
    hf_token: str | None
    data_dir: Path
    wmloc_task_id: str | None

    def require_api_key(self) -> str:
        """Return the API key or raise ConfigError with the fix."""
        if not self.api_key or self.api_key == API_KEY_PLACEHOLDER:
            raise ConfigError("HACKATHON_API_KEY is not set: add it to .env (never commit it).")
        return self.api_key

    def require_wmloc_task_id(self) -> str:
        """Return the localization task id or raise ConfigError with the fix."""
        if not self.wmloc_task_id:
            raise ConfigError(f"{WMLOC_TASK_ENV} is not set: add the server task id to .env.")
        return self.wmloc_task_id


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines of a .env file; a missing file gives an empty dict."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_settings(env_file: Path | None = DEFAULT_ENV_FILE) -> Settings:
    """Merge `env_file` and the process environment (environment wins) into Settings."""
    merged = read_env_file(env_file) if env_file is not None else {}
    merged.update({key: value for key, value in os.environ.items() if value})

    def optional(key: str) -> str | None:
        value = merged.get(key, "").strip()
        return value or None

    data_dir = Path(optional("HACKATHON_DATA_DIR") or DEFAULT_DATA_DIR).expanduser()
    return Settings(
        api_key=optional("HACKATHON_API_KEY"),
        server_url=(optional("HACKATHON_SERVER_URL") or DEFAULT_SERVER_URL).rstrip("/"),
        hf_token=optional("HF_TOKEN"),
        data_dir=data_dir,
        wmloc_task_id=optional(WMLOC_TASK_ENV),
    )

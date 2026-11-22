"""Named constants of the experiment lab (paths, statuses, JURECA fixed parameters)."""

from __future__ import annotations

from pathlib import Path

# --- Layout (all relative to the repository root) ---
EXPERIMENTS_DIRNAME = "experiments"
RUNS_DIRNAME = "runs"
TEMPLATES_DIRNAME = "_templates"
REGISTRY_FILENAME = "registry.jsonl"
IDEA_TEMPLATE_FILENAME = "idea.md"
BOARD_FILENAME = "board.html"
JOB_TEMPLATE_RELATIVE = Path("jureca") / "lab_job.sbatch.tmpl"

# --- Files written in every run directory ---
SPEC_SNAPSHOT_FILENAME = "spec.toml"
RUN_META_FILENAME = "run.json"
STDOUT_LOG_FILENAME = "stdout.log"
STDERR_LOG_FILENAME = "stderr.log"
JOB_SCRIPT_FILENAME = "job.sbatch"
EXIT_CODE_FILENAME = "exit_code"
STARTED_AT_FILENAME = "started_at"
FINISHED_AT_FILENAME = "finished_at"

# --- Run statuses ---
STATUS_QUEUED = "queued"
"""Submitted to Slurm, not started."""
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
"""JURECA only: the remote job ended with exit code 0 but the files are not fetched yet."""
STATUS_DONE = "done"
"""Ended, expected files present and collected: the only status a submission accepts."""
STATUS_FAILED = "failed"
ACTIVE_STATUSES = frozenset({STATUS_QUEUED, STATUS_RUNNING})

# --- Execution places ---
WHERE_LOCAL = "local"
WHERE_JURECA = "jureca"
WHERE_CHOICES = (WHERE_LOCAL, WHERE_JURECA)

# --- Spec rules ---
MAX_VARIANTS = 4
"""A JURECA node exposes 4 A100: one variant per GPU, all inside a single job."""
MIN_GPUS = 1
SLUG_MAX_LENGTH = 48
PLACEHOLDER_OUT = "out"
PLACEHOLDER_RUN_ID = "run_id"
PLACEHOLDER_VARIANT = "variant"
PLACEHOLDER_VARIANT_INDEX = "variant_index"
PLACEHOLDER_TASK_DIR = "task_dir"
PLACEHOLDER_TEAM_DIR = "team_dir"
COMMON_PLACEHOLDERS = frozenset(
    {PLACEHOLDER_OUT, PLACEHOLDER_RUN_ID, PLACEHOLDER_VARIANT, PLACEHOLDER_VARIANT_INDEX}
)
JURECA_ONLY_PLACEHOLDERS = frozenset({PLACEHOLDER_TASK_DIR, PLACEHOLDER_TEAM_DIR})
FORBIDDEN_TOKENS = frozenset({"submit", "query-ep", "--yes"})
"""Tokens a spec step may never contain: they spend the irreplaceable budget or submit.

The lab runs steps unattended (and on a cluster); spending or submitting stays a human gesture
through the task CLI or `lab submit --yes`."""

# --- JURECA (values fixed by the organisers, see docs/hackathon/jureca.md) ---
JURECA_PROJECT = "trainingXXXX"  # placeholder: the Slurm account given by the organisers
JURECA_SCRATCH_ROOT = Path("/p/scratch") / JURECA_PROJECT
DEFAULT_SSH_HOST = "juelich"
REMOTE_REPO_DIRNAME = "Tempete"
REMOTE_RUNS_DIRNAME = "lab_runs"
CODE_SNAPSHOT_DIRNAME = "code"
"""Copy of the synced checkout taken at submission, under the run root: the job runs this frozen
code, not the shared `Tempete/` that a teammate may re-sync while the job waits for the node."""
SYNC_EXCLUDES = (
    "/.venv",
    ".venv",
    "/data",
    "/weights",
    "/third_party",
    "/runs",
    ".env",
    ".git",
    "__pycache__",
    "*.pyc",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
)
"""rsync exclusions. Top-level folders are anchored so `src/**/data.py` style names survive."""
SSH_CONNECT_TIMEOUT_S = 30
SSH_COMMAND_TIMEOUT_S = 600.0
RSYNC_TIMEOUT_S = 3600.0
"""A fetched unlearning candidate is ~344 MB over a conference Wi-Fi: be generous."""
HOST_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
SLURM_TIME_PATTERN = r"(\d{1,2}-)?\d{1,3}:\d{2}:\d{2}"
SLURM_FAILURE_STATES = (
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
)

# --- Board ---
BOARD_METRIC_COLUMNS = 3
"""Metrics shown per run in the board (the primary one first)."""
RUN_ID_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

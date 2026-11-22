"""The versioned run registry `experiments/registry.jsonl`.

Append-only, one JSON snapshot of a run per line. A run changes state several times (running,
finished, scored); each change appends a NEW full snapshot, and the reader keeps, per `run_id`,
the snapshot with the greatest `updated_at`. This is deliberate: the file is committed and merged
by three people, so `experiments/.gitattributes` gives it `merge=union` (like the model-equality
answers store); with union merge, line order is arbitrary but nothing is ever lost, and the
timestamp, not the position, decides which snapshot wins.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path

from hackathon.lab.errors import RegistryError

JsonValue = str | int | float | bool | None
MetricMap = dict[str, float]


@dataclass(frozen=True)
class RunRecord:
    """State of one run (one variant of one experiment) at one point in time.

    Attributes:
        run_id: Unique id, `<UTC timestamp>-<task tag>-<spec id>[-vN]`.
        group_id: Id shared by the variants launched together (equals run_id without sweep).
        task: Canonical task name.
        idea: Idea label copied from the spec.
        spec_id: Id of the spec the run came from.
        variant: Variant name (`v1`..`v4`).
        params: Sweep parameters of this variant.
        sha: Git commit of the code that ran, or `unknown`.
        dirty: The working tree had uncommitted changes (the SHA alone does not reproduce it).
        where: `local` or `jureca`.
        status: queued | running | finished | done | failed.
        created_at: Creation time (ISO 8601, UTC).
        updated_at: Time of this snapshot; the greatest one wins.
        started_at: Start of the execution, when known.
        finished_at: End of the execution, when known.
        duration_s: Execution time in seconds, when known.
        exit_code: Exit code of the last executed step.
        metrics: Local metrics (never the public score).
        primary_metric: Name of the metric that ranks runs on the board.
        primary_value: Its value in this run.
        higher_is_better: Direction of the primary metric.
        submission_file: Path of the submission file (relative to the repo when inside it).
        submission_sha256: Hash of the submission file.
        jureca_job_id: Slurm job id for a JURECA run.
        remote_dir: Output directory of the run on JURECA.
        public_score: Public leaderboard score (30 % of the data) once submitted.
        previous_best: Server-reported previous best, when given.
        public_submitted_at: When the submission attempt was made (a failure counts too).
        submit_message: Server message or failure reason of the submission.
        note: Free text of the lab (why a run failed, ...).
    """

    run_id: str
    group_id: str
    task: str
    idea: str
    spec_id: str
    variant: str
    sha: str
    where: str
    status: str
    created_at: str
    updated_at: str
    dirty: bool = False
    params: dict[str, JsonValue] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    exit_code: int | None = None
    metrics: MetricMap = field(default_factory=dict)
    primary_metric: str | None = None
    primary_value: float | None = None
    higher_is_better: bool = True
    submission_file: str | None = None
    submission_sha256: str | None = None
    jureca_job_id: str | None = None
    remote_dir: str | None = None
    public_score: float | None = None
    previous_best: float | None = None
    public_submitted_at: str | None = None
    submit_message: str = ""
    note: str = ""

    def to_json_line(self) -> str:
        """One compact JSON line (no trailing newline)."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> RunRecord:
        """Build a record from a decoded JSON object; unknown keys are ignored (forward compat).

        Raises:
            RegistryError: a required key is missing.
        """
        known = {f.name for f in fields(cls)}
        data = {key: value for key, value in raw.items() if key in known}
        try:
            return cls(**data)  # type: ignore[arg-type]
        except TypeError as exc:
            raise RegistryError(f"registry record is incomplete: {exc}") from exc


def utc_now_iso(now: datetime | None = None) -> str:
    """ISO 8601 UTC timestamp with microseconds (so snapshots of one run order strictly)."""
    return (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="microseconds")


class Registry:
    """Reader/writer of the registry file.

    Args:
        path: The `registry.jsonl` file (created on first append).
        clock: Time source, overridden in tests.
    """

    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None) -> None:
        self.path = path
        self._clock = clock or (lambda: datetime.now(UTC))

    def snapshots(self) -> list[RunRecord]:
        """Every line of the file, in file order.

        Raises:
            RegistryError: a line is not a JSON object or lacks required fields (the message
                gives the line number so a bad merge can be fixed by hand).
        """
        if not self.path.is_file():
            return []
        records: list[RunRecord] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise RegistryError(f"{self.path}:{number}: not valid JSON ({exc})") from exc
            if not isinstance(raw, dict):
                raise RegistryError(f"{self.path}:{number}: expected a JSON object")
            try:
                records.append(RunRecord.from_mapping(raw))
            except RegistryError as exc:
                raise RegistryError(f"{self.path}:{number}: {exc}") from exc
        return records

    def latest(self) -> dict[str, RunRecord]:
        """Current state of every run: the snapshot with the greatest `updated_at` per run id.

        Ties keep the later line. The result is ordered by creation time, oldest first.
        """
        current: dict[str, RunRecord] = {}
        for record in self.snapshots():
            known = current.get(record.run_id)
            if known is None or record.updated_at >= known.updated_at:
                current[record.run_id] = record
        return dict(sorted(current.items(), key=lambda item: (item[1].created_at, item[0])))

    def append(self, record: RunRecord) -> None:
        """Append one snapshot (creates the file and its directory)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json_line() + "\n")

    def update(self, record: RunRecord, **changes: object) -> RunRecord:
        """Append a new snapshot of `record` with `changes` applied; returns it.

        `updated_at` is set to now, but never below the snapshot being replaced, so a skewed clock
        cannot make an old snapshot win.
        """
        stamp = max(utc_now_iso(self._clock()), record.updated_at)
        updated = replace(record, **changes, updated_at=stamp)  # type: ignore[arg-type]
        self.append(updated)
        return updated

    def find(self, reference: str) -> list[RunRecord]:
        """Resolve a run id, a group id, or a unique prefix to the matching runs.

        An exact run id wins; otherwise every run of the group `reference` (the variants launched
        together); otherwise the runs whose id starts with `reference`.

        Raises:
            RegistryError: nothing matches, or a prefix matches runs of several groups.
        """
        runs = self.latest()
        if reference in runs:
            return [runs[reference]]
        group = [r for r in runs.values() if r.group_id == reference]
        if group:
            return group
        prefixed = [r for r in runs.values() if r.run_id.startswith(reference)]
        if not prefixed:
            raise RegistryError(f"no run '{reference}' in {self.path}")
        if len({r.group_id for r in prefixed}) > 1:
            names = ", ".join(r.run_id for r in prefixed[:5])
            raise RegistryError(f"'{reference}' is ambiguous: {names}")
        return prefixed

    def get(self, reference: str) -> RunRecord:
        """Resolve `reference` to exactly one run.

        Raises:
            RegistryError: unknown, or it names several variants (the message lists them).
        """
        matches = self.find(reference)
        if len(matches) > 1:
            names = ", ".join(r.run_id for r in matches)
            raise RegistryError(f"'{reference}' names several runs, pick one: {names}")
        return matches[0]


def runs_of_task(records: Iterable[RunRecord], task: str | None) -> list[RunRecord]:
    """Filter `records` to one task (all of them when `task` is None)."""
    return [r for r in records if task is None or r.task == task]

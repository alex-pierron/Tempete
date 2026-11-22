"""Write slides/deck_data.json (score curves of the deck and the report) from our submission log.

The log is the JSON-lines file our submission queue appended to (one line per submission: task_id,
file, submitted_at in UTC, score). Only the time, the task, the score and the file stem are kept.

    python3 slides/make_deck_data.py <path/to/log.jsonl>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_FILE = HERE / "deck_data.json"
PARIS = timezone(timedelta(hours=1))
"""Central European Time on 21-22 November (no daylight saving)."""
TASK_KEYS: dict[str, str] = {
    "31-unlearning": "unlearning",
    "33-model-equality": "model_equality",
    "34-watermark-localisation-image": "wmloc",
}
SECONDS_PER_DAY = 86400.0


class LogFormatError(ValueError):
    """A log line misses a field the deck needs."""


def day_fraction(moment: datetime) -> float:
    """Paris time as a spreadsheet serial: 0.75 = 18:00 on the first day, 1.25 = 06:00 the next."""
    local = moment.astimezone(PARIS)
    days = (local.date() - datetime(2026, 11, 21, tzinfo=PARIS).date()).days
    seconds = local.hour * 3600 + local.minute * 60 + local.second
    return days + seconds / SECONDS_PER_DAY


def read_log(path: Path) -> dict[str, list[dict[str, object]]]:
    """Scored submissions per task, in time order."""
    curves: dict[str, list[dict[str, object]]] = {key: [] for key in TASK_KEYS.values()}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            task, file_name, sent_at = row["task_id"], row["file"], row["submitted_at"]
        except KeyError as error:
            raise LogFormatError(f"line {number}: missing {error}") from error
        score = row.get("score")
        if task not in TASK_KEYS or not row.get("ok") or score is None:
            continue
        moment = datetime.fromisoformat(str(sent_at))
        curves[TASK_KEYS[task]].append({
            "time": moment.astimezone(PARIS).strftime("%H:%M"),
            "x": round(day_fraction(moment), 6),
            "score": float(score),
            "name": Path(str(file_name)).stem,
        })
    for points in curves.values():
        points.sort(key=lambda point: float(str(point["x"])))
    return curves


def main(argv: list[str]) -> int:
    """Read the log given on the command line and write deck_data.json."""
    if len(argv) != 2:
        print(__doc__)
        return 2
    curves = read_log(Path(argv[1]))
    document = {
        "note": "Scored public submissions of team Tempete (Paris time), from our submission log.",
        "submissions": curves,
    }
    OUTPUT_FILE.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    counts = {task: len(points) for task, points in curves.items()}
    print(OUTPUT_FILE, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

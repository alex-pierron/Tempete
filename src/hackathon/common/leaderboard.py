"""Snapshot of the public leaderboard: where the team stands on every task, and the gap to the top.

The page (`{server}/leaderboard_page`) holds 4 `<table class="board ...">`: the overall ranking, then
one table per task (Rank, Team, Best score, Last improved) in the order Watermark Localization,
Model Equality, Unlearning. Every call appends one JSON line to
`artifacts/leaderboard/snapshots.jsonl` (versioned: the history is the team's record) and rewrites
`artifacts/leaderboard/latest.md`. Public scores only cover ~30 % of the data: use them to situate
the team, never to tune a method.

CLI: python -m hackathon.common.leaderboard [--team "Tempête"] [--no-save]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import requests

from hackathon.common.config import LEADERBOARD_PATH, REPO_ROOT, load_settings
from hackathon.common.errors import HackathonError

TEAM_NAME_ENV = "HACKATHON_TEAM_NAME"
DEFAULT_TEAM_NAME = "Tempête"
TASK_TABLE_NAMES = ("wmloc", "model_equality", "unlearning")
"""Order of the per-task tables on the page (after the overall table)."""
TOP_SHOWN = 5
FETCH_TIMEOUT_S = 30.0
SNAPSHOT_DIR = REPO_ROOT / "artifacts" / "leaderboard"
_TABLE = re.compile(r'<table class="board[^"]*">(.*?)</table>', re.DOTALL)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


class LeaderboardError(HackathonError):
    """The page could not be fetched or no longer has the expected tables."""


@dataclass(frozen=True)
class TaskStanding:
    """The team's position on one task (rank None when the team has no score yet)."""

    task: str
    teams_scored: int
    rank: int | None
    score: float | None
    best: float | None
    gap_to_best: float | None
    top: list[tuple[str, float]]


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", fragment))).strip()


def _score(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_task_tables(page: str) -> dict[str, list[tuple[str, float]]]:
    """Map task -> [(team, best score)] sorted as on the page; raises if the layout changed."""
    tables = _TABLE.findall(page)
    if len(tables) < 1 + len(TASK_TABLE_NAMES):
        raise LeaderboardError(f"expected {1 + len(TASK_TABLE_NAMES)} tables, found {len(tables)}")
    result: dict[str, list[tuple[str, float]]] = {}
    for name, table in zip(TASK_TABLE_NAMES, tables[1:], strict=False):
        rows: list[tuple[str, float]] = []
        for row in _ROW.findall(table):
            cells = [_text(cell) for cell in _CELL.findall(row)]
            if len(cells) >= 3 and (score := _score(cells[2])) is not None:
                rows.append((cells[1], score))
        result[name] = rows
    return result


def standings(tables: dict[str, list[tuple[str, float]]], team: str) -> list[TaskStanding]:
    """The team's rank, score and gap to the best score on every task."""
    out: list[TaskStanding] = []
    for task, rows in tables.items():
        best = rows[0][1] if rows else None
        mine = next(((i + 1, s) for i, (name, s) in enumerate(rows) if name == team), None)
        out.append(TaskStanding(
            task=task, teams_scored=len(rows), rank=mine[0] if mine else None,
            score=mine[1] if mine else None, best=best,
            gap_to_best=round(best - mine[1], 4) if mine and best is not None else None,
            top=rows[:TOP_SHOWN],
        ))  # fmt: skip
    return out


def render_markdown(items: list[TaskStanding], team: str, taken_at: str) -> str:
    """Short Markdown summary for humans and reports."""
    lines = [f"# Leaderboard public : {team} ({taken_at})", "",
             "| Tâche | Notre rang | Notre score | Meilleur | Écart | Top 5 |",
             "|---|---|---|---|---|---|"]
    for item in items:
        top = ", ".join(f"{name} {score:.2f}" for name, score in item.top)
        rank = f"{item.rank}/{item.teams_scored}" if item.rank else f"– /{item.teams_scored}"
        score = f"{item.score:.4f}" if item.score is not None else "pas soumis"
        gap = f"{item.gap_to_best:+.2f}" if item.gap_to_best is not None else "–"
        lines.append(f"| {item.task} | {rank} | {score} | {item.best} | {gap} | {top} |")
    lines += ["", "Scores publics (~30 % des données) : pour se situer, jamais pour régler."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default=os.environ.get(TEAM_NAME_ENV, DEFAULT_TEAM_NAME))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings()
    try:
        reply = requests.get(settings.server_url + LEADERBOARD_PATH, timeout=FETCH_TIMEOUT_S)
        reply.raise_for_status()
        items = standings(parse_task_tables(reply.text), args.team)
    except (requests.RequestException, LeaderboardError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    taken_at = datetime.now(UTC).isoformat(timespec="seconds")
    markdown = render_markdown(items, args.team, taken_at)
    print(markdown)
    if not args.no_save:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        with (SNAPSHOT_DIR / "snapshots.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"taken_at": taken_at, "team": args.team,
                                     "standings": [asdict(i) for i in items]},
                                    ensure_ascii=False) + "\n")
        (SNAPSHOT_DIR / "latest.md").write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The board: every run of the registry in a terminal table and a static HTML page.

Two "best" notions are shown on purpose and never mixed. The LOCAL best (by each run's primary
metric, computed on our own validation) is what decides which idea to keep. The PUBLIC best is
the leaderboard score on 30 % of the hidden data: it is information, not a selection criterion
(the final ranking uses the other 70 %, and the subjects insist the public score can collapse).
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hackathon.lab.constants import BOARD_FILENAME, BOARD_METRIC_COLUMNS, STATUS_DONE
from hackathon.lab.context import LabContext
from hackathon.lab.registry import RunRecord, runs_of_task

SHA_DISPLAY_LENGTH = 7
EMPTY_CELL = "-"
SIGNIFICANT_DIGITS = ".4g"
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
TEXT_COLUMNS = (
    "run",
    "task",
    "idea",
    "where",
    "status",
    "sha",
    "local metrics",
    "public",
    "time",
)
IDEA_MAX_CHARS = 28
GUIDANCE = (
    "Decide on the LOCAL metric. The public score covers 30 % of the hidden data: "
    "information only, never a target."
)


@dataclass(frozen=True)
class Best:
    """The best run of a task by one criterion."""

    run: RunRecord
    label: str
    value: float


def format_number(value: float) -> str:
    """Compact number: 4 significant digits."""
    return format(value, SIGNIFICANT_DIGITS)


def format_duration(seconds: float | None) -> str:
    """`83` -> `1m23s`; None -> `-`."""
    if seconds is None:
        return EMPTY_CELL
    whole = round(seconds)
    if whole < SECONDS_PER_MINUTE:
        return f"{whole}s"
    if whole < SECONDS_PER_HOUR:
        return f"{whole // SECONDS_PER_MINUTE}m{whole % SECONDS_PER_MINUTE:02d}s"
    return f"{whole // SECONDS_PER_HOUR}h{whole % SECONDS_PER_HOUR // SECONDS_PER_MINUTE:02d}m"


def metrics_text(record: RunRecord) -> str:
    """The primary metric first, then up to BOARD_METRIC_COLUMNS - 1 others, `name=value`."""
    names = sorted(record.metrics, key=lambda n: (n != record.primary_metric, n))
    shown = [f"{n}={format_number(record.metrics[n])}" for n in names[:BOARD_METRIC_COLUMNS]]
    hidden = len(names) - len(shown)
    return " ".join(shown) + (f" (+{hidden})" if hidden > 0 else "") if shown else EMPTY_CELL


def public_text(record: RunRecord) -> str:
    """Public score, `failed` for a rejected attempt, `-` when never submitted."""
    if record.public_score is not None:
        return format_number(record.public_score)
    return "rejected" if record.public_submitted_at else EMPTY_CELL


def sha_text(record: RunRecord) -> str:
    """Short SHA, starred when the tree was dirty (the SHA alone does not reproduce the run)."""
    short = record.sha[:SHA_DISPLAY_LENGTH]
    return short + ("*" if record.dirty else "")


def best_local(records: Sequence[RunRecord]) -> list[Best]:
    """Best finished run per primary metric name (a task may rank by several)."""
    ranked: dict[str, Best] = {}
    for record in records:
        if record.status != STATUS_DONE or record.primary_metric is None:
            continue
        if record.primary_value is None:
            continue
        key = record.primary_metric
        known = ranked.get(key)
        better = known is None or (
            record.primary_value > known.value
            if record.higher_is_better
            else record.primary_value < known.value
        )
        if better:
            ranked[key] = Best(record, key, record.primary_value)
    return list(ranked.values())


def best_public(records: Sequence[RunRecord]) -> Best | None:
    """Run with the highest public score (the leaderboard keeps the best score)."""
    scored = [r for r in records if r.public_score is not None]
    if not scored:
        return None
    top = max(scored, key=lambda r: float(r.public_score or 0.0))
    return Best(top, "public", float(top.public_score or 0.0))


def _row(record: RunRecord) -> list[str]:
    idea = (
        record.idea
        if len(record.idea) <= IDEA_MAX_CHARS
        else record.idea[: IDEA_MAX_CHARS - 1] + "~"
    )
    label = f"{idea} [{record.variant}]" if record.params else idea
    return [
        record.run_id,
        record.task,
        label,
        record.where,
        record.status,
        sha_text(record),
        metrics_text(record),
        public_text(record),
        format_duration(record.duration_s),
    ]


def _tasks_in(records: Sequence[RunRecord]) -> list[str]:
    return sorted({r.task for r in records})


def render_text(records: Sequence[RunRecord], task: str | None = None) -> str:
    """Aligned terminal table, then the best runs per task."""
    shown = runs_of_task(records, task)
    if not shown:
        return "no run in the registry yet" + (f" for task '{task}'" if task else "") + "."
    rows = [list(TEXT_COLUMNS), *(_row(r) for r in shown)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(TEXT_COLUMNS))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    lines.append("")
    for name in _tasks_in(shown):
        group = runs_of_task(shown, name)
        lines.append(f"{name}: {len(group)} run(s)")
        for best in best_local(group):
            lines.append(
                f"  best local  ({best.label}) : {format_number(best.value)}  {best.run.run_id}"
            )
        public = best_public(group)
        if public:
            lines.append(
                f"  best public (info only)  : {format_number(public.value)}  {public.run.run_id}"
            )
    lines.append("")
    lines.append(GUIDANCE)
    return "\n".join(lines)


HTML_STYLE = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#dcdcd4;--card:#fff;--accent:#1f5fbf;--bad:#b3261e;--good:#1e7a3a}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecece6;--muted:#9a9a92;--line:#33332f;--card:#1f1f1c;--accent:#7aa7f0;--bad:#f2766d;--good:#6fcf8a}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto}
h1{font-size:1.4rem;margin:0 0 4px}h2{font-size:1.05rem;margin:28px 0 8px}
p.note{color:var(--muted);margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card b{display:block}.card span{color:var(--muted);font-size:.9rem}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600}
td.mono{font-family:ui-monospace,Menlo,monospace;font-size:.82rem}
.status-done{color:var(--good)}.status-failed{color:var(--bad)}
"""


def _cell(text: str, css: str = "") -> str:
    attr = f' class="{css}"' if css else ""
    return f"<td{attr}>{html.escape(text)}</td>"


def render_html(
    records: Sequence[RunRecord], task: str | None = None, generated_at: datetime | None = None
) -> str:
    """Self-contained static page (no script, no external resource); every value is escaped."""
    shown = runs_of_task(records, task)
    cards: list[str] = []
    for name in _tasks_in(shown):
        group = runs_of_task(shown, name)
        body = [f"<b>{html.escape(name)}</b><span>{len(group)} run(s)</span>"]
        for best in best_local(group):
            body.append(
                f"<span>best local ({html.escape(best.label)}): "
                f"{html.escape(format_number(best.value))} - {html.escape(best.run.run_id)}</span>"
            )
        public = best_public(group)
        if public:
            body.append(
                f"<span>best public (info only): {html.escape(format_number(public.value))} - "
                f"{html.escape(public.run.run_id)}</span>"
            )
        cards.append(f'<div class="card">{"<br>".join(body)}</div>')
    header = "".join(f"<th>{html.escape(name)}</th>" for name in TEXT_COLUMNS)
    rows = []
    for record in shown:
        cells = _row(record)
        rendered = "".join(
            _cell(text, "mono" if index in (0, 5) else f"status-{text}" if index == 4 else "")
            for index, text in enumerate(cells)
        )
        rows.append(f"<tr>{rendered}</tr>")
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    scope = f" - {html.escape(task)}" if task else ""
    empty = "" if rows else "<p class='note'>No run in the registry yet.</p>"
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Lab board</title><style>{HTML_STYLE}</style></head><body><main>"
        f"<h1>Experiment board{scope}</h1>"
        f'<p class="note">{html.escape(GUIDANCE)} Generated {html.escape(stamp)}.</p>'
        f'<div class="cards">{"".join(cards)}</div>'
        f'<h2>Runs</h2>{empty}<div class="wrap"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div></main></body></html>\n"
    )


def write_board(ctx: LabContext, task: str | None = None) -> tuple[str, Path]:
    """Render the registry: returns the terminal text and the path of the written HTML page."""
    records = list(ctx.registry.latest().values())
    page = ctx.runs_dir / BOARD_FILENAME
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render_html(records, task, ctx.clock()), encoding="utf-8")
    return render_text(records, task), page

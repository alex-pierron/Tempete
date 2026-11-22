"""Assemble a self-contained team-B2 HTML report (`report-<N>.html`): dashboard first, then evidence.

Text comes from `narrative-<N>.json` (human-written, French); every number in the dashboard and in
the CV table is read from the `b2_train cv` outputs and from the leaderboard snapshot, so the page
cannot drift from the measurements.

Usage:
    python -m hackathon.wmloc.research.b2_report --report-dir reports/phase2/wmloc/team-b2 \
        --salvo 1 --tags ps_base ps_content_only
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path
from typing import Any

from hackathon.wmloc.research.b2_train import results_dir
from hackathon.wmloc.research.build_report import STYLE

REPORT_TITLE = "Watermark Localization, equipe B2 : segmenteur appris sur features gelees"
LEADERBOARD_SNAPSHOT = Path("artifacts/leaderboard/latest.md")
LEADERBOARD_ROW = re.compile(
    r"^\|\s*wmloc\s*\|\s*(?P<rank>\d+/\d+)\s*\|\s*(?P<score>[\d.]+)\s*\|\s*(?P<best>[\d.]+)\s*\|"
)
EXTRA_STYLE = """
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .big{font-size:30px;font-weight:650;line-height:1.1}.card .lab{font-size:12.5px;color:var(--muted)}
"""
TABLE_COLUMNS: tuple[str, ...] = ("all", "full", "rect")


def _esc(value: object) -> str:
    return html.escape(str(value))


def leaderboard_line() -> dict[str, str]:
    """Our wmloc rank, score and the best score from the latest leaderboard snapshot."""
    if not LEADERBOARD_SNAPSHOT.is_file():
        return {}
    for line in LEADERBOARD_SNAPSHOT.read_text(encoding="utf-8").splitlines():
        match = LEADERBOARD_ROW.match(line)
        if match:
            return match.groupdict()
    return {}


def _figure(figures: Path, name: str, captions: dict[str, str]) -> str:
    path = figures / name
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    caption = captions.get(name, "")
    return (
        f'<figure><img alt="{_esc(name)}" src="data:image/png;base64,{encoded}">'
        f"<figcaption>{_esc(caption)}</figcaption></figure>"
    )


def _cards(cards: list[tuple[str, str]]) -> str:
    return (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><div class="big">{_esc(big)}</div><div class="lab">{_esc(lab)}</div></div>'
            for big, lab in cards
        )
        + "</div>"
    )


def _cv_table(tags: list[str]) -> str:
    """One row per (CV run, method): mIoU overall, full regime, rectangle regime."""
    rows = []
    for tag in tags:
        path = results_dir() / f"{tag}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        head = payload.get("head", {})
        config = payload["config"]
        description = (
            f"dropout {head.get('dropout')}, dino {head.get('dino')}, epochs {config['epochs']}, "
            f"lovasz {config['lovasz_weight']}"
        )
        for method, report in payload["reports"].items():
            cells = "".join(f'<td class="num">{report[c]:.3f}</td>' for c in TABLE_COLUMNS)
            tta = report.get("tta_all")
            cells += (
                f'<td class="num">{tta:.3f}</td>' if tta is not None else '<td class="num">-</td>'
            )
            rows.append(
                f"<tr><td><code>{_esc(tag)}</code></td><td>{_esc(method)}</td>{cells}"
                f"<td>{_esc(description)}</td></tr>"
            )
    return (
        "<table><tr><th>run</th><th>methode</th><th>mIoU total</th><th>regime full</th>"
        "<th>regime rectangle</th><th>total avec symetries (TTA)</th><th>reglages</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _tracks_table(
    tracks: list[dict[str, str]], keys: tuple[str, ...], heads: tuple[str, ...]
) -> str:
    rows = "".join(
        "<tr>" + "".join(f"<td>{_esc(t.get(k, ''))}</td>" for k in keys) + "</tr>" for t in tracks
    )
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in heads) + f"</tr>{rows}</table>"


def build(report_dir: Path, salvo: int, tags: list[str]) -> Path:
    """Write `report-<salvo>.html` and return its path."""
    narrative = json.loads((report_dir / f"narrative-{salvo}.json").read_text(encoding="utf-8"))
    figures = report_dir / "figures"
    captions: dict[str, str] = narrative.get("captions", {})
    board = leaderboard_line() if narrative.get("leaderboard_card", True) else {}
    hero = narrative.get("hero", "fig_dashboard_bars.png")
    cards = [(c["big"], c["label"]) for c in narrative["cards"]]
    if board:
        cards.append(
            (f"{board['rank']} / {board['score']}", f"classement public, meilleur {board['best']}")
        )
    ordered = narrative.get("figure_order", sorted(p.name for p in figures.glob("*.png")))
    parts = [
        f"<h1>{_esc(REPORT_TITLE)}</h1>",
        f'<p class="muted">{_esc(narrative["subtitle"])}</p>',
        _cards(cards),
        _figure(figures, hero, captions),
        "<h2>1. Resume decisionnel</h2>",
        '<div class="box"><ul>'
        + "".join(f"<li>{_esc(line)}</li>" for line in narrative["summary"])
        + "</ul></div>",
        "<h2>2. Pistes explorees</h2>",
        _tracks_table(
            narrative["tracks"],
            ("piste", "hypothese", "source", "fait", "resultat", "verdict"),
            ("piste", "hypothese", "source", "ce qui a ete fait", "resultat chiffre", "verdict"),
        ),
        "<h2>3. Metriques</h2>",
        (
            '<p class="muted">mIoU officiel, validation croisee 5 plis (seed 2026) sur les 510 '
            "images reelles du train ; toutes les valeurs sont hors-pli.</p>"
        ),
        _cv_table(tags),
        "".join(f"<p>{_esc(t)}</p>" for t in narrative.get("metrics_notes", [])),
        "<h2>4. Visuels</h2>",
        "".join(_figure(figures, n, captions) for n in ordered if n != hero),
        "<h2>5. Interpretabilite</h2>",
        "".join(f"<p>{_esc(t)}</p>" for t in narrative["interpretability"]),
        "<h2>6. Classement des pistes</h2>",
        _tracks_table(
            narrative["ranking"],
            ("piste", "gain", "cout", "risque", "verdict"),
            ("piste", "gain attendu", "cout", "risque", "verdict"),
        ),
        "<h2>7. Reproductibilite</h2>",
        "".join(f"<p>{_esc(t)}</p>" for t in narrative["reproducibility_text"]),
        "<pre>" + _esc("\n".join(narrative["reproducibility_commands"])) + "</pre>",
    ]
    document = (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(REPORT_TITLE)}</title><style>{STYLE}{EXTRA_STYLE}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )
    out = report_dir / f"report-{salvo}.html"
    out.write_text(document, encoding="utf-8")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the team B2 HTML report.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--salvo", type=int, required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    args: Any = parser.parse_args()
    print(build(args.report_dir, args.salvo, args.tags))


if __name__ == "__main__":
    main()

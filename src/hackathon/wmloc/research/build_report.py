"""Assemble the self-contained HTML report of team A from metrics.json, figures and narrative.json.

`narrative.json` holds the human-written parts (summary, tracks, interpretation, ranking,
reproducibility); every number in the tables comes from metrics.json so the two cannot drift.

Usage: python -m hackathon.wmloc.research.build_report --report-dir reports/phase1/wmloc/team-a
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

from hackathon.wmloc.constants import KNOWN_METHODS

REPORT_TITLE = "Watermark Localization, equipe A : localiser sans apprentissage"
FIGURE_CAPTIONS: dict[str, str] = {
    "fig_switch_agreement.png": "Accord des bits de l'image entiere avec le message connu : les deux "
    "regimes se separent presque parfaitement (AUC proche de 1) pour chaque methode.",
    "fig_miou_by_rule.png": "mIoU hors-pli de chaque regle, par methode et par regime ; la barre "
    "d'erreur est l'ecart-type sur les plis. Les regles constantes sont les references.",
    "fig_iou_distribution.png": "Distribution du mIoU par image du pipeline deployable : "
    "masse proche de 0,45 (retombee sur la constante) et queue de vraies reussites.",
    "fig_overlays_success.png": "Meilleures images hors-pli (vert : verite, rouge : prediction).",
    "fig_overlays_failure.png": "Pires images hors-pli (vert : verite, rouge : prediction).",
    "fig_scale_space_pixelseal.png": "Pixel Seal : l'echelle de sonde qui contraste le mieux suit "
    "la taille du rectangle, comme le predit l'adequation d'echelle.",
    "fig_scale_space_trustmark.png": "TrustMark : meme test que pour Pixel Seal.",
    "fig_evidence_maps_pixelseal.png": "Pixel Seal, regime rectangle : image + verite, z-score de "
    "la carte par position (256), z-score multi-echelle.",
    "fig_evidence_maps_trustmark.png": "TrustMark, regime rectangle : memes cartes.",
}
STYLE = """
:root{--bg:#fff;--fg:#1b1f24;--muted:#57606a;--line:#d0d7de;--card:#f6f8fa;--accent:#0b5cad}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9da7b3;--line:#30363d;--card:#161b22;--accent:#58a6ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0}
main{max-width:1020px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:16px;margin:22px 0 6px}p,li{max-width:80ch}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:12.5px;table-layout:auto;overflow-wrap:anywhere}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--card)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:14px 0}figure img{max-width:100%;height:auto;border:1px solid var(--line)}
figcaption{font-size:13px;color:var(--muted);max-width:80ch}
code,pre{background:var(--card);border-radius:4px;font-size:12.5px}pre{padding:10px;overflow-x:auto}code{padding:1px 4px}
.box{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 14px}
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _figure(figures: Path, name: str) -> str:
    path = figures / name
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    caption = FIGURE_CAPTIONS.get(name, name)
    return (
        f'<figure><img alt="{_esc(name)}" src="data:image/png;base64,{encoded}">'
        f"<figcaption>{_esc(caption)}</figcaption></figure>"
    )


def _rule_table(metrics: dict[str, Any], key: str) -> str:
    methods = [m for m in KNOWN_METHODS if m in metrics]
    names: list[str] = []
    for method in methods:
        for name in metrics[method][key]:
            if name not in names:
                names.append(name)
    head = "".join(f"<th>{_esc(m)}</th>" for m in methods)
    rows = []
    for name in names:
        cells = []
        for method in methods:
            entry = metrics[method][key].get(name)
            cells.append(
                f'<td class="num">{entry["mean"]:.3f} &plusmn; {entry["std"]:.3f}</td>'
                if entry
                else '<td class="num">n/a</td>'
            )
        rows.append(f"<tr><td><code>{_esc(name)}</code></td>{''.join(cells)}</tr>")
    return f"<table><tr><th>regle</th>{head}</tr>{''.join(rows)}</table>"


def _summary_table(metrics: dict[str, Any]) -> str:
    methods = [m for m in KNOWN_METHODS if m in metrics]
    rows = []
    labels = (
        ("baselines.all_zeros", "tout a zero"),
        ("baselines.all_ones", "tout a un"),
        ("baselines.oracle_switch", "aiguillage parfait + constantes"),
        ("baselines.global_switch_cv", "aiguillage global appris (CV) + constantes"),
        ("deployable_full_regime", "pipeline retenu, regime full"),
        ("deployable_rect_regime", "pipeline retenu, regime rectangle"),
        ("deployable_cv_miou", "pipeline retenu, total (hors-pli)"),
    )
    for path, label in labels:
        cells = []
        for method in methods:
            value: Any = metrics[method]
            for part in path.split("."):
                value = value[part]
            cells.append(f'<td class="num">{value:.3f}</td>')
        rows.append(f"<tr><td>{_esc(label)}</td>{''.join(cells)}</tr>")
    head = "".join(f"<th>{_esc(m)}</th>" for m in methods)
    return f"<table><tr><th>mIoU</th>{head}</tr>{''.join(rows)}</table>"


def _switch_table(metrics: dict[str, Any]) -> str:
    rows = []
    for method in KNOWN_METHODS:
        if method not in metrics:
            continue
        switch = metrics[method]["switch"]
        cuts = ", ".join(f"{c:.3f}" for c in switch["cuts_per_fold"])
        rows.append(
            f'<tr><td>{_esc(method)}</td><td class="num">{switch["accuracy"]:.3f}</td>'
            f'<td class="num">{switch["auc"]:.4f}</td><td>{_esc(cuts)}</td>'
            f"<td><code>{_esc(metrics[method]['best_full_rule'])}</code> / "
            f"<code>{_esc(metrics[method]['best_rect_rule'])}</code></td></tr>"
        )
    return (
        "<table><tr><th>methode</th><th>precision aiguillage</th><th>AUC</th>"
        "<th>seuil appris par pli</th><th>regles retenues (full / rect)</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _tracks_table(tracks: list[dict[str, str]]) -> str:
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{_esc(track.get(key, ''))}</td>"
            for key in ("piste", "hypothese", "source", "fait", "resultat", "verdict")
        )
        + "</tr>"
        for track in tracks
    )
    return (
        "<table><tr><th>piste</th><th>hypothese</th><th>source</th><th>ce qui a ete fait</th>"
        f"<th>resultat chiffre</th><th>verdict</th></tr>{rows}</table>"
    )


def _paragraphs(items: list[str]) -> str:
    return "".join(f"<p>{_esc(item)}</p>" for item in items)


def build(report_dir: Path) -> Path:
    """Write `report.html` in `report_dir` and return its path."""
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    narrative = json.loads((report_dir / "narrative.json").read_text(encoding="utf-8"))
    figures = report_dir / "figures"
    figure_names = sorted(p.name for p in figures.glob("*.png")) if figures.is_dir() else []
    parts = [
        f"<h1>{_esc(REPORT_TITLE)}</h1>",
        f'<p class="muted">{_esc(narrative["subtitle"])}</p>',
        "<h2>1. Resume decisionnel</h2>",
        '<div class="box"><ul>'
        + "".join(f"<li>{_esc(line)}</li>" for line in narrative["summary"])
        + "</ul></div>",
        "<h2>2. Pistes explorees</h2>",
        _tracks_table(narrative["tracks"]),
        "<h2>3. Metriques</h2>",
        (
            "<h3>3.1 mIoU hors-pli par methode et par regime (validation croisee, "
            f"{metrics[next(m for m in KNOWN_METHODS if m in metrics)]['folds']} plis)</h3>"
        ),
        (
            '<p class="muted">Metrique officielle recalculee localement (parite verifiee avec le kit) ; '
            "chaque valeur est hors-pli, le seuil etant ajuste sur les plis d'entrainement.</p>"
        ),
        _summary_table(metrics),
        "<h3>3.2 Aiguillage de regime</h3>",
        _switch_table(metrics),
        "<h3>3.3 Regles du regime full (hors-pli, moyenne et ecart-type sur les plis)</h3>",
        _rule_table(metrics, "full_rules"),
        "<h3>3.4 Regles du regime rectangle</h3>",
        _rule_table(metrics, "rect_rules"),
        _paragraphs(narrative.get("metrics_notes", [])),
        "<h2>4. Visuels</h2>",
        "".join(_figure(figures, name) for name in figure_names),
        "<h2>5. Interpretabilite</h2>",
        _paragraphs(narrative["interpretability"]),
        "<h2>6. Classement des pistes</h2>",
        _tracks_table(narrative["ranking"]),
        "<h2>7. Reproductibilite</h2>",
        _paragraphs(narrative["reproducibility_text"]),
        "<pre>" + _esc("\n".join(narrative["reproducibility_commands"])) + "</pre>",
    ]
    document = (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(REPORT_TITLE)}</title><style>{STYLE}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )
    out = report_dir / "report.html"
    out.write_text(document, encoding="utf-8")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the team A HTML report.")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.report_dir))


if __name__ == "__main__":
    main()

"""Standalone HTML report (figures embedded as base64) from `metrics.json` and a narrative file.

Tables are generated from the metrics so no number is retyped; the prose (summary, verdicts,
interpretation, ranking, reproducibility) comes from a JSON narrative written by the analyst after
reading the numbers. Section order is the one required by `docs/hackathon/research_loop.md`.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;--blue:#2a78d6;--card:#f9f9f7}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--card:#222220}}
@media print{:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;--card:#f9f9f7}}
body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,
sans-serif;margin:0 auto;max-width:980px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid
var(--grid);padding-bottom:4px}h3{font-size:14px;margin:18px 0 6px}
.sub{color:var(--ink2);margin:0 0 16px}
table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:12px}
th,td{border-bottom:1px solid var(--grid);padding:4px 6px;text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:12px 0}figure img{max-width:100%;height:auto;border:1px solid var(--grid)}
figcaption{color:var(--ink2);font-size:12px;margin-top:4px}
.card{background:var(--card);border:1px solid var(--grid);padding:10px 14px;margin:10px 0}
code,pre{font:12px ui-monospace,Menlo,monospace}pre{background:var(--card);padding:8px;
overflow-x:auto;border:1px solid var(--grid)}
.verdict{font-weight:600;color:var(--blue)}
.reply{font:11px ui-monospace,Menlo,monospace;color:var(--ink2)}
ul{margin:4px 0 8px 18px;padding:0}
"""
MAX_REPLY_CHARS = 150


def _figure(path: Path, caption: str) -> str:
    """One `<figure>` with a base64 PNG and its caption."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}">'
            f"<figcaption>{caption}</figcaption></figure>")


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for position, cell in enumerate(row):
            klass = ' class="n"' if position >= numeric_from and isinstance(cell, (int, float)) else ""
            cells.append(f"<td{klass}>{_fmt(cell) if not isinstance(cell, str) else cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def data_table(metrics: Mapping[str, Any]) -> str:
    """Per-endpoint counts, determinism and length."""
    rows = []
    for endpoint, info in metrics["data_summary"].items():
        rows.append([f"{endpoint} ({metrics['endpoints'][endpoint].split(' ', 1)[1]})", info["n"],
                     info["identical_repeat_rate"], info["mean_words"], info["median_words"],
                     info["empty"]])
    return _table(["endpoint", "answers", "repeat identical (share of prompts)", "mean words",
                   "median words", "empty"], rows)


def pooled_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Pooled channel AUC table."""
    ordered = sorted(rows, key=lambda r: -(r["auc"] or 0))
    return _table(["channel (pooled)", "AUC", "mean z same", "mean z different", "Cohen d"],
                  [[r["kernel"], r["auc"], r["mean_same"], r["mean_different"], r["cohens_d"]]
                   for r in ordered])


def loeo_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Leave-one-endpoint-out table."""
    return _table(
        ["channel set", "method", "MCC", "95 % CI (pairs bootstrap)", "false same", "false diff.",
         "wrong pairs"],
        [[r["set"], r["method"], r["mcc"], f"[{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}]",
          r["false_same"], r["false_different"], ", ".join(r["wrong_pairs"][:6])] for r in rows])


def sweep_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """MCC versus n table."""
    return _table(
        ["method", "answers / endpoint", "draws", "MCC mean", "sd", "p10", "p90",
         "share MCC >= 0.8"],
        [[r["method"], r["answers_per_endpoint"], r["draws"], r["mcc_mean"],
          r["mcc_sd"], r["mcc_p10"], r["mcc_p90"], r["share_mcc_at_least_0_8"]] for r in rows])


def anchor_table(
    data: Mapping[str, Mapping[str, Mapping[str, float | None]]], endpoints: Mapping[str, str],
) -> str:
    """z of plain S against each other endpoint, by prompt category (one table per kernel)."""
    blocks = []
    for kernel, per_category in data.items():
        names = list(next(iter(per_category.values())))
        rows = [[category, *[values[e] for e in names]] for category, values in per_category.items()]
        headers = ["catégorie", *[endpoints[e].split(" ", 1)[1] for e in names]]
        blocks.append(f"<h3>z de S plain contre chaque endpoint, par catégorie : noyau "
                      f"{html.escape(kernel)}</h3>{_table(headers, rows)}")
    return "".join(blocks)


def examples_html(examples: Sequence[Mapping[str, Any]], endpoints: Mapping[str, str]) -> str:
    """Raw replies of every endpoint for a few prompts."""
    blocks = []
    for item in examples:
        rows = "".join(
            f"<tr><td>{html.escape(endpoints[e])}</td><td class='reply'>"
            f"{html.escape(text[:MAX_REPLY_CHARS])}</td></tr>"
            for e, text in item["replies"].items())
        blocks.append(f"<h3>{html.escape(item['prompt_id'])}: {html.escape(item['prompt'])}</h3>"
                      f"<table>{rows}</table>")
    return "".join(blocks)


def build_html(
    metrics: Mapping[str, Any], narrative: Mapping[str, Any], figures_dir: Path,
) -> str:
    """Assemble the report."""
    def fig(name: str, caption_key: str) -> str:
        path = figures_dir / name
        return _figure(path, narrative["captions"][caption_key]) if path.exists() else ""

    parts = [(f"<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(narrative['title'])}</title><style>{CSS}</style></head><body>"),
             f"<h1>{html.escape(narrative['title'])}</h1>",
             f"<p class='sub'>{narrative['subtitle']}</p>",
             "<h2>1. Résumé décisionnel</h2><div class='card'><ul>"
             + "".join(f"<li>{line}</li>" for line in narrative["summary"]) + "</ul></div>",
             "<h2>2. Pistes explorées</h2>"]
    for piste in narrative["pistes"]:
        parts.append(f"<div class='card'><h3>{piste['name']}</h3>"
                     f"<p><b>Hypothèse</b> : {piste['hypothesis']}<br><b>Source</b> : "
                     f"{piste['source']}<br><b>Fait</b> : {piste['done']}<br><b>Résultat</b> : "
                     f"{piste['result']}<br><b>Verdict</b> : <span class='verdict'>"
                     f"{piste['verdict']}</span></p></div>")
    parts += ["<h2>3. Toutes les métriques</h2>", narrative["metrics_intro"],
              "<h3>3.1 Données collectées et déterminisme</h3>", data_table(metrics),
              narrative["data_note"],
              ("<h3>3.2 Canaux regroupes sur toutes les categories : separation modele different / "
              "meme modele (proxy : les 36 paires des ref, 10 meme modele)</h3>"),
              pooled_table(metrics["pooled"]),
              "<h3>3.3 Idem sur les 4 catégories stables (arith, completion, fact, format)</h3>",
              pooled_table(metrics["pooled_stable_categories"]),
              ("<h3>3.4 MCC en validation croisee laisser-un-endpoint-dehors (proxy du MCC modele "
              "officiel)</h3>"), loeo_table(metrics["loeo"]),
              (f"<p>Kit tel quel, sans aucun ajustement (poids a priori, seuil "
              f"{metrics['kit_default']['tau']}): MCC = {_fmt(metrics['kit_default']['mcc'])}. "
              f"Baseline triviale (tout « même » ou tout « différent ») : MCC = 0.</p>")]
    if "mcc_vs_n" in metrics:
        parts += [("<h3>3.5 MCC en fonction du nombre de reponses par endpoint "
                  "(sous-echantillonnage de prompts)</h3>"), sweep_table(metrics["mcc_vs_n"])]
    parts += ["<h3>3.6 Séparation par catégorie de prompts</h3>",
              anchor_table(metrics["anchor_by_category"], metrics["endpoints"])]
    parts += ["<h2>4. Visuels</h2>"]
    for name, key in (("matrix_prior.png", "matrix_prior"), ("matrix_calibrated.png",
                                                              "matrix_calibrated"),
                      ("dendrogram.png", "dendrogram"), ("channel_auc.png", "channel_auc"),
                      ("category_kernel_auc.png", "category_kernel"), ("roc.png", "roc"),
                      ("traps.png", "traps"), ("mcc_vs_n.png", "mcc_vs_n"),
                      ("z_vs_n.png", "z_vs_n")):
        parts.append(fig(name, key))
    parts += ["<h2>5. Interprétabilité</h2>", narrative["interpretation"],
              "<h3>Réponses brutes (repeat 0) à quelques prompts</h3>",
              examples_html(metrics["examples"], metrics["endpoints"]),
              "<h2>6. Classement des pistes</h2>",
              _table(["piste", "gain attendu", "coût", "risque"], narrative["ranking"],
                     numeric_from=99),
              "<h2>7. Reproductibilité</h2>", narrative["repro"], "</body></html>"]
    return "".join(parts)


def main() -> int:
    """CLI: `--metrics`, `--narrative`, `--figures`, `--out`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--narrative", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    narrative = json.loads(args.narrative.read_text(encoding="utf-8"))
    args.out.write_text(build_html(metrics, narrative, args.figures), encoding="utf-8")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

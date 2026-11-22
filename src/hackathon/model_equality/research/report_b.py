"""Self-contained HTML report of team B (figures embedded as base64), from `metrics.json`.

Layout follows `docs/hackathon/research_loop.md`: decision summary, explored leads, metrics,
visuals, interpretability, ranking, reproducibility. Narrative text lives in `report_text.py`
(it is judgement, so it is written by hand against the numbers); this module only renders.
"""

from __future__ import annotations

import base64
import html
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hackathon.model_equality.prompts.bank import build_bank
from hackathon.model_equality.research import report_text
from hackathon.model_equality.research.candidates import candidate_bank
from hackathon.model_equality.research.figures_b import COLUMN_ORDER, endpoint_label

STYLE = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;
--warn:#eb6834;--good:#0c7a50}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232321;
--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3935;--accent:#3987e5;--warn:#d95926;--good:#0ca30c}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232321;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3935;
--accent:#3987e5;--warn:#d95926;--good:#0ca30c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",
Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}h3{font-size:15px;margin:20px 0 6px}
.meta{color:var(--ink2);font-size:13px}
.summary{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
padding:12px 16px;border-radius:6px}
.summary li{margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:left;vertical-align:top}
th{background:var(--card)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:14px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:6px;padding:8px}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{font-size:12.5px;color:#52514e;margin-top:6px}
code,pre{font-family:ui-monospace,Menlo,monospace;font-size:12px}
pre{background:var(--card);border:1px solid var(--line);padding:8px;border-radius:6px;overflow-x:auto}
.verdict-keep{color:var(--good);font-weight:600}.verdict-drop{color:var(--warn);font-weight:600}
.verdict-dig{color:var(--accent);font-weight:600}
.resp{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;white-space:pre-wrap;word-break:break-word}
@media print{body{font-size:12px}main{padding:0}figure{break-inside:avoid}h2{break-after:avoid}}
"""


def embed_png(path: Path) -> str:
    """Data URI of a PNG file."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure_html(figures: Path, name: str, caption: str) -> str:
    """One `<figure>` with an embedded image and its caption (what to conclude)."""
    return (f'<figure><img alt="{html.escape(caption[:90])}" src="{embed_png(figures / name)}">'
            f"<figcaption>{caption}</figcaption></figure>")


def table_html(header: Sequence[str], rows: Sequence[Sequence[str]], numeric_from: int = 1) -> str:
    """HTML table; columns from `numeric_from` are right aligned."""
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{cell}</td>'
            for i, cell in enumerate(row)) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def fmt(value: float | None, digits: int = 2) -> str:
    """Number to text, `-` for missing / NaN."""
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def counts_table(res: Mapping[str, Any]) -> str:
    """Answers per endpoint with its truth."""
    rows = [[endpoint_label(res["truth"], e), e, str(res["answers_per_endpoint"].get(e, 0))]
            for e in COLUMN_ORDER]
    return table_html(["endpoint (truth)", "id", "answers used"], rows, numeric_from=2)


def group_table_html(res: Mapping[str, Any]) -> str:
    """AUC of every prompt group, per kernel, with 5-95 % interval over prompts."""
    rows: list[list[str]] = []
    for entry in sorted(res["groups"], key=lambda g: (g["group"], g["kernel"])):
        rows.append([
            entry["group"], entry["kernel"], str(entry["prompts"]),
            f'{fmt(entry["auc_hard"])} [{fmt(entry["auc_hard_low"])}, {fmt(entry["auc_hard_high"])}]',
            f'{fmt(entry["auc_easy"])} [{fmt(entry["auc_easy_low"])}, {fmt(entry["auc_easy_high"])}]',
        ])
    return table_html(["group", "kernel", "prompts", "AUC hard (T1,T2 vs S)",
                       "AUC easy (T3,T4 vs S)"], rows, numeric_from=2)


def style_table_html(res: Mapping[str, Any], subset: str) -> str:
    """t statistics of every style feature for every endpoint."""
    header = ["feature"] + [endpoint_label(res["truth"], e) for e in COLUMN_ORDER]
    rows = [[row["feature"]] + [fmt(row["t"].get(e), 1) for e in COLUMN_ORDER]
            for row in res["style"][subset]]
    return table_html(header, rows)


def adherence_table_html(res: Mapping[str, Any]) -> str:
    """Short-reply counts to constrained prompts."""
    header = ["family"] + [endpoint_label(res["truth"], e) for e in COLUMN_ORDER]
    rows = []
    for family, per in res["adherence"].items():
        rows.append([family] + [
            f"{per[e][0]}/{per[e][1]}" if e in per else "-" for e in COLUMN_ORDER])
    return table_html(header, rows)


def agreement_table_html(res: Mapping[str, Any]) -> str:
    """Mean agreement within S, between S endpoints and between S and each T model."""
    names = ("within_S", "S_S", "S_T1", "S_T2", "S_T3", "S_T4")
    rows = [[kernel] + [fmt(vals.get(n), 3) for n in names]
            for kernel, vals in res["agreement"].items()]
    return table_html(["kernel", "same S endpoint", "S vs S", "S vs T1 (fp8)", "S vs T2 (LoRA 0.3)",
                       "S vs T3 (LoRA 0.7)", "S vs T4 (LoRA 1.5)"], rows)


def top_prompts_html(res: Mapping[str, Any]) -> str:
    """Ranked prompts with their text and raw scores (see the reliability caveat in the text)."""
    texts = {**{p.id: p.text for p in build_bank()}, **{p.id: p.text for p in candidate_bank()}}
    rows = [[html.escape(r["prompt"]), html.escape(texts.get(r["prompt"], "")[:110]), r["kernel"],
             fmt(r["excess_hard"], 3), fmt(r["excess_easy"], 3), fmt(r["disguise_leak"], 3),
             fmt(r["score"], 3)] for r in res["top_prompts"]]
    return table_html(["prompt", "text", "kernel", "excess T1/T2", "excess T3/T4",
                       "worst disguise", "score"], rows, numeric_from=3)


BUDGET_TABLE_POINTS: tuple[int, ...] = (12, 24, 48, 72, 96)


def budget_table_html(res: Mapping[str, Any], kernel: str) -> str:
    """AUC at a few budgets for each number of draws per prompt, both statistics."""
    rows: list[list[str]] = []
    for role in ("hard", "easy"):
        for statistic in ("pair", "pool"):
            for repeats in range(1, 4):
                points = res["repeat_vs_diversity"].get(
                    f"{kernel}/{role}/r{repeats}/{statistic}", [])
                by_budget = {point["budget"]: point for point in points}
                cells = [
                    f'{fmt(by_budget[n]["auc_mean"])} ({fmt(by_budget[n]["auc_low"])}-'
                    f'{fmt(by_budget[n]["auc_high"])}; k={by_budget[n]["prompts"]}/'
                    f'{by_budget[n]["pool"]})' if n in by_budget else "-"
                    for n in BUDGET_TABLE_POINTS]
                rows.append([role, statistic, str(repeats)] + cells)
    header = ["pairs", "statistic", "draws/prompt"] + [f"N={n}" for n in BUDGET_TABLE_POINTS]
    return table_html(header, rows, numeric_from=3)


def signature_table_html(examples: Sequence[Mapping[str, Any]], truth: Mapping[str, Any]) -> str:
    """Replies of the nine endpoints to a few prompts (the signature of each variant)."""
    parts: list[str] = []
    for example in examples:
        rows = [[endpoint_label(truth, e),
                 f'<span class="resp">{html.escape(example["replies"].get(e, "-")[:230])}</span>']
                for e in COLUMN_ORDER]
        parts.append(f"<h3>{html.escape(example['prompt_id'])}: "
                     f"<span class=\"meta\">{html.escape(example['text'][:140])}</span></h3>"
                     + table_html(["endpoint", "reply (draw 0, truncated)"], rows, numeric_from=9))
    return "".join(parts)


def render(res: Mapping[str, Any], figures: Path, examples: Sequence[Mapping[str, Any]],
           provenance: Mapping[str, str]) -> str:
    """The full HTML document."""
    text = report_text.sections(res, provenance)
    kernel = report_text.HEADLINE_KERNEL
    captions = report_text.captions(res)
    body = [
        f"<h1>{text['title']}</h1>",
        f'<p class="meta">{text["meta"]}</p>',
        "<h2>1. Résumé décisionnel</h2>",
        f'<div class="summary">{text["summary"]}</div>',
        "<h2>2. Pistes explorées</h2>", text["leads"],
        "<h2>3. Métriques</h2>", text["metrics_intro"],
        "<h3>3.1 Volume de données</h3>", counts_table(res),
        "<h3>3.2 Pouvoir de séparation par groupe de prompts et par noyau</h3>",
        text["group_intro"], group_table_html(res),
        "<h3>3.2b Effet à détecter : accord moyen entre endpoints</h3>", text["agreement_intro"],
        agreement_table_html(res),
        "<h3>3.3 Canaux de style : t de chaque endpoint contre le pool S</h3>",
        text["style_intro"],
        "<p><b>Tous les prompts</b></p>", style_table_html(res, "all prompts"),
        "<p><b>Prompts à réponse courte</b></p>", style_table_html(res, "short-answer prompts"),
        "<h3>3.4 Respect de la contrainte (réponses de 2 mots au plus)</h3>", text["adherence_intro"],
        adherence_table_html(res),
        "<h3>3.5 Répétitions contre diversité et courbe budget</h3>", text["budget_intro"],
        budget_table_html(res, kernel),
        "<h3>3.6 Fiabilité par prompt (split-half)</h3>", text["reliability"],
        "<h2>4. Visuels</h2>",
        figure_html(figures, "fig1_heatmap.png", captions["fig1"]),
        figure_html(figures, "fig2_group_auc.png", captions["fig2"]),
        figure_html(figures, "fig5_length_signature.png", captions["fig5"]),
        figure_html(figures, "fig8_adherence.png", captions["fig8"]),
        figure_html(figures, "fig7_style_t.png", captions["fig7"]),
        figure_html(figures, "fig3_budget_curve.png", captions["fig3"]),
        figure_html(figures, "fig4_prompt_ranking.png", captions["fig4"]),
    ]
    if (figures / "fig9_echo_compliance.png").is_file():
        body.append(figure_html(figures, "fig9_echo_compliance.png", captions["fig9"]))
    if (figures / "fig6_adaptive.png").is_file():
        body.append(figure_html(figures, "fig6_adaptive.png", captions["fig6"]))
    body += [
        "<h3>4.1 Signature de chaque variante : réponses réelles</h3>",
        signature_table_html(examples, res["truth"]),
        "<h3>4.2 Classement brut des prompts</h3>", text["top_intro"], top_prompts_html(res),
        "<h2>5. Interprétabilité</h2>", text["interpretability"],
        "<h2>6. Classement des pistes</h2>", text["ranking"],
        "<h2>7. Plan de dépense ep recommandé</h2>", text["plan"],
        "<h2>8. Nouveaux prompts proposés</h2>", text["prompts"],
        "<h2>9. Reproductibilité</h2>", text["repro"],
        "<h2>10. En cours et suite</h2>", text["in_progress"],
    ]
    return (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>Model Equality, équipe B, phase 1'
            f"</title><style>{STYLE}</style></head><body><main>{''.join(body)}</main></body></html>")


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m ...report_b <team-b dir> <provenance.json>`: write `report.html` next to metrics."""
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(args[0])
    res = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    examples = json.loads((out / "examples.json").read_text(encoding="utf-8"))
    (out / "report.html").write_text(render(res, out / "figures", examples, provenance),
                                     encoding="utf-8")
    print(f"wrote {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

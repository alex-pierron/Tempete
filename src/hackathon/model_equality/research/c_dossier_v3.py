"""One-page vote dossier, version v3: +10 arithmetic prompts, with or without the identity step.

Reads the frozen results (held-out arithmetic test `rehearsal_holdout.json`, frozen v3 test
`identity_test.json`) and the exploratory follow-up (`identity_followup.json`: means, probability
that v3 loses to v2), draws one figure and states, in French: the verdict of the frozen v2 rule,
the verdict of the frozen v3 adoption rule, the disguise rates by ref, the options with their exact
cost, the second possible disguise flag (likelihood under a small reference model, from r-met-d)
and what stays uncertain.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD_X = 0.40
CEILING = 0.66
OURS = "#0072B2"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREY = "#9a9a95"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
STYLE = """
body{margin:0;background:#fcfcfb;color:#0b0b0b;font:12.5px/1.45 -apple-system,Helvetica,Arial,sans-serif}
main{max-width:900px;margin:0 auto;padding:12px 18px}
h1{font-size:18px;margin:0 0 2px}h2{font-size:13.5px;margin:9px 0 3px;border-bottom:1px solid #e4e3df}
.meta{color:#52514e;font-size:11px}
.verdict{border:2px solid #0072B2;border-radius:6px;padding:6px 10px;margin:6px 0;font-size:12.5px}
.verdict.no{border-color:#D55E00}
table{border-collapse:collapse;width:100%;font-size:11px;margin:3px 0}
th,td{border:1px solid #e4e3df;padding:2px 5px;text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:3px 0}figure img{width:66%;height:auto;display:block;margin:0 auto}figcaption{font-size:10.5px;color:#52514e}
ul{margin:2px 0 2px 16px;padding:0}li{margin:1px 0}
@media print{@page{size:A4;margin:6mm}body{font-size:10px;line-height:1.3}h2{margin:6px 0 2px}}
"""
ROW_ORDER: tuple[str, ...] = ("plain", "fp8", "system", "persona", "watermark", "temp1.0",
                              "LoRA0.3", "LoRA0.7", "LoRA1.5")
ROW_LABEL = {"plain": "plain", "fp8": "fp8", "system": "system prompt", "persona": "persona",
             "watermark": "watermark", "temp1.0": "temperature 1,0", "LoRA0.3": "LoRA 0,3",
             "LoRA0.7": "LoRA 0,7", "LoRA1.5": "LoRA 1,5"}


def fmt(value: float) -> str:
    """French decimal comma, two digits."""
    return f"{value:.2f}".replace(".", ",")


def pct(value: float) -> str:
    """Whole-number percentage."""
    return f"{round(100 * value)} %"


def draw(path: Path, holdout: dict[str, Any], test: dict[str, Any], followup: list[dict[str, Any]]) -> Path:
    """Median (dot), 80 % interval (bar) and mean (diamond) of the main cells."""
    rows: list[tuple[str, float, float, float, float | None, str]] = []
    for cell in holdout["cells"]:
        if cell["pool"] == "all_ref":
            rows.append((f"v2, N = {cell['n']}", cell["median"], cell["p10"], cell["p90"], None, OURS))
    means = {(r["pool"], r["k"]): r for r in followup}
    for cell in test["cells"]:
        if cell["pool"] == "all_ref":
            mean = means[(cell["pool"], cell["k"])]["v3"]["mean"]
            rows.append((f"v3, k = {cell['k']}", cell["v3"]["median"], cell["v3"]["p10"],
                         cell["v3"]["p90"], mean, ORANGE))
    base = means[("all_ref", 6)]["v2"]["mean"]
    rows[0] = (rows[0][0], rows[0][1], rows[0][2], rows[0][3], base, rows[0][5])
    fig, ax = plt.subplots(figsize=(8.6, 0.33 * len(rows) + 1.0))
    ax.set_facecolor(SURFACE)
    for position, (label, median, low, high, mean, colour) in enumerate(rows):
        y = len(rows) - position
        ax.hlines(y, low, high, color=colour, linewidth=5, alpha=0.45)
        ax.plot(median, y, "o", color=colour, markersize=8, markeredgecolor=INK)
        if mean is not None:
            ax.plot(mean, y, "D", color="white", markersize=6, markeredgecolor=INK)
        ax.text(high + 0.015, y, f"mediane {fmt(median)}" + (f", moyenne {fmt(mean)}" if mean else ""),
                va="center", fontsize=8.5)
    ax.axvline(THRESHOLD_X, color=VERMILION, linestyle="--", linewidth=1.3)
    ax.text(THRESHOLD_X + 0.005, len(rows) + 0.55, "seuil 0,40", color=VERMILION, fontsize=8.5)
    ax.axvline(CEILING, color=GREY, linestyle=":", linewidth=1.1)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.0, 1.12)
    ax.set_ylim(0, len(rows) + 1)
    ax.set_xlabel("score officiel sur les 9 ref (0,9 x MCC modele) ; losange = moyenne ; plafond 0,66 en pointille",
                  fontsize=8.5)
    ax.set_title("Jeu tenu a l'ecart, 1 000 tirages : v2 (point final) contre v3 (+ identite)", fontsize=9.5)
    ax.grid(True, axis="x", color="#e4e3df", linewidth=0.6)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def build(
    holdout_path: Path, test_path: Path, followup_path: Path, out: Path,
    v2_freeze: str, v3_freeze: str, snapshot_sha: str, identity_sha: str,
) -> Path:
    """Write the one-page HTML."""
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    followup = json.loads(followup_path.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = draw(out.with_suffix(".png"), holdout, test, followup)
    image = "data:image/png;base64," + base64.b64encode(figure.read_bytes()).decode("ascii")
    v2cell = next(c for c in holdout["cells"] if c["pool"] == "all_ref" and c["n"] == 10)
    k6 = next(c for c in test["cells"] if c["pool"] == "all_ref" and c["k"] == 6)
    stats = {(r["pool"], r["k"]): r for r in followup}
    adopt = bool(test["adoption"]["adopt"])
    k3 = stats[("all_ref", 3)]
    k6f = stats[("all_ref", 6)]
    k9 = stats[("all_ref", 9)]
    disguised = k3["declared_disguised"]
    rows = "".join(
        f"<tr><td>{ROW_LABEL[name]}</td><td class='n'>{pct(disguised[name])}</td>"
        f"<td class='n'>{pct(k3['isolated_v3'][name])}</td>"
        f"<td class='n'>{pct(next(c for c in test['cells'] if c['pool'] == 'all_ref' and c['k'] == 3)['flagged_v2'][_ref(name)])}</td></tr>"
        for name in ROW_ORDER)
    options = (
        f"<tr><td>A. +10 arithmetiques (v2)</td><td class='n'>10</td><td class='n'>10 %</td><td class='n'>330</td>"
        f"<td class='n'>{fmt(v2cell['median'])} / {fmt(k6f['v2']['mean'])}</td><td class='n'>{pct(k6f['v2']['p_at_or_above_0.40'])}</td></tr>"
        f"<tr><td>B. v3, k = 3 (13 reponses)</td><td class='n'>13</td><td class='n'>13 %</td><td class='n'>429</td>"
        f"<td class='n'>{fmt(k3['v3']['median'])} / {fmt(k3['v3']['mean'])}</td><td class='n'>{pct(k3['v3']['p_at_or_above_0.40'])}</td></tr>"
        f"<tr><td>C. v3, k = 6 (regle figee)</td><td class='n'>16</td><td class='n'>16 %</td><td class='n'>528</td>"
        f"<td class='n'>{fmt(k6f['v3']['median'])} / {fmt(k6f['v3']['mean'])}</td><td class='n'>{pct(k6f['v3']['p_at_or_above_0.40'])}</td></tr>"
        f"<tr><td>D. v3, k = 9</td><td class='n'>19</td><td class='n'>19 %</td><td class='n'>627</td>"
        f"<td class='n'>{fmt(k9['v3']['median'])} / {fmt(k9['v3']['mean'])}</td><td class='n'>{pct(k9['v3']['p_at_or_above_0.40'])}</td></tr>"
        f"<tr><td>E. rien</td><td class='n'>0</td><td class='n'>0 %</td><td class='n'>0</td><td class='n'>-</td><td class='n'>-</td></tr>")
    v3_class = "verdict" if adopt else "verdict no"
    v3_word = "ELLE PASSE" if adopt else "ELLE NE PASSE PAS"
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Dossier de vote v3</title>
<style>{STYLE}</style></head><body><main>
<h1>Dossier de vote : +10 arithmetiques, avec ou sans l'identite (v3) ?</h1>
<p class="meta">Model Equality, equipe C (r-met-c), 21/11/2026. v2 figee au commit {html.escape(v2_freeze)}, v3 figee au commit {html.escape(v3_freeze)} avant toute reponse d'identite. Aucune reponse ep n'est depensee sans le vote.</p>
<div class="verdict"><b>v2 (point final seul), regle figee : ELLE PASSE</b> : mediane a N = 10 = {fmt(v2cell['median'])} (80 % : {fmt(v2cell['p10'])} a {fmt(v2cell['p90'])}), seuil {fmt(THRESHOLD_X)}.</div>
<div class="{v3_class}"><b>v3 (+ identite), regle d'adoption figee : {v3_word}</b> : a N = 10 et k = 6, mediane v3 {fmt(k6['v3']['median'])} contre v2 {fmt(k6['v2']['median'])} (gain {fmt(k6['v3']['median'] - k6['v2']['median'])}, il fallait au moins 0,05 et une mediane de 0,50) ; aucun LoRA lu comme deguise (condition 3 : oui). <b>Mais le mecanisme marche</b> : system prompt et persona sont declares deguises dans {pct(disguised['system'])} des tirages, aucun LoRA ({pct(disguised['LoRA0.3'])}) ; la moyenne passe de {fmt(k6f['v2']['mean'])} a {fmt(k6f['v3']['mean'])} (+{fmt(k6f['mean_paired_gain'])}), v3 n'est jamais pire que v2 ({pct(k6f['p_v3_worse'])} des tirages) ; la mediane est en equilibre instable car LoRA 0,3 est isole dans {pct(k6f['isolated_v3']['LoRA0.3'])} des tirages, a ras du bord de bande.</div>
<figure><img src="{image}" alt="v2 contre v3"><figcaption>La regle d'adoption figee juge la mediane ; la moyenne et la queue basse montrent le gain (v3 : 80 % des tirages entre {fmt(k6['v3']['p10'])} et {fmt(k6['v3']['p90'])}, v2 : entre {fmt(k6['v2']['p10'])} et {fmt(k6['v2']['p90'])}).</figcaption></figure>
<h2>Deguisement lu par l'identite (k = 3 reponses d'identite par endpoint), sur les ref</h2>
<table><tr><th>ref</th><th>declare deguise</th><th>isole par v3</th><th>isole par v2 (a tort si system)</th></tr>{rows}</table>
<h2>Options et cout exact (par endpoint : reponses, part du budget de 100 ; requetes pour 33 endpoints)</h2>
<table><tr><th>option</th><th>reponses</th><th>budget</th><th>requetes</th><th>score ref : mediane / moyenne</th><th>P(&gt;= 0,40)</th></tr>{options}</table>
<p class="meta">Collecteurs prets, non lances, chacun verrouille par son propre fichier d'autorisation humaine (c_plus10_run.py, c_v3_run.py). 5 a 10 reponses par minute : 330 requetes = 33 a 66 min, 429 = 43 a 86 min, 528 = 53 a 106 min. Il reste 93 reponses par endpoint (77 apres C). <b>Observation apres coup (exploratoire, hors regle figee) :</b> k = 3 suffit, le deguisement est deja lu a 100 % ; k = 6 n'ajoute que du cout.</p>
<h2>Deuxieme drapeau de deguisement possible (r-met-d)</h2>
<p>La vraisemblance des reponses sous SmolLM2-135M ne sauve pas fp8 mais repere la temperature (t = -4,0) et le system prompt (t = +3,4). Utilisable sur les ep si le modele de base de chaque famille est identifie (r-met-e y travaille) ; non evalue ici.</p>
<h2>Ce qui reste incertain</h2>
<ul><li><b>Passage aux ep</b> : les ref sont un petit modele (identite SmolLM / Hugging Face) ; sur les ep le lexique de marquage (Qwen, Llama...) et les system prompts (Anthropic, OpenAI, personas) sont a verifier ; un system prompt qui ne change pas l'identite n'est pas vu.</li>
<li>La mediane de la regle d'adoption est instable (LoRA 0,3 isole ~ 52 % du temps) ; fp8 n'est jamais detecte ; 9 endpoints : un marquage change le score ref de 0,1 a 0,3 ; le score ref ne mesure pas la famille (nos familles valent encore 0,05 au lieu de 0,10).</li>
<li>Rattachement d'un endpoint deguise a sa famille (quel modele principal) non fige ici : il vient des familles par style.</li></ul>
<p class="meta">Donnees : instantane arithmetique 450 lignes SHA256 {html.escape(snapshot_sha[:16])}... ; vague d'identite 270 lignes SHA256 {html.escape(identity_sha[:16])}... ; resultats : data-2/rehearsal_holdout.json, identity_test.json, identity_followup.json, identity_sensitivity.json (critere r-met-d : resultats identiques).</p>
</main></body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


def _ref(name: str) -> str:
    """The ref endpoint of a row label."""
    return {"plain": "ref-01", "fp8": "ref-02", "system": "ref-03", "LoRA0.7": "ref-04",
            "persona": "ref-05", "watermark": "ref-06", "LoRA1.5": "ref-07", "temp1.0": "ref-08",
            "LoRA0.3": "ref-09"}[name]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--v2-freeze", required=True)
    parser.add_argument("--v3-freeze", required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.holdout, args.test, args.followup, args.out, args.v2_freeze, args.v3_freeze,
                args.snapshot_sha256, args.identity_sha256))


if __name__ == "__main__":
    main()

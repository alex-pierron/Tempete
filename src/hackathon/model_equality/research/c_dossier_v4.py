"""One-page vote dossier, version v4: the configuration frozen to reach a median of 0.60 on the ref.

Reads the development grid (`tune_v4b_dev.json`, cells of the frozen edge) and the ONE confirmation
test (`confirmation_v4.json`), draws a figure (median, 80 % interval, mean) against the target
and states in French: the verdict of the frozen rule, the confirmation numbers with and without
the temperature copy, the isolation and disguise rates per ref, the exact cost on the ep of every
configuration, and what stays uncertain.
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

TARGET = 0.60
DELTA = 0.20
FLOOR = 0.45
OURS = "#0072B2"
GREY = "#9a9a95"
VERMILION = "#D55E00"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
KIND = {"ref-01": "plain", "ref-02": "fp8", "ref-03": "system prompt", "ref-04": "LoRA 0,7",
        "ref-05": "persona", "ref-06": "watermark", "ref-07": "LoRA 1,5", "ref-08": "temperature 1,0",
        "ref-09": "LoRA 0,3"}
COSTS = (("N = 10", 13, 429), ("N = 15 (primaire)", 18, 594), ("N = 20", 23, 759))
STYLE = """
body{margin:0;background:#fcfcfb;color:#0b0b0b;font:12.5px/1.45 -apple-system,Helvetica,Arial,sans-serif}
main{max-width:900px;margin:0 auto;padding:12px 18px}
h1{font-size:18px;margin:0 0 2px}h2{font-size:13.5px;margin:9px 0 3px;border-bottom:1px solid #e4e3df}
.meta{color:#52514e;font-size:11px}
.verdict{border:2px solid #0072B2;border-radius:6px;padding:6px 10px;margin:6px 0;font-size:13px}
.verdict.no{border-color:#D55E00}
table{border-collapse:collapse;width:100%;font-size:11px;margin:3px 0}
th,td{border:1px solid #e4e3df;padding:2px 5px;text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:3px 0}figure img{width:70%;height:auto;display:block;margin:0 auto}
ul{margin:2px 0 2px 16px;padding:0}li{margin:1px 0}
@media print{@page{size:A4;margin:6mm}body{font-size:10px;line-height:1.3}h2{margin:6px 0 2px}}
"""


def fmt(value: float) -> str:
    """French decimal comma, two digits."""
    return f"{value:.2f}".replace(".", ",")


def pct(value: float) -> str:
    """Whole-number percentage."""
    return f"{round(100 * value)} %"


def frozen_dev_cells(dev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The development cells of the frozen edge on all 9 ref."""
    return [c for c in dev if c["pool"] == "all_ref" and abs(c["delta"] - DELTA) < 1e-9
            and abs(c["floor"] - FLOOR) < 1e-9]


def draw(path: Path, dev: list[dict[str, Any]], confirmation: dict[str, Any]) -> Path:
    """Median (dot), 80 % interval (bar), mean (diamond) of development and confirmation cells."""
    rows: list[tuple[str, dict[str, Any], str]] = []
    rows += [(f"dev, N = {c['n']}", c, GREY) for c in frozen_dev_cells(dev)]
    primary = confirmation["rule"]["n"]
    for cell in confirmation["cells"]:
        if cell["pool"] == "all_ref":
            tag = " (primaire)" if cell["n"] == primary else ""
            rows.append((f"CONFIRMATION 9 ref, N = {cell['n']}{tag}", cell, OURS))
    for cell in confirmation["cells"]:
        if cell["pool"] == "without_temperature_ref" and cell["n"] == primary:
            rows.append((f"CONFIRMATION 8 ref (sans temp.), N = {cell['n']}", cell, OURS))
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(rows) + 1.0))
    ax.set_facecolor(SURFACE)
    for position, (_label, cell, colour) in enumerate(rows):
        y = len(rows) - position
        ax.hlines(y, cell["p10"], cell["p90"], color=colour, linewidth=5, alpha=0.45)
        ax.plot(cell["median"], y, "o", color=colour, markersize=8, markeredgecolor=INK)
        ax.plot(cell["mean"], y, "D", color="white", markersize=6, markeredgecolor=INK)
        ax.text(cell["p90"] + 0.015, y, f"mediane {fmt(cell['median'])}, moyenne {fmt(cell['mean'])}",
                va="center", fontsize=8.5)
    ax.axvline(TARGET, color=VERMILION, linestyle="--", linewidth=1.3)
    ax.text(TARGET + 0.005, len(rows) + 0.55, "cible 0,60", color=VERMILION, fontsize=8.5)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(0, len(rows) + 1)
    ax.set_xlabel("score officiel sur les ref (0,9 x MCC modele) ; losange = moyenne ; plafond 0,66 (9 ref) / 0,63 (8 ref)",
                  fontsize=8.5)
    ax.set_title("Gris = developpement (optimiste), bleu = confirmation sur 25 prompts jamais vus", fontsize=9.5)
    ax.grid(True, axis="x", color="#e4e3df", linewidth=0.6)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def build(
    dev_path: Path, confirmation_path: Path, out: Path, freeze_sha: str, file_sha: str,
    identity_sha: str,
) -> Path:
    """Write the one-page HTML."""
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = draw(out.with_suffix(".png"), dev, confirmation)
    image = "data:image/png;base64," + base64.b64encode(figure.read_bytes()).decode("ascii")
    primary_n = confirmation["rule"]["n"]
    cells = confirmation["cells"]
    primary = next(c for c in cells if c["pool"] == "all_ref" and c["n"] == primary_n)
    passed = bool(confirmation["verdict_median_reaches_target"])
    css = "verdict" if passed else "verdict no"
    word = "ELLE ATTEINT 0,60" if passed else "ELLE N'ATTEINT PAS 0,60"
    rows = "".join(
        f"<tr><td>{'9 ref' if c['pool'] == 'all_ref' else '8 ref, sans temperature 1,0'}</td>"
        f"<td class='n'>{c['n']}</td><td class='n'>{fmt(c['median'])}</td>"
        f"<td class='n'>{fmt(c['p10'])} a {fmt(c['p90'])}</td><td class='n'>{fmt(c['mean'])}</td>"
        f"<td class='n'>{pct(c['probability_at_or_above_target'])}</td></tr>" for c in cells)
    per_ref = "".join(
        f"<tr><td>{KIND[e]}</td><td class='n'>{pct(primary['declared_disguised'][e])}</td>"
        f"<td class='n'>{pct(primary['isolated'][e])}</td></tr>" for e in KIND)
    costs = "".join(
        f"<tr><td>{label}</td><td class='n'>{answers}</td><td class='n'>{answers} %</td>"
        f"<td class='n'>{requests}</td><td class='n'>{100 - 7 - answers}</td></tr>"
        for label, answers, requests in COSTS)
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Dossier de vote v4</title>
<style>{STYLE}</style></head><body><main>
<h1>Dossier de vote : la configuration v4 atteint-elle 0,60 sur les ref ?</h1>
<p class="meta">Model Equality, equipe C (r-met-c), 21/11/2026. Configuration figee au commit {html.escape(freeze_sha)} avant la lecture des prompts 26 a 50 ; un seul test ; aucune reponse ep n'est depensee sans le vote.</p>
<div class="{css}"><b>Regle figee : {word}</b> : mediane a N = {primary_n} sur 9 ref, prompts de confirmation = {fmt(primary['median'])} (80 % : {fmt(primary['p10'])} a {fmt(primary['p90'])}), moyenne {fmt(primary['mean'])}, probabilite d'atteindre 0,60 : {pct(primary['probability_at_or_above_target'])}. Cible exigee par l'equipe humaine : 0,60.</div>
<figure><img src="{image}" alt="developpement et confirmation"></figure>
<h2>Confirmation (25 prompts jamais vus, 9 ref x 2 repetitions, 1 000 tirages)</h2>
<table><tr><th>ref</th><th>N</th><th>mediane</th><th>intervalle 80 %</th><th>moyenne</th><th>P(&gt;= 0,60)</th></tr>{rows}</table>
<h2>Deguisement lu par l'identite et isolement, par ref (N = {primary_n})</h2>
<table><tr><th>ref</th><th>declare deguise</th><th>isole par v4</th></tr>{per_ref}</table>
<h2>Cout exact sur les ep de chaque configuration (par endpoint, budget de 100 dont 7 deja utilises)</h2>
<table><tr><th>configuration</th><th>reponses</th><th>part du budget</th><th>requetes (33 ep)</th><th>reste apres</th></tr>{costs}</table>
<p class="meta">N prompts arithmetiques + 3 questions d'identite ; 5 a 10 reponses par minute (594 requetes : 1 a 2 h). Collecteur pret, <b>non lance</b>, verrouille par plus10_v4_authorisation.json (N choisi par les humains).</p>
<h2>Ce qui reste incertain</h2>
<ul><li><b>Passage aux ep</b> : les ref sont un petit modele ; le bord est relatif a la famille (25e percentile + 0,20, plancher 0,45) pour suivre les taux de base Qwen et Llama, mais seul le cas d'une famille est testable ici ; un system prompt qui ne change pas l'identite n'est pas vu.</li>
<li>LoRA 0,3 reste pres du bord ; fp8 jamais detecte (plafond 0,66) ; 9 endpoints, un marquage change le score de 0,1 a 0,3 ; le score ref ne mesure pas la famille (nos familles valent encore 0,05 au lieu de 0,10).</li>
<li>La vague d'identite utilisee est celle du developpement ; la piste 3 (vraisemblance de r-met-d : temperature, system prompt) n'est pas dans la v4 et demande le modele de base des ep (r-met-e).</li></ul>
<p class="meta">Donnees : fichier tenu a l'ecart SHA256 {html.escape(file_sha[:16])}... ; identite SHA256 {html.escape(identity_sha[:16])}... ; resultats : data-2/confirmation_v4.json ; developpement : data-2/tune_v4b_dev.json.</p>
</main></body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--freeze-sha", required=True)
    parser.add_argument("--file-sha256", required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.dev, args.confirmation, args.out, args.freeze_sha, args.file_sha256,
                args.identity_sha256))


if __name__ == "__main__":
    main()

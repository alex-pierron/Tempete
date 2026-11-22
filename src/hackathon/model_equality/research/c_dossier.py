"""One-page vote dossier: should the +10 ep answers per endpoint be spent? (French, HTML -> PDF).

Reads the development grid and, when it exists, the held-out result written by
`c_rehearsal holdout`; draws one figure (median and 80 % interval of every cell against the frozen
threshold and the ceilings); states the frozen rule, the verdict, the exact cost and what stays
uncertain. Without a held-out result the page says so and shows the development cells only.
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

from hackathon.model_equality.research.c_rehearsal import KIND

THRESHOLD_X = 0.40
CEILING_FP8_LUMPED = 0.66
BAND_LOW = 0.55
OURS = "#0072B2"
GREY = "#9a9a95"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
REQUESTS = 330
ANSWERS_PER_ENDPOINT = 10
BUDGET_PER_ENDPOINT = 100
LEFT_PER_ENDPOINT = 93
RESERVE = 10
SECONDS_PER_ANSWER_LOW, SECONDS_PER_ANSWER_HIGH = 6, 12
STYLE = """
body{margin:0;background:#fcfcfb;color:#0b0b0b;font:12.5px/1.45 -apple-system,Helvetica,Arial,sans-serif}
main{max-width:900px;margin:0 auto;padding:14px 18px}
h1{font-size:19px;margin:0 0 2px}h2{font-size:14px;margin:12px 0 4px;border-bottom:1px solid #e4e3df}
.meta{color:#52514e;font-size:11.5px}
.verdict{border:2px solid #0072B2;border-radius:6px;padding:8px 12px;margin:8px 0;font-size:14px}
.verdict.no{border-color:#D55E00}.verdict.wait{border-color:#E69F00}
table{border-collapse:collapse;width:100%;font-size:11.5px;margin:4px 0}
th,td{border:1px solid #e4e3df;padding:3px 6px;text-align:left}
td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:4px 0}figure img{width:86%;height:auto;display:block;margin:0 auto}figcaption{font-size:11px;color:#52514e}
ul{margin:3px 0 3px 18px;padding:0}li{margin:2px 0}
@media print{@page{size:A4;margin:7mm}body{font-size:10.5px;line-height:1.35}h2{margin:8px 0 3px}}
"""


def fmt(value: float) -> str:
    """French decimal comma, two digits."""
    return f"{value:.2f}".replace(".", ",")


def cell_label(cell: dict[str, Any], prefix: str) -> str:
    """Row label of a cell."""
    names = {"primary": "9 ref (bank)", "secondary": "7 ref (d4)", "all_ref": "9 ref",
             "without_temperature_ref": "8 ref, sans temperature 1,0"}
    return f"{prefix} {names.get(cell['pool'], cell['pool'])}, N = {cell['n']}"


def draw(path: Path, dev: list[dict[str, Any]], holdout: dict[str, Any] | None) -> Path:
    """Median and 80 % interval of every cell, frozen threshold and ceiling."""
    rows: list[tuple[str, dict[str, Any], str]] = []
    chosen = [c for c in dev if abs(float(c["band_low"]) - BAND_LOW) < 1e-9]
    rows += [(cell_label(c, "dev"), c, GREY) for c in chosen]
    if holdout:
        rows += [(cell_label(c, "TEST"), c, OURS) for c in holdout["cells"]]
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(rows) + 1.1))
    ax.set_facecolor(SURFACE)
    for position, (label, cell, colour) in enumerate(rows):
        y = len(rows) - position
        ax.hlines(y, cell["p10"], cell["p90"], color=colour, linewidth=5, alpha=0.45)
        ax.plot(cell["median"], y, "o", color=colour, markersize=8, markeredgecolor=INK)
        ax.text(max(cell["p90"], cell["median"]) + 0.015, y, fmt(cell["median"]), va="center", fontsize=9)
    ax.axvline(THRESHOLD_X, color=VERMILION, linestyle="--", linewidth=1.4)
    ax.text(THRESHOLD_X + 0.005, len(rows) + 0.55, "seuil fige X = 0,40", color=VERMILION, fontsize=9)
    ax.axvline(CEILING_FP8_LUMPED, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(CEILING_FP8_LUMPED + 0.005, 0.35, "plafond 0,66 (fp8 groupe)", color=GREY, fontsize=8.5)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(-0.02, 0.95)
    ax.set_ylim(0, len(rows) + 1)
    ax.set_xlabel("score officiel sur les ref (0,9 x MCC modele ; la famille n'est pas mesurable)", fontsize=9)
    ax.set_title("Mediane (point) et intervalle a 80 % : gris = developpement, bleu = jeu tenu a l'ecart",
                 fontsize=10)
    ax.grid(True, axis="x", color="#e4e3df", linewidth=0.6)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def build(
    dev_path: Path, holdout_path: Path | None, freeze_sha: str, holdout_sha: str | None,
    out: Path,
) -> Path:
    """Write the one-page HTML."""
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    holdout = (json.loads(holdout_path.read_text(encoding="utf-8"))
               if holdout_path and holdout_path.is_file() else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = draw(out.with_suffix(".png"), dev, holdout)
    image = "data:image/png;base64," + base64.b64encode(figure.read_bytes()).decode("ascii")
    if holdout:
        first = next(c for c in holdout["cells"] if c["pool"] == "all_ref" and c["n"] == 10)
        passed = bool(holdout["verdict_spend_plus_10"])
        css = "verdict" if passed else "verdict no"
        verdict = (f"<b>Verdict de la regle figee : {'ELLE PASSE' if passed else 'ELLE NE PASSE PAS'}</b> "
                   f": mediane a N = 10 sur le jeu tenu a l'ecart = {fmt(first['median'])} "
                   f"(intervalle a 80 % {fmt(first['p10'])} a {fmt(first['p90'])}), seuil {fmt(THRESHOLD_X)}. "
                   f"Aucune reponse ep n'est depensee sans le vote de l'equipe.")
        rows = "".join(
            f"<tr><td>{html.escape(cell_label(c, 'TEST'))}</td><td class='n'>{fmt(c['median'])}</td>"
            f"<td class='n'>{fmt(c['p10'])} a {fmt(c['p90'])}</td>"
            f"<td class='n'>{fmt(c['probability_at_or_above_X'])}</td></tr>" for c in holdout["cells"])
        marks = ", ".join(f"{KIND[e]} {round(100 * f)} %" for e, f in first["flag_frequency"].items())
        second = next((c for c in holdout["cells"] if c["pool"] == "all_ref" and c["n"] == 20), None)
        flat = (f" A N = 20 la mediane est {fmt(second['median'])} : plus de prompts n'ameliorent pas, "
                f"car la copie a system prompt est de plus en plus souvent marquee a tort."
                if second and second["median"] <= first["median"] else "")
        held = (f"<h2>Resultat sur le jeu tenu a l'ecart ({holdout['holdout_prompts']} prompts complets, instantane fige)</h2>"
                f"<table><tr><th>cellule</th><th>mediane</th><th>intervalle 80 %</th><th>P(>= 0,40)</th></tr>{rows}</table>"
                f"<p><b>Marquages a N = 10</b> (part des tirages ou l'endpoint est isole) : {html.escape(marks)}. "
                f"La temperature 1,0 est bien neutralisee ; la copie a system prompt est le faux positif.{flat}</p>"
                f"<p class='meta'>instantane : 450 lignes, SHA256 {html.escape(holdout_sha or 'inconnu')}</p>")
    else:
        css, verdict = "verdict wait", ("<b>Jeu tenu a l'ecart pas encore recu.</b> Ce dossier montre le "
                                        "developpement (gris) ; le verdict est celui du test final.")
        held = ""
    chosen = [c for c in dev if abs(float(c["band_low"]) - BAND_LOW) < 1e-9]
    dev_rows = "".join(
        f"<tr><td>{html.escape(cell_label(c, 'dev'))}</td><td class='n'>{fmt(c['median'])}</td>"
        f"<td class='n'>{fmt(c['p10'])} a {fmt(c['p90'])}</td>"
        f"<td class='n'>{fmt(c['probability_at_or_above_X'])}</td></tr>" for c in chosen)
    dev_block = "" if holdout else (
        "<h2>Developpement (donnees ref existantes)</h2>"
        "<table><tr><th>cellule</th><th>mediane</th><th>intervalle 80 %</th><th>P(>= 0,40)</th></tr>"
        f"{dev_rows}</table>")
    minutes_low = REQUESTS * SECONDS_PER_ANSWER_LOW / 60
    minutes_high = REQUESTS * SECONDS_PER_ANSWER_HIGH / 60
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Dossier de vote +10</title>
<style>{STYLE}</style></head><body><main>
<h1>Dossier de vote : depenser +10 reponses ep par endpoint ?</h1>
<p class="meta">Model Equality, equipe C (r-met-c), 21/11/2026. Chaine figee au commit {html.escape(freeze_sha)} avant l'arrivee du jeu tenu a l'ecart ; le reglage n'a utilise que les donnees ref de developpement.</p>
<div class="{css}">{verdict}</div>
<h2>Ce que l'on teste</h2>
<p>Regle figee : sur 10 prompts arithmetiques d'une ligne, un endpoint dont au moins 55 % des reponses finissent par un point est isole (LoRA) ; tous les autres (copies a system prompt, persona, watermark, temperature 1,0, et fp8) forment le modele principal. On compte le score officiel contre la vraie partition des 9 ref (1 000 tirages). <b>Depenser +10 si la mediane a N = 10 est >= 0,40</b> (60 % du plafond 0,66 : fp8 n'est pas detectable).</p>
<figure><img src="{image}" alt="medianes et intervalles"><figcaption>Developpement : bord de bande 0,55 choisi sur ces donnees (optimiste). Test : regle figee, aucune retouche.</figcaption></figure>
{dev_block}
{held}
<h2>Cout exact de +10</h2>
<ul><li><b>{REQUESTS} reponses</b> = 10 prompts figes x 33 endpoints ; <b>{ANSWERS_PER_ENDPOINT} % du budget</b> par endpoint ({ANSWERS_PER_ENDPOINT} sur {BUDGET_PER_ENDPOINT}) ; il reste {LEFT_PER_ENDPOINT} par endpoint avant, {LEFT_PER_ENDPOINT - ANSWERS_PER_ENDPOINT} apres ({LEFT_PER_ENDPOINT - ANSWERS_PER_ENDPOINT - RESERVE} depensables apres la reserve de {RESERVE}).</li>
<li>Duree : {minutes_low:.0f} a {minutes_high:.0f} minutes a 5 a 10 reponses par minute (debit mesure sous charge) ; collecteur pret, <b>non lance</b>, refuse sans fichier d'autorisation humaine.</li>
<li>Gain attendu sur les ep (simulation du plan chiffre, a prendre comme ordre de grandeur) : environ 0,39 a +10 contre 0,05 aujourd'hui ; +20 : environ 0,50.</li></ul>
<h2>Ce qui reste incertain</h2>
<ul><li><b>Passage aux ep</b> : les ref sont un petit modele unique ; les familles Qwen et Llama des ep ne sont pas testees, la statistique (point final) peut y avoir un autre niveau de base.</li>
<li>LoRA 0,3 est a 60 a 72 % de points, juste au-dessus du bord 0,55 : TPR attendu 0,65 a 0,75. LoRA 1,5 et 0,7 sont attrapes en partie. fp8 jamais.</li>
<li>Nos familles valent encore 0,05 au lieu de 0,10 ; le score ref ne mesure pas la famille. Le score public n'est qu'un sous-ensemble.</li>
<li>Peu de paires sur les ref (9 endpoints) : un seul marquage change le score de 0,1 a 0,3.</li></ul>
</main></body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, default=None)
    parser.add_argument("--freeze-sha", required=True)
    parser.add_argument("--holdout-sha256", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.dev, args.holdout, args.freeze_sha, args.holdout_sha256, args.out))


if __name__ == "__main__":
    main()

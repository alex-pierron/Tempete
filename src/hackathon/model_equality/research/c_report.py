"""Self-contained HTML report of team C (figures embedded as base64), salvo by salvo.

`python -m hackathon.model_equality.research.c_report --number 1 --plan-dir ... --out ...` reads the
planner outputs (`plan.json`, `posterior.npz`), the ep answers and the ref answers, recomputes every
number and figure, and writes `report-<N>.html`. Narrative text is hand-written judgement and lives
in `c_report_text.py`; this module only computes and renders.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research import c_figures as figs
from hackathon.model_equality.research import c_report_text as text
from hackathon.model_equality.research.c_data import (
    AgreementData,
    clique_components,
    comparable,
    load_agreement,
)
from hackathon.model_equality.research.c_oracle import (
    cross_tabulate_many,
    official_score_from_cells,
    public_score_draws,
)
from hackathon.model_equality.research.c_pairs import (
    Partition,
    pair_index,
    pair_states,
    read_partition,
)
from hackathon.model_equality.research.c_ref_power import load_long_units, power_curve

FloatArray = npt.NDArray[np.float64]
PUBLIC_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
PREDICTIVE_DRAWS = 60
REPORT_SEED = 20261121
LEADERBOARD: Mapping[str, float] = {
    "1er (0,75)": 0.75, "Eugenia": 0.40, "Fable 4.2": 0.40, "SLZ": 0.20, "3V1B": 0.10,
}
REF_TRUTH: Mapping[str, str] = {
    "ref-01": "S", "ref-02": "T1", "ref-03": "S", "ref-04": "T3", "ref-05": "S", "ref-06": "S",
    "ref-07": "T4", "ref-08": "S", "ref-09": "T2",
}
REF_KIND: Mapping[str, str] = {
    "ref-01": "plain", "ref-02": "fp8", "ref-03": "system", "ref-04": "LoRA 0.7",
    "ref-05": "persona", "ref-06": "watermark", "ref-07": "LoRA 1.5", "ref-08": "temp 1.0",
    "ref-09": "LoRA 0.3",
}
REQUESTS_PER_SECOND = 0.25
OPENING_WORDS: frozenset[str] = frozenset({"sure", "certainly", "here"})
REF_POWER_SIZES: tuple[int, ...] = (1, 2, 5, 10, 20)
STYLE = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#0072B2;
--warn:#D55E00;--good:#009E73}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232321;
--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3935}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",
Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}h3{font-size:15px;margin:18px 0 6px}
.meta{color:var(--ink2);font-size:13px}
.summary{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
padding:12px 16px;border-radius:6px}.summary li{margin:4px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card .big{font-size:30px;font-weight:700;line-height:1.1}.card .lab{font-size:12.5px;color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:left;vertical-align:top}
th{background:var(--card)}td.num{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:14px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:6px;padding:8px}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{font-size:12.5px;color:#52514e;margin-top:6px}
code,pre{font-family:ui-monospace,Menlo,monospace;font-size:12px}
pre{background:var(--card);border:1px solid var(--line);padding:8px;border-radius:6px;overflow-x:auto}
.keep{color:var(--good);font-weight:600}.drop{color:var(--warn);font-weight:600}
@media print{body{font-size:12px}main{padding:0}figure{break-inside:avoid}h2{break-after:avoid}}
"""


def embed(path: Path) -> str:
    """Data URI of a PNG."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, caption: str) -> str:
    """`<figure>` with an embedded image and a caption saying what to conclude."""
    return (f'<figure><img src="{embed(path)}" alt="{html.escape(caption[:80])}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Plain HTML table."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def ref_agreement_stats(store_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Exact-agreement rates on the ref: by pair class (SS, ST, TT) and of each endpoint with itself."""
    by_prompt: dict[str, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    for name in ("reference_answers.jsonl", "reference_answers_b.jsonl", "reference_answers_m1.jsonl"):
        path = store_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_prompt[row["id"]][row["endpoint"]][int(row["repeat"])] = row["response"]
    totals: dict[str, list[int]] = {"SS": [0, 0], "ST": [0, 0], "TT": [0, 0]}
    for per_endpoint in by_prompt.values():
        for left, right in combinations(sorted(REF_TRUTH), 2):
            if 0 in per_endpoint.get(left, {}) and 0 in per_endpoint.get(right, {}):
                same = REF_TRUTH[left] == REF_TRUTH[right]
                kind = "SS" if same else ("ST" if "S" in (REF_TRUTH[left], REF_TRUTH[right]) else "TT")
                totals[kind][0] += comparable(per_endpoint[left][0]) == comparable(per_endpoint[right][0])
                totals[kind][1] += 1
    pair_rates = {k: v[0] / v[1] for k, v in totals.items() if v[1]}
    self_rates: dict[str, float] = {}
    for endpoint in sorted(REF_TRUTH):
        agree = count = 0
        for per_endpoint in by_prompt.values():
            draws = per_endpoint.get(endpoint, {})
            if 0 in draws and 1 in draws:
                agree += comparable(draws[0]) == comparable(draws[1])
                count += 1
        self_rates[f"{endpoint[4:]} {REF_KIND[endpoint]}"] = agree / count if count else 0.0
    return pair_rates, self_rates


def e1_family_constraint(rows: Sequence[Sequence[str]], index_size: int) -> tuple[list[int], list[float]]:
    """MCC_family of the submitted partition if its larger family were two true families."""
    from hackathon.model_equality.research.c_oracle import mcc_from_counts

    index = pair_index(index_size)
    labels = [r[1] for r in rows]
    larger = Counter(labels).most_common(1)[0][0]
    family = np.array([0 if label == larger else 1 for label in labels])
    guess = family[index.first] == family[index.second]
    members = np.flatnonzero(family == 0)
    sizes = list(range(3, len(members) // 2 + 1))
    values: list[float] = []
    for small in sizes:
        truth_label = np.full(index_size, 2)
        truth_label[members[:small]] = 0
        truth_label[members[small:]] = 1
        truth = truth_label[index.first] == truth_label[index.second]
        tp, fp = float((truth & guess).sum()), float((~truth & guess).sum())
        fn, tn = float((truth & ~guess).sum()), float((~truth & ~guess).sum())
        values.append(float(mcc_from_counts(np.array([tp]), np.array([fp]), np.array([fn]),
                                            np.array([tn]))[0]))
    return sizes, values


def posterior_numbers(
    plan_dir: Path, candidates: Mapping[str, Partition]
) -> dict[str, Any]:
    """Posterior mean / sd of the full score and predictive public-score levels per candidate."""
    posterior = np.load(plan_dir / "posterior.npz")
    index = pair_index(33)
    states = pair_states(posterior["family"], posterior["model"], index)
    weights = posterior["weights"]
    names = list(candidates)
    submitted = np.stack([candidates[n].states(index) for n in names])
    cells = cross_tabulate_many(states, submitted)
    full = official_score_from_cells(cells)
    rng = np.random.default_rng(REPORT_SEED)
    out: dict[str, Any] = {
        "names": names, "mean": [], "sd": [], "levels": {}, "p_above": {},
        "families": np.array([len(set(f.tolist())) for f in posterior["family"]], dtype=float),
        "models": np.array([len(set(m.tolist())) for m in posterior["model"]], dtype=float),
        "weights": weights,
    }
    for position, name in enumerate(names):
        mean = float((weights * full[:, position]).sum())
        out["mean"].append(mean)
        out["sd"].append(float(np.sqrt((weights * (full[:, position] - mean) ** 2).sum())))
        draws = public_score_draws(cells[:, position, :], PREDICTIVE_DRAWS, rng, 0.3, step=0.05)
        levels = [float((weights[:, None] * (np.abs(draws - lo) < 0.025)).sum() / draws.shape[1])
                  for lo in PUBLIC_LEVELS[:-1]]
        levels.append(float((weights[:, None] * (draws > 0.275)).sum() / draws.shape[1]))
        out["levels"][name] = levels
        out["p_above"][name] = float((weights[:, None] * (draws > 0.075)).sum() / draws.shape[1])
    out["mean"] = np.array(out["mean"])
    out["sd"] = np.array(out["sd"])
    return out


def distinct_per_prompt(data: AgreementData) -> list[int]:
    """Distinct normalised answers per prompt."""
    return [len(Counter(comparable(data.texts[p][e]) for e in data.endpoints))
            for p in data.prompts]


def opening_category(answer: str) -> str:
    """'sure', 'certainly', 'here' or 'other': the first word of an answer, punctuation removed."""
    match = re.match(r"\W*([A-Za-z]+)", answer)
    word = match.group(1).lower() if match else ""
    return word if word in OPENING_WORDS else "other"


def long_style_counts(
    data: AgreementData, prompt_ids: Sequence[str], creators: Sequence[str]
) -> dict[str, dict[str, dict[str, int]]]:
    """Opening-word counts per complete long prompt and per stated-creator group."""
    group_of = {e: ("Qwen-like" if c == "alibaba" else "Llama-like" if c == "meta" else "autres")
                for e, c in zip(data.endpoints, creators, strict=True)}
    out: dict[str, dict[str, dict[str, int]]] = {}
    for prompt in prompt_ids:
        if prompt not in data.texts:
            continue
        table: dict[str, dict[str, int]] = {g: {} for g in ("Qwen-like", "Llama-like", "autres")}
        for endpoint, answer in data.texts[prompt].items():
            category = opening_category(answer)
            table[group_of[endpoint]][category] = table[group_of[endpoint]].get(category, 0) + 1
        out[prompt] = table
    return out


def long_form_summary(data: AgreementData, prompt_ids: Sequence[str]) -> dict[str, Any]:
    """Per long prompt: distinct answers, size of the largest identical group, mean length."""
    out: dict[str, Any] = {}
    for pid in prompt_ids:
        if pid not in data.texts:
            continue
        counts = Counter(comparable(t) for t in data.texts[pid].values())
        out[pid] = {
            "distinct": len(counts), "largest_identical_group": counts.most_common(1)[0][1],
            "mean_words": float(np.mean([len(t.split()) for t in data.texts[pid].values()])),
        }
    return out


def build(
    number: int, plan_dir: Path, answers: Path, store_dir: Path, out_dir: Path,
    observed: Mapping[str, float], pending: Sequence[str], sim_curves: Sequence[Sequence[float]],
    extra_candidates: Mapping[str, Path], long_prompt_ids: Sequence[str],
    abc_path: Path | None = None,
) -> Path:
    """Write `report-<number>.html` and its figures; returns the HTML path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / f"figures-{number}"
    fig_dir.mkdir(exist_ok=True)
    plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    data = load_agreement(answers, repeat=0, drop_incomplete=True)
    short_prompts = tuple(p for p in data.prompts if p not in set(long_prompt_ids))
    short = AgreementData(data.endpoints, short_prompts, {p: data.texts[p] for p in short_prompts})
    agreement = short.agreement_count()
    creators = short.creators
    names = data.endpoints
    candidates: dict[str, Partition] = {
        name: Partition.from_labels([r[1] for r in info["rows"]], [r[2] for r in info["rows"]])
        for name, info in plan["known_candidates"].items()
    }
    best_rows = plan["best_response"]
    candidates["best_response"] = Partition.from_labels(
        [r[1] for r in best_rows], [r[2] for r in best_rows])
    for name, path in extra_candidates.items():
        candidates[name] = read_partition(path)
    numbers = posterior_numbers(plan_dir, candidates)
    ref_pair, ref_self = ref_agreement_stats(store_dir)
    sizes, family_mcc = e1_family_constraint(plan["known_candidates"]["e1"]["rows"], 33)
    posterior = np.load(plan_dir / "posterior.npz")
    f: dict[str, Path] = {}
    abc: dict[str, Any] = (json.loads(abc_path.read_text(encoding="utf-8")) if abc_path
                           and abc_path.is_file() else {})
    expected = None
    if "carve_all5" in abc.get("candidates", {}):
        info = abc["candidates"]["carve_all5"]
        expected = ("sonde 4 (attendu, H*)", float(info["mean_full"]), float(info["sd_full"]))
    f["dash"] = figs.fig_dashboard_bars(fig_dir / "dashboard.png", LEADERBOARD, observed, expected)
    f["decomp"] = figs.fig_score_decomposition(fig_dir / "decomposition.png", [
        ("E1 et E1b (public)", 0.5, 0.0), ("1er du classement (0,75)", 1.0, 0.8),
        ("Eugenia / Fable (0,40)", 1.0, 0.33)])
    f["famc"] = figs.fig_family_constraint(fig_dir / "family_constraint.png", sizes, family_mcc,
                                           (0.45, 0.55))
    f["agree"], order = figs.fig_agreement_matrix(fig_dir / "agreement.png", agreement, names,
                                                  creators, len(short_prompts))
    f["dendro"] = figs.fig_dendrogram(fig_dir / "dendrogram.png", agreement, names, creators,
                                      len(short_prompts))
    f["info"] = figs.fig_prompt_information(fig_dir / "prompt_info.png", list(short_prompts),
                                            distinct_per_prompt(short))
    f["ref"] = figs.fig_ref_agreement(fig_dir / "ref_agreement.png", ref_pair, ref_self)
    f["lib"] = figs.fig_library_structure(fig_dir / "library.png", numbers["families"],
                                          numbers["models"], numbers["weights"])
    f["marg"] = figs.fig_marginals(fig_dir / "marginals.png", posterior["p_same_family"],
                                   posterior["p_same_model"], order, names)
    f["graph_e1"] = figs.fig_partition_graph(
        fig_dir / "graph_e1.png", plan["known_candidates"]["e1"]["rows"],
        "E1 : deux familles, un modele chacune (23 + 10)")
    first_probe = next(iter(extra_candidates), "best_response")
    probe_rows = [[n, str(f_id), str(m_id)] for n, f_id, m_id in candidates[first_probe].to_rows()]
    f["graph_p3"] = figs.fig_partition_graph(
        fig_dir / "graph_probe.png", probe_rows,
        f"{first_probe} : familles E1, modeles = consensus du posterieur")
    if "p4" in candidates:
        rows4 = [[n, str(a), str(b)] for n, a, b in candidates["p4"].to_rows()]
        f["graph_p4"] = figs.fig_partition_graph(
            fig_dir / "graph_p4.png", rows4,
            "Sonde 4 : familles E1, un modele par famille sauf 5 suspects isoles")
    f["cands"] = figs.fig_candidates(fig_dir / "candidates.png", numbers["names"], numbers["mean"],
                                     numbers["sd"], observed={k.lower(): v for k, v in observed.items()})
    shown = {n: numbers["levels"][n] for n in ("e1", "e1b", "p3") if n in numbers["levels"]}
    f["pred"] = figs.fig_predictive(fig_dir / "predictive.png", PUBLIC_LEVELS, shown)
    top = plan["top_probes"][:8]
    f["voi"] = figs.fig_voi(fig_dir / "voi.png", [f"sonde {k + 1}" for k in range(len(top))],
                            np.array([t["voi"] for t in top]),
                            np.array([t["mutual_bits"] for t in top]))
    if sim_curves:
        f["sim"] = figs.fig_closed_loop(fig_dir / "closed_loop.png", sim_curves,
                                        [f"verite {k + 1}" for k in range(len(sim_curves))])
    power = power_curve(load_long_units(store_dir), REF_POWER_SIZES, np.random.default_rng(REPORT_SEED))
    f["refpower"] = figs.fig_ref_power(fig_dir / "ref_power.png", power)
    style = long_style_counts(data, long_prompt_ids, list(creators))
    if style:
        f["longstyle"] = figs.fig_long_style(fig_dir / "long_style.png", list(style), style)
    if abc.get("candidates"):
        names_abc = list(abc["candidates"])
        f["abc"] = figs.fig_candidates(
            fig_dir / "abc.png", names_abc,
            np.array([abc["candidates"][n]["mean_full"] for n in names_abc]),
            np.array([abc["candidates"][n]["sd_full"] for n in names_abc]),
            observed={k.lower(): v for k, v in observed.items()},
            title="Hypothese H* conditionnee aux trois 0,05 : isoler 5 suspects domine ; isoler les personas est moins bon")
    f["hist"] = figs.fig_history(fig_dir / "history.png", list(observed.items()), list(pending))
    long_info = long_form_summary(data, long_prompt_ids)
    body = text.render(
        number=number, figures={k: figure(v, text.CAPTIONS[k]) for k, v in f.items()
                                if k in text.CAPTIONS},
        numbers=numbers, plan=plan, observed=observed, pending=list(pending), long_info=long_info,
        agreement=agreement, names=list(names), creators=list(creators), ref_pair=ref_pair,
        ref_self=ref_self, table=table, cliques=clique_components(agreement, 5),
        ref_power=power, style_counts=style, abc=abc,
        probe_name=first_probe, e=html.escape, regex=re,
    )
    page = (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>Model Equality, equipe C, salve '
            f'{number}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>')
    target = out_dir / f"report-{number}.html"
    target.write_text(page, encoding="utf-8")
    return target


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observed", action="append", default=[], help="NAME=SCORE")
    parser.add_argument("--pending", action="append", default=[])
    parser.add_argument("--probe", action="append", default=[], help="NAME=PATH.csv")
    parser.add_argument("--sim-curve", action="append", default=[], help="comma-separated scores")
    parser.add_argument("--long-prompt", action="append", default=[])
    parser.add_argument("--abc", type=Path, default=None, help="abc.json from c_abc_run")
    args = parser.parse_args()
    observed = {k: float(v) for k, v in (item.split("=") for item in args.observed)}
    probes = {k: Path(v) for k, v in (item.split("=") for item in args.probe)}
    curves = [[float(x) for x in c.split(",")] for c in args.sim_curve]
    path = build(args.number, args.plan_dir, args.answers, args.store_dir, args.out, observed,
                 args.pending, curves, probes, args.long_prompt, args.abc)
    print(path)


if __name__ == "__main__":
    main()

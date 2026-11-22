"""Build the team-B phase-1 deliverables: `metrics.json` and `figures/` (the HTML is `report_b`).

Usage (repo root)::

    PYTHONPATH=src python -m hackathon.model_equality.research.build_phase1 \
        --answers artifacts/model_equality/reference_answers_b.jsonl \
        --team-answers /path/to/team_a/reference_answers.jsonl \
        --practice data/hackathon/hf/model_equality/practice_answers.csv \
        --out reports/phase1/model_equality/team-b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hackathon.model_equality.research import adaptive, analysis_b, figures_b, style_features
from hackathon.model_equality.research.candidates import ECHO_EXPECTED
from hackathon.model_equality.research.discrimination import (
    REF_ENDPOINTS,
    agreement_tensor,
    group_answers,
    group_label,
    read_truth,
)
from hackathon.model_equality.store import Answer

WORD_PATTERN = re.compile(r"\S+")
LENGTH_PANEL_GROUPS: tuple[str, ...] = ("yesno", "numeric", "style", "count")
EXIT_OK = 0
TOP_PROMPTS = 15


def reply_lengths(answers: Sequence[Answer], groups: Sequence[str]) -> dict[str, dict[str, list[int]]]:
    """Word counts of replies per `{group: {endpoint: [..]}}`."""
    out: dict[str, dict[str, list[int]]] = {g: defaultdict(list) for g in groups}
    for answer in answers:
        group = group_label(answer.prompt_id)
        if group in out and answer.endpoint in REF_ENDPOINTS:
            out[group][answer.endpoint].append(len(WORD_PATTERN.findall(answer.response)))
    return {g: dict(per) for g, per in out.items()}


ADHERENCE_GROUPS: tuple[str, ...] = ("yesno", "choice", "numeric", "neutral")
ADHERENCE_MAX_WORDS = 2
"""A reply of at most this many words obeys "answer only ..." / "single word" constraints."""


def adherence_counts(
    answers: Sequence[Answer], groups: Sequence[str]
) -> dict[str, dict[str, tuple[int, int]]]:
    """`{group: {endpoint: (short replies, replies)}}` for constrained-answer prompts."""
    out: dict[str, dict[str, list[int]]] = {g: defaultdict(lambda: [0, 0]) for g in groups}
    for answer in answers:
        group = group_label(answer.prompt_id)
        if group in out and answer.endpoint in REF_ENDPOINTS:
            cell = out[group][answer.endpoint]
            cell[1] += 1
            cell[0] += int(len(WORD_PATTERN.findall(answer.response)) <= ADHERENCE_MAX_WORDS)
    return {g: {e: (v[0], v[1]) for e, v in per.items()} for g, per in out.items()}


EXAMPLE_PROMPTS: tuple[str, ...] = (
    "cand-yesno-000", "cand-numeric-000", "cand-style-000", "cand-count-000", "cand-neutral-000",
)


def signature_examples(answers: Sequence[Answer]) -> list[dict[str, Any]]:
    """Draw-0 replies of the nine endpoints to a few prompts, for the signature table."""
    out: list[dict[str, Any]] = []
    for prompt_id in EXAMPLE_PROMPTS:
        rows = [a for a in answers if a.prompt_id == prompt_id and a.repeat == 0]
        if rows:
            out.append({"prompt_id": prompt_id, "text": rows[0].prompt,
                        "replies": {a.endpoint: a.response for a in rows}})
    return out


def _loose(text: str) -> str:
    """Reply normalised for the loose compliance test: case, spaces, quotes, final period."""
    return text.strip().strip("`\"'").strip().rstrip(".").strip().lower()


def echo_compliance(answers: Sequence[Answer]) -> dict[str, dict[str, tuple[int, int]]]:
    """`{strict|loose: {endpoint: (compliant replies, replies)}}` on the echo (copy) prompts.

    strict: the reply equals the expected text after trimming whitespace. loose: also ignores
    case, quotes, code fences and a final period (what a plain `Yes.` would do).
    """
    out: dict[str, dict[str, list[int]]] = {
        "strict": defaultdict(lambda: [0, 0]), "loose": defaultdict(lambda: [0, 0])}
    for answer in answers:
        expected = ECHO_EXPECTED.get(answer.prompt_id)
        if expected is None or answer.endpoint not in REF_ENDPOINTS:
            continue
        reply = answer.response.strip()
        for mode, ok in (("strict", reply == expected),
                         ("loose", _loose(reply) == _loose(expected))):
            out[mode][answer.endpoint][1] += 1
            out[mode][answer.endpoint][0] += int(ok)
    return {m: {e: (v[0], v[1]) for e, v in per.items()} for m, per in out.items()}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--answers", type=Path, required=True, help="team B ref store")
    parser.add_argument("--team-answers", type=Path, default=None, help="team A ref store (read only)")
    parser.add_argument("--practice", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def build(answers: Sequence[Answer], practice: Path, out: Path) -> dict[str, Any]:
    """Compute every metric, write `metrics.json` and the figures, return the metrics."""
    truth = read_truth(practice)
    res = analysis_b.run_analysis(answers, truth)
    grouped0 = group_answers(answers, REF_ENDPOINTS, repeats=[0])
    grouped1 = group_answers(answers, REF_ENDPOINTS, repeats=[1])
    prompts = res["prompts"]
    kernel = analysis_b.HEADLINE_KERNEL
    t0 = agreement_tensor(grouped0, prompts, REF_ENDPOINTS, kernel)
    t1 = agreement_tensor(grouped1, prompts, REF_ENDPOINTS, kernel)
    summary = adaptive.simulate(t0, t1, truth)
    res["adaptive"] = None if summary is None else asdict(summary)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figures_b.figure_heatmap(res, figures / "fig1_heatmap.png")
    figures_b.figure_group_auc(
        res, figures / "fig2_group_auc.png",
        ("onehot_key", "hamming_char", "spectrum", "length"))
    figures_b.figure_budget_curve(res, figures / "fig3_budget_curve.png", kernel)
    figures_b.figure_prompt_ranking(res, figures / "fig4_prompt_ranking.png", kernel)
    lengths = reply_lengths(answers, LENGTH_PANEL_GROUPS)
    figures_b.figure_length_signature(
        {g: v for g, v in lengths.items() if v}, res["truth"], figures / "fig5_length_signature.png")
    if summary is not None:
        figures_b.figure_adaptive([asdict(s) for s in summary.steps], asdict(summary),
                                  figures / "fig6_adaptive.png")
    short_families = {"yesno", "choice", "numeric", "arith", "bank:arith", "bank:fact", "bank:format"}
    subsets = {
        "all prompts": None,
        "short-answer prompts": lambda pid: group_label(pid) in short_families,
    }
    style_tables: dict[str, list[dict[str, Any]]] = {}
    for name, keep in subsets.items():
        style_tables[name] = [
            {"feature": r.feature, "t": r.t, "mean_delta": r.mean_delta, "prompts": r.prompts}
            for r in style_features.style_tests(answers, truth, REF_ENDPOINTS, keep)
        ]
    kernel_for = {g: figures_b.GROUP_KERNEL_KEY if g in figures_b.KEY_GROUPS
                  else figures_b.GROUP_KERNEL_TEXT for g in {r["group"] for r in res["per_prompt"]}}
    res["top_prompts"] = analysis_b.top_prompts(res["per_prompt"], kernel_for, TOP_PROMPTS)
    res["style"] = style_tables
    figures_b.figure_style_t(style_tables, res["truth"], figures / "fig7_style_t.png")
    adherence = adherence_counts(answers, ADHERENCE_GROUPS)
    res["adherence"] = {g: {e: list(v) for e, v in per.items()} for g, per in adherence.items()}
    figures_b.figure_adherence({g: v for g, v in adherence.items() if v}, res["truth"],
                               figures / "fig8_adherence.png")
    compliance = echo_compliance(answers)
    res["echo_compliance"] = {m: {e: list(v) for e, v in per.items()}
                              for m, per in compliance.items()}
    if compliance["strict"]:
        figures_b.figure_adherence(
            compliance, res["truth"], figures / "fig9_echo_compliance.png",
            xlabel="share of replies that obey the copy instruction")
    res["lengths_median"] = {
        g: {e: (sorted(v)[len(v) // 2] if v else None) for e, v in per.items()}
        for g, per in lengths.items()
    }
    (out / "examples.json").write_text(
        json.dumps(signature_examples(answers), indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    paths = [args.answers] + ([args.team_answers] if args.team_answers else [])
    answers = analysis_b.load_reference_answers(paths)
    build(answers, args.practice, args.out)
    print(f"wrote {args.out}/metrics.json and figures from {len(answers)} answers")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

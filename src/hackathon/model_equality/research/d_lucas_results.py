"""Assemble `lucas_results.json`: the answers to Lucas' proposals, in his format (team D).

Lucas' format (report of 21/11/2026, sections 3.4 and 6): which questions separate fp8 (ref-02) from the
undisguised model (ref-01), and which of them resist the disguises. Blocks: A1 (identity flag), B
(families), C (models, fp8 first), plus the MAUVE test of his colleague.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hackathon.model_equality.research.d_identity import (
    draw_curve,
    marking_rates,
    marks_by_endpoint,
    verdict,
)
from hackathon.model_equality.research.d_lucas import (
    block_b_summary,
    evaluate_block_c,
)
from hackathon.model_equality.store import Answer, AnswerStore

BLOCK_C_PREFIX = "d-lucC"
STAGE_ONE_ROWS_PER_ENDPOINT = 5
STAGE_ONE_ROWS_EXPECTED = 36 * 2 * STAGE_ONE_ROWS_PER_ENDPOINT
"""Block C stage 1: 36 questions x 2 endpoints (ref-01, ref-02) x 5 draws."""

FROZEN_COMMITS = {
    "block_A_identity_criterion": "85a279f",
    "blocks_B_C_and_retention_rules": "9c67355",
    "mauve_rule": "34e1521",
}


def load(path: Path) -> list[Answer]:
    """Answers of a store file (empty when the file does not exist yet)."""
    return list(AnswerStore(path).load().values()) if path.exists() else []


def block_a(identity: Sequence[Answer]) -> dict[str, Any]:
    """Block A1: marking rate per ref, verdict and replies needed."""
    if not identity:
        return {}
    marks = marks_by_endpoint(identity)
    rates = marking_rates(marks)
    curve = draw_curve(marks)
    return {"marking_rate": rates, "verdict": verdict(rates), "minimal_draws_per_endpoint": curve.minimal_draws,
            "rows": len(identity)}


def block_c(lucas: Sequence[Answer]) -> dict[str, Any]:
    """Block C: per-question stage results and the three lists Lucas asked for."""
    results = evaluate_block_c(lucas)
    questions = [asdict(r) for r in results if r.stage1_distance is not None]
    stage_one_rows = sum(1 for a in lucas if a.prompt_id.startswith(BLOCK_C_PREFIX) and a.repeat < STAGE_ONE_ROWS_PER_ENDPOINT)
    return {
        "status": {
            "stage1_rows": stage_one_rows,
            "stage1_rows_expected": STAGE_ONE_ROWS_EXPECTED,
            "stage1_complete": stage_one_rows >= STAGE_ONE_ROWS_EXPECTED,
            "questions_evaluated": len(questions),
        },
        "questions": questions,
        "retained_fp8_vs_plain": [r.prompt_id for r in results if r.retained],
        "confirmed_on_independent_draws": [r.prompt_id for r in results if r.confirmed],
        "resist_disguises": [r.prompt_id for r in results if r.resists],
        "not_resisting": [r.prompt_id for r in results if r.confirmed and r.resists is False],
    }


def build(root: Path, out: Path) -> dict[str, Any]:
    """Read the stores and results files of `root` and write the JSON at `out`."""
    lucas = load(root / "artifacts/model_equality/reference_answers_lucas.jsonl")
    identity = load(root / "artifacts/model_equality/reference_answers_identity.jsonl")
    report_dir = root / "reports/phase2/model_equality/team-d"
    mauve = json.loads((report_dir / "mauve_results.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "frozen_before_answers": FROZEN_COMMITS,
        "block_A1_identity": block_a(identity),
        "block_B_families": block_b_summary(lucas) if lucas else {},
        "block_C_models": block_c(lucas) if lucas else {},
        "mauve": mauve,
        "caveats": [
            "ref are ONE family: block B only measures stability on ref; the cross-family contrast needs the ep.",
            ("block C compares ref-01 with ref-02 on 5 draws each: with so few draws a question can pass "
             "stage 1 by chance, hence the confirmation on independent draws."),
            ("likelihood tests with the local SmolLM2-135M were run on the WRONG base model (the ref are "
             "SmolLM2-360M-Instruct) and are withdrawn for fp8."),
        ],
    }
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def summarise(result: Mapping[str, Any]) -> str:
    """One-paragraph text of the headline numbers."""
    c = result.get("block_C_models", {})
    return (f"retained {len(c.get('retained_fp8_vs_plain', []))}, confirmed "
            f"{len(c.get('confirmed_on_independent_draws', []))}, resisting {len(c.get('resist_disguises', []))}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[4]
    data = build(base, base / "reports/phase2/model_equality/team-d/lucas_results.json")
    print(summarise(data))

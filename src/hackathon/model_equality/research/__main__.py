"""`python -m hackathon.model_equality.research`: run the phase-1 analysis and write the outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hackathon.model_equality.pairwise import DEFAULT_PERMUTATIONS
from hackathon.model_equality.prompts import expand_phase, get_phase
from hackathon.model_equality.research.dataset import load_reference_answers, load_truth
from hackathon.model_equality.research.figures import build_all
from hackathon.model_equality.research.pipeline import (
    SWEEP_DRAWS,
    SWEEP_TARGETS,
    run_research,
    write_metrics,
)
from hackathon.model_equality.store import REPO_ROOT, default_answers_path

DEFAULT_TRUTH = REPO_ROOT / "data" / "hackathon" / "hf" / "model_equality" / "practice_answers.csv"
DEFAULT_OUT = REPO_ROOT / "reports" / "phase1" / "model_equality" / "team-a"


def main() -> int:
    """Run the analysis; write `metrics.json` and `figures/*.png` under `--out`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=default_answers_path(reference=True))
    parser.add_argument("--practice-answers", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--no-sweep", action="store_true", help="skip the MCC-versus-n sweep")
    parser.add_argument("--targets", type=int, nargs="+", default=list(SWEEP_TARGETS),
                        help="answers per endpoint at which the MCC-versus-n sweep is measured")
    parser.add_argument("--draws", type=int, default=SWEEP_DRAWS, help="random draws per target")
    parser.add_argument("--sweep-from", type=Path, default=None,
                        help="reuse `mcc_vs_n` and `z_vs_n` from the metrics.json of an earlier "
                             "sweep run (the sweep is the only slow step)")
    parser.add_argument("--phases", nargs="+", default=None,
                        help="analyse only the (prompt, repeat) units of these phases, e.g. ref-1")
    args = parser.parse_args()
    truth = load_truth(args.practice_answers)
    answers = load_reference_answers(args.answers, truth.endpoints)
    if args.phases:
        units = {(unit.prompt.id, unit.repeat) for name in args.phases
                 for unit in expand_phase(get_phase(name))}
        answers = [a for a in answers if (a.prompt_id, a.repeat) in units]
    metrics, artifacts = run_research(answers, truth, args.permutations, sweep=not args.no_sweep,
                                      sweep_targets=args.targets, sweep_draws=args.draws)
    if args.sweep_from is not None:
        earlier = json.loads(args.sweep_from.read_text(encoding="utf-8"))
        metrics["mcc_vs_n"] = earlier["mcc_vs_n"]
        metrics["z_vs_n"] = earlier["z_vs_n"]
        metrics["sweep_source"] = str(args.sweep_from)
    write_metrics(metrics, args.out / "metrics.json")
    figures = build_all(metrics, artifacts, truth, args.out / "figures")
    print(f"metrics: {args.out / 'metrics.json'} ({metrics['duration_s']} s)")
    print("figures: " + ", ".join(sorted(path.name for path in figures.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

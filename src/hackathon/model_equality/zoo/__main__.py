"""`python -m hackathon.model_equality.zoo <command>`: Mac-side helpers of the zoo.

Commands:
    bundle    write the prompt bundle shipped to the cluster (from the committed prompt files)
    evaluate  frozen rules v2 / v3 / likelihood flag on the zoo answers -> evaluation.json
    export    official-format answers + truth + checksums for team C
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hackathon.model_equality.zoo.prompts import PromptBundle, build_bundle

REPO_ROOT = Path(__file__).resolve().parents[4]
HOLDOUT_PROMPTS = REPO_ROOT / "artifacts" / "model_equality" / "holdout_prompts.jsonl"
TEAM_C_REPORTS = REPO_ROOT / "reports" / "phase2" / "model_equality" / "team-c"
PLUS10_PROMPTS = TEAM_C_REPORTS / "plus10_prompts.json"
IDENTITY_PROMPTS = TEAM_C_REPORTS / "identity_prompts.json"
MICRO_ARITHMETIC = 4


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the sub-command."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle", help="write the prompt bundle JSON")
    bundle.add_argument("--out", type=Path, required=True)
    bundle.add_argument("--micro", action="store_true",
                        help="4 arithmetic + 1 identity prompt (preflight run)")
    evaluation = commands.add_parser("evaluate", help="frozen rules on the zoo answers")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--replicates", type=int, default=None)
    evaluation.add_argument("--overlay", type=Path, default=None,
                            help="second run directory whose endpoints replace the main ones")
    export = commands.add_parser("export", help="deliverables for team C")
    export.add_argument("--data", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--no-combined", action="store_true",
                        help="skip zoo_answers.jsonl (the per-family files already hold the rows)")
    args = parser.parse_args(argv)
    if args.command == "bundle":
        built = build_bundle(HOLDOUT_PROMPTS, PLUS10_PROMPTS, IDENTITY_PROMPTS)
        if args.micro:
            built = PromptBundle(built.arithmetic[:MICRO_ARITHMETIC], built.identity[:1])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(built.to_json(), indent=1), encoding="utf-8")
        print(f"bundle: {len(built.arithmetic)} arithmetic, {len(built.identity)} identity "
              f"prompts, {len(built.units())} units -> {args.out}")
    elif args.command == "evaluate":
        from hackathon.model_equality.zoo.evaluate import (
            REPLICATES,
            evaluate_all,
            likelihood_tables,
            load_zoo,
        )

        data = load_zoo(args.data, overlay=args.overlay)
        payload = evaluate_all(data, replicates=args.replicates or REPLICATES)
        payload["likelihood"] = likelihood_tables(args.data, data, args.overlay)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"evaluation: {len(payload['cells'])} cells over {payload['families']} -> {args.out}")  # type: ignore[arg-type]
    else:
        from hackathon.model_equality.zoo.export import export_for_team_c

        manifest = export_for_team_c(args.data, args.out, combined=not args.no_combined)
        print("export:", json.dumps(manifest["files"], indent=0)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())

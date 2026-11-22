"""Command line of the model-equality kit: `python -m hackathon.model_equality <command>`.

Commands (all offline except `budget`, `query-ref`, `query-ep` and `submit`):

    budget            server-side budget per endpoint next to what the local store holds
    plan              exact cost per endpoint of a phase, before any request
    query-ref         query ref-01..ref-09 (unlimited, never scored)
    query-ep          query ep-01..ep-33 (irreplaceable budget: refuses to run without --yes)
    analyze           statistics, calibration on ref, distances, two-level clustering, report
    score-practice    score a ref-* labelling like the official score_practice.py
    make-submission   build a submission CSV from an analysis (labels or re-cut distances)
    validate          check a submission file against the server's format rules
    submit            validate, then send (only with --yes; 30 minute cooldown per attempt)

Exit codes: 0 success, 1 expected failure (bad config, invalid file, server error), 2 refused
because a mandatory confirmation (--yes) is missing.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from hackathon.common.config import Settings, load_settings
from hackathon.common.errors import HackathonError
from hackathon.model_equality import analysis as analysis_module
from hackathon.model_equality.analysis import CALIBRATION_FILENAME, Calibration
from hackathon.model_equality.client import (
    DEFAULT_RESERVE,
    ENDPOINT_LIMIT_DEFAULT,
    MAX_PARALLEL,
    ApiClient,
    BudgetPolicy,
    HttpSession,
    run_jobs,
)
from hackathon.model_equality.clustering import cut_two_levels
from hackathon.model_equality.pairwise import DEFAULT_PERMUTATIONS
from hackathon.model_equality.planning import (
    REFERENCE_ENDPOINTS,
    REFERENCE_PREFIX,
    SCORED_ENDPOINTS,
    SCORED_PREFIX,
    build_plan,
    format_plan,
)
from hackathon.model_equality.prompts import PHASES, expand_phase, get_phase
from hackathon.model_equality.scoring import practice_score, read_label_rows
from hackathon.model_equality.store import AnswerStore, default_answers_path
from hackathon.model_equality.submission import (
    describe_clusters,
    rows_from_labeling,
    submit,
    validate_submission,
    write_submission,
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_REFUSED = 2
DEFAULT_ANALYSIS_DIR = Path("artifacts") / "model_equality" / "analysis"
DEFAULT_SUBMISSION_PATH = Path("artifacts") / "model_equality" / "submissions" / "submission.csv"
PRACTICE_ANSWERS_RELATIVE = Path("hf") / "model_equality" / "practice_answers.csv"
TARGET_EP = "ep"
TARGET_REF = "ref"
Out = Callable[[str], None]


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with one sub-parser per command."""
    parser = argparse.ArgumentParser(prog="python -m hackathon.model_equality",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_store_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--answers", type=Path, default=None,
                         help="answers JSONL (default: $MET_ANSWERS_FILE, or "
                              "$MET_REFERENCE_ANSWERS_FILE for ref, else artifacts/model_equality/)")

    def add_parallel_option(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--max-parallel", type=int, default=MAX_PARALLEL,
                         choices=range(1, MAX_PARALLEL + 1), metavar=f"1..{MAX_PARALLEL}",
                         help=f"simultaneous requests (the team shares {MAX_PARALLEL}: leave "
                              "places to the teammates who query at the same time)")

    def add_selection_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--phase", required=True, choices=sorted(PHASES))
        sub.add_argument("--endpoints", nargs="+", default=None, help="restrict to these endpoints")
        sub.add_argument("--categories", nargs="+", default=None, help="restrict to these categories")

    def add_budget_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--reserve", type=int, default=DEFAULT_RESERVE,
                         help=f"answers per endpoint never spent (default {DEFAULT_RESERVE})")
        sub.add_argument("--limit", type=int, default=ENDPOINT_LIMIT_DEFAULT)

    budget = commands.add_parser("budget", help="server budget per endpoint vs local store")
    add_store_options(budget)

    plan = commands.add_parser("plan", help="exact per-endpoint cost of a phase (offline)")
    add_selection_options(plan)
    add_budget_options(plan)
    add_store_options(plan)
    plan.add_argument("--online", action="store_true",
                      help="also read the server budget (GET /models, costs nothing)")

    query_ref = commands.add_parser("query-ref", help="query the ref endpoints (unlimited)")
    add_selection_options(query_ref)
    add_parallel_option(query_ref)
    add_store_options(query_ref)

    query_ep = commands.add_parser("query-ep", help="query the scored endpoints (needs --yes)")
    add_selection_options(query_ep)
    add_budget_options(query_ep)
    add_store_options(query_ep)
    add_parallel_option(query_ep)
    query_ep.add_argument("--yes", action="store_true", help="really spend budget")

    analyze = commands.add_parser("analyze", help="offline analysis and clustering")
    analyze.add_argument("--target", choices=[TARGET_EP, TARGET_REF], default=TARGET_EP)
    analyze.add_argument("--answers", type=Path, default=None, help="answers of the target")
    analyze.add_argument("--ref-answers", type=Path, default=None, help="ref answers (calibration)")
    analyze.add_argument("--practice-answers", type=Path, default=None,
                         help="official practice_answers.csv (truth of ref-*)")
    analyze.add_argument("--calibrate", action="store_true",
                         help="calibrate weights and thresholds on the ref store first")
    analyze.add_argument("--calibration", type=Path, default=None,
                         help=f"saved calibration (default <out-dir>/{CALIBRATION_FILENAME} if present)")
    analyze.add_argument("--ref-like", nargs="+", default=None, choices=sorted(PHASES),
                         help="keep only the (prompt, repeat) units of these phases in the ref "
                              "calibration, to match the sample shape of the scored endpoints")
    analyze.add_argument("--units-of", nargs="+", default=None, choices=sorted(PHASES),
                         help="keep only the units of these phases in the target")
    analyze.add_argument("--endpoints", nargs="+", default=None,
                         help="restrict the analysis to these endpoints (partial data)")
    analyze.add_argument("--tau-model", type=float, default=None)
    analyze.add_argument("--tau-family", type=float, default=None)
    analyze.add_argument("--no-strip-boilerplate", action="store_true",
                         help="keep persona / system-prompt openers and sign-offs in the answers")
    analyze.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    analyze.add_argument("--out-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)

    score = commands.add_parser("score-practice", help="score a ref-* labelling (official logic)")
    score.add_argument("file", type=Path)
    score.add_argument("--answers", type=Path, default=None, help="practice_answers.csv")

    make = commands.add_parser("make-submission", help="build a submission CSV from an analysis")
    source = make.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels", type=Path, help="labels_ep.csv written by analyze")
    source.add_argument("--distances", type=Path, help="distance_ep.csv: re-cut with thresholds")
    make.add_argument("--tau-model", type=float, default=None)
    make.add_argument("--tau-family", type=float, default=None)
    make.add_argument("--out", type=Path, default=DEFAULT_SUBMISSION_PATH)

    validate = commands.add_parser("validate", help="check a submission file")
    validate.add_argument("file", type=Path)

    submit_parser = commands.add_parser("submit", help="validate then send (needs --yes)")
    submit_parser.add_argument("file", type=Path)
    submit_parser.add_argument("--yes", action="store_true", help="really send the file")
    return parser


def _practice_path(args_path: Path | None, settings: Settings) -> Path:
    """Where practice_answers.csv lives (explicit option, else under the data directory)."""
    return args_path if args_path is not None else settings.data_dir / PRACTICE_ANSWERS_RELATIVE


def _units_of(phase_names: Sequence[str] | None) -> set[tuple[str, int]] | None:
    """Union of the (prompt id, repeat) units of the named phases, or None for no filter."""
    if phase_names is None:
        return None
    return {(unit.prompt.id, unit.repeat)
            for name in phase_names for unit in expand_phase(get_phase(name))}


def _resolve_endpoints(requested: Sequence[str] | None, default: Sequence[str],
                       prefix: str) -> list[str]:
    """Validate a user endpoint list against the expected family of endpoints."""
    chosen = list(requested) if requested else list(default)
    wrong = [e for e in chosen if not e.startswith(prefix)]
    if wrong:
        raise HackathonError(f"endpoints {wrong} do not start with {prefix!r} for this command")
    return chosen


def _make_client(settings: Settings, session: HttpSession | None, out: Out) -> ApiClient:
    return ApiClient(settings, session=session, log=out)


def _cmd_budget(args: argparse.Namespace, client: ApiClient, out: Out) -> int:
    budgets = client.list_budgets()
    stored = {
        TARGET_EP: AnswerStore(args.answers or default_answers_path(reference=False)).count_by_endpoint(),
        TARGET_REF: AnswerStore(default_answers_path(reference=True)).count_by_endpoint(),
    }
    # Columns are space-separated and wide enough for the 1 000 000 ref limit: glued columns
    # ("187999813" = used 187 + left 999813) were once misread as 55 million answers used.
    out(f"{'endpoint':<9} {'limit':>9} {'used':>9} {'left':>9} {'stored':>8}  note")
    for endpoint, info in sorted(budgets.items()):
        mine = stored[TARGET_EP if endpoint.startswith(SCORED_PREFIX) else TARGET_REF][endpoint]
        note = ""
        if endpoint.startswith(SCORED_PREFIX) and info.used > mine:
            note = f"server used {info.used} > stored {mine}: pull the team store / answers not saved"
        out(f"{endpoint:<9} {'-' if info.limit is None else info.limit:>9} {info.used:>9} "
            f"{'-' if info.left is None else info.left:>9} {mine:>8}  {note}")
    return EXIT_OK


def _cmd_plan(args: argparse.Namespace, settings: Settings, session: HttpSession | None,
              out: Out) -> int:
    phase = get_phase(args.phase)
    reference = args.phase.startswith("ref-")
    prefix, default = (REFERENCE_PREFIX, REFERENCE_ENDPOINTS) if reference else (
        SCORED_PREFIX, SCORED_ENDPOINTS)
    endpoints = _resolve_endpoints(args.endpoints, default, prefix)
    store = AnswerStore(args.answers or default_answers_path(reference=reference))
    policy = None if reference else BudgetPolicy(args.limit, args.reserve)
    server_left = None
    if args.online:
        budgets = _make_client(settings, session, out).list_budgets()
        server_left = {e: b.left for e, b in budgets.items() if b.left is not None}
    plan = build_plan(expand_phase(phase, categories=args.categories), endpoints, store.load(),
                      policy, server_left)
    out(f"phase {phase.name}: {phase.description}")
    out(format_plan(plan, policy))
    return EXIT_OK


def _cmd_query(args: argparse.Namespace, settings: Settings, session: HttpSession | None,
               out: Out, *, reference: bool) -> int:
    phase = get_phase(args.phase)
    if reference != phase.name.startswith("ref-"):
        raise HackathonError(
            f"phase {phase.name} cannot be used with {'query-ref' if reference else 'query-ep'}")
    prefix, default = (REFERENCE_PREFIX, REFERENCE_ENDPOINTS) if reference else (
        SCORED_PREFIX, SCORED_ENDPOINTS)
    endpoints = _resolve_endpoints(args.endpoints, default, prefix)
    store = AnswerStore(args.answers or default_answers_path(reference=reference))
    policy = None if reference else BudgetPolicy(args.limit, args.reserve)
    units = expand_phase(phase, categories=args.categories)
    if not reference and not args.yes:
        plan = build_plan(units, endpoints, store.load(), policy)
        out(format_plan(plan, policy))
        out("REFUSED: ep-* answers are irreplaceable (100 per endpoint for the whole competition, "
            "shared by the team). Re-run with --yes to spend the budget shown above.")
        return EXIT_REFUSED
    client = _make_client(settings, session, out)
    budgets = client.list_budgets()
    unknown = [e for e in endpoints if e not in budgets]
    if unknown:
        raise HackathonError(f"unknown endpoints {unknown}; the server lists: {sorted(budgets)}")
    server_left = {e: b.left for e, b in budgets.items() if b.left is not None and e in endpoints}
    if reference:
        ref_limits = [limit for e in endpoints if (limit := budgets[e].limit) is not None]
        if ref_limits:
            # The docs call ref-* unlimited, but on 21/11/2026 the server reported a limit of 100
            # for them too: plan inside it rather than let the server cut the phase at random.
            policy = BudgetPolicy(min(ref_limits), 0)
    plan = build_plan(units, endpoints, store.load(), policy,
                      None if policy is None else server_left)
    out(format_plan(plan, policy))
    report = run_jobs(client, store, plan.jobs, caps=plan.caps(),
                      max_parallel=args.max_parallel, progress=out)
    out("run: " + report.summary())
    for failure in report.failed:
        out(f"FAILED {failure}")
    return EXIT_OK if not report.failed else EXIT_FAILURE


def _cmd_analyze(args: argparse.Namespace, settings: Settings, out: Out) -> int:
    practice = _practice_path(args.practice_answers, settings)
    ref_store = AnswerStore(args.ref_answers or default_answers_path(reference=True))
    out_dir: Path = args.out_dir
    ref_units = _units_of(args.ref_like)
    target_units = _units_of(args.units_of)
    calibration: Calibration | None = None
    if args.target == TARGET_REF or args.calibrate:
        truth = read_label_rows(practice, REFERENCE_PREFIX, with_kind=True)
        if not truth:
            raise HackathonError(f"no ref-* truth rows in {practice}")
        ref_answers = list(ref_store.load().values())
        calibration, _diagnostics, ref_analysis = analysis_module.calibrate_on_reference(
            ref_answers, truth, units=ref_units, n_permutations=args.permutations,
            strip=not args.no_strip_boilerplate)
        labels = [(truth[e][0], truth[e][1]) for e in ref_analysis.endpoints]
        default_run = analysis_module.run_analysis(
            ref_answers, ref_analysis.endpoints, units=ref_units,
            tau_model=analysis_module.DEFAULT_TAU_MODEL, n_permutations=args.permutations,
            strip=not args.no_strip_boilerplate)
        out(f"reference model MCC, prior weights + default cut {analysis_module.DEFAULT_TAU_MODEL}: "
            f"{analysis_module.model_mcc(default_run, labels):.3f}")
        out(f"reference model MCC, calibrated weights + cut {calibration.tau_model:.2f}: "
            f"{analysis_module.model_mcc(ref_analysis, labels):.3f} (IN-SAMPLE, optimistic)")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / CALIBRATION_FILENAME).write_text(calibration.to_json(), encoding="utf-8")
        if args.target == TARGET_REF:
            for path in analysis_module.write_outputs(ref_analysis, ref_answers, out_dir, "ref"):
                out(f"wrote {path}")
            out((out_dir / "report_ref.txt").read_text(encoding="utf-8"))
            return EXIT_OK
    elif args.calibration is not None or (out_dir / CALIBRATION_FILENAME).is_file():
        path = args.calibration or out_dir / CALIBRATION_FILENAME
        calibration = Calibration.from_json(path.read_text(encoding="utf-8"))
        out(f"calibration loaded from {path}")
    else:
        out("WARNING: no calibration (run with --calibrate once ref answers exist): "
            "prior weights and the default cut are used")
    answers = list(AnswerStore(args.answers or default_answers_path(reference=False)).load().values())
    result = analysis_module.run_analysis(
        answers, _resolve_endpoints(args.endpoints, SCORED_ENDPOINTS, SCORED_PREFIX), calibration=calibration, units=target_units,
        tau_model=args.tau_model, tau_family=args.tau_family, n_permutations=args.permutations,
        strip=not args.no_strip_boilerplate)
    for path in analysis_module.write_outputs(result, answers, out_dir, "ep"):
        out(f"wrote {path}")
    out((out_dir / "report_ep.txt").read_text(encoding="utf-8"))
    return EXIT_OK


def _cmd_score_practice(args: argparse.Namespace, settings: Settings, out: Out) -> int:
    out(practice_score(args.file, _practice_path(args.answers, settings)).render())
    return EXIT_OK


def _cmd_make_submission(args: argparse.Namespace, out: Out) -> int:
    if args.labels is not None:
        labels = read_label_rows(args.labels, SCORED_PREFIX)
        rows = rows_from_labeling({e: (f, m) for e, (f, m, _) in labels.items()})
    else:
        endpoints, distance = analysis_module.read_matrix(args.distances)
        family_path = args.distances.with_name(
            args.distances.name.replace("distance_", analysis_module.FAMILY_DISTANCE_PREFIX, 1))
        family_distance = (analysis_module.read_matrix(family_path)[1]
                           if family_path.is_file() else distance)
        calibration_file = args.distances.parent / CALIBRATION_FILENAME
        calibration = (Calibration.from_json(calibration_file.read_text(encoding="utf-8"))
                       if calibration_file.is_file() else None)
        tau = args.tau_model if args.tau_model is not None else (
            calibration.tau_model if calibration else analysis_module.DEFAULT_TAU_MODEL)
        floor = calibration.family_floor if calibration else 0.0
        clustering = cut_two_levels(distance, family_distance, tau, args.tau_family, floor)
        families, models = analysis_module.family_names(clustering)
        rows = [(e, f, m) for e, f, m in zip(endpoints, families, models, strict=True)]
        out(f"cut at tau_model={clustering.tau_model:.2f}, tau_family={clustering.tau_family:.2f}")
    write_submission(rows, args.out)
    validated = validate_submission(args.out)
    out(describe_clusters(validated))
    out(f"wrote {args.out} (valid)")
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace, out: Out) -> int:
    rows = validate_submission(args.file)
    out(describe_clusters(rows))
    out(f"{args.file}: valid")
    return EXIT_OK


def _cmd_submit(args: argparse.Namespace, settings: Settings, out: Out) -> int:
    result = submit(args.file, settings, confirm=args.yes)
    if result is None:
        out(f"{args.file}: valid. DRY RUN: nothing sent. Re-run with --yes to submit "
            "(starts a 30 minute cooldown, a failure included).")
        return EXIT_REFUSED
    out(f"ok={result.ok} score={result.score} previous_best={result.previous_best} "
        f"improved={result.improved} message={result.message!r}")
    return EXIT_OK if result.ok else EXIT_FAILURE


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    session: HttpSession | None = None,
    out: Out = print,
) -> int:
    """Run one command.

    Args:
        argv: Arguments (defaults to `sys.argv[1:]`).
        settings: Settings override (tests); defaults to `load_settings()`.
        session: HTTP session override (tests); defaults to a real `requests.Session`.
        out: Line sink for normal output.

    Returns:
        The process exit code (see the module docstring).
    """
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    try:
        resolved = settings if settings is not None else load_settings()
        command: str = args.command
        if command == "budget":
            code = _cmd_budget(args, _make_client(resolved, session, out), out)
        elif command == "plan":
            code = _cmd_plan(args, resolved, session, out)
        elif command in ("query-ref", "query-ep"):
            code = _cmd_query(args, resolved, session, out, reference=command == "query-ref")
        elif command == "analyze":
            code = _cmd_analyze(args, resolved, out)
        elif command == "score-practice":
            code = _cmd_score_practice(args, resolved, out)
        elif command == "make-submission":
            code = _cmd_make_submission(args, out)
        elif command == "validate":
            code = _cmd_validate(args, out)
        else:
            code = _cmd_submit(args, resolved, out)
    except HackathonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(f"({time.monotonic() - started:.1f} s)", file=sys.stderr)
    return code

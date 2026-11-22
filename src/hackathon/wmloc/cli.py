"""CLI of the localization task: `python -m hackathon.wmloc <command>`.

Commands: fetch, synth, bench, predict, validate, submit. Nothing is ever sent to the server
unless `submit` receives `--yes`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from hackathon.common.config import Settings, load_settings
from hackathon.common.errors import HackathonError
from hackathon.common.submit import submit_file
from hackathon.wmloc.bench import (
    LOCALIZER_FACTORIES,
    LocalizerReport,
    compute_outputs,
    format_reports,
    run_bench,
)
from hackathon.wmloc.constants import (
    DEFAULT_SEED,
    KIT_TASK_ID,
    METHOD_PIXELSEAL,
    RELEASE_SUBDIR,
    WMLOC_DATA_SUBDIR,
)
from hackathon.wmloc.fetch import fetch_dataset
from hackathon.wmloc.local_backend import DEVICE_AUTO, load_local_pixelseal
from hackathon.wmloc.localizers import (
    ConstantLocalizer,
    Localizer,
    PixelMapLocalizer,
    RoutingLocalizer,
    ScoreLocalizer,
    WatermarkBackend,
)
from hackathon.wmloc.manifest import (
    ManifestSample,
    manifest_from_release,
    read_manifest,
    write_manifest,
)
from hackathon.wmloc.submission import save_submission, validate_submission_file
from hackathon.wmloc.synth import (
    EDIT_TYPES,
    EMBED_CROP,
    EMBED_MODES,
    generate_samples,
    list_source_images,
    write_synth_dataset,
)

DEFAULT_N_PER_EDIT = 24
DEFAULT_BENCH_LOCALIZERS = "all_ones,pixel_map_s4,sw_max_96_128"
DEFAULT_BACKEND_FACTORY = "hackathon.wmloc.kit_backend:make_backend"
DEFAULT_SOURCE_FOLDERS: tuple[str, ...] = ("kodak", "div2k_valid", "coco_val_subset")
EXIT_OK = 0
EXIT_ERROR = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hackathon.wmloc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download the gated dataset with HF_TOKEN")
    fetch.add_argument("--repo-id", default=None, help="override the dataset repo id")
    fetch.add_argument("--dest", type=Path, default=None)
    fetch.add_argument("--no-unpack", action="store_true", help="download only")

    prepare = commands.add_parser(
        "prepare", help="write manifest.json for the official release splits"
    )
    prepare.add_argument(
        "--release", type=Path, default=None, help="folder holding train/ and test/"
    )
    prepare.add_argument("--splits", default="train,test")

    synth = commands.add_parser("synth", help="generate the local synthetic bench (Pixel Seal)")
    synth.add_argument("--out", type=Path, required=True)
    synth.add_argument("--n-per-edit", type=int, default=DEFAULT_N_PER_EDIT)
    synth.add_argument("--seed", type=int, default=DEFAULT_SEED)
    synth.add_argument("--embed-mode", choices=EMBED_MODES, default=EMBED_CROP)
    synth.add_argument("--source-dir", type=Path, action="append", default=None)
    synth.add_argument("--device", default=DEVICE_AUTO)

    bench = commands.add_parser("bench", help="mIoU of localizers on a labelled manifest")
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument(
        "--localizers",
        default=DEFAULT_BENCH_LOCALIZERS,
        help=f"comma list among {sorted(LOCALIZER_FACTORIES)}",
    )
    bench.add_argument("--methods", default=METHOD_PIXELSEAL, help="comma list of methods")
    bench.add_argument("--device", default=DEVICE_AUTO)
    bench.add_argument(
        "--backend-factory",
        default=DEFAULT_BACKEND_FACTORY,
        help="module:function(method, device) -> backend or None",
    )
    bench.add_argument("--limit", type=int, default=None, help="first N images only")
    bench.add_argument(
        "--per-edit", type=int, default=None, help="first N images of each edit type only"
    )
    bench.add_argument("--json", type=Path, default=None, help="also write the report as JSON")

    predict = commands.add_parser("predict", help="write a validated submission .pt")
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--localizer", default="sw_max_96_128")
    predict.add_argument(
        "--threshold", type=float, default=None, help="agreement cut (default: the localizer's own)"
    )
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--device", default=DEVICE_AUTO)
    predict.add_argument(
        "--backend-factory",
        default=DEFAULT_BACKEND_FACTORY,
        help="module:function(method, device) -> backend or None",
    )

    validate = commands.add_parser("validate", help="check a submission file")
    validate.add_argument("file", type=Path)
    validate.add_argument(
        "--manifest", type=Path, default=None, help="require exactly the names of this manifest"
    )

    submit = commands.add_parser("submit", help="validate, then send with --yes")
    submit.add_argument("file", type=Path)
    submit.add_argument("--manifest", type=Path, default=None)
    submit.add_argument("--yes", action="store_true", help="really send the file")
    return parser


def _default_sources(settings: Settings) -> list[Path]:
    root = settings.data_dir.parent
    return [root / folder for folder in DEFAULT_SOURCE_FOLDERS]


def _load_backend_factory(spec: str) -> Callable[[str, str], WatermarkBackend | None]:
    """Import `module:function`, the extension point that wires the starter-kit wrappers."""
    module_name, _, function_name = spec.partition(":")
    try:
        factory: Callable[[str, str], WatermarkBackend | None] = getattr(
            importlib.import_module(module_name), function_name
        )
    except (ImportError, AttributeError, ValueError) as exc:
        raise HackathonError(f"cannot load backend factory {spec!r}: {exc}") from exc
    return factory


def _cmd_fetch(args: argparse.Namespace, settings: Settings) -> int:
    result = fetch_dataset(
        settings, repo_id=args.repo_id, destination=args.dest, unpack=not args.no_unpack
    )
    print(f"repo: {result.repo_id}\ndestination: {result.destination}")
    print(
        f"unpacked now: {list(result.extracted)}; already unpacked: "
        f"{list(result.already_extracted)}; absent archives: {list(result.missing)}"
    )
    return EXIT_OK


def _cmd_synth(args: argparse.Namespace, settings: Settings) -> int:
    backend = load_local_pixelseal(args.device)
    sources = list_source_images(args.source_dir or _default_sources(settings))
    started = time.monotonic()
    count = write_synth_dataset(
        args.out,
        generate_samples(sources, backend.embed, args.n_per_edit, args.seed, args.embed_mode),
    )
    print(
        f"wrote {count} samples ({', '.join(EDIT_TYPES)}) to {args.out} "
        f"in {time.monotonic() - started:.0f} s (embed mode {args.embed_mode}, "
        f"seed {args.seed}, device {backend.device})"
    )
    return EXIT_OK


def _take_per_edit(samples: list[ManifestSample], count: int) -> list[ManifestSample]:
    """First `count` samples of each edit type, original order kept."""
    seen: dict[str | None, int] = {}
    kept: list[ManifestSample] = []
    for sample in samples:
        seen[sample.edit] = seen.get(sample.edit, 0) + 1
        if seen[sample.edit] <= count:
            kept.append(sample)
    return kept


def _cmd_prepare(args: argparse.Namespace, settings: Settings) -> int:
    release = args.release or settings.data_dir / WMLOC_DATA_SUBDIR / RELEASE_SUBDIR
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        samples = manifest_from_release(release / split)
        write_manifest(release / split / "manifest.json", samples)
        counts = Counter(sample.method for sample in samples)
        print(f"{split}: {len(samples)} samples {dict(counts)} -> {release / split}/manifest.json")
    return EXIT_OK


def _cmd_bench(args: argparse.Namespace, settings: Settings) -> int:
    all_samples = read_manifest(args.manifest)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    factory = _load_backend_factory(args.backend_factory)
    names = [name.strip() for name in args.localizers.split(",") if name.strip()]
    done: list[LocalizerReport] = []
    for method in methods:
        samples = [s for s in all_samples if s.method == method]
        if args.per_edit is not None:
            samples = _take_per_edit(samples, args.per_edit)
        samples = samples[: args.limit]
        if not samples:
            raise HackathonError(f"no {method} sample in the manifest")
        backend = factory(method, args.device)
        if backend is None:
            raise HackathonError(f"the backend factory returned no backend for {method}")

        def show(report: LocalizerReport, method: str = method) -> None:
            """Print and persist each localizer as soon as it finishes (a crash keeps the rest)."""
            report.name = f"{report.name}[{method}]"
            done.append(report)
            print(format_reports([report]), flush=True)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(
                    json.dumps([asdict(r) for r in done], indent=1), encoding="utf-8"
                )

        run_bench(names, backend, samples, args.manifest.parent, on_report=show)
    return EXIT_OK


def _cmd_predict(args: argparse.Namespace, settings: Settings) -> int:
    samples = read_manifest(args.manifest)
    if args.localizer not in LOCALIZER_FACTORIES:
        raise HackathonError(
            f"unknown localizer {args.localizer!r}; known: {sorted(LOCALIZER_FACTORIES)}"
        )
    factory = _load_backend_factory(args.backend_factory)
    routes: dict[str, Localizer] = {}
    for method in sorted({sample.method for sample in samples}):
        backend = factory(method, args.device)
        if backend is None:
            continue
        chosen = LOCALIZER_FACTORIES[args.localizer](backend)
        if isinstance(chosen, PixelMapLocalizer) and backend.map_stride is None:
            raise HackathonError(
                f"{args.localizer} needs a spatial logit map but the {method} "
                "backend is decode-only: use a sw_* or rect_refine_* localizer"
            )
        if args.threshold is not None and isinstance(chosen, ScoreLocalizer):
            chosen.threshold = args.threshold
        routes[method] = chosen
    # Methods without a backend yet (TrustMark, StegaStamp) get the all-ones constant: with the
    # truth mostly watermarked it is the best constant, and it keeps the file complete.
    router = RoutingLocalizer(routes, ConstantLocalizer(1))
    masks = compute_outputs(router, samples, args.manifest.parent, scores=False)
    names = [sample.name for sample in samples]
    report = save_submission(args.out, names, masks, expected_names=names)
    routed = sum(1 for s in samples if s.method in routes)
    print(
        f"wrote {args.out}: {report.n_images} images, {report.n_bytes} bytes, "
        f"{report.watermarked_fraction:.3f} of cells marked; {routed} images localized by "
        f"{args.localizer} (methods {sorted(routes)}), {len(samples) - routed} by the "
        "all-ones fallback"
    )
    return EXIT_OK


def _expected_names(manifest: Path | None) -> list[str] | None:
    return [s.name for s in read_manifest(manifest)] if manifest else None


def _cmd_validate(args: argparse.Namespace, settings: Settings) -> int:
    report = validate_submission_file(args.file, _expected_names(args.manifest))
    print(
        f"VALID: {report.n_images} images, shape {report.shape}, {report.dtype}, "
        f"{report.n_bytes} bytes, {report.watermarked_fraction:.3f} of cells marked"
    )
    for warning in report.warnings:
        print(f"warning: {warning}")
    return EXIT_OK


def _cmd_submit(args: argparse.Namespace, settings: Settings) -> int:
    report = validate_submission_file(args.file, _expected_names(args.manifest))
    print(f"valid: {report.n_images} images, {report.n_bytes} bytes")
    if not args.yes:
        print("[dry run] nothing sent; add --yes to submit")
        return EXIT_OK
    task_id = settings.wmloc_task_id or KIT_TASK_ID
    result = submit_file(task_id, args.file, settings, confirm=True)
    print(json.dumps(asdict(result), indent=2))
    return EXIT_OK if result.ok else EXIT_ERROR


COMMANDS = {
    "fetch": _cmd_fetch,
    "prepare": _cmd_prepare,
    "synth": _cmd_synth,
    "bench": _cmd_bench,
    "predict": _cmd_predict,
    "validate": _cmd_validate,
    "submit": _cmd_submit,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point; expected failures print `error: ...` and return 1."""
    args = _build_parser().parse_args(argv)
    torch.set_grad_enabled(False)
    try:
        return COMMANDS[args.command](args, load_settings())
    except HackathonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

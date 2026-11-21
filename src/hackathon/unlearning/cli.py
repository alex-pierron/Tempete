"""Command line of the unlearning task: fetch, info, proxy-eval, run, validate, submit.

All paths default under `<HACKATHON_DATA_DIR>/unlearning` (see hackathon.common.config). Nothing
is sent to the server unless `submit --yes` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import timm
import torch

from hackathon.common.config import Settings, load_settings
from hackathon.common.errors import HackathonError
from hackathon.unlearning.alterations import ALTERATIONS
from hackathon.unlearning.configs import (
    METHOD_NAMES,
    PRESETS,
    UnlearnConfig,
    config_to_dict,
    load_config,
)
from hackathon.unlearning.constants import (
    BYTES_PER_MEGABYTE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    FORGET_IMAGES_FILENAME,
    MODEL_FILENAME,
    OUTPUT_SUBDIR,
    default_task_dir,
)
from hackathon.unlearning.data import (
    is_full_forget_set,
    load_forget_images,
    load_proxy_images,
    select_subset,
)
from hackathon.unlearning.fetch import fetch_task_files, model_matches_official
from hackathon.unlearning.methods import unlearn_with_history
from hackathon.unlearning.model import build_model, load_model, select_device
from hackathon.unlearning.proxy_eval import (
    ProxyReport,
    cache_fingerprint,
    evaluate_proxy,
    format_report,
    load_logits_cache,
    save_logits_cache,
)
from hackathon.unlearning.submission import save_submission, submit_validated
from hackathon.unlearning.validate import validate_submission

CACHE_SUBDIR = "cache"
ORIGINAL_LOGITS_CACHE = "orig_logits.pt"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
SECONDS_FORMAT = "{:.1f} s"


def _task_paths(settings: Settings, task_dir: Path | None) -> tuple[Path, Path, Path]:
    """(task_dir, model_path, forget_path)."""
    directory = task_dir or default_task_dir(settings.data_dir)
    return directory, directory / MODEL_FILENAME, directory / FORGET_IMAGES_FILENAME


def _proxy_root(settings: Settings, explicit: str | None) -> Path:
    """Folder holding coco_val_subset / div2k_valid / kodak: `<repo>/data` by default."""
    return Path(explicit) if explicit else settings.data_dir.parent


def _run_proxy_eval(
    *,
    original: torch.nn.Module,
    candidate: torch.nn.Module,
    forget: torch.Tensor,
    retain: torch.Tensor,
    variants: tuple[str, ...],
    seed: int,
    batch_size: int,
    task_dir: Path,
) -> ProxyReport:
    """Proxy report with the original logits cached on disk across invocations."""
    device = next(original.parameters()).device
    fingerprint = cache_fingerprint(forget, retain, variants, seed, device)
    cache_path = task_dir / CACHE_SUBDIR / ORIGINAL_LOGITS_CACHE
    cache = load_logits_cache(cache_path, fingerprint)
    known = set(cache)
    report = evaluate_proxy(
        original, candidate, forget, retain, variants=variants, seed=seed, batch_size=batch_size,
        original_logits_cache=cache,
    )  # fmt: skip
    if set(cache) != known:
        save_logits_cache(cache_path, fingerprint, cache)
    return report


def cmd_fetch(args: argparse.Namespace, settings: Settings) -> int:
    """Download and verify model.pt and forget_images.npz."""
    task_dir, _, _ = _task_paths(settings, args.task_dir)
    started = time.monotonic()
    for item in fetch_task_files(task_dir, settings.hf_token):
        state = "downloaded" if item.downloaded else "already present"
        print(f"{item.path}  {item.size_bytes} bytes  sha256 {item.sha256}  ({state})")
    print(SECONDS_FORMAT.format(time.monotonic() - started))
    return 0


def cmd_info(args: argparse.Namespace, settings: Settings) -> int:
    """Print what is on disk, the environment and the model."""
    task_dir, model_path, forget_path = _task_paths(settings, args.task_dir)
    print(f"task dir        : {task_dir}")
    print(f"torch / timm    : {torch.__version__} / {timm.__version__}")
    print(f"device (auto)   : {select_device(args.device)}")
    if model_path.is_file():
        official = "official sha256 OK" if model_matches_official(model_path) else "NOT OFFICIAL"
        print(f"model.pt        : {model_path.stat().st_size} bytes, {official}")
        parameters = sum(p.numel() for p in build_model().parameters())
        print(f"architecture    : vit_base_patch16_224, 100 classes, {parameters:,} parameters")
    else:
        print("model.pt        : missing (run fetch)")
    if forget_path.is_file():
        forget = load_forget_images(forget_path)
        full = "full set" if is_full_forget_set(forget) else "NOT 3500 images"
        print(f"forget images   : {tuple(forget.shape)} {forget.dtype}, {full}")
    else:
        print("forget_images   : missing (run fetch)")
    return 0


def cmd_proxy_eval(args: argparse.Namespace, settings: Settings) -> int:
    """Proxy-evaluate a candidate state dict (default: the original model = identity baseline)."""
    task_dir, model_path, forget_path = _task_paths(settings, args.task_dir)
    device = select_device(args.device)
    forget = load_forget_images(forget_path, args.forget_limit)
    retain = load_proxy_images(_proxy_root(settings, args.proxy_root), args.retain_count, args.seed)
    original = load_model(model_path, device)
    candidate = load_model(args.candidate, device) if args.candidate else original
    started = time.monotonic()
    report = _run_proxy_eval(
        original=original, candidate=candidate, forget=forget, retain=retain,
        variants=tuple(args.variants), seed=args.seed, batch_size=args.batch_size,
        task_dir=task_dir,
    )  # fmt: skip
    label = str(args.candidate) if args.candidate else "original (identity baseline)"
    print(format_report(report, f"candidate: {label} | device {device}"))
    print(SECONDS_FORMAT.format(time.monotonic() - started))
    if args.json_out:
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    """Unlearn, save a validated submission file, and print the proxy table."""
    task_dir, model_path, forget_path = _task_paths(settings, args.task_dir)
    overrides: dict[str, object] = {}
    for assignment in args.set or []:
        key, separator, value = assignment.partition("=")
        if not separator:
            raise HackathonError(f"--set expects key=value, got '{assignment}'")
        overrides[key.strip()] = value
    if args.method:
        overrides["method"] = args.method
    if args.device:
        overrides["device"] = args.device
    config: UnlearnConfig = load_config(args.config, overrides)
    if config.proxy_root is None:
        config = load_config(
            args.config, {**overrides, "proxy_root": str(_proxy_root(settings, args.proxy_root))}
        )
    device = select_device(config.device)
    print(f"method={config.method} device={device} config={json.dumps(config_to_dict(config))}")

    forget = select_subset(
        load_forget_images(forget_path), config.forget_subset or 10**9, config.seed
    )
    original = load_model(model_path, device)
    started = time.monotonic()
    unlearned, history = unlearn_with_history(original, forget, config)
    train_seconds = time.monotonic() - started
    print(f"unlearning finished in {train_seconds:.1f} s on {device}; history: {history}")

    stamp = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
    output = args.out or (task_dir / OUTPUT_SUBDIR / f"{config.method}_{stamp}.pt")
    report_file = save_submission(unlearned, output)
    print(
        f"saved {output} ({report_file.size_bytes / BYTES_PER_MEGABYTE:.1f} MB, "
        f"{report_file.parameter_count:,} parameters), validated"
    )

    proxy: ProxyReport | None = None
    if not args.no_eval:
        retain = load_proxy_images(Path(str(config.proxy_root)), config.retain_count, config.seed)
        proxy = _run_proxy_eval(
            original=original, candidate=unlearned, forget=forget, retain=retain,
            variants=tuple(ALTERATIONS), seed=config.seed, batch_size=args.eval_batch_size,
            task_dir=task_dir,
        )  # fmt: skip
        print(format_report(proxy, f"proxy evaluation of {config.method}"))
    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "config": config_to_dict(config),
                "history": history,
                "train_seconds": train_seconds,
                "device": str(device),
                "proxy": proxy.to_dict() if proxy else None,
                "file": asdict(report_file) | {"path": str(report_file.path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def cmd_validate(args: argparse.Namespace, settings: Settings) -> int:
    """Validate a submission file."""
    del settings
    report = validate_submission(args.file)
    print(
        f"VALID: {report.path} ({report.size_bytes / BYTES_PER_MEGABYTE:.1f} MB, "
        f"{report.tensor_count} tensors, {report.parameter_count:,} parameters)"
    )
    return 0


def cmd_submit(args: argparse.Namespace, settings: Settings) -> int:
    """Validate then, only with --yes, send the file to the server."""
    result = submit_validated(args.file, settings, confirm=args.yes)
    if result is None:
        print(f"[dry run] {args.file} is valid; add --yes to send it to the server.")
        return 0
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with one sub-command per step."""
    parser = argparse.ArgumentParser(prog="python -m hackathon.unlearning", description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=None, help="override the task data dir")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: automatic)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download model.pt and forget_images.npz").set_defaults(
        func=cmd_fetch
    )
    sub.add_parser("info", help="show files, environment, model").set_defaults(func=cmd_info)

    proxy = sub.add_parser("proxy-eval", help="local proxy metrics of a candidate vs the original")
    proxy.add_argument("--candidate", type=Path, default=None, help="state dict (.pt) to evaluate")
    proxy.add_argument("--forget-limit", type=int, default=None, help="use the first N images")
    proxy.add_argument("--retain-count", type=int, default=64)
    proxy.add_argument("--proxy-root", default=None)
    proxy.add_argument(
        "--variants", nargs="+", default=list(ALTERATIONS), choices=list(ALTERATIONS)
    )
    proxy.add_argument("--seed", type=int, default=DEFAULT_SEED)
    proxy.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    proxy.add_argument("--json-out", type=Path, default=None)
    proxy.set_defaults(func=cmd_proxy_eval)

    run = sub.add_parser("run", help="unlearn, save and proxy-evaluate")
    run.add_argument("--method", choices=METHOD_NAMES, default=None)
    run.add_argument("--config", default=None, help=f"preset {sorted(PRESETS)} or JSON file")
    run.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config field")
    run.add_argument("--out", type=Path, default=None, help="submission file to write")
    run.add_argument("--proxy-root", default=None)
    run.add_argument("--no-eval", action="store_true", help="skip the proxy evaluation")
    run.add_argument("--eval-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate", help="validate a submission file")
    validate.add_argument("file", type=Path)
    validate.set_defaults(func=cmd_validate)

    submit = sub.add_parser("submit", help="validate then send (only with --yes)")
    submit.add_argument("file", type=Path)
    submit.add_argument("--yes", action="store_true", help="really send the file")
    submit.set_defaults(func=cmd_submit)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; expected failures print `error: ...` and exit 1."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args, load_settings()))
    except HackathonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

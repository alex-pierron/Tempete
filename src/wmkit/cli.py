"""Command line interface: `python -m wmkit <command>` (or the `wmkit` script).

Commands: list, embed, extract, attack, eval, batch-embed, batch-extract. Exit status: 0 success, 1 runtime failure
(message on stderr), 2 usage error (argparse).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from wmkit.attacks import SUITES, get_attack, get_suite, list_attacks, reseed_attack
from wmkit.batch import (
    ImageShape,
    embed_folder,
    extract_folder,
    read_expected_messages,
    score_extractions,
)
from wmkit.config import get_device
from wmkit.eval import evaluate, summarize
from wmkit.io import list_images, load_image, save_image
from wmkit.messages import bits_to_hex, hex_to_bits, random_messages
from wmkit.methods import get_watermarker, list_methods
from wmkit.metrics import DETECTION_NEG_LOG10_P, bit_accuracy, neg_log10_pvalue
from wmkit.types import ATTACK_GROUPS, Watermarker

EXIT_OK: int = 0
EXIT_FAILURE: int = 1
DEFAULT_SEED: int = 0
SUMMARY_FLOAT_FORMAT: str = ".3f"
LIST_ALL_SUITE: str = "all"
RANDOM_MESSAGE_KEYWORD: str = "random"
"""`--message random` draws a fresh payload from `--seed`, like omitting `--message`."""
RUNTIME_ERRORS: tuple[type[Exception], ...] = (
    KeyError, ValueError, OSError, ImportError, RuntimeError,
)
"""Errors reported as a one-line message instead of a traceback."""
AUTO_BATCH_LABEL: str = "auto"
BATCH_EMBED_CSV_HEADER: str = "image,message_hex"
BATCH_EXTRACT_CSV_HEADER: str = "image,bits"
BATCH_EXTRACT_SCORED_CSV_HEADER: str = "image,bits,bit_acc,neg_log10_p,detected"
SCORE_FLOAT_FORMAT: str = ".4f"


def _format_error(error: Exception) -> str:
    """Readable text of an exception (KeyError would otherwise print its repr)."""
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the `wmkit` command."""
    parser = argparse.ArgumentParser(
        prog="wmkit", description="Image watermarking toolkit: embed, extract, attack, evaluate."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    commands.add_parser("list", help="list methods, attacks by group and suites")

    embed = commands.add_parser("embed", help="watermark an image and print the message (hex)")
    embed.add_argument("--method", required=True)
    embed.add_argument("--in", dest="input_path", required=True, type=Path)
    embed.add_argument("--out", dest="output_path", required=True, type=Path)
    embed.add_argument(
        "--message",
        help="hex payload, or 'random' (the default when omitted) to draw one from --seed",
    )
    embed.add_argument("--seed", type=int, default=DEFAULT_SEED)

    extract = commands.add_parser("extract", help="read the message of an image")
    extract.add_argument("--method", required=True)
    extract.add_argument("--in", dest="input_path", required=True, type=Path)
    extract.add_argument("--message", help="expected hex payload: also prints the detection stats")

    attack = commands.add_parser("attack", help="apply a named attack to an image")
    attack.add_argument("--attack", required=True)
    attack.add_argument("--in", dest="input_path", required=True, type=Path)
    attack.add_argument("--out", dest="output_path", required=True, type=Path)
    attack.add_argument("--seed", type=int, default=DEFAULT_SEED)

    evaluation = commands.add_parser("eval", help="evaluate methods against an attack suite")
    evaluation.add_argument("--method", required=True, help="comma separated method names")
    evaluation.add_argument("--images", required=True, type=Path, help="folder of images")
    evaluation.add_argument("--suite", choices=list(SUITES), default="quick")
    evaluation.add_argument("--max-images", type=int, default=None)
    evaluation.add_argument("--seed", type=int, default=DEFAULT_SEED)
    evaluation.add_argument("--lpips", action="store_true", help="also compute LPIPS")
    evaluation.add_argument("--out", dest="output_path", required=True, type=Path,
                            help="CSV file receiving the full long-format results")

    batch_embed = commands.add_parser(
        "batch-embed", help="watermark every image of a folder, by batches"
    )
    batch_embed.add_argument("--method", required=True)
    batch_embed.add_argument("--images", required=True, type=Path, help="folder of images")
    batch_embed.add_argument("--out", dest="output_dir", required=True, type=Path,
                             help="folder receiving the watermarked images (PNG)")
    _add_batch_size_argument(batch_embed)
    batch_embed.add_argument("--seed", type=int, default=DEFAULT_SEED)
    batch_embed.add_argument(
        "--messages-out", dest="messages_path", type=Path, default=None,
        help="also write one hex message per line (image order): feed it to --expected",
    )

    batch_extract = commands.add_parser(
        "batch-extract", help="read the message of every image of a folder, by batches"
    )
    batch_extract.add_argument("--method", required=True)
    batch_extract.add_argument("--images", required=True, type=Path, help="folder of images")
    _add_batch_size_argument(batch_extract)
    batch_extract.add_argument(
        "--expected", dest="expected_path", type=Path, default=None,
        help="file with one expected hex message per line, in image order: "
             "also prints the detection stats",
    )
    return parser


def _add_batch_size_argument(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--batch-size", type=int, default=None,
        help="images per batch; 0 or omitted = automatic from the free memory",
    )


def _command_list() -> int:
    print("Methods")
    for name, description in list_methods().items():
        print(f"  {name:<18} {description}")
    print("\nAttacks")
    for group in ATTACK_GROUPS:
        print(f"  {group}: {', '.join(list_attacks(group))}")
    print("\nSuites")
    for name, attack_names in SUITES.items():
        listing = f"{len(attack_names)} attacks" if name == LIST_ALL_SUITE else (
            f"{len(attack_names)}: {', '.join(attack_names)}"
        )
        print(f"  {name}: {listing}")
    return EXIT_OK


def _load_watermarker(name: str) -> Watermarker:
    return get_watermarker(name, device=str(get_device()))


def _command_embed(arguments: argparse.Namespace) -> int:
    watermarker = _load_watermarker(arguments.method)
    image = load_image(arguments.input_path)
    if arguments.message is None or arguments.message.strip().lower() == RANDOM_MESSAGE_KEYWORD:
        generator = torch.Generator().manual_seed(arguments.seed)
        message = random_messages(1, watermarker.n_bits, generator)
    else:
        message = hex_to_bits(arguments.message, watermarker.n_bits)[None]
    device = get_device()
    with torch.no_grad():
        marked = watermarker.embed(image[None].to(device), message.to(device))
    save_image(marked[0].cpu(), arguments.output_path)
    print(bits_to_hex(message[0]))
    return EXIT_OK


def _command_extract(arguments: argparse.Namespace) -> int:
    watermarker = _load_watermarker(arguments.method)
    image = load_image(arguments.input_path)
    with torch.no_grad():
        soft_bits = watermarker.extract(image[None].to(get_device())).cpu()
    print(f"bits: {bits_to_hex(soft_bits[0])}")
    if arguments.message is not None:
        expected = hex_to_bits(arguments.message, watermarker.n_bits)[None]
        accuracy = bit_accuracy(soft_bits, expected)
        n_correct = torch.round(accuracy * watermarker.n_bits)
        score = float(neg_log10_pvalue(watermarker.n_bits, n_correct)[0])
        print(f"bit_acc: {float(accuracy[0]):.4f}")
        print(f"neg_log10_p: {score:.4f}")
        print(f"detected: {score > DETECTION_NEG_LOG10_P}")
    return EXIT_OK


def _command_attack(arguments: argparse.Namespace) -> int:
    spec = get_attack(arguments.attack)
    reseed_attack(spec, arguments.seed)
    image = load_image(arguments.input_path)
    device = get_device()
    with torch.no_grad():
        attacked = spec.fn(image[None].to(device))
    save_image(attacked[0].cpu(), arguments.output_path)
    height, width = image.shape[-2:]
    new_height, new_width = attacked.shape[-2:]
    print(f"{spec.name}: {width}x{height} -> {new_width}x{new_height}, "
          f"saved to {arguments.output_path}")
    return EXIT_OK


def _command_eval(arguments: argparse.Namespace) -> int:
    method_names = [name.strip() for name in arguments.method.split(",") if name.strip()]
    if not method_names:
        raise ValueError("--method needs at least one method name")
    paths = list_images(arguments.images, arguments.max_images)
    if not paths:
        raise ValueError(f"no image (png, jpg, jpeg, webp) found in {arguments.images}")
    images = [(path.name, load_image(path)) for path in paths]
    device = str(get_device())
    frames: list[pd.DataFrame] = []
    for name in method_names:
        watermarker = get_watermarker(name, device=device)
        frames.append(evaluate(
            watermarker, images, get_suite(arguments.suite), seed=arguments.seed, device=device,
            with_lpips=arguments.lpips,
        ))
    results = pd.concat(frames, ignore_index=True)
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(arguments.output_path, index=False)
    print(f"{len(results)} rows, {len(images)} image(s), suite {arguments.suite!r} "
          f"-> {arguments.output_path}\n")
    print(summarize(results).to_markdown(index=False, floatfmt=SUMMARY_FLOAT_FORMAT))
    return EXIT_OK


def _describe_batches(requested: int | None, used: dict[ImageShape, int]) -> str:
    """One line on the batch size: requested value and what each image shape ran with."""
    mode = AUTO_BATCH_LABEL if not requested else str(requested)
    per_shape = ", ".join(f"{width}x{height}:{size}" for (height, width), size in used.items())
    return f"batch size {mode} ({per_shape})"


def _command_batch_embed(arguments: argparse.Namespace) -> int:
    watermarker = _load_watermarker(arguments.method)
    result = embed_folder(
        watermarker, arguments.images, arguments.output_dir,
        batch_size=arguments.batch_size, seed=arguments.seed, device=get_device(),
    )
    print(f"{len(result.records)} image(s) watermarked with {watermarker.name} "
          f"-> {arguments.output_dir}; {_describe_batches(arguments.batch_size, result.batch_sizes)}")
    print(BATCH_EMBED_CSV_HEADER)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    for record in result.records:
        writer.writerow([record.image_name, record.message_hex])
    if arguments.messages_path is not None:
        arguments.messages_path.parent.mkdir(parents=True, exist_ok=True)
        # Write 'output_name,hex' so --expected pairs by file name, not by position: the sort
        # order of the output folder can differ from the inputs once extensions become .png.
        arguments.messages_path.write_text(
            "".join(f"{record.output_path.name},{record.message_hex}\n"
                    for record in result.records),
            encoding="utf-8",
        )
    return EXIT_OK


def _command_batch_extract(arguments: argparse.Namespace) -> int:
    watermarker = _load_watermarker(arguments.method)
    result = extract_folder(
        watermarker, arguments.images, batch_size=arguments.batch_size, device=get_device()
    )
    names = result.image_names
    hexes = [bits_to_hex(row) for row in result.soft_bits]
    print(f"{len(names)} image(s) read with {watermarker.name}; "
          f"{_describe_batches(arguments.batch_size, result.batch_sizes)}")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    if arguments.expected_path is None:
        print(BATCH_EXTRACT_CSV_HEADER)
        for name, bits_hex in zip(names, hexes, strict=True):
            writer.writerow([name, bits_hex])
        return EXIT_OK
    expected = read_expected_messages(arguments.expected_path, watermarker.n_bits, names)
    scores = score_extractions(result.soft_bits, expected)
    print(BATCH_EXTRACT_SCORED_CSV_HEADER)
    for name, bits_hex, score in zip(names, hexes, scores, strict=True):
        writer.writerow([name, bits_hex, f"{score.bit_accuracy:{SCORE_FLOAT_FORMAT}}",
                         f"{score.neg_log10_p:{SCORE_FLOAT_FORMAT}}", score.detected])
    mean_accuracy = sum(score.bit_accuracy for score in scores) / len(scores)
    min_accuracy = min(score.bit_accuracy for score in scores)
    mean_score = sum(score.neg_log10_p for score in scores) / len(scores)
    n_detected = sum(score.detected for score in scores)
    print(f"summary: images={len(scores)} mean_bit_acc={mean_accuracy:{SCORE_FLOAT_FORMAT}} "
          f"min_bit_acc={min_accuracy:{SCORE_FLOAT_FORMAT}} "
          f"mean_neg_log10_p={mean_score:{SCORE_FLOAT_FORMAT}} "
          f"detected={n_detected}/{len(scores)}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit status."""
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "list":
            return _command_list()
        if arguments.command == "embed":
            return _command_embed(arguments)
        if arguments.command == "extract":
            return _command_extract(arguments)
        if arguments.command == "attack":
            return _command_attack(arguments)
        if arguments.command == "batch-embed":
            return _command_batch_embed(arguments)
        if arguments.command == "batch-extract":
            return _command_batch_extract(arguments)
        return _command_eval(arguments)
    except RUNTIME_ERRORS as error:
        print(f"error: {_format_error(error)}", file=sys.stderr)
        return EXIT_FAILURE

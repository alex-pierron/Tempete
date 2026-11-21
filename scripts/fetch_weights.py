"""Pre-download the Meta Seal checkpoints and print their sizes (idempotent).

Each checkpoint is downloaded (resumable), reduced to its inference-only `model` entry
(`<weights_dir>/seal/<card>.slim.pth`) and the full upstream file is deleted unless
`--keep-original` is given. The weights directory follows `$WMKIT_WEIGHTS_DIR`.

By default the three small cards are fetched. Chunky Seal publishes a 13.4 GB checkpoint: it
is fetched only when named explicitly (`--cards chunkyseal`), and the free-disk guard of
`wmkit.seal.ensure_checkpoint` still applies.

Usage:
    WMKIT_WEIGHTS_DIR=/path/to/weights python scripts/fetch_weights.py
    python scripts/fetch_weights.py --cards pixelseal
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from collections.abc import Sequence

from wmkit.config import weights_dir
from wmkit.seal import (
    CHUNKYSEAL_CARD,
    SEAL_CARDS,
    SealError,
    ensure_checkpoint,
    get_card_info,
    resolve_card,
)

DEFAULT_CARDS: tuple[str, ...] = tuple(card for card in SEAL_CARDS if card != CHUNKYSEAL_CARD)
HTTP_TIMEOUT_SECONDS = 30.0
HTTP_USER_AGENT = "wmkit-fetch/0.1"
HUGGINGFACE_HOST = "https://huggingface.co/"
BYTES_PER_MB = 1e6
EXIT_OK = 0
EXIT_FAILURE = 1


def _remote_size_bytes(url: str) -> int | None:
    """Published size of the original checkpoint (HEAD, redirects followed); None if unknown."""
    if url.startswith(HUGGINGFACE_HOST):
        url = url.replace("/facebook/video_seal/", "/facebook/video_seal/resolve/main/")
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return int(response.headers["Content-Length"])
    except (OSError, KeyError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Meta Seal checkpoints (idempotent).")
    parser.add_argument(
        "--cards",
        nargs="+",
        default=list(DEFAULT_CARDS),
        help=f"cards to fetch (default: {' '.join(DEFAULT_CARDS)}; chunkyseal is 13.4 GB)",
    )
    parser.add_argument(
        "--keep-original", action="store_true", help="keep the full upstream checkpoint files"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints one line per card and returns a process exit code."""
    args = build_parser().parse_args(argv)
    status = EXIT_OK
    for requested in args.cards:
        try:
            card = resolve_card(requested)
            info = get_card_info(card)
            published = _remote_size_bytes(info.checkpoint_url)
            path = ensure_checkpoint(card, keep_original=args.keep_original)
        except SealError as error:
            print(f"error: {requested}: {error}", file=sys.stderr)
            status = EXIT_FAILURE
            continue
        published_text = "n/a" if published is None else f"{published / BYTES_PER_MB:.0f} MB"
        print(
            f"{card}: {info.n_bits} bits, published {published_text}, "
            f"local inference weights {path.stat().st_size / BYTES_PER_MB:.0f} MB ({path})"
        )
    free_gb = shutil.disk_usage(weights_dir()).free / 1024**3
    print(f"free disk on the weights volume: {free_gb:.1f} GiB")
    return status


if __name__ == "__main__":
    sys.exit(main())

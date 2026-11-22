"""Registry of watermarking methods, resolved lazily by name.

Each entry points to a factory `factory(n_bits: int | None = None, device: str = "cpu",
**kwargs) -> Watermarker` living in the module that owns the method. Imports are lazy so
that a missing optional dependency (e.g. videoseal) only breaks the method that needs it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from wmkit.types import Watermarker


@dataclass(frozen=True)
class MethodEntry:
    """Location of a watermarker factory and the default kwargs passed to it."""

    module: str
    factory: str
    defaults: dict[str, Any] = field(default_factory=dict)
    description: str = ""


METHODS: dict[str, MethodEntry] = {
    # Meta Seal family, pretrained (owner: wmkit.seal)
    "pixelseal": MethodEntry("wmkit.seal", "load_seal", {"card": "pixelseal"},
                             "Meta Pixel Seal, 256 bits, SOTA 2025-12"),
    "videoseal": MethodEntry("wmkit.seal", "load_seal", {"card": "videoseal"},
                             "Meta Video Seal v1.0, 256 bits"),
    "videoseal0": MethodEntry("wmkit.seal", "load_seal", {"card": "videoseal_0.0"},
                              "Meta Video Seal v0.0, 96 bits (legacy)"),
    "chunkyseal": MethodEntry("wmkit.seal", "load_seal", {"card": "chunkyseal"},
                              "Meta Chunky Seal, 1024 bits"),
    # Classical, training-free baselines (owner: wmkit.classical)
    "dct_qim": MethodEntry("wmkit.classical.dct_qim", "build", {},
                           "Blockwise DCT mid-frequency QIM on luminance"),
    "spread_spectrum": MethodEntry("wmkit.classical.spread_spectrum", "build", {},
                                   "Additive pseudo-random spread spectrum, correlation decoding"),
    "dwt_dct_svd": MethodEntry("wmkit.classical.dwt_dct_svd", "build", {},
                               "DWT-DCT-SVD (invisible-watermark style) on luminance"),
    # Small trainable model reproducing the Pixel Seal recipe (owner: wmkit.neural)
    "minipixelseal": MethodEntry("wmkit.neural.model", "build", {},
                                 "Mini Pixel Seal recipe, trained locally (REPORTE - non implemente)"),
}


def list_methods() -> dict[str, str]:
    """Return {method name: description}."""
    return {name: entry.description for name, entry in METHODS.items()}


def get_watermarker(name: str, n_bits: int | None = None, device: str = "cpu",
                    **kwargs: Any) -> Watermarker:
    """Instantiate the watermarker registered under `name`.

    Args:
        name: A key of METHODS.
        n_bits: Payload size; None keeps the method's native size.
        device: Torch device string.
        **kwargs: Overrides of the entry defaults.

    Raises:
        KeyError: Unknown method name.
        ImportError: The owning module or one of its dependencies is unavailable.
    """
    if name not in METHODS:
        known = ", ".join(sorted(METHODS))
        raise KeyError(f"unknown watermarking method {name!r}; known: {known}")
    entry = METHODS[name]
    module = importlib.import_module(entry.module)
    factory = getattr(module, entry.factory)
    params = {**entry.defaults, **kwargs}
    watermarker: Watermarker = factory(n_bits=n_bits, device=device, **params)
    # The registry key is the canonical identity used by the CLI (--method) and by the
    # eval DataFrame `method` column; override any native name (e.g. a Seal card id such
    # as "videoseal_1.0") so CLI and CSV always agree.
    watermarker.name = name
    return watermarker

"""Turn X-LOC-C SAM test maps into an assemble.py overlay (applied AFTER c_rect and the PS occlusion overlay).

Each input npz holds <method> (0/1 masks, 340 x 128 x 128), names_<method> and apply_<method>; only the apply images
go into the overlay, with confidence 1.0 (above the 0.85 overlay gate).

Usage: python sam_overlay.py <out.pt> <method>=<sam_test.npz> [<method>=<sam_test.npz> ...]
"""

from __future__ import annotations

import sys

import numpy as np
import torch

OVERLAY_CONFIDENCE = 1.0


def main() -> None:
    """Write the overlay and print the count per method."""
    out = sys.argv[1]
    names: list[str] = []
    masks: list[np.ndarray] = []
    for spec in sys.argv[2:]:
        method, _, path = spec.partition("=")
        z = np.load(path)
        apply = z[f"apply_{method}"].astype(bool)
        chosen = np.flatnonzero(apply)
        names += [str(z[f"names_{method}"][i]) for i in chosen]
        masks += [(z[method][i] > 0).astype(np.uint8) for i in chosen]
        print(f"{method}: {len(chosen)} SAM masks from {path}")
    if len(set(names)) != len(names):
        raise SystemExit("duplicate image names across inputs")
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)),
                "confidence": torch.full((len(names),), OVERLAY_CONFIDENCE)}, out)
    print(f"overlay {out}: {len(names)} masks")


if __name__ == "__main__":
    main()

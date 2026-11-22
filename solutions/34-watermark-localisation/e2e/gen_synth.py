"""X-LOC-B (JURECA CPU): synthetic composites WITH their messages, for end-to-end training (b2_synth recipe).

The team's synthetic shards keep the images but not the random messages, and the e2e model needs the message
for its agreement maps. This regenerates samples with the same recipe as the deployed heads (real mask shapes,
FULL = watermark all then restore a real hole, RECT = watermark the bbox crop and paste it inside a real
region), on CPU, and keeps the message. Each sample carries the b2 fold of the real image its shape came from,
so a training fold can drop the shapes of its own held-out images.

Usage: python gen_synth.py <method> <max samples> <seed> <seconds> <out.npz>
Env: PYTHONPATH=<r-wml-b2>/src, HACKATHON_DATA_DIR, WMLOC_B2_CACHE (team_b2/feat, for clean_images.npz).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from hackathon.wmloc.research.b2_synth import SynthError, sample_pair
from hackathon.wmloc.research.b2_synth_features import TRUSTMARK_MAX_ASPECT, clean_crop, clean_pool, real_library
from hackathon.wmloc.research.kit import N_BITS_BY_METHOD, load_wrapper

THREADS = 2


def main() -> None:
    """Generate until the sample count or the time budget is reached, then save atomically."""
    method, count, seed, seconds, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), Path(sys.argv[5])
    start = time.time()
    torch.set_num_threads(THREADS)
    wrapper = load_wrapper(method, "cpu")
    library = real_library()
    sources = clean_pool()
    n_bits = N_BITS_BY_METHOD[method]
    max_aspect = TRUSTMARK_MAX_ASPECT if method == "trustmark" else None
    rng = np.random.default_rng(seed)
    images, masks, fulls, folds, messages = [], [], [], [], []
    print(f"{method} seed {seed}: ready [{time.time() - start:.0f} s]", flush=True)
    while len(images) < count and time.time() - start < seconds:
        bits = rng.integers(0, 2, n_bits).astype(np.uint8)
        clean = clean_crop(sources[int(rng.integers(len(sources)))], rng)
        try:
            image, label, is_full, fold = sample_pair(clean, bits, library, wrapper.embed, rng, max_aspect=max_aspect)
        except SynthError:
            continue
        images.append(image)
        masks.append(label)
        fulls.append(is_full)
        folds.append(fold)
        messages.append(bits)
    if not images:
        raise SystemExit(f"{method} seed {seed}: no sample in {seconds} s")
    temporary = out.with_name(out.name + ".tmp.npz")
    np.savez(temporary, images=np.stack(images), masks=np.stack(masks), full=np.array(fulls),
             fold=np.array(folds, dtype=np.int64), messages=np.stack(messages))
    temporary.replace(out)
    print(f"{method} seed {seed}: {len(images)} samples -> {out} [{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
    main()

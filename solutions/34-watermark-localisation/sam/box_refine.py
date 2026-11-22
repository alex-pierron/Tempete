"""X-LOC-C (JURECA): refine decoder-found rectangles (c_rect) to the exact embedding crop, PS and TM.

In the rect regime the patch itself was embedded, so the soft margin (mean of logit x known-message sign) of a
crop peaks at the exact box. c_rect boxes match the truth at IoU ~0.8 on train: this coordinate ascent moves
each edge (and the whole box) by 8, 4, 2, 1 px, keeping a move when the margin increases, all candidate moves of
an iteration decoded in one batch. Aspect limits of the method are enforced.

Usage: python box_refine.py <boxes.json> <out.json>
  boxes.json: [{"split", "name", "method", "message", "box": [top, left, bottom, right]}, ...]
Env: WMLOC_ROOT (release/ and models/), WMLOC_KIT (dir holding the kit package `wmloc`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(os.environ["WMLOC_ROOT"])
sys.path.insert(0, os.environ["WMLOC_KIT"])
from wmloc.watermarks import load_watermark  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 512
STEPS_PX = (8, 4, 2, 1)
MAX_ITER_PER_STEP = 10
MIN_SIDE = 8
MAX_ASPECT = {"pixelseal": 16.0, "trustmark": 2.0}
DECODE_RES = 256


def moves(box: tuple[int, int, int, int], step: int) -> list[tuple[int, int, int, int]]:
    """Edge moves, shifts and uniform inflations of a box."""
    t, l, b, r = box
    out = []
    for dt, dl, db, dr in (
        (-step, 0, 0, 0), (step, 0, 0, 0), (0, -step, 0, 0), (0, step, 0, 0),
        (0, 0, -step, 0), (0, 0, step, 0), (0, 0, 0, -step), (0, 0, 0, step),
        (-step, 0, -step, 0), (step, 0, step, 0), (0, -step, 0, -step), (0, step, 0, step),
        (-step, -step, step, step), (step, step, -step, -step),
    ):
        out.append((t + dt, l + dl, b + db, r + dr))
    return out


def valid(box: tuple[int, int, int, int], method: str) -> bool:
    """Inside the image, sides >= MIN_SIDE, aspect within the method's limit."""
    t, l, b, r = box
    if t < 0 or l < 0 or b > SIZE or r > SIZE:
        return False
    h, w = b - t, r - l
    if h < MIN_SIDE or w < MIN_SIDE:
        return False
    return max(h, w) / min(h, w) <= MAX_ASPECT[method] * 1.05


@torch.no_grad()
def margins(wm: object, x: torch.Tensor, boxes: list[tuple[int, int, int, int]], sign: torch.Tensor) -> np.ndarray:
    """Soft margin and agreement of each crop (batched decode)."""
    crops = [F.interpolate(x[None, :, t:b, l:r], size=(DECODE_RES, DECODE_RES), mode="bilinear",
                           align_corners=False, antialias=True) for t, l, b, r in boxes]
    lg = wm.logits(torch.cat(crops)).float()  # type: ignore[attr-defined]
    return torch.stack([(lg * sign).mean(1), ((lg > 0) == (sign > 0)).float().mean(1)], 1).cpu().numpy()


def main() -> None:
    """Refine every listed box."""
    items = json.loads(Path(sys.argv[1]).read_text())
    out: list[dict[str, object]] = []
    start = time.time()
    for method in ("pixelseal", "trustmark"):
        todo = [it for it in items if it["method"] == method]
        if not todo:
            continue
        wm = load_watermark(method, str(ROOT / "models"), device=DEVICE)
        for it in todo:
            img = np.asarray(Image.open(ROOT / "release" / it["split"] / "images" / f"{it['name']}.png").convert("RGB"))
            x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(DEVICE)
            sign = torch.tensor([2 * int(c) - 1 for c in it["message"]], dtype=torch.float32, device=DEVICE)
            box = tuple(int(v) for v in it["box"])
            cur = margins(wm, x, [box], sign)[0]
            start_margin = float(cur[0])
            for step in STEPS_PX:
                for _ in range(MAX_ITER_PER_STEP):
                    cand = [c for c in moves(box, step) if valid(c, method)]
                    if not cand:
                        break
                    sc = margins(wm, x, cand, sign)
                    j = int(np.argmax(sc[:, 0]))
                    if sc[j, 0] <= cur[0]:
                        break
                    box, cur = cand[j], sc[j]
            out.append({**it, "refined": list(box), "margin0": start_margin, "margin": float(cur[0]),
                        "agreement": float(cur[1])})
        print(f"{method}: {len(todo)} boxes [{time.time() - start:.0f} s]", flush=True)
        del wm
    Path(sys.argv[2]).write_text(json.dumps(out))
    print(f"done {len(out)} [{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
    main()

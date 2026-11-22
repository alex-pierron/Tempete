"""X-LOC-C: build a test candidate = ll_v14 with the SAM object chosen on gated rect images.

Same rule as sam_select.py (parameters given on the command line, chosen there out of fold):
gate = predicted rect regime (share of head cells >= 0.5 below 0.5) and fill-of-bbox of the head mask
(thresholded at the deployed rect threshold) < FILL_T, and selection score >= SCORE_T.

Usage: python sam_build.py <sam_test.npz> '<json {method: [fill_t, score_t, target, parts]}>' <base.pt> <out.pt>
       [export_prefix]   (a method absent from the JSON is left as in the base)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sam_select import DECISION, RECT_THRESHOLD, ROOT, choose, fill_of_bbox, load_candidates  # noqa: E402

TEST_MAPS = {
    "pixelseal": (ROOT / "loclab" / "x23b" / "ens_ps_3.npz", "pixelseal", "names_pixelseal"),
    "trustmark": (ROOT / "loclab" / "x23b" / "ens_tm_2.npz", "trustmark", "names_trustmark"),
    "stegastamp": (ROOT / "team_final" / "ss_split7_u1x5_zoom2.npz", "test_finetuned", "names_test"),
}
# ll_v14 overlays (last writers in its assembly): an image they cover keeps the ll_v14 mask, exactly as
# if they were re-applied after the SAM replacement.
# Only c_rect (decoder-fitted rectangles) keeps priority; the PS occlusion overlay (full-regime holes, 14 of its
# 180 images predicted rect) is overwritten by SAM on the retained images (directeur, 08:00).
OVERLAYS = (ROOT / "overlays" / "c_rect_overlay_tmshrink2.pt",)
OVERLAY_CONFIDENCE = 0.85


def overlay_names() -> set[str]:
    """Test images an ll_v14 overlay overwrites (confidence >= 0.85)."""
    out: set[str] = set()
    for path in OVERLAYS:
        o = torch.load(path, weights_only=True)
        out.update(str(n) for n, c in zip(o["image_names"], o["confidence"], strict=True) if float(c) >= OVERLAY_CONFIDENCE)
    return out


def main() -> None:
    """Write the candidate .pt and print how many images changed, by method."""
    cands = load_candidates(Path(sys.argv[1]))
    configs = json.loads(sys.argv[2])
    base = torch.load(sys.argv[3], weights_only=True)
    names = list(base["image_names"])
    masks = base["masks"].clone().to(torch.uint8)
    pos = {n: i for i, n in enumerate(names)}
    changed: dict[str, int] = {}
    protected = overlay_names()
    skipped_overlay = 0
    cells_changed = 0
    for method, (path, key, nkey) in TEST_MAPS.items():
        if method not in configs:
            continue
        fill_t, score_t, kind, parts = float(configs[method][0]), float(configs[method][1]), str(configs[method][2]), int(configs[method][3])
        z = np.load(path)
        all_names = [str(n) for n in z[nkey]]
        export = np.zeros((len(all_names), 128, 128), np.uint8)
        apply = np.zeros(len(all_names), bool)
        for j, (n, p) in enumerate(zip(z[nkey], z[key], strict=True)):
            n = str(n)
            if n not in cands:
                continue
            p = p.astype(np.float32)
            pred_rect = (p >= DECISION).mean() < DECISION
            fill = fill_of_bbox(p >= RECT_THRESHOLD[method])
            m, s = choose(p, cands[n], 0.0, parts, 0.0, kind)
            export[j] = m
            if pred_rect and fill < fill_t and s >= score_t and m.any():
                if n in protected:
                    skipped_overlay += 1
                    continue
                apply[j] = True
                new = torch.from_numpy(m.astype(np.uint8))
                cells_changed += int((new != masks[pos[n]]).sum())
                masks[pos[n]] = new
                changed[method] = changed.get(method, 0) + 1
        if len(sys.argv) > 5:
            prefix = Path(sys.argv[5])
            tmp = prefix.parent / f"{prefix.name}_{method}_test.tmp.npz"
            np.savez_compressed(tmp, **{method: export, f"names_{method}": np.array(all_names),
                                        f"apply_{method}": apply})
            tmp.replace(prefix.parent / f"{prefix.name}_{method}_test.npz")
    torch.save({"image_names": names, "masks": masks}, sys.argv[4])
    print(f"changed images {changed} (total {sum(changed.values())}), cells changed {cells_changed}, "
          f"kept by an ll_v14 overlay {skipped_overlay} (overlay set {len(protected)})")


if __name__ == "__main__":
    main()

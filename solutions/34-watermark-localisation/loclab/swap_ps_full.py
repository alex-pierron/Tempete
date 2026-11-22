"""Replace the PS full masks of a base submission by a new fused map, only where the base still equals the old
occlusion overlay (so SAM / c_rect decisions made after it are kept). Usage: base.pt new_map.npz out.pt"""
import sys
import numpy as np
import torch

OVERLAY = __import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/overlays/rect_ps_occlusion_overlay.pt"
base_path, map_path, out_path = sys.argv[1:4]
base = torch.load(base_path, weights_only=False)
names = list(base["image_names"])
masks = base["masks"].clone()
pos = {n: i for i, n in enumerate(names)}
ov = torch.load(OVERLAY, weights_only=False)
ov_mask = {n: ov["masks"][i].bool() for i, n in enumerate(ov["image_names"])}
z = np.load(map_path)
tn = [str(v) for v in z["names_pixelseal"]]
full = z["full_pixelseal"]
new = z["pixelseal"] >= 0.5
replaced, kept_changed, cells = 0, 0, 0
for k, n in enumerate(tn):
    if not full[k]:
        continue
    i = pos[n]
    cur = masks[i].bool()
    if n in ov_mask and not torch.equal(cur, ov_mask[n]):
        kept_changed += 1  # SAM / c_rect touched it after the overlay: keep the base
        continue
    m = torch.from_numpy(new[k])
    cells += int((m != cur).sum())
    masks[i] = m.to(masks.dtype)
    replaced += 1
torch.save({"image_names": base["image_names"], "masks": masks}, out_path)
print(f"replaced {replaced} PS full masks ({cells} cells changed), kept {kept_changed} touched after the overlay")
mean_hole_old = np.mean([1 - base['masks'][pos[n]].float().mean().item() for k, n in enumerate(tn) if full[k]])
mean_hole_new = np.mean([1 - masks[pos[n]].float().mean().item() for k, n in enumerate(tn) if full[k]])
print(f"mean predicted hole share on these images: old {mean_hole_old:.4f} new {mean_hole_new:.4f}")

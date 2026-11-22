"""Apply the soft expected-mIoU decision (T = 1, thresholds 0.20-0.90) to the raw fused PS full test map."""
import sys
import numpy as np
T = np.round(np.arange(0.2, 0.91, 0.05), 2)
z = np.load(sys.argv[1])
raw, full, names = z["raw_pixelseal"], z["full_pixelseal"], z["names_pixelseal"]
out = z["pixelseal"].copy()
ts = []
for i in np.flatnonzero(full):
    q = raw[i]
    best, bt = -1.0, 0.5
    for t in T:
        m = q >= t
        sc = 0.0
        for pc, mc in ((q, m), (1 - q, ~m)):
            inter = pc[mc].sum(); union = mc.sum() + pc.sum() - inter
            sc += inter / union if union > 0 else 1.0
        if sc > best:
            best, bt = sc, t
    out[i] = np.where(q >= bt, 1.0, 0.0)
    ts.append(bt)
np.savez(sys.argv[2], pixelseal=out, names_pixelseal=names, full_pixelseal=full)
print(f"soft-IoU thresholds on {len(ts)} full test images: median {np.median(ts):.2f} q10 {np.quantile(ts, .1):.2f} q90 {np.quantile(ts, .9):.2f}")

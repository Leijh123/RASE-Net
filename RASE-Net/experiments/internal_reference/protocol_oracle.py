import math
import numpy as np
from scipy import ndimage

def oracle(pred, gt):
    """Independent audit of the ORIGINAL formulas, including undefined values."""
    p = np.asarray(pred, bool)
    g = np.asarray(gt, bool)
    npred = int(p.sum())
    ngt = int(g.sum())
    tp = int((p & g).sum())
    v = dict(dice=2 * tp / (npred + ngt + 1e-06), iou=tp / (npred + ngt - tp + 1e-06), precision=tp / (npred + 1e-06), recall=tp / (ngt + 1e-06))
    if not npred:
        return dict(v, hd95=float('nan'), assd=float('nan'), nsd=float('nan'))
    a = p ^ ndimage.binary_erosion(p, structure=np.ones((3, 3)), border_value=0)
    b = g ^ ndimage.binary_erosion(g, structure=np.ones((3, 3)), border_value=0)
    ab = ndimage.distance_transform_edt(~g)[a]
    ba = ndimage.distance_transform_edt(~p)[b]
    return dict(v, hd95=max(float(np.quantile(ab, 0.95)), float(np.quantile(ba, 0.95))), assd=(math.fsum(ab) / len(ab) + math.fsum(ba) / len(ba)) / 2, nsd=(int((ab <= 2).sum()) + int((ba <= 2).sum())) / (len(ab) + len(ba)))

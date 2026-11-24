import numpy as np
from scipy import ndimage

def seg_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return {
        'dice': dice,
        'iou': tp / (tp + fp + fn + 1e-6),
        'precision': tp / (tp + fp + 1e-6),
        'recall': tp / (tp + fn + 1e-6),
        'hd95': _hd95(pred, gt),
        'assd': _assd(pred, gt),
        'nsd': _nsd(pred, gt, tol=2.0),
    }

def _surface(m):
    if m.sum() == 0:
        return m
    er = ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=0)
    return m & ~er

def _surf_dists(a, b):
    if a.sum() == 0 or b.sum() == 0:
        return None, None
    sa, sb = _surface(a), _surface(b)
    db = ndimage.distance_transform_edt(~b)
    da = ndimage.distance_transform_edt(~a)
    return db[sa], da[sb]

def _hd95(a, b):
    d_ab, d_ba = _surf_dists(a, b)
    if d_ab is None:
        return float('nan')
    return max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))

def _assd(a, b):
    d_ab, d_ba = _surf_dists(a, b)
    if d_ab is None:
        return float('nan')
    return (d_ab.mean() + d_ba.mean()) / 2.0

def _nsd(a, b, tol=2.0):
    d_ab, d_ba = _surf_dists(a, b)
    if d_ab is None:
        return float('nan')
    return ((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba))

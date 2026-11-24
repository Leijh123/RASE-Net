import torch
from torch.nn import functional as F
def align(ex, pred):
    """Reference 256-grid centroid translation: no scale, no rotation."""
    H = ex.shape[-1]
    eb = ex > 0.5
    if eb.sum() < 4:
        return ex * 0
    pb = pred > 0.5
    if pb.sum() < 4:
        pb = pred >= pred.max() * 0.5
        if pb.sum() < 1:
            return ex
    (py, px) = [v.float().mean() for v in torch.nonzero(pb, as_tuple=True)]
    (ey, exx) = [v.float().mean() for v in torch.nonzero(eb, as_tuple=True)]
    theta = torch.tensor([[[1.0, 0.0, 2 * (exx - px) / H], [0.0, 1.0, 2 * (ey - py) / H]]], dtype=ex.dtype, device=ex.device)
    grid = F.affine_grid(theta, (1, 1, H, H), align_corners=False)
    return F.grid_sample(ex[None, None], grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0, 0]

def retrieval(q_emb, q_layout, candidates, embeddings, layouts, K, mode, random_state):
    if not candidates:
        return []
    if mode == 'random':
        return random_state.sample(candidates, min(K, len(candidates)))
    c = torch.as_tensor(candidates)
    similarity = (embeddings[c] @ q_emb).numpy()
    if mode != 'appearance':
        import numpy as np
        d = np.linalg.norm(layouts[candidates] - q_layout, axis=1)
        similarity += np.exp(-d / (d.mean() + 1e-06))
    return [candidates[i] for i in (-similarity).argsort(kind='stable')[:K]]

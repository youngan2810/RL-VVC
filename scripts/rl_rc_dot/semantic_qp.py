"""Causal source-detection ROI -> CTU-aligned five-action QP templates.

Experimental D3QN adapter; original PPO qpdev1-qbins and native QPA are unchanged.
Only source-frame detections may be used before the corresponding encoding step.
"""
from __future__ import annotations
import numpy as np

FRAME_W, FRAME_H, CTU = 480, 320, 32
CTU_ROWS, CTU_COLS = FRAME_H // CTU, FRAME_W // CTU
COARSE_ROWS, COARSE_COLS = 5, 7
# (roi_delta, background_delta); values are codec QP integers.
TEMPLATES = ((0, 0), (-1, 1), (-2, 2), (-1, 0), (-2, 1))


def source_ctu_roi(boxes):
    """Binary CTU occupancy from SOURCE-car boxes only; strict input validation."""
    a = np.asarray(boxes, dtype=np.float64)
    if not a.size:
        return np.zeros((CTU_ROWS, CTU_COLS), dtype=bool)
    if a.ndim != 2 or a.shape[1] != 5 or not np.isfinite(a).all():
        raise ValueError('Expected finite [N,5] source detections')
    if ((a[:, 0:2] < 0).any() or (a[:, 2] > FRAME_W).any() or
            (a[:, 3] > FRAME_H).any() or (a[:, 2:4] <= a[:, 0:2]).any() or
            (a[:, 4] < 0).any() or (a[:, 4] > 1).any()):
        raise ValueError('Invalid source detection box')
    occupied = np.zeros((CTU_ROWS, CTU_COLS), dtype=bool)
    for x1, y1, x2, y2, _ in a:
        col0, col1 = int(x1 // CTU), int(np.ceil(x2 / CTU))
        row0, row1 = int(y1 // CTU), int(np.ceil(y2 / CTU))
        occupied[row0:row1, col0:col1] = True
    return occupied


def coarse_roi(ctu_roi):
    """OR occupancy into the exact 5x7 spatial heads used by qpdev1-qbins."""
    arr = np.asarray(ctu_roi)
    if arr.shape != (CTU_ROWS, CTU_COLS) or arr.dtype != np.bool_:
        raise ValueError('Expected bool CTU ROI [10,15]')
    result = np.zeros((COARSE_ROWS, COARSE_COLS), dtype=bool)
    for y in range(CTU_ROWS):
        for x in range(CTU_COLS):
            if arr[y, x]:
                result[y * COARSE_ROWS // CTU_ROWS, x * COARSE_COLS // CTU_COLS] = True
    return result


def expanded_roi(coarse):
    """Preview the effective CTU-aligned region mask [20,30]."""
    m = np.asarray(coarse)
    if m.shape != (5, 7):
        raise ValueError('Expected [5,7] ROI')
    ys = np.arange(CTU_ROWS) * COARSE_ROWS // CTU_ROWS
    xs = np.arange(CTU_COLS) * COARSE_COLS // CTU_COLS
    return np.repeat(np.repeat(m[ys[:, None], xs[None, :]], 2, axis=0), 2, axis=1).astype(np.float32)


def template_to_qpbins(action, coarse):
    """Map one DQN action to 36 categorical heads without defeating QG centering.

    qpdev1-qbins uses: final_delta=global+spatial-round(mean(spatial)).
    Setting global=round(mean(spatial)) preserves the intended integer template.
    A no-car frame uses zero deltas (rather than penalizing all background).
    """
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < len(TEMPLATES):
        raise ValueError('Action must be a template index 0..4')
    m = np.asarray(coarse)
    if m.shape != (5, 7) or m.dtype != np.bool_:
        raise ValueError('Expected bool coarse ROI [5,7]')
    roi, background = TEMPLATES[int(action)] if m.any() else (0, 0)
    values = np.where(m, roi, background).astype(np.int64)
    ys = np.arange(CTU_ROWS) * COARSE_ROWS // CTU_ROWS
    xs = np.arange(CTU_COLS) * COARSE_COLS // CTU_COLS
    mean = float(values[ys[:, None], xs[None, :]].mean())
    center = int(np.sign(mean) * np.floor(abs(mean) + 0.5))
    if abs(center) > 2:
        raise ValueError('Centered QP exceeds available global bin')
    return np.r_[center + 2, (values + 2).ravel()].astype(np.int64)


def augment_observation(native_obs, coarse):
    """Add source-based ROI channel to the unchanged encoder state (32 + 6 maps)."""
    if set(native_obs) != {'global', 'mb'}:
        raise ValueError('Unexpected native observation keys')
    glob = np.asarray(native_obs['global'], dtype=np.float32)
    local = np.asarray(native_obs['mb'], dtype=np.float32)
    if glob.shape != (32,) or local.shape != (6, 20, 30):
        raise ValueError('Requires qpdev1-qbins development observation')
    if not (np.isfinite(glob).all() and np.isfinite(local).all()):
        raise ValueError('Non-finite native observation')
    return {'global': glob.copy(), 'mb': np.concatenate([local, expanded_roi(coarse)[None]], axis=0)}

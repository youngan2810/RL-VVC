"""Independent Dueling Double DQN QP Markov decision process for VVenC.

No PPO observation, QP-bin, centering or PPO reward functions are imported.
The native VVenC picture-boundary action bridge is the only shared controller ABI.
All current-picture semantic features use SOURCE detections before encoding.
"""
from __future__ import annotations
import math
import numpy as np

ROWS, COLS = 10, 15  # 480x320 at CTU32
GLOBAL_DIM, MAP_CHANNELS = 25, 5
QP_OFFSETS = (-2, -1, 0, 1, 2)
N_ACTIONS = len(QP_OFFSETS)**2
NEUTRAL_ACTION = 12   # (roi=0, background=0)
REWARD_RATE_WEIGHT = 0.15
REWARD_EXCESS_WEIGHT = 3.0
REWARD_RATE_SLACK = 0.02


def semantic_importance(source_boxes):
    """Confidence-weighted box-area overlaps with each CTU; [10,15] in [0,1].

    The boxes must be available at the encoder *before* the current POC is coded.
    This oracle/teacher feature is not a free or decoder-side signal.
    """
    boxes = np.asarray(source_boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((ROWS, COLS), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 5 or not np.isfinite(boxes).all():
        raise ValueError('Expected finite source car boxes with shape [N,5]')
    if ((boxes[:, :2] < 0).any() or (boxes[:, 2] > 480).any() or
            (boxes[:, 3] > 320).any() or (boxes[:, 2:4] <= boxes[:, :2]).any() or
            (boxes[:, 4] < 0).any() or (boxes[:, 4] > 1).any()):
        raise ValueError('Invalid source detection boxes')
    xs, ys = np.arange(COLS) * 32, np.arange(ROWS) * 32
    importance = np.zeros((ROWS, COLS), dtype=np.float64)
    for x1,y1,x2,y2,confidence in boxes:
        wx = np.maximum(0., np.minimum(xs+32,x2)-np.maximum(xs,x1))/32
        wy = np.maximum(0., np.minimum(ys+32,y2)-np.maximum(ys,y1))/32
        importance = np.maximum(importance,confidence*wy[:,None]*wx[None,:])
    return importance.astype(np.float32)


def action_pair(action):
    if isinstance(action,(bool,np.bool_)) or not isinstance(action,(int,np.integer)) or not 0 <= action < N_ACTIONS:
        raise ValueError('D3QN action must be an integer from 0 to 24')
    return QP_OFFSETS[int(action)//5], QP_OFFSETS[int(action)%5]


def qp_action_map(action, importance):
    """Direct CTU32/QG-compatible 20x30 native ΔQP map, no PPO intermediate."""
    mask = np.asarray(importance)
    if mask.shape != (ROWS,COLS) or not np.isfinite(mask).all() or (mask<0).any() or (mask>1).any():
        raise ValueError('Expected source semantic CTU map in [0,1]')
    roi, bg = action_pair(action)
    delta = np.where(mask>0,roi,bg).astype(np.float64)
    return np.repeat(np.repeat(delta,2,axis=0),2,axis=1)


def legal_actions(importance):
    """Deduplicate identical effective ΔQP maps, e.g. a frame with no objects."""
    seen=set()
    allowed=np.zeros(N_ACTIONS,dtype=bool)
    for a in range(N_ACTIONS):
        m=qp_action_map(a,importance)
        signature=m.tobytes()
        if signature not in seen:
            allowed[a]=True
            seen.add(signature)
    if not allowed[NEUTRAL_ACTION]:
        # In no-ROI frames (0,-2) etc are ordered earlier, so the valid neutral
        # action must explicitly remain legal. Deduplicate against it first.
        raise AssertionError('Neutral action unexpectedly masked')
    return allowed


def _pool_ctu(map_20x30):
    return np.asarray(map_20x30,dtype=np.float64).reshape(ROWS,2,COLS,2).mean(axis=(1,3))


def observation(event, source_boxes, *, frames, target_bps, completed,
                previous_action=NEUTRAL_ACTION, previous_au_bits=0,previous_f1=0.):
    """Causal observation: 25 global values + five [10,15] maps + legal mask."""
    g=np.asarray(event.get('global'),dtype=np.float64)
    raw=np.asarray(event.get('maps'),dtype=np.float64)
    if g.shape!=(16,) or raw.shape!=(4,600) or not np.isfinite(g).all() or not np.isfinite(raw).all():
        raise ValueError('Unexpected native VVenC state schema')
    if (not isinstance(completed,int) or frames<1 or not 0<=completed<frames or
            not np.isfinite(target_bps) or target_bps<=0 or previous_au_bits<0 or
            not 0<=previous_f1<=1):
        raise ValueError('Invalid episode or rate context')
    if int(event.get('step',-1)) != completed:
        raise ValueError('Native state and completed frame count diverge')
    imp=semantic_importance(source_boxes)
    maps=np.empty((MAP_CHANNELS,ROWS,COLS),dtype=np.float32)
    for j in range(3):
        vals=_pool_ctu(np.log1p(np.maximum(raw[j].reshape(20,30),0)))
        maps[j]=np.clip((vals-vals.mean())/(vals.std()+1e-6),-5,5)
    qpa_inverse=_pool_ctu(np.maximum(raw[3].reshape(20,30),1e-12))
    maps[3]=np.clip((4-6*np.log2(qpa_inverse))/63,0,1)
    maps[4]=imp
    budget=target_bps*frames/30
    rate_per_frame=target_bps/30
    last_qp=g[3] if completed else g[0]
    stats=np.array([
        g[0]/63., completed/frames, math.log(target_bps/120000.),
        g[2]/budget, (g[2]-completed*rate_per_frame)/budget,
        1.-completed/frames, last_qp/63.,g[4]/63.,
        g[5],g[6],g[7],g[8]/50.,g[9],
        g[10],g[11],np.log1p(max(g[15],0))/12.,
        g[14]/2., np.clip(g[13],-3,3)/3.,
        np.clip(previous_au_bits/rate_per_frame,0,20)/5.,
        (int(previous_action)-NEUTRAL_ACTION)/12.,previous_f1,
        float((imp>0).mean()),float(imp.mean()),
        math.log1p(np.asarray(source_boxes).size//5)/5.,
        (int(event.get('lookahead_last_poc',0))-int(event.get('poc',0)))/10.
    ],dtype=np.float32)
    if stats.shape!=(GLOBAL_DIM,) or not np.isfinite(stats).all() or not np.isfinite(maps).all():
        raise ValueError('Nonfinite D3QN observation')
    return {'global':stats, 'maps':maps, 'legal':legal_actions(imp)},imp


def f1_from_counts(tp,fp,fn):
    if any((not isinstance(x,(int,np.integer)) or x<0) for x in (tp,fp,fn)):
        raise ValueError('Invalid detection matching counts')
    return 2.*tp/max(2*tp+fp+fn,1)


def rate_debt(cumulative_bits, completed, *, frames, target_bps, slack=REWARD_RATE_SLACK):
    """Nonnegative percentage-point debt against per-frame ideal cumulative rate."""
    if not 0<=completed<=frames or frames<1 or target_bps<=0 or cumulative_bits<0 or slack<0:
        raise ValueError('Invalid rate debt inputs')
    total_budget=frames*target_bps/30
    return max(0.,(cumulative_bits-(1.+slack)*completed*target_bps/30)/total_budget)


def reward_step(f1, au_bits, previous_bits, completed, *, frames, target_bps,
                rate_weight=REWARD_RATE_WEIGHT, excess_weight=REWARD_EXCESS_WEIGHT,
                slack=REWARD_RATE_SLACK):
    """New per-picture reward; sum exactly equals declared clip objective at gamma=1.

    Sum r = mean F1 - rate_weight*(B/Btarget)
            - excess_weight*max(0, B/Btarget - 1 - slack).
    """
    if not 0<=f1<=1 or au_bits<0 or previous_bits<0 or frames<1 or not 0<=completed<frames:
        raise ValueError('Invalid reward data')
    old=rate_debt(previous_bits,completed,frames=frames,target_bps=target_bps,slack=slack)
    new=rate_debt(previous_bits+au_bits,completed+1,frames=frames,target_bps=target_bps,slack=slack)
    budget=frames*target_bps/30
    return (f1/frames-rate_weight*au_bits/budget-excess_weight*(new-old)),new


def clip_objective(frame_f1, final_bits, *, frames, target_bps,
                   rate_weight=REWARD_RATE_WEIGHT,excess_weight=REWARD_EXCESS_WEIGHT,
                   slack=REWARD_RATE_SLACK):
    if len(frame_f1)!=frames or not np.isfinite(frame_f1).all():
        raise ValueError('Incomplete frame F1 trajectory')
    ratio=final_bits/(frames*target_bps/30)
    return float(np.mean(frame_f1)-rate_weight*ratio-
                 excess_weight*max(0.,ratio-1.-slack))

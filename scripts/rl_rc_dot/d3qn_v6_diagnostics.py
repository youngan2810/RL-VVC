"""Research diagnostics with explicit data-validity rules.

No generated counterfactual samples are used for training. Native paired
interventions exist exclusively as evaluation/audit evidence.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def policy_summary(states, q_values, legal_masks, *, action_ids=(2, 9)):
    q = np.asarray(q_values, dtype=np.float64)
    legal = np.asarray(legal_masks, dtype=np.bool_)
    if (q.ndim != 2 or q.shape[1] != 25 or q.shape != legal.shape or
            not np.isfinite(q).all() or not legal.any(axis=1).all()):
        raise ValueError('Invalid Q matrix or legal mask')
    scores = np.where(legal, q, -np.inf)
    greedy = scores.argmax(axis=1)
    hist = np.bincount(greedy, minlength=25)
    active = hist[hist > 0]
    frac = active / max(active.sum(), 1)
    entropy = -float(np.sum(frac * np.log(frac))) if len(frac) else 0.
    ordered = np.sort(scores, axis=1)
    qgap = np.asarray([np.sort(scores[i, legal[i]])[-1] -
                       np.sort(scores[i, legal[i]])[-2] if legal[i].sum() > 1 else 0
                       for i in range(len(scores))], dtype=np.float64)
    return {'states': len(scores), 'greedy_histogram': hist.tolist(),
            'dominant_action': int(hist.argmax()),
            'dominant_fraction': float(hist.max() / len(scores)),
            'greedy_entropy': entropy,
            'top12_gap_median': float(np.median(qgap)),
            'pair_legal_fraction': float(np.mean(legal[:, action_ids].all(axis=1))),
            'pair_qgap_median': float(np.median(q[:, action_ids[0]] - q[:, action_ids[1]]))}


def compare_matched_pair(branch2, branch9, *, action_step, scale=25., atol=1e-10):
    """Strict matched-prefix check for recorded native video trajectories.

    Both branches must encode the same prefix and share a *pre-action* state.
    Suffix action sequence may diverge: report it, never suppress it.
    No assumption is made about counterfactual optimal Q*.
    """
    a, b = branch2, branch9
    if a.get('video') != b.get('video') or a.get('rate_kbps') != b.get('rate_kbps'):
        raise ValueError('Paired probe must use identical video and bitrate')
    rows_a, rows_b = a['frames'], b['frames']
    if len(rows_a) != len(rows_b) or not 0 <= action_step < len(rows_a):
        raise ValueError('Invalid trajectory length or probe position')
    keys = ('poc', 'action', 'au_bits', 'frame_f1', 'cumulative_bits')
    for i in range(action_step):
        for key in keys:
            x, y = rows_a[i][key], rows_b[i][key]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(float(x)-float(y)) > atol:
                    raise ValueError(f'Prefix mismatch at t={i}, {key}')
            elif x != y:
                raise ValueError(f'Prefix mismatch at t={i}, {key}')
    if rows_a[action_step]['action'] != 2 or rows_b[action_step]['action'] != 9:
        raise ValueError('Probe actions do not match planned intervention')
    if not bool(a.get('same_pre_action_state')) or not bool(b.get('same_pre_action_state')):
        raise ValueError('Pre-action state parity not established')
    d_q = float(a['q2_minus_q9'])
    if not np.isclose(d_q, float(b['q2_minus_q9']), atol=1e-6):
        raise ValueError('Q-gap differs at shared state')
    d_j = float(a['objective']) - float(b['objective'])
    suffix_a = [r['action'] for r in rows_a[action_step+1:]]
    suffix_b = [r['action'] for r in rows_b[action_step+1:]]
    return {'video': a['video'], 'rate_kbps': a['rate_kbps'],
            'coding_step': action_step, 'poc': rows_a[action_step]['poc'],
            'delta_Q_2_minus_9': d_q, 'delta_J_2_minus_9': d_j,
            'delta_return_scaled': scale * d_j,
            'ranking_conflict': bool(d_q * d_j < 0 and abs(d_j) > 1e-5),
            'suffix_action_sequences_equal': suffix_a == suffix_b,
            'suffix_action_disagreements': sum(x != y for x, y in zip(suffix_a, suffix_b)),
            'prefix_verified': True, 'same_pre_action_state': True,
            'outcome_type': 'frozen_policy_continuation_not_optimal_q'}


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

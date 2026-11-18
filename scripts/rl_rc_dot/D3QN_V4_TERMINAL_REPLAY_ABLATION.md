# D3QN V4 — Terminal-Aware Replay: controlled ablation

## Status and scientific claim

**Implemented; NOT validated by a new VVenC/YOLO training run.** This is a
single-change ablation of V3 on the same BDD100K Mini Train16/Val4 data,
5,120 training frames, 256-frame episodes, and 120-kbps Val4 monitoring.
V3 sources and old SAFE ZIPs remain reproducible and untouched. Do not
interpret lower TD loss or better terminal calibration as proof of better
video coding.

### Evidence motivating this change

Audits of the *existing* checkpoints at 5,120 training steps (one seed):

| Measurement | V3 n=1 | V3 n=3 |
|---|---:|---:|
| Replay terminal transitions | 20/5120 | 60/5120 |
| Mean terminal predicted Q (scaled reward) | 1.001955 | 2.433443 |
| Mean terminal target (no bootstrap) | 0.080026 | 0.160758 |
| Mean terminal TD target minus Q | -0.921928 | -2.272685 |
| Mean replay Q | 0.893670 | 2.451404 |
| Mean realized behavior MC return | 9.429884 | 9.533586 |

The transition at the terminal boundary is handled **correctly** in source:
`done=True` removes the bootstrap term. The network does *not* match
those rare terminal targets. This points to a learning/calibration issue,
**not** a proven bug in terminal flag computation. Behavior-policy
Monte Carlo returns are **not** ground-truth optimal Q-values.
Large negative rewards at GOP/intra-related positions warrant separate
study, but this ablation intentionally does not modify the reward.

Prior Val4 (same target 120 kbps): best V3 n=3 J=0.651504, best V3 n=1
J=0.645517, fixed ROI (-2,0) J=0.658950. Both learned policies were
below fixed ROI. These are development validation results, not Test30,
BD-rate or mAP generalization evidence.

## Single intended algorithmic difference

- V3: uniform sample 64 replay transitions without replacement.
- V4: draw `min(4, available terminal rows)` terminal transitions
  **without replacement**, then fill the rest of the same 64-sample
  minibatch with nonterminal transitions, also without replacement.
  Shuffle the sampled indices before forming tensors.
- If no terminal sample exists, the batch is entirely nonterminal.
  With only three terminal rows, use three; **never fabricate or repeat**
  a transition within a minibatch.
- No importance weights are applied. This deliberately changes the
  **training distribution**; interpret as terminal-oversampling ablation,
  not an unbiased estimator of the V3 uniform replay objective.
- The original replay storage/order, transition payload, n-step
  aggregation, reward values, action/legal mask, optimizer, target network
  and Polyak updates are unchanged.

All key unchanged settings: `n_step=3`, `gamma=1`,
`reward_scale=25`, `target_tau=0.005`, `lr=5e-5`,
`batch_size=64`, `warmup=512`, `train_every=4`,
`epsilon floor=0.2, decay=10240`, `seed=20261111`.
All 16 train and 4 validation videos are locked using existing
`dataset_lock.json`. Test30 is deliberately held out.

## New entry points / isolation

- `d3qn_agent_v4.py`: V4 terminal-stratified replay plus V4-only
  checkpoint type `d3qn-native-checkpoint/v4`.
- `d3qn_native_runner_v4.py`: V4-only experiment identity
  `native-masked-d3qn-qp/v4-terminal-aware`; CLI
  `--terminal-quota 4`; logs `mean_update.sampled_terminal`.
- `test_d3qn_native_v4.py`: CPU tests for exact quota, disjoint
  samples, sparse fallback, RNG reproducibility after checkpoint,
  n-step terminal flush, optimization/legal-mask behavior, and strict
  V3/V4 checkpoint isolation.
- `notebooks/VVC_D3QN_V4_TerminalReplay_Mini16_Val4_Kaggle.ipynb`:
  pinned source SHA, toolchain preflight, V4 fresh training, five Val4
  checkpoints, baselines, SAFE ZIP.
- Reusing an **old V3 or V3 n-step=1 checkpoint in V4 is forbidden**.
  V4 resumes only from a matching V4 checkpoint and identity.

### Quick CPU checks

```bash
python -m py_compile scripts/rl_rc_dot/d3qn_agent_v4.py \
    scripts/rl_rc_dot/d3qn_native_runner_v4.py \
    scripts/rl_rc_dot/test_d3qn_native_v4.py
python -m unittest scripts.rl_rc_dot.test_d3qn_native_v2 \
    scripts.rl_rc_dot.test_d3qn_native_v3 \
    scripts.rl_rc_dot.test_d3qn_native_v4 -v
```

### Kaggle run

Open the V4 notebook, add the same BDD100K dataset, enable GPU and
Internet, execute all cells; default `RESUME_SAFE_ZIP=None` forces a
fresh run. The notebook exports
`D3QN_V4_TERMINAL_REPLAY_MINI16_VAL4_SAFE.zip`.

Check `train_history.json` for `mean_update.sampled_terminal`
after warmup. In practice, with n=3, after at least two complete
256-frame episodes there are >= 6 terminal transitions and V4 should
sample four of them per optimizer step. This is a **sampling audit**, not
an improvement metric.

### Acceptance criteria

1. V4 unit tests and real native VVenC/VVdeC/YOLO preflight pass.
2. Completed Train16 coverage, 20 episodes, five checkpoint milestones
   with four validation videos and decoder parity. Zero Test30 access.
3. Compared to V3, report terminal Q versus terminal target,
   early/mid/late Monte Carlo-minus-Q calibration **with the behavior
   policy caveat**, action concentration, F1, actual kbps, and J.
4. **Primary criterion:** improved Val4 objective, ideally exceeding
   both V3 best 0.651504 and fixed ROI 0.658950 at the same target;
   no claim of success based on Q-error alone.
5. If V4 only improves terminal fit but not J, this hypothesis is
   insufficient; next experiments must be separately controlled and
   consider state representation, reward spikes, or longer horizons.

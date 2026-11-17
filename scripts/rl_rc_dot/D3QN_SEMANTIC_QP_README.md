# Experimental D3QN semantic-QP branch (not a paper result)

This branch is a **non-invasive feasibility prototype** for task-aware QP
allocation. It does not change VVenC C++, the existing `qpdev1-qbins`
PPO agent, or any Test30/Test100 data. No performance gains are claimed.

## Research decision

**Dueling Double DQN is a reasonable first off-policy control** if the
policy chooses **one of five discrete spatial QP templates per picture**.
It is **not** an appropriate drop-in replacement for the original
`MultiDiscrete([5]*36)` policy, where naive DQN would have `5^36` joint
actions. Changing the action space and adding semantic state also changes
the *method*, not just the optimizer. An honest algorithm comparison needs
an additional **PPO with exactly the same five actions and same state**.

Before claiming D3QN is effective, demonstrate that **fixed semantic ROI**
templates beat the strong native QPA and fixed-QP controls **at measured,
matched bitrate**, not merely the same target bitrate.

## Causal state/action contract

- The unmodified `VVenCEnv(..., agent_profile='qpdev1-qbins')` supplies
  its 32 global state features and six 20x30 encoder feature maps.
- A source-frame YOLO detector is run **before encoding the current POC**,
  and its boxes are transformed into a seventh `20x30` semantic map.
  This is a costly **encoder-side semantic prior / oracle diagnostic**,
  *not* a deployable detector-free solution. It changes the runtime
  assumptions relative to the existing encoder-only policy.
- The detector result is cached for the existing source-versus-reconstruction
  pseudo-label reward, to avoid two source inference passes.
- The D3QN action is `Discrete(5)`:
  `A0 (ROI=0, BG=0)`, `A1 (-1,+1)`, `A2 (-2,+2)`,
  `A3 (-1,0)`, `A4 (-2,+1)`.
  If no car is detected, all actions become a *zero* ΔQP map.
- A source box marks all intersecting 32x32 CTUs. CTUs are OR-pooled onto
  **5x7 coarse regions**; this enlarges small ROIs. Actions align with the
  original 20x30 map and exact 36-bin QG mapping.
- QP remains **native QPA + integer ΔQP**. To neutralize the native
  `map_qp_bins` spatial-centering operation, the global head exactly
  equals the rounded coarse map mean. Unit tests check the *actual*
  post-mapping 20x30 values against the intended template.
- Reward is unchanged from `qpdev1-qbins`:
  `mean_frame_F1 - 0.1 * abs(log(actual_bitrate / target))`;
  per-frame increments telescope to the final clip score.
- The D3QN uses Double-Q bootstrap, Dueling value/advantage streams,
  online/target networks, bounded replay buffer, Huber loss,
  gradient clipping, epsilon-greedy exploration, and an explicit
  fixed 256-step episodic horizon. `gamma=1` preserves the undiscounted
  clip objective; rewards are scaled positively by 10 for learning only.
- No access to current reconstructed detections, coded future statistics,
  or Test split is given to the policy before its action.

## Minimal tests (no native encoder required)

```bash
python -m unittest scripts.rl_rc_dot.test_semantic_d3qn -v
```

This is a **unit test only**. It does not establish native build,
CUDA/YOLO environment correctness, VVdeC parity, bitrate feasibility or
learning quality.

## Native probe (run on the existing validated Kaggle environment)

First run the repository's existing native preflight for the repaired
`native-qpa-single-delta/2` action contract. Install the existing pinned
requirements and preserve baseline/dataset locks. Then use **a new output
directory for each run**:

```bash
python -m scripts.rl_rc_dot.d3qn_experiment \
  --phase probe \
  --input-root /kaggle/input/datasets/robikscube/driving-video-with-object-tracking \
  --encoder /path/to/repaired/vvencapp \
  --decoder /path/to/vvdecapp \
  --yolo-repo /path/to/pinned/yolov5 \
  --weights /path/to/pinned/yolov5n.pt \
  --output /kaggle/working/d3qn_probe_v1 \
  --device cuda:0 --target-kbps 120 --val-videos 1
```

The `probe` phase evaluates the native QPA anchor and the four fixed
semantic templates, writing `validation.json`, full native logs and
the FFmpeg-independent VVdeC parity result. **This is not a rate-matched
RD comparison**. Run multiple actual-rate points and interpolate the
rate-task curve before deciding that headroom exists.

## D3QN pilot (only if a useful ROI effect is confirmed)

```bash
python -m scripts.rl_rc_dot.d3qn_experiment \
  --phase train \
  --input-root /kaggle/input/datasets/robikscube/driving-video-with-object-tracking \
  --encoder /path/to/repaired/vvencapp \
  --decoder /path/to/vvdecapp \
  --yolo-repo /path/to/pinned/yolov5 \
  --weights /path/to/pinned/yolov5n.pt \
  --output /kaggle/working/d3qn_train_v1 \
  --device cuda:0 --train-steps 5120 --target-kbps 120 --val-videos 1
```

This is a **smoke/learning pilot**, not full scientific training:
5120 encoded frames on Mini16 are unlikely to establish reliable
generalization. The checkpoint is a prototype. The runner does not yet
support validated resume/best-checkpoint selection or multi-seed runs.
Do not run Test30/Test100 or claim results until native preflight, verified
validation at multiple bitrates, robust training diagnostics, and fair
algorithm ablations have passed.

## Go/no-go for a scientific experiment

1. Unit tests, integer requested-QG / lambda audit, VVenC zero-action
   byte parity, and **independent VVdeC decoding** all pass.
2. Fixed semantic templates achieve a reproducible detection benefit
   at **matched measured bitrate** vs native QPA, fixed global offsets,
   and a simpler non-RL ROI heuristic.
3. D3QN learns a state-dependent action policy, beats fixed ROI and
   strong baselines on untouched Validation, has nontrivial action/Q-gap
   diagnostics, and obeys the predeclared bitrate acceptance criteria.
4. Add a matched `PPO + Discrete(5) + same semantic map` control before
   attributing any improvements to *D3QN instead of PPO*.
5. Count source-YOLO feature generation and policy inference in encoder
   runtime; explicitly separate task-retention pseudo-F1 from detector
   accuracy against BDD100K ground-truth annotations.

Only after those checks should QP policy be finalized and Partition/MTS
controllers be designed.

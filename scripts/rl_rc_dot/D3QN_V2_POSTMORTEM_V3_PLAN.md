# D3QN V2 pilot postmortem and V3 stabilization plan (Mini16/Val4)

## Scope and provenance

This report uses the **real 5,120-transition** V2 learning-check exported as
`train_history.json`, `monitor_history.json`, `baseline.json` and
`notebook_manifest.json`, with source commit
`bc214cd6a10dc9098d9e0534279037bbc094ce5c` and seed `20261111`.
The locked Mini uses **Train16 and Val4** only. **Test30 remains untouched.**
The historical V2 implementation and checkpoints remain unmodified for audit.

All means below are **macro averages over the four locked validation videos**,
120 kbps target, 256 pictures per video, source-YOLO/reconstruction-YOLO
pseudo-label *frame F1*. They are **not** ground-truth BDD100K detection mAP
or matched-rate BD-rate.

## Actual observed performance

| checkpoint | mean frame F1 | mean actual kbps | objective J |
|:--|--:|--:|--:|
| QPA anchor | 0.770815 | 117.870 | 0.623477 |
| Fixed ΔQP=-2 (entire frame) | 0.793599 | 119.859 | 0.643776 |
| D3QN @1024 | 0.772605 | 116.894 | 0.626488 |
| D3QN @2048 | 0.767966 | 117.960 | 0.620516 |
| **D3QN @3072** | **0.798227** | 118.288 | **0.650367** |
| D3QN @4096 | 0.780210 | 116.183 | 0.634981 |
| D3QN @5120 | 0.781997 | 116.304 | 0.636617 |

V2 best observed Val4 checkpoint was **3072**, NOT 5120.
The advantage over fixed -2 at 3072 is only **0.004628 F1** (0.46pp)
and +0.006591 objective; it is strongly influenced by individual videos,
so cannot be claimed as robust rate-matched gain.

The deterministic greedy ROI-present action histogram across 1024 Val frames:
- step 1024: action 14 = (ROI 0, BG +2), **1002/1024**.
- step 2048: action 13 = (0,+1), **652/1024**.
- step 3072: action 19 = (+1,+2), **878/1024**.
- step 4096: action 19 = (+1,+2), **1011/1024**.
- step 5120: action 19 = (+1,+2), **943/1024**.

All 1024 Validation frames in each milestone have source-YOLO ROI.
Thus the pilot cannot establish behavior on no-object frames.

The mean training batch |TD| increased from about 0.1965 in the first
1024 transitions to 0.2806 in the last 1024; valid Q-max from 0.0863
to 0.9446. This is **consistent with unstable value fitting**, not
proof that DDQN diverged. V2 epsilon declined to approximately 0.05
after only 5120 steps, while the 25-action state-action space saw
only 16 unique training videos (20 episodes).

## Diagnosed vs unproven causes

**Confirmed source-level weaknesses**
1. Exploration ends very early (linear 5120-step epsilon schedule, floor 0.05).
   A high action-count during exploratory TRAIN does not demonstrate
   useful action-dependent policy on greedy VAL.
2. One-step bootstrap with gamma=1 and hard target synchronization
   every 250 SGD updates can produce volatile targets with short,
   correlated, mixed-rate trajectories. This is a plausible mechanism,
   NOT a causal attribution established by the single seed.
3. The previous action entered the global state as scalar
   `(action_id-12)/12`. This imposes an arbitrary ordering instead of
   representing the two physically meaningful ROI/BG offsets.
4. The notebook previously considered the latest checkpoint rather
   than selecting the best by frozen Val4 objective; the result demonstrates
   why periodic checkpoint selection matters.
5. The D3QN V2 runner imported `evaluate.verify_decoded`, pulling in
   PPO/SB3 dependencies despite algorithmic independence.

**Hypotheses requiring controlled native ablations**
- Since the reward contains a bitrate cost and an excess penalty,
  the greedy action (+1,+2) may be a shortcut to bit savings when
  source-YOLO pseudo-F1 is already high; counterfactual fixed-action
  measurements must test whether a context-dependent agent adds value.
- With only four Val videos at one target rate, the observed
  concentration into action 19 could be generalization failure,
  a nearly optimal fixed policy, or a limited action/reward definition.
- Temporal propagation from reference-coded pictures can delay the
  effect of per-picture QP actions; three-step replay may improve
  credit assignment but needs an n_step=1 control experiment.

## V3 changes (new protocol, no V2 checkpoint reuse)

Files:
- `d3qn_agent_v3.py`: n-step=3 return, fixed finite-horizon gamma=1,
  Polyak target update tau=0.005, LR=5e-5, slower epsilon schedule
  (1.0 → 0.20 over 10240 transitions; 0.60 at 5120). Full replay and
  both neural networks still checkpointed at each 256-frame boundary.
- `d3qn_native_env_v3.py`: replace redundant remaining-progress and
  ordinal previous-action features with **previous ΔQP_ROI and ΔQP_BG**.
  Preserve 25 global features and five native/semantic spatial maps.
- `d3qn_native_runner_v3.py`: Train16 permutation per epoch unchanged;
  sample 50/80/120/160/200 kbps **equally often** across episodes instead
  of uncontrolled random targets. Expose `--n-step`, `--target-tau`,
  `--learning-rate`, `--epsilon-floor`, `--epsilon-decay-steps`
  to permit controlled training-only ablations; each is part of strict
  checkpoint experiment identity. Add non-RL controls
  `roi_plus1_bg_plus2` (=action19), `roi_minus2_bg_zero` (=action2).
- `d3qn_verify_decoded.py`: independent VVdeC parity without PPO import.
- `test_d3qn_native_v3.py`: CPU regression tests for n-step terminal
  flush, legal-action Q-gap, Polyak update, action-factor observation
  and V3-only checkpoint resume.

**Unchanged on purpose:** source detector YOLOv5n, frozen dataset IDs and
hashes, coding-order native VVenC QPA and QG/RDO contract, 25 ROI/BG
action pairs, F1/bitrate clip objective, gamma=1. The native encoder
and decoder are NOT altered. This isolates an improved value-learning
method without pretending that a newly tuned reward was validated.

V3 is a fresh experiment. Do NOT load a V2 checkpoint into a V3 agent;
the V3 checkpoint contains a separate protocol and strict
source/dataset/algorithm identity.

## Source and paper rationale

- van Hasselt et al., **Deep Reinforcement Learning with Double
  Q-Learning**, AAAI 2016, https://doi.org/10.1609/aaai.v30i1.10295
  (online argmax, target evaluation, reduce maximization bias).
- Wang et al., **Dueling Network Architectures for Deep Reinforcement
  Learning**, ICML 2016, https://proceedings.mlr.press/v48/wangf16.html
  (value/advantage factorization and generalization among similar actions).
- Fedus et al., **Revisiting Fundamentals of Experience Replay**, ICML
  2020, https://proceedings.mlr.press/v119/fedus20a.html
  (n-step returns and replay ratio empirical ablations).
- Schaul et al., **Prioritized Experience Replay**, ICLR 2016,
  https://mlanthology.org/iclr/2016/schaul2016iclr-prioritized/
  (prioritized replay is a candidate for **future ablation**; not added
  in V3 because it introduces new sampling bias and IS correction).
- Gadot et al., **RL-RC-DoT: A Block-level RL Agent for Task-Aware
  Video Compression**, CVPR 2025,
  https://research.nvidia.com/publication/2025-06_rl-rc-dot-block-level-rl-agent-task-aware-video-compression
  (long-term task and bitrate effects of per-block QP).
- **Reinforcement Learning based Low Delay Rate Control for HEVC
  Region of Interest Coding**, 2022,
  https://ieeexplore.ieee.org/abstract/document/9950043
  (frame rate control and ROI/non-ROI offset as two decisions).
- **Hierarchical Reinforcement Learning-Based Adaptive Initial QP
  Selection and Rate Control for H.266/VVC**, Electronics 2024,
  https://doi.org/10.3390/electronics13245028
  (state includes rate/complexity; QP decisions need bitrate-aware design).

No paper is claimed to prescribe *these exact* V3 hyperparameters; tau,
n-step horizon and slower epsilon are explicit hypotheses to validate.

## Next gated experiment

1. Python unit tests V2+V3 and **real** VVenC/QG and VVdeC
   preflight must pass on Kaggle; GitHub CI alone does not prove this.
2. Fresh V3 Mini16, five chunks of 1024 native transitions
   = 20 episodes, 256 frames each; no re-use of V2 checkpoint.
3. Fixed Val4 @120 kbps after each chunk. Save ALL five V3
   checkpoints. Select best by **mean Val4 objective**, not by last update.
4. Measure QPA, fixed global -2, fixed (+1,+2) and fixed (-2,0)
   once on the same Val4; compare paired per-video F1, objective,
   measured bitrate, QP action entropy, and action19 concentration.
5. If V3 improves, verify a controlled `--n-step 1` vs
   `--n-step 3` ablation with identical split/seed/other parameters.
   Do not claim algorithm or rate-matched improvements from one seed.
6. **No Test30** until architecture, reward, action budget, best-checkpoint
   policy and follow-up validation have been locked.

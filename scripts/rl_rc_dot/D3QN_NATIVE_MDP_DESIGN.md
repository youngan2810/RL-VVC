# D3QN Native QP v2 — independent research design

**Status:** engineering prototype for a new MDP, not trained evidence. Source base:
`main` at `377ed0ae316c9ea74d836b697bae8bfd383398ae`.
This is a separate research branch, not a rename of the old PPO adapter.
No code in this branch changes native VVenC C++ or the established PPO tests.

## Research question

Can a **masked Dueling Double DQN** allocate spatially differentiated
integer QP corrections for machine-object detection, while retaining the
VVenC native QPA/RDO pipeline, better than native VVenC QPA and simple
non-RL ROI policies at comparable *measured* bitrate?

This is a hypothesis, not a claim. It separates **algorithm** from
**state/action/reward design**.

## Independent Markov decision process (MDP)

**Time step:** one coding-order picture from the existing native VVenC
action bridge. One `NativeEpisode.step` consumes a full `20x30` integer
ΔQP action map. CTU32 values are duplicated over the corresponding
2×2 grid of 16×16 cells so QG aggregation cannot erase integer actions.
No 36-category PPO mapping or frame-centering is involved.

**Observation:** 25 normalized global features and five spatial
`10x15` maps:

- Global features: current slice QP, progress, target rate, actual
  cumulative budget consumption and debt, remaining fraction,
  previous picture/cumulative QP, coding-mode fractions, last PSNR/SSIM,
  motion/transform bit fractions, frame complexity, I/P/B context,
  native rate discrepancy, previous picture AU bits, previous action,
  previous task F1, current semantic ROI occupancy/confidence/count,
  and permitted lookahead length.
- Spatial channels: source energy, intra complexity, temporal
  propagation, native QPA QP estimate, and source-frame YOLO
  confidence-weighted car-box overlap.
- Legal-action mask: 25 boolean entries computed from the actual
  semantic ROI, excluding redundant actions. Neutral action (0,0)
  is always retained.

**Causality:** the current YOLO semantic map is computed from the
*source* frame before encoding; no reconstructed detections are fed
into the current decision. Previous-picture F1 is available from
the actual completed codec step. The source detector **runs at the
encoder** and must be counted in the reported compute cost. It is an
oracle/teacher prior for the feasibility stage, not a zero-cost
deployment assumption.

**Action:** one discrete choice out of 25 Cartesian products:

`(ΔQP_ROI, ΔQP_BACKGROUND) ∈ {-2,-1,0,+1,+2}²`.

This includes (0,0) = native QPA untouched; equal offsets = frame
wide changes; unequal offsets = spatial bit redistribution. The map
is generated directly on the CTU32 grid and passed to the native QG
hook. Action masking deduplicates identical spatial maps (for example,
no-car frames), and is respected in ε-greedy exploration AND
Double-DQN bootstrap. The action family is deliberately finite
for initial DQN identification, but **is not inherited** from PPO
or the earlier 5 hand-picked ROI templates.

**Reward:** true post-codec per-picture task F1 and actual access-unit
bits. The declared clip objective is

`J = mean_frame_F1 - 0.15*(B_actual/B_target)
     - 3.0*max(0, B_actual/B_target - 1.02)`.

Each step uses `F1_t/N`, a per-AU rate price and a potential
difference in cumulative overspend. With `gamma=1`, the sum of
rewards equals the above final objective **exactly** for a full
episode. The environment asserts this after every completed clip.
The constants (0.15,3.0,2%) are *initial hypotheses*, not
validated optima. Choose final values on Train/Validation only,
never Test.

**Detector metric:** source YOLO versus reconstructed YOLO IoU
matches, as in the existing protocol, to preserve evaluation
comparability. This remains *pseudo-label retention*, not BDD100K
ground-truth mAP. For publication include manual annotation-based
detection evaluation where possible.

**Learning:** masked Dueling Double DQN, off-policy replay capacity
10k, Huber TD loss, target sync, Adam (1e-4), gradient clipping,
epsilon-greedy exploration, and finite-horizon `gamma=1`.
The replay stores observations AND next-state legal masks.
Training updates reuse real native transitions. DQN is not claimed
to dominate PPO; verify training stability with multiple seeds,
Q gaps, action frequencies, TD errors, and live codec outcomes.

## Why this is not the previous PPO or five-template adapter

- No import of `VVenCEnv`, `qp_development.state`,
  `qp_development.reward_increment`, `map_qp_bins`, or `QpBinsPolicy`.
- New `D3QNNativeEnv` reads the native VVenC state directly,
  queries source YOLO, applies its own action map, computes
  its own task/rate reward, and writes a complete trajectory.
- Existing `NativeEpisode`, `VideoCatalog`, original
  one-to-one detection matching and independent VVdeC verification
  are retained as **shared experimental infrastructure**.
- The earlier PR is a **technical adapter feasibility prototype**,
  not the method selected here.

## Suggested validation gates

1. Unit tests:
   `python -m unittest scripts.rl_rc_dot.test_d3qn_native_v2 -v`.
   Confirm shape, integer action, rate objective telescoping,
   masking, valid Q-values and finite replay update.
2. Existing preflight: verify native QG QP, lambda/RDO, zero-QP
   bitstream parity, and independent VVdeC reconstruction.
3. Fixed-action oracle headroom: test native QPA, frame-wide ±1,
   ROI/background combinations, and a simpler task-aware heuristic.
   Compare at multiple **actual bitrate** points, not only equal
   rate targets. Use fixed Validation clips.
4. Train limited 5120 native transitions ONLY after headroom
   exists. Validate 25-action policy against controls; inspect
   state-dependent action changes and full bitstream hashes.
5. Add a **PPO agent trained on this identical v2 MDP** before
   claiming D3QN beats PPO as an algorithm. Earlier PPO with
   different state/action/reward is merely a historical baseline.
6. Test30/Test100 must remain untouched until the method,
   hyperparameters and checkpoint selection are locked.

## Execution (requires the existing verified Kaggle environment)

```bash
python -m scripts.rl_rc_dot.d3qn_native_runner \
  --phase probe \
  --input-root /kaggle/input/datasets/robikscube/driving-video-with-object-tracking \
  --encoder /path/to/repaired/vvencapp \
  --decoder /path/to/vvdecapp \
  --yolo-repo /path/to/pinned/yolov5 \
  --weights /path/to/pinned/yolov5n.pt \
  --output /kaggle/working/d3qn_v2_probe_001 \
  --device cuda:0 --val-videos 1 --val-rates 120

python -m scripts.rl_rc_dot.d3qn_native_runner \
  --phase train \
  --input-root /kaggle/input/datasets/robikscube/driving-video-with-object-tracking \
  --encoder /path/to/repaired/vvencapp \
  --decoder /path/to/vvdecapp \
  --yolo-repo /path/to/pinned/yolov5 \
  --weights /path/to/pinned/yolov5n.pt \
  --output /kaggle/working/d3qn_v2_train_001 \
  --device cuda:0 --train-steps 5120 --val-videos 1 --val-rates 120
```

The default native probe uses six *representative* QP actions
(not all 25) and one Validation video/rate to limit cost. It is
a functionality check, not a performance conclusion. The code
does not yet implement checkpoint resume, optimized early stopping,
ground-truth evaluation, full-rate curves, or full multi-seed sweep.
An experiment missing those checks is not a paper-ready result.

## Long-term expansion

Keep the shared encoder state, bit-budget accounting, task metrics,
native action audit and replay schema extensible. Later introduce
Partition and MTS as *separate legal masked action heads* with
their own temporal scope and candidate search/pruning; do not
force VVenC coding tools into the same 25-value QP action set.
A hierarchical coordinator is justified only when tasks are
demonstrably coupled across scales.

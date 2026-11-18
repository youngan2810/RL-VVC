# D3QN V5-A — causal coding-cycle state (controlled V4 ablation)

## Status

Source, runner, notebook, and automated tests are implemented on a **separate
experimental branch**. **No V5 native VVenC training or Val4 improvement is
claimed.** Keep V2/V3/V4 code and their historical checkpoints immutable.

This document separates **measured results** from **causal hypotheses**.

## Evidence from V4 offline audit (source commit ace1db10...)

The reproducible locked Mini16/Val4 V4 experiment completed 20×256=5120
training frames. The best V4 Val4 objective was **J=0.6437757086**
(1024-step checkpoint), versus best V3 n-step=3 **J=0.6515035**
and constant ROI (-2, BG 0) **J=0.658950367** at target 120 kbps.
These are Mini Val4 development results, **not** Test30/BD-rate results.

Independent offline reconstruction verified that reward over each of the 20
episodes telescopes to the exact clip objective:
`mean(F1) - 0.15 * bitrate_ratio - 3 * max(0,rate_ratio-1.02)`.

- V4 raw single-frame reward (scaled ×25) at coding positions t mod 32=0
  has mean absolute magnitude **3.309737**, versus **0.242140** elsewhere.
- Absolute post-training Double DQN TD error at t mod 32=0:
  **4.388790**, versus **0.579193** elsewhere. These 160 of 5120
  transitions contribute roughly 19.6% of total absolute TD error.
- Huber error above 1 at boundary: **96.25%**, elsewhere **8.39%**.
  This is a diagnostic; it does *not* prove Huber is incorrect.
- V4 best/latest action rankings on the exact same 1024 replay states
  agree **0%** (best: action0 on all, latest: action17 on 831/1024).
  Action2 remained legal on 952 states but was not selected greedy.
  We do **not** have counterfactual rewards for unchosen actions.
- Supervised behavior-MC return prediction (held-out videos, same Dueling
  backbone) achieved R² **0.5378** with MSE, **0.5739** with Huber;
  a simple progress+rate Ridge benchmark achieved **0.7020**. Thus
  direct-value supervision partially succeeds, but neither replacing
  Huber nor blaming bootstrapping alone is supported.
- V4 terminal replay 4/64 improves some terminal calibration but fails
  to improve objective or collapse. Leave this fixed for V5-A.
- At 5120 steps the epsilon schedule is still ~0.60, so behavior MC is
  **not** an optimal-action Q label.

## V5-A change — and ONLY experimental algorithmic change

The V4 global state had 25 features including video progress
`completed/frames`, but no explicit periodic encoder **coding-step**
feature. With the encoder configured for `GOPSize=32` and
`intraPeriod=32`, add three causal features:

```python
phase = (coding_step % 32) / 32
cycle_sin = sin(2*pi*phase)
cycle_cos = cos(2*pi*phase)
cycle_boundary = float(coding_step % 32 == 0)
```

The observation now has 28 globals and exactly the previous five 10×15
spatial maps and 25-action legal mask. Input network first linear layer
widens from 825 to 828; convolutional stack, remaining MLP, value/
advantage heads, loss, optimizer, target updates, and value discounting
remain unchanged. **Initialization is controlled:** with equal random seeds,
the original 825 first-layer columns, bias, and all other V4 network
parameters are bit-identical at initialization. The three appended
columns start at zero but receive gradients. V5 initially predicts the
same Q-values as V4 for every legal action, even before V5 training.

**Causality:** `coding_step == event['step'] == completed` is asserted
before action. This is the **encoder's coding decision order**, not the
display-order `poc`. `cycle_boundary` is **not** claimed to identify
a specific actual I/CRA picture; native picture type is not annotated
for these samples. No unencoded image, future F1 or decoder result is
introduced into this state.

**Keep fixed:** `n_step=3`, `gamma=1`, `tau=0.005`,
`lr=5e-5`, `reward_scale=25`, `batch_size=64`,
`warmup=512`, `train_every=4`, `terminal_quota=4`,
`seed=20261111`, 25 ROI/BG QP actions, epsilon schedule,
same 16 training and 4 validation videos, native 480×320/30fps encoder,
trained QP control and independent VVdeC parity verification.

## Source changes

- `d3qn_state_v5.py`: causal feature transform and 28-D input contract;
  validates original 25-D state is not otherwise modified
- `d3qn_native_env_v5.py`: uses V3 native environment and semantic
  source-detector pipeline; extends only the pre-action observation
  and terminal placeholder; requires native step alignment
- `d3qn_agent_v5.py`: 28-D Dueling Q network; V4's n-step/Double DQN,
  Polyak, and terminal-aware replay logic inherited unchanged; strict
  v5-specific replay checks and fully versioned checkpoint
- `d3qn_native_runner_v5.py`: native runner with unique
  `native-masked-d3qn-qp/v5-coding-cycle` protocol, source fingerprint,
  state schema and coding-period identity; Train16/Val4 only
- `test_d3qn_native_v5.py`: CPU unit tests for cycle positions,
  codec configuration, legal action mask, exact n-step/terminal flush,
  optimizer smoke, seeded V4/V5 network initialization parity,
  isolated phase-conditioned action learning on a synthetic reward,
  actual 256-step episode checkpoint/resume equivalence,
  strict V4/V5 checkpoint isolation, and runtime dimensions.
  Old V2/V3/V4 regression tests retained.
- `test_d3qn_v5_notebook.py`: checks notebook source pin, syntax, SAFE
  and audit/validation safety contracts.
- `notebooks/VVC_D3QN_V5_CodingCycle_Mini16_Val4_Kaggle.ipynb`:
  pinned source, preflight and independent decoded parity, 5120-step
  Train16/Val4, 5 checkpoints, fixed action baselines, robust SAFE ZIP
  resume limited to V5, and V5 28-D Q and 32-step TD audit.

## Validation gates (must pass before interpreting V5 result)

1. Run all V2/V3/V4/V5 CPU regressions and notebook syntax/protocol
   checks in GitHub Actions; inspect actual PASS results.
2. Kaggle: source SHA, dataset lock, encoder/detector binary hash, 32-frame
   native action mapping and independently decoded frame parity PASS.
3. Fresh Train16 run, no V4 checkpoint resume, all 20 episodes and five
   Val4 checkpoints. No test split.
4. Report Val4 `J`, F1, actual kbps and action distribution at all 5
   milestones, compare to **both** previous learned policies and fixed
   ROI(-2,0).
5. Report `cycle_td` mean abs TD for t mod 32==0 versus other positions.
   It is a descriptive within-policy replay comparison; compare V4 with
   awareness of different data distributions, not a causal estimate.
6. If cycle-position TD improves but objective/policy ranking does not,
   **V5-A hypothesis is insufficient**. Follow up with a *new separately
   controlled experiment* on credit assignment, state sufficiency, or
   loss conditioning; do not silently change reward, Huber, epsilon and
   replay simultaneously.

## Important limitations

This is a deliberately minimal **scientifically identifiable** code fix.
No evidence proves a single root cause for action collapse or a guaranteed
gain from new cycle features. Repeated videos, one small dataset and a
single seed limit external validity. No counterfactual unchosen-action
reward labels, nor actual per-picture I/CRA labels, are in the audit.
Do not represent V5 results as beating the fixed ROI baseline until
measured on exactly the same locked validation protocol.

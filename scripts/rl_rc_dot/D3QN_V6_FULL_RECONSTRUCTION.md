# D3QN V6 Action-Conditioned Reconstruction — Design, implementation, and scientific gates

**Status:** source and automated CPU tests implemented; **native VVC training has NOT been run**. The human researcher runs the pinned Kaggle notebook. Branch: `experiment/d3qn-v6-action-conditioned-reconstruction`. Historical V3/V4/V5 files and checkpoints are not modified.

## 1. Research motivation and evidence boundary

From the V3–V5 technical postmortem:

| Evidence | Observation | Interpretation |
|---|---|---|
| V5 replay greedy | Action 9 in 4,911/4,943 fully legal states (99.35%) | State-dependent action preferences are weak; legal mask is not the cause there |
| V5 validation | J=0.649158 versus fixed action 2 J=0.658950 | V5 did not improve Val4 objective |
| V5 terminal | Mean abs TD≈1.614 on 60 terminal n-step samples | Value fitting remains difficult even without terminal bootstrap |
| V5 boundary | Mean abs TD≈3.952 boundary vs≈0.561 elsewhere | Sharp position-dependent error; does not alone prove Huber/gamma are wrong |
| V5 native Gate F | 2 action-ranking conflicts among 7 non-negligible pairs | Known misranking occurs under fixed-2 prefix, not yet confirmed on greedy V5 state distribution |
| V5 cycle state | New boundary duplicates a prior state variable on 5,120 replay states | Adding more periodic scalars is not a convincing state fix |

**No single causal root cause has been proven.** The chosen V6 tests a **coherent coupled replacement for the learning system**, not a one-factor ablation. Do not claim this reconstruction is better until native validation supports it.

## 2. Considered alternatives and selected design

| Option | Benefits | Important limitations | Decision |
|---|---|---|---|
| Continue V5, increase terminal quota | Very small change | V4 tested 4/64; still collapsed, induces replay distribution bias | Rejected as primary approach |
| Add more phase/cycle features | Low implementation cost | Existing cycle features partly redundant; no physical reference dependency | Rejected pending causal feature evidence |
| PPO replacement | Different exploration/control | Prior PPO operated a different MDP; no comparable causal evidence | Not selected |
| Offline contextual-bandit labels from logged F1 | Simple supervision | Logged actions do not supply counterfactual reward at same state; long-term effects ignored | Rejected |
| **Action-conditioned Double DQN + coverage-aware replay** | Explicit physical ROI/BG action coordinates, state-action interaction, better action support, auditable Q gaps | Needs native validation; changes multiple learning factors; no observed counterfactual learning labels | **Selected for V6 pilot** |

### Preserved from V5/V3 (because established checks were PASS)

- Native `NativeEpisode` action bridge, QG/RDO-integrated `20×30` ΔQP map, same integer offset set, masking of identical effective maps and VVdeC parity.
- Locked BDD100K Mini Train16/Val4 and 256-picture videos. Test30 remains untouched.
- Existing `D3QNNativeEnvV3` source-YOLO pre-action observation, five `10×15` maps, native RC feedback, past F1, pseudo-label matching and native AU accounting.
- Exact clip objective **`J = mean(pseudo-F1) - .15*rate_ratio - 3*max(0,rate_ratio-1.02)`**, potential-difference rewards, gamma=1, scaled reward×25.
- Known baselines QPA, fixed global(-2), fixed2=(-2,0), fixed9=(-1,+2), and independent validation.

### Replaced and why

**Observation architecture:** Return to the 25 causal V3 globals and five spatial maps, **drop the redundant V5 cycle triplet** from the learned representation. Separate global/codec encoder from spatial conv encoder, then fuse a learned state context. This does not pretend actual temporal-layer or reference graph metadata is available. The native event may require future extension once causality is verified.

**Q-value model:** A physical five-coordinate representation of each action `(ROI/2, BG/2, (ROI-BG)/4, (ROI+BG)/4, ROI*BG/4)` passes through an MLP action encoder. A learned state-dependent 32-D context multiplies the action embeddings to produce **explicit action-conditioned advantages**. A separate state-value head is combined using the **mean over legal actions**, retaining masking. The network can distinguish different action rankings for similar state values. Its capacity to do so is tested on synthetic two-action data; that is not proof of VVC performance.

**Data collection:** Epsilon-greedy exploration chooses among **least-visited legal actions within the target bitrate** (rather than uniform random action selection). Logged rates are still 50, 80, 120, 160 and 200 kbps. All experience uses the same on-policy native codec stream until the current action; no invented counterfactual outcomes enter the replay.

**Replay:** Blend uniform, inverse-action-frequency and coding-boundary balanced sampling (coefficients .45 and .10), with **normalized inverse-probability sample weights**. This targets action support without repeated terminal-only oversampling. Because weights are self-normalized by minibatch maxima, the gradient has a different stochastic scale but not a guarantee of unbiased finite-sample estimates. Sampling with replacement can yield duplicates; diagnostic counters expose this.

**Credit horizon:** `n_step=8` (V5 was 3), gamma=1, same reward. This is a hypothesis to test longer dependencies, **not a demonstrated fix**. TD targets remain off-policy Double DQN targets with Polyak τ=.005, learning rate 5e-5, batch64, warmup512, train_every4, Huber loss, gradient clipping. Keep loss/reward physics unchanged.

**Strict experiment identity:** New `native-masked-d3qn-qp/v6-action-conditioned` protocol, checkpoint version, optimizer/replay/RNG serialization and source/dataset/encoder/detector SHA. A V5 checkpoint is intentionally incompatible with V6 training. Historical frozen V5 checkpoint loading is permitted **only in the separate native diagnostic program**.

## 3. Code map

| File | Purpose |
|---|---|
| `d3qn_v6_agent.py` | Action-conditioned Q network, exploration, action-boundary replay with IS weights, n-step Double DQN, full checkpoint/resume |
| `d3qn_v6_native_runner.py` | Native training/validation/probes with locked Train16/Val4 and fixed controls |
| `d3qn_v6_diagnostics.py` | Policy collapse/Q-gap reports and strict matched-prefix paired-result evaluator |
| `d3qn_v6_native_pair_probe.py` | **Greedy V5 or V6 prefix** native 2-vs-9 intervention, frozen suffix, full video and VVdeC parity |
| `test_d3qn_v6_reconstruction.py` | Synthetic two-action/DDQN test; state, algebra, gradient, replay, n-step, resume, coverage checks |
| `test_d3qn_v6_diagnostics.py` | Matched-prefix Q-return ranking and invalid-pair rejection |
| `notebooks/VVC_D3QN_V6_ActionConditioned_Mini16_Val4_Kaggle.ipynb` | Kaggle source pin, native preflight, Train16/Val4, fixed baselines, SAFE resume, summary |

## 4. Native diagnostics and causality

`python -m scripts.rl_rc_dot.d3qn_v6_native_pair_probe --help` lists the interface. This probe should be run **separately** with a V5 `best_checkpoint.pt` plus BDD100K Train16 and native toolchain:

1. Run frozen V5 greedy at coding steps `0..t-1` in two independent encodes.
2. Confirm the source, actual native observation and prefix outputs are identical.
3. At step `t`, try action2 and action9; check both legal and physically distinct QP maps.
4. Continue both branches using frozen greedy V5 for the remaining pictures.
5. Decode independently, record frame F1, actual AU bits, `J`, Q-gap and prefix/suffix parity.
6. Compare `Q(s,2)-Q(s,9)` with `25*(J2-J9)` and report sign conflicts.

**Do not mistake this for native codec state snapshot/reset**: both episodes re-encode prefixes from the start. The suffix greedy action sequence may differ because the codec state differs after intervention; report disagreements, never silently force equality. This comparison is under frozen V5 continuation, **not optimal Q***.

**Encoder rebuild safety:** If the SHA differs from the frozen checkpoint's encoder, the program normally aborts. An explicit `--allow-encoder-mismatch --reproduction-audit <10_encoder_rebuild_audit.json>` requires a prior Fixed2 Val4 functional reproduction PASS; this remains a disclosed limitation, **not bit-exact equivalence**.

## 5. Local verification status

| Check | Status |
|---|---|
| Python syntax/import (V6 new modules) | PASS locally |
| V6 unit tests (input/gradient/algebra/action geometry) | PASS locally |
| Action-balanced replay/n-step/terminal tests | PASS locally |
| Synthetic known-optimal-action **real Double DQN optimizer** | PASS locally (synthetic only) |
| Full 256-picture checkpoint/resume parameter equivalence | PASS locally |
| Matched-prefix result validator tests | PASS locally |
| Existing native V3/V4/V5 regression tests | **Not run in this local container** (full original checkout/native dependencies unavailable); mandatory in Kaggle Gate B |
| VVenC/VVdeC/YOLO live preflight | **Not run locally**; mandatory Kaggle Gate B |
| Kaggle V6 training and Val4 objective | **NOT RUN** (user responsibility) |
| Greedy-V5-prefix native intervention on Train16 | **NOT RUN** (user responsibility) |

**Local PASS ≠ Kaggle training PASS ≠ native coding improvement.**

## 6. Research protocol and acceptance

- **Stage 0:** Real native preflight including action application and VVdeC decoder parity. Must pass before any RL training.
- **Stage 1:** Fresh V6 Train16, baseline Val4, 20×256 transitions and five independent Val4 monitor checkpoints. Select best by mean `J` and keep latest.
- **Stage 2:** Compare J, pseudo-F1, actual kbps, per-video deltas, state-conditional action changes and Q ranking against fixed2/fixed9. Action entropy by itself is not a success metric.
- **Stage 3:** Greedy-V5-prefix matched interventions; test if historical fixed2-prefix misranking generalizes to the learned policy's own state distribution.
- **Stage 4:** If V6 improves, run controlled ablations (coverage only, horizon 3 vs 8, original vs action-conditioned Q) and multiple seeds. V6 changes several factors simultaneously; it does **not identify which factor caused a gain**.
- **Stage 5:** Only after freezing design, independently evaluate Test30, diverse rates and ground-truth detector annotations; do not use Test30 for tuning.

The **minimum promising empirical gate** is reproducible native Val4 `J > 0.6589504` against fixed2 and meaningful context-conditioned **ranking correctness** under sampled native actions. This single-seed Val4 threshold is a development requirement, not a publication/generalization claim.

## 7. Reproducibility / Kaggle inputs

The notebook must be pinned to the **new source commit SHA** (not V5 `f10ec5de...`). Add one Kaggle Input: `robikscube/driving-video-with-object-tracking` (BDD100K videos). GPU ON, Internet ON. Do not add the historical V5 SAFE ZIP to V6 training; V6 needs a fresh checkpoint. Resume only from matching V6 SAFE ZIP with the exact source commit and experiment identity. The notebook exports `D3QN_V6_ACTION_CONDITIONED_MINI16_VAL4_SAFE.zip`.

After the human training run return the V6 SAFE ZIP (manifest, train/baseline/monitor histories, best/latest/at-step checkpoints, logs, action counts, decoder audits). If training fails return `last_failure.json` and Kaggle cell logs.

## 8. Limitations

- V6 remains a finite-state observation of an encoder with long-range dependencies; the Markov assumption is not established.
- A model with richer action embeddings cannot manufacture counterfactual rewards absent from collected data.
- `n_step=8` does not eliminate longer-horizon TD bootstrapping, and boundary/terminal Q fitting may still be poor.
- Five target rates × 25 actions × multiple contexts remain data-intensive; balanced exploration does not guarantee sufficient effective support.
- No learned V6 native metrics exist at writing. Do not claim action collapse has been resolved.
- Pseudo-YOLO source/recon F1 is not labeled BDD100K detection mAP; source teacher detection compute is not free.

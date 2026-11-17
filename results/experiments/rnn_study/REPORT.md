# Model 6 — Vanilla RNN: Final Research and Completion Report

**Project:** Language Models From Scratch  
**Model:** 6 — Single-layer Elman (Vanilla) RNN  
**Report date:** 16 November 2026  
**Status:** **COMPLETED — study formally closed**  
**Canonical closure:** D-62  
**Post-closure analysis:** D-63

---

## 1. Executive summary

The Vanilla RNN research stage is complete. The architecture was specified and approved, implemented, independently verified, tested, studied under a predeclared experimental plan, evaluated once on the protected test split, analysed, and formally closed by D-62. D-63 added an accepted extended analysis without reopening the experiment.

The model is a single-layer Elman RNN with hidden size 128 and 23,451 trainable parameters. The study focused on recurrent-state dynamics, temporal Jacobian products, loss-projected gradients, context dependence, and the evolution of the learning signal. Predictive performance was recorded as an outcome, not treated as the primary scientific objective.

### Final performance

| Metric | Result |
|---|---:|
| Best development NLL | **2.0520 ± 0.0011** |
| One-time test NLL | **2.0628 ± 0.0010** |
| Training NLL at best epoch | **1.9454 ± 0.0020** |
| Final runs | **10 seeds (0–9), all completed** |
| Selected epochs | 30–38; 3/10 selected epoch 38 |
| Training configuration | η = 0.1; 38 epochs; batch size 32 names |

Uncertainties above are standard errors across the ten seeds, as reported in the project evidence. They do not represent all sources of uncertainty.

### Main scientific outcomes

1. At initialisation, typical temporal Jacobian products contracted at approximately **−0.355 log-norm units per recurrent step**, close to the predeclared mean-field prediction.
2. During training, the recurrent matrix moved away from orthogonality and became expansive in its largest singular direction: **σ_max(W) increased from 1.00 to 3.58**. Hidden-state saturation also increased, from about **2.4% to roughly 19%** at the best epoch.
3. Typical-direction attenuation weakened in absolute terms during training, reaching approximately **−0.13 per step** at the best epoch. This reflects the combined effects of the recurrent matrix and tanh derivative factors; the experiment does not provide a clean additive causal decomposition of their contributions.
4. In the BOS-restart context intervention on the predeclared F₈ subset, the NLL penalty decreased as more recent context was retained: **5.987 nats at d=1** to **0.016 nats at d=8**.
5. The estimated long-distance share of the learning signal for W remained around 0.11 from epoch 1 to epoch 38. D-63 therefore supersedes D-60's earlier wording: M6-HC1 is **partly supported**, not simply unsupported.
6. The EOS-versus-letter split suggests that much of the measured long-context dependence is associated with predicting the end of a name. This split is exploratory and must not be promoted to a confirmatory conclusion.

The study does **not** establish long-range memory, generalisation to long sequences, or that gradient attenuation causes the observed context-use pattern.

---

## 2. Research motivation and scope

The Vanilla RNN was introduced as the first model in the project with a recurrent hidden state and parameters shared across time. Unlike the preceding fixed-context models, it processes the entire available prefix through repeated applications of the same recurrent transformation.

The model served three purposes:

- examine how information and gradients propagate through repeated recurrent transformations;
- establish a baseline for later gated recurrent architectures;
- separate predictive performance from the underlying mechanisms of recurrent-state dynamics.

The research agenda was approved before implementation (D-38). The three core questions were:

| ID | Research question | Main evidence |
|---|---|---|
| M6-RQA | How do temporal Jacobian products and loss-projected gradients attenuate with distance, at initialisation and during training, and how do the recurrent matrix and derivative factors relate to this behaviour? | H1, G1, G2 and counterfactual comparisons |
| M6-RQB | Under a BOS-restart intervention, how much does development NLL degrade when characters more than d steps back are removed, by target position and name length? | C1 and per-position development NLL |
| M6-RQC | How does the distance profile of the learning signal for W evolve, and how does it relate temporally to measured context dependence? | G2, G3 and C1 across training |

The data contain at most 16 recurrent steps per name, with a mean of approximately 7.1. Long-range memory was explicitly outside the scope of the study.

---

## 3. Architecture and implementation

### 3.1 Model definition

The implemented model is a single-layer Elman RNN with:

- hidden size **H = 128**;
- direct character-to-hidden lookup matrix U;
- recurrent matrix W shared across all time steps;
- tanh nonlinearity;
- zero initial hidden state;
- output projection V and output bias c;
- manual recurrence and full backpropagation through time.

For a valid input step t:

\[
 a_t = U[\hat{x}_t] + W h_{t-1} + b, \qquad h_t = \tanh(a_t), \qquad z_t = Vh_t+c.
\]

At padded positions the hidden state is frozen at its last valid value. Each name starts independently from h₀ = 0; hidden states are not carried between names.

### 3.2 Parameter count

| Parameter | Shape | Count |
|---|---:|---:|
| U | 27 × 128 | 3,456 |
| W | 128 × 128 | 16,384 |
| b | 128 | 128 |
| V | 27 × 128 | 3,456 |
| c | 27 | 27 |
| **Total** | | **23,451** |

The implementation uses five raw `nn.Parameter` tensors. High-level recurrent modules and packed-sequence helpers were excluded by the approved specification.

### 3.3 Initialisation

- W = gQ, where Q is Haar-random orthogonal and g = 1.0.
- U is sampled i.i.d. from N(0, s²), with s² = 0.5.
- b, V and c are zero-initialised.
- Initialisation and data-order random streams are separated.
- The initialisation gate retained g = 1 and s² = 0.5 (D-57).

Because V is initially zero, the initial loss is exactly ln(27), and the loss gradient reaching U and W is zero at step zero.

### 3.4 Deliberately excluded components

The approved architecture contains no gates, extra recurrent layers, dropout, normalisation, learned initial state, truncated BPTT, gradient clipping, attention, or residual connections. These exclusions preserve the Vanilla RNN mechanism under study.

---

## 4. Training protocol

The accepted training specification fixed the following procedure:

| Component | Final setting |
|---|---|
| Optimiser | Plain SGD |
| Learning rate | η = 0.1 |
| Momentum / weight decay | None |
| Gradient handling | Full BPTT; no truncation; no clipping |
| Batch | 32 complete names |
| Sampling | EpochSampler, without replacement |
| Steps per epoch | 802 (801 full batches and one final batch of 8) |
| Loss normalisation | Token-summed NLL divided by fixed C = B·M/N = 227.976… |
| Epoch budget | 38 |
| Checkpoint selection | Lowest development NLL; earliest epoch on ties |
| Device / precision | CPU, float32 |
| Thread settings | intra-op 4; inter-op 8 |
| Evaluation | Full train and dev NLL each epoch |

The fixed loss denominator was selected to keep target weighting independent of name length. The training specification explains that per-batch token averaging would alter the relative weight of targets from longer names, which matter to the context analyses.

The final training runs used seeds 0–9. Separate seed ranges were used for the pilot, initialisation gate, and benchmark. The final experiment did not use the test split for model selection.

---

## 5. Experimental workflow and verification

The work proceeded through the following accepted stages:

| Stage | Purpose | Outcome |
|---|---|---|
| Research agenda | Fix questions, hypotheses and scope | Accepted, D-38 |
| Architecture/training/diagnostics specifications | Define model, training and measurements | Approved, D-44–D-48 |
| Implementation | Complete implementation milestones | Accepted, D-50 |
| Independent verification | Specification-derived tests, source review and mutation testing | Accepted, D-51 |
| Probe and execution setup | Fixed probe, thread benchmark and diagnostic timing | Completed, D-54–D-56 |
| Initialisation gate | Check initialisation against declared expectations | g and s² retained, D-57 |
| Learning-rate pilot | Select learning rate | η = 0.1, D-58 |
| Epoch-budget extension | Select final epoch budget | E = 38, D-59 |
| Final research runs | Ten seeds with predeclared diagnostics | 10/10 completed, D-60 |
| One-time test evaluation | Evaluate frozen checkpoints on protected test split | Completed, D-61 and D-62 |
| Extended analysis | Analyse temporal gradients, optimisation and generation | Accepted, D-63 |

### Verification evidence recorded by the project

- Five specification documents and three source modules were reported for Model 6.
- The reconstruction report counts 257 test functions and 16 orchestration scripts.
- An independent specification-derived verification suite passed **48/48** tests (D-51).
- Mutation testing caught **38/38** planted defects (D-51).
- Native test results reported by the user: **540 passed** for Model 6 suites and **1,860 passed, 19 skipped** for the full repository (D-51).
- The study's run verifier was reported clean for all 25 training runs (12 pilot, 3 extension, 10 final) (D-60).
- Pilot and extension runs at the same settings were bit-identical over their shared epochs (D-59).
- Mac and Linux summary outputs agreed to within 5e−16 relative, which is floating-point rounding (D-63; `analysis/summary.json` and `analysis/summary_mac.json`).

**Evidence qualification:** these verification counts and outcomes are recorded project evidence. The tests and experiments were not rerun for this report, and the native test logs, the independent review report and the mutation list were not stored as standalone artefacts. See *Verification and provenance* at the end.

---

## 6. Diagnostics and measurement design

The final research runs used a fixed probe of 1,024 training names, stratified by sequence length, with population weights. Each final run recorded 49 measurement points: step zero, ten early optimisation points, and the end of each of 38 epochs.

| Diagnostic | Measurement | Purpose |
|---|---|---|
| H1 | Hidden-state RMS, saturation and tanh derivative statistics | Characterise the hidden-state operating regime |
| G1 | Exact temporal Jacobian products | Measure typical-direction and selected worst-case propagation |
| W-only counterfactual | Norm of W raised to powers | Describe recurrent-matrix-only amplification/attenuation |
| D/W counterfactual estimate | Product of observed derivative factors with recurrent gain | Compare observed behaviour with simplified reference curves |
| G2 | Loss-projected temporal gradient attenuation | Measure propagation in the actual loss-gradient direction |
| G3 | Distance-wise decomposition of the learning signal for W | Estimate where the recurrent-matrix learning signal originates |
| C1 | BOS-restart context intervention | Measure prediction change when older context is removed |
| O1–O3 | Gradient, parameter, activation and optimisation histories | Describe training dynamics and numerical behaviour |

The specification explicitly treats the attenuation curves as counterfactual comparisons, not as an additive causal decomposition. G3 is estimated from a fixed probe rather than computed as an exact full-training-set decomposition.

---

## 7. Quantitative results

### 7.1 Predictive performance

| Metric | Mean ± seed SE |
|---|---:|
| Best dev NLL | 2.0520 ± 0.0011 |
| Test NLL | 2.0628 ± 0.0010 |
| Train NLL at best epoch | 1.9454 ± 0.0020 |

The best development epochs ranged from 30 to 38. Three seeds selected the final epoch 38. Test evaluation was performed once after the checkpoints and relevant code were frozen. No model selection or rerun was based on test results.

### 7.2 Initial temporal attenuation

At initialisation, the log normalised Frobenius norm of temporal Jacobian products decreased approximately linearly with recurrent distance, at about **−0.355 per step**. This was close to the predeclared mean-field prediction, supporting M6-HA1.

### 7.3 Evolution of W and hidden-state saturation

During training:

- σ_max(W): **1.00 → 1.81** by epoch 1 → **3.58 ± 0.02** by epoch 38;
- σ_min(W): **1.00 → approximately 0.010**;
- spectral radius: **1.00 → approximately 1.79**;
- normalised Frobenius scale g_W: **1.000 → 1.604**;
- hidden-state saturation: approximately **2.4% → 18–19%**;
- mean-square tanh derivative at the best epoch: approximately **0.31**, compared with approximately **0.49** at initialisation.

The typical-direction attenuation rate weakened in absolute terms, from about −0.355 per step at initialisation to about −0.13 per step at the best epoch. The recurrent matrix became more expansive, while the tanh derivative factors reduced local gains. The accepted evidence supports a change in the combined operating regime; it does not justify a precise additive attribution of attenuation to W versus D.

### 7.4 Effective context under BOS-restart

For F₈ (280 late targets from 182 long names), the measured NLL increase when retaining only the last d characters was:

| Retained context d | ΔNLL (nats per target) |
|---:|---:|
| 1 | 5.987 |
| 2 | 4.247 |
| 3 | 2.506 |
| 4 | 1.150 |
| 5 | 0.566 |
| 6 | 0.194 |
| 7 | 0.049 |
| 8 | 0.016 |

The effect decreases strongly as more recent context is retained. At d=8, the name-clustered standard error is 0.023, larger than the seed standard error of 0.005. The appropriate conclusion is that the mean observed effect at d=8 is small; uncertainty across names is material.

The intervention both removes earlier characters and restarts the sequence from BOS. It therefore changes position information and can create inputs unlike those seen in training. It is not a pure measurement of stored character memory.

### 7.5 EOS versus letter targets (exploratory)

An exploratory split of F₈ showed substantially larger BOS-restart effects for end-of-name targets than for letter targets. At d=3, the reported values were 3.63 nats for EOS targets and 0.43 for letter targets. This split was added after the primary results and is explicitly exploratory. It is consistent with the possibility that much of the measured long-context dependence concerns end-of-name prediction, but it does not establish that the model fails to use distant letters.

### 7.6 Learning signal over training

The estimated long-distance share of the learning signal for W, Π̂(Δ ≥ 3), was approximately:

- 0.089 after the first optimiser step;
- a peak around 0.19 at step 32;
- approximately 0.108 at epoch 1;
- approximately 0.113 at epoch 38.

The sign of this share was reproduced in both probe halves for all ten seeds at the listed points. However, late in training the cancellation index κ was approximately 0.014 and the split-half cosine was lower (between 0.34 and 0.70 after step 128), so late-training per-distance estimates require caution.

D-63 corrected the interpretation of M6-HC1: concentration at short distances is observed, while a progressive growth of the long-distance share throughout training is not. The final status is **partly supported**.

### 7.7 Optimisation behaviour

Development NLL improved from approximately 2.238 at epoch 1 to 2.056 at epoch 38. The runs showed a broad plateau from around epoch 30, mild overfitting, and no recorded instability. The per-epoch gradient norms did not show a late training spike. The gradient for W is exactly zero at the first optimiser step because of the zero initialisation of V.

### 7.8 Generation

The extended analysis recorded 120 generated samples at temperature 1, with mean length 6.09 compared with 6.12 in the training data; none were unfinished. This is a qualitative sanity check, not a separate evaluation of sample quality.

---

## 8. Accepted findings and hypothesis status

| Finding | Final status | Qualification |
|---|---|---|
| M6-HA1: initial typical-direction attenuation is near the mean-field rate | **Supported** | Directly measured at initialisation |
| M6-HA2: training moves W away from orthogonality and raises its largest singular value | **Supported** | Direct spectral evidence across seeds |
| M6-HA3: derivative factors explain attenuation relative to W-only growth | **Supported with interpretive qualification** | Absolute attenuation weakened; same-position clause was not directly tested; no additive decomposition |
| Some temporal directions are amplified at the best epoch | **Supported** | Worst-case spectral norms can grow while typical directions contract |
| M6-HB1: context-removal degradation falls with d and is small by d=8 | **Supported** | Conditional on F₈ and the BOS-restart intervention |
| M6-HB2: degradation is larger at late targets in long names | **Supported, with exploratory split** | Position reset and EOS composition complicate interpretation |
| M6-HC1: short-distance learning signal dominates; long-distance share grows throughout training | **Partly supported** | D-63 supersedes D-60 wording; no monotonic/progressive growth established |
| M6-HC2: behavioural context dependence emerges after long-distance learning signal | **Weakly informative association only** | No operational threshold; no causal claim |
| EOS/letter split | **Exploratory** | Post-result analysis; not confirmatory |
| Predictive performance | **Descriptively recorded** | Not inferentially compared with WaveNet or other models |
| Optimisation behaviour | **Accepted descriptive finding** | Plateau, mild overfitting, no observed instability |

---

## 9. Limitations and unresolved questions

The study is complete, but completion does not mean every limitation or engineering note was fixed. The following items remain part of the scientific record and should accompany any use of the results.

### 9.1 Methodological limitations

- One architecture arm and one initialisation; no manipulated factor separates data limitations from gradient limitations.
- Findings are descriptive. No hypothesis tests, confidence intervals or multiplicity correction were predeclared for the research hypotheses.
- The learning rate was selected from a four-point grid.
- The epoch-budget rule depends on an epsilon derived from a three-seed pilot; three final runs selected the last permitted epoch.
- Comparison with Models 3–5 and WaveNet is descriptive only. Training units, learning rates, budgets and context differ.

### 9.2 Measurement limitations

- Recurrent distance, target position and name length are confounded in this dataset.
- G3 is a probe estimate; late-training cancellation and lower split-half agreement weaken fine-grained distance attribution.
- BOS-restart changes both character context and position and may create out-of-distribution sequences.
- F₈ is a selected subset of late targets in long names, not a representative sample of all targets.
- Seeds vary initialisation and data order, not the fixed probe.
- The G2-versus-G1 comparison has a positive Jensen bias; only the sign change is interpreted.

### 9.3 Dataset limitations

- Maximum sequence length is 16 recurrent steps. No long-sequence or long-range-memory claim is supported.
- The first hidden state is the same for every name because all sequences start from the same BOS embedding and zero state.
- Late positions contain a high proportion of EOS targets.

### 9.4 Documented implementation limitations left unfixed

The Stage 9 record (D-51) explicitly retained several minor limitations rather than silently changing accepted infrastructure. These include:

- no run-time R≤σ check for the post-hoc best-epoch spectral point, and no propagation of an invalid post-hoc point into metadata;
- weighted quantiles can shift by one element at an exact cumulative-weight boundary;
- `config.json` stores run identity while the full configuration is in checkpoints;
- the initialisation gate does not call `validate_research_run` before constructing models;
- a directory left by a crash before the epoch-zero row can be reused;
- the final two Stage 9 fixes did not receive a completed independent re-review;
- a minor arithmetic discrepancy in a printed expected tanh-derivative value.

These are documented limitations, not evidence that the final runs failed. They should not be described as fixed unless a later authorised change and verification record establishes that.

### 9.5 Open research questions

- Is the plateau in context use limited by gradient attenuation or by the dataset?
- How much of the BOS-restart degradation is position information rather than stored character information?
- How would context dependence compare with Models 3–5 under a shared protocol?
- Which findings transfer to gated recurrent architectures?

These questions remain open or deferred. They do not prevent closure of the Vanilla RNN study.

---

## 10. Decision history and formal closure

The Vanilla RNN stage progressed from agenda to closure through the following decisions:

| Decision | Date | Accepted outcome |
|---|---|---|
| D-38–D-43 | 13 Nov 2026 | Research agenda, architecture/training/initialisation direction, diagnostics and documentation conventions |
| D-44 | 14 Nov 2026 | RNN architecture specification v0.2 approved |
| D-45 | 14 Nov 2026 | Diagnostics specification v0.3 approved |
| D-46 | 14 Nov 2026 | Training specification v0.3 approved |
| D-47 | 14 Nov 2026 | Initialisation-gate amendment accepted |
| D-48 | 14 Nov 2026 | Integrated implementation contract v0.2 approved |
| D-49 | 14 Nov 2026 | Implementation plan v0.2 approved |
| D-50 | 14 Nov 2026 | Implementation milestones MS0–MS8 completed; declared deviations recorded |
| D-51 | 15 Nov 2026 | Independent verification, source review, mutation testing and native test outcomes accepted |
| D-52–D-56 | 15 Nov 2026 | Study acceptance, probe construction, thread benchmark and diagnostic timing completed |
| D-57 | 15 Nov 2026 | Initialisation gate accepted; g=1 and s²=0.5 retained |
| D-58 | 15 Nov 2026 | Learning-rate pilot accepted; η=0.1 selected |
| D-59 | 15 Nov 2026 | Epoch-budget extension accepted; E=38 selected |
| D-60 | 15 Nov 2026 | Final research results accepted; 10/10 final runs complete |
| D-61 | 15 Nov 2026 | One-time test evaluation authorised and completed |
| **D-62** | **15 Nov 2026** | **Formal Vanilla RNN study completion and closure accepted by the user** |
| D-63 | 15 Nov 2026 | Extended analysis and correction to M6-HC1 accepted |

D-62 is the formal closure decision. D-63 adds analysis to the closed study; it does not reopen training or test evaluation.

---

## 11. Final completion checklist

| Deliverable | Status |
|---|---|
| Research agenda and hypotheses | **Completed / accepted** |
| Architecture specification | **Approved** |
| Training specification | **Approved** |
| Diagnostics specification | **Approved** |
| Integrated implementation contract and plan | **Approved** |
| Vanilla RNN implementation | **Completed / accepted** |
| Independent specification-derived verification | **Accepted; 48/48 passed** |
| Mutation testing | **Accepted; 38/38 caught** |
| Native test outcomes | **Recorded as passed** |
| Probe and frozen study conditions | **Completed** |
| Initialisation gate | **Completed** |
| Learning-rate pilot | **Completed** |
| Epoch-budget selection | **Completed** |
| Ten final research runs | **Completed; 10/10** |
| Run verification | **Reported clean for all 25 runs** |
| Predeclared research analysis | **Completed / accepted** |
| One-time protected test evaluation | **Completed** |
| Extended analysis | **Completed / accepted** |
| Findings, limitations and open questions | **Documented** |
| Formal user-approved closure | **Completed — D-62** |

---

## 12. Final status statement

**Model 6 — Vanilla RNN is COMPLETED and FORMALLY CLOSED.** The architecture, implementation, verification, controlled study, one-time test evaluation and accepted analyses are complete. The final performance and mechanistic findings are recorded above, together with the limitations and unresolved questions that constrain their interpretation.

Any new Vanilla RNN experiment, or any change to the frozen study infrastructure, would require a new explicit decision and a separately recorded protocol.

---

## Verification and provenance

This report was first written from a reconstruction of the project's research record. Before publication, every number in it was checked against the study's own files, in this folder:

- **Checked and matching:** dev, test and train NLL and the best epochs (`test_evaluation/result.json`, `analysis/summary.json`); the W spectrum, hidden-state saturation and tanh derivatives, the attenuation rates, the full BOS-restart table with its seed and name-clustered standard errors, the EOS/letter split, the long-distance share Π̂(Δ ≥ 3), κ, the learning curves and the generation statistics (`analysis/summary.json`, `analysis/extended_summary.json`); the steps per epoch and the loss constant C (recomputed from the data).
- **Corrected in this check:** the split-half cosine range after step 128 (§7.6) and the Mac/Linux agreement (§5). Both were slightly misquoted; the conclusions are unchanged.
- **Recorded only:** the test-suite counts, the 48/48 independent verification and the 38/38 mutation result (§5). No standalone logs of these were kept, so they cannot be checked from the published files.
- **Integrity:** all 42 code, specification and utility files pinned in `study_conditions.json` still match their recorded SHA-256.

**Decision identifiers.** The report cites decisions D-38 to D-63. They are rows of the project's decision table, which records each decision, its date and its rationale before the corresponding result was produced. The table is published as `DECISIONS.md` at the repository root.

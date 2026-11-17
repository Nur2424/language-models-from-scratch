# Model 6 — Vanilla RNN: Training Specification (Stage 4)

## Status (current, v0.3)

1. **DRAFT for review. Not approved and not installed in the repository.** It does not authorise implementation, benchmarks, training, pilots or experiments.
2. **Governs:**
   - the reference training procedure of Model 6 in `models/recurrent/rnn_training.py` (M6-TRN17);
   - the training-side obligations towards the approved diagnostics;
   - the pilot that fixes the learning rate and the training horizon.
3. **Depends on** (both binding; where this draft and they disagree, they govern):
   - `models/recurrent/rnn_spec.md` v0.2, **APPROVED** (D-44; file SHA-256 `3961afdd…97f1`);
   - `models/recurrent/rnn_diagnostics_spec.md` v0.3, **APPROVED** (D-45; SHA-256 `271d4bf8…1bb2`). **Diagnostic definitions live only there.** This document references them and never redefines them.
4. **Governing decisions:** D-38 – D-45; in particular D-40 (provisional training direction M6-TRN1 – TRN9) and D-41 (provisional initialisation). Record SHA-256 `57033788…102b`.
5. **Resolves:** the 11 Stage 4 items of `rnn_spec.md` §18.4: M6-TRN10 – TRN16, TRN18, TRN19, M6-INI6 (gate), M6-EXP3. It also confirms or amends D-40 (M6-TRN1 – TRN9).
6. **Test split:** protected. Nothing here builds or evaluates Model 6 test data.
7. **Proposed location:** `models/recurrent/rnn_training_spec.md` (not created).

**Version:** 0.3 (2026-11-14)

### Labels

**[T]** mathematics; **[I]** implementation requirement; **[O]** observation from existing records; **[E]** estimate, not a measurement; **PROVISIONAL (D-40/D-41)**; **APPROVED (D-44/D-45)**; **PROPOSED**; **PENDING (Later)**. Each decision states whether it follows an **established project convention** or is a **new recommendation**.

### Amendment log

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-11-14 | Initial draft |
| v0.2 | 2026-11-14 | After the Models 3–5 reconstruction. §4 mathematics corrected (fixed-θ and marginal statements); 30-epoch learning-rate pilot; ln 27 divergence rule; near-tie rule; hybrid horizon; per-step parameter check; Git guard |
| v0.3 | 2026-11-14 | Made consistent with the approved Stage 5 (D-45). (1) Diagnostic hook moved **after** `optimizer.step()` and the post-step check; the epoch-end hook placed after evaluation and before checkpoint writes (v0.2 placed it before the step: a contradiction). (2) Category D (diverged) now **ends the run**, so that "failed run" in Stage 5 O3 has one meaning. (3) Resume truncation of diagnostic records specified. (4) Pilots run **no** research diagnostics. (5) One parameter-hash definition shared with Stage 5. (6) Per-position dev NLL by position and Stage 5 length stratum. (7) Full W spectrum at every epoch. (8) Order of gates: probe construction → thread benchmark → diagnostic timing → initialisation gate → pilots. (9) Extension-run instability rule. (10) Post-hoc best-epoch spectral measurement specified as read-only. (11) References to Stage 5 items updated from PENDING to APPROVED (D-45) |

---

## 1. Notation and data facts

| Symbol | Meaning | Value (training split) |
|---|---|---|
| N | Training names: rows of the training `NextTokenSequences` | 25,640 (record §5.3) |
| M | Training targets Σ_β n_β | 182,666 (record §5.4) |
| n̄ | M/N | 7.124259 |
| B | Names per batch | 32 |
| **S** | Steps per epoch: ⌈N/B⌉ | **802 = 801 full batches + 1 batch of r = 8 names** |
| E | `max_epochs`, fixed by §7.4 | — |
| ℓ_i, L_data | Per-target NLL; (1/M) Σ_i ℓ_i | — |
| N_dev, M_dev | Development names and targets | 3,196; 22,801 |

`train_rnn` recomputes N, M, S and r from the data and records them. An implementation must refuse to train if they differ from the values recorded in the approved protocol. This guards against silently using a different split.

---

## 2. Sequence preparation (M6-TRN2, M6-TRN3)

**Rule.**
1. `load_names` → `split_names` with the shared data section (seed 42, `duplicate_policy: group`, fractions 0.8 / 0.1 / 0.1).
2. `Vocabulary.from_names(splits.train)`; `check_vocab_coverage` on dev.
3. `make_next_token_sequences` for train and dev, once per run, on CPU, `max_len=None`.
4. **Test sequences are never built.**

**Per batch:** trim to T_b = max_{β∈b} n_β, read from the CPU copy of `lengths`.

**No length bucketing:** it would make batches non-uniform subsets and remove the marginal unbiasedness of §4.3.

**Status:** established convention (record §5.3 – 5.4; `mlp_spec.md` §4.6); trimming PROVISIONAL (D-40), confirmed here.

**Rationale [T]:** valid logits do not depend on padding (`rnn_spec.md` §4.4). Trimming is deterministic; it may change last-bit rounding compared with untrimmed runs (`rnn_spec.md` §13.1 F).

---

## 3. Batching (M6-TRN1, M6-TRN11)

**Rule.**
- `EpochSampler(N, batch_size=32, seed=derive_seed(seed, "data"))`.
- Epoch e (0-based) uses `sampler.batches(e)`: a permutation determined by (seed, e, N).
- Every name appears once per epoch: 801 batches of 32 names, then **one batch of 8 names, which is kept**.

**Status:** established convention (record §5.5; `utils/batching.py`; D-40).

**Difference from Models 3–5:** the unit sampled is a name (2 – 16 targets), not a one-target window. This is why §4 is needed.

---

## 4. Loss normalisation (M6-TRN4, M6-TRN10, M6-TRN11)

### 4.1 Rule (PROPOSED)

    L_b(θ) = (1/C) Σ_{β∈b} Σ_{t≤n_β} ℓ_{β,t}(θ)  =  token_nll(model(x), y, reduction="sum") / C
    C = B · M / N = 730,664 / 3,205 = 227.976287…

- C is computed once, in float64, from the training data and `batch_size`.
- It is the same for every step, the final batch included.
- It is recorded and checked on resume.
- It is not a configuration field.

### 4.2 Implications [T]

**(a) Maximum likelihood.**
- Σ_t log p(y_t | prefix) = log P_θ(name), so Σ_i ℓ_i is the negative log-likelihood of the training names.
- Option B is that objective up to a constant.
- A per-name mean is **not** a likelihood.

**(b) Complete batches, at fixed θ.**
- Under a uniform permutation each full batch is marginally a uniform B-subset of the names, so P(β ∈ b) = B/N.
- Therefore E[∇L_b(θ)] = (B/N)(M/C)∇L_data(θ) = ∇L_data(θ).
- This holds at parameters fixed independently of the permutation.

**(c) Final partial batch.**
- r = 8 names, a uniform 8-subset, so E[∇L_{b_S}(θ)] = (r/B)∇L_data(θ).
- Per-target weight stays 1/C. The final step is a quarter step **in proportion to its content**. This is not a bias.
- With mean reduction over the batch's own targets, its ≈ 57 targets [E] would each get ≈ 4× the weight of a target in a full batch.

**(d) Unequal name lengths.**
- Every target has weight 1/C, whatever the length of its name.
- Under a per-batch token mean, weight is 1/m_b. Since m_b includes the name's own length (≈ n + 220.85 in a full batch), targets of a 16-target name receive ≈ 6% less weight than those of a 2-target name (first-order).
- That bias would fall on the late targets that M6-RQB and M6-RQC study.

**(e) Epoch-level weighting, at fixed θ.**
- Σ_{b∈epoch} L_b(θ) = (N/B)·L_data(θ), exactly, for any permutation.
- This is a statement about weights.

**(f) Parameters changing between steps.**
- θ_j depends on the batches already drawn. Given that history, batch j is a uniform subset of the **unseen** names, so E[∇L_{b_j}(θ_j) | history] is the gradient over those names, **not** ∇L_data(θ_j).
- One epoch of SGD differs from one full-batch step of size (N/B)η by O(η²) terms. This is the accepted behaviour of `EpochSampler` (random reshuffling) and holds equally for Models 3–5.
- **Correction:** `mlp_spec.md` §4.6 states per-step unbiasedness without these qualifications. The correct statement is (b).

**(g) Learning-rate units.**
- The expected full-batch step at fixed θ is η∇L_data, as for Models 3–5 with mean reduction over 32 one-target windows.
- So η means the same expected step per unit gradient. It does **not** imply the same dynamics: noise, curvature, steps per epoch and the recurrent gradient structure all differ.

**(h) Diagnostics.**
- G3 uses its own unnormalised weighted probe loss, so its shares are invariant to C (`rnn_diagnostics_spec.md` §4.4).
- M6-O1 gradient norms carry a constant 1/C and are comparable within the study.

### 4.3 Status, precedent and difference

- Option B: PROVISIONAL (D-40), confirmed here.
- C = B·M/N and weight 1/C for the final batch: **new** (PROPOSED).
- Models 3–5 use mean reduction over the actual batch (`mlp.py` line 681, `batchnorm_mlp.py` line 840, `wavenet.py` line 972). That is harmless for one-target windows, biased for names.
- **Rejected:** C = 228 as a literal (0.01% step error, origin unstated); C = B (η units 7.12× larger).

---

## 5. Optimiser, step order and hooks (M6-TRN5, M6-TRN20)

**Optimiser.** `torch.optim.SGD(params, lr=η, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False)`; full BPTT through autograd (M6-TRN6); empty optimiser state.

**Order within each step [I] (PROPOSED):**
1. `idx` ← next sampler batch (CPU). T_b and m_b come from `lengths_cpu[idx]`.
2. `x, y` ← `inputs[idx_dev, :T_b]`, `targets[idx_dev, :T_b]`.
3. `optimizer.zero_grad(set_to_none=True)`.
4. `logits = model(x)`.
5. `loss_sum = token_nll(logits, y, reduction="sum")`.
6. **If `loss_sum` is not finite → Category A** (§12).
7. `(loss_sum / C).backward()`.
8. Per-parameter gradient L2 norms (U, W, b, V, c) and the global norm, accumulated in float64 on the host. **If not finite → Category A**, naming the parameters via `nonfinite_gradients`.
9. `optimizer.step()`.
10. **Post-step parameter finiteness** (one fused reduction). **If not finite → Category A.**
11. `running.update_from_sum(float(loss_sum), m_b)`.
12. Append the M6-O1 step record (§12.2).
13. **Step diagnostic hook:** `plan.after_step(model, sigma=completed_steps, epoch=e)` if a plan is given (`rnn_diagnostics_spec.md` §5.2). It is called after the step, after the post-step check, with σ = steps completed so far.

**At the end of epoch e [I] (PROPOSED):**
1. Parameter finiteness (implied by step 10; asserted again).
2. Full train and dev NLL (§8). A non-finite NLL → Category A.
3. Divergence test (§12.1, Category D).
4. M6-O2 records (§8.4), including the parameter hash (§10.4).
5. **Epoch-end diagnostic hook:** `plan.at_epoch_end(model, sigma, completed_epochs=e+1)`, after evaluation and **before any checkpoint write** (`rnn_diagnostics_spec.md` §5.2).
6. Selection and checkpoints, in the order of §10.3.
7. If Category D: end the run (§12.1) after writing the history row and `checkpoint_last.pt`.

**Before the first step:**
1. Epoch-0 evaluation row (§8.3), O2 at k = 0, parameter hash.
2. Initial diagnostic point: `plan.at_start(model)` (σ = 0, P₀).
3. Then step 1.

**Status.**
- SGD settings: PROVISIONAL (D-40) and convention (`mlp.py` lines 661–704; record §8.4).
- Step order mirrors `mlp_spec.md` §4.6.
- **New:** every-step gradient norms; the post-step check; the hook positions required by D-45.

**Hook contract.** The plan receives the model read-only and returns nothing. Training never reads anything the plan produces. Isolation requirements and tests I1 – I8 are those of `rnn_diagnostics_spec.md` §11.

---

## 6. Learning rate and schedule (M6-TRN12, M6-TRN13)

**Schedule: constant** (PROPOSED). M6-RQC tracks changes over training, and a schedule would add an external intervention. Precedent: constant learning rate in every executed run (record §8.4; `lr_pilot/conditions.json` and `final/conditions.json`, `lr_decay_epoch: None`).

**Value: selected by pilot P (§7.3)** on the grid {0.03, 0.1, 0.3, 1.0} with the precedent boundary extension (×≈3, at most two rounds; 0.01 → 0.003 downward, 3.0 → 10.0 upward). No planning value is used for any decision.

Rationale:
- Under C = B·M/N the units match the WaveNet study, where 0.03 was the interior optimum for four ≈ 22k-parameter arms (D-22).
- The RNN's gradient noise, curvature and recurrent accumulation make the optimum unpredictable. The grid spans ×33, and extension reaches ×1,000.

---

## 7. Pilot, horizon and stopping (M6-TRN14, M6-EXP3)

### 7.1 Stopping (convention)

Exactly `max_epochs` epochs, unless a run ends by §12.1. No patience-based early stopping; best checkpoint chosen afterwards (record §5.7).

### 7.2 Minimum observation horizon E_min = 30 (PROPOSED)

- **Reason:** M6-RQC needs a training timeline that a plateau rule may cut short.
- **Evidence:** under the plateau-selected budget of 26 epochs, 10 of 40 WaveNet final runs had their best dev NLL at the last epoch (A 0, B 3, C 3, D 4), and mean dev NLL was still falling by 0.003 – 0.005 nats over the last five epochs (`final/analysis/dev/per_seed_dev.csv`, `learning_curves.csv`) [O].
- **Why 30:** the precedent cap (`wavenet_spec.md` §16.5); above every WaveNet selected epoch (17 – 26).

### 7.3 Pilot P — learning rate (PROPOSED)

| Item | Rule | Basis |
|---|---|---|
| Prerequisites | All gates of §15, steps 1 – 6 | — |
| Candidates, seeds | Grid of §6; seeds 100, 101, 102 | Convention (record §8.5; D-20) |
| Horizon | 30 epochs, constant η | **New** (precedent 10) |
| Recorded | Epoch-0 row and epochs 1 – 30: running, train and dev NLL; O1 every step; O2 every epoch; outcome category | Convention plus O1/O2 |
| Diagnostics | **No research diagnostics** (no plan; `diagnostics=None`) | Recommendation of `rnn_diagnostics_spec.md` §5.1, adopted |
| Eligibility | Ineligible if any seed ends by Category A or D (§12.1) | A: D-16; D: new |
| Primary rule | η̂ = the eligible candidate with the lowest mean full-dev NLL at epoch 30 over the seeds | Convention form |
| Boundary extension | If η̂ is at a boundary of the evaluated grid: extend ×≈3 that way, same seeds and horizon, at most two rounds; still at a boundary → inconclusive; an ineligible point counts as evaluated | Convention (A2) |
| Near-tie rule | After the grid is final: select the **smallest** eligible η with D̄_η(30) ≤ D̄_η̂(30) + max(10⁻³, SE_η̂(30)), SE = sample SD/√3 | **New**; ε form from D-23 |
| Human review | Flags (§12.3) reviewed before acceptance. The review checks validity only: it may declare the pilot inconclusive with written reasons; it may not choose another η | D-16, D-22, with scope stated |
| Data | Train and dev; test never built | Convention |

**Why 30 and not 10.**
- *Stability over the horizon of use:* without clipping, gradient growth can begin only after W has changed (M6-E2 bounds it at initialisation). Every final run uses at least 30 epochs, so the pilot must cover 30.
- *Ranking at the horizon of use:* ranking at 10 epochs "may favour learning rates that improve quickly early" (record §8.5.1, §12).
- Steps per epoch is **not** used as the argument.

**Why the near-tie rule.** The WaveNet arm-D near-tie (0.03 against 0.1) was settled after seeing results (D-22); this predeclares the resolution, and the smaller rate is the conservative choice without clipping. **Weakness:** SE from three seeds is imprecise, which the D-23 floor of 10⁻³ partly addresses.

**Why ln 27.** It is the loss of the initialised model (M6-E4), fixed by the model rather than chosen. The precedent recorded finite blow-ups (WaveNet arm A at η = 1.0, dev NLL 14.50) only descriptively.

### 7.4 Training horizon E: hybrid rule (PROPOSED, M6-TRN14)

From the P runs at the selected η (seeds 100 – 102, epochs 1 – 30), compute e* by `wavenet_spec.md` §16.5 Steps 2 – 3:
- D̄(e), SD(e), SE(e) = SD/√3;
- ε(e) = max(10⁻³, SE(e));
- e* = the earliest e with D̄(e) ≤ min_e′ D̄(e′) + ε(e).

Then:
1. If e*₃₀ < 30: **E = 30.**
2. If e*₃₀ = 30: **extension run X.** Selected η, seeds 100 – 102, 60 epochs, from scratch, no research diagnostics.
   - **Integrity:** epochs 1 – 30 must be bit-identical (parameter hashes, §10.4) to the P runs; a mismatch stops X for review.
   - **Stability:** if any X seed ends by Category A or D, no budget is selected; record inconclusive; human decision.
   - Otherwise compute e*₆₀:
     - if e*₆₀ < 60: **E = max(30, e*₆₀)**;
     - if e*₆₀ = 60: **inconclusive**, human decision.
3. An SE-derived ε at e* requires explicit human review (as D-32).

**Alternatives:**
- plateau-only (`max_epochs` = e*): risks cutting the RQC timeline;
- fixed 60: twice the cost, mostly post-plateau epochs.

Final checkpoints are still selected by best dev NLL (§8), so extra epochs never worsen the reported model.

### 7.5 Pilot versus final protocol

- Pilots set **only** η and E. They may not change the architecture, initialisation, loss or hypotheses.
- Pilot results are FINDING (pilot) evidence for hyperparameter selection only.
- Final runs (M6-EXP1, EXP2, Later) use final seeds disjoint from 100 – 102, the full Stage 5 plan and a frozen protocol decision recorded before the first final run.

### 7.6 Cost (not measured)

- P: 4 × 3 × 30 = 360 run-epochs, plus 90 per extension round (at most 2).
- X: 180 run-epochs, only if needed.
- **Reference only:** WaveNet-study epochs took a median 3.3 – 7.9 s (LR pilot) and 4.6 – 12.3 s with the per-forward monitor (CPU, 1 thread). The RNN's Python loop over time steps may differ. The benchmark (§11) measures it.

---

## 8. Evaluation, selection and per-epoch records (M6-TRN9, M6-TRN15, M6-TRN21)

### 8.1 Three NLLs per epoch (convention, record §5.6)

1. Running train NLL: from `loss_sum` and m_b per step.
2. Full train NLL.
3. Full dev NLL.

All token-weighted via `NLLAccumulator`, never a mean of batch means.

### 8.2 `evaluate_nll` (PROPOSED)

- Names in a fixed **length-sorted order** (stable by n_β, ties by row index).
- Chunks of `eval_batch_size` = **1,024 names**, each trimmed to its longest name.
- Runs under `torch.no_grad()`; the model's mode is restored.
- [T] The token sum does not depend on order or chunking; sorting changes only rounding (`rnn_spec.md` §13.1 B).
- `eval_batch_size` is fixed within the study and part of the resume check.

### 8.3 Epoch-0 row (PROPOSED, M6-TRN21)

- Before any step: full train and dev NLL, O2, the parameter hash; running NLL = `null`; `completed_epochs = 0`.
- **Never selectable.**
- Required by O2 ("including epoch 0"), and the reference for the divergence rule.
- **New:** the precedent histories begin at epoch 1 (`mlp.py` line 720).

### 8.4 M6-O2 records (content fixed by `rnn_spec.md` §12.7 and `rnn_diagnostics_spec.md` §4.6)

Per completed epoch k = 0 … E, in the history row:
- the three NLLs;
- **per-position dev NLL** as sums and counts per t = 1 … 16 and per Stage 5 length stratum (n ∈ {2–5, 6–7, 8–9, 10–16}), computed by `evaluate_position_nll` in the same pass structure as §8.2;
- L2 norms of U, W, b, V, c;
- **all 128 singular values of W** (float64, from a float64 CPU copy of W);
- spectral radius of W;
- ‖WᵀW/g² − I‖_F with g = `recurrent_gain`;
- the parameter hash (§10.4);
- the divergence flag;
- seconds.

These are read-only computations involving no RNG.

### 8.5 Selection (convention)

- Best = the completed epoch k ≥ 1 with the lowest full dev NLL, by strict `<` on the float64 value; ties keep the earliest (`mlp.py` line 717; record §5.7).
- `model_best.pt` is written on strict improvement.
- The test split is never used.

---

## 9. Initialisation gate (M6-INI6, M6-INI7)

**Values.** g = 1.0 and s² = 0.5 remain provisional (D-41). Nothing changes them automatically.

**Gate M6-INI7 (PROPOSED).** No training step of a pilot (P, X) or final run is taken until all of the following are recorded:
1. The Model 6 implementation is accepted (UMIS §20).
2. The probe is constructed under its own authorisation (`rnn_diagnostics_spec.md` §3.1), with its SHA-256, N_ℓ and the F₈ count recorded.
3. The `rnn_spec.md` §7.2 procedure has been run under authorisation, on that probe, with initialisation-only seeds 1000 – 1009 (D-45, M6-DGN10), and its report recorded as [O].
4. A recorded decision either **retains** g = 1 and s² = 0.5 (confirming D-41's values for training), or **changes** them under the §7.2 change rule.

**Exempt:** unit tests on fixtures, and the timing benchmark of §11, whose parameters and results are discarded and never used for any decision except the thread count.

**Sufficiency.**
- At g = 1 the Jacobian products cannot grow at initialisation (M6-E2). Instability can arise only as W changes, and that is handled by §12.
- No further initialisation precaution is needed.
- Orthogonality is verified by `rnn_spec.md` §13.2 I3.

---

## 10. Reproducibility, checkpoints, resume and records (M6-TRN16, M6-TRN23)

### 10.1 Random streams (convention, record §5.5)

- `derive_seed(s, "init")` → the constructor's generator;
- `derive_seed(s, "data")` → `EpochSampler`;
- `derive_seed(s, "generation")` → sampling.

`train_rnn` and the diagnostic plan consume no global RNG. The probe uses its own stream `derive_seed(42, "m6-probe")` (D-45), drawn once at construction and never during training.

### 10.2 Determinism

CPU, float32, fixed software stack and fixed thread counts (§11): runs are bit-identical. This is the basis of resume equivalence and of the §7.4 integrity check. MPS and cross-version runs make no guarantee.

### 10.3 Checkpoints (convention, `mlp_spec.md` §14)

- `checkpoint_last.pt` (resume) at every epoch boundary;
- `model_best.pt` on strict improvement;
- `model_final.pt` after the last completed epoch, including a run ended by Category D.

**Write order at the end of epoch e:**
1. evaluate;
2. O2;
3. the epoch-end diagnostic hook;
4. `model_best.pt` (if improved);
5. history row;
6. `checkpoint_last.pt` with `epoch = e + 1`, which is the **commit point**;
7. after the last epoch: `model_final.pt`.

Checkpoints store the configuration of §10.5 and `extra = {"best_epoch", "sampler", "base_seed", "param_sha256"}`.

### 10.4 Parameter hash (one definition, shared with Stage 5)

`param_sha256` = SHA-256 over the float32 bytes, C-contiguous, of U, W, b, V, c in that order. It is identical to `rnn_diagnostics_spec.md` §5.3. It is recorded:
- in every history row;
- in every checkpoint's `extra`;
- by the plan at every diagnostic point.

### 10.5 Recorded configuration (PROPOSED)

    {"experiment": <resolved ExperimentConfig>, "model": asdict(RNNConfig),
     "training": asdict(RNNTrainingConfig),
     "derived": {"loss_normalizer_C": C, "N": N, "M": M, "N_dev": N_dev, "M_dev": M_dev,
                 "steps_per_epoch": S, "final_batch_names": r,
                 "intra_op_threads": k, "inter_op_threads": k′, "dtype": "float32", "device": "cpu"},
     "diagnostics": null | {"probe_sha256": …, "schema": "m6_diagnostics/3"}}

Run metadata adds `environment_info()`, the torch version and SHA-256 checksums of the code files used.

### 10.6 Resume (PROPOSED; mechanism as convention)

Resume is at epoch boundaries only. Steps:
1. Load `checkpoint_last.pt`.
2. Compatibility check. Any difference raises `IncompatibleCheckpointError`. Fields checked: `RNNConfig`; `lr`, `max_epochs`, `batch_size`, `eval_batch_size`; C; N; M; S; sampler and base seeds; vocabulary; dtype; device type; intra-op and inter-op thread counts; diagnostics presence and probe SHA-256.
3. `restore_training_state`.
4. `EpochSampler.from_state_dict`.
5. **Truncate** to the committed state K = `epoch` in the checkpoint, σ_K = its `step`:
   - `history.jsonl` rows with `completed_epochs ≤ K`;
   - step records with step ≤ σ_K;
   - **diagnostic records:** the plan's `truncate(sigma=σ_K, completed_epochs=K)` keeps points with σ ≤ σ_K whose kind is not an uncommitted epoch end, and removes or overwrites any later point and its `raw/` file (`rnn_diagnostics_spec.md` §5.2).
6. Continue to the **original** `max_epochs`.

Consequences:
- An epoch interrupted after its epoch-end diagnostic point but before `checkpoint_last.pt` is re-run, and its point is rewritten bit-identically on CPU.
- **Thread counts in the check:** a resume under different counts would continue validly but would no longer be bit-identical to an uninterrupted run.

### 10.7 Post-hoc best-epoch spectral measurement (Stage 5 E_sel)

**Required only if the best epoch is neither epoch 1 nor E**, since those epoch ends are already spectral points.

**Procedure (read-only):**
1. Load `model_best.pt` with `weights_only=True`.
2. Verify its `param_sha256` equals both the checkpoint's `extra` value and the epoch-end diagnostic point of that epoch.
3. Compute the spectral measurement on a float64 CPU copy.
4. Append a point of kind `posthoc_best` to the run's **diagnostic** records.

**The checkpoint, `history.jsonl`, step records and all other training records are not modified.** If the hashes disagree, it refuses (`rnn_diagnostics_spec.md` §5.3).

### 10.8 Git (D-15)

`RunManager.create` calls `utils.run_manager.git_info()`, which starts `git` (`utils/run_manager.py` lines 91–106, 205). Every Model 6 runner and **every Model 6 test** that creates a `RunManager` must first install the project's Git guard, as `training/wavenet_study/study_common.py` `disable_git` and `tests/training/wavenet_study/conftest.py` do.

**Recorded gap, not fixed here:** `tests/models/feedforward/test_mlp.py` (line 114), `test_batchnorm_mlp.py` (line 129), `tests/models/hierarchical/test_wavenet.py` (line 135) and `tests/utils/test_run_manager.py` call `RunManager.create` or `git_info` without the guard. A fix needs its own decision.

---

## 11. Device, precision, threads and timing (M6-TRN18, M6-TRN19, M6-TRN22)

**Rule (PROPOSED).** Pilots and final runs: CPU, float32. Intra-op and inter-op thread counts are fixed for the whole study and recorded.

**Thread benchmark (PROPOSED; precedent D-24).**
- Run once after implementation acceptance and probe construction, before the diagnostic timing and before the initialisation gate.
- Workload: one training epoch plus its evaluation, with seed 999, η = 0.1, no research diagnostics, at intra-op counts 1, 2, 4 and 8 (inter-op at the PyTorch default, recorded).
- Wall time per configuration is recorded.
- Its parameters and results are discarded. The thread count is decided by a recorded decision (recommendation: the fastest), **not automatically**.

**Evidence:** in the WaveNet benchmark one thread was fastest (estimated round-0 time 2,444 s against 2,665, 2,716 and 3,091 s; `lr_pilot/benchmark/*.json`). The final runs used intra-op 1, inter-op 8 (`final/conditions.json`).

**Diagnostic timing (`rnn_diagnostics_spec.md` §7.3, M6-DGN13):** measured after the thread decision, with the chosen counts.

**MPS (M6-TRN19):** not used for pilots or final runs (no determinism guarantee; launch overhead on tiny kernels; an uncontrolled second device). Forward-level checks only, under `rnn_spec.md` §13.1 C. **CUDA:** not available.

**Tolerances:** as approved in `rnn_spec.md` §13.1 (bitwise for identical CPU runs and resume; A = 1e‑9 in float64; B = C = 1e‑5 in float32).

---

## 12. Gradient handling, finiteness and failure handling (M6-TRN7, M6-TRN8, M6-TRN24)

**No clipping** (PROVISIONAL, D-40; confirmed here). No clipping, rescaling or loss scaling; any change needs a new decision (UMIS §13).
- **Rationale:** gradient growth is part of what Model 6 studies (M6-RQA, HA2), and clipping would hide it.
- It is not recommended merely because the model is recurrent: sequences are at most 16 steps, and products are bounded at initialisation (M6-E2).

### 12.1 Failure handling (Stage 4: what training does)

| Category | Definition | Checked | Action in `train_rnn` |
|---|---|---|---|
| **A — mechanical** | Non-finite `loss_sum`, gradient norm, parameter after a step, or evaluated NLL | Every step / evaluation | Raise `FloatingPointError`. The run ends with status `failed_mechanical`. `checkpoint_last.pt` from the previous boundary is untouched. Nothing of the failing epoch is committed |
| **D — diverged** | Finite, but full-train NLL > ln 27 at a completed epoch k ≥ 1 | Every epoch end | Record the flag in the history row; complete that epoch's commit (§10.3); write `model_final.pt`; **end the run** with status `failed_diverged` |

No automatic retry, clipping or η change. A retry requires explicit authorisation and a written reason (D-18).

**Why D ends the run:** a model worse than its initial state has no research use in that run. Ending it gives Stage 5's "failed run" (O3) a single meaning: a run ended by Stage 4 failure handling.

### 12.2 Per-step record (M6-O1)

- epoch, cumulative step, batch index;
- names, m_b, T_b;
- token-mean batch NLL `loss_sum/m_b` and the optimised loss `loss_sum/C`;
- pre-step global gradient norm and the five per-parameter gradient norms;
- finite flag.

Every step, no sampling. One host synchronisation per step [E: negligible]. File format follows the run layout of §10; values are float64.

### 12.3 Pilot flags (Stage 4; for the pilot review only)

Threshold-free, as D-16:
- full-dev NLL rose between consecutive epochs (and "repeated" if in ≥ 2 epochs);
- running train NLL rose;
- dev NLL at epoch 30 above the seed's best.

### 12.4 Relation to Stage 5 O3

Stage 5 **describes** runs (`rnn_diagnostics_spec.md` §4.6):
- runs ended by §12.1 are "failed runs", excluded from research aggregates;
- epoch events and the per-epoch gradient-norm profile (max, median, max/median of the O1 global norm) are descriptive, with no spike multiple (D-45).

Stage 4 **acts** only by §12.1. Neither stage uses the other's categories to change its own behaviour.

---

## 13. Training interface: `models/recurrent/rnn_training.py` (M6-TRN17, M6-TRN26)

`rnn.py` holds architecture and model behaviour only (D-44). `rnn_training.py` uses `forward` only. The diagnostic plan lives in `rnn_diagnostics.py` (D-44, D-45).

**`RNNTrainingConfig`** (frozen dataclass):

| Field | Type | Default | Validation | Value |
|---|---|---|---|---|
| `lr` | float | required | finite, > 0 | from pilot P |
| `max_epochs` | int | required | ≥ 1 | from §7.4 |
| `batch_size` | int | 32 | ≥ 1 | PROVISIONAL (D-40) |
| `eval_batch_size` | int | 1,024 | ≥ 1 | PROPOSED |

**Fixed in code:** SGD settings; C = B·M/N; constant schedule; no clipping; Category D ends the run.

**Functions (PROPOSED):**

```python
def loss_normalizer(training: RNNTrainingConfig, train: NextTokenSequences) -> float  # B·M/N
def evaluate_nll(model, sequences, *, batch_size: int) -> NLLAccumulator
def evaluate_position_nll(model, sequences, *, batch_size: int) -> PositionNLL   # sums, counts per (t, stratum)
def train_rnn(model, train, dev, *, training: RNNTrainingConfig, seed: int,
              run: RunManager | None = None, resume: bool = False,
              config: Mapping[str, Any] | None = None,
              diagnostics: "RNNDiagnosticsPlan | None" = None) -> RNNTrainingResult
```

- `train_rnn` calls the plan exactly at the three hook points of §5 and on resume (§10.6), and nowhere else.
- `RNNTrainingResult`: history, best epoch, best dev NLL, final status (`completed`, `failed_mechanical`, `failed_diverged`), C.

**Not in `rnn_training.py`** (UMIS §12): pilots, seed loops, frozen conditions, summaries, plotting, Git handling. Orchestration belongs to a future `training/rnn_study/` package, created by a separate decision.

---

## 14. Training-procedure tests (PROPOSED; additions to `rnn_spec.md` §13.2 T1 – T8)

`tests/models/recurrent/test_rnn_training.py`: CPU, small fixtures, Git guard installed.

| # | Test | Criterion |
|---|---|---|
| T9 | `loss_normalizer` = B·M/N; recorded | exact |
| T10 | Static weighting at fixed parameters: Σ_b L_b·C equals the total token NLL; Σ_b ∇L_b equals (N/B)∇L_data | `rnn_spec.md` §13.1 A |
| T11 | Final batch: r names; loss `loss_sum/C` | exact |
| T12 | Injected NaN in loss, gradient, post-step parameter and evaluation: Category A at the right point; for loss and gradient, parameters bitwise unchanged; status `failed_mechanical` | bitwise / exact |
| T13 | Injected divergence: flag set exactly when train NLL > ln 27 at k ≥ 1; epoch committed; `model_final.pt` written; run ends with status `failed_diverged` | exact |
| T14 | History rows k = 0 … E; k = 0 never selected; strict `<` keeps the earliest of an injected tie | exact |
| T15 | Resume at an epoch boundary equals an uninterrupted run: parameters, history, step records, parameter hashes, **and diagnostic records** (with a plan) | bitwise |
| T16 | Resume with a crash injected after the epoch-end diagnostic point and before `checkpoint_last.pt`: the point is rewritten once, not duplicated | exact |
| T17 | Resume rejects each changed field of §10.6 | error |
| T18 | Hook order: the plan observes σ = completed steps after the update; the epoch-end call precedes the checkpoint write; the parameter hashes at the hooks equal those in the history and checkpoints | exact |
| T19 | Length-sorted chunked evaluation equals unsorted single-chunk evaluation; position NLL totals equal the full NLL | A (float64), B (float32) |
| T20 | Step records per epoch: count = ⌈N/B⌉, Σ m_b = M, Σ names = N | exact |
| T21 | Trimmed and untrimmed batches give the same trajectory in float64 | A |
| T22 | `train_rnn` leaves the global RNG unchanged and never starts `git` (the guard raises if attempted) | exact |
| T23 | Post-hoc best spectral measurement: refuses a hash mismatch; leaves checkpoint and training-record bytes unchanged | exact |
| T24 | Training with `diagnostics=None` and with a full plan: bitwise identical (= `rnn_diagnostics_spec.md` I1) | bitwise |

---

## 15. Gates and sequencing

Each step needs its own authorisation and record:
1. **Stage 4 approval** (this document).
2. Implementation authorisation; implementation; acceptance (`rnn_spec.md` §17; `rnn_diagnostics_spec.md` §10 – §11; §14 here).
3. **Probe construction** (`rnn_diagnostics_spec.md` §3.1): SHA-256, N_ℓ, F₈ count.
4. **Thread benchmark** (§11) and the thread-count decision.
5. **Diagnostic timing** (M6-DGN13).
6. **Initialisation statistics** (`rnn_spec.md` §7.2) and gate decision M6-INI7.
7. Pilot P; review; η decision.
8. If required, extension X; review; E decision.
9. Final-protocol decision (M6-EXP1, EXP2, failed-seed policy); final runs with the full Stage 5 plan.

---

## 16. Decision table

| ID | Decision | Proposed rule | Evidence / rationale | Status |
|---|---|---|---|---|
| M6-TRN1 | Batching | Names, B = 32, `EpochSampler` data stream | Record §5.5 | PROVISIONAL → confirm |
| M6-TRN11 | Final batch | Kept (8 names), per-target weight 1/C; S = 802 | §5.5; §4.2 c | PROPOSED |
| M6-TRN2/3 | Sequences | Built once; per-batch trim; no bucketing | §5.4; spec §4.4 | PROVISIONAL → confirm |
| M6-TRN4/10 | Loss | Token sum / C, C = B·M/N | §4.2: likelihood, length-neutral weights, fixed-θ unbiasedness | PROPOSED |
| M6-TRN5/20 | Optimiser and step order | Plain SGD; §5 order; hooks after the step and before checkpoint writes | Convention; D-45 §5.2 | PROPOSED |
| M6-TRN13 | Schedule | Constant | Convention; RQC | PROPOSED |
| M6-TRN12 | Learning rate | Pilot P: grid + extension; 30 epochs; seeds 100 – 102; A/D ineligible; near-tie → smaller η; validity-only review | Grid, rule, seeds: convention; horizon, D, near-tie: new | PROPOSED; value from P |
| M6-EXP3 | Pilot design | P and, only if needed, X; no research diagnostics | §7; D-45 §5.1 | PROPOSED |
| M6-TRN14 | Horizon | E = 30, or max(30, e*₆₀) after X; inconclusive at 60 or if X fails | WaveNet final-run evidence; RQC | PROPOSED |
| M6-TRN9 | Selection | Lowest full dev NLL, k ≥ 1, strict <, earliest | Convention | PROVISIONAL → confirm |
| M6-TRN15 | Evaluation | Token-weighted; length-sorted chunks of 1,024 names | Weighting: convention; order and size: new | PROPOSED |
| M6-TRN21 | Epoch-0 row | Recorded; never selectable | O2; §12.1 | PROPOSED |
| (O2 content) | Per-epoch records | §8.4 (per-position × stratum dev NLL; full W spectrum) | `rnn_spec.md` §12.7; D-45 | PROPOSED (implements approved content) |
| M6-INI6/7 | Initialisation | Values retained; gate §9 after probe construction | D-41; D-45 | Values provisional; gate PROPOSED |
| M6-TRN16 | Resume | Epoch boundaries; extended compatibility check; diagnostic truncation | Convention; D-45 §5.2 | PROPOSED |
| (hash) | Parameter hash | One definition (§10.4), shared with Stage 5 | D-45 §5.3 | PROPOSED |
| (post-hoc) | Best-epoch spectral | Read-only procedure §10.7 | D-45 §5.1, §5.3 | PROPOSED |
| M6-TRN23 | Records | §10.5; checksums; Git guard in runners and tests | D-15 | PROPOSED |
| M6-TRN18/22 | Device, threads | CPU float32; threads fixed; benchmark then decision | D-16 A3, D-24 | PROPOSED; value from benchmark |
| M6-TRN19 | MPS | Not for training | Record §5.5 | PROPOSED |
| M6-TRN7/8 | Clipping; norms | None; every-step norms | D-40 | PROVISIONAL → confirm |
| M6-TRN24 | Failure handling | A and D end the run; pilot flags threshold-free; separate from O3 | D-16; ln 27 from M6-E4 | PROPOSED |
| M6-TRN17/26 | Interface | §13 | D-44 | PROPOSED |
| — | Placement | Separate `rnn_training_spec.md` | — | PROPOSED |

**Later (not Stage 4):** M6-EXP1, M6-EXP2, failed-final-seed policy. **STANDING:** M6-EXP4, EXP5.

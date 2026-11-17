# Model Specification — WaveNet (hierarchical context merging)

**Status (current, v0.5):** Both variants (`wavenet`, `wavenet_batchnorm`) are **implemented and formally accepted** (Model 5; D-12 in `PROJECT_RESEARCH_RECORD.md`). All design decisions Q1–Q12 are approved (§18). The learning rate is **0.03 for each of the four study arms** (LR pilot completed; accepted 2026-11-11, D-22). The common epoch budget remains **PENDING PILOT** under §16.5; the approved operational protocol for that pilot is `training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md` (§16.5.1; D-23, D-26). The budget pilot's implementation and execution are **not yet authorised**. The v0.4 status line is preserved verbatim below the amendment log.
**Update (v0.6, 2026-11-12, D-31):** the design of the **final controlled experiment** is amended: primary endpoint test NLL at the dev-selected checkpoint, ten paired final seeds (0–9), estimation-first reporting with pre-declared comparison families and a smallest effect size of interest (§16.1, §16.3, §16.6.1). The LR pilot (§16.4) and the common epoch-budget pilot (§16.5, §16.5.1, its protocol) are **unchanged**: they select on development NLL with seeds 100–102 and never evaluate the test split. The other statements of the v0.5 status line above are as written on 2026-11-11.
**Status clarification (v0.6 addendum, 2026-11-12, D-31 addendum):** the v0.5 status line's statement that the budget pilot's "implementation and execution are **not yet authorised**" describes 2026-11-11 and is retained as historical. Since then the pilot's implementation has been accepted (D-27, D-29), and the epoch-budget pilot **has been executed** (user-reported, 2026-11-12). Its selection under §16.5 (the common `max_epochs`) and the human approval of that budget **remain pending**. No pilot result is recorded in this specification.
**Update (v0.7, 2026-11-12, D-32):** the common epoch budget is decided. Current status:
1. LR selection: completed (η = 0.03 for A–D; D-22).
2. Epoch-budget pilot: completed.
3. Common budget: **`max_epochs` = 26** for every arm, formally approved (D-32; §16.5).
4. Final controlled experiment: **not yet executed**.
5. Final-experiment configuration freeze: **pending**.
6. Final seeds: 0–9 (D-31; §16.6.1).
7. Final test evaluation: **not yet performed**.

The "pending" statements above (v0.5 status, v0.6 addendum) are retained as written for their dates.

**Update (v0.8, 2026-11-17): the final controlled experiment is complete and the study is closed.** Current status, superseding items 4, 5 and 7 of the v0.7 list:
1. Final controlled experiment: **executed** on 2026-11-12. 40 runs (arms A–D × paired seeds 0–9), all completed and independently verified.
2. Final-experiment freeze: **done** before any test access (`results/experiments/wavenet_study/final/test_freeze/freeze.json`).
3. Final test evaluation: **performed once**, 40 of 40 runs (`test_freeze/evaluation_ledger.jsonl`).
4. Results: reported in `results/experiments/wavenet_study/final/analysis/final/REPORT.md` and accepted (D-35); the study was closed on 2026-11-13 (D-36). No further final runs or test evaluations take place.
5. Acceptance criteria (§17): ticked at closure.

*Decision identifiers.* The identifiers D-xx, the references to `PROJECT_RESEARCH_RECORD.md` and the record sections cited in this document refer to the project's internal research record, which is not published. They are kept so that the history of this specification stays traceable. The published evidence for every decision is in `results/experiments/wavenet_study/` and in `training/wavenet_study/`.

**Governing standard:** `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md`
**Template:** `MODEL_SPECIFICATION_TEMPLATE.md` (its 18 sections map onto the 18 required sections below; see §18.4)
**Structural references:** `models/feedforward/mlp_spec.md` v0.3 and `models/feedforward/batchnorm_mlp_spec.md` v0.2 (both accepted). "As in Model 3/4, §x" means that section applies unchanged, with names replaced.
**Historical investigation:** Phase 1 report of 2026-11-11 (sources `old_models/04_wavenet.py`, `old_notebooks/04_wavenet.ipynb`, git commit `cf0410d`).

### Amendment log
| Version | Date | Change | Decision |
|---|---|---|---|
| v0.4 | 2026-11-11 | Q1–Q12 approved; experimental protocol §16 (LR pilot §16.4, budget procedure §16.5) | D-08 – D-11 |
| **v0.5** | 2026-11-11 | (1) Accepted learning rate 0.03 for arms A, B, C and D recorded in §7.2, §8, §16.4, §16.5 and §18.3 (D-22). (2) §16.4: operational definition of stability as applied by the LR pilot (D-16) recorded. (3) §16.5.1 added: approved operational amendments of the common epoch-budget pilot, with the canonical protocol in `training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md` (D-23). The §16.5 formulas and extension rules are **unchanged**. (4) §11: step-0 description of `log10_update_to_param_std` corrected (all parameters at step 0, not only `out.weight`). (5) §17 and §18.3 status updated. | D-22, D-23 |
| **v0.5 addendum** | 2026-11-11 | (1) Header: current status stated for v0.5 (Model 5 accepted, D-12); the v0.4 status line moved verbatim to just below this log. (2) §1.3: current scope distinguished from the v0.4 implementation-task scope (retained verbatim). (3) §20: current file status recorded; v0.4 heading and "currently empty" wording retained and marked historical; notes added to §17 and §18.2 for the same reason. (4) §16.5.1: approved interpretations I-A (G5/G6 mismatch), I-B (finiteness checks in every forward), I-C (G7 independent of G4) and the category B resolution procedure. §16.5 formulas and extension rules unchanged. | D-26 |
| **v0.6** | 2026-11-12 | Final controlled experiment only: (1) §16.1: primary and secondary endpoints (test NLL at the dev-selected checkpoint; best dev NLL). (2) §16.3: final seed identifiers 0–9. (3) New §16.6.1: ten paired seeds, estimation-first reporting (paired t-based 95% CI primary; sign-flip test as sensitivity analysis), comparison families with Holm adjustment of p-values within families, SESOI = 0.01 nats per character, learning-rate limitation, learning-behaviour analyses. (4) §16.6 and §18.1 Q6: the five-seed design and the "consistent advantage" rule retained as historical text, marked as superseded as the primary criterion. §16.4, §16.5, §16.5.1 and the epoch-budget pilot protocol are **unchanged**. | D-31 |
| **v0.6 addendum** | 2026-11-12 | Clarifications only: (1) §16.6.1: paired t-test p-values are the primary inferential p-values, Holm-adjusted within each family; the exact sign-flip test is a sensitivity analysis reported separately; paired 95% CIs remain the primary estimation framework. (2) §16.6.1: context-frequency buckets are defined and frozen on development data and applied to test predictions only as a secondary exploratory analysis within the single final test evaluation. (3) Header, §17 and §18.3: stale v0.5 status of the budget pilot clarified, with the v0.5 wording retained. §16.4, §16.5, §16.5.1 and the protocol unchanged. | D-31 addendum |
| **v0.7** | 2026-11-12 | Common epoch budget recorded: `max_epochs` = **26** for every arm (selected epochs A 23, B 25, C 25, D 26; η = 0.03 for all; common budget = max = 26). Status notes added in the header, §1.3, §7.2, §16.5, §17 and §18.3; the earlier "PENDING PILOT" wording is retained and marked. The §16.5 procedure, §16.5.1 and the epoch-budget pilot protocol are **unchanged**. | D-32 |
| **v0.8** | 2026-11-17 | Status update only: the final controlled experiment, the freeze and the one-time test evaluation are recorded as done, and the study as closed (header). A note explains that decision identifiers refer to the internal, unpublished research record. §17 boxes ticked at closure. No design, procedure or result statement is changed. | D-35, D-36 |

Notes on v0.5:
- **Checksum consequence (accepted).** The closed LR pilot (`results/experiments/wavenet_study/lr_pilot/conditions.json`) froze this file's v0.4 SHA-256 `915b1b3178bfb749df284e61b78515b665f506e8a6651704bfbdb75576ac01c2` as a blocking checksum. From v0.5 on, this file no longer matches that record. This is an accepted, documented consequence of keeping the LR pilot immutable: its frozen conditions are not regenerated or rewritten, and no LR-pilot command is run again. In the budget pilot, specification files are recorded but non-blocking (DEC-1).
- (First v0.5 revision.) Text superseded by v0.5 is retained and marked "v0.4" wherever it is changed: pending learning-rate values now decided (§7.2, §8, §18.3, header) and the incorrect step-0 description (§11). All other v0.5 changes are additions (§16, §16.4, §16.5 inputs, §16.5.1, §17). The §16.5 formulas, the extension rule and every other v0.4 statement are unchanged. The v0.4 statements that implementation is "not authorised" (header, §20) and that `wavenet.py` is "currently empty" (§20) predate the acceptance of Model 5 (D-12); they were left unchanged by the first v0.5 revision and are resolved by the v0.5 addendum (D-26) without deleting the v0.4 wording.

**Historical status line (v0.4, verbatim; superseded by the v0.5 status at the top):** DRAFT v0.4. Implementation is **not authorised**. All design decisions (Q1–Q12) are approved (2026-11-11, §18). The learning rate (per arm) and the common epoch budget are **PENDING PILOT** under the approved procedures of §16.4–16.5; neither may be chosen before those pilots. No pilot or training result exists for this model, and none is reported here.

Labels:
- **[T] theory:** follows from the equations.
- **[I] implementation requirement:** verified by tests (§14, §15).
- **[O] observation:** measured in scratch checks on the group split (seed 42, train-only vocabulary), **without** training on real data. Preliminary.
- **[L] legacy:** historical. Leaky split, single seed, sampling with replacement. **Not comparable** with current results.

The specification defines **two separately identifiable models**, which share one implementation file:

| Identifier | `model.name` | Normalisation | Parameters | Role |
|---|---|---|---|---|
| **WaveNet (baseline)** | `wavenet` | none | **21,989** (§5, confirmed) | Model 5 baseline |
| **WaveNet-BN (variant)** | `wavenet_batchnorm` | BatchNorm at every level | **22,397** (§5, confirmed) | separate experimental variant; not part of the baseline |

Sections marked **[BN]** apply only to the variant; everything else applies to both. Both variants share one hierarchical forward implementation (§7.1); they differ only in whether the per-level normalisation step exists.

---

## 1. Purpose and scope

### 1.1 Purpose
Model `P(x_t | x_{t-8}, …, x_{t-1})` over 27 classes with an 8-character, BOS-filled window. The window is combined **hierarchically**: adjacent pairs, then pairs of pairs, then the whole window. The model should test whether hierarchical composition helps relative to a flat MLP with the same context and approximately the same number of parameters (§16).

### 1.2 What this model is, and is not
- **It is** the receptive-field structure of WaveNet (van den Oord et al., 2016): a binary tree of kernel-2 merges. For the single prediction made from a window, this tree computes exactly the output of a stack of dilated causal convolutions (kernel 2, dilations 1, 2, 4) evaluated at the last position. [O] Verified numerically in float64: max difference 1.1e-16 (§14.3).
- **It is not** the full WaveNet. There are no gated activations (tanh ⊙ σ), no residual or skip connections, no causal-convolution modules, no μ-law output, and no conditioning. The name "WaveNet" refers to the receptive-field structure only. Every report must say so.

### 1.3 Scope
**Current scope (v0.5).**
- The two models above, their model-local training procedure, tests, configuration files: implemented and accepted (D-12).
- The experimental protocol (§16): the LR pilot (§16.4) has been run and its result accepted (D-22); the common epoch-budget pilot (§16.5, §16.5.1) has an approved protocol (D-23, D-26) and is pending; final runs (§16.6) are pending. *(v0.7, D-32: the epoch-budget pilot is completed and the common budget of 26 epochs is approved; final runs remain pending.)* Each experimental stage requires its own explicit authorisation, and experiment code lives in `training/wavenet_study/`, not in this model's implementation.
- Still out of scope: changes to Models 1–4, to shared utilities, or to accepted specifications (other than by documented amendment); gated, residual or convolutional variants.

**v0.4 scope of the implementation task (historical, verbatim).**
- In scope: the two models above, their model-local training procedure, tests (including the new independent reference tests, §14), configuration files and the experimental protocol (§16).
- Out of scope: changes to Models 1–4, to shared utilities, or to accepted specifications; any training or pilot run; gated, residual or convolutional variants.

(The v0.4 exclusion of "any training or pilot run" is read as applying to the implementation task; the pilots of §16 are separately authorised experimental stages.)

### 1.4 Historical lessons applied (Phase 1)
| Historical limitation [L] | Treatment here |
|---|---|
| Leaky split, single seed, sampling with replacement | group split, multiple paired seeds, `EpochSampler` |
| Notebook and script differ (output ×0.1, split, schedule, RNG) | one specification; zero output layer; explicit seeds |
| Conditional `squeeze`; `view` on possibly non-contiguous inputs; BN buffers changing shape | fixed shapes, `reshape`, explicit final flatten (§4) |
| Unbiased-variance BatchNorm; EMA inference statistics | biased batch variance; exact recalibration (§3.6) |
| Weak comparisons (single seed, confounds) | controlled design with paired seeds (§16) |
| "Kaiming" initialisation was really LeCun; claims of "no parameter cost" | stated correctly (§5, §6) |

Historical numbers are context only. They are not expectations for the new protocol.

## 2. Architecture overview

```
contexts [B, 8] (int64)
  └─ emb: Embedding(27, 10) ............................. [B, 8, 10]
level 1: merge_pairs ..................................... [B, 4, 20]
         hidden1: Linear(20, 68, bias=False) ............. [B, 4, 68]
         [BN] bn1: LevelBatchNorm1d(68) .................. [B, 4, 68]
         act1: Tanh ...................................... [B, 4, 68]
level 2: merge_pairs ..................................... [B, 2, 136]
         hidden2: Linear(136, 68, bias=False) ............ [B, 2, 68]
         [BN] bn2 ........................................ [B, 2, 68]
         act2: Tanh ...................................... [B, 2, 68]
level 3: merge_pairs ..................................... [B, 1, 136]
         hidden3: Linear(136, 68, bias=False) ............ [B, 1, 68]
         [BN] bn3 ........................................ [B, 1, 68]
         act3: Tanh ...................................... [B, 1, 68]
flatten: reshape(B, 68), unconditional ................... [B, 68]
out: Linear(68, 27) with bias ............................ [B, 27] logits
```

- **Weight sharing:** within a level, the same `hidden_ℓ` (and, [BN], the same `bn_ℓ`) is applied to every group. Level ℓ is therefore a stride-2, kernel-2 convolution over the level-(ℓ−1) sequence.
- **No hidden biases** in either model. For the baseline this is the approved choice. For [BN], β provides the shift.
- **Last-level shape (APPROVED, Q2):** every level uses a uniform 3-D representation. The merge always returns 3-D, so level 3 gives `[B, 1, 136]` and `[B, 1, 68]`. After the third hierarchical transformation, one explicit, **unconditional** `reshape` from `[B, 1, 68]` to `[B, 68]` precedes the output projection. No dimension is ever squeezed conditionally, and there is no special last-level merge. Every shape is documented (§4.2) and tested (§15.2).
- Submodules have flat top-level names (`emb`, `hidden1…3`, `bn1…3`, `act1…3`, `out`), so `gradient_summaries(group_by="module")` and `parameter_breakdown` give per-level groups with no changes to shared code (§11).

## 3. Mathematical formulation

### 3.1 Notation
- `V = 27`, `k = 8` (context length), `L = log₂ k = 3` (levels), `d = 10`, `H = 68`.
- `G_ℓ = k / 2^ℓ` is the number of groups at level ℓ (4, 2, 1).
- `C_ℓ` is the input width of `hidden_ℓ`: `C_1 = 2d = 20`, `C_ℓ = 2H = 136` for ℓ ≥ 2.
- Ids `x_{n,p} ∈ {0, …, 26}` for positions p = 1…8, oldest first. Position 8 is the token immediately before the target. PAD (id 27) never enters the model.

### 3.2 Pair merging
For `X ∈ ℝ^{B×T×C}` with T even:

    merge(X)[b, j, :] = [ X[b, 2j, :] ; X[b, 2j+1, :] ]      j = 0, …, T/2 − 1   (0-based)

The output is in `ℝ^{B×(T/2)×2C}`: the earlier position occupies the first C features and the later position the next C. `merge` is a fixed linear rearrangement (a permutation of entries), so its gradient is the inverse rearrangement.

### 3.3 Baseline forward (normalisation = none)

    h⁰_{n,p}   = E[x_{n,p}]                                ∈ ℝ^{d},  p = 1..8      (h⁰ ∈ ℝ^{B×8×d})
    m^ℓ        = merge(h^{ℓ−1})                            ∈ ℝ^{B×G_ℓ×C_ℓ}
    a^ℓ_{n,j}  = W_ℓ m^ℓ_{n,j}                             ∈ ℝ^{H},  W_ℓ ∈ ℝ^{H×C_ℓ}  (shared over j)
    h^ℓ        = tanh(a^ℓ)                                 ∈ ℝ^{B×G_ℓ×H}
    z_n        = W_o h³_{n,1} + b_o                        ∈ ℝ^{V},  W_o ∈ ℝ^{V×H}
    P(x_t = c | window) = softmax(z_n)_c

Written out:

    h¹_{n,j} = tanh(W₁ [u_{2j−1}; u_{2j}])                      j = 1..4  sees positions {2j−1, 2j}
    h²_{n,j} = tanh(W₂ [h¹_{2j−1}; h¹_{2j}])                    j = 1..2  sees positions {4j−3, …, 4j}
    h³_n     = tanh(W₃ [h²_1; h²_2])                                      sees positions {1, …, 8}

### 3.4 Receptive field [T]
- Group j at level ℓ depends exactly on positions `{(j−1)·2^ℓ + 1, …, j·2^ℓ}`, and on no other position.
- The receptive field doubles per level. After L levels it covers `2^L = k` positions.
- The merges have no parameters, but each level adds a Linear layer (and, [BN], a BatchNorm). The receptive-field growth therefore has a parameter cost; the historical claim "no parameter cost" is incorrect.
- Within level 1 the features are position-agnostic, because W₁ is shared. Order across groups is restored by the concatenation at the next level.
- [O] Autograd dependence sets on a random instance: {5, 6} for level-1 group 3, {1–4} for level-2 group 1, {1–8} for the output.

### 3.5 Loss and gradients [T]
Per-example loss `ℓ_n = logsumexp(z_n) − z_{n,y_n}`. Mean over the actual batch: `L_b = (1/|b|) Σ_n ℓ_n`. With `g_n = (p_n − e_{y_n})/|b|`:

    ∂L/∂W_o = Σ_n g_n h³_nᵀ,   ∂L/∂b_o = Σ_n g_n,   δ³_n = (W_oᵀ g_n) ⊙ (1 − (h³_n)²)
    ∂L/∂W_ℓ = Σ_n Σ_j δ^ℓ_{n,j} (m^ℓ_{n,j})ᵀ        (sum over groups: weight sharing)
    ∂L/∂m^ℓ_{n,j} = W_ℓᵀ δ^ℓ_{n,j};  split into its two halves → ∂L/∂h^{ℓ−1} at positions 2j−1, 2j
    δ^{ℓ−1}_{n,i} = (∂L/∂h^{ℓ−1}_{n,i}) ⊙ (1 − (h^{ℓ−1}_{n,i})²)
    ∂L/∂E[v] = Σ_n Σ_{p: x_{n,p}=v} ∂L/∂h⁰_{n,p}

- Every level's weight gradient **accumulates over groups**, and the embedding gradient accumulates over all positions where a token occurs (e.g. eight times for an all-BOS window).
- [BN] The BatchNorm backward of Model 4 (`batchnorm_mlp_spec.md` §4.3) applies per level. It is pooled over the `B·G_ℓ` normalised rows: `Σ_{n,j} ∂L/∂a^ℓ_{n,j} = 0` per unit, and the forward is approximately invariant to positive rescaling of each row of `W_ℓ`.

### 3.6 [BN] Normalisation at level ℓ

**Train mode.** Statistics are pooled per unit `c` over all `R_ℓ = B·G_ℓ` rows (examples × groups):

    μ^ℓ_c  = (1/R_ℓ) Σ_{n,j} a^ℓ_{n,j,c}
    σ²^ℓ_c = (1/R_ℓ) Σ_{n,j} (a^ℓ_{n,j,c} − μ^ℓ_c)²                  (biased)
    y^ℓ    = γ_ℓ ⊙ (a^ℓ − μ^ℓ)/√(σ²^ℓ + ε) + β_ℓ,   h^ℓ = tanh(y^ℓ)

Pooling over groups matches the weight sharing: one feature map has one normalisation, never per-position statistics. Per-position statistics were the historical "buggy" BatchNorm; they are an explicitly different normaliser and are excluded.

**Eval mode.** `y^ℓ = γ_ℓ ⊙ (a^ℓ − m^ℓ)/√(v^ℓ + ε) + β_ℓ`, with the authoritative `inference_mean` / `inference_var` of level ℓ.

**Constants:** ε = 1e-5 and momentum = 0.1 (EMA, diagnostic only), as in Model 4.

**Exact inference statistics (recalibration).** With fixed parameters θ and all M training windows, the population statistics of level ℓ are the mean and **biased** variance (divisor `M·G_ℓ`) of `a^ℓ`. Here `a^ℓ` is computed with **levels 1…ℓ−1 already in eval mode with their newly recalibrated statistics**. This is the statistics of the network exactly as it is evaluated. Consequences:
- Levels are calibrated **sequentially** (APPROVED, Q4): level 1 (independent of any normalisation), install; then level 2, using the installed level-1 statistics; install; then level 3. That is L streaming passes over the training windows.
- **Why sequential calibration is necessary** [T]. The pre-activation `a^ℓ = W_ℓ merge(h^{ℓ−1})` depends on `h^{ℓ−1}`, and in eval mode `h^{ℓ−1}` depends on level ℓ−1's inference statistics `(m^{ℓ−1}, v^{ℓ−1})`. Changing those statistics changes the whole distribution of `a^ℓ`. Level ℓ's statistics are only correct for the network that is actually evaluated if they are computed **after** the lower levels hold their final, new statistics. The alternatives each describe a network that is never evaluated:
  - computing all levels in one pass with the **old** lower-level inference statistics (from the previous epoch, or placeholders 0/1 at the first epoch) gives level ≥ 2 statistics of a network whose lower levels no longer exist;
  - computing them with the **EMA** buffers ties the authoritative statistics to the diagnostic EMA, which the approved Model 4 design forbids;
  - computing them with **train-mode** batch statistics makes them batch-dependent and updates the EMA.

  Only the sequential procedure has the full-batch consistency property below, which is used as a test oracle.
- [T] **Full-batch consistency:** if one training batch contains all windows, then after recalibration the eval-mode outputs equal the train-mode outputs for the same parameters. Level by level the eval statistics equal the batch statistics, because the lower levels' outputs coincide. This is a test oracle (§14.6).

### 3.7 Conditional-entropy bound
- [O] For k = 8 on the group split: `M_train = 182,666`; 49,968 distinct training contexts (34,724 of them singletons); `Ĥ(Y|W₈) = 1.4096`; **21.6% of dev windows have a context unseen in training** (0.82% for k = 3).
- `Ĥ(Y|W₈)` lower-bounds the fixed-parameter full-train NLL of any window-only predictor: the baseline in both modes, and [BN] in eval mode. It is very loose because of the singletons, so it serves as a violation detector, not a target.
- It does not bound dev NLL, mini-batch losses, or train-mode [BN] predictions (which are batch-dependent).
- [O] The oldest pair (positions 1–2) is (BOS, BOS) in 91.7% of windows, and the whole window is BOS in 14.0%. Many level-1 inputs at early positions are therefore one constant vector.

## 4. Tensor shape trace and shape safety

### 4.1 `merge_pairs` (public, module-level function in `wavenet.py`)

```python
def merge_pairs(x: torch.Tensor) -> torch.Tensor:
    """[B, T, C] → [B, T // 2, 2C]; out[b, j] = cat(x[b, 2j], x[b, 2j + 1])."""
```

Requirements [I]:
1. **Validation:**
   - `x` must be a `torch.Tensor` (`TypeError`);
   - `x.ndim == 3` (`ValueError`);
   - `T = x.shape[1]` must be even and ≥ 2 (`ValueError` naming T);
   - `C = x.shape[2]` must be ≥ 1 (`ValueError`);
   - `B = 0` is allowed.
2. **Implementation:** `x.reshape(B, T // 2, 2 * C)`. `reshape` is contiguity-safe: it returns a view when the strides allow and copies otherwise. `view` is prohibited because it fails on non-contiguous input ([O] confirmed).
3. **Never squeezes;** the output is always 3-D.
4. **Ordering:** earlier position first, as §3.2. It is checked against an index-coded oracle (§14.1).
5. **Dtype and device preserved;** no copy to another device; autograd flows (`reshape` is differentiable).

### 4.2 Full trace (B = batch size; baseline values)

| Step | Operation | Input shape | Output shape |
|---|---|---|---|
| 0 | validate `contexts` | `[B, 8]` int64 | — |
| 1 | `emb` | `[B, 8]` | `[B, 8, 10]` |
| 2 | `merge_pairs` | `[B, 8, 10]` | `[B, 4, 20]` |
| 3 | `hidden1` (matmul over the last dim) | `[B, 4, 20]` | `[B, 4, 68]` |
| 4 | [BN] `bn1` (pooled over `B·4` rows) | `[B, 4, 68]` | `[B, 4, 68]` |
| 5 | `act1` | `[B, 4, 68]` | `[B, 4, 68]` |
| 6 | `merge_pairs` | `[B, 4, 68]` | `[B, 2, 136]` |
| 7 | `hidden2` | `[B, 2, 136]` | `[B, 2, 68]` |
| 8 | [BN] `bn2` (`B·2` rows) | `[B, 2, 68]` | `[B, 2, 68]` |
| 9 | `act2` | `[B, 2, 68]` | `[B, 2, 68]` |
| 10 | `merge_pairs` | `[B, 2, 68]` | `[B, 1, 136]` |
| 11 | `hidden3` | `[B, 1, 136]` | `[B, 1, 68]` |
| 12 | [BN] `bn3` (`B` rows) | `[B, 1, 68]` | `[B, 1, 68]` |
| 13 | `act3` | `[B, 1, 68]` | `[B, 1, 68]` |
| 14 | flatten: `h.reshape(B, H)` | `[B, 1, 68]` | `[B, 68]` |
| 15 | `out` | `[B, 68]` | `[B, 27]` |

- After every level, `forward` asserts the expected shape `[B, G_ℓ, H]`, so a shape error raises instead of broadcasting silently.
- Step 14 is unconditional. `G_L = 1` holds by construction, because `k = 2^L` is validated.

## 5. Parameter count derivation

General formulas, with `k = 2^L`:

    baseline:  V·d  +  (2d)·H  +  (L−1)·(2H)·H  +  H·V + V
    [BN]:      baseline  +  L·2H        (γ_ℓ, β_ℓ per level)

| Group | Module | Shape | Baseline | [BN] |
|---|---|---|---|---|
| E | `emb.weight` | `[27, 10]` | 270 | 270 |
| W₁ | `hidden1.weight` | `[68, 20]` | 1,360 | 1,360 |
| W₂ | `hidden2.weight` | `[68, 136]` | 9,248 | 9,248 |
| W₃ | `hidden3.weight` | `[68, 136]` | 9,248 | 9,248 |
| γ, β | `bn1–3.weight/bias` | 3 × 2 × `[68]` | — | 408 |
| W_o | `out.weight` | `[27, 68]` | 1,836 | 1,836 |
| b_o | `out.bias` | `[27]` | 27 | 27 |
| **Total** | | | **21,989** | **22,397** |

- **Confirmed (Q1):** baseline **21,989**; [BN] **22,397**. The v0.1 brief's "22,397 for the baseline" was the historical model *with* BatchNorm. No integer H gives 22,397 without BatchNorm (H = 68 gives 21,989; H = 69 gives 22,584), and hidden biases (22,193) are excluded by design.
- **Difference between the variants:** +408 = `L · 2H`, the γ and β of each level, about +1.9%. BatchNorm buffers are not parameters (§10).
- **Comparison arms (§16)**, with the counts derived for the accepted implementations:

  | Arm | Parameters | Difference vs WaveNet baseline |
  |---|---|---|
  | Model 3 MLP with `context_length = 8` | 21,897 | −92 |
  | Model 4 BN-MLP with `context_length = 8` | 22,097 | +108 |
  | WaveNet | 21,989 | — |
  | WaveNet-BN | 22,397 | +408 |

  Every pair differs by less than 2.3%.

## 6. Initialization

All draws use `make_generator(derive_seed(seed, "init"))` on CPU, in the order `E`, `W₁`, `W₂`, `W₃`, before any `.to(device)`. Modules are allocated with `nn.utils.skip_init`, and every parameter (and, [BN], every buffer) is set explicitly. The global RNG is never touched.

| Tensor | Initialisation | Rationale |
|---|---|---|
| `E` | `N(0, 1)` | as Models 3–4 |
| `W_ℓ` | LeCun normal `N(0, 1/C_ℓ)`: std `1/√20 ≈ 0.2236` (ℓ = 1), `1/√136 ≈ 0.0857` (ℓ = 2, 3) | unit-variance pre-activation for unit-variance inputs |
| `W_o`, `b_o` | exactly 0 | uniform initial prediction; no historical ×0.1 |
| [BN] `γ_ℓ` = 1, `β_ℓ` = 0; EMA `running_mean` = 0, `running_var` = 1, `num_batches_tracked` = 0; `inference_mean` = 0, `inference_var` = 1, `num_calibrations` = 0 | | as Model 4 |

**Pre-activation scale by level** [T, O]:
- **Level 1:** inputs are N(0,1) embeddings, so Var(a¹) ≈ 1.
- **Higher levels:** inputs are tanh outputs, so `Var(a^{ℓ+1}) = E[tanh²(a^ℓ)] < Var(a^ℓ)`. LeCun without a tanh gain therefore shrinks the scale: theory gives std ≈ 1, 0.628, 0.486 for levels 1–3.
- [O] Measured at initialisation on the training windows (mean over 5 seeds):

  | | Level 1 | Level 2 | Level 3 |
  |---|---|---|---|
  | Pre-activation std | 0.929 | 0.579 | 0.473 |
  | tanh saturation (\|h\| > 0.97) | 2.8% | 0.04% | 0.00% |

  Level 1 is below 1 because constant BOS pairs reduce the spread.
- This is **intended and documented, not corrected:** the approved initialisation does not include a gain. It is monitored per level (§11) and is the subject of experiment E4.
- [BN] At initialisation the normalised std is 1.000 at every level, with saturation 6.3%, 5.4% and 4.0%.

**Zero output layer** [T]. Logits are exactly 0 for every input, in both models and both modes, so the initial loss is exactly `ln 27`. On the first backward pass only `W_o` and `b_o` receive gradient: the gradients of E, W₁₋₃ and [BN] γ, β are **exactly zero**, because every path passes through `W_o = 0`. From the second step every group receives gradient. There is no symmetry problem, since E and W_ℓ are random. This is intended, as in Models 3–4.

**Why these choices:**
- They isolate the architectural change. The embedding, LeCun and zero-output conventions are identical to the flat comparison arms.
- They remove the historical notebook/script initialisation discrepancy.
- They make the initial state analytically known (`ln 27`), which gives exact tests.

**How they are tested (§15.1, §15.4):**
- exact zeros;
- determinism under the init generator;
- global-RNG isolation;
- sampling-error tolerances per level: `|std(W_ℓ)·√C_ℓ − 1| < 0.05` for W₂, W₃ (n = 9,248), < 0.10 for W₁ (n = 1,360, standard error ≈ 1.9%); `|std(E) − 1| < 0.2`;
- a non-default configuration (`context_length = 16`, `hidden_dim = 32`) to check that the scale follows `C_ℓ`;
- initial loss `ln 27` and exactly zero first-step gradients.

## 7. Configuration schema

### 7.1 Architecture: `WaveNetConfig` (`model.params`)

```python
@dataclass(frozen=True)
class WaveNetConfig:
    context_length: int = 8          # APPROVED; power of two >= 2 (validated); L = log2(context_length)
    embedding_dim: int = 10          # APPROVED; int >= 1
    hidden_dim: int = 68             # APPROVED; int >= 1
    normalization: str = "none"      # "none" (baseline) | "batchnorm" (variant); any other value rejected
```

- The merge factor is a constant (2) and is not configurable. ε, momentum and initialisation are code constants.
- **Identification (APPROVED, Q3):** one implementation file (`models/hierarchical/wavenet.py`) and one model class, `WaveNetLanguageModel`. The class has **one** hierarchical forward implementation; the per-level normalisation step is present only when `normalization == "batchnorm"`. No subclass, registry or factory is introduced. The two variants are kept apart by:
  - **Configuration files:** `configs/models/wavenet.yaml` (`name: wavenet`, `normalization: none`) and `configs/models/wavenet_batchnorm.yaml` (`name: wavenet_batchnorm`, `normalization: batchnorm`).
  - **Model identity (APPROVED, Q11):** a read-only property `model.model_name`, computed on every access from `self.config.normalization` (`"none"` → `"wavenet"`, `"batchnorm"` → `"wavenet_batchnorm"`). `config.normalization` is the **single authoritative source**. The name is never stored as a separate attribute, has no setter, and is not a constructor argument, so it cannot disagree with the normalisation actually built. (`WaveNetConfig` is frozen, so `normalization` cannot change after construction.) The derived name is passed to `RunManager.create`, written to every checkpoint as `config["model_name"]` and to `config.json`, and printed in every report row.
  - **Run directories:** `results/runs/wavenet/` and `results/runs/wavenet_batchnorm/`.
  - **Checkpoints:** identity is validated on every load and resume (§10.1). Any mismatch raises `IncompatibleCheckpointError` with a message naming both identities.
  - **Runs:** `train_wavenet` rejects a `run` whose `run.model_name` differs from `model.model_name` (`ValueError`, before any step), and an `experiment` config whose `model.name` differs from `model.model_name`. A WaveNet run directory can therefore never receive WaveNet-BN checkpoints, or the reverse.
  - **Experiment arguments:** an experiment script that loads `wavenet.yaml` must obtain `normalization == "none"`, and one that loads `wavenet_batchnorm.yaml` must obtain `"batchnorm"`. A test checks this pairing (§15.2).

### 7.2 Training: `WaveNetTrainingConfig` (model-local, not part of `ExperimentConfig`)

| Field | Default | Validation | Status |
|---|---|---|---|
| `lr` | required | finite > 0 | **0.03** for each arm (LR pilot, accepted 2026-11-11, D-22; §16.4). v0.4: PENDING PILOT (separate per model; §16) |
| `max_epochs` | required | int ≥ 1 | **PENDING PILOT** (common budget for all arms; Q7). *v0.7: **26** for every arm in the final controlled experiment (epoch-budget pilot, approved D-32; §16.5)* |
| `batch_size` | 32 | int ≥ 1; ≥ 2 for [BN] (checked in `train_wavenet`, §8) | APPROVED |
| `lr_decay_epoch` | None | None or 1 ≤ value < `max_epochs` | kept for parity with Models 3–4; None in the pilot |
| `lr_decay_factor` | 0.1 | 0 < value ≤ 1 | as Models 3–4 |
| `eval_batch_size` | 4096 | int ≥ 1 | evaluation and recalibration chunking |

The optimiser (plain SGD, momentum 0, weight decay 0) and the loss reduction (mean) are fixed in code.

### 7.3 Recorded configuration
`run_config(experiment, model_config, training)` returns:

```
{"experiment": …, "model_name": "wavenet" | "wavenet_batchnorm", "model": asdict(WaveNetConfig),
 "training": asdict(WaveNetTrainingConfig), "batchnorm": BATCHNORM_RECORD or None}
```

`BATCHNORM_RECORD = {"eps": 1e-5, "momentum": 0.1, "inference_statistics": "full_train_biased_sequential"}` is set for [BN] and is `None` for the baseline.

### 7.4 Shared sections
`reproducibility`, `data`, `checkpoint`, `diagnostics` and `generation` are as Models 3–4 (seeds via `derive_seed(seed, "init" | "data" | "generation")`; `generation.max_new_tokens` is used as `max_length`).

## 8. Training protocol

The model-local `train_wavenet(model, train, dev, *, training, seed, run=None, resume=False, diagnostics=None, config=None) -> WaveNetTrainingResult` mirrors `train_mlp` / `train_batchnorm_mlp`. There is no shared trainer and no import from `models.*`.

- **Data:** `make_context_windows(splits.*, vocab, context_length=8)`; `vocab = Vocabulary.from_names(splits.train)`; group split, seed 42.
- **Batches:** `EpochSampler(M, batch_size, seed=derive_seed(seed, "data"))`. Sampling without replacement; the final partial batch is kept ([O] 5,708 batches of 32 plus one of 10 per epoch).
- **Batch-size rule:**
  - Baseline: no batch dependence, so any batch ≥ 1 is valid.
  - [BN]: every training batch must hold ≥ 2 windows, because level 3 pools over B rows only. `train_wavenet` rejects `M mod batch_size == 1`, or `batch_size < 2`, before any step, and train-mode `forward` rejects `B < 2` before any BatchNorm call (no buffer change).
- **Objective:** `token_nll(logits, y, reduction="mean")` over the actual batch size. Epoch metrics are example-weighted through `NLLAccumulator`.
- **Step order** (as Models 3–4): forward → loss → finite-loss check → backward → finite-gradient check (one fused sync; groups named on failure) → diagnostics → `optimizer.step()`. A non-finite value raises `FloatingPointError` before the step.
- **Mode discipline:**
  - `model.train()` is used only for the SGD forwards.
  - Evaluation, diagnostics, generation and [BN] recalibration never run a train-mode BatchNorm forward.
  - [BN] `num_batches_tracked == steps` at every level after training (a test).
- **Learning rate:** epoch-based and stateless (`lr`, optionally `× lr_decay_factor` from `lr_decay_epoch`). The value is 0.03, constant, for both variants (arms C and D) and for the flat arms A and B of the study (D-22, v0.5). v0.4: PENDING PILOT.
- **Epoch end:**
  1. [BN] sequential recalibration (§3.6, §8.1);
  2. full-train and full-dev NLL in eval mode;
  3. epoch-end diagnostics on dev;
  4. `model_best.pt` if dev strictly improved (ties keep the earliest epoch);
  5. history row;
  6. `checkpoint_last.pt`;
  7. after the last epoch, `model_final.pt`.
- **Three reported NLLs** (as Models 3–4): running train NLL, fixed-parameter full-train NLL, full-dev NLL. For the baseline, the running NLL differs from the full-train NLL only because parameters change during the epoch. For [BN] it also contains the train/eval normalisation mismatch.
- **No early stopping;** a fixed budget. Best checkpoint = lowest full-dev NLL.

### 8.1 [BN] Recalibration algorithm (`calibrate_batchnorm(model, windows, *, batch_size)`)
For ℓ = 1…L, in order:
1. Initialise per-unit `(n, mean, M2)` in float64 on CPU.
2. Process training windows in sequential chunks of `batch_size`. For each chunk:
   - compute `a^ℓ` via the **private** method `model._level_preactivations(contexts, level=ℓ)` (under `no_grad`; not part of the public API, Q10). It shares the forward's level code, runs levels < ℓ with their **eval-mode** normalisation (`F.batch_norm` with the inference buffers) regardless of `model.training`, and never calls a BatchNorm module in train mode;
   - reshape to `[n_c·G_ℓ, H]`;
   - move to CPU, *then* cast to float64 (`.to(device="cpu").to(dtype=torch.float64)`; the MPS lesson from Model 4);
   - compute the centred two-pass chunk moments;
   - Chan-merge them into the running totals;
   - release the chunk.

   The `[M·G_ℓ, H]` matrix is never retained, and no list of chunks or chunk statistics is kept.
3. `m = mean`, `v = M2 / (M·G_ℓ)`. Check finiteness and `v ≥ 0` (`FloatingPointError` naming the level and epoch). Install with `bn_ℓ.set_inference_statistics(m, v)`, cast to the buffer dtype; this increments that level's `num_calibrations`.

The procedure never reads or writes the EMA buffers, parameters or `model.training`. Its result does not depend on previous buffer values. Cost: about L partial forwards over the training set, which is cheap relative to an epoch.

## 9. Evaluation and generation

- **Evaluation:** `evaluate_nll(model, windows, *, batch_size) -> NLLAccumulator`. Eval mode and `no_grad`; every window, including the final partial chunk; the original mode restored in `finally`.
- **Test protocol:** the test split is evaluated **once** per final run, on `model_best.pt`, after all configuration decisions are frozen (§16). It is never used in pilots.
- **Generation:** `generate(num_samples, *, max_length, generator=None, temperature=1.0, top_k=None, top_p=None) -> list[WaveNetSample]`:
  - eval mode, with the previous mode restored in `finally`, for both models;
  - the context starts as `[0]·8` and shifts as `context[1:] + [token]`, identical to `make_context_windows(k=8)` (a test);
  - EOS and `max_length` semantics as Models 1–4: at most `max_length` ordinary characters and at most `max_length + 1` draws;
  - an explicit generator, by convention `make_generator(derive_seed(seed, "generation"))`;
  - `num_samples = 1` is valid (eval mode has no batch dependence);
  - parameters and buffers are unchanged.

## 10. Checkpointing and resume

As Models 3–4 (`mlp_spec.md` §14, `batchnorm_mlp_spec.md` §14), using `utils.checkpoint` (`weights_only=True`):
- **Kinds:** `checkpoint_last.pt` (`kind="resume"`, every epoch), `model_best.pt` (`kind="best"`, strict improvement), `model_final.pt` (`kind="final"`).
- **Write order per epoch:** best → history → last; then final.
- **Contents:**
  - `model_state` (all parameters and, [BN], all buffers);
  - `optimizer` (resume only);
  - `step`;
  - `epoch` = completed epochs;
  - `best_metric` / `best_metric_name = "dev_nll"`;
  - `rng_state` (resume only);
  - `config = run_config(...)` (§7.3);
  - `vocab`;
  - `extra = {"best_epoch", "sampler", "base_seed"}`.
- **Resume:** continue from the completed epoch under the **original** `max_epochs` and schedule; history truncated to completed epochs. Changes to any of these are rejected with `IncompatibleCheckpointError`:
  - the model config (including `normalization`) and `model_name`;
  - `lr`, `max_epochs`, `batch_size`, `lr_decay_epoch`, `lr_decay_factor`;
  - the sampler seed and base seed;
  - the vocabulary and dataset size;
  - the `batchnorm` record.
- **`from_checkpoint(path, *, expected_vocab=None, expected_model_name=None, map_location="cpu")`** constructs the variant from the stored `config["model"]` and then applies §10.1. It rejects:
  - a wrong `model_class`;
  - missing or mismatching vocabulary;
  - an invalid or shape-mismatched config;
  - a `batchnorm` record inconsistent with `normalization`, or a stored `model_name` inconsistent with either;
  - a `model_name` different from `expected_model_name`, when given;
  - [BN] any level with `num_calibrations == 0`, or levels with unequal counters.

### 10.1 Identity validation (APPROVED, Q11)
The stored identity is `ckpt["config"]["model_name"]`, written from `model.model_name` at save time. The receiving model's identity is its derived `model.model_name`. Checks, in this order, each raising `IncompatibleCheckpointError` with both values in the message:

1. **Stored self-consistency:** the stored `model_name` must equal the name derived from the stored `config["model"]["normalization"]`, and the stored `batchnorm` record must be `BATCHNORM_RECORD` for `"batchnorm"` and `None` for `"none"`.
2. **Receiving model:**
   - in `from_checkpoint`, the model is built from the stored configuration, and its derived `model_name` must equal the stored one (a guard against a corrupted or hand-edited checkpoint);
   - on resume, the caller's existing model must have the same `model_name` (and the same full model config) as the checkpoint.
3. **Caller expectation:** if `expected_model_name` is given, it must equal the stored `model_name`. An unknown value (not `"wavenet"` or `"wavenet_batchnorm"`) raises `ValueError`.

Tests (§15.7):
- a round trip preserves `model_name`;
- loading a WaveNet-BN checkpoint with `expected_model_name="wavenet"` fails, and the reverse;
- a checkpoint edited so that `model_name` disagrees with `normalization` fails;
- a checkpoint whose `batchnorm` record disagrees with `normalization` fails;
- resuming a WaveNet checkpoint with a WaveNet-BN model fails;
- passing a `wavenet_batchnorm` run to a WaveNet model fails;
- `model.model_name` has no setter (assignment raises `AttributeError`).

## 11. Diagnostics

Shared tools only (`gradient_summaries`, `global_grad_norm`, `nonfinite_gradients`, `ParameterSnapshot`, `ActivationMonitor`, `tanh_saturation`), plus small model-local summaries. No shared utility is extended, and no public intermediate-activation API is added (Q10): per-level activations are observed through the named modules (`hidden1–3`, [BN] `bn1–3`, `act1–3`) with forward hooks, which is what `ActivationMonitor` already does.

**Required per-level records** (Q8): pre-activation mean and std (`hidden_ℓ` outputs), activation distribution summary (`act_ℓ` outputs: mean, std, min, max), tanh saturation fraction and per-unit saturation, and gradient norms per level. These make the documented scale shrinkage of the baseline (§6) visible over training; they are recorded, never corrected.

**Step level** (scheduled by `DiagnosticsSchedule`; written to `<run>/diagnostics/steps.jsonl`):
- **Gradients:** `global_grad_norm`; `gradient_summaries(group_by="module")` with groups `emb`, `hidden1`, `hidden2`, `hidden3`, [BN] `bn1–3`, `out`, i.e. per-level gradient norms; `nonfinite_gradients`.
- **Updates:** `ParameterSnapshot.update_stats` gives `update_norm`, `relative_update` (‖Δθ‖/‖θ‖) and `log10_update_to_param_std` (the measured update/data ratio) per parameter.
  - At step 0 the ratio is `None` for **every** parameter (v0.5 correction; v0.4 stated "for `out.weight`, whose std is 0" only). Reason: the output layer is zero-initialised (§6), so at step 0 `∂L/∂h = W_oᵀδ = 0`; every parameter upstream of the output layer has zero gradient and zero update (`update_std = 0`), and `out.weight` and `out.bias` have zero pre-step norm and standard deviation. The shared utility returns `None` unless both standard deviations are positive. Observed in all 60 LR-pilot runs, for all four arms, only at step 0. It is recorded as `null`, not as an error. At step 0, `relative_update` of `out.weight` (and of other zero-norm parameters) is a finite, ε-regularised value (order 10¹⁰ for `out.weight`); it is recorded unchanged. In the budget pilot these step-0 values are classified as expected structural observations only under the exact conditions of the protocol (§16.5.1).
- **Activations:** an `ActivationMonitor` attached during the SGD forward, on `hidden1–3` (pre-activations), [BN] `bn1–3` (normalised) and `act1–3` (tanh, with `tanh_saturation(0.97)`).

**Epoch end** (eval mode, on dev; stored in the `EpochRecord`):
- `activations`: the same monitor, plus per-level per-unit saturation summaries (fraction over windows × groups for each of the 68 units: min, median, max, and the number of units ≥ 0.99).
- `output`:
  - mean and std of the dev logits;
  - mean predictive entropy;
  - mean max-probability;
  - ‖W_o‖ and ‖b_o‖.
- `levels`: per level, the row-norm summary of `W_ℓ` (mean, min, max).
- [BN] `batchnorm`, per level:
  - EMA vs inference statistics (mean/max |z|, median/max |log((v_EMA+ε)/(v+ε))|);
  - γ and β summaries;
  - `num_calibrations` and `num_batches_tracked`.

**Overhead rules** (as Model 4): no retained graphs; parameter snapshots only on scheduled steps; mode restored in `finally`; no train-mode forward.

## 12. Error handling

| Condition | Error | Where |
|---|---|---|
| `contexts` not a tensor; dtype ≠ int64 | `TypeError` | `forward`, `_level_preactivations` |
| `ndim ≠ 2`, `shape[1] ≠ context_length`; wrong device | `ValueError` ("move them explicitly") | same |
| PAD id or id outside `[0, 27)` | `ValueError` (dedicated PAD message); empty-safe check | same |
| [BN] train mode with `B < 2` | `ValueError` before any BatchNorm call | `forward` |
| `merge_pairs`: non-tensor / `ndim ≠ 3` / odd or < 2 T / C < 1 | `TypeError` / `ValueError` | `merge_pairs` |
| Shape after a level ≠ `[B, G_ℓ, H]` | `RuntimeError` (internal invariant) | `forward` |
| Invalid config (`context_length` not a power of two ≥ 2; unknown `normalization`; non-int) | `ValueError` | `WaveNetConfig` |
| Invalid training config; [BN] `batch_size < 2` or `M mod batch_size == 1` | `ValueError` | `WaveNetTrainingConfig`, `train_wavenet` |
| Non-finite loss or gradient | `FloatingPointError` naming epoch, step, groups; before `optimizer.step()` | `train_wavenet` |
| [BN] non-finite recalibration | `FloatingPointError` naming level and epoch; nothing installed | `calibrate_batchnorm` |
| Checkpoint incompatibility, including any identity mismatch (§10.1) | `IncompatibleCheckpointError` naming both values | `from_checkpoint`, resume |
| `run.model_name` or experiment `model.name` ≠ `model.model_name`; unknown `expected_model_name` | `ValueError` before any step / on call | `train_wavenet`, `from_checkpoint` |
| Assignment to `model.model_name` | `AttributeError` (read-only property) | model |
| private `_level_preactivations` with `level ∉ {1, …, L}` | `ValueError` (internal guard) | `_level_preactivations` |

## 13. Device compatibility

- **Placement:** parameters and buffers are created and initialised on CPU, then moved with `model.to(device)`. `forward` never moves data, and windows are moved by the training and evaluation helpers.
- **Backends:** CPU, CUDA and MPS. Every operation used (`Embedding`, `Linear` on 3-D input, `reshape`, `tanh`, BatchNorm, `F.batch_norm`, SGD, `multinomial`) exists on all three.
- **Determinism:** bit-exact on CPU. CUDA needs `configure_determinism` (atomic adds in the embedding backward). MPS gives no guarantee.
- **MPS float64:** no float64 on device. All float64 work (recalibration moments, diagnostic summaries) uses `.to(device="cpu").to(dtype=torch.float64)`, never a single combined `.to(cpu, float64)` (Model 4 MPS fix). Device tests cover these helpers with device-originating tensors.
- **Device tests** (skipped when unavailable): a few epochs of training, recalibration, evaluation and generation on CUDA/MPS matching CPU within 1e-5; device-mismatch errors; checkpoint `map_location`.

## 14. Independent reference tests (new project standard)

**Test categories.** The specification distinguishes four kinds of evidence; each test item in §14–§15 belongs to exactly one:

| Category | Question answered | Where | Uses model internals? |
|---|---|---|---|
| **Independent reference tests** | Does the model compute the same numbers as an independently written implementation of the equations? | §14; `test_wavenet_reference.py` with `wavenet_reference.py` | no: cloned parameters, public `forward`, forward hooks only |
| **Mathematical correctness tests** | Does the model satisfy properties that follow from the equations (receptive field, equivariance, zero-output gradients, BatchNorm identities, entropy bound, parameter formula)? | §15.1; `test_wavenet.py` | public API and hooks |
| **Unit and integration tests** | Do contracts, errors, training, recalibration, checkpoints, resume, generation and devices behave as specified? | §15.2–15.8; `test_wavenet.py` | may use the private calibration mechanism where memory discipline must be checked (§15.5) |
| **Experimental comparisons** | Which arm generalises better under a controlled protocol? | §16; outside the test suite | not tests; never part of acceptance by passing |

**Location:**
- `tests/models/hierarchical/wavenet_reference.py`: a helper module; its name does not start with `test_`, so it is not collected.
- `tests/models/hierarchical/test_wavenet_reference.py`: the tests.

**Independence rules [I]:**
1. The reference module imports only `torch`, `math` and `typing`. An AST test fails if it imports anything from `models`, from `utils`, or `wavenet`.
2. Reference functions take **tensors only**: cloned parameters (`{name: t.detach().clone().double()}` from `model.state_dict()`), contexts and statistics. They never receive the model object, never call `forward` or any model method, and never use model helpers (including `merge_pairs`).
3. Model-side values are obtained only from the public `forward` / `generate` and from **forward hooks** on named submodules (`hidden1–3` input = merged tensor; outputs of `hidden_ℓ`, `bn_ℓ`, `act_ℓ`, `out`).
4. Parameter-name mapping, documented in the reference module:

   | State-dict key | Reference symbol |
   |---|---|
   | `emb.weight` | E |
   | `hidden{ℓ}.weight` | W_ℓ, shape `[H, C_ℓ]` |
   | `out.weight` / `out.bias` | W_o / b_o |
   | `bn{ℓ}.weight` / `bias` / `inference_mean` / `inference_var` | γ_ℓ / β_ℓ / m_ℓ / v_ℓ |

5. Precision: high-precision comparisons run on CPU in float64 (model `.double()`, reference float64). Float32 comparisons are secondary.

**Reference functions** (specified here; written in the test helper):
- `ref_merge(x)`: `torch.stack([torch.cat([x[:, 2j], x[:, 2j+1]], -1) for j in range(T//2)], 1)` (explicit loop and gather).
- `ref_forward(params, contexts, *, normalization, mode, stats=None) -> dict`:
  - explicit Python loops over levels **and groups**: `a[:, j] = m[:, j] @ W_ℓ.T` per group j;
  - manual BatchNorm (pooled `mean((0, 1))`, biased `var`, ε = 1e-5) in train mode, or the given statistics in eval mode;
  - `tanh`, and explicit final indexing `h[:, 0, :]`;
  - returns every intermediate (`e`, `m1`, `a1`, `y1`, `h1`, …, `logits`).
- `ref_conv_forward(params, contexts, *, normalization, mode, stats=None)`: the secondary reference (§14.3).
- `ref_loss(logits, y)`: `(torch.logsumexp(logits, 1) − logits.gather(1, y[:, None])[:, 0]).mean()`.
- `ref_calibrate(params, contexts)`: all windows at once in float64; levels sequential (§3.6). Returns `{ℓ: (m_ℓ, v_ℓ)}` with `v = var(unbiased=False)` over `M·G_ℓ` rows.

### 14.1 A. Pair merging
- **Index-coded tensor:** `x[b, t, c] = 1000·b + 10·t + c` (float64, B = 3, T = 8, C = 4). The model's `merge_pairs(x)` must equal `ref_merge(x)` **exactly** (`torch.equal`), and spot entries must equal hand values; e.g. `out[1, 2, 5] = x[1, 5, 1] = 1051`.
- **Explicit concatenation:** equality with `torch.cat([x[:, ::2], x[:, 1::2]], -1)` (an independent second formulation).
- **Non-contiguous inputs:** `x.transpose(0, 1).contiguous().transpose(0, 1)`, a sliced `x[:, :, ::2]`, and an expanded tensor; the results equal the contiguous result.
- **Invalid inputs:** T ∈ {0, 1, 3, 7} → `ValueError`; `ndim ∈ {2, 4}` → `ValueError`; non-tensor → `TypeError`. `B = 0` → `[0, T/2, 2C]`.
- **Gradients:** `gradcheck` (float64) of `merge_pairs`, and the backward of `sum(w ⊙ merge(x))` equals `ref_merge`'s inverse arrangement of `w`.
- **Repeated levels:** three merges of an index-coded `[B, 8, 1]` tensor give `[B, 1, 8]` with features in the order `t = 0…7`.

### 14.2 B. Hierarchical forward (explicit-loop reference)
- **Baseline:** for random parameters (generic `W_o`, `b_o` set under `no_grad`), every hooked intermediate (`m1–3`, `a1–3`, `h1–3`) and the logits equal `ref_forward` (float64, atol 1e-12, rtol 0).
- **[BN] train mode:** one batch (B = 6); hooks also on `bn1–3`; same tolerance.
- **[BN] eval mode:** after installing arbitrary inference statistics (and garbage EMA values); equal to `ref_forward(mode="eval", stats=…)` (atol 1e-12).
- **Hand-computed case:** `d = 1`, `H = 1`, `k = 4` (L = 2), small integer embeddings and weights; logits compared with values written out in the test.
- **Float32 secondary check:** baseline-sized model in float32, logits atol 1e-5.

### 14.3 C. Convolutional equivalence (secondary reference)
- **Layout:** sequences as `[B, C, T]` (`x.permute(0, 2, 1)`).
- **Weight mapping:** with `W_ℓ ∈ ℝ^{H×2C}` in PyTorch `[out, in]` layout, whose first C input columns act on the earlier position:

      K_ℓ = W_ℓ.view(H, 2, C).permute(0, 2, 1)     # [H, C, 2]
      K_ℓ[o, c, 0] = W_ℓ[o, c]        (earlier position)
      K_ℓ[o, c, 1] = W_ℓ[o, C + c]    (later position)

- **(i) Strided form:** level ℓ is `F.conv1d(x, K_ℓ, stride=2)`, then BatchNorm ([BN]: `F.batch_norm` on `[B, H, G_ℓ]`, which pools over batch and length) and `tanh`. It must match the hooked intermediates (permuted back) within atol 1e-12. [O] Strided mapping verified: max difference 8.9e-16.
- **(ii) Dilated causal form (baseline only):** `F.conv1d(seq, K_ℓ, dilation=2^{ℓ−1})` with no padding, for ℓ = 1, 2, 3. This shrinks length 8 → 7 → 5 → 1, and the final position equals the tree's level-3 output. [O] Verified: max difference 1.1e-16.

  For [BN] the dilated form is **not** equivalent: its statistics would pool over a different set of positions. It is used for the baseline only, and the specification states this explicitly.

### 14.4 D. Mathematical verification against the reference
Intrinsic properties (receptive field, equivariance, parameter formula, entropy bound) are model-property tests and are listed in §15.1. The reference comparisons are:
- **Receptive field (reference-derived):** the dependence sets of `ref_forward` intermediates, computed by autograd on the reference graph, equal those measured on the model (§15.1). The reference thereby fixes the expected sets independently of the model.
- **Loss:** `token_nll(model(x), y)` equals `ref_loss(ref_logits, y)` (float64, atol 1e-12).
- **Gradients:**
  - autograd through `ref_forward` with cloned leaf parameters equals the model's `.grad` for every parameter (float64, atol 1e-10). This covers the baseline and [BN] in train mode, including group accumulation and the all-BOS window (8 contributions to `E[0]`);
  - `torch.autograd.gradcheck` on the model with `functional_call` (float64, small H).
- **Parameter counts (reference-side):** the reference module provides `ref_parameter_count(V, d, H, L, normalization)` (the §5 formula). It equals `count_parameters(model)` for the default and two non-default configurations. Expected values 21,989 and 22,397.
- **Tolerances:**

  | Comparison | Tolerance |
  |---|---|
  | merge equality | exact |
  | float64 forward / intermediates / loss | atol 1e-12 |
  | float64 gradients | atol 1e-10 |
  | float32 logits | atol 1e-5 |
  | float32 loss | atol 1e-6 |
  | calibration, float32 model | rel 1e-6 |
  | calibration, float64 model | atol 1e-12 |
  | sampling tolerances | §6 |

### 14.5 E. [BN] variant
- **Normalisation equations:** the hooked `bn_ℓ` outputs equal the manual pooled biased normalisation over `B·G_ℓ` rows. They also equal `F.batch_norm` on the `[B, H, G_ℓ]` layout, an independent primitive.
- **Train vs eval:** in train mode the outputs depend on the other windows in the batch, and the EMA update matches `0.9·old + 0.1·μ` / `0.9·old + 0.1·(unbiased pooled var)`, also under `no_grad`. In eval mode outputs are batch-independent and ignore the EMA.
- **Calibration correctness:** after `calibrate_batchnorm`, each level's `(inference_mean, inference_var)` equals `ref_calibrate` (sequential levels, all windows at once). It also differs from a deliberately wrong single-pass calibration (lower levels with placeholder statistics), which confirms the sequential dependency is exercised.
- **Partition independence:** chunk sizes 1, 7, 64, M, M+5 give the same statistics.
- **Isolation:** calibration leaves parameters, EMA buffers, `num_batches_tracked` and `model.training` unchanged, and its result is independent of the previous inference and EMA values.
- **Full-batch consistency:** on the tiny fixture (one batch of 9), the train-mode logits of a copy equal the eval-mode logits after calibration (atol 1e-5, float32; 1e-12, float64).
- **Per-level statistics:** the diagnostic summaries (EMA vs inference, γ, β) equal values recomputed from the buffers by the reference.

## 15. Mathematical correctness, unit and integration tests

File `tests/models/hierarchical/test_wavenet.py`. Fixtures:
- SMALL: the 12 names of Models 3–4; 63 windows at k = 8.
- TINY: `["ab", "ac", "ab"]`; 9 windows; Ĥ = 0.212171.
- V27.
- Real data: skipped if absent.

Each test that differs by model is parametrised over `normalization ∈ {"none", "batchnorm"}`.

### 15.1 Mathematical correctness tests (model properties)
- **Receptive field (autograd):** for a level-ℓ unit of group j, `∂h^ℓ_{n,j,c}/∂e_{n,p}` is nonzero only for `p ∈ {(j−1)2^ℓ+1, …, j·2^ℓ}`, for every level and group (eval mode; [BN] with installed statistics). Embeddings are observed via a hook on `emb`.
- **Receptive field (perturbation):** changing tokens outside a unit's field leaves that unit bit-identical; changing any of the 8 tokens changes the logits (generic parameters).
- **Weight sharing / equivariance:** permuting the four level-1 pairs of the input permutes the level-1 outputs identically (baseline; [BN] eval mode; [BN] train mode too, since pooled statistics are permutation-invariant).
- **Position sensitivity:** swapping the two tokens of one pair changes the output (generic parameters), so order within a pair is used.
- **Zero-output initialisation:** initial loss `ln 27` in both modes; first-step gradients exactly zero for E, W₁₋₃ and [BN] γ, β; nonzero for every group after one step.
- **[BN] identities:** per level and unit, `Σ_{n,j} ∂L/∂a^ℓ_{n,j} = 0` (float64, atol 1e-12); γ and β gradients equal `Σ g ⊙ â` and `Σ g`; with ε = 1e-30 on a float64 copy, scaling a row of `W_ℓ` by 5 leaves the train-mode output unchanged (atol 1e-10; PyTorch rejects ε = 0 in train mode, as found in Model 4).
- **Parameter formula:** `count_parameters` and `parameter_breakdown` equal the §5 formula: `emb` 270, `hidden1` 1,360, `hidden2` 9,248, `hidden3` 9,248, [BN] `bn1–3` 136 each, `out` 1,863; totals 21,989 / 22,397; plus two non-default configurations.
- **Conditional-entropy bound:** fixed-parameter full-train NLL ≥ `Ĥ(Y|W) − 1e-6` (baseline in both modes; [BN] in eval mode after recalibration), for random and trained parameters, on the tiny and small fixtures.
- **Finite differences:** `torch.autograd.gradcheck` on the model via `functional_call` (float64, small H), in train mode.

### 15.2 Construction, configuration and shapes
- **Config validation:**
  - `context_length ∈ {0, 1, 3, 6, 12, True, "8"}` rejected; {2, 4, 8, 16} accepted;
  - `normalization ∈ {"", "BatchNorm", None, "layernorm"}` rejected;
  - non-int dims rejected.
- **Training-config validation,** including [BN] `batch_size = 1` rejected (but accepted for the baseline).
- **YAML:** both config files load with `model_config_cls=WaveNetConfig`; `model.name` and `normalization` match (`wavenet` ↔ `none`, `wavenet_batchnorm` ↔ `batchnorm`); unknown keys rejected.
- **Module inventory:**
  - baseline: exactly `emb`, `hidden1–3`, `act1–3`, `out`, with no BatchNorm module and no buffers;
  - [BN]: additionally `bn1–3`, each an `nn.BatchNorm1d` subclass with ε 1e-5 and momentum 0.1, and the six buffers per level;
  - `hidden_ℓ.bias is None`.
- `wavenet.py` imports nothing from `models.*`.

### 15.3 Input contract
- **Shape trace:** forward hooks on `hidden1–3`, [BN] `bn1–3`, `act1–3` and `out` (plus forward pre-hooks for the merged inputs) record exactly the shapes of §4.2 for B ∈ {1, 2, 32}: `[B, 4, 20]`, `[B, 4, 68]`, `[B, 2, 136]`, `[B, 2, 68]`, `[B, 1, 136]`, `[B, 1, 68]`, `[B, 68]` (input of `out`), `[B, 27]`. For B = 1 the level-3 output is still `[1, 1, 68]` (no squeeze).
- **Identity pairing:** `wavenet.yaml` yields `normalization == "none"` and `model_name == "wavenet"`; `wavenet_batchnorm.yaml` yields `"batchnorm"` and `"wavenet_batchnorm"`.
- `[B, 8]` → `[B, 27]` float32 in both modes.
- Eval-mode `[0, 8]` → `[0, 27]`. [BN] train-mode `B ∈ {0, 1}` → `ValueError` with no buffer change. Baseline train-mode `B = 1` is valid.
- Wrong ndim, width, dtype, device; PAD and out-of-range ids (empty-safe check).
- The public API contains no intermediate-activation method; `level_preactivations` does not exist as a public attribute (Q10).

### 15.4 Initialisation (§6)
- Exact zeros and ones.
- Determinism under the init generator.
- Global RNG untouched.
- Per-level LeCun sampling tolerances, including a non-default configuration.
- Initial loss and first-step gradients: covered by the mathematical correctness test of §15.1 (not duplicated here).

### 15.5 Training, recalibration and diagnostics (integration)
- **Tiny-fixture overfitting:** lr 0.5, 500 epochs, batch 32 (one batch of 9). Full-train NLL − Ĥ lies in `[−1e-6, 1e-2]` for both models.
  - [O] Scratch measurement with the approved procedure, gaps for seeds 0–2: baseline 5.2e-4, 6.3e-4, 7.3e-4; [BN] 6.0e-4, 6.2e-4, 6.7e-4. That is a ≥ 13× margin.
  - This is a debugging check, not a quality claim.
- **Mean reduction:** with 42 windows (batches 32 + 10), training equals an independent loop with per-batch means, not sums divided by 32.
- **Example-weighted running NLL;** evaluation coverage for any chunk size.
- **Determinism:** same seed gives bit-identical parameters and buffers; a different seed gives a different result.
- **Schedule** recomputed after resume. **Selection ties** keep the earliest epoch.
- **Non-finite guards:** an injected NaN loss and an injected NaN gradient (hook on `hidden2.weight`) each raise before the step; parameters are unchanged and the error names the group.
- [BN] **Batch-size guard:** SMALL with `batch_size = 31` (31, 31, 1) is rejected before any step. The same configuration is accepted for the baseline.
- [BN] **Epoch-end order:** `[steps…, calibrate (levels 1, 2, 3 in order), evaluate train, evaluate dev]`; `num_calibrations == epochs completed` at every level.
- [BN] **Mode discipline:** `num_batches_tracked == steps` at every level after training with all diagnostics enabled.
- [BN] **Streaming recalibration and memory:** the private `_level_preactivations` is wrapped on the instance. Calls are sequential and level-ordered (all level-1 chunks, then level 2, then level 3); each call covers at most `batch_size` windows; the row counts sum to M per level; a weak reference shows each chunk is released before the next call; no list of chunks is retained (peak activation memory is one chunk).
- [BN] **Non-finite calibration:** names the level and epoch; nothing is installed; the run is marked failed; `checkpoint_last.pt` is from the previous epoch.
- **Mode restoration:** `evaluate_nll`, `per_unit_saturation` and `generate` restore `model.training` for both starting modes, also on exceptions; buffers unchanged.
- **Diagnostics:** step records contain per-level gradient groups (`hidden1–3`, [BN] `bn1–3`) and activations. Epoch records contain `activations`, `output`, `levels` and, for [BN], `batchnorm`. Trajectories are bit-identical with and without diagnostics.

### 15.6 Generation
- Contexts used during generation equal `make_context_windows([text], k=8)` for finished samples, including the eight-BOS start.
- EOS and `max_length` semantics, including the draw count.
- Eval mode even when the model is in train mode; `num_samples = 1`.
- [BN] uses the inference statistics (output changes when they change, and is unaffected by the EMA).
- Reproducible with an explicit generator; text never contains `.`; state unchanged.

### 15.7 Persistence
- Contents of the three checkpoint kinds, including `normalization` and the `batchnorm` record (None for the baseline).
- Write order best → history → last, then final.
- Round trip for best and final: identical parameters, buffers, eval logits and seeded generation.
- `from_checkpoint` rejections (§10), including a baseline checkpoint that claims a `batchnorm` record, and [BN] with any level uncalibrated.
- Identity validation: every case listed in §10.1.
- Bit-exact resume of `max_epochs = 4` interrupted after 2 epochs (parameters, all buffers, history, best epoch).
- History truncation after a crash between the history append and the checkpoint save.
- Incompatible resume, one case per field in §10, including `normalization`.

### 15.8 Devices and real data
- Device tests as in §13, including a regression test of the float64 helpers with device-originating tensors.
- Real data (skipped if absent):
  - 182,666 training windows; `min(batch_sizes) = 10`;
  - parameter counts 21,989 / 22,397;
  - initial loss `ln 27`;
  - [BN] calibration on a 20k-window subset equal to the reference.

## 16. Experimental protocol (outside the implementation task)

No experiment is run under this specification, and no result is assumed.
*(v0.5: this sentence dates from v0.4. The LR pilot has since been run under §16.4 and its outcome accepted (§16.4 Outcome, D-22). No other experiment has been run.)*

### 16.1 Questions
- **Primary (architecture):** does hierarchical composition give lower development NLL than a flat MLP, at matched context (k = 8) and approximately matched parameters?
- **Secondary (normalisation ablation):** what is the effect of BatchNorm on the hierarchical model and on the flat model?

These are reported separately. An architecture comparison never mixes normalisation regimes.

*(v0.6, D-31: the primary question above is v0.4 wording. For the final controlled experiment the primary endpoint is **test NLL**, in nats per character, evaluated once per final run on the checkpoint selected by development NLL; best development NLL is the secondary endpoint (§16.6.1). Development NLL remains the selection metric of the LR pilot (§16.4), the common epoch-budget pilot (§16.5) and checkpoint selection. The architecture factor compares the one-layer flat MLP with the three-level hierarchical tree.)*

### 16.2 Arms

| Arm | Model and configuration | Parameters | Status |
|---|---|---|---|
| A | Model 3 MLP, `MLPConfig(context_length=8)` (accepted code, config override only) | 21,897 | required |
| B | Model 4 BN-MLP, `BatchNormMLPConfig(context_length=8)` (accepted code, config override only) | 22,097 | required |
| C | WaveNet, `normalization="none"` | 21,989 | required |
| D | WaveNet-BN, `normalization="batchnorm"` | 22,397 | required |

|  | Flat | Hierarchical |
|---|---|---|
| No BatchNorm | A: MLP, `context_length=8` | C: WaveNet |
| BatchNorm | B: BatchNorm MLP, `context_length=8` | D: WaveNet-BN |

**The complete 2 × 2 design is adopted (APPROVED, Q5).** The flat controls use the accepted Model 3 and Model 4 implementations unchanged; only `context_length` is overridden through configuration. Models 3 and 4 are not modified. The comparisons are:

| Comparison | Arms | Purpose |
|---|---|---|
| Architecture, without normalisation | **C − A** | |
| Architecture, with normalisation | **D − B** | |
| Normalisation ablations | B − A (flat), D − C (hierarchical) | |
| Interaction | (D − B) − (C − A) | |
| Combined change | C − B | reported, with the explicit statement that it changes **both** architecture (flat → hierarchical) **and** normalisation (BatchNorm → none), so it cannot be attributed to either factor |

### 16.3 Controls (identical for every arm)
- Group split (seed 42); train-only vocabulary; `make_context_windows(k=8)`; `EpochSampler`, batch 32, final partial batch kept.
- Plain SGD; mean reduction; epoch-based constant LR unless a schedule is approved.
- The same `max_epochs` for all arms (Q7).
- Best checkpoint by full-dev NLL with the earliest-epoch tie-break.
- Token-weighted NLL via `NLLAccumulator`.
- The same `eval_batch_size`.
- For BatchNorm arms, eval-mode statistics are exact recalibrated population statistics.
- **Paired conditions** (APPROVED, Q6): every arm uses the identical data split, the same seed identifiers (0–4), the identical sampler ordering (`derive_seed(s, "data")`; same M and batch size, so identical batch sequences) and the same evaluation procedure. Initialisation streams use the same `derive_seed(s, "init")` but differ by architecture, as they must. *(v0.6, D-31: final seed identifiers are 0–9; "0–4" above is v0.4 wording. The pilots keep seeds 100–102.)*

### 16.4 Learning-rate pilot (per arm, identical protocol)
- **Grid and seeds:** {0.03, 0.1, 0.3, 1.0}; seeds 100–102; 10 epochs; constant LR; train and dev only.
- **Selection:** mean full-dev NLL at epoch 10 among stable candidates.
  - *Operational definition of "stable" (as applied by the LR pilot, D-16; recorded in v0.5):* a candidate is excluded only if it is **mechanically unstable** on any seed (a `FloatingPointError` during training: non-finite loss, gradient or recalibration; or a non-finite recorded NLL). Threshold-free optimisation observations (dev-NLL rises, repeated rises, running-train rises, epoch-10 dev NLL above the seed's best) make a candidate *flagged for review*: it remains eligible, and the selection is provisional until a human has reviewed the flags. No numerical thresholds are used.
- **Reports:** per candidate, mean ± std of the epoch-10 dev NLL; best dev NLL and its epoch per seed; non-finite events; gradient spikes and oscillations.
- **Boundary rule:** extend by ×≈3 in the boundary direction, for at most two rounds. If the value is still at a boundary, the pilot is inconclusive and a revised experiment is proposed.
- **Arms A and B** are piloted **again at k = 8.** Their k = 3 pilot results (Models 3–4) do not transfer.
- The learning rate is selected **per arm**. Each architecture gets its best stable rate under the same protocol, so no arm is disadvantaged by a rate tuned for another.
- **Order:** the four LR pilots come first; the budget pilot (§16.5) uses their selected rates; neither the rates nor the budget are chosen before the corresponding pilot has been run and recorded.
- **Schedule:** constant LR in the LR pilot, the budget pilot and the final runs, for every arm. Any schedule (e.g. a step decay) would need separate approval and a new pilot.
- **Outcome (v0.5).** The LR pilot was run on 2026-11-11 (60 runs: grid {0.03, 0.1, 0.3, 1.0} plus one boundary-extension round at 0.01; seeds 100–102; 10 epochs; CPU, 1 thread). In every arm the lowest mean epoch-10 dev NLL was at 0.03, an interior point; all 0.03 candidates were flagged for review and were accepted after human review. **Accepted: η = 0.03 for arms A, B, C and D (D-22).** For arm D, 0.03 and 0.1 were nearly indistinguishable (2.0784 vs 2.0791, mixed paired per-seed differences); 0.03 was retained under the predefined rule and for its lower seed variation. Results: `results/experiments/wavenet_study/lr_pilot/`; details in `PROJECT_RESEARCH_RECORD.md` §8.5.1.

### 16.5 Common epoch budget (APPROVED procedure, Q7; value PENDING PILOT)
All four arms use the **same** final `max_epochs`, determined by this procedure. Final training is not authorised until it has been run and recorded.

**Outcome (v0.7, D-32, 2026-11-12).** The heading's "value PENDING PILOT" is v0.4–v0.6 wording. The procedure below was run under the accepted protocol (§16.5.1), with pilot seeds 100–102 and η = 0.03 for every arm.
- Selected epochs e*: **A 23, B 25, C 25, D 26**. Step 5: max(23, 25, 25, 26) = **26**.
- No arm required the extension to 60. A, B and D had SE-derived ε at e* and were reviewed by a human; C's ε was floor-derived.
- **Accepted: `max_epochs` = 26 for every arm in the final controlled experiment (D-32).** This accepts the pilot's common budget under this procedure; it does not claim that 26 epochs is optimal in general.
- The final runs are not yet executed and still require their own authorisation. Results: `results/experiments/wavenet_study/epoch_budget_pilot/`; details in `PROJECT_RESEARCH_RECORD.md` D-32.

**Inputs.** (v0.5: `η_a = 0.03` for every arm, D-22.) For each arm a ∈ {A, B, C, D}: the learning rate `η_a` selected by its LR pilot (§16.4), constant schedule, batch 32, pilot seeds **100, 101, 102**, train and dev only (the test split is never loaded).

**Step 1: pilot runs.** Train each arm with each pilot seed for `max_epochs = 30`, recording the full-dev NLL `D_{a,s}(e)` at the end of every epoch e = 1, …, 30. This is the fixed-parameter, eval-mode quantity of §8; BatchNorm arms are recalibrated first. Epochs are counted as completed epochs (1-based).

**Step 2: curve statistics,** per arm and epoch:

    D̄_a(e)  = (1/3) Σ_s D_{a,s}(e)                                (mean over the three seeds)
    SD_a(e) = sqrt( (1/2) Σ_s (D_{a,s}(e) − D̄_a(e))² )            (sample std, divisor n − 1 = 2)
    SE_a(e) = SD_a(e) / √3                                         (standard error at epoch e)
    D̄*_a    = min_e D̄_a(e);   ê_a = smallest e attaining D̄*_a     (ties → earliest)

**Step 3: selected epoch at the current cap.** With the tolerance evaluated **at the candidate epoch**, over the epochs e = 1, …, `E_cap` evaluated so far:

    ε_a(e) = max(1e-3, SE_a(e))
    e*_a   = min { e ≤ E_cap : D̄_a(e) ≤ D̄*_a + ε_a(e) }

`e*_a` is the earliest epoch whose mean dev NLL is within tolerance of the arm's best mean. It always exists, because `ê_a` satisfies the condition (`ε ≥ 1e-3 > 0`), and `e*_a ≤ ê_a ≤ E_cap`.

**Step 4: extension decision (APPROVED, Q12).** "Still improving at the current cap" uses the **same** tolerance as Step 3:
- If `e*_a < E_cap`, the arm has reached an acceptable plateau. It needs no extension, and `e*_a` is its selected epoch.
- If `e*_a = E_cap`, no earlier epoch is within tolerance of the best mean, so the arm is still improving at the cap:
  - at `E_cap = 30`, **that arm** is re-run for `max_epochs = 60` with the same three seeds and its `η_a`, and Steps 2–4 are recomputed for it over e = 1, …, 60 (`E_cap = 60`);
  - at `E_cap = 60`, the budget pilot is declared **inconclusive**: all curves are reported, no budget is selected, and further approval is requested.
- **One extension at most** (30 → 60). The same rule is applied identically at both caps.
- The re-run starts from scratch: resume across a changed budget is rejected by design (§10). On CPU its first 30 epochs are bit-identical to the 30-epoch run, so the extension equals continued training; on accelerators they agree up to backend nondeterminism, and the 60-epoch curves are the ones used.
- Arms with `e*_a < 30` keep their 30-epoch curves and selected epochs.

Under this definition, a curve still falling by amounts below the tolerance counts as a plateau. For example, a mean curve falling 1e-4 per epoch over 30 epochs (with SE ≤ 1e-3) has `ê = 30` but `e* = 20 < 30`, so it is not extended.

**Step 5: common budget.** `max_epochs = max_a e*_a` over all four arms. This single value is used by every arm in the final runs (§16.6), with the arm's own `η_a`. Taking the maximum means no arm is stopped before its own selected epoch. Final checkpoints are selected by best dev NLL (no early stopping), so arms with a smaller `e*_a` are not penalised by the extra epochs.

**Step 6: record.** For each arm: `η_a`, the per-seed curves `D_{a,s}(e)`, `D̄_a`, `SD_a`, `SE_a`, `ε_a`, `D̄*_a`, `ê_a`, `e*_a` at each cap evaluated, and whether the extension to 60 was used. Also the common `max_epochs`, or "inconclusive".

**Worked illustration of the rule (hypothetical numbers, not results).** If an arm had `D̄* = 2.050` at `ê = 24`, and at epoch 18 had `D̄ = 2.0508` with `SE = 0.0006`, then `ε(18) = max(0.001, 0.0006) = 0.001` and `2.0508 ≤ 2.051`. Epoch 18 qualifies, and `e* = 18` if no earlier epoch qualifies.

### 16.5.1 Approved operational amendments (v0.5, D-23)
The rule above (Steps 1–6, including the extension rule of Step 4) is unchanged. The operational protocol that applies it is `training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md`, which is canonical for every operational detail. In summary:
- **Inputs and conditions:** η = 0.03 for A–D; seeds 100–102; CPU with 1 intra-op thread and the LR pilot's inter-op thread count; step diagnostics as in the LR pilot (option B); new results folder `results/experiments/wavenet_study/epoch_budget_pilot/` with its own frozen conditions; the LR pilot is closed and only read. Execution-affecting code, configurations and data are blocking checksums; documents are recorded but non-blocking.
- **Test protection:** no test windows are constructed and the test split is never evaluated; splits are formed only for the disjointness and vocabulary-coverage checks.
- **Computation:** statistics for an arm only when all three seeds are complete and pass every integrity gate; full precision; optimisation flags descriptive only. For every arm and epoch the summary reports D̄, SD, SE, ε, each seed's value and the ε source: `SE-derived` (SE > 0.001), `floor-derived` (SE < 0.001) or `floor (SE equal)` (SE = 0.001). A selected epoch with an SE-derived tolerance requires explicit human review before the budget is accepted.
- **Stability categories:** A (mechanical instability: non-finite loss, gradients, parameters after a step, recalibration statistics or module outputs; checked at every step, read-only) halts the pilot with no retry, no budget and no extension; B (diagnostic computation failure) blocks acceptance but not execution and is never instability; C (finite observations, including the exactly specified step-0 structural nulls of §11 and `batchnorm: null` in non-BN arms) is descriptive.
- **Integrity:** per-epoch hashes of all parameters and buffers; G5 (epochs 1–10 bit-identical to the LR pilot's LR-0.03 runs) and G6 (epochs 1–30 of a 60-epoch run bit-identical to the 30-epoch run), checked by the runner (stopping the run as `integrity_blocked`) and independently by the summary script.
- **Execution:** staged (validation stage with seed 100, validation review, completion stage, selection review, extension stage only on explicit authorisation, final review); explicit retries from scratch with a written reason; selection mode only by an explicit flag.
- **Addendum (D-26):** (I-A) a G5/G6 mismatch stops the affected run immediately and makes the overall sweep `integrity_blocked`; no subsequent stage proceeds until human review; a retry requires explicit authorisation, a written reason and a fresh run from scratch under identical frozen conditions. (I-B) The read-only finiteness checks run in every forward pass, training and evaluation; evaluation never modifies parameters, buffers or training state. (I-C) G4 (checkpoint loading and consistency with the recorded history) and G7 (state-hash records exist and match the corresponding checkpoint states) are independent gates, tested separately. **Category B resolution:** every event gets a unique identifier and a preserved record (run, arm, seed, epoch, step, diagnostic, observed value, reason, evidence; initial status `unresolved`); a human records `accepted_as_expected`, `corrected` (any new execution separately authorised) or `rejected`, with reviewer, timestamp, rationale and evidence; no event disappears from the summary; unresolved or rejected events block acceptance (protocol §9.7).

### 16.6 Final runs and reporting
*(v0.6, D-31: the list below is the v0.4 design, retained as written. Where it differs from §16.6.1 — the seed count (five → ten), the endpoint, and the "consistent advantage" rule as the primary criterion — §16.6.1 governs. Per-seed and per-arm reporting, checkpoint selection, the frozen-before-test rule, parameter accounting and labelling remain in force, with "five" read as "ten".)*
- **Seeds (APPROVED, Q6):** five final seeds, 0–4, for every arm. *(Superseded by §16.6.1: ten seeds, 0–9.)*
- **Per seed, per arm (every seed reported individually):** best dev NLL, its epoch, train NLL at that epoch, and (after freezing) test NLL.
- **Per arm:** parameters; mean and standard deviation across the five seeds of best dev NLL and of its epoch; train/dev gap.
- **Per comparison of §16.2:** the five per-seed paired differences (same seed identifier in both arms), their mean, standard deviation and standard error, and the number of seeds sharing the sign of the mean.
- **Conservative decision rule (descriptive, not a significance test):** a comparison is described as a "consistent advantage" only if the mean paired difference exceeds two standard errors **and** all five seeds share its sign; otherwise it is described as "no reliable difference at this budget". This rule is a pre-declared reporting convention. It is **not** a formal statistical significance test, makes no claim about p-values or error rates, and is reported as such. *(Superseded as the primary criterion by §16.6.1 (D-31); retained as historical. The number of seeds sharing the sign of the mean remains a reported descriptor. A result is never described as statistically significant because it meets this rule.)*
- **Test NLL** is reported once per arm and seed, on `model_best.pt`, only after all decisions above are frozen. It is never used for selection.
- **Parameter accounting:** report the counts of every arm. The BatchNorm arms carry +200 (flat) / +408 (hierarchical) γ, β parameters, at most 2.3% between any two arms. Differences of that size are stated, not adjusted for.
- **Labelling:** every table states "WaveNet = hierarchical receptive-field structure only (no gates, residuals or skips)". Historical numbers [L] may appear only in a separate, clearly labelled context table.

### 16.6.1 Final controlled experiment: endpoints, seeds and statistical reporting (APPROVED, v0.6, D-31, 2026-11-12)

**Scope.** This section applies **only to the final controlled experiment**. The LR pilot (§16.4), the common epoch-budget pilot (§16.5, §16.5.1) and `training/wavenet_study/EPOCH_BUDGET_PILOT_PROTOCOL.md` are unchanged. They select on development NLL with pilot seeds 100–102, never evaluate the test split, and are not re-run or extended because of this section.

**Selection versus evaluation.**

| Stage | Selection metric | Seeds | Test split |
|---|---|---|---|
| LR pilot (§16.4) | mean dev NLL at epoch 10 | 100–102 | never used |
| Epoch-budget pilot (§16.5) | per-epoch mean dev NLL, §16.5 rule | 100–102 | never used |
| Final runs: checkpoint | best full-dev NLL, earliest-epoch tie-break (`model_best.pt`; §16.3) | 0–9 | not used for selection |
| Final runs: endpoints | none (evaluation only) | 0–9 | evaluated once per run, after every decision is frozen |

**Endpoints.**
- **Primary:** test NLL in nats per character (token-weighted, as for dev; §16.3), evaluated **once** per final run on `model_best.pt`, the checkpoint selected by development NLL, only after every decision is frozen. It is never used for any selection.
- **Secondary:** best development NLL of the run.

**Seeds.** Ten paired final seeds, **0–9**, for every arm, under the paired conditions of §16.3.

**Paired differences.** For a comparison X − Y and seed s: `d_s = NLL(X, s) − NLL(Y, s)`; negative values favour X. For the architecture × normalisation interaction: `d_s = (D_s − B_s) − (C_s − A_s)`.

**Reported for every pre-declared comparison** (estimation first):
- every per-seed difference `d_s`;
- the mean paired difference, the sample standard deviation (divisor n − 1), the standard error (SD/√n);
- the two-sided 95% paired t confidence interval, `mean ± t(0.975, n − 1) · SE` (with n = 10, t(0.975, 9) ≈ 2.262), which is the **primary uncertainty interval**;
- the number of seeds sharing the sign of the mean (descriptive);
- optionally, an exact sign-flip permutation test as a **sensitivity analysis**.

**Comparison families (pre-declared).**

| Family | Comparisons | Role |
|---|---|---|
| Primary: architecture | C − A, D − B | one-layer flat MLP vs three-level hierarchical tree, without and with BatchNorm |
| Secondary: normalisation and interaction | B − A, D − C, (D − B) − (C − A) | effect of BatchNorm per architecture; whether it differs between them |
| Descriptive | C − B | both factors change; no causal interpretation (§16.2) |

- p-values, where reported, are **Holm-adjusted within each inferential family** (primary; secondary). The descriptive comparison is not part of either family.
- The 95% confidence intervals are **nominal, per comparison**. They do not provide simultaneous, family-wise coverage, and are not presented as if they did.
- An averaged architecture main effect is not reported unless the interaction results justify such a summary; the two architecture simple effects are reported in any case.

**Smallest effect size of interest (SESOI): 0.01 nats per character.** A project-specific practical threshold chosen by convention, not a universal constant. The 95% CI of a paired difference is read as:
- entirely below −0.01: practically meaningful improvement for X;
- entirely above +0.01: practically meaningful improvement for Y;
- entirely within [−0.01, +0.01]: evidence of practical equivalence;
- otherwise: inconclusive regarding practical importance.

A non-significant result, or an interval that includes zero, is **not** evidence of equivalence.

**The v0.4 decision rule.** The "consistent advantage" rule of §16.6 (mean paired difference above two standard errors and all seeds sharing its sign) is no longer the primary criterion. It is retained as historical text; seed-sign consistency remains a descriptive statistic. No result is called statistically significant because it meets that rule.

**Learning rate.** The per-arm learning rates selected by §16.4 (η = 0.03 for A–D) are retained. [LIMITATION] The grid is coarse (×≈3 spacing) and the rate was selected at 10 epochs, so every conclusion is conditional on the selected rates. A learning-rate robustness experiment is not part of this design and would require a separate decision and authorisation.

**Learning-behaviour analyses.** Analyses of existing records come before any additional training.
- **From measurements the final runs already log:** training and development NLL trajectories; the generalisation gap; update-to-parameter ratios (step diagnostics, §11); per-level activation scales and tanh saturation (epoch-end diagnostics, §11).
- **Requiring new post-training analysis code, without new training:** context-frequency-dependent predictive performance, from per-target NLL computed with the saved `model_best.pt` checkpoints.

**D-31 addendum (2026-11-12): clarifications.** These clarify the reporting above; they change no endpoint, seed, family or SESOI.
- **Statistical tests.**
  - Paired 95% confidence intervals remain the **primary estimation framework**.
  - **Paired t-test p-values** (two-sided, on the per-seed differences `d_s`, the test that corresponds to the paired t interval) are the **primary inferential p-values**.
  - Holm correction is applied to those paired t-test p-values **within each declared comparison family** (primary; secondary).
  - The exact sign-flip permutation test is a **sensitivity analysis**, reported separately from the primary p-values.
- **Context-frequency analysis.**
  - Development-set predictions are used for the initial exploration and to define the context-frequency buckets. The bucket definitions are then **frozen**.
  - The frozen definitions are applied to test predictions only as a **secondary, exploratory analysis**, within the single planned final test-evaluation pass of the primary endpoint. No additional test evaluation is introduced.
  - Test results are **never** used to change buckets, models, hyperparameters, checkpoints or any later experimental decision.

### 16.7 Optional experiments
- **E1:** per-level saturation and gradient flow over training.
- **E2:** EMA vs exact statistics in D.
- **E3:** sensitivity to H at fixed k.
- **E4:** per-level scale shrinkage (§6) and its evolution in C.
- **E5:** k = 16 (L = 4), for which the configuration is already supported.

## 17. Acceptance criteria
*(v0.5: Model 5 was formally accepted as a whole (D-12). The boxes below are the v0.4 checklist, retained as written; they were not ticked individually in this document. The last-but-one item remains open for the budget pilot.)*

*(v0.8: the boxes are now ticked. Every criterion was met when Model 5 was accepted (D-12), except the pilots, which were completed with the common budget of 26 epochs (D-32); the final experiment then ran and was closed (D-35, D-36).)*

- [x] The two variants are separately identifiable (`model_name` `wavenet` / `wavenet_batchnorm`; `normalization` recorded in configs, checkpoints, run directories and reports) while sharing one hierarchical forward implementation
- [x] Architecture and shape trace exactly as §2/§4: no conditional squeeze; contiguity-safe merge; invalid lengths rejected; per-level shape assertions
- [x] Parameter counts 21,989 / 22,397, with breakdowns as §5
- [x] Initialisation as §6 for every parameter and buffer; no PyTorch default survives; global RNG untouched
- [x] Zero-output behaviour verified; initial loss `ln 27`
- [x] [BN] pooled biased train statistics; EMA diagnostic only; exact, sequential, streaming recalibration; batch-of-1 guard; train mode only in SGD forwards
- [x] Training semantics of Models 3–4 preserved: sampler, mean reduction, step order and non-finite guards, three NLLs, selection, checkpoints and resume, history truncation
- [x] Independent reference tests (§14) pass, with the reference module importing nothing from `models.*` and never calling model methods
- [x] Mathematical correctness tests (§15.1) pass
- [x] Unit and integration tests (§15.2–15.8) pass on CPU; device tests on available accelerators (MPS on the user's Mac)
- [x] No public intermediate-activation API; per-level diagnostics via named modules and hooks
- [x] Model identity: `model.model_name` derived only from `config.normalization`; persisted in every checkpoint; validated on load and resume (§10.1); mismatches raise clear errors
- [x] Ruff clean (`--line-length 120`) on new files; no unauthorised file changed
- [x] LR pilots (§16.4) and then the budget pilot (§16.5) run and recorded for all four arms before any final result; one common `max_epochs` used by all arms (**PENDING PILOT**). v0.5: LR pilots completed and accepted (η = 0.03 for A–D, D-22); budget pilot protocol approved (D-23), not yet implemented or run. *(v0.6 addendum, D-31 addendum: the v0.5 note is historical. The budget pilot has since been implemented (D-27, D-29) and executed (user-reported, 2026-11-12); the common `max_epochs` and its human approval remain pending.)* *(v0.7: budget pilot completed; common `max_epochs` = 26 approved (D-32). The final runs have not been executed.)*
- [x] User has run and reviewed locally; explicit approval to proceed

## 18. Open questions and unresolved decisions

### 18.1 Approved (from the brief)
| Item | Decision |
|---|---|
| Architecture | k = 8, merge factor 2, L = 3, d = 10, H = 68, tanh, Linear per level shared across groups, 27 logits; no residuals, skips, gates or conv modules |
| Baseline normalisation | none |
| Variant | BatchNorm at each level; no hidden biases; explicit normalisation and calibration semantics |
| Initialisation | `E ~ N(0, 1)`; LeCun `W_ℓ`; no hidden biases; `W_o = b_o = 0`; no ×0.1 |
| Shape safety | explicit shapes; no conditional squeeze; contiguity-safe; invalid lengths rejected; order preserved |
| Training | SGD without momentum; `EpochSampler`; batch 32 without replacement; partial batch kept; batch ≥ 2 where required; token-weighted NLL; best by dev; epoch-boundary resume; deterministic seeds; CPU/CUDA/MPS |
| LR and budget | by controlled pilot |
| Testing | independent PyTorch reference tests (§14) |
| Experiments | identical splits, multiple seeds, same evaluation, protected test; architecture and normalisation comparisons separated |
| Q1 parameter counts | baseline 21,989; [BN] 22,397 (+408 = 3 levels × (68 γ + 68 β)) |
| Q2 shapes | uniform 3-D at every level; one explicit unconditional reshape `[B, 1, 68] → [B, 68]` before the output projection; every shape documented and tested |
| Q3 file structure | one file `models/hierarchical/wavenet.py`; one class with one hierarchical forward; two identities (`wavenet`, `wavenet_batchnorm`) with separate configs, checkpoints and run directories |
| Q4 recalibration | sequential per level, after lower levels hold their new statistics; streaming CPU float64 with transfer-then-cast (MPS-safe) |
| Q5 design | complete 2 × 2 (A–D); flat controls use accepted Models 3–4 unchanged; C − B reported as a combined change |
| Q6 seeds | five final seeds (0–4); paired conditions; per-seed, mean, std, paired differences; conservative descriptive decision rule, not a significance test *(v0.6: amended by D-31 — ten final seeds (0–9), primary endpoint test NLL, estimation-first reporting with comparison families and SESOI 0.01; §16.6.1)* |
| Q8 scale shrinkage | accepted as a property of the specified initialisation; no scaling corrections; per-level statistics recorded |
| Q9 reference tests | `tests/models/hierarchical/wavenet_reference.py`, `test_wavenet_reference.py`, `test_wavenet.py`; no `models.*` imports in the reference |
| Q10 intermediate activations | no public `level_preactivations`; named modules and hooks; a private `_level_preactivations` used only by recalibration |
| Q7 epoch budget | procedure of §16.5: pilot seeds 100–102; cap 30; one extension to 60 for an arm still improving at 30; still improving at 60 → inconclusive; `ε(e) = max(1e-3, SE(e))` at the candidate epoch; per-arm earliest epoch within tolerance; common budget = maximum over the four arms, used by all arms |
| Q12 "still improving" | tolerance-based: at the current cap, compute the selected epoch `e*_a` (§16.5 Step 3); extend once (30 → 60) only if `e*_a` equals the cap; otherwise the arm has plateaued; the same rule at 30 and 60; still requiring extension at 60 → inconclusive |
| Q11 model identity | read-only `model.model_name` derived from `config.normalization` (single source); persisted as `config["model_name"]`; validated on load and resume against the receiving model; optional `expected_model_name`; `IncompatibleCheckpointError` on any mismatch (§10.1) |

### 18.2 Pending approval
None. Q1–Q12 are all approved (§18.1). The specification as a whole still requires the final review and an explicit implementation authorisation.
*(v0.5: the sentence above is v0.4 wording. The implementation was subsequently authorised, completed and accepted (D-12).)*

### 18.3 Pending pilot
- `η_a` for each arm A–D: LR pilot, §16.4. **Resolved (v0.5): 0.03 for A, B, C and D (D-22, 2026-11-11).**
- The common `max_epochs`: budget pilot, §16.5, which runs after the LR pilots. **Still pending;** protocol approved (§16.5.1, D-23), implementation and execution not yet authorised. *(v0.6 addendum, D-31 addendum: the last clause is v0.5 wording. The pilot has since been implemented (D-27, D-29) and executed (user-reported, 2026-11-12); the common `max_epochs` is still pending its §16.5 selection and human approval.)* **Resolved (v0.7): 26 for every arm (D-32, 2026-11-12).**

Neither may be chosen before its pilot.

### 18.4 Mapping to `MODEL_SPECIFICATION_TEMPLATE.md`
- Identity / role: §1.
- Architecture: §2.
- Mathematics: §3, §5.
- Tensor contract: §4.
- PyTorch policy: §19 below.
- Initialisation: §6.
- Forward: §4.2.
- Configuration: §7.
- Numerical stability: §12, §13.
- Complexity: §19.
- Diagnostics: §11.
- Tests: §14, §15.
- Checkpoints: §10.
- Generation: §9.
- Scope: §1.3, §20.
- Acceptance: §17.
- Open decisions: §18.

## 19. PyTorch policy and complexity (template completeness)

**Permitted:**
- `nn.Module`, `nn.Embedding`, `nn.Linear(bias=False)` on 3-D input, `nn.Tanh`;
- [BN] an `nn.BatchNorm1d` subclass applied to `[B·G, H]`, and `F.batch_norm` (eval);
- `reshape`, `torch.optim.SGD`;
- in tests: `F.conv1d`, `gradcheck`, `functional_call`.

**Prohibited:**
- `nn.Conv1d` or `F.conv1d` **in the model** (they would hide the merge mechanism; allowed only as a test reference);
- `view` in `merge_pairs`; conditional `squeeze`;
- per-position BatchNorm statistics; PyTorch default initialisation; hidden biases;
- momentum, Adam, lr schedulers; dropout and weight decay;
- the global RNG in model code;
- float64 on device.

**Complexity:**
- Multiply-adds per window, forward: level 1: 4 × 20 × 68 = 5,440; level 2: 2 × 136 × 68 = 18,496; level 3: 9,248; output: 1,836. Total ≈ 35k, about 3× the k = 3 MLP. About 3× that for forward plus backward.
- Memory per batch of 32 is negligible.
- Recalibration costs about L partial passes over the training set.
- Not timed; it will be measured during implementation and reported as an observation.

## 20. File structure
*(v0.4 heading, historical: "Proposed file structure (implementation not authorised)".)*

**Current status (v0.5):** the files below exist and Model 5 is accepted (D-12): `models/hierarchical/wavenet.py` (1,037 lines), `configs/models/wavenet.yaml`, `configs/models/wavenet_batchnorm.yaml`, `tests/models/hierarchical/wavenet_reference.py`, `test_wavenet_reference.py` and `test_wavenet.py`. Study code lives in `training/wavenet_study/` (the closed LR-pilot scripts and, as future implementation targets not yet created, the budget-pilot files named in the protocol). The listing below is the v0.4 plan, retained verbatim; its "(currently empty)" annotation describes the state before implementation.

```
models/hierarchical/wavenet.py            (currently empty) model, configs, merge_pairs, LevelBatchNorm1d,
                                           calibrate_batchnorm, evaluate_nll, per_unit_saturation,
                                           train_wavenet, run_config, from_checkpoint
models/hierarchical/wavenet_spec.md       this specification
configs/models/wavenet.yaml               name: wavenet,            normalization: none
configs/models/wavenet_batchnorm.yaml     name: wavenet_batchnorm,  normalization: batchnorm
tests/models/hierarchical/wavenet_reference.py       independent reference (torch/math/typing only)
tests/models/hierarchical/test_wavenet_reference.py  §14
tests/models/hierarchical/test_wavenet.py            §15
```

Not touched: Models 1–4 and their tests, shared utilities, existing specifications, repository configuration.

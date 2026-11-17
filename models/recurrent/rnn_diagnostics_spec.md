# Model 6 — Vanilla RNN: Research Diagnostics Specification (Stage 5)

## Status (current, v0.3)

1. **DRAFT for review.** Nothing in this document is approved. It does not authorise implementation, probe construction, measurement, training, pilots or experiments.
2. **Governs:**
   - the research diagnostics M6-H1, M6-G1 – G3 and M6-C1;
   - the reporting of M6-O1 – O3;
   - the open Stage 5 decisions M6-DGN1 – DGN14 of `rnn_spec.md` §18.4.
3. **Depends on:** `models/recurrent/rnn_spec.md` v0.2, **APPROVED** (D-44; file SHA-256 `3961afdd…97f1`). Its §7.2, §12 and §13 are binding here. The one intended deviation is marked **AMENDMENT PROPOSED** (§4.6). Until a recorded decision adopts it, the accepted text applies.
4. **Governing decisions:** D-38 (agenda), D-42 (diagnostic framework), D-44 (approval of `rnn_spec.md`). Record SHA-256 `dc787619…1be6`, last verified 2026-11-14 in this session; latest decision D-44.
5. **Stage 4 dependency: Stage 4 is NOT APPROVED.** The record contains no Stage 4 decision; the training specification is an unapproved workspace draft. This document uses only symbols for training quantities, and takes nothing from that draft:
   - S = optimisation steps in epoch 1;
   - E = `max_epochs`;
   - the step and epoch-end hooks (§5.2);
   - resume truncation (§5.2);
   - the failure handling referenced by O3 (§4.6).
6. **Probe:** **not constructed and not accessed.** Its row IDs, stratum sizes and SHA-256 are produced only by the authorised construction step (§3.1).
7. **Test split:** never used. The only data counts quoted here (§3.2, §4.5) come from dev-only metadata of the closed WaveNet study.
8. **Proposed location:** `models/recurrent/rnn_diagnostics_spec.md` (not created).

**Version:** 0.3 (2026-11-14)

### Labels

**[T]** mathematics; **[I]** implementation requirement; **[O]** observation from existing records; **[E]** estimate or assumption, not a measurement; **[H]** hypothesis; **ACCEPTED (D-xx)**; **APPROVED (D-44)**; **PROPOSED**; **AMENDMENT PROPOSED**; **OPTIONAL**; **PENDING**.

### Amendment log

| Version | Date | Change | Decision |
|---|---|---|---|
| v0.1 | 2026-11-14 | Initial draft | — |
| v0.2 | 2026-11-14 | Per-stratum primary aggregation; weights derived; G2 numerical rule; corrected G1 cost; 4-point spectral schedule; isolation (hook sanitisation, tests I1 – I8); reduced outputs; G3 population option (a) full training split | — |
| v0.3 | 2026-11-14 | (1) **G3 stays on the weighted probe (D-42).** Option (a) and identifier M6-DGN20 are withdrawn. A split-half reliability analysis (M6-DGN21) is added, with deterministic halves, full per-half reporting, no pass/fail threshold, and an explicit distinction between the probe estimate and the exact full-training decomposition. (2) **C1:** F₈ primary, F₇ secondary; the exact F₈ count is PENDING until probe construction; the v0.2 bounds on F₈ are removed. (3) **O3:** descriptive categories kept; separation from Stage 4 stated once (§4.6). (4) **Consistency:** the G1 start-position table is restored to the primary outputs, as D-42 requires ("by distance and start position"); duplicated rules removed; every new item labelled. | — |

---

## 1. Purpose and scope

**Purpose.** Measure exactly and reproducibly what the accepted research questions need, without changing training:
- **M6-RQA:** temporal propagation and its W-versus-D mechanism;
- **M6-RQB:** effective context use;
- **M6-RQC:** gradient reach versus behavioural reach over training.

**Out of scope:**
- training-failure handling (Stage 4);
- experiment design (M6-EXP1, EXP2; Later);
- implementation;
- probe construction (a separate authorised step).

### 1.1 Mapping to the research questions (canonical: `rnn_spec.md` §2.3)

| Diagnostic | M6-RQA | M6-RQB | M6-RQC | Hypotheses |
|---|---|---|---|---|
| H1 hidden-state dynamics | regime by position and over training | — | context | HA1 (transient), HA3 |
| G1 Jacobian products, with ω (W only) and δ̂ (D only) | **primary** | — | context | HA1, HA3 |
| G2 loss-projected attenuation | **primary** | — | supporting | [E] alignment topic |
| G3 recurrent-gradient decomposition (probe estimate) | — | — | **primary (gradient reach)** | HC1, HC2 |
| C1 BOS-restart | — | **primary** | **primary (behavioural reach)** | HB1, HB2, HC2 |
| O2 W spectrum, per-position dev NLL | HA2 | supporting | supporting | HA2 |
| O1 – O3 | optimisation conditions under which the questions are answered | | | |

---

## 2. Accepted framework and its status

| Element | Source | Status here |
|---|---|---|
| H1: RMS, saturation \|h\| > 0.97, mean and mean-square tanh′; position × length stratum; position 1 separate; training probe | D-42; `rnn_spec.md` §12.2 | ACCEPTED; strata PROPOSED (§3) |
| G1: exact P_{t,s}; φ = ‖P‖_F/√H **by distance and start position**; exact spectral norm at selected points; ω(Δ); no estimator; probe | D-42; §12.3 | ACCEPTED; reporting PROPOSED (§4.2) |
| G2: R = ‖Pᵀr‖/‖r‖; unweighted and magnitude-weighted; only exact or near-exact zeros of r excluded, and counted; probe | D-42; §12.4 | ACCEPTED; exclusion rule PROPOSED (§4.3) |
| G3: T_{t,k}; signed projection shares summing to 1 (primary); cancellation index; norm mass; two-pass algorithm; **probe** | D-42; §12.5 | ACCEPTED; weighting and reliability analysis PROPOSED (§4.4) |
| C1: BOS-restart; affected t ≥ d + 2; one fixed target set across d; position strata; dev; selected checkpoints; shared-start algorithm | D-42; §12.6 | ACCEPTED; target sets PROPOSED (§4.5) |
| O1 per step; O2 per epoch; O3 categories defined before any run | D-42; §12.7 | ACCEPTED; O3 PROPOSED (§4.6) |
| Float64 CPU copy; no retained graphs; no RNG use; training bit-identical with diagnostics on and off; analytic diagnostics verified against autograd; G3 sum equals the gradient | D-42; §12.1; M6-IMP3; §13.2 | ACCEPTED / APPROVED |
| G3 and C1 at the same points; points defined in steps, possibly within epoch 1; G1 spectral on a subset | §12.8 | APPROVED (D-44) |
| §7.2 initial-statistics procedure | §7.2 (M6-DGN8) | APPROVED (D-44) |

**Clarifications (no change of meaning).**
1. G2 is defined from step 1 onward (V ≠ 0 after the first update). D-42's "from epoch 1" therefore includes points within epoch 1.
2. Running G3 and C1 at every evaluation point is a permitted reading of D-42's "selected epochs / selected checkpoints": all points are selected.

---

## 3. Probe (M6-DGN1, M6-DGN2, M6-DGN15)

### 3.1 Definition and construction (PROPOSED)

- **Population:** rows of the training `NextTokenSequences`, in split order, i = 0 … N − 1, N = 25,640. Duplicate names are distinct rows, as in training.
- **Length strata** by valid steps n = L + 1: S1 = {2 – 5}, S2 = {6 – 7}, S3 = {8 – 9}, S4 = {10 – 16}. Fixed before any model output exists. The same strata are used for H1, G1, G2, G3 and C1.
- **Allocation:** n_ℓ = min(256, N_ℓ), so the whole stratum is taken if it is smaller.
- **Draw:**
  1. Let R_ℓ be the stratum's row indices, sorted ascending.
  2. `g = make_generator(derive_seed(42, "m6-probe"))`, used for S1, S2, S3, S4 in that order.
  3. Rows of stratum ℓ = `sorted(R_ℓ[torch.randperm(len(R_ℓ), generator=g)[:n_ℓ]])`.
  - Seed 42 is the split seed: the probe is a property of the split. **One probe for every run and seed.**
- **Design weight:** w_ℓ = N_ℓ / n_ℓ.
- **Half label (for §4.4):** within each stratum, the sorted probe rows at positions 0, 2, 4, … form half A and those at positions 1, 3, 5, … form half B. **No further random draw.**
- **Record:** `probe.json`, canonical JSON with sorted keys:
  - schema `"m6_probe/1"`;
  - split fingerprint (seed, policy, fractions, N, M);
  - strata;
  - N_ℓ, n_ℓ, w_ℓ;
  - rows, names and half labels;
  - SHA-256.
- **Construction step (PENDING authorisation):** builds the probe and records:
  - its SHA-256;
  - N_ℓ and n_ℓ;
  - the exact dev count of F₈ (§4.5);

  all in a D-entry, before the initialisation gate. Nothing is constructed by this document.

### 3.2 Expected stratum shares (dev-only; for planning only)

Source: `results/experiments/wavenet_study/final/analysis/dev/bucket_definitions.json` (SHA-256 `f0a57b48…`), category `real_context_length`, r = min(t − 1, 8). Dev targets at position t equal dev names with n ≥ t. Hence the exact dev names by n [O]:

| n | 2 | 3 | 4 | 5 | 6 | 7 | ≥ 8 |
|---|---|---|---|---|---|---|---|
| Dev names | 0 | 8 | 52 | 262 | 740 | 987 | 1,147 |

- Dev shares: S1 10.1%, S2 54.0%, S3 ∪ S4 35.9%. The dev split between S3 and S4 is not given by this metadata.
- Training shares are expected to be similar under the random group split [E]. They are **not** used for any decision; the exact N_ℓ come from construction.

### 3.3 Weights and estimators [T]

Within stratum ℓ the rows are a simple random sample without replacement, so the inclusion probability is π_ℓ = n_ℓ/N_ℓ and the weight is w_ℓ = 1/π_ℓ.

- **Totals:** Σ_probe w x is unbiased (Horvitz–Thompson) for the population total Σ_pop x.
- **Means, ratios and shares** (Σ w x / Σ w y): consistent ratio estimators, with bias of order 1/n_ℓ.
- **Weighted quantiles:** consistent; the weighted empirical CDF is unbiased pointwise.
- **Efficiency:** the effective sample size of a pooled estimate is n_eff = (Σw)²/Σw². With illustrative N_ℓ ≈ (2,583; 13,855; 8,000; 1,200) [E], n_eff ≈ 637 of 1,024. Each stratum alone keeps n_ℓ.

### 3.4 Aggregation (PROPOSED, M6-DGN15)

| Aggregation | Estimates | Use |
|---|---|---|
| **Per stratum** (unweighted within ℓ) | Behaviour conditional on name length | **Primary for H1, G1, G2** |
| **Population-weighted** (w_ℓ) | Training-population quantities | Secondary for H1, G1, G2; **primary for G3** |
| Equal-stratum averaging | No defined population | **Not used** |

- **Why per stratum is primary for H1, G1 and G2 [T].** A pooled curve over Δ changes its stratum composition as Δ grows: Δ ≥ 9 exists only in S4. It can therefore bend for compositional rather than dynamical reasons.
- **Why G3 is weighted.** G3 decomposes the learning signal of the training objective, so the probe must be reweighted to the training population.

---

## 4. Definitions

Notation as `rnn_spec.md` §4. Everything is computed on the float64 CPU copy (§6) at the parameters θ_p of evaluation point p (§5), on valid positions only. A pair (t, s) means 1 ≤ s < t ≤ n_β.

**Conventions:**
- Counts (elements and names) accompany every statistic. There is no minimum cell size and no suppression.
- **Weighted quantile** at level q: the smallest x_(j), in ascending order, with cumulative weight ≥ q·Σw. Ties are broken by (stratum, row, t, s). Quantiles reported: q ∈ {0.25, 0.5, 0.75}.
- Natural logarithms; log(0) = −inf is stored and counted.

### 4.1 H1 — hidden-state dynamics (ACCEPTED; reporting PROPOSED)

Per position t = 1 … 16 and stratum ℓ, over A_{t,ℓ} = {(β, j): β ∈ ℓ, n_β ≥ t}:

| Quantity | Definition |
|---|---|
| ρ | (mean h²)^{1/2} |
| S | mean 1[\|h\| > 0.97] |
| d̄ | mean (1 − h²) |
| d² | mean (1 − h²)² |

- Name count and element count are reported.
- Position 1 is flagged: one vector shared by every name (M6-E4).
- **Primary:** per (t, ℓ). **Secondary:** per t, weighted.
- **H1 on dev (M6-DGN7): not measured** (PROPOSED). Same distribution of names; no research question needs it. OPTIONAL at E_sel.

### 4.2 G1 — exact Jacobian products (ACCEPTED; reporting PROPOSED)

| Quantity | Definition | When |
|---|---|---|
| φ_{t,s} | ‖P_{t,s}‖_F/√H | every point |
| σ_{t,s} | ‖P_{t,s}‖₂, exact SVD | E_sel |
| ω(Δ) | ‖W^Δ‖_F/√H, Δ = 1 … 15 | every point (ACCEPTED) |
| δ̂_{t,s} | ∏_{k=s+1}^{t} (g_W² · (1/H) Σ_j (1 − h_{k,j}²)²)^{1/2}, g_W = ‖W‖_F/√H | every point (PROPOSED, M6-DGN18) |

[T] For independent Haar matrices with RMS singular value g_W, E‖D_t W_t ⋯ D_{s+1} W_{s+1}‖_F²/H = ∏ g_W² mean(D_k²). δ̂ is the product norm that the observed derivatives alone would give under an isotropic W. At θ₀ it equals the §7.2 Level-2 prediction φ̂.

**Reported** (all primary):
- (i) per (stratum, Δ): quantiles of log φ, and of log σ at E_sel, with quantiles of log δ̂;
- (ii) **per (start s, Δ), population-weighted:** quantiles of log φ, as D-42 requires;
- (iii) ω(Δ).

**Secondary:** pooled per Δ, weighted, with the composition caveat (§3.4).

The comparison of log φ, log ω and log δ̂ is descriptive, not an additive decomposition (§12).

### 4.3 G2 — loss-projected attenuation (ACCEPTED; numerical rule PROPOSED, M6-DGN5)

r_t = Vᵀ g_t with g_t = p_t − e_{y_t}. For s ≤ t: R_{t,s} = ‖P_{t,s}ᵀ r_t‖ / ‖r_t‖.

**Rule (no threshold parameter):**
1. p_k = exp(log_softmax(z)_k) in float64. Set g_k = p_k for k ≠ y, and g_y = −Σ_{k≠y} p_k. This avoids cancellation when p_y ≈ 1.
2. r = Vᵀ g. **Exclude the target iff every component of r is exactly 0**, and count it. This happens at V = 0 and, for a full-rank V, not otherwise, since p > 0.
3. Otherwise compute u = r/‖r‖ with a scaled norm (divide by max_k |r_k| first), so neither underflow nor subnormals arise. Run the chain on u: R_{t,s} = ‖P_{t,s}ᵀ u‖.

[T] R is scale-invariant. A small but non-zero ‖r‖ (a well-predicted target) gives a valid R and is **kept**; excluding it would bias the unweighted aggregate toward poorly predicted targets.

**Reported:**
- primary per (stratum, Δ): quantiles of log R;
- magnitude-weighted R̄(Δ) = Σ ‖r_t‖ R_{t,s} / Σ ‖r_t‖ (weighted by w_ℓ in pooled form);
- exact-zero exclusion counts;
- minimum ‖r_t‖.

[T] For a uniformly random direction of r, E R² = φ², so log R − log φ measures alignment (the [E] topic). Computed at every point with σ ≥ 1.

### 4.4 G3 — recurrent-gradient decomposition, probe estimate (ACCEPTED; reliability analysis PROPOSED, M6-DGN21)

**Population and estimand.** G3 uses the **stratified 1,024-name training probe with population weights** (D-42). With w_β = w_ℓ(β):

    T_{t,k} = w_β · (D_k P_{t,k}ᵀ r_t) h_{k−1}ᵀ,   2 ≤ k ≤ t ≤ n_β,   Δ = t − k ∈ {0, …, 14}
    Ĝ = Σ T_{t,k} = ∂L̂/∂W,   L̂ = Σ_β w_β Σ_t ℓ_{β,t}   [T]
    π̂(Δ) = Σ_{t−k=Δ} ⟨T_{t,k}, Ĝ⟩_F / ‖Ĝ‖_F²,   Σ_Δ π̂(Δ) = 1   [T]
    π̂⁺(Δ), π̂⁻(Δ): the same sums over pairs with ⟨T, Ĝ⟩ > 0 and < 0 respectively
    κ̂_c = ‖Ĝ‖_F / Σ ‖T_{t,k}‖_F,   Q̂(Δ) = Σ_{t−k=Δ} ‖T_{t,k}‖_F,   Π̂(Δ ≥ m) = Σ_{Δ≥m} π̂(Δ)

- **Estimate versus exact decomposition [T].**
  - Ĝ is an unbiased (Horvitz–Thompson) estimator of the full-training-set gradient G_train at θ_p, scaled to the population total.
  - π̂, κ̂_c and Π̂ are **ratio estimates** of the shares of the exact decomposition of G_train. They are consistent, not unbiased, and carry probe sampling error.
  - In particular E‖Ĝ‖² = ‖G_train‖² + tr Var(Ĝ). When G_train is small relative to per-name gradients, as is expected later in training, Ĝ and therefore π̂ can be dominated by sampling noise.
  - Every G3 output is therefore labelled **"probe estimate"**. The exact full-training decomposition is **not computed**. No statement about G_train is made except through these estimates and their reliability analysis.
- **Signs:** π̂(Δ) is reported signed. π̂⁺ and π̂⁻ are reported separately; absolute values are never used. A negative π̂(Δ) means that distance's contributions oppose Ĝ.
- **Long-distance share:** Π̂(Δ ≥ 3). The boundary 3 comes from M6-HC1 ("distances ≤ 2").
- **Not computed at σ = 0**, where G ≡ 0 (M6-E4).

**Split-half reliability analysis (PROPOSED, M6-DGN21).**
- **Halves:** fixed at construction by alternating positions within each stratum's sorted probe rows (§3.1). No new random sampling. Each half is a stratified sample with m_{ℓ,h} rows per stratum (128 each when n_ℓ = 256).
- **Half weights:** w_{ℓ,h} = N_ℓ / m_{ℓ,h}. Each half therefore estimates the same population total independently of the other half's rows.
- **Per half h ∈ {A, B}**, at every point, computed exactly as above with half weights and the half's own Ĝ_h:
  - π̂_h(Δ), π̂⁺_h(Δ), π̂⁻_h(Δ);
  - κ̂_{c,h};
  - Π̂_h(Δ ≥ 3);
  - ‖Ĝ_h‖_F.
- **Between-half agreement**, descriptive only:
  - the gradient-direction cosine cos(Ĝ_A, Ĝ_B) = ⟨Ĝ_A, Ĝ_B⟩_F / (‖Ĝ_A‖_F ‖Ĝ_B‖_F);
  - per Δ, the half difference π̂_A(Δ) − π̂_B(Δ) and the half-sample spread s(Δ) = |π̂_A(Δ) − π̂_B(Δ)|/2 (the two-group form of the random-group variance estimator; approximate, because the halves are systematic subsamples of a random sample);
  - the same for Π̂(Δ ≥ 3) and κ̂_c.
- **No pass/fail threshold.**
  - A quantity is labelled **"sign not reproduced across halves"** when the two halves give opposite signs, a qualitative descriptor.
  - Every G3 figure and table shows the full-probe estimate together with both halves.
  - Directional statements about G3 (including those bearing on HC1 and HC2) are made only for quantities whose **sign** agrees in both halves. Any other quantity is reported as **not reproduced**, never omitted.
  - cos(Ĝ_A, Ĝ_B) is shown next to every G3 result. A negative cosine is reported as "halves estimate opposed gradient directions". Lower agreement indicates the noise-dominated regime of the estimate above. No cosine value is used as a cut-off.
- **This is a measurement limitation, not a change of research question:** M6-RQC still concerns the learning signal of the training objective, estimated from the probe.

### 4.5 C1 — BOS-restart (ACCEPTED; target sets PROPOSED, M6-DGN4)

ΔNLL_t(d) = −log p^{(d)}[y_t] + log p_t[y_t], for dev targets with t ≥ d + 2, where d is the number of characters kept. Targets with t ≤ d + 1 are exact no-ops and excluded (M6-E6).

**Dev target counts by position [O]** (dev-only metadata, §3.2): positions 1 – 8 have 3,196, 3,196, 3,196, 3,188, 3,136, 2,874, 2,134 and 1,147 targets; positions ≥ 9 have **734** together; the total is 22,801.

**Fixed target sets:**
- **Primary: F₈ = {t ≥ 10}, d = 1 … 8.** d = 8 is required by M6-HB1 ("d ≥ 8"), and t ≥ 10 is the smallest set affected for every d ≤ 8. This is the single fixed set across d that D-42 requires.
  - **Its exact size is PENDING:** it is counted and recorded at the authorised construction step (§3.1). It is not inferred here.
- **Secondary: F₇ = {t ≥ 9}, d = 1 … 7**, exactly 734 targets [O].
- **Archival:** every other F_D = {t ≥ D + 2} is computable from the stored per-target values, never selected after seeing results.
- **Matrix:** per (t, d, stratum), sums and counts. Positions t are the position strata of D-42.

**Uncertainty (PROPOSED), clustered by name.** For a set F and a given d, with N_c names, name-level sums s_i and counts m_i:
- mean = Σ s_i / Σ m_i;
- SE² = (N_c/(N_c − 1)) · Σ_i (s_i − mean·m_i)² / (Σ m_i)².

Target and name counts are always reported. **Per-target values are mandatory** (§8).

**Interpretation:** see §12, items 5 and 6.

**OPTIONAL (M6-DGN19):** the data bound Ĥ(Y | last d chars) − Ĥ(Y | last d chars, t) on training counts, d ≤ 4, which indicates how much absolute position can matter in the data.

### 4.6 O1 – O3: Stage 5 reporting (separate from Stage 4)

| | Stage 4 (training procedure; unapproved draft) | Stage 5 (this document) |
|---|---|---|
| Question | What does training **do** when something fails? | How are runs and epochs **described** in the research results? |
| Examples | Raise on non-finite values; stop, record, no retry | Classify, list and describe; never alter training |
| Effect on data | Determines whether a run completes | Determines only whether a run enters research aggregates |

**O1 and O2** are collected by the training procedure. Stage 5 fixes their reporting:
- **per-position dev NLL (M6-DGN11):** each t = 1 … 16 and each stratum, sums and counts;
- **W spectrum:** all 128 singular values per epoch, float64.

**O3 (M6-DGN9, PROPOSED):**

| Category | Definition | Use |
|---|---|---|
| Failed run | Ended by Stage 4's failure handling, whatever categories Stage 4 adopts | Excluded from research aggregates; listed with its reason |
| Epoch events | Full-dev NLL rose between consecutive epochs; running train NLL rose between consecutive epochs | Descriptive; never a reason to exclude data |
| Gradient-norm profile | Per epoch: max, median and max/median of the per-step global gradient norm | Descriptive numbers |

**AMENDMENT PROPOSED:** the "gradient-spike multiple" in the `rnn_spec.md` §12.7 candidate list is withdrawn. No spike threshold exists.

---

## 5. Cadence and alignment (M6-DGN3, M6-DGN12, M6-DGN14)

### 5.1 Evaluation points (PROPOSED)

| Set | Completed steps σ | Diagnostics |
|---|---|---|
| P₀ | 0 | H1, G1-F, G1-σ, ω, δ̂ (G2, G3 undefined; C1 not run) |
| P_early | 2^j < S (1, 2, 4, …, 512 if S = 802) | H1, G1-F, ω, δ̂, G2, G3 (with halves), C1 |
| P_epoch | the end of each epoch k = 1 … E | H1, G1-F, ω, δ̂, G2, G3 (with halves), C1 |
| E_sel | 0; end of epoch 1; best epoch (post-hoc); end of epoch E | + G1-σ |

**Reasons:**
- **Log-spaced early points:** M6-HC1 concerns early training; the first steps change V and c only (M6-E4).
- **Every epoch end:** gives the training timeline.
- **E_sel:** the start, the end of the fastest phase, the selected model and the end, which is sufficient for the worst-case/average comparison. HA2 concerns W's spectrum, which O2 records every epoch.
- **OPTIONAL:** spectral at epochs 2^j ≤ E.
- **Pilots:** research diagnostics are defined for research runs; pilot use is decided by Stage 4 or Later. Recommendation: O1 – O3 only.

### 5.2 Hook and resume (M6-DGN14, PROPOSED)

- **Mechanism:** a model-local `RNNDiagnosticsPlan`, because `DiagnosticsSchedule` only expresses "every n steps".
- **Placement:**
  - step points after `optimizer.step()` and the post-step checks;
  - epoch-end points after evaluation and before any checkpoint write.
- **Interface:** `train_rnn(..., diagnostics=plan)`. The plan receives a read-only model reference and the counters, and returns nothing.
- **Resume:** diagnostic records are keyed by `point_id` and truncated together with the history. **Cross-stage requirement for Stage 4.**

### 5.3 Alignment (M6-DGN12, PROPOSED)

- Current parameters at each point.
- `param_sha256`, over the float32 bytes of (U, W, b, V, c) in that order, is recorded before and after every point.
- Best-epoch spectral norms are computed post-hoc from `model_best.pt`, only if its hash equals the recorded epoch-end hash.

---

## 6. Numerical and precision policy (part of M6-DGN6)

- **Computation:** float64 on CPU. States are recomputed with the shared `_recurrence` on the copy, never cast from float32.
- **Accumulation:** float64; cross-name totals with `math.fsum`.
- **Stable G2 and G3 inputs:** per §4.3 (one rule, used by both).
- **Storage:** float64. JSON uses round-trip `repr`; arrays go in `.npz`; nothing is rounded.
- **Tolerances:** `rnn_spec.md` §13.1, criterion A.

---

## 7. Algorithms and cost

### 7.1 Algorithms

| Diagnostic | Algorithm (float64 copy) |
|---|---|
| H1 | One `_recurrence` over the probe; masked reductions |
| G1-F | Per name and start s: P ← D_{s+1}W (a scaling); for t = s + 2 … n: P ← D_t(W P), one H × H product. Starts may be stacked |
| G1-σ | `torch.linalg.matrix_norm(P, ord=2)` per pair |
| ω | Repeated products W^Δ |
| δ̂ | Products of per-step factors from the probe states |
| G2 | Per target: u from §4.3; for k = t … s + 1: u ← Wᵀ(D_k u) |
| G3 | Pass 1: Ĝ, Ĝ_A, Ĝ_B by weighted BPTT. Pass 2: the G2 vector chains; per position compute Ĝ h_{k−1} and Ĝ_h h_{k−1} once; ⟨T, Ĝ⟩ = w_β (D_k u)ᵀ(Ĝ h_{k−1}) and likewise per half with half weights |
| C1 | Shared-start restarts j = 3 … n per dev name, batched |

### 7.2 Arithmetic [T] and illustrative counts [E]

Per name with n steps:
- pairs: n(n − 1)/2;
- G1-F products: (n − 1)(n − 2)/2, at H³ = 2,097,152 MACs each;
- one SVD per pair, ≈ (8/3)H³ ≈ 5.6 MFLOP each plus O(H²) iterations.

Probe illustration (dev length shares; S3 assumed evenly split over n = 8, 9; S4 over n = 10 – 13 in ratio 4 : 3 : 2 : 1) [E]: ≈ 29,500 pairs and ≈ 22,600 products for 4 × 256 names.

| Diagnostic | Per point [E] |
|---|---|
| G1-F | ≈ 47 G MACs |
| G1-σ | ≈ 29,500 SVDs ≈ 165 GFLOP |
| G2 | ≈ 0.5 G MACs |
| G3 (full probe + both halves) | ≈ 1 – 2 G MACs (chains shared; per-half inner products O(H) per pair plus O(H²) per position) |
| C1 | ≈ 1.5 G MACs |
| H1 | ≈ 0.15 G MACs |

### 7.3 Memory [T] and runtime

- **Memory:** at most 15 H × H float64 matrices per name (≈ 2 MB) for G1; three H × H gradient estimates for G3; one name's states. Well under 100 MB.
- **Runtime: not measured.** Small LAPACK SVDs run far below peak.
- **M6-DGN13 (PROPOSED):** one timing measurement of each diagnostic on the untrained model (seed 1000), under its own authorisation, after probe construction and before research runs. It informs feasibility only; no automatic change.

---

## 8. Outputs (M6-DGN6, PROPOSED)

**Study level:** `probe/probe.json` (§3.1); `diagnostics_schema.json` (`"m6_diagnostics/3"`).

**Per run** (`<run_dir>/diagnostics/`):

| File | Content |
|---|---|
| `plan.json` | Resolved points, probe SHA-256, schema, thread counts |
| `points.jsonl` | Per point: id, σ, completed epochs, kind, `param_sha256` and RNG-state hash (before and after), diagnostics run, run-time check values, seconds, `valid`, error text |
| `primary.jsonl` | Per point, the primary aggregates of §4: H1 per (t, ℓ); G1 per (ℓ, Δ) and per (s, Δ), ω, δ̂; G2 per (ℓ, Δ), R̄, exclusions; G3 full-probe and per-half π̂, π̂±, Π̂(Δ ≥ 3), κ̂_c, ‖Ĝ‖, cos(Ĝ_A, Ĝ_B), s(Δ); C1 F₈ and F₇ curves with clustered SEs and counts |
| `raw/point_{id:04d}.npz` | Per probe pair: row, stratum, half, s, t, log φ, log σ (E_sel), log R, log δ̂, ‖D_k u‖·‖h_{k−1}‖, ⟨T, Ĝ⟩, ⟨T_h, Ĝ_h⟩. Per dev target: row, t, stratum, ΔNLL for every affected d |
| `o3.json` | Run-level classification |

- **Derived later from `raw/`** (not stored separately): pooled curves, per-stratum G3, archival F_D sets, any other table.
- **Storage [E]:** about 1 – 2 MB per point, about 50 – 80 MB per run.

---

## 9. Primary analysis set (M6-DGN16, PROPOSED)

| RQ | Predeclared summaries |
|---|---|
| M6-RQA | A1: per-stratum log φ(Δ) with ω and δ̂, at P₀, the end of epoch 1 and the best epoch; the (s, Δ) table at the same points. A2: H1 saturation and d² per (t, ℓ). A3: log σ against log φ at E_sel. A4: W singular values over epochs |
| M6-RQB | B1: F₈ (primary) and F₇ curves at the best epoch, with clustered SEs and counts. B2: the (t, d) matrix by stratum at the best epoch |
| M6-RQC | C1: Π̂(Δ ≥ 3) and π̂(Δ) over all points (log-step axis), **full probe with both halves and cos(Ĝ_A, Ĝ_B)**. C2: F₈ and F₇ curves over all points. C3: both on one axis per seed |

Everything else is archival. M6-HC2 is assessed descriptively (§12).

---

## 10. Validation

### 10.1 Implementation tests (additions to `rnn_spec.md` §13.2 M1 – M11 and T5; PROPOSED)

| # | Test | Criterion |
|---|---|---|
| D1 | Probe: deterministic rows and SHA-256; strata, census rule, weights, half labels on a fixture | exact |
| D2 | Weighted quantile and tie order; Horvitz–Thompson total equals the population total in expectation on an enumerable fixture | exact |
| D3 | Plan resolution for several S and E (S a power of two; E = 1; best = final) | exact |
| D4 | Alignment: epoch-end hash equals the checkpoint hash; post-hoc best refuses a mismatch | exact |
| D5 | G3: Σ T equals autograd ∂L̂/∂W with weights, for the full probe and for each half; each Σπ̂ = 1 | A |
| D6 | G2 rule: r = 0 excluded; p_y = 1 − 10⁻²⁰ handled; components near 10⁻²⁰⁰ give the same R as rescaled to order 1 | A |
| D7 | δ̂ at θ₀ equals the §7.2 φ̂ | A |
| D8 | C1: F₈ and F₇ membership; clustered SE against a hand computation; matrix totals | exact / A |
| D9 | Stored values reload bit-identically | exact |
| D10 | Injected exception and failed check: point `valid: false`; training bitwise identical to a run without diagnostics | bitwise |
| D11 | Halves partition each stratum; half weights N_ℓ/m_{ℓ,h}; cos and s(Δ) against hand computation | exact / A |

### 10.2 Run-time checks (every point; mandatory)

| Check | Pass |
|---|---|
| ‖Σ T − Ĝ_autograd‖_F ≤ 1e‑9 ‖Ĝ_autograd‖_F, full probe and each half | A |
| \|Σ_Δ π̂(Δ) − 1\| ≤ 1e‑9 / κ̂_c, full probe and each half | A |
| φ ≤ σ(1 + 1e‑12) and R ≤ σ(1 + 1e‑12) for every pair, at E_sel | [T] |
| A j = 2 restart per dev name reproduces the full logits | A (M6-E6) |
| H1 counts equal the valid cells | exact |
| `param_sha256` and RNG-state hash unchanged across the call | exact |

These checks verify the computation; they are not reliability thresholds (§4.4).

**On failure (M6-DGN17, PROPOSED):** the point is recorded `valid: false`, training continues, and results that use the point require a recorded human review (precedent: category B, D-23/D-26).

### 10.3 Initial-statistics seeds (M6-DGN10, PROPOSED)

Seeds 1000 – 1009. The relative standard error of an across-seed SD is then ≈ 1/√18 ≈ 0.24 [T, normal approximation]. Alternative: 20 seeds (≈ 0.16), at double the cost.

---

## 11. Isolation (PROPOSED)

### 11.1 Requirements

| Risk | Requirement |
|---|---|
| Parameter values, dtype, device | Read the training model only via `deepcopy` under `torch.no_grad()`; convert only the copy |
| `.grad` copied by `deepcopy` | Set the copy's `.grad = None` |
| **Hooks copied by `deepcopy`** | Clear forward, forward-pre, backward, backward-pre and state-dict hooks on every submodule of the copy; assert they are empty |
| Autograd graph | Copied parameters are leaves; autograd (§10.2) only on the copy, inside `torch.enable_grad()`, restored by context manager |
| RNG | No generator created or consumed; RNG-state hash recorded before and after |
| Grad mode, threads, deterministic flags | Asserted unchanged after each call |
| Optimiser, sampler | Never passed to diagnostics |
| Files | Written only under `diagnostics/` |
| Exceptions | Caught at the hook boundary (`Exception` only) and recorded |

### 11.2 Required tests (implementation stage; not run)

| # | Test |
|---|---|
| I1 (= `rnn_spec.md` T5) | Training with and without a full plan (non-empty P₀, P_early, P_epoch, E_sel): bitwise identical parameters, step records, history, checkpoints and final global RNG state |
| I2 | Per call, before and after: parameter and `.grad` bytes; `requires_grad`; dtype and device; `model.training`; hook dictionaries; `torch.is_grad_enabled()`; `torch.get_rng_state()`; `torch.get_num_threads()`; `torch.get_num_interop_threads()`; `torch.are_deterministic_algorithms_enabled()` |
| I3 | Copy independence: changing the copy leaves the original unchanged; the copy has no hooks, no `.grad`, float64, CPU |
| I4 | A counting forward hook on the training model is not triggered by a diagnostic call |
| I5 | After the autograd check, the training `.grad` is bitwise unchanged and no new `grad_fn` is attached to a training tensor |
| I6 | Failure isolation (= D10) |
| I7 | Interrupted and resumed run: identical training and identical diagnostic files, with no duplicated points |
| I8 | No file is written outside `diagnostics/` |

---

## 12. Limitations and interpretation rules

1. **Distance, position and length are confounded.** Large Δ exists only in long names at late positions. Primary results are per stratum; pooled curves carry the composition caveat (§3.4).
2. **Small cells** are reported with counts and are not the basis of conclusions. No minimum is imposed.
3. **φ, ω and δ̂ are counterfactual comparisons,** not an additive decomposition.
4. **G2 and G3 describe one parameter state,** not the stochastic steps taken.
5. **G3 is a probe estimate** of the exact full-training decomposition and carries sampling error, which may dominate when the full gradient is small. Its reliability is shown by the split-half analysis (§4.4). Quantities whose signs disagree between halves are reported as not reproduced.
6. **C1 measures removed characters plus position reset** (BOS-restart). It is not a pure storage measure. The optional data bound indicates the possible positional share.
7. **F₈ is a subpopulation** of long-name, late targets whose exact size is PENDING (§4.5). HB1 conclusions are conditional on F₈, its size and its clustered SE.
8. **Short sequences** (n ≤ 16).
9. **One fixed probe.** Seeds vary initialisation and data order, not the probe.
10. **M6-HC2 has no operational threshold** in the accepted agenda. It is assessed descriptively. A formal criterion would need an agenda amendment.

---

## 13. Decision table

| ID | Decision | Recommendation | Status |
|---|---|---|---|
| M6-DGN1 | Probe | 4 strata × min(256, N_ℓ); `derive_seed(42, "m6-probe")`; `probe.json` + SHA-256 at the authorised construction step | PROPOSED |
| M6-DGN2 | Strata | n ∈ {2–5, 6–7, 8–9, 10–16}; positions individually; start positions reported per (s, Δ) | PROPOSED |
| M6-DGN15 | Aggregation | Per stratum primary for H1, G1, G2; population-weighted primary for G3 and secondary elsewhere; no equal-stratum averaging | PROPOSED |
| (shared probe) | — | H1, G1, G2, G3 on the probe (D-42); C1 on dev (D-42) | ACCEPTED (D-42) |
| M6-DGN21 | G3 reliability | Deterministic alternating halves; per-half π̂, π̂±, κ̂_c, Π̂, ‖Ĝ‖; cos(Ĝ_A, Ĝ_B); half spread; sign-agreement labelling; no threshold | PROPOSED |
| M6-DGN3 | Cadence | P₀; P_early (powers of two < S); every epoch end | PROPOSED |
| (spectral) | E_sel | {0, end of epoch 1, best (post-hoc), final}; powers of two OPTIONAL | PROPOSED |
| M6-DGN4 | C1 | F₈ primary (count PENDING until construction); F₇ secondary (734); full (t, d, stratum) matrix; clustered SEs; per-target values mandatory | PROPOSED |
| M6-DGN5 | G2 | Stable g; exclusion only for r = 0 exactly; scaled norms | PROPOSED |
| (G3 signs) | — | Signed π̂, π̂±, Π̂(Δ ≥ 3), Q̂, κ̂_c; run-time identity checks | PROPOSED |
| M6-DGN7 | H1 on dev | Not measured; OPTIONAL at E_sel | PROPOSED |
| M6-DGN6 | Outputs, precision | §6, §8 | PROPOSED |
| M6-DGN12 | Alignment | §5.3 | PROPOSED |
| M6-DGN13 | Timing | One measurement after construction, before research runs; no automatic change | PROPOSED |
| M6-DGN14 | Hook, resume | §5.2 (resume truncation is a cross-stage requirement for Stage 4) | PROPOSED |
| (isolation) | — | §11; tests I1 – I8 | PROPOSED |
| M6-DGN9 | O3 | §4.6; no spike multiple | PROPOSED; **AMENDMENT PROPOSED** to `rnn_spec.md` §12.7 candidates |
| M6-DGN10 | §7.2 seeds | 1000 – 1009 | PROPOSED |
| M6-DGN11 | Per-position dev NLL | Each t × stratum | PROPOSED |
| M6-DGN16 | Primary analysis set | §9 | PROPOSED |
| M6-DGN17 | Diagnostic failures | Invalid point; training continues; review before use | PROPOSED |
| M6-DGN18 | δ̂ | Every point | PROPOSED |
| M6-DGN19 | Positional data bound | Once, OPTIONAL | OPTIONAL |
| M6-DGN20 | (v0.2 G3 population option) | **Withdrawn** | — |
| (mandatory set) | — | **Mandatory:** H1 (probe); G1-F, ω, δ̂ and the (s, Δ) table; G1-σ at E_sel; G2; G3 with halves; C1 including per-target values; O2 reporting; O3; run-time checks; isolation records. **Optional:** H1 on dev; spectral at powers of two; DGN19 | PROPOSED |

**PENDING outside this document:**
- probe construction, the exact N_ℓ and the F₈ count (authorised step);
- Stage 4 (S, E, hook locations, resume truncation, failure handling, pilot use);
- the timing measurement.

---

## Appendix A — Historical diagnostic methodology (Models 3–5)

**spec** = specification; **impl** = implementation; **exec** = executed (records); **acc** = later accepted decision.

| Topic | spec | impl | exec | acc |
|---|---|---|---|---|
| Activation statistics, saturation | MLP §12: epoch-end dev `ActivationMonitor` + per-unit saturation; step activations on scheduled steps. BN §12: also train-mode step activations. WaveNet §11: per-level records (Q8) | `utils/diagnostics/activations.py` (`ActivationMonitor`; `tanh_saturation` line 38); `mlp.py` lines 714–716 | Step-level activations **off** in every study run (`conditions.json` `diagnostics.activations: false`); epoch-end dev activations in every history row | D-17, D-33 |
| Parameter and gradient norms | Step-level `global_grad_norm`, `gradient_summaries`, `ParameterSnapshot` on scheduled steps | `utils/diagnostics/gradients.py` lines 24, 56; `parameters.py` line 50; `mlp.py` lines 693–700 | Every 100 steps: ≈ 1,485 rows per final run (≈ 4.7 MB per WaveNet run) | D-17; record §8.7: ≈ 1% of steps sampled |
| Finiteness, instability | Loss and gradient checks before every step; D-16 categories, no thresholds | `mlp.py` lines 526–535, 683 | Budget pilot and final runs: per-forward and post-step monitor; epoch time rose ≈ 40–55% between the pilots (also includes state hashing) | D-16, D-26 |
| Weights and initialisation | [O] scratch measurements in specs (`mlp_spec.md` §7; `wavenet_spec.md` §6) | Epoch-end W row norms | Recorded per epoch | No initialisation gate |
| Per-position / per-context | None during training | `training/wavenet_study/context_buckets.py` (`MIN_SUPPORT = 500`) | Post-hoc on dev; once on test | D-31 addendum; exploratory |
| Evaluation frequency | Three NLLs every epoch | `mlp.py` lines 709–715 | Every epoch | Record §5.6 |
| Storage and reproducibility | `steps.jsonl`, `history.jsonl` | `RunManager`, `append_jsonl` | Plus state hashes, timing, hash-chained manifests; `math.fsum` in F2 | D-23, D-33 |
| Diagnostics do not change training | Stated | `test_mlp.py` lines 675–690; `test_wavenet.py` from line 850; `test_finiteness_monitor.py` lines 120–158 | Fixture tests (final-state equality) | Record §8.7 |
| Use in conclusions | — | — | **Final report used NLL and context strata only; logged step and epoch diagnostics never analysed** | Record §8.11 open question "Learning-behaviour analysis of the logged diagnostics" |

**Lessons applied:**
1. Predeclare the analysis (§9).
2. Measure completely where it is cheap and selectively where it is expensive.
3. No thresholds not grounded in the model or an accepted hypothesis.
4. Prove isolation, including RNG state, hooks and failures.
5. Store raw values for re-analysis.

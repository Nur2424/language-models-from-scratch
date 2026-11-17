# Model 6 — Vanilla RNN (single-layer Elman): Specification

## Status (current: v0.2 APPROVED, D-44, 2026-11-14)

1. **APPROVED** by the user on 2026-11-14 and recorded as **D-44** in `PROJECT_RESEARCH_RECORD.md`. The approved content is the v0.2 draft with SHA-256 `b9c4f093cb0d0c459ba2f00f21fb7d3561cf041834b3824f1b6c3f46a1a0517d`. After approval only this status header, the labels table and the amendment log were updated (D-43 (1), current-status header). Every line from "## 1. Model identity" to the end of the file is byte-identical to the approved content.
2. **Governing decisions:** D-38 – D-44 in `PROJECT_RESEARCH_RECORD.md` §11. Where this specification and those decisions disagree, the decisions govern.
3. **Scope of the approval (D-44):** every item in §18.2 and §18.3, including the four departures documented in the v0.2 review: (a) criterion A tolerance 1e‑9 instead of the v0.1 proposal 1e‑12; (b) float32 invariance tolerance B = 1e‑5 instead of 1e‑6, with float32 tests at g = 1 only; (c) `forward` accepts B = 0; (d) finished generation samples leave the active set. The draft labels "APPROVED FOR v0.1 / v0.2" and "PROPOSED" in the body now read as **APPROVED (D-44)**. All identifiers in §18 are final.
4. **Not changed by the approval:** the D-40 training directions (M6-TRN1 – TRN9) remain PROVISIONAL until the Stage 4 decision; g = 1 and s² = 0.5 remain provisional until the §7.2 gate (M6-INI6). Every §18.4 item remains PENDING at its stage (S4, S5, Later) or STANDING.
5. **Research agenda:** §2.3 is the **canonical** statement of the Model 6 agenda; record §4.1 is the historical snapshot of 2026-11-13 (D-44).
6. **Test split:** protected (record §5.7). No Model 6 test evaluation is authorised.
7. **Implementation:** NOT authorised; it requires its own decision (UMIS §20). Training, pilots and experiments are NOT authorised.

**Version:** 0.2 (2026-11-14) — **APPROVED (D-44)**
**Governing standard:** `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md` (UMIS)
**Template:** `MODEL_SPECIFICATION_TEMPLATE.md` (all 18 sections; mapping in §18.5)
**Predecessor specifications used as precedents:** `models/hierarchical/wavenet_spec.md`, `models/feedforward/mlp_spec.md`, `models/feedforward/neural_bigram_spec.md`

### Labels

| Label | Meaning |
|---|---|
| **[T]** | Established mathematics or a derivation from the definitions in this document |
| **[I]** | Implementation requirement |
| **[O]** | Observation from a measurement. **None exist in v0.2.** |
| **[H]** | Hypothesis: untested; never a finding |
| **[E]** | Exploratory topic: no defensible prediction |
| **ACCEPTED (D-xx)** | Recorded global decision |
| **PROVISIONAL (D-40/D-41)** | Agreed direction, binding for drafting, to be confirmed or amended by the decision approving this specification (record, note under the status labels) |
| **APPROVED FOR v0.1 / v0.2** | Approved by the user on 2026-11-14 for inclusion in that draft; recorded globally only at specification approval |
| **APPROVED (D-44)** | Specification approved as a whole on 2026-11-14; applies to every item labelled APPROVED FOR v0.1 / v0.2 or PROPOSED in the body |
| **PROPOSED** | New in this draft; awaiting the user's decision |
| **PENDING (S4 / S5 / Later)** | Unresolved; must not be assumed. S4 = Stage 4 training specification; S5 = Stage 5 diagnostic specification; Later = after both |
| **STANDING** | A status that holds without a further decision (test-split protection; reserved study not authorised) |

### Amendment log

| Version | Date | Change | Decision |
|---|---|---|---|
| v0.1 | 2026-11-14 | Initial draft: all 18 template sections; research agenda (§2.3) identical to record §4.1; architecture, contract, initialisation, configuration, diagnostics mathematics, test and verification plan, acceptance checklist, decision tables | D-38 – D-43; user approvals of 2026-11-14 (§18.2) |
| v0.2 | 2026-11-14 | Shared core extended to `_step` / `_logits` / `_recurrence` with optional pre-activations (§5, §6.3, §8); analytic diagnostics on a float64 CPU copy (§12.1); pre-activation statistics moved out of M6-H1 into §7.2; §7.2 rewritten as a predeclared procedure with per-position mean-field predictions and a two-level Jacobian prediction (the v0.1 slope fit and its minimum pair count are withdrawn); numerical acceptance policy with documented rationale (§13.1); perturbed-parameter fixture rule; deterministic generation tests 8a – 8c; additional masking, initialisation, validation and diagnostic tests; §12.9 split into complexity, operation counts and (unavailable) runtime, G3 cost corrected; exact shared-start C1 algorithm (§12.6); training procedure moved to `rnn_training.py` (§1, §16); `tanh_saturation` location corrected (§1, §6.4); padded-position logits defined (§4.1, §5); generation contract, sample class, constructor, input validation, configuration and test layout finalised as PROPOSED (§5, §9, §14 – §16); identifier assignment completed (§18); v0.1 pending-item count corrected (v0.1 §18.4 had 23 rows: 21 open decisions and 2 standing statuses, not 25) | User approvals of 2026-11-14 (v0.2 review); D-38 – D-43 |
| v0.2 (status update) | 2026-11-14 | Status header, labels table and amendment log only: approval recorded. Body (§1 – §18) byte-identical to the approved content, SHA-256 `b9c4f093…a0517d` | D-44 |

---

## 1. Model identity

| Field | Value |
|---|---|
| Model name | `rnn` (`model.name` in `configs/models/rnn.yaml`; M6-IMP5, PROPOSED) |
| File paths | `models/recurrent/rnn.py` — architecture and model behaviour only (`RNNConfig`, `RNNSample`, `RNNLanguageModel`, `IncompatibleCheckpointError`); `models/recurrent/rnn_training.py` — model-local reference training procedure (`RNNTrainingConfig`, `train_rnn`, `evaluate_nll`) (M6-TRN17, APPROVED FOR v0.2); `models/recurrent/rnn_diagnostics.py` — analytic diagnostics (M6-IMP3) |
| Model family | Recurrent |
| Exact architecture variant | Single-layer Elman recurrent network with tanh hidden units, one hidden bias, direct character-to-hidden lookup, zero initial state and a linear output projection to 27 classes (M6-ARC1 – ARC4; D-39) |
| Specification version | v0.2 (DRAFT) |
| Reference implementation(s) | Independent test reference `tests/models/recurrent/rnn_reference.py` (§13.2). No historical RNN exists in the repository (record §3, §14), so UMIS §20 step 1 is N/A. |
| Shared utility dependencies | `utils/vocabulary.py`; `utils/data.py` (`load_names`, `split_names`, `make_next_token_sequences`, `NextTokenSequences`); `utils/batching.py` (`EpochSampler`); `utils/metrics.py` (`token_nll`, `NLLAccumulator`, `count_parameters`); `utils/reproducibility.py` (`derive_seed`, `make_generator`); `utils/config.py` (`load_config(..., model_config_cls=...)`, `default_config_paths`); `utils/checkpoint.py`; `utils/run_manager.py`; `utils/generation.py` (`sample_next_token`); `utils/device.py`; `utils/statistics.py` (`tensor_moments`, `Moments`); `utils/diagnostics/activations.py` (`tanh_saturation` only); `utils/diagnostics/gradients.py`; `utils/diagnostics/parameters.py` (`parameter_summaries` only) |

---

## 2. Role and motivation

### 2.1 Problem addressed

Next-character prediction on the names dataset, conditioning on the **whole prefix** of a name rather than on a fixed window (Models 3–5). The model reads a name one character at a time, carries a hidden state, and predicts the next character (or EOS) at every step.

### 2.2 Why this architecture?

- It is the first model in the programme with a **recurrent state** and **parameters shared across time**. Information and gradients must travel through repeated applications of the same transformation.
- It exposes the mechanisms named in D-38: hidden-state dynamics, temporal Jacobian products, attenuation of gradients through time, and the relationship between learning signal and context use.
- It is the baseline for the later gated models (GRU, LSTM), which change exactly these mechanisms.
- Understanding the mechanism is the primary objective; predictive performance is an evaluation outcome (D-38).

### 2.3 Research agenda (D-38)

This subsection reproduces record §4.1 (2026-11-13) without change of meaning. **When this specification is approved, this subsection becomes canonical and record §4.1 becomes the historical snapshot (record §4.1 status note).**

**Core research questions (ACCEPTED, D-38)**

| ID | Question | Programme RQ |
|---|---|---|
| M6-RQA | Temporal propagation and its mechanism: on actual name sequences, how do exact temporal Jacobian products and loss-projected gradients attenuate with distance, at initialisation and during training, and how much of the attenuation is attributable to the state-dependent derivatives D_t versus the recurrent matrix W? Sub-questions: the hidden-state regime by position and over training; attenuation curves; W-versus-D attribution. | RQ4 |
| M6-RQB | Effective context use: under the BOS-restart intervention, how much does development NLL degrade when characters more than d steps back are removed, and how does this depend on target position and name length? | RQ7 |
| M6-RQC | Gradient reach versus behavioural reach: during training, how does the distance profile of the learning signal for W evolve, and does the emergence of long-range context dependence follow, precede or fail to follow the growth of long-distance learning signal? | RQ4, RQ6 |

**Mathematical expectations to verify (not findings) [T]**
- M6-E1: J_t = ∂h_t/∂h_{t−1} = D_t W with D_t = diag(1 − h_t²); P_{t,s} = J_t J_{t−1} ⋯ J_{s+1}; ∂ℓ_t/∂h_s = P_{t,s}ᵀ r_t with r_t = Vᵀ(p_t − e_{y_t}); the hidden-gradient recursion δ_s = r_s + Wᵀ D_{s+1} δ_{s+1}; Σ_{t≥k} T_{t,k} = ∂L/∂W.
- M6-E2: σ_max(P_{t,s}) ≤ g^{t−s} ∏_k ‖D_k‖₂. With W = g·(orthogonal) and g = 1, the products cannot grow at initialisation.
- M6-E3: with orthogonal W at initialisation, W itself preserves norms. Any attenuation of the products D_t W therefore arises from the derivative factors D_t (0 < tanh′ ≤ 1). The full recurrence is **not** norm-preserving.
- M6-E4: h_1 = tanh(U[BOS] + b) is identical for every name. With V = 0, every loss gradient reaching the hidden states, U and W is exactly zero at initialisation, and the first update changes only V and c. ∂L/∂W never receives a contribution from step 1, because h_0 = 0.
- M6-E5 (mean-field approximation; an approximate expectation, not a finding): at g = 1 and s² = 0.5, stationary pre-activation SD ≈ 0.93, saturation ≈ 2.5%, mean-square per-step gradient gain ≈ 0.49. It is expected to be imperfect at early positions and with 27 reused input rows.
- M6-E6: C1 is an exact no-op for targets with t ≤ d + 1.
- M6-E7: the G3 projection shares sum exactly to 1.

**Hypotheses [H] (untested; none is a finding)**
- M6-HA1: at initialisation, log of the normalised Frobenius norm of P_{t,s} decreases approximately linearly with distance after a transient over the first positions, at a per-step rate near the mean-field value.
- M6-HA2: training moves W away from orthogonality and raises σ_max(W) above 1.
- M6-HA3: where saturation rises during training, attenuation at a given distance strengthens at the same positions, relative to the W-only counterfactual.
- M6-HB1: on a fixed target set, mean dev-NLL degradation decreases with d and is small for d ≥ 8.
- M6-HB2: at fixed d, degradation is larger for later targets in long names.
- M6-HC1: early in training the learning signal for W is concentrated at distances ≤ 2; its long-distance share grows during training.
- M6-HC2: behavioural dependence at distance d appears only after the corresponding long-distance learning signal becomes non-negligible. This is an association; it cannot establish causation.

**Exploratory topics [E]**
- Whether loss gradients align with the most contracting directions (loss-projected vs direction-averaged attenuation).
- How much of the BOS-restart degradation reflects lost position information rather than lost characters.

**Open questions (not committed; may become future projects)**
- Whether a plateau in context use is limited by gradient attenuation or by the data. Not separable without a manipulated factor, e.g. the reserved recurrent-gain study (D-41).
- A descriptive comparison of the RNN's context dependence with Models 3–5 and the training-data context-entropy bounds.
- How gated architectures (GRU, LSTM) change these behaviours, and which observations transfer.

**Constraint.** Names give at most 16 recurrent steps (mean 7.1). No conclusion about long-range memory or long sequences is in scope.

**Mapping of questions to diagnostics (§12):** M6-RQA → M6-H1, M6-G1, M6-G2 and the W-only counterfactual (§12.3); M6-RQB → M6-C1 and per-position dev NLL (M6-O2); M6-RQC → M6-G2, M6-G3, M6-C1 and per-position dev NLL over training. M6-O1 – O3 document the optimisation conditions under which the questions are answered. **Operationalisation of M6-HA1 (PROPOSED, APPROVED FOR v0.2 as part of §7.2):** the "transient" is the predicted mean-field transient of §7.2, and the predeclared M6-HA1 assessment is the §7.2 report at g = 1, s² = 0.5.

---

## 3. Exact architecture definition

| Item | Definition |
|---|---|
| Architecture definition | Single-layer Elman RNN: a_t = U[x_t] + W h_{t−1} + b; h_t = tanh(a_t) at valid steps; z_t = V h_t + c (§4.1) |
| Included components | Input lookup U; recurrent matrix W; hidden bias b; output projection V and output bias c; zero initial state; validity masking of padded steps |
| Excluded components | Gates; additional recurrent layers; bidirectionality; dropout; normalisation of any kind; a separate embedding followed by an input matrix; a learned initial state; a second (input-side) recurrent bias; truncated BPTT; hidden-state carry-over between names; attention; residual connections |
| Causal or bidirectional | Causal: z_t depends only on x_1 … x_t (§4.4) |
| Autoregressive | Yes: the model predicts y_t = next character (or EOS) from the prefix; generation feeds each sampled character back in (§15) |
| Deliberate simplifications | One layer; tanh only; full BPTT through PyTorch autograd (no truncation, sequences ≤ 16 steps); per-name processing with h_0 = 0 |

Accepted basis: M6-ARC1 – M6-ARC6 (D-39). Hidden size, padding semantics and parameter representation: M6-ARC7 – M6-ARC9 (APPROVED FOR v0.1).

---

## 4. Mathematical formulation

### 4.0 Symbols and indexing

| Symbol | Meaning |
|---|---|
| H | Hidden size; **H = 128** (M6-ARC7, APPROVED FOR v0.1) |
| K = 27 | Output classes: id 0 = `.` (BOS/EOS), ids 1–26 = letters (`Vocabulary.num_classes`) |
| Input ids | 0–26 at valid positions (BOS and letters). **PAD = 27** marks padded positions and is **never looked up** (M6-ARC8). `IGNORE_INDEX = −100` marks padded targets. |
| β | Batch row (one name), β = 1 … B |
| t | Position, t = 1 … T in mathematics (index t − 1 in code). T is the padded width of the batch. |
| n_β | Number of valid steps of row β: n_β = L_β + 1 for a name of L_β characters (`NextTokenSequences.lengths`) |
| m_{β,t} | Validity: m_{β,t} = 1 if t ≤ n_β, else 0 |
| x_{β,t} | Input id: x_{β,1} = 0 (BOS); x_{β,t} = character t − 1 for 2 ≤ t ≤ n_β; 27 (PAD) for t > n_β |
| y_{β,t} | Target id: y_{β,t} = x_{β,t+1} for t < n_β; y_{β,n_β} = 0 (EOS); −100 for t > n_β |
| h_t ∈ ℝᴴ | Hidden state after consuming x_t; h_0 = 0 |
| e_k | k-th standard basis vector of ℝᴷ |

Example (record §5.4; `utils/data.py`): "emma" → inputs `[0, 5, 13, 13, 1, 27…]`, targets `[5, 13, 13, 1, 0, −100…]`, n = 5.

Dataset facts used for scale (record §5): longest training name 15 characters, so n ≤ 16; 25,640 training names; 182,666 training and 22,801 development targets; mean 182,666 / 25,640 ≈ 7.12 steps per name.

### 4.1 Core equations

For each row β (index omitted), with h_0 = 0:

- a_t = U[x̂_t] + W h_{t−1} + b
- h̃_t = tanh(a_t)
- h_t = h̃_t if m_t = 1, and h_t = h_{t−1} if m_t = 0
- z_t = V h_t + c
- p_t = softmax(z_t)

where x̂_t = x_t at valid positions and x̂_t = 0 at padded positions (a safe in-range id whose row is looked up but whose result is discarded by the selection; §4.3, M6-ARC8). U[x] denotes row x of U, taken as a column vector in ℝᴴ.

At valid positions h_t = tanh(U[x_t] + W h_{t−1} + b), the standard Elman recurrence. At padded positions the state is **frozen** at the last valid state, h_t = h_{n_β}.

**Padded-position logits (M6-IMP10, APPROVED FOR v0.2).** For t > n_β, z_t = V h_{n_β} + c: the logits are computed from the frozen final state and are finite. They have **no predictive meaning**: their targets are −100, and they are excluded from every loss, metric and diagnostic. They equal z_{n_β} in exact arithmetic.

**Code orientation [I].** States are row vectors in code (`h` of shape [B, H]): `a = U[x̂] + h_prev @ W.T + b` and `z = h @ V.T + c`. W[i, j] is the weight from unit j at step t − 1 to unit i at step t; V[k, j] is the weight from hidden unit j to class k.

### 4.2 Loss and reported metric

- Per-target loss: ℓ_{β,t} = −log p_{β,t}[y_{β,t}] for m_{β,t} = 1.
- **Training loss (Option B, D-40, M6-TRN4):** L_batch = (1/C) · Σ_β Σ_{t ≤ n_β} ℓ_{β,t}, computed as `token_nll(logits, targets, reduction="sum") / C`. Padded targets (−100) are excluded by `token_nll`. **C is a fixed constant whose definition and value are PENDING (M6-TRN10, S4).** For Option B to be unbiased up to a constant, C must not depend on the batch [T].
- **Reported metric:** token-weighted mean NLL in nats per character over all valid targets of a split, via `NLLAccumulator` (record §5.6). It does not depend on C.
- With C = (batch size) × (mean steps per training name), the expected step equals η·∇(mean NLL), the same learning-rate units as Models 3–5 [T]. This is a property of one candidate, not a decision.

### 4.3 Parameterisation (M6-ARC9, APPROVED FOR v0.1)

Five raw `nn.Parameter` tensors; no `nn.Embedding`, `nn.Linear` or recurrent module.

| Parameter (code name) | Symbol | Shape | Count at H = 128 | Role |
|---|---|---|---|---|
| `input_lookup` | U | [27, H] | 3,456 | Row x is the input contribution of id x (BOS and 26 letters). No PAD row. |
| `recurrent_weight` | W | [H, H] | 16,384 | Hidden-to-hidden transformation, shared across all steps |
| `hidden_bias` | b | [H] | 128 | The single hidden bias (M6-ARC1) |
| `output_weight` | V | [27, H] | 3,456 | Hidden-to-logit projection |
| `output_bias` | c | [27] | 27 | Logit bias |
| **Total** | | | **23,451** | P(H) = H² + 55H + 27 [T] |

U, W and b are shared across all time steps and all names (UMIS §9; tested, §13.2).

*Why direct lookup [T]:* an embedding E ∈ ℝ^{27×d} followed by an input matrix W_x ∈ ℝ^{H×d} with no nonlinearity in between is the single map U = E W_xᵀ of rank ≤ d. The direct lookup is the same family without the rank constraint (M6-ARC2, D-39).

*Why no learned initial state [T]:* x_1 is always BOS, so the effect of a learned h_0 at t = 1 is W h_0, which a shift of U[BOS] reproduces exactly. A learned h_0 would add parameters without adding a mechanism (M6-ARC3).

### 4.4 State, information flow and gradients

**Causality [T].** By induction, h_t is a function of x_1 … x_t only, so z_t depends only on the prefix. Changing any x_{t'} with t' > t leaves z_1 … z_t unchanged.

**Mathematical invariance to padding and batch composition [T] (mandatory, M6-IMP8).** Row β's valid states and logits depend only on row β's inputs: the recurrence is row-wise, padded steps are selected out, and no operation mixes rows. Hence valid logits are invariant (in exact arithmetic) to the padded width T and to which other names share the batch. Bitwise identity of the floating-point results across different batch shapes is **not** implied and is not required (§13.1).

**Temporal Jacobian [T].** At valid steps, J_t = ∂h_t/∂h_{t−1} = D_t W with D_t = diag(1 − h_t²). D_t uses the new state h_t. For s < t ≤ n:

- P_{t,s} = ∂h_t/∂h_s = J_t J_{t−1} ⋯ J_{s+1} (newest Jacobian on the left); P_{s,s} = I.

At padded steps the selection makes ∂h_t/∂h_{t−1} = I; no loss term exists there, so padded steps never contribute to any gradient [T].

**Backpropagation through time [T].** For one name with valid steps 1 … n and unnormalised loss L = Σ_{t≤n} ℓ_t:

- r_t = ∂ℓ_t/∂h_t = Vᵀ(p_t − e_{y_t})
- δ_t = ∂L/∂h_t = r_t + Wᵀ D_{t+1} δ_{t+1}, with δ_{n+1} = 0
- g_t = ∂L/∂a_t = D_t δ_t
- ∂L/∂W = Σ_{t=2}^{n} g_t h_{t−1}ᵀ (the t = 1 term vanishes because h_0 = 0)
- ∂L/∂U[x] = Σ_{t: x_t = x} g_t (row x of the gradient of U); ∂L/∂b = Σ_t g_t
- ∂L/∂V = Σ_t (p_t − e_{y_t}) h_tᵀ; ∂L/∂c = Σ_t (p_t − e_{y_t})
- ∂ℓ_t/∂h_s = P_{t,s}ᵀ r_t, so δ_s = Σ_{t≥s} P_{t,s}ᵀ r_t.

For a batch, gradients are summed over names and multiplied by 1/C (Option B). Training uses **full BPTT through PyTorch autograd** (M6-ARC5); the formulas above are used only by diagnostics and tests.

**Vanishing and growth [T].** Since 0 < tanh′ ≤ 1, ‖J_t‖₂ ≤ ‖W‖₂ and ‖P_{t,s}‖₂ ≤ ∏_{k=s+1}^{t} ‖D_k‖₂ ‖W‖₂. With W = g·Q (Q orthogonal) at initialisation, ‖P_{t,s}‖₂ ≤ g^{t−s} ∏ ‖D_k‖₂ (M6-E2). Orthogonality makes W norm-preserving, not the recurrence: the products contract whenever D_k < I (M6-E3).

**No carry-over [T].** Every name starts from h_0 = 0 (M6-ARC3). Batches never share state.

### 4.5 Assumptions and limitations

- Sequences are short (n ≤ 16, mean ≈ 7.1). Long-range claims are out of scope (§2.3 constraint).
- h_1 is identical for every name (M6-E4); position-1 statistics describe one vector.
- With V = 0 at initialisation, all loss gradients reaching the hidden states, U and W are exactly zero at step 0 (M6-E4). Initial temporal propagation is characterised with loss-independent Jacobian products (§12.3). For the same reason every logit is exactly 0 at initialisation, which makes equivalence tests on an initialised model vacuous (§13.1, M6-IMP9).
- Logits at padded positions are defined but meaningless (§4.1).
- Float32 is the training default; analytic diagnostics use float64 on CPU (§10, §12).

### 4.6 Training procedure (D-40; PROVISIONAL)

The model-local reference procedure `train_rnn` lives in `models/recurrent/rnn_training.py` (M6-TRN17, APPROVED FOR v0.2) and follows the record's selection conventions:

| ID | Rule | Status |
|---|---|---|
| M6-TRN1 | Training examples are complete names; 32 names per batch; `EpochSampler(N_train_names, 32, seed=derive_seed(seed, "data"))` | PROVISIONAL (D-40) |
| M6-TRN2 | Sequences built once per split with `make_next_token_sequences` | PROVISIONAL (D-40) |
| M6-TRN3 | Each batch is trimmed to its longest name (width = max n_β in the batch) before `forward`; trimming is an efficiency step only (§4.4) | PROVISIONAL (D-40) |
| M6-TRN4 | Option-B loss, `token_nll(..., reduction="sum") / C` | PROVISIONAL (D-40); C PENDING (M6-TRN10, S4) |
| M6-TRN5 | Plain SGD, no momentum, no weight decay | PROVISIONAL (D-40) |
| M6-TRN6 | Full BPTT through autograd; no truncation | PROVISIONAL (D-40); ACCEPTED as architecture (D-39) |
| M6-TRN7 | No gradient clipping by default; any clipping requires a new decision and lives in the training loop, never in `forward` (UMIS §13) | PROVISIONAL (D-40) |
| M6-TRN8 | The pre-step global gradient norm is recorded at every optimisation step (part of M6-O1) | PROVISIONAL (D-40) |
| M6-TRN9 | Three NLLs per epoch (running train, full train, full dev); the best checkpoint is the lowest full dev NLL, earliest epoch on ties (record §5.6–5.7) | PROVISIONAL (D-40) |

Non-finite guard [I] (PROPOSED, precedent `train_mlp`): a non-finite loss or gradient raises `FloatingPointError` naming the epoch and step, before `optimizer.step()`. All training values are PENDING (S4): §18.4.

---

## 5. Tensor and interface contract

| Item | Shape | Dtype | Device | Meaning |
|---|---|---|---|---|
| `inputs` | [B, T] | `torch.long` | model's device | Right-padded ids: column 0 is BOS; valid ids 0–26; padded positions exactly 27 |
| Validity m (internal) | [B, T] | `torch.bool` | model's device | Derived from `inputs != 27` |
| States h (internal) | [B, T, H] | parameter dtype | model's device | Returned by `_recurrence`; frozen after n_β |
| Pre-activations a (internal, optional) | [B, T, H] | parameter dtype | model's device | Returned by `_recurrence` only when `return_preactivations=True`; never collected on the training path. At padded positions a_t is computed from the safe id and has no meaning |
| `logits` (output) | [B, T, 27] | parameter dtype | model's device | z_t at every position; meaningful only where m = 1; padded positions per §4.1 |
| `targets` (training) | [B, T] | `torch.long` | model's device | From `NextTokenSequences`; −100 where m = 0 |

- **Interface (M6-IMP1, APPROVED FOR v0.1):** `forward(inputs) -> logits`. Validity is derived from PAD positions; lengths are not passed. Hidden states are not exposed publicly (precedent: WaveNet Q10).
- **Output representation:** logits (UMIS §6.1); log-softmax and loss outside `forward`.
- **Special tokens:** id 0 is BOS as an input (position 1 only, in data) and EOS as a target. PAD is input-only and never a target or a looked-up row.
- **Dtype:** parameters are created in float32 (UMIS §10). `model.to(torch.float64)` gives a float64 model; the forward path follows the parameter dtype and hard-codes no dtype (§10).
- **Variable lengths:** any amount of right padding is accepted; trimming is the caller's choice (M6-TRN3).

### 5.1 Input validation (M6-IMP6, PROPOSED)

`forward` (and `_recurrence`) raise before any computation:

| Condition | Error |
|---|---|
| `inputs` is not a `torch.Tensor` | `TypeError` |
| dtype is not `torch.long` | `ValueError` |
| `inputs.ndim != 2` | `ValueError` |
| T = 0 (no BOS column) | `ValueError` |
| `inputs.device` differs from the parameters' device (no hidden transfer, UMIS §5) | `ValueError` |
| any id outside 0 … 27 | `ValueError` |
| any row whose column 0 is not BOS (0) | `ValueError` |
| any PAD followed later in its row by a non-PAD (padding not right-contiguous) | `ValueError` |

Consequences [T]: column 0 is BOS, so every row has n_β ≥ 1; BOS appears only in column 0 in data, but a later id 0 is a legal input and is not rejected (id 0 is a valid input id). **B = 0 is allowed** and returns an empty tensor of shape [0, T, 27] (precedent: `MLPLanguageModel`; v0.1 rejected it). Error messages name the offending row where applicable.

### 5.2 Construction (M6-IMP11, PROPOSED)

`RNNLanguageModel(vocab: Vocabulary, config: RNNConfig = RNNConfig(), *, generator: torch.Generator)` (precedent `MLPLanguageModel`):

- `vocab` must be a `Vocabulary` (`TypeError` otherwise); K = `vocab.num_classes` sizes U and V; PAD id from `vocab.pad_id`; BOS/EOS id from the vocabulary.
- `config` must be an `RNNConfig` (`TypeError` otherwise).
- `generator` must be a CPU `torch.Generator` (`TypeError` / `ValueError` otherwise); every initial draw comes from it (§7.1). Callers pass `make_generator(derive_seed(seed, "init"))`.
- Attributes: `vocab`, `config`, read-only property `device`.

---

## 6. PyTorch implementation policy

### 6.1 Permitted

- `nn.Module`, `nn.Parameter`; tensor indexing (`U[ids]`) or `torch.nn.functional.embedding` for the lookup;
- `@` / `torch.matmul`, `torch.tanh`, `torch.where`, `torch.stack`, `torch.zeros`;
- `torch.linalg.qr` (initialisation only) and `torch.linalg.svdvals` / `torch.linalg.matrix_norm` (diagnostics only);
- `torch.optim.SGD`; `utils.metrics.token_nll`; PyTorch autograd for training.

### 6.2 Prohibited

- `nn.RNN`, `nn.RNNCell`, `nn.GRU`, `nn.GRUCell`, `nn.LSTM`, `nn.LSTMCell`;
- `torch.nn.utils.rnn` (packed or padded sequence helpers);
- `torch._VF` recurrent kernels or any fused or compiled replacement of the recurrence;
- `nn.Embedding` and `nn.Linear` for the five parameters (M6-ARC9);
- PyTorch default initialisation of any parameter.

A source-scan test enforces §6.2 (§13.2).

### 6.3 Manual implementation requirements [I] (M6-IMP2, APPROVED FOR v0.1 and v0.2)

- **`_step(h_prev, x̂_t, m_t) -> (h_t, a_t)`** is the only place the recurrence equation exists. Returning a_t costs nothing: it is computed inside `_step` anyway.
- **`_logits(h) -> h @ Vᵀ + c`** is the only place the output projection exists. It accepts any leading shape ([B, H] or [B, T, H]).
- **`_recurrence(inputs, *, return_preactivations: bool = False) -> RecurrenceOutput`**, a model-local named tuple with fields `states` [B, T, H], `valid` [B, T] and `preactivations` [B, T, H] or `None`. It validates inputs (§5.1), derives m and x̂, loops `_step` over t, and collects pre-activations only when requested.
- `forward(inputs) = _logits(_recurrence(inputs).states)`; it never requests pre-activations.
- `generate` uses `_step` and `_logits` (§15). Diagnostics use `_recurrence` (§12.1).
- **Autograd is preserved.** The model never detaches, clones-without-grad or casts states; when gradients are enabled the full graph through every step is kept, so that autograd-based verification (§13.2) applies to the training computation itself. Under `torch.no_grad()` no graph is built.
- The validity selection uses `torch.where(m_t, h̃_t, h_{t−1})` on the **state**, not on the pre-activation.
- The only deliberate duplicate of the equations is the independent test reference (§13.2).

### 6.4 Shared utility integration

| Utility | Use |
|---|---|
| `Vocabulary` | ids, `num_classes` (27), `pad_id` (27), BOS/EOS id 0, `check_vocab_coverage` |
| `make_next_token_sequences` | sequences per split (record §5.4) |
| `EpochSampler` | name-level sampling (M6-TRN1) |
| `token_nll`, `NLLAccumulator` | training loss (`"sum"`) and reported metrics |
| `derive_seed`, `make_generator` | `"init"`, `"data"`, `"generation"` streams; no global reseeding |
| `load_config(..., model_config_cls=RNNConfig)`, `default_config_paths("rnn")` | architecture configuration |
| `save_checkpoint`, `load_checkpoint`, `RunManager` | checkpoints and run folders |
| `sample_next_token` | sampling in `generate`; imported at module level in `rnn.py` (precedent `mlp.py`) |
| `resolve_device`, `validate_dtype` | device policy; float64 is unavailable on MPS |
| `tensor_moments`, `Moments.merge` (`utils/statistics.py`) | masked statistics for M6-H1 and §7.2 |
| `tanh_saturation(0.97)` (`utils/diagnostics/activations.py`; returns the predicate \|y\| > 0.97) | saturation indicator for M6-H1 and §7.2 |
| `global_grad_norm`, `gradient_summaries`, `nonfinite_gradients` (`utils/diagnostics/gradients.py`), `parameter_summaries` (`utils/diagnostics/parameters.py`) | M6-O1, M6-O2 |
| **Not used** | `ActivationMonitor` (aggregates across the T calls and has no mask); `ParameterSnapshot` (under plain SGD the update is η times the gradient) (D-42). Whether `DiagnosticsSchedule` is used: PENDING (M6-DGN14, S5) |

---

## 7. Initialisation and parameter sharing

| Parameter | Initialisation | Status | Rationale |
|---|---|---|---|
| W | W = g · Q, Q a Haar-random orthogonal H × H matrix | M6-INI1, PROVISIONAL (D-41); construction APPROVED FOR v0.1 | All singular values and the spectral radius of W equal g: one interpretable gain (D-41) |
| U | Entries i.i.d. N(0, s²) | M6-INI2, PROVISIONAL (D-41) | Sets the input drive into the pre-activations |
| b | 0 | M6-INI3, PROVISIONAL (D-41) | Project convention |
| V, c | 0 | M6-INI4, PROVISIONAL (D-41) | Initial loss exactly ln 27 (Models 3–5 convention); consequences in M6-E4 |
| Random source | The CPU generator passed to the constructor (`make_generator(derive_seed(seed, "init"))`); no global reseeding; nothing in `forward` | M6-INI5, PROVISIONAL (D-41) | UMIS §9, §14 |
| Values | **g = 1.0, s² = 0.5 (provisional)** | M6-INI6, PROVISIONAL (D-41); configurability APPROVED FOR v0.1 | Reported against M6-E5 by the procedure in §7.2; final values PENDING (S4 entry gate) |

### 7.1 Construction [I] (APPROVED FOR v0.1)

Draw order is fixed; all draws on CPU from the constructor's generator:

1. Z ∈ ℝ^{H×H}, Z_ij ~ N(0, 1), drawn in **float64**.
2. (Q, R) = `torch.linalg.qr(Z)` (reduced).
3. Sign correction: Q ← Q · diag(σ), with σ_i = sign(R_ii), and σ_i = +1 where R_ii = 0. This makes Q Haar-distributed (Mezzadri 2007) and the factorisation unique.
4. W = g · Q, then cast to float32.
5. U = s · N(0, 1)^{27×H}, drawn next in float64 and cast to float32.
6. b, V, c set to zero.

The model is created on CPU; the caller moves it to the target device. Float32 casting perturbs orthogonality at about 1e‑7 relative; the measured deviation is reported in M6-O2 at epoch 0.

**Configurability (M6-INI6, APPROVED FOR v0.1):** g and s² are `RNNConfig` fields `recurrent_gain` and `input_init_variance` with provisional defaults 1.0 and 0.5. Non-default values are used only under an explicit decision (D-41 reserves g as a possible later factor; not authorised).

### 7.2 Initial-statistics procedure (M6-DGN8, APPROVED FOR v0.2)

Purpose: describe the model at the provisional values before any training, against predictions declared here in advance. Nothing in this procedure selects or changes a value.

**1. Fixed values.** g = 1, s² = 0.5. No observation changes them automatically.

**2. Models.** Constructed for the primary training seed and for a predeclared set of initialisation-only seeds (set PENDING, M6-DGN10, S5). No optimiser step is taken.

**3. Data.** The fixed training probe (M6-DGN1, S5). Valid positions only; position 1 reported separately (M6-E4). The test split is never used.

**4. Computation.** Analytic quantities on a float64 CPU copy of each model (§12.1), using `_recurrence(..., return_preactivations=True)`.

**5. Measured [O], per seed.** By position t and length stratum (M6-DGN2, S5), with counts:
- pre-activation mean and variance; RMS state; saturation fraction |h| > 0.97; mean and mean-square of tanh′ = 1 − h²;
- for every probe pair (t, s): exact log φ_{t,s} = log(‖P_{t,s}‖_F/√H) and exact log σ_{t,s} = log ‖P_{t,s}‖₂;
- singular values of W, ‖WᵀW/g² − I‖_F; sample mean and variance of U.

**6. Predicted [T], declared now.**

*Per-position mean field.* With b = 0 and h_0 = 0: q_1 = s²; q_t = g²·E[tanh²(√q_{t−1} Z)] + s², Z ~ N(0, 1). Then pre-activation SD √q_t, saturation 2Φ(−atanh(0.97)/√q_t), and χ_t = g²·E[(1 − tanh²(√q_t Z))²]. At g = 1, s² = 0.5 (Gauss–Hermite quadrature, 200 nodes):

| t | q_t | SD | Saturation | E[tanh′] | χ_t |
|---|---|---|---|---|---|
| 1 | 0.5000 | 0.707 | 0.31% | 0.726 | 0.592 |
| 2 | 0.7737 | 0.880 | 1.74% | 0.652 | 0.511 |
| 3 | 0.8481 | 0.921 | 2.31% | 0.636 | 0.494 |
| 4 | 0.8645 | 0.930 | 2.44% | 0.632 | 0.491 |
| 5 | 0.8680 | 0.932 | 2.47% | 0.631 | 0.490 |
| ≥ 6 | 0.8689 | 0.932 | 2.48% | 0.631 | 0.490 |

These reproduce M6-E5; the transient is predicted to be essentially complete by t ≈ 4–5.

*Jacobian products, two levels.*
- Level 1 (a priori): log φ^MF_{t,s} = ½ Σ_{k=s+1}^{t} log χ_k.
- Level 2 (factorised, using the row's observed derivatives): log φ̂_{t,s} = ½ Σ_{k=s+1}^{t} log(g² · (1/H) Σ_j (1 − h_{k,j}²)²). For independent Haar matrices W_k it is exact in expectation, because E[Wᵀ D² W] = (tr D²/H)·I [T]; the model uses one W at every step, so the difference measures the effect of tying and of correlations.
- Decomposition [T]: log φ_{t,s} − log φ^MF_{t,s} = (log φ_{t,s} − log φ̂_{t,s}) + (log φ̂_{t,s} − log φ^MF_{t,s}).

**7. Reported.** Observed minus predicted, per position and per (Δ, s), including the decomposition, each shown next to the spread of the same quantity across the initialisation seeds. Analytic sampling scale at t = 1: q_1 is the sample variance of the H entries of U[BOS] (with b = 0), whose SD is ≈ s²√(2/H) ≈ 0.0625 at H = 128. **No fitted slopes, no minimum counts, no pass/fail thresholds.**

**8. Use.** This report is the predeclared M6-HA1 assessment at initialisation (§2.3).

**9. Change rule.** Any proposal to change g or s² requires a separate recorded decision **before any training step**. The proposal must cite the observations reported here, identify which part of the D-41 rationale they contradict and why that prevents M6-RQA or M6-RQC from being measured (for example, products indistinguishable from zero in float64 at distances within the name lengths, or derivatives D_k essentially zero), and give the alternative values' Level 1 predictions.
- Deviations from mean-field alone are **not** grounds for a change: they are findings about M6-E5 and M6-HA1.
- Construction defects (W not orthogonal within §13.2 tolerances, wrong U variance) are implementation errors to fix, **not** grounds for changing values.
- After any change, M6-HA1 statements at the new values are exploratory unless predeclared again and measured on fresh initialisation seeds.

### 7.3 Parameter sharing

One W, one U and one b are used at every step of every name; there is no tying between input and output (U and V are separate). Tested in §13.2.

---

## 8. Forward-pass specification

| Stage | Operation | Input → output shape | Parameters | Masking / boundary | Numerics |
|---|---|---|---|---|---|
| 1 | Validate inputs (§5.1) | [B, T] → — | — | Rejects invalid padding | — |
| 2 | m = inputs ≠ 27; x̂ = where(m, inputs, 0) | [B, T] → [B, T], [B, T] | — | PAD replaced by a safe id | — |
| 3 | h_0 = zeros | → [B, H] | — | Parameter dtype and device | — |
| 4 | For t = 1 … T: (h_t, a_t) = `_step`(h_{t−1}, x̂_t, m_t) | [B, H] → [B, H], [B, H] | U, W, b | Frozen where m_t = 0 | tanh bounded in (−1, 1) |
| 5 | Stack states; stack pre-activations only if requested | → [B, T, H] | — | — | — |
| 6 | z = `_logits`(h) | [B, T, H] → [B, T, 27] | V, c | Padded positions: frozen-state logits (§4.1) | Finite |

`_step(h_prev, x̂_t, m_t)`: a_t = U[x̂_t] + h_prev @ Wᵀ + b; h̃_t = tanh(a_t); returns (where(m_t, h̃_t, h_prev), a_t).

`_recurrence(inputs, *, return_preactivations=False)` returns `RecurrenceOutput(states, valid, preactivations | None)`. `forward` returns `_logits(states)` only. With gradients enabled the graph is kept through every step; under `torch.no_grad()` none is built (§6.3).

---

## 9. Configuration

### 9.1 `RNNConfig` (architecture; `configs/models/rnn.yaml`, `model.params`)

Frozen dataclass in `rnn.py` (precedent `MLPConfig`).

| Field | Type | Default | Validation | Status |
|---|---|---|---|---|
| `hidden_dim` | int | 128 | int ≥ 1 (bool rejected) | APPROVED FOR v0.1 (M6-ARC7) |
| `recurrent_gain` | float | 1.0 | finite, > 0 (bool rejected) | Configurability APPROVED FOR v0.1; value PROVISIONAL (D-41) |
| `input_init_variance` | float | 0.5 | finite, > 0 (bool rejected) | Configurability APPROVED FOR v0.1; value PROVISIONAL (D-41) |

Vocabulary sizes come from `Vocabulary` (27 lookup rows = `num_classes`, because PAD is never looked up). Unknown keys are rejected by `load_config` (strict).

**Configuration file (M6-IMP5, PROPOSED).** `configs/models/rnn.yaml`, following `configs/models/mlp.yaml`: a header comment naming `models/recurrent/rnn.py`, this specification and its version, the load call `load_config(default_config_paths("rnn"), model_config_cls=RNNConfig)`, the parameter count, and a note that training settings live in `RNNTrainingConfig`; then

```yaml
model:
  name: rnn
  params:
    hidden_dim: 128
    recurrent_gain: 1.0        # provisional (D-41)
    input_init_variance: 0.5   # provisional (D-41)
```

### 9.2 `RNNTrainingConfig` (model-local, in `rnn_training.py`; precedent `MLPTrainingConfig`)

| Field | Status |
|---|---|
| `batch_size` | 32 names (PROVISIONAL, D-40) |
| `lr` | PENDING (M6-TRN12, S4) |
| `lr_schedule` | PENDING (M6-TRN13, S4) |
| `max_epochs` | PENDING (M6-TRN14, S4) |
| `eval_batch_size` | PENDING (M6-TRN15, S4) |
| Normalisation constant C | PENDING (M6-TRN10, S4) |
| Partial-batch policy | PENDING (M6-TRN11, S4) |

Field names other than `batch_size` are placeholders until Stage 4. Seed, split seed, device and generation cap come from `configs/base.yaml`. The training device is PENDING (M6-TRN18, S4).

---

## 10. Numerical stability and edge cases

| Case | Intended behaviour | Test |
|---|---|---|
| Softmax / log-softmax | Via `token_nll` (cross-entropy on logits; stable) | Finite loss for extreme logits |
| Hidden state | tanh keeps h ∈ (−1, 1); no overflow in the recurrence | Large-gain fixture stays finite |
| Gradient growth | Possible if ‖W‖ grows; not clipped (M6-TRN7); visible in M6-O1 | Non-finite guard test |
| NaN / Inf | `FloatingPointError` before `optimizer.step()` (PROPOSED) | Injected NaN loss and gradient |
| Shortest names | n = 2 (one character plus EOS) | Included in fixtures |
| No padding | All rows of equal length | Covered |
| Heavy padding | T much larger than every n_β | Padding-invariance tests |
| Single-row batch | B = 1 | Covered |
| Empty batch | B = 0 returns [0, T, 27] | Validation tests |
| Invalid inputs | Errors listed in §5.1 | Validation tests |
| Products in diagnostics | Computed in float64 on CPU; reported on a log scale to avoid underflow | Diagnostic tests |
| Device dtype | float64 unavailable on MPS (`validate_dtype`); diagnostics and Haar construction run on CPU | Device tests |
| Precision | Float32 default for training (UMIS §10); the forward path follows the parameter dtype and hard-codes none | Float64 reference tests; dtype test |

---

## 11. Complexity and resource profile

All figures are [T] arithmetic, not measurements.

| Quantity | Formula | H = 128 |
|---|---|---|
| Parameters | H² + 55H + 27 | 23,451 |
| Forward MACs per valid step | H² + 27H (lookup has no MACs) | 19,840 |
| Forward MACs per epoch (valid steps only) | 182,666 × (H² + 27H) | ≈ 3.6 G |
| Training per epoch (forward + backward, ≈ 3×, rule of thumb) | | ≈ 11 G MACs, plus padded steps |
| Per-epoch evaluation (full train + dev NLL, M6-TRN9) | (182,666 + 22,801) × (H² + 27H) | ≈ 4.1 G MACs |
| Steps per epoch | ⌈25,640 / 32⌉ if the partial batch is kept (M6-TRN11, S4) | 802 (801 if dropped) |
| Sequence scaling | Linear in T; sequential over t (no parallelism across time) | — |

**Activation memory [T].** Under autograd, per step: the saved h_{t−1} (for the matmul), the saved tanh output and the mask, ≈ 2·B·H floats; plus the stacked states (B·T·H), the logits (B·T·27) and the softmax saves. At B = 32, T = 16, H = 128 in float32 this is under 2 MB per batch. Pre-activations, if requested, add B·T·H; the training path never requests them.

**Bottleneck:** the Python loop over t with small matrix products. Per-step overhead may dominate, and MPS kernel-launch overhead may make CPU faster. Runtime will be measured, not assumed. Diagnostic costs: §12.9.

---

## 12. Diagnostics (D-42; framework ACCEPTED, parameters PENDING S5)

### 12.1 Common rules [I]

- **Data:** M6-H1 and M6-G1 – G3 use a **fixed training probe** (size, stored name IDs and length stratification PENDING, M6-DGN1, S5). M6-C1 uses **dev** data. M6-O2 uses train and dev. **The test split is never used.**
- **Computation (M6-IMP3, APPROVED FOR v0.2):** analytic diagnostics run the shared `_recurrence` under `torch.no_grad()` on a **float64 CPU deep copy** of the model. No computation graph is created or retained. The training model, its parameters, gradients, optimiser, sampler and mode are never touched.
- **Masking:** only valid steps enter any statistic; position 1 is reported separately (M6-E4).
- **No side effects:** diagnostics consume no random numbers from any stream and change no model, optimiser or sampler state. Training with diagnostics on and off is bit-identical (tested).
- **Exactness:** G1 – G3 and C1 are exact. No approximate estimator (random-vector trace estimation, power iteration, sampling of pairs) may replace them without approval.
- **Location:** `models/recurrent/rnn_diagnostics.py` (M6-IMP3), except M6-O1 and M6-O2, which are collected by `train_rnn` in `rnn_training.py`. How `train_rnn` invokes the diagnostics module (hook interface) is PENDING (M6-DGN14, S5).

### 12.2 M6-H1 — hidden-state dynamics

For probe rows in length stratum ℓ (strata PENDING, M6-DGN2, S5) and position t, over the valid (row, unit) pairs A_{t,ℓ} = {(β, j): n_β ≥ t}:

- RMS state ρ_{t,ℓ} = ( mean h_{β,t,j}² )^{1/2}
- saturation S_{t,ℓ} = mean 1[|h_{β,t,j}| > 0.97] (`tanh_saturation`)
- mean derivative d̄_{t,ℓ} = mean (1 − h²), and mean-square derivative d²_{t,ℓ} = mean (1 − h²)²
- counts |A_{t,ℓ}|

Pre-activation statistics are **not** part of M6-H1 (D-42); they are collected only by §7.2. Whether M6-H1 is also reported on dev: PENDING (M6-DGN7, S5).

### 12.3 M6-G1 — exact loss-independent Jacobian products

For each probe row and each pair 1 ≤ s < t ≤ n_β, with distance Δ = t − s:

- **Exact normalised Frobenius norm:** φ_{t,s} = ‖P_{t,s}‖_F / √H.
- **Exact spectral norm:** σ_{t,s} = ‖P_{t,s}‖₂ (largest singular value, from a full LAPACK SVD).
- **Per-step rate:** λ_{t,s} = φ_{t,s}^{1/Δ}.
- **W-only counterfactual** (for M6-RQA attribution and M6-HA3): ω(Δ) = ‖W^Δ‖_F / √H, computed from W alone. At initialisation with orthogonal W and g = 1, ω(Δ) = 1 exactly [T].

Computation: for each start s, accumulate P_{s+1,s} = J_{s+1}, then P_{t,s} = (D_t W) P_{t−1,s}, recording φ at each t; σ from `torch.linalg.matrix_norm(P, ord=2)`. Aggregation: median and quartiles of log φ (and log σ) by Δ and by start s, with counts. Pairs involving padded steps never enter.

Frobenius and spectral norms are distinct quantities: φ² is the mean squared singular value (average amplification); σ is the worst-case amplification. Both are exact. Their cadences may differ (§12.8).

### 12.4 M6-G2 — loss-projected attenuation (from epoch 1)

r_t = Vᵀ(p_t − e_{y_t}) (float64). For s ≤ t ≤ n_β:

- R_{t,s} = ‖P_{t,s}ᵀ r_t‖ / ‖r_t‖ (so R_{t,t} = 1)
- **unweighted** aggregate: median and quartiles of R by Δ;
- **magnitude-weighted** aggregate: R̄(Δ) = Σ ‖P_{t,s}ᵀ r_t‖ / Σ ‖r_t‖ over pairs with that Δ.

Targets with ‖r_t‖ ≤ τ are excluded and counted; τ is PENDING (M6-DGN5, S5). At epoch 0, r_t ≡ 0 (M6-E4), so M6-G2 is not computed. Computation: backward vector chain v ← Wᵀ(D_k v) from k = t down to s + 1 (no matrices needed). R_{t,s} ≤ σ_{t,s} [T].

### 12.5 M6-G3 — exact additive decomposition of the recurrent gradient

On the training probe, with unnormalised probe loss Σ ℓ:

- T_{t,k} = (D_k P_{t,k}ᵀ r_t) h_{k−1}ᵀ for 2 ≤ k ≤ t ≤ n_β; Δ = t − k ≥ 0 (Δ = 0 is the direct term). Terms with k = 1 are zero (h_0 = 0).
- **Exact identity:** G = Σ_β Σ_{t,k} T_{t,k} = ∂(Σ ℓ)/∂W (M6-E1); tested against autograd.
- **Primary — projection shares:** π(Δ) = Σ_{t−k=Δ} ⟨T_{t,k}, G⟩_F / ‖G‖_F², with Σ_Δ π(Δ) = 1 exactly (M6-E7). π(Δ) may be negative (that distance opposes the net update).
- **Cancellation index:** κ_c = ‖G‖_F / Σ ‖T_{t,k}‖_F ∈ [0, 1].
- **Secondary — norm mass:** Q(Δ) = Σ_{t−k=Δ} ‖T_{t,k}‖_F, with ‖T_{t,k}‖_F = ‖D_k P_{t,k}ᵀ r_t‖ · ‖h_{k−1}‖.

Computation: pass 1 computes G by the δ recursion (§4.4) over the whole probe; pass 2 recomputes the G2 vector chains u_{t,k} = P_{t,k}ᵀ r_t and, with G h_{k−1} computed once per position, evaluates ⟨T_{t,k}, G⟩ = (D_k u_{t,k})ᵀ (G h_{k−1}) in Θ(H) per pair. The shares are invariant to the 1/C scaling. G3 describes the full-probe gradient at an evaluation point; it does not replay the stochastic training steps.

### 12.6 M6-C1 — BOS-restart effective-context intervention (dev)

**Definition.** For a dev target at position t (prediction of y_t from h_t) and prefix length d ≥ 1 (number of characters kept):

- kept inputs: x_{t−d+1} … x_t (the characters c_{t−d} … c_{t−1});
- restart: h_0 = 0, then BOS, then the kept inputs (d + 1 steps through the shared `_step`); z^{(d)} from the final state;
- ΔNLL_t(d) = −log p^{(d)}[y_t] − (−log p_t[y_t]).

Affected targets satisfy **t ≥ d + 2**; for t ≤ d + 1 the intervention is an exact no-op (M6-E6).

**Computation: shared-start algorithm (APPROVED FOR v0.2) [T].** For each dev name and each start j with 3 ≤ j ≤ n, one restart run feeds BOS, x_j, x_{j+1}, …, x_n through `_step` (a batch of rows `[BOS, x_j, …, x_n]` padded to a common width). By causality, the state after consuming x_t is exactly the restart state for target t with d = t − j + 1. One run per j therefore yields ΔNLL_t(d) for every affected (t, d) with t − d + 1 = j. Starts j ≤ 2 are the no-op of M6-E6 and are not run. This computes the same exact quantities as the definition with fewer steps; agreement is a verification test (§13.2).

**Reporting.** On a **fixed target set** {t ≥ d_max + 2} for d = 1 … d_max, plus position and length strata with counts. d_max and the target set are PENDING (M6-DGN4, S5); the shared-start runs produce all d regardless. Pitfall: BOS-restart also removes position information ([E] topic in §2.3).

### 12.7 M6-O1 – O3 — optimisation behaviour

| ID | Content | When |
|---|---|---|
| M6-O1 | Batch loss (sum / C); batch token count; pre-step global gradient norm (M6-TRN8); per-parameter gradient norms (U, W, b, V, c); finiteness | Every optimisation step |
| M6-O2 | Three NLLs; per-position dev NLL (grouping PENDING, M6-DGN11, S5); per-parameter norms; singular values of W (max, min, median) and spectral radius; ‖WᵀW/g² − I‖_F | Every epoch, including epoch 0 |
| M6-O3 | Instability categories, defined before any run, including pilots. Candidates (PROPOSED, precedent D-16): *mechanically unstable* (non-finite loss, gradient or parameter); *flagged for review* (dev NLL rises between epochs; global gradient norm above a multiple of its running median). Categories and the multiple PENDING (M6-DGN9, S5). | Continuous |

Relative update η‖∇θ‖/‖θ‖ is derived from M6-O1 and M6-O2 (D-42).

### 12.8 When diagnostics run (cadence PENDING, M6-DGN3, S5)

Fixed by this specification [I]:
- M6-O1 at every optimisation step; M6-O2 at every epoch end and at epoch 0.
- Epoch 0 runs the §7.2 procedure; M6-G2 and M6-G3 are identically zero there (M6-E4) and are not computed; M6-C1 is not run before training.

PROPOSED principles for Stage 5 (not values):
- **M6-G3 and M6-C1 are evaluated at the same points**, because M6-RQC compares their timing.
- Evaluation points are defined in optimisation steps and may include points within epoch 1 (M6-HC1 concerns early training).
- M6-G1 spectral, the most expensive diagnostic, may run at a subset E_sel of the evaluation points E_all.

All specific points, which parameters are analysed at each point (current or also best checkpoint, M6-DGN12), and the scheduling mechanism (M6-DGN14) are PENDING (S5).

### 12.9 Diagnostic cost (APPROVED FOR v0.2 as structure)

Notation: n = valid steps in a name; π_n = n(n − 1)/2 pairs; one step costs H² + 27H = 19,840 MACs at H = 128.

**(a) Theoretical complexity per name [T]**

| Diagnostic | Complexity | Dominant operation |
|---|---|---|
| M6-H1 | Θ(n·H²) forward + Θ(n·H) statistics | the probe forward pass |
| M6-G1 Frobenius | Θ(n²·H³) | one H × H by H × H product per pair |
| M6-G1 spectral | Θ(n²·H³), larger constant | one full SVD per pair (bidiagonalisation ≈ ⁴⁄₃H³ MACs + O(H²) iterations) |
| M6-G2 | Θ(n²·H²) | one matrix–vector product per (t, s) pair |
| M6-G3 | Θ(n²·H²) | the G2 vector chains (recomputed in pass 2), one BPTT pass for G and one G·h_{k−1} per position; Θ(H) extra per pair |
| M6-C1 (shared-start) | Θ(n²·H²) | Σ_{j=3}^{n} (n − j + 2) = n(n − 1)/2 − 1 recurrent steps per name |

**(b) Estimated operation counts [T].** Assumptions: mean n ≈ 7.12; pairs per name ≈ 23 (depends on the length spread; exactly computable from the probe's IDs once M6-DGN1 is fixed); illustrative probe of 1,024 names (size PENDING).

| Item | Per name | Illustration |
|---|---|---|
| Reference: one training epoch | — | ≈ 11 G MACs + padded steps; plus ≈ 4.1 G evaluation |
| M6-H1 | ≈ 0.14 M | ≈ 0.15 G (1,024 names) |
| M6-G1 Frobenius | ≈ 23 × H³ ≈ 48 M | ≈ 49 G |
| M6-G1 spectral | ≈ 23 SVDs | ≈ 23,500 SVDs of 128 × 128 in float64 |
| M6-G2 | ≈ 23 × H² ≈ 0.38 M | ≈ 0.4 G |
| M6-G3 | ≈ 0.8 M (two passes) | ≈ 0.8 G |
| M6-C1, all d | ≈ 24 steps ≈ 0.5 M | ≈ 1.5 G (3,196 dev names) |

**(c) Measured runtime: not available.** No diagnostic has been run. MAC counts do not predict wall time between the Python-loop training computation (small [32, 128] × [128, 128] products) and dense float64 diagnostics (large products), which differ greatly in efficiency per MAC. Per-diagnostic timing on the untrained model is a Stage 5 item requiring its own authorisation (M6-DGN13).

Memory: one H × H product and one name's states at a time (well under 1 MB per name). Output format and storage precision PENDING (M6-DGN6, S5).

---

## 13. Test plan

### 13.1 Numerical acceptance policy (M6-IMP8, principles APPROVED FOR v0.2; values PROPOSED)

**Principles.**
1. **Mathematical correctness and mathematical invariance are mandatory.** They are established in float64 against the independent reference. A floating-point tolerance is never widened to excuse a mathematical discrepancy; any widening requires a recorded decision.
2. **Bitwise determinism** is required for identical calls (same inputs, shape, dtype, device).
3. **Bitwise identity across different batch shapes is observational [O]**, not an acceptance requirement: matrix-multiply kernels may accumulate in a different order for different shapes.

**Criteria.** |x − x_ref| ≤ atol + rtol·|x_ref| elementwise.

| Code | Comparison | atol = rtol | Fixtures |
|---|---|---|---|
| A | float64 model vs float64 independent reference (states, logits, pre-activations, analytic diagnostics vs autograd) | 1e‑9 | g ∈ {1, 1.2}, T ≤ 16, H ≤ 128 |
| B | float32 vs float32 across padding widths and batch compositions (supplementary invariance check) | 1e‑5 | g = 1 |
| C | float32 model vs float64 reference (precision) | 1e‑5 | g = 1 |
| D | identical calls | bitwise | any |
| E | states at t ≤ n_β when only T changes and B is fixed (CPU) | bitwise | any |
| F | logits across different batch shapes | recorded [O] only | — |
| FD | central finite differences (float64, step 1e‑6) vs autograd, per parameter element | rtol 1e‑6, atol 1e‑8 | small H |

**Numerical rationale [T].**
- *Bug signatures.* Row mixing, padding leakage or a misplaced mask change values by O(10⁻²) or more (U entries are O(0.7)); an accidental float32 cast inside the float64 path produces errors of order 10⁻⁷. Every tolerance below sits under these.
- *A (rigorous bound).* Per element per step, a_t is a dot product of length H plus two additions, so its rounding error is at most γ_{H+2}·(Σ_j |W_ij h_j| + |U| + |b|) with γ_{H+2} = (H + 2)u, u = 2⁻⁵³ ≈ 1.1·10⁻¹⁶. By Cauchy–Schwarz Σ_j |W_ij h_j| ≤ ‖W_i‖₂‖h‖₂ ≤ g√H; tanh is 1-Lipschitz. In 2-norm, the per-step rounding is ≤ √H · γ_{H+2} · (g√H + 2) ≈ 2.2·10⁻¹² at g = 1, H = 128, and errors propagate with gain ≤ ‖W‖₂ = g. Over 16 steps the accumulated bound is ≈ 3.5·10⁻¹¹ at g = 1 and ≈ 2.2·10⁻¹⁰ at g = 1.2 (Σ_{k=0}^{15} 1.2^k ≈ 87); with V rows of norm ≈ 1 the logit bound is of the same order, and the reference carries its own error of the same order. The worst-case difference is therefore ≲ 5·10⁻¹⁰. **1e‑9 is the smallest round value above this rigorous bound.** It is 100× looser than the v0.1 proposal (1e‑12), which lay below the worst-case bound although typical errors (random-walk accumulation ≈ √(16·H)·u·scale ≈ 10⁻¹³) are far smaller. 1e‑9 remains ≥ 60× below float32 contamination and ≥ 10⁷× below logic bugs.
- *B and C (statistical bound).* With u₃₂ = 2⁻²⁴ ≈ 6·10⁻⁸, worst-case bounds (≈ 10⁻³) are too loose to be useful, so these criteria use the typical-error model: per step ≈ √(H+2)·u₃₂·O(1) ≈ 7·10⁻⁷, accumulating as √16 at g = 1 to ≈ 3·10⁻⁶ per path, ≈ 6·10⁻⁶ between two paths. 1e‑5 covers this; float32 tests are therefore run at g = 1 only. It remains ≥ 100× below logic bugs. **A failure of B or C is investigated against the float64 reference (criterion A); it is never resolved by loosening the tolerance without a recorded decision.**
- *E (structural).* Per-step operands have shape [B, H] whatever T is, so padding columns never enter the computation of valid states; any difference indicates a defect.
- *FD.* Truncation error O(h²) ≈ 10⁻¹²; rounding error ≈ u·|L|/h ≈ 10⁻¹⁰·|L|; with fixture losses O(10) the absolute error is ≲ 10⁻⁹, well inside atol 1e‑8 and rtol 1e‑6.
- *G3 share sum.* |Σπ − 1| is bounded by rounding relative to Σ‖T‖ = ‖G‖/κ_c, i.e. amplified by 1/κ_c; fixtures report κ_c, and criterion A applies while κ_c ≥ 10⁻³.

**Fixture rule (M6-IMP9, APPROVED FOR v0.2).** With V = c = 0 every logit is exactly 0, so equivalence tests on an initialised model are vacuous. All equivalence tests (§13.2 R1 – R8, M1 – M11, generation tests) therefore use **perturbed parameters** drawn from a test-local generator: U entries N(0, s²); W = g·Q (Haar, as §7.1); b entries N(0, 0.1²); V entries N(0, 1/H) (logits of order 1); c entries N(0, 0.1²). Float64 tests run at g = 1 and g = 1.2; float32 tests at g = 1. Only the initialisation tests (I1 – I5) use the initialised model.

### 13.2 Test inventory

**Files (M6-IMP4, PROPOSED).** In `tests/models/recurrent/` (no `__init__.py`, precedent `tests/models/hierarchical/`):

| File | Content |
|---|---|
| `rnn_reference.py` | Independent reference: imports only `torch`; never imports `models.*` or `utils.*`; receives plain tensors (U, W, b, V, c, inputs), never the model object; processes each name **separately, unbatched, only up to its own length**, with an explicit loop written from §4.1; float64. Provides states, pre-activations, logits, losses, Jacobian products and the §4.4 gradient formulas. |
| `test_rnn_reference.py` | Tests of the reference itself against hand-computed values (precedent `test_wavenet_reference.py`) |
| `test_rnn.py` | Configuration, construction, validation, forward, invariance, initialisation, generation, checkpoint, source scan |
| `test_rnn_training.py` | `train_rnn`, `evaluate_nll`, determinism, selection, guards, overfitting check, diagnostics on/off identity |
| `test_rnn_diagnostics.py` | M6-H1, M6-G1 – G3, M6-C1, §7.2 helpers, isolation from the training model |

**Configuration, construction and validation** (`test_rnn.py`)

| # | Test | Criterion |
|---|---|---|
| V1 | Valid construction; parameter names, shapes and count 23,451 at H = 128 (`count_parameters`); parameters float32 on CPU | exact |
| V2 | Invalid `RNNConfig` values (non-int or < 1 `hidden_dim`; bool; non-finite or ≤ 0 gain or variance) raise `ValueError`; unknown YAML keys rejected by `load_config` | error raised |
| V3 | Constructor type checks: non-`Vocabulary`, non-`RNNConfig`, non-generator, non-CPU generator | error raised |
| V4 | Every §5.1 condition raises the specified error; B = 0 returns [0, T, 27] | error / shape |
| V5 | `configs/models/rnn.yaml` loads through `load_config(default_config_paths("rnn"), model_config_cls=RNNConfig)` to the defaults | exact |

**Forward, masking and invariance** (`test_rnn.py`; perturbed fixtures)

| # | Test | Criterion |
|---|---|---|
| R1 | Reference equivalence: valid states, pre-activations and logits of every row, across several batch compositions and padding widths, match the per-name reference (this establishes correctness and mathematical invariance) | A |
| R2 | Padding invariance in float32: valid logits unchanged by extra right padding; valid states bitwise unchanged with B fixed | B; E |
| R3 | Batch-composition invariance in float32: a name alone vs inside different batches | B; bitwise result recorded [O] (F) |
| R4 | Float32 precision against the float64 reference | C |
| R5 | Determinism: identical calls give identical outputs | D |
| R6 | Causality: changing x_{t'} for t' > t leaves z_1 … z_t unchanged | A (and D for identical shapes) |
| R7 | Frozen state: for t > n_β, h_t equals h_{n_β} | bitwise |
| R8 | Padded-position logits equal z_{n_β} of the same row and are finite | A in float64; B in float32 |
| R9 | Parameter sharing: after replacing W by W′ = W + ε e_i e_jᵀ, the perturbed model's pre-activations satisfy a′_t − (U[x_t] + W h′_{t−1} + b) = ε h′_{t−1,j} e_i at every valid t ≥ 2 (one W used at every step), and a′_1 = a_1 | A; a′_1 bitwise |
| R10 | Hand-computed H = 2 example: two steps of a_t, h_t, z_t match closed-form values | A |
| R11 | Pre-activations are `None` unless requested; `forward` never requests them | exact |
| R12 | Dtype: a float64 model returns float64 logits and passes R1; no float32 tensor is created in the forward path of a float64 model (checked by R1 at tolerance A, which a float32 intermediate would fail) | dtype exact; A |
| R13 | Train and eval mode give identical outputs (no dropout or normalisation) | D |
| R14 | Source scan: no prohibited module or helper from §6.2 in `rnn.py`, `rnn_training.py` or `rnn_diagnostics.py` | none found |

**Initialisation** (`test_rnn.py`; initialised model)

| # | Test | Criterion |
|---|---|---|
| I1 | Initial loss equals ln 27 | \|Δ\| ≤ 4·u·ln 27 in the evaluation dtype |
| I2 | M6-E4: gradients of U, W, b exactly zero; one SGD step changes only V and c (U, W, b bitwise unchanged); h_1 identical for all rows | exact / bitwise |
| I3 | Haar construction: float64 Q with ‖QᵀQ − I‖_max ≤ 1e‑12; float32 W singular values within 1e‑5 of g; diag R > 0 after sign correction; b, V, c exactly zero | stated tolerances |
| I4 | Draws: same generator seed gives bitwise identical parameters; different seeds differ; draw order W then U; the global RNG state is unchanged by construction | bitwise / exact |
| I5 | Sample variance of U within a statistical tolerance of s² (27·H entries; tolerance 5 standard errors of the variance estimator, s²·√(2/(27H))·5) | stated |

**Generation** (`test_rnn.py`; perturbed fixtures; no random sampling)

| # | Test | Criterion |
|---|---|---|
| 8a | Teacher-forced step loop vs full forward: a fixed list of names (shortest "a"; a 15-character name; repeated letters; names differing only in their last character). Path 1: `forward` on the padded batch. Path 2: per name, h_0 = 0, `_step` with BOS then each known character, `_logits` after each step. Compared at every valid position | A (float64), B (float32); states bitwise for B = 1 (E) |
| 8b | `generate` with a scripted sampler: `rnn.sample_next_token` is replaced by a script returning predetermined tokens per row and recording the logits it receives. Three rows: "ann" then EOS; "b" then EOS; a 6-character sequence with `max_length = 6`, whose 7th draw is not EOS. Checks: recorded logits equal `forward([BOS] + prefix)` at the matching positions; texts "ann", "b" and the 6 characters (never containing `.`); `finished` True, True, False; the script is never called for a row after it has finished; at most `max_length + 1` draws per row; parameters, `model.training` and the global RNG unchanged; no autograd graph built | A / B; exact |
| 8c | Greedy decoding (`temperature=0`, deterministic): each generated text, fed back through `forward`, has argmax equal to the generated character at every position and EOS at the end when `finished` is True. Fixture validity: the top-two logit gap is ≥ 1e‑3 at every step, otherwise a fixed alternative fixture seed is used (guards against ties; not a tolerance) | exact |
| 8d | Argument validation: `num_samples` and `max_length` must be positive ints (bool rejected) | error raised |

**Checkpoint** (`test_rnn.py`)

| # | Test | Criterion |
|---|---|---|
| K1 | Save via the training procedure's checkpoint format and reload with `from_checkpoint`: logits bitwise identical | D |
| K2 | Wrong model class, different vocabulary, invalid or missing model config, mismatched weight shapes raise `IncompatibleCheckpointError` | error raised |

**Mathematical verification** (`test_rnn_reference.py`, `test_rnn_diagnostics.py`; perturbed fixtures; float64)

| # | Test | Criterion |
|---|---|---|
| M1 | Analytic Jacobian products P_{t,s} (§12.3) vs `torch.autograd.functional.jacobian` of the `_step` chain | A |
| M2 | G2 vectors P_{t,s}ᵀ r_t vs autograd ∂ℓ_t/∂h_s | A |
| M3 | BPTT formulas (§4.4) for U, W, b, V, c vs autograd on the model's own forward | A |
| M4 | G3: Σ T_{t,k} equals `W.grad` of the unnormalised probe loss; Σ π(Δ) = 1 | A (κ_c reported) |
| M5 | Finite differences for every parameter group on a small fixture | FD |
| M6 | Diagnostics isolation: after any diagnostic call, the training model's parameters, gradients, dtype, device and mode, and all RNG states (global and the training generators), are unchanged; diagnostics ran on a float64 CPU copy | bitwise / exact |
| M7 | M6-H1 statistics equal statistics computed directly from reference states with explicit masks | A |
| M8 | M6-C1 no-op: for t ≤ d + 1 the restart reproduces the full logits (M6-E6) | A |
| M9 | M6-C1 shared-start results equal the per-(t, d) definition run separately for every affected (t, d) | A |
| M10 | Padded positions and pairs never enter any diagnostic (fixtures with heavy padding give identical diagnostics to unpadded inputs) | A |
| M11 | §7.2 mean-field predictions: the implementation reproduces the q_t / χ_t table of §7.2 | \|Δ\| ≤ 1e‑4 (table rounding) |

**Training procedure** (`test_rnn_training.py`)

| # | Test | Criterion |
|---|---|---|
| T1 | Tiny-fixture overfitting (precedent `mlp_spec.md` §13.3): fixture `["ab", "ac", "ab"]`; the prefix-conditional entropy bound is 3 · H(2/3, 1/3) / 9 ≈ 0.2122 nats per target [T]; full-train NLL approaches it within a test-local margin after a test-local number of steps at a test-local learning rate (all fixed with an [O] value at implementation). A debugging check, not a quality claim | stated at implementation |
| T2 | Determinism: two `train_rnn` runs with the same seed give bitwise identical parameters and histories on CPU | D |
| T3 | Selection: lowest dev NLL; on an injected exact tie the earliest epoch is kept | exact |
| T4 | Non-finite guard: injected NaN loss or gradient raises `FloatingPointError`; parameters bitwise identical to their values before the failing step | bitwise |
| T5 | Diagnostics on/off: a short run with diagnostics enabled and disabled gives bitwise identical parameters, losses and sampler state | D |
| T6 | `evaluate_nll` equals the manually computed token-weighted NLL and is independent of the evaluation batch size | A (float64); B (float32) |
| T7 | Option-B loss equals `token_nll(..., reduction="sum") / C` and is unaffected by per-batch trimming | A |
| T8 | Mode and RNG: evaluation and diagnostics leave `model.training` and all RNG states as they found them | exact |

General device checks: CPU for everything; MPS (and CUDA where available) for the training-path operations under criterion C, with no bitwise expectation; float64 only on CPU. MPS tolerance and determinism expectations are PENDING (M6-TRN19, S4).

### 13.3 Learning and behaviour checks

Covered by T1 – T8 above. No learning-quality claim is made by any test.

### 13.4 Experiments (outside the implementation task)

**None is authorised.** PENDING items: whether any controlled study follows (M6-EXP1, Later); number of seeds (M6-EXP2, Later); learning-rate and budget pilot design (M6-EXP3, S4). The reserved recurrent-gain study (D-41) is not authorised (M6-EXP5, STANDING). The test split remains protected (M6-EXP4, STANDING).

---

## 14. Checkpoint and serialisation contract

- **Learned state:** the five parameters. No buffers.
- **Written by** `train_rnn` (`rnn_training.py`) through `save_checkpoint`, with metadata: `model_class` (`"RNNLanguageModel"`), `Vocabulary.to_dict()`, `config["model"]` (the `RNNConfig` as plain data) and the training configuration (precedent `train_mlp`).
- **Kinds:** best (lowest dev NLL), final, resume.
- **Resume state** (PROPOSED fields; semantics PENDING, M6-TRN16, S4): epoch, best metric and epoch, `EpochSampler` state, RNG states, optimiser state (empty for plain SGD without momentum).
- **Loading (M6-IMP11, PROPOSED):** `RNNLanguageModel.from_checkpoint(path, *, expected_vocab=None, map_location="cpu")`, with `weights_only=True`. Restores weights only; continuing training uses `train_rnn(..., resume=True)` (semantics S4). Raises the model-local `IncompatibleCheckpointError` (subclass of `ValueError`, duplicated on purpose, precedent `mlp.py`) for: a different model class; missing or invalid vocabulary metadata; a vocabulary differing from `expected_vocab`; a missing or invalid model configuration; weights not matching it.
- **Round-trip tests:** K1, K2 (§13.2).

---

## 15. Generation (M6-IMP7, PROPOSED)

| Item | Specification |
|---|---|
| Entry point | `generate(num_samples, *, max_length, generator=None, temperature=1.0, top_k=None, top_p=None) -> list[RNNSample]` (mirrors `MLPLanguageModel.generate`) |
| Arguments | `num_samples`: positive int; `max_length`: positive int, **required**, no default — project callers pass `generation.max_new_tokens` from `configs/base.yaml` (32); sampling arguments validated by the shared utility |
| `max_length` semantics | The maximum number of **ordinary characters** in a sample; **EOS is not counted**. A row that has `max_length` characters gets one further draw: EOS gives `finished=True`; any other token is discarded and the row stops with `finished=False`. Hence at most `max_length + 1` draws and `max_length + 1` calls of `_step` per row. |
| Procedure (per active row) | h_0 = 0; h ← `_step`(h, BOS); repeat: logits = `_logits`(h); token = `sample_next_token`(logits, …); if token = EOS: finished; elif row length = `max_length`: stop, truncated; else append token and h ← `_step`(h, token) |
| Validity in `_step` | Always all-true for active rows (no padding during generation) |
| Finished rows | Removed from the active set (precedent `MLPLanguageModel`): their state is no longer advanced and `sample_next_token` is never called for them again. Rows advance together while active. |
| Recurrence | Exactly the training recurrence: the same `_step` and `_logits` (M6-IMP2) |
| Sampling | `utils.generation.sample_next_token`, imported at module level; `temperature=0` is greedy. With `generator=None` PyTorch's default generator is consumed (precedent); all project callers and tests pass an explicit generator (`make_generator(derive_seed(seed, "generation"))`) or use `temperature=0` |
| Mode and gradients | Runs under `torch.no_grad()`; parameters and `model.training` are not modified |
| Degenerate distributions | `SamplingError` from the shared utility |
| Sample record | `RNNSample`, frozen dataclass in `rnn.py` with `text: str` (characters between BOS and EOS, or up to `max_length`; never contains `.`) and `finished: bool` (True iff EOS was drawn). Same semantics as `MLPSample`, duplicated on purpose |

Tests: 8a – 8d (§13.2).

---

## 16. Implementation scope and boundaries

Implementation is **not authorised** by this draft. When authorised, the scope is:

| File | Content | Status |
|---|---|---|
| `models/recurrent/rnn.py` | `RNNConfig`, `RNNSample`, `IncompatibleCheckpointError`, `RNNLanguageModel` (parameters, initialisation, `_step`, `_logits`, `_recurrence`, `forward`, `generate`, `from_checkpoint`) | Architecture and model behaviour only (UMIS §12) |
| `models/recurrent/rnn_training.py` | `RNNTrainingConfig`, `train_rnn`, `evaluate_nll`, M6-O1 and M6-O2 collection | M6-TRN17, APPROVED FOR v0.2; contents beyond §4.6 PENDING (S4) |
| `models/recurrent/rnn_diagnostics.py` | M6-H1, M6-G1 – G3, M6-C1, the §7.2 procedure, probe handling | M6-IMP3; function interface PENDING (S5) |
| `configs/models/rnn.yaml` | §9.1 | M6-IMP5, PROPOSED |
| `tests/models/recurrent/` | `rnn_reference.py`, `test_rnn_reference.py`, `test_rnn.py`, `test_rnn_training.py`, `test_rnn_diagnostics.py` (§13.2) | M6-IMP4, PROPOSED |

No shared trainer and no change to `utils/`, `configs/base.yaml`, other models, existing tests, the WaveNet study code (`training/wavenet_study/`) or `requirements.txt`. `gru.py` and `lstm.py` are untouched. No sweeps, no multi-seed runners, no plotting, no notebooks; any future study orchestration belongs in its own `training/<study>/` package under a separate decision.

---

## 17. Acceptance checklist

To be **ticked explicitly** at acceptance (D-43 (1)).

**Template criteria**
- [ ] Exact architecture variant implemented as specified (§3, §4)
- [ ] Defining mechanisms explicit and inspectable (single `_step`, single `_logits`)
- [ ] Public API and tensor contracts documented (§5, §15)
- [ ] Configuration validated (§9)
- [ ] Initialisation justified and implemented as §7.1
- [ ] Numerical and edge-case behaviour handled (§10)
- [ ] Shared utilities used appropriately (§6.4)
- [ ] General tests pass (§13.2 V1 – V5, R1 – R14, K1 – K2)
- [ ] Architecture-specific mathematical tests pass (§13.2 M1 – M11)
- [ ] Checkpoint and generation behaviour verified (§14, §15; K1 – K2, 8a – 8d)
- [ ] CPU and available accelerator checks completed
- [ ] No unresolved material code-review issue
- [ ] User has run and reviewed the implementation locally
- [ ] User explicitly approves completion

**Model 6 criteria**
- [ ] Independent reference imports only `torch` and agrees with the model under criterion A (R1)
- [ ] Mathematical invariance, causality, frozen state and padded-logit tests pass (R1 – R3, R6 – R8)
- [ ] All equivalence tests use perturbed parameters (M6-IMP9)
- [ ] Deterministic generation tests pass (8a – 8d)
- [ ] Analytic Jacobians, G2, BPTT formulas and G3 agree with autograd (M1 – M4); finite differences pass (M5)
- [ ] Diagnostics are isolated from the training model (M6) and training is bit-identical with diagnostics on and off (T5)
- [ ] C1 shared-start equals the definition (M9)
- [ ] Initialisation tests pass (I1 – I5)
- [ ] Prohibited-module source scan passes (R14)
- [ ] §7.2 initial statistics reported as [O] at g = 1, s² = 0.5, with deviations from the predeclared predictions, and no automatic change
- [ ] All Stage 4 and Stage 5 items needed before training resolved by recorded decisions
- [ ] Test split untouched

---

## 18. Decisions

**Identifier note.** Identifiers recorded in D-38 – D-43 are final: M6-RQA – RQC, M6-E1 – E7, M6-HA1 – HC2, M6-ARC1 – ARC6, M6-TRN1 – TRN9, M6-INI1 – INI6, M6-H1, M6-G1 – G3, M6-C1, M6-O1 – O3. **Every other identifier below is a PROPOSED identifier** until this specification is approved, independently of the status of its content. Prefixes follow D-43 (2); test numbers (V, R, I, K, M, T, 8a – 8d) are section-local labels, not decision identifiers.

### 18.1 Accepted global decisions (record §11)

| Local IDs | Content | Global |
|---|---|---|
| M6-RQA – RQC; M6-E1 – E7; M6-HA1 – HC2 | Research agenda (expectations and hypotheses are not decisions) | D-38 |
| M6-ARC1 | Single-layer Elman recurrence, tanh, one hidden bias | D-39 |
| M6-ARC2 | Direct character-to-hidden lookup | D-39 |
| M6-ARC3 | Zero initial hidden state; no carry-over between names | D-39 |
| M6-ARC4 | Linear output projection to 27 classes | D-39 |
| M6-ARC5 | Full sequences via `make_next_token_sequences`; full BPTT through autograd, no truncation | D-39 |
| M6-ARC6 | Manual step-by-step recurrence; `nn.RNN` prohibited | D-39 |
| M6-TRN1 – TRN9 | Training methodology (§4.6) | D-40 (PROVISIONAL) |
| M6-INI1 – INI6 | Initialisation direction and provisional values (§7) | D-41 (PROVISIONAL) |
| M6-H1, M6-G1 – G3, M6-C1, M6-O1 – O3 | Diagnostic framework (§12) | D-42 |
| — | Documentation and identifier conventions | D-43 |

### 18.2 Approved by the user (2026-11-14); to be recorded at specification approval

| ID | Item | Approved for |
|---|---|---|
| M6-ARC7 | H = 128 | v0.1 |
| M6-ARC8 | PAD never looked up; right-padded positions freeze the hidden state; invalid targets excluded from the loss | v0.1 |
| M6-ARC9 | Five raw `nn.Parameter` tensors with explicit matrix operations and a manual recurrence | v0.1 |
| M6-IMP1 | `forward(inputs)` derives validity from PAD positions and returns logits only | v0.1 |
| M6-IMP2 | One shared core: `_step` (returns state and pre-activation), `_logits`, `_recurrence` with optional pre-activation collection disabled on the training path; autograd preserved, no detaching inside the model (§6.3) | v0.1; extended v0.2 |
| M6-IMP3 | Diagnostics in `models/recurrent/rnn_diagnostics.py`; analytic diagnostics on a float64 CPU copy, never altering the training model or consuming any RNG stream (§12.1) | v0.1 (subject to final review); extended v0.2 |
| M6-INI1, M6-INI6 (details) | Deterministic CPU float64 Haar construction (§7.1); configurable `recurrent_gain` and `input_init_variance`, provisional defaults 1.0 and 0.5 | v0.1 |
| M6-IMP8 (principles) | Mathematical invariance mandatory; bitwise determinism for identical calls; bitwise identity across batch shapes observational; float64-reference, float32-invariance and precision criteria subject to documented rationale (§13.1; the numerical values are PROPOSED) | v0.2 |
| M6-IMP9 | Perturbed-parameter fixture rule (§13.1) | v0.2 |
| M6-IMP10 | Padded-position logits computed from the frozen final state; no predictive meaning; excluded from all losses and metrics (§4.1) | v0.2 |
| M6-TRN17 | Training procedure in `models/recurrent/rnn_training.py` (`RNNTrainingConfig`, `train_rnn`, `evaluate_nll`); `rnn.py` limited to architecture and model behaviour; no shared trainer | v0.2 |
| M6-DGN8 | Predeclared initial-statistics procedure (§7.2) | v0.2 |
| — (§13.2) | Deterministic generation tests 8a – 8c; additional masking, initialisation, validation and diagnostic tests | v0.2 |
| — (§12.6, §12.9) | Shared-start C1 algorithm with verification test M9; corrected G3 complexity; separation of complexity, operation counts and runtime; no approximations | v0.2 |
| — (§1, §6.4) | `tanh_saturation` located in `utils/diagnostics/activations.py` | v0.2 |

### 18.3 Proposed in this draft (awaiting decision at specification approval)

| ID | Proposal |
|---|---|
| M6-IMP4 | Test file layout (§13.2, §16) |
| M6-IMP5 | `configs/models/rnn.yaml` (§9.1) |
| M6-IMP6 | Input-validation rules, including B = 0 allowed (§5.1) |
| M6-IMP7 | Generation contract: `max_length` counts ordinary characters only, one extra draw for EOS, finished rows leave the active set, `RNNSample(text, finished)` (§15) |
| M6-IMP8 (values) | Tolerances A = 1e‑9, B = C = 1e‑5, FD rtol 1e‑6 / atol 1e‑8, with the rationale of §13.1 (A changed from the v0.1 proposal of 1e‑12) |
| M6-IMP11 | Constructor signature, model-local `IncompatibleCheckpointError`, `from_checkpoint` contract (§5.2, §14) |
| M6-DGN9 (candidates) | Candidate O3 categories (§12.7); final categories PENDING (S5) |
| — | Non-finite guard (§4.6); checkpoint resume fields (§14); §12.8 cadence principles |

### 18.4 Pending (must not be assumed)

| ID | Item | Gate |
|---|---|---|
| M6-TRN10 | Definition and value of C | S4 |
| M6-TRN11 | Partial-batch treatment | S4 |
| M6-TRN12 | Learning rate | S4 |
| M6-TRN13 | Learning-rate schedule | S4 |
| M6-TRN14 | Epoch budget, and whether a pilot sets it with the learning rate | S4 |
| M6-TRN15 | Evaluation batch size | S4 |
| M6-TRN16 | Resume semantics (including diagnostic evaluation points across a resume) | S4 |
| M6-TRN18 | Training device (CPU vs MPS) | S4 |
| M6-TRN19 | MPS tolerance and determinism expectations | S4 |
| M6-INI6 (values) | Final g and s² (provisional 1.0 and 0.5); decided after §7.2 has been run, before any training step | S4 entry gate |
| M6-EXP3 | Pilot design (learning rate, budget) | S4 |
| M6-DGN1 | Probe size, stored name IDs, length stratification | S5 |
| M6-DGN2 | Length strata boundaries | S5 |
| M6-DGN3 | Evaluation points E_all (in optimisation steps) and E_sel | S5 |
| M6-DGN4 | C1 d_max and fixed target set | S5 |
| M6-DGN5 | G2 exclusion threshold τ | S5 |
| M6-DGN6 | Diagnostic output format, files and storage precision | S5 |
| M6-DGN7 | Whether M6-H1 is also reported on dev | S5 |
| M6-DGN9 | O3 instability categories and gradient-spike multiple (before any run, including pilots) | S5 |
| M6-DGN10 | Initialisation-only seed set for §7.2 | S5 |
| M6-DGN11 | Grouping of per-position dev NLL (M6-O2) | S5 |
| M6-DGN12 | Parameters analysed at an evaluation point (current; whether the best checkpoint is also analysed) | S5 |
| M6-DGN13 | Per-diagnostic timing measurement on the untrained model (requires authorisation) | S5 |
| M6-DGN14 | Diagnostic scheduling mechanism (`DiagnosticsSchedule` or model-local) and the `train_rnn` diagnostics hook interface | S5 |
| M6-EXP1 | Whether any controlled study follows implementation | Later |
| M6-EXP2 | Number of seeds | Later |
| M6-EXP4 | Test split protected; no Model 6 test evaluation authorised; any future use requires its own decision and a freeze protocol | STANDING |
| M6-EXP5 | Reserved recurrent-gain study (D-41): not authorised | STANDING |

**Count:** 28 rows — 26 open decisions (11 for Stage 4, 13 for Stage 5, 2 later) and 2 standing statuses. (v0.1 §18.4 had 23 rows: 21 open decisions and 2 standing statuses. v0.2 adds M6-TRN19 and M6-DGN10 – DGN14 and withdraws the v0.1 "minimum pair count" parameter of M6-DGN8 together with the slope fit it served.)

### 18.5 Template mapping

All 18 template sections are present with the template's numbering. Additions inside them: §2.3 carries the D-38 agenda; §4.0 (symbols), §4.6 (training procedure), §5.1 (input validation), §5.2 (construction) and §7.2 (initial-statistics procedure) are subsections; §13.1 holds the numerical acceptance policy and §13.2 the full test inventory; §13.4 lists experiments outside the implementation task (precedent `mlp_spec.md` §13.4); §18 is split into accepted, approved, proposed and pending tables (precedent `wavenet_spec.md` §18).

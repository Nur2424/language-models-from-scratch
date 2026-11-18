# Transformer (Models 01 and 02): Specification

**Status:** approved.
**Code:** `models/attention/transformer.py` (both models). **Configs:** `configs/models/transformer_2017.yaml`, `configs/models/gpt2_style.yaml`.
**Study:** Transformer Part I, a controlled comparison with the vanilla RNN and WaveNet with BatchNorm at about 22–24k parameters, under the shared name-batch protocol (`training/transformer_study/`).

Labels: **[T]** mathematics; **[I]** implementation requirement; **[R]** taken from a reference (paper or reference code); **[A]** adaptation to our task; **[C]** experimental control.

---

## 1. Model identity

| | Model 01 | Model 02 |
|---|---|---|
| Name in code | `transformer_2017` | `gpt2_style` |
| Full name | 2017 Transformer, decoder-only adaptation | GPT-2-style decoder-only Transformer |
| References | Vaswani et al. (2017), *Attention Is All You Need*; tensor2tensor as reference code | Radford et al. (2019), *Language Models are Unsupervised Multitask Learners*, and its released code; Radford et al. (2018, GPT-1) as lineage |
| Task | Next-character prediction on names, autoregressive | Same |

Model 01 is **not** the original encoder–decoder. It is the decoder stack without cross-attention, which is the same computation as encoder blocks with a causal mask.

## 2. Role and motivation

### 2.1 Problem addressed

Every earlier model decides by its structure which characters influence a prediction: the MLP by a fixed window, the WaveNet tree by fixed pairwise merging, the RNN by one state rewritten at every step. Attention decides by content: each position takes a weighted average of earlier positions, with weights computed from the characters themselves, and reaches any earlier character in one step.

### 2.2 Why these two architectures

- **Model 01** keeps what defines the 2017 design: sinusoidal positions, √d embedding scaling, post-LayerNorm, ReLU, tied embeddings.
- **Model 02** keeps what defines GPT-2: learned positions, pre-LayerNorm with a final LayerNorm, GELU, biases throughout, the scaled residual initialisation.
- At the same shape, the two differ only in this bundle, so their comparison compares the two **designs**, not single components (Part II isolates components).

### 2.3 Research questions served

T-RQ1 to T-RQ5 of the study (attention vs recurrence and hierarchy at equal capacity; where along a name; 2017 vs GPT-2 design; use of distant characters; attention by distance; cost). This specification defines the models and their diagnostics interface, not the study.

## 3. Exact architecture definition

Shared by both models:

| Component | Definition |
|---|---|
| Vocabulary | K = 27 output classes (`.` for BOS/EOS and a–z); input PAD = 27 is never looked up (§5) |
| Context | T ≤ 16 positions: BOS plus up to 15 characters |
| Width, depth, heads | d, L, H with d_k = d / H; H = 3 for every shape (study plan) |
| Attention | Causal multi-head scaled dot-product self-attention; no cross-attention |
| FFN | Two linear maps d → 4d → d with biases |
| Output | Tied: logits = h Eᵀ, with E the token embedding [27, d]; **no output bias** [R] |
| Dropout | None [C] |

Per model:

| Component | Model 01 | Model 02 |
|---|---|---|
| Token embedding | E[x] · √d [R] | E[x] [R] |
| Positions | Sinusoidal, base 10,000, added [R]; requires even d | Learned P [16, d], added [R] |
| Q, K, V, output projection | No biases [R] | With biases [R] |
| Activation | ReLU [R] | GELU, tanh approximation [R] |
| LayerNorm placement | Post: x ← LN(x + sublayer(x)) [R] | Pre: x ← x + sublayer(LN(x)) [R] |
| Final LayerNorm | None [R] | Yes, before the output [R] |
| LayerNorm ε | 1e-6 [R] | 1e-5 [R] |

Adaptations and controls:

| Original choice | Here | Label | Reason |
|---|---|---|---|
| Encoder–decoder, cross-attention | Decoder only | [A] | No source sequence |
| Bidirectional encoder attention | Causal mask | [A] | Autoregressive prediction |
| Dropout 0.1 | None | [C] | Consistent with every earlier model |
| Label smoothing 0.1 (Model 01) | None | [C] | Keeps NLL comparable |
| Adam with warm-up (Model 01); Adam (Model 02) | Shared plain-SGD protocol | [C] | Comparable training; AdamW checked separately in the pilot |
| Context 512 / 1024 | 16 | [A] | Names are at most 15 characters |

## 4. Mathematical formulation

### 4.1 Core equations [T]

Input ids x₀ … x_{T−1}, with x₀ = BOS. Embedding:

```latex
\text{Model 01: } h^{(0)}_t = \sqrt{d}\,E[\hat x_t] + \mathrm{PE}(p_t) \qquad
\text{Model 02: } h^{(0)}_t = E[\hat x_t] + P[p_t]
```

with p_t the position of row t (default p_t = t; §4.4) and x̂_t = x_t at valid positions, 0 at padding (§5).

Block ℓ, Model 01 (post-LN):

```latex
u = \mathrm{LN}_1\big(h + \mathrm{MHA}(h)\big), \qquad h' = \mathrm{LN}_2\big(u + \mathrm{FFN}(u)\big)
```

Block ℓ, Model 02 (pre-LN):

```latex
u = h + \mathrm{MHA}(\mathrm{LN}_1(h)), \qquad h' = u + \mathrm{FFN}(\mathrm{LN}_2(u))
```

Output: Model 01 z_t = h_t^{(L)} Eᵀ; Model 02 z_t = LN_f(h_t^{(L)}) Eᵀ.

### 4.2 Component equations [T]

Multi-head attention, heads h = 1 … H, d_k = d / H:

```latex
Q = XW_Q (+b_Q),\ K = XW_K (+b_K),\ V = XW_V (+b_V); \quad
\text{head}_h = \mathrm{softmax}\!\Big(\frac{Q_h K_h^\top}{\sqrt{d_k}} + M\Big) V_h; \quad
\mathrm{MHA}(X) = [\text{head}_1,\dots,\text{head}_H]\,W_O (+b_O)
```

Causal mask: M_{ts} = 0 if s ≤ t, −∞ if s > t (rows t = queries, columns s = keys).

FFN: φ(xW₁ + b₁)W₂ + b₂, with φ = ReLU (Model 01) or GELU_tanh(z) = ½ z (1 + tanh(√(2/π) (z + 0.044715 z³))) (Model 02).

LayerNorm over the d features of one position: LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β, σ² the biased variance.

Sinusoidal encoding, i = 0 … d/2 − 1: PE(p, 2i) = sin(ω_i p), PE(p, 2i + 1) = cos(ω_i p), ω_i = 10000^(−2i/d).

### 4.3 Parameterization

| Model | Per block | Embeddings and output | Total |
|---|---|---|---|
| 01 | 12d² + 9d (attention 4d²; FFN 8d² + 5d; two LNs 4d) | 27d (tied) | L(12d² + 9d) + 27d |
| 02 | 12d² + 13d (attention 4d² + 4d; FFN 8d² + 5d; two LNs 4d) | 27d (tied) + 16d (positions) + 2d (final LN) | L(12d² + 13d) + 45d |

Study shapes (H = 3):

| Shape | d_k | Model 01 | Model 02 |
|---|---|---|---|
| L = 1, d = 42 | 14 | 22,680 | 23,604 |
| L = 2, d = 30 | 10 | 22,950 | 23,730 |
| L = 3, d = 24 | 8 | 22,032 | 22,752 |

### 4.4 State and information flow

- No state is carried between names or between steps. Each forward pass reads one whole (padded) name.
- **Leakage-free by induction [T]:** the output at row t depends only on x̂₀ … x̂_t. Embeddings, FFN, residual additions and LayerNorm act on one row at a time; attention at row t mixes only rows s ≤ t. Hence one parallel pass equals T separate passes over the prefixes.
- Row 0 (BOS) attends only to itself, so the first prediction is the same for every name.
- **Positions as an input [I]:** `forward` accepts an optional `positions` tensor [B, T] (default 0 … T − 1). Attention order is always row order; `positions` only selects the positional vectors. This enables the position-preserving truncation diagnostic (§12) without a second code path.

### 4.5 Assumptions and limitations

- T ≤ 16: no claim about longer sequences.
- Sinusoidal base 10,000 at d ≤ 42: most frequency pairs barely turn over 16 positions (for d = 30, only pairs 0–4 turn by more than 1 radian), so Model 01's positional signal is concentrated in about 10 dimensions. Faithful, recorded, not changed.
- Learned positions 13–15 are trained by fewer than 40 names in the whole dataset; Model 02's late position vectors stay close to initialisation.
- Attention weights are not a measure of influence.

## 5. Tensor and interface contract

| Item | Contract |
|---|---|
| `forward(inputs, positions=None)` | inputs [B, T] int64, 1 ≤ T ≤ 16, values 0 … 26 at valid positions and exactly PAD (27) after them, right-contiguous; returns logits [B, T, 27] float32 |
| Padding | PAD is replaced by 0 before the embedding lookup; outputs at padded rows are defined but meaningless and must be excluded from the loss |
| `positions` | [B, T] int64 in 0 … 15; default `arange(T)`; must be non-decreasing along T at valid rows |
| `forward_with_internals(inputs, positions=None)` | Same logits, plus per layer: attention weights [B, H, T, T], residual stream before each sublayer [B, T, d], LayerNorm inputs; used only by diagnostics and tests |
| Device, dtype | CPU float32 for the study; float64 for reference tests |
| Errors | `ValueError` for T > 16, ids out of range, PAD not right-contiguous, position out of range |

## 6. PyTorch implementation policy

### 6.1 Permitted

`nn.Parameter`, `nn.Module`, `nn.ModuleList`, tensor operations, `torch.softmax`, `torch.tanh`, `torch.relu`, `masked_fill`. All weights are raw `nn.Parameter` tensors (as in the RNN), so no PyTorch default initialisation ever runs.

### 6.2 Prohibited

`nn.MultiheadAttention`, `nn.Transformer*`, `F.scaled_dot_product_attention`, `nn.LayerNorm`, `nn.GELU`, `nn.functional.layer_norm`, `nn.functional.gelu` (all written by hand), `nn.Linear`, packaged language-model modules, dropout.

### 6.3 Manual implementation requirements [I]

- One function each for causal attention, LayerNorm, GELU and the sinusoidal table; both models call the same functions. No second implementation anywhere in `models/`.
- The mask uses −∞ (`masked_fill` with `float("-inf")`) and is built once per T; every row keeps its diagonal, so no row is fully masked.
- Weight tying: the output uses the **same tensor** as the token embedding (not a copy).

### 6.4 Shared utility integration

`utils.vocabulary.Vocabulary` (K, BOS/EOS, PAD), `utils.data.make_next_token_sequences` (inputs, targets, mask), `utils.reproducibility.make_generator` / `derive_seed(seed, "init")` for initialisation, `utils.checkpoint` for serialization.

## 7. Initialization and parameter sharing

All randomness comes from one `torch.Generator` (`derive_seed(seed, "init")`); nothing else is random.

| Parameter | Model 01 [R] | Model 02 [R] |
|---|---|---|
| Token embedding E | N(0, d^−1) (std d^−½) | N(0, 0.02²) |
| Positions | Fixed sinusoidal (not a parameter) | N(0, 0.01²) |
| W_Q, W_K, W_V | Glorot uniform on the full [d, d] matrix | N(0, 0.02²) |
| W_O | Glorot uniform | N(0, (0.02 / √(2L))²) |
| W₁ | Glorot uniform | N(0, 0.02²) |
| W₂ | Glorot uniform | N(0, (0.02 / √(2L))²) |
| All biases | 0 (FFN only) | 0 |
| LayerNorm γ, β | 1, 0 | 1, 0 |

Sharing: one E for input and output; within a layer all positions share all weights; layers share nothing.

**Expected initial statistics [T]** (checked by tests, §13.2):
- Model 02: scaled attention-score std ≈ 0.01 at d = 30 (nearly uniform attention); logits are small (std of order √d · 0.02 ≈ 0.1), so the initial loss is just above ln 27: measured ln 27 + 0.014 at d = 30, L = 2 (256 random names).
- Model 01: scaled attention-score std of order 1; the post-LN output has unit scale and E has std d^−½, so logits have std of order 1 and the initial loss is clearly above ln 27: measured ln 27 + 0.68 at d = 30, L = 2. Unlike the earlier models, neither starts at exactly ln 27, because tying forbids a zero output layer.
- Token signal vs position signal per dimension (Model 01): ≈ 1.0 vs ≈ 0.71.

## 8. Forward-pass specification

1. Validate inputs (§5); x̂ = inputs with PAD → 0; default positions.
2. Embedding and positions (§4.1).
3. For each block: attention sublayer, FFN sublayer, with the model's LN placement.
4. Model 02: final LN.
5. Logits = h Eᵀ.

Shape trace for d = 30, H = 3, B names: [B, T] → [B, T, 30] → Q, K, V [B, 3, T, 10] → scores [B, 3, T, T] → heads [B, 3, T, 10] → concat [B, T, 30] → FFN [B, T, 120] → [B, T, 30] → logits [B, T, 27].

## 9. Configuration

`TransformerConfig(variant ∈ {"transformer_2017", "gpt2_style"}, d_model, num_layers, num_heads=3, max_len=16)`, frozen dataclass; d_model divisible by num_heads; d_model even for Model 01. d_ff = 4 d_model, LN ε, activation, LN placement, biases and initialisation are fixed per variant in code. Shape values in the YAML files are provisional until the shape pilot.

## 10. Numerical stability and edge cases

| Case | Handling |
|---|---|
| Fully masked row | Impossible by construction (diagonal); asserted in tests |
| Softmax overflow | Subtract the row maximum (`torch.softmax` does) |
| LN of a constant vector | σ² = 0 → output β; ε prevents division by zero |
| T = 1 (BOS only) | Valid: one row, attends to itself |
| B = 0 | Returns [0, T, 27] |
| Non-finite values | Detected by the training driver (category A), not by the model |

## 11. Complexity and resource profile

Per name and layer: projections 4·T·d² multiply-adds, scores and mixing 2·T²·d, FFN 8·T·d². At d = 30, T = 16, L = 2 this is about 0.25 M multiply-adds per name forward, of the same order as the RNN. Memory is dominated by the attention weights [B, H, T, T], negligible at T = 16. Measured times are reported by the study (T-RQ5).

## 12. Diagnostics

Provided by `forward_with_internals` (§5); computed by the study's scripts, not inside training:
- initial statistics (§7);
- per layer: residual stream norm, LN input std, gradient norm (from the training driver);
- attention weights by distance and entropy per head (T-RQ4);
- **BOS-restart** (positions reset: the truncated prefix starts at BOS, position 0) and **position-preserving truncation** (proposed design: BOS stays at position 0, the removed characters are deleted, and the kept characters keep their original positions via `positions`), on the F₈ subset (T-RQ3).

## 13. Test plan

### 13.1 General tests

- Output shape, dtype, device; B = 0 and T = 1.
- Exact parameter counts for the three study shapes, both models (§4.3).
- Input validation errors (§5).
- Determinism: same seed → identical parameters and outputs.
- Weight tying: output weight is the embedding tensor itself (same storage); one gradient reaches it from both uses.
- Prohibited modules absent (§6.2), checked over `model.modules()` and the source.

### 13.2 Mathematical verification

- **Independent reference implementation** in float64 (`tests/models/attention/transformer_reference.py`): loops over heads and positions, its own softmax, LayerNorm and GELU; logits equal to the model's to 1e-12 for both models and all three shapes.
- **No leakage:** changing every token after row t (characters or PAD) leaves rows ≤ t unchanged (to float32 rounding, 1e-6); the output at row t equals a forward pass on the prefix of length t + 1 alone.
- **Mask:** attention weights are exactly 0 above the diagonal; every row sums to 1.
- **Padding:** outputs at valid rows do not depend on what PAD is replaced with.
- **Sinusoidal:** values against the formula; the shift property PE(p + k) = R(k) PE(p) per frequency pair; PE(p) · PE(q) depends only on p − q.
- **LayerNorm, GELU, ReLU:** against their formulas; LN scale invariance LN(cx) = LN(x).
- **Structure:** post-LN vs pre-LN order verified with a hand-built two-block example; no final LN in Model 01.
- **Initialisation statistics:** empirical std of each parameter group within tolerance of §7 (over seeds); initial loss and attention-score std as in §7.
- **`positions`:** default equals `arange(T)`; with explicit positions equal to the default, outputs are identical.
- **Gradient check:** `torch.autograd.gradcheck` in float64 on a tiny configuration (d = 6, H = 3, L = 2) for both models.

### 13.3 Learning and behaviour checks

- Overfitting check: on 32 names, 300 SGD steps reduce the training NLL below 1.0 for both models.
- Generation (§15) produces only valid ids, stops at EOS, and never exceeds 15 characters.

## 14. Checkpoint and serialization contract

Checkpoints through `utils.checkpoint`: `model_state_dict`, `TransformerConfig`, the vocabulary, the parameter SHA-256. `from_checkpoint` rebuilds the model and refuses a different variant, shape or vocabulary. The sinusoidal table is not stored (recomputed).

## 15. Generation or inference behaviour

`generate(num_samples, generator, max_chars=15, temperature=1.0)`: start from BOS; at each step feed the whole prefix, sample from the last row's softmax; stop at EOS. A sample reaching 15 characters without EOS is marked unfinished.

## 16. Implementation scope and boundaries

- **In scope:** `models/attention/transformer.py`, the two YAML configs, `tests/models/attention/` (unit tests and the reference implementation).
- **Out of scope:** training loops (in `training/transformer_study/`), any change to shared utilities, other models or published studies; KV caching; dropout; mixed precision.

## 17. Acceptance criteria

- [x] Specification approved.
- [ ] All tests of §13 pass in the workspace and natively on the author's Mac.
- [ ] Reference implementation agreement to 1e-12 for both models and the three shapes.
- [ ] Parameter counts match §4.3 exactly.
- [ ] Completion approved by the author before the study runs start.

## 18. Open decisions

| # | Decision | Proposal |
|---|---|---|
| 1 | Position-preserving truncation design | BOS at position 0, removed characters deleted, kept characters at original positions (§12); to confirm |
| 2 | GPT-2 residual scaling | The paper describes it; reimplementations use 0.02 / √(2L) on W_O and W₂; adopted |
| 3 | Initial loss above ln 27 | Accepted and recorded (§7); the divergence rule (train NLL > ln 27 at an epoch ≥ 1) is unchanged |
| 4 | Model 01 LayerNorm ε = 1e-6 | Kept. Consistent with TensorFlow-style reference implementations (ε inside the square root, 1e-6); not confirmed in the tensor2tensor source itself. Immaterial at this scale: LayerNorm inputs have variance of order 1 |

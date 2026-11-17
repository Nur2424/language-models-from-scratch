# Model Specification — BatchNorm MLP

**Status:** v0.2, implemented, tested and accepted. All design decisions are approved (2026-11-10, including B1–B5 of v0.1). The k = 3 learning-rate pilot (E0, §13.4) was run on 2026-11-17 by `training/mlp_study/run_mlp_study.py`, and the training settings that were **PENDING PILOT** are now decided: **η = 0.03, constant, `max_epochs` = 26** (inherited from the MLP) (§18). Results are in `results/experiments/mlp_study/`. The markers **PENDING PILOT** in the body below are kept as the record of what was open when the specification was approved.
**Governing standard:** `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md`
**Template:** `MODEL_SPECIFICATION_TEMPLATE.md`
**Structural reference:** `models/feedforward/mlp_spec.md` v0.3 (accepted). Wherever this document says "as in the MLP", the MLP spec section named applies unchanged, with `MLP` names replaced by the names defined here.

Labels, as in the MLP specification:
- **[T] theory:** follows from the equations under the stated assumptions.
- **[I] implementation requirement:** must hold for the code; verified by tests (§13).
- **[O] observation:** measured in scratch checks (group split, seed 42, train-only vocabulary), never in a recorded run. Preliminary; not a substitute for tests.
- **[L] legacy:** from the scripts and notebooks of the first version of this project, available in the repository history. Uncontrolled (leaky split, sampling with replacement, unseeded or differently seeded RNG, different normalisation conventions). **Not comparable** with current results.

Status markers: **APPROVED**, **PENDING PILOT**, **PENDING APPROVAL**.

The document separates three layers:
- **Architecture** (§3, §4.1–4.5, §7, §8): what the model computes in each mode.
- **Reference training procedure** (§4.6, §4.7, §9, §14): one model-local, deterministic mini-batch SGD procedure with per-epoch recalibration.
- **Experiments** (§13.4): the learning-rate pilot and optional investigations, outside the implementation task.

## 1. Model identity

- **Model name:** `batchnorm_mlp`
- **File path:** `models/feedforward/batchnorm_mlp.py`
- **Model family:** feedforward neural language models
- **Exact architecture variant:** the accepted MLP (Model 3) with batch normalisation inserted between the hidden linear map and the tanh:
  - three previous tokens looked up in **one shared** embedding table `E ∈ ℝ^{27×10}`, concatenated to 30 dimensions;
  - `Linear(30, 200, bias=False)` (no hidden bias: β takes its role);
  - `BatchNorm1d(200)`: per-unit normalisation over the batch in training; fixed, recalibrated full-training-set statistics in evaluation;
  - `tanh`, then `Linear(200, 27)` with bias, giving 27 logits;
  - no direct input-to-output connections, no dropout, no regularisation;
  - **12,097** trainable parameters (200 more than the MLP).
- **Specification version:** 0.2 (draft)
- **Reference implementation(s):** the BatchNorm phases of the activations-and-BatchNorm script and notebook, and the flat k = 3 BatchNorm MLP of the WaveNet notebook, from the first version of this project, available in the repository history [L].
- **Shared utility dependencies:** exactly as the MLP (§1 of `mlp_spec.md`). No shared utility provides per-feature streaming moments (`utils.statistics.Moments` summarises all elements of a tensor together), so the per-unit merge of §4.7 is model-local.

## 2. Role and motivation

### 2.1 Problem addressed
The same as Model 3: `P(x_t | x_{t-3}, x_{t-2}, x_{t-1})` over 27 classes, with BOS-filled windows.

### 2.2 Why this architecture?
Batch normalisation fixes the scale and offset of every hidden pre-activation *during training*, whatever `W₁` and `E` do. It is the project's first model whose function depends on the **mode**:
- in **train mode** each example's output also depends on the other examples in its batch;
- in **eval mode** it is a deterministic function of the window, using statistics estimated beforehand.

Compared with Model 3, the only change is the normalisation layer (and the removal of the redundant hidden bias). Data, initialisation of the shared parameters, optimiser, batching, budget and selection rules are identical, so differences in results can be attributed to BatchNorm, up to the independently selected learning rate (§13.4).

### 2.3 Learning and research questions
- Established [T]: in train mode the normalised pre-activations have zero mean and unit (biased) variance per unit within each batch; the forward pass is invariant to positive rescaling of each row of `W₁` up to ε (§4.3); `b₁` would receive exactly zero gradient.
- Questions for experiments (§13.4):
  1. Which learning rate gives a stable baseline? (PENDING PILOT)
  2. How do dev NLL, saturation and gradient flow compare with Model 3 at equal budget?
  3. How far are PyTorch's EMA running statistics from the exact population statistics at each epoch end?
  4. How large is the train-mode/eval-mode gap (running training NLL vs fixed-parameter full-train NLL)?

## 3. Exact architecture definition

- **Architecture definition:** `e_i = E[x_i]`; `u = [e₁; e₂; e₃]`; `a = W₁u`; `y = BN(a)`; `h = tanh(y)`; `z = W₂h + b₂`.
- **Included components:**
  - `nn.Embedding(27, 10)`, shared across positions;
  - concatenation in position order (oldest first);
  - `nn.Linear(30, 200, bias=False)`;
  - `CalibratedBatchNorm1d(200)`: a minimal subclass of `nn.BatchNorm1d` (§3.1);
  - `nn.Tanh()`;
  - `nn.Linear(200, 27)` with bias.
- **Excluded components:** the hidden bias `b₁`; direct connections; additional hidden layers; dropout and weight decay; weight tying; a PAD row; positional embeddings; BatchNorm anywhere other than before the tanh; BatchNorm on the output logits.
- **Causal behaviour:** causal (three previous tokens only).
- **Autoregressive generation:** yes, always in eval mode (§15).
- **Deliberate simplifications:** fixed context length, one hidden layer, float32 parameters, fixed ε and momentum.

### 3.1 The normalisation module (APPROVED, B1)

The approved decisions require two sets of statistics: PyTorch's EMA (kept for diagnostics) and recalibrated full-training-set statistics (authoritative for evaluation and generation). The recalibration must not update the EMA or depend on it. This draft keeps them in **separate buffers of one module**:

```python
class CalibratedBatchNorm1d(nn.BatchNorm1d):
    # Parameters (from nn.BatchNorm1d): weight = γ [H], bias = β [H]
    # EMA buffers (from nn.BatchNorm1d, diagnostic only): running_mean [H], running_var [H], num_batches_tracked []
    # Authoritative inference buffers (added):            inference_mean [H], inference_var [H], num_calibrations [] (int64)
    def forward(self, a):            # a: [B, H]
        if self.training:            # batch statistics (biased variance); updates the EMA buffers
            return super().forward(a)
        return F.batch_norm(a, self.inference_mean, self.inference_var, self.weight, self.bias,
                            training=False, momentum=0.0, eps=self.eps)   # never reads or writes the EMA
    def set_inference_statistics(self, mean, var) -> None   # validates, copies, increments num_calibrations
```

Why this structure:
- The EMA evolves exactly as in standard PyTorch training, uninterrupted across epochs, and is never overwritten.
- Eval-mode normalisation reads only `inference_mean` / `inference_var`; nothing can silently fall back to the EMA.
- The module stays an `nn.BatchNorm1d` (one registered submodule, `bn`), so hooks, `parameter_breakdown` and `state_dict` work as usual; all buffers are checkpointed automatically.

Alternatives considered (not recommended):
- **(a) Overwrite `running_mean`/`running_var` with the calibrated values.** Simplest, but the next epoch's EMA would continue from the calibrated values, so the EMA would depend on the recalibration, violating the approved separation.
- **(b) Swap the EMA out and back in around evaluation.** Satisfies the separation but is stateful and error-prone.

## 4. Mathematical formulation

Symbols as in the MLP: `V = 27`, `k = 3`, `d = 10`, `H = 200`, `fan_in = k·d = 30`, `ε = 1e-5`, `ρ = 0.1` (momentum). Ids lie in `{0, …, 26}`; PAD (27) never enters the model.

### 4.1 Core equations (architecture)

For a batch of contexts `x_n = (x_{n,1}, x_{n,2}, x_{n,3})`, n = 1…B:

    e_{n,i} = E[x_{n,i}]            ∈ ℝ^{10}
    u_n     = [e_{n,1}; e_{n,2}; e_{n,3}]  ∈ ℝ^{30}
    a_n     = W₁ u_n                ∈ ℝ^{200}      (W₁ ∈ ℝ^{200×30}, no bias)
    y_n     = BN(a_n)               ∈ ℝ^{200}      (mode-dependent, below)
    h_n     = tanh(y_n)
    z_n     = W₂ h_n + b₂           ∈ ℝ^{27}       (W₂ ∈ ℝ^{27×200}, b₂ ∈ ℝ^{27})

The position blocks `W₁[:, 10p:10p+10]` are as in the MLP (§4.1 there).

**Train mode** (per unit j, over the batch):

    μ_j  = (1/B) Σ_n a_{nj}
    σ²_j = (1/B) Σ_n (a_{nj} − μ_j)²                 (biased batch variance)
    â_{nj} = (a_{nj} − μ_j) / √(σ²_j + ε)
    y_{nj} = γ_j â_{nj} + β_j

**Eval mode** (per unit j, per example; no dependence on other examples):

    y_{nj} = γ_j (a_{nj} − m_j) / √(v_j + ε) + β_j

where `m = inference_mean` and `v = inference_var`: the exact mean and biased variance of the pre-BatchNorm activations over the full training set under the current parameters (§4.7).

### 4.2 Component roles

- **γ (scale) and β (shift):** after normalisation, every unit can still take any affine form `γ â + β`; with `γ = σ`, `β = μ` the layer reproduces the identity on that batch, so no expressive power is lost relative to an affine pre-activation.
- **β replaces b₁** [T]. In train mode a constant added to `a_{·j}` is removed by subtracting `μ_j`; in eval mode it is removed by `m_j`, which is recalibrated with the same parameters. `b₁` would therefore be redundant (its train-mode gradient is exactly zero, §4.3).
- **ε:** keeps the division finite when a unit is (nearly) constant over a batch. It bounds the gain `γ_j/√(σ²_j + ε)` by `γ_j/√ε ≈ 316 γ_j`. For unit-scale pre-activations it is negligible.

### 4.3 Gradients (train mode)

Let `g_{nj} = ∂L/∂y_{nj}` for the batch loss `L`. Then, per unit j:

    ∂L/∂γ_j = Σ_n g_{nj} â_{nj},      ∂L/∂β_j = Σ_n g_{nj}
    ∂L/∂a_{nj} = (γ_j / √(σ²_j + ε)) · ( g_{nj} − (1/B) Σ_m g_{mj} − â_{nj} · (1/B) Σ_m g_{mj} â_{mj} )

Consequences [T]:
1. **Cross-example coupling.** The gradient of each example depends on the whole batch (through μ, σ² and the two batch means). The per-example losses are not independent functions of the parameters.
2. **Zero-sum:** `Σ_n ∂L/∂a_{nj} = 0` exactly. A hidden bias would get zero gradient. [O] With a float64 check: `max |Σ_n ∂L/∂a_n| = 2.2e-16`.
3. **Approximate scale invariance:** for `c > 0`, the train-mode output with `cW₁` equals the output with `W₁` when ε = 0. With ε > 0 the identity holds approximately ([O] c = 5: max output difference 9.8e-5). Correspondingly, `Σ_n â_{nj} ∂L/∂a_{nj} = (γ_j/√(σ²_j+ε)) Σ_n g_{nj} â_{nj} · ε/(σ²_j+ε) ≈ 0`, so each row gradient of `W₁` is nearly orthogonal to the row. Under SGD the row norms of `W₁` tend to grow, and the effective step on the direction of a row scales like `η/‖w_j‖²`. The initial scale of `W₁` thus affects training only through this effective learning rate and through ε.
4. **Shared-embedding accumulation:** unchanged from the MLP (§4.2 there), with `δ_n = ∂L/∂a_n` now given by the formula above.

### 4.4 Parameterization

| Group | Module | Shape (PyTorch) | Count | Role |
|---|---|---|---|---|
| `E` | `emb.weight` | `[27, 10]` | 270 | token embeddings, shared across positions |
| `W₁` | `hidden.weight` | `[200, 30]` | 6,000 | three position blocks; **no bias** |
| `γ` | `bn.weight` | `[200]` | 200 | BatchNorm scale |
| `β` | `bn.bias` | `[200]` | 200 | BatchNorm shift |
| `W₂` | `out.weight` | `[27, 200]` | 5,400 | output projection |
| `b₂` | `out.bias` | `[27]` | 27 | output bias |
| **total** | | | **12,097** | |

General formula: `V·d + (k·d)·H + 2H + H·V + V`.

**Buffers** (not parameters, not trained, all in `state_dict`):

| Buffer | Shape | Dtype | Meaning |
|---|---|---|---|
| `bn.running_mean`, `bn.running_var` | `[200]` | float32 | PyTorch EMA (diagnostic only) |
| `bn.num_batches_tracked` | `[]` | int64 | number of train-mode BN forwards |
| `bn.inference_mean`, `bn.inference_var` | `[200]` | float32 | authoritative eval statistics (§4.7) |
| `bn.num_calibrations` | `[]` | int64 | number of recalibrations performed; 0 = never calibrated |

### 4.5 Assumptions, limits and the conditional-entropy bound

The dataset numbers are the MLP's (§4.5 there): `M_train = 182,666`, `M_dev = 22,801`, 5,402 distinct training contexts, `Ĥ(Y|W) ≈ 1.8681`, `H(Y) ≈ 2.8224`, ≈ 0.82% unseen dev contexts.

- **The bound applies to the eval-mode model.** In eval mode the prediction is a function of the window alone, so the fixed-parameter full-train NLL satisfies `L_train ≥ Ĥ(Y|W)`, exactly as for the MLP. It remains a lower bound, not an achievable target, and is not a bound on dev NLL.
- **It does not apply to train-mode predictions.** In train mode a prediction also depends on the other windows in the batch, so it is not a window-only function. The running training NLL (§4.6) is therefore not covered by the bound and is not compared with it.
- **Non-convexity:** as for the MLP; results depend on seeds (three seeds for final results).
- **Repeated-BOS encoding and representational limits:** as for the MLP.

### 4.6 Reference training procedure

Everything in §4.6 of `mlp_spec.md` applies unchanged, except as follows.

**Mode discipline** [I] (APPROVED):
- `model.train()` is used **only** for the forward passes of SGD steps.
- Recalibration, evaluation, per-unit saturation, epoch-end diagnostics and generation never run a train-mode forward of `bn`. A train-mode forward updates the EMA buffers even under `torch.no_grad()` ([O] confirmed for PyTorch 2.x).

**Batch-size constraint** [I] (APPROVED): every training batch must contain at least 2 windows (a batch of 1 has zero batch variance; PyTorch raises in train mode, and a batch of 0 silently increments `num_batches_tracked`). Therefore:
- `BatchNormMLPTrainingConfig.batch_size ≥ 2`;
- before training (and on resume), `train_batchnorm_mlp` checks `min(sampler.batch_sizes()) ≥ 2`, i.e. rejects `M_train mod batch_size == 1`, and raises `ValueError` naming `M_train` and `batch_size`;
- `forward` in train mode also raises `ValueError` if `B < 2` (§8).
- [O] With `M_train = 182,666` and batch size 32: 5,708 batches of 32 and one of 10. The constraint is satisfied.

**Partial batch** [T]:
- The final batch of 10 is kept (approved infrastructure policy) and normalised with its own statistics. These are noisier: [O] at initialisation, the RMS error of the batch mean relative to the population std is 0.18 for B = 32 and 0.31 for B = 10; for the batch std, 0.13 and 0.24.
- Unlike for the MLP, the partial batch does **not** give the same expected gradient as a full batch: in train mode the per-batch objective itself depends on the batch size. It is one step per epoch, with weight 10/182,666.
- Because evaluation uses recalibrated statistics (§4.7), the partial batch does not affect evaluation, unlike PyTorch's EMA, in which the final batch of each epoch would carry weight ρ = 0.1.

**Per-step objective** (APPROVED): mean reduction over the actual batch size, as for the MLP:

    L_b(θ) = (1/|b|) Σ_{i∈b} ℓ_i(θ; b),     token_nll(model(x_b), y_b, reduction="mean")

where `ℓ_i(θ; b)` now depends on the batch `b` through μ and σ².

**Order of operations within one step** [I]: identical to the MLP (§4.6 there: forward → loss → finite-loss check → backward → finite-gradient check → diagnostics → `optimizer.step()`). Note: the forward (step 1) has already updated the EMA buffers when a later check fails. The run stops, so this affects only the in-memory model; no checkpoint contains it.

**Epoch end** [I] (APPROVED order):
1. Finish the epoch's SGD steps.
2. **Recalibrate** (§4.7) on all training windows with the current fixed parameters, installing `inference_mean` and `inference_var`.
3. Evaluate full-train and full-dev NLL in eval mode (uses the new inference statistics), plus the epoch-end diagnostics (§12).
4. Then selection and writes exactly as the MLP: `model_best.pt` if dev strictly improved → history row → `checkpoint_last.pt` → after the last epoch `model_final.pt` (§14).

**Reported quantities.** As for the MLP, three are kept distinct:
1. **Running training NLL:** train mode, batch statistics, parameters changing during the epoch.
2. **Full-training NLL:** eval mode, fixed end-of-epoch parameters, recalibrated statistics.
3. **Full-development NLL:** the same, on dev.

For this model the gap between (1) and (2) also contains the train/eval normalisation mismatch.

**Learning rate and schedule:** epoch-based and stateless, as for the MLP. The value is **PENDING PILOT**, selected independently of the MLP (§13.4). Constant LR in the pilot.

**Epoch budget:** the **same** `max_epochs` as the final accepted MLP configuration (APPROVED), which is itself PENDING PILOT. No early stopping.

**Selection and test:** as for the MLP: best = lowest full-dev NLL, earliest epoch on ties; test once on `model_best.pt`.

**Determinism:** as for the MLP. Recalibration is deterministic given the parameters (fixed chunk order, float64 merge on CPU). `nn.BatchNorm1d` adds no randomness.

### 4.7 Recalibration of the inference statistics (APPROVED)

**Definition.** With fixed parameters θ and all training windows `w_1 … w_M`, let `a_n = W₁ u_n ∈ ℝ^{200}` be the pre-BatchNorm activation of window n. For every unit j:

    m_j = (1/M) Σ_{n=1}^{M} a_{nj}
    v_j = (1/M) Σ_{n=1}^{M} (a_{nj} − m_j)²            (biased population variance, divisor M)

These are installed as `inference_mean` and `inference_var`.

**Why the biased divisor** [T, O]: train mode normalises with the biased batch variance. With the biased population variance, a full-batch model (one batch containing all windows) gives **identical** train-mode and eval-mode outputs for the same parameters. [O] On the tiny fixture with one batch of 9: max logit difference 3.3e-6 (float32) with divisor M, but 0.57 with divisor M−1. At M = 182,666 the two divisors differ by a factor 1.0000055.

**Algorithm** [I]: a numerically stable streaming computation, independent of how the windows are chunked up to floating-point precision. Chunks are processed **sequentially**, one at a time:

1. For each chunk `c` of at most `eval_batch_size` consecutive training windows (in index order), compute `A_c = W₁ [E[x]]` (the pre-BN activations, shape `[n_c, 200]`) on the model device under `torch.no_grad()`, **without calling `bn`**. This uses the model-local method `preactivations(contexts)` (embedding → concatenation → `hidden`), which has no mode-dependent behaviour.
2. Move `A_c` to CPU as float64. Compute the chunk statistics with a centred two-pass:
   `n_c`, `mean_c = A_c.mean(0)`, `M2_c = ((A_c − mean_c)²).sum(0)` (each `[200]`).
3. Merge into the running totals `(n, mean, M2)` with the Chan et al. parallel update:

       n'    = n + n_c
       δ     = mean_c − mean
       mean' = mean + δ · n_c / n'
       M2'   = M2 + M2_c + δ² · n · n_c / n'

4. After the last chunk: `m = mean`, `v = M2 / M`. Check that `n == M`, that all values are finite, and that `v ≥ 0`; otherwise raise `FloatingPointError` naming the epoch.
5. Install with `bn.set_inference_statistics(m.float(), v.float())`, which copies into `inference_mean`/`inference_var` on the model device and increments `num_calibrations`.

**Memory discipline** [I] (APPROVED):
- Chunks are processed strictly sequentially: compute `A_c`, compute its per-unit moments `(n_c, mean_c, M2_c)`, merge them into the running totals, then release `A_c` (both the device tensor and its CPU float64 copy) before the next chunk is computed.
- The only state carried between chunks is `(n, mean, M2)`: one integer and two `[200]` float64 vectors.
- The complete `[M, 200]` activation matrix is **never** materialised, on the device or on CPU, and no list of chunks or chunk statistics is accumulated. Peak memory is one chunk (≤ `eval_batch_size × 200` values on the device plus the same in float64 on CPU) plus O(H).

Requirements:
- **Never** average batch means or batch variances, and never compute `E[a²] − E[a]²`.
- The recalibration **does not read or write** `running_mean`, `running_var` or `num_batches_tracked`, and its result does not depend on their values or on the previous inference statistics.
- It does not change `model.training` (it never calls `bn`). It runs under `no_grad` and keeps no graph.
- Precision: the merge is in float64 on CPU on every backend (MPS has no float64); only `A_c` itself is computed in float32 on the model device. The result is independent of `eval_batch_size` up to float32 rounding of `A_c` (per-row matmul results can differ in the last bits between batch sizes on some backends).
- Cost [T]: one forward through embedding and `hidden` over `M_train` windows (≈ 6.2k multiply-adds per window), plus a host transfer of `M_train × 200` float32 values on accelerators (≈ 146 MB per epoch). Cheaper than the full-train evaluation pass that follows.

**Before the first recalibration**, `inference_mean = 0` and `inference_var = 1` (§7). A freshly constructed model can be evaluated (with `W₂ = 0` its logits are 0 in both modes), but every checkpoint written by training has `num_calibrations ≥ 1`.

## 5. Tensor and interface contract

| Item | Shape | Dtype | Device | Meaning |
|---|---|---|---|---|
| `contexts` (input) | `[B, 3]` | int64 | model device | ids in `[0, 27)`, oldest first; PAD rejected |
| `emb(contexts)` | `[B, 3, 10]` | float32 | model device | embeddings |
| `u` | `[B, 30]` | float32 | model device | position-major concatenation |
| `a` (pre-BN) | `[B, 200]` | float32 | model device | `hidden(u)`, no bias |
| `y` (post-BN) | `[B, 200]` | float32 | model device | `bn(a)` |
| `h` | `[B, 200]` | float32 | model device | `act(y)` |
| logits (output) | `[B, 27]` | float32 | model device | `out(h)`; logits, not probabilities |
| batch statistics μ, σ² | `[200]` | float32 | model device | train mode only, internal to `bn` |
| `inference_mean`, `inference_var` | `[200]` | float32 | model device | authoritative eval statistics |
| calibration moments | `[200]` | float64 | CPU | streaming merge state (§4.7) |

- **Batch size:** train mode requires `B ≥ 2`. Eval mode accepts `B ≥ 0`; `B = 0` returns `[0, 27]`.
- **Padding, masks, special tokens, device behaviour:** as for the MLP.

### Public API (to implement in `models/feedforward/batchnorm_mlp.py`)

```python
BN_EPS = 1e-5                    # APPROVED constant (not configurable)
BN_MOMENTUM = 0.1                # APPROVED constant (EMA, diagnostic only)

@dataclass(frozen=True)
class BatchNormMLPConfig:        # model.params in configs/models/batchnorm_mlp.yaml
    context_length: int = 3      # APPROVED; validated int ≥ 1
    embedding_dim: int = 10      # APPROVED; validated int ≥ 1
    hidden_dim: int = 200        # APPROVED; validated int ≥ 1

@dataclass(frozen=True)
class BatchNormMLPTrainingConfig:    # model-local; NOT part of ExperimentConfig
    lr: float                        # PENDING PILOT; required, finite > 0
    max_epochs: int                  # same value as the final MLP config (PENDING PILOT there)
    batch_size: int = 32             # APPROVED; validated int ≥ 2
    lr_decay_epoch: int | None = None    # schedule hook kept identical to the MLP; None in the pilot
    lr_decay_factor: float = 0.1
    eval_batch_size: int = 4096      # evaluation and recalibration chunking only

@dataclass(frozen=True)
class BatchNormMLPSample:            # same semantics as MLPSample
    text: str
    finished: bool

@dataclass(frozen=True)
class EpochRecord:                   # one row of history.jsonl
    epoch: int; lr: float; steps: int
    running_train_nll: float         # (1) train mode, batch statistics, changing parameters
    train_nll: float                 # (2) eval mode, recalibrated statistics, fixed parameters
    dev_nll: float                   # (3) eval mode, recalibrated statistics, fixed parameters
    is_best: bool; seconds: float
    activations: dict                # epoch-end dev activation summary (§12)
    batchnorm: dict                  # epoch-end BatchNorm summary (§12)

@dataclass(frozen=True)
class BatchNormMLPTrainingResult:
    best_epoch: int; best_dev_nll: float; epochs_completed: int; history: list[EpochRecord]

class CalibratedBatchNorm1d(nn.BatchNorm1d):   # §3.1
    def set_inference_statistics(self, mean: torch.Tensor, var: torch.Tensor) -> None

class BatchNormMLPLanguageModel(nn.Module):
    def __init__(self, vocab, config: BatchNormMLPConfig = BatchNormMLPConfig(), *, generator: torch.Generator)
        # submodules emb, hidden (bias=False), bn (CalibratedBatchNorm1d), act (nn.Tanh), out
    def forward(self, contexts: LongTensor[B, k]) -> FloatTensor[B, V]
    def preactivations(self, contexts: LongTensor[B, k]) -> FloatTensor[B, H]   # pre-BN a; mode-independent; validated
    @torch.no_grad()
    def generate(self, num_samples, *, max_length, generator=None, temperature=1.0, top_k=None, top_p=None)
        -> list[BatchNormMLPSample]   # eval mode inside; previous mode restored in `finally`
    @classmethod
    def from_checkpoint(cls, path, *, expected_vocab=None, map_location="cpu") -> "BatchNormMLPLanguageModel"

def calibrate_batchnorm(model, windows: ContextWindows, *, batch_size: int) -> dict
    # §4.7; installs inference statistics; returns {"num_windows": M, "num_calibrations": k}

def evaluate_nll(model, windows: ContextWindows, *, batch_size: int) -> NLLAccumulator
    # eval mode, no_grad, every example; original mode restored in `finally`; never updates the EMA

def per_unit_saturation(model, windows, *, threshold: float = 0.97, batch_size: int) -> torch.Tensor   # [H]
    # eval mode, no_grad; original mode restored in `finally`

def train_batchnorm_mlp(model, train: ContextWindows, dev: ContextWindows, *,
                        training: BatchNormMLPTrainingConfig, seed: int, run: RunManager | None = None,
                        resume: bool = False, diagnostics: DiagnosticsSchedule | None = None,
                        config: Mapping | None = None) -> BatchNormMLPTrainingResult
```

All other names, arguments and behaviours (sample dataclass, `IncompatibleCheckpointError`, history truncation, `run_config`) follow `mlp.py`, with the class and function names above.

## 6. PyTorch implementation policy

### 6.1 Permitted operations and modules
Those of the MLP (§6.1 there), plus:
- `nn.BatchNorm1d` (subclassed as in §3.1), with `eps=BN_EPS`, `momentum=BN_MOMENTUM`, `affine=True`, `track_running_stats=True` passed explicitly;
- `torch.nn.functional.batch_norm` with `training=False` for eval-mode normalisation;
- `nn.Linear(..., bias=False)`.

### 6.2 Prohibited operations and modules
Those of the MLP (§6.2 there), plus:
- relying on any PyTorch default value of a BatchNorm parameter or buffer (all are set explicitly, §7);
- `momentum=None` (cumulative averaging) and `track_running_stats=False`;
- any train-mode forward outside an SGD step (including "warm-up" passes to fill statistics);
- writing `running_mean` / `running_var` / `num_batches_tracked` anywhere in model code (only PyTorch's train-mode forward updates them);
- reading the EMA buffers in eval-mode normalisation;
- estimating inference statistics by averaging batch statistics or via `E[a²] − E[a]²`;
- a hidden bias.

### 6.3 Manual implementation requirements
Those of the MLP, plus: `CalibratedBatchNorm1d`; `preactivations`; the streaming per-unit merge and `calibrate_batchnorm`; the batch-size guards; eval-mode handling in `generate`; the BatchNorm epoch summary.

### 6.4 Shared utility integration
As for the MLP. No shared utility is extended. The per-unit streaming merge is model-local (§1). `batchnorm_mlp.py` imports nothing from `models.*` (independent of `mlp.py`, APPROVED); the duplication of the MLP's training procedure is accepted.

## 7. Initialization and parameter sharing

All APPROVED. Random draws use `make_generator(derive_seed(seed, "init"))` on CPU, in the order `E` then `W₁`, before any `.to(device)`. Modules are created with `nn.utils.skip_init` (as in the MLP), so every parameter **and buffer** is set explicitly:

| Tensor | Shape | Initialization | Rationale |
|---|---|---|---|
| `E` | `[27, 10]` | `N(0, 1)` | as the MLP |
| `W₁` | `[H, k·d]` | `N(0, 1/fan_in)`, std `1/√fan_in`; baseline `1/√30` | as the MLP; keeps `a` at unit scale |
| `γ` (`bn.weight`) | `[200]` | 1 | BN starts as pure normalisation |
| `β` (`bn.bias`) | `[200]` | 0 | same |
| `W₂` | `[27, 200]` | 0 | as the MLP: uniform initial prediction |
| `b₂` | `[27]` | 0 | same |
| `running_mean` | `[200]` | 0 | standard EMA start |
| `running_var` | `[200]` | 1 | standard EMA start |
| `num_batches_tracked` | `[]` | 0 | |
| `inference_mean` | `[200]` | 0 | placeholder until the first recalibration |
| `inference_var` | `[200]` | 1 | placeholder until the first recalibration |
| `num_calibrations` | `[]` | 0 | "never calibrated" |

**Role of the `W₁` scale with BatchNorm** [T]: in train mode the forward is (up to ε) invariant to the scale of each row of `W₁` (§4.3), so the LeCun scale is kept for consistency with Model 3; it now matters only through the effective learning rate. [O] At initialisation on the training windows the pre-activations have per-unit population mean |μ| ≈ 0.25 and std ≈ 0.87 on average, so BN initially re-centres and mildly rescales.

**Zero output initialisation** [T]: with `W₂ = 0`, `b₂ = 0`, the logits are exactly 0 **in both modes**, whatever the normalisation does. The initial loss is exactly `ln 27` in train and eval mode. On the first backward pass:

| Group | First-step gradient |
|---|---|
| `b₂` | `Σ_n g_n` (nonzero) |
| `W₂` | `Σ_n g_n h_nᵀ` (nonzero) |
| `γ`, `β`, `W₁`, `E` | exactly zero (every path passes through `W₂ = 0`) |

From the second step every group receives gradient. This is intended, as in the MLP. No symmetry problem: `E` and `W₁` are random, so `h` differs across units.

**Parameter sharing:** `E` shared across positions; no other sharing or tying.

## 8. Forward-pass specification

1. **Validate** [I], exactly as the MLP (§8 there): tensor type, int64, `ndim == 2`, `shape[1] == context_length`, device, empty-safe range check with the dedicated PAD message.
2. **Train-mode batch size:** if `self.training` and `B < 2`, raise `ValueError` ("BatchNorm needs at least 2 windows per training batch"). This check precedes `bn`, so no buffer is touched.
3. **Embed:** `e = self.emb(contexts)` → `[B, 3, 10]`.
4. **Concatenate:** `u = e.reshape(B, 30)`.
5. **Pre-activation:** `a = self.hidden(u)` → `[B, 200]` (no bias).
6. **Normalise:** `y = self.bn(a)`: train mode → batch statistics and EMA update; eval mode → inference statistics only (§3.1).
7. **Nonlinearity:** `h = self.act(y)`.
8. **Output:** `z = self.out(h)` → `[B, 27]` logits.

`preactivations(contexts)` performs steps 1, 3, 4 and 5 only and returns `a`. Both `forward` and `preactivations` use the same internal code for steps 1–5, so the calibrated `a` is exactly the `a` that `bn` sees.

## 9. Configuration

**Architecture:** `model.params` → `BatchNormMLPConfig` (`configs/models/batchnorm_mlp.yaml`): `context_length: 3`, `embedding_dim: 10`, `hidden_dim: 200` (APPROVED). ε and momentum are code constants, not config fields (APPROVED). Unknown keys are rejected by `load_config`.

**Training:** `BatchNormMLPTrainingConfig` (model-local, as the MLP):

| Field | Default | Validation | Status |
|---|---|---|---|
| `lr` | required | finite, > 0 | **PENDING PILOT** (independent of the MLP) |
| `max_epochs` | required | int ≥ 1 | equal to the final MLP value (APPROVED rule; value PENDING PILOT) |
| `batch_size` | 32 | int **≥ 2** | APPROVED |
| `lr_decay_epoch` | None | None or 1 ≤ value < `max_epochs` | kept for parity with the MLP; None in the pilot |
| `lr_decay_factor` | 0.1 | 0 < value ≤ 1 | as the MLP |
| `eval_batch_size` | 4096 | int ≥ 1 | chunking for evaluation **and** recalibration |

Optimiser (plain SGD, momentum 0, weight decay 0) and loss reduction (mean) are fixed in code.

**Recorded constants.** `run_config(...)` stores `{"experiment": …, "model": asdict(BatchNormMLPConfig), "training": asdict(BatchNormMLPTrainingConfig), "batchnorm": {"eps": 1e-5, "momentum": 0.1, "inference_statistics": "full_train_biased"}}` in `config.json` and every checkpoint. `from_checkpoint` and resume reject a mismatch in the `batchnorm` entry.

**Reproducibility, data, checkpoints, diagnostics, generation:** as for the MLP (§9 there). `generation.max_new_tokens` is used as `max_length`. Pilot seeds 100–102; final seeds 0–2.

## 10. Numerical stability and edge cases

| Case | Behaviour | Test |
|---|---|---|
| Softmax / log-softmax | stable primitives only (as the MLP) | extreme logits finite |
| Initial loss | exactly `ln 27` in both modes | §13.2 |
| Unit constant over a batch | `σ² = 0`; ε keeps the division finite; output `β_j` | dedicated test (identical windows) |
| Unit constant over the training set | `v_j = 0`; eval output `β_j` | covered by the same test in eval mode |
| Train-mode batch of 0 or 1 | `ValueError` before `bn` (no buffer change) | dedicated test |
| Training data with `M mod batch_size == 1` | `ValueError` before any step | dedicated test |
| Eval-mode empty batch | `[0, 3]` → `[0, 27]` | dedicated test |
| Final partial batch (10) | normalised with its own statistics; mean loss over 10 | §13.2 |
| Recalibration precision | float64 CPU merge; partition-independent | §13.2 |
| Non-finite calibration result | `FloatingPointError` naming the epoch | dedicated test |
| Non-finite loss or gradient | as the MLP: `FloatingPointError` before `optimizer.step()` | dedicated tests |
| EMA buffers after a failed step | updated by that step's forward; the run stops and no checkpoint contains them | documented |
| Large learning rate | no stability claim; `W₁` row norms grow under BN (§4.3); pilot | experiment |
| Precision | float32 parameters and buffers; float64 for `gradcheck` (CPU) and the calibration merge | — |
| Backend | BatchNorm, `F.batch_norm` exist on CPU, CUDA and MPS. CUDA determinism as the MLP (`configure_determinism`). MPS: no float64 on device, hence the CPU merge | device tests |

## 11. Complexity and resource profile

- **Parameters:** 12,097 (§4.4); buffers: 802 values (four `[200]` vectors and two scalar counters).
- **Per example (training):** as the MLP minus the hidden bias, plus BatchNorm: O(H) for statistics and normalisation forward and backward. About 11.4k multiply-adds forward.
- **Per epoch:** the MLP's cost (5,709 SGD steps, a full-train and a full-dev evaluation) plus one recalibration pass (embedding and `hidden` only over the training set). The recalibration skips BatchNorm, tanh and the output layer, so it costs less than one evaluation pass (roughly half the multiply-adds per window, plus the host transfer on accelerators). Not measured; it will be timed during implementation and reported as an observation.
- **Memory:** a recalibration chunk of 4096 × 200 values (3.3 MB float32, 6.6 MB float64 on CPU); moments are `[200]` vectors.
- **Main bottleneck:** Python overhead per SGD step, as for the MLP.

## 12. Diagnostics

**Step level** (scheduled, as the MLP, written to `<run>/diagnostics/steps.jsonl`):
- `"gradients"`: `global_grad_norm`; `gradient_summaries(group_by="module")` with groups `emb`, `hidden`, `bn`, `out`; `nonfinite_gradients`.
- `"parameters"`: `ParameterSnapshot` update statistics.
- `"activations"`: an `ActivationMonitor` attached during the SGD forward (train mode, so these are **batch-statistics** activations) on `hidden` (pre-BN), `bn` (post-BN) and `act` (tanh, with `tanh_saturation(0.97)`). No extra forward pass.

**Epoch end** (always with `run`, after recalibration, all in eval mode on **dev**; stored in the `EpochRecord`):
- `activations`: `ActivationMonitor` summaries for `hidden`, `bn`, `act` (saturation) during the dev evaluation, plus the per-unit saturation summary (min, median, max, units ≥ 0.99), as the MLP.
- `batchnorm`:
  - `ema_vs_inference`: per unit, `z_j = (running_mean_j − inference_mean_j) / √(inference_var_j + ε)` and `r_j = log(running_var_j / inference_var_j)`; recorded as mean |z|, max |z|, median |r|, max |r|. This measures how far PyTorch's EMA is from the exact statistics at the epoch end.
  - `gamma` and `beta`: mean, std, min, max.
  - `w1_row_norm`: mean, min, max of the row norms of `W₁` (scale-invariance drift, §4.3).
  - `num_calibrations` and `num_batches_tracked`.

**Mode restoration and overhead:** `evaluate_nll`, `per_unit_saturation` and `generate` record `model.training` on entry and restore it in `finally`. Diagnostics keep no graphs, take parameter snapshots only on scheduled steps, and never run a train-mode forward. No shared utility is extended.

**Experiment-specific** (§13.4): comparison with Model 3 curves; EMA-vs-exact discrepancy over training; saturation dynamics; `W₁` norm growth; γ and β distributions.

## 13. Test plan

File: `tests/models/feedforward/test_batchnorm_mlp.py`. Fixtures as the MLP (SMALL, TINY with Ĥ = 0.212171, V27, real data skipped if absent).

### 13.1 General tests
- [x] Construction; `BatchNormMLPConfig` and `BatchNormMLPTrainingConfig` validation (including `batch_size = 1` rejected); YAML integration; unknown keys rejected
- [x] Input contract as the MLP (shapes, dtype, device, range, PAD, empty-safe range check); eval mode `[0, 3]` → `[0, 27]`; train mode `B ∈ {0, 1}` → `ValueError` with **no buffer change**
- [x] Parameter count 12,097 with breakdown `emb` 270, `hidden` 6,000, `bn` 400, `out` 5,427; general formula for a non-default config; exactly six parameter tensors; `hidden.bias is None`
- [x] Buffers: the six of §4.4 exist with the stated shapes and dtypes and are in `state_dict`
- [x] Explicit initialisation (§7): every parameter and buffer has its stated value; `E`, `W₁` depend only on the init generator; construction does not touch the global RNG
- [x] `batchnorm_mlp.py` imports nothing from `models.*`; `bn` is a `CalibratedBatchNorm1d` and an `nn.BatchNorm1d` with `eps == 1e-5`, `momentum == 0.1`

### 13.2 Mathematical verification
1. **Train-mode oracle:** for random parameters, the model equals a hand-written `onehot·E` → concat → `W₁` → `(a − mean)/√(biased var + ε)·γ + β` → tanh → `W₂`, `b₂` (atol 1e-5).
2. **Eval-mode oracle:** after installing arbitrary inference statistics, the model equals the same pipeline with `(a − m)/√(v + ε)`; the EMA buffers do not affect the output (set them to garbage; output unchanged).
3. **Batch dependence:** in eval mode, the logits of a window are identical alone and inside any batch; in train mode they change when the rest of the batch changes.
4. **EMA update rule:** one train-mode forward updates `running_mean ← 0.9·running_mean + 0.1·μ_B`, `running_var ← 0.9·running_var + 0.1·(unbiased batch variance)`, `num_batches_tracked += 1`, also under `torch.no_grad()`; inference buffers unchanged.
5. **Embedding sharing, position blocks, position sensitivity, logit contract:** as MLP tests 1–3 and 5, in eval mode with installed statistics.
6. **Shared-embedding gradient accumulation:** autograd `∂/∂E` equals the hand computation using the §4.3 BatchNorm backward formula (float64, train mode, a batch with repeated tokens).
7. **BatchNorm identities** (float64, train mode): `Σ_n ∂L/∂a_n = 0` (atol 1e-12); `∂L/∂γ`, `∂L/∂β` match §4.3; with ε set to 0 on a float64 copy, scaling a row of `W₁` by 5 leaves the output unchanged (atol 1e-10).
8. **Finite differences:** `gradcheck` in float64 via `functional_call` over all six parameter groups, in train mode (batch statistics).
9. **Zero output initialisation:** initial full-batch loss `ln 27` in both modes; first-step gradients of `E`, `W₁`, `γ`, `β` exactly zero; `∇W₂`, `∇b₂` match the hand formulas; after one step all groups have gradient.
10. **Initialisation statistics:** as the MLP test 9 (E and `W₁` sampling tolerances, including the non-default `fan_in = 80` case; pre-activation std bound applied to the mean over 20 seeds). γ = 1, β = 0 exactly.
11. **Mean reduction over the actual batch size:** with 42 windows (batches 32 and 10), `train_batchnorm_mlp` equals an independent re-implementation using the per-batch mean, not `sum/32`.
12. **Example-weighted running NLL:** as the MLP test 11.
13. **Evaluation coverage:** as the MLP test 12 (every example, any chunk size), in eval mode.
14. **Recalibration correctness:** after `calibrate_batchnorm`, `inference_mean`/`inference_var` equal `a.mean(0)` and `a.var(0, unbiased=False)` computed in float64 from all pre-activations at once (rel 1e-6, float32 model).
15. **Recalibration streaming and partition independence:** chunk sizes 1, 7, 64, M and M+5 give the same statistics (rel 1e-6 for a float32 model; atol 1e-12 for a float64 model on CPU). A dataset with a large common offset (`|mean| ≫ std`, e.g. a constant added to a unit through hand-set `W₁`) still gives an accurate variance (guards against the `E[a²] − E[a]²` form). A wrapped `preactivations` records every call: calls are sequential, each returns at most `batch_size` rows, the row counts sum to M; a weak reference to each returned chunk is dead before the next call (the chunk was released, not retained).
16. **Recalibration isolation:** recalibration leaves `running_mean`, `running_var`, `num_batches_tracked`, all parameters and `model.training` unchanged in both starting modes; its result does not depend on the previous inference statistics or EMA values (set both to garbage first); `num_calibrations` increments by one.
17. **Full-batch consistency:** on the tiny fixture (one batch of 9), the train-mode logits of a copy of the model equal the eval-mode logits after recalibration (atol 1e-5).
18. **Conditional-entropy bound (eval mode):** fixed-parameter full-train NLL in eval mode after recalibration is ≥ Ĥ(Y|W) − 1e-6, for random and trained parameters (tiny and small sets).

### 13.3 Learning and behaviour checks
- **Tiny-fixture overfitting:** `["ab", "ac", "ab"]`, lr 0.5, 500 epochs, batch 32 (one batch of 9). Eval-mode full-train NLL − Ĥ lies in `[−1e-6, 1e-2]`. [O] Scratch measurement with the approved procedure: gaps 7.9e-4, 7.2e-4, 7.5e-4 for seeds 0–2, a ≥ 12× margin. A debugging check, not a quality claim.
- **Batch-size guard:** training data with `M mod batch_size == 1` raises `ValueError` before any step, with parameters and buffers unchanged.
- **Epoch-end order:** recalibration happens after the last step of each epoch and before evaluation (a monkeypatched recorder sees `[steps…, calibrate, evaluate train, evaluate dev]` per epoch); `num_calibrations == epochs completed`.
- **Mode discipline:** across a full `train_batchnorm_mlp` call, `num_batches_tracked` equals the number of SGD steps exactly (no extra train-mode forward anywhere: evaluation, calibration, diagnostics).
- **Determinism, schedule, selection with ties, non-finite guards, three distinct NLLs:** as the MLP tests, adapted. The running NLL is now expected to differ from the full-train NLL even with a zero learning-rate update (normalisation mismatch).
- **Mode restoration:** `evaluate_nll`, `per_unit_saturation`, `generate` restore `model.training` for both starting modes, also when they raise part-way; they leave every buffer unchanged.
- **Generation:**
  - runs in eval mode even if the model is in train mode, and restores train mode afterwards; uses the inference statistics (output changes when they change; unaffected by the EMA);
  - works with `num_samples = 1` (a train-mode batch of 1 would fail);
  - context shifting equals `make_context_windows`; EOS and `max_length` semantics as Models 1–3; reproducible with an explicit generator; parameters and buffers unchanged.
- **Diagnostics:** step records contain the `bn` gradient group and `hidden`/`bn`/`act` activations; epoch records contain the `batchnorm` summary; diagnostics never change the training trajectory (bit-identical parameters with and without them).
- **Persistence:**
  - `model_best.pt` / `model_final.pt` round trip: identical parameters **and buffers** (including inference statistics and `num_calibrations`), identical eval-mode logits and seeded generation;
  - `from_checkpoint` rejects: wrong class, vocabulary mismatch, invalid or shape-mismatched config, a `batchnorm` constants mismatch, and `num_calibrations == 0`;
  - contents, write order, history truncation after a crash between history and checkpoint, and the resume-compatibility rejections: as the MLP;
  - **bit-exact resume:** `max_epochs = 4` straight vs interrupted after 2 and resumed with the same configuration: identical parameters, all six buffers, history and best epoch (CPU).
- **Devices:** forward, a few training epochs, recalibration, evaluation and generation on CUDA and MPS when available, matching CPU within 1e-5; device-mismatch errors.

### 13.4 Experiments (outside the implementation task)

**E0. Learning-rate pilot** (APPROVED protocol; required before final runs; *run on 2026-11-17; outcome in §18 and `results/experiments/mlp_study/pilot.json`. Instability was defined as a non-finite loss or gradient, or a dev NLL at epoch 10 above ln 27; spikes and oscillations are reported but exclude nothing. Final seeds 0–9 instead of 0–2.*):
- Grid η ∈ {0.03, 0.1, 0.3, 1.0}; seeds 100, 101, 102 for every candidate; constant LR; `max_epochs = 10` for every candidate.
- Primary criterion: mean full-dev NLL (eval mode, recalibrated) at the end of epoch 10 across the three seeds, among stable candidates.
- Reported for every candidate: mean and std of the epoch-10 dev NLL; best dev NLL per seed and its epoch; any non-finite loss or gradient; significant gradient spikes and loss oscillations; and, for this model, the epoch-10 `ema_vs_inference` summary.
- **Boundary rule** (APPROVED, B4):
  - Start with {0.03, 0.1, 0.3, 1.0}.
  - If the selected η (lowest mean epoch-10 dev NLL among stable candidates) is at a boundary of the grid evaluated so far, extend the grid by one candidate in that direction with ×≈3 spacing (1.0 → 3.0 → 10.0 upward; 0.03 → 0.01 → 0.003 downward).
  - Extended candidates use the same seeds 100–102, the same 10-epoch constant-LR protocol, the same criterion and the same reports.
  - **At most two extension rounds.** If after two rounds the selected η is still at a boundary, no learning rate is selected: the pilot is reported as **inconclusive**, with all results, and a revised experiment is proposed for approval.
  - Every extension round and its reason are recorded with the pilot results.
- If every candidate is unstable, the pilot is inconclusive and a revised grid is proposed for approval.
- Test data untouched; seeds 0–2 reserved.

**Final runs:** seeds 0–2, selected η, constant LR unless a schedule is approved later, `max_epochs` equal to the final MLP value. Report best dev NLL mean ± std, then test NLL (once per seed, `model_best.pt`), next to Model 3 under the same budget and seeds.

Optional:
- **E1:** EMA vs exact statistics over training (the effect of using EMA-only inference on dev NLL, evaluated without changing the authoritative procedure).
- **E2:** saturation, pre-activation and gradient-flow comparison with Model 3.
- **E3:** `W₁` row-norm growth and effective learning rate.
- **E4:** batch-size sensitivity of the train/eval gap.

## 14. Checkpoint and serialization contract

All of §14 of `mlp_spec.md` applies unchanged (checkpoint kinds and API constraints; contents of `checkpoint_last.pt`, `model_best.pt`, `model_final.pt`; write order; resume semantics, compatibility table and procedure; history truncation; resume vs best), with these additions:

- **Buffers:** `model_state` contains all six BatchNorm buffers. They are restored by `restore_model(strict=True)` and `restore_training_state`, so resume continues the EMA exactly and keeps the authoritative inference statistics.
- **Calibrated checkpoints:** every checkpoint is written after the epoch's recalibration, so `model_best.pt` holds exactly the statistics with which its dev NLL was measured. `num_calibrations` equals the number of completed epochs in `checkpoint_last.pt` and `model_final.pt`, and the best epoch + 1 in `model_best.pt`.
- **Compatibility on resume and in `from_checkpoint`:** in addition to the MLP's eight fields, the `batchnorm` constants entry (§9) must match. On resume, `min(sampler.batch_sizes()) ≥ 2` is re-checked.
- **Training trajectory and buffers** [T]: train-mode forwards use batch statistics only, so the optimisation trajectory does not depend on any buffer. The buffers are still restored, because evaluation and selection depend on the inference statistics and the EMA history is part of the diagnostic record.
- `from_checkpoint` rejects `num_calibrations == 0`, since such a model has placeholder inference statistics.

## 15. Generation or inference behaviour

As for the MLP (§15 there), with these differences:
- **Mode:** `generate` records `model.training`, switches to eval mode, samples, and restores the previous mode in `finally`. Normalisation uses `inference_mean` / `inference_var`; the EMA buffers are never read or written.
- **Batch size:** any number of active samples, including 1, is valid because eval mode has no batch dependence. Results for a sample do not depend on how many other samples are drawn alongside it, apart from the shared generator stream.
- **Prerequisite:** generation is meaningful after at least one recalibration; models loaded with `from_checkpoint` always satisfy this.

## 16. Implementation scope and boundaries

Implement in `models/feedforward/batchnorm_mlp.py`:
- `BN_EPS`, `BN_MOMENTUM`; `BatchNormMLPConfig`, `BatchNormMLPTrainingConfig`, `BatchNormMLPSample`, `EpochRecord`, `BatchNormMLPTrainingResult`, `IncompatibleCheckpointError`;
- `CalibratedBatchNorm1d`, `BatchNormMLPLanguageModel`;
- `calibrate_batchnorm`, `evaluate_nll`, `per_unit_saturation`, `train_batchnorm_mlp`, `run_config`;
- the model-local helpers (token validation, streaming merge, history truncation, BatchNorm summary).

`train_batchnorm_mlp` is a model-local reference procedure. It mirrors `train_mlp` (APPROVED: no shared trainer, no import from `mlp.py`); it must not grow into a framework.

**Supporting files:**
1. `configs/models/batchnorm_mlp.yaml`: `model: {name: batchnorm_mlp, params: {context_length: 3, embedding_dim: 10, hidden_dim: 200}}`.
2. `tests/models/feedforward/test_batchnorm_mlp.py`: the §13.1–13.3 tests.

**Not in scope:** shared utility changes; changes to `mlp.py`, other models or specifications; running the pilot or final experiments; notebooks and plots.

## 17. Acceptance criteria

- [x] Exact architecture: shared `nn.Embedding(27, 10)` → concatenation → `nn.Linear(30, 200, bias=False)` → `CalibratedBatchNorm1d(200)` (ε = 1e-5, momentum = 0.1) → `nn.Tanh` → `nn.Linear(200, 27)`; 12,097 parameters; six buffers
- [x] Every parameter and buffer explicitly initialised as in §7; no PyTorch default survives
- [x] Train mode: biased batch statistics and EMA update; eval mode: inference statistics only, batch-independent
- [x] Recalibration at every epoch end: exact full-training-set mean and biased variance by the streaming float64 merge; chunks processed sequentially and released, the full activation matrix never retained; partition-independent; does not touch the EMA, parameters or mode; authoritative for evaluation and generation
- [x] Train mode only for SGD forwards (`num_batches_tracked == steps`); batches of fewer than 2 windows rejected
- [x] Zero-output behaviour verified in both modes
- [x] MLP training semantics preserved: windows, `EpochSampler`, mean reduction over the actual batch size, plain SGD, epoch-based LR, non-finite guard order, three distinct NLLs, selection rule, test once on `model_best.pt`
- [x] Checkpoints with all buffers; write order; bit-exact resume including buffers; history truncation; compatibility checks including the BatchNorm constants
- [x] Diagnostics as §12; mode restored in `finally`; no shared utility extended
- [x] Mathematical tests 1–18 and behaviour checks pass on CPU and available accelerators (MPS on the user's Mac)
- [x] Learning-rate pilot (E0) run under the approved protocol, including the at-most-two-rounds boundary rule, and its decision recorded before any final result (2026-11-17; §18)
- [x] No unresolved material code-review issue; implementation run and reviewed locally by the project author; completion approved by the project author

## 18. Open decisions

### Approved (2026-11-10)

| Item | Decision |
|---|---|
| Architecture | Embedding → concatenation → `Linear(30, 200, bias=False)` → `BatchNorm1d(200)` → tanh → `Linear(200, 27)`; 12,097 parameters |
| Hidden bias | removed; β provides the shift |
| Initialisation | `E ~ N(0,1)`; `W₁` LeCun std `1/√fan_in`; `W₂ = 0`; `b₂ = 0`; `γ = 1`; `β = 0`; `running_mean = 0`; `running_var = 1` |
| Constants | ε = 1e-5, momentum = 0.1, fixed in code |
| Inference statistics | recomputed over the complete training set at every epoch end; biased variance (divisor M); numerically stable streaming algorithm; partition-independent |
| EMA | kept for diagnostic comparison; never authoritative |
| Batch size | batches of fewer than two windows rejected |
| Modes | train mode only for SGD forwards; calibration, evaluation, diagnostics, generation never update running statistics; generation in eval mode with the previous mode restored |
| Infrastructure | MLP training procedure, checkpoints, resume, history, split and evaluation conventions preserved |
| Learning rate | independent pilot: {0.03, 0.1, 0.3, 1.0}, seeds 100–102, 10 epochs, constant LR, mean dev NLL at epoch 10, test untouched |
| Pilot boundary rule (B4) | extend in the boundary direction with ×≈3 spacing, same seeds and protocol; at most two extension rounds; still at a boundary → inconclusive, revised experiment proposed |
| Statistics separation (B1) | `CalibratedBatchNorm1d` with separate `inference_mean` / `inference_var` buffers; the EMA buffers are never overwritten (§3.1) |
| Calibration counter (B2) | `num_calibrations` buffer; `from_checkpoint` rejects uncalibrated checkpoints (§4.4, §14) |
| Recorded constants (B3) | `config["batchnorm"]` holds ε, momentum and the inference-statistics method; checked on resume and load (§9) |
| Schedule fields (B5) | `lr_decay_epoch` / `lr_decay_factor` kept for parity with the MLP; no effect unless set |
| Recalibration memory | chunks processed sequentially; per-unit moments merged; each chunk released before the next; the full activation matrix never retained (§4.7) |
| Budget | same final `max_epochs` as the accepted MLP |
| Code structure | independent `batchnorm_mlp.py`; `mlp.py` unchanged; no shared trainer |

### Decided by the pilot (2026-11-17)

| Item | Decision | Evidence |
|---|---|---|
| `lr` | **0.03** | Mean dev NLL at epoch 10 over seeds 100–102: 0.01 → 2.2062, **0.03 → 2.1659**, 0.1 → 2.1953, 0.3 → 2.2552, 1.0 → 2.3597; all stable. 0.03 was at the lower edge of the first grid, so one extension round (B4) added 0.01 (`pilot.json`) |
| `max_epochs` | **26** | The MLP's decided budget, as this specification requires |
| Schedule | **constant** | No schedule pilot is defined for this model, so the constant rate stands. The MLP's final setting has a ×0.1 drop at epoch 13, so the two models' final results differ in schedule as well as in BatchNorm |

Final runs at these settings, seeds 0–9: best dev NLL 2.1357 ± 0.0007, test NLL 2.1455 ± 0.0009, test evaluated once (`final.json`, `test.json`). Both models under one identical constant protocol are compared in `paired.json`.

### Pending pilot (as approved, kept for the record)

| Item | Rule |
|---|---|
| `lr` | E0 (§13.4) |
| `max_epochs` | inherited from the MLP pilot outcome |

### Pending approval

None. B1–B5 of v0.1 are approved (B4 with the two-round limit). The document was approved and the model implemented and accepted.

### Conflicts noted

1. **The MLP spec's unbiasedness argument (§4.6 there) does not transfer.** With BatchNorm the per-batch objective depends on the batch size, so the final partial batch is a slightly different objective, not just a noisier estimate. The approved policy (keep it, mean over its actual size) is unchanged; the difference is documented (§4.6).
2. **`ActivationMonitor` aggregates over all elements.** Per-unit BatchNorm statistics (EMA vs inference) are computed model-locally (§12), as per-unit saturation was for the MLP.
3. **`utils.statistics.Moments` is per-tensor, not per-feature.** The calibration merge is model-local (§4.7); shared utilities are not extended.
4. **MPS has no float64.** The calibration merge runs on CPU in float64 for every backend (§4.7).

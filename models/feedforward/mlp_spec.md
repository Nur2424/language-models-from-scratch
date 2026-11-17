# Model Specification — MLP (Bengio-Style Feedforward Character Model)

**Status:** v0.3, implemented, tested and accepted. The architecture, initialisation and training procedure are as specified here. The k = 3 learning-rate pilot (E0, §13.4) was run on 2026-11-17 by `training/mlp_study/run_mlp_study.py`, and the training settings that were **PENDING PILOT** are now decided: **η = 0.03, `max_epochs` = 26, one ×0.1 drop after epoch 13** (§18). Results are in `results/experiments/mlp_study/`. The markers **PENDING PILOT** in the body below are kept as the record of what was open when the specification was approved.
**Governing standard:** `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md`
**Template:** `MODEL_SPECIFICATION_TEMPLATE.md`
**Predecessors:** `models/statistical/bigram_spec.md` and `models/feedforward/neural_bigram_spec.md` (both accepted)

Labels used throughout, as in the Neural Bigram specification:
- **[T] theory:** follows from the equations under the stated assumptions.
- **[I] implementation requirement:** must hold for the code; verified by tests (§13).
- **[O] observation:** measured in scratch runs on the group split (seed 42, train-only vocabulary). Preliminary; not a recorded run; never a substitute for tests.
- **[L] legacy:** from the scripts and notebooks of the first version of this project, available in the repository history. Uncontrolled: different initialisation, unseeded batches, sampling with replacement, and a leaky split. **Not comparable** with current results.

Status markers for decisions: **APPROVED**, **PENDING PILOT**, **PENDING APPROVAL**.

The document separates three layers:
- **Architecture** (§3, §4.1–4.4, §7, §8): what the model computes.
- **Reference training procedure** (§4.6, §9, §14): one model-local, deterministic mini-batch SGD procedure.
- **Experiments** (§13.4): the learning-rate pilot and optional investigations, outside the implementation task.

## 1. Model identity

- **Model name:** `mlp`
- **File path:** `models/feedforward/mlp.py`
- **Model family:** feedforward neural language models
- **Exact architecture variant:** a fixed-context feedforward model in the style of Bengio et al. (2003).
  - Three previous tokens are looked up in **one shared** embedding table `E ∈ ℝ^{27×10}`.
  - The three embeddings are concatenated into a 30-dimensional vector.
  - One hidden layer, `tanh(W₁x + b₁)` with 200 units.
  - A linear output layer, `W₂h + b₂`, gives 27 logits.
  - No direct input-to-output connections. No normalisation, dropout or regularisation.
  - 11,897 trainable parameters.
- **Specification version:** 0.3 (draft)
- **Reference implementation(s):** the MLP script and notebook of the first version of this project, and its initialisation experiments, available in the repository history [L].
- **Shared utility dependencies:**
  - data and vocabulary: `utils.vocabulary`, `utils.data.make_context_windows` / `ContextWindows`, `utils.batching.EpochSampler`;
  - losses and metrics: `utils.metrics` (`token_nll`, `NLLAccumulator`, `count_parameters`, `parameter_breakdown`);
  - sampling and RNG: `utils.generation.sample_next_token`, `utils.reproducibility` (`derive_seed`, `make_generator`, `capture_rng_state`);
  - persistence and runs: `utils.checkpoint`, `utils.run_manager.RunManager`, `utils.io` (`read_jsonl`, `atomic_write_text`, `dumps_json`), `utils.config`;
  - diagnostics: `utils.diagnostics` (`ActivationMonitor`, `tanh_saturation`, `global_grad_norm`, `gradient_summaries`, `nonfinite_gradients`, `ParameterSnapshot`, `DiagnosticsSchedule`).

## 2. Role and motivation

### 2.1 Problem addressed
Model `P(x_t | x_{t-3}, x_{t-2}, x_{t-1})` over the 27 output classes, where positions before the start of a name are filled with BOS.

### 2.2 Why this architecture?
Two mechanisms are new relative to the Neural Bigram (`z = W[x_{t-1}]`, an unfactorized `V × V` table):

1. **Longer context with parameter sharing.** A count model or an unfactorized table over 3-character contexts would need one row per context. 27³ = 19,683 contexts are possible and [O] 5,402 occur in training. The MLP instead maps every context through the *same* embedding table and hidden layer. Evidence about one context changes the predictions for others, which makes generalisation to unseen contexts possible: [O] 0.82% of dev windows have contexts never seen in training.
2. **A nonlinear hidden representation.** Without the tanh, the composition of two linear maps would itself be linear, so the hidden layer would add nothing that a single linear map couldn't express. The tanh makes the hidden layer a nonlinear feature extractor.

It is also the project's first model trained with mini-batches. It is the first user of the shared `make_context_windows` and `EpochSampler`, and the first with epoch-boundary resume.

### 2.3 Learning and research questions
- Established [T]: the training objective is non-convex, unlike the Neural Bigram's convex objective. The full-train NLL of *any* model that depends only on the 3-character window is bounded below by the empirical conditional entropy Ĥ(Y|W) (§4.5).
- Questions for experiments (§13.4):
  1. Which learning rate and schedule give a stable, reproducible baseline? (PENDING PILOT)
  2. How large is the gap between this finite MLP's train NLL and Ĥ(Y|W), and between train and dev?
  3. How do saturation, pre-activation scale and update ratios evolve?
  4. What geometry do the embeddings learn?

## 3. Exact architecture definition

- **Architecture definition:** `e_i = E[x_i]` for i = 1, 2, 3; `h = tanh(W₁[e₁; e₂; e₃] + b₁)`; `z = W₂h + b₂`; `P(· | x) = softmax(z)`.
- **Included components:**
  - `nn.Embedding(27, 10)`, shared across the three positions;
  - concatenation in position order (oldest token first);
  - `nn.Linear(30, 200)` followed by `nn.Tanh()`;
  - `nn.Linear(200, 27)`.

  Softmax and log-softmax are applied by the loss and by sampling, not inside `forward`.
- **Excluded components:**
  - direct input-to-output connections;
  - more than one hidden layer;
  - normalisation of any kind (that is Model 4);
  - dropout and weight decay (regularisation);
  - weight tying between `E` and `W₂`;
  - a PAD embedding row;
  - positional embeddings (position is encoded by the concatenation slot).
- **Causal or bidirectional behaviour:** causal. Only the three previous tokens are used.
- **Autoregressive or non-autoregressive behaviour:** autoregressive generation.
- **Deliberate simplifications:** fixed context length; one hidden layer; float32 parameters.

## 4. Mathematical formulation

Symbols:
- `V = vocab.num_classes = 27`, `k = 3` (context length), `d = 10` (embedding dimension), `H = 200` (hidden width).
- Token ids lie in `{0, …, 26}`; id 0 is `.` (BOS in contexts, EOS as a target). PAD (id 27) never enters the model.

### 4.1 Core equations (architecture)

For a context `x = (x₁, x₂, x₃)`, with x₁ the oldest token:

    e_i = E[x_i]                    ∈ ℝ^{10},      E ∈ ℝ^{27×10}
    u   = [e₁; e₂; e₃]              ∈ ℝ^{30}       (concatenation)
    a   = W₁ u + b₁                 ∈ ℝ^{200},     W₁ ∈ ℝ^{200×30}, b₁ ∈ ℝ^{200}
    h   = tanh(a)                   ∈ (−1, 1)^{200}
    z   = W₂ h + b₂                 ∈ ℝ^{27},      W₂ ∈ ℝ^{27×200}, b₂ ∈ ℝ^{27}
    P(x_t = j | x) = softmax(z)_j

**Position-specific blocks** [T]. PyTorch's `nn.Linear` stores its weight as `[out_features, in_features]`. Writing `W₁ = [W₁⁽¹⁾ | W₁⁽²⁾ | W₁⁽³⁾]` with

    W₁⁽¹⁾ = W₁[:, 0:10],   W₁⁽²⁾ = W₁[:, 10:20],   W₁⁽³⁾ = W₁[:, 20:30],

the pre-activation decomposes as

    a = W₁⁽¹⁾ E[x₁] + W₁⁽²⁾ E[x₂] + W₁⁽³⁾ E[x₃] + b₁.

The embedding *table* is shared, but each position has its own projection block. A character contributes differently depending on where it sits in the context, which is what makes the model position-sensitive.

**One-hot form** [T]. `E[x_i] = Eᵀ onehot(x_i)`, so `a = Σ_p W₁⁽ᵖ⁾ Eᵀ onehot(x_p) + b₁`. This is the test oracle for the lookup (§13.2).

### 4.2 Component equations and their roles

**Embedding.** Maps each discrete token to a dense 10-dimensional vector. Shared across positions: a character has one representation, whose use is specialised by the position blocks of `W₁`.

**Concatenation.** Preserves order. Swapping two context tokens changes `u`, and generically changes `z`.

**Hidden layer.** An affine map followed by tanh.
- tanh is odd, bounded in (−1, 1), with derivative `1 − h²`.
- The derivative is 1 at 0 and tends to 0 as |a| grows. Units with |a| ≳ 2.1 (|h| > 0.97) are **saturated** and pass back almost no gradient.

**Output layer.** An affine map to 27 logits. The loss is `ℓ = logsumexp(z) − z_y`.

**Cross-entropy and its gradient** [T]. For a batch `b` with targets `y_n`:

    ∂ℓ_b/∂z_n  = (p_n − e_{y_n}) / |b|                          (mean reduction, §4.6)
    ∂ℓ_b/∂W₂   = Σ_n (∂ℓ_b/∂z_n) h_nᵀ,          ∂ℓ_b/∂b₂ = Σ_n ∂ℓ_b/∂z_n
    δ_n        = (W₂ᵀ ∂ℓ_b/∂z_n) ⊙ (1 − h_n²)                  (= ∂ℓ_b/∂a_n)
    ∂ℓ_b/∂W₁   = Σ_n δ_n u_nᵀ,                  ∂ℓ_b/∂b₁ = Σ_n δ_n
    ∂ℓ_b/∂u_n  = W₁ᵀ δ_n,   block p:  ∂ℓ_b/∂e_{n,p} = W₁⁽ᵖ⁾ᵀ δ_n

**Shared-embedding gradient accumulation** [T]. Each row of `E` sums the contributions of every position where its token occurs:

    ∂ℓ_b/∂E[v] = Σ_n Σ_{p : x_{n,p} = v} W₁⁽ᵖ⁾ᵀ δ_n

- A window containing the same token twice, e.g. `[0, 0, 0]`, adds that row's gradient once *per position*.
- The implementation must accumulate these contributions, not overwrite them. `nn.Embedding`'s backward pass is a scatter-add, which accumulates correctly. This is tested (§13.2).
- [O] On the train windows, row 0 (BOS) occurs in 153,840 position slots, 'a' in 66,892 and 'q' in 601. Rows are updated in proportion to frequency.

### 4.3 Parameterization

| Group | Module | Shape (PyTorch) | Count | Role |
|---|---|---|---|---|
| `E` | `emb.weight` | `[27, 10]` | 270 | token embeddings, shared across positions |
| `W₁` | `hidden.weight` | `[200, 30]` | 6,000 | three position blocks `[:, 10p:10p+10]` |
| `b₁` | `hidden.bias` | `[200]` | 200 | hidden bias |
| `W₂` | `out.weight` | `[27, 200]` | 5,400 | output projection |
| `b₂` | `out.bias` | `[27]` | 27 | output bias |
| **total** | | | **11,897** | |

General formula: `V·d + (k·d)·H + H + H·V + V`.

### 4.4 State and information flow
- No recurrent state: each prediction depends only on its window.
- Information flows embedding → concatenation → hidden → output.
- **Information is shared across contexts** through `E`, `W₁`, `b₁`, `W₂` and `b₂`. This is the defining difference from the Neural Bigram, where each context had a private row.

### 4.5 Assumptions, limits and the conditional-entropy bound

**Dataset measurements** [O]: group split, seed 42, vocabulary from train, context windows with k = 3 and repeated BOS.

| Quantity | Value |
|---|---|
| Training windows `M_train` | 182,666 |
| Development windows `M_dev` | 22,801 |
| Distinct training contexts | 5,402 (of 27³ = 19,683 possible) |
| Empirical conditional entropy Ĥ(Y \| W) | ≈ 1.8681 nats |
| Target unigram entropy H(Y) | ≈ 2.8224 nats |
| Dev windows with a context unseen in training | ≈ 0.82% |
| Reference: ln 27 (uniform prediction) | 3.2958 nats |

**Conditional-entropy bound** [T]. Let `q_w` be the empirical distribution of the next token after context window `w` in the training data, and `N_w` the number of times `w` occurs. For any conditional distribution `p(· | w)` indexed by the observed windows,

    L_train(p) = Σ_w (N_w / M) [ H(q_w) + KL(q_w ‖ p(· | w)) ]  ≥  Σ_w (N_w / M) H(q_w)  =  Ĥ(Y | W).

- The **minimum** is reached exactly when `p(· | w) = q_w` for every observed `w`. The count-based conditional estimator achieves it.
- Any other model that can represent those conditional distributions could also achieve it. For a softmax model, any `q_w` with zero entries (unseen next tokens) is reachable only as a limit, with logits tending to −∞.
- **What the bound is:** a lower bound on the full-train NLL, evaluated with fixed parameters, for any model whose predictions depend only on the 3-character context.
- **What the bound is not:**
  - **not** a guaranteed achievable score for this finite MLP: its 11,897 parameters and smooth function class need not represent all 5,402 conditionals;
  - **not** a bound on development NLL;
  - **not** a bound on individual mini-batch losses.
- [O] 1,141 training contexts occur once and contribute zero entropy, which makes the bound loose. Its main use is as a **violation detector**: a fixed-parameter full-train NLL below Ĥ(Y|W) − ε proves a bug, such as target leakage into the window or an off-by-one window (§13.2).
- [O] On the tiny fixture `["ab", "ac", "ab"]`, Ĥ(Y|W) = (2 ln 1.5 + ln 3)/9 = 0.212171. The approved training procedure approached it from above: gap 7.5e-4 after 500 epochs at lr 0.5, and 2.1e-4 after 2,000.

**Representational limits** [T]. The 27 logit columns, viewed as vectors over the observed contexts, must lie in the span of the 200 hidden features plus a constant. Nothing guarantees this for H = 200. Exact representation is neither expected nor the goal; generalisation is.

**Repeated-BOS encoding** [T]. Names never contain `.`, and windows never cross names. So in every window id 0 appears only as a leading run, and the number of leading zeros encodes how many characters have been seen. The encoding is injective. As an inductive bias, `E[0]` serves both as "start of name" and "empty slot". The position blocks keep the two distinguishable.

**Non-convexity** [T]. Unlike the Neural Bigram, the objective is non-convex. Results depend on the initialisation seed and the data order, so final results need several seeds (§13.4).

### 4.6 Reference training procedure (separate from the architecture)

**Data** (APPROVED):
- `train = make_context_windows(splits.train, vocab, context_length=3)`;
- `dev = make_context_windows(splits.dev, vocab, context_length=3)`;
- `vocab = Vocabulary.from_names(splits.train)`, after `check_vocab_coverage`.

**Batching** (APPROVED):
- `sampler = EpochSampler(train.num_examples, batch_size=32, seed=derive_seed(seed, "data"))`.
- Epoch `e` (0-based) uses `sampler.batches(e)`: sampling without replacement, each training window exactly once per epoch, final partial batch kept.
- [O] 5,709 batches per epoch: 5,708 of size 32 and one of size 10.
- The order of epoch `e` is a function of (seed, e, N) only.

**Per-step objective** (APPROVED): mean reduction over the **actual** batch size, including the final partial batch:

    L_b(θ) = (1 / |b|) Σ_{i ∈ b} ℓ_i(θ),     implemented as token_nll(model(x_b), y_b, reduction="mean")

- The summed loss of the final batch is **not** divided by the nominal 32.
- [T] Under a uniformly random permutation, every batch, including the partial one, is a uniformly random subset, so 𝔼[∇L_b] = ∇L_data with L_data = (1/M) Σ_i ℓ_i. Every step is an unbiased estimate of the same gradient, so the learning rate means the same thing at every step.
- The partial step has higher variance: one step per epoch, weight 10/182,666 ≈ 0.005%.

**Update** (APPROVED): plain SGD, `torch.optim.SGD(model.parameters(), lr=η_e, momentum=0, weight_decay=0)`. The step is `θ ← θ − η_e ∇L_b(θ)`. Gradients are cleared with `optimizer.zero_grad(set_to_none=True)`. No regularisation.

**Order of operations within one step** [I] (APPROVED). Each training step runs exactly in this order:

1. **Forward:** `logits = model(x_b)`.
2. **Loss computation:** `loss = token_nll(logits, y_b, reduction="mean")`.
3. **Finite-loss check:** if `loss` is not finite, raise `FloatingPointError`.
4. **Backward:** `loss.backward()`.
5. **Finite-gradient check:** if any gradient element is NaN or ±inf, raise `FloatingPointError`.
6. **Diagnostics:** only on steps scheduled by `DiagnosticsSchedule` (§12).
7. **Optimiser step:** `optimizer.step()`.

Requirements:
- A non-finite loss or gradient stops training **before** `optimizer.step()`, so parameters are never updated with a non-finite value. Training never silently continues or skips the batch.
- The error message names the 0-based epoch, the cumulative step index and, for gradients, the affected parameter groups (`emb`, `hidden`, `out`) with NaN and inf counts from `nonfinite_gradients(model)`.
- The step-5 check runs on every step. To keep its cost to one host synchronisation, it uses a single fused reduction (for example `torch.isfinite` of the sum of per-parameter gradient norms), and calls `nonfinite_gradients` only after a failure, to build the message.
- With `run` given, the exception propagates through `RunManager.track()`, which records the run as `failed`. The last `checkpoint_last.pt` (the previous epoch boundary) is left untouched.
- For an activation-diagnostics step, the `ActivationMonitor` must already be attached during step 1. Its summaries are only *recorded* at step 6.

**Learning rate and schedule** (**PENDING PILOT**, §13.4):
- The learning rate depends only on the epoch, so it is stateless and recomputed on resume:

      η_e = lr                       if lr_decay_epoch is None or e < lr_decay_epoch
      η_e = lr · lr_decay_factor     otherwise

- It is set on every parameter group at the start of epoch `e`. No `torch.optim.lr_scheduler` is used.
- The candidates to compare after the pilot are constant LR (`lr_decay_epoch = None`) and one ×0.1 step at 50% of the epoch budget.

**Epoch budget** (**PENDING PILOT**): train for exactly `max_epochs` epochs; no patience-based early stopping. A provisional planning value is **30 epochs**. That is not an approved value, and not a claim that 30 epochs is sufficient or converged.

**Reported quantities.** Three are kept strictly separate:
1. **Running training NLL:** example-weighted over the epoch, accumulated by `NLLAccumulator.update(logits.detach(), y_b)` while the parameters change during the epoch.
2. **Full-training NLL:** `evaluate_nll(model, train)` with fixed end-of-epoch parameters, in eval mode, under `no_grad`.
3. **Full-development NLL:** the same, on dev, with the same fixed parameters.

Epoch metrics are always example-weighted through `NLLAccumulator`. Batch means are never averaged.

**Selection and test** (APPROVED):
- The best checkpoint is the epoch with the lowest full-dev NLL. Exact ties go to the earliest epoch (strict `<` comparison).
- The test set is evaluated **once**, after every configuration and selection decision, using `model_best.pt`.

**Determinism.**
- Initialisation uses `make_generator(derive_seed(seed, "init"))` on CPU; the parameters are created and initialised on CPU, then moved.
- Data order is described above. Nothing uses the global torch RNG: there is no dropout, and generation uses its own generator.
- On a fixed CPU software stack, runs are bit-identical. CUDA requires `configure_determinism(True)`, because `nn.Embedding`'s backward uses atomic additions. MPS gives no guarantee.

## 5. Tensor and interface contract

| Item | Shape | Dtype | Device | Meaning |
|---|---|---|---|---|
| `contexts` (forward input) | `[B, 3]` | int64 | model device | token ids in `[0, 27)`, oldest first; PAD rejected |
| `emb(contexts)` | `[B, 3, 10]` | float32 | model device | embeddings |
| `u` | `[B, 30]` | float32 | model device | `emb(contexts).reshape(B, 30)` (position-major) |
| `a` (hidden pre-activation) | `[B, 200]` | float32 | model device | `hidden(u)` |
| `h` | `[B, 200]` | float32 | model device | `act(a)` |
| logits (forward output) | `[B, 27]` | float32 | model device | `out(h)`: **logits**, not probabilities |
| targets (training) | `[B]` | int64 | model device | next-token ids; never `IGNORE_INDEX` |
| `ContextWindows.contexts` | `[M, 3]` | int64 | CPU | from `make_context_windows` |
| sampler batch | `[b]` | int64 | CPU | indices into the windows; moved by the caller |

- **Dimensions:** `B` is the batch size (≥ 0; `B = 0` returns `[0, 27]`); k = 3; V = 27.
- **Input token representation:** fixed-length windows from `make_context_windows`, left-filled with repeated BOS.
- **Output representation:** logits.
- **Variable-length sequences:** not applicable. Every example is a fixed-length window.
- **Padding convention:** PAD never reaches the model. `forward` raises `ValueError` for PAD (dedicated message) and for any id outside `[0, 27)`.
- **Mask convention:** none; every window is a real example.
- **Special tokens:** BOS (0) fills missing left context; EOS (0) is a target.
- **Device behaviour:** `forward` raises `ValueError` if `contexts` isn't on the model's device. It never moves data.

### Public API (to implement in `models/feedforward/mlp.py`)

```python
@dataclass(frozen=True)
class MLPConfig:                     # model.params in configs/models/mlp.yaml
    context_length: int = 3          # APPROVED baseline; validated int ≥ 1
    embedding_dim: int = 10          # APPROVED baseline; validated int ≥ 1
    hidden_dim: int = 200            # APPROVED baseline; validated int ≥ 1

@dataclass(frozen=True)
class MLPTrainingConfig:             # model-local; NOT part of ExperimentConfig (§9)
    lr: float                        # PENDING PILOT; required, finite > 0
    max_epochs: int                  # PENDING PILOT; required, int ≥ 1 (provisional 30)
    batch_size: int = 32             # APPROVED
    lr_decay_epoch: int | None = None    # PENDING PILOT; None = constant LR; else 1 ≤ value < max_epochs
    lr_decay_factor: float = 0.1     # used only when lr_decay_epoch is set; 0 < value ≤ 1
    eval_batch_size: int = 4096      # evaluation chunking only; must not change any decision (§10)

@dataclass(frozen=True)
class MLPSample:                     # model-specific (as NeuralBigramSample)
    text: str                        # characters between BOS and EOS; never contains "."
    finished: bool                   # True iff EOS was drawn

@dataclass(frozen=True)
class EpochRecord:                   # one row of history.jsonl
    epoch: int                       # 0-based index of the completed epoch
    lr: float
    steps: int                       # cumulative optimiser steps after this epoch
    running_train_nll: float         # (1) parameters changing during the epoch
    train_nll: float                 # (2) fixed end-of-epoch parameters
    dev_nll: float                   # (3) fixed end-of-epoch parameters
    is_best: bool
    seconds: float
    activations: dict                # epoch-end dev activation summary (§12)

@dataclass(frozen=True)
class MLPTrainingResult:
    best_epoch: int
    best_dev_nll: float
    epochs_completed: int
    history: list[EpochRecord]       # history of this call, after any resume truncation

class MLPLanguageModel(nn.Module):
    def __init__(self, vocab: Vocabulary, config: MLPConfig = MLPConfig(), *, generator: torch.Generator) -> None
        # submodules emb (nn.Embedding), hidden (nn.Linear), act (nn.Tanh), out (nn.Linear);
        # created on CPU and explicitly initialised from `generator` (§7)
    num_classes: int; context_length: int
    def forward(self, contexts: LongTensor[B, k]) -> FloatTensor[B, V]
    @torch.no_grad()
    def generate(self, num_samples: int, *, max_length: int, generator: torch.Generator | None = None,
                 temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None) -> list[MLPSample]
    @classmethod
    def from_checkpoint(cls, path, *, expected_vocab: Vocabulary | None = None,
                        map_location="cpu") -> "MLPLanguageModel"     # weights only, from any checkpoint kind

def evaluate_nll(model: MLPLanguageModel, windows: ContextWindows, *, batch_size: int) -> NLLAccumulator
    # eval mode, no_grad, every example (including the final partial chunk);
    # restores the model's original train/eval mode in a `finally` block, even if an exception occurs

def per_unit_saturation(model: MLPLanguageModel, windows: ContextWindows, *, threshold: float = 0.97,
                        batch_size: int) -> torch.Tensor     # [H] fraction of windows with |h| > threshold, per unit
    # eval mode, no_grad; restores the original train/eval mode in a `finally` block, even if an exception occurs

def train_mlp(model: MLPLanguageModel, train: ContextWindows, dev: ContextWindows, *,
              training: MLPTrainingConfig, seed: int, run: RunManager | None = None,
              resume: bool = False, diagnostics: DiagnosticsSchedule | None = None,
              config: Mapping | None = None) -> MLPTrainingResult
    # §4.6 procedure. `seed` is the base seed (data order = derive_seed(seed, "data")).
    # With `run`: writes model_best.pt / checkpoint_last.pt / model_final.pt and history.jsonl (§14).
    # `config` = resolved experiment config dict stored with checkpoints and run metadata.
```

Loss computation is not a model method. `train_mlp` uses `utils.metrics.token_nll`.

## 6. PyTorch implementation policy

### 6.1 Permitted operations and modules
- `nn.Module`, `nn.Embedding`, `nn.Linear` and `nn.Tanh`.
- Explicit initialisation through `torch.nn.init` functions or in-place `normal_` / `zero_` with an explicit generator, under `no_grad`.
- Tensor reshaping and indexing.
- `torch.softmax` and `torch.log_softmax`, and `F.cross_entropy` through the shared `token_nll`.
- `torch.optim.SGD` with `momentum=0` and `weight_decay=0`.
- `torch.autograd.gradcheck`, `torch.func.functional_call` and `F.one_hot` in tests.

`nn.Tanh` must be a registered submodule (`self.act`), so that `ActivationMonitor` can hook it.

### 6.2 Prohibited operations and modules
- **Any PyTorch default initialisation surviving construction.** `nn.Linear`'s default is `kaiming_uniform(a=√5)` with a uniform bias; `nn.Embedding`'s default draws from the global RNG. Every parameter must be overwritten explicitly (§7).
- `nn.Sequential` hiding the forward order is discouraged. The forward pass must read as §8.
- Normalisation layers, dropout and weight decay.
- Momentum, Adam/AdamW and `torch.optim.lr_scheduler` in the baseline.
- `padding_idx` and a PAD row.
- Packaged language-model modules; `nn.RNN`, `nn.LSTM`, `nn.GRU` and `nn.Transformer*`.
- Hand-written `exp(z)/Σexp(z)`.
- Probability clamping.
- Global RNG reseeding in `__init__` or `forward`.
- Loss computation or optimiser steps inside `forward`.

### 6.3 Manual implementation requirements
Written explicitly in `mlp.py`:
- `MLPConfig` and `MLPTrainingConfig` validation;
- explicit initialisation of all five parameter groups;
- context-id validation;
- the forward order (embedding → reshape to `[B, k·d]` → hidden → tanh → out);
- `train_mlp`: epoch loop, schedule, evaluation, selection, checkpoints, history and resume truncation;
- `evaluate_nll` and `per_unit_saturation`;
- the generation loop;
- `from_checkpoint` with vocabulary and configuration validation.

### 6.4 Shared utility integration
- **Windows:** `make_context_windows`. The MLP does not build windows itself.
- **Batches:** `EpochSampler`.
- **Loss and NLL:** `token_nll(reduction="mean")` for steps; `NLLAccumulator` for all epoch metrics.
- **Generation:** `sample_next_token`.
- **RNG:** `derive_seed` with the streams `"init"`, `"data"` and `"generation"`, and `make_generator`.
- **Checkpoints:** `save_checkpoint`, `load_checkpoint`, `restore_model` and `restore_training_state`.
- **Runs:** `RunManager` (`append_history`, `checkpoint_path`, `track`); `utils.io.read_jsonl` / `atomic_write_text` / `dumps_json` for history truncation (§14).
- **Diagnostics:** the shared tools in §12.
- **Config:** `load_config(..., model_config_cls=MLPConfig)`.

Model-local by approved convention, with no shared changes:
- **Duplicated on purpose:** the sample dataclass, token-id validation, checkpoint vocabulary validation, `evaluate_nll` and `per_unit_saturation`.
- **Not created:** no shared trainer, no `training/` package, and no `ExperimentConfig` changes.

## 7. Initialization and parameter sharing

All APPROVED. Every draw uses `make_generator(derive_seed(seed, "init"))` on CPU before any `.to(device)`.

| Parameter | Shape | Initialization | Rationale |
|---|---|---|---|
| `E` (`emb.weight`) | `[27, 10]` | `N(0, 1)` | unit-scale inputs; distinct characters are distinguishable from the start |
| `W₁` (`hidden.weight`) | `[H, k·d]`; baseline `[200, 30]` | LeCun normal `N(0, 1/fan_in)`, std `1/√fan_in`, with `fan_in = context_length · embedding_dim`; baseline `fan_in = 30`, std `1/√30 ≈ 0.1826` | keeps hidden pre-activations at unit scale (below) |
| `b₁` (`hidden.bias`) | `[200]` | exactly 0 | zero-mean pre-activations; no PyTorch default survives |
| `W₂` (`out.weight`) | `[27, 200]` | exactly 0 | initial prediction exactly uniform (below) |
| `b₂` (`out.bias`) | `[27]` | exactly 0 | same |

**General rule for `W₁`** [I]. `MLPConfig` lets `context_length` and `embedding_dim` vary, so the scale must be computed from the configuration, never hard-coded:

    fan_in = context_length · embedding_dim          (= in_features of `hidden`)
    W₁ ~ N(0, 1/fan_in),   std = 1 / √fan_in

The approved baseline (k = 3, d = 10) has `fan_in = 30`, so std = `1/√30 ≈ 0.1826`. For example, k = 8, d = 10 gives `fan_in = 80` and std ≈ 0.1118. `hidden_dim` does not enter the formula, because it is the fan-out. All other initialisation decisions are unchanged and do not depend on the configuration: `E ~ N(0, 1)`, `b₁ = 0`, `W₂ = 0`, `b₂ = 0`.

**Hidden pre-activation scale** [T]. Given an input `u`, `a_j = Σ_i W₁[j,i] u_i + b₁[j]` with `b₁ = 0`. With independent zero-mean weights of variance `1/fan_in`, `Var(a_j | u) = ‖u‖²/fan_in`. With embedding entries of variance 1, `𝔼‖u‖² = k·d = fan_in`, so `Var(a_j) ≈ 1` and std(a) ≈ 1 for every configuration. The baseline is the case `fan_in = 30`.

The saturation probability is `P(|a| > atanh(0.97) ≈ 2.09) ≈ 3.6%` for a standard normal.
- [O] On the training windows: pre-activation std **1.052** and saturated fraction **4.97%**; no unit is saturated on ≥ 99% of windows.
- Both are slightly above theory because repeated BOS makes the entries of `u` correlated and `‖u‖²` varies, so the distribution is a scale mixture with heavier tails.

Suitability: tanh is near-linear for |a| ≲ 1 and saturates beyond ≈ 2. Unit scale puts most units in the responsive, still nonlinear range.
- [L] The legacy unscaled `N(0, 1)` weights gave std ≈ 5.5 and about 70% saturated activations.
- The "(5/3)/√fan_in" tanh gain targets variance preservation in *deep* stacks. It brings no benefit for one hidden layer and would give about 21% saturation.

**Zero output initialisation** [T]: intended behaviour, not a defect.
- With `W₂ = 0` and `b₂ = 0`, every logit is 0, so `softmax(z)` is uniform (1/27).
- The initial cross-entropy is exactly `ln 27 = 3.295837` for every example, up to floating-point tolerance ([O] 3.295837 on the full train set).

**First backward pass** [T]. Let `g_n = (p_n − e_{y_n})/|b|`.

| Group | First-step gradient | Value |
|---|---|---|
| `b₂` | `Σ_n g_n` (= uniform − batch target frequencies) | nonzero ([O] norm 0.23) |
| `W₂` | `Σ_n g_n h_nᵀ` (`h` depends on random `E` and `W₁`) | nonzero ([O] norm 1.7) |
| `b₁`, `W₁`, `E` | contain the factor `W₂ᵀ g_n = 0` | **exactly zero** |

- After the first update, `W₂ = −η ∇W₂`, which is generally nonzero, so from the second step every group receives gradient.
- [O] At η = 0.1, ‖W₂‖ was 0.17 after step 1 and 1.9 after 100 steps. Gradient norms at step 1: E 1.8e-2, W₁ 3.4e-2, b₁ 8.6e-3.
- **There is no symmetry problem.** Hidden units are already distinct through random `W₁` and `E`. The columns of `W₂` receive different gradients through different `h`, and every hidden unit then receives a different backpropagated signal.
- Random output initialisation must **not** be introduced to avoid this behaviour.

**Parameter sharing.** `E` is shared across the three positions (one table, three lookups). No tying between `E` and `W₂`. No other sharing.

Historical initialisation alternatives (unscaled N(0,1), the "fixes" with ×0.01 and ×0.2 scaling, the tanh gain) belong in experiments (§13.4), not in `MLPConfig`.

## 8. Forward-pass specification

1. **Validate** [I] (always):
   - `contexts` is a `torch.Tensor`, otherwise `TypeError`;
   - dtype int64, otherwise `TypeError`;
   - `ndim == 2` and `shape[1] == context_length`, otherwise `ValueError`;
   - on the model's device, otherwise `ValueError`;
   - every id in `[0, 27)`, otherwise `ValueError`, with the dedicated message "padding must not reach the model" for PAD (27).

   The range check costs one host sync, as for the Neural Bigram.

   **Empty batches** [I]. An input of shape `[0, context_length]` is valid and returns logits of shape `[0, 27]`:
   - The type, dtype, dimensionality, context-width and device checks are metadata-only and **always** apply, including when `B = 0`. A `[0, 2]` input, a float input or an input on the wrong device is still rejected.
   - The range check must be **safe for empty tensors.** It must not call `min()`/`max()` (or `amin`/`amax`) on an empty tensor, which raise an error. Either skip the range check when `contexts.numel() == 0`, or use reductions defined on empty inputs, e.g. `(contexts == pad_id).any()` and `((contexts < 0) | (contexts >= 27)).any()`, which return `False`.
   - Steps 2–6 then run normally on the empty tensor.
2. **Embed:** `e = self.emb(contexts)`: `[B, 3]` → `[B, 3, 10]`.
3. **Concatenate:** `u = e.reshape(B, 30)`: position-major, so `u[:, 10p:10p+10] = E[x_{p+1}]`.
4. **Hidden pre-activation:** `a = self.hidden(u)`: `[B, 200]`.
5. **Nonlinearity:** `h = self.act(a)`: `[B, 200]`.
6. **Output:** `z = self.out(h)`: `[B, 27]`. Return the logits; no softmax, loss or device transfer.

## 9. Configuration

Configuration is split by layer. **No unsupported top-level config section is added.** The shared `ExperimentConfig` (`utils/config.py`) has no `training` section, and adding one is not approved.

**Architecture: `model.params` → `MLPConfig`** (`configs/models/mlp.yaml`)

| Field | Type | Default | Validation | Status |
|---|---|---|---|---|
| `context_length` | int | 3 | int ≥ 1 (bool rejected) | APPROVED |
| `embedding_dim` | int | 10 | int ≥ 1 | APPROVED |
| `hidden_dim` | int | 200 | int ≥ 1 | APPROVED |

The baseline uses exactly the defaults. Other values are for experiments only. Initialisation is fixed in code, not configurable.

**Training: `MLPTrainingConfig`** (model-local; constructed by the experiment script)

| Field | Type | Default | Validation | Status |
|---|---|---|---|---|
| `lr` | float | required | finite, > 0 | **PENDING PILOT** |
| `max_epochs` | int | required | ≥ 1 | **PENDING PILOT** (provisional planning value 30) |
| `batch_size` | int | 32 | ≥ 1 | APPROVED (32) |
| `lr_decay_epoch` | int \| None | None | None or 1 ≤ value < `max_epochs` | **PENDING PILOT** (constant or 50% of budget) |
| `lr_decay_factor` | float | 0.1 | 0 < value ≤ 1 | APPROVED as the candidate factor; use PENDING PILOT |
| `eval_batch_size` | int | 4096 | ≥ 1 | implementation detail |

- Optimiser, momentum (0), weight decay (0) and loss reduction (mean) are **fixed in code**, not fields, because they are APPROVED invariants of the baseline.
- `MLPTrainingConfig` is serialised with `dataclasses.asdict` into `config.json` and every checkpoint (§14). Because it is not part of `ExperimentConfig`, it is not loaded with `load_config`; scripts construct it in code.

**Reproducibility** (shared `reproducibility` section): `seed` is the base seed, with streams `derive_seed(seed, "init" | "data" | "generation")`. `deterministic`: `configure_determinism` for CUDA runs. Pilot seeds 100–102; final reporting seeds 0–2 (§13.4).

**Data** (shared `data` section, fixed for every model): `split_seed: 42`, `duplicate_policy: group`, `split_fractions: [0.8, 0.1, 0.1]`.

**Evaluation and checkpoints** (shared `checkpoint` section):
- `save_best: true` and `save_final: true` are honoured.
- `checkpoint_last.pt` is written at **every epoch boundary** whenever `run` is given. The shared field `resume_every_n_steps` is step-based, is not used by this model, and should stay `null` (§10, conflict note).

**Diagnostics** (shared `diagnostics` section → `DiagnosticsSchedule`): step-level collection cadence (§12).

**Generation** (shared `generation` section): `temperature`, `top_k`, `top_p` and `num_samples` are passed through. `max_new_tokens` is used as `max_length`, the maximum number of ordinary characters, with the same semantics as Models 1–2.

## 10. Numerical stability and edge cases

| Case | Behaviour | Test |
|---|---|---|
| Softmax / log-softmax | only stable primitives through `token_nll` and `sample_next_token` | extreme logits give a finite loss |
| Initial loss | exactly `ln 27` (zero logits) | §13.2 |
| tanh saturation | expected in part (≈5% at init [O]); monitored, not prevented | diagnostics (§12) |
| Zero gradient to `E`, `W₁`, `b₁` at step 0 | intended (§7) | §13.2 |
| PAD, negative or ≥ 27 ids | `ValueError` | dedicated tests |
| Wrong dtype, shape or device | `TypeError` / `ValueError` | dedicated tests |
| Empty batch `[0, 3]` | `forward` returns `[0, 27]`; the range check is empty-safe (no `min`/`max` on an empty tensor); metadata checks still apply; training never forms empty batches (§8) | dedicated test |
| Final partial batch (10 windows) | mean over 10 (not over 32) | §13.2 |
| Evaluation chunking | `NLLAccumulator` sums are float64; float32 per-chunk sums mean the result can vary with `eval_batch_size` at the ~1e-7 level. Selection ties are decided on these values, so `eval_batch_size` must be fixed per study | §13.2 |
| Non-finite loss or gradient | `train_mlp` raises `FloatingPointError` **before** `optimizer.step()`, naming the epoch, the step and (for gradients) the affected parameter groups; never silently continues (order in §4.6) | dedicated tests (injected NaN loss; injected NaN gradient) |
| Large learning rate | may diverge or oscillate; no stability guarantee is claimed; detected by the non-finite check and studied in the pilot | experiment |
| Precision | float32 parameters; float64 for `gradcheck` (CPU only) | — |
| Backend | `nn.Embedding`, `nn.Linear`, tanh, SGD and `multinomial` all exist on CPU, CUDA and MPS; bit-exact determinism only on CPU (§4.6) | device tests |

## 11. Complexity and resource profile

- **Parameter count:** 11,897 (§4.3).
- **Time complexity per example:** an embedding gather of 30 values; hidden 30·200 = 6,000 multiply-adds; output 200·27 = 5,400. About 11.4k multiply-adds forward, and about 3× that for forward plus backward.
- **Epoch:** about 182,666 × 34k ≈ 6.2 GFLOP-equivalent multiply-adds. [O] About 4.1 s per epoch on the scratch container's CPU (one measurement; hardware-dependent; not a benchmark).
- **Memory complexity:** a batch of 32 is negligible. Evaluation with `eval_batch_size = 4096` holds `4096 × 200` activations (~3.3 MB).
- **Sequence-length scaling:** none within the model; window construction is O(M·k).
- **Main bottlenecks:** Python overhead per step (5,709 steps per epoch); the per-call range-check sync.
- **Device considerations:** windows live on CPU. `train_mlp` moves the full window tensors to the model device once, then indexes them with sampler indices moved to that device.

## 12. Diagnostics

**Step level.** Only on steps where `diagnostics.should_collect(step, kind)` is true. Records go to `run.diagnostics_dir / "steps.jsonl"`.
- `"gradients"`: `global_grad_norm(model)`; `gradient_summaries(model, group_by="module")`, giving the groups `emb`, `hidden` and `out`; `nonfinite_gradients(model)`.
- `"parameters"`: `ParameterSnapshot.capture(model)` before the step and `.update_stats(model)` after it. These are measured update norms and relative updates per parameter.
- `"activations"`: an `ActivationMonitor` on the current batch (see below).

**Epoch end.** Always collected when `run` is given; cheap. With fixed parameters, on **dev**, during or alongside `evaluate_nll`:
- `ActivationMonitor(model, module_names=["hidden", "act"], saturation={"act": tanh_saturation(0.97)})` gives the pre-activation (`hidden` output) mean and std, the tanh output statistics, the saturation fraction, and NaN/inf counts.
- `per_unit_saturation(model, dev, threshold=0.97)` gives a `[200]` vector. Its summary is stored in `EpochRecord.activations`: min, median, max, and the number of units with saturation ≥ 0.99. `ActivationMonitor` aggregates over elements, so this per-unit statistic is model-local; shared diagnostics are not extended for this model.

The summaries are scalars or `[H]` vectors. No graphs or large tensors are retained, and monitors are removed (context manager).

**Mode restoration** [I]. `evaluate_nll` and `per_unit_saturation` switch the model to eval mode and run under `torch.no_grad()`. Both record `was_training = model.training` on entry and restore it in a `finally` block, so the original train/eval mode is restored **even if an exception occurs** (e.g. a validation error or a device error part-way through). The model has no mode-dependent layers today, but the contract is kept so that callers in the training loop never depend on it.

**Overhead limits** [I]. The diagnostic scope is unchanged; no shared utility is extended.
- **No retained graphs:** every recorded quantity is a detached Python scalar or a detached `[H]` tensor. Diagnostics run on `.detach()`ed tensors or under `no_grad`. No output, activation or loss tensor that requires grad is kept beyond its step.
- **No routine full-parameter copies to CPU:** `ParameterSnapshot.capture` clones parameters on their own device (`p.detach().clone()`). It is called **only** on steps scheduled for `"parameters"`, never on every step. Only summary scalars are transferred to the host. The model is ≈ 11.9k parameters, so even a scheduled snapshot is ≈ 48 KB.
- **Unscheduled steps** pay only for the always-on finite-loss and finite-gradient checks of §4.6 (one host sync each).
- **Epoch-end activation and per-unit statistics** use the dev forward pass in chunks of `eval_batch_size` and accumulate running sums. Full `[M_dev, H]` activation matrices are never stored.

**Experiment-specific** (§13.4, not part of the standard run): embedding geometry (PCA or 2D); saturation dynamics over steps; initialisation comparisons; per-unit activation visualisations and histograms; the gap to Ĥ(Y|W).

## 13. Test plan

File: `tests/models/feedforward/test_mlp.py`.

Fixtures:
- **Small set:** `["emma", "olivia", "ava", "isabella", "sophia", "mia", "liam", "noah", "anna", "anna", "eva", "bo"]`.
- **Tiny set:** `["ab", "ac", "ab"]`, with Ĥ(Y|W) = 0.212171.
- **Real data:** used only in tests marked to skip if `data/names.txt` is absent.

### 13.1 General tests
- [x] Construction; `MLPConfig` and `MLPTrainingConfig` validation (types, ranges, bool rejection); YAML integration (`load_config(default_config_paths("mlp"), model_config_cls=MLPConfig)`; unknown keys rejected)
- [x] Input contract: `[B, 3]` → `[B, 27]` float32; wrong ndim or width; non-int64; device mismatch (meta device on CPU); out-of-range and PAD rejected with the dedicated message
- [x] Empty batch: `[0, 3]` int64 → `[0, 27]` without error (the range check is empty-safe); `[0, 2]`, `[0]`, a float `[0, 3]` and a `[0, 3]` on the wrong device are still rejected
- [x] Parameter count 11,897 with the breakdown `emb` 270, `hidden` 6,200, `out` 5,427 (via `parameter_breakdown`) and the general formula for a non-default config; all trainable; exactly five parameter tensors
- [x] No surviving defaults: `b₁`, `W₂`, `b₂` exactly zero; `E` and `W₁` are deterministic functions of the init generator (same seed gives identical, different seed gives different), and neither depends on global RNG state
- [x] Train and eval mode give identical outputs (no dropout or normalisation)
- [x] Checkpoints (§14); determinism; CPU; CUDA and MPS where available (skipped otherwise)
- [x] `mlp.py` imports nothing from `models.statistical` or `models.feedforward.neural_bigram`; `nn.Tanh` is a registered submodule

### 13.2 Mathematical verification
1. **Embedding sharing:** a single embedding table. For a context with the same token at two positions, both slices of `u` are identical. Editing `E[v]` changes `u` at every position where `v` occurs and nowhere else.
2. **Position blocks:** perturbing `W₁[:, 10p:10p+10]` changes the output only through position p, i.e. `a` equals `Σ_p W₁⁽ᵖ⁾ E[x_p] + b₁` with blocks taken from the `[out, in]` layout.
3. **Position sensitivity:** for generic parameters (random `W₂` set under `no_grad`), permuting a non-constant context changes the logits.
4. **Forward oracle:** for random parameters, the model equals the explicit `onehot · E` → concatenation → matmul with `W₁ᵀ` → `+b₁` → tanh → matmul with `W₂ᵀ` → `+b₂` (atol 1e-6).
5. **Logit contract and cross-entropy:** `token_nll(model(x), y, "mean")` equals a hand-written `mean(logsumexp(z) − z_y)`. On a 2-example fixture with hand-set parameters, it equals a hand-computed value.
6. **Shared-embedding gradient accumulation:** autograd's `∂/∂E` equals `Σ_n Σ_{p: x_{n,p}=v} W₁⁽ᵖ⁾ᵀ δ_n` computed by hand. This includes the window `[0, 0, 0]`, three contributions to row 0 (atol 1e-7).
7. **Finite differences:** `torch.autograd.gradcheck` on a float64 copy via `functional_call` over all five parameter groups.
8. **Zero output initialisation:**
   - initial full-batch loss = `ln 27` (atol 1e-6);
   - first-step gradients of `E`, `W₁` and `b₁` exactly zero;
   - `∇W₂ = Σ_n g_n h_nᵀ` and `∇b₂ = Σ_n g_n` (atol 1e-7);
   - after one SGD step, every group has a nonzero gradient.
9. **Initialisation statistics.** Robust tolerances derived from sampling error, not from particular draws:
   - **Embedding std:** `|std(E) − 1| < 0.2` (standard error ≈ 0.043, n = 270).
   - **Embedding mean:** `|mean(E)| < 0.3` (standard error ≈ 0.061).
   - **W₁ std:** `|std(W₁)·√30 − 1| < 0.05` (standard error ≈ 0.9%, n = 6,000).
   - **W₁ mean:** `|mean(W₁)| < 0.012` (≈ 5 standard errors).
   - **Pre-activation std** on the small set's windows: within `1 ± 0.15`.
   - **Saturation fraction:** *reported*, not asserted tightly.
   - **Non-default configuration:** for `MLPConfig(context_length=8, embedding_dim=10, hidden_dim=200)` (fan_in = 80; `W₁` has 16,000 entries), `|std(W₁)·√80 − 1| < 0.05` and `|mean(W₁)| < 0.01`. The test also asserts that the scale is **not** the baseline `1/√30`: `std(W₁)·√30` lies outside `1 ± 0.05`. `E`, `b₁`, `W₂` and `b₂` keep their configuration-independent initialisation (`b₁`, `W₂`, `b₂` exactly zero).
10. **Mean reduction:** for a batch of size 10 (the final partial size) and one of size 32, the step gradient equals the gradient of `(1/|b|) Σ ℓ_i`, **not** `(1/32) Σ ℓ_i`.
11. **Example-weighted epoch NLL:** the running NLL from `train_mlp` equals `Σ_b |b|·L_b / M` recomputed from per-batch values. On a hand example with unequal batch sizes it differs from the mean of batch means.
12. **Evaluation coverage:** `evaluate_nll` with a `batch_size` that doesn't divide M gives `num_tokens == M` and equals one full pass (rel 1e-6).
13. **Conditional-entropy bound:** the full-train NLL with fixed parameters is ≥ Ĥ(Y|W) − 1e-6 for random and for trained parameters (tiny and small sets).

### 13.3 Learning and behaviour checks
- **Tiny-fixture overfitting:** `["ab", "ac", "ab"]`, lr 0.5, 500 epochs, batch 32 (one batch of 9 per epoch).
  - Full-train NLL − Ĥ(Y|W) lies in `[−1e-6, 1e-2]` ([O] 7.5e-4 at 500 epochs, so the threshold gives a ≥ 13× margin).
  - This is a *debugging* check, not a quality claim.
- **Determinism:** two `train_mlp` runs with the same seed give bit-identical parameters and histories (CPU). Different seeds give different parameters.
- **Schedule:** with `lr_decay_epoch = 2`, the optimiser LR is `lr` in epochs 0–1 and `lr·factor` from epoch 2; it is recomputed identically after resume.
- **Selection:** the best epoch is the lowest dev NLL. On an injected exact tie, the earliest epoch is kept.
- **Non-finite guard:**
  - an injected NaN loss raises `FloatingPointError` naming the epoch and step;
  - an injected NaN gradient (e.g. a backward hook on `hidden.weight`) raises `FloatingPointError` naming the epoch, the step and the `hidden` group;
  - in both cases the parameters are bit-identical to their values before the failing step, which proves `optimizer.step()` was not called.
- **Mode restoration:** `evaluate_nll` and `per_unit_saturation` leave `model.training` as it was, for both starting modes, and also when they raise part-way (e.g. windows on the wrong device).
- **Generation:**
  - the contexts used while generating a name equal `make_context_windows([name]).contexts` for that name, including the repeated-BOS start;
  - EOS stops a sample;
  - `max_length` semantics as in Models 1–2 (`finished=False` on truncation; at most `max_length + 1` draws);
  - `text` never contains `.`;
  - reproducible with an explicit generator;
  - parameters and training mode are unchanged.
- **Persistence:**
  - `model_best.pt` / `model_final.pt` round trip: identical parameters, logits and seeded generation;
  - `from_checkpoint` rejects a vocabulary mismatch, a wrong model class and an incompatible `MLPConfig` (shape mismatch);
  - **epoch-boundary resume:** both runs use the **same** `MLPTrainingConfig` with `max_epochs = 4`.
    - Run A trains 4 epochs straight.
    - Run B is interrupted after two completed epochs, by an exception injected at the start of epoch 2. It then resumes from `checkpoint_last.pt` with the unchanged four-epoch configuration and completes epochs 2–3.
    - Result: bit-exact parameters, identical history with no duplicate rows, identical best epoch (CPU).
  - **Resume compatibility:** resuming with any changed field listed in §14 raises `IncompatibleCheckpointError`, one parametrised case per field.
  - **History truncation:** simulate a crash between `append_history` and the `checkpoint_last.pt` save of epoch `e` (e.g. an exception injected into that save). The history then contains a row for epoch `e` that the checkpoint does not record. On resume, that row is removed, epoch `e` is re-run, and the final history equals that of an uninterrupted run, with exactly one row per epoch.
  - **Write order:** within each epoch, the history row is appended before `checkpoint_last.pt` is saved (§14).
- **Devices:** forward, a few training steps, evaluation and generation on CUDA and MPS when available, matching CPU within 1e-5 for a fixed number of steps; device-mismatch errors.

Correctness evidence comes from §13.2. A good dev NLL is **not** evidence of correctness.

### 13.4 Experiments (outside the implementation task)

**E0. Learning-rate pilot.** Required before final runs. *Run on 2026-11-17; outcome in §18 and `results/experiments/mlp_study/pilot.json`, `budget.json`, `schedule.json`.*

*Rules added before the pilot ran (2026-11-17), where this specification left them open:* unstable means a non-finite loss or gradient, or a dev NLL at the end of the pilot above ln 27; spikes and oscillations are reported for every candidate but exclude nothing. The boundary rule of the BatchNorm MLP (at most two ×≈3 extension rounds) is applied to the MLP as well. Budget: at the selected η, seeds 100–102, 30 epochs; E = the earliest epoch whose mean dev NLL is within max(0.001, SE) of the minimum; if that is epoch 30, extend to 60 epochs. Final seeds 0–9 instead of 0–2.

*Setup:*
- Grid η ∈ {0.03, 0.1, 0.3, 1.0}.
- Pilot seeds 100, 101, 102 for every candidate.
- **Constant LR**, `max_epochs = 10` for every candidate. All candidates use the same training duration; candidates are never compared at different durations.

*Primary selection criterion:* the **mean full-development NLL at the end of epoch 10** (fixed end-of-epoch parameters, §4.6 quantity 3), averaged over the three pilot seeds. The selected η is the stable candidate with the lowest value.

*Reported for every candidate,* whether or not it is selected:
- the mean and standard deviation (across the three seeds) of the epoch-10 development NLL;
- the best development NLL for each seed;
- the epoch at which each seed's best development NLL occurred;
- any non-finite loss or gradient, with seed, epoch and step;
- significant gradient spikes (global gradient norm from the step diagnostics) and loss oscillations (running train NLL and per-epoch dev NLL).

The best-dev and best-epoch figures are descriptive only. They do not replace the epoch-10 criterion.

*Stability and inconclusive outcomes:*
- A candidate is unstable if any seed produces a non-finite loss or gradient, or shows significant spikes or oscillations recorded above.
- Unstable candidates are not selected.
- If **every** candidate is unstable, no learning rate is selected automatically. The pilot is recorded as **inconclusive**, and a revised grid is proposed for approval before anything else is run.

*Schedule follow-up:*
- Uses the **selected** η, the **same** pilot seeds 100–102, the same budget for both arms, and the **same** evaluation criterion (mean end-of-budget full-dev NLL across seeds, with the same additional reports).
- Compares constant LR against one ×0.1 step at 50% of that budget.
- The final `lr`, `lr_decay_epoch` and `max_epochs` are chosen from E0, and the rationale is recorded.

*Data discipline:*
- Train and dev only. **The test split is never loaded for evaluation** during the pilot or the schedule follow-up.
- Seeds 0–2 are reserved for final reporting and are not used in the pilot.

**Final runs:** seeds 0, 1, 2 with the selected configuration. Report the mean ± std of best dev NLL, then test NLL (evaluated once per seed on `model_best.pt`).

Optional:
- **E1:** gap to Ĥ(Y|W) and train/dev gap over training.
- **E2:** historical initialisation comparisons (unscaled N(0,1); W₂×0.01; W₁×0.2; tanh gain), set in experiment code.
- **E3:** saturation and pre-activation dynamics.
- **E4:** embedding geometry.
- **E5:** `embedding_dim` against `hidden_dim` variations, with parameter counts reported.

## 14. Checkpoint and serialization contract

All checkpoints use `utils.checkpoint.save_checkpoint` and are loaded with `weights_only=True`.

**API constraints** (from `utils/checkpoint.py`):
- `kind="resume"` **requires** `optimizer`, `step` and `rng_state`, and accepts `epoch`, `best_metric`, `best_metric_name`, `config`, `vocab` and `extra`.
- `kind="best"` **requires** `best_metric`.
- `kind="best"` and `kind="final"` **refuse** optimizer, scheduler and scaler state.

**`checkpoint_last.pt`** (`kind="resume"`): written at every epoch boundary when `run` is given.
- `model`: model state (`emb.weight`, `hidden.weight`/`bias`, `out.weight`/`bias`).
- `optimizer`: SGD state. It is empty for momentum-free SGD, but **required by the API** and stored.
- `step`: cumulative optimiser steps.
- `epoch`: the **number of completed epochs**, which is also the 0-based index of the next epoch.
- `best_metric`: best dev NLL so far; `best_metric_name = "dev_nll"`.
- `rng_state`: `capture_rng_state()`. It is required by the API; the procedure consumes no global RNG.
- `config`: `{"experiment": <resolved experiment config dict>, "model": asdict(MLPConfig), "training": asdict(MLPTrainingConfig)}`.
- `vocab`: `vocab.to_dict()`.
- `extra`: `{"best_epoch": <0-based epoch>, "sampler": sampler.state_dict(), "base_seed": seed}`.

**`model_best.pt`** (`kind="best"`):
- written whenever dev NLL strictly improves;
- contains weights, `best_metric`, `best_metric_name`, `epoch` (completed epochs at that point), `step`, `config`, `vocab` and `extra={"best_epoch": e}`;
- **used for test evaluation and generation.**

**`model_final.pt`** (`kind="final"`): written after the last configured epoch, regardless of whether it is the best. Weights, `epoch`, `step`, `config` and `vocab`.

**Write order at the end of epoch `e`:**
1. evaluate train and dev;
2. if dev improved, save `model_best.pt`;
3. `run.append_history(EpochRecord)`;
4. save `checkpoint_last.pt` with `epoch = e + 1`;
5. after the last epoch, save `model_final.pt`.

**Why the history row comes before `checkpoint_last.pt`** [I]. The checkpoint is the commit point for an epoch.
- A crash between steps 3 and 4 leaves the history with **one extra row** (epoch `e`), while the checkpoint still records `e` completed epochs. On resume, truncation keeps only rows with `epoch < e`, which removes the extra row. Epoch `e` is then re-run and its row rewritten.
- With the opposite order, a crash between the two writes would leave the checkpoint at `e + 1` with **no** row for epoch `e`. Truncation can remove rows but cannot recreate one, so that row would be lost permanently.
- In every crash case, the history therefore holds at least every completed epoch, and truncation makes it exactly those.

On CPU an interrupted epoch re-runs bit-identically, so re-saving `model_best.pt` and re-appending the row give the same content.

**Resume semantics** [I]. The checkpoint stores the **original resolved training configuration** (`config["training"]`, including `max_epochs`) and, separately, the **number of completed epochs** (`epoch`). These are different quantities:
- `max_epochs` is the run's fixed budget. It is preserved across resumes.
- `completed_epochs` is only the point to continue from. It is **never** used as, or written back as, a replacement `max_epochs`.

Example: a run configured with `max_epochs = 4` that stops after two epochs has a checkpoint with `epoch = 2` and `config["training"]["max_epochs"] = 4`. It is resumed with the **original four-epoch configuration**, runs epochs 2 and 3, and then writes `model_final.pt`. Resuming with `max_epochs = 2` (or any other value) is an incompatible change and is rejected.

**Resume compatibility.** The current call is compared with the checkpoint. Any difference raises `IncompatibleCheckpointError` naming the field and both values:

| Field | Compared against |
|---|---|
| Model architecture | `config["model"]` vs `asdict(model_config)` (`context_length`, `embedding_dim`, `hidden_dim`) |
| Learning rate | `config["training"]["lr"]` |
| Maximum epoch budget | `config["training"]["max_epochs"]` |
| Batch size | `config["training"]["batch_size"]` and the sampler state's batch size |
| LR decay epoch and factor | `config["training"]["lr_decay_epoch"]`, `["lr_decay_factor"]` |
| Sampler seed | the sampler state's seed vs `derive_seed(seed, "data")` of the current call (and `extra["base_seed"]` vs `seed`) |
| Vocabulary | `ckpt["vocab"]` vs `vocab.to_dict()` |
| Dataset size | the sampler state's `num_examples` vs `train.num_examples` |

`eval_batch_size` affects only evaluation chunking and is not part of this check. It is still recorded, and must be held fixed within a study (§10).

**Resume procedure** (`train_mlp(..., resume=True)`):
1. Load `checkpoint_last.pt` (`expected_kind="resume"`).
2. Run the compatibility check above before touching the model or optimiser.
3. `restore_training_state(ckpt, model=model, optimizer=optimizer)` restores model, optimizer and RNG and returns `TrainingProgress`.
4. `EpochSampler.from_state_dict(ckpt["extra"]["sampler"], num_examples=M, batch_size=B)` additionally rejects a changed N or B. No mutable sampler state exists: epoch order = f(seed, epoch, N).
5. **Truncate history:** `RunManager` has no truncation method, so `train_mlp` performs a model-local operation on `run.history_path`. It reads the rows with `utils.io.read_jsonl`, keeps rows whose `epoch < completed_epochs`, and rewrites the file atomically with `utils.io.atomic_write_text` (one `dumps_json(row, indent=None)` per line). No shared utility is modified.
6. Continue from epoch index `completed_epochs` up to the **original** `max_epochs`, with η recomputed for every remaining epoch from the original epoch-based schedule (§4.6). The best metric and best epoch are restored from the checkpoint.

**Resume versus best.** "Resume" restores the full *training state* to continue optimising. "Best" loads *weights only*, via `from_checkpoint`, for test evaluation or generation, with no optimiser and no history.

**Validation in `from_checkpoint`:**
- requires `vocab` metadata;
- `model_class == "MLPLanguageModel"`;
- `MLPConfig` is rebuilt from `config["model"]`;
- `expected_vocab` must match if given;
- the load is strict (`restore_model(strict=True)`), and shape mismatches raise `IncompatibleCheckpointError`.

## 15. Generation or inference behaviour

- **Inference entry point:** `generate(num_samples, *, max_length, generator=None, temperature=1.0, top_k=None, top_p=None)`, under `@torch.no_grad()`.
- **Prediction procedure:** every sample starts with context `[0, 0, 0]`. At each step: `logits = model(contexts)` → `sample_next_token(logits, …, generator)`. If the token is EOS, the sample is finished. Otherwise the token is appended and the context shifts: `context ← context[1:] + [token]`.
  - This is exactly the rule used by `make_context_windows`, and a test checks the equality.
  - Samples are drawn in parallel; one draw per active sample per step.
- **Start/end tokens:** BOS fills the initial context; EOS stops a sample; neither appears in `text`.
- **Sampling controls:** the defaults sample the model distribution exactly; temperature, top-k and top-p use the shared semantics; `temperature = 0` is greedy.
- **RNG:** an explicit generator, by convention `make_generator(derive_seed(seed, "generation"))`.
- **Evaluation mode:** the model has no mode-dependent behaviour; `generate` does not change `self.training`.
- **Maximum length:** identical to Models 1–2. At most `max_length` ordinary characters. A sample that reaches it gets one more draw: EOS gives `finished=True`, anything else returns it truncated with `finished=False`. At most `max_length + 1` draws per sample.
- **Degenerate distributions:** none for finite parameters (strictly positive softmax).

## 16. Implementation scope and boundaries

Implement in `models/feedforward/mlp.py`:
- `MLPConfig`, `MLPTrainingConfig`, `MLPSample`, `EpochRecord`, `MLPTrainingResult`, `IncompatibleCheckpointError`;
- `MLPLanguageModel`;
- `evaluate_nll`, `per_unit_saturation`, `train_mlp`;
- the token-id validation and history-truncation helpers.

`train_mlp` is a model-local reference procedure (approved: no shared trainer at this stage). It must not grow into a general framework: no optimiser choice, no plugins, no callbacks.

**Necessary supporting files:**
1. `configs/models/mlp.yaml`: `model: {name: mlp, params: {context_length: 3, embedding_dim: 10, hidden_dim: 200}}`.
2. `tests/models/feedforward/test_mlp.py`: the §13.1–13.3 tests.

**Not in scope:**
- shared utility changes;
- a `training/` package or an `ExperimentConfig` `training` section;
- changes to the statistical bigram, Neural Bigram, or any specification;
- `batchnorm_mlp.py`;
- running the pilot or the final experiments;
- notebooks and plots.

## 17. Acceptance criteria

- [x] Exact architecture: shared `nn.Embedding(27, 10)` → concatenation (position-major) → `nn.Linear(30, 200)` → `nn.Tanh` → `nn.Linear(200, 27)`; 11,897 parameters; no direct connections
- [x] Every parameter explicitly initialised as in §7 (`E` ~ N(0,1), `W₁` ~ N(0, 1/fan_in) with `fan_in = context_length · embedding_dim`, i.e. N(0,1/30) for the baseline, `b₁ = W₂ = b₂ = 0`) from the `"init"` generator on CPU; no PyTorch default survives
- [x] Zero-output behaviour verified: initial loss `ln 27`; first-step zero gradients for `E`, `W₁`, `b₁`; all groups active after one step
- [x] Input validation, including PAD rejection and empty-safe range checking for `B = 0`; logits contract
- [x] Non-finite loss or gradient raises `FloatingPointError` before `optimizer.step()`, in the §4.6 order
- [x] `evaluate_nll` and `per_unit_saturation` restore the original train/eval mode, including on exceptions
- [x] Windows from `make_context_windows` and batches from `EpochSampler` (without replacement, partial batch kept)
- [x] Mean reduction over the actual batch size; epoch metrics example-weighted with `NLLAccumulator`; the three reported NLLs kept distinct
- [x] Plain SGD with no momentum or weight decay; epoch-based stateless learning rate
- [x] Best checkpoint by lowest dev NLL with earliest-epoch tie-break; test evaluated once on `model_best.pt`
- [x] Checkpoint contents and write order as in §14; epoch-boundary resume bit-exact on CPU under the original `max_epochs`; incompatible resume changes rejected; history truncated on resume
- [x] Standard diagnostics as in §12; no shared diagnostics extended
- [x] Mathematical tests 1–13 pass; learning and behaviour checks pass
- [x] CPU and available accelerators (MPS on the user's Mac) checked
- [x] Learning-rate pilot (E0) run and its decision recorded before any final result is reported (2026-11-17; §18)
- [x] No unresolved material code-review issue
- [x] Implementation run and reviewed locally by the project author
- [x] Completion approved by the project author

## 18. Open decisions

### Approved

| Item | Decision |
|---|---|
| Architecture | k = 3, d = 10, H = 200, shared embedding, concatenation, tanh hidden layer, linear output, both biases, no direct connections, 11,897 parameters |
| Context encoding | repeated BOS; `make_context_windows`; PAD never enters the model |
| PyTorch modules | `nn.Embedding`, `nn.Linear`, `nn.Tanh` (registered submodule), with explicit initialisation |
| Initialisation | `E` ~ N(0,1); `W₁` LeCun normal std 1/√fan_in with `fan_in = context_length · embedding_dim` (baseline 1/√30); `b₁ = 0`; `W₂ = 0`; `b₂ = 0` |
| Empty batches | `[0, k]` → `[0, 27]`; metadata checks always apply; range check empty-safe |
| Non-finite guard (P3) | forward → loss → finite-loss check → backward → finite-gradient check → diagnostics → `optimizer.step()`; raise `FloatingPointError` naming epoch, step and affected groups; never continue |
| Optimiser | plain SGD; no momentum; no weight decay or regularisation |
| Batching | batch 32; `EpochSampler`; without replacement; partial batch kept; per-epoch derived order |
| Loss reduction | mean over the actual batch size, including the partial batch |
| Epoch reporting | `NLLAccumulator`, example-weighted; running, train and dev NLL kept distinct |
| Stopping and selection | fixed epoch budget; no patience; best = lowest dev NLL, earliest on ties; test once on `model_best.pt` |
| Training-loop ownership | model-local `train_mlp`; no shared trainer; no `ExperimentConfig` change |
| Configuration | validated `MLPConfig(context_length=3, embedding_dim=10, hidden_dim=200)` |
| Resume | epoch-boundary `checkpoint_last.pt`; continue from `completed_epochs` under the original `max_epochs` and schedule; reject changes to architecture, lr, `max_epochs`, batch size, decay epoch/factor, sampler seed, vocabulary or dataset size; history truncated to completed epochs |
| Diagnostics | shared tools; per-unit saturation model-local; evaluation helpers restore train/eval mode in `finally`; no retained graphs; parameter snapshots only on scheduled steps |
| Generation length (P1) | `generation.max_new_tokens` is used as `max_length`, consistent with Models 1 and 2 |
| Step-diagnostics destination (P2) | `run.diagnostics_dir / "steps.jsonl"`; `history.jsonl` keeps one row per epoch |

### Decided by the pilot (2026-11-17)

| Item | Decision | Evidence |
|---|---|---|
| `lr` | **0.03** | Mean dev NLL at epoch 10 over seeds 100–102: 0.01 → 2.2024, **0.03 → 2.1634**, 0.1 → 2.2014, 0.3 → 2.2979, 1.0 → 2.5038; all stable. 0.03 was at the lower edge of the first grid, so one extension round added 0.01 (`pilot.json`) |
| `max_epochs` | **26** | At η = 0.03 over 30 epochs the mean dev NLL is lowest at epoch 28 (2.1374); the earliest epoch within max(0.001, SE) = 0.0048 of it is 26 (2.1392). The tolerance came from the SE of three seeds; with the 0.001 floor alone the rule gives 28 (`budget.json`) |
| `lr_decay_epoch` | **13** (×0.1) | Mean dev NLL at epoch 26 over seeds 100–102: constant 2.1392 ± 0.0048, drop at epoch 13 **2.1221 ± 0.0015**, lower in all three seeds (`schedule.json`) |

Final runs at these settings, seeds 0–9: best dev NLL 2.1232 ± 0.0009, test NLL 2.1291 ± 0.0007, test evaluated once (`final.json`, `test.json`).

### Pending pilot (as approved, kept for the record)

| Item | Candidates | Rule |
|---|---|---|
| `lr` | {0.03, 0.1, 0.3, 1.0} | E0 (§13.4): lowest mean full-dev NLL at the end of epoch 10 across seeds 100–102, constant LR, among stable candidates; if all are unstable, the pilot is inconclusive and a revised grid goes for approval |
| Schedule (`lr_decay_epoch`) | constant (None) or one ×0.1 step at 50% of the budget | E0 follow-up: selected η, same seeds 100–102, same criterion, dev only |
| `max_epochs` | provisional planning value 30 (not approved, not a convergence claim) | set from pilot curves; the same budget-setting rule applies to later models |

### Pending approval

None. P1–P3 from v0.1 are APPROVED (see the Approved table). The document was approved and the model implemented and accepted.

### Conflicts noted (template, UMIS and existing utilities)

1. **No shared `training` config section.** Training settings live in the model-local `MLPTrainingConfig` and are serialised into `config.json` and checkpoints. They can't be loaded through `load_config`. Consistent with the approved scope; a known limitation.
2. **`CheckpointConfig.resume_every_n_steps` is step-based.** This model's resume cadence is per epoch, so the field is unused and stays `null`.
3. **`RunManager` has no history truncation.** It is handled model-locally with existing `utils.io` helpers (§14); no shared change.
4. **The template's "dropout and train/eval" items** don't apply: the model has no mode-dependent behaviour, and that is tested.
5. **The template's §13.1 "padding, lengths and masks"** is adapted: fixed windows, no masks, and PAD rejection is tested.

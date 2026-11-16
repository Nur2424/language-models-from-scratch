# Model Specification — Neural Bigram (Bias-Free Softmax-Regression Bigram)

**Status:** v0.6, implemented, tested and accepted. This specification describes the implementation in `models/feedforward/neural_bigram.py`. All design decisions are recorded in §18 (D1–D9, Q1–Q4, implementation details I1–I4). v0.5 corrected the gradient-norm bound; v0.6 corrected the stability wording, called the gradient-norm value a supremum, and renamed "full-rank" to "unfactorized, full-table parameterisation".
**Governing standard:** `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md`
**Template:** `MODEL_SPECIFICATION_TEMPLATE.md`
**Predecessor:** `models/statistical/bigram_spec.md` (accepted)

Every substantive statement carries one of three labels:
- **[T] theory:** follows from the equations under the stated assumptions.
- **[I] implementation requirement:** must hold for the code and is verified by tests (§13).
- **[O] observation:** measured once in scratch runs on the group split (seed 42, train-only fit, float64). Preliminary; not a recorded run, and never a substitute for the tests.

The document separates three layers:
- **Architecture** (§3, §4.1–4.4, §8): what the model computes.
- **Reference training procedure** (§4.6, §16): one minimal, deterministic, model-specific fitting procedure.
- **Experiments** (§13.4): optional investigations, not part of the implementation.

## 1. Model identity

- **Model name:** `neural_bigram`
- **File path:** `models/feedforward/neural_bigram.py`
- **Model family:** feedforward neural language models (degenerate case: no hidden layer)
- **Exact architecture variant:** bias-free softmax regression on the previous token, with an unfactorized, full-table parameterisation (one free logit per `(context, next token)` cell; "full table" refers to the parameterisation, not to the algebraic rank of `W`). One trainable matrix `W ∈ ℝ^{V×V}`, initialised to zero. The logits are the row of `W` selected by the previous token: `z = W[x_{t-1}]`, and `P(x_t = j | x_{t-1} = i) = softmax(W_i)_j`.
  - This is mathematically identical to a bias-free linear layer applied to the one-hot encoding of `x_{t-1}`.
  - No bias, no embedding dimension, no hidden layer, no random initialisation.
- **Specification version:** 0.6 (implemented and accepted)
- **Reference implementation(s):** an earlier neural-bigram script from the first version of this project, available in the repository history (historical, not authoritative). Differences from this spec: N(0,1) initialisation; an L2 term `0.01·mean(W²)` folded into the reported loss; the claim that `lr = 50` "overshoots". That claim is not reproduced from zero initialisation (§4.5).
- **Shared utility dependencies:**
  - `utils.vocabulary` and `utils.data.make_next_token_sequences`;
  - `utils.metrics` (`token_nll`, `NLLAccumulator`, `count_parameters`);
  - `utils.generation.sample_next_token`;
  - `utils.reproducibility` (`make_generator`, `derive_seed`, `capture_rng_state`);
  - `utils.checkpoint` and `utils.config`;
  - `utils.diagnostics`.

## 2. Role and motivation

### 2.1 Problem addressed
Model `P(x_t | x_{t-1})` over the 27 output classes, exactly as the statistical bigram does. The difference: each conditional distribution is represented by trainable logits and fitted by gradient descent on cross-entropy, instead of being computed from counts.

### 2.2 Why this architecture?
It is the smallest change that turns the count model into a neural one. Kept the same as the statistical bigram:
- the model class (one categorical distribution per context);
- the conditioning information;
- the data.

What changes:
- the **parameterisation** (unconstrained logits instead of probabilities);
- the **fitting procedure** (iterative optimisation instead of a closed form).

That isolates the mechanisms this step should make concrete: softmax, cross-entropy, gradients, and when optimisation recovers the count solution. It is also the project's first `nn.Module`, so it is the first integration test of autograd, an optimiser, resumable checkpoints and diagnostics.

What it deliberately does **not** add is any sharing of parameters between contexts. Each logit `W_ij` belongs to exactly one context row. The next model (MLP) introduces a shared low-dimensional embedding, which factorises the logit table. Keeping this model unfactorized (a full table) and bias-free keeps the two steps separate (§4.4).

### 2.3 Learning and research questions
Established [T]:
- The training objective is convex, and its infimum equals the statistical MLE training NLL.
- On this dataset the infimum is not attained by any finite `W` (§4.5).
- With an L2 penalty the minimiser exists, is unique and is finite.
- Unseen targets in observed contexts are actively suppressed by the gradient. Rows of contexts never observed receive no gradient (§4.5).

Questions for the experiments in §13.4: how closely and how fast gradient descent approaches the MLE; how learning rate, initialisation and weight decay change the solution; how much probability unseen transitions keep over training, and what that does to dev NLL.

## 3. Exact architecture definition

- **Architecture definition:** `z = W[x_{t-1}] ∈ ℝ^V`; `P(· | x_{t-1}) = softmax(z)`.
- **Included components:**
  - one trainable logit table `W` (`[V, V]`), initialised to zero;
  - row lookup by previous-token id;
  - token-id validation.

  Softmax and log-softmax are applied by the loss and by sampling, not inside `forward`.
- **Excluded components:**
  - bias vector (D1);
  - random initialisation (D2);
  - embedding of dimension `d < V`;
  - hidden layers and nonlinearities;
  - normalisation and dropout;
  - context longer than one token;
  - weight tying;
  - regularisation as part of the model (regularisation is a training-procedure option, §4.6).
- **Causal or bidirectional behaviour:** causal. `forward` sees only the previous token.
- **Autoregressive or non-autoregressive behaviour:** autoregressive generation.
- **Deliberate simplifications:** unfactorized, full-table parameterisation with no parameter sharing between contexts; float32 parameters.

## 4. Mathematical formulation

Symbols:
- `V = vocab.num_classes = 27`. Token ids `i, j ∈ {0, …, V-1}`; id 0 is `.` (BOS as a context, EOS as a target).
- Training transitions `(i_m, j_m)`, `m = 1..M`, with `M = Σ_names (len + 1)`. [O] `M = 182,666` on the train split.
- Counts `C_ij`, context totals `N_i = Σ_j C_ij`, empirical rows `q_ij = C_ij / N_i` for `N_i > 0` (the statistical MLE).

### 4.1 Core equations (architecture)

    z_m   = W[i_m]                                    ∈ ℝ^V       (row lookup)
    p_m   = softmax(z_m),   p_mj = exp(z_mj) / Σ_k exp(z_mk)
    ℓ_m   = -log p_m,j_m  = logsumexp(z_m) - z_m,j_m              (cross-entropy of one transition)
    L(W)  = (1/M) Σ_m ℓ_m                                         (token-weighted mean, nats/token)

One-hot equivalence [T]: `W[i] = e_iᵀ W`, where `e_i` is the one-hot row, so `z = onehot(x_{t-1}) · W`. That is a linear layer without bias. The lookup is the implementation; the matrix product is a test oracle (§13.2).

### 4.2 Component equations and their roles

**Softmax: logits → probabilities.** `exp` maps ℝ to (0, ∞), and normalising makes each row sum to 1.
- Every finite `z` gives a strictly positive distribution.
- Softmax is invariant to shifting a row by a constant, `softmax(z + c·1) = softmax(z)`, so a row has `V` parameters but `V − 1` degrees of freedom.
- Up to that constant, logits are log-probabilities: `log p_j = z_j − logsumexp(z)`.

**Cross-entropy: prediction error.** `ℓ = −log p_target` is 0 when the observed token gets probability 1 and grows without bound as that probability goes to 0. Its average over the data is the token-weighted NLL that every model in the project reports.

**Count form, the sufficient statistic** [T]. Grouping the `M` terms by context gives an exact identity, checked numerically [O, error 0.0]:

    L(W) = -(1/M) Σ_i Σ_j C_ij log softmax(W_i)_j
         = Σ_i (N_i / M) · [ H(q_i) + KL(q_i ‖ softmax(W_i)) ]

- The loss depends on the data only through `C`, as in the statistical model.
- Without a bias, the objective is a sum of independent per-row problems.
- Rows with `N_i = 0` contribute nothing.
- Training uses the pair form `L(W)` (D9). The count form is a test oracle only.

**Gradient** [T]. Using `∂ℓ/∂z = p − e_target`:

    ∂L/∂W_ij = (1/M) (N_i p_ij − C_ij) = (N_i / M) (p_ij − q_ij)          for N_i > 0
    ∂L/∂W_ij = 0                                                         for N_i = 0

Properties:
- Each gradient row sums to 0, because `Σ_j p_ij = Σ_j q_ij = 1`.
- In a batch, only rows of contexts present in the batch receive gradient.
- Gradient descent raises `W_ij` where the model under-predicts (`p_ij < q_ij`) and lowers it where it over-predicts.

### 4.3 Parameterization

| Group | Shape | Count | Init | Role |
|---|---|---|---|---|
| `transition_logits` (`W`) | `[V, V]` | `V² = 729` | zeros | row `i` = next-token logits after context `i` |

There is no bias parameter. Effective degrees of freedom are `V(V − 1) = 702`, because each row is defined only up to an additive constant. Zero initialisation together with gradient descent fixes that constant (§4.5, gauge).

### 4.4 State and information flow
- The only input is the previous-token id. There is no hidden state.
- **No parameter is shared between contexts.** Training evidence for context `i` changes only row `i`, and never the predictions for any context `k ≠ i`.
- Within a row, softmax couples the targets: raising one logit lowers every other probability in that row. This coupling is what suppresses unseen targets in observed contexts (§4.5).
- Transfer *between* contexts (for example, similar letters behaving similarly) first becomes possible with the MLP's shared embedding.

### 4.5 Relationship to the statistical MLE, and assumptions

**Convexity** [T]. `L` is convex in `W`: log-sum-exp is convex and the other terms are linear. Every stationary point is a global minimiser; there are no spurious local minima. `L` is not strictly convex on all of `ℝ^{V×V}`, because of the row-shift directions.

**Infimum** [T]. By Gibbs' inequality `KL ≥ 0`, with equality iff `softmax(W_i) = q_i`. Hence

    inf_W L(W) = Σ_i (N_i/M) H(q_i) = statistical bigram training NLL (α = 0)        [O] 2.4536 nats

and the excess loss `L(W) − L_MLE = Σ_i (N_i/M) KL(q_i ‖ p_i) ≥ 0` measures exactly how far the model is from the count solution.

**Attainment** [T]. `softmax(W_i) = q_i` for a finite `W_i` requires `q_ij > 0` for every `j`.
- [O] 112 of the 729 cells have `C_ij = 0`.
- `C_00 = 0` holds for every dataset of non-empty names: BOS is never followed by EOS.

The infimum is therefore never attained, and the MLE is reproduced only as a limit. Equivalence holds in this precise sense, for full-batch gradient descent satisfying the step-size condition below:
- (a) `L(W_t) → L_MLE`;
- (b) `p_ij(t) → q_ij` on observed cells;
- (c) `p_ij(t) → 0` on unseen cells of observed contexts, which requires `W_ij → −∞` relative to the row.

No equivalence is claimed for finite training time, mini-batch noise, or regularised objectives.

**Unseen transitions: two different situations** [T].

1. *Observed context `i` (`N_i > 0`), unobserved target `j` (`C_ij = 0`).* The gradient is

       ∂L/∂W_ij = (N_i / M) p_ij > 0,

   strictly positive, because `p_ij > 0` for finite `W`. Gradient descent on `L_data` (`λ = 0`) therefore **actively decreases** `W_ij` at every step. The model receives no positive evidence for `j` after `i`, but the softmax normalisation turns the evidence for the other targets in the same row into evidence *against* `j`. That is learned suppression, not an absence of learning. With `λ = 0` the probability tends to 0 but never reaches it. With `λ > 0` the decay term `−ηλW_ij` opposes the suppression, and the logit converges to the finite value given by stationarity (§4.6).
   - Heuristic rate [T]: once the row's log-normaliser has nearly settled, `W_ij ← W_ij − η(N_i/M)p_ij` gives `p_ij(t) ≈ 1/(η (N_i/M) t)`.
   - [O] For BOS → EOS with `η = 25`: `t·P(.|.) = 0.274` at `t = 10⁴` and `10⁵`, against a prediction of `1/(25·0.1404) = 0.285`.

2. *Unobserved context `i` (`N_i = 0`).* Row `i` receives **zero gradient**, so it never changes: zero initialisation leaves it at `W_i = 0`, the uniform distribution `1/V`. With weight decay the update is `W_i ← (1 − ηλ)W_i`, which also keeps a zero row at zero ([O] verified exactly).
   - When the vocabulary is built from the training split, every context occurs (`N_i > 0` for all `i`), so this case arises only with a vocabulary larger than the training character set.
   - The statistical bigram leaves the same row *undefined* under α = 0.

**Consequence for held-out data** [T]. Dev transitions unseen in train ([O] 9 events) are exactly case 1. Their probability shrinks like 1/t under unregularised training, so the dev NLL eventually grows like `(#events/M_dev)·log t`, and early stopping acts as implicit smoothing.
- [O] Dev NLL at 100 / 1,000 / 5,000 / 20,000 steps: 2.4897 / 2.4583 / 2.4569 / 2.4571.
- The statistical bigram's dev NLL is 2.4571 at α = 1 and lies between 2.4570 and 2.4577 for α in [0.1, 2]. **The numerical match is not an equivalence.**

**Smoothness and the step-size condition** [T]. The Hessian of `L` is block-diagonal with blocks `(N_i/M)(diag(p_i) − p_i p_iᵀ)`.
- The largest eigenvalue of `diag(p) − ppᵀ` is at most ½: for a unit vector `v`, `vᵀ(diag p − ppᵀ)v = Var_p(v) ≤ (max v − min v)²/4 ≤ ½`. [O] 0.492 was the largest over 2000 random `p`.
- So `L` is `ℓ`-smooth with `ℓ ≤ max_i N_i/(2M)` ([O] 0.1488/2 = 0.0744, context `'a'`), and `J = L + (λ/2)‖W‖²` is `(ℓ + λ)`-smooth.
- Under that condition, full-batch gradient descent with `0 < η < 2/(ℓ + λ)` decreases **the optimised objective `J`** monotonically and converges to its infimum. For `λ = 0`, `J = L_data` and the condition is [O] `η < 26.9`.
- For `λ > 0` the guarantee is about `J` only; the data loss `L_data` alone may increase (§4.6).

This is a **sufficient** condition, not a stability boundary. Outside it there is no convergence guarantee and no stability guarantee.
- [O] From zero initialisation with `λ = 0`, `η = 50` and `η = 100` also converged; `η = 200` was slow; `η ∈ {400, 800, 1600}` oscillated without reducing the loss over 2,000 steps.
- [O] In all of these tested runs the parameters and losses stayed finite, with no NaN. That is an **empirical observation about these runs only, not a theoretical guarantee**. The gradient bound below limits each step's size, but it does not make arbitrary learning rates or training durations safe:
  - with `λ = 0`, `‖W_T − W_0‖_F ≤ T·η·√2`, which is finite for finite `T` in exact arithmetic but grows without limit in `η·T`; float32 logits can then overflow, and overflowing logits can produce `inf`/NaN in log-sum-exp;
  - with `λ > 0`, the decay factor is `(1 − ηλ)`, so for `ηλ > 2` the weight component itself can grow geometrically.

**Gradient bound** [T]. Row `i` of the gradient is `∇_{W_i} L = (N_i/M)(p_i − q_i)`. Since `p_i` and `q_i` are both probability vectors, `‖p_i − q_i‖₂ ≤ √2` (equality when both are one-hot on different tokens). Hence:

    per row:      ‖∇_{W_i} L‖₂  ≤ √2 · N_i/M
    full matrix:  ‖∇L(W)‖_F     = ( Σ_i ‖∇_{W_i} L‖₂² )^{1/2}  ≤  √( 2 Σ_i (N_i/M)² )  ≤  √2

The last step uses `Σ_i N_i/M = 1`, so `Σ_i (N_i/M)² ≤ 1`.
- The per-row bound `√2·N_i/M` does **not** carry over to the whole matrix with `max_i` in place of `N_i`. The Frobenius norm sums over rows, so `√2·max_i N_i/M` is not a valid global bound.
  - [O] On the train split, `√2·max_i N_i/M = 0.210`, but the supremum of the gradient norm over all finite `W` is 0.294.
  - The supremum is approached, not attained. `‖p_i − q_i‖²` is convex in `p_i`, so its maximum over the simplex lies at a vertex, the one-hot vector on row `i`'s least-likely target. Finite logits give strictly positive probabilities and never reach that vertex, so 0.294 is approached as those logits grow without limit.
  - The valid bound `√(2 Σ_i (N_i/M)²)` is 0.395.
  - [O] At zero initialisation the actual gradient norm is 0.074.
- Contrast with the step-size condition above, where `max_i N_i/(2M)` *is* correct. That bounds the spectral norm of a block-diagonal Hessian, which equals the largest block's norm; it is not a sum over rows.
- With weight decay, the gradient of `J` adds `λW`, which is unbounded in `W`. In exact arithmetic and within the step-size condition, `J` decreases, so the iterates stay in a sublevel set of `J`. That set is bounded because `J ≥ (λ/2)‖W‖²`. Outside the condition no such bound is claimed.

**Convergence speed depends on context frequency** [T]. Row `i`'s curvature scales with `N_i/M`, so rare contexts converge slowest.
- [O] `N_q/M = 0.0012`, against `N_a/M = 0.149`.
- [O] After 20,000 steps at `η ≈ 25.5`, the largest `|p − q|` on observed cells was 1.4e-3 for `'q'`, against 6e-17 for `'a'`.

**Zero initialisation fixes the gauge** [T]. Gradient rows sum to 0, and zero initialisation starts every row at sum 0. Without weight decay the update keeps the row sum unchanged; with coupled weight decay it multiplies the row sum by `(1 − ηλ)`. Either way, every row of `W` sums to 0 at every step.
- The trajectory stays in the subspace `{Σ_j W_ij = 0}`, which removes the shift non-identifiability. **Logits remain zero-mean (centred) within each context row.**
- The invariant applies to the logits, not to the log-probabilities. `log P(j | i) = W_ij − logsumexp(W_i)`, so `Σ_j log P(j | i) = −V·logsumexp(W_i)`. That sum is negative and generally not zero; at zero init it equals `−V log V`.
- [O] Row-sum drift was below 1e-13. This is tested as an invariant (§13.2).

### 4.6 Reference training procedure (separate from the architecture)

This is the one procedure the implementation provides (D4, D5). It is model-specific and minimal, not a general framework.
- **Data:** flat pairs `(contexts [M], targets [M])` from `make_bigram_pairs`, which builds them from the shared `make_next_token_sequences` as `inputs[mask]`, `targets[mask]` (D3). Padding never reaches the model.
- **Two quantities, never conflated:**
  - **Data loss** `L_data(W) = L(W)`: mean cross-entropy over the pairs, computed by `utils.metrics.token_nll(model(contexts), targets, reduction="mean")` (D9). This is the quantity reported, compared with the statistical bigram, and used for evaluation.
  - **Optimised objective** `J(W) = L_data(W) + (λ/2)‖W‖²_F`, with `λ = 0` by default (D6). This is what each SGD step decreases. When `λ = 0`, `J = L_data`.
- **What is guaranteed** [T]:
  - For `η < 2/(ℓ + λ)`, `J` is non-increasing at every step.
  - For `λ > 0` that implies **nothing** about `L_data` on its own. A step may trade higher data loss for a smaller norm.
  - [O] On the §13 fixture from zero initialisation with `λ = 0.2`, `η = 4`, `L_data` rose by 0.106 in one step, while `J` never rose by more than float32 rounding (5e-8).
  - [O] After an unregularised fit, switching on `λ = 0.05` raised `L_data` from 0.2775 to 0.4708 while `J` fell from 4.08 to 0.63.
- No monotonicity requirement is ever placed on `L_data` when `λ > 0`.
- **Optimiser:** full-batch `torch.optim.SGD(model.parameters(), lr=η, momentum=0, weight_decay=λ)`. SGD's coupled weight decay adds `λW` to the gradient, so each step is exactly

      W ← W − η (∇L(W) + λ W)

  which is gradient descent on `J`. AdamW, momentum, schedules and mini-batches are not part of the reference procedure.
- **Determinism:** no data RNG (full batch), no initialisation RNG (zeros). On a fixed device and software stack, two runs give identical parameters.
- **Returned summary (minimal, Q4):** the data loss `L` (without the penalty) before the first update and after the last update. No per-step history is kept. Experiment scripts that need curves can call `train_full_batch(steps=1)` repeatedly, or record inside their own loop. Because full-batch SGD with `momentum=0` keeps no optimiser state, `k` calls with `steps=1` are bit-identical to one call with `steps=k` [I, tested]. `TrainingResult` reports **`L_data` only**. The penalty is never folded into it; callers who need `J` compute `final_loss + (λ/2)‖W‖²` from the model's parameters.
- **With `λ > 0`** [T]: `J` is strictly convex, so the minimiser `W*_λ` exists, is unique and is finite.
  - Stationarity gives `(N_i/M)(p_i − q_i) + λW*_i = 0`.
  - Hence every row of `W*_λ` sums to 0, and `W*_ij = (N_i/(λM))(q_ij − p_ij)`.
  - Unseen cells get finite negative logits.
  - Rows with small `N_i` stay closer to uniform.

  This is qualitatively like additive smoothing, which is also stronger for small `N_i`, but it is **not** the same estimator; no closed form maps `λ` to `α`.
  - [O] `λ = 1e-4` gave dev NLL 2.4757 and `λ = 1e-3` gave 2.5912, both worse than unregularised.

## 5. Tensor and interface contract

| Item | Shape | Dtype | Device | Meaning |
|---|---|---|---|---|
| `contexts` (forward input) | `[*S]`, any shape | int64 | model device | previous-token ids in `[0, V)` |
| `transition_logits` (parameter) | `[V, V]` | float32 | model device | `W` |
| logits (forward output) | `[*S, V]` | float32 | model device | unnormalised next-token scores (**logits**, not probabilities) |
| `transition_probabilities()` | `[V, V]` | float32 | model device | `softmax(W, dim=1)`, no grad |
| `transition_log_probabilities()` | `[V, V]` | float32 | model device | `log_softmax(W, dim=1)`, no grad |
| training targets | `[M]` | int64 | model device | next-token ids; pair form never contains `IGNORE_INDEX` |

- **Batch / sequence / vocabulary dimensions:** `*S` is any batch shape. The canonical training form is `[M]` (flat pairs). `[B, T]` inputs are accepted for compatibility with the UMIS sequence convention, provided they contain no padding.
- **Input token representation:** integer ids from the shared `Vocabulary`.
- **Output representation:** logits.
- **Variable-length sequences:** not needed (pairs).
- **Padding convention:** padding must never reach the model (D3). `forward` raises `ValueError` for PAD (`vocab.pad_id`, a dedicated message) and for any id outside `[0, V)` (D8).
- **Mask convention:** N/A.
- **Length handling:** each name contributes `len + 1` pairs: BOS → x₁, the internal transitions, and x_L → EOS.
- **Special-token handling:** row 0 is the BOS context; column 0 is the EOS target. `W[0, 0]` (BOS → EOS) is never observed.
- **Device behaviour:** see §11.

### Public API (to implement)

```python
@dataclass(frozen=True)
class NeuralBigramConfig:       # model.params in configs/models/neural_bigram.yaml
    """No fields: the architecture has no hyperparameters (V comes from the vocabulary).
    Exists so that any key under model.params is rejected as unknown."""

@dataclass(frozen=True)
class NeuralBigramSample:       # model-specific (D7); same semantics as the statistical GeneratedName
    text: str                   # characters between BOS and EOS; never contains "."
    finished: bool              # True iff EOS was drawn

@dataclass(frozen=True)
class TrainingResult:
    initial_loss: float         # L_data(W_0): data loss before the first update; NOT the objective J
    final_loss: float           # L_data(W_steps): data loss after the last update; NOT the objective J
    # With weight_decay > 0, final_loss may exceed initial_loss even though J decreased.

class NeuralBigramLanguageModel(nn.Module):
    def __init__(self, vocab: Vocabulary) -> None                   # W = zeros([V, V]), on CPU
    @classmethod
    def from_config(cls, vocab, config: NeuralBigramConfig) -> "NeuralBigramLanguageModel"
    num_classes: int                                                 # V
    def forward(self, contexts: LongTensor[*S]) -> FloatTensor[*S, V]    # logits = W[contexts]
    def transition_probabilities(self) -> FloatTensor[V, V]              # softmax(W, 1), no grad
    def transition_log_probabilities(self) -> FloatTensor[V, V]          # log_softmax(W, 1), no grad
    @torch.no_grad()
    def generate(self, num_samples: int, *, max_length: int, generator=None,
                 temperature=1.0, top_k=None, top_p=None) -> list[NeuralBigramSample]
    @classmethod
    def from_checkpoint(cls, path, *, expected_vocab: Vocabulary | None = None,
                        map_location="cpu") -> "NeuralBigramLanguageModel"

def make_bigram_pairs(names: Sequence[str], vocab: Vocabulary) -> tuple[LongTensor[M], LongTensor[M]]

def train_full_batch(model, contexts, targets, *, lr: float, steps: int,
                     weight_decay: float = 0.0) -> TrainingResult
    # §4.6. Mutates model parameters in place. Validates: lr finite > 0, steps int ≥ 1,
    # weight_decay finite ≥ 0, contexts/targets the same shape [M] with M ≥ 1, on the model's device.
```

Evaluation is not a model method. Use `NLLAccumulator` (or `token_nll(reduction="sum")`), which applies because the neural NLL is always finite.

## 6. PyTorch implementation policy

### 6.1 Permitted operations and modules
- `nn.Module` and `nn.Parameter`.
- `torch.zeros` for initialisation.
- Integer advanced indexing `W[contexts]`; its backward pass scatter-adds into the selected rows.
- `torch.softmax` / `torch.log_softmax`, and `F.cross_entropy` through the shared `token_nll`.
- `torch.optim.SGD` with `momentum=0` and `weight_decay=λ`, in `train_full_batch` only.
- `torch.autograd.gradcheck` and `torch.nn.functional.one_hot` (test oracle) in tests.

### 6.2 Prohibited operations and modules
- `nn.Embedding` (it *is* this model, packaged) and `nn.Linear` (it hides the one-hot equivalence and adds a bias by default).
- Any bias parameter.
- Random initialisation of any kind.
- `AdamW`, and any decoupled weight decay.
- Momentum, learning-rate schedules and mini-batching in the reference procedure.
- Packaged language-model modules.
- A hand-written `exp(z)/exp(z).sum()`.
- Probability clamping.
- Global RNG reseeding.
- Loss computation or optimiser steps inside `forward`.

### 6.3 Manual implementation requirements
Written explicitly in `neural_bigram.py`:
- the zero-initialised parameter table and the lookup `forward`;
- token-id validation;
- the pair adapter;
- `train_full_batch` (zero-grad → forward → `token_nll` → backward → step; returns initial and final data loss);
- the generation loop;
- checkpoint loading with vocabulary validation.

### 6.4 Shared utility integration
- **Data:** `Vocabulary`, and `make_next_token_sequences` (pairs via `mask`).
- **Metrics:** `token_nll(reduction="mean")` for training; `NLLAccumulator` for evaluation; `count_parameters`.
- **Generation:** `sample_next_token`, with an explicit generator from `make_generator(derive_seed(seed, "generation"))`.
- **Device:** `resolve_device` in calling code, then `model.to(device)`.
- **Checkpoint:** `save_checkpoint` with kinds `"best"`, `"final"` and `"resume"`; `load_checkpoint`; `restore_model`.
- **Config:** `load_config(..., model_config_cls=NeuralBigramConfig)`.
- **Diagnostics:** `global_grad_norm`, `gradient_summaries`, `ParameterSnapshot`.

`NeuralBigramSample` is model-specific; neither the statistical bigram nor shared utilities are modified (D7).

## 7. Initialization and parameter sharing

| Parameter group | Shape | Initialization | Rationale |
|---|---|---|---|
| `transition_logits` | `[V, V]` | `torch.zeros`, float32, created on CPU | Every row starts as the uniform distribution, so the initial loss is exactly `log V` (3.2958 for `V = 27`). Deterministic, no RNG. Symmetry breaking is unnecessary because the problem is convex and rows receive different gradients from different counts. Zero row sums are preserved by the training procedure (§4.5). |

- No bias, no parameter tying, no constraints, no random-initialisation option (D2).
- Initialisation experiments (E2) would set `W` explicitly in experiment code, outside the model.
- Initialisation never happens in `forward`.

## 8. Forward-pass specification

1. **Validate** [I] (always, D8):
   - `contexts` must be a tensor of dtype int64 (otherwise `TypeError`);
   - `contexts` must be on the model's device (otherwise `ValueError`; data is never moved silently);
   - every id must lie in `[0, V)` (otherwise `ValueError`); a PAD id gets the message "padding must not reach the model".

   This needs one `min`/`max` reduction and a host sync. The cost is accepted.
2. **Look up**: `logits = self.transition_logits[contexts]`, mapping `[*S]` to `[*S, V]`. Gradients flow only to the selected rows.
3. **Return the logits**: no softmax, no loss, no device transfer.

```python
def forward(self, contexts):
    _check_context_ids(contexts, num_classes=self.num_classes, pad_id=self.vocab.pad_id)
    return self.transition_logits[contexts]
```

## 9. Configuration

**Architecture configuration.** `NeuralBigramConfig` has no fields. `configs/models/neural_bigram.yaml` contains `model: {name: neural_bigram, params: {}}`, and any key under `params` is rejected as unknown.

**Reference-training settings.** These are arguments of `train_full_batch`. They are not architecture configuration, and not part of `model.params`.

| Argument | Type | Default | Validation | Meaning |
|---|---|---|---|---|
| `lr` | float | required | finite, > 0 | SGD step size `η` |
| `steps` | int | required | ≥ 1 | number of full-batch updates |
| `weight_decay` | float | `0.0` | finite, ≥ 0 | coupled L2 coefficient `λ` in `J = L + (λ/2)‖W‖²` |

- `lr` and `steps` have no defaults, because no values are justified for every dataset. §4.5 gives the sufficient condition `η < 2/(ℓ + λ)`, which callers can compute from the data.
- The shared config has no `training` section (gap G5), so these settings are function arguments only and are not loadable from YAML (Q1, approved). No shared configuration changes.

## 10. Numerical stability and edge cases

| Case | Behaviour | Test |
|---|---|---|
| Softmax / log-softmax | only stable primitives (log-sum-exp with max subtraction inside PyTorch) | logits of ±1e4 give a finite loss and valid probabilities |
| Zero probabilities | impossible for finite `W`: every probability is > 0, so the NLL is always finite | unseen-target probabilities stay positive throughout training |
| Unseen target, observed context | data-loss gradient `(N_i/M)p_ij > 0`. With `λ = 0` the logit decreases at every step. With `λ > 0` the decay term `−ηλW_ij` opposes this and the logit converges to a finite value (it may even rise; [O] −3.13 → −1.76 when decay is switched on after an unregularised fit) | §13.2 item 6 |
| Data loss versus objective with `λ > 0` | `J` non-increasing (within the step-size condition); `L_data` may increase. Only `J` is checked for monotonicity | §13.3 |
| Unseen context | zero gradient; row stays exactly 0 (uniform), with or without weight decay | §13.2 item 7 |
| Very negative logits after long training | grow like `−log t`; well within float32 range | — |
| PAD, negative or `≥ V` context | `ValueError` naming the id | dedicated tests |
| Non-int64 contexts | `TypeError` | dedicated test |
| Empty batch `[0]` | `forward` returns `[0, V]`; `train_full_batch` rejects `M = 0` | dedicated tests |
| Large learning rate | **No theoretical guarantee** of finite parameters or absence of NaN for arbitrary `η` or training length (§4.5). The data-loss gradient is bounded, `‖∇L_data‖_F ≤ √(2 Σ_i (N_i/M)²) ≤ √2`, which limits each step's size but not the total drift or float32 overflow. With `λ > 0` and `ηλ > 2`, iterates can diverge. [O] Tested runs up to `η = 1600` (2,000 steps, `λ = 0`) stayed finite and oscillated. Not treated as an error by the model; studied in E3 | experiment E3 |
| Precision | float32 parameters; loss accumulated per the shared metrics (float64 in `NLLAccumulator`); `gradcheck` on a float64 copy, CPU only | — |
| Backend | indexing, softmax, SGD and `multinomial` exist on CPU, CUDA and MPS | device tests |

## 11. Complexity and resource profile

- **Parameter count:** `V² = 729`.
- **Time complexity:** forward `O(|S|·V)` (gather); loss `O(|S|·V)`; one full-batch step `O(M·V)` ([O] about 4.9M elements).
- **Memory complexity:** full-batch logits `M × V × 4` bytes ≈ 20 MB, plus the same for their gradient.
- **Sequence-length scaling:** N/A (pairs).
- **Main bottlenecks:** none at this size. Training time is roughly `steps × O(M·V)`.
- **Device considerations:** parameters are created on CPU, then the model is moved with `model.to(device)`. Contexts and targets must already be on the model's device; neither `forward` nor `train_full_batch` moves data, and both raise on a mismatch. Generation samples with a CPU generator through `sample_next_token`, which moves the `[B, V]` probabilities to the generator's device.

## 12. Diagnostics

Shared tools, collected only on steps chosen by `DiagnosticsSchedule` (in experiment code, not inside `train_full_batch`):
- `global_grad_norm`;
- `gradient_summaries` (the single parameter forms one group);
- `ParameterSnapshot` update norms;
- `nonfinite_gradients`.

Architecture-specific diagnostics, computed from `W`, its gradient and `C`, all reduced to scalars or `[V]` vectors so no graph is retained:
- **excess loss** `Σ_i (N_i/M) KL(q_i ‖ p_i)`, the distance to the MLE;
- **per-row KL** and **per-row gradient norm**, to show how convergence depends on `N_i`;
- **max `|p − q|`** on observed cells;
- **unseen-target mass** per row and `t·p_ij`, for the 1/t law in observed contexts;
- **row sums of `W`**, an invariant equal to 0;
- **logit range** `min W` / `max W`.

## 13. Test plan

File: `tests/models/feedforward/test_neural_bigram.py`. The reference fixture is the statistical bigram's:

    names = ["a", "ab"], ids . = 0, a = 1, b = 2, V = 3, M = 5
    pairs (prev → next): .→a, a→., .→a, a→b, b→.
    C = [[0,2,0],[1,0,1],[1,0,0]],  q = [[0,1,0],[½,0,½],[1,0,0]],  MLE NLL = 2 ln 2 / 5 = 0.277259

All hand values below were verified [O] in float64.

### 13.1 General tests
- [x] Valid construction; `from_config`; YAML integration (`params: {}` loads; any key under `params` is rejected)
- [x] Invalid construction: a vocabulary of the wrong type
- [x] Output shape and dtype: `[B] → [B, V]`, `[B, T] → [B, T, V]`, `[0] → [0, V]`, float32
- [x] Token validation: PAD, negative and `≥ V` ids raise `ValueError`; non-int64 raises `TypeError`
- [x] Pair adapter: `make_bigram_pairs` gives `Σ (len + 1)` pairs, in the same order as the encoded names, containing no PAD or `IGNORE_INDEX`
- [x] Finite outputs, also for extreme logits (±1e4)
- [x] Finite gradients
- [x] Parameter count: `count_parameters(model).total == V²` (729 for `V = 27`), all trainable; exactly one parameter; no bias
- [x] Initialisation: `W` is exactly zero; initial loss equals `ln V`
- [x] Train/eval: identical outputs in `train()` and `eval()`
- [x] `train_full_batch` argument validation: lr, steps, weight_decay, shapes, device mismatch, `M = 0`
- [x] Determinism: two `train_full_batch` runs from fresh models give bit-identical parameters and histories (CPU)
- [x] Checkpoint save/load (§14)
- [x] Generation reproducible with a fixed generator
- [x] CPU execution
- [x] CUDA / MPS execution where available (skipped otherwise)

### 13.2 Mathematical verification
1. **One-hot equivalence:** `model(x) == F.one_hot(x, V).float() @ W` exactly, for arbitrary `W` set under `no_grad`.
2. **Softmax normalisation:** rows of `transition_probabilities()` sum to 1 (atol 1e-6), and every entry is > 0.
3. **Hand-computed cross-entropy:**
   - at zero init the loss is `ln 3 = 1.098612`;
   - with `W` zero except `W[0, 1] = ln 2`, the loss is `(2 ln 2 + 3 ln 3)/5 = 0.936426`.
4. **Analytic gradient:** the autograd gradient equals `(N_i p_ij − C_ij)/M`.
   - At zero init: rows `[2, −4, 2]/15`, `[−1, 2, −1]/15`, `[−2, 1, 1]/15`.
   - With the item-3 `W`: `[0.1, −0.2, 0.1]`, `[−1/15, 2/15, −1/15]`, `[−2/15, 1/15, 1/15]`.
   - Every gradient row sums to 0.
5. **Finite differences:** `torch.autograd.gradcheck` on a float64 copy with the fixture pairs.
6. **Unseen target in an observed context:** at zero init, the gradient of `W[0, 0]` and `W[0, 2]` (BOS → EOS, BOS → b) is `(2/5)(1/3) = 2/15 > 0`. After one SGD step both logits have decreased.
7. **Unseen context:** vocabulary `(a, b, c)` fitted on the fixture, where `c` never occurs.
   - Row `c` has an exactly zero gradient.
   - After training with `λ = 0` and with `λ > 0`, row `c` is still exactly 0, and its probabilities are exactly uniform (1/4).
8. **Gradient flow:** a batch containing only context 0 gives exactly zero gradient in rows 1..V−1 and a nonzero row 0.
9. **Count-form oracle** (D9): pair-form loss = `−(1/M) Σ C_ij log softmax(W)_ij` (rel 1e-6). Counts come from `models.statistical.bigram.count_transitions`. This read-only, **test-only** dependency is approved. `neural_bigram.py` itself must not import anything from the statistical bigram [I].
10. **Excess-loss identity:** `L(W) − L_MLE = Σ_i (N_i/M) KL(q_i ‖ p_i)` for arbitrary `W` (atol 1e-5 in float32), with `L_MLE` from the accepted `BigramLanguageModel(α=0).evaluate`.
11. **Zero row-sum invariant:** after `train_full_batch` with `λ = 0` and with `λ > 0`, every row of `W` sums to 0 (atol 1e-5).
12. **Regularised SGD update:** for `λ > 0` and from a non-zero `W` (so the decay term is active), one call `train_full_batch(steps=1, weight_decay=λ)` gives exactly the manual update `W − η(∇L_data(W) + λW)`, where `∇L_data` comes from autograd on the data loss alone. Tolerance: atol 1e-6 in float32 ([O] 2e-16 in float64). The same test with `λ = 0` reduces to `W − η∇L_data(W)`.
13. **Weight-decay stationarity:** after convergence with `λ > 0` on the fixture, `(N_i/M)(p_i − q_i) + λW_i ≈ 0` for observed rows (atol 1e-5).

### 13.3 Learning and behaviour checks
- **Overfit the tiny fixture:**
  - Setup: zero init, `train_full_batch(lr=5, steps=5000)`, float32. The sufficient bound for this fixture is `2/ℓ = 2/(0.4/2) = 10`.
  - Expected [O]: excess loss 8.0e-5 and max `|p − q|` 1.3e-4 on observed cells.
  - Test tolerances: excess `< 1e-3`; `|p − q| < 1e-2` on observed cells; every unseen-target probability `> 0` and `< 1e-3`.
- **Monotone descent of the objective:**
  - Procedure: repeat `train_full_batch(steps=1, weight_decay=λ)` on the fixture, and after each call compute `J_t = final_loss + (λ/2)‖W‖²` from the model's parameters.
  - Assertion: for `λ = 0` and for `λ > 0`, with `η < 2/(ℓ + λ)` (fixture `ℓ = 0.2`), the sequence `J_t` is non-increasing, allowing 1e-6 for float32 rounding ([O] max observed increase 1.1e-7).
  - No monotonicity assertion is made on `final_loss` when `λ > 0`.
  - A companion test documents the distinction: with `λ = 0.2`, `η = 4` from zero init, `J_t` is non-increasing while `L_data` increases on at least one step ([O] +0.106).
- **Step composition:** 50 calls with `steps=1` give parameters bit-identical to one call with `steps=50` (CPU), for both `λ = 0` and `λ > 0`.
- **Generation and EOS** (the statistical bigram's semantics):
  - `text` never contains `.`, and contains only vocabulary characters;
  - a model trained on `["ab"]` to near-determinism gives `("a", False)` for `max_length = 1` and `("ab", True)` for `max_length = 2`;
  - output is reproducible with a fixed generator.
- A successful training run is **not** evidence of correctness. The correctness evidence is §13.2.

### 13.4 Experiments (optional; outside the implementation task)
These use `train_full_batch` and the diagnostics of §12. How their results are recorded is deferred until E1 is designed (Q3). Experiment scripts may record every step.

Essential first experiment:
- **E1. Recovery of the MLE.**
  - Setup: group split, seed 42; zero init; `lr = 25` (inside the sufficient bound); `λ = 0`; up to 10⁵ steps; logs at geometric steps.
  - Metrics: excess loss; per-row KL against `N_i`; max `|p − q|`; unseen-target mass and `t·p`; dev NLL. Compare with the statistical bigram at α = 0 and on an α grid.
  - Expectations [T]: monotone decrease of the excess loss toward 0; rare contexts converge last; unseen-target mass ~1/t; dev NLL eventually rising like `(9/22801) log t`.
  - *Run:* a reduced version of E1 (up to 10⁵ steps; train and dev NLL, excess loss, max |p − q| on observed cells, unseen-target mass, P(EOS | BOS)) is implemented in `training/bigram_study/run_bigram_study.py`, with results in `results/experiments/bigram_study/results.json`.

Optional:
- **E2. Initialisation:** zero against Gaussian `W` set in experiment code, with several seeds.
- **E3. Learning rate:** `η` from 1 to 400, on both sides of the sufficient bound and the empirical breakdown.
- **E4. Weight decay:** `λ` on a log grid. Measure zero row sums and dev NLL, and compare with the best additive-smoothing α.
- **E5. Optimiser:** Adam against full-batch SGD, including Adam's fixed-lr floor.
- **E6. Mini-batch SGD:** needs shared batching (G1).

## 14. Checkpoint and serialization contract

- **Learned state:** `state_dict() = {"transition_logits": [V, V] float32}`. No buffers.
- **Required metadata:** `vocab = vocab.to_dict()` and `config` (resolved experiment config, or `{"model": {"name": "neural_bigram", "params": {}}}`).
- **Kinds (shared semantics):**
  - `"best"`: weights at the best dev NLL;
  - `"final"`: weights at the end of training;
  - `"resume"`: weights, SGD state, step and RNG states.
- **Loading:** `NeuralBigramLanguageModel.from_checkpoint(path, expected_vocab=...)` uses `weights_only=True` and then:
  - requires `ckpt["vocab"]` and builds the `Vocabulary` from it;
  - rejects a mismatch with `expected_vocab`;
  - checks `model_class`;
  - runs `restore_model(strict=True)`.

  The vocabulary check is required: a same-size vocabulary with a different character order would otherwise load silently (gap G4).
- **Round-trip tests:**
  - `final`: identical parameters, logits, probabilities and seeded generation;
  - `resume`: after `k` reference steps, save; restore into fresh objects; continuing must equal the uninterrupted run bit-for-bit on CPU;
  - rejected: vocabulary mismatch, wrong model class, wrong `V` (shape).

## 15. Generation or inference behaviour

- **Inference entry point:** `generate(num_samples, *, max_length, generator=None, temperature=1.0, top_k=None, top_p=None)`, decorated with `@torch.no_grad()`.
- **Prediction procedure:** ancestral sampling `x_t ~ softmax(W[x_{t-1}])`. Unfinished samples are drawn in parallel, and the logits go to `sample_next_token`.
- **Start/end-token behaviour:** every sample starts at BOS and stops at EOS; neither appears in `text`.
- **Sampling controls:** the defaults sample the model distribution exactly. Temperature, top-k and top-p use the shared semantics; `temperature = 0` is greedy.
- **RNG:** an explicit generator, by convention `make_generator(derive_seed(seed, "generation"))`.
- **Evaluation mode:** the model has no mode-dependent behaviour; `generate` does not change `self.training`.
- **Maximum length (identical to the statistical bigram):**
  - at most `max_length` ordinary characters;
  - a sample that reaches `max_length` gets one more draw, and EOS on that draw gives `finished = True`;
  - otherwise the sample is returned truncated with `finished = False`;
  - at most `max_length + 1` draws per sample.
- **Degenerate distributions:** none. Every row is a strictly positive distribution for finite `W`. An unseen context's row is uniform, not undefined.

## 16. Implementation scope and boundaries

Implement in `models/feedforward/neural_bigram.py`:
- `NeuralBigramConfig`, `NeuralBigramSample` and `TrainingResult`;
- `NeuralBigramLanguageModel`;
- `make_bigram_pairs` and `train_full_batch`;
- the token-id validation helper.

`train_full_batch` is a model-specific reference procedure, allowed by UMIS §12 as a small helper intrinsic to the model (D4). It must stay minimal: no logging, checkpointing, run management, diagnostics scheduling or mini-batching.

**Necessary supporting files:**
1. `configs/models/neural_bigram.yaml`: required by `default_config_paths("neural_bigram")`.
2. `tests/models/feedforward/test_neural_bigram.py`: the §13.1–13.3 tests.

**Not in scope:**
- shared utility changes (including `GeneratedName`, D7, and the vocabulary check in `restore_model`, G4);
- any change to the statistical bigram;
- a general training framework;
- experiments E1–E6, plotting, notebooks;
- the MLP or any other model.

## 17. Acceptance criteria

- [x] Exact variant: bias-free `[V, V]` logit table, zero-initialised, lookup `forward`, logits out (D1, D2)
- [x] Mechanism explicit: no `nn.Embedding`, `nn.Linear`, bias or random initialisation
- [x] Token ids always validated, including the range check and the PAD message (D8)
- [x] Training data are flat pairs from the shared sequence utility; padding never reaches the model (D3)
- [x] `train_full_batch` implements exactly `W ← W − η(∇L + λW)` with full-batch SGD; `λ = 0` by default; no AdamW, momentum or schedules (D4–D6)
- [x] Training uses pair cross-entropy via `token_nll`; the count form is used only as a test oracle (D9)
- [x] `train_full_batch` returns only `TrainingResult(initial_loss, final_loss)`, both **data loss `L_data`**, never the objective `J`; no per-step history (Q4); `steps=1` calls compose bit-identically
- [x] The monotone-descent test checks `J = L_data + (λ/2)‖W‖²`; no monotonic-data-loss requirement with `λ > 0`; the regularised update `W − η(∇L_data + λW)` is verified exactly
- [x] Training settings are function arguments only; no shared configuration change (Q1); empty typed `NeuralBigramConfig` rejects stray keys (Q2)
- [x] `neural_bigram.py` imports nothing from `models.statistical`; only the tests use `count_transitions`
- [x] `NeuralBigramSample` is model-specific; the statistical bigram and shared utilities are unchanged (D7)
- [x] Public API and tensor contracts documented in docstrings
- [x] Configuration validated (empty `params`)
- [x] General tests (§13.1) pass
- [x] Mathematical tests 1–13 (§13.2) pass, including unseen-target suppression and unseen-context invariance
- [x] Learning and behaviour checks (§13.3) pass
- [x] Checkpoint round trips, including bit-exact `resume`, verified
- [x] CPU and available accelerators (MPS on the user's Mac) checked
- [x] No unresolved material code-review issue
- [x] Implementation run and reviewed locally by the project author
- [x] Completion approved by the project author

## 18. Open decisions

### Approved

| # | Question | Decision |
|---|---|---|
| D1 | Bias | **No bias.** `W ∈ ℝ^{V×V}` only; logits = `W[x_{t-1}]` |
| D2 | Initialisation | **Zeros.** No random-initialisation option in the model; initialisation is investigated experimentally later |
| D3 | Data representation | **Flat previous/next pairs** from the shared sequence utilities; padding never reaches the model |
| D4 | Training loop | **Minimal reference procedure** `train_full_batch` in the model file; no general framework; no shared training changes |
| D5 | Optimiser | **Full-batch SGD**: transparent and deterministic |
| D6 | Regularisation | **None by default.** Explicit coupled L2 via SGD `weight_decay` for controlled experiments; no AdamW |
| D7 | `GeneratedName` | **Model-specific dataclass** (`NeuralBigramSample`); no change to the statistical bigram or utilities |
| D8 | Forward validation | **Always** validate dtype and range |
| D9 | Count-weighted loss | Pair cross-entropy for training; the count form is a **test oracle only** |

### Approved after review (v0.2 → v0.3)

| # | Question | Decision |
|---|---|---|
| Q1 | Where `lr`, `steps` and `weight_decay` live | **Function arguments** of `train_full_batch`; no shared configuration changes |
| Q2 | Empty `NeuralBigramConfig` | **Keep** the empty typed dataclass, so unexpected `model.params` keys are rejected |
| Q3 | How E1 results are recorded | **Deferred** until E1 is designed |
| Q4 | Training-history granularity | **No per-step history** in the reference procedure (`TrainingResult` holds initial and final loss only); experiment scripts may record every step |
| — | Test-only use of `count_transitions` from the statistical bigram | **Approved** for tests only; the implementation stays independent of the statistical bigram |

### Implementation details approved (v0.3 → v0.4)

| # | Detail | Decision |
|---|---|---|
| I1 | Gradient reset | `optimizer.zero_grad(set_to_none=True)` before every step |
| I2 | Device mismatch | `forward` and `train_full_batch` both raise `ValueError`; data is never moved silently |
| I3 | Checkpoint loading | `from_checkpoint` restores model weights only, from any checkpoint kind. Optimiser state is restored separately by the caller (shared `restore_training_state`). Full-batch SGD with `momentum=0` holds no state between updates |
| I4 | Reported loss versus objective | `TrainingResult` reports `L_data` only; tests check monotonicity of `J`; no `L_data` monotonicity is required with `λ > 0` |

No design questions remain open. The implementation was accepted.


# Model Specification — Statistical Character Bigram

**Status:** Implemented, tested and accepted. This specification describes the implementation in `models/statistical/bigram.py`.
**Governing standard:** `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md`
**Template:** `MODEL_SPECIFICATION_TEMPLATE.md`

## 1. Model identity

- **Model name:** `bigram`
- **File path:** `models/statistical/bigram.py`
- **Model family:** statistical (count-based) language models
- **Exact architecture variant:** first-order character Markov model with a shared BOS/EOS boundary token. Parameters are estimated in closed form from transition counts, either unsmoothed (maximum likelihood) or with additive smoothing.
- **Specification version:** 1.0
- **Reference implementation(s):** an earlier counting script from the first version of this project, available in the repository history (historical; not authoritative)
- **Shared utility dependencies:** `utils.vocabulary`, `utils.data.encode_names`, `utils.device.resolve_device`, `utils.generation.sample_next_token`, `utils.metrics.perplexity` / `bits_per_token`, `utils.checkpoint`, `utils.config`

## 2. Role and motivation

### 2.1 Problem addressed
Model the probability of a name, character by character, using only the previous character as context.

### 2.2 Why this architecture?
It is the simplest autoregressive language model with a closed-form estimate, so it is the first point of the model progression. It makes three things concrete that every later model shares:
- the BOS/EOS framing,
- token-weighted NLL,
- generation by sampling from a conditional distribution.

It also exposes a problem that neural models hide: with unsmoothed estimates, unseen events have zero probability and infinite held-out NLL. Finally, it is the first integration test of the shared infrastructure.

### 2.3 Learning and research questions
- *Established:* the MLE row `C_i· / N_i` maximises the training likelihood. Smoothing trades training fit for non-zero mass on unseen transitions.
- *Observed on this dataset* (group split, seed 42, train-only fit; sanity check, not a recorded run): α = 0 gives dev NLL = +∞ with 9 zero-probability dev events, and α = 1 gives dev NLL ≈ 2.457.
- *Open (experiment decision):* how α should be selected on dev for the controlled comparison with later models.

## 3. Exact architecture definition

- **Architecture definition:** `P(x_1..x_L, EOS) = P(x_1 | BOS) · Π_{t=2..L} P(x_t | x_{t-1}) · P(EOS | x_L)`, with one categorical distribution per context token.
- **Included components:** transition counting, row-normalised estimation (with optional additive smoothing), likelihood evaluation, ancestral sampling, persistence.
- **Excluded components:** neural parameters, gradients, optimizers, interpolation or back-off with lower-order models, Kneser–Ney or other discounting, and any context longer than one token.
- **Causal or bidirectional behaviour:** causal (left to right).
- **Autoregressive or non-autoregressive behaviour:** autoregressive.
- **Deliberate simplifications:** a single smoothing pseudo-count α shared by all cells, and float32 probabilities on every device.

## 4. Mathematical formulation

Symbols:
- `V = vocab.num_classes`: the output vocabulary size (27 for the names data).
- `i, j ∈ {0..V-1}`: token ids. Id 0 is `.` (BOS when used as a context, EOS when used as a target).
- `C_ij`: the number of training transitions `i → j`.
- `N_i = Σ_j C_ij`.

### 4.1 Core equations
Maximum likelihood (`alpha = 0`):

    P(j | i) = C_ij / N_i                       for N_i > 0
    row i undefined (stored as all zeros)       for N_i = 0

Additive smoothing (`alpha > 0`):

    P_α(j | i) = (C_ij + α) / (N_i + α V)

Only `alpha = 0` is a maximum-likelihood estimate. `alpha > 0` adds α pseudo-counts to every cell before each row is normalised, which moves mass from observed toward unobserved transitions. The effect is strong when `αV` is large relative to `N_i` and vanishes as `N_i` grows. Equivalently, it is the posterior mean of each row under a symmetric Dirichlet(α) prior. For a context unseen in training (`N_i = 0`), smoothing gives the uniform row `1/V`.

### 4.2 Component equations
- **Counting:** each name `x_1..x_L` (`L ≥ 1`) is framed as `0, x_1, …, x_L, 0` and contributes `L + 1` transitions. Each pair `(i, j)` is mapped to the flat index `i·V + j`, and `bincount` over the `V²` cells gives `C`.
- **Log-probabilities:** `log P` is computed with `torch.log`, so `log 0 = -inf` exactly.
- **Evaluation:** see §10.

### 4.3 Parameterization
The only learned state is the count matrix `C ∈ ℕ^{V×V}`. This gives `V²` cells (729 for `V = 27`), but `C` is a sufficient statistic, not a set of trainable parameters. Probabilities are derived from `C` and α, cached, and never stored separately.

### 4.4 State and information flow
The state carried during generation is the previous token. There is no memory beyond one token.

### 4.5 Assumptions and limitations
- The Markov assumption ignores all context beyond one character.
- Under MLE, held-out data containing any unseen transition has NLL = +∞.
- Smoothing assigns positive probability to the empty name (`P(. | .) > 0`), even though the dataset contains none.

**Count-matrix structure.** A matrix is a valid fitted count matrix exactly when it is non-negative and integer and satisfies all of:
- `Σ_j C_0j > 0` (at least one name);
- `C_00 = 0` (no empty name);
- flow conservation (row sums equal column sums for every token);
- every used token is reachable from token 0 along positive-count edges.

Sufficiency follows from the Eulerian-circuit argument documented in `check_fitted_counts`: a balanced, connected multigraph has a closed walk using every edge once, and cutting it at each visit to token 0 yields the names.

## 5. Tensor and interface contract

| Item | Shape | Dtype | Device | Meaning |
|---|---|---|---|---|
| `fit` / `evaluate` input | list of `str` | — | — | names (non-empty, in-vocabulary characters) |
| `count_transitions` output | `[V, V]` | int64 | CPU | raw counts, row = previous, column = next |
| `counts()` | `[V, V]` | int64 | model device | copy of `C` |
| `probabilities()` | `[V, V]` | float32 | model device | copy of `P(j | i)` |
| `log_probabilities()` | `[V, V]` | float32 | model device | copy of `log P`, `-inf` where `P = 0` |
| `defined_contexts()` | `[V]` | bool | model device | rows that are proper distributions |
| `sample_next` input | `[B]` | int64 | any (moved to model device) | previous-token ids in `[0, V)` |
| `sample_next` output | `[B]` | int64 | model device | sampled next-token ids |
| `evaluate` output | `BigramEvaluation` | Python numbers | — | see §10 |
| `generate` output | `list[GeneratedName]` | — | — | `(text, finished)` per sample |

- **Batch / sequence / vocabulary dimensions:** `B` = number of contexts (or samples) and `V = vocab.num_classes`. There is no sequence tensor: names are handled as strings.
- **Input token representation:** characters are mapped through the shared `Vocabulary`. No second character-to-index map exists.
- **Output representation:** probabilities and log-probabilities, not logits. Log-probabilities are passed to the shared sampler as logits, since `softmax(log p) = p`.
- **Variable-length sequence support:** yes, because each name is counted independently.
- **Padding convention:** not used. PAD (id 27) never appears; the model needs no padded tensors.
- **Mask convention:** N/A.
- **Length handling:** a name of length `L` contributes `L + 1` predicted tokens, including EOS.
- **Special-token handling:** `.` (id 0) is BOS as row 0 and EOS as column 0. A name containing `.` is rejected by the vocabulary.
- **Device behaviour:** see §11.

## 6. PyTorch implementation policy

### 6.1 Permitted operations and modules
Tensor creation, `torch.bincount`, `reshape`, elementwise arithmetic, `torch.where`, `torch.log`, reductions, indexing, comparisons, `torch.isneginf` / `isfinite`, and dtype/device casts. Sampling goes through `utils.generation.sample_next_token`, which uses `softmax` and `multinomial`.

### 6.2 Prohibited operations and modules
`nn.Module`, `nn.Parameter`, autograd, any optimizer, neural layers, probability clamping or epsilon-flooring, and fallback distributions for undefined rows.

### 6.3 Manual implementation requirements
Counting, estimation (both formulas), zero-probability accounting, count-matrix validation, and the generation loop are written explicitly in `bigram.py`.

### 6.4 Shared utility integration
- **`Vocabulary`:** token ids, `encode` / `decode`, and `to_dict` / `from_dict` for persistence.
- **`encode_names(add_bos=True, add_eos=True)`:** framing for counting.
- **`resolve_device`:** constructor and `to()`. An explicit, unavailable device raises an error.
- **`sample_next_token`:** temperature, top-k and top-p, with an explicit generator.
- **`perplexity`, `bits_per_token`:** derived evaluation quantities.
- **`save_checkpoint`, `load_checkpoint`, `restore_model`:** persistence through the `state_dict` / `load_state_dict` protocol. This is duck-typed; there is no `nn.Module`.
- **`load_config(..., model_config_cls=BigramConfig)`:** configuration.

Not used, deliberately:
- **`NLLAccumulator`** rejects non-finite sums by design, and MLE evaluation can legitimately be +∞.
- **`token_nll`** re-normalises through log-softmax and returns NaN for an all-zero row. Tests still use it as an independent cross-check of the smoothed NLL.

## 7. Initialization and parameter sharing

| Parameter group | Shape | Initialization | Rationale |
|---|---|---|---|
| counts `C` | `[V, V]` | `None` until `fit` / `load` | no random initialization; estimation is closed form |

No randomness during fitting, no parameter tying, and no RNG streams beyond the generation generator. Calling `fit` again replaces the previous state.

## 8. Forward-pass specification

There is no forward pass. The operations, in order:

1. **`fit(names)`**
   1. Validate the names (non-empty list of non-empty strings; a single string is rejected).
   2. Count transitions on CPU, giving `[V, V]` int64.
   3. Move the counts to the model device.
   4. Estimate `P` in float32.
   5. Validate `P` (§10).
   6. Cache `log P`.
2. **`evaluate(names)`**
   1. Count the held-out transitions `E` on CPU, as `[V, V]` float64.
   2. Copy `log P` to CPU as float64 (one `[V, V]` transfer).
   3. Reduce as in §10.
3. **`sample_next(contexts)`**
   1. Validate the shape, dtype and range of the contexts.
   2. Raise if any context is undefined.
   3. Take `logits = log P[contexts]` (`[B, V]`) and call `sample_next_token`.
4. **`generate(n, max_length)`**: see §15.

## 9. Configuration

| Field | Type | Default | Validation | Meaning |
|---|---|---|---|---|
| `alpha` | float | `0.0` | finite, `≥ 0`; `bool` / `str` / `None` rejected | additive smoothing pseudo-count |

- Config file: `configs/models/bigram.yaml` (`model.params.alpha: 0.0`), loaded with `load_config(default_config_paths("bigram"), model_config_cls=BigramConfig)`.
- `alpha = 0.0` is the specified default (MLE), not a tuned value.
- An `alpha > 0` whose smoothed probabilities underflow to 0 in float32 (e.g. `1e-46`) is rejected at fit time.
- `device` is a constructor argument, not a model hyperparameter. The shared `device.dtype` setting does not apply: probabilities are always float32.

## 10. Numerical stability and edge cases

| Case | Behaviour | Test |
|---|---|---|
| Unseen transition, α = 0 | `P = 0` exactly, `log P = -inf`; no clamping | `test_zero_probabilities_are_exact_and_log_is_neg_inf` |
| Evaluated event with `P = 0` | counted in `zero_probability_events`; `total_nll = mean_nll = +inf`; kept in `num_tokens` | `test_zero_probability_events_give_inf_not_nan` |
| `0 · log 0` | never formed: the sum runs only over cells with `E_ij > 0` and `P_ij > 0` | same test (asserts not NaN) |
| Context with `N_i = 0`, α = 0 | row stored as zeros, `defined_contexts()[i] = False`; evaluation counts its events as zero-probability; `sample_next` raises `UndefinedDistributionError` | `test_row_without_observed_transitions`, `test_unseen_context_counts_as_zero_probability`, `test_undefined_row_raises_clearly` |
| Same context, α > 0 | uniform row `1/V` | `test_row_without_observed_transitions` |
| Tiny α underflowing float32 | `ValueError` at fit | `test_alpha_that_underflows_float32_is_rejected` |
| Empty input / empty name / single string / non-string | `ValueError` / `TypeError` | `test_fit_rejects_invalid_data` |
| Unknown character | `VocabularyError`. It is not scored as a zero-probability event, because it has no id. | `test_unknown_characters_are_rejected_not_scored` |
| Probability validation | finite, non-negative, defined rows sum to 1 (`atol = 1e-5`) | checked on every fit/load |

Evaluation formula:

    total_nll        = - Σ_{i,j: E_ij > 0, P_ij > 0} E_ij · log P_ij      (+inf if zero_probability_events > 0)
    num_tokens       = Σ_ij E_ij = Σ_names (len(name) + 1)                 (EOS included)
    zero_prob_events = Σ_{i,j: E_ij > 0, P_ij = 0} E_ij
    mean_nll         = total_nll / num_tokens                               (token-weighted)

Precision: log-probabilities are float32 (relative error ≈ 1e-7), and the sum is accumulated in float64.

## 11. Complexity and resource profile

- **Parameter count:** none trainable. The state is `V²` int64 counts plus two cached `V²` float32 matrices (≈ 14 KB for `V = 27`).
- **Time complexity:** fit and evaluate are `O(Σ (len + 1))` for counting plus `O(V²)` for estimation and reduction. `generate` makes at most `max_length + 1` sampling steps, each `O(B·V)`.
- **Memory complexity:** `O(V²)` state plus `O(total tokens)` temporary id lists during counting.
- **Sequence-length scaling:** linear.
- **Main bottlenecks:** encoding names in Python, and one host sync per generation step.
- **Device considerations:**
  - Counting and the final NLL reduction run on CPU, because the inputs are strings and the reduction is `V×V` float64. MPS has no float64.
  - Estimation, cached probabilities and sampling run on the model device: CPU, CUDA, or MPS where available.
  - With a CPU generator, the shared sampler moves the `[B, V]` probabilities to the generator's device.

## 12. Diagnostics

Not applicable in the neural sense: there are no gradients, activations or updates. The informative quantities are part of the API:
- `zero_probability_events`;
- `defined_contexts()`;
- the number of zero cells in `probabilities()`;
- `num_training_names` and `num_training_transitions`, stored in the checkpoint's `extra`.

## 13. Test plan

File: `tests/models/statistical/test_bigram.py`. CUDA and MPS tests are skipped when unavailable.

### 13.1 General tests
- [x] Valid construction and configuration (`TestConstruction`, including YAML config integration)
- [x] Invalid configuration raises clear errors (alpha, vocabulary type, device, config typos)
- [x] Input/output shapes and dtypes (int64 counts, float32 probabilities, `[B]` samples)
- [x] Vocabulary and special-token behaviour (shared ids, `.` framing, unknown characters)
- [x] Padding, lengths, and masks: N/A (no padded tensors); lengths are covered by the `L + 1` token count
- [x] Finite outputs (smoothed models); +∞ without NaN for MLE zero events
- [x] Finite gradients for neural models: N/A
- [x] Expected parameter count and trainability: N/A (no parameters); state shape tested
- [x] Train/eval behaviour: N/A
- [x] Checkpoint save/load (`TestPersistence`)
- [x] Controlled-seed reproducibility (generation with `make_generator`)
- [x] CPU execution (`TestDevices[cpu]`)
- [x] CUDA/MPS execution where available: parametrized; MPS runs on the author's Mac, CUDA is skipped where unavailable

### 13.2 Mathematical verification
Hand-computed reference, `names = ["a", "ab"]`, ids `. = 0, a = 1, b = 2`:

    C = [[0, 2, 0],      P_MLE = [[0,   1, 0  ],      P_{α=1} = [[1/5, 3/5, 1/5],
         [1, 0, 1],               [1/2, 0, 1/2],                 [2/5, 1/5, 2/5],
         [1, 0, 0]]               [1,   0, 0  ]]                 [2/4, 1/4, 1/4]]

**Model-specific mathematical tests:**
1. Exact count matrix (every entry), the `anna` framing, counts independent of order, and one BOS transition per name.
2. Exact MLE and α = 1 matrices; the smoothing formula on random α; rows sum to 1; the empty-row smoothed value `1/V`.
3. Training NLL = `2 ln 2 / 5` over 5 tokens. The token-weighted mean (0.2773) differs from the mean of per-name NLLs (0.2888).
4. EOS included: evaluating `["a"]` gives `ln 2` over 2 tokens.
5. Zero events: `["b"]` under MLE gives +∞ with 1 event and no NaN. `["bb", "a"]` gives 2 events with 5 tokens in the denominator. Under α = 1, `["b"]` gives exactly `ln 10`.
6. Independent cross-checks: a brute-force sum of `log transition_probability`, and the shared `token_nll` on log-probabilities.
7. Flow conservation holds for fitted counts. It is necessary but not sufficient: a balanced detached cycle passes it and is rejected by `check_fitted_counts`. 200 random fitted matrices all pass the full check.
8. Generation:
   - a deterministic model always produces `"ab"`;
   - `max_length` boundaries (1 gives `("a", False)`, 2 gives `("ab", True)`);
   - MLE samples use only observed transitions;
   - a fixed-seed 20,000-draw frequency check;
   - greedy decoding;
   - the distribution passed to the sampler equals `P`.

### 13.3 Learning and behaviour checks
Statistical model: the hand-computed checks above, and the explicit unseen-event and unseen-context behaviour in §10. Tolerances: exact for counts and MLE matrices on the tiny fixture; `rel = 1e-6` (float32 log-probabilities) for NLL values; `1e-5` across devices.

## 14. Checkpoint and serialization contract

- **Entry points:** `save(path, *, config=None)` writes a shared `"final"` checkpoint. `load(path, *, device="cpu", expected_vocab=None)` reads it with `weights_only=True`.
- **Why "final":** a fitted count model has no training trajectory, so it is neither "best" (selected on dev) nor "resume".
- **`state_dict()` contents:**
  - `format = "bigram/v1"`;
  - `vocab` (`Vocabulary.to_dict()`: characters and ordering);
  - `alpha`;
  - `counts` (a `[V, V]` int64 CPU clone, which never aliases the model).
- **Derived values are not saved:** probabilities and log-probabilities are recomputed on load.
- **Checkpoint extras:** `extra = {num_training_names, num_training_transitions}`. `config` defaults to the model config; a full resolved experiment config can be passed instead.
- **Validation in `load_state_dict` (strict only):**
  - exact key set and supported format;
  - a valid vocabulary equal to the model's;
  - an equal alpha;
  - counts that are a tensor of shape `[V, V]`, int64 and non-negative;
  - `check_fitted_counts`: at least one BOS transition, `C_00 = 0`, flow conservation, and every used token reachable from BOS. Unused vocabulary tokens are allowed.
- **Validation in `load`:** the checkpoint kind is `"final"`; the model class is `BigramLanguageModel`; the checkpoint's vocabulary metadata agrees with the state; `expected_vocab` matches if given.
- **Round-trip tests:** counts, probabilities, vocabulary order, alpha, evaluation results and seeded generation are identical after reload. Corrupted, incompatible and wrong-class checkpoints are rejected.

## 15. Generation or inference behaviour

- **Inference entry point:** `generate(num_samples, *, max_length, generator=None, temperature=1.0, top_k=None, top_p=None)`. One-step sampling: `sample_next(contexts, ...)`.
- **Prediction procedure:** ancestral sampling `x_t ~ P(· | x_{t-1})`. Unfinished samples are drawn in parallel, one draw per active sample per step.
- **Start/end-token behaviour:** every sample starts from BOS and stops when EOS is drawn. BOS and EOS never appear in `text`.
- **Sampling method and supported controls:** with the defaults, samples follow `P` exactly. Temperature, top-k and top-p (via the shared sampler) sample from a modified distribution. `temperature = 0` means greedy decoding.
- **Explicit RNG/generator handling:** pass a `torch.Generator`; the project convention is a CPU generator from `make_generator(derive_seed(seed, "generation"))`. `None` uses the global torch RNG.
- **Evaluation-mode requirements:** none. Generation never modifies counts or probabilities.
- **Maximum sequence length and stopping conditions:**
  - `max_length` is the maximum number of ordinary characters and is required.
  - A sample with `max_length` characters gets one more draw. EOS on that draw means a complete name with `finished = True`; anything else means the sample is returned truncated with `finished = False`.
  - At most `max_length + 1` draws are made per sample.
- **Invalid or degenerate distribution behaviour:**
  - An undefined context raises `UndefinedDistributionError`.
  - Such a row is unreachable from BOS in a fitted model: `C_ij > 0` implies `N_j ≥ 1`.
  - Invalid arguments raise `ValueError`.

## 16. Implementation scope and boundaries

Implemented in `models/statistical/bigram.py` only. There is no training loop, data splitting, evaluation pipeline, plotting or notebook.

**Necessary supporting-file changes and justification:**
1. `configs/models/bigram.yaml`: required by `default_config_paths("bigram")`.
2. `tests/models/statistical/test_bigram.py`: the model tests.
3. No shared utility was modified. Known friction: `save_checkpoint` / `restore_model` are annotated `nn.Module`, so the duck-typed call carries `# type: ignore`. A `Protocol` annotation in `utils/checkpoint.py` is proposed but not made.

## 17. Acceptance criteria

- [x] Exact architecture variant implemented as specified
- [x] Defining mechanisms explicit and inspectable
- [x] Public API and tensor contracts documented
- [x] Configuration validated
- [x] Initialization justified (none; closed form)
- [x] Numerical and edge-case behaviour handled
- [x] Shared utilities used appropriately
- [x] General tests pass (CPU)
- [x] Architecture-specific mathematical tests pass
- [x] Checkpoint/inference behaviour verified
- [x] CPU and available accelerator checks completed (CPU and MPS)
- [x] No unresolved material code-review issue known
- [x] Implementation run and reviewed locally by the project author
- [x] Completion approved by the project author

## 18. Open decisions

| Question | Options | Recommendation | Decision |
|---|---|---|---|
| α for controlled comparisons | fixed α = 0; fixed α = 1; select α on dev | select on dev from a small grid, report test once | Open. Not needed by the model itself. `training/bigram_study/` reports a descriptive dev sweep over α; selection belongs to a future controlled comparison of models |
| Handling MLE's +∞ dev NLL in reports | report as +∞ with the event count; exclude MLE from NLL tables | report +∞ and the count; use a smoothed variant for numeric comparison | Adopted: +∞ is reported together with the number of zero-probability events, as `evaluate` returns them; smoothed variants are used for numeric comparison |
| Type annotation of shared checkpoint `model` argument | keep `nn.Module` + `type: ignore`; `Protocol` | `Protocol` (one-line change in `utils/checkpoint.py`) | Closed: kept as is (`nn.Module` with `type: ignore`); the shared utilities are not changed for this |

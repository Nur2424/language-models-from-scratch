# Model 6 — Vanilla RNN: Integrated Implementation Contract (Stage 6)

## Status (current, v0.2)

1. **DRAFT for review. Not approved.** It authorises nothing: no implementation, probe construction, benchmark, measurement, training, pilot or experiment.
2. **Purpose:** one internally consistent implementation contract for Model 6, assembled from the three approved specifications. It defines:
   - component responsibilities and the module dependency order;
   - every public interface and signature;
   - configuration ownership;
   - the full run lifecycle;
   - the file map;
   - the integrated acceptance checklist;
   - the register of inconsistencies and their resolution.

   **It changes no scientific behaviour.**
3. **Authoritative sources** (verified on the Mac, 2026-11-14):

| Source | Version | SHA-256 | Decision |
|---|---|---|---|
| `models/recurrent/rnn_spec.md` (architecture) | v0.2 | `3961afdd12169da59c7a170f1e8783a88dda856d79ee753a7200f014372197f1` | D-44; §12.7 amended by D-45; §7.2 step 2 subject to the proposed amendment of §9.1 |
| `models/recurrent/rnn_training_spec.md` (training) | v0.3 | `61573eb3d8e5cc24f97905d8617f92bb0d061b34f63f0431b70bf6ab52114822` | D-46 |
| `models/recurrent/rnn_diagnostics_spec.md` (diagnostics) | v0.3 | `271d4bf8fbba269257742d336685bb213c2a04a1720887d5c22057efb83b48c0` | D-45 |
| `PROJECT_RESEARCH_RECORD.md` | — | `a65527b8e1101304fb0d89eece84dca37f8557a0d228b442d02e0c0e68ec5bdc` | D-38 – D-46 |

4. **Precedence:**
   1. recorded decisions;
   2. each approved specification within its domain (architecture → model behaviour; training → training procedure; diagnostics → diagnostic definitions);
   3. this document.

   This document overrides neither 1 nor 2.
5. **Labels:**
   - **[I]** a requirement taken from an approved source (cited);
   - **CLARIFICATION (C2 – C8)** — how approved parts connect, approved in direction by the user on 2026-11-14 and binding once this document is approved;
   - **AMENDMENT A6-1** — a change to approved content, which takes effect only by a recorded decision (§9.1);
   - **OPEN** — an unresolved item requiring a decision (§10). Never assumed.
6. **Test split:** protected. Nothing here builds or evaluates test data.
7. **Proposed location:** `models/recurrent/rnn_integration_spec.md` (not created).

**Version:** 0.2 (2026-11-14)

### Amendment log

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-11-14 | Initial Stage 6 draft |
| v0.2 | 2026-11-14 | User decisions applied. A6-1 refined: the gate uses seeds 1000 – 1009 **exclusively**; pilot seeds stay separate; the gate must be completed and approved before any pilot; formal record text proposed (§9.1). C2 – C8 accepted and specified exactly: canonical parameter-hash bytes (§4.3); probe validation (§4.4); diagnostic-failure handling that never reports success after a failure (§4.6); aborted runs (§3.4); fixed device enforcement (§5.2). Full lifecycle checked (§3). Execution points table for G1 – G3 and C1 against D-45/D-46 (§4.2). Pilot scope stated (§3.3), with one **OPEN** item where the user's refinement and D-46 differ (U1). Mandatory/optional switches explicit (§4.7). Complete signatures (§6). Checklist and X1 – X6 refined (§8) |

---

## 1. Components and responsibilities

| Component | File | Owns | Never does |
|---|---|---|---|
| Model | `models/recurrent/rnn.py` | Parameters; Haar initialisation; `_step`, `_logits`, `_recurrence`, `forward`, `generate`, `from_checkpoint`; input validation (arch §5 – §8, §14, §15) | Training, evaluation loops, diagnostics, run files |
| Diagnostics | `models/recurrent/rnn_diagnostics.py` | Shared definitions (`LENGTH_STRATA`, `param_sha256`, `float64_copy`); probe; H1, G1 – G3, C1; the arch §7.2 initial-statistics procedure; `RNNDiagnosticsPlan`; post-hoc spectral measurement; diagnostic files (diag §3 – §11) | Modifying training state; using any RNG (except the probe construction draw); importing `rnn_training.py` |
| Training | `models/recurrent/rnn_training.py` | `RNNTrainingConfig`; C; step loop; evaluation; O1 and O2; selection; checkpoints; resume; failure handling; research-run configuration validation (training §2 – §13; C8) | Defining diagnostics; reading diagnostic results; calling `_step`, `_logits`, `_recurrence` |
| Configuration | `configs/models/rnn.yaml` (+ unchanged `configs/base.yaml`) | Architecture parameters; shared infrastructure | Training or diagnostic values |
| Orchestration | future `training/rnn_study/` (**design is the user's decision**; U2) | Building data, models, plans and configs; seeds; thread settings; Git guard; frozen conditions; the gate, pilots and research runs | Defining training or diagnostic semantics |

### 1.1 Dependency order (CLARIFICATION C4)

```
utils/*  ←  rnn.py  ←  rnn_diagnostics.py  ←  rnn_training.py  ←  orchestration (future)
```

- `rnn_diagnostics.py` imports `rnn.py` and `utils/*` only.
- `rnn_training.py` imports `rnn.py`, `rnn_diagnostics.py` (`LENGTH_STRATA`, `param_sha256`, `RNNDiagnosticsPlan`) and `utils/*`.
- Shared definitions therefore exist once, in the lower module. That module's responsibility (diag §3.1, §5.3; training §8.4, §10.4) is consistent with the approved file list (arch §16). Enforced by X6.

---

## 2. Architecture integration

Every need maps to an approved interface; **no new public model API**.

| Need | Consumer | Interface (arch section) |
|---|---|---|
| Logits for loss and evaluation | training | `forward(inputs) -> [B, T, 27]`; padded logits from the frozen state (§5, §4.1) |
| Batches of 8 names; any T_b; B = 0 allowed | training | §5.1 |
| States for H1, G1, G2, G3, δ̂ | diagnostics | `_recurrence(inputs).states`, `.valid` (§6.3, §8) |
| Pre-activations for the §7.2 procedure | diagnostics | `_recurrence(..., return_preactivations=True)` (§6.3) |
| p_t, r_t | diagnostics | `_logits(states)` (§6.3) |
| BOS-restart (C1) | diagnostics | `_step(h_prev, x̂_t, m_t)` with all-true validity, then `_logits` (§6.3, §12.6) |
| Parameter values | diagnostics; training O2; hash | named parameters `input_lookup` (U), `recurrent_weight` (W), `hidden_bias` (b), `output_weight` (V), `output_bias` (c) (§4.3) |
| Construction | orchestration; gate | `RNNLanguageModel(vocab, config, *, generator=make_generator(derive_seed(s, "init")))` (§5.2, §7.1) |
| Loading the best checkpoint | post-hoc | `RNNLanguageModel.from_checkpoint(path, *, expected_vocab, map_location="cpu")` (§14) |
| Autograd on the float64 copy | G3 run-time check | Graph kept when gradients are enabled (§6.3) |

Diagnostics may call the private `_step`, `_logits` and `_recurrence` (M6-IMP2). Training may not (training §13).

---

## 3. Lifecycle

### 3.1 Overview, in gate order (training §15; D-46)

| # | Step | Uses | Produces | Gate |
|---|---|---|---|---|
| L1 | Implementation acceptance | §8 | Decision | — |
| L2 | **Probe construction** | `build_probe` (§4.4) on the training split | `probe.json` + SHA-256, N_ℓ, F₈ count → recorded | Authorisation |
| L3 | Thread benchmark (timing only; parameters discarded) | `train_rnn`, seed 999, η = 0.1, one epoch, no plan, threads 1 / 2 / 4 / 8 | Timings → thread decision | Authorisation; decision |
| L4 | Diagnostic timing (M6-DGN13) | All diagnostics on the untrained model, seed 1000, chosen threads | Timings → recorded | Authorisation |
| L5 | **Initialisation gate** | `initial_statistics` (§4.5) with seeds **1000 – 1009 exclusively** (A6-1) | Report → **gate decision** on g, s² (retain or change) | Authorisation; **decision before any pilot** |
| L6 | Pilot P (and extension X if required) | `train_rnn`, seeds 100 – 102, `diagnostics=None` | η decision; E decision | Authorisation; decisions |
| L7 | Research (final) runs | `train_rnn` with a full `RNNDiagnosticsPlan` | Training and diagnostic records | M6-EXP1, EXP2, failed-seed policy (Later) |
| L8 | Post-hoc best-epoch spectral | `posthoc_best_spectral` | Diagnostic point `posthoc_best` | After each L7 run, if best ∉ {1, E} |

### 3.2 One `train_rnn` call (training §5, §8, §10, §12)

```
validate inputs; compute N, M, S, r, C; refuse on mismatch with the recorded protocol values
if diagnostics: the plan was validated at construction (§4.4); verify plan.max_epochs == E and plan.steps_per_epoch == S
epoch 0: evaluate train/dev → O2(k=0) incl. param_sha256 → history row k=0 → [plan.at_start(model)]          σ=0
for e in 0 … E−1:
  each step: training §5 items 1–12 → [plan.after_step(model, sigma=σ, epoch=e)]
  epoch end: param check → evaluate → Category D test → O2 + param_sha256
             → [plan.at_epoch_end(model, sigma=σ, completed_epochs=e+1)]                                    before writes
             → model_best.pt (strict <, k ≥ 1) → history row → checkpoint_last.pt (commit)
             → if D: model_final.pt; status failed_diverged; stop
after E: model_final.pt; status completed
return RNNTrainingResult(status, best_epoch, best_dev_nll, history, C, diagnostics_status)
```

### 3.3 Pilot scope (training §7.3, D-46; one OPEN item)

- **Approved (D-46):** pilots and extension runs use `diagnostics=None`, so no research diagnostics run. They record O1 at every step and O2 at every epoch, **including the epoch-0 row** (k = 0: full train and dev NLL, parameter norms, all 128 singular values of W, ‖WᵀW/g² − I‖_F, `param_sha256`; training §8.3 – §8.4).
- **Reading adopted here:** the user's requirement that "each pilot run must still record its own initial diagnostic point" is satisfied by this **epoch-0 O2 record**, which every pilot run writes. This is consistent with D-46.
- **OPEN U1:** if the requirement means the Stage 5 P₀ diagnostic point (H1, G1-F, G1-σ, ω, δ̂ on the probe), pilots would need a plan restricted to P₀. That **changes D-46 §7.3** and requires an explicit amendment. It is not assumed here (§10).

### 3.4 Aborted runs (CLARIFICATION C7)

- **Definition:** a run is **aborted before first commit** if it ends, by crash, kill or exception, before `checkpoint_last.pt` exists.
- **Rule:**
  - its directory is preserved unchanged as evidence;
  - `train_rnn(..., resume=True)` on it raises `ResumeError` ("no committed checkpoint"), as does `resume=False`, because the directory holds state (precedent `mlp.py`);
  - any retry is a **new run directory from scratch**, under explicit authorisation with a written reason (D-18).
- **Status recording:** where the process survives (an exception), `RunManager.track()` records `failed`. Where it does not survive (a kill), the run's metadata remains `running`; orchestration must classify it as aborted (U2).

### 3.5 Resume (training §10.6; diag §5.2)

- Only from `checkpoint_last.pt` (K completed epochs, σ_K steps).
- The compatibility check includes the diagnostics presence and probe SHA-256.
- Truncation:
  1. history rows k ≤ K;
  2. step records ≤ σ_K;
  3. `plan.truncate(sigma=σ_K, completed_epochs=K)`.
- **P₀ is never re-run on resume:** K ≥ 1 always, and the P₀ record has σ = 0 ≤ σ_K, so it is kept.

---

## 4. Diagnostics integration

### 4.1 Plan interface (CLARIFICATION C5, C6)

```python
class RNNDiagnosticsPlan:
    def __init__(self, *, probe: Probe, train: NextTokenSequences, train_names: Sequence[str],
                 dev: NextTokenSequences, steps_per_epoch: int, max_epochs: int,
                 recurrent_gain: float, out_dir: Path,
                 optional: OptionalDiagnostics = OptionalDiagnostics()) -> None
    def at_start(self, model: RNNLanguageModel) -> None
    def after_step(self, model: RNNLanguageModel, *, sigma: int, epoch: int) -> None
    def at_epoch_end(self, model: RNNLanguageModel, *, sigma: int, completed_epochs: int) -> None
    def truncate(self, *, sigma: int, completed_epochs: int) -> None
    @property
    def status(self) -> DiagnosticsStatus   # "complete" | "incomplete" (any invalid mandatory point)
```

The constructor:
- validates the probe against the data (§4.4) and raises `ProbeMismatchError` before training;
- resolves the point sets (§4.2);
- writes `plan.json`.

### 4.2 Execution points (D-45 diag §5.1; D-46 training §5)

| Point | Call (training position) | σ | H1 | G1-F, ω, δ̂ | G1-σ | G2 | G3 (+ halves) | C1 |
|---|---|---|---|---|---|---|---|---|
| P₀ | `at_start`: after the epoch-0 row, before step 1 | 0 | ✓ | ✓ | ✓ | — (undefined) | — (G ≡ 0) | — |
| P_early | `after_step`: after the update, post-step check, running NLL and O1 | 2^j < S (1 … 512 at S = 802) | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| P_epoch | `at_epoch_end`: after evaluation, D test and O2; before checkpoint writes | cumulative at the end of epoch k = 1 … E | ✓ | ✓ | only k = 1 and k = E | ✓ | ✓ | ✓ |
| Post-hoc best | `posthoc_best_spectral`, after training | — | — | — | ✓ if best ∉ {1, E} | — | — | — |

- `after_step` is called at every step and acts only for σ ∈ P_early (strict < S, so σ = S is handled only by `at_epoch_end`).
- G3 and C1 always run at the same points (arch §12.8).
- Optional diagnostics are added only by the switches of §4.7.

### 4.3 Canonical parameter-hash bytes (CLARIFICATION C3; definition of training §10.4 = diag §5.3)

```python
PARAM_ORDER = ("input_lookup", "recurrent_weight", "hidden_bias", "output_weight", "output_bias")  # U, W, b, V, c

def param_sha256(model: RNNLanguageModel) -> str
```

Exact procedure:
1. For each name in `PARAM_ORDER`, take the parameter tensor `p` of the model (the training model, never a float64 copy).
2. **Shape:** require `p.shape` to equal the architecture shape: U `(27, H)`, W `(H, H)`, b `(H,)`, V `(27, H)`, c `(27,)`. Otherwise raise `ValueError`. Shapes are not hashed; they are fixed by the validated `RNNConfig`.
3. **Values:** `a = p.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()`. A float32 parameter is copied exactly. A float64 parameter (tests only; research runs are float32, C8) is rounded to nearest-even, which is deterministic. The hash then identifies the rounded values, not the float64 state.
4. **Bytes:** `a.astype("<f4", copy=False).tobytes(order="C")`: little-endian IEEE-754 binary32, row-major.
5. **Digest:** SHA-256 over the concatenation of the five byte strings in `PARAM_ORDER`. No separators, no names, no shapes, exactly as the approved text ("float32 bytes, C-contiguous, of U, W, b, V, c in that order"). Returned as 64 lowercase hex digits.

**Determinism [T]:** the bytes are a pure function of the parameter values. Signed zeros are distinguished, which bitwise comparison requires. Hashes are taken only after the finiteness checks, so NaN payloads cannot occur.

**Used in:**
- every history row;
- every checkpoint's `extra["param_sha256"]`;
- every diagnostic point, before and after;
- the post-hoc verification.

### 4.4 Probe identity (CLARIFICATION C5)

```python
@dataclass(frozen=True)
class Probe: rows: tuple[int, ...]; names: tuple[str, ...]; strata: tuple[int, ...]; halves: tuple[str, ...]
             weights_by_stratum: tuple[float, ...]; population_by_stratum: tuple[int, ...]
             split_info: Mapping[str, Any]; sha256: str
def build_probe(train: NextTokenSequences, train_names: Sequence[str], split_info: Mapping[str, Any],
                out_path: Path) -> Probe                       # L2 only; diag §3.1
def load_probe(path: Path) -> Probe                            # verifies schema "m6_probe/1" and its SHA-256
def validate_probe(probe: Probe, train: NextTokenSequences, train_names: Sequence[str],
                   split_info: Mapping[str, Any]) -> None      # raises ProbeMismatchError
```

`validate_probe` rejects, before any training or gate computation, when any of the following fail:
1. the SHA-256 recomputed from the canonical JSON equals the stored value;
2. `split_info` (split seed, duplicate policy, fractions, N, M) equals the current data;
3. **for every probe row i, `train_names[i]` equals the stored name;**
4. the stratum of every row, recomputed from `train.lengths[i]` with `LENGTH_STRATA`, equals the stored stratum;
5. N_ℓ recomputed from the data equals the stored population counts.

It is called by the plan constructor and by `initial_statistics`.

### 4.5 Diagnostic function signatures (definitions in diag §4; procedure in arch §7.2)

```python
LENGTH_STRATA: tuple[tuple[int, int], ...] = ((2, 5), (6, 7), (8, 9), (10, 16))
def float64_copy(model: RNNLanguageModel) -> RNNLanguageModel        # deepcopy; clear all hooks; .grad=None; cpu float64
def h1_statistics(copy: RNNLanguageModel, probe: Probe, train: NextTokenSequences) -> H1Result
def g1_products(copy, probe, train, *, spectral: bool) -> G1Result  # φ, σ (if spectral), δ̂, ω(Δ)
def g2_attenuation(copy, probe, train) -> G2Result                  # stable g; exact-zero exclusion
def g3_decomposition(copy, probe, train) -> G3Result                # full probe + halves A/B; run-time checks
def c1_restart(copy, dev: NextTokenSequences) -> C1Result           # per-target ΔNLL; F8, F7; clustered SEs
def mean_field_table(gain: float, input_variance: float, *, max_position: int = 16) -> list[MeanFieldRow]
def initial_statistics(vocab: Vocabulary, config: RNNConfig, probe: Probe, train: NextTokenSequences,
                       train_names: Sequence[str], split_info: Mapping[str, Any],
                       seeds: Sequence[int] = tuple(range(1000, 1010)), out_dir: Path = ...) -> InitialStatisticsReport
def posthoc_best_spectral(run_dir: Path, probe: Probe, train: NextTokenSequences) -> None
def positional_data_bound(train: NextTokenSequences, *, max_d: int = 4) -> list[BoundRow]   # OPTIONAL (M6-DGN19)
```

- **`initial_statistics`** (A6-1) builds each model with `RNNLanguageModel(vocab, config, generator=make_generator(derive_seed(s, "init")))` for s in 1000 – 1009, with no optimiser step, and produces the arch §7.2 report. It refuses any other seed set unless a decision amends A6-1. Pilot seeds 100 – 102 are not accepted.
- **`posthoc_best_spectral`** opens only the diagnostic files for writing (training §10.7):
  1. load `model_best.pt` (`weights_only=True`);
  2. require `extra["param_sha256"]` == `param_sha256(loaded)` == the hash of that epoch's epoch-end point;
  3. compute G1-σ on `float64_copy`;
  4. append a `posthoc_best` point.

### 4.6 Diagnostic failures (CLARIFICATION C6; policy M6-DGN17, D-45)

- **Inside a plan method:** any `Exception` raised by a diagnostic computation, and any failed run-time consistency check (diag §10.2), is handled as follows:
  1. The point is written with `valid: false`, the diagnostic name, the exception type and message, the traceback, and the failing check's values.
  2. The plan's `status` becomes `incomplete` if the point is mandatory (§4.7).
  3. A warning is logged.
  4. Control returns to training. **Training continues**, because diagnostics cannot affect it (diag §11).
- **Never reported as success:**
  - `RNNTrainingResult.diagnostics_status`, the run's `config.json`/metadata entry `diagnostics_status`, and `o3.json` all carry `incomplete` with the list of invalid points;
  - a run with status `incomplete` **cannot be used for research conclusions** until a recorded human review states the cause (D-45).
- **Recording failure:** if writing the failure record itself fails (an I/O error), the exception **propagates** and the run fails. The policy cannot be honoured without the record.
- **Escapes:** an exception that escapes a plan method is a plan defect and fails the run. `train_rnn` does not catch it.

### 4.7 Mandatory and optional (diag §13; D-45)

```python
@dataclass(frozen=True)
class OptionalDiagnostics:
    h1_on_dev: bool = False                 # diag §4.1, at E_sel
    spectral_at_powers_of_two: bool = False # diag §5.1
```

- **Mandatory:** always run at the points of §4.2; cannot be switched off for a research run. They are H1 (probe), G1-F, ω, δ̂, G1-σ at E_sel, G2, G3 with halves, C1 including per-target values, O2 reporting, O3, the run-time checks and the isolation records.
- **Optional:** off by default. Enabling one is recorded in `plan.json`. The positional data bound is a separate one-off function, not part of runs.

### 4.8 Isolation (diag §11)

Requirements and tests D:I1 – D:I8 are unchanged. The O2 computations in `train_rnn` follow the same rules (read-only, no RNG, no graph) and are covered by D:I1 = T:T24 = A:T5.

---

## 5. Configuration integration

### 5.1 Ownership

| Layer | Location | Contents | Owner | Recorded as |
|---|---|---|---|---|
| Shared | `configs/base.yaml` (unchanged) | data, split, seed, device, dtype, logging, checkpoint, generation, shared diagnostics schedule | project | `config["experiment"]` |
| Architecture | `configs/models/rnn.yaml` | `model.name: rnn`; `hidden_dim: 128`, `recurrent_gain: 1.0`, `input_init_variance: 0.5` (arch §9.1) | `rnn.py` (`RNNConfig`) | `config["model"]` |
| Training | `RNNTrainingConfig` in code (no YAML; precedent) | lr, max_epochs, batch_size = 32, eval_batch_size = 1,024 | `rnn_training.py` | `config["training"]`, `config["derived"]` |
| Diagnostics | `RNNDiagnosticsPlan` + `OptionalDiagnostics` in code | probe path and hash, S, E, gain, out_dir, optional switches | `rnn_diagnostics.py` | `config["diagnostics"]`, `plan.json` |
| Run setup | orchestration (U2) | seeds, threads, run directories, Git guard | user's design | run metadata |

### 5.2 Research-run validation (CLARIFICATION C8)

```python
def validate_research_run(experiment: ExperimentConfig, model: RNNLanguageModel,
                          training: RNNTrainingConfig) -> None    # in rnn_training.py; raises ResearchConfigError
```

It refuses unless **all** of the following hold:

| Setting | Required |
|---|---|
| `experiment.device.device` | exactly `"cpu"`. **`"auto"` is rejected** explicitly, as is any other value |
| `experiment.device.dtype` | `"float32"` |
| model parameters | device `cpu`, dtype `torch.float32` |
| `experiment.diagnostics.enabled` (shared `DiagnosticsSchedule`) | `false` |
| `experiment.checkpoint.resume_every_n_steps` | `null` |
| `torch.get_num_threads()`, `torch.get_num_interop_threads()` | equal to the decided thread counts (passed by orchestration) |
| `training.batch_size`, `training.eval_batch_size` | 32, 1,024 |

- Called by orchestration before the gate's model construction, before the benchmark (thread counts excepted, since they are varied there), and before every pilot and research run.
- `train_rnn` itself stays usable in tests with float64 on CPU (training §14 T10, T21). The research restriction lives in this function, tested by X2.

---

## 6. File map and public interfaces (definitive)

| File | Public interface | Specification |
|---|---|---|
| `models/recurrent/rnn.py` | `RNNConfig(hidden_dim=128, recurrent_gain=1.0, input_init_variance=0.5)`; `RNNSample(text, finished)`; `RecurrenceOutput(states, valid, preactivations)`; `IncompatibleCheckpointError`; `RNNLanguageModel(vocab, config=RNNConfig(), *, generator)` with `forward(inputs)`, `generate(num_samples, *, max_length, generator=None, temperature=1.0, top_k=None, top_p=None)`, `from_checkpoint(path, *, expected_vocab=None, map_location="cpu")`, property `device`; private `_step(h_prev, x_hat_t, m_t) -> (h_t, a_t)`, `_logits(h)`, `_recurrence(inputs, *, return_preactivations=False)` | arch §5 – §9, §14, §15 |
| `models/recurrent/rnn_diagnostics.py` | §4.1, §4.3 – §4.7: `LENGTH_STRATA`, `PARAM_ORDER`, `param_sha256`, `float64_copy`, `Probe`, `build_probe`, `load_probe`, `validate_probe`, `ProbeMismatchError`, `h1_statistics`, `g1_products`, `g2_attenuation`, `g3_decomposition`, `c1_restart`, `mean_field_table`, `initial_statistics`, `OptionalDiagnostics`, `RNNDiagnosticsPlan`, `DiagnosticsStatus`, `posthoc_best_spectral`, `positional_data_bound`; result dataclasses; writers for schema `m6_diagnostics/3` | diag §3 – §11; arch §7.2 |
| `models/recurrent/rnn_training.py` | `RNNTrainingConfig(lr, max_epochs, batch_size=32, eval_batch_size=1024)`; `RNNTrainingResult(status, best_epoch, best_dev_nll, history, loss_normalizer, diagnostics_status)`; `loss_normalizer(training, train) -> float`; `evaluate_nll(model, sequences, *, batch_size) -> NLLAccumulator`; `evaluate_position_nll(model, sequences, *, batch_size) -> PositionNLL`; `train_rnn(model, train, dev, *, training, seed, run=None, resume=False, config=None, diagnostics=None) -> RNNTrainingResult`; `validate_research_run(...)`; `ResumeError`, `ResearchConfigError` | training §2 – §13; §3, §5.2 here |
| `configs/models/rnn.yaml` | §5.1 | arch §9.1 |
| `tests/models/recurrent/conftest.py` | Git guard for every Model 6 test (as `tests/training/wavenet_study/conftest.py`) | training §10.8; D-15 |
| `tests/models/recurrent/rnn_reference.py` | Independent reference (imports only `torch`) | arch §13.2 |
| `tests/models/recurrent/test_rnn_reference.py` | Reference self-tests; A:M1 – A:M5 | arch §13.2 |
| `tests/models/recurrent/test_rnn.py` | A:V1 – V5, A:R1 – R14, A:I1 – I5, A:8a – 8d, A:K1 – K2 | arch §13.2 |
| `tests/models/recurrent/test_rnn_diagnostics.py` | A:M6 – M11, D:D1 – D11, D:I1 – I8, X5, X6 | arch §13.2; diag §10 – §11 |
| `tests/models/recurrent/test_rnn_training.py` | A:T1 – T8, T:T9 – T24, X1 – X4 | arch §13.2; training §14 |

- **Untouched:** `gru.py`, `lstm.py`, shared utilities, `base.yaml`, existing tests.
- **Not designed here:** orchestration (U2).

---

## 7. Values fixed later by approved procedures

| Value | Procedure | Lifecycle step |
|---|---|---|
| Probe SHA-256, N_ℓ, F₈ count | diag §3.1 | L2 |
| Thread counts | training §11 | L3 |
| g, s² | arch §7.2 as amended by A6-1; training §9 | L5 |
| η | training §7.3 | L6 |
| E | training §7.4 | L6 |
| Final seeds; failed-seed policy | M6-EXP1, EXP2 (Later) | before L7 |

---

## 8. Integrated acceptance checklist

Labels (CLARIFICATION C2), used consistently in this document and in test names: **A:** architecture (arch §13.2), **T:** training (training §14), **D:** diagnostics (diag §10 – §11), **X:** integration (below). Duplicates are listed once with all labels.

| # | Area | Criterion | Tests | Tolerance |
|---|---|---|---|---|
| 1 | Mathematical correctness | Model equals the independent reference (states, pre-activations, logits) across compositions and padding | A:R1, A:R10 | arch §13.1 A |
| 2 | | Analytic Jacobians, G2 vectors, BPTT formulas, G3 identity against autograd; finite differences | A:M1 – M5, D:D5 | A; FD |
| 3 | | Static loss weighting; C = B·M/N; final batch | T:T9 – T11 | A / exact |
| 4 | Architecture behaviour | Construction, validation, configuration, shapes, float64 support | A:V1 – V5, A:R11, A:R12 | exact |
| 5 | | Causality, padding and batch invariance, frozen state, padded logits, sharing, train/eval identity | A:R2 – R9, A:R13 | A, B, E |
| 6 | | Initialisation (ln 27, M6-E4, Haar, draw order, U variance) | A:I1 – I5 | as stated |
| 7 | | Generation | A:8a – 8d | A / B / exact |
| 8 | | Checkpoint round trip, incompatibility | A:K1, A:K2 | bitwise / error |
| 9 | | Prohibited-module scan | A:R14 | none found |
| 10 | Training correctness | Step order; Category A at each check; Category D ends the run; selection (k = 0 excluded; earliest tie) | A:T3, A:T4, T:T12 – T14 | exact / bitwise |
| 11 | | Evaluation weighting, order and chunk independence; per-position totals | A:T6, T:T19 | A / B |
| 12 | | Trimming invariance; step-record totals | A:T7, T:T20, T:T21 | A / exact |
| 13 | | Overfitting fixture (debugging check) | A:T1 | test-local |
| 14 | Numerical stability | Non-finite guards; finite outputs; large-gain fixture | A:T4, T:T12, arch §10 | — |
| 15 | Diagnostic correctness | Probe determinism, strata, weights, halves; weighted quantiles; plan resolution | D:D1 – D3, D:D11 | exact |
| 16 | | G2 rule; δ̂ = φ̂ at θ₀; C1 sets and clustered SEs; H1 | D:D6 – D8, A:M7 – M11 | A / exact |
| 17 | | Run-time consistency checks detect injected faults | D:D10 | — |
| 18 | | Post-hoc spectral: hash check; training records byte-identical | T:T23 | exact |
| 19 | Reproducibility | CPU determinism; global RNG untouched; no Git started | A:T2, A:T8, T:T22 | bitwise / exact |
| 20 | | Storage round trip; hash alignment across history, checkpoints, diagnostics; **canonical hash bytes on a hand fixture** (§4.3) | D:D4, D:D9, T:T18, X1 | exact |
| 21 | Resume equivalence | Interrupted and resumed run equals uninterrupted, including diagnostic files | T:T15 – T17, D:I7 | bitwise |
| 22 | Diagnostic isolation | Training identical with and without a full plan | A:T5 = T:T24 = D:I1 | bitwise |
| 23 | | Per-call invariants; copy independence; hook sanitisation; graph isolation; failure isolation; no stray files | D:I2 – I6, D:I8, D:D10 | exact / bitwise |
| 24 | Integration | **X1** (refined): end-to-end fixture run with a full plan. The hook call sequence and σ values equal §4.2 exactly; P₀ precedes step 1; epoch-end points precede `checkpoint_last.pt` (file timestamps and order log); all files validate against schema `m6_diagnostics/3`; `param_sha256` on a hand-constructed parameter set equals a precomputed digest | X1 | exact |
| 25 | | **X2** (refined): `validate_research_run` rejects `device: auto`, any non-cpu device, dtype ≠ float32, shared diagnostics enabled, step-based resume, wrong batch sizes, wrong thread counts; accepts the approved configuration; `rnn.yaml` loads to the approved defaults | X2, A:V5 | error / exact |
| 26 | | **X3** (refined, C6): an exception injected in each mandatory diagnostic produces a `valid: false` point with traceback, `status = incomplete` in the result, metadata and `o3.json`, and training bit-identical to a run without diagnostics; an injected I/O failure while writing the failure record propagates and fails the run; an escaping exception fails the run | X3 | exact |
| 27 | | **X4** (C7): a crash before the first commit leaves the directory unchanged; `resume=True` raises `ResumeError`; `resume=False` refuses the directory; a fresh directory runs normally | X4 | exact |
| 28 | | **X5** (C5): `validate_probe` rejects each mismatch type (hash, split info, a changed name, a changed length, a changed N_ℓ), before any training step or gate computation; `initial_statistics` refuses a seed set other than 1000 – 1009 | X5 | error |
| 29 | | **X6** (C4): source scan confirms the dependency order (§1.1); `rnn_training.py` never calls `_step`, `_logits` or `_recurrence` | X6 | none violated |
| 30 | Process | Architecture acceptance checklist (arch §17) ticked explicitly; the user has run and reviewed the tests locally; the user approves | — | — |

---

## 9. Conflict register and resolutions

### 9.1 C1 — initialisation gate seeds (AMENDMENT A6-1)

- **Conflict:** arch §7.2 step 2 constructs models "for the primary training seed and for a predeclared set of initialisation-only seeds". The gate (training §9, D-46) must pass before any pilot, while final seeds are a Later decision. No "primary training seed" exists at gate time.
- **Resolution (user direction, 2026-11-14):**
  - the gate uses seeds **1000 – 1009 exclusively** (D-45, M6-DGN10);
  - pilot seeds 100 – 102 stay separate and are never part of the gate;
  - the gate is completed **and approved** before any pilot begins;
  - each pilot run records its own initial state in its epoch-0 O2 record (§3.3; see U1);
  - each research run's P₀ point records the initial state for its own seed.
- **Formal amendment, proposed record text** (to be appended as the next decision; **not written** by this draft):

> **D-47 (proposed)** | 2026-11-14 | **Amendment A6-1 to `models/recurrent/rnn_spec.md` §7.2 (initial-statistics procedure), approved under D-44.** Step 2 ("Constructed for the primary training seed and for a predeclared set of initialisation-only seeds") is replaced, for all purposes, by: "The procedure constructs models for the initialisation-only seeds **1000 – 1009 exclusively** (D-45, M6-DGN10), each with `make_generator(derive_seed(s, "init"))`, and takes no optimiser step. Pilot seeds (100 – 102) and research-run seeds are not part of the procedure. The procedure, its report and the gate decision M6-INI7 (`rnn_training_spec.md` §9) are completed and approved **before any pilot begins**. The initial state of each pilot run is recorded by that run's epoch-0 record (`rnn_training_spec.md` §8.3 – §8.4); the initial state of each research run is recorded by its P₀ diagnostic point (`rnn_diagnostics_spec.md` §5.1)." **Authority:** the approved specification text (SHA-256 `3961afdd…97f1`) is preserved unchanged as the historical approved version; where its §7.2 step 2 differs, this decision governs. No other part of §7.2 changes (fixed values, measurements, predictions, reporting, change rule). | ACCEPTED (on user approval) | No "primary training seed" exists before the gate, which must precede pilots (D-46) | Gate defined unambiguously; `rnn_spec.md` not edited; any editorial pointer in that file needs a separate authorised step | `rnn_spec.md` §7.2; `rnn_training_spec.md` §9; `rnn_diagnostics_spec.md` §10.3; `rnn_integration_spec.md` §9.1 |

### 9.2 C2 – C8 (CLARIFICATIONS, accepted in direction by the user)

| # | Clarification | Where specified |
|---|---|---|
| C2 | Source-prefixed test labels A:, T:, D:, X: everywhere | §6, §8 |
| C3 | Canonical bytes of `param_sha256` | §4.3 |
| C4 | Shared definitions in `rnn_diagnostics.py`; dependency order | §1.1, §4.5 |
| C5 | Probe identity validated against the training data; mismatches rejected before training or gate | §4.4 |
| C6 | Diagnostic failures recorded, flagged `incomplete`, never reported as success; recording failure fails the run | §4.6 |
| C7 | Abort before the first commit; retry only in a new directory | §3.4 |
| C8 | Research runs reject `device: auto`; fixed CPU float32 configuration enforced | §5.2 |

### 9.3 Items needing no change

| # | Finding | Type |
|---|---|---|
| C9 | Arch §12.9 cost estimates superseded by diag §7.2 (non-normative) | Documentary |
| C10 | Arch §12 / §18.4 "PENDING" labels; §12.7 spike multiple | Resolved by D-45, D-46 |
| C11 | "DRAFT" headers of the training and diagnostics specifications; diagnostics header says Stage 4 not approved | Resolved by D-45, D-46 (hash integrity) |
| C12 | Overlapping tests across the three specifications | Merged in §8 |
| C13 | Arch §14 "resume semantics PENDING" | Resolved by D-46 |
| C14 | Existing Model 1–5 tests start Git without a guard | Documentary; outside Model 6 |

**Consistency check of execution points:** §4.2 matches D-45 (diag §5.1: P₀, P_early, P_epoch, E_sel = {0, end of epoch 1, best, E}) and D-46 (training §5 hook positions; §7.3 no diagnostics in pilots; §10.6 truncation; §10.7 post-hoc). No contradiction found.

---

## 10. Open items requiring the user's decision

| # | Item | Why open | Options |
|---|---|---|---|
| **U1** | Meaning of "each pilot run must still record its own initial diagnostic point" | D-46 §7.3 approves **no research diagnostics** in pilots. The epoch-0 O2 record satisfies the requirement without amendment. Running the Stage 5 P₀ point in pilots would amend D-46 | (a) epoch-0 O2 record (adopted in this draft; no amendment); (b) P₀ point in pilots (requires an amendment of D-46 §7.3 and a plan restricted to P₀; adds 1 P₀ evaluation per pilot run) |
| U2 | Orchestration package design (`training/rnn_study/`): frozen conditions, run directories, classification of killed runs as aborted, thread setup, gate and pilot drivers | User's decision (not designed here) | — |
| U3 | Recording A6-1 as D-47 | Requires the user's approval | Text in §9.1 |
| U4 | M6-EXP1, M6-EXP2, failed-seed policy | Later, before L7 | — |
| U5 | Git guard for the existing Model 1–5 tests (C14) | Outside Model 6 | Optional decision |
| U6 | Implementation authorisation and its staging | Separate decision | Recommended order: `rnn.py` → `rnn_diagnostics.py` → `rnn_training.py`, each with its tests |

---

## 11. Readiness

- **Specification consistency:** after A6-1 and C2 – C8, no contradiction remains between the three approved specifications. The one remaining interpretive question (U1) is listed, not assumed.
- **Implementability:** every need maps to an approved model interface. Every public function, its owner file and its signature are fixed (§4 – §6). Configuration ownership is explicit (§5). Every lifecycle step has a defined input, output and gate (§3).
- **Before implementation:** approval of this document, recording of A6-1 (D-47), a decision on U1, and a separate implementation authorisation. Orchestration (U2) is needed before the first execution step (L2 probe construction), not before implementation of the three modules.

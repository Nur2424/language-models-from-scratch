# Model 6 — Vanilla RNN: Implementation Plan (Stage 7)

## Status (current, v0.2)

1. **DRAFT for review. Not approved.** This plan authorises nothing. It does not authorise implementation, source changes, test runs, probe construction, benchmarks, the initialisation gate, training, pilots or experiments. Each implementation milestone needs its own explicit authorisation (P7-1). Each research execution step needs its own authorisation (§3).
2. **Purpose:** how the approved Model 6 contract will be built, verified and run:
   - milestone sequence;
   - exact file scope per milestone;
   - import boundaries;
   - test and independent mathematical verification strategy;
   - orchestration;
   - resume, interruption and partial-record recovery;
   - acceptance criteria;
   - risks;
   - the separation of implementation authorisation from research-run authorisation.

   **It changes nothing that is approved.** Rules the approved documents do not contain are labelled **[P]**, a proposal binding once this plan is approved, or listed as unresolved in §13.
3. **Authoritative sources** (verified on the Mac, 2026-11-14):

| Source | SHA-256 | Decision |
|---|---|---|
| `models/recurrent/rnn_spec.md` v0.2 | `3961afdd12169da59c7a170f1e8783a88dda856d79ee753a7200f014372197f1` | D-44; §12.7 amended by D-45; §7.2 step 2 amended by D-47 |
| `models/recurrent/rnn_training_spec.md` v0.3 | `61573eb3d8e5cc24f97905d8617f92bb0d061b34f63f0431b70bf6ab52114822` | D-46 |
| `models/recurrent/rnn_diagnostics_spec.md` v0.3 | `271d4bf8fbba269257742d336685bb213c2a04a1720887d5c22057efb83b48c0` | D-45 |
| `models/recurrent/rnn_integration_spec.md` v0.2 | `ef264a4b16f0dc9fe6dbf192bb0f7e831313cd701f67e81a365c5cb44016c323` | D-48 |
| `PROJECT_RESEARCH_RECORD.md` | `08f1005717771312ea340bb3a9ff38a6f72828ebc19e9fb2ed852a2c65e1e66b` | D-01 – D-48 |
| `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md` (UMIS) | — | §20 workflow |

4. **Precedence:**
   1. recorded decisions;
   2. the approved specifications;
   3. the integrated contract (D-48);
   4. this plan.
5. **Labels:**
   - **[C]** an approved requirement (cited);
   - **[P]** a proposal of this plan;
   - **[D]** a Stage 7 decision of the user (P7-1 – P7-8, 2026-11-14);
   - **OPEN** an item still requiring a decision (§13).
6. **Test split:** protected. Nothing in this plan builds, loads or evaluates test data.
7. **Proposed location:** `models/recurrent/rnn_implementation_plan.md` (not created).

**Version:** 0.2 (2026-11-14)

### Amendment log

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-11-14 | Initial Stage 7 draft |
| v0.2 | 2026-11-14 | User decisions P7-1 – P7-8 applied (§1). Infrastructure reuse decided after inspecting the WaveNet module (§4.3). Milestones re-cut into ten with exact file scopes (§5). Development procedure on the Mac repository with scope snapshots (§6). Decision verification mechanism (§8.4). Resume, interruption and per-format partial-record recovery (§9). Optional diagnostics deferred (§11). Slow tests organised by directory (§7.5). Implementation and research authorisation separated (§3). New open items (§13) |

---

## 1. Resolved Stage 7 decisions [D]

| # | Decision (user, 2026-11-14) | Where applied |
|---|---|---|
| P7-1 | **One explicit implementation authorisation per milestone.** Each milestone has a defined scope, deliverables, tests and acceptance criteria. No milestone starts before the previous one is approved | §5 |
| P7-2 | Reuse existing utilities **only where genuinely generic and importable unchanged**. Dependencies inspected first. No change to frozen Model 5 files or shared infrastructure. Where reuse would couple Model 6 to study-specific code, use the smallest independent Model 6 alternative | §4.3 |
| P7-3 | **No new authorisation to resume** an already-authorised run from a valid checkpoint, provided its run identity, seed, configuration and relevant execution settings are unchanged. Any configuration change, new run or change in scientific conditions requires the appropriate authorisation | §9.2 |
| P7-4 | **Fail-safe partial-record recovery:** preserve the original as evidence; validate all complete records; never silently discard, rewrite or overwrite evidence; record every recovery action and its result; abort the run if safe recovery cannot be established; an exact procedure per record format | §9.3 – §9.5 |
| P7-5 | **Development in the actual repository on the Mac**, with narrowly scoped changes and hash verification. No Git. No file outside the authorised milestone scope | §6 |
| P7-6 | **Optional diagnostics deferred.** Mandatory diagnostics are implemented and verified first. Optional diagnostics stay disabled and need separate approval | §11 |
| P7-7 | **Drivers fail closed** when required research decisions are absent or not accepted in the research record. A robust, explicitly tested mechanism; no fragile substring matching; no parallel approval system | §8.4 |
| P7-8 | **`pytest.ini` is not modified.** Slow tests are identified by test organisation and explicit commands | §7.5 |

---

## 2. Starting point (observed, read-only, 2026-11-14)

- `models/recurrent/rnn.py`, `gru.py` and `lstm.py` are **empty placeholder files** (0 bytes; SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
- `tests/models/recurrent/`, `tests/training/rnn_study/`, `training/rnn_study/` and `configs/models/rnn.yaml` do not exist.
- Test packages have no `__init__.py`. `pytest.ini` uses `pythonpath = .` and `--import-mode=importlib`, and tests import helpers by module path (for example `import tests.models.hierarchical.wavenet_reference as ref`).
- `utils/config.py` has **no model registry**: `configs/models/<name>.yaml` is located by name, so adding `rnn.yaml` touches no shared file. `configs/models/_template.yaml` gives the layout (`model.name`, `model.params`).
- `utils/io.py`:
  - JSON and checkpoint writes are **atomic** (temp file → fsync → `os.replace`);
  - `append_jsonl` writes one line per call with `flush`, **without fsync**, so a kill or power loss can leave a partial last line.
- `RunManager` statuses are `created`, `running`, `completed`, `failed` and `interrupted`. `track()` maps `KeyboardInterrupt` to `interrupted` and any other exception to `failed`. A killed process leaves `running`.
- **Execution environments:**
  - the bridge shell on the Mac is a **Linux VM (aarch64, Python 3.10, numpy 2.2.6, no torch)**;
  - the Claude workspace has torch 2.10 (Python 3.13);
  - the user's native macOS environment is the only one that matches the research runs.

  Consequences: §6.4 and OPEN O-1.

---

## 3. Implementation authorisation versus research-run authorisation

| | Implementation authorisation | Research execution authorisation |
|---|---|---|
| **Unit** | One milestone MS0 – MS9 (P7-1) | One lifecycle step L2 – L8 [C: integration §3.1], or one named set of runs within it |
| **Permits** | Creating and modifying **only** the files listed for that milestone (§5); test runs as stated in the authorisation (O-1) | Running one named driver with named parameters on project data, producing research records |
| **Never permits** | Reading `data/` or `results/` contents; building the probe; running any driver on project data; benchmarks; the gate; training; pilots; research runs; any test-split access | Changing source, configuration or specifications; any run not named; any test-split access |
| **Prerequisites** | Previous milestone accepted | All module milestones accepted (MS6 = L1); MS7 – MS9 accepted; frozen conditions recorded; every decision the driver needs present and accepted in the record (§8.4) |
| **Ends with** | Milestone acceptance by the user, then a record entry if the user instructs one | Recorded results; a decision fixing any value the step determines (probe hash, threads, g/s², η, E) |
| **Resume** | — | No new authorisation if identity and conditions are unchanged (P7-3, §9.2) |
| **Retry after abort** | — | New authorisation, new run directory, written reason [C: C7; D-18] |

Accepting MS6 is the **implementation acceptance** of lifecycle step L1. It authorises no research execution.

---

## 4. Dependencies and import boundaries

### 4.1 Dependency order [C: integration §1.1]

```
utils/*  ←  rnn.py  ←  rnn_diagnostics.py  ←  rnn_training.py  ←  training/rnn_study/*
```

### 4.2 Allowed imports

| Module | May import | Must not import |
|---|---|---|
| `models/recurrent/rnn.py` | `torch`, `math`, standard library, `utils.vocabulary`, `utils.reproducibility`, `utils.checkpoint`, `utils.device` (as the spec requires) | `rnn_diagnostics`, `rnn_training`, `training.*`, `numpy` (not needed), prohibited torch modules (A:R14) |
| `models/recurrent/rnn_diagnostics.py` | the above + `models.recurrent.rnn`, `numpy`, `utils.data`, `utils.io` | `rnn_training`, `training.*` |
| `models/recurrent/rnn_training.py` | the above + `models.recurrent.rnn_diagnostics` (`LENGTH_STRATA`, `PARAM_ORDER`, `param_sha256`, `RNNDiagnosticsPlan`, `DiagnosticsStatus`), `utils.batching`, `utils.run_manager`, `utils.metrics` | `training.*`; private `_step`, `_logits`, `_recurrence` |
| `training/rnn_study/*` | Model 6 modules; `utils.*`; from `training.wavenet_study.study_common` **only** the items of §4.3 | any other WaveNet study module or item |
| `tests/models/recurrent/rnn_reference.py` | `torch`, `math` **only** | everything else (independence) |
| Model 6 tests | the module under test, the reference, `utils.*`, the Git guard of §4.3 | `data/`, `results/`, other models' tests |

Enforced by static AST scans: X6 (modules), MS1 independence test (reference), O8 (orchestration).

### 4.3 Reuse of existing infrastructure (P7-2; inspected 2026-11-14)

`training/wavenet_study/study_common.py` (frozen, D-34) imports only the standard library at module level. `utils` and `torch` are imported inside functions. Importing it executes no study code and no Git.

| Item | Dependencies | Assessment | Plan |
|---|---|---|---|
| `disable_git`, `GitBlockedError`, `GIT_DISABLED` | stdlib; `utils.run_manager` (imported inside the function) | Generic: refuses any `git` process at `Popen._execute_child`, `os.system`, `posix_spawn*`, `spawn*`, `exec*`; stubs `git_info`; idempotent; tested by `tests/training/wavenet_study/test_git_guard.py`. No study semantics | **Import unchanged** |
| `HashChainedLog`, `LedgerError`, `canonical_json`, `GENESIS` | stdlib | Generic: append-only JSONL with SHA-256 chain, fsync on append, refuses an incomplete last line. No study semantics | **Import unchanged** |
| `AttemptStore` | `PilotPaths`, module-level `ATTEMPT_STATUSES` (`completed`, `mechanically_unstable`, `integrity_blocked`, `error`, `crashed`, `interrupted`), `pilot_result.json` paths, invocation records | **Study-specific**: rejects Model 6 statuses (`failed_mechanical`, `failed_diverged`, `aborted_before_commit`, `unrecoverable`); cannot be used unchanged | **Model 6 alternative** `training/rnn_study/attempts.py` (§8.3), built on the imported `HashChainedLog` |
| `PilotLock`, `lock_held_elsewhere` | `PilotPaths`; raises `Blocked("another pilot command …")` | Semi-generic; pilot-specific types and messages | **Model 6 alternative** (≈ 30 lines of `fcntl.flock` in `training/rnn_study/common.py`) |
| `verify_or_freeze`, `checksum_record`, `environment_record` | `PilotPaths`, WaveNet nonblocking paths | Study-specific | **Model 6 alternative** `training/rnn_study/conditions.py` |

**Residual coupling (stated, not hidden):**
- Model 6 code depends on the **import path** `training.wavenet_study.study_common` for seven generic items. The file is frozen (D-34), so its behaviour cannot change without a recorded decision, which would then also be reviewed for Model 6.
- If the user prefers **zero** cross-study coupling, the smallest alternative is a Model 6 copy of the two pieces: about 70 lines for the guard and about 50 for the log. It would need its own copy of the guard tests (OPEN O-2).

---

## 5. Milestones, exact file scope and acceptance criteria

Every milestone follows UMIS §20 steps 5 – 12:
1. coding brief (in the authorisation message);
2. implementation inside the file scope;
3. source review against the cited specification clauses, with a **traceability table** (clause → code location → test) in the milestone report;
4. automated tests (§7);
5. **the user runs the tests natively** and reports the result;
6. fixes inside the same scope;
7. **explicit acceptance by the user**.

**Scope rule [D: P7-5]:**
- "May create" and "may modify" lists are exhaustive.
- `__pycache__` directories produced by the user's test runs are the only other permitted changes.
- Any other change detected by the scope snapshot (§6.2) stops the milestone.
- A file accepted in an earlier milestone is modified only where a later milestone lists it. The earlier tests are then re-run as part of the later acceptance.

### MS0 — Test scaffolding and Git guard

| | |
|---|---|
| **Scope** | Guard and synthetic fixtures for all Model 6 tests |
| **May create** | `tests/models/recurrent/conftest.py`; `tests/models/recurrent/test_rnn_scaffolding.py` |
| **May modify** | — |
| **Content** | `conftest.py` calls `disable_git()` at import (§4.3), before anything imports `RunManager`. Synthetic fixtures: a small alphabet and invented names; the 27-symbol vocabulary built from `.` + a – z without reading `data/`; tiny configurations; float64 helpers |
| **Tests** | Guard active (`subprocess.run(["git", "--version"])` raises `GitBlockedError` without starting a process); `utils.run_manager.git_info()` returns the disabled record; AST scan: no test or fixture under `tests/models/recurrent/` names `data/`, `results/`, `load_names` or `split_names` |
| **Acceptance** | All MS0 tests pass natively; the scope snapshot shows only the two new files; the user accepts |

### MS1 — Independent reference

| | |
|---|---|
| **Scope** | Torch-only reference written from the specification **before** `rnn.py` has content |
| **May create** | `tests/models/recurrent/rnn_reference.py`; `tests/models/recurrent/test_rnn_reference.py` |
| **May modify** | — |
| **Content** | Explicit loops over time and batch: recurrence with frozen state on padding; logits; analytic Jacobians ∂h_t/∂h_{t−1}; BPTT gradients for U, W, b, V, c; G2 vectors (p − e_y); the G3 per-distance decomposition [C: arch §13.2 A:M1 – M5; diag D:D5] |
| **Tests** | A:M1 – M5: analytic Jacobians and BPTT gradients against autograd (tolerance A, float64) and central finite differences (float64); G3 decomposition sums to the autograd gradient; independence AST scan (imports `torch`, `math` only; no reference to `models.recurrent`) |
| **Acceptance** | All pass natively; the reference was written without access to any `rnn.py` content (`rnn.py` is still empty, confirmed by its SHA-256 in the snapshot); the user accepts |

### MS2 — Model (`rnn.py`) and architecture configuration

| | |
|---|---|
| **Scope** | Architecture and model behaviour [C: arch §5 – §9, §14, §15] |
| **May create** | `configs/models/rnn.yaml`; `tests/models/recurrent/test_rnn.py`; `tests/models/recurrent/slow/test_rnn_full_size.py` |
| **May modify** | `models/recurrent/rnn.py` (currently empty) |
| **Tests** | A:V1 – V5, A:R1 – R14 (including R1/R10 against the MS1 reference), A:I1 – I5, A:8a – 8d, A:K1 – K2; full-size checks (H = 128; 23,451 parameters; ln 27 at initialisation; Haar orthogonality) in `slow/` |
| **Acceptance** | Integration checklist rows 1, 4 – 9 and the A:M parts of row 2 pass natively; traceability table complete for arch §5 – §15; `gru.py` and `lstm.py` unchanged (hash); the user accepts |

### MS3 — Mandatory diagnostic functions

| | |
|---|---|
| **Scope** | `rnn_diagnostics.py` functions, mandatory only [C: diag §3 – §4, §10; integration §4.3 – §4.5] |
| **May create** | `models/recurrent/rnn_diagnostics.py`; `tests/models/recurrent/test_rnn_diagnostics.py`; `tests/models/recurrent/slow/test_rnn_diagnostics_full_size.py` |
| **May modify** | — |
| **Content** | `LENGTH_STRATA`, `PARAM_ORDER`, `param_sha256` (C3 exact bytes), `float64_copy`, `Probe`, `build_probe`, `load_probe`, `validate_probe`, `ProbeMismatchError`, `h1_statistics`, `g1_products`, `g2_attenuation`, `g3_decomposition`, `c1_restart`, `mean_field_table`, `initial_statistics` (seeds 1000 – 1009 only, D-47), result dataclasses. **Not created:** `positional_data_bound` (optional, deferred by P7-6; §11) |
| **Tests** | D:D1 – D3 (probe determinism, strata, weights, halves; **synthetic data**), D:D5, D:D6 – D8, A:M6 – M11, D:D4, D:D9 (storage round trip, hash alignment), the `param_sha256` hand fixture with a precomputed digest, X5 (probe mismatch rejection; seed-set refusal) |
| **Acceptance** | Rows 2 (D:D5), 15, 16, 20 (D:D4, D:D9) pass natively; `initial_statistics` refuses any seed set other than 1000 – 1009; no call path reads `data/`; the user accepts |

### MS4 — Diagnostics plan, records and failure handling

| | |
|---|---|
| **Scope** | Stateful plan, writers, truncation and failure path [C: diag §5, §8, §11; integration §4.1, §4.2, §4.6, §4.7] |
| **May create** | `tests/models/recurrent/test_rnn_diagnostics_plan.py` |
| **May modify** | `models/recurrent/rnn_diagnostics.py`: **additions only**; MS3 definitions unchanged (verified by comparing each MS3 top-level definition's source before and after, by AST, in the milestone report) |
| **Content** | `OptionalDiagnostics` (both switches; **any `True` raises `ValueError("optional diagnostic not authorised")`** at plan construction, P7-6), `RNNDiagnosticsPlan` (`at_start`, `after_step`, `at_epoch_end`, `truncate`, `status`), `DiagnosticsStatus`, writers for schema `m6_diagnostics/3` (`plan.json`, `points.jsonl`, `primary.jsonl`, `raw/point_{id:04d}.npz`, `o3.json`), `posthoc_best_spectral`, and the C6 failure path. **Write rules [P]:** JSON and `.npz` files atomically (temp → fsync → replace, via `utils.io` where it provides this); `.jsonl` lines with flush **and fsync**; a raw file is written before its `points.jsonl` line, so every point line refers to a complete raw file |
| **Tests** | D:D10 (injected faults detected), D:D11 (plan resolution: P₀, P_early σ = 2^j < S, P_epoch, G1-σ only at E_sel), writer round trip, `truncate` keeps exactly σ ≤ σ_K, invalid-point records on injected exceptions, failure-record I/O error propagates, optional switch refusal |
| **Acceptance** | Row 17 and the MS4 unit tests pass natively; MS3 tests re-run and pass; the user accepts |

### MS5 — Training (`rnn_training.py`)

| | |
|---|---|
| **Scope** | Training, evaluation, selection, checkpoints, resume, failure categories, research-run validation [C: training §2 – §13; integration §3.2, §3.4, §3.5, §5.2] |
| **May create** | `models/recurrent/rnn_training.py`; `tests/models/recurrent/test_rnn_training.py`; `tests/models/recurrent/test_rnn_isolation.py`; `tests/models/recurrent/slow/test_rnn_training_full_size.py` |
| **May modify** | — |
| **Content** | `RNNTrainingConfig`, `RNNTrainingResult`, `loss_normalizer`, `evaluate_nll`, `evaluate_position_nll`, `train_rnn`, `validate_research_run`, `ResumeError`, `ResearchConfigError`. **Partial-record behaviour [P, P7-4]:** `train_rnn(..., resume=True)` raises `ResumeError` if any `.jsonl` record it truncates ends with an incomplete line. Recovery of such files is the orchestration preflight's job (§9.4), so the training module never discards bytes it has not validated. **Step-record file name** (not fixed by the training specification §12.2): OPEN O-3 |
| **Tests** | A:T1 – T8, T:T9 – T24, D:I1 – I8 (isolation, in their own file), X2 (research configuration rejection), X4 (abort before first commit), the partial-line refusal |
| **Acceptance** | Rows 3, 10 – 14, 18, 19, 21 – 23 pass natively; resume bitwise tests (T:T15 – T17) pass at one thread; the user accepts |

### MS6 — Integration and implementation acceptance (L1)

| | |
|---|---|
| **Scope** | Cross-module properties; complete checklist |
| **May create** | `tests/models/recurrent/test_rnn_integration.py` |
| **May modify** | Only files of MS2 – MS5, and only to fix defects found by MS6 tests, each fix listed with its reason |
| **Tests** | X1 (end-to-end hook order and σ values; P₀ before step 1; epoch-end points before `checkpoint_last.pt`; schema validation; hash hand fixture), X3 (diagnostic failure handling, C6), X6 (dependency order and import boundaries of §4.2) |
| **Acceptance** | **All 30 integration checklist rows** pass natively, with fast and slow suites run by the commands of §7.5; architecture checklist (arch §17) ticked; independent review completed (§7.3) and its findings resolved or accepted by the user; the user records **implementation acceptance (L1)**. Execution remains unauthorised |

### MS7 — Orchestration infrastructure

| | |
|---|---|
| **Scope** | Study-level mechanisms, no lifecycle driver |
| **May create** | `training/rnn_study/common.py` (paths, guard installation, lock, thread setup), `training/rnn_study/decisions.py` (§8.4), `training/rnn_study/conditions.py` (frozen conditions, study conditions), `training/rnn_study/attempts.py` (§8.3), `training/rnn_study/recovery.py` (§9.4); `tests/training/rnn_study/conftest.py` (guard; synthetic study root in a temporary directory); `tests/training/rnn_study/test_decisions.py`, `test_conditions.py`, `test_attempts.py`, `test_recovery.py`, `test_lock.py` |
| **May modify** | — |
| **Tests** | §8.4 decision-verification tests; frozen-condition mismatch → `integrity_blocked`; attempt events, statuses, counting rule, orphans; lock exclusivity across processes; recovery procedures of §9.5 for every format, including **subprocess kills** (SIGKILL) before and after the first commit and during an append |
| **Acceptance** | All pass natively; no file outside the scope changed; no read of `data/` or `results/` (static scan); the user accepts |

### MS8 — Lifecycle drivers

| | |
|---|---|
| **Scope** | One driver per lifecycle step, with summaries and verifier; **no execution on project data** |
| **May create** | `training/rnn_study/build_probe.py` (L2), `thread_benchmark.py` (L3), `diagnostic_timing.py` (L4), `init_gate.py` (L5), `lr_pilot.py` and `extension_x.py` (L6), `research_runs.py` (L7), `posthoc_spectral.py` (L8), `summarize_gate.py`, `summarize_pilot.py`, `verify_runs.py`; `tests/training/rnn_study/test_drivers.py`, `test_verify_runs.py` |
| **May modify** | — |
| **Tests** | O1 – O9 (§8.5) on synthetic data and temporary study roots; dry-run mode of each driver on synthetic data |
| **Acceptance** | All pass natively; each driver refuses on every missing prerequisite (§8.2); the user accepts |

### MS9 — Execution readiness (no execution)

| | |
|---|---|
| **Scope** | Prepare, not perform, the first research step |
| **May create** | The study root `results/experiments/rnn_study/` [P] with `study_conditions.json` holding the frozen conditions (§8.1), the decision references available so far (D-44 – D-48 and the implementation-acceptance decision) and `record_reviewed_through` |
| **May modify** | — |
| **Acceptance** | Frozen-condition hashes verified; `decisions.py` validates every pinned decision against the live record; the authorisation request for L2 (probe construction) is drafted. **L2 is not run** until the user authorises it |

---

## 6. Development procedure on the Mac repository [D: P7-5]

### 6.1 Writing files

- Source and test files are written **directly into the repository** through the bridge. Short files are written in one step. Larger files are written with an exact-content write and then verified.
- Existing files are changed only by read-modify-write scripts with exact-match assertions (each `old` text must occur exactly once). Content is never retyped from earlier tool output.
- After every write, the file's SHA-256 is read back and listed in the milestone report.
- No parallel copy is kept as a source of record. The repository on the Mac is the only one.

### 6.2 Scope snapshots

- **Before** the first write of a milestone and **after** its last write, a snapshot is taken and compared:
  - SHA-256 of every file in the repository, **excluding** `.git/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `data/` and `results/`;
  - for `data/` and `results/`, only path, size and modification time (**contents are not read**).
- The snapshot is stored outside the repository.
- The difference must equal the milestone's create and modify lists. Anything else stops the milestone and is reported.

### 6.3 No Git

No Git command is run, directly or through any tool. `.git/` is neither read nor listed.

### 6.4 Where tests run (OPEN O-1)

- The bridge shell is a Linux VM without torch, and dependencies may not be installed, so **Model 6 tests cannot run there**.
- The user's **native macOS run is authoritative** for every acceptance (UMIS §20 step 9). Exact commands are given in each milestone report (§7.5).
- Optional pre-check: if a milestone authorisation permits it, the same tests can run first in a disposable **workspace mirror** of the repository's source and test files, excluding `data/`, `results/` and `.git/`. This catches defects before the user's run. It is never used as acceptance evidence.

---

## 7. Test strategy and independent mathematical verification

### 7.1 Layers

1. **Specification tests:** every label in integration spec §8 at its tolerance (A = 1e-9 float64; B = C = 1e-5 float32) [C: arch §13.1]. Duplicate labels share one test named with all labels.
2. **Independent mathematical verification** (§7.2).
3. **Fault injection:**
   - non-finite values at every Category A check;
   - Category D;
   - exceptions in each mandatory diagnostic;
   - I/O failure while writing a failure record;
   - subprocess kills before and after the first commit and during an append.
4. **Bitwise equivalence** at one thread: resume (T:T15 – T17), diagnostic isolation (A:T5 = T:T24 = D:I1), pilot-extension prefix, recovery (§9.5).
5. **Static scans:** prohibited modules (A:R14), dependency order and import boundaries (X6, §4.2), reference independence (MS1), no `data/`/`results/` access in tests, no test-split access in orchestration (O8).

### 7.2 Independent mathematical verification

| Check | Method | Independence |
|---|---|---|
| Forward recurrence, padding, logits | Model vs `rnn_reference.py` (explicit loops) | Reference written from the spec before the model exists (MS1) |
| Jacobians, BPTT gradients | Reference analytic formulas vs autograd vs central finite differences (float64, step and tolerance fixed in the brief) | Three methods with different failure modes |
| G1 products, ω, δ̂ | Explicit product of Jacobians on tiny synthetic cases vs `g1_products`; δ̂ = φ̂ at θ₀ (D:D7) | Closed-form special cases (for example W = 0 or W orthogonal with saturating inputs) |
| G2 vectors and exclusion | p − e_y in float64 vs reference; exact-zero exclusion on constructed cases | Constructed inputs |
| G3 decomposition | Σ over Δ equals the autograd gradient (also a run-time check) | Autograd as oracle |
| C1 restart | Restart NLL vs explicit re-run of the recurrence from BOS on the truncated context | Reference recurrence |
| Initialisation | NLL = ln 27 exactly at V = c = 0; WᵀW = g²I; U variance (statistical bound fixed in the spec) | Analytic values |
| Loss normalisation | C = 730,664/3,205 recomputed from the approved counts (no data read); per-target weight 1/C on synthetic batches | Arithmetic |
| Probe weights and halves | Hand-computed Horvitz–Thompson totals on a synthetic population | Hand computation |

### 7.3 Independent review

- At MS6 a **separate reviewer** checks the full source against the four approved documents and reports deviations with file and line references. The reviewer is a separate agent session that took no part in implementation.
- The review covers contract conformance, untested clauses, silent assumptions, RNG use, test-split exposure and Git exposure.
- Findings go to the user. Fixes stay within MS6 scope.

### 7.4 Synthetic data rule

No Model 6 test reads `data/`, the training split, the probe file or any research record. Full-size tests use synthetic inputs at full width. The only file outside the test tree a test may read is the approved specifications and the research record, read-only, by the decision-parser test in MS7 (§8.4).

### 7.5 Slow tests [D: P7-8]

- Slow tests live in **`slow/` subdirectories** (`tests/models/recurrent/slow/`) with normal `test_*.py` names. A plain `pytest` therefore still runs everything, and nothing is silently skipped.
- `pytest.ini` is not modified.
- Explicit commands, given in every milestone report:

| Purpose | Command |
|---|---|
| Fast Model 6 suite | `python -m pytest tests/models/recurrent --ignore=tests/models/recurrent/slow` |
| Slow Model 6 suite | `python -m pytest tests/models/recurrent/slow` |
| Full Model 6 suite (acceptance) | `python -m pytest tests/models/recurrent` |
| Orchestration (MS7 – MS8) | `python -m pytest tests/training/rnn_study` |
| Whole repository (regression check that nothing else changed) | `python -m pytest` |

---

## 8. Orchestration design (`training/rnn_study/`)

### 8.1 Study root and conditions [P]

The study root is `results/experiments/rnn_study/`. It holds:

| Path | Content | Write rule |
|---|---|---|
| `study_conditions.json` | Frozen conditions (§8.1.1); decision references (§8.4); values fixed by decisions; `record_reviewed_through` | Written by a human-authorised step; read-only to drivers |
| `manifest.jsonl` | Attempt events (hash chain) | `HashChainedLog` (fsync) |
| `invocations.jsonl` | One record per driver invocation, with the manifest head | `HashChainedLog` |
| `probe/probe.json` | Probe (L2) | Atomic; immutable after recording |
| `<step>/runs/<run_id>/` | One run directory per attempt (`RunManager` layout) | as §9.5 |
| `.lock` | Exclusive `flock` | — |

**8.1.1 Frozen conditions:**
- SHA-256 of the three Model 6 modules, `rnn.yaml`, the four specification files, every `training/rnn_study/*.py`, the imported `study_common.py` and each imported `utils` module;
- Python, torch and numpy versions; platform; CPU description.

Every driver verifies them before acting. A mismatch makes the driver end as `integrity_blocked`, with nothing written to run directories.

### 8.2 Prerequisites per driver (fail closed)

| Driver | Required decisions and values (verified by §8.4) |
|---|---|
| all | Git guard active; lock held; frozen conditions match; no unreconciled attempt; no orphan run directory; implementation acceptance (L1) |
| `build_probe.py` | Authorisation of L2; no existing `probe.json` |
| `thread_benchmark.py` | Probe decision (hash, N_ℓ, F₈ count); authorisation of L3 |
| `diagnostic_timing.py` | Thread-count decision; authorisation of L4 |
| `init_gate.py` | Timing recorded; authorisation of L5; seed set exactly 1000 – 1009 (D-47) |
| `lr_pilot.py` | **Gate decision (g, s²) approved** (D-47); authorisation of L6 |
| `extension_x.py` | Pilot decision; the D-46 extension condition met; authorisation |
| `research_runs.py` | η and E decisions; M6-EXP1, M6-EXP2 and failed-seed policy decisions (U4); authorisation of L7; `validate_research_run` passes |
| `posthoc_spectral.py` | Run status `completed`; `model_best.pt` present; authorisation of L8 (may be granted together with L7) |

### 8.3 Attempt records and classification (`attempts.py`)

**Two status layers.** The training outcome comes from `train_rnn` (D-46). The process outcome is assigned by orchestration. Each attempt is keyed by (step, arm, seed), for example (`pilot`, η = 0.1, seed 101). Manifest events per attempt are: one `launched`; then `resumed` (zero or more); then at most one `finished` (written by the process) or `registered` (written by a later invocation that reconciled it).

| Status | Assigned when | Final? | Continuation |
|---|---|---|---|
| `completed` | `train_rnn` returned `completed` | yes | — |
| `failed_mechanical` | Category A | yes | **none**; pilot candidate ineligible; research seed under U4 |
| `failed_diverged` | Category D | yes | as above |
| `aborted_before_commit` | ended (exception or kill) with no `checkpoint_last.pt` [C: C7] | yes | **retry** only by new authorisation in a new run directory; old directory preserved |
| `interrupted` | `KeyboardInterrupt` after ≥ 1 commit | no | **resume** (§9.2) |
| `crashed` | lock free, metadata `running`, no result, ≥ 1 commit | no | **resume** (§9.2) |
| `error` | non-Category A/D exception after ≥ 1 commit (defect, I/O) | no | resume only if the cause is external and conditions are unchanged (§9.2); otherwise void → new authorisation |
| `unrecoverable` | the recovery preflight (§9.4) cannot establish safe recovery | yes | new authorisation, new run directory; directory preserved as evidence |
| `integrity_blocked` | frozen conditions or a decision check fail before launch or resume | no | human decision; nothing was run |

**Reconciliation** happens at every invocation, under the lock:
- attempts that are `launched` but not closed are classified by the table and closed with `registered`;
- run directories without an attempt record block the driver.

**Counting rule:** exactly one attempt counts per key, namely the last with a final status in {`completed`, `failed_mechanical`, `failed_diverged`}. Aborted and unrecoverable attempts are listed in summaries and never pooled. A resume is a new session of the same attempt, not a new attempt.

### 8.4 Decision verification (`decisions.py`) [D: P7-7]

**Source:** the decision table of `PROJECT_RESEARCH_RECORD.md` §11. **No new approval file is introduced.** `study_conditions.json` only **references** decisions.

**Parsing (structural, not substring):**
1. Locate the line exactly `## 11. Decisions and Rationale` (exactly once in the file) and the next line exactly `## 12. …` that ends the section.
2. Within it, find exactly one table whose header cells equal (`ID`, `Date`, `Decision`, `Status`, `Reason`, `Consequences`, `Reference`), followed by its separator row. The table ends at the first line not starting with `|`. Any `| D-` line outside it in §11 → error.
3. Split each row into cells with a **code-span-aware splitter**. A `|` inside backticks or escaped as `\|` is not a separator. Each row must have **exactly 7 cells**. (Verified on the current record: 49 rows, all 7 cells. A naive split gives 9 cells in one row.)
4. IDs must match `D-\d{2,}( addendum)?` and be unique. Rows must stay in file order.

**Acceptance test for one referenced decision:**
1. The ID exists.
2. Its Status cell matches exactly `^ACCEPTED \(user (approval|authorisation|decision), \d{4}-\d{2}-\d{2}\)$`. Any qualifier, `PENDING`, `PROPOSED`, partial acceptance or other wording → **fail closed**. D-44 – D-48 already use this form. Future execution-relevant decisions are written in it [P].
3. The SHA-256 of the row's exact text (UTF-8, without the newline) equals the value pinned in `study_conditions.json`. This detects any edit of an accepted row.
4. **Staleness:** the table contains no decision after the ID pinned as `record_reviewed_through`. A newly recorded decision therefore blocks every driver until a human-authorised update of `study_conditions.json` has considered it. This requires no text matching and has no false negatives.

**Failure:** `DecisionError` names the ID and the failed condition. The driver exits `integrity_blocked`. Nothing is written to run directories.

**Tests (MS7):** synthetic record fixtures for:
- a valid table;
- a missing ID;
- statuses with `PENDING`, other qualifiers or `PROPOSED`;
- an edited row (hash mismatch);
- rows with 6 or 8 cells;
- `|` inside a code span (accepted) and escaped `\|` (accepted);
- a duplicate ID;
- a missing or renamed heading;
- two tables;
- a decision beyond `record_reviewed_through`.

Plus one read-only test that the **real** record parses and that D-47 and D-48 pass with their current row hashes. That test is updated by the authorised step that pins them.

### 8.5 Orchestration tests (MS7 – MS8)

| ID | Property |
|---|---|
| O1 | Git guard active in every driver entry point |
| O2 | Every driver refuses for every missing prerequisite of §8.2 |
| O3 | Frozen-condition or decision failure → `integrity_blocked`; nothing written to run directories |
| O4 | SIGKILL before the first commit → `aborted_before_commit`; resume refused; retry needs a new directory |
| O5 | SIGKILL after a commit → `crashed`; resume (with recovery preflight) gives records bit-identical to an uninterrupted run |
| O6 | Category A and D outcomes never retried; pilot candidate ineligible |
| O7 | Counting rule; orphans block |
| O8 | Static scan: `training/rnn_study/` never builds or reads the test split |
| O9 | `init_gate.py` refuses any seed set other than 1000 – 1009; `lr_pilot.py` refuses without the gate decision |
| O10 | Recovery: each format and each failure case of §9.5 |

### 8.6 Threads and environment

- Each driver process sets intra- and inter-op thread counts **before any tensor operation** (inter-op can be set once per process).
- One run per process.
- Research runs execute natively on the Mac only. Bit-identity is claimed within one machine and environment.

---

## 9. Resume, interruption and partial-record recovery

### 9.1 Interruption

| Event | Effect | Attempt status |
|---|---|---|
| `Ctrl-C` (`KeyboardInterrupt`) | `RunManager.track()` records `interrupted`; nothing of the current epoch is committed | `interrupted` (or `aborted_before_commit` if no commit yet) |
| SIGKILL, SIGTERM, power loss, crash, macOS ending the process | Metadata stays `running` | `crashed` (or `aborted_before_commit`) on reconciliation |
| Python exception other than Category A/D | `RunManager` records `failed` | `error` (or `aborted_before_commit`) |
| Category A / D | Run ends by design [C: D-46] | `failed_mechanical` / `failed_diverged` |
| macOS sleep that suspends but does not end the process | Run continues | none; wall-clock timings note the gap |

No signal handler is installed [P]: fewer mechanisms, and every abnormal end is handled by reconciliation.

### 9.2 Resume without new authorisation [D: P7-3]

Resume of an attempt in status `interrupted`, `crashed` or (external cause only) `error` needs no new authorisation if **all** of the following are unchanged. Otherwise the driver refuses and a new authorisation is needed.

| Identity element | Checked against |
|---|---|
| Run identity | Same run directory, `run_id`, attempt key, manifest record |
| Seed | Attempt key and checkpoint |
| Configuration | `config.json` (experiment, model, training, derived, diagnostics) equal in canonical JSON; the approved compatibility fields [C: training §10.6 step 2], including C, N, M, S, vocabulary, dtype, device, thread counts, diagnostics presence and probe SHA-256 |
| Execution settings | Frozen conditions (§8.1.1: code, specifications, environment versions, platform) |
| Scientific conditions | Every decision referenced by the driver still passes §8.4 |
| Checkpoint validity | `checkpoint_last.pt` loads (`weights_only=True`); its `param_sha256` equals that of history row K |

The resume procedure, under the lock:
1. Reconciliation.
2. Identity checks.
3. **Recovery preflight** (§9.4).
4. `train_rnn(..., resume=True)`, with the approved truncation [C: training §10.6].
5. **Recovery post-check** (§9.4).
6. A `resumed` manifest event with the recovery record's hash.

### 9.3 Principles of recovery [D: P7-4]

1. **Preserve first:** before anything in a run directory changes, every record file that resume may change is copied byte for byte into `recovery/<session_id>/original/` inside the run directory. The copy's SHA-256 must equal the original's, otherwise abort.
2. **Validate all complete records** before changing any file. **Two phases:** all files are validated first, and only then is any file changed.
3. **Never silently discard:** the approved truncation [C: training §10.6 step 5; diag §5.2] still removes records beyond the commit point from the **active** files. Their bytes remain in the preserved copy, and every removal is listed in the recovery record.
4. **Record every action:** `recovery.jsonl` (hash-chained, in the run directory) receives one entry per session. Each entry gives:
   - the session id and commit point (K, σ_K);
   - per file: original size and SHA-256, the preserved copy's path and SHA-256, the number of complete records kept and removed, the partial tail's byte count and SHA-256, the new size and SHA-256;
   - the result (`recovered` or `aborted`) and the reason.
5. **Abort when unsafe:** if any check fails, no active file is changed, the attempt becomes `unrecoverable`, and the directory is left for inspection.

### 9.4 Preflight and post-check (`recovery.py`)

**Preflight** (before `train_rnn(resume=True)`):
1. Determine the commit point from `checkpoint_last.pt`: K = `epoch`, σ_K = `step`.
2. Preserve every file of §9.5 marked "resume-changed", as §9.3 (1).
3. Validate each file by its format rule (§9.5). In particular:
   - every **complete** line (terminated by `\n`) parses as one JSON object of the expected schema;
   - keys are in the required order;
   - every record required **up to the commit point** is present (no gap).
4. A **partial tail** (bytes after the last `\n`) is acceptable only if:
   - (a) it is the last bytes of the file;
   - (b) all records up to the commit point are complete before it (so the tail necessarily belongs after the commit point, because records ≤ σ_K precede the commit in write order [C: training §10.3 item 6; integration §3.2]).

   It is never parsed or repaired.
5. If every file passes, remove each partial tail from its active file: write the exact byte prefix up to and including the last `\n`, atomically (temp → fsync → replace), and verify that the new file equals `original[:offset]`.
6. Append the preflight entry to `recovery.jsonl`.

**`train_rnn(resume=True)`** then performs the approved truncation of complete records beyond the commit point. It refuses any remaining partial line (MS5).

**Post-check** (after `train_rnn` returns or ends):
1. Each truncated active file must be a byte prefix of its preserved original up to the commit point.
2. Records **re-generated** by the resumed run that overlap preserved originals beyond the commit point are compared byte for byte. The approved behaviour is a bit-identical rewrite on CPU [C: training §10.6 consequences; T:T16]. A match is recorded. A **mismatch is recorded, and the run is flagged for human review** before any use; it is not silently accepted.
3. Append the post-check entry to `recovery.jsonl`.

### 9.5 Procedure per record format

| File | Writer and write rule | Resume-changed? | Validation (complete records) | Partial content | Kept / removed |
|---|---|---|---|---|---|
| `history.jsonl` | `RunManager.append_history` (one line, flush) | yes | JSON object per line; `completed_epochs` contiguous from 0; schema of training §8.3 – §8.4; row K's `param_sha256` equals `checkpoint_last.pt`'s | partial last line → preflight (§9.4 step 4) | keep rows with `completed_epochs ≤ K`; later rows removed by the approved truncation, preserved |
| Step records (O1; file name OPEN O-3) | one line per step, flush **and fsync** [P] | yes | JSON per line; cumulative step contiguous from 1; exactly σ_K records ≤ σ_K | as above | keep step ≤ σ_K |
| `points.jsonl` (diagnostics) | one line per point, flush and fsync (MS4) | yes | JSON per line; `point_id`, σ and kind match `plan.json`'s resolved points in order; every point with σ ≤ σ_K (excluding an uncommitted epoch end) present; before/after hashes present | as above | approved `truncate` [C: diag §5.2] |
| `primary.jsonl` | as `points.jsonl` | yes | one record per `points.jsonl` record ≤ commit, same `point_id`s | as above | as above |
| `raw/point_{id:04d}.npz` | atomic (temp → fsync → replace), before its `points.jsonl` line | yes (files beyond commit) | each raw file referenced by a kept point loads with `allow_pickle=False` and holds the arrays of diag §8 | cannot be partial (atomic); stray temp files (`.point_*.tmp`) are **moved** into `recovery/<session>/original/` | raw files of removed points are **moved** into the preserved set, never deleted |
| `plan.json` | atomic | no | parses; equals the plan reconstructed for the resume (probe hash, S, E, gain, points) | n/a | unchanged |
| `o3.json` | atomic; written at run end | no (rewritten at the run's end) | parses if present | n/a | a pre-existing `o3.json` is preserved before the end-of-run write |
| `config.json`, `metadata.json` | `RunManager`, atomic | metadata gains a session | parse; identity checks (§9.2) | n/a | unchanged except the `RunManager` session entry |
| `checkpoint_last.pt`, `model_best.pt`, `model_final.pt` | atomic | overwritten by the resumed run at its next commit, as approved | load with `weights_only=True`; `param_sha256` alignment with history | stray temp files moved into the preserved set | the pre-resume checkpoints are part of the preserved set |
| `logs/*.log` (text) | appended | appended | not parsed | left as is; the resumed session starts with a new line and a session marker | never truncated |
| `recovery.jsonl` | `HashChainedLog` | appended | chain verifies | incomplete last line → **abort** (`unrecoverable`); evidence of a fault during recovery itself | — |
| Study `manifest.jsonl`, `invocations.jsonl` | `HashChainedLog` (fsync) | appended | chain verifies (`HashChainedLog.read`) | incomplete last line → **all drivers blocked**; human decision (chain of custody is never repaired automatically) | — |
| `probe/probe.json`, `study_conditions.json` | atomic; human-authorised | no | SHA-256 and schema | n/a | never changed by recovery |

---

## 10. Diagnostic runtime and computational risks

Planning figures only. **The binding numbers come from L4** (M6-DGN13). A reduction of mandatory diagnostics needs an amended decision.

| Item | Count / size | Risk | Mitigation |
|---|---|---|---|
| Points per research run | P₀ 1 + P_early 10 (σ = 2^0 … 2^9 < 802) + P_epoch E → **41 (E = 30) to 71 (E = 60)** | Diagnostics dominate run time | Measured at L4 and reported before L7 |
| G1 products | Stage 5 estimate ≈ 47 G float64 multiply–accumulates per point | Python overhead on 128 × 128 products may make each run take minutes to about an hour (not measured) | Batched products over names and start positions; cumulative products; verified against the reference before any optimisation; no approximation |
| G1-σ | E_sel points only (+ post-hoc) | Many SVDs of 128 × 128 matrices | Batched |
| G3 | Per-Δ accumulation on probe and halves | Memory if per-pair vectors were kept | One parameter-sized vector (23,451 values) per Δ; run-time identity check |
| C1 | F₈ (count from L2) and F₇ (734) × d ≤ 8 | Small | Batched restarts |
| Storage | `raw/` per point; preserved copies on recovery | Disk use | Measured at L4; recovery copies counted |
| Training | 802 steps/epoch; pilots 12 runs × 30 epochs (+ up to 2 extension rounds; X: 3 × 60) | CPU wall time | Thread benchmark L3 |

---

## 11. Mandatory and optional work [D: P7-6]

| Work | Class |
|---|---|
| MS0 – MS6 (modules, configuration, 30 checklist rows, X1 – X6) | **Mandatory** |
| Mandatory diagnostics [C: integration §4.7]: H1 on the probe; G1-F, ω, δ̂; G1-σ at E_sel; G2; G3 with halves; C1 with per-target values; O2; O3; run-time checks; isolation records | **Mandatory**, implemented and verified first (MS3, MS4) |
| `OptionalDiagnostics` type | Created (part of the approved plan signature); **both switches disabled; enabling either raises `ValueError`** until a separate approval |
| `h1_on_dev`, `spectral_at_powers_of_two` computations | **Deferred**; not implemented |
| `positional_data_bound` (M6-DGN19) | **Deferred**; not created. This is a deliberate deviation from the integration spec §6 file map, authorised by P7-6, to be stated in the decision approving this plan |
| MS7 – MS9 orchestration | **Mandatory** before any execution |
| Extra summaries, plots, performance work beyond L4 needs | Optional |
| RQ analysis and reporting; test evaluator | Out of scope (later stages) |
| Git guard for Model 1–5 tests | Outside Model 6 (D-48 U5); not touched |

---

## 12. Risks

| # | Risk | Consequence | Mitigation |
|---|---|---|---|
| R1 | Hook order drifts from integration §4.2 | Diagnostics misaligned with checkpoints | X1 call-sequence test |
| R2 | Parameter-hash mismatch across records | Post-hoc refusal; lost alignment | One `param_sha256` (C3, C4); hand fixture; T:T18, T:T23 |
| R3 | Probe or data drift after L2 | Wrong names diagnosed | `validate_probe` before every computation (C5); probe hash frozen |
| R4 | Diagnostic exception reported as success | Invalid evidence | C6 path; X3; `incomplete` blocks analysis |
| R5 | Partial last line after a kill | Unsafe resume | §9 preflight; MS5 refusal; O10 |
| R6 | Kill during checkpoint or raw write | Corrupt file | Atomic writes; stray temp files preserved |
| R7 | Category D epoch still runs `at_epoch_end` | Expected (integration §3.2) | X1 includes a diverging fixture |
| R8 | Inter-op threads set too late | Error or wrong threads | Set at process start; checked by `validate_research_run` |
| R9 | `device: auto` or float64 in a research run | Non-comparable runs | `validate_research_run` (C8); X2 |
| R10 | Git started indirectly via `RunManager.create` | Policy breach | Guard first in tests (MS0) and drivers (O1) |
| R11 | A test reads project data | Probe or test-split exposure | Synthetic-only rule; static scans |
| R12 | Approved spec headers still read "DRAFT" | Confusion during implementation | Briefs cite D-44 – D-48 as status authority |
| R13 | Tests cannot run in the bridge VM | Delays; temptation to accept unverified code | Native run mandatory for acceptance; optional mirror pre-check (O-1) |
| R14 | Cross-study import coupling (§4.3) | A future change to the frozen WaveNet file affects Model 6 | Frozen status (D-34); frozen-condition hash includes `study_common.py`; O-2 offers zero coupling |
| R15 | `record_reviewed_through` blocks drivers after every new decision | Friction mid-study | Intended fail-closed behaviour; the update is part of each decision's authorised step |
| R16 | Resumed run not bit-identical in its re-run epoch | Evidence conflict | Post-check comparison; flagged for human review |

---

## 13. Unresolved decisions

| # | Item | Options | Recommendation |
|---|---|---|---|
| O-1 | **Test execution before your native run** | (a) only your native macOS runs; (b) each milestone authorisation may also permit a pre-check run in a disposable workspace mirror (source and tests only; no `data/`, `results/`, `.git/`) | (b), as a pre-check only; native run remains the acceptance evidence |
| O-2 | Cross-study coupling for the Git guard and hash-chained log | (a) import unchanged from `study_common.py` (§4.3); (b) Model 6 copies (≈ 120 lines) with copied tests | (a): generic, tested, frozen; one implementation of a policy-critical guard |
| O-3 | Step-record file name and location (training spec §12.2 says only "follows the run layout of §10") | (a) `diagnostics/steps.jsonl` (`RunManager.diagnostics_dir`, the WaveNet precedent) with the research diagnostics in `research_diagnostics/`; (b) another layout | (a), keeping training records and research diagnostics in separate directories |
| O-4 | Status wording for execution-relevant decisions | Adopt the exact form `ACCEPTED (user approval, YYYY-MM-DD)` (also `authorisation` or `decision`), as already used by D-44 – D-48 | Adopt |
| O-5 | Recording this plan | A decision approving the plan, stating P7-1 – P7-8, O-1 – O-4 and the P7-6 deviation from the integration spec §6 file map | Record on approval |
| U4 | M6-EXP1, M6-EXP2, failed-seed policy | — | Before L7 (D-48) |

---

## 14. Readiness

- **Plan completeness:** every approved interface has a milestone, an exact file scope and tests. Every lifecycle step has a driver, prerequisites, decision checks and attempt classification. Every record format has a recovery rule. Every risk has a mitigation or an open item.
- **Before MS0:** approval of this plan, decisions O-1 – O-4, and the **MS0 implementation authorisation**.
- **Before any research execution:** MS0 – MS9 accepted, then a separate authorisation per lifecycle step (§3).

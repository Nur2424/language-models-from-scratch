# WaveNet Study — Common Epoch-Budget Pilot Protocol

**Status:** APPROVED protocol (human review, 2026-11-11; decisions D-23 and D-26 in `PROJECT_RESEARCH_RECORD.md`). **Implementation and execution are NOT authorised.** No budget-pilot run, test or benchmark has been performed under this protocol, and no result is reported here.
**Version:** 1.1 (2026-11-11; see the change log).
**Role:** the canonical, detailed operational protocol and implementation reference for the common epoch-budget pilot of the WaveNet study.
**Authority and precedence:**
- The scientific rule is `models/hierarchical/wavenet_spec.md` §16.5 (APPROVED, Q7, Q12). Its formulas and extension rules are reproduced verbatim in §6–§7 and are **not** changed by this document. If this document and §16.5 ever disagree on the selection formula or the extension rule, §16.5 governs and the disagreement is a documentation error to be reported.
- The formal amendment that adopts this protocol is `wavenet_spec.md` v0.5 (§16.5.1 and the amendment log). Decisions, rationale, provenance and status are recorded in `PROJECT_RESEARCH_RECORD.md` (D-22, D-23, D-24, D-26).
- Approved operational decisions referenced here: DEC-1 – DEC-12, AM-1 – AM-7, clarifications 1–8, resolutions R1–R8, additional clarifications A and B (D-23), and the approved interpretations I-A (integrity mismatch), I-B (finiteness checks in every forward pass), I-C (G7 as an independent gate) and the category B resolution procedure (D-26) (all 2026-11-11; see §18).

### Labels

| Label | Meaning |
|---|---|
| **[ORIG]** | Original accepted rule (source given) |
| **[APPROVED]** | Newly approved operational decision (identifier given) |
| **[DERIVED]** | Mathematical or technical consequence of the rules or of the accepted code; not a new rule |
| **[IMPL]** | Requirement on the future implementation that follows from an approved decision |
| **[LIMITATION]** | Known constraint or weakness |

---

## 1. Objective and questions

- [ORIG] §16.5: all four arms use the **same** final `max_epochs`, determined by this procedure. Final training is not authorised until it has been run and recorded.
- **Primary question:** what is the smallest common training budget at which every arm reaches its §16.5 plateau at its accepted learning rate?
- **Descriptive (no decisions depend on them):** the shape of each arm's dev curve, the train/dev gap over training, and numerical and optimisation behaviour beyond 10 epochs.
- [ORIG] §16.4: this is pilot evidence. It is **not** an architecture comparison and answers no research question about architecture or normalisation.

## 2. Arms and fixed configuration

| Arm | Model and configuration | Parameters | η_a | Source |
|---|---|---|---|---|
| A | Model 3 MLP, `mlp.yaml` + `model.params.context_length=8` | 21,897 | 0.03 | §16.2; D-22 |
| B | Model 4 BN-MLP, `batchnorm_mlp.yaml` + `model.params.context_length=8` | 22,097 | 0.03 | §16.2; D-22 |
| C | WaveNet, `wavenet.yaml` (`normalization: none`) | 21,989 | 0.03 | §16.2; D-22 |
| D | WaveNet-BN, `wavenet_batchnorm.yaml` (`normalization: batchnorm`) | 22,397 | 0.03 | §16.2; D-22 |

- [ORIG] §16.3, §16.5 inputs: group split (`split_seed` 42); train-only vocabulary; `make_context_windows(k=8)`; `EpochSampler`, batch 32, final partial batch kept; plain SGD (momentum 0, no weight decay); mean reduction; **constant** learning rate; best checkpoint by full-dev NLL with the earliest-epoch tie-break; token-weighted NLL; `eval_batch_size` 4096; exact recalibrated statistics for BatchNorm arms; pilot seeds **100, 101, 102**; train and dev only.
- [ORIG] D-16 (A3): CPU, fixed thread count.
- [APPROVED] DEC-3, clarification 7: intra-op threads = **1**; inter-op threads frozen at the value recorded by the LR pilot (8). Both are frozen and blocking.
- [APPROVED] DEC-4: step diagnostics **option B**, identical to the LR pilot: `{"enabled": true, "every_n_steps": 100, "gradients": true, "parameters": true, "activations": false}`. Epoch-end diagnostics are part of the accepted trainers and always run.
- [ORIG] §16.4: no learning-rate schedule. Any schedule requires separate approval and a new pilot.
- [ORIG] D-22: the learning rates are fixed inputs. They are not re-selected in this pilot.

## 3. Training procedure

- [ORIG] §16.5 Step 1: each arm with each pilot seed is trained for `max_epochs = 30` (12 runs), recording the full-dev NLL `D_{a,s}(e)` at the end of every epoch e = 1, …, 30. This is the fixed-parameter, eval-mode quantity of `wavenet_spec.md` §8; BatchNorm arms are recalibrated first. Epochs are counted as completed epochs (1-based).
- [ORIG] §8: no early stopping. The accepted trainers write `model_best.pt` (strict improvement), `checkpoint_last.pt` (every epoch) and `model_final.pt` (after the last epoch); per-epoch order: best → history → last; then final.
- [APPROVED] DEC-5: no resume. Every attempt trains from scratch.
- [APPROVED] AM-1 (Option B): the accepted training functions `train_mlp`, `train_batchnorm_mlp` and `train_wavenet` are used unchanged, through new pilot code (§12.1). No model, utility, configuration or accepted script is modified.
- **Record conventions** [DERIVED from the LR-pilot records]: the `epoch` field of history rows and step records is **0-based**; `step` is the 0-based global optimiser step. Epoch e (1-based) in this protocol is history `epoch = e − 1`. There are 5,709 optimiser steps per epoch; with option B, step records exist for every global step divisible by 100 (58 records in the first epoch, 57 in each later one; 571 over 10 epochs).

## 4. Data protection (test split)

- [ORIG] §9, §16.5: the test split is never used in pilots.
- [APPROVED] DEC-11 (operational definition): the name splits may be formed only for the split-disjointness and vocabulary-coverage checks. **No test windows are ever constructed and no function evaluates the test split.** The summary script reads no test fields.
- [IMPL] An automated test asserts that the pilot's set-up path never calls `make_context_windows` on the test split and never evaluates it (§16).
- [APPROVED] **DEC-11 amendment A1** (approved in principle during implementation; documented 2026-11-12, D-28 in `PROJECT_RESEARCH_RECORD.md`). The operational definition above is unchanged except that it permits exactly **one further operation** on the test names, so three are permitted in total: (1) the split-disjointness check (`Splits.assert_disjoint`); (2) the vocabulary-coverage check (`check_vocab_coverage`); (3) **the split fingerprint** (`Splits.fingerprint`). Nothing else is permitted.
  - *Operation:* in each child process's set-up (`epoch_budget_pilot.build_arm`), after the splits are formed and checked, `Splits.fingerprint()` (`utils/data.py`) computes, for each of train, dev and test, the number of names and an order-sensitive SHA-256 of the name strings, together with the split seed, fractions and duplicate policy. Combined with `dataset_fingerprint("data/names.txt")`, it is written once into the run's `metadata.json` (field `dataset`) through `RunManager.create(dataset_fingerprint=…)`.
  - *Why:* provenance. It records, with every run, the sizes and digests of the partition that run used, without exposing the test names. It also keeps the budget-pilot run records identical in form to the closed LR pilot's: the accepted `lr_pilot.py` makes the same call, and every LR-pilot run records a fingerprint in the same form.
  - *What it permits, and what it does not do:* the recorded sizes and digests make it **possible** to compare the train/dev/test partition of any two runs, within this pilot or with the LR pilot. **No such comparison is performed by the implementation.** No integrity gate, summary, statistic, selection or decision reads the fingerprint. It is a record, not a check and not a gate. Partition identity is therefore not verified through the fingerprint.
  - *What is actually enforced (independently of the fingerprint, within this pilot only):* the partition is a deterministic function of `data/names.txt`, the code of `utils/data.py` and the `data` configuration (path, split fractions, split seed 42, duplicate policy). `data/names.txt` and every `utils/**/*.py` are blocking checksums of the frozen conditions (§10.1). The resolved configuration, including its `data` block, is compared with the frozen record by every command and by G2. A change to any of these inputs is refused. Nothing automatically compares the budget pilot's partition with the LR pilot's; G5 compares training trajectories, not partitions.
  - *Information recorded:* sizes and SHA-256 digests only. No test name, character, window, prediction, loss or metric is stored, printed or returned.
  - *Still prohibited:* constructing test windows; any forward pass, loss, NLL, metric or prediction on the test names; using test names or the fingerprint for training, evaluation, model selection, budget selection, hyperparameter decisions, diagnostics or summaries. The summary script still reads no test fields and does not load, split or fingerprint the data. The protected test evaluation (once per arm and seed after all decisions are frozen; `wavenet_spec.md` §16.6) is not affected by this amendment and remains unauthorised.
  - *Compliance:* implementation `training/wavenet_study/epoch_budget_pilot.py` (`build_arm`) and `utils/data.py` (`Splits.fingerprint`, `dataset_fingerprint`), both unchanged by this amendment; tests `tests/training/wavenet_study/test_epoch_budget_pilot.py::test_the_test_split_is_never_turned_into_windows` (windows only for train and dev; no direct test-split access in the runner source), `tests/training/wavenet_study/test_summarize_epoch_budget_pilot.py::test_summary_never_reads_or_builds_data_splits`, `tests/utils/test_data.py::test_fingerprints`.
  - [LIMITATION] No test asserts specifically that the runner records only sizes and digests for the test split; this is established by code inspection. The approved vocabulary-coverage check (operation 2), if it ever fails, names up to three example names of the offending split in its error message; with the accepted split it passes.

## 5. Run matrix and stages

### 5.1 Runs
- Cap 30: 4 arms × 3 seeds = 12 runs.
- Cap 60 (only for arms that require extension under §7 and only after authorisation): 3 runs per extended arm.

### 5.2 Stages (descriptive names; no numbering) — [APPROVED] DEC-8
1. **Validation stage:** seed 100 for arms A, B, C and D at cap 30.
2. **Validation review (human):** execution integrity, runner-side and summary-side G5, category A/B records, finiteness-check cost. No selection statistics are shown (status-only mode).
3. **Completion stage:** seeds 101 and 102 for all arms at cap 30.
4. **Selection review (human):** summary in selection mode (§13); e*, ε sources, extension requirements.
5. **Extension stage:** only for arms the rule marks as still improving at 30, and only on an explicit, separately authorised command.
6. **Final review (human):** common budget accepted, or the pilot declared inconclusive or halted.

Each stage starts only after explicit human authorisation of that stage.

## 6. Epoch-budget selection rule — verbatim from `wavenet_spec.md` §16.5 [ORIG]

(Step 1 is reproduced in §3 and Step 4 in §7 of this document. In the quoted text, "§16.6" refers to `wavenet_spec.md` §16.6.)

**Step 2: curve statistics,** per arm and epoch:

    D̄_a(e)  = (1/3) Σ_s D_{a,s}(e)                                (mean over the three seeds)
    SD_a(e) = sqrt( (1/2) Σ_s (D_{a,s}(e) − D̄_a(e))² )            (sample std, divisor n − 1 = 2)
    SE_a(e) = SD_a(e) / √3                                         (standard error at epoch e)
    D̄*_a    = min_e D̄_a(e);   ê_a = smallest e attaining D̄*_a     (ties → earliest)

**Step 3: selected epoch at the current cap.** With the tolerance evaluated **at the candidate epoch**, over the epochs e = 1, …, `E_cap` evaluated so far:

    ε_a(e) = max(1e-3, SE_a(e))
    e*_a   = min { e ≤ E_cap : D̄_a(e) ≤ D̄*_a + ε_a(e) }

`e*_a` is the earliest epoch whose mean dev NLL is within tolerance of the arm's best mean. It always exists, because `ê_a` satisfies the condition (`ε ≥ 1e-3 > 0`), and `e*_a ≤ ê_a ≤ E_cap`.

**Step 5: common budget.** `max_epochs = max_a e*_a` over all four arms. This single value is used by every arm in the final runs (§16.6), with the arm's own `η_a`. Taking the maximum means no arm is stopped before its own selected epoch. Final checkpoints are selected by best dev NLL (no early stopping), so arms with a smaller `e*_a` are not penalised by the extra epochs.

**Step 6: record.** For each arm: `η_a`, the per-seed curves `D_{a,s}(e)`, `D̄_a`, `SD_a`, `SE_a`, `ε_a`, `D̄*_a`, `ê_a`, `e*_a` at each cap evaluated, and whether the extension to 60 was used. Also the common `max_epochs`, or "inconclusive".

### 6.1 Approved operational rules for the computation (they do not change the formula)
- [APPROVED] DEC-12: an arm's statistics are computed only when **all three seeds** at the relevant cap are `completed` and pass every integrity gate (§10). No partial or two-seed statistics.
- [APPROVED] DEC-12, clarification 8: all calculations use the full-precision values stored in `history.jsonl` (`dev_nll`), never rounded values. Comparisons use `≤` exactly as written.
- [APPROVED] DEC-7: optimisation flags (dev-NLL rises, running-train rises, etc.) are reported descriptively and play **no** part in Steps 2–5.
- [APPROVED] clarification 6, R3 — **ε-source label**, for every arm and epoch:
  - `SE-derived` when `SE_a(e) > 0.001`;
  - `floor-derived` when `SE_a(e) < 0.001`;
  - `floor (SE equal)` when `SE_a(e) == 0.001` (exact float comparison).
  The formula is unchanged: `ε = max(0.001, SE)`.
- [APPROVED] clarification 6, R3: if any arm's selected epoch `e*_a` has an **SE-derived** label, the common budget requires explicit human review before acceptance. `floor-derived` and `floor (SE equal)` do not trigger this requirement.
- [APPROVED] R5, D-26: while any category B event (§9.3, §9.7) is `unresolved` or `rejected`, the common budget is **not calculated and not accepted** (§13.3).

## 7. Extension rules — verbatim from `wavenet_spec.md` §16.5 Step 4 [ORIG] (Q12)

"Still improving at the current cap" uses the **same** tolerance as Step 3:
- If `e*_a < E_cap`, the arm has reached an acceptable plateau. It needs no extension, and `e*_a` is its selected epoch.
- If `e*_a = E_cap`, no earlier epoch is within tolerance of the best mean, so the arm is still improving at the cap:
  - at `E_cap = 30`, **that arm** is re-run for `max_epochs = 60` with the same three seeds and its `η_a`, and Steps 2–4 are recomputed for it over e = 1, …, 60 (`E_cap = 60`);
  - at `E_cap = 60`, the budget pilot is declared **inconclusive**: all curves are reported, no budget is selected, and further approval is requested.
- **One extension at most** (30 → 60). The same rule is applied identically at both caps.
- The re-run starts from scratch: resume across a changed budget is rejected by design (§10). On CPU its first 30 epochs are bit-identical to the 30-epoch run, so the extension equals continued training; on accelerators they agree up to backend nondeterminism, and the 60-epoch curves are the ones used.
- Arms with `e*_a < 30` keep their 30-epoch curves and selected epochs.

Under this definition, a curve still falling by amounts below the tolerance counts as a plateau. For example, a mean curve falling 1e-4 per epoch over 30 epochs (with SE ≤ 1e-3) has `ê = 30` but `e* = 20 < 30`, so it is not extended.

(In the quoted text, "§10" and "§16.6" refer to `wavenet_spec.md` §10 (checkpointing and resume) and §16.6 (final runs).)

### 7.1 Approved operational rules for extensions
- [APPROVED] DEC-8: the rule determines **which** arms require extension; an extension is executed **only** on an explicit command after human authorisation.
- [APPROVED] clarification 5: no extension is performed after any category A event (§9.4).
- [APPROVED] DEC-10, R6 (G6, §10.3–§10.4): on CPU the bit-identity statement above is not assumed; it is verified for every extended run.
- The 30-epoch runs of an extended arm are kept as raw evidence; the 60-epoch curves are the ones used for that arm.

## 8. Derived consequences [DERIVED]

1. **Existence and order.** `e*_a` always exists and `e*_a ≤ ê_a ≤ E_cap` (stated in §16.5).
2. **Extended arms select in [30, 60].** Because `D̄*_60 ≤ D̄*_30`, any epoch `e < 30` that failed at cap 30 satisfies `D̄(e) > D̄*_30 + ε(e) ≥ D̄*_60 + ε(e)`, so it fails again at cap 60. Hence `e*_60 ≥ 30`.
3. **Range of a conclusive budget.** `max_epochs ∈ [1, 59]`; if any arm was extended, `max_epochs ≥ 30`. (`e*_a = 60` at cap 60 makes the pilot inconclusive.)
4. **A single deviating seed can make an epoch qualify.** If two seeds are at `x` and one at `x + Δ` with `Δ/3 > 10⁻³`, then `D̄ = x + Δ/3`, `SD = Δ/√3`, `SE = Δ/3`, and the condition `D̄ ≤ D̄* + SE` reduces to `x ≤ D̄*`, however large Δ is. This is why an SE-derived selected epoch requires human review (§6.1).
5. **Floor regime.** If `SE_a(e) ≤ 10⁻³` at every epoch, the rule reduces to a fixed absolute tolerance of 10⁻³ nats.
6. **The first k epochs do not depend on the cap.** In all three accepted trainers `max_epochs` is used only in `range(start_epoch, max_epochs)`, in the final `model_final.pt` save and in the result object; the learning rate is stateless and constant. Epochs 1…k of a run with cap 30 or 60 are therefore computed by the same operations as in a run with cap 10. This is the basis of G5 and G6.
7. **Non-finite updates reduce to non-finite parameters.** For plain SGD, `Δθ = −η·g` with `0 < η < 1`: a finite gradient gives a finite update. A non-finite update can arise only from a non-finite gradient (A2) or from overflow of `θ + Δθ`; the post-step parameter check (§9.2) detects both.
8. **Step-0 undefined ratios are structural.** Every arm initialises the output layer to zero (`W_o = 0`, `b_o = 0`; initial loss `ln 27`). Then `∂L/∂h = W_oᵀ δ = 0`, so every parameter upstream of the output layer has zero gradient and zero update at step 0 (`update_std = 0`), and `out.weight` and `out.bias` have zero pre-step norm. The shared utility (`utils/diagnostics/parameters.py`) returns `log10_update_to_param_std = None` unless both standard deviations are > 0. Observed in all 60 LR-pilot runs: this field is `null` at step 0 for **every** parameter of every arm, and nowhere else; `relative_update` of `out.weight` at step 0 is a finite, ε-regularised value of order 10¹⁰.
9. **Bit-identity is conditional** on the same software stack, platform and thread configuration; the frozen, blocking conditions enforce this (§10.1).

## 9. Stability and diagnostic policy

### 9.1 Categories — [APPROVED] clarification 4

| Category | Members | Consequence |
|---|---|---|
| **A. Mechanical instability** | A1 non-finite loss; A2 non-finite gradients; A3 non-finite parameter updates (detected as non-finite parameters after the step; §8, item 7); A4 non-finite recalibration statistics; A5 non-finite model activations (intermediate module outputs) | Run status `mechanically_unstable`; pilot halts (§9.4) |
| **B. Diagnostic computation failure** | A derived diagnostic that cannot be computed reliably, e.g. an undefined ratio not covered by §9.3 | Blocks acceptance; human review; **not** instability (§9.5) |
| **C. Ordinary finite observations** | Everything else, including the structural cases of §9.3 | Recorded descriptively; no effect on selection |

- [ORIG] carried over from the LR pilot (record §8.5, D-16): a non-finite recorded running-train, train or dev NLL is category A.
- [ORIG] D-18: a `FloatingPointError` outside the training phase, and any other exception, is a software `error`, not category A.
- No numerical thresholds are used anywhere in this classification.

### 9.2 Per-step finiteness checks — [APPROVED] R2, clarification B
[IMPL] The pilot code must:
1. check **every model parameter** for finiteness **immediately after every optimiser step** (global optimiser post-step hook);
2. check the output of **every leaf module** for finiteness **during every forward pass, training and evaluation alike** ([APPROVED] I-B; forward hooks): `emb`; `hidden` (A, B) or `hidden1`–`hidden3` (C, D); `bn` (B) or `bn1`–`bn3` (D); `act` / `act1`–`act3`; `out`. Each check runs on the module's output **before** the next module transforms it, so a non-finite pre-activation is detected before `tanh` (or BatchNorm) can mask it (e.g. `tanh(±∞) = ±1`). Reshapes and pair merges between modules do not alter values and need no separate check;
3. retain the accepted trainers' existing loss, gradient and recalibration checks unchanged;
4. be **read-only**: no clipping, replacement, rescaling or other numerical intervention, and no change to the optimiser, the model or the order of operations. In evaluation forwards (full-train and full-dev evaluation, epoch-end diagnostics, recalibration passes that call modules) the checks must not modify parameters, buffers or any training state (optimiser, step counters, RNG, BatchNorm EMA or inference statistics, `model.training`);
5. on detection, raise `FloatingPointError` immediately, recording the 1-based epoch, the 0-based global step, the parameter or module name, the kind of failure (`parameter_after_step` or `module_output`) and whether the forward was a training or evaluation forward; never wait for the next loss computation and never continue training (clarification B);
6. measure and report its own runtime cost: cumulative seconds spent in the checks and the number of checks, per epoch, in `timing.json` and in the summary.

G5 (§10.4) verifies that these checks do not change the numerical trajectory. A G5 mismatch is never resolved by weakening the equality requirement.

### 9.3 Category C structural cases (step-0 exception) — [APPROVED] R1
A null or ε-regularised derived value is category C **only** when one of these exact conditions holds:
1. **Step-0 update ratio.** Record with `step == 0` (and `epoch == 0`), field `updates.per_parameter.<p>.log10_update_to_param_std` is `null`, and in the same record either `update_std == 0.0` exactly or `param_norm == 0.0` exactly (the conditions of §8, item 8). This exception applies **exclusively to step 0**.
2. **Non-BN arms.** A history field `batchnorm` that is `null` in an arm whose `normalization` is `none` (arm C; arm A records no such field).

Further rules:
- Finite ε-regularised values (e.g. `relative_update` when `param_norm == 0.0` at step 0) are **preserved and reported** as recorded, labelled as ε-regularised; they are never reinterpreted, replaced or discarded.
- Any other null, missing or undefined derived diagnostic value is **category B**, including a `log10_update_to_param_std` null at any step > 0 and a `param_norm == 0.0` at any step > 0.

### 9.4 Response to category A — [APPROVED] clarification 5
- The affected run stops immediately (it raises inside training and is recorded as `mechanically_unstable`).
- The pilot **halts**: no further run is launched; every later `run` or extension command is refused.
- All completed runs and all records are preserved.
- Mechanically unstable runs are never retried.
- No epoch budget is computed from incomplete seed data, and no extension is performed.
- Human review is requested. There is no automatic fallback learning rate; any continuation requires a new, approved amendment.
- [DERIVED] An arm with an unstable seed has no `e*_a` (DEC-12); because the budget is a maximum over all four arms, no common budget can be computed.

### 9.5 Response to category B — [APPROVED] R5
- The diagnostic failure is recorded with its exact location (run, 1-based epoch, 0-based global step where applicable, field path).
- Training continues while the model's numerical state remains valid; scheduled runs continue.
- The common budget is not calculated or accepted while any category B event is `unresolved` or `rejected` (§9.7); human review is required before acceptance.
- If a failure prevents an integrity gate from being evaluated, that gate remains **unresolved** (not passed).
- Category B is never classified as mechanical instability.
- Every category B event is recorded and resolved under the procedure of §9.7.
- [DERIVED, LIMITATION] An **exception** raised inside the accepted trainers' diagnostic code cannot be caught without modifying those trainers; it ends the run and is classified as a software `error` (D-18). Category B therefore covers diagnostic values that are recorded as null or undefined.

### 9.6 Optimisation flags
- [APPROVED] DEC-7: dev-NLL rises (with epoch and magnitude), repeated rises, running-train rises, best epoch, and whether dev NLL is still improving at the cap are reported per run, descriptively.

### 9.7 Category B resolution procedure — [APPROVED] D-26
**Event record.** Every category B event receives a **unique identifier** and preserves:
- run (run directory and attempt id), arm, seed, cap; 1-based epoch and 0-based global step where applicable;
- diagnostic name (full field path);
- the observed value, or the undefined result (e.g. `null`, missing field, failed computation);
- the reason for classification (which rule of §9.3 the value fails);
- relevant evidence (the record line or file, and the values of related fields in the same record);
- initial status: **`unresolved`**.

**Resolution.** A human reviewer must explicitly record exactly one outcome:
1. **`accepted_as_expected`** — the evidence establishes that the observation is mathematically expected;
2. **`corrected`** — an identified software or diagnostic issue was corrected. Any new execution requires separate authorisation and must respect the frozen-condition rules (§10.1; a corrected execution-affecting file is a blocking checksum change) and the retry rules (§12.3);
3. **`rejected`** — the diagnostic result is invalid or cannot be trusted.

Every resolution preserves: reviewer identification; timestamp; rationale; evidence supporting the decision.

**Rules.**
- No category B event may silently disappear: every event, with its full history (initial record and resolution, if any), appears in every later summary.
- `accepted_as_expected` is not permission to ignore an unexplained numerical failure: it applies only to the specific event whose expected nature the evidence establishes, and never to a non-finite value (category A).
- **Unresolved or rejected category B events block acceptance** of the common budget (§13.3, §14). `accepted_as_expected` and `corrected` events do not block by themselves; a `corrected` event whose correction requires new evidence leaves acceptance blocked until that evidence exists under separately authorised execution.

[IMPL] Events are written append-only to `<pilot>/category_b_events.jsonl` (identifier, record fields, `status: unresolved`); resolutions are appended to `<pilot>/category_b_resolutions.jsonl` (event identifier, outcome, reviewer, timestamp, rationale, evidence) by an explicit command that requires all four fields. Neither file is ever rewritten. The summary script derives each event's current status from both files and reports every event.

## 10. Reproducibility, frozen conditions and integrity gates

### 10.1 Frozen conditions — [APPROVED] clarification 7, AM-7, DEC-1, DEC-2
The first `run` command writes `results/experiments/wavenet_study/epoch_budget_pilot/conditions.json`. It contains:
- the protocol: seeds, caps 30/60, the ε formula and labels, stage definitions, protocol version;
- `η_a = 0.03` per arm with provenance: D-22 and the SHA-256 of the LR pilot's `summary.json` (`5e52cc5bb8795b57daf640e4178724f1bf02186d392be83d1ac76be9c9eeccad`);
- SHA-256 of the 12 LR-0.03 reference runs' `history.jsonl`, `diagnostics/steps.jsonl` and `model_final.pt` (36 files) in the closed LR-pilot folder;
- every arm's fully resolved configuration (except the seed) and the expected trainer defaults;
- the diagnostics policy (option B) and the finiteness-check policy (§9.2);
- intra-op and inter-op thread counts;
- the software and hardware environment (Python, PyTorch, NumPy, PyYAML, platform, machine, CPU);
- checksums in two classes:
  - **blocking** (execution-affecting): `models/feedforward/mlp.py`, `models/feedforward/batchnorm_mlp.py`, `models/hierarchical/wavenet.py`, every `utils/**/*.py`, `configs/base.yaml`, the four arm configs, `data/names.txt`, `training/wavenet_study/study_common.py`, `epoch_budget_pilot.py`, `summarize_epoch_budget_pilot.py`;
  - **recorded, non-blocking** (documents): `wavenet_spec.md`, `mlp_spec.md`, `batchnorm_mlp_spec.md`, the UMIS, this protocol, `PROJECT_RESEARCH_RECORD.md`. A difference is reported, never blocking.

Every later command and every child process compares itself with the frozen record and refuses to run on any difference in a blocking item, the environment, the threads or the configuration.

The LR-pilot folder is **closed** (DEC-2): it is only ever read, never written, and no LR-pilot command is run again. Its own frozen conditions are not regenerated (§17).

### 10.2 State hashes — [APPROVED] AM-3, clarification 3, additional clarification A
- [IMPL] At the end of every epoch, **at exactly the point when the history row is written** (after epoch-end recalibration, evaluation and epoch-end diagnostics; per §3 write order, after `model_best.pt` and before `checkpoint_last.pt`), the runner records one SHA-256 over the model state in `<run>/state_hashes.jsonl` (`{"epoch": <0-based>, "sha256": ...}`).
- **Content:** every entry of `model.state_dict()` — all parameters and all (persistent) buffers, including BatchNorm EMA buffers (`running_mean`, `running_var`), inference statistics, `num_batches_tracked` and calibration counters.
- **Serialisation (deterministic):** entries in ascending order of their names; for each entry, the UTF-8 bytes of `name`, `dtype` and `shape`, each NUL-terminated, then the byte length of the tensor data as an 8-byte little-endian integer, then the exact bytes of `tensor.detach().cpu().contiguous()`.
- [IMPL] A test asserts that no model in the study has non-persistent buffers; otherwise the hash definition must be revisited before use.
- No additional checkpoints are written.
- **Reference states** (clarification A): references are derived from the corresponding model state only, never from checkpoint metadata or optimiser state.
  - G5 reference: the `model_state` stored in the LR-pilot run's `model_final.pt` (the state after epoch 10; nothing modifies the model between the epoch-10 history write and the final save), hashed with the same serialisation, after verifying the file against its frozen SHA-256.
  - G6 reference: the 30-epoch run's own recorded epoch-30 hash, cross-checked against the `model_state` of its `model_final.pt`.

### 10.3 Integrity gates

| Gate | Requirement | Source |
|---|---|---|
| G1 | History has exactly one row per epoch up to the run's cap, `epoch` = 0 … cap − 1 consecutively | LR pilot, carried over |
| G2 | Run configuration and checksums match the frozen record | LR pilot, carried over |
| G3 | Run metadata status is consistent with the attempt record | LR pilot, carried over |
| G4 | **Checkpoint loading and consistency with the recorded training history:** `model_best.pt`, `model_final.pt` and `checkpoint_last.pt` load (`weights_only`) and their recorded epoch, step, best metric and best epoch agree with `history.jsonl` | LR pilot, carried over |
| G5 | Epochs 1–10 of every budget-pilot run (cap 30 or 60) equal the LR-pilot run at LR 0.03 with the same arm and seed | [APPROVED] DEC-10, AM-2, R6 |
| G6 | Epochs 1–30 of every 60-epoch run equal the 30-epoch run with the same arm and seed | [APPROVED] DEC-10, AM-2, R6 |
| G7 | **State-hash records:** `state_hashes.jsonl` contains exactly one record for every completed epoch, and the recorded hash equals the hash (§10.2 serialisation) of the corresponding checkpoint's `model_state` wherever a checkpoint holds that epoch's state (`model_final.pt`: last epoch; `model_best.pt`: best epoch; `checkpoint_last.pt`: last completed epoch) | [APPROVED] I-C (independent gate) |

G4 and G7 are independent gates, evaluated and reported separately; neither is a sub-condition of the other.

### 10.4 G5 and G6 comparison definition — [APPROVED] AM-2, clarifications 2 and 8, R1, R6
For G5, k = 10; for G6, k = 30. Compared:
1. **History rows** with `epoch` 0 … k − 1: every leaf field except `seconds` (the only wall-clock field in history rows). The two rows must have identical sets of field paths.
2. **Step records** with `epoch` 0 … k − 1, matched by their explicit `(epoch, step)` identifiers, never by position or by inferred step counts. The sets of `(epoch, step)` identifiers must be identical, and every matched pair must have identical sets of field paths and identical values. (Step records contain no wall-clock fields; verified in the LR-pilot records.)
3. **State hash** after epoch k (0-based `epoch = k − 1`) against the reference state of §10.2.

Equality rules:
- Numbers: exact equality of the IEEE-754 float64 bit patterns as parsed from JSON (Python's float representation round-trips exactly), at full precision. Integers: exact equality. Strings and booleans: exact equality.
- **Non-finite values are rejected:** any NaN or ±Inf in a compared record fails the gate; NaN never matches NaN.
- **JSON nulls** match only where the null is a structural category C case of §9.3 at the same location in both records. A null anywhere else fails the gate (and is category B). A missing field on either side always fails.
- Run configuration and metadata may differ only in `training.max_epochs`, the pilot block, identifiers and timestamps; all other resolved configuration must be identical.

Two independent checks (R6):
- **Runner-side:** G5 is checked at the end of epoch 10 of every run; G6 at the end of epoch 30 of every 60-epoch run. On a mismatch the run stops immediately, all records are preserved, and the attempt is recorded with the explicit status `integrity_blocked`, including the first mismatching record and field; the overall sweep becomes `integrity_blocked` (§10.5).
- **Summary-side:** the summary script independently recomputes G5 and G6 from the saved histories, step records, state hashes and reference files. It never relies on a success flag written by the runner. A runner record and a summary result that disagree are themselves a blocking inconsistency.

A mismatch requires human review. There is no automatic retry.

### 10.5 Response to a G5/G6 integrity mismatch — [APPROVED] I-A
- The affected run stops immediately; its attempt is recorded as `integrity_blocked`.
- The overall sweep becomes **`integrity_blocked`**: no further run is launched and no subsequent stage proceeds until human review.
- All completed runs and all evidence are preserved.
- No automatic retry. A retry requires explicit authorisation, a written reason, and a fresh run from scratch under identical frozen conditions.

## 11. Diagnostics collected

| Diagnostic | Class | Source |
|---|---|---|
| Per-epoch running-train, full-train and full-dev NLL; best epoch | Essential | Accepted trainers (history) |
| Loss, gradient and recalibration non-finite guards | Essential | Accepted trainers |
| Per-step parameter and module-output finiteness checks (§9.2) | Essential | [APPROVED] R2 |
| Gradient norms and parameter-update statistics every 100 steps (option B) | Essential | [APPROVED] DEC-4 |
| BatchNorm EMA vs inference statistics, calibration counters (B, D) | Essential | Accepted trainers (epoch end) |
| Epoch-end dev activation, saturation, output and per-level summaries | Recorded | Accepted trainers (epoch end) |
| Per-epoch state hashes (§10.2) | Essential | [APPROVED] AM-3 |
| Timing and throughput; finiteness-check cost | Essential (operational) | Pilot code |
| Peak process RSS | Recorded, not analysed | Pilot code |
| Step-level activation statistics | Off | [APPROVED] DEC-4 (finiteness checks are not activation statistics) |

## 12. Execution, attempts, failure and retry

### 12.1 Code organisation — [APPROVED] AM-1, R7
Future implementation targets (**not yet created; creation not authorised**):
- `training/wavenet_study/study_common.py` — copy-based extraction of genuinely shared, pilot-independent functionality: git guard, checksum and JSON helpers, file lock, generic manifest and attempt store with a configurable key schema, step timer, failure classification, arm construction, a generalised single-run function parameterised by a pilot context (results folder, stage label, specification reference, checksum lists, record keys), state hashing, finiteness checks and G5/G6 comparison functions.
- `training/wavenet_study/epoch_budget_pilot.py` — orchestration and protocol configuration of this pilot only.
- `training/wavenet_study/summarize_epoch_budget_pilot.py` — status-only and selection summaries, independent gate recomputation.

`training/wavenet_study/lr_pilot.py` (SHA-256 `b77d76844a6c09dfd905e97a764bc929ab7591417ce08b6c55d2b1dfa95e27bc`) and `summarize_lr_pilot.py` (`6402c34366a681642a86c91233cbad0302f5264c8ea8e9af4bd1e37a01573a8d`) remain **byte-identical** and are **never imported** by the new pilot. Evidence for this choice (DEC-9 investigation): `lr_pilot.py` binds its attempt machinery to the LR-pilot folder through module globals, keys attempts by `(arm, lr, seed)` without a cap, launches children that train the hard-coded 10 epochs, and labels runs `"stage": "lr_pilot"`.

### 12.2 Results folder
`results/experiments/wavenet_study/epoch_budget_pilot/`: `conditions.json`, `manifest.jsonl`, `invocations.jsonl`, `attempts/`, `logs/`, `runs/<model_name>/run_NNN/`, `summary.md`, `summary.json`, `epochs.csv`, `.runner.lock`. Each run directory additionally holds `state_hashes.jsonl`, `timing.json` (including finiteness-check cost), `pilot_result.json` and the runner-side integrity record.

### 12.3 Attempts — [APPROVED] DEC-5 (adapted from D-18)
- Attempt key `(arm, cap, seed)`; attempt id includes the cap.
- Manifest rows `launched`, `finished`, `registered` (for attempts whose runner died); one child process per run, sequential, under an exclusive file lock; orphan run directories detected and blocking; every attempt kept and listed in summaries.

| Status | Meaning | Final? | Effect |
|---|---|---|---|
| `completed` | Training finished | yes | — |
| `mechanically_unstable` | Category A during training | yes; never retried | Pilot halts (§9.4) |
| `integrity_blocked` | Runner-side G5/G6 mismatch | no | Sweep becomes `integrity_blocked`; no later stage until human review; retry only with explicit authorisation, a written reason and a fresh run (§10.5) |
| `error` | Software or infrastructure exception (with phase and traceback) | no | Sweep stops; explicit reasoned retry |
| `crashed` | Child ended without a result record | no | Sweep stops; explicit reasoned retry |
| `interrupted` | Ctrl-C | no | Sweep stops; explicit reasoned retry |

- Retries: never automatic; require an explicit command, a written reason and matching frozen conditions; restart from scratch; `completed` and `mechanically_unstable` attempts are never retried.
- A software error is never interpreted as evidence about the learning rate or the budget.

## 13. Summary outputs

### 13.1 Modes — [APPROVED] DEC-12
- **Status-only mode** (default): attempts, statuses, integrity gates, category A/B records, epochs completed, timing and finiteness-check cost. No dev or train NLL, curves, curve-derived flags, best epochs, statistics or selections in any output.
- **Selection mode:** only with an explicit command-line flag, and only when its prerequisites hold: all required runs at the evaluated cap(s) `completed`; G1–G7 recomputed and passed; no category A event; no unresolved attempt. Category B events do not prevent the statistics from being displayed; `unresolved` or `rejected` events prevent the budget from being calculated or accepted (§9.7, §13.3).

### 13.2 Content
- `summary.json` (machine-readable), `summary.md` (review tables), `epochs.csv` (one row per arm, cap, seed and epoch).
- Per arm and epoch (clarification 6): `D̄`, `SD`, `SE`, `ε`, ε-source label, and each seed's `D_{a,s}(e)`.
- Per arm: `D̄*`, `ê`, `e*` at each cap evaluated, ε-source label at `e*`, extension required/used, per-seed best epochs and optimisation flags (descriptive).
- Gate results (G1–G7 reported separately) with the first mismatch for any failure; category A and C records with locations; ε-regularised values (§9.3); **every category B event** with its identifier, record fields and current status and resolution (§9.7).
- Timing per arm, including finiteness-check cost; peak RSS (recorded, not analysed).
- The common `max_epochs`, or one of: `inconclusive` (§7), `halted: mechanical instability` (§9.4), `withheld: unresolved or rejected category B` (§9.5, §9.7), `withheld: integrity` (§10), `requires human review: SE-derived e*` (§6.1).

### 13.3 Withholding rules
- No budget is calculated while: any arm lacks three valid seeds at its cap; any gate is failed or unresolved; any category A event exists; any category B event is `unresolved` or `rejected` (§9.7).
- A calculated budget with an SE-derived `e*` is labelled as requiring explicit human review.

## 14. Acceptance criteria

The common epoch budget is accepted only when **all** hold:
1. Every required run is `completed`; no unresolved attempt; no category A event.
2. G1–G7 pass, recomputed independently by the summary script, with runner-side and summary-side G5/G6 results in agreement.
3. No category B event `unresolved` or `rejected`; every category B event resolved under §9.7 and listed in the summary.
4. `e*_a` computed for every arm at its final cap; the pilot is not inconclusive.
5. Any SE-derived `e*_a` explicitly reviewed and accepted by a human.
6. Test-split protection verified (code inspection and automated test).
7. Finiteness-check cost reported.
8. Human approval of the common `max_epochs`, recorded as a new decision in `PROJECT_RESEARCH_RECORD.md` §11; the specification's pending budget value updated by amendment.

## 15. Known limitations [LIMITATION]

- Three seeds: SE is itself imprecise; a single deviating seed can drive the outcome (§8, item 4).
- The budget is selected on dev data; the protected test set mitigates but does not remove selection effects.
- Final runs use seeds 0–4, different from the pilot seeds; a final run's best epoch may lie beyond `max_epochs`.
- The learning rate was selected at 10 epochs and is not revisited at 30–60 epochs.
- The 10⁻³ floor is a convention, not a derived quantity.
- Bit-identity is guaranteed only on CPU with an identical software stack and thread configuration.
- Step diagnostics sample 1% of steps; per-step finiteness checks cover numerical failure, not optimisation behaviour.
- Exceptions inside the accepted trainers' diagnostics end a run as software errors (§9.5).
- Per-step finiteness checks add runtime; their cost is measured but timings are operational, not efficiency evidence.
- One dataset, one optimiser, constant learning rate, one context length.

## 16. Testing requirements for the future implementation — [APPROVED] R8

All tests use **synthetic fixtures and fault injection** in a separate workspace. Actual budget-pilot training is never used for testing. Required coverage:
1. **State hash and G7:** determinism across repeated calls; sensitivity to tensor names, ordering, shapes, dtypes and single-bit changes in tensor bytes; inclusion of every parameter and buffer (BatchNorm EMA and inference buffers, counters); absence of non-persistent buffers; hash timing at the history write. **G7 tested independently of G4:** a missing, duplicated or extra state-hash record, and a hash that disagrees with a checkpoint's `model_state`, each fail G7 while G4 still passes; conversely a checkpoint whose epoch, step or best metric disagrees with the history fails G4 while G7 still passes.
2. **G5/G6:** exact matching on identical records; synthetic single-field mismatches in history rows, step records (including a missing or extra `(epoch, step)`) and state hashes; NaN-vs-NaN and ±Inf rejected; structural nulls accepted only at §9.3 locations; arbitrary nulls and missing fields rejected; runner-side stop with `integrity_blocked`; summary-side independent recomputation, including disagreement with a forged runner success flag.
3. **Categories A/B/C:** injection of non-finite loss, gradients, parameters after a step, module outputs (including an infinite pre-activation before `tanh`) and recalibration statistics; exact epoch, step and name attribution; step-0 exception applied only at step 0 and only under its exact conditions; ε-regularised values preserved.
4. **Halt on category A:** the sweep stops immediately; no later run or extension can be launched; unstable attempts cannot be retried; no budget computed. **G5/G6 mismatch:** run stopped, sweep `integrity_blocked`, later stages refused until review, retry refused without authorisation and reason.
5. **Category B:** recorded with location and unique identifier; training and scheduled runs continue; budget withheld; never classified as instability; gates that cannot be evaluated remain unresolved. Resolution procedure: each outcome (`accepted_as_expected`, `corrected`, `rejected`) recorded only with reviewer, timestamp, rationale and evidence (missing fields refused); `unresolved` and `rejected` block acceptance; no event disappears from any summary; resolution files append-only.
6. **Read-only checks:** trajectories with and without the finiteness checks and state hashing are bit-identical on synthetic data; evaluation forwards with the checks leave parameters, buffers and training state unchanged; checks fire in evaluation forwards as well as training forwards.
7. **Selection mode:** refused without the flag; refused when any prerequisite of §13.1 fails; status-only outputs contain no learning metrics.
8. **ε-source labels:** `SE-derived`, `floor-derived` and `floor (SE equal)` on constructed values; review requirement triggered only by an SE-derived `e*`.
9. **Selection rule:** hand-computed synthetic curves for Steps 2–5, ties, the worked illustration of §16.5, extension at cap 30, inconclusive at cap 60, `e*_60 ≥ 30`.
10. **Test-split protection:** no test windows constructed, no test evaluation.
11. **Attempts:** registration, simultaneous invocations, interruption, crash, runner killed mid-run, orphan detection, explicit retry rules (reason required, completed/unstable refused).
12. **Frozen conditions and checksums:** blocking on code, configuration, data, thread or environment changes; documents recorded but non-blocking; reference-file hashes verified.
13. **Numerical parity:** exact parity of the new single-run path against reference runs produced by the accepted trainers on synthetic fixtures; parity of LR-pilot reference hashing.

## 17. Relation to the closed LR pilot

- The LR pilot (`results/experiments/wavenet_study/lr_pilot/`) is closed and immutable. Its frozen `conditions.json` records SHA-256 `915b1b3178bfb749df284e61b78515b665f506e8a6651704bfbdb75576ac01c2` for `models/hierarchical/wavenet_spec.md` (v0.4). After the v0.5 amendment that file no longer matches; this is an **accepted, documented consequence** of keeping the old experiment immutable. The LR pilot's frozen conditions are never regenerated or rewritten, and its scripts are not run again.
- The LR pilot's results, summaries, checkpoints, histories, logs and manifests are read only (as G5 references, verified against frozen hashes).

## 18. Decision provenance

| Identifier | Content | Where applied |
|---|---|---|
| §16.5 (Q7, Q12) | Selection formula, extension rule | §6, §7 |
| D-16 | Stability categories of the LR pilot; mechanical-instability definition | §9.1 |
| D-18 | Retry policy, blocking on mismatches, code freeze | §10.1, §12.3 |
| D-22 | η = 0.03 for A–D | §2 |
| DEC-1 | Documents recorded, non-blocking; code, configs, data blocking | §10.1 |
| DEC-2 | New results folder; LR pilot closed | §10.1, §17 |
| DEC-3 | Intra-op threads 1, environment identical to the LR pilot | §2 |
| DEC-4 | Diagnostics option B | §2 |
| DEC-5 | Explicit retries from scratch, no resume | §3, §12.3 |
| DEC-6 / clarification 5 | Category A halts the pilot | §9.4 |
| DEC-7 | Flags descriptive only | §6.1, §9.6 |
| DEC-8 | Stage gates; extensions only after authorisation | §5.2, §7.1 |
| DEC-9 / AM-1 / R7 | Option B code organisation; file names | §12.1 |
| DEC-10 / AM-2 / R6 | G5, G6; runner- and summary-side checks | §10.3–10.5 |
| I-A (D-26) | G5/G6 mismatch: run stops, sweep `integrity_blocked`, retry only explicit | §10.5, §12.3 |
| I-B (D-26) | Finiteness checks in every training and evaluation forward; evaluation read-only | §9.2 |
| I-C (D-26) | G7 independent of G4 | §10.3, §16 |
| D-26 | Category B resolution procedure | §9.7, §13, §14 |
| DEC-11 | Test-split protection | §4 |
| DEC-11 amendment A1 (D-28) | Split fingerprint as the third and last permitted operation on the test names; provenance only | §4 |
| D-30 | A1 wording correction: the fingerprint permits a partition comparison, but none is performed and it is no gate; the partition's inputs are gated within this pilot | §4, change log |
| DEC-12 | Three seeds required; full precision; explicit selection flag | §6.1, §13.1 |
| AM-3 / clarification 3 / A | State hashes and their timing | §10.2 |
| AM-4 / clarification 4 | Categories A/B/C | §9.1 |
| AM-5 / clarification 5 | Halt on category A | §9.4 |
| AM-6 / clarification 6 / R3 | ε reporting, labels, SE-derived review | §6.1 |
| AM-7 / clarification 7 | Frozen conditions content | §10.1 |
| Clarification 8 | Explicit `(epoch, step)` matching; full precision | §10.4 |
| R1 | Step-0 structural exception | §9.3 |
| R2 / clarification B | Per-step finiteness checks; attribution | §9.2 |
| R4 | This file is the canonical protocol | header |
| R5 | Category B blocks acceptance, not execution | §9.5 |
| R8 | Testing requirements | §16 |

## Change log
- **1.0 (2026-11-11):** initial approved protocol (D-23). Documentation only; no implementation or execution.
- **1.1 (2026-11-11):** approved interpretations I-A, I-B, I-C and the category B resolution procedure (D-26): §9.2, §9.5, new §9.7, §10.3 (G4 and G7 defined separately), §10.5, §12.3, §13.2, §13.3, §14, §16, §18. §16.5 formulas and extension rules unchanged. Documentation only; no implementation or execution.
- **1.1 addendum (2026-11-12):** DEC-11 amendment A1 (D-28): §4, §18. Documents an operation the accepted implementation already performs; no code changed, and operational content, formulas and stage rules are unchanged. The version number stays 1.1 because the accepted implementation records protocol version 1.1 in its frozen conditions. The status line, §12.1 and §16 still describe the implementation as future work; it has been implemented and accepted in five stages, the last on 2026-11-12 (D-27). That wording is retained, not rewritten (see `PROJECT_RESEARCH_RECORD.md` §12). No pilot run has been performed.
- **1.1 addendum, wording correction (2026-11-12, D-30):** in the DEC-11 amendment A1 (§4), two sentences overstated what the fingerprint does, so they were replaced.
  - The *Why* item said the fingerprint "ties every run to the exact partition it used and records that the protected partition is present and unchanged". It records the partition's sizes and digests; it does not establish that anything is unchanged.
  - The item "*What it is used to verify:* identity of the train/dev/test partition across runs and pilots, by comparison of the recorded sizes and digests" is replaced by "*What it permits, and what it does not do*": the fingerprint permits such a comparison, but no comparison is performed, and it is not a gate.
  - A new item, "*What is actually enforced*", states the existing blocking-checksum and G2 controls on the inputs of the partition, within this pilot only.

  The permitted operation and the A1 scope are unchanged; no code was changed.

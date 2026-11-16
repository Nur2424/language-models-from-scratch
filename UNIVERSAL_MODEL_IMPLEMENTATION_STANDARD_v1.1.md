# Universal Model Implementation Standard (UMIS)
 
**Project:** `language-models-from-scratch`
**Status:** Approved project standard. It is internal to this repository, not an external or community standard.

## 1. Purpose

This standard defines the mathematical, implementation, testing, and experimental requirements for every model in this repository. The project aims to understand and independently implement the defining mechanisms of language-model architectures—not merely call high-level framework APIs.

Every implementation should make clear what computation the architecture performs, why it is structured that way, how information and gradients flow, how parameters are initialized, what numerical/computational limitations exist, and whether the implementation behaves as its mathematical specification predicts.

**Guiding principle: Understand the mechanism. Implement the mechanism. Verify the mechanism.**

## 2. Scope and philosophy

### 2.1 What we implement

We independently implement each architecture's defining mechanisms: for example, recurrent state updates and gates, attention, positional information, temporal convolutions, state-space recurrences, and mixture-of-experts routing. The implementation must expose the mechanism being studied.

### 2.2 What we do not reinvent

PyTorch is the numerical computing and automatic-differentiation foundation. We do not reimplement tensor storage, tensor arithmetic, matrix multiplication, autograd, backpropagation engines, device runtimes, GPU kernels, or standard optimizers when the optimizer itself is not the subject of study.

Use low-level PyTorch primitives where appropriate, without hiding the architecture's central computation behind a packaged module.

### 2.3 Correctness before optimization

The first implementation prioritizes clarity, mathematical transparency, and testability over speed. Fused or optimized alternatives may be explored later, but must not silently replace the educational implementation or obscure its core mechanism.

## 3. Repository boundaries

| Location | Responsibility |
|---|---|
| `models/` | Architecture implementations, their specifications (`*_spec.md`), model-specific configuration classes, and each model's reference training procedure and diagnostics |
| `utils/` | Shared vocabulary, data, device, reproducibility, metrics, checkpoint, generation, diagnostics, and run utilities |
| `configs/` | Shared (`base.yaml`) and model-specific (`models/*.yaml`) configuration files |
| `data/` | The names dataset |
| `training/` | Study orchestration: pilots, final runs, run verification, protected test evaluation, and analysis for each controlled study |
| `tests/` | Automated utility, model, architecture-specific, and study-infrastructure tests |
| `results/` | Run metadata, metrics, analysis, and generated samples |

Model scripts are implemented first. Notebooks are a later phase and are outside an individual model task unless explicitly requested.

## 4. PyTorch abstraction policy

### 4.1 Generally permitted

Subject to each model's specification:
- `torch.Tensor` and tensor creation functions.
- Tensor arithmetic, indexing, slicing, reshaping, broadcasting, reductions, and concatenation.
- Low-level operations such as `torch.matmul`, `torch.einsum`, suitable `torch.nn.functional` primitives, and stable numerical functions.
- `torch.autograd`, parameter registration, and `nn.Module`.
- Standard optimizers such as those in `torch.optim`.
- CPU, CUDA, and MPS execution.
- Standard serialization primitives through shared checkpoint utilities.

A low-level primitive must not be used as a disguised packaged implementation of the architecture.

### 4.2 Generally prohibited shortcuts

Unless a model-specific specification explicitly justifies an exception, do not use high-level modules that conceal the mechanism being implemented. Examples:
- `nn.RNN`
- `nn.GRU`
- `nn.LSTM`
- `nn.MultiheadAttention`
- `nn.TransformerEncoder`
- `nn.TransformerDecoder`

Every model specification must list its exact permitted and prohibited modules and operations.

### 4.3 Functional operations

Functional operations are acceptable when they represent low-level mathematical primitives. For example, manually implemented attention may use matrix multiplication and softmax, but must explicitly construct projections, attention scores, masking, weighting, and output projection.

## 5. Hardware independence and device policy

All implementations must be hardware-independent.

- Never hardcode `.cuda()`, `.to("cuda")`, `.to("mps")`, or a particular device as a global assumption.
- Use the shared device utility, including `resolve_device("auto")`, where device resolution is needed.
- CPU is the baseline; support CUDA and Apple MPS when available and required operations are supported.
- Create tensors on the appropriate device; keep inputs, parameters, buffers, and intermediate tensors device-compatible.
- Avoid hidden device transfers inside `forward`.
- Document backend limitations and provide alternatives where practical.

Floating-point results may differ slightly across CPU, CUDA, and MPS. Reproducibility is expected within a fixed hardware/software environment; bitwise identity across backends is not required.

## 6. Model API and tensor contracts

### 6.1 Neural models

Neural models should normally subclass `torch.nn.Module`. Their primary computation is exposed through `forward`, which returns **logits** unless a justified architecture-specific contract says otherwise. Loss computation, optimizer steps, training loops, and general evaluation orchestration remain outside the model class (see §12).

### 6.2 Statistical models

Statistical models are not forced into `nn.Module`. They use a standalone interface appropriate to their estimation procedure, learned state, prediction, evaluation, and generation.

### 6.3 Shape conventions

Unless the architecture requires and documents another representation:
- Context inputs: documented context representation, commonly `[B, V]`.
- Sequence token inputs: `[B, T]`.
- Sequence logits: `[B, T, V]`.

`B` is batch size, `T` sequence length, and `V` vocabulary size. Every public method documents shapes, dtypes, devices, and semantic meaning.

### 6.4 Input/output contract

Each specification states accepted input representation, dtype/device, variable-length support, padding and mask semantics, length handling, output shape/dtype, and whether outputs are logits, probabilities, log-probabilities, states, or another quantity.

## 7. Vocabulary and data conventions

Use the shared `Vocabulary` utility and current character-level names-dataset conventions:
- `.` is the shared BOS/EOS token, ID `0`.
- Current output vocabulary size is 27.
- Padding is input-only, with PAD ID `27`; it is not an output class.
- `IGNORE_INDEX = -100` is used for ignored target positions where applicable.
- Vocabulary construction and model fitting use training data only.
- Unknown characters in validation/test data must be handled explicitly; never silently extend vocabulary using held-out data.

Use shared vocabulary and sequence utilities instead of duplicating their logic. Document and justify deviations.

## 8. Mathematical specification before coding

Before implementation, define:
1. Exact architecture variant and purpose.
2. Defining equations and component roles.
3. Forward data flow and important tensor dimensions.
4. Trainable parameters and parameter-sharing rules.
5. Initialization choices and rationale.
6. Masking, padding, and sequence-boundary semantics.
7. Computational complexity and limitations.

The implementation must match the written specification. If the design changes, update the specification and tests.

## 9. Initialization and parameterization

Initialization must be architecture-specific and justified. State the method for each important parameter group and explain its rationale, particularly for signal propagation, recurrent stability, residual paths, and attention scaling. Do not rely blindly on defaults. Document parameter sharing and test that shared parameters are genuinely shared. Keep initialization out of `forward`; never reseed global RNG state inside constructors or `forward`.

Use shared reproducibility utilities and independent random streams where appropriate.

## 10. Numerical stability and precision

Consider stable softmax/log-softmax, logarithms of zero, division by zero, empty reductions, overflow/underflow, exponential saturation, masking values, fully masked rows, NaN/Inf propagation, accumulation precision, and backend-specific behaviour.

Use `float32` as the initial neural-model default unless the specification justifies otherwise. Mixed precision is a future optional optimization, not a baseline requirement. Do not silently clamp invalid values without documenting the mathematical consequences.

## 11. Configuration

Model-specific settings must be explicit and validated. Use shared configuration infrastructure and YAML conventions. Document defaults, valid ranges, divisibility constraints, incompatible options, and the rationale for meaningful defaults. Avoid unexplained magic constants. Keep architecture configuration separate from experiment/training configuration where appropriate. Save configuration with checkpoints and run metadata.

## 12. Training and evaluation boundaries

Model scripts define model behaviour, not the full experiment pipeline. General training loops, optimizer orchestration, dataset downloading/splitting, sweeps, multi-seed orchestration, general evaluation pipelines, reporting, plotting, and notebook explanations belong outside individual model scripts.

Small model-specific helpers are acceptable when intrinsic to the architecture. Avoid duplicating general infrastructure.

Each model keeps one minimal reference training procedure next to its architecture: a module-level function in the model file (for example `train_mlp`) or a companion module (for example `rnn_training.py`). It is separate from the model class and fits a single run. Everything that spans runs (pilots, seed loops, frozen conditions, verification, test evaluation, analysis) lives in `training/`.

## 13. Dropout, normalization, and optimization

- Dropout must respect train/eval mode and be disabled during evaluation and generation.
- Implement normalization mechanisms explicitly when understanding the mechanism is part of the project. The model specification states which mechanisms are manual and which standard primitives are allowed.
- Standard PyTorch optimizers are permitted initially.
- Gradient clipping may be configured by future training infrastructure; it does not belong inside `forward`.
- Document the purpose and effect of optimization choices.

## 14. Reproducibility and randomness

Use shared reproducibility utilities. Do not reseed global RNG state inside model construction or `forward`. Keep initialization, data-ordering, and generation randomness conceptually separate where supported. Generation/sampling should accept an explicit generator or use shared utilities. Record seeds and environment details. Document nondeterminism, including backend limitations.

## 15. Diagnostics

Use shared diagnostics where useful: parameter counts/summaries, gradient norms, activation statistics, non-finite checks, and parameter-update measurements. Add architecture-specific diagnostics where informative (for example, gate statistics, attention statistics, recurrent-state norms, expert load balance, or state stability).

Diagnostics should be optional, avoid retaining computation graphs or large tensors, and not materially alter model behaviour.

## 16. Checkpoints and model state

Neural checkpoints should use shared checkpoint infrastructure and preserve model parameters/buffers, configuration, relevant vocabulary/token metadata, and training state when resumability is requested. Statistical models save learned state and interpretation metadata, such as vocabulary ordering, smoothing settings, and fitted counts/probabilities.

Loading validates compatibility and raises clear errors. Test checkpoint round trips.

## 17. Testing and verification

Every model requires general and architecture-specific tests.

### 17.1 General tests

Where applicable, test valid construction, invalid configuration, shapes, dtypes, devices, vocabulary/special tokens, padding/masks, finite outputs and gradients, parameter counts/trainability, train/eval behaviour, checkpoint compatibility, controlled-seed determinism, CPU, and supported accelerators including MPS where available.

### 17.2 Mathematical and architecture-specific tests

Verify the defining mechanism, not merely that the model runs. Depending on architecture, test hand-computed numerical references, probability normalization, equivalence of sequential/parallel formulations, causality/no future leakage, masking, finite-difference gradients, state transitions, gate/routing behaviour, parameter sharing, and edge cases.

### 17.3 Learning and behaviour checks

Neural models should be tested on a deliberately small batch or tiny dataset to verify that they can overfit it under a reasonable setup. This is a debugging check, not a performance claim.

Statistical models require checks against manually computed counts, probabilities, likelihoods, and generation behaviour. Unseen events must be handled and tested explicitly.

### 17.4 Test discipline

Tests should be deterministic where practical, use small fixtures, and not depend on external downloads. Shape tests alone are insufficient. Do not proceed while material failures remain unexplained.

## 18. Fair model comparison

Perfect fairness between different architectures is impossible because their computational structures and inductive biases differ. Use controlled protocols and report limitations honestly.

### 18.1 Shared controls

Where applicable, use the same dataset, fixed train/dev/test splits, duplicate policy, character vocabulary/tokenization, token-weighted NLL definition, held-out test set, dev-based checkpoint selection, generation protocol, and multiple independent seeds. Report mean and standard deviation. Record tokens processed and actual computational cost. All data-dependent fitting must respect split boundaries.

### 18.2 Two complementary protocols

**Parameter-matched:** compare configurations with approximately comparable parameter counts. This asks how architectures use similar parameter budgets.

**Compute-aware:** compare under a documented, approximately comparable computational budget. This asks how performance relates to computational cost. Equal training steps must not be assumed to mean equal compute.

These protocols answer different questions and must not be conflated.

### 18.3 Best-achievable performance

Architecture-specific tuning may be reported separately and clearly distinguished from controlled comparisons. Do not claim universal superiority from one budget, dataset, or tuning procedure.

### 18.4 Experimental transparency

Record at least:
- Model name and architecture variant.
- Parameter count.
- Dataset and split identifiers.
- Random seed.
- Device, hardware/backend, and relevant software environment.
- Configuration.
- Training tokens processed and steps.
- Elapsed time and throughput where available.
- Peak memory where available.
- Best development NLL and selection information.
- Final test NLL, evaluated only after model selection.

When results differ, investigate architecture, parameter count, training duration, optimization, tuning, random variation, and hardware/numerical differences. Distinguish observations from explanations. Exact budgets and tuning ranges will be decided after implementations and computational profiles are understood.

## 19. Code quality

Prefer readable, explicit, testable code; clear names and public interfaces; small coherent functions; useful type hints and error messages; and comments that explain reasoning or non-obvious choices.

Avoid overengineering, hypothetical frameworks, unnecessary inheritance, duplicated utilities, global mutable state, mega-functions, opaque cleverness, and premature optimization.

## 20. Model implementation workflow

For each architecture:
1. Inspect relevant historical implementation and references.
2. Discuss architecture and mathematical foundations.
3. Agree on exact variant and model-specific specification.
4. Define permitted/prohibited PyTorch abstractions.
5. Prepare a focused coding-agent brief.
6. Implement only the current model in its designated script.
7. Review actual source against the specification.
8. Add and run automated tests.
9. User runs implementation locally.
10. Verify numerical behaviour, gradients, edge cases, and supported devices.
11. Fix and retest material issues.
12. Obtain explicit user approval before starting the next model.

One model at a time. Do not proceed merely because a script runs.

## 21. Acceptance criteria

A model is complete only when its specification is reviewed; the defining mechanism is explicit; code follows this standard; code review finds no unresolved material issue; general and architecture-specific tests pass; numerical and edge-case behaviour is examined; CPU and available accelerator checks are completed; configuration/checkpoint behaviour is verified where applicable; and the user explicitly approves completion.

## 22. Guiding principle

> Understand the mechanism. Implement the mechanism. Verify the mechanism.

Plausible output alone does not establish correctness. Every architecture must be supported by mathematical reasoning, explicit code, and tests of its defining behaviour.
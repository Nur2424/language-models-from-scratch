# Model Specification Template

**Purpose:** The common structure of every model specification in this repository. One copy is completed per architecture, before any code for that architecture is written.  
**Governing standard:** `UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md` (UMIS, the project's internal implementation standard)

## How this template is used

- A specification is a copy of this file saved next to the model as `models/<family>/<model>_spec.md`.
- All 18 sections are kept, with their numbers, so that specifications can be compared section by section and cited as "§n". A section that does not apply is marked N/A with a one-line reason.
- The specification is written and reviewed first. Implementation starts only when no essential section is vague, and consequential choices are settled in §18, not during coding.
- The specification stays authoritative after implementation: if the design changes, the specification and the tests change with it (UMIS §8).

**Labels.** Specifications mark their statements so that mathematics, requirements and measurements are never confused:

| Label | Meaning |
|---|---|
| **[T]** | Theory: follows from the equations under the stated assumptions |
| **[I]** | Implementation requirement: must hold for the code and is verified by a test |
| **[O]** | Observation: a measured value, with its source stated |
| **[H]** | Hypothesis: untested, never a finding |

## 1. Model identity

- **Model name:**
- **File path:**
- **Model family:**
- **Exact architecture variant:**
- **Specification version:**
- **Reference implementation(s):**
- **Shared utility dependencies:**

## 2. Role and motivation

### 2.1 Problem addressed
The modelling problem and the role of this architecture in it.

### 2.2 Why this architecture?
What it contributes to the model progression and which mechanism it makes available for study.

### 2.3 Learning and research questions
The questions that the implementation and its experiments should help answer. Established properties are kept separate from hypotheses.

## 3. Exact architecture definition

- **Architecture definition:**
- **Included components:**
- **Excluded components:**
- **Causal or bidirectional behaviour:**
- **Autoregressive or non-autoregressive behaviour:**
- **Deliberate simplifications:**

A broad label ("RNN", "Transformer") is not a definition: this section names the actual variant and explains why it was selected.

## 4. Mathematical formulation

All symbols are defined before use.

### 4.1 Core equations
The equations that define the architecture.

### 4.2 Component equations
The operation and role of each major component.

### 4.3 Parameterization
The trainable parameter groups, their dimensions and their roles.

### 4.4 State and information flow
How information moves through the model: recurrent state, residual paths, memory, attention, routing or receptive fields, as applicable.

### 4.5 Assumptions and limitations
Assumptions, approximations and known limitations.

## 5. Tensor and interface contract

| Item | Shape | Dtype | Device | Meaning |
|---|---|---|---|---|
| Input | | | | |
| Intermediate 1 | | | | |
| Intermediate 2 | | | | |
| Output | | | | |

Also specified here:
- **Batch / sequence / vocabulary dimensions:**
- **Input token representation:**
- **Output representation:** logits / probabilities / log-probabilities / other
- **Variable-length sequence support:**
- **Padding convention:**
- **Mask convention:**
- **Length handling:**
- **Special-token handling:**
- **Device behaviour:**

Neural sequence models return logits of shape `[B, T, V]` unless a justified exception is documented. Statistical models define a standalone interface suited to their estimation procedure.

## 6. PyTorch implementation policy

### 6.1 Permitted operations and modules
The exact primitives that may be used, with the reasoning behind the important choices.

### 6.2 Prohibited operations and modules
The high-level modules and shortcuts that would conceal the mechanism under study.

### 6.3 Manual implementation requirements
The computations that must be written out explicitly.

### 6.4 Shared utility integration
The existing utilities the model relies on: `Vocabulary`, data and sequence builders, device resolution, reproducibility and RNG streams, generation and sampling, metrics, diagnostics, checkpoints and configuration. Shared functionality is not duplicated without a stated reason.

## 7. Initialization and parameter sharing

| Parameter group | Shape | Initialization | Rationale |
|---|---|---|---|
| | | | |

This section also covers bias and embedding initialization, residual- or recurrent-specific initialization, parameter tying, parameter constraints and any dedicated RNG requirements.

## 8. Forward-pass specification

The execution order of the forward pass. For each stage:
1. Operation.
2. Input shape.
3. Output shape.
4. Parameters used.
5. Masking and boundary behaviour.
6. Numerical considerations.

Pseudocode may be included; it must agree with the equations of §4.

## 9. Configuration

| Field | Type | Default | Validation | Meaning |
|---|---|---|---|---|
| | | | | |

Required and optional fields, valid ranges, divisibility constraints, incompatible options, the rationale for each default, and which settings are architectural as opposed to training or experimental. No constant is left unexplained.

## 10. Numerical stability and edge cases

Risks to consider:
- Softmax/log-softmax stability.
- Logarithms of zero and zero-probability events.
- Division by zero and empty reductions.
- Empty sequences or batches.
- Fully masked positions.
- Very long sequences.
- Exponential/recurrent-state growth or decay.
- Overflow, underflow, NaN, and Inf.
- Precision and backend support.
- Boundary conditions and sequence starts/ends.

For each edge case that matters, the specification states the intended behaviour and the test that checks it.

## 11. Complexity and resource profile

- **Parameter count:** formula or method.
- **Time complexity:** per forward pass and relevant dimensions.
- **Memory complexity:** activations/intermediates.
- **Sequence-length scaling:**
- **Main bottlenecks:**
- **Device considerations:**

Theoretical complexity is kept separate from measured runtime and memory.

## 12. Diagnostics

The general and architecture-specific diagnostics that are useful for this model: parameter and gradient norms, activation statistics, recurrent-state norms, gate distributions, attention statistics, expert load balance, state stability, and so on. For each one: when it is collected, and how it avoids retaining computation graphs or large tensors.

## 13. Test plan

### 13.1 General tests

- [ ] Valid construction and configuration.
- [ ] Invalid configuration raises clear errors.
- [ ] Input/output shapes and dtypes.
- [ ] Vocabulary and special-token behaviour.
- [ ] Padding, lengths, and masks.
- [ ] Finite outputs.
- [ ] Finite gradients for neural models.
- [ ] Expected parameter count and trainability.
- [ ] Train/eval behaviour where applicable.
- [ ] Checkpoint save/load.
- [ ] Controlled-seed reproducibility.
- [ ] CPU execution.
- [ ] CUDA/MPS execution where available and supported.

### 13.2 Mathematical verification

Small, decisive tests that verify the equations themselves: hand-computed references, comparison with an independent reference implementation, probability normalization, sequential/parallel equivalence, causality (no future leakage), masking, finite-difference gradients, state transitions, routing invariants, parameter sharing and edge cases, as applicable.

**Model-specific mathematical tests:**
1.
2.
3.

### 13.3 Learning and behaviour checks

For a neural model: a small-batch overfitting check. For a statistical model: hand-computed count, probability and likelihood checks, and explicit unseen-event behaviour.

**Expected result and tolerance:**

## 14. Checkpoint and serialization contract

The learned state, the configuration and vocabulary metadata that must travel with it, buffers and other non-parameter state, resumable training state where relevant, compatibility validation, and the round-trip tests. Shared checkpoint infrastructure is used where applicable.

## 15. Generation or inference behaviour

If applicable:
- Inference entry point.
- Prediction procedure.
- Start/end-token behaviour.
- Sampling method and supported controls.
- Explicit RNG/generator handling.
- Evaluation-mode requirements.
- Maximum sequence length and stopping conditions.
- Invalid or degenerate distribution behaviour.

If not applicable, the reason is stated.

## 16. Implementation scope and boundaries

The implementation of a specification is limited to the model-specific code in its designated file or files, plus narrowly scoped tests and configuration.

The following are outside a model's scope unless the specification explicitly includes them: general training loops, dataset downloading or splitting, sweeps, multi-seed runners, general evaluation and reporting, plotting, notebooks, duplicated shared utilities, generic frameworks and changes to other models. (A model's own reference training procedure is in scope; see UMIS §12.)

**Necessary supporting-file changes and justification:**
1.
2.

## 17. Acceptance criteria

- [ ] Exact architecture variant implemented as specified.
- [ ] Defining mechanisms explicit and inspectable.
- [ ] Public API and tensor contracts documented.
- [ ] Configuration validated.
- [ ] Initialization justified.
- [ ] Numerical and edge-case behaviour handled.
- [ ] Shared utilities used appropriately.
- [ ] General tests pass.
- [ ] Architecture-specific mathematical tests pass.
- [ ] Checkpoint/inference behaviour verified where applicable.
- [ ] CPU and available accelerator checks completed.
- [ ] No unresolved material code-review issue.
- [ ] Implementation run and reviewed locally by the project author.
- [ ] Completion explicitly approved by the project author before the next model is started.

## 18. Open decisions

Every consequential architectural choice is recorded here with its options and its decision. None is made silently during coding.

| Question | Options | Recommendation | Decision |
|---|---|---|---|
| | | | | 

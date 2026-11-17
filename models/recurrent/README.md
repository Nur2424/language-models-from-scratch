# Vanilla RNN

A single-layer Elman recurrent network that reads a whole name one character at a time and carries a hidden state from each character to the next. Unlike the fixed-window models, its prediction can in principle depend on every earlier character of the name.

- Code: [`rnn.py`](rnn.py) (model), [`rnn_training.py`](rnn_training.py) (reference training procedure), [`rnn_diagnostics.py`](rnn_diagnostics.py) (research diagnostics)
- Design documents: see [Documents](#documents) below
- Tests: [`tests/models/recurrent/`](../../tests/models/recurrent/), including an independent reference implementation and full-size tests in `slow/`
- Results: the [RNN study](../../training/rnn_study/)

## The model

For one name, starting from h₀ = 0:

```latex
a_t = U[x_t] + W h_{t-1} + b, \qquad h_t = \tanh(a_t), \qquad z_t = V h_t + c
```

- x_t is the current character (the first input is the start token `.`), h_t the hidden state, z_t the logits for the next character, or for the end of the name.
- **Parameters:** U [27 × 128] maps each character straight into the hidden space, with no separate embedding; W [128 × 128] is the recurrent matrix; b, V [27 × 128] and c complete it. Total **23,451** = H² + 55H + 27 with H = 128.
- **Shared across time.** The same U, W and b are applied at every step of every name.
- **Written out by hand.** `nn.RNN` and PyTorch's packed-sequence helpers are not used. The recurrence exists in exactly one method (`_step`), and the training, generation and diagnostics all go through it.
- **Padding.** In a batch of names of different lengths, a finished name's state is frozen at its last character, and the padding positions are excluded from the loss. The padding id is never looked up.

### Initialisation

- **W = Q, a random orthogonal matrix** (all its singular values are 1), so W alone neither grows nor shrinks the state.
- **U ~ N(0, 0.5),** with b = 0.
- **V = 0 and c = 0,** so every name starts at exactly the uniform loss ln 27. A consequence worth knowing: at the very first step the loss gradient reaching U, W and b is exactly zero, so the first update changes only V and c.

Orthogonal W does not make the recurrence norm-preserving. Each step multiplies by D_t W, where D_t holds the tanh derivatives, which are below 1. At initialisation the theory predicts that products of these Jacobians shrink by a factor of about 0.70 per step. Measuring that, and how it changes during training, is the main subject of the study.

### Training

Plain SGD with **full backpropagation through time** (no truncation, no gradient clipping), batches of 32 whole names, and a loss equal to the summed per-character NLL divided by a fixed constant. That constant makes every character count equally, whatever the length of its name, and gives the learning rate the same units as for the fixed-window models. The study selected η = 0.1 and a budget of 38 epochs.

## Documents

The RNN was specified in five documents before any code was written:

| Document | Contents | Approved |
|---|---|---|
| [`rnn_spec.md`](rnn_spec.md) | Architecture, mathematics, initialisation, the research questions and hypotheses, the test plan | 2026-11-14 (v0.2). One later amendment is not written into the file: the initialisation check uses seeds 1000–1009 only and takes no training step |
| [`rnn_diagnostics_spec.md`](rnn_diagnostics_spec.md) | The research diagnostics: hidden-state statistics, exact Jacobian products, loss-projected gradients, the decomposition of the recurrent gradient by distance, and the BOS-restart context test | 2026-11-14 (v0.3) |
| [`rnn_training_spec.md`](rnn_training_spec.md) | Batching, the loss constant, the learning-rate pilot, the epoch-budget rule and failure handling | 2026-11-14 (v0.3) |
| [`rnn_integration_spec.md`](rnn_integration_spec.md) | One consistent contract for how the three fit together: interfaces, file map, run lifecycle | 2026-11-14 (v0.2) |
| [`rnn_implementation_plan.md`](rnn_implementation_plan.md) | Milestones, file scope per milestone, verification plan | 2026-11-14 (v0.2) |

**Why some headers still say "DRAFT".** Four of these documents open with "DRAFT for review. Not approved"; only `rnn_spec.md` records its approval. All five were approved on the dates above, but their text was frozen at that moment. The RNN study records the hash of the four specifications and of the code, and refuses to run if any of them changes, so updating those headers would break the study's own reproducibility check. The implementation plan is not checked that way, but it was left untouched as well, so that all five documents read exactly as they did when approved. The approval dates above are the current status.

The documents also mention two empty placeholder files, `gru.py` and `lstm.py`, which they required to stay untouched. Those placeholders were deleted later, when this folder was published; no GRU or LSTM was ever written, and the RNN study does not check them.

The implementation was then accepted on 2026-11-15, after independent verification:

- a test suite written from the specifications alone, by a separate review that had taken no part in the implementation (48 of 48 passed);
- an independent source review, whose major findings were fixed;
- 38 deliberately planted defects, every one caught by the tests.

**Decision identifiers.** The documents cite decision identifiers (D-xx) and "the record". They refer to the project's internal research record. Its decision table is published as [`DECISIONS.md`](../../DECISIONS.md); the rest of the record is not. The published evidence for each decision is in the documents themselves and in [`results/experiments/rnn_study/`](../../results/experiments/rnn_study/).

## Results in brief

From the [RNN study](../../training/rnn_study/), ten seeds:

- **Test NLL 2.0628 ± 0.0010** (dev 2.0520 ± 0.0011). It was trained under a different protocol from the fixed-window models (name batches, η = 0.1, 38 epochs), so the number is not a ranking against them.
- **Training makes W expansive.** Its largest singular value grows from 1.0 to 3.6, while tanh saturation rises from about 2% to about 19%. Typical gradient directions still shrink with distance, but much more slowly than at initialisation.
- **The trained model uses little distant context.** Removing all but the last 8 characters changes the NLL of late characters in long names by only 0.016 nats. Most of the longer-range dependence concerns predicting where the name ends, not recalling distant letters (exploratory).

The study's report and its limitations are in the [RNN study](../../training/rnn_study/).

## Limitation

Names have at most 16 characters, so nothing here says anything about long-range memory. The study measures how gradients and context dependence behave over at most 16 steps.

## Run the tests

```bash
python3 -m pytest -q tests/models/recurrent    # includes the full-size tests in slow/
```

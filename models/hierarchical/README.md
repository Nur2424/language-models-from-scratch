# WaveNet (hierarchical context merging)

An 8-character context combined as a binary tree: adjacent characters are merged in pairs, then pairs of pairs, then the whole window. It comes in two variants, without and with batch normalisation at every level.

- Code: [`wavenet.py`](wavenet.py) (both variants)
- Specification: [`wavenet_spec.md`](wavenet_spec.md)
- Tests: [`tests/models/hierarchical/`](../../tests/models/hierarchical/), including an independent reference implementation
- Results: the [WaveNet study](../../training/wavenet_study/), a controlled 2 × 2 experiment against flat MLPs with the same context

## What "WaveNet" means here

The name refers to the **receptive-field structure** of WaveNet (van den Oord et al., 2016) and nothing else. There are no gated activations, no residual or skip connections, no convolution modules, no μ-law output and no conditioning.

That structure is exact, not approximate. For the single prediction made from a window, the tree computes the same output as a stack of dilated causal convolutions with kernel 2 and dilations 1, 2 and 4, evaluated at the last position. The tests check this numerically in float64, to a maximum difference of 1.1 × 10⁻¹⁶.

## The model

```
characters [B, 8] ─ embedding (27 × 10) ─────────────────── [B, 8, 10]
level 1   merge adjacent pairs ─ Linear(20 → 68) ─ tanh ── [B, 4, 68]
level 2   merge adjacent pairs ─ Linear(136 → 68) ─ tanh ─ [B, 2, 68]
level 3   merge adjacent pairs ─ Linear(136 → 68) ─ tanh ─ [B, 1, 68]
output    Linear(68 → 27) ───────────────────────────────── [B, 27] logits
```

- **Weight sharing within a level.** Level 1 applies one weight matrix to all four pairs, and level 2 one to both pairs of pairs. A flat MLP instead gives every position its own weights.
- **No hidden biases,** by design. The BatchNorm variant inserts batch normalisation between each linear layer and its tanh, and its shift β plays the role of the bias.
- **Parameters:** 21,989 without BatchNorm, 22,397 with it. The width 68 was chosen so that the model has about as many parameters as the flat MLP with the same 8-character context (21,897).
- **Initialisation:** embedding N(0, 1); each level's weights LeCun normal (std 1/√fan-in); output layer zero, so the initial loss is exactly ln 27.
- **BatchNorm variant:** as in the [BatchNorm MLP](../feedforward/README.md#batchnorm-mlp), the statistics used for evaluation are recomputed exactly over the whole training set at the end of every epoch, never taken from a moving average. Here this is done level by level, each level using the already recalibrated levels below it.

## Results

The model was evaluated in the WaveNet study: a 2 × 2 experiment crossing the architecture (flat MLP or tree) with normalisation (none or BatchNorm), all with the same 8-character context and the same protocol: η = 0.03, 26 epochs, plain SGD, batch 32, ten paired seeds. Both the learning rate and the budget were selected by pilots on separate seeds. The test split was evaluated once, after a formal freeze.

Test NLL in nats per character, mean over seeds 0–9:

| | No BatchNorm | BatchNorm |
|---|---|---|
| Flat MLP, k = 8 | A: 2.0903 | B: 2.0743 |
| Tree (this model) | C: **2.0685** | D: **2.0581** |

Paired differences on test, mean with 95% confidence interval:

- **Tree minus flat, without BatchNorm (C − A):** −0.0218 [−0.0259, −0.0177], lower in 10 of 10 seeds.
- **Tree minus flat, with BatchNorm (D − B):** −0.0162 [−0.0196, −0.0127], lower in 10 of 10 seeds.

Both lie entirely beyond the study's predeclared smallest effect of interest, 0.01 nats per character.

Two things this does **not** show:

- **It does not isolate "hierarchy".** Going from the flat MLP to the tree changes several things at once: one hidden layer becomes three, the width drops from 200 to 68, weights become shared across positions, and the arithmetic per prediction rises by about 1.6×. The result belongs to that bundle, not to any one part of it.
- **It does not come for free.** The tree's lower NLL costs training time. On one laptop CPU core, C took about 1.6 times as long as A, and D about 2.7 times.

The full analysis, including the BatchNorm comparisons, the interaction between the two factors and an exploratory breakdown by how often a context occurs in training, is in the [WaveNet study](../../training/wavenet_study/).

## Limitation

The context is fixed at 8 characters, and every prediction is computed from scratch from its window. The [RNN](../recurrent/README.md) instead reads a whole name with a state carried from character to character.

## Run the tests

```bash
python3 -m pytest -q tests/models/hierarchical
```

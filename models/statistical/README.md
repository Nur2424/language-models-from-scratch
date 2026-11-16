# Statistical bigram

The simplest language model in this repository: the next character depends only on the previous one, and the probabilities are counted, not learned.

- Code: [`bigram.py`](bigram.py)
- Specification: [`bigram_spec.md`](bigram_spec.md)
- Tests: [`tests/models/statistical/test_bigram.py`](../../tests/models/statistical/test_bigram.py)
- Results: [`training/bigram_study/`](../../training/bigram_study/) writes [`results/experiments/bigram_study/results.json`](../../results/experiments/bigram_study/results.json)

## The idea

Every name is framed with the boundary token `.`, which serves as both start and end. The name `anna` becomes the transitions

```
. → a → n → n → a → .
```

Count every transition in the training names into a 27 × 27 table C, where C_ij is how often character j followed character i. Dividing each row by its total gives the probability of each next character:

```latex
P(j \mid i) = \frac{C_{ij}}{N_i}, \qquad N_i = \sum_j C_{ij}
```

This is the maximum-likelihood estimate: no other table of probabilities gives the training names a higher likelihood. There are no parameters to train and no gradients; fitting is one counting pass.

## The problem it exposes: unseen transitions

A transition that never occurs in training gets probability exactly 0. If it then occurs in the dev names, the model assigns those names likelihood 0, and the negative log-likelihood is infinite.

On this dataset that happens 9 times in the dev split, so the unsmoothed model's dev NLL is +∞. The code does not hide this by clamping probabilities to a small epsilon, because that would silently change the model. Instead `evaluate` reports +∞ together with the number of zero-probability events.

The standard fix is additive smoothing: add α pseudo-counts to every cell before normalising.

```latex
P_\alpha(j \mid i) = \frac{C_{ij} + \alpha}{N_i + 27\,\alpha}
```

Smoothing gives up a little training likelihood in exchange for non-zero probability on every transition.

## Results

Standard split (25,640 training names, 3,196 dev names). NLL is in nats per character, averaged over every predicted character including the end of the name. For reference, a uniform guess over 27 classes scores ln 27 = 3.2958.

| α | Train NLL | Dev NLL | Zero-probability dev events |
|---|---|---|---|
| 0 (unsmoothed) | 2.4536 | +∞ | 9 |
| 0.01 | 2.4536 | 2.4582 | 0 |
| 0.03 | 2.4536 | 2.4578 | 0 |
| 0.1 | 2.4537 | 2.4573 | 0 |
| 0.3 | 2.4538 | **2.4570** | 0 |
| 1 | 2.4544 | 2.4571 | 0 |
| 3 | 2.4563 | 2.4586 | 0 |
| 10 | 2.4641 | 2.4661 | 0 |

What the table shows:

- Any α > 0 removes the infinity. Between α = 0.1 and α = 1 the dev NLL hardly changes (2.4570 to 2.4573).
- Large α hurts: at α = 10 the 270 pseudo-counts per row noticeably flatten the rare contexts, and both train and dev NLL rise.
- 2.457 nats per character is a perplexity of about 11.7: on average the model is as uncertain as a uniform choice among 11 or 12 characters.

This is a descriptive sweep on the dev split. No α is selected for a comparison here, and the test split is not used.

Ten samples from the unsmoothed model (fixed generator): `ash`, `tine`, `sta`, `s`, `hilarlliomehaka`, `jei`, `m`, `a`, `fwadekatos`, `kan`.

## Limitation

The model sees one character of context. After `n` it predicts the same distribution whether the name so far is `an`, `ann` or `jonathan`, so it cannot learn that a name rarely has three identical letters in a row, or that `anna` is likely to end after the second `a`. The [neural bigram](../feedforward/README.md) reaches the same solution by gradient descent; models with more context start with the MLP.

## Run the tests

```bash
python3 -m pytest -q tests/models/statistical
```

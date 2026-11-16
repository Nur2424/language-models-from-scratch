# Feedforward models

This folder holds the feedforward language models: the neural bigram, the MLP and the BatchNorm MLP. This page currently covers the neural bigram; the other two are added with their code.

## Neural bigram

The statistical bigram's probabilities, learned by gradient descent instead of counted.

- Code: [`neural_bigram.py`](neural_bigram.py)
- Specification: [`neural_bigram_spec.md`](neural_bigram_spec.md)
- Tests: [`tests/models/feedforward/test_neural_bigram.py`](../../tests/models/feedforward/test_neural_bigram.py)
- Results: [`training/bigram_study/`](../../training/bigram_study/) writes [`results/experiments/bigram_study/results.json`](../../results/experiments/bigram_study/results.json)

### The model

One trainable 27 × 27 table W. Row i holds the logits of the next character after character i:

```latex
z = W[x_{t-1}], \qquad P(x_t = j \mid x_{t-1} = i) = \mathrm{softmax}(W_i)_j
```

This is the same as a linear layer without bias applied to a one-hot encoding of the previous character. There is no hidden layer and no embedding: 729 parameters, one free logit per (previous, next) pair. W starts at zero, so every row starts as the uniform distribution and the initial loss is exactly ln 27 = 3.2958.

Training is plain full-batch gradient descent on the mean cross-entropy over all 182,666 training transitions, with learning rate 25 and no weight decay.

### Why it learns the counting solution

The gradient of the loss with respect to row i has a simple form. With p_i the model's row and q_i the counted probabilities of the statistical bigram:

```latex
\nabla_{W_i} L = \frac{N_i}{M}\,(p_i - q_i)
```

Gradient descent moves each row toward the counts, weighted by how often that context occurs. The loss is convex, and the lowest value it can approach is exactly the statistical bigram's training NLL, 2.4536.

There is a catch. A transition never seen in training (q_ij = 0) still has p_ij > 0, so its gradient is always positive and the logit is pushed down at every step. Softmax can only reach probability 0 as a logit goes to −∞, so the model never reaches the counting solution; it only approaches it. The mass on unseen transitions decays roughly like 1/t.

### Results

Same split as the statistical bigram. Train and dev NLL in nats per character, measured at increasing numbers of full-batch steps:

| Steps | Train NLL | Dev NLL | Mass on unseen transitions |
|---|---|---|---|
| 0 | 3.2958 | 3.2958 | 0.048 |
| 10 | 2.7194 | 2.7208 | 0.037 |
| 100 | 2.4879 | 2.4897 | 0.012 |
| 1,000 | 2.4565 | 2.4583 | 0.0019 |
| 5,000 | 2.4541 | **2.4569** | 0.00048 |
| 20,000 | 2.4538 | 2.4571 | 0.00014 |
| 100,000 | 2.4536 | 2.4576 | 0.000030 |

What the run shows:

- **It converges to the counting solution.** After 100,000 steps the train NLL is 2.4536, within 3 × 10⁻⁵ of the statistical bigram's 2.4536, and the largest difference between a learned and a counted probability on observed transitions is 0.00025.
- **The 1/t decay is visible.** Doubling the steps from 50,000 to 100,000 halves the mass on unseen transitions (5.9 × 10⁻⁵ to 3.0 × 10⁻⁵). For the transition start → end, which never occurs, steps × probability settles at 0.275, close to the 0.285 that the specification's 1/t estimate predicts.
- **Stopping early acts as smoothing.** Dev NLL is lowest around 5,000 steps (2.4569) and then slowly rises, because the 9 dev transitions unseen in training keep losing probability. This is the same trade-off that α controls in the statistical bigram, whose best dev NLL in the sweep is 2.4570. The numbers are close, but the two are different mechanisms, not the same model.
- With the same generator, the trained model produces the same ten samples as the statistical bigram: `ash`, `tine`, `sta`, `s`, `hilarlliomehaka`, `jei`, `m`, `a`, `fwadekatos`, `kan`. Its probabilities are close enough to the counts that every draw lands on the same character.

The run is deterministic (zero initialisation, full batch), so there are no seeds to repeat. Nothing is selected on these results, and the test split is not used. The full run takes about 25 minutes on a laptop CPU, almost all of it spent on the slow approach after step 5,000.

### Limitation

It has exactly the statistical bigram's information: one previous character. Its value is as the bridge from counting to learning; more context starts with the MLP.

### Run the tests

```bash
python3 -m pytest -q tests/models/feedforward/test_neural_bigram.py
```

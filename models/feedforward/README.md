# Feedforward models

This folder holds the three feedforward language models. Each one predicts the next character from a fixed window of previous characters, with no state carried between predictions.

- [Neural bigram](#neural-bigram): one previous character, a 27 × 27 table learned by gradient descent.
- [MLP](#mlp): three previous characters, embedded and passed through one hidden layer.
- [BatchNorm MLP](#batchnorm-mlp): the same MLP with batch normalisation before the hidden nonlinearity.

All results on this page use the same split of the names dataset and report negative log-likelihood (NLL) in nats per character, including the end of the name. A uniform guess over the 27 classes scores ln 27 = 3.2958.

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

It has exactly the statistical bigram's information: one previous character. Its value is as the bridge from counting to learning; more context starts with the [MLP](#mlp).


## MLP

A character-level version of the neural language model of Bengio et al. (2003): the previous three characters are embedded, concatenated and passed through one tanh hidden layer.

- Code: [`mlp.py`](mlp.py)
- Specification: [`mlp_spec.md`](mlp_spec.md)
- Tests: [`tests/models/feedforward/test_mlp.py`](../../tests/models/feedforward/test_mlp.py)
- Results: [`training/mlp_study/`](../../training/mlp_study/) writes [`results/experiments/mlp_study/`](../../results/experiments/mlp_study/)

### The model

Each of the k = 3 previous characters is looked up in a shared embedding table E (27 × 10). The three vectors are concatenated into one vector x of length 30, which goes through one hidden layer and a linear output:

```latex
x = [E_{c_1};\, E_{c_2};\, E_{c_3}] \in \mathbb{R}^{30}, \qquad
h = \tanh(W_1 x + b_1) \in \mathbb{R}^{200}, \qquad
z = W_2 h + b_2 \in \mathbb{R}^{27}
```

At the start of a name the missing characters are filled with the boundary token, so the first prediction sees `. . .`.

- **Parameters:** 270 (embedding) + 6,200 (hidden) + 5,427 (output) = **11,897**.
- **Shared embedding, separate positions.** The embedding table is shared across the three positions, but each position has its own block of W₁, so the model knows where in the window a character sits.
- **Initialisation.** E ~ N(0, 1); W₁ ~ N(0, 1/30) so that the pre-activations start with roughly unit variance; b₁ = 0; W₂ = b₂ = 0, so every name starts at exactly the uniform loss ln 27.
- **Training.** Plain SGD (no momentum, no weight decay), mini-batches of 32 windows drawn without replacement, mean loss over the batch. The best epoch is chosen by dev NLL.

### Choosing the training settings

The specification fixed the procedure before any run, and the study applied it mechanically (all runs on seeds 100–102, disjoint from the final seeds):

1. **Learning rate.** Ten epochs at each of η ∈ {0.03, 0.1, 0.3, 1.0}; mean dev NLL at epoch 10: 2.1634, 2.2014, 2.2979, 2.5038. The winner, 0.03, was at the edge of the grid, so 0.01 was added (2.2024). **η = 0.03.** At η = 1.0 the dev NLL rose in 5 or 6 of the 10 epochs, but every candidate stayed finite.
2. **Epoch budget.** Thirty epochs at η = 0.03. The mean dev NLL is lowest at epoch 28; the earliest epoch within max(0.001, SE) of that minimum is 26. **E = 26.**
3. **Schedule.** A constant rate against one ×0.1 drop after epoch 13: mean dev NLL at epoch 26 of 2.1392 against 2.1221, lower in all three seeds. **Drop after epoch 13.**

### Results

Ten seeds (0–9) at the selected settings, mean ± standard error over seeds:

| | Best dev NLL | Test NLL |
|---|---|---|
| MLP, k = 3 | 2.1232 ± 0.0009 | **2.1291 ± 0.0007** |

The test split was evaluated once, after every setting had been fixed. 2.129 nats per character is a perplexity of about 8.4. The bigram models reach about 2.457 on dev, so three characters of context instead of one are worth about 0.33 nats per character.

## BatchNorm MLP

The same MLP, with batch normalisation between the hidden linear layer and the tanh.

- Code: [`batchnorm_mlp.py`](batchnorm_mlp.py)
- Specification: [`batchnorm_mlp_spec.md`](batchnorm_mlp_spec.md)
- Tests: [`tests/models/feedforward/test_batchnorm_mlp.py`](../../tests/models/feedforward/test_batchnorm_mlp.py)
- Results: as for the MLP

### The model

```latex
h = \tanh\big(\gamma \odot \tfrac{W_1 x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\big)
```

During training, μ and σ² are the mean and variance of the current batch. The hidden bias b₁ is removed, because subtracting μ would cancel it, and β takes its place. Parameters: **12,097** (200 fewer for b₁, 400 more for γ and β).

The one design choice that departs from the standard implementation is how the statistics for evaluation are obtained:

- PyTorch's `BatchNorm1d` keeps an exponential moving average of batch statistics during training and uses it at evaluation. That average depends on the order of recent batches and on the momentum.
- Here the evaluation statistics are **recomputed exactly over the whole training set at the end of every epoch**, with a streaming float64 computation that never holds all activations at once. The moving average is still kept, but only to compare against.

So the evaluated model is a fixed function of its parameters, not of the last few batches it happened to see.

### Training settings

Chosen by the same pilot, independently of the MLP. Mean dev NLL at epoch 10 for η ∈ {0.01, 0.03, 0.1, 0.3, 1.0}: 2.2062, **2.1659**, 2.1953, 2.2552, 2.3597. **η = 0.03**, again after one extension to 0.01. The specification gives the BatchNorm MLP the MLP's budget, **26 epochs**, and a **constant** rate, because no schedule pilot is defined for it.

### Results

| | Best dev NLL | Test NLL |
|---|---|---|
| BatchNorm MLP, k = 3 | 2.1357 ± 0.0007 | **2.1455 ± 0.0009** |

These final results should not be read as "BatchNorm makes the MLP worse". The two final configurations differ twice: the BatchNorm MLP has normalisation, and the MLP has the learning-rate drop. The comparison below holds everything else fixed.

## Context length and BatchNorm: a paired comparison

The [WaveNet study](../../training/wavenet_study/) trained both models with a context of k = 8 characters (its arms A and B), with η = 0.03, a constant rate and 26 epochs. The MLP study retrained both at k = 3 under exactly that protocol and with the same ten seeds. The number of training windows does not depend on k, so each seed sees exactly the same sequence of batches at both context lengths. Each k = 3 run therefore pairs with one k = 8 run, and differences are taken seed by seed.

As a check that the protocol really is identical, the study first retrained arm A at k = 8 for seed 0: its best dev NLL, 2.07868446439438, matches the WaveNet study's recorded value bit for bit.

Best dev NLL, mean ± standard error over the ten paired seeds:

| | k = 3 | k = 8 | k = 8 minus k = 3 |
|---|---|---|---|
| MLP | 2.1389 ± 0.0008 | 2.0806 | **−0.0583 ± 0.0011** (10 of 10 seeds) |
| BatchNorm MLP | 2.1357 ± 0.0007 | 2.0656 | **−0.0701 ± 0.0005** (10 of 10 seeds) |
| BatchNorm minus MLP | −0.0033 ± 0.0009 (9 of 10) | −0.0150 ± 0.0010 | |

What it shows:

- **Context is the larger lever.** Five more characters lower the dev NLL by 0.06 to 0.07 nats per character, in every seed. That is three to four times the size of the architectural differences measured in the WaveNet study.
- **BatchNorm's benefit grows with context** (exploratory). At k = 3 it helps by 0.003 nats; at k = 8, by 0.015. The difference, 0.0118 ± 0.0013, has the same sign in all ten seeds. A plausible reading is that normalisation matters more when the hidden layer has more inputs (80 rather than 30), but the study was not designed to test that, so it is an observation, not a finding.

## Limitations of the MLP results

- **Best dev NLL is slightly optimistic.** The dev split also chose each run's best epoch. Test NLL, measured once, is 0.006 (MLP) and 0.010 (BatchNorm MLP) higher.
- **Some runs stop while still improving.** The best epoch is the last one in 5 of 10 final MLP runs, 4 of 10 paired MLP runs and 2 of 10 BatchNorm MLP runs.
- **The budget depends on three seeds.** The 26-epoch budget used a tolerance taken from the standard error of three pilot seeds. With the fixed 0.001 floor alone, the same rule gives 28.
- **The learning rate was ranked after 10 epochs,** as the specification prescribes, not at the full budget.
- **Seeds vary initialisation and batch order only,** never the split, so differences smaller than about 0.01 should be read with the test-minus-dev gap in mind.

## Run the tests

```bash
python3 -m pytest -q tests/models/feedforward
```

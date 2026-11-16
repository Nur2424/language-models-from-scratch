# Bigram study

One script that fits the statistical bigram for a range of smoothing values and trains the neural bigram by full-batch gradient descent, on the standard split.

```bash
python3 training/bigram_study/run_bigram_study.py                    # full run, about 25 minutes on a laptop CPU
python3 training/bigram_study/run_bigram_study.py --max-steps 1000   # quick check, a few seconds
```

It writes [`results/experiments/bigram_study/results.json`](../../results/experiments/bigram_study/results.json): the environment, the dataset and split fingerprints, train and dev NLL for every α and every measured step, convergence measures for the neural bigram, and ten samples from each model.

Both models are deterministic, so a rerun reproduces the same numbers on the same machine and library versions. Nothing is selected or tuned, and the test split is never used.

The results are explained in the model READMEs: [statistical bigram](../../models/statistical/README.md) and [neural bigram](../../models/feedforward/README.md).

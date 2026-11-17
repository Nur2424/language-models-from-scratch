# WaveNet study

A controlled experiment on whether combining an 8-character context **hierarchically** (the [WaveNet](../../models/hierarchical/README.md) tree) gives lower NLL than combining it **flat** (the [MLP](../../models/feedforward/README.md#mlp)), with and without batch normalisation.

Everything here was fixed before it ran: the learning rate and epoch budget by pilots on separate seeds, the comparisons and their statistics in advance, and the test split evaluated exactly once, after a formal freeze.

## Design

Four arms, all with an 8-character context, about 22,000 parameters, and identical training: plain SGD, η = 0.03, batch 32, 26 epochs, best epoch chosen on dev.

| Arm | Architecture | BatchNorm | Parameters |
|---|---|---|---|
| A | Flat MLP (one hidden layer, width 200) | no | 21,897 |
| B | Flat MLP | yes | 22,097 |
| C | Tree (three levels, width 68) | no | 21,989 |
| D | Tree | yes | 22,397 |

Every arm is trained with the same ten seeds (0–9). For a given seed all four arms see exactly the same sequence of batches, so the arms are compared seed by seed (paired), which removes variation that comes from data order.

The comparisons were declared in advance:

- **Primary:** C − A and D − B, tree against flat, without and with BatchNorm.
- **Secondary:** B − A and D − C (BatchNorm within each architecture), and their interaction.

For each one: mean paired difference, 95% confidence interval, a paired t-test with Holm correction within its family, and an exact sign-flip test as a check. A difference counts as practically meaningful only if its whole interval lies beyond **0.01 nats per character**, a threshold also fixed in advance.

## Pipeline

| Step | Module | What it did |
|---|---|---|
| Learning-rate pilot | `lr_pilot.py`, `summarize_lr_pilot.py` | η ∈ {0.01, 0.03, 0.1, 0.3, 1.0}, seeds 100–102, 10 epochs, all four arms: 0.03 for every arm |
| Epoch-budget pilot | `epoch_budget_pilot.py`, `summarize_epoch_budget_pilot.py` | Seeds 100–102 at η = 0.03, up to 30 epochs; plateau epochs A 23, B 25, C 25, D 26; common budget 26. Protocol: [`EPOCH_BUDGET_PILOT_PROTOCOL.md`](EPOCH_BUDGET_PILOT_PROTOCOL.md) |
| Final runs | `final_experiment.py` | 40 runs (4 arms × seeds 0–9) under frozen conditions; seed 0 first, reviewed, then the rest |
| Independent verification | `verify_final_runs.py` | Recomputes every integrity check from the files on disk, never trusting the runner's own status |
| Freeze | `evaluate_final_test.py freeze` | Fixes every selected checkpoint and code hash before any test data is touched |
| Test evaluation | `evaluate_final_test.py evaluate` | Each run evaluated once on the test split; a ledger refuses any repeat |
| Analysis and report | `analyze_final_experiment.py`, `final_statistics.py`, `context_buckets.py` | Dev analysis before the freeze; the final report after the test evaluation |

`study_common.py` holds what the steps share: frozen conditions, hash-chained run manifests, integrity gates, and a guard that keeps the study code from running git.

## Results

Test NLL, mean over seeds 0–9: **A 2.0903, B 2.0743, C 2.0685, D 2.0581.** Paired differences:

| Comparison | Mean difference | 95% CI | Seeds in favour | Verdict at 0.01 |
|---|---|---|---|---|
| C − A (tree vs flat, no BN) | −0.0218 | [−0.0259, −0.0177] | 10 of 10 | meaningful |
| D − B (tree vs flat, BN) | −0.0162 | [−0.0196, −0.0127] | 10 of 10 | meaningful |
| B − A (BN in the flat MLP) | −0.0159 | [−0.0198, −0.0121] | 10 of 10 | meaningful |
| D − C (BN in the tree) | −0.0103 | [−0.0142, −0.0065] | 10 of 10 | inconclusive |
| Interaction | +0.0056 | [+0.0003, +0.0109] | 8 of 10 | inconclusive |

- The tree had lower test NLL than the flat MLP with the same context, with and without BatchNorm, in every seed, by more than the predeclared threshold.
- BatchNorm clearly helped the flat MLP. In the tree its effect is real but its size relative to 0.01 is unresolved.
- The interaction is positive: BatchNorm helped the flat MLP more than the tree. Because it is not clearly negligible, no averaged "architecture effect" is reported.
- An exploratory breakdown suggests the tree's advantage is largest for contexts that are rare or unseen in training. It is exploratory only: the categories were defined on dev, there is no multiplicity correction, and the categories overlap.

The full report, with every number, is [`REPORT.md`](../../results/experiments/wavenet_study/final/analysis/final/REPORT.md).

## Limitations

- **"Tree versus flat" is a bundle.** Depth (one layer versus three), width (200 versus 68), weight sharing across positions and compute all change together. The result belongs to that bundle, not to "hierarchy" alone.
- **The results are conditional on the protocol:** η = 0.03 chosen at 10 epochs on a coarse grid, a 26-epoch budget, plain SGD, batch 32, CPU float32, this dataset and this split. In 12 of the 40 runs the best epoch is the last one.
- **Seeds vary initialisation and data order, not the split.**
- **Runtime:** with diagnostics on, on one laptop core, B took 1.3×, C 1.6× and D 2.7× the training time of A.
- **Exploratory is not causal:** the context-stratified results suggest where the tree helps, not why.

## What is published

| Path | Contents |
|---|---|
| `results/experiments/wavenet_study/lr_pilot/` | Conditions, manifest, invocation log, per-epoch CSV, summaries, thread benchmark |
| `results/experiments/wavenet_study/epoch_budget_pilot/` | The same for the budget pilot |
| `results/experiments/wavenet_study/final/` | Frozen conditions, manifest, invocation log, the seed-0 validation review |
| `.../final/analysis/dev/` | Context definitions (frozen on dev), learning curves, per-seed dev results |
| `.../final/analysis/final/` | `REPORT.md`, `report.json`, per-seed results, comparisons, context strata, runtime |
| `.../final/test_freeze/` | The freeze, the evaluation ledger, one `result.json` per run |

Not published: checkpoints, per-step logs, attempt records and the per-target test NLL arrays (about 430 MB in total). The verifier and the freeze check need those files, so the checks below can be rerun only with the complete local results. What is published is the record of the runs and the full analysis.

## Checking the records

With the complete results present, from the repository root:

```bash
python3 training/wavenet_study/verify_final_runs.py        # every training-record check, read-only
python3 training/wavenet_study/evaluate_final_test.py verify   # the freeze and the one-time test evaluation
```

The study's own tests do not need the results:

```bash
python3 -m pytest -q tests/training/wavenet_study    # about 3 minutes
```

## A note on decision identifiers

Some module comments and the epoch-budget protocol cite decision identifiers (D-xx) and "the record". They refer to the project's internal research record, which is not published. The published evidence for each decision is in the files above.

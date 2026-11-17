# WaveNet study — learning-rate pilot summary

Protocol: wavenet_spec.md v0.4 §16.4. Seeds [100, 101, 102]; 10 epochs; constant LR; selection metric = mean full-dev NLL at epoch 10, computed only when an arm's required grid is complete and valid. Test data never used. Epochs are 1-based.
Frozen conditions: `6db534b4e3a46e8b`; summary script `6402c34366a68164`.

## Blocking conditions

- none

## Attempts that did not complete (every one, including retried ones)

none

## Arm A: Flat-8 MLP, no BatchNorm (21,897 parameters), selection

| LR | seed | attempts | state | epoch-10 dev NLL | best dev (epoch) | flags | largest within-epoch grad max/median (epoch) | problems |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 100 | 1 | final (completed) | 2.1027 | 2.1027 (10) | dev_nll_rose | 1.4 (1) | — |
| 0.03 | 101 | 1 | final (completed) | 2.1034 | 2.1034 (10) | dev_nll_rose | 1.4 (3) | — |
| 0.03 | 102 | 1 | final (completed) | 2.1050 | 2.1050 (10) | — | 1.5 (1) | — |
| 0.1 | 100 | 1 | final (completed) | 2.1199 | 2.1199 (10) | dev_nll_rose,repeated_dev_deterioration | 1.3 (1) | — |
| 0.1 | 101 | 1 | final (completed) | 2.1276 | 2.1267 (9) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.3 (2) | — |
| 0.1 | 102 | 1 | final (completed) | 2.1289 | 2.1276 (7) | dev_nll_rose,epoch10_not_best | 1.5 (1) | — |
| 0.3 | 100 | 1 | final (completed) | 2.2307 | 2.2015 (9) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose,epoch10_not_best | 1.6 (1) | — |
| 0.3 | 101 | 1 | final (completed) | 2.2365 | 2.2095 (7) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.4 (1) | — |
| 0.3 | 102 | 1 | final (completed) | 2.2357 | 2.1973 (7) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.7 (1) | — |
| 1 | 100 | 1 | final (completed) | 13.6484 | 11.3919 (8) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose,epoch10_not_best | 6.9 (2) | — |
| 1 | 101 | 1 | final (completed) | 15.4814 | 2.3989 (1) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose,epoch10_not_best | 3.2 (8) | — |
| 1 | 102 | 1 | final (completed) | 14.3773 | 11.0939 (7) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose,epoch10_not_best | 2.5 (8) | — |
| 0.01 | 100 | 1 | final (completed) | 2.1570 | 2.1570 (10) | — | 1.4 (6) | — |
| 0.01 | 101 | 1 | final (completed) | 2.1520 | 2.1520 (10) | — | 1.5 (3) | — |
| 0.01 | 102 | 1 | final (completed) | 2.1644 | 2.1644 (10) | — | 1.5 (1) | — |

| LR | category | mean ± sample sd of epoch-10 dev NLL |
|---|---|---|
| 0.01 | no_flags_raised | 2.1578 ± 0.0062 |
| 0.03 | flagged_for_review | 2.1037 ± 0.0012 |
| 0.1 | flagged_for_review | 2.1255 ± 0.0049 |
| 0.3 | flagged_for_review | 2.2343 ± 0.0031 |
| 1 | flagged_for_review | 14.5024 ± 0.9229 |

**Decision: interior.** provisional selection is interior to the evaluated grid; flagged candidates must be reviewed before the selection is accepted
Provisional LR 0.03 (flagged_for_review); lowest unflagged LR: 0.01.

## Arm B: Flat-8 MLP, BatchNorm (22,097 parameters), selection

| LR | seed | attempts | state | epoch-10 dev NLL | best dev (epoch) | flags | largest within-epoch grad max/median (epoch) | problems |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 100 | 1 | final (completed) | 2.0935 | 2.0935 (10) | dev_nll_rose | 1.3 (1) | — |
| 0.03 | 101 | 1 | final (completed) | 2.0972 | 2.0972 (10) | — | 1.2 (9) | — |
| 0.03 | 102 | 1 | final (completed) | 2.0987 | 2.0987 (10) | — | 1.2 (1) | — |
| 0.1 | 100 | 1 | final (completed) | 2.1015 | 2.1015 (10) | dev_nll_rose | 1.3 (1) | — |
| 0.1 | 101 | 1 | final (completed) | 2.1134 | 2.1134 (10) | dev_nll_rose,repeated_dev_deterioration | 1.2 (9) | — |
| 0.1 | 102 | 1 | final (completed) | 2.1167 | 2.1128 (7) | dev_nll_rose,epoch10_not_best | 1.2 (1) | — |
| 0.3 | 100 | 1 | final (completed) | 2.1632 | 2.1632 (10) | dev_nll_rose,repeated_dev_deterioration | 1.6 (1) | — |
| 0.3 | 101 | 1 | final (completed) | 2.1800 | 2.1689 (9) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.5 (1) | — |
| 0.3 | 102 | 1 | final (completed) | 2.1747 | 2.1747 (10) | dev_nll_rose,repeated_dev_deterioration | 1.5 (1) | — |
| 1 | 100 | 1 | final (completed) | 2.2986 | 2.2710 (6) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 2.6 (1) | — |
| 1 | 101 | 1 | final (completed) | 2.3046 | 2.2454 (9) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose,epoch10_not_best | 3.0 (1) | — |
| 1 | 102 | 1 | final (completed) | 2.2856 | 2.2855 (4) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 2.4 (1) | — |
| 0.01 | 100 | 1 | final (completed) | 2.1552 | 2.1552 (10) | — | 1.3 (9) | — |
| 0.01 | 101 | 1 | final (completed) | 2.1494 | 2.1494 (10) | — | 1.3 (9) | — |
| 0.01 | 102 | 1 | final (completed) | 2.1502 | 2.1502 (10) | — | 1.2 (1) | — |

| LR | category | mean ± sample sd of epoch-10 dev NLL |
|---|---|---|
| 0.01 | no_flags_raised | 2.1516 ± 0.0031 |
| 0.03 | flagged_for_review | 2.0965 ± 0.0027 |
| 0.1 | flagged_for_review | 2.1105 ± 0.0080 |
| 0.3 | flagged_for_review | 2.1727 ± 0.0086 |
| 1 | flagged_for_review | 2.2963 ± 0.0097 |

**Decision: interior.** provisional selection is interior to the evaluated grid; flagged candidates must be reviewed before the selection is accepted
Provisional LR 0.03 (flagged_for_review); lowest unflagged LR: 0.01.

## Arm C: Hierarchical WaveNet, no BatchNorm (21,989 parameters), selection

| LR | seed | attempts | state | epoch-10 dev NLL | best dev (epoch) | flags | largest within-epoch grad max/median (epoch) | problems |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 100 | 1 | final (completed) | 2.0862 | 2.0862 (10) | — | 1.7 (1) | — |
| 0.03 | 101 | 1 | final (completed) | 2.0912 | 2.0912 (10) | dev_nll_rose | 1.5 (1) | — |
| 0.03 | 102 | 1 | final (completed) | 2.0873 | 2.0873 (10) | — | 1.5 (1) | — |
| 0.1 | 100 | 1 | final (completed) | 2.1038 | 2.1005 (9) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.3 (1) | — |
| 0.1 | 101 | 1 | final (completed) | 2.0994 | 2.0994 (10) | dev_nll_rose,repeated_dev_deterioration | 1.4 (7) | — |
| 0.1 | 102 | 1 | final (completed) | 2.1031 | 2.1024 (7) | dev_nll_rose,epoch10_not_best | 1.5 (1) | — |
| 0.3 | 100 | 1 | final (completed) | 2.2048 | 2.2048 (10) | dev_nll_rose,repeated_dev_deterioration | 1.3 (5) | — |
| 0.3 | 101 | 1 | final (completed) | 2.2308 | 2.2283 (7) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.3 (9) | — |
| 0.3 | 102 | 1 | final (completed) | 2.2293 | 2.2087 (4) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.4 (1) | — |
| 1 | 100 | 1 | final (completed) | 3.5030 | 3.5030 (10) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose | 2.2 (8) | — |
| 1 | 101 | 1 | final (completed) | 3.8778 | 3.8778 (10) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose | 12.7 (10) | — |
| 1 | 102 | 1 | final (completed) | 4.2878 | 3.6991 (7) | dev_nll_rose,repeated_dev_deterioration,running_train_nll_rose,epoch10_not_best | 2.3 (4) | — |
| 0.01 | 100 | 1 | final (completed) | 2.1421 | 2.1421 (10) | — | 1.7 (1) | — |
| 0.01 | 101 | 1 | final (completed) | 2.1398 | 2.1398 (10) | — | 1.4 (1) | — |
| 0.01 | 102 | 1 | final (completed) | 2.1451 | 2.1451 (10) | — | 1.5 (2) | — |

| LR | category | mean ± sample sd of epoch-10 dev NLL |
|---|---|---|
| 0.01 | no_flags_raised | 2.1423 ± 0.0027 |
| 0.03 | flagged_for_review | 2.0882 ± 0.0027 |
| 0.1 | flagged_for_review | 2.1021 ± 0.0023 |
| 0.3 | flagged_for_review | 2.2216 ± 0.0146 |
| 1 | flagged_for_review | 3.8895 ± 0.3925 |

**Decision: interior.** provisional selection is interior to the evaluated grid; flagged candidates must be reviewed before the selection is accepted
Provisional LR 0.03 (flagged_for_review); lowest unflagged LR: 0.01.

## Arm D: Hierarchical WaveNet, BatchNorm (22,397 parameters), selection

| LR | seed | attempts | state | epoch-10 dev NLL | best dev (epoch) | flags | largest within-epoch grad max/median (epoch) | problems |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 100 | 1 | final (completed) | 2.0778 | 2.0778 (10) | — | 1.5 (1) | — |
| 0.03 | 101 | 1 | final (completed) | 2.0791 | 2.0791 (10) | dev_nll_rose | 1.3 (1) | — |
| 0.03 | 102 | 1 | final (completed) | 2.0782 | 2.0782 (10) | — | 1.4 (1) | — |
| 0.1 | 100 | 1 | final (completed) | 2.0715 | 2.0715 (10) | — | 1.3 (6) | — |
| 0.1 | 101 | 1 | final (completed) | 2.0894 | 2.0850 (9) | dev_nll_rose,epoch10_not_best | 1.2 (2) | — |
| 0.1 | 102 | 1 | final (completed) | 2.0765 | 2.0765 (10) | dev_nll_rose,repeated_dev_deterioration | 1.3 (1) | — |
| 0.3 | 100 | 1 | final (completed) | 2.1062 | 2.1062 (10) | dev_nll_rose | 1.4 (1) | — |
| 0.3 | 101 | 1 | final (completed) | 2.1099 | 2.0954 (9) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.3 (1) | — |
| 0.3 | 102 | 1 | final (completed) | 2.1030 | 2.1030 (10) | dev_nll_rose,repeated_dev_deterioration | 1.3 (5) | — |
| 1 | 100 | 1 | final (completed) | 2.1716 | 2.1612 (9) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.6 (1) | — |
| 1 | 101 | 1 | final (completed) | 2.1627 | 2.1463 (9) | dev_nll_rose,repeated_dev_deterioration,epoch10_not_best | 1.7 (1) | — |
| 1 | 102 | 1 | final (completed) | 2.1525 | 2.1525 (10) | dev_nll_rose,repeated_dev_deterioration | 1.4 (1) | — |
| 0.01 | 100 | 1 | final (completed) | 2.1062 | 2.1062 (10) | — | 1.6 (1) | — |
| 0.01 | 101 | 1 | final (completed) | 2.1031 | 2.1031 (10) | — | 1.5 (1) | — |
| 0.01 | 102 | 1 | final (completed) | 2.1086 | 2.1086 (10) | — | 1.4 (1) | — |

| LR | category | mean ± sample sd of epoch-10 dev NLL |
|---|---|---|
| 0.01 | no_flags_raised | 2.1060 ± 0.0028 |
| 0.03 | flagged_for_review | 2.0784 ± 0.0006 |
| 0.1 | flagged_for_review | 2.0791 ± 0.0093 |
| 0.3 | flagged_for_review | 2.1064 ± 0.0034 |
| 1 | flagged_for_review | 2.1623 ± 0.0096 |

**Decision: interior.** provisional selection is interior to the evaluated grid; flagged candidates must be reviewed before the selection is accepted
Provisional LR 0.03 (flagged_for_review); lowest unflagged LR: 0.01.

## Compute (means over final runs; operational, not efficiency evidence)

| arm | LR | params | training tokens | set-up s | trainer epoch s | measured SGD-phase s | est. diag overhead s | est. tok/s without diag | idealized tok/s | measured epoch-end s | process RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.01 | 21,897 | 1826660 | 1.1 | 3.22 | 2.74 | 0.141 | 70707 | 73597 | 0.49 | 354 |
| A | 0.03 | 21,897 | 1826660 | 1.1 | 3.38 | 2.90 | 0.149 | 66632 | 71122 | 0.48 | 353 |
| A | 0.1 | 21,897 | 1826660 | 1.1 | 3.34 | 2.83 | 0.146 | 68536 | 72668 | 0.52 | 347 |
| A | 0.3 | 21,897 | 1826660 | 1.1 | 3.41 | 2.91 | 0.152 | 66738 | 71971 | 0.50 | 336 |
| A | 1 | 21,897 | 1826660 | 1.1 | 3.25 | 2.78 | 0.134 | 69287 | 72963 | 0.47 | 339 |
| B | 0.01 | 22,097 | 1826660 | 1.2 | 4.15 | 3.33 | 0.173 | 58476 | 63358 | 0.82 | 329 |
| B | 0.03 | 22,097 | 1826660 | 1.2 | 3.85 | 3.11 | 0.155 | 62005 | 64657 | 0.74 | 365 |
| B | 0.1 | 22,097 | 1826660 | 1.1 | 3.93 | 3.21 | 0.159 | 60289 | 63372 | 0.73 | 395 |
| B | 0.3 | 22,097 | 1826660 | 1.2 | 4.16 | 3.33 | 0.184 | 59324 | 64330 | 0.84 | 350 |
| B | 1 | 22,097 | 1826660 | 1.2 | 4.08 | 3.27 | 0.178 | 59908 | 64176 | 0.81 | 327 |
| C | 0.01 | 21,989 | 1826660 | 1.2 | 5.16 | 4.05 | 0.162 | 47833 | 50746 | 1.20 | 339 |
| C | 0.03 | 21,989 | 1826660 | 1.2 | 4.88 | 3.86 | 0.156 | 49448 | 50742 | 1.10 | 372 |
| C | 0.1 | 21,989 | 1826660 | 1.2 | 5.22 | 4.12 | 0.170 | 46671 | 49804 | 1.19 | 355 |
| C | 0.3 | 21,989 | 1826660 | 1.1 | 5.14 | 4.03 | 0.166 | 47489 | 49790 | 1.20 | 371 |
| C | 1 | 21,989 | 1826660 | 1.3 | 4.99 | 3.96 | 0.168 | 48295 | 51142 | 1.12 | 319 |
| D | 0.01 | 22,397 | 1826660 | 1.1 | 7.72 | 5.22 | 0.265 | 36910 | 37715 | 2.59 | 358 |
| D | 0.03 | 22,397 | 1826660 | 1.1 | 7.74 | 5.30 | 0.251 | 36241 | 37188 | 2.54 | 386 |
| D | 0.1 | 22,397 | 1826660 | 1.2 | 8.79 | 5.86 | 0.319 | 33267 | 35670 | 3.04 | 387 |
| D | 0.3 | 22,397 | 1826660 | 1.2 | 8.24 | 5.60 | 0.286 | 34531 | 36113 | 2.75 | 317 |
| D | 1 | 22,397 | 1826660 | 1.1 | 8.02 | 5.47 | 0.283 | 35352 | 37140 | 2.65 | 329 |

Timing definitions are stored in every run's `timing.json`. Epoch-end time includes the trainers' arm-specific epoch-end diagnostics. Process RSS is dominated by the PyTorch import, not the model.
Gradient ratios are within-run (scale-free); raw gradient norms are not comparable between BatchNorm and non-BatchNorm arms. Flags are optimisation observations for review, not evidence of instability.

# Language Models From Scratch

Character-level language models, built one at a time from the bigram to the recurrent network. Each model is implemented by hand in PyTorch, specified before it is written, and studied in a controlled experiment. The aim is to understand **why** each architectural step helps, not only how much.

Every model predicts the next character of a name, trained on the same dataset ([`data/names.txt`](data/), 32,033 names) and the same fixed split (80 / 10 / 10 over distinct names, seed 42). All results are negative log-likelihood (NLL) in nats per character, so they are directly comparable in units. Lower is better, and the uniform distribution over 27 characters gives ln 27 = 3.296.

## Models and results

| Model | Context | Parameters | Result (nats / char) | Details |
|---|---|---|---|---|
| Statistical bigram | 1 character | 729 counts | dev 2.457 (α = 0.3) | [models/statistical](models/statistical/) |
| Neural bigram | 1 character | 729 | dev 2.457 | [models/feedforward](models/feedforward/) |
| MLP | 3 characters | 11,897 | **test 2.129** ± 0.001 | [models/feedforward](models/feedforward/), [study](training/mlp_study/) |
| BatchNorm MLP | 3 characters | 12,097 | **test 2.146** ± 0.001 | [models/feedforward](models/feedforward/), [study](training/mlp_study/) |
| Flat MLP | 8 characters | 21,897 | **test 2.090** | [study](training/wavenet_study/) |
| WaveNet-style tree | 8 characters | 21,989 | **test 2.069** | [models/hierarchical](models/hierarchical/), [study](training/wavenet_study/) |
| WaveNet-style tree + BatchNorm | 8 characters | 22,397 | **test 2.058** | [models/hierarchical](models/hierarchical/), [study](training/wavenet_study/) |
| Vanilla RNN | whole name | 23,451 | **test 2.063** ± 0.001 | [models/recurrent](models/recurrent/), [study](training/rnn_study/) |

Test results are means over ten seeds, evaluated once after every setting was fixed. The two bigram models were studied on dev only.

The numbers share units, but not always a training protocol. The MLPs, the flat MLP and the trees were trained under one common protocol, so differences between them are controlled comparisons. The RNN was trained on whole names with its own learning rate and budget, so its result is not a ranking against the others.

## Main findings

- **More context helps, with diminishing returns.** One character of context gives 2.457. Three give 2.129, about 0.33 nats better, and eight give 2.06–2.09.
- **Two estimators, one model.** The count-based bigram and the gradient-trained neural bigram converge to the same distribution and produce the same samples from the same generator. The one systematic difference: the neural model can only push unseen transitions towards zero, and their probability mass shrinks like 1/t with the number of steps.
- **Hierarchy beats a flat layer at equal size.** With the same 8 characters and about the same number of parameters, the tree beats the flat MLP by 0.022 nats without BatchNorm and 0.016 with it, in 10 of 10 seeds. Both differences lie beyond the predeclared smallest effect of interest (0.01). The change bundles depth, width and weight sharing together, so the result belongs to the bundle, not to "hierarchy" alone.
- **BatchNorm's benefit grows with context** (exploratory). It helps by 0.003 nats at 3 characters of context and by 0.015 at 8.
- **The RNN uses little distant context.** Training makes the recurrent matrix expansive: its largest singular value goes from 1.0 to 3.6. Even so, gradients still shrink with distance, and keeping only the last 8 characters of a long name raises the NLL of its late characters by only 0.016 nats. Most of the longer-range dependence concerns predicting where a name ends (exploratory).

Each study's README and report give the full results, their uncertainty and their limitations.

## How the research is done

- **Specified before implemented.** Each model has a specification (architecture, mathematics, initialisation, training, diagnostics, tests) written from [a common template](MODEL_SPECIFICATION_TEMPLATE.md) and approved before any code. The implementation follows an [internal standard](UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md): no packaged layers such as `nn.RNN` or `nn.Transformer`, a single implementation of each computation, and tests for every model, including independent reference implementations for the WaveNet tree and the RNN.
- **Predeclared studies.** Questions, hypotheses, endpoints and decision rules are fixed before the runs. Learning rates and budgets are chosen by pilots on separate seeds, and final runs use ten seeds, paired across the arms being compared.
- **The test split is used once.** It is evaluated only after every setting is fixed (in the WaveNet and RNN studies, after the checkpoints and code are frozen by hash), and nothing is selected on it.
- **Decisions are recorded as they are made.** Each design and study decision is logged with its date and rationale, and design decisions are recorded before the results they govern. The log is published in [`DECISIONS.md`](DECISIONS.md), and the specifications and reports cite it by identifier (D-xx).
- **Results are published with their evidence.** Each study folder in [`results/experiments/`](results/experiments/) holds the conditions, per-run histories, trained models, analysis scripts and outputs. Only files too large for a repository are kept outside it, and each study README says which ones.

## Repository layout

```
models/        one folder per model family: code, specification and README
training/      one folder per study: drivers, protocols and README
results/       published study results (results/experiments/<study>/)
tests/         tests for every model, utility and study driver
utils/         shared infrastructure: data, splits, metrics, checkpoints, diagnostics
configs/       base and per-model configuration
data/          the dataset and its provenance
```

## Getting started

```bash
pip install -r requirements-dev.txt    # Python ≥ 3.10, PyTorch ≥ 2.0
python3 -m pytest -q                   # the full test suite
```

Everything runs on a laptop CPU. Each study README describes its stages and how to check its results.

## Next step

The next stage of the project is the Transformer. It will follow the same path: a research agenda and specification first, then a hand-written implementation, then a controlled study of what attention adds over the recurrent and hierarchical models.

## Licence

[MIT](LICENSE) © 2026 Nur Taabaldiev

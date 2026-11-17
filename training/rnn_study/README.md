# RNN study

The experiment on the [vanilla RNN](../../models/recurrent/): how gradients and context dependence behave across at most 16 recurrent steps, at initialisation and during training. Predictive performance is recorded, but it is not the main question.

- **Full report:** [`results/experiments/rnn_study/REPORT.md`](../../results/experiments/rnn_study/REPORT.md). It covers the questions and hypotheses, every result with its uncertainty, the hypothesis status, limitations and the decision history.
- **Specifications:** the training procedure and the diagnostics are fixed in [`rnn_training_spec.md`](../../models/recurrent/rnn_training_spec.md) and [`rnn_diagnostics_spec.md`](../../models/recurrent/rnn_diagnostics_spec.md), approved before any run.
- **Tests:** [`tests/training/rnn_study/`](../../tests/training/rnn_study/).

## Main results

Ten final runs (seeds 0–9), η = 0.1, 38 epochs, batches of 32 whole names; mean ± standard error over seeds.

- **Test NLL 2.0628 ± 0.0010**, evaluated once, after the checkpoints and code were frozen. Dev 2.0520 ± 0.0011.
- **At initialisation**, typical Jacobian products shrink by about 0.355 log-units per step, close to the predicted rate.
- **Training makes W expansive.** σ_max(W) goes from 1.00 to 3.58 and saturation from about 2% to about 19%. Typical directions still shrink, but more slowly, at about 0.13 per step.
- **Little use of distant context.** On late targets of long names, keeping only the last 8 characters raises the NLL by 0.016 nats. Most of the longer-range effect concerns predicting the end of the name (exploratory).
- **The long-distance share of W's learning signal** stays near 0.11 from epoch 1 to epoch 38. It is dominated by short distances but does not grow steadily, so the hypothesis that it grows is only partly supported.

The test NLL comes from a different protocol from the other models (name batches, its own learning rate and budget), so it is not a ranking against them.

## How the study ran

Every stage was authorised by a recorded decision before it ran. Each script refuses to run if any pinned file has changed, if its decision is missing, or if its output already exists.

| Stage | Script | What it did | Outcome |
|---|---|---|---|
| L2 | `build_probe.py` | Fixed probe of 1,024 training names, stratified by length, used by every diagnostic | `probe/probe.json` |
| L3 | `thread_benchmark.py` | Timing of one epoch at 1, 2, 4 and 8 threads (seed 999) | 4 intra-op threads chosen |
| L4 | `diagnostic_timing.py` | Timing of each diagnostic on the untrained model | Feasibility only |
| L5 | `init_gate.py`, `summarize_gate.py` | Initial statistics against the theory, seeds 1000–1009, no training step | g = 1 and s² = 0.5 kept |
| L6 | `lr_pilot.py`, `summarize_pilot.py` | η ∈ {0.03, 0.1, 0.3, 1.0} × seeds 100–102, 30 epochs | η = 0.1 (best dev 2.0528) |
| L6 | `extension_x.py` | η = 0.1, seeds 100–102, 60 epochs, because the pilot was still improving at epoch 30 | Budget E = 38 |
| L7 | `research_runs.py` | The ten final runs with the full diagnostics, 49 measurement points each | 10/10 completed |
| L8 | `posthoc_spectral.py` | Jacobian spectra at the best epoch, for the 7 runs whose best epoch is not the last | One extra point per run |
| Test | [`evaluate_test.py`](../../results/experiments/rnn_study/test_evaluation/evaluate_test.py) | `freeze`, then a single `evaluate` | `test_evaluation/result.json` |
| Analysis | [`compute_summary.py`, `compute_extended.py`](../../results/experiments/rnn_study/analysis/) | Read-only summaries of the ten final runs | `analysis/*.json` |

The remaining modules are infrastructure:

- `common.py`: paths, locking and threads.
- `conditions.py`: the hash check against `study_conditions.json`.
- `decisions.py`: the check against the decision table.
- `attempts.py`: an append-only, hash-chained log of every run attempt (`manifest.jsonl`), with each command recorded in `invocations.jsonl`.
- `recovery.py`: safe resumption after a crash.
- `verify_runs.py`: an independent, read-only check of all run records.

All 25 training runs completed on the first attempt: the manifest records 25 launches and 25 completions, and no retries. Each pilot run at η = 0.1 was bit-identical to the matching extension run over their 30 shared epochs.

## What is published

Everything is in [`results/experiments/rnn_study/`](../../results/experiments/rnn_study/):

- the study conditions (every pinned hash and decided value), the attempt manifest and the invocation log;
- the probe, the benchmark, the timings and the initial-statistics report;
- for each of the 25 runs: config, metadata, per-epoch history, and the best, final and last checkpoints;
- for the ten final runs, the research diagnostics at all 49 points: `primary.jsonl`, `points.jsonl`, `plan.json` and `o3.json`;
- the test freeze and the test result;
- the analysis scripts and their outputs, including the same summary computed on a second machine (`summary_mac.json`).

Two kinds of file are not on GitHub because of their size: the raw per-row diagnostic arrays (`research_diagnostics/raw/`, 3.9 GB) and the per-step training logs (`diagnostics/steps.jsonl`, 226 MB). They are kept unchanged and will be archived in full, with a permanent DOI.

## Checking the results

Any published model can be loaded and evaluated directly. For example, seed 0:

```python
import models.recurrent.rnn_training as rt
from models.recurrent.rnn import RNNLanguageModel
from training.rnn_study.common import research_experiment_config
from utils.data import load_names, make_next_token_sequences, split_names
from utils.vocabulary import Vocabulary

cfg = research_experiment_config().data
splits = split_names(load_names(cfg.path), fractions=cfg.split_fractions, seed=cfg.split_seed,
                     duplicate_policy=cfg.duplicate_policy)
vocab = Vocabulary.from_names(splits.train)
model = RNNLanguageModel.from_checkpoint(
    "results/experiments/rnn_study/L7_research/runs/rnn/run_001/model_best.pt", expected_vocab=vocab)
dev = make_next_token_sequences(splits.dev, vocab)
print(rt.evaluate_nll(model, dev, batch_size=1024).mean_nll)   # 2.0508911298778014, as recorded
```

Run directories `run_001` to `run_010` hold seeds 0 to 9. Each seed's recorded dev and test NLL are in `test_evaluation/result.json`.

Some checks need more than GitHub holds:

- **`verify_runs.py` and the analysis scripts** also read the raw arrays and the step logs, so they need the full archive.
- **The study drivers** check every decision they depend on against the decision table in `PROJECT_RESEARCH_RECORD.md`. That record is kept locally. Its decision table is published as `DECISIONS.md` at the repository root, with each row byte for byte as the drivers pinned it. To let the drivers run, copy it into place first: `cp DECISIONS.md PROJECT_RESEARCH_RECORD.md`. They also refuse to overwrite existing outputs, so a rerun goes into a new study folder (`--root`).

## Run the tests

```bash
python3 -m pytest -q tests/training/rnn_study
```

The tests use synthetic records and temporary folders. They never read `results/` or the real data.

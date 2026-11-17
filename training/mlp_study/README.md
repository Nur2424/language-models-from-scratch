# MLP study

Trains the [MLP and the BatchNorm MLP](../../models/feedforward/README.md) at their own context length, k = 3, under rules that were written down before any result existed, and evaluates the test split once.

The model specifications left the learning rate, schedule and epoch budget open until a pilot was run. This study runs that pilot, fixes those settings, trains the final runs, and adds a paired comparison with the k = 8 runs of the WaveNet study.

## Stages

Each stage is one command, refuses to start until the previous one is complete, and writes one JSON file to [`results/experiments/mlp_study/`](../../results/experiments/mlp_study/).

| Stage | What it does | Output |
|---|---|---|
| `repro` | Retrains WaveNet-study arm A (MLP, k = 8, seed 0) and requires its best dev NLL to match the recorded value bit for bit | `repro.json` |
| `pilot` | For each model: η ∈ {0.03, 0.1, 0.3, 1.0}, seeds 100–102, 10 epochs; extends the grid if the winner is at an edge | `pilot.json` |
| `budget` | MLP at the selected η, 30 epochs; the epoch budget by the plateau rule | `budget.json` |
| `schedule` | MLP: constant rate against one ×0.1 drop at half the budget | `schedule.json` |
| `final` | Both models at their selected settings, seeds 0–9 | `final.json` |
| `paired` | Both models at k = 3 under the WaveNet protocol (η = 0.03, 26 epochs, constant), seeds 0–9, compared seed by seed with arms A and B | `paired.json` |
| `test` | The single evaluation of the test split, on every final run's best checkpoint; refuses to run twice | `test.json` |

The exact rules (what counts as unstable, the boundary extension, the budget rule) are written at the top of [`run_mlp_study.py`](run_mlp_study.py). `runs_index.json` lists every training run with its full dev-NLL curve.

## Running it

From the repository root:

```bash
python3 training/mlp_study/run_mlp_study.py repro
python3 training/mlp_study/run_mlp_study.py pilot
python3 training/mlp_study/run_mlp_study.py budget
python3 training/mlp_study/run_mlp_study.py schedule
python3 training/mlp_study/run_mlp_study.py final
python3 training/mlp_study/run_mlp_study.py paired
python3 training/mlp_study/run_mlp_study.py test
```

All stages together took about an hour on a laptop CPU. The `repro` and `paired` stages read the WaveNet study's per-seed results (`results/experiments/wavenet_study/final/analysis/final/per_seed.csv`), which are published with that study. Training uses one intra-op thread, as the WaveNet study did, so a rerun on the same machine and library versions reproduces every number bit for bit. If a stage is interrupted, running it again skips the runs that already finished. Checkpoints and per-step records are written to `results/experiments/mlp_study/runs/`, which is not published.

## Outcome

- η = 0.03 for both models; budget 26 epochs; the MLP keeps a ×0.1 drop after epoch 13, the BatchNorm MLP a constant rate.
- Test NLL over seeds 0–9: MLP 2.1291 ± 0.0007, BatchNorm MLP 2.1455 ± 0.0009.
- Going from 3 to 8 characters of context lowers best dev NLL by 0.058 (MLP) and 0.070 (BatchNorm MLP), in all ten paired seeds.

The results and their limitations are discussed in [`models/feedforward/README.md`](../../models/feedforward/README.md).

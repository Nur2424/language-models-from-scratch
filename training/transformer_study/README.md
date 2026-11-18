# Transformer Part I study

A controlled comparison at roughly equal capacity (22–24k parameters): do the [two Transformers](../../models/attention/) model names better than the [vanilla RNN](../../models/recurrent/) and [WaveNet with BatchNorm](../../models/hierarchical/), when all four are trained the same way?

- **Full report:** [`results/experiments/transformer_study/REPORT.md`](../../results/experiments/transformer_study/REPORT.md). It covers the questions and hypotheses, every pilot and decision, the results with their uncertainty, the hypothesis status and the limitations.
- **Specifications:** the models in [`transformer_spec.md`](../../models/attention/transformer_spec.md); the re-trained WaveNet arm in [`wavenet_d_arm_spec.md`](wavenet_d_arm_spec.md). Both were approved before any run.
- **Tests:** [`tests/training/transformer_study/`](../../tests/training/transformer_study/).

## Main results

Ten final runs per arm (seeds 0–9), all trained with the RNN's protocol: batches of 32 whole names, the same name batches for the same seed in every arm, plain SGD. Mean ± standard error over seeds; test evaluated once, after a freeze.

| Arm | Parameters | η, epochs | Test NLL |
|---|---:|---|---:|
| Model 02 (GPT-2 style) | 23,604 | 0.1, 48 | **2.0439 ± 0.0014** |
| Vanilla RNN (published runs) | 23,451 | 0.1, 38 | 2.0628 ± 0.0010 |
| Model 01 (2017 Transformer) | 22,680 | 0.1, 53 | 2.0644 ± 0.0006 |
| WaveNet with BatchNorm (D) | 22,397 | 0.1, 30 | 2.0653 ± 0.0016 |

- **Model 02 beats the RNN** by 0.019 nats (95% interval −0.023 to −0.015, all 10 seeds), more than the smallest effect of interest (0.01).
- **Model 02 is not equivalent to D, as had been expected:** it is ahead by 0.021 (90% interval −0.026 to −0.016), outside the ±0.01 margin.
- **Model 01 is 0.020 worse than Model 02** at the same shape and learning rate, on every seed, and level with the RNN and D.
- **The advantage is not long-range.** On dev, Model 02's whole lead over D lies in the first eight characters of a name, where D's window already sees everything; late in long names it is not ahead.
- **The Transformers rely on absolute position** more than on distant characters: removing early characters costs little if the rest keep their positions.

## How the study ran

All commands run from the repository root. Pilots use seeds 100–102 and dev data only; final runs use seeds 0–9.

| Stage | Command | What it did | Outcome |
|---|---|---|---|
| D pilot | `wavenet_d_arm.py pilot`, `summarize-pilot` | η ∈ {0.03, 0.1, 0.3, 1.0} × 3 seeds, 30 epochs | η = 0.1, E = 30 |
| D final | `wavenet_d_arm.py final --lr 0.1 --epochs 30` | Ten final runs | Dev 2.0562 ± 0.0011 |
| Shape pilot | `transformer_runs.py shape-pilot`, `summarize-shape-pilot` | Model 02 at 1 × 42, 2 × 30 and 3 × 24 (layers × width), each with the η grid × 3 seeds | 1 × 42, η = 0.1; still improving at epoch 30 |
| Model 02 extension | `transformer_runs.py extension --variant gpt2_style --lr 0.1`, `summarize-extension` | 60 epochs × 3 seeds | E = 48 |
| Model 01 pilot | `transformer_runs.py lr-pilot`, `summarize-lr-pilot` | η grid × 3 seeds at 1 × 42 | η = 0.1 (η = 1.0 diverged) |
| Model 01 extension | `transformer_runs.py extension --variant transformer_2017 --lr 0.1`, `summarize-extension` | 60 epochs × 3 seeds | E = 53 |
| AdamW check | `transformer_runs.py adamw-check`, `summarize-adamw-check` | Model 02, AdamW η ∈ {3e-4, 1e-3, 3e-3} × 3 seeds | Feasibility only; protocol unchanged |
| Final runs | `transformer_runs.py final --variant V --lr 0.1 --epochs E` | Ten runs each of Model 02 (E = 48) and Model 01 (E = 53) | 20/20 completed |
| Dev diagnostics | `dev_diagnostics.py run`, `summarize`, `timing` | Dev NLL by position, BOS-restart and position-preserving truncation, attention by distance, epoch timing | `dev_diagnostics/` |
| Freeze | `test_evaluation.py freeze` | Re-checks every final checkpoint against its recorded dev NLL and parameter hash; records all hashes once | `freeze.json` |
| Test | `test_evaluation.py evaluate` | Verifies the frozen hashes, builds the test split and evaluates it once | `result.json` |
| Analysis | `test_evaluation.py analyze` | The predeclared statistics | `analysis.json` |
| Position analysis | `position_analysis.py` | Dev NLL differences by position with name-clustered standard errors | `position_paired.json` |

The 105 training runs (22 for D, 36 shape pilot, 12 Model 01 pilot, 9 AdamW, 6 extension, 20 final) all completed; the only failures are the three Model 01 pilot runs at η = 1.0, which diverged as the rules anticipate. The selection rules, and the four epoch-budget decisions that needed explicit review, are set out in the report (§3).

**Modules:**

- `wavenet_d_arm.py`: arm D, and the shared protocol (batches, loss constant, pilot and budget rules) that the Transformer driver imports unchanged.
- `transformer_runs.py`: Models 01 and 02.
- `dev_diagnostics.py`, `position_analysis.py`: dev-only measurements of trained checkpoints.
- `test_evaluation.py`: the only code that builds the test split.
- `run_lock.py`: both training drivers hold `<root>/.runner.lock` while they run, so a second training process on the same results folder is refused. It was added after all runs had finished (see the report, §11).

Every training command skips completed runs and resumes interrupted ones at the last epoch boundary. Nothing except `test_evaluation.py evaluate` reads the test split.

## What is published

Everything in [`results/experiments/transformer_study/`](../../results/experiments/transformer_study/), about 39 MB:

- for each of the 105 runs: `config.json`, `metadata.json`, the per-epoch `history.jsonl` (train and dev NLL, dev NLL by position, parameter hash, time) and the best, final and last checkpoints;
- the summaries of the pilots, the extensions and the AdamW check;
- the dev diagnostics (per run and summarised), the timing and the position analysis;
- the freeze, the test result (per seed, with test NLL by position) and the analysis.

The vanilla RNN's runs are not copied here: the study reads them from the [RNN study](../../results/experiments/rnn_study/), and the freeze records the hash of the RNN's published test result. The study plan itself is kept locally; its questions, hypotheses, rules and decisions are all set out in the report.

## Checking the results

Any final model can be loaded and evaluated on dev directly. For example, Model 02, seed 0:

```python
import training.transformer_study.transformer_runs as tr
from models.attention.transformer import TransformerLanguageModel

data = tr.load_study_data()
model = TransformerLanguageModel.from_checkpoint(
    "results/experiments/transformer_study/final/gpt2_style/seed_0/model_best.pt",
    expected_vocab=data.vocab, expected_variant="gpt2_style")
print(tr.evaluate(model, data.dev)["nll"])   # recorded: 2.0313545724302573
```

On another machine or PyTorch version the last digits may differ at the level of float32 rounding (about 1e-8).

- **The statistics** can be recomputed from the published test result at any time: `python3 training/transformer_study/test_evaluation.py analyze` rewrites `analysis.json` from `result.json`, and must give the same numbers.
- **The dev diagnostics and the position analysis** read only the published checkpoints and the dev split, so they can be rerun as they are.
- **The test evaluation cannot be repeated in place, by design:** it refuses to run while `result.json` exists. Each seed's test NLL is in `result.json`.
- **A full rerun** goes into a new folder with `--root`, stage by stage in the order above. On the same machine and PyTorch version, a run with the same seed reproduces its history bit for bit.

## Run the tests

```bash
python3 -m pytest -q tests/training/transformer_study
```

The tests use synthetic data, small subsets or the dev split, and temporary folders. They never read `results/` or the test split.

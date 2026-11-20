# Figures

The figures of the main README, built only from published result files, so anyone with a clone can rebuild them.

```bash
python3 figures/make_figures.py          # writes figures/out/readme_fig{1,2}.{png,pdf}
python3 -m pytest -q tests/figures       # the plotted numbers equal the studies' reported numbers
```

| Figure | Shows | Source files |
|---|---|---|
| `readme_fig1` | (a) the predeclared primary comparisons, paired by seed; (b) the sensitivity of WaveNet+BN to the training protocol | `wavenet_study/final/analysis/final/per_seed.csv`, `transformer_study/test_evaluation/result.json` |
| `readme_fig2` | (a) WaveNet vs MLP by context frequency (test set, exploratory); (b) Transformer (GPT-2) vs WaveNet+BN by position (dev set) | `wavenet_study/final/analysis/final/context_test.csv`, `transformer_study/dev_diagnostics/position_by_index.json` |

All source paths are under `results/experiments/`.

## Model names

- **MLP**, **MLP+BN**: flat MLP with an 8-character context, without and with batch normalisation (WaveNet-study arms A and B; 21,897 and 22,097 parameters).
- **WaveNet**, **WaveNet+BN**: the hierarchical (tree) character model of this project, not the audio WaveNet, without and with batch normalisation (arms C and D; 21,989 and 22,397 parameters).
- **RNN**: vanilla (Elman) RNN, hidden size 128 (23,451 parameters).
- **Transformer (2017)**, **Transformer (GPT-2)**: decoder-only Transformers following the 2017 and the GPT-2 designs, 1 layer, width 42 (22,680 and 23,604 parameters). Neither is the original model.

## Style

One system for every figure, in [`style.py`](style.py):

- **Typeface:** Source Sans 3, Regular and Semibold. The font files and their licence (SIL Open Font License 1.1) are in [`fonts/`](fonts/) and loaded from there, so every machine renders the same glyphs.
- **Canvas:** every figure is exactly 9 inches wide, at 220 dpi, with no automatic cropping. On GitHub all figures therefore display at the same scale.
- **Type scale:**
  - 11.5 pt Semibold for panel titles;
  - 9.5 pt for row and axis labels;
  - 8.5 pt for ticks, legends and status lines;
  - 8 pt for annotations.

  Semibold is used only for titles, protocol names and the words TEST SET and DEV SET.
- **Data colours:**
  - blue and green: the Okabe–Ito colour-blind-safe palette;
  - charcoal: a methodological control.
- **Output:** files are written without timestamps, so rebuilding on the same machine gives byte-identical files.
- **Matplotlib version:** the committed images were built with Matplotlib 3.10.9. Other versions lay out multi-line labels slightly differently, so their output can differ by a few pixels.

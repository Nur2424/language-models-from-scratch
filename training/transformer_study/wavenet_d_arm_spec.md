# WaveNet-BatchNorm (arm D) under the shared protocol: driver specification

Part of the Transformer Part I comparative study. Status: approved 2026-11-17; pilot and final runs completed the same day.

## 1. Purpose

Re-train WaveNet with BatchNorm (arm D of the WaveNet study) under the shared name-batch protocol used by the vanilla RNN and the two Transformers, so that every Part I comparison is paired by seed and trained the same way. The published WaveNet study and its results are unchanged; this is a new arm with its own runs.

## 2. Reused unchanged

| Component | Source |
|---|---|
| Model | `WaveNetLanguageModel` with `WaveNetConfig(context_length=8, embedding_dim=10, hidden_dim=68, normalization="batchnorm")`: 22,397 parameters |
| Initialisation | `generator=make_generator(derive_seed(seed, "init"))`, as in every WaveNet study driver, so seed s starts from exactly the parameters of WaveNet-study D seed s |
| BatchNorm inference statistics | `calibrate_batchnorm(model, train_windows, batch_size=4096)` after every epoch: exact, sequential, full training set |
| Evaluation | `evaluate_nll(model, windows, batch_size=4096)`, token-weighted, in eval mode |
| Data | `load_names` → `split_names` (seed 42, fractions 0.8/0.1/0.1, `duplicate_policy="group"`); `Vocabulary.from_names(splits.train)` |
| Windows | `make_context_windows(names, vocab, context_length=8)`: 8 BOS tokens at the start of every name, one window per target, EOS included |
| Name batches | `EpochSampler(N, 32, seed=derive_seed(seed, "data"))` over the N = 25,640 training names, exactly as in `rnn_training.py` |
| Loss constant | C = B · M / N = 32 · 182,666 / 25,640 = 227.976…, computed as in `rnn_training.loss_normalizer` |

`wavenet.py` and `train_wavenet` are not modified and not used for training; only the functions above are called.

## 3. Batch construction

- `train_windows = make_context_windows(splits.train, vocab, context_length=8)`. Its rows for training name n are contiguous: rows `off[n]` to `off[n] + len(n)` inclusive, where `off` is the cumulative sum of `len + 1` over the names in order.
- At each step the sampler yields name indices `idx` (32 names; the last batch of an epoch has 8). The step's windows are the rows of those names, concatenated in the order of `idx`.
- So a step sees W = Σ (len(n) + 1) windows over its names, about 228 on average, varying from batch to batch.
- **Pairing:** for a given seed, the sequence of name batches is identical to the RNN's and the Transformers', step by step.

## 4. Training step

1. `model.train()`; logits = `model(contexts)` for the step's W windows. BatchNorm uses the statistics of these W windows.
2. Loss = (summed NLL over the W targets) / C. With C fixed, every target has the same weight whatever its name's length, as for the RNN.
3. Finite-loss check → backward → finite-gradient check → `torch.optim.SGD(lr=η, momentum=0, weight_decay=0)` step.
4. No clipping, no warm-up, constant η. CPU, float32, intra-op threads as decided for the study.

## 5. End of epoch

1. `calibrate_batchnorm` on the full training windows (installs exact inference statistics; the EMA is not used).
2. Full train NLL and full dev NLL with `evaluate_nll` (eval mode).
3. `model_best.pt` if dev NLL strictly improves (so the earliest epoch wins ties); history row; `checkpoint_last.pt`. After the last epoch, `model_final.pt`.

**History row:** epoch, η, train NLL, dev NLL, dev NLL by target position (sums and counts, for T-RQ1b), running training loss, parameter hash, epoch time.

## 6. Pilot, budget and final runs

- **Learning-rate pilot:** η ∈ {0.03, 0.1, 0.3, 1.0}, seeds 100–102, 30 epochs, constant η. Selection by the RNN pilot rule (training spec §7.3): the eligible η with the lowest mean full-dev NLL **at epoch 30**; then the near-tie rule, the smallest eligible η within max(10⁻³, SE) of it. A run that fails mechanically (category A) or diverges (category D: train NLL > ln 27 at an epoch ≥ 1, which ends the run) makes its η ineligible. Boundary extension ×≈3 in the boundary's direction, same seeds and horizon, at most two rounds; still at a boundary → inconclusive.
- **Epoch budget:** the RNN training spec §7.4 hybrid rule, unchanged: e* from the pilot at the selected η with ε = max(0.001, SE); e* < 30 → E = 30; e* = 30 → extension X (60 epochs, seeds 100–102, epochs 1–30 bit-identical to the pilot by parameter hash), then E = max(30, e*₆₀), or inconclusive if e*₆₀ = 60. An SE-derived ε requires explicit review.
- **Final runs:** seeds 0–9 at the selected η and E. Test split not touched until the Part I freeze.

## 7. Differences from the WaveNet study's D (for the report)

| | WaveNet study | This arm |
|---|---|---|
| Batch unit | 32 windows, sampled over all 182,666 windows | 32 names (≈ 228 windows), sampled over names |
| Steps per epoch | 5,709 | 802 |
| Loss | mean over the batch | sum / C (≈ mean over ≈ 228 windows) |
| BatchNorm batch statistics | 32 windows | ≈ 228 windows of 32 names (windows of one name are correlated) |
| η, epochs | 0.03, 26 | from this arm's pilot and budget rule |

The objective (mean NLL over all training targets) and the model are identical; only the optimisation path differs.

## 8. Code

- Driver: [`wavenet_d_arm.py`](wavenet_d_arm.py), with the stages `pilot`, `summarize-pilot`, `extension`, `summarize-extension` and `final`. Runs are written under `results/experiments/transformer_study/wavenet_d/`.
- Every run records an epoch-0 row (the initialised model, NLL = ln 27), so the history has epochs 0 to E.

## 9. Tests

| ID | Test |
|---|---|
| D1 | For a given seed and epoch, the name-index batches equal those of `rnn_training`'s sampler, batch by batch |
| D2 | Window rows of each name are contiguous and equal `make_context_windows` rows; a batch has exactly Σ(len + 1) windows, one per target of its names, EOS included |
| D3 | C equals `rnn_training.loss_normalizer` for the same training set; with all-zero logits the step loss equals W · ln 27 / C |
| D4 | Initial parameters equal a `WaveNetLanguageModel` built with `make_generator(derive_seed(seed, "init"))` |
| D5 | Train mode uses batch statistics; after `calibrate_batchnorm`, eval-mode outputs use the installed statistics and do not depend on batch composition |
| D6 | Reported train and dev NLL equal `evaluate_nll` on the full windows |
| D7 | Determinism: two runs of 2 epochs on a small subset with the same seed give bitwise-identical histories, parameters and checkpoints |
| D8 | Resume at an epoch boundary equals an uninterrupted run (history and parameters, bitwise) |
| D9 | Best-checkpoint selection: strict improvement, earliest epoch on ties |
| D10 | Per-position dev NLL sums and counts add up to the overall dev NLL sum and target count |

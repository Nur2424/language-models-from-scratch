# Transformer Part I: Report

**Question:** at roughly equal capacity (22–24k parameters) and under one shared training protocol, does attention model names better than recurrence (the vanilla RNN) and hierarchical merging (WaveNet with BatchNorm)?

**Answer:** yes, but only in the GPT-2-style design. Model 02 reaches a test NLL of **2.0439**, about 0.019 nats below the RNN and 0.021 below WaveNet, on every seed. The 2017 design (Model 01), at the same shape and learning rate, gives that advantage away and lands with the RNN and WaveNet. The advantage does not come from reaching further back than WaveNet's 8-character window: it lies almost entirely in the first eight characters of a name, which WaveNet also sees in full.

Contents: [1 Design](#1-design) · [2 Models](#2-models) · [3 Configuration](#3-configuration) · [4 Final runs](#4-final-runs) · [5 Test results](#5-test-results) · [6 Where the advantage comes from](#6-where-the-advantage-comes-from-t-rq1b) · [7 Context and attention](#7-context-use-and-attention-t-rq3-t-rq4) · [8 Cost](#8-cost-t-rq5) · [9 Hypotheses](#9-hypothesis-status) · [10 Limitations](#10-limitations) · [11 Integrity](#11-integrity-and-reproducibility)

---

## 1. Design

### Arms

| Arm | What it is | Parameters | Source |
|---|---|---:|---|
| **Model 02** | GPT-2-style decoder-only Transformer | 23,604 | New. Declared the primary comparator before any test access |
| **Model 01** | 2017 Transformer, decoder-only adaptation | 22,680 | New |
| **RNN** | Vanilla (Elman) RNN, H = 128 | 23,451 | Reused: the ten published final runs of the [RNN study](../rnn_study/REPORT.md), unchanged |
| **D** | WaveNet with BatchNorm (arm D of the WaveNet study) | 22,397 | Re-trained here under the shared protocol; `models/hierarchical/wavenet.py` unchanged |

The published RNN and WaveNet studies are not changed by this study.

### Shared protocol

Every arm is trained as the RNN was:

- **Data:** `data/names.txt`, split 80/10/10 over distinct names with seed 42; 27 symbols. Train: 25,640 names and 182,666 targets.
- **Batches:** 32 whole names, from the RNN's `EpochSampler` with data seed `derive_seed(seed, "data")`. Seed s therefore gives every arm the same name batches at every step, and the arms are **paired by seed**.
- **Loss:** summed NLL over the batch's targets divided by a fixed C = B·M/N = 227.976, so every character counts equally whatever the length of its name.
- **Optimiser:** plain SGD with a constant learning rate; no momentum, weight decay, clipping or warm-up. CPU, float32.
- **Selection:** each run keeps the epoch with the lowest dev NLL (earliest on ties).

For D, which predicts each character from a fixed 8-character window, each batch of 32 names is fed as all its windows (about 228 per batch). The windows of a batch therefore share names, so D's BatchNorm statistics come from correlated windows (§10).

### Endpoint and questions

**Endpoint:** test NLL in nats per character, mean over ten final seeds (0–9), evaluated once after a freeze. **Smallest effect of interest (SESOI):** 0.01 nats.

| ID | Question | Status | Hypothesis, declared before any test access |
|---|---|---|---|
| T-RQ1 | Does Model 02 reach a lower test NLL than (a) the RNN and (b) D? | Primary | **H1a:** lower than the RNN by at least 0.01. **H1b** (main expectation): equivalent to D within ±0.01 |
| T-RQ1b | Where along the name does any advantage over D appear? | Secondary, dev | **H1c:** no gap while D's window holds the whole prefix (i ≤ 7 characters read); a gap may start at i = 8 and grow from i = 9 |
| T-RQ2 | Do Model 01 and Model 02 differ at the shared shape? | Primary | Two-sided, no direction |
| T-RQ3 | Does the Transformer use distant characters more than the RNN? | Secondary, dev | **H3:** a larger BOS-restart penalty at d = 8 than the RNN's 0.016 |
| T-RQ4 | How is attention distributed by distance? | Exploratory, dev | None |
| T-RQ5 | What does each arm cost to train? | Descriptive | None |

**Statistics:** paired differences over seeds with 95% t intervals. T-RQ1a and T-RQ2 are two-sided t tests. The comparison with D (H1b) is an equivalence test (TOST) with margin ±0.01: equivalence is claimed only if the 90% interval lies entirely inside ±0.01. Holm's correction runs over these three p-values (the TOST p is the larger of its two one-sided p-values). Every other pair is descriptive. Per-target diagnostics use standard errors clustered by name.

## 2. Models

Both Transformers are written by hand in [`models/attention/transformer.py`](../../../models/attention/transformer.py), specified in [`transformer_spec.md`](../../../models/attention/transformer_spec.md), and described in the [models README](../../../models/attention/). They share every computation and differ only in a bundle of seven design choices:

|  | Model 01 (2017) | Model 02 (GPT-2 style) |
|---|---|---|
| Positions | Sinusoidal, base 10,000 | Learned, 16 × d |
| Embedding scaling | × √d | None |
| LayerNorm | Post-LN, no final LN, ε = 1e-6 | Pre-LN with a final LN, ε = 1e-5 |
| Activation | ReLU | GELU (tanh approximation) |
| Attention biases | None | On Q, K, V and output |
| Initialisation | Glorot uniform; embedding N(0, 1/d) | N(0, 0.02²); W_O and W₂ scaled by 1/√(2L); positions N(0, 0.01²) |
| Output | Tied to the embedding, no bias | Tied to the embedding, no bias |

Common to both: causal multi-head self-attention scaled by 1/√d_k, a feed-forward block d → 4d → d with biases, and a context of 16 positions. Adaptations to this task: decoder only (no source sequence), causal mask. Controls: no dropout or label smoothing, and the shared SGD protocol instead of Adam (AdamW is checked separately, §3.5).

**Shape:** 1 layer, d = 42, 3 heads of 14, chosen by a pilot (§3.2).

**Verification before any run:** an independent float64 reference implementation agrees with both models to 1e-12 at all three candidate shapes. The model tests (40, run on the author's Mac) also cover causality (changing later tokens leaves earlier outputs unchanged), prefix equivalence, padding, parameter counts, initial statistics and gradcheck.

**Initial loss.** Tying the output to the embedding rules out a zero output layer, so neither model starts at exactly ln 27 = 3.296, unlike the RNN. Epoch-0 train NLL over the ten final seeds: Model 02 3.30–3.33; Model 01 3.94–4.73.

## 3. Configuration

All pilots use seeds 100–102, 30 epochs and dev data only, with the RNN study's rules (RNN training spec §7.3–7.4):

- **Learning rate:** grid {0.03, 0.1, 0.3, 1.0}; the eligible rate with the lowest mean dev NLL at epoch 30 is the primary choice; the **near-tie rule** then takes the smallest rate within max(0.001, SE of the primary) of it. A run whose train NLL exceeds ln 27 after epoch 0 has diverged, which makes its rate ineligible.
- **Epoch budget:** e* is the earliest epoch whose mean dev NLL is within ε = max(0.001, SE) of the minimum. If e* < 30, E = 30. If e* = 30, a 60-epoch extension runs (its epochs 1–30 must be bit-identical to the pilot), and E = max(30, e*₆₀). An ε derived from the SE rather than the 0.001 floor requires explicit review.

### 3.1 Arm D

| η | Mean dev NLL at epoch 30 | SE |
|---|---:|---:|
| 0.03 | 2.0754 | 0.0024 |
| 0.1 | 2.0537 | 0.0011 |
| 0.3 | 2.0529 | 0.0039 |
| 1.0 | 2.0624 | 0.0054 |

Primary 0.3; the near-tie margin 0.0039 selects **η = 0.1**. All 12 runs completed. e* = 28 with the floor ε, so **E = 30**.

### 3.2 Shape (Model 02)

Three shapes inside the budget, each with 3 heads: 1 layer at d = 42, 2 layers at d = 30, 3 layers at d = 24 (four layers cannot meet the budget without changing d_ff). Each shape got its own learning-rate pilot; 36 runs, all completed.

| Shape | η = 0.03 | η = 0.1 | η = 0.3 | η = 1.0 | Selected η |
|---|---:|---:|---:|---:|---:|
| 1 layer, d = 42 | 2.0804 | **2.0453** | 2.0441 | 2.0844 | 0.1 |
| 2 layers, d = 30 | 2.0991 | 2.0568 | **2.0531** | 2.0841 | 0.3 |
| 3 layers, d = 24 | 2.1139 | 2.0691 | **2.0638** | 2.0840 | 0.3 |

Mean dev NLL at epoch 30. For 1 layer the primary rate was 0.3; the near-tie rule (margin 0.0012) selected 0.1, which was inside the margin by only 0.00005. In practice the two rates are equivalent.

Shapes are compared at their own selected rates, with a paired tie margin of max(0.001, SE of the seed differences). Two layers are 0.0078 worse than one (margin 0.0062) and three layers 0.0185 worse (margin 0.0042), so no shape tied and **1 × 42** was chosen. This search was given only to the Transformer; the reference models' shapes were fixed by design (§10).

### 3.3 Model 01 learning rate

At the chosen shape:

| η | Mean dev NLL at epoch 30 | Outcome |
|---|---:|---|
| 0.03 | 2.1051 | Completed |
| 0.1 | **2.0713** | Completed, selected |
| 0.3 | 2.0814 | Completed |
| 1.0 | — | Diverged in all 3 seeds |

At η = 1.0 the runs first trained well (train NLL 2.17–2.18 by epoch 4–5), degraded slowly, and exploded at epochs 18–20. A diverged rate counts as evaluated but ineligible, so 0.1 is not a grid boundary. Model 02 trained stably at η = 1.0. This fits the usual account of post-LN, which has no clean identity path for the gradient and tolerates large steps worse than pre-LN; it is an observation from three seeds, not a tested finding.

### 3.4 Epoch budgets

Both Transformers were still improving at epoch 30. All four Transformer decisions had an SE-derived ε and were reviewed:

| Decision | Rule result | ε (SE) | Review | E |
|---|---|---:|---|---:|
| Model 02, pilot | e* = 30 | 0.0049 | No effect: e* = 30 with either ε | Extension |
| Model 02, extension | e* = 48 | 0.0034 | Accepted: epochs 48–60 lie within seed noise (minimum 2.0266 at epoch 53) | **48** |
| Model 01, pilot | e* = 28 | 0.0059 | **Rejected**: the minimum mean is at epoch 30 itself, so the model was still improving | Extension |
| Model 01, extension | e* = 53 | 0.0047 | Accepted: epochs 53–60 lie within seed noise (minimum 2.0536 at epoch 58) | **53** |

Both extensions reproduced epochs 1–30 of their pilots bit for bit (by parameter hash) in all three seeds. With the floor ε alone, e* would have been 53 for Model 02 and 55 for Model 01; since every final run keeps its own best-dev epoch, the difference matters little.

### 3.5 AdamW check (feasibility only)

Does plain SGD handicap the Transformer? Model 02 at 1 × 42, AdamW with betas (0.9, 0.999), ε = 1e-8 and no weight decay, 30 epochs:

| Optimiser | Mean dev NLL at epoch 30 | SE |
|---|---:|---:|
| AdamW 3e-4 | 2.0573 | 0.0002 |
| AdamW 1e-3 | 2.0384 | 0.0017 |
| AdamW 3e-3 | 2.0364 | 0.0044 |
| SGD 0.1 (shape pilot) | 2.0453 | 0.0049 |

At an equal 30 epochs AdamW is about 0.007–0.009 better. SGD given its full 48 epochs reaches 2.0283 on the extension runs, below every AdamW value here. SGD makes the Transformer slower, not worse; the protocol is unchanged. AdamW's best rate sits at the top of its grid, which was not extended.

## 4. Final runs

Ten seeds (0–9) per arm; all 30 new runs completed without failure or divergence.

| Arm | η | E | Best dev NLL | Best epochs | Train NLL at best epoch |
|---|---:|---:|---:|---|---:|
| Model 02 | 0.1 | 48 | 2.0315 ± 0.0008 | 41–48 | 1.9444 ± 0.0013 |
| RNN (published) | 0.1 | 38 | 2.0520 ± 0.0011 | 30–38 | 1.9454 ± 0.0020 |
| D | 0.1 | 30 | 2.0562 ± 0.0011 | 25–30 | 1.9564 ± 0.0019 |
| Model 01 | 0.1 | 53 | 2.0581 ± 0.0010 | 44–53 | 1.9875 ± 0.0010 |

Mean ± standard error over seeds. Best dev NLL selected each checkpoint, so it is optimistic; the comparison is made on test (§5). Descriptively, Model 02 and the RNN fit the training set equally well (1.944 and 1.945) but Model 02 is 0.020 lower on dev, while Model 01 fits the training set least well of the four.

## 5. Test results

**Freeze:** before the test split was built, `test_evaluation.py freeze` re-evaluated all 30 new final checkpoints, checked each against its recorded dev NLL and parameter hash, and recorded the hashes of the checkpoints, of 11 code files and of the RNN's published result. **Evaluation:** `test_evaluation.py evaluate` verified those hashes and evaluated the test split once (3,197 names; 22,679 targets). The RNN's test values are the published ones, reused.

| Arm | Test NLL | Dev NLL |
|---|---:|---:|
| **Model 02** | **2.0439 ± 0.0014** | 2.0315 |
| RNN | 2.0628 ± 0.0010 | 2.0520 |
| Model 01 | 2.0644 ± 0.0006 | 2.0581 |
| D | 2.0653 ± 0.0016 | 2.0562 |

Every arm is 0.006–0.012 higher on test than on dev.

### Primary comparisons

In `analysis.json` these are named `T-RQ1a`, `T-RQ1b_equivalence` and `T-RQ2`; the second is the equivalence test of T-RQ1 against D, not the position question T-RQ1b of §6.

| Comparison | Mean difference | Interval | Seeds where the first is lower | p (Holm) | Result |
|---|---:|---|---:|---:|---|
| T-RQ1, vs RNN: Model 02 − RNN | −0.0189 | 95%: [−0.0228, −0.0151] | 10/10 | 3.0 × 10⁻⁶ | **H1a supported:** lower by more than 0.01 |
| T-RQ1, vs D (equivalence): Model 02 − D | −0.0214 | 90%: [−0.0263, −0.0165] | 10/10 | TOST 0.999 | **H1b not supported:** not equivalent |
| T-RQ2: Model 01 − Model 02 | +0.0205 | 95%: [+0.0170, +0.0240] | 0/10 | 9.9 × 10⁻⁷ | **Model 01 higher** |

**Reading H1b.** Equivalence was the declared test and it fails: the 90% interval lies entirely below −0.01, outside the margin. Model 02 is therefore ahead of D by more than the smallest effect of interest. This direction is read from the interval; no superiority test against D was declared.

**Reading T-RQ2.** At the same shape, learning rate and protocol, the GPT-2-style design is 0.020 nats better than the 2017 design on every seed. The two differ in seven choices at once, so this compares designs, not any single component.

### Other pairs (descriptive)

| Difference | Mean | 95% interval | Seeds where the first is lower |
|---|---:|---|---:|
| RNN − Model 01 | −0.0015 | [−0.0040, +0.0009] | 6/10 |
| RNN − D | −0.0025 | [−0.0065, +0.0016] | 5/10 |
| D − Model 01 | +0.0009 | [−0.0027, +0.0046] | 4/10 |

Model 01, the RNN and D are within 0.0025 of each other and no interval excludes zero.

## 6. Where the advantage comes from (T-RQ1b)

Dev data only. For every target, the NLL difference between two arms, averaged over the ten paired seeds, grouped by i, the number of characters already read. D's 8-character window holds the whole prefix and the start token for i ≤ 7, loses the start token at i = 8, and loses characters from i = 9. Standard errors are clustered by name ([`position_analysis.py`](../../../training/transformer_study/position_analysis.py)); in brackets, the number of seeds in which the first arm is lower.

| Difference | i ≤ 7 (22,067 targets) | i = 8 (454) | i ≥ 9 (280 targets, 182 names) | All |
|---|---|---|---|---|
| Model 02 − D | −0.026 ± 0.002 (10/10) | −0.018 ± 0.022 (7/10) | +0.051 ± 0.033 (2/10) | −0.025 ± 0.002 |
| Model 02 − RNN | −0.022 ± 0.002 (10/10) | +0.001 ± 0.023 (6/10) | +0.045 ± 0.029 (1/10) | −0.020 ± 0.002 |
| Model 01 − D | +0.001 ± 0.002 (5/10) | +0.017 ± 0.020 (3/10) | +0.082 ± 0.032 (0/10) | +0.002 ± 0.002 |
| Model 02 − Model 01 | −0.026 ± 0.002 (10/10) | −0.035 ± 0.015 (7/10) | −0.031 ± 0.018 (9/10) | −0.027 ± 0.002 |

- **H1c is not supported.** It expected no gap for i ≤ 7. Instead, Model 02's whole lead over D is there: positions i ≤ 7 carry about 100% of the dev gap, with the largest per-target gaps at i = 3–5 (−0.04 to −0.05).
- **Late in the name, where D loses characters, Model 02 is not ahead.** At i ≥ 9 the estimate points the other way (+0.051), but with 182 names it is 1.5 standard errors from zero: inconclusive. The same holds against the RNN.
- **Model 02's lead over Model 01 is spread evenly** across positions.

**Reading.** At this size, attention does not win by seeing further back than an 8-character window. It wins by predicting better from short prefixes that D also sees in full. Why both Transformers are comparatively weak at late positions is not tested here; late positions have few training examples (only 29 of the 25,640 training names reach positions 13–15).

## 7. Context use and attention (T-RQ3, T-RQ4)

Dev data, means over seeds 0–9, computed by one shared script ([`dev_diagnostics.py`](../../../training/transformer_study/dev_diagnostics.py)). A test confirms that its BOS-restart code computes exactly what the RNN study's own function computes (to 1e-12), and on the RNN's checkpoints it gives the RNN study's published values.

**BOS-restart on F₈** (the RNN study's subset: 280 late targets in 182 long names). Keep only the last d characters, restart from the start token, and measure the increase in NLL:

| Arm and intervention | d = 1 | d = 3 | d = 5 | d = 8 |
|---|---:|---:|---:|---:|
| RNN, BOS-restart | 5.987 | 2.506 | 0.566 | 0.016 ± 0.023 |
| Model 01, BOS-restart | 5.146 | 2.345 | 0.530 | −0.045 ± 0.025 |
| Model 02, BOS-restart | 5.431 | 2.501 | 0.612 | −0.028 ± 0.032 |
| Model 01, positions kept | 1.643 | 0.088 | 0.008 | −0.004 ± 0.009 |
| Model 02, positions kept | 2.501 | 0.225 | 0.050 | 0.023 ± 0.015 |

At d = 8, ± the name-clustered standard error, averaged over seeds as in the RNN study.

- **H3 is not supported as declared.** The BOS-restart penalty at d = 8 is negative for both Transformers, not larger than the RNN's 0.016. With name-clustered errors of 0.02–0.03, none of the three values is distinguishable from zero.
- **But BOS-restart is confounded for the Transformers.** Restarting resets positions, so a late character is presented at an early position. At d = 8 this *lowers* the NLL of letter targets (−0.15 for Model 01, −0.18 for Model 02), plausibly because the model no longer expects the name to end, while end-of-name targets get slightly worse (+0.01, +0.05). The RNN has no positions, so the confound is specific to the Transformers.
- **The position-preserving variant** removes the same characters but keeps every remaining character at its original position, with the start token at position 0. With positions kept, keeping only the last 3 characters costs the Transformers 0.09–0.23 nats, against 2.3–2.5 with positions reset. For the Transformers, most of the restart penalty is the lost position, not the lost characters. This variant was declared in the study plan as a second intervention; H3 itself is kept and judged exactly as declared.

**Attention** (one layer, three heads, averaged over dev queries after the start token):

- By distance: the current character receives 19–26% of the weight, the previous one 37–45%, the one before that 17–21%, and the weight falls quickly beyond.
- On the start token: 16–19% for Model 01 and 22% for Model 02.
- Entropy: 0.76–0.85 nats for Model 01 and 1.15–1.21 for Model 02; Model 01's heads are sharper.

Attention weights describe where a head looks, not what influences the prediction; they are reported descriptively.

## 8. Cost (T-RQ5)

One training epoch of each arm, timed back to back on the author's Mac under the same thread settings, without evaluation: **D 0.9 s, RNN 1.3 s, Model 01 1.6 s, Model 02 2.1 s.** A full final run is therefore about 27 s of training for D, 49 s for the RNN, 85 s for Model 01 and 100 s for Model 02. Model 02's advantage costs about twice the RNN's training time and nearly four times D's: each epoch is slower (2.1 s against 1.3 s and 0.9 s), and it trains for more epochs (48 against 38 and 30).

## 9. Hypothesis status

| Hypothesis | Status | Evidence |
|---|---|---|
| H1a: Model 02 below the RNN by ≥ 0.01 | **Supported** | −0.0189, 95% interval entirely beyond −0.01; Holm p = 3.0 × 10⁻⁶ |
| H1b: Model 02 equivalent to D within ±0.01 | **Not supported** | 90% interval [−0.0263, −0.0165] lies outside the margin, on Model 02's side |
| H1c: no gap to D while D sees the whole prefix | **Not supported** | The gap is −0.026 ± 0.002 at i ≤ 7, and not established at i ≥ 9 |
| T-RQ2: Model 01 vs Model 02 | **Model 01 higher by 0.020** | Two-sided; Holm p = 9.9 × 10⁻⁷ |
| H3: larger BOS-restart penalty than the RNN at d = 8 | **Not supported**, confounded | Negative penalties; the position-preserving variant is the cleaner view |
| T-RQ4, T-RQ5 | Descriptive | §7, §8 |

## 10. Limitations

1. **Shape search for the Transformer only.** The reference models' shapes were fixed by design, so Model 02 had a selection step the others did not. The selected shape won its pilot clearly, by 0.008 and 0.019 nats.
2. **Model 01 was not searched.** It used the shape chosen for Model 02; a different shape might suit the 2017 design better.
3. **Two near-equal learning rates.** Model 02's η = 0.1 was chosen over 0.3 by 0.00005 inside the near-tie margin.
4. **Reviewed budgets.** All four Transformer budget decisions had an SE-derived ε and were decided by review (§3.4); one was rejected.
5. **Different budgets.** D trains for 30 epochs, the RNN 38, Model 02 48 and Model 01 53, each by the same rule.
6. **One optimiser.** Plain SGD for every arm. AdamW is about 0.008 better for Model 02 at 30 epochs, and its grid was not extended.
7. **D under a new protocol.** D's BatchNorm statistics come from about 228 correlated windows of 32 names, not 32 independent windows as in the WaveNet study, and it takes 802 steps per epoch instead of 5,709. Its result here is not a replacement for the WaveNet study's.
8. **The divergence rule** was written for models that start near ln 27. Model 01 starts above it (3.94–4.73), so a very short run could be flagged while improving. In the real runs this cannot happen: after one epoch Model 01's train NLL is 2.27–2.29.
9. **Positions in Model 01.** Over 16 positions most sinusoid pairs barely change, so at d = 42 its positional signal lives in about 14 of 42 dimensions. Model 02's position vectors for positions 13–15 are trained by only 29 names.
10. **Bundle comparison.** Model 01 vs Model 02 compares seven choices at once.
11. **Model 01's LayerNorm ε = 1e-6** matches TensorFlow-style reference implementations but was not confirmed in the tensor2tensor source. At this scale it is immaterial: LayerNorm inputs have variance of order 1.
12. **Short sequences.** Names have at most 15 characters. Nothing here concerns long-range dependencies.
13. **One dataset, ten seeds.** Seeds vary initialisation and data order; the split is fixed.

## 11. Integrity and reproducibility

- **Tests**, run on the author's Mac: 40 model tests and 35 study tests ([`tests/models/attention/`](../../../tests/models/attention/), [`tests/training/transformer_study/`](../../../tests/training/transformer_study/)). The study tests use synthetic data or the dev split; none reads the test split.
- **Pairing:** a test confirms that the name batches equal the RNN's, batch by batch, for the same seed.
- **Determinism:** two runs with the same seed give bit-identical histories, and both extensions reproduced their pilots' first 30 epochs bit for bit.
- **One test evaluation:** the test split is built only by `test_evaluation.py evaluate`, which refuses to run twice and refuses if any frozen hash has changed.
- **Diagnostics:** the shared script gives the RNN study's published BOS-restart values on the RNN's checkpoints; the position analysis reproduces every arm's recorded dev NLL.
- **Incident.** The first start of the extension stage launched three copies at once, because earlier starts survived closing the terminal. They wrote into the same extension folders. All copies were stopped, that folder was deleted, and a single chain was started from scratch; its epochs 1–30 are bit-identical to the pilots, which confirms the clean rerun. Nothing else was affected. Both training drivers now take a study lock and refuse to start while another training process holds it. The lock was added after all runs had finished; it is the only change to the drivers after the runs.

**What is published** in this folder: every run's config, metadata, per-epoch history (with dev NLL by position) and its best, final and last checkpoints; the pilot, extension and AdamW summaries; the dev diagnostics, timing and position analysis; and the freeze, test result and analysis. How to rerun or check any of it is in the [study README](../../../training/transformer_study/).

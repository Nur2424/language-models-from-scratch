# Transformers

Two decoder-only, character-level Transformers that predict the next character of a name from every earlier character of it. Unlike the recurrent and hierarchical models, which fix by their structure how earlier characters reach a prediction, attention decides by content: each position takes a weighted average of all earlier positions, with weights computed from the characters themselves, and reaches any of them in one step.

- **Code:** [`transformer.py`](transformer.py), both models in one file
- **Specification:** [`transformer_spec.md`](transformer_spec.md), approved before any code was written
- **Configs:** [`configs/models/transformer_2017.yaml`](../../configs/models/transformer_2017.yaml), [`configs/models/gpt2_style.yaml`](../../configs/models/gpt2_style.yaml)
- **Tests:** [`tests/models/attention/`](../../tests/models/attention/), including an independent reference implementation
- **Results:** the [Transformer Part I study](../../training/transformer_study/)

## The two models

| | Model 01: 2017 Transformer | Model 02: GPT-2 style |
|---|---|---|
| Name in code | `transformer_2017` | `gpt2_style` |
| Reference | Vaswani et al. (2017), *Attention Is All You Need*, as a decoder-only adaptation | Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* |
| Positions | Sinusoidal, base 10,000 | Learned, one vector per position |
| Embedding scaling | × √d | None |
| LayerNorm | Post-LN: x ← LN(x + sublayer(x)); no final LN; ε = 1e-6 | Pre-LN: x ← x + sublayer(LN(x)); final LN; ε = 1e-5 |
| Activation | ReLU | GELU (tanh approximation) |
| Attention biases | None | On Q, K, V and the output projection |
| Initialisation | Glorot uniform; embedding N(0, 1/d) | N(0, 0.02²); W_O and W₂ scaled by 1/√(2L); positions N(0, 0.01²) |

Shared by both: causal multi-head self-attention scaled by 1/√d_k; a feed-forward block d → 4d → d with biases; the output tied to the token embedding (logits = h Eᵀ, no output bias); no dropout; a context of 16 positions (the start token and up to 15 characters).

**Model 01 is not the original encoder–decoder.** It is the 2017 decoder stack without cross-attention, which is the same computation as encoder blocks with a causal mask. Dropout, label smoothing and the Adam warm-up schedule of the original are left out on purpose, so that every model in this project is trained the same way.

**The two models differ in a bundle of seven choices.** Comparing them compares the two designs, not any single component.

## One block

For Model 02, with h the residual stream [T, d]:

```latex
h \leftarrow h + \mathrm{Attn}(\mathrm{LN}_1(h)), \qquad h \leftarrow h + W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}_2(h) + b_1) + b_2
```

and for Model 01:

```latex
h \leftarrow \mathrm{LN}_1(h + \mathrm{Attn}(h)), \qquad h \leftarrow \mathrm{LN}_2\big(h + W_2\,\mathrm{ReLU}(W_1 h + b_1) + b_2\big)
```

with, per head, Attn = softmax(Q Kᵀ / √d_k + M) V, where M is −∞ above the diagonal, so position t sees only positions ≤ t.

- **Written out by hand.** No `nn.MultiheadAttention`, `nn.Transformer*`, `nn.LayerNorm`, `nn.GELU` or `nn.Linear`. Attention, the mask, LayerNorm, GELU and the sinusoidal table each exist once in `transformer.py`, and both models use them.
- **Padding.** The padding id is replaced by 0 before the embedding lookup and excluded from the loss. Because of the causal mask, real positions never attend to padding.
- **Positions can be set explicitly.** `forward(inputs, positions=None)` accepts non-decreasing positions per row. The study uses this to remove early characters while keeping the later ones at their original positions.

## Parameters

At width d, depth L and 27 output classes:

- Model 01: L(12d² + 9d) + 27d
- Model 02: L(12d² + 13d) + 27d + 16d + 2d

The study's shape, chosen by a pilot, is **1 layer, d = 42, 3 heads of 14**: 22,680 parameters for Model 01 and 23,604 for Model 02, inside the 22–24k budget of the vanilla RNN (23,451) and WaveNet with BatchNorm (22,397).

**Initial loss.** Tying the output to the embedding rules out a zero output layer, so neither model starts at exactly ln 27, unlike the RNN. Model 02 starts just above it; Model 01, whose post-LN output has unit scale, starts clearly above it.

## Using a trained model

```python
import training.transformer_study.transformer_runs as tr
from models.attention.transformer import TransformerLanguageModel
from utils.reproducibility import make_generator

data = tr.load_study_data()                       # the fixed split and vocabulary
model = TransformerLanguageModel.from_checkpoint(
    "results/experiments/transformer_study/final/gpt2_style/seed_0/model_best.pt",
    expected_vocab=data.vocab, expected_variant="gpt2_style")
print(tr.evaluate(model, data.dev)["nll"])        # dev NLL of seed 0, recorded as 2.03135457
print([s.text for s in model.generate(5, generator=make_generator(0))])
```

`from_checkpoint` refuses a checkpoint of another class, variant, shape or vocabulary. `generate` samples from the start token, re-reads the whole prefix at every step and stops at the end token; a sample that reaches 15 characters without it is marked unfinished. `forward_with_internals` also returns the attention weights and the residual stream of every layer, for diagnostics.

## Results in brief

From the [Transformer Part I study](../../training/transformer_study/), ten seeds each, under the RNN's training protocol:

- **Model 02: test NLL 2.0439 ± 0.0014,** lower than the vanilla RNN (2.0628) by 0.019 and than WaveNet with BatchNorm (2.0653) by 0.021, on every seed.
- **Model 01: test NLL 2.0644 ± 0.0006,** 0.020 above Model 02 at the same shape and learning rate, and level with the RNN and WaveNet.
- **The advantage is not long-range.** Model 02's lead over WaveNet lies in the first eight characters of a name, which WaveNet's window also sees in full.
- **Both models rely heavily on absolute position.** Removing early characters while keeping the rest at their positions costs little; resetting positions costs a lot.

## Limitation

Names have at most 15 characters, so the models never see more than 16 positions. Nothing here concerns long contexts.

## Run the tests

```bash
python3 -m pytest -q tests/models/attention
```

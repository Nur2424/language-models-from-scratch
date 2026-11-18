"""Tests for ``models/attention/transformer.py`` (spec §13). Synthetic data only: no test reads
``data/`` or ``results/``."""
from __future__ import annotations

import inspect
import math
import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.attention.transformer as tr
from models.attention.transformer import (
    TransformerConfig,
    TransformerLanguageModel,
    causal_mask,
    gelu_tanh,
    layer_norm,
    parameter_count,
    sinusoidal_table,
)
from tests.models.attention.transformer_reference import reference_logits
from utils.checkpoint import save_checkpoint
from utils.data import make_next_token_sequences
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary

VARIANTS = tr.VARIANTS
SHAPES = [(42, 1), (30, 2), (24, 3)]
EXPECTED = {("transformer_2017", 42, 1): 22_680, ("transformer_2017", 30, 2): 22_950,
            ("transformer_2017", 24, 3): 22_032, ("gpt2_style", 42, 1): 23_604,
            ("gpt2_style", 30, 2): 23_730, ("gpt2_style", 24, 3): 22_752}
LN27 = math.log(27)


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return Vocabulary.from_names(["abcdefghijklmnopqrstuvwxyz"])


def make(vocab, variant="gpt2_style", d=30, L=2, seed=0) -> TransformerLanguageModel:
    return TransformerLanguageModel(vocab, TransformerConfig(variant, d, L), generator=make_generator(seed))


def random_names(n: int, seed: int, lo: int = 2, hi: int = 15) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(lo, hi))) for _ in range(n)]


def batch(vocab, n=6, seed=0) -> torch.Tensor:
    return make_next_token_sequences(random_names(n, seed), vocab, max_len=16).inputs


# ------------------------------------------------------------------ §13.1 general
@pytest.mark.parametrize("variant", VARIANTS)
def test_shapes_and_edge_cases(vocab, variant) -> None:
    m = make(vocab, variant)
    x = batch(vocab)
    out = m(x)
    assert out.shape == (6, 16, 27) and out.dtype == torch.float32
    assert m(torch.zeros(1, 1, dtype=torch.long)).shape == (1, 1, 27)
    assert m(torch.zeros(0, 5, dtype=torch.long)).shape == (0, 5, 27)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("d,L", SHAPES)
def test_parameter_counts(vocab, variant, d, L) -> None:
    m = make(vocab, variant, d, L)
    assert m.num_parameters == parameter_count(m.config) == EXPECTED[(variant, d, L)]


def test_input_validation(vocab) -> None:
    m = make(vocab)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))
    with pytest.raises(ValueError):
        m(torch.tensor([[0, 28]]))
    with pytest.raises(ValueError):
        m(torch.tensor([[0, 27, 3]]))           # a valid id after PAD
    x = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError):
        m(x, positions=torch.tensor([[0, 1, 16]]))
    with pytest.raises(ValueError):
        m(x, positions=torch.tensor([[0, 2, 1]]))
    with pytest.raises(ValueError):
        TransformerConfig("transformer_2017", 31, 2, num_heads=1)  # odd width for sinusoids
    with pytest.raises(ValueError):
        TransformerConfig("gpt2_style", 30, 2, num_heads=4)


@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism(vocab, variant) -> None:
    a, b, c = make(vocab, variant, seed=3), make(vocab, variant, seed=3), make(vocab, variant, seed=4)
    x = batch(vocab)
    assert torch.equal(a(x), b(x))
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a(x), c(x))


@pytest.mark.parametrize("variant", VARIANTS)
def test_weight_tying(vocab, variant) -> None:
    m = make(vocab, variant)
    shapes = [tuple(p.shape) for p in m.parameters()]
    assert shapes.count((27, m.config.d_model)) == 1        # one matrix serves input and output
    x = torch.tensor([[0, 1, 2]])                           # 'z' (id 26) is not an input here
    with torch.no_grad():
        before = m(x).clone()
        m.token_embedding[26] += torch.randn(m.config.d_model, generator=torch.Generator().manual_seed(0))
        after = m(x)
    assert not torch.allclose(before[..., 26], after[..., 26])   # the output reads E itself
    assert torch.allclose(before[..., :26], after[..., :26])


def test_prohibited_modules(vocab) -> None:
    for variant in VARIANTS:
        for mod in make(vocab, variant).modules():
            assert not isinstance(mod, (nn.MultiheadAttention, nn.LayerNorm, nn.GELU, nn.Dropout,
                                        nn.Transformer, nn.TransformerEncoderLayer, nn.Linear))
    src = inspect.getsource(tr)
    for banned in ("MultiheadAttention", "scaled_dot_product_attention", "nn.LayerNorm", "F.layer_norm",
                   "F.gelu", "nn.GELU", "Dropout"):
        assert banned not in src


# ------------------------------------------------------------------ §13.2 mathematics
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("d,L", SHAPES)
def test_reference_implementation(vocab, variant, d, L) -> None:
    m = make(vocab, variant, d, L, seed=11).double()
    with torch.no_grad():   # give every bias and LayerNorm a non-trivial value
        for name, p in m.named_parameters():
            if "b_" in name or "beta" in name or "gamma" in name:
                p.add_(0.1 * torch.randn(p.shape, generator=make_generator(len(name)), dtype=torch.float64))
    x = batch(vocab, n=3, seed=d)
    ref = reference_logits(m, x)
    assert torch.allclose(m(x), ref, rtol=0, atol=1e-12)
    pos = torch.tensor([[0] + list(range(5, 20))[:15]] * 3).clamp(max=15)
    pos = torch.cummax(pos, dim=1).values
    assert torch.allclose(m(x, positions=pos), reference_logits(m, x, pos), rtol=0, atol=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_leakage_and_prefix_equivalence(vocab, variant) -> None:
    m = make(vocab, variant, seed=5)
    x = batch(vocab, n=4, seed=1)
    full = m(x)
    gen = torch.Generator().manual_seed(0)
    for t in (0, 3, 7):
        y = x.clone()
        rows = x[:, t] != vocab.pad_id          # rows still valid at t: replace every later token
        y[rows, t + 1:] = torch.randint(1, 27, (int(rows.sum()), 16 - t - 1), generator=gen)
        assert torch.allclose(m(y)[:, : t + 1], full[:, : t + 1], atol=1e-6)
        prefix = m(x[:, : t + 1])
        valid = x[:, t] != vocab.pad_id
        assert torch.allclose(prefix[valid, t], full[valid, t], atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_mask_and_attention_rows(vocab, variant) -> None:
    m = make(vocab, variant, seed=2)
    _, rec = m.forward_with_internals(batch(vocab))
    for w in rec["attention"]:
        assert torch.all(w.masked_select(causal_mask(16)) == 0)
        assert torch.allclose(w.sum(-1), torch.ones(w.shape[:-1]), atol=1e-6)
    assert not causal_mask(16).all(dim=1).any()      # no fully masked row


@pytest.mark.parametrize("variant", VARIANTS)
def test_padding_content_is_irrelevant(vocab, variant) -> None:
    m = make(vocab, variant, seed=6)
    x = batch(vocab, n=5, seed=2)
    valid = x != vocab.pad_id
    y = torch.where(valid, x, torch.randint(1, 27, x.shape, generator=torch.Generator().manual_seed(1)))
    assert torch.allclose(m(x)[valid], m(y)[valid], atol=1e-6)


def test_sinusoids() -> None:
    d = 30
    pe = sinusoidal_table(16, d, torch.float64)
    p, i = 7, 4
    assert pe[p, 2 * i] == pytest.approx(math.sin(p / 10000 ** (2 * i / d)), abs=1e-15)
    assert pe[p, 2 * i + 1] == pytest.approx(math.cos(p / 10000 ** (2 * i / d)), abs=1e-15)
    for k in (1, 3):                     # shift property: PE(p + k) = R(k) PE(p) per frequency pair
        for i in range(d // 2):
            w = 10000 ** (-2 * i / d)
            R = torch.tensor([[math.cos(w * k), math.sin(w * k)], [-math.sin(w * k), math.cos(w * k)]],
                             dtype=torch.float64)
            for p in range(16 - k):
                assert torch.allclose(R @ pe[p, 2 * i: 2 * i + 2], pe[p + k, 2 * i: 2 * i + 2], atol=1e-12)
    dots = pe @ pe.T                      # depends only on p − q
    assert torch.allclose(torch.diagonal(dots, 2), torch.full((14,), float(dots[0, 2]), dtype=torch.float64))


def test_layer_norm_gelu_relu() -> None:
    x = torch.randn(4, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    g, b = torch.rand(7, dtype=torch.float64), torch.rand(7, dtype=torch.float64)
    ref = F.layer_norm(x, (7,), g, b, eps=1e-5)              # oracle in the test only
    assert torch.allclose(layer_norm(x, g, b, 1e-5), ref, atol=1e-12)
    ones, zeros = torch.ones(7, dtype=torch.float64), torch.zeros(7, dtype=torch.float64)
    assert torch.allclose(layer_norm(10 * x, ones, zeros, 0.0), layer_norm(x, ones, zeros, 0.0), atol=1e-12)
    assert torch.allclose(gelu_tanh(x), F.gelu(x, approximate="tanh"), atol=1e-12)
    assert gelu_tanh(torch.tensor(-1.0)).item() == pytest.approx(-0.1588, abs=1e-4)


def test_post_ln_and_pre_ln_structure(vocab) -> None:
    x = batch(vocab, n=2)
    m1 = make(vocab, "transformer_2017")
    _, r1 = m1.forward_with_internals(x)
    assert "final_ln_input" not in r1
    out = m1._embed(x, None)                                   # a post-LN block output is normalised
    for block in m1.blocks:
        out = block(out, causal_mask(16))
        assert torch.allclose(out.mean(-1), torch.zeros(out.shape[:-1]), atol=1e-5)
        assert torch.allclose(out.var(-1, unbiased=False), torch.ones(out.shape[:-1]), atol=1e-3)
    m2 = make(vocab, "gpt2_style")
    _, r2 = m2.forward_with_internals(x)
    assert "final_ln_input" in r2
    assert not torch.allclose(r2["final_ln_input"].var(-1, unbiased=False), torch.ones(2, 16), atol=1e-3)


def test_initialisation_statistics(vocab) -> None:
    d, L = 30, 2
    m1, m2 = make(vocab, "transformer_2017", d, L, seed=1), make(vocab, "gpt2_style", d, L, seed=1)
    assert m1.token_embedding.std().item() == pytest.approx(d ** -0.5, rel=0.15)
    assert m2.token_embedding.std().item() == pytest.approx(0.02, rel=0.15)
    assert m2.position_embedding.std().item() == pytest.approx(0.01, rel=0.15)
    for blk in m2.blocks:
        assert blk.w_q.std().item() == pytest.approx(0.02, rel=0.15)
        assert blk.w_o.std().item() == pytest.approx(0.02 / math.sqrt(2 * L), rel=0.15)
        assert blk.w_2.std().item() == pytest.approx(0.02 / math.sqrt(2 * L), rel=0.15)
    for blk in m1.blocks:
        bound = math.sqrt(6 / (d + 4 * d))
        assert blk.w_1.abs().max().item() <= bound
        assert blk.w_1.var().item() == pytest.approx(2 / (d + 4 * d), rel=0.15)
        assert blk.w_q.var().item() == pytest.approx(1 / d, rel=0.15)
    x = make_next_token_sequences(random_names(64, 9), vocab, max_len=16)
    for model, lo, hi in ((m2, 0.0, 0.03), (m1, 0.15, 1.2)):
        logits = model(x.inputs)
        loss = F.cross_entropy(logits[x.mask], x.targets[x.mask]).item()
        assert lo <= loss - LN27 <= hi
    # scaled attention-score spread at layer 1: tiny for GPT-2 (near-uniform attention), order 1 for 2017
    for model, lo, hi in ((m2, 0.0, 0.05), (m1, 0.3, 3.0)):
        blk = model.blocks[0]
        h = model._embed(x.inputs, None)
        a_in = blk.ln_1(h) if model.config.pre_ln else h
        q, k = blk._linear(a_in, blk.w_q, blk.b_q), blk._linear(a_in, blk.w_k, blk.b_k)
        s = (q.view(64, 16, 3, 10)[..., 0, :] @ k.view(64, 16, 3, 10)[..., 0, :].transpose(1, 2)) / math.sqrt(10)
        assert lo <= s.std().item() <= hi


@pytest.mark.parametrize("variant", VARIANTS)
def test_default_positions(vocab, variant) -> None:
    m = make(vocab, variant)
    x = batch(vocab)
    assert torch.equal(m(x), m(x, positions=torch.arange(16).expand(6, 16).clone()))


@pytest.mark.parametrize("variant", VARIANTS)
def test_gradcheck(vocab, variant) -> None:
    m = make(vocab, variant, d=6, L=2).double()
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.add_(0.05 * torch.randn(p.shape, dtype=torch.float64, generator=make_generator(len(name))))
    names = [n for n, _ in m.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_(True) for p in m.parameters())
    x = batch(vocab, n=2, seed=4)[:, :6]
    x = torch.where(x == vocab.pad_id, torch.zeros_like(x), x)

    def f(*ps):
        return torch.func.functional_call(m, dict(zip(names, ps)), (x,)).sum()

    assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-5)


# ------------------------------------------------------------------ §13.3 behaviour
@pytest.mark.parametrize("variant", VARIANTS)
def test_can_overfit_a_small_batch(vocab, variant) -> None:
    torch.manual_seed(0)
    m = make(vocab, variant, seed=1)
    data = make_next_token_sequences(random_names(32, 3), vocab, max_len=16)
    opt = torch.optim.SGD(m.parameters(), lr=0.3)
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(m(data.inputs)[data.mask], data.targets[data.mask])
        loss.backward()
        opt.step()
    assert loss.item() < 1.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_generation(vocab, variant) -> None:
    m = make(vocab, variant, seed=2)
    s1 = m.generate(20, generator=torch.Generator().manual_seed(0))
    s2 = m.generate(20, generator=torch.Generator().manual_seed(0))
    assert s1 == s2
    for s in s1:
        assert "." not in s.text and len(s.text) <= 15 and set(s.text) <= set("abcdefghijklmnopqrstuvwxyz")
        assert s.finished or len(s.text) == 15


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_round_trip(vocab, variant, tmp_path) -> None:
    m = make(vocab, variant, d=24, L=3, seed=8)
    path = save_checkpoint(tmp_path / "m.pt", kind="final", model=m, config={"model": m.config_dict()},
                           vocab=vocab.to_dict())
    back = TransformerLanguageModel.from_checkpoint(path, expected_vocab=vocab, expected_variant=variant)
    x = batch(vocab)
    assert torch.equal(back(x), m(x))
    other = [v for v in VARIANTS if v != variant][0]
    with pytest.raises(tr.IncompatibleCheckpointError):
        TransformerLanguageModel.from_checkpoint(path, expected_variant=other)

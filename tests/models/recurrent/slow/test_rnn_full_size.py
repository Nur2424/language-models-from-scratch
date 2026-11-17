"""MS2 full-width checks of the production Model 6 at the approved H = 128 (slow suite).

Milestone MS2 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (approved, D-49): the
"full-size checks (H = 128; 23,451 parameters; ln 27 at initialisation; Haar orthogonality)"
kept apart from the fast suite (P7-8). Run with ``python3 -m pytest tests/models/recurrent/slow``.

Criteria as in ``test_rnn.py`` (``rnn_spec.md`` §13.1): A 1e‑9 (float64, g ∈ {1, 1.2},
H ≤ 128, T ≤ 16) and C 1e‑5 (float32 vs float64 reference, g = 1).
"""

from __future__ import annotations

import math

import pytest
import torch

import tests.models.recurrent.rnn_reference as ref
from models.recurrent.rnn import RNNConfig, RNNLanguageModel, _haar_orthogonal
from tests.models.recurrent.conftest import SYNTHETIC_NAMES
from utils.metrics import count_parameters, token_nll
from utils.reproducibility import make_generator

H = 128
PAD = 27
PARAM_NAMES = ("input_lookup", "recurrent_weight", "hidden_bias", "output_weight", "output_bias")
NAMES = tuple(n for n in SYNTHETIC_NAMES if len(n) <= 15)


def _close(actual, expected, tol, what):
    a, e = actual.detach().to(torch.float64), expected.detach().to(torch.float64)
    excess = float(((a - e).abs() - (tol + tol * e.abs())).max())
    assert excess <= 0.0, f"{what}: exceeded by {excess:.3e}"


def _batch(names, width=None):
    rows = [[0] + [ord(ch) - 96 for ch in n] for n in names]
    T = max(len(r) for r in rows) if width is None else width
    return torch.tensor([r + [PAD] * (T - len(r)) for r in rows], dtype=torch.long)


def _params(g: float, seed: int) -> ref.Params:
    gen = make_generator(seed)
    Q, R = torch.linalg.qr(torch.randn(H, H, generator=gen, dtype=torch.float64))
    W = g * (Q * torch.sign(torch.diagonal(R)).unsqueeze(0))
    U = math.sqrt(0.5) * torch.randn(27, H, generator=gen, dtype=torch.float64)
    b = 0.1 * torch.randn(H, generator=gen, dtype=torch.float64)
    V = torch.randn(27, H, generator=gen, dtype=torch.float64) / math.sqrt(H)
    c = 0.1 * torch.randn(27, generator=gen, dtype=torch.float64)
    return ref.Params(U.contiguous(), W.contiguous(), b, V, c)


def _model(p: ref.Params, vocab, dtype):
    m = RNNLanguageModel(vocab, generator=make_generator(7)).to(dtype)
    with torch.no_grad():
        for name, value in zip(PARAM_NAMES, p.as_tuple()):
            getattr(m, name).copy_(value.to(dtype))
    return m


def test_full_size_parameter_count(full_vocab) -> None:
    model = RNNLanguageModel(full_vocab, generator=make_generator(1))
    assert count_parameters(model).total == 23_451 == H * H + 55 * H + 27


@pytest.mark.parametrize("g", [1.0, 1.2])
def test_full_size_reference_equivalence_float64(full_vocab, g) -> None:
    """A:R1 at H = 128: states, pre-activations and all logits against the reference (A)."""
    p = _params(g, 500 + int(10 * g))
    model = _model(p, full_vocab, torch.float64)
    inputs = _batch(NAMES, 18)
    out = model._recurrence(inputs, return_preactivations=True)
    exp = ref.forward_batch(p, inputs)
    v = exp.valid
    _close(out.states, exp.states, 1e-9, "states")
    _close(out.preactivations[v], exp.pre[v], 1e-9, "pre-activations")
    _close(model(inputs), exp.logits, 1e-9, "logits")


def test_full_size_float32_precision(full_vocab) -> None:
    """A:R4 at H = 128 (criterion C, g = 1)."""
    p = _params(1.0, 510)
    model = _model(p, full_vocab, torch.float32)
    inputs = _batch(NAMES)
    exp = ref.forward_batch(p, inputs)
    v = exp.valid
    _close(model(inputs)[v], exp.logits[v], 1e-5, "float32 logits")


def test_full_size_initial_loss_is_ln27(full_vocab) -> None:
    """A:I1 at H = 128 with the default configuration (float32)."""
    model = RNNLanguageModel(full_vocab, generator=make_generator(11))
    inputs = _batch(NAMES)
    targets = ref.forward_batch(_params(1.0, 0), inputs).targets
    with torch.no_grad():
        loss = token_nll(model(inputs), targets, reduction="mean")
    assert abs(float(loss) - math.log(27)) <= 4 * (torch.finfo(torch.float32).eps / 2) * math.log(27)


@pytest.mark.parametrize("g", [1.0, 1.2])
def test_full_size_haar_orthogonality(full_vocab, g) -> None:
    """A:I3 at H = 128: ‖QᵀQ − I‖_max ≤ 1e‑12 (float64 Q); float32 W singular values within 1e‑5
    of g; W = (g·Q) in float32 bitwise; b, V, c zero."""
    Q = _haar_orthogonal(H, make_generator(21))
    assert float((Q.T @ Q - torch.eye(H, dtype=torch.float64)).abs().max()) <= 1e-12
    model = RNNLanguageModel(full_vocab, RNNConfig(recurrent_gain=g), generator=make_generator(21))
    assert torch.equal(model.recurrent_weight.detach(), (g * Q).to(torch.float32))
    sv = torch.linalg.svdvals(model.recurrent_weight.detach().to(torch.float64))
    assert float((sv - g).abs().max()) <= 1e-5
    assert all(torch.count_nonzero(getattr(model, n)) == 0 for n in PARAM_NAMES[2:])


def test_full_size_input_variance(full_vocab) -> None:
    """A:I5 at H = 128 (3,456 entries): |var − s²| ≤ 5·s²·√(2/(27H))."""
    model = RNNLanguageModel(full_vocab, generator=make_generator(43))
    var = float(model.input_lookup.detach().to(torch.float64).var(unbiased=True))
    assert abs(var - 0.5) <= 5 * 0.5 * math.sqrt(2 / (27 * H)), var

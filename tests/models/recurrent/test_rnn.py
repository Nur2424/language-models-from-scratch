"""MS2 tests of the production Model 6 (``models/recurrent/rnn.py``).

Milestone MS2 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (approved, D-49);
``rnn_spec.md`` v0.2 (D-44) §5 – §10, §13 – §15; ``rnn_integration_spec.md`` v0.2 (D-48) §2, §6.

Labels follow ``rnn_spec.md`` §13.2 with the source prefix ``A:`` (D-48, C2): configuration
V1 – V5, forward and invariance R1 – R14, initialisation I1 – I5, generation 8a – 8d,
checkpoint K1 – K2, and the model-facing mathematical checks M1 – M5 (row 2 of the integrated
checklist). Full-width (H = 128) checks are in ``slow/test_rnn_full_size.py``.

Verification baseline: the accepted MS1 reference ``rnn_reference.py`` (independent, float64,
per name). Expected values are never derived from the model under test, except in tests whose
subject is a property of the model itself (determinism, invariance, round trip).

Tolerances (§13.1): **A** 1e‑9 float64 (atol = rtol); **B** and **C** 1e‑5 float32 (atol = rtol),
float32 tests at g = 1 only; **D** bitwise for identical calls; **E** bitwise for states when only
T changes and B is fixed; **FD** step 1e‑6, rtol 1e‑6, atol 1e‑8. Equivalence tests use
**perturbed, non-zero parameters** (fixture rule M6-IMP9); only I1 – I5 use the initialised model.
"""

from __future__ import annotations

import ast
import itertools
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import models.recurrent.rnn as rnn_module
import tests.models.recurrent.rnn_reference as ref
from models.recurrent.rnn import (
    IncompatibleCheckpointError,
    RecurrenceOutput,
    RNNConfig,
    RNNLanguageModel,
    RNNSample,
    _haar_orthogonal,
)
from tests.models.recurrent.conftest import SYNTHETIC_NAMES
from utils.checkpoint import save_checkpoint
from utils.config import ConfigError, default_config_paths, load_config
from utils.metrics import count_parameters, token_nll
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary

PAD = 27
TOL = {"A": dict(atol=1e-9, rtol=1e-9), "B": dict(atol=1e-5, rtol=1e-5), "C": dict(atol=1e-5, rtol=1e-5)}
FD_STEP, FD_RTOL, FD_ATOL = 1e-6, 1e-6, 1e-8
GAINS = (1.0, 1.2)
PARAM_NAMES = ("input_lookup", "recurrent_weight", "hidden_bias", "output_weight", "output_bias")
REPO = Path(rnn_module.__file__).resolve().parents[2]

#: Synthetic names with n = L + 1 ≤ 16 (tolerance A is derived for T ≤ 16).
NAMES = tuple(n for n in SYNTHETIC_NAMES if len(n) <= 15)


# =========================================================================== helpers
def close(actual: torch.Tensor, expected: torch.Tensor, crit: str, what: str = "") -> None:
    """Criterion A/B/C: |x − x_ref| ≤ atol + rtol·|x_ref| elementwise; dtypes are compared as given
    (the float64 reference is never cast down)."""
    assert actual.shape == expected.shape, (what, tuple(actual.shape), tuple(expected.shape))
    a, e = actual.detach().to(torch.float64).cpu(), expected.detach().to(torch.float64).cpu()
    bound = TOL[crit]["atol"] + TOL[crit]["rtol"] * e.abs()
    excess = float(((a - e).abs() - bound).max()) if a.numel() else -1.0
    assert excess <= 0.0, f"{what}: criterion {crit} exceeded by {excess:.3e}"


def encode(name: str) -> list[int]:
    return [0] + [ord(ch) - ord("a") + 1 for ch in name]


def padded_batch(names, width: int | None = None) -> torch.Tensor:
    rows = [encode(n) for n in names]
    T = max((len(r) for r in rows), default=1) if width is None else width
    return torch.tensor([r + [PAD] * (T - len(r)) for r in rows], dtype=torch.long).reshape(len(rows), T)


def lengths_of(inputs: torch.Tensor) -> list[int]:
    return (inputs != PAD).sum(dim=1).tolist()


def reference_params(H: int, g: float, gen: torch.Generator) -> ref.Params:
    """Perturbed float64 parameters (fixture rule M6-IMP9, §13.1), drawn test-locally."""
    A = torch.randn(H, H, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    W = g * (Q * torch.sign(torch.diagonal(R)).unsqueeze(0))
    U = math.sqrt(0.5) * torch.randn(27, H, generator=gen, dtype=torch.float64)
    b = 0.1 * torch.randn(H, generator=gen, dtype=torch.float64)
    V = torch.randn(27, H, generator=gen, dtype=torch.float64) / math.sqrt(H)
    c = 0.1 * torch.randn(27, generator=gen, dtype=torch.float64)
    return ref.Params(U.contiguous(), W.contiguous(), b, V, c)


def model_with(p: ref.Params, vocab: Vocabulary, dtype: torch.dtype) -> RNNLanguageModel:
    """A model whose five parameters are set to the reference fixture (cast to ``dtype``)."""
    model = RNNLanguageModel(vocab, RNNConfig(hidden_dim=p.H), generator=make_generator(7)).to(dtype)
    with torch.no_grad():
        for name, value in zip(PARAM_NAMES, p.as_tuple()):
            getattr(model, name).copy_(value.to(dtype))
    return model


@pytest.fixture(params=GAINS, ids=lambda g: f"g{g}")
def gain(request) -> float:
    return float(request.param)


@pytest.fixture
def fixture(tiny_hidden_dim, gain, generator_factory, full_vocab):
    """(reference params, float64 model) with identical perturbed parameters."""
    p = reference_params(tiny_hidden_dim, gain, generator_factory(100 + int(10 * gain) + tiny_hidden_dim))
    return p, model_with(p, full_vocab, torch.float64)


@pytest.fixture
def fixture32(tiny_hidden_dim, generator_factory, full_vocab):
    """(reference params, float32 model) at g = 1 (float32 tests run at g = 1 only, §13.1)."""
    p = reference_params(tiny_hidden_dim, 1.0, generator_factory(200 + tiny_hidden_dim))
    return p, model_with(p, full_vocab, torch.float32)


def test_fixtures_are_non_trivial(fixture) -> None:
    """Guard (M6-IMP9): V and c are non-zero and the model's logits vary across classes."""
    p, model = fixture
    with torch.no_grad():
        assert float(model.output_weight.abs().min()) > 0 and float(model.output_bias.abs().max()) > 0
        logits = model(padded_batch(NAMES))
        assert float(logits[padded_batch(NAMES) != PAD].std(dim=1).min()) > 1e-2


# =========================================================================== A:V configuration
def test_V1_construction_names_shapes_count_dtype(full_vocab) -> None:
    """A:V1: five raw parameters with the approved names and shapes; 23,451 at H = 128;
    float32 on CPU; no buffers; P(H) = H² + 55H + 27."""
    model = RNNLanguageModel(full_vocab, generator=make_generator(1))
    named = dict(model.named_parameters())
    assert tuple(named) == PARAM_NAMES
    H = 128
    assert [tuple(t.shape) for t in named.values()] == [(27, H), (H, H), (H,), (27, H), (27,)]
    assert count_parameters(model).total == count_parameters(model).trainable == 23_451
    assert all(t.dtype == torch.float32 and t.device.type == "cpu" for t in named.values())
    assert list(model.buffers()) == []
    assert model.config == RNNConfig() and model.device.type == "cpu"
    for H in (1, 4, 8):
        m = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=H), generator=make_generator(1))
        assert count_parameters(m).total == H * H + 55 * H + 27


@pytest.mark.parametrize(
    "kwargs",
    [dict(hidden_dim=0), dict(hidden_dim=-3), dict(hidden_dim=2.0), dict(hidden_dim=True), dict(hidden_dim="8"),
     dict(recurrent_gain=0.0), dict(recurrent_gain=-1.0), dict(recurrent_gain=float("nan")),
     dict(recurrent_gain=float("inf")), dict(recurrent_gain=True), dict(recurrent_gain="1"),
     dict(input_init_variance=0.0), dict(input_init_variance=-0.5), dict(input_init_variance=float("inf")),
     dict(input_init_variance=False)],
)
def test_V2_invalid_config_values(kwargs) -> None:
    """A:V2: invalid ``RNNConfig`` values raise ``ValueError``."""
    with pytest.raises(ValueError):
        RNNConfig(**kwargs)


def test_V2_valid_config_values_are_normalised() -> None:
    cfg = RNNConfig(hidden_dim=16, recurrent_gain=1, input_init_variance=2)
    assert cfg.recurrent_gain == 1.0 and isinstance(cfg.recurrent_gain, float)
    assert isinstance(cfg.input_init_variance, float)


def test_V3_constructor_type_checks(full_vocab) -> None:
    """A:V3: non-Vocabulary, non-RNNConfig, non-generator and non-CPU generator are rejected."""
    with pytest.raises(TypeError):
        RNNLanguageModel("abc", generator=make_generator(1))
    with pytest.raises(TypeError):
        RNNLanguageModel(full_vocab, {"hidden_dim": 8}, generator=make_generator(1))
    with pytest.raises(TypeError):
        RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=8), generator=1)
    with pytest.raises(TypeError):
        RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=8))  # generator is required

    class NonCPUGenerator(torch.Generator):  # reports a non-CPU device without needing an accelerator
        @property
        def device(self) -> torch.device:
            return torch.device("cuda")

    with pytest.raises(ValueError, match="CPU generator"):
        RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=8), generator=NonCPUGenerator())


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ([[0, 1]], TypeError, "torch.Tensor"),
        (torch.tensor([[0, 1]], dtype=torch.int32), ValueError, "dtype"),
        (torch.tensor([0, 1], dtype=torch.long), ValueError, "2-D"),
        (torch.zeros(2, 0, dtype=torch.long), ValueError, "T = 0"),
        (torch.tensor([[0, 1], [0, 28]], dtype=torch.long), ValueError, "row 1"),
        (torch.tensor([[0, 1], [0, -1]], dtype=torch.long), ValueError, "row 1"),
        (torch.tensor([[0, 1], [2, 1]], dtype=torch.long), ValueError, "row 1: column 0"),
        (torch.tensor([[0, 1, PAD], [0, PAD, 3]], dtype=torch.long), ValueError, "row 1: padding"),
        (torch.tensor([[PAD, PAD]], dtype=torch.long), ValueError, "row 0: column 0"),
    ],
    ids=["not-tensor", "int32", "1-D", "T0", "id-28", "negative", "no-bos", "non-contiguous", "all-pad"],
)
def test_V4_input_validation(full_vocab, inputs, error, message) -> None:
    """A:V4: every §5.1 condition raises the specified error, naming the offending row."""
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=4), generator=make_generator(1))
    with pytest.raises(error, match=message):
        model(inputs)
    with pytest.raises(error, match=message):
        model._recurrence(inputs)


def test_V4_device_mismatch_and_empty_batch(full_vocab) -> None:
    """A:V4: inputs on another device raise ``ValueError`` (no hidden transfer); B = 0 returns
    an empty [0, T, 27] tensor in the parameter dtype; a later id 0 is a legal input."""
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=4), generator=make_generator(1))
    with pytest.raises(ValueError, match="device|on meta"):
        model(torch.zeros(1, 3, dtype=torch.long, device="meta"))
    out = model(torch.zeros(0, 5, dtype=torch.long))
    assert out.shape == (0, 5, 27) and out.dtype == torch.float32
    assert model(torch.tensor([[0, 3, 0, 4]], dtype=torch.long)).shape == (1, 4, 27)


def test_V5_yaml_loads_to_defaults() -> None:
    """A:V5: ``configs/models/rnn.yaml`` loads to the ``RNNConfig`` defaults; unknown keys and
    invalid values are rejected."""
    cfg = load_config(default_config_paths("rnn"), model_config_cls=RNNConfig)
    assert cfg.model.name == "rnn"
    assert cfg.model.params == RNNConfig()
    with pytest.raises(ConfigError):
        load_config(default_config_paths("rnn"), overrides=["model.params.dropout=0.1"], model_config_cls=RNNConfig)
    with pytest.raises((ConfigError, ValueError)):
        load_config(default_config_paths("rnn"), overrides=["model.params.hidden_dim=0"], model_config_cls=RNNConfig)


# =========================================================================== A:R forward and invariance
@pytest.mark.parametrize(
    "batch",
    [NAMES, ("a",), ("ludmeriaxo", "a", "zo"), ("kestravelunio",) * 2 + ("bex",)],
    ids=["all", "single", "mixed", "repeated"],
)
@pytest.mark.parametrize("width", [None, 19], ids=["tight", "wide"])
def test_R1_reference_equivalence(fixture, batch, width) -> None:
    """A:R1 (criterion A): valid states, pre-activations and logits of every row, and the
    padded-position logits, equal the independent per-name reference across batch
    compositions and padding widths."""
    p, model = fixture
    inputs = padded_batch(batch, width)
    out = model._recurrence(inputs, return_preactivations=True)
    logits = model(inputs)
    exp = ref.forward_batch(p, inputs)
    assert torch.equal(out.valid, exp.valid)
    v = exp.valid
    close(out.states[v], exp.states[v], "A", "valid states")
    close(out.preactivations[v], exp.pre[v], "A", "valid pre-activations")
    close(logits[v], exp.logits[v], "A", "valid logits")
    close(logits, exp.logits, "A", "all logits (padded: frozen-state logits)")
    close(out.states, exp.states, "A", "all states (padded: frozen)")


def test_R2_padding_invariance_float32(fixture32) -> None:
    """A:R2: in float32, valid logits are unchanged by extra right padding (B); valid states are
    bitwise unchanged with B fixed (E)."""
    _, model = fixture32
    base = padded_batch(NAMES)
    out0 = model._recurrence(base)
    z0 = model(base)
    for width in (17, 25):
        wide = padded_batch(NAMES, width)
        out1, z1 = model._recurrence(wide), model(wide)
        for beta, n in enumerate(lengths_of(base)):
            assert torch.equal(out1.states[beta, :n], out0.states[beta, :n]), "criterion E"
            close(z1[beta, :n], z0[beta, :n], "B", "valid logits under padding")


def test_R3_batch_composition_invariance_float32(fixture32, record_property) -> None:
    """A:R3: a name alone vs inside different batches (B); bitwise identity is recorded [O] only
    (criterion F), never asserted."""
    _, model = fixture32
    target = "obelandrinas"
    alone = model(padded_batch([target]))[0]
    n = len(target) + 1
    bitwise = []
    for batch in (["a", target], [target, "zo", "corvinalt"], list(NAMES)):
        z = model(padded_batch(batch))[batch.index(target), :n]
        close(z, alone[:n], "B", f"composition {batch[:3]}")
        bitwise.append(bool(torch.equal(z, alone[:n])))
    record_property("R3_bitwise_identical_across_compositions", bitwise)


def test_R4_float32_precision_against_float64_reference(fixture32) -> None:
    """A:R4 (criterion C): the float32 model against the float64 reference at g = 1."""
    p, model = fixture32
    inputs = padded_batch(NAMES)
    exp = ref.forward_batch(p, inputs)
    out = model._recurrence(inputs, return_preactivations=True)
    v = exp.valid
    assert model(inputs).dtype == torch.float32
    close(out.states[v], exp.states[v], "C", "states")
    close(out.preactivations[v], exp.pre[v], "C", "pre-activations")
    close(model(inputs)[v], exp.logits[v], "C", "logits")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
def test_R5_determinism(fixture, dtype) -> None:
    """A:R5 (criterion D): identical calls give bitwise identical outputs."""
    model = fixture[1].to(dtype)
    inputs = padded_batch(NAMES)
    a, b = model._recurrence(inputs, return_preactivations=True), model._recurrence(inputs, return_preactivations=True)
    assert torch.equal(model(inputs), model(inputs))
    assert torch.equal(a.states, b.states) and torch.equal(a.preactivations, b.preactivations)


def test_R6_causality(fixture) -> None:
    """A:R6: changing x_{t'} for t' > t leaves z_1 … z_t unchanged (bitwise for identical
    shapes, D), and changes z_{t'}; the new logits match the reference (A)."""
    p, model = fixture
    base = padded_batch(["kestravelunio", "mavra"])
    z0 = model(base)
    for t_change in (3, 8, 13):
        alt = base.clone()
        alt[0, t_change - 1] = int(alt[0, t_change - 1]) % 26 + 1
        z1 = model(alt)
        assert torch.equal(z1[0, : t_change - 1], z0[0, : t_change - 1])
        assert torch.equal(z1[1], z0[1])
        assert not torch.equal(z1[0, t_change - 1], z0[0, t_change - 1])
        close(z1, ref.forward_batch(p, alt).logits, "A", "changed input")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
def test_R7_frozen_state(fixture, dtype) -> None:
    """A:R7 (bitwise): for t > n_β, h_t equals h_{n_β}."""
    model = fixture[1].to(dtype)
    inputs = padded_batch(NAMES, 20)
    states = model._recurrence(inputs).states
    for beta, n in enumerate(lengths_of(inputs)):
        for t in range(n, 20):
            assert torch.equal(states[beta, t], states[beta, n - 1])


def test_R8_padded_logits(fixture, fixture32) -> None:
    """A:R8: padded-position logits equal z_{n_β} of the same row and are finite (A in float64,
    B in float32)."""
    for (_, model), crit in ((fixture, "A"), (fixture32, "B")):
        inputs = padded_batch(NAMES, 20)
        z = model(inputs)
        assert torch.isfinite(z).all()
        for beta, n in enumerate(lengths_of(inputs)):
            close(z[beta, n:], z[beta, n - 1].expand(20 - n, -1), crit, "padded logits")


def test_R9_parameter_sharing(fixture) -> None:
    """A:R9: after W′ = W + ε e_i e_jᵀ, a′_t − (U[x_t] + W h′_{t−1} + b) = ε h′_{t−1,j} e_i at every
    valid t ≥ 2 (one W at every step), and a′_1 = a_1 bitwise."""
    p, model = fixture
    inputs = padded_batch(NAMES)
    base = model._recurrence(inputs, return_preactivations=True)
    eps, i, j = 0.25, 1, 2
    with torch.no_grad():
        model.recurrent_weight[i, j] += eps
    try:
        out = model._recurrence(inputs, return_preactivations=True)
    finally:
        with torch.no_grad():
            model.recurrent_weight[i, j] -= eps
    assert torch.equal(out.preactivations[:, 0], base.preactivations[:, 0])
    H = p.H
    e_i = torch.zeros(H, dtype=torch.float64)
    e_i[i] = 1.0
    for beta, n in enumerate(lengths_of(inputs)):
        for t in range(2, n + 1):
            h_prev = out.states[beta, t - 2]
            unperturbed = p.U[int(inputs[beta, t - 1])] + p.W @ h_prev + p.b
            close(out.preactivations[beta, t - 1] - unperturbed, eps * h_prev[j] * e_i, "A", f"row {beta} t {t}")


def test_R10_hand_computed_H2(full_vocab) -> None:
    """A:R10 (criterion A): H = 2, input [BOS, 'b'] (ids [0, 2]); a_t, h_t and all 27 logits of
    both steps recomputed with Python floats."""
    U = [[0.1 * ((k * 7) % 11 - 5), 0.05 * ((k * 3) % 13 - 6)] for k in range(27)]
    W = [[0.8, -0.3], [0.4, 0.9]]
    b = [0.02, -0.04]
    V = [[0.07 * ((k * 5) % 9 - 4), -0.06 * ((k * 2) % 7 - 3)] for k in range(27)]
    c = [0.01 * ((k * 4) % 5 - 2) for k in range(27)]
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=2), generator=make_generator(3)).to(torch.float64)
    with torch.no_grad():
        for name, value in zip(PARAM_NAMES, (U, W, b, V, c)):
            getattr(model, name).copy_(torch.tensor(value, dtype=torch.float64))
    h = [0.0, 0.0]
    exp_a, exp_h, exp_z = [], [], []
    for x in (0, 2):
        a = [U[x][i] + W[i][0] * h[0] + W[i][1] * h[1] + b[i] for i in range(2)]
        h = [math.tanh(v) for v in a]
        exp_a.append(a)
        exp_h.append(h)
        exp_z.append([V[k][0] * h[0] + V[k][1] * h[1] + c[k] for k in range(27)])
    out = model._recurrence(torch.tensor([[0, 2]]), return_preactivations=True)
    t64 = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    close(out.preactivations[0], t64(exp_a), "A", "a")
    close(out.states[0], t64(exp_h), "A", "h")
    close(model(torch.tensor([[0, 2]]))[0], t64(exp_z), "A", "z")


def test_R11_preactivations_only_on_request(fixture, monkeypatch) -> None:
    """A:R11: pre-activations are ``None`` unless requested, and ``forward`` never requests them."""
    _, model = fixture
    inputs = padded_batch(NAMES)
    out = model._recurrence(inputs)
    assert isinstance(out, RecurrenceOutput) and out.preactivations is None
    assert model._recurrence(inputs, return_preactivations=True).preactivations.shape == out.states.shape
    calls = []
    original = model._recurrence

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_recurrence", spy)
    model(inputs)
    assert calls == [{}]


def test_R12_dtype_follows_parameters(fixture) -> None:
    """A:R12: a float64 model returns float64 logits (and passes R1 at tolerance A); a float32
    model returns float32. No dtype is hard-coded in the forward path."""
    p, model = fixture
    inputs = padded_batch(NAMES)
    z64 = model(inputs)
    assert z64.dtype == torch.float64 and model._recurrence(inputs).states.dtype == torch.float64
    close(z64, ref.forward_batch(p, inputs).logits, "A", "float64 logits")
    assert model.to(torch.float32)(inputs).dtype == torch.float32


def test_R13_train_and_eval_identical(fixture) -> None:
    """A:R13 (D): no dropout or normalisation; train and eval mode give identical outputs."""
    _, model = fixture
    inputs = padded_batch(NAMES)
    model.train()
    z_train = model(inputs)
    model.eval()
    assert torch.equal(model(inputs), z_train)


_PROHIBITED_ATTRS = {"RNN", "RNNCell", "GRU", "GRUCell", "LSTM", "LSTMCell", "Embedding", "Linear",
                     "_VF", "compile", "script", "trace", "reset_parameters", "pack_padded_sequence",
                     "pad_packed_sequence", "PackedSequence"}
_PROHIBITED_PREFIXES = ("torch.nn.utils.rnn", "nn.utils.rnn", "torch._VF", "torch.jit", "torch.nn.init", "nn.init")


def _chain(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_R14_prohibited_modules_source_scan() -> None:
    """A:R14: no prohibited module or helper from §6.2 in any ``models/recurrent/rnn*.py``
    (``rnn.py`` now; ``rnn_diagnostics.py`` and ``rnn_training.py`` automatically once they exist)."""
    files = sorted((REPO / "models" / "recurrent").glob("rnn*.py"))
    assert files and files[0].name == "rnn.py"
    problems = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain = _chain(node)
                if node.attr in _PROHIBITED_ATTRS or chain.startswith(_PROHIBITED_PREFIXES):
                    problems.append(f"{path.name}:{node.lineno} {chain}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                bad = [n for n in names if n and (n.startswith(("torch.nn.utils.rnn", "torch._VF", "torch.jit"))
                                                  or n in _PROHIBITED_ATTRS)]
                problems += [f"{path.name}:{node.lineno} import {n}" for n in bad]
    assert not problems, problems


# =========================================================================== A:I initialisation
def test_I1_initial_loss_is_ln27(full_vocab) -> None:
    """A:I1: with V = c = 0 the mean NLL over valid targets equals ln 27 within 4·u·ln 27 in the
    evaluation dtype (float32 model)."""
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=8), generator=make_generator(11))
    inputs = padded_batch(NAMES)
    targets = ref.forward_batch(reference_params(8, 1.0, make_generator(0)), inputs).targets
    with torch.no_grad():
        loss = token_nll(model(inputs), targets, reduction="mean")
    u = torch.finfo(torch.float32).eps / 2
    assert loss.dtype == torch.float32
    assert abs(float(loss) - math.log(27)) <= 4 * u * math.log(27)


def test_I2_zero_output_layer_consequences(full_vocab) -> None:
    """A:I2 (M6-E4): at initialisation the gradients of U, W and b are exactly zero; one SGD step
    changes only V and c (U, W, b bitwise unchanged); h_1 is identical for every row."""
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=8), generator=make_generator(12))
    inputs = padded_batch(NAMES)
    targets = ref.forward_batch(reference_params(8, 1.0, make_generator(0)), inputs).targets
    before = {n: t.detach().clone() for n, t in model.named_parameters()}
    token_nll(model(inputs), targets, reduction="sum").backward()
    for name in ("input_lookup", "recurrent_weight", "hidden_bias"):
        assert torch.count_nonzero(getattr(model, name).grad) == 0, name
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    for name in ("input_lookup", "recurrent_weight", "hidden_bias"):
        assert torch.equal(getattr(model, name).detach(), before[name]), name
    for name in ("output_weight", "output_bias"):
        assert not torch.equal(getattr(model, name).detach(), before[name]), name
    h1 = model._recurrence(inputs).states[:, 0]
    assert all(torch.equal(h1[i], h1[0]) for i in range(h1.shape[0]))


@pytest.mark.parametrize("g", GAINS)
def test_I3_haar_construction(full_vocab, g) -> None:
    """A:I3: float64 Q with ‖QᵀQ − I‖_max ≤ 1e‑12; diag R > 0 after the sign correction; W equals
    (g·Q) cast to float32 bitwise; float32 singular values within 1e‑5 of g; b, V, c exactly zero."""
    H, seed = 8, 21
    Q = _haar_orthogonal(H, make_generator(seed))
    assert Q.dtype == torch.float64
    assert float((Q.T @ Q - torch.eye(H, dtype=torch.float64)).abs().max()) <= 1e-12
    Z = torch.randn(H, H, generator=make_generator(seed), dtype=torch.float64)  # the same first draw
    R = Q.T @ Z  # Z = Q R with the corrected Q, so R = Qᵀ Z
    assert bool((torch.diagonal(R) > 0).all())
    assert float(torch.tril(R, diagonal=-1).abs().max()) <= 1e-12
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=H, recurrent_gain=g), generator=make_generator(seed))
    assert torch.equal(model.recurrent_weight.detach(), (g * Q).to(torch.float32))
    sv = torch.linalg.svdvals(model.recurrent_weight.detach().to(torch.float64))
    assert float((sv - g).abs().max()) <= 1e-5
    for name in ("hidden_bias", "output_weight", "output_bias"):
        assert torch.count_nonzero(getattr(model, name)) == 0


def test_I4_draws_order_and_rng_isolation(full_vocab) -> None:
    """A:I4: the same generator seed gives bitwise identical parameters, different seeds differ;
    draw order W (H×H normal draws) then U (27×H normal draws), both float64; the global RNG is
    not touched by construction."""
    H, s2 = 8, 0.5
    state = torch.random.get_rng_state()
    a = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=H), generator=make_generator(31))
    b = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=H), generator=make_generator(31))
    c = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=H), generator=make_generator(32))
    assert torch.equal(torch.random.get_rng_state(), state)
    for name in PARAM_NAMES:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert not torch.equal(a.recurrent_weight, c.recurrent_weight)
    assert not torch.equal(a.input_lookup, c.input_lookup)
    gen = make_generator(31)
    torch.randn(H, H, generator=gen, dtype=torch.float64)  # W's draw comes first
    U = (math.sqrt(s2) * torch.randn(27, H, generator=gen, dtype=torch.float64)).to(torch.float32)
    assert torch.equal(a.input_lookup.detach(), U)


def test_I5_input_lookup_variance(full_vocab) -> None:
    """A:I5: the sample variance of U is within 5 standard errors of s²:
    |var − s²| ≤ 5·s²·√(2/(27H)) (27·H entries)."""
    for H, s2 in ((8, 0.5), (32, 0.5), (32, 2.0)):
        model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=H, input_init_variance=s2),
                                 generator=make_generator(41 + H))
        u = model.input_lookup.detach().to(torch.float64)
        var = float(u.var(unbiased=True))
        assert abs(var - s2) <= 5 * s2 * math.sqrt(2 / (27 * H)), (H, s2, var)


# =========================================================================== A:8 generation
GEN_NAMES = ("a", "pqrstuvwxyz", "hhhhhhhhh", "mavra", "mavrb", "kestravelunio", "ludmeriaxo")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_8a_teacher_forced_step_loop_equals_forward(fixture, fixture32, dtype) -> None:
    """A:8a: Path 1 ``forward`` on the padded batch; Path 2 per name h_0 = 0, ``_step`` with BOS
    then each character, ``_logits`` after each step. Equal at every valid position (A in float64,
    B in float32); states bitwise equal for a single-row batch (E)."""
    _, model = fixture if dtype == torch.float64 else fixture32
    crit = "A" if dtype == torch.float64 else "B"
    names = list(GEN_NAMES) + ["obelandrinas"]  # includes a 15-character-class long name
    z_batch = model(padded_batch(names))
    for beta, name in enumerate(names):
        ids = encode(name)
        h = torch.zeros(1, model.hidden_dim, dtype=dtype)
        step_states, step_logits = [], []
        for x in ids:
            h, _ = model._step(h, torch.tensor([x]), torch.tensor([True]))
            step_states.append(h[0])
            step_logits.append(model._logits(h)[0])
        close(torch.stack(step_logits), z_batch[beta, : len(ids)], crit, name)
        single = model._recurrence(padded_batch([name])).states[0]
        assert torch.equal(torch.stack(step_states), single), f"criterion E for {name}"


class _Script:
    """Scripted sampler for A:8b: returns predetermined tokens per sample and records the logits."""

    def __init__(self, plans: list[list[int]]) -> None:
        self.plans, self.calls, self.seen = plans, 0, {}

    def __call__(self, logits, *, temperature, top_k, top_p, generator):
        assert not torch.is_grad_enabled() and not logits.requires_grad
        active = [k for k, plan in enumerate(self.plans) if len(plan) > self.calls]
        assert logits.shape[0] == len(active), (self.calls, logits.shape, active)  # finished rows never sampled
        for row, k in enumerate(active):
            self.seen[(k, self.calls)] = logits[row].detach().clone()
        out = torch.tensor([self.plans[k][self.calls] for k in active], dtype=torch.long)
        self.calls += 1
        return out


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_8b_generate_with_scripted_sampler(fixture, fixture32, monkeypatch, dtype) -> None:
    """A:8b: rows "ann"+EOS, "b"+EOS, and six characters with ``max_length = 6`` whose 7th draw is
    not EOS. Recorded logits equal ``forward([BOS] + prefix)`` at the matching positions; texts,
    ``finished`` flags, at most ``max_length + 1`` draws, no draws after a row finishes;
    parameters, ``model.training`` and the global RNG unchanged; no autograd graph."""
    _, model = fixture if dtype == torch.float64 else fixture32
    crit = "A" if dtype == torch.float64 else "B"
    E = 0
    plans = [encode("ann")[1:] + [E], encode("b")[1:] + [E], encode("cdefgh")[1:] + [encode("z")[1]]]
    script = _Script(plans)
    monkeypatch.setattr(rnn_module, "sample_next_token", script)
    before = {n: t.detach().clone() for n, t in model.named_parameters()}
    model.train()
    rng = torch.random.get_rng_state()
    samples = model.generate(3, max_length=6, generator=make_generator(5))
    assert samples == [RNNSample("ann", True), RNNSample("b", True), RNNSample("cdefgh", False)]
    assert all("." not in s.text for s in samples)
    assert script.calls == 7  # the longest row: max_length + 1 draws
    assert sorted(k for k, _ in script.seen if k == 2) == [2] * 7
    assert len([k for k, _ in script.seen if k == 0]) == 4 and len([k for k, _ in script.seen if k == 1]) == 2
    for (k, i), logits in script.seen.items():
        prefix = [0] + plans[k][:i]
        expected = model(torch.tensor([prefix]))[0, -1]
        close(logits, expected, crit, f"sample {k} draw {i}")
    assert model.training is True
    assert torch.equal(torch.random.get_rng_state(), rng)
    for n, t in model.named_parameters():
        assert torch.equal(t.detach(), before[n]) and t.grad is None


def test_8c_greedy_decoding_is_self_consistent(full_vocab, generator_factory) -> None:
    """A:8c: with ``temperature=0`` each generated text, fed back through ``forward``, has its
    argmax equal to the generated character at every position, and EOS at the end when
    ``finished``. Fixture validity: a top-two logit gap ≥ 1e‑3 at every step; otherwise the next
    fixed alternative seed is used (guards against ties; not a tolerance)."""
    for attempt in range(10):
        p = reference_params(8, 1.0, generator_factory(300 + attempt))
        model = model_with(p, full_vocab, torch.float64)
        (sample,) = model.generate(1, max_length=12, temperature=0)
        ids = encode(sample.text)
        with torch.no_grad():
            z = model(torch.tensor([ids]))[0]
        top2 = torch.topk(z, 2, dim=1).values
        if float((top2[:, 0] - top2[:, 1]).min()) >= 1e-3:
            break
    else:  # pragma: no cover
        pytest.fail("no tie-free greedy fixture among 10 seeds")
    for i, ch in enumerate(sample.text):
        assert int(z[i].argmax()) == ord(ch) - ord("a") + 1
    last = int(z[len(sample.text)].argmax())
    assert (last == 0) == sample.finished
    many = model.generate(4, max_length=12, temperature=0)
    assert all(s == sample for s in many)  # greedy is deterministic across rows


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_8d_argument_validation(full_vocab, bad) -> None:
    """A:8d: ``num_samples`` and ``max_length`` must be positive ints (bool rejected);
    ``max_length`` is required."""
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=4), generator=make_generator(1))
    with pytest.raises(ValueError):
        model.generate(bad, max_length=5, temperature=0)
    with pytest.raises(ValueError):
        model.generate(2, max_length=bad, temperature=0)
    with pytest.raises(TypeError):
        model.generate(2, temperature=0)


def test_8_generation_with_explicit_generator_is_reproducible(fixture32) -> None:
    """§15: with an explicit generator, sampling is reproducible and leaves the global RNG alone;
    no text contains the boundary symbol; at most ``max_length`` characters."""
    _, model = fixture32
    rng = torch.random.get_rng_state()
    a = model.generate(16, max_length=8, generator=make_generator(9))
    b = model.generate(16, max_length=8, generator=make_generator(9))
    assert a == b and torch.equal(torch.random.get_rng_state(), rng)
    assert all(len(s.text) <= 8 and "." not in s.text for s in a)
    assert all(s.finished or len(s.text) == 8 for s in a)


# =========================================================================== A:K checkpoint
def _save(path: Path, model: torch.nn.Module, vocab, config, kind: str = "best") -> Path:
    extra = dict(best_metric=1.0) if kind == "best" else {}
    return save_checkpoint(path, kind=kind, model=model, config=config, vocab=vocab, **extra)


@pytest.mark.parametrize("kind", ["best", "final"])
def test_K1_checkpoint_round_trip(fixture32, full_vocab, tmp_path, kind) -> None:
    """A:K1 (D): saved through ``utils.checkpoint.save_checkpoint`` with the §14 metadata and
    reloaded with ``from_checkpoint``: logits bitwise identical."""
    _, model = fixture32
    path = _save(tmp_path / f"model_{kind}.pt", model, full_vocab.to_dict(),
                 {"model": asdict(model.config)}, kind)
    loaded = RNNLanguageModel.from_checkpoint(path, expected_vocab=full_vocab)
    assert isinstance(loaded, RNNLanguageModel) and loaded.config == model.config
    inputs = padded_batch(NAMES)
    assert torch.equal(loaded(inputs), model(inputs))


def test_K2_incompatible_checkpoints(full_vocab, tmp_path) -> None:
    """A:K2: wrong model class, different vocabulary, missing or invalid vocabulary, missing or
    invalid model config and mismatched weight shapes raise ``IncompatibleCheckpointError``."""
    model = RNNLanguageModel(full_vocab, RNNConfig(hidden_dim=8), generator=make_generator(1))
    good_cfg = {"model": asdict(model.config)}
    vocab = full_vocab.to_dict()

    class OtherModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    cases = {
        "wrong-class": _save(tmp_path / "a.pt", OtherModel(), vocab, good_cfg),
        "no-vocab": _save(tmp_path / "b.pt", model, None, good_cfg),
        "bad-vocab": _save(tmp_path / "c.pt", model, {"chars": ["a", "a"], "boundary": "."}, good_cfg),
        "no-config": _save(tmp_path / "d.pt", model, vocab, None),
        "no-model-config": _save(tmp_path / "e.pt", model, vocab, {"training": {}}),
        "invalid-config": _save(tmp_path / "f.pt", model, vocab, {"model": {"hidden_dim": 0}}),
        "unknown-key": _save(tmp_path / "g.pt", model, vocab, {"model": {**asdict(model.config), "dropout": 0.1}}),
        "shape-mismatch": _save(tmp_path / "h.pt", model, vocab, {"model": {**asdict(model.config), "hidden_dim": 16}}),
    }
    for label, path in cases.items():
        with pytest.raises(IncompatibleCheckpointError):
            RNNLanguageModel.from_checkpoint(path)
            pytest.fail(label)
    ok = _save(tmp_path / "ok.pt", model, vocab, good_cfg)
    with pytest.raises(IncompatibleCheckpointError, match="vocabulary mismatch"):
        RNNLanguageModel.from_checkpoint(ok, expected_vocab=Vocabulary.from_names(["xyz"]))


# =========================================================================== A:M model-facing mathematics
def test_M1_model_step_chain_jacobians(fixture) -> None:
    """A:M1 (criterion A): the autograd Jacobian of the model's ``_step`` chain h_s ↦ h_t equals the
    analytic product P_{t,s} = J_t ⋯ J_{s+1} of the independent reference, for every 0 ≤ s < t ≤ n."""
    p, model = fixture
    name = "corvinalt"
    f = ref.forward_name(p, encode(name))
    states = model._recurrence(padded_batch([name])).states[0].detach()
    for s in range(0, f.n):
        h_s = torch.zeros(p.H, dtype=torch.float64) if s == 0 else states[s - 1]
        for t in range(s + 1, f.n + 1):

            def chain(h, s=s, t=t):
                h = h.unsqueeze(0)
                for k in range(s + 1, t + 1):
                    h, _ = model._step(h, torch.tensor([f.inputs[k - 1]]), torch.tensor([True]))
                return h[0]

            J = torch.autograd.functional.jacobian(chain, h_s.clone())
            close(J, ref.jacobian_product(p, f, t, s), "A", f"P_{{{t},{s}}}")


def test_M2_model_loss_projected_vectors(fixture) -> None:
    """A:M2 (criterion A): autograd ∂ℓ_t/∂h_s through the model's ``_step`` and ``_logits`` equals
    the reference's P_{t,s}ᵀ r_t, for every 1 ≤ s ≤ t ≤ n."""
    p, model = fixture
    name = "wyndamorphelus"
    f = ref.forward_name(p, encode(name))
    states = model._recurrence(padded_batch([name])).states[0].detach()
    for s in range(1, f.n + 1):
        for t in range(s, f.n + 1):
            h = states[s - 1].clone().requires_grad_(True)
            ht = h.unsqueeze(0)
            for k in range(s + 1, t + 1):
                ht, _ = model._step(ht, torch.tensor([f.inputs[k - 1]]), torch.tensor([True]))
            loss = token_nll(model._logits(ht), torch.tensor([f.targets[t - 1]]), reduction="sum")
            (grad,) = torch.autograd.grad(loss, h)
            close(grad, ref.loss_projected_vector(p, f, t, s), "A", f"u_{{{t},{s}}}")


@pytest.mark.parametrize("width", [None, 21], ids=["tight", "padded"])
def test_M3_model_gradients_equal_bptt_formulas(fixture, width) -> None:
    """A:M3 (criterion A): autograd gradients of the unnormalised loss Σ ℓ on the model's own
    ``forward`` (through ``token_nll``, padded targets ignored) equal the §4.4 BPTT formulas of the
    reference for U, W, b, V and c; also the loss values agree."""
    p, model = fixture
    inputs = padded_batch(NAMES, width)
    exp = ref.forward_batch(p, inputs)
    model.zero_grad(set_to_none=True)
    loss = token_nll(model(inputs), exp.targets, reduction="sum")
    close(loss, ref.total_loss(p, inputs), "A", "loss")
    loss.backward()
    analytic = ref.bptt_gradients(p, inputs)
    for symbol, name in zip("UWbVc", PARAM_NAMES):
        close(getattr(model, name).grad, analytic[symbol], "A", f"∂L/∂{symbol}")


def test_M4_model_recurrent_gradient_equals_g3_sum(fixture) -> None:
    """A:M4 (criterion A): ``W.grad`` of the model's unnormalised loss equals Σ T_{t,k} of the
    reference's G3 decomposition; Σ π(Δ) = 1; κ_c is reported (A applies while κ_c ≥ 1e‑3)."""
    p, model = fixture
    inputs = padded_batch(NAMES, 18)
    dec = ref.g3_decomposition(p, inputs)
    model.zero_grad(set_to_none=True)
    token_nll(model(inputs), ref.forward_batch(p, inputs).targets, reduction="sum").backward()
    assert dec.kappa >= 1e-3, dec.kappa
    close(model.recurrent_weight.grad, dec.G, "A", "W.grad vs Σ T")
    assert abs(math.fsum(dec.share.values()) - 1.0) <= 1e-9


def test_M5_model_finite_differences(generator_factory, full_vocab) -> None:
    """A:M5 (FD): central differences of the model's float64 loss for every element of the five
    parameters (H = 3, g = 1.2) against the model's autograd gradient (rtol 1e‑6, atol 1e‑8)."""
    p = reference_params(3, 1.2, generator_factory(400))
    model = model_with(p, full_vocab, torch.float64)
    inputs = padded_batch(["zo", "bex", "mavra", "hhhhhhhhh"], 11)
    targets = ref.forward_batch(p, inputs).targets
    model.zero_grad(set_to_none=True)
    token_nll(model(inputs), targets, reduction="sum").backward()
    checked = 0
    with torch.no_grad():
        for name in PARAM_NAMES:
            param = getattr(model, name)
            grad = param.grad.clone()
            for idx in itertools.product(*(range(d) for d in param.shape)):
                orig = float(param[idx])
                param[idx] = orig + FD_STEP
                plus = float(token_nll(model(inputs), targets, reduction="sum"))
                param[idx] = orig - FD_STEP
                minus = float(token_nll(model(inputs), targets, reduction="sum"))
                param[idx] = orig
                fd, an = (plus - minus) / (2 * FD_STEP), float(grad[idx])
                assert abs(fd - an) <= FD_ATOL + FD_RTOL * abs(an), f"{name}{list(idx)}: FD {fd} vs {an}"
                checked += 1
    assert checked == 3 * 3 + 55 * 3 + 27


# =========================================================================== accelerator check
@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="MPS not available")
def test_accelerator_forward_against_reference_mps(fixture32) -> None:
    """§13.2 general device checks: the float32 forward path on MPS against the float64 reference
    (criterion C), with no bitwise expectation. MPS is not used for training (D-46)."""
    p, model = fixture32
    inputs = padded_batch(NAMES)
    mps_model = model.to("mps")
    z = mps_model(inputs.to("mps"))
    assert z.device.type == "mps"
    close(z.cpu(), ref.forward_batch(p, inputs).logits, "C", "MPS logits")

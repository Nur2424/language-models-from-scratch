"""Tests for models/feedforward/mlp.py, following the specification (v0.3) §13.

Fixtures
--------
SMALL  12 names (63 windows, k = 3), used for most behavioural tests.
TINY   ["ab", "ac", "ab"]: 9 windows; empirical conditional entropy
       Ĥ(Y|W) = (2 ln 1.5 + ln 3) / 9 = 0.212171 (only the context [., ., a] is uncertain).
V27    the full 27-class vocabulary, used where the spec's statistics assume V = 27.

Mathematical tests (§13.2) are numbered as in the specification.
"""

from __future__ import annotations

import ast
import math
import string
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.func import functional_call

import models.feedforward.mlp as mlp_module
from models.feedforward.mlp import (
    EpochRecord,
    IncompatibleCheckpointError,
    MLPConfig,
    MLPLanguageModel,
    MLPSample,
    MLPTrainingConfig,
    MLPTrainingResult,
    evaluate_nll,
    per_unit_saturation,
    run_config,
    train_mlp,
)
from utils.batching import EpochSampler
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import ConfigError, default_config_paths, load_config
from utils.data import ContextWindows, load_names, make_context_windows, split_names
from utils.device import is_mps_available
from utils.diagnostics.schedule import DiagnosticsSchedule
from utils.io import read_jsonl
from utils.metrics import NLLAccumulator, count_parameters, parameter_breakdown, token_nll
from utils.paths import DEFAULT_NAMES_PATH
from utils.reproducibility import derive_seed, make_generator
from utils.run_manager import RunManager, RunStatus
from utils.vocabulary import Vocabulary

LN27 = math.log(27.0)
SMALL = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "liam", "noah", "anna", "anna", "eva", "bo"]
TINY = ["ab", "ac", "ab"]
H_TINY = (2 * math.log(1.5) + math.log(3.0)) / 9
DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("mps", id="mps", marks=pytest.mark.skipif(not is_mps_available(), reason="MPS unavailable")),
]


# ------------------------------------------------------------------ helpers
@pytest.fixture
def v27() -> Vocabulary:
    return Vocabulary.from_names(list(string.ascii_lowercase))


@pytest.fixture
def small_vocab() -> Vocabulary:
    return Vocabulary.from_names(SMALL)


@pytest.fixture
def small(small_vocab) -> ContextWindows:
    return make_context_windows(SMALL, small_vocab, context_length=3)


BASELINE = MLPConfig()


def _model(vocab: Vocabulary, seed: int = 0, config: MLPConfig = BASELINE) -> MLPLanguageModel:
    return MLPLanguageModel(vocab, config, generator=make_generator(seed))


def _randomise_output(model: MLPLanguageModel, seed: int = 123) -> MLPLanguageModel:
    """Give b₁, W₂ and b₂ generic (non-zero) values, for tests that need a general parameter point."""
    g = make_generator(seed)
    with torch.no_grad():
        for p in (model.hidden.bias, model.out.weight, model.out.bias):
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.5)
    return model


def _subset(w: ContextWindows, n: int) -> ContextWindows:
    return ContextWindows(contexts=w.contexts[:n], targets=w.targets[:n], name_index=w.name_index[:n])


def _conditional_entropy(w: ContextWindows) -> float:
    """Ĥ(Y|W) = Σ_w (N_w/M) H(q_w), computed by counting (independent of the model)."""
    by_ctx: dict[tuple, Counter] = defaultdict(Counter)
    for c, y in zip(w.contexts.tolist(), w.targets.tolist()):
        by_ctx[tuple(c)][y] += 1
    total = 0.0
    for counts in by_ctx.values():
        n = sum(counts.values())
        total -= sum(k * math.log(k / n) for k in counts.values())
    return total / w.num_examples


def _run(tmp_path: Path) -> RunManager:
    return RunManager.create("mlp", config={}, results_root=tmp_path)


def _rows(history) -> list[dict]:
    """History rows without the (non-deterministic) wall-clock field."""
    out = []
    for r in history:
        d = asdict(r) if isinstance(r, EpochRecord) else dict(r)
        d.pop("seconds")
        out.append(d)
    return out


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_same_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k, value in a.items():
        assert torch.equal(value.cpu(), b[k].cpu()), k


def _manual_epochs(model, w, *, batch_size, lr, seed, epochs, denominator=None):
    """Independent re-implementation of the §4.6 loop. Returns the per-batch (size, mean loss) pairs."""
    sampler = EpochSampler(w.num_examples, batch_size, seed=derive_seed(seed, "data"))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    batches = []
    for e in range(epochs):
        for idx in sampler.batches(e):
            opt.zero_grad(set_to_none=True)
            total = token_nll(model(w.contexts[idx]), w.targets[idx], reduction="sum")
            loss = total / (len(idx) if denominator is None else denominator)
            batches.append((len(idx), float(total.detach()) / len(idx)))
            loss.backward()
            opt.step()
    return batches


# ============================================================ §13.1 general
class TestConfigAndConstruction:
    def test_defaults_and_fan_in(self):
        c = MLPConfig()
        assert (c.context_length, c.embedding_dim, c.hidden_dim, c.fan_in) == (3, 10, 200, 30)

    @pytest.mark.parametrize("field", ["context_length", "embedding_dim", "hidden_dim"])
    @pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3", None])
    def test_mlp_config_validation(self, field, bad):
        with pytest.raises(ValueError, match=field):
            MLPConfig(**{field: bad})

    def test_training_config_valid_and_normalised(self):
        t = MLPTrainingConfig(lr=1, max_epochs=4, lr_decay_epoch=2, lr_decay_factor=1)
        assert isinstance(t.lr, float) and isinstance(t.lr_decay_factor, float)
        assert (t.batch_size, t.lr_decay_epoch, t.eval_batch_size) == (32, 2, 4096)
        assert MLPTrainingConfig(lr=0.1, max_epochs=1).lr_decay_epoch is None

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"lr": 0}, "lr"), ({"lr": -0.1}, "lr"), ({"lr": float("nan")}, "lr"),
            ({"lr": float("inf")}, "lr"), ({"lr": True}, "lr"),
            ({"max_epochs": 0}, "max_epochs"), ({"max_epochs": True}, "max_epochs"), ({"max_epochs": 2.0}, "max_epochs"),
            ({"batch_size": 0}, "batch_size"), ({"eval_batch_size": 0}, "eval_batch_size"),
            ({"lr_decay_epoch": 0}, "lr_decay_epoch"), ({"lr_decay_epoch": 4}, "lr_decay_epoch"),
            ({"lr_decay_factor": 0.0}, "lr_decay_factor"), ({"lr_decay_factor": 1.5}, "lr_decay_factor"),
            ({"lr_decay_factor": float("nan")}, "lr_decay_factor"),
        ],
    )
    def test_training_config_validation(self, kwargs, match):
        base = {"lr": 0.1, "max_epochs": 4}
        with pytest.raises(ValueError, match=match):
            MLPTrainingConfig(**{**base, **kwargs})

    def test_constructor_validation(self, v27):
        with pytest.raises(TypeError):
            MLPLanguageModel(v27)  # generator is required
        with pytest.raises(TypeError, match="generator"):
            MLPLanguageModel(v27, generator=0)
        with pytest.raises(TypeError, match="vocab"):
            MLPLanguageModel("abc", generator=make_generator(0))
        with pytest.raises(TypeError, match="MLPConfig"):
            MLPLanguageModel(v27, {"context_length": 3}, generator=make_generator(0))

    def test_yaml_config(self):
        cfg = load_config(default_config_paths("mlp"), model_config_cls=MLPConfig)
        assert cfg.model.name == "mlp"
        assert cfg.model.params == MLPConfig()
        with pytest.raises(ConfigError, match="unknown"):
            load_config(default_config_paths("mlp"), overrides=["model.params.dropout=0.1"], model_config_cls=MLPConfig)
        with pytest.raises(ConfigError):
            load_config(default_config_paths("mlp"), overrides=["model.params.hidden_dim=0"], model_config_cls=MLPConfig)


class TestInputContract:
    def test_shapes_and_dtype(self, v27):
        m = _model(v27)
        z = m(torch.tensor([[0, 0, 0], [1, 2, 3]]))
        assert z.shape == (2, 27) and z.dtype == torch.float32

    def test_empty_batch(self, v27):
        m = _model(v27)
        z = m(torch.zeros(0, 3, dtype=torch.long))
        assert z.shape == (0, 27)
        # the metadata checks still apply when B = 0
        with pytest.raises(ValueError, match="shape"):
            m(torch.zeros(0, 2, dtype=torch.long))
        with pytest.raises(ValueError, match="shape"):
            m(torch.zeros(0, dtype=torch.long))
        with pytest.raises(TypeError, match="int64"):
            m(torch.zeros(0, 3))
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(0, 3, dtype=torch.long, device="meta"))

    @pytest.mark.parametrize("shape", [(3,), (2, 2), (2, 4), (1, 3, 1)])
    def test_wrong_shape(self, v27, shape):
        with pytest.raises(ValueError, match="shape"):
            _model(v27)(torch.zeros(shape, dtype=torch.long))

    def test_type_and_dtype(self, v27):
        m = _model(v27)
        with pytest.raises(TypeError, match="torch.Tensor"):
            m([[0, 0, 0]])
        with pytest.raises(TypeError, match="int64"):
            m(torch.zeros(1, 3, dtype=torch.int32))

    def test_device_mismatch(self, v27):
        with pytest.raises(ValueError, match="move them explicitly"):
            _model(v27)(torch.zeros(2, 3, dtype=torch.long, device="meta"))

    def test_padding_rejected(self, v27):
        with pytest.raises(ValueError, match="padding must not reach the model"):
            _model(v27)(torch.tensor([[0, 0, v27.pad_id]]))

    @pytest.mark.parametrize("bad", [-1, 28, 100])
    def test_out_of_range_rejected(self, v27, bad):
        with pytest.raises(ValueError, match="outside"):
            _model(v27)(torch.tensor([[0, bad, 1]]))

    def test_train_eval_identical(self, v27):
        m = _randomise_output(_model(v27))
        x = torch.tensor([[0, 0, 1], [4, 5, 6]])
        m.train()
        a = m(x)
        m.eval()
        assert torch.equal(a, m(x))


class TestParameters:
    def test_count_and_breakdown(self, v27):
        m = _model(v27)
        assert count_parameters(m).total == count_parameters(m).trainable == 11_897
        assert {k: c.total for k, c in parameter_breakdown(m).items()} == {"emb": 270, "hidden": 6_200, "out": 5_427}
        names = [n for n, _ in m.named_parameters()]
        assert names == ["emb.weight", "hidden.weight", "hidden.bias", "out.weight", "out.bias"]
        assert tuple(m.hidden.weight.shape) == (200, 30) and tuple(m.out.weight.shape) == (27, 200)

    def test_general_formula(self, v27):
        k, d, H, V = 5, 7, 11, 27
        m = _model(v27, config=MLPConfig(context_length=k, embedding_dim=d, hidden_dim=H))
        assert count_parameters(m).total == V * d + (k * d) * H + H + H * V + V

    def test_tanh_is_registered_submodule(self, v27):
        m = _model(v27)
        assert isinstance(dict(m.named_modules())["act"], nn.Tanh)
        assert isinstance(m.emb, nn.Embedding) and m.emb.padding_idx is None

    def test_module_imports_no_other_model(self):
        tree = ast.parse(Path(mlp_module.__file__).read_text())
        imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [i for i in imported if i and i.startswith("models")]


class TestInitialisation:
    def test_no_default_survives(self, v27):
        m = _model(v27)
        for p in (m.hidden.bias, m.out.weight, m.out.bias):
            assert torch.count_nonzero(p) == 0
        assert torch.count_nonzero(m.emb.weight) == 270 and torch.count_nonzero(m.hidden.weight) == 6000

    def test_deterministic_in_init_generator_only(self, v27):
        torch.manual_seed(1)
        a = _model(v27, seed=5)
        torch.manual_seed(2)  # a different global RNG state must not matter
        b = _model(v27, seed=5)
        c = _model(v27, seed=6)
        _assert_same_state(_state(a), _state(b))
        assert not torch.equal(a.emb.weight, c.emb.weight)
        assert not torch.equal(a.hidden.weight, c.hidden.weight)

    def test_construction_does_not_touch_global_rng(self, v27):
        before = torch.get_rng_state()
        _model(v27)
        assert torch.equal(before, torch.get_rng_state())


# ====================================================== §13.2 mathematical
def _capture(model: MLPLanguageModel, name: str, x: torch.Tensor, *, inputs: bool = False) -> torch.Tensor:
    store = {}
    module = dict(model.named_modules())[name]
    h = module.register_forward_hook(lambda _m, i, o: store.__setitem__("t", (i[0] if inputs else o).detach()))
    try:
        model(x)
    finally:
        h.remove()
    return store["t"]


class TestMathematics:
    def test_1_embedding_sharing(self, v27):
        m = _model(v27)
        assert sum(isinstance(mod, nn.Embedding) for mod in m.modules()) == 1
        x = torch.tensor([[3, 3, 7]])
        u = _capture(m, "hidden", x, inputs=True)[0]
        assert torch.equal(u[0:10], u[10:20]) and torch.equal(u[0:10], m.emb.weight[3])
        assert torch.equal(u[20:30], m.emb.weight[7])
        with torch.no_grad():
            m.emb.weight[3] += 1.0
        u2 = _capture(m, "hidden", x, inputs=True)[0]
        assert torch.allclose(u2[:20] - u[:20], torch.ones(20)) and torch.equal(u2[20:], u[20:])

    def test_2_position_blocks(self, v27):
        m = _randomise_output(_model(v27))
        x = torch.tensor([[2, 9, 14]])
        W1, b1, E = m.hidden.weight, m.hidden.bias, m.emb.weight
        a = _capture(m, "hidden", x)[0]
        expected = sum(W1[:, 10 * p:10 * p + 10] @ E[x[0, p]] for p in range(3)) + b1
        assert torch.allclose(a, expected, atol=1e-6)
        for p in range(3):  # perturbing block p changes a by Δ_p · E[x_p] only
            delta = torch.randn(200, 10, generator=make_generator(p))
            with torch.no_grad():
                m.hidden.weight[:, 10 * p:10 * p + 10] += delta
            a_new = _capture(m, "hidden", x)[0]
            assert torch.allclose(a_new - a, delta @ E[x[0, p]], atol=1e-5)
            a = a_new

    def test_3_position_sensitivity(self, v27):
        m = _randomise_output(_model(v27))
        z = m(torch.tensor([[1, 2, 3], [3, 2, 1], [2, 1, 3]]))
        assert not torch.allclose(z[0], z[1]) and not torch.allclose(z[0], z[2])

    def test_4_forward_oracle(self, v27):
        m = _randomise_output(_model(v27))
        x = torch.tensor([[0, 0, 0], [0, 0, 5], [1, 2, 3], [26, 26, 1]])
        onehot = torch.nn.functional.one_hot(x, 27).float()  # [B, 3, 27]
        u = (onehot @ m.emb.weight).reshape(4, 30)
        h = torch.tanh(u @ m.hidden.weight.T + m.hidden.bias)
        z = h @ m.out.weight.T + m.out.bias
        assert torch.allclose(m(x), z, atol=1e-6)

    def test_5_logit_contract_and_cross_entropy(self, v27):
        m = _randomise_output(_model(v27))
        x, y = torch.tensor([[0, 0, 0], [1, 2, 3], [4, 4, 4]]), torch.tensor([1, 0, 26])
        z = m(x)
        hand = (torch.logsumexp(z, dim=1) - z[torch.arange(3), y]).mean()
        assert torch.allclose(token_nll(z, y, reduction="mean"), hand, atol=1e-6)
        # hand-set point: W₂ = 0, b₂ = c ⇒ every row of logits equals c
        c = [0.1 * j for j in range(27)]
        with torch.no_grad():
            m.out.weight.zero_()
            m.out.bias.copy_(torch.tensor(c))
        lse = math.log(sum(math.exp(v) for v in c))
        expected = ((lse - c[1]) + (lse - c[2])) / 2
        loss = token_nll(m(torch.tensor([[0, 0, 0], [5, 6, 7]])), torch.tensor([1, 2]), reduction="mean")
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_6_shared_embedding_gradient_accumulation(self, v27):
        m = _randomise_output(_model(v27)).double()
        x = torch.tensor([[0, 0, 0], [0, 0, 5], [5, 7, 5], [1, 2, 3]])
        y = torch.tensor([5, 7, 0, 4])
        m.zero_grad()
        token_nll(m(x), y, reduction="mean").backward()
        with torch.no_grad():
            u = m.emb.weight[x].reshape(4, 30)
            h = torch.tanh(u @ m.hidden.weight.T + m.hidden.bias)
            p = torch.softmax(h @ m.out.weight.T + m.out.bias, dim=1)
            g = (p - torch.nn.functional.one_hot(y, 27).double()) / 4  # ∂L/∂z
            delta = (g @ m.out.weight) * (1 - h ** 2)  # ∂L/∂a
            expected = torch.zeros_like(m.emb.weight)
            for n in range(4):
                for pos in range(3):  # accumulate: one contribution per occurrence
                    expected[x[n, pos]] += m.hidden.weight[:, 10 * pos:10 * pos + 10].T @ delta[n]
        assert torch.allclose(m.emb.weight.grad, expected, atol=1e-12)
        assert expected[0].abs().sum() > 0  # row 0 really received five contributions

    def test_7_finite_difference_gradcheck(self, v27):
        m = _randomise_output(_model(v27, config=MLPConfig(hidden_dim=8))).double()
        x, y = torch.tensor([[0, 0, 0], [0, 1, 2], [2, 2, 3]]), torch.tensor([1, 3, 0])
        names = [n for n, _ in m.named_parameters()]
        params = tuple(p.detach().clone().requires_grad_(True) for p in m.parameters())

        def f(*ps):
            return token_nll(functional_call(m, dict(zip(names, ps)), (x,)), y, reduction="mean")

        assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-6)

    def test_8_zero_output_initialisation(self, v27, small):
        m = _model(v27)
        w = make_context_windows(SMALL, v27, context_length=3)
        assert evaluate_nll(m, w, batch_size=17).mean_nll == pytest.approx(LN27, abs=1e-6)
        x, y = w.contexts[:32], w.targets[:32]
        loss = token_nll(m(x), y, reduction="mean")
        loss.backward()
        for p in (m.emb.weight, m.hidden.weight, m.hidden.bias):
            assert torch.count_nonzero(p.grad) == 0  # exactly zero: every path passes through W₂ = 0
        with torch.no_grad():
            h = torch.tanh(m.hidden(m.emb(x).reshape(32, 30)))
            g = (torch.full((32, 27), 1 / 27) - torch.nn.functional.one_hot(y, 27)) / 32
        assert torch.allclose(m.out.weight.grad, g.T @ h, atol=1e-7)
        assert torch.allclose(m.out.bias.grad, g.sum(0), atol=1e-7)
        torch.optim.SGD(m.parameters(), lr=0.1).step()
        m.zero_grad()
        token_nll(m(x), y, reduction="mean").backward()
        for name, p in m.named_parameters():
            assert torch.count_nonzero(p.grad) > 0, name

    def test_9_initialisation_statistics(self, v27):
        m = _model(v27, seed=0)
        E, W1 = m.emb.weight, m.hidden.weight
        assert abs(E.std().item() - 1) < 0.2 and abs(E.mean().item()) < 0.3
        assert abs(W1.std().item() * math.sqrt(30) - 1) < 0.05 and abs(W1.mean().item()) < 0.012
        # Pre-activation std on the small set. For a single seed it varies substantially with the 27×10
        # draw of E (observed 0.78–1.32 over 200 seeds), so the bound is applied to the mean over 20 seeds.
        w = make_context_windows(SMALL, v27, context_length=3)
        stds, sats = [], []
        for s in range(20):
            mm = _model(v27, seed=1000 + s)
            a = _capture(mm, "hidden", w.contexts)
            stds.append(a.std().item())
            sats.append((torch.tanh(a).abs() > 0.97).float().mean().item())
        assert abs(sum(stds) / len(stds) - 1) < 0.15
        assert all(0 <= s < 1 for s in sats)  # saturation is reported, not asserted tightly

    def test_9_initialisation_scale_non_default_config(self, v27):
        m = _model(v27, seed=0, config=MLPConfig(context_length=8, embedding_dim=10, hidden_dim=200))
        W1 = m.hidden.weight
        assert W1.shape == (200, 80)
        assert abs(W1.std().item() * math.sqrt(80) - 1) < 0.05 and abs(W1.mean().item()) < 0.01
        assert abs(W1.std().item() * math.sqrt(30) - 1) > 0.05  # not the baseline 1/√30 scale
        assert abs(m.emb.weight.std().item() - 1) < 0.2
        for p in (m.hidden.bias, m.out.weight, m.out.bias):
            assert torch.count_nonzero(p) == 0

    @pytest.mark.parametrize("n_windows", [42])  # batches of 32 and 10
    def test_10_mean_reduction_over_actual_batch_size(self, small_vocab, small, n_windows):
        w = _subset(small, n_windows)
        assert EpochSampler(n_windows, 32, seed=0).batch_sizes() == [32, 10]
        trained = _model(small_vocab)
        train_mlp(trained, w, w, training=MLPTrainingConfig(lr=0.5, max_epochs=2), seed=3)
        reference = _model(small_vocab)
        _manual_epochs(reference, w, batch_size=32, lr=0.5, seed=3, epochs=2)
        for (n, a), b in zip(trained.named_parameters(), reference.parameters()):
            assert torch.allclose(a, b, rtol=0, atol=1e-6), n
        nominal = _model(small_vocab)
        _manual_epochs(nominal, w, batch_size=32, lr=0.5, seed=3, epochs=2, denominator=32)
        assert (trained.out.bias - nominal.out.bias).abs().max() > 1e-3  # sum/32 is a different procedure

    def test_11_running_nll_is_example_weighted(self, small_vocab, small):
        w = _subset(small, 42)
        result = train_mlp(_model(small_vocab), w, w, training=MLPTrainingConfig(lr=0.5, max_epochs=1), seed=3)
        batches = _manual_epochs(_model(small_vocab), w, batch_size=32, lr=0.5, seed=3, epochs=1)
        weighted = sum(n * loss for n, loss in batches) / 42
        mean_of_means = sum(loss for _, loss in batches) / len(batches)
        assert result.history[0].running_train_nll == pytest.approx(weighted, rel=1e-6)
        assert abs(weighted - mean_of_means) > 1e-4

    @pytest.mark.parametrize("batch_size", [1, 7, 62, 63, 64, 4096])
    def test_12_evaluation_covers_every_example(self, small_vocab, small, batch_size):
        m = _randomise_output(_model(small_vocab))
        acc = evaluate_nll(m, small, batch_size=batch_size)
        assert acc.num_tokens == small.num_examples == 63
        with torch.no_grad():
            full = token_nll(m(small.contexts), small.targets, reduction="mean").item()
        assert acc.mean_nll == pytest.approx(full, rel=1e-6)

    def test_13_conditional_entropy_bound(self, small_vocab, small):
        tiny_vocab = Vocabulary.from_names(TINY)
        tiny = make_context_windows(TINY, tiny_vocab, context_length=3)
        assert _conditional_entropy(tiny) == pytest.approx(H_TINY, abs=1e-12)
        for vocab, w in ((tiny_vocab, tiny), (small_vocab, small)):
            bound = _conditional_entropy(w)
            for seed in range(3):
                m = _randomise_output(_model(vocab, seed=seed), seed=seed)
                assert evaluate_nll(m, w, batch_size=64).mean_nll >= bound - 1e-6
            m = _model(vocab)
            train_mlp(m, w, w, training=MLPTrainingConfig(lr=0.5, max_epochs=30, batch_size=8), seed=0)
            assert evaluate_nll(m, w, batch_size=64).mean_nll >= bound - 1e-6


# ========================================================= §13.3 behaviour
class TestTraining:
    def test_overfit_tiny_fixture(self):
        vocab = Vocabulary.from_names(TINY)
        w = make_context_windows(TINY, vocab, context_length=3)
        m = _model(vocab)
        train_mlp(m, w, w, training=MLPTrainingConfig(lr=0.5, max_epochs=500), seed=0)
        gap = evaluate_nll(m, w, batch_size=64).mean_nll - H_TINY
        assert -1e-6 <= gap <= 1e-2

    def test_result_and_history(self, small_vocab, small):
        r = train_mlp(_model(small_vocab), small, small, training=MLPTrainingConfig(lr=0.3, max_epochs=3), seed=1)
        assert isinstance(r, MLPTrainingResult) and r.epochs_completed == 3
        assert [h.epoch for h in r.history] == [0, 1, 2]
        assert [h.steps for h in r.history] == [2, 4, 6]  # 63 windows → batches 32 + 31
        for h in r.history:
            acts = h.activations
            assert {"hidden", "act", "per_unit_saturation"} <= acts.keys()
            assert "saturation_fraction" in acts["act"]
            assert acts["hidden"]["numel"] == 63 * 200
        best = min(range(3), key=lambda e: (r.history[e].dev_nll, e))
        assert r.best_epoch == best and r.best_dev_nll == r.history[best].dev_nll

    def test_three_nlls_are_distinct_quantities(self, small_vocab, small):
        r = train_mlp(_model(small_vocab), small, small, training=MLPTrainingConfig(lr=0.5, max_epochs=1), seed=0)
        h = r.history[0]
        assert h.train_nll == h.dev_nll  # same windows, same fixed parameters
        assert h.running_train_nll != h.train_nll  # parameters changed during the epoch

    def test_determinism(self, small_vocab, small):
        cfg = MLPTrainingConfig(lr=0.3, max_epochs=3)
        a, b, c = _model(small_vocab), _model(small_vocab), _model(small_vocab)
        ra = train_mlp(a, small, small, training=cfg, seed=7)
        rb = train_mlp(b, small, small, training=cfg, seed=7)
        rc = train_mlp(c, small, small, training=cfg, seed=8)
        _assert_same_state(_state(a), _state(b))
        assert _rows(ra.history) == _rows(rb.history)
        assert not torch.equal(a.hidden.weight, c.hidden.weight) and _rows(ra.history) != _rows(rc.history)

    def test_schedule(self, small_vocab, small):
        cfg = MLPTrainingConfig(lr=0.4, max_epochs=4, lr_decay_epoch=2, lr_decay_factor=0.1)
        r = train_mlp(_model(small_vocab), small, small, training=cfg, seed=0)
        assert [h.lr for h in r.history] == [0.4, 0.4, 0.4 * 0.1, 0.4 * 0.1]
        r = train_mlp(_model(small_vocab), small, small, training=replace(cfg, lr_decay_epoch=None), seed=0)
        assert [h.lr for h in r.history] == [0.4] * 4

    def test_selection_ties_keep_earliest_epoch(self, small_vocab, small, tmp_path, monkeypatch):
        dev = _subset(small, 40)
        values = iter([3.0, 2.0, 2.0, 2.5])
        real = mlp_module.evaluate_nll

        def fake(model, windows, *, batch_size):
            acc = real(model, windows, batch_size=batch_size)
            if windows is dev:
                acc = NLLAccumulator()
                acc.update_from_sum(next(values) * 40, 40)
            return acc

        monkeypatch.setattr(mlp_module, "evaluate_nll", fake)
        run = _run(tmp_path)
        r = train_mlp(_model(small_vocab), small, dev, training=MLPTrainingConfig(lr=0.1, max_epochs=4),
                      seed=0, run=run)
        assert [h.is_best for h in r.history] == [True, True, False, False]
        assert (r.best_epoch, r.best_dev_nll) == (1, 2.0)
        best = load_checkpoint(run.checkpoint_path("best"))
        assert best["extra"]["best_epoch"] == 1 and best["best_metric"] == 2.0 and best["epoch"] == 2
        assert load_checkpoint(run.checkpoint_path("resume"))["extra"]["best_epoch"] == 1

    def test_argument_validation(self, small_vocab, small, tmp_path):
        m, cfg = _model(small_vocab), MLPTrainingConfig(lr=0.1, max_epochs=1)
        with pytest.raises(ValueError, match="resume=True requires a run"):
            train_mlp(m, small, small, training=cfg, seed=0, resume=True)
        with pytest.raises(ValueError, match="pass a run"):
            train_mlp(m, small, small, training=cfg, seed=0, diagnostics=DiagnosticsSchedule(enabled=True))
        with pytest.raises(TypeError, match="MLPTrainingConfig"):
            train_mlp(m, small, small, training={"lr": 0.1}, seed=0)
        w2 = make_context_windows(SMALL, small_vocab, context_length=2)
        with pytest.raises(ValueError, match="context_length"):
            train_mlp(m, w2, w2, training=cfg, seed=0)
        run = _run(tmp_path)
        train_mlp(m, small, small, training=cfg, seed=0, run=run)
        with pytest.raises(ValueError, match="resume=True"):  # never silently restart into an existing run
            train_mlp(m, small, small, training=cfg, seed=0, run=run)


class TestNonFiniteGuard:
    def test_nan_loss_stops_before_step(self, small_vocab, small, monkeypatch):
        m = _model(small_vocab)
        real, calls, before = mlp_module.token_nll, [0], {}

        def poisoned(logits, targets, *, reduction):
            calls[0] += 1
            loss = real(logits, targets, reduction=reduction)
            if calls[0] == 4:  # epoch 1, step 3
                before.update(_state(m))
                return loss * float("nan")
            return loss

        monkeypatch.setattr(mlp_module, "token_nll", poisoned)
        with pytest.raises(FloatingPointError, match=r"non-finite loss .* at epoch 1, step 3"):
            train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.1, max_epochs=3), seed=0)
        _assert_same_state(before, _state(m))  # optimizer.step() was not called

    def test_nan_gradient_stops_before_step(self, small_vocab, small):
        m = _model(small_vocab)
        calls, before = [0], {}

        def hook(grad):
            calls[0] += 1
            if calls[0] == 3:  # epoch 1, step 2
                before.update(_state(m))
                return grad * float("nan")
            return grad

        m.hidden.weight.register_hook(hook)
        with pytest.raises(FloatingPointError, match=r"epoch 1, step 2 in parameter groups \['hidden'\]") as info:
            train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.1, max_epochs=3), seed=0)
        assert "hidden.weight" in str(info.value) and "nan" in str(info.value)
        _assert_same_state(before, _state(m))

    def test_failure_is_recorded_and_last_checkpoint_kept(self, small_vocab, small, tmp_path):
        m, run = _model(small_vocab), _run(tmp_path)
        calls = [0]

        def hook(grad):
            calls[0] += 1
            return grad * float("inf") if calls[0] == 3 else grad

        m.hidden.weight.register_hook(hook)
        with pytest.raises(FloatingPointError), run.track():
            train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.1, max_epochs=3), seed=0, run=run)
        assert run.status is RunStatus.FAILED
        assert load_checkpoint(run.checkpoint_path("resume"))["epoch"] == 1
        assert not run.checkpoint_path("final").exists()


class TestEvaluationHelpers:
    @pytest.mark.parametrize("fn", ["evaluate_nll", "per_unit_saturation"])
    @pytest.mark.parametrize("training", [True, False])
    def test_mode_restored(self, small_vocab, small, fn, training):
        m = _model(small_vocab)
        m.train(training)
        getattr(mlp_module, fn)(m, small, batch_size=10)
        assert m.training is training
        bad = ContextWindows(contexts=small.contexts.clone(), targets=small.targets, name_index=small.name_index)
        bad.contexts[40, 0] = small_vocab.pad_id  # fails part-way through (third chunk)
        with pytest.raises(ValueError, match="padding"):
            getattr(mlp_module, fn)(m, bad, batch_size=10)
        assert m.training is training

    def test_per_unit_saturation_matches_hand_computation(self, small_vocab, small):
        m = _randomise_output(_model(small_vocab))
        with torch.no_grad():
            m.hidden.weight.mul_(3.0)  # make saturation common
            h = torch.tanh(m.hidden(m.emb(small.contexts).reshape(63, 30)))
        expected = (h.abs() > 0.97).double().mean(0)
        got = per_unit_saturation(m, small, batch_size=10)
        assert got.shape == (200,) and got.dtype == torch.float64
        assert torch.allclose(got, expected, atol=1e-12)
        with pytest.raises(ValueError, match="threshold"):
            per_unit_saturation(m, small, threshold=1.0, batch_size=10)

    def test_evaluate_nll_validation(self, small_vocab, small):
        m = _model(small_vocab)
        with pytest.raises(ValueError, match="batch_size"):
            evaluate_nll(m, small, batch_size=0)
        with pytest.raises(TypeError, match="ContextWindows"):
            evaluate_nll(m, small.contexts, batch_size=4)


class TestDiagnostics:
    def test_step_diagnostics(self, small_vocab, small, tmp_path):
        cfg = MLPTrainingConfig(lr=0.3, max_epochs=3)
        plain = _model(small_vocab)
        train_mlp(plain, small, small, training=cfg, seed=0)
        m, run = _model(small_vocab), _run(tmp_path)
        schedule = DiagnosticsSchedule(enabled=True, every_n_steps=2, gradients=True, parameters=True,
                                       activations=True)
        train_mlp(m, small, small, training=cfg, seed=0, run=run, diagnostics=schedule)
        _assert_same_state(_state(plain), _state(m))  # diagnostics never change training
        rows = read_jsonl(run.diagnostics_dir / "steps.jsonl")
        assert [r["step"] for r in rows] == [0, 2, 4]
        r0 = rows[0]
        assert set(r0["gradients"]) == {"emb", "hidden", "out"}
        assert r0["gradients"]["emb"]["norm"] == 0.0 and r0["gradients"]["out"]["norm"] > 0  # §7, step 0
        assert set(r0["activations"]) == {"hidden", "act"} and "saturation_fraction" in r0["activations"]["act"]
        assert "global" in r0["updates"] and r0["grad_global_norm"] > 0
        assert r0["nonfinite_gradients"] == {}
        assert len(read_jsonl(run.history_path)) == 3  # history stays one row per epoch


# =============================================================== generation
class TestGeneration:
    def test_contexts_shift_like_context_windows(self, small_vocab, small):
        m = _model(small_vocab)
        train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.5, max_epochs=20), seed=0)
        seen: list[list[int]] = []
        handle = m.register_forward_pre_hook(lambda _m, args: seen.extend(args[0].tolist()))
        try:
            for s in range(10):
                seen.clear()
                (sample,) = m.generate(1, max_length=20, generator=make_generator(s))
                if sample.finished and sample.text:
                    assert seen == make_context_windows([sample.text], small_vocab, context_length=3).contexts.tolist()
        finally:
            handle.remove()

    def test_eos_and_max_length(self, small_vocab):
        m = _model(small_vocab)
        with torch.no_grad():
            m.out.bias[0] = -1e4  # EOS never drawn
        calls = [0]
        h = m.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        out = m.generate(4, max_length=5, generator=make_generator(0))
        h.remove()
        assert all(len(s.text) == 5 and not s.finished for s in out)
        assert calls[0] == 6  # max_length + 1 draws (batched across samples)
        with torch.no_grad():
            m.out.bias[0] = 1e4  # EOS always drawn
        assert m.generate(3, max_length=5, generator=make_generator(0)) == [MLPSample("", True)] * 3

    def test_valid_text_reproducibility_and_unchanged_model(self, small_vocab, small):
        m = _model(small_vocab)
        train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.5, max_epochs=5), seed=0)
        m.train()
        before = _state(m)
        a = m.generate(20, max_length=12, generator=make_generator(9))
        b = m.generate(20, max_length=12, generator=make_generator(9))
        assert a == b
        assert all("." not in s.text and set(s.text) <= set(small_vocab.chars) for s in a)
        _assert_same_state(before, _state(m))
        assert m.training
        assert m.generate(2, max_length=12, temperature=0) == m.generate(2, max_length=12, temperature=0)

    @pytest.mark.parametrize("kwargs", [{"num_samples": 0}, {"num_samples": True}, {"max_length": 0},
                                        {"max_length": 2.5}])
    def test_invalid_arguments(self, small_vocab, kwargs):
        args = {"num_samples": 2, "max_length": 5, **kwargs}
        with pytest.raises(ValueError):
            _model(small_vocab).generate(args.pop("num_samples"), **args)


# ============================================================== persistence
@pytest.fixture
def interrupted(small_vocab, small, tmp_path, monkeypatch):
    """A run with max_epochs = 4 interrupted after two completed epochs (exception at the start of epoch 2)."""
    cfg = MLPTrainingConfig(lr=0.4, max_epochs=4, lr_decay_epoch=2)
    run = _run(tmp_path / "b")
    real = mlp_module._learning_rate_at

    def interrupt(epoch, training):
        if epoch == 2:
            raise KeyboardInterrupt("simulated interruption")
        return real(epoch, training)

    with monkeypatch.context() as mp:
        mp.setattr(mlp_module, "_learning_rate_at", interrupt)
        with pytest.raises(KeyboardInterrupt), run.track():
            train_mlp(_model(small_vocab), small, small, training=cfg, seed=11, run=run, config={"note": "b"})
    assert run.status is RunStatus.INTERRUPTED
    return run, cfg


class TestCheckpoints:
    def test_contents(self, small_vocab, small, tmp_path):
        run = _run(tmp_path)
        cfg = MLPTrainingConfig(lr=0.3, max_epochs=2)
        r = train_mlp(_model(small_vocab), small, small, training=cfg, seed=5, run=run, config={"exp": 1})
        last = load_checkpoint(run.checkpoint_path("resume"), expected_kind="resume")
        assert (last["epoch"], last["step"], last["best_metric_name"]) == (2, 4, "dev_nll")
        assert last["best_metric"] == r.best_dev_nll and last["optimizer_state"] is not None
        assert last["rng_state"] is not None and last["vocab"] == small_vocab.to_dict()
        assert last["config"] == run_config({"exp": 1}, MLPConfig(), cfg)
        assert last["extra"] == {"best_epoch": r.best_epoch, "base_seed": 5,
                                 "sampler": EpochSampler(63, 32, seed=derive_seed(5, "data")).state_dict()}
        best = load_checkpoint(run.checkpoint_path("best"), expected_kind="best")
        assert best["optimizer_state"] is None and best["extra"]["best_epoch"] == r.best_epoch
        final = load_checkpoint(run.checkpoint_path("final"), expected_kind="final")
        assert (final["epoch"], final["step"], final["optimizer_state"]) == (2, 4, None)
        assert _rows(read_jsonl(run.history_path)) == _rows(r.history)

    def test_write_order(self, small_vocab, small, tmp_path, monkeypatch):
        run, events = _run(tmp_path), []
        real_save, real_append = mlp_module.save_checkpoint, RunManager.append_history

        def save(path, *, kind, **kw):
            events.append((kind, kw.get("epoch")))
            return real_save(path, kind=kind, **kw)

        def append(self, record):
            events.append(("history", record["epoch"]))
            return real_append(self, record)

        monkeypatch.setattr(mlp_module, "save_checkpoint", save)
        monkeypatch.setattr(RunManager, "append_history", append)
        dev = _subset(small, 40)
        r = train_mlp(_model(small_vocab), small, dev, training=MLPTrainingConfig(lr=0.3, max_epochs=3), seed=0,
                      run=run)
        expected = []
        for h in r.history:
            if h.is_best:
                expected.append(("best", h.epoch + 1))
            expected += [("history", h.epoch), ("resume", h.epoch + 1)]
        assert events == expected + [("final", 3)]

    @pytest.mark.parametrize("kind", ["best", "final"])
    def test_weights_round_trip(self, small_vocab, small, tmp_path, kind):
        run = _run(tmp_path)
        m = _model(small_vocab)
        train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.5, max_epochs=2), seed=0, run=run)
        loaded = MLPLanguageModel.from_checkpoint(run.checkpoint_path(kind), expected_vocab=small_vocab)
        if kind == "final":
            _assert_same_state(_state(m), _state(loaded))
            assert torch.equal(m(small.contexts), loaded(small.contexts))
            assert m.generate(5, max_length=8, generator=make_generator(1)) == \
                loaded.generate(5, max_length=8, generator=make_generator(1))
        assert loaded.config == MLPConfig()

    def test_from_checkpoint_rejects_incompatible(self, small_vocab, small, tmp_path):
        run = _run(tmp_path)
        m = _model(small_vocab)
        train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.5, max_epochs=1), seed=0, run=run)
        path = run.checkpoint_path("final")
        with pytest.raises(IncompatibleCheckpointError, match="vocabulary mismatch"):
            MLPLanguageModel.from_checkpoint(path, expected_vocab=Vocabulary.from_names(["xyz"]))
        cases = {
            "wrong_class": (nn.Linear(2, 2), {"model": asdict(MLPConfig())}, small_vocab.to_dict(), "not a"),
            "no_vocab": (m, {"model": asdict(MLPConfig())}, None, "vocabulary metadata"),
            "no_config": (m, None, small_vocab.to_dict(), "config"),
            "bad_config": (m, {"model": {"context_length": 0, "embedding_dim": 10, "hidden_dim": 200}},
                           small_vocab.to_dict(), "invalid model config"),
            "shape_mismatch": (m, {"model": asdict(MLPConfig(hidden_dim=100))}, small_vocab.to_dict(),
                               "do not match"),
        }
        for name, (model, config, vocab, match) in cases.items():
            p = tmp_path / f"{name}.pt"
            save_checkpoint(p, kind="final", model=model, config=config, vocab=vocab)
            with pytest.raises(IncompatibleCheckpointError, match=match):
                MLPLanguageModel.from_checkpoint(p)


class TestResume:
    def test_resume_is_bit_exact(self, small_vocab, small, tmp_path, interrupted):
        run_b, cfg = interrupted
        assert load_checkpoint(run_b.checkpoint_path("resume"))["epoch"] == 2
        run_a = _run(tmp_path / "a")
        a = _model(small_vocab)
        ra = train_mlp(a, small, small, training=cfg, seed=11, run=run_a, config={"note": "b"})
        b = _model(small_vocab, seed=999)  # different initial weights: the checkpoint must restore them
        rb = train_mlp(b, small, small, training=cfg, seed=11, run=run_b, resume=True, config={"note": "b"})
        _assert_same_state(_state(a), _state(b))
        assert _rows(ra.history) == _rows(rb.history) and len(rb.history) == 4
        assert [h.lr for h in rb.history] == [0.4, 0.4, 0.4 * 0.1, 0.4 * 0.1]
        assert (ra.best_epoch, ra.best_dev_nll) == (rb.best_epoch, rb.best_dev_nll)
        assert _rows(read_jsonl(run_a.history_path)) == _rows(read_jsonl(run_b.history_path))
        for kind in ("best", "final"):
            _assert_same_state(load_checkpoint(run_a.checkpoint_path(kind))["model_state"],
                               load_checkpoint(run_b.checkpoint_path(kind))["model_state"])

    def test_history_truncated_after_crash_between_history_and_checkpoint(self, small_vocab, small, tmp_path,
                                                                           monkeypatch):
        cfg = MLPTrainingConfig(lr=0.4, max_epochs=4)
        reference = train_mlp(_model(small_vocab), small, small, training=cfg, seed=2)
        run = _run(tmp_path)
        real = mlp_module.save_checkpoint

        def crash(path, *, kind, **kw):
            if kind == "resume" and kw["epoch"] == 2:
                raise RuntimeError("simulated crash after append_history")
            return real(path, kind=kind, **kw)

        with monkeypatch.context() as mp:
            mp.setattr(mlp_module, "save_checkpoint", crash)
            with pytest.raises(RuntimeError, match="simulated crash"):
                train_mlp(_model(small_vocab), small, small, training=cfg, seed=2, run=run)
        assert [r["epoch"] for r in read_jsonl(run.history_path)] == [0, 1]  # one row more than committed
        assert load_checkpoint(run.checkpoint_path("resume"))["epoch"] == 1
        r = train_mlp(_model(small_vocab), small, small, training=cfg, seed=2, run=run, resume=True)
        rows = read_jsonl(run.history_path)
        assert [row["epoch"] for row in rows] == [0, 1, 2, 3]
        assert _rows(rows) == _rows(reference.history) == _rows(r.history)

    @pytest.mark.parametrize(
        "change, match",
        [
            ("model", "model config"), ("lr", "training.lr"), ("max_epochs", "training.max_epochs"),
            ("batch_size", "training.batch_size"), ("lr_decay_epoch", "training.lr_decay_epoch"),
            ("lr_decay_factor", "training.lr_decay_factor"), ("seed", "seed"), ("vocabulary", "vocabulary"),
            ("dataset_size", "dataset size"),
        ],
    )
    def test_incompatible_resume_rejected(self, small_vocab, small, interrupted, change, match):
        run, cfg = interrupted
        vocab, model_cfg, train, seed = small_vocab, MLPConfig(), small, 11
        if change == "model":
            model_cfg = MLPConfig(hidden_dim=50)
        elif change == "seed":
            seed = 12
        elif change == "vocabulary":
            vocab = Vocabulary.from_names(SMALL + ["xyz"])
            train = make_context_windows(SMALL, vocab, context_length=3)
        elif change == "dataset_size":
            train = _subset(small, 60)
        elif change == "lr_decay_epoch":
            cfg = replace(cfg, lr_decay_epoch=1)
        else:
            cfg = replace(cfg, **{change: {"lr": 0.5, "max_epochs": 6, "batch_size": 16,
                                           "lr_decay_factor": 0.5}[change]})
        model = MLPLanguageModel(vocab, model_cfg, generator=make_generator(0))
        with pytest.raises(IncompatibleCheckpointError, match=match):
            train_mlp(model, train, small, training=cfg, seed=seed, run=run, resume=True, config={"note": "b"})

    def test_completed_epochs_is_not_a_budget(self, small_vocab, small, interrupted):
        run, cfg = interrupted
        # Resuming with the completed count as max_epochs is a changed budget, not a shortcut.
        with pytest.raises(IncompatibleCheckpointError, match="max_epochs"):
            train_mlp(_model(small_vocab), small, small, training=replace(cfg, max_epochs=2, lr_decay_epoch=1),
                      seed=11, run=run, resume=True)
        r = train_mlp(_model(small_vocab), small, small, training=cfg, seed=11, run=run, resume=True)
        assert r.epochs_completed == 4 and [h.epoch for h in r.history] == [0, 1, 2, 3]


# ================================================================== devices
@pytest.mark.parametrize("device", DEVICES)
class TestDevices:
    def test_training_evaluation_generation_on_device(self, device, small_vocab, small):
        cfg = MLPTrainingConfig(lr=0.3, max_epochs=1)
        cpu = _model(small_vocab)
        train_mlp(cpu, small, small, training=cfg, seed=0)
        m = _model(small_vocab).to(device)
        r = train_mlp(m, small, small, training=cfg, seed=0)
        assert m.device.type == device
        for (n, a), b in zip(m.named_parameters(), cpu.parameters()):
            assert torch.allclose(a.cpu(), b, atol=1e-5), n
        assert r.history[0].dev_nll == pytest.approx(evaluate_nll(cpu, small, batch_size=64).mean_nll, abs=1e-5)
        samples = m.generate(4, max_length=10, generator=make_generator(0))
        assert len(samples) == 4 and all("." not in s.text for s in samples)
        assert per_unit_saturation(m, small, batch_size=16).shape == (200,)

    def test_device_mismatch_raises(self, device, small_vocab):
        m = _model(small_vocab).to(device)
        other = "meta" if device != "meta" else "cpu"
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(1, 3, dtype=torch.long, device=other))

    def test_checkpoint_to_device(self, device, small_vocab, small, tmp_path):
        run = _run(tmp_path)
        m = _model(small_vocab).to(device)
        train_mlp(m, small, small, training=MLPTrainingConfig(lr=0.3, max_epochs=1), seed=0, run=run)
        loaded = MLPLanguageModel.from_checkpoint(run.checkpoint_path("final"), map_location=device)
        assert loaded.device.type == device
        x = small.contexts[:8].to(device)
        assert torch.allclose(loaded(x), m(x), atol=1e-6)


# ================================================================ real data
@pytest.mark.skipif(not DEFAULT_NAMES_PATH.is_file(), reason="names dataset not present")
def test_real_dataset_windows_and_initial_state():
    splits = split_names(load_names(DEFAULT_NAMES_PATH), fractions=(0.8, 0.1, 0.1), seed=42,
                         duplicate_policy="group")
    vocab = Vocabulary.from_names(splits.train)
    train = make_context_windows(splits.train, vocab, context_length=3)
    dev = make_context_windows(splits.dev, vocab, context_length=3)
    assert (train.num_examples, dev.num_examples) == (182_666, 22_801)
    m = MLPLanguageModel(vocab, generator=make_generator(derive_seed(0, "init")))
    assert count_parameters(m).total == 11_897
    assert evaluate_nll(m, train, batch_size=4096).mean_nll == pytest.approx(LN27, abs=1e-6)
    assert EpochSampler(train.num_examples, 32, seed=0).batch_sizes()[-1] == 10

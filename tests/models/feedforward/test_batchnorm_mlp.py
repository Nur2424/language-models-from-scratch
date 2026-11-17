"""Tests for models/feedforward/batchnorm_mlp.py, following the specification (v0.2) §13.

Fixtures
--------
SMALL  12 names (63 windows, k = 3), used for most behavioural tests.
TINY   ["ab", "ac", "ab"]: 9 windows; Ĥ(Y|W) = (2 ln 1.5 + ln 3) / 9 = 0.212171.
V27    the full 27-class vocabulary.

Mathematical tests (§13.2) are numbered as in the specification.
"""

from __future__ import annotations

import ast
import copy
import gc
import math
import string
import weakref
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.func import functional_call

import models.feedforward.batchnorm_mlp as bm
from models.feedforward.batchnorm_mlp import (
    BATCHNORM_RECORD,
    BatchNormMLPConfig,
    BatchNormMLPLanguageModel,
    BatchNormMLPSample,
    BatchNormMLPTrainingConfig,
    BatchNormMLPTrainingResult,
    CalibratedBatchNorm1d,
    EpochRecord,
    IncompatibleCheckpointError,
    calibrate_batchnorm,
    evaluate_nll,
    per_unit_saturation,
    run_config,
    train_batchnorm_mlp,
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
EPS = 1e-5
BASELINE = BatchNormMLPConfig()
BUFFERS = ("running_mean", "running_var", "num_batches_tracked", "inference_mean", "inference_var",
           "num_calibrations")
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


def _model(vocab: Vocabulary, seed: int = 0, config: BatchNormMLPConfig = BASELINE) -> BatchNormMLPLanguageModel:
    return BatchNormMLPLanguageModel(vocab, config, generator=make_generator(seed))


def _randomise(model: BatchNormMLPLanguageModel, seed: int = 123) -> BatchNormMLPLanguageModel:
    """Give γ, β, W₂ and b₂ generic values, for tests that need a general parameter point."""
    g = make_generator(seed)
    with torch.no_grad():
        model.bn.weight.copy_(1.0 + 0.3 * torch.randn(model.bn.weight.shape, generator=g))
        for p in (model.bn.bias, model.out.weight, model.out.bias):
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.5)
    return model


def _install(model: BatchNormMLPLanguageModel, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Install arbitrary (but valid) inference statistics."""
    g = make_generator(seed)
    H = model.config.hidden_dim
    m, v = torch.randn(H, generator=g), torch.rand(H, generator=g) + 0.5
    model.bn.set_inference_statistics(m, v)
    return m, v


def _subset(w: ContextWindows, n: int) -> ContextWindows:
    return ContextWindows(contexts=w.contexts[:n], targets=w.targets[:n], name_index=w.name_index[:n])


def _conditional_entropy(w: ContextWindows) -> float:
    by_ctx: dict[tuple, Counter] = defaultdict(Counter)
    for c, y in zip(w.contexts.tolist(), w.targets.tolist()):
        by_ctx[tuple(c)][y] += 1
    total = 0.0
    for counts in by_ctx.values():
        n = sum(counts.values())
        total -= sum(k * math.log(k / n) for k in counts.values())
    return total / w.num_examples


def _run(tmp_path: Path) -> RunManager:
    return RunManager.create("batchnorm_mlp", config={}, results_root=tmp_path)


def _rows(history) -> list[dict]:
    out = []
    for r in history:
        d = asdict(r) if isinstance(r, EpochRecord) else dict(r)
        d.pop("seconds")
        out.append(d)
    return out


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Parameters **and** buffers."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_same_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k, value in a.items():
        assert torch.equal(value.cpu(), b[k].cpu()), k


def _buffers(model) -> dict[str, torch.Tensor]:
    return {n: getattr(model.bn, n).detach().clone() for n in BUFFERS}


def _cfg(**kw) -> BatchNormMLPTrainingConfig:
    return BatchNormMLPTrainingConfig(**{"lr": 0.1, "max_epochs": 1, **kw})


def _bn_train_reference(a: torch.Tensor, gamma, beta) -> tuple[torch.Tensor, torch.Tensor]:
    mu, var = a.mean(0), a.var(0, unbiased=False)
    ahat = (a - mu) / torch.sqrt(var + EPS)
    return gamma * ahat + beta, ahat


def _manual_epochs(model, w, *, batch_size, lr, seed, epochs, denominator=None):
    """Independent re-implementation of the SGD loop (train mode). Returns per-batch (size, mean loss)."""
    sampler = EpochSampler(w.num_examples, batch_size, seed=derive_seed(seed, "data"))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    model.train()
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
        c = BatchNormMLPConfig()
        assert (c.context_length, c.embedding_dim, c.hidden_dim, c.fan_in) == (3, 10, 200, 30)

    @pytest.mark.parametrize("field", ["context_length", "embedding_dim", "hidden_dim"])
    @pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3", None])
    def test_config_validation(self, field, bad):
        with pytest.raises(ValueError, match=field):
            BatchNormMLPConfig(**{field: bad})

    def test_training_config_valid(self):
        t = BatchNormMLPTrainingConfig(lr=1, max_epochs=4, lr_decay_epoch=2, lr_decay_factor=1, batch_size=2)
        assert isinstance(t.lr, float) and isinstance(t.lr_decay_factor, float)
        assert (t.batch_size, t.eval_batch_size) == (2, 4096)
        assert BatchNormMLPTrainingConfig(lr=0.1, max_epochs=1).batch_size == 32

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"lr": 0}, "lr"), ({"lr": float("nan")}, "lr"), ({"lr": True}, "lr"),
            ({"max_epochs": 0}, "max_epochs"), ({"max_epochs": 2.0}, "max_epochs"),
            ({"batch_size": 1}, "batch_size"), ({"batch_size": 0}, "batch_size"), ({"batch_size": True}, "batch_size"),
            ({"eval_batch_size": 0}, "eval_batch_size"),
            ({"lr_decay_epoch": 0}, "lr_decay_epoch"), ({"lr_decay_epoch": 4}, "lr_decay_epoch"),
            ({"lr_decay_factor": 0.0}, "lr_decay_factor"), ({"lr_decay_factor": 1.5}, "lr_decay_factor"),
        ],
    )
    def test_training_config_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BatchNormMLPTrainingConfig(**{"lr": 0.1, "max_epochs": 4, **kwargs})

    def test_constructor_validation(self, v27):
        with pytest.raises(TypeError):
            BatchNormMLPLanguageModel(v27)  # generator is required
        with pytest.raises(TypeError, match="generator"):
            BatchNormMLPLanguageModel(v27, generator=0)
        with pytest.raises(TypeError, match="vocab"):
            BatchNormMLPLanguageModel("abc", generator=make_generator(0))
        with pytest.raises(TypeError, match="BatchNormMLPConfig"):
            BatchNormMLPLanguageModel(v27, {"context_length": 3}, generator=make_generator(0))

    def test_yaml_config(self):
        cfg = load_config(default_config_paths("batchnorm_mlp"), model_config_cls=BatchNormMLPConfig)
        assert cfg.model.name == "batchnorm_mlp" and cfg.model.params == BatchNormMLPConfig()
        with pytest.raises(ConfigError, match="unknown"):
            load_config(default_config_paths("batchnorm_mlp"), overrides=["model.params.eps=0.1"],
                        model_config_cls=BatchNormMLPConfig)
        with pytest.raises(ConfigError):
            load_config(default_config_paths("batchnorm_mlp"), overrides=["model.params.hidden_dim=0"],
                        model_config_cls=BatchNormMLPConfig)


class TestInputContract:
    def test_shapes_and_dtype(self, v27):
        m = _model(v27)
        for mode in (True, False):
            m.train(mode)
            z = m(torch.tensor([[0, 0, 0], [1, 2, 3]]))
            assert z.shape == (2, 27) and z.dtype == torch.float32

    def test_eval_empty_batch_and_metadata_checks(self, v27):
        m = _model(v27).eval()
        assert m(torch.zeros(0, 3, dtype=torch.long)).shape == (0, 27)
        for bad, err in (((0, 2), ValueError), ((0,), ValueError)):
            with pytest.raises(err, match="shape"):
                m(torch.zeros(bad, dtype=torch.long))
        with pytest.raises(TypeError, match="int64"):
            m(torch.zeros(0, 3))
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(0, 3, dtype=torch.long, device="meta"))

    @pytest.mark.parametrize("batch", [0, 1])
    def test_train_mode_batch_below_two_rejected_without_buffer_change(self, v27, batch):
        m = _model(v27).train()
        before = _state(m)
        with pytest.raises(ValueError, match="at least 2 windows"):
            m(torch.zeros(batch, 3, dtype=torch.long))
        _assert_same_state(before, _state(m))

    @pytest.mark.parametrize("shape", [(3,), (2, 2), (2, 4), (1, 3, 1)])
    def test_wrong_shape(self, v27, shape):
        with pytest.raises(ValueError, match="shape"):
            _model(v27).eval()(torch.zeros(shape, dtype=torch.long))

    def test_type_dtype_device(self, v27):
        m = _model(v27)
        with pytest.raises(TypeError, match="torch.Tensor"):
            m([[0, 0, 0], [0, 0, 1]])
        with pytest.raises(TypeError, match="int64"):
            m(torch.zeros(2, 3, dtype=torch.int32))
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(2, 3, dtype=torch.long, device="meta"))

    def test_padding_and_range(self, v27):
        m = _model(v27)
        with pytest.raises(ValueError, match="padding must not reach the model"):
            m(torch.tensor([[0, 0, v27.pad_id], [0, 0, 1]]))
        for bad in (-1, 28):
            with pytest.raises(ValueError, match="outside"):
                m(torch.tensor([[0, bad, 1], [0, 0, 1]]))
        with pytest.raises(ValueError, match="padding"):
            m.preactivations(torch.tensor([[0, 0, v27.pad_id]]))


class TestParametersAndBuffers:
    def test_count_and_breakdown(self, v27):
        m = _model(v27)
        assert count_parameters(m).total == count_parameters(m).trainable == 12_097
        assert {k: c.total for k, c in parameter_breakdown(m).items()} == {
            "emb": 270, "hidden": 6_000, "bn": 400, "out": 5_427}
        assert [n for n, _ in m.named_parameters()] == [
            "emb.weight", "hidden.weight", "bn.weight", "bn.bias", "out.weight", "out.bias"]
        assert m.hidden.bias is None

    def test_general_formula(self, v27):
        k, d, H, V = 5, 7, 11, 27
        m = _model(v27, config=BatchNormMLPConfig(context_length=k, embedding_dim=d, hidden_dim=H))
        assert count_parameters(m).total == V * d + (k * d) * H + 2 * H + H * V + V

    def test_buffers(self, v27):
        m = _model(v27)
        state = m.state_dict()
        for name, shape, dtype in (("running_mean", (200,), torch.float32), ("running_var", (200,), torch.float32),
                                   ("num_batches_tracked", (), torch.int64),
                                   ("inference_mean", (200,), torch.float32),
                                   ("inference_var", (200,), torch.float32), ("num_calibrations", (), torch.int64)):
            t = state[f"bn.{name}"]
            assert tuple(t.shape) == shape and t.dtype == dtype, name

    def test_module_types_and_constants(self, v27):
        m = _model(v27)
        assert isinstance(m.bn, CalibratedBatchNorm1d) and isinstance(m.bn, nn.BatchNorm1d)
        assert (m.bn.eps, m.bn.momentum, m.bn.affine, m.bn.track_running_stats) == (1e-5, 0.1, True, True)
        assert isinstance(dict(m.named_modules())["act"], nn.Tanh)
        assert BATCHNORM_RECORD == {"eps": 1e-5, "momentum": 0.1, "inference_statistics": "full_train_biased"}

    def test_module_imports_no_other_model(self):
        tree = ast.parse(Path(bm.__file__).read_text())
        imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [i for i in imported if i and i.startswith("models")]


class TestInitialisation:
    def test_every_parameter_and_buffer_explicit(self, v27):
        m = _model(v27)
        bn = m.bn
        assert torch.equal(bn.weight, torch.ones(200)) and torch.equal(bn.bias, torch.zeros(200))
        assert torch.equal(bn.running_mean, torch.zeros(200)) and torch.equal(bn.running_var, torch.ones(200))
        assert torch.equal(bn.inference_mean, torch.zeros(200)) and torch.equal(bn.inference_var, torch.ones(200))
        assert int(bn.num_batches_tracked) == 0 and int(bn.num_calibrations) == 0
        assert torch.count_nonzero(m.out.weight) == 0 and torch.count_nonzero(m.out.bias) == 0
        assert torch.count_nonzero(m.emb.weight) == 270 and torch.count_nonzero(m.hidden.weight) == 6000

    def test_deterministic_in_init_generator_only(self, v27):
        torch.manual_seed(1)
        a = _model(v27, seed=5)
        torch.manual_seed(2)
        b = _model(v27, seed=5)
        c = _model(v27, seed=6)
        _assert_same_state(_state(a), _state(b))
        assert not torch.equal(a.emb.weight, c.emb.weight) and not torch.equal(a.hidden.weight, c.hidden.weight)

    def test_construction_does_not_touch_global_rng(self, v27):
        before = torch.get_rng_state()
        _model(v27)
        assert torch.equal(before, torch.get_rng_state())


# ====================================================== §13.2 mathematical
def _pipeline(m, x, mean=None, var=None):
    """Hand-written forward: onehot·E → concat → W₁ → BN (batch stats or given stats) → tanh → W₂, b₂."""
    onehot = torch.nn.functional.one_hot(x, m.num_classes).to(m.emb.weight.dtype)
    a = (onehot @ m.emb.weight).reshape(x.shape[0], -1) @ m.hidden.weight.T
    if mean is None:
        y, _ = _bn_train_reference(a, m.bn.weight, m.bn.bias)
    else:
        y = m.bn.weight * (a - mean) / torch.sqrt(var + EPS) + m.bn.bias
    return torch.tanh(y) @ m.out.weight.T + m.out.bias


class TestMathematics:
    X = torch.tensor([[0, 0, 0], [0, 0, 5], [1, 2, 3], [26, 26, 1], [5, 5, 5], [3, 9, 14]])

    def test_1_train_mode_oracle(self, v27):
        m = _randomise(_model(v27)).train()
        assert torch.allclose(m(self.X), _pipeline(m, self.X), atol=1e-5)

    def test_2_eval_mode_oracle_ignores_ema(self, v27):
        m = _randomise(_model(v27)).eval()
        mean, var = _install(m)
        z = m(self.X)
        assert torch.allclose(z, _pipeline(m, self.X, mean, var), atol=1e-5)
        with torch.no_grad():
            m.bn.running_mean.fill_(1e3)
            m.bn.running_var.fill_(1e-3)
        assert torch.equal(m(self.X), z)

    def test_3_batch_dependence(self, v27):
        m = _randomise(_model(v27))
        _install(m)
        m.eval()
        full = m(self.X)
        for i in range(len(self.X)):
            assert torch.allclose(m(self.X[i:i + 1]), full[i:i + 1], atol=1e-6)
        m.train()
        a = m(self.X[:3])
        b = m(torch.cat([self.X[:2], self.X[3:5]]))
        assert not torch.allclose(a[:2], b[:2])  # same windows, different batch → different outputs

    def test_4_ema_update_rule_also_under_no_grad(self, v27):
        m = _randomise(_model(v27)).train()
        inf_before = (m.bn.inference_mean.clone(), m.bn.inference_var.clone())
        a = m.preactivations(self.X).detach()
        with torch.no_grad():
            m(self.X)
        assert torch.allclose(m.bn.running_mean, 0.1 * a.mean(0), atol=1e-6)
        assert torch.allclose(m.bn.running_var, 0.9 + 0.1 * a.var(0, unbiased=True), atol=1e-6)
        assert int(m.bn.num_batches_tracked) == 1
        assert torch.equal(m.bn.inference_mean, inf_before[0]) and torch.equal(m.bn.inference_var, inf_before[1])

    def test_5_sharing_blocks_sensitivity_logits(self, v27):
        m = _randomise(_model(v27))
        _install(m)
        m.eval()
        assert sum(isinstance(mod, nn.Embedding) for mod in m.modules()) == 1
        a = m.preactivations(torch.tensor([[3, 3, 7]]))[0]
        W1, E = m.hidden.weight, m.emb.weight
        assert torch.allclose(a, W1[:, 0:10] @ E[3] + W1[:, 10:20] @ E[3] + W1[:, 20:30] @ E[7], atol=1e-5)
        z = m(torch.tensor([[1, 2, 3], [3, 2, 1]]))
        assert not torch.allclose(z[0], z[1])
        y = torch.tensor([1, 0])
        hand = (torch.logsumexp(z, dim=1) - z[torch.arange(2), y]).mean()
        assert torch.allclose(token_nll(z, y, reduction="mean"), hand, atol=1e-6)

    def test_6_shared_embedding_gradient_accumulation(self, v27):
        m = _randomise(_model(v27)).double().train()
        x = torch.tensor([[0, 0, 0], [0, 0, 5], [5, 7, 5], [1, 2, 3]])
        y = torch.tensor([5, 7, 0, 4])
        m.zero_grad()
        token_nll(m(x), y, reduction="mean").backward()
        with torch.no_grad():
            E, W1, W2, b2, gamma, beta = (m.emb.weight, m.hidden.weight, m.out.weight, m.out.bias,
                                          m.bn.weight, m.bn.bias)
            a = E[x].reshape(4, 30) @ W1.T
            var = a.var(0, unbiased=False)
            yb, ahat = _bn_train_reference(a, gamma, beta)
            h = torch.tanh(yb)
            p = torch.softmax(h @ W2.T + b2, dim=1)
            gz = (p - torch.nn.functional.one_hot(y, 27).double()) / 4
            gy = (gz @ W2) * (1 - h ** 2)
            ga = gamma / torch.sqrt(var + EPS) * (gy - gy.mean(0) - ahat * (gy * ahat).mean(0))  # §4.3
            expected = torch.zeros_like(E)
            for n in range(4):
                for pos in range(3):
                    expected[x[n, pos]] += W1[:, 10 * pos:10 * pos + 10].T @ ga[n]
        assert torch.allclose(m.emb.weight.grad, expected, atol=1e-12)

    def test_7_batchnorm_identities(self, v27):
        m = _randomise(_model(v27)).double().train()
        x, y = self.X, torch.tensor([1, 3, 0, 5, 2, 7])
        a = m.preactivations(x).detach().requires_grad_(True)
        out = m.bn(a)
        out.retain_grad()
        loss = token_nll(m.out(m.act(out)), y, reduction="mean")
        loss.backward()
        g = out.grad
        _, ahat = _bn_train_reference(a.detach(), m.bn.weight, m.bn.bias)
        assert a.grad.sum(0).abs().max() < 1e-12  # zero-sum: a hidden bias would get no gradient
        assert torch.allclose(m.bn.weight.grad, (g * ahat).sum(0), atol=1e-12)
        assert torch.allclose(m.bn.bias.grad, g.sum(0), atol=1e-12)
        # Scale invariance, exact as ε → 0. PyTorch rejects ε = 0 in train mode, so ε = 1e-30 is used:
        # its effect (~ε/σ² ≈ 1e-30) is far below the 1e-10 tolerance.
        m0 = copy.deepcopy(m)
        m0.bn.eps = 1e-30
        m5 = copy.deepcopy(m0)
        with torch.no_grad():
            m5.hidden.weight[4] *= 5.0
        assert torch.allclose(m0(x), m5(x), atol=1e-10)

    def test_8_finite_difference_gradcheck_train_mode(self, v27):
        m = _randomise(_model(v27, config=BatchNormMLPConfig(hidden_dim=8))).double().train()
        x, y = torch.tensor([[0, 0, 0], [0, 1, 2], [2, 2, 3], [4, 1, 0]]), torch.tensor([1, 3, 0, 2])
        names = [n for n, _ in m.named_parameters()]
        params = tuple(p.detach().clone().requires_grad_(True) for p in m.parameters())
        assert len(params) == 6

        def f(*ps):
            return token_nll(functional_call(m, dict(zip(names, ps)), (x,)), y, reduction="mean")

        assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-6)

    def test_9_zero_output_initialisation(self, v27):
        w = make_context_windows(SMALL, v27, context_length=3)
        m = _model(v27)
        assert evaluate_nll(m, w, batch_size=17).mean_nll == pytest.approx(LN27, abs=1e-6)  # eval mode
        x, y = w.contexts[:32], w.targets[:32]
        m.train()
        loss = token_nll(m(x), y, reduction="mean")  # train mode
        assert loss.item() == pytest.approx(LN27, abs=1e-6)
        loss.backward()
        for p in (m.emb.weight, m.hidden.weight, m.bn.weight, m.bn.bias):
            assert torch.count_nonzero(p.grad) == 0
        with torch.no_grad():
            h = torch.tanh(_bn_train_reference(m.preactivations(x), m.bn.weight, m.bn.bias)[0])
            g = (torch.full((32, 27), 1 / 27) - torch.nn.functional.one_hot(y, 27)) / 32
        assert torch.allclose(m.out.weight.grad, g.T @ h, atol=1e-6)
        assert torch.allclose(m.out.bias.grad, g.sum(0), atol=1e-7)
        torch.optim.SGD(m.parameters(), lr=0.1).step()
        m.zero_grad()
        token_nll(m(x), y, reduction="mean").backward()
        for name, p in m.named_parameters():
            assert torch.count_nonzero(p.grad) > 0, name

    def test_10_initialisation_statistics(self, v27):
        m = _model(v27, seed=0)
        E, W1 = m.emb.weight, m.hidden.weight
        assert abs(E.std().item() - 1) < 0.2 and abs(E.mean().item()) < 0.3
        assert abs(W1.std().item() * math.sqrt(30) - 1) < 0.05 and abs(W1.mean().item()) < 0.012
        w = make_context_windows(SMALL, v27, context_length=3)
        stds = [_model(v27, seed=1000 + s).preactivations(w.contexts).std().item() for s in range(20)]
        assert abs(sum(stds) / len(stds) - 1) < 0.15  # mean over seeds (single-seed spread is large)
        k8 = _model(v27, seed=0, config=BatchNormMLPConfig(context_length=8))
        assert k8.hidden.weight.shape == (200, 80)
        assert abs(k8.hidden.weight.std().item() * math.sqrt(80) - 1) < 0.05
        assert abs(k8.hidden.weight.mean().item()) < 0.01
        assert abs(k8.hidden.weight.std().item() * math.sqrt(30) - 1) > 0.05
        assert torch.equal(k8.bn.weight, torch.ones(200)) and torch.equal(k8.bn.bias, torch.zeros(200))

    def test_11_mean_reduction_over_actual_batch_size(self, small_vocab, small):
        w = _subset(small, 42)
        assert EpochSampler(42, 32, seed=0).batch_sizes() == [32, 10]
        trained = _model(small_vocab)
        train_batchnorm_mlp(trained, w, w, training=_cfg(lr=0.5, max_epochs=2), seed=3)
        reference = _model(small_vocab)
        _manual_epochs(reference, w, batch_size=32, lr=0.5, seed=3, epochs=2)
        for (n, a), b in zip(trained.named_parameters(), reference.parameters()):
            assert torch.allclose(a, b, rtol=0, atol=1e-6), n
        nominal = _model(small_vocab)
        _manual_epochs(nominal, w, batch_size=32, lr=0.5, seed=3, epochs=2, denominator=32)
        assert (trained.out.bias - nominal.out.bias).abs().max() > 1e-3

    def test_12_running_nll_is_example_weighted(self, small_vocab, small):
        w = _subset(small, 42)
        r = train_batchnorm_mlp(_model(small_vocab), w, w, training=_cfg(lr=0.5), seed=3)
        batches = _manual_epochs(_model(small_vocab), w, batch_size=32, lr=0.5, seed=3, epochs=1)
        weighted = sum(n * loss for n, loss in batches) / 42
        assert r.history[0].running_train_nll == pytest.approx(weighted, rel=1e-6)
        assert abs(weighted - sum(loss for _, loss in batches) / len(batches)) > 1e-4

    @pytest.mark.parametrize("batch_size", [1, 7, 62, 63, 64, 4096])
    def test_13_evaluation_covers_every_example(self, small_vocab, small, batch_size):
        m = _randomise(_model(small_vocab))
        _install(m)
        acc = evaluate_nll(m, small, batch_size=batch_size)
        assert acc.num_tokens == small.num_examples == 63
        with torch.no_grad():
            full = token_nll(m.eval()(small.contexts), small.targets, reduction="mean").item()
        assert acc.mean_nll == pytest.approx(full, rel=1e-6)

    def test_14_recalibration_correctness(self, small_vocab, small):
        m = _randomise(_model(small_vocab))
        out = calibrate_batchnorm(m, small, batch_size=10)
        a = m.preactivations(small.contexts).detach().double()
        assert out == {"num_windows": 63, "num_calibrations": 1}
        assert torch.allclose(m.bn.inference_mean.double(), a.mean(0), rtol=1e-6, atol=1e-7)
        assert torch.allclose(m.bn.inference_var.double(), a.var(0, unbiased=False), rtol=1e-6, atol=0)

    @pytest.mark.parametrize("dtype, tol", [(torch.float32, {"rtol": 1e-6, "atol": 1e-7}),
                                            (torch.float64, {"rtol": 0, "atol": 1e-12})])
    def test_15_partition_independence(self, small_vocab, small, dtype, tol):
        m = _randomise(_model(small_vocab)).to(dtype)
        results = []
        for bs in (1, 7, 64, 63, 68):
            calibrate_batchnorm(m, small, batch_size=bs)
            results.append((m.bn.inference_mean.clone(), m.bn.inference_var.clone()))
        for mean, var in results[1:]:
            assert torch.allclose(mean, results[0][0], **tol) and torch.allclose(var, results[0][1], **tol)

    def test_15_large_offset_is_accurate(self, small_vocab, small):
        m = _model(small_vocab)
        with torch.no_grad():  # unit 0 reads E[:, 0] at the last position: |mean| ≫ std
            m.emb.weight[:, 0] = 1.0e4 + torch.arange(m.num_classes, dtype=torch.float32) * 0.01
            m.hidden.weight[0].zero_()
            m.hidden.weight[0, 20] = 1.0
        calibrate_batchnorm(m, small, batch_size=5)
        a = m.preactivations(small.contexts).detach().double()[:, 0]
        exact = a.var(unbiased=False).item()
        assert exact > 0 and m.bn.inference_var[0].item() == pytest.approx(exact, rel=1e-6)
        naive32 = (a.float().square().mean() - a.float().mean().square()).item()
        assert abs(naive32 - exact) > 10 * exact  # the E[a²] − E[a]² form would be useless here

    def test_15_sequential_chunks_released(self, small_vocab, small, monkeypatch):
        m = _randomise(_model(small_vocab))
        original = m.preactivations
        refs: list[weakref.ref] = []
        sizes: list[int] = []

        def wrapped(contexts):
            if refs:
                gc.collect()
                assert refs[-1]() is None, "previous chunk still alive: chunks must be released"
            out = original(contexts)
            sizes.append(out.shape[0])
            refs.append(weakref.ref(out))
            return out

        monkeypatch.setattr(m, "preactivations", wrapped)
        calibrate_batchnorm(m, small, batch_size=10)
        gc.collect()
        assert refs[-1]() is None
        assert sizes == [10] * 6 + [3] and sum(sizes) == 63

    @pytest.mark.parametrize("training", [True, False])
    def test_16_recalibration_isolation(self, small_vocab, small, training):
        m = _randomise(_model(small_vocab))
        reference = copy.deepcopy(m)
        calibrate_batchnorm(reference, small, batch_size=8)
        with torch.no_grad():
            m.bn.running_mean.fill_(123.0)
            m.bn.running_var.fill_(0.5)
            m.bn.num_batches_tracked.fill_(17)
            m.bn.inference_mean.fill_(-9.0)
            m.bn.inference_var.fill_(42.0)
        m.train(training)
        params = {n: p.detach().clone() for n, p in m.named_parameters()}
        calibrate_batchnorm(m, small, batch_size=8)
        assert m.training is training
        assert torch.equal(m.bn.running_mean, torch.full((200,), 123.0))
        assert torch.equal(m.bn.running_var, torch.full((200,), 0.5)) and int(m.bn.num_batches_tracked) == 17
        for n, p in m.named_parameters():
            assert torch.equal(p, params[n]), n
        assert torch.equal(m.bn.inference_mean, reference.bn.inference_mean)
        assert torch.equal(m.bn.inference_var, reference.bn.inference_var)
        assert int(m.bn.num_calibrations) == 1

    def test_17_full_batch_consistency(self):
        vocab = Vocabulary.from_names(TINY)
        w = make_context_windows(TINY, vocab, context_length=3)
        m = _randomise(_model(vocab))
        z_train = copy.deepcopy(m).train()(w.contexts)
        calibrate_batchnorm(m, w, batch_size=4)
        assert torch.allclose(m.eval()(w.contexts), z_train, atol=1e-5)

    def test_18_conditional_entropy_bound_eval_mode(self, small_vocab, small):
        tiny_vocab = Vocabulary.from_names(TINY)
        tiny = make_context_windows(TINY, tiny_vocab, context_length=3)
        assert _conditional_entropy(tiny) == pytest.approx(H_TINY, abs=1e-12)
        for vocab, w in ((tiny_vocab, tiny), (small_vocab, small)):
            bound = _conditional_entropy(w)
            for seed in range(3):
                m = _randomise(_model(vocab, seed=seed), seed=seed)
                calibrate_batchnorm(m, w, batch_size=64)
                assert evaluate_nll(m, w, batch_size=64).mean_nll >= bound - 1e-6
            m = _model(vocab)
            train_batchnorm_mlp(m, w, w, training=_cfg(lr=0.5, max_epochs=30, batch_size=5), seed=0)  # no batch of 1
            assert evaluate_nll(m, w, batch_size=64).mean_nll >= bound - 1e-6


# ========================================================= §13.3 behaviour
class TestTraining:
    def test_overfit_tiny_fixture(self):
        vocab = Vocabulary.from_names(TINY)
        w = make_context_windows(TINY, vocab, context_length=3)
        m = _model(vocab)
        train_batchnorm_mlp(m, w, w, training=_cfg(lr=0.5, max_epochs=500), seed=0)
        gap = evaluate_nll(m, w, batch_size=64).mean_nll - H_TINY
        assert -1e-6 <= gap <= 1e-2

    def test_batch_size_guard(self, small_vocab, small):
        m = _model(small_vocab)
        before = _state(m)
        assert EpochSampler(63, 31, seed=0).batch_sizes() == [31, 31, 1]
        with pytest.raises(ValueError, match="at least 2 windows"):
            train_batchnorm_mlp(m, small, small, training=_cfg(batch_size=31), seed=0)
        _assert_same_state(before, _state(m))

    def test_epoch_end_order_and_calibration_count(self, small_vocab, small, monkeypatch):
        events: list[str] = []
        real_nll, real_cal, real_eval = bm.token_nll, bm.calibrate_batchnorm, bm.evaluate_nll
        dev = _subset(small, 40)

        def nll(*a, **k):
            events.append("step")
            return real_nll(*a, **k)

        def cal(*a, **k):
            events.append("calibrate")
            return real_cal(*a, **k)

        def ev(model, windows, *, batch_size):
            events.append("eval_train" if windows is small else "eval_dev")
            return real_eval(model, windows, batch_size=batch_size)

        monkeypatch.setattr(bm, "token_nll", nll)
        monkeypatch.setattr(bm, "calibrate_batchnorm", cal)
        monkeypatch.setattr(bm, "evaluate_nll", ev)
        m = _model(small_vocab)
        train_batchnorm_mlp(m, small, dev, training=_cfg(max_epochs=3), seed=0)
        assert events == ["step", "step", "calibrate", "eval_train", "eval_dev"] * 3
        assert int(m.bn.num_calibrations) == 3

    def test_mode_discipline_train_forwards_only_in_sgd(self, small_vocab, small, tmp_path):
        m, run = _model(small_vocab), _run(tmp_path)
        schedule = DiagnosticsSchedule(enabled=True, every_n_steps=1, gradients=True, parameters=True,
                                       activations=True)
        r = train_batchnorm_mlp(m, small, small, training=_cfg(lr=0.3, max_epochs=3), seed=0, run=run,
                                diagnostics=schedule)
        assert int(m.bn.num_batches_tracked) == r.history[-1].steps == 6

    def test_result_history_and_summaries(self, small_vocab, small):
        r = train_batchnorm_mlp(_model(small_vocab), small, small, training=_cfg(lr=0.3, max_epochs=3), seed=1)
        assert isinstance(r, BatchNormMLPTrainingResult) and r.epochs_completed == 3
        assert [h.steps for h in r.history] == [2, 4, 6]
        for i, h in enumerate(r.history):
            assert {"hidden", "bn", "act", "per_unit_saturation"} <= h.activations.keys()
            assert h.activations["bn"]["numel"] == 63 * 200
            b = h.batchnorm
            assert set(b) == {"ema_vs_inference", "gamma", "beta", "w1_row_norm", "num_calibrations",
                              "num_batches_tracked"}
            assert b["num_calibrations"] == i + 1 and b["num_batches_tracked"] == 2 * (i + 1)
            assert set(b["ema_vs_inference"]) == {"mean_abs_z", "max_abs_z", "median_abs_log_var_ratio",
                                                  "max_abs_log_var_ratio"}
        best = min(range(3), key=lambda e: (r.history[e].dev_nll, e))
        assert r.best_epoch == best and r.best_dev_nll == r.history[best].dev_nll

    def test_three_nlls_and_train_eval_mismatch(self, small_vocab, small):
        r = train_batchnorm_mlp(_model(small_vocab), small, small, training=_cfg(lr=0.5), seed=0)
        h = r.history[0]
        assert h.train_nll == h.dev_nll and h.running_train_nll != h.train_nll
        # fixed parameters: train-mode batch NLL differs from calibrated eval-mode NLL (normalisation mismatch)
        m = _randomise(_model(small_vocab))
        tm = copy.deepcopy(m).train()
        acc = NLLAccumulator()
        with torch.no_grad():
            for idx in EpochSampler(63, 32, seed=0).batches(0):
                acc.update(tm(small.contexts[idx]), small.targets[idx])
        calibrate_batchnorm(m, small, batch_size=64)
        assert abs(acc.mean_nll - evaluate_nll(m, small, batch_size=64).mean_nll) > 1e-4

    def test_determinism(self, small_vocab, small):
        cfg = _cfg(lr=0.3, max_epochs=3)
        a, b, c = _model(small_vocab), _model(small_vocab), _model(small_vocab)
        ra = train_batchnorm_mlp(a, small, small, training=cfg, seed=7)
        rb = train_batchnorm_mlp(b, small, small, training=cfg, seed=7)
        rc = train_batchnorm_mlp(c, small, small, training=cfg, seed=8)
        _assert_same_state(_state(a), _state(b))
        assert _rows(ra.history) == _rows(rb.history)
        assert not torch.equal(a.hidden.weight, c.hidden.weight) and _rows(ra.history) != _rows(rc.history)

    def test_schedule(self, small_vocab, small):
        cfg = _cfg(lr=0.4, max_epochs=4, lr_decay_epoch=2)
        r = train_batchnorm_mlp(_model(small_vocab), small, small, training=cfg, seed=0)
        assert [h.lr for h in r.history] == [0.4, 0.4, 0.4 * 0.1, 0.4 * 0.1]
        r = train_batchnorm_mlp(_model(small_vocab), small, small, training=replace(cfg, lr_decay_epoch=None), seed=0)
        assert [h.lr for h in r.history] == [0.4] * 4

    def test_selection_ties_keep_earliest_epoch(self, small_vocab, small, tmp_path, monkeypatch):
        dev = _subset(small, 40)
        values = iter([3.0, 2.0, 2.0, 2.5])
        real = bm.evaluate_nll

        def fake(model, windows, *, batch_size):
            acc = real(model, windows, batch_size=batch_size)
            if windows is dev:
                acc = NLLAccumulator()
                acc.update_from_sum(next(values) * 40, 40)
            return acc

        monkeypatch.setattr(bm, "evaluate_nll", fake)
        run = _run(tmp_path)
        r = train_batchnorm_mlp(_model(small_vocab), small, dev, training=_cfg(max_epochs=4), seed=0, run=run)
        assert [h.is_best for h in r.history] == [True, True, False, False]
        best = load_checkpoint(run.checkpoint_path("best"))
        assert best["extra"]["best_epoch"] == 1 and best["epoch"] == 2
        assert int(best["model_state"]["bn.num_calibrations"]) == 2  # best epoch + 1

    def test_argument_validation(self, small_vocab, small, tmp_path):
        m, cfg = _model(small_vocab), _cfg()
        with pytest.raises(ValueError, match="resume=True requires a run"):
            train_batchnorm_mlp(m, small, small, training=cfg, seed=0, resume=True)
        with pytest.raises(ValueError, match="pass a run"):
            train_batchnorm_mlp(m, small, small, training=cfg, seed=0, diagnostics=DiagnosticsSchedule(enabled=True))
        with pytest.raises(TypeError, match="BatchNormMLPTrainingConfig"):
            train_batchnorm_mlp(m, small, small, training={"lr": 0.1}, seed=0)
        run = _run(tmp_path)
        train_batchnorm_mlp(m, small, small, training=cfg, seed=0, run=run)
        with pytest.raises(ValueError, match="resume=True"):
            train_batchnorm_mlp(m, small, small, training=cfg, seed=0, run=run)


class TestNonFiniteGuards:
    def test_nan_loss_stops_before_step(self, small_vocab, small, monkeypatch):
        m = _model(small_vocab)
        real, calls, before = bm.token_nll, [0], {}

        def poisoned(logits, targets, *, reduction):
            calls[0] += 1
            loss = real(logits, targets, reduction=reduction)
            if calls[0] == 4:  # epoch 1, step 3
                before.update({n: p.detach().clone() for n, p in m.named_parameters()})
                return loss * float("nan")
            return loss

        monkeypatch.setattr(bm, "token_nll", poisoned)
        with pytest.raises(FloatingPointError, match=r"non-finite loss .* at epoch 1, step 3"):
            train_batchnorm_mlp(m, small, small, training=_cfg(max_epochs=3), seed=0)
        for n, p in m.named_parameters():  # optimizer.step() was not called
            assert torch.equal(p, before[n]), n

    def test_nan_gradient_stops_before_step(self, small_vocab, small):
        m = _model(small_vocab)
        calls, before = [0], {}

        def hook(grad):
            calls[0] += 1
            if calls[0] == 3:  # epoch 1, step 2
                before.update({n: p.detach().clone() for n, p in m.named_parameters()})
                return grad * float("nan")
            return grad

        m.hidden.weight.register_hook(hook)
        with pytest.raises(FloatingPointError, match=r"epoch 1, step 2 in parameter groups \['hidden'\]"):
            train_batchnorm_mlp(m, small, small, training=_cfg(max_epochs=3), seed=0)
        for n, p in m.named_parameters():
            assert torch.equal(p, before[n]), n

    def test_non_finite_calibration_names_epoch(self, small_vocab, small, monkeypatch, tmp_path):
        real, calls = bm._chunk_moments, [0]

        def poisoned(a):
            calls[0] += 1
            n, mean, m2 = real(a)
            return (n, mean * float("nan"), m2) if calls[0] == 2 else (n, mean, m2)

        monkeypatch.setattr(bm, "_chunk_moments", poisoned)
        m, run = _model(small_vocab), _run(tmp_path)
        with pytest.raises(FloatingPointError, match="recalibration failed at epoch 1"), run.track():
            train_batchnorm_mlp(m, small, small, training=_cfg(max_epochs=3), seed=0, run=run)
        assert int(m.bn.num_calibrations) == 1  # the failed statistics were not installed
        assert run.status is RunStatus.FAILED
        assert load_checkpoint(run.checkpoint_path("resume"))["epoch"] == 1

    def test_direct_calibration_rejects_non_finite(self, small_vocab, small):
        m = _model(small_vocab)
        with torch.no_grad():
            m.emb.weight[0, 0] = float("inf")
        with pytest.raises(FloatingPointError, match="non-finite"):
            calibrate_batchnorm(m, small, batch_size=16)
        assert int(m.bn.num_calibrations) == 0


class TestModeRestoration:
    @pytest.mark.parametrize("fn", ["evaluate_nll", "per_unit_saturation"])
    @pytest.mark.parametrize("training", [True, False])
    def test_helpers(self, small_vocab, small, fn, training):
        m = _randomise(_model(small_vocab))
        _install(m)
        m.train(training)
        buffers = _buffers(m)
        getattr(bm, fn)(m, small, batch_size=10)
        assert m.training is training
        bad = ContextWindows(contexts=small.contexts.clone(), targets=small.targets, name_index=small.name_index)
        bad.contexts[40, 0] = small_vocab.pad_id
        with pytest.raises(ValueError, match="padding"):
            getattr(bm, fn)(m, bad, batch_size=10)
        assert m.training is training
        for k, v in _buffers(m).items():
            assert torch.equal(v, buffers[k]), k

    @pytest.mark.parametrize("training", [True, False])
    def test_generate(self, small_vocab, training):
        m = _randomise(_model(small_vocab))
        _install(m)
        m.train(training)
        buffers = _buffers(m)
        m.generate(3, max_length=6, generator=make_generator(0))
        assert m.training is training
        with pytest.raises(Exception, match="temperature"):  # raises part-way (first draw)
            m.generate(2, max_length=6, temperature=-1.0)
        assert m.training is training
        for k, v in _buffers(m).items():
            assert torch.equal(v, buffers[k]), k


class TestDiagnostics:
    def test_step_and_epoch_diagnostics(self, small_vocab, small, tmp_path):
        cfg = _cfg(lr=0.3, max_epochs=3)
        plain = _model(small_vocab)
        train_batchnorm_mlp(plain, small, small, training=cfg, seed=0)
        m, run = _model(small_vocab), _run(tmp_path)
        schedule = DiagnosticsSchedule(enabled=True, every_n_steps=2, gradients=True, parameters=True,
                                       activations=True)
        r = train_batchnorm_mlp(m, small, small, training=cfg, seed=0, run=run, diagnostics=schedule)
        _assert_same_state(_state(plain), _state(m))  # diagnostics never change training (incl. buffers)
        rows = read_jsonl(run.diagnostics_dir / "steps.jsonl")
        assert [x["step"] for x in rows] == [0, 2, 4]
        r0 = rows[0]
        assert set(r0["gradients"]) == {"emb", "hidden", "bn", "out"}
        assert r0["gradients"]["bn"]["norm"] == 0.0 and r0["gradients"]["out"]["norm"] > 0  # §7, step 0
        assert set(r0["activations"]) == {"hidden", "bn", "act"}
        assert r0["nonfinite_gradients"] == {} and "global" in r0["updates"]
        assert [x["batchnorm"]["num_calibrations"] for x in read_jsonl(run.history_path)] == [1, 2, 3]
        assert r.history[-1].batchnorm["ema_vs_inference"]["max_abs_z"] >= 0


# =============================================================== generation
class TestGeneration:
    def test_contexts_shift_like_context_windows(self, small_vocab, small):
        m = _model(small_vocab)
        train_batchnorm_mlp(m, small, small, training=_cfg(lr=0.5, max_epochs=20), seed=0)
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

    def test_eval_mode_inference_statistics_and_single_sample(self, small_vocab):
        m = _randomise(_model(small_vocab))
        _install(m)
        m.train()  # a train-mode batch of 1 would fail; generation must use eval mode
        a = m.generate(1, max_length=8, generator=make_generator(3))
        assert m.training
        with torch.no_grad():
            m.bn.running_mean.fill_(50.0)  # EMA is not used
        assert m.generate(1, max_length=8, generator=make_generator(3)) == a
        seen = []
        h = m.act.register_forward_hook(lambda _m, _i, o: seen.append(o.detach().clone()))
        m.generate(1, max_length=1, generator=make_generator(3))
        with torch.no_grad():
            m.bn.inference_mean.add_(1.0)  # inference statistics are used
        m.generate(1, max_length=1, generator=make_generator(3))
        h.remove()
        assert not torch.allclose(seen[0], seen[-1])

    def test_eos_and_max_length(self, small_vocab):
        m = _model(small_vocab)
        with torch.no_grad():
            m.out.bias[0] = -1e4
        calls = [0]
        h = m.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        out = m.generate(4, max_length=5, generator=make_generator(0))
        h.remove()
        assert all(len(s.text) == 5 and not s.finished for s in out) and calls[0] == 6
        with torch.no_grad():
            m.out.bias[0] = 1e4
        assert m.generate(3, max_length=5, generator=make_generator(0)) == [BatchNormMLPSample("", True)] * 3

    def test_valid_text_reproducibility_and_unchanged_state(self, small_vocab, small):
        m = _model(small_vocab)
        train_batchnorm_mlp(m, small, small, training=_cfg(lr=0.5, max_epochs=5), seed=0)
        before = _state(m)
        a = m.generate(20, max_length=12, generator=make_generator(9))
        assert a == m.generate(20, max_length=12, generator=make_generator(9))
        assert all("." not in s.text and set(s.text) <= set(small_vocab.chars) for s in a)
        _assert_same_state(before, _state(m))

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
    cfg = _cfg(lr=0.4, max_epochs=4, lr_decay_epoch=2)
    run = _run(tmp_path / "b")
    real = bm._learning_rate_at

    def interrupt(epoch, training):
        if epoch == 2:
            raise KeyboardInterrupt("simulated interruption")
        return real(epoch, training)

    with monkeypatch.context() as mp:
        mp.setattr(bm, "_learning_rate_at", interrupt)
        with pytest.raises(KeyboardInterrupt), run.track():
            train_batchnorm_mlp(_model(small_vocab), small, small, training=cfg, seed=11, run=run,
                                config={"note": "b"})
    return run, cfg


class TestCheckpoints:
    def test_contents(self, small_vocab, small, tmp_path):
        run = _run(tmp_path)
        cfg = _cfg(lr=0.3, max_epochs=2)
        r = train_batchnorm_mlp(_model(small_vocab), small, small, training=cfg, seed=5, run=run, config={"exp": 1})
        last = load_checkpoint(run.checkpoint_path("resume"), expected_kind="resume")
        assert (last["epoch"], last["step"], last["best_metric_name"]) == (2, 4, "dev_nll")
        assert last["config"] == run_config({"exp": 1}, BatchNormMLPConfig(), cfg)
        assert last["config"]["batchnorm"] == BATCHNORM_RECORD
        assert last["extra"] == {"best_epoch": r.best_epoch, "base_seed": 5,
                                 "sampler": EpochSampler(63, 32, seed=derive_seed(5, "data")).state_dict()}
        assert {f"bn.{b}" for b in BUFFERS} <= last["model_state"].keys()
        assert int(last["model_state"]["bn.num_calibrations"]) == 2
        assert int(last["model_state"]["bn.num_batches_tracked"]) == 4
        final = load_checkpoint(run.checkpoint_path("final"), expected_kind="final")
        assert int(final["model_state"]["bn.num_calibrations"]) == 2 and final["optimizer_state"] is None
        assert _rows(read_jsonl(run.history_path)) == _rows(r.history)

    def test_write_order(self, small_vocab, small, tmp_path, monkeypatch):
        run, events = _run(tmp_path), []
        real_save, real_append = bm.save_checkpoint, RunManager.append_history

        def save(path, *, kind, **kw):
            events.append((kind, kw.get("epoch")))
            return real_save(path, kind=kind, **kw)

        def append(self, record):
            events.append(("history", record["epoch"]))
            return real_append(self, record)

        monkeypatch.setattr(bm, "save_checkpoint", save)
        monkeypatch.setattr(RunManager, "append_history", append)
        r = train_batchnorm_mlp(_model(small_vocab), small, _subset(small, 40), training=_cfg(lr=0.3, max_epochs=3),
                                seed=0, run=run)
        expected = []
        for h in r.history:
            if h.is_best:
                expected.append(("best", h.epoch + 1))
            expected += [("history", h.epoch), ("resume", h.epoch + 1)]
        assert events == expected + [("final", 3)]

    @pytest.mark.parametrize("kind", ["best", "final"])
    def test_round_trip_with_buffers(self, small_vocab, small, tmp_path, kind):
        run = _run(tmp_path)
        m = _model(small_vocab)
        train_batchnorm_mlp(m, small, small, training=_cfg(lr=0.5, max_epochs=2), seed=0, run=run)
        loaded = BatchNormMLPLanguageModel.from_checkpoint(run.checkpoint_path(kind), expected_vocab=small_vocab)
        assert loaded.config == BatchNormMLPConfig()
        if kind == "final":
            _assert_same_state(_state(m), _state(loaded))
            assert torch.equal(m.eval()(small.contexts), loaded.eval()(small.contexts))
            assert m.generate(5, max_length=8, generator=make_generator(1)) == \
                loaded.generate(5, max_length=8, generator=make_generator(1))

    def test_from_checkpoint_rejects_incompatible(self, small_vocab, small, tmp_path):
        run = _run(tmp_path)
        m = _model(small_vocab)
        train_batchnorm_mlp(m, small, small, training=_cfg(lr=0.5), seed=0, run=run)
        with pytest.raises(IncompatibleCheckpointError, match="vocabulary mismatch"):
            BatchNormMLPLanguageModel.from_checkpoint(run.checkpoint_path("final"),
                                                      expected_vocab=Vocabulary.from_names(["xyz"]))
        good = {"model": asdict(BatchNormMLPConfig()), "batchnorm": dict(BATCHNORM_RECORD)}
        fresh = _model(small_vocab)
        cases = {
            "wrong_class": (nn.Linear(2, 2), good, small_vocab.to_dict(), "not a"),
            "no_vocab": (m, good, None, "vocabulary metadata"),
            "no_config": (m, None, small_vocab.to_dict(), "config"),
            "bad_config": (m, {**good, "model": {"context_length": 0, "embedding_dim": 10, "hidden_dim": 200}},
                           small_vocab.to_dict(), "invalid model config"),
            "shape_mismatch": (m, {**good, "model": asdict(BatchNormMLPConfig(hidden_dim=100))},
                               small_vocab.to_dict(), "do not match"),
            "bn_constants": (m, {**good, "batchnorm": {**BATCHNORM_RECORD, "eps": 1e-3}}, small_vocab.to_dict(),
                             "BatchNorm constants"),
            "uncalibrated": (fresh, good, small_vocab.to_dict(), "never calibrated"),
        }
        for name, (model, config, vocab, match) in cases.items():
            p = tmp_path / f"{name}.pt"
            save_checkpoint(p, kind="final", model=model, config=config, vocab=vocab)
            with pytest.raises(IncompatibleCheckpointError, match=match):
                BatchNormMLPLanguageModel.from_checkpoint(p)


class TestResume:
    def test_resume_is_bit_exact_including_buffers(self, small_vocab, small, tmp_path, interrupted):
        run_b, cfg = interrupted
        ckpt = load_checkpoint(run_b.checkpoint_path("resume"))
        assert ckpt["epoch"] == 2 and int(ckpt["model_state"]["bn.num_calibrations"]) == 2
        run_a = _run(tmp_path / "a")
        a = _model(small_vocab)
        ra = train_batchnorm_mlp(a, small, small, training=cfg, seed=11, run=run_a, config={"note": "b"})
        b = _model(small_vocab, seed=999)  # different initial state: the checkpoint must restore everything
        with torch.no_grad():
            b.bn.running_mean.fill_(5.0)
        rb = train_batchnorm_mlp(b, small, small, training=cfg, seed=11, run=run_b, resume=True,
                                 config={"note": "b"})
        _assert_same_state(_state(a), _state(b))  # parameters and all six buffers
        assert _rows(ra.history) == _rows(rb.history) and len(rb.history) == 4
        assert (ra.best_epoch, ra.best_dev_nll) == (rb.best_epoch, rb.best_dev_nll)
        for kind in ("best", "final"):
            _assert_same_state(load_checkpoint(run_a.checkpoint_path(kind))["model_state"],
                               load_checkpoint(run_b.checkpoint_path(kind))["model_state"])

    def test_history_truncated_after_crash(self, small_vocab, small, tmp_path, monkeypatch):
        cfg = _cfg(lr=0.4, max_epochs=4)
        reference = train_batchnorm_mlp(_model(small_vocab), small, small, training=cfg, seed=2)
        run = _run(tmp_path)
        real = bm.save_checkpoint

        def crash(path, *, kind, **kw):
            if kind == "resume" and kw["epoch"] == 2:
                raise RuntimeError("simulated crash after append_history")
            return real(path, kind=kind, **kw)

        with monkeypatch.context() as mp:
            mp.setattr(bm, "save_checkpoint", crash)
            with pytest.raises(RuntimeError, match="simulated crash"):
                train_batchnorm_mlp(_model(small_vocab), small, small, training=cfg, seed=2, run=run)
        assert [r["epoch"] for r in read_jsonl(run.history_path)] == [0, 1]
        r = train_batchnorm_mlp(_model(small_vocab), small, small, training=cfg, seed=2, run=run, resume=True)
        rows = read_jsonl(run.history_path)
        assert [row["epoch"] for row in rows] == [0, 1, 2, 3]
        assert _rows(rows) == _rows(reference.history) == _rows(r.history)

    @pytest.mark.parametrize(
        "change, match",
        [("model", "model config"), ("lr", "training.lr"), ("max_epochs", "training.max_epochs"),
         ("batch_size", "training.batch_size"), ("lr_decay_epoch", "training.lr_decay_epoch"),
         ("lr_decay_factor", "training.lr_decay_factor"), ("seed", "seed"), ("vocabulary", "vocabulary"),
         ("dataset_size", "dataset size"), ("batchnorm", "batchnorm constants")],
    )
    def test_incompatible_resume_rejected(self, small_vocab, small, interrupted, monkeypatch, change, match):
        run, cfg = interrupted
        vocab, model_cfg, train, seed = small_vocab, BatchNormMLPConfig(), small, 11
        if change == "model":
            model_cfg = BatchNormMLPConfig(hidden_dim=50)
        elif change == "seed":
            seed = 12
        elif change == "vocabulary":
            vocab = Vocabulary.from_names(SMALL + ["xyz"])
            train = make_context_windows(SMALL, vocab, context_length=3)
        elif change == "dataset_size":
            train = _subset(small, 60)
        elif change == "lr_decay_epoch":
            cfg = replace(cfg, lr_decay_epoch=1)
        elif change == "batchnorm":
            monkeypatch.setattr(bm, "BATCHNORM_RECORD", {**BATCHNORM_RECORD, "momentum": 0.2})
        else:
            cfg = replace(cfg, **{change: {"lr": 0.5, "max_epochs": 6, "batch_size": 16,
                                           "lr_decay_factor": 0.5}[change]})
        model = BatchNormMLPLanguageModel(vocab, model_cfg, generator=make_generator(0))
        with pytest.raises(IncompatibleCheckpointError, match=match):
            train_batchnorm_mlp(model, train, small, training=cfg, seed=seed, run=run, resume=True,
                                config={"note": "b"})

    def test_completed_epochs_is_not_a_budget(self, small_vocab, small, interrupted):
        run, cfg = interrupted
        with pytest.raises(IncompatibleCheckpointError, match="max_epochs"):
            train_batchnorm_mlp(_model(small_vocab), small, small,
                                training=replace(cfg, max_epochs=2, lr_decay_epoch=1), seed=11, run=run, resume=True)
        r = train_batchnorm_mlp(_model(small_vocab), small, small, training=cfg, seed=11, run=run, resume=True)
        assert r.epochs_completed == 4 and [h.epoch for h in r.history] == [0, 1, 2, 3]


# ================================================================== devices
@pytest.mark.parametrize("device", DEVICES)
class TestDevices:
    def test_training_calibration_evaluation_generation(self, device, small_vocab, small):
        cfg = _cfg(lr=0.3, max_epochs=2)
        cpu = _model(small_vocab)
        train_batchnorm_mlp(cpu, small, small, training=cfg, seed=0)
        m = _model(small_vocab).to(device)
        r = train_batchnorm_mlp(m, small, small, training=cfg, seed=0)
        assert m.device.type == device
        for (n, a), b in zip(m.named_parameters(), cpu.parameters()):
            assert torch.allclose(a.cpu(), b, atol=1e-5), n
        for name in ("inference_mean", "inference_var", "running_mean", "running_var"):
            assert torch.allclose(getattr(m.bn, name).cpu(), getattr(cpu.bn, name), atol=1e-5), name
        assert r.history[-1].dev_nll == pytest.approx(evaluate_nll(cpu, small, batch_size=64).mean_nll, abs=1e-5)
        samples = m.generate(4, max_length=10, generator=make_generator(0))
        assert len(samples) == 4 and all("." not in s.text for s in samples)
        assert per_unit_saturation(m, small, batch_size=16).shape == (200,)

    def test_chunk_moments_and_summaries_from_device_tensors(self, device, small_vocab):
        # Regression (MPS): device → CPU transfer must happen before the float64 cast (MPS has no float64).
        a_cpu = torch.randn(37, 200, generator=make_generator(4))
        n, mean, m2 = bm._chunk_moments(a_cpu.to(device))
        n_ref, mean_ref, m2_ref = bm._chunk_moments(a_cpu)
        assert n == n_ref == 37
        for got, ref in ((mean, mean_ref), (m2, m2_ref)):
            assert got.device.type == "cpu" and got.dtype == torch.float64
            assert torch.equal(got, ref)  # identical float32 values in, identical float64 moments out
        m = _randomise(_model(small_vocab))
        _install(m)
        cpu_summary = bm._batchnorm_summary(m)
        assert bm._batchnorm_summary(m.to(device)) == cpu_summary
        assert bm._summary(m.bn.weight) == bm._summary(m.bn.weight.cpu())

    def test_device_mismatch_raises(self, device, small_vocab):
        m = _model(small_vocab).to(device)
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(2, 3, dtype=torch.long, device="meta"))

    def test_checkpoint_to_device(self, device, small_vocab, small, tmp_path):
        run = _run(tmp_path)
        m = _model(small_vocab).to(device)
        train_batchnorm_mlp(m, small, small, training=_cfg(lr=0.3), seed=0, run=run)
        loaded = BatchNormMLPLanguageModel.from_checkpoint(run.checkpoint_path("final"), map_location=device)
        assert loaded.device.type == device and loaded.bn.inference_mean.device.type == device
        x = small.contexts[:8].to(device)
        assert torch.allclose(loaded.eval()(x), m.eval()(x), atol=1e-6)


# ================================================================ real data
@pytest.mark.skipif(not DEFAULT_NAMES_PATH.is_file(), reason="names dataset not present")
def test_real_dataset_initial_state_and_calibration():
    splits = split_names(load_names(DEFAULT_NAMES_PATH), fractions=(0.8, 0.1, 0.1), seed=42,
                         duplicate_policy="group")
    vocab = Vocabulary.from_names(splits.train)
    train = make_context_windows(splits.train, vocab, context_length=3)
    assert train.num_examples == 182_666
    assert min(EpochSampler(train.num_examples, 32, seed=0).batch_sizes()) == 10  # the batch-size guard passes
    m = BatchNormMLPLanguageModel(vocab, generator=make_generator(derive_seed(0, "init")))
    assert count_parameters(m).total == 12_097
    assert evaluate_nll(m, train, batch_size=4096).mean_nll == pytest.approx(LN27, abs=1e-6)
    part = _subset(train, 20_000)
    calibrate_batchnorm(m, part, batch_size=4096)
    a = m.preactivations(part.contexts).detach().double()
    assert torch.allclose(m.bn.inference_var.double(), a.var(0, unbiased=False), rtol=1e-6, atol=0)

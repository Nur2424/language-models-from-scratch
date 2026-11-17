"""Mathematical correctness, unit and integration tests for models/hierarchical/wavenet.py (spec v0.4 §15).

Fixtures: SMALL (12 names, 63 windows at k = 8), TINY (["ab", "ac", "ab"], 9 windows,
Ĥ(Y|W) = 0.212171), V27 (full vocabulary); real data only if present. Tests that differ
by variant are parametrised over ``normalization ∈ {"none", "batchnorm"}``.
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

import models.hierarchical.wavenet as wn
from models.hierarchical.wavenet import (
    BATCHNORM_RECORD,
    EpochRecord,
    IncompatibleCheckpointError,
    LevelBatchNorm1d,
    WaveNetConfig,
    WaveNetLanguageModel,
    WaveNetSample,
    WaveNetTrainingConfig,
    WaveNetTrainingResult,
    calibrate_batchnorm,
    evaluate_nll,
    merge_pairs,
    per_unit_saturation,
    run_config,
    train_wavenet,
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
NORMS = ["none", "batchnorm"]
BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked", "inference_mean", "inference_var",
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
    return make_context_windows(SMALL, small_vocab, context_length=8)


def _model(vocab, normalization="none", seed=0, **kw) -> WaveNetLanguageModel:
    return WaveNetLanguageModel(vocab, WaveNetConfig(normalization=normalization, **kw), generator=make_generator(seed))


def _generic(model, seed=11):
    g = make_generator(seed)
    with torch.no_grad():
        for p in (model.out.weight, model.out.bias):
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.7)
        if model.uses_batchnorm:
            for level in model.levels:
                bn = model._bn(level)
                bn.weight.copy_(1.0 + 0.3 * torch.randn(bn.weight.shape, generator=g, dtype=bn.weight.dtype))
                bn.bias.copy_(0.3 * torch.randn(bn.bias.shape, generator=g, dtype=bn.bias.dtype))
    return model


def _install(model, seed=5):
    g = make_generator(seed)
    for level in model.levels:
        H, dtype = model.config.hidden_dim, model.out.weight.dtype
        model._bn(level).set_inference_statistics(torch.randn(H, generator=g, dtype=dtype),
                                                  torch.rand(H, generator=g, dtype=dtype) + 0.4)


def _prepare_eval(model):
    """Generic parameters; BatchNorm variant gets installed inference statistics; eval mode."""
    _generic(model)
    if model.uses_batchnorm:
        _install(model)
    return model.eval()


def _subset(w: ContextWindows, n: int) -> ContextWindows:
    return ContextWindows(contexts=w.contexts[:n], targets=w.targets[:n], name_index=w.name_index[:n])


def _conditional_entropy(w: ContextWindows) -> float:
    by_ctx: dict = defaultdict(Counter)
    for c, y in zip(w.contexts.tolist(), w.targets.tolist()):
        by_ctx[tuple(c)][y] += 1
    total = 0.0
    for counts in by_ctx.values():
        n = sum(counts.values())
        total -= sum(k * math.log(k / n) for k in counts.values())
    return total / w.num_examples


def _run(tmp_path: Path, name: str) -> RunManager:
    return RunManager.create(name, config={}, results_root=tmp_path)


def _rows(history) -> list[dict]:
    out = []
    for r in history:
        d = asdict(r) if isinstance(r, EpochRecord) else dict(r)
        d.pop("seconds")
        out.append(d)
    return out


def _state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_same_state(a, b):
    assert a.keys() == b.keys()
    for k, value in a.items():
        assert torch.equal(value.cpu(), b[k].cpu()), k


def _cfg(**kw) -> WaveNetTrainingConfig:
    return WaveNetTrainingConfig(**{"lr": 0.1, "max_epochs": 1, **kw})


def _hooked(model, x, names):
    seen = {}
    handles = []
    for name in names:
        def hook(_m, _i, o, name=name):
            seen[name] = o
        handles.append(dict(model.named_modules())[name].register_forward_hook(hook))
    try:
        logits = model(x)
    finally:
        for h in handles:
            h.remove()
    return seen, logits


def _manual_epochs(model, w, *, batch_size, lr, seed, epochs, denominator=None):
    sampler = EpochSampler(w.num_examples, batch_size, seed=derive_seed(seed, "data"))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    model.train()
    batches = []
    for e in range(epochs):
        for idx in sampler.batches(e):
            opt.zero_grad(set_to_none=True)
            total = token_nll(model(w.contexts[idx]), w.targets[idx], reduction="sum")
            batches.append((len(idx), float(total.detach()) / len(idx)))
            (total / (len(idx) if denominator is None else denominator)).backward()
            opt.step()
    return batches


X8 = torch.tensor([[0] * 8, [0, 0, 0, 0, 0, 0, 5, 9], [1, 2, 3, 4, 5, 6, 7, 8], [26, 3, 3, 1, 0, 14, 2, 2],
                   [9, 9, 9, 9, 9, 9, 9, 9], [4, 0, 7, 0, 7, 0, 4, 1]])


# ======================================= §15.1 mathematical correctness tests
class TestMathematicalCorrectness:
    @pytest.mark.parametrize("normalization", NORMS)
    def test_receptive_field_autograd(self, v27, normalization):
        m = _prepare_eval(_model(v27, normalization)).double()
        x = torch.randint(1, 27, (2, 8), generator=make_generator(1))
        for level in m.levels:
            for group in range(1, m.config.level_groups(level) + 1):
                kept = {}

                def keep_e(_m, _i, o, kept=kept):
                    o.retain_grad()
                    kept["e"] = o

                def keep_h(_m, _i, o, kept=kept):
                    kept["h"] = o

                handles = [m.emb.register_forward_hook(keep_e), m._act(level).register_forward_hook(keep_h)]
                try:
                    m(x)
                finally:
                    for h in handles:
                        h.remove()
                kept["h"][:, group - 1].sum().backward()
                deps = [p + 1 for p in range(8) if kept["e"].grad[:, p].abs().sum() > 0]
                width = 2 ** level
                assert deps == list(range((group - 1) * width + 1, group * width + 1)), (level, group)

    @pytest.mark.parametrize("normalization", NORMS)
    def test_receptive_field_perturbation(self, v27, normalization):
        m = _prepare_eval(_model(v27, normalization))
        x = torch.randint(1, 27, (1, 8), generator=make_generator(2))
        names = [f"act{lvl}" for lvl in m.levels]
        base, base_logits = _hooked(m, x, names)
        for p in range(8):
            y = x.clone()
            y[0, p] = (y[0, p] % 26) + 1
            seen, logits = _hooked(m, y, names)
            assert not torch.allclose(logits, base_logits)  # every position reaches the output
            for level in m.levels:
                g = p // 2 ** level  # the only group of this level that sees position p
                for j in range(m.config.level_groups(level)):
                    same = torch.equal(seen[f"act{level}"][:, j], base[f"act{level}"][:, j])
                    assert same == (j != g), (p, level, j)

    @pytest.mark.parametrize("normalization, mode", [("none", "eval"), ("batchnorm", "eval"), ("batchnorm", "train")])
    def test_level1_pair_permutation_equivariance(self, v27, normalization, mode):
        m = _prepare_eval(_model(v27, normalization))
        m.train(mode == "train")
        perm = [2, 0, 3, 1]
        xp = X8.view(6, 4, 2)[:, perm].reshape(6, 8)
        a, _ = _hooked(m, X8, ["act1"])
        b, _ = _hooked(m, xp, ["act1"])
        assert torch.allclose(b["act1"], a["act1"][:, perm], atol=1e-6)

    @pytest.mark.parametrize("normalization", NORMS)
    def test_order_within_a_pair_matters(self, v27, normalization):
        m = _prepare_eval(_model(v27, normalization))
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        y = torch.tensor([[2, 1, 3, 4, 5, 6, 7, 8]])
        assert not torch.allclose(m(x), m(y))

    @pytest.mark.parametrize("normalization", NORMS)
    def test_zero_output_initialisation(self, v27, normalization):
        w = make_context_windows(SMALL, v27, context_length=8)
        m = _model(v27, normalization)
        assert evaluate_nll(m, w, batch_size=17).mean_nll == pytest.approx(LN27, abs=1e-6)
        m.train()
        loss = token_nll(m(w.contexts[:32]), w.targets[:32], reduction="mean")
        assert loss.item() == pytest.approx(LN27, abs=1e-6)
        loss.backward()
        for name, p in m.named_parameters():
            if not name.startswith("out."):
                assert torch.count_nonzero(p.grad) == 0, name
        assert torch.count_nonzero(m.out.weight.grad) > 0 and torch.count_nonzero(m.out.bias.grad) > 0
        torch.optim.SGD(m.parameters(), lr=0.1).step()
        m.zero_grad()
        token_nll(m(w.contexts[:32]), w.targets[:32], reduction="mean").backward()
        for name, p in m.named_parameters():
            assert torch.count_nonzero(p.grad) > 0, name

    def test_batchnorm_identities(self, v27):
        m = _generic(_model(v27, "batchnorm")).double().train()
        y = torch.tensor([1, 3, 0, 5, 2, 7])
        kept = {}

        def keep(name):
            def hook(_m, _i, o):
                o.retain_grad()
                kept[name] = o
            return hook

        handles = [m._hidden(lvl).register_forward_hook(keep(f"a{lvl}")) for lvl in m.levels]
        handles += [m._bn(lvl).register_forward_hook(keep(f"y{lvl}")) for lvl in m.levels]
        try:
            token_nll(m(X8), y, reduction="mean").backward()
        finally:
            for h in handles:
                h.remove()
        for lvl in m.levels:
            a, g = kept[f"a{lvl}"], kept[f"y{lvl}"].grad
            assert a.grad.sum(dim=(0, 1)).abs().max() < 1e-12
            mean, var = a.mean(dim=(0, 1)), a.var(dim=(0, 1), unbiased=False)
            ahat = (a - mean) / torch.sqrt(var + 1e-5)
            assert torch.allclose(m._bn(lvl).weight.grad, (g * ahat).sum(dim=(0, 1)), atol=1e-12)
            assert torch.allclose(m._bn(lvl).bias.grad, g.sum(dim=(0, 1)), atol=1e-12)
        # scale invariance: exact as ε → 0 (PyTorch rejects ε = 0 in train mode; 1e-30 is negligible)
        m0 = copy.deepcopy(m)
        for lvl in m0.levels:
            m0._bn(lvl).eps = 1e-30
        m5 = copy.deepcopy(m0)
        with torch.no_grad():
            m5.hidden2.weight[7] *= 5.0
        assert torch.allclose(m0(X8), m5(X8), atol=1e-10)

    @pytest.mark.parametrize("normalization", NORMS)
    def test_parameter_formula_and_breakdown(self, v27, normalization):
        m = _model(v27, normalization)
        expected = {"emb": 270, "hidden1": 1_360, "hidden2": 9_248, "hidden3": 9_248, "out": 1_863}
        if normalization == "batchnorm":
            expected.update({"bn1": 136, "bn2": 136, "bn3": 136})
        assert {k: c.total for k, c in parameter_breakdown(m).items()} == expected
        assert count_parameters(m).total == count_parameters(m).trainable == {"none": 21_989,
                                                                             "batchnorm": 22_397}[normalization]
        for kw in ({"context_length": 16, "hidden_dim": 32}, {"context_length": 2, "embedding_dim": 4}):
            c = WaveNetConfig(normalization=normalization, **kw)
            d, H, L = c.embedding_dim, c.hidden_dim, c.num_levels
            formula = 27 * d + 2 * d * H + (L - 1) * 2 * H * H + H * 27 + 27 + (L * 2 * H if normalization ==
                                                                                 "batchnorm" else 0)
            assert count_parameters(_model(v27, normalization, **kw)).total == formula

    @pytest.mark.parametrize("normalization", NORMS)
    def test_conditional_entropy_bound(self, small_vocab, small, normalization):
        tiny_vocab = Vocabulary.from_names(TINY)
        tiny = make_context_windows(TINY, tiny_vocab, context_length=8)
        assert _conditional_entropy(tiny) == pytest.approx(H_TINY, abs=1e-12)
        for vocab, w in ((tiny_vocab, tiny), (small_vocab, small)):
            bound = _conditional_entropy(w)
            for seed in range(3):
                m = _generic(_model(vocab, normalization, seed=seed), seed=seed)
                if normalization == "batchnorm":
                    calibrate_batchnorm(m, w, batch_size=64)
                assert evaluate_nll(m, w, batch_size=64).mean_nll >= bound - 1e-6
            m = _model(vocab, normalization)
            train_wavenet(m, w, w, training=_cfg(lr=0.5, max_epochs=30, batch_size=5), seed=0)
            assert evaluate_nll(m, w, batch_size=64).mean_nll >= bound - 1e-6

    @pytest.mark.parametrize("normalization", NORMS)
    def test_gradcheck(self, v27, normalization):
        m = _generic(_model(v27, normalization, hidden_dim=4)).double().train()
        x, y = X8[:4], torch.tensor([1, 3, 0, 2])
        names = [n for n, _ in m.named_parameters()]
        params = tuple(p.detach().clone().requires_grad_(True) for p in m.parameters())

        def f(*ps):
            return token_nll(functional_call(m, dict(zip(names, ps)), (x,)), y, reduction="mean")

        assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-6)


# ============================== §15.2 construction, configuration and shapes
class TestConfiguration:
    @pytest.mark.parametrize("value", [0, 1, 3, 6, 12, True, "8", 2.0])
    def test_context_length_rejected(self, value):
        with pytest.raises(ValueError, match="context_length"):
            WaveNetConfig(context_length=value)

    @pytest.mark.parametrize("value", [2, 4, 8, 16])
    def test_context_length_accepted(self, value):
        c = WaveNetConfig(context_length=value)
        assert c.num_levels == int(math.log2(value))

    @pytest.mark.parametrize("value", ["", "BatchNorm", None, "layernorm", 1])
    def test_normalization_rejected(self, value):
        with pytest.raises(ValueError, match="normalization"):
            WaveNetConfig(normalization=value)

    @pytest.mark.parametrize("field", ["embedding_dim", "hidden_dim"])
    @pytest.mark.parametrize("bad", [0, -1, True, 2.5, "10"])
    def test_dims_rejected(self, field, bad):
        with pytest.raises(ValueError, match=field):
            WaveNetConfig(**{field: bad})

    @pytest.mark.parametrize(
        "kwargs, match",
        [({"lr": 0}, "lr"), ({"lr": float("inf")}, "lr"), ({"max_epochs": 0}, "max_epochs"),
         ({"batch_size": 0}, "batch_size"), ({"eval_batch_size": 0}, "eval_batch_size"),
         ({"lr_decay_epoch": 4}, "lr_decay_epoch"), ({"lr_decay_factor": 0.0}, "lr_decay_factor")],
    )
    def test_training_config_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WaveNetTrainingConfig(**{"lr": 0.1, "max_epochs": 4, **kwargs})

    def test_batch_size_one_baseline_ok_batchnorm_rejected(self, small_vocab, small):
        cfg = _cfg(batch_size=1)
        train_wavenet(_model(small_vocab), _subset(small, 4), _subset(small, 4), training=cfg, seed=0)
        with pytest.raises(ValueError, match="batch_size >= 2"):
            train_wavenet(_model(small_vocab, "batchnorm"), small, small, training=cfg, seed=0)

    @pytest.mark.parametrize("name, normalization", [("wavenet", "none"), ("wavenet_batchnorm", "batchnorm")])
    def test_yaml_identity_pairing(self, v27, name, normalization):
        cfg = load_config(default_config_paths(name), model_config_cls=WaveNetConfig)
        assert cfg.model.name == name and cfg.model.params.normalization == normalization
        assert cfg.model.params == WaveNetConfig(normalization=normalization)
        assert WaveNetLanguageModel(v27, cfg.model.params, generator=make_generator(0)).model_name == name
        with pytest.raises(ConfigError, match="unknown"):
            load_config(default_config_paths(name), overrides=["model.params.gain=2"], model_config_cls=WaveNetConfig)

    def test_module_inventory(self, v27):
        base = _model(v27)
        names = {n for n, _ in base.named_children()}
        assert names == {"emb", "hidden1", "hidden2", "hidden3", "act1", "act2", "act3", "out"}
        assert not any(isinstance(mod, nn.BatchNorm1d) for mod in base.modules()) and not list(base.buffers())
        bn = _model(v27, "batchnorm")
        assert {n for n, _ in bn.named_children()} == names | {"bn1", "bn2", "bn3"}
        for lvl in bn.levels:
            mod = bn._bn(lvl)
            assert isinstance(mod, LevelBatchNorm1d) and isinstance(mod, nn.BatchNorm1d)
            assert (mod.eps, mod.momentum) == (1e-5, 0.1)
            assert {n for n, _ in mod.named_buffers()} == set(BN_BUFFERS)
        for m in (base, bn):
            assert all(m._hidden(lvl).bias is None for lvl in m.levels)

    def test_module_imports_no_other_model(self):
        tree = ast.parse(Path(wn.__file__).read_text())
        imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [i for i in imported if i and i.startswith("models")]

    @pytest.mark.parametrize("normalization", NORMS)
    @pytest.mark.parametrize("B", [1, 2, 32])
    def test_shape_trace(self, v27, normalization, B):
        m = _prepare_eval(_model(v27, normalization))
        shapes = {}
        handles = []
        for name, mod in m.named_children():
            if name == "emb":
                continue

            def hook(_m, inputs, output, name=name):
                shapes[f"{name}.in"] = tuple(inputs[0].shape)
                shapes[f"{name}.out"] = tuple(output.shape)

            handles.append(mod.register_forward_hook(hook))
        try:
            x = torch.randint(0, 27, (B, 8), generator=make_generator(B))
            assert tuple(m(x).shape) == (B, 27)
        finally:
            for h in handles:
                h.remove()
        expected = {"hidden1.in": (B, 4, 20), "hidden1.out": (B, 4, 68), "act1.out": (B, 4, 68),
                    "hidden2.in": (B, 2, 136), "hidden2.out": (B, 2, 68), "act2.out": (B, 2, 68),
                    "hidden3.in": (B, 1, 136), "hidden3.out": (B, 1, 68), "act3.out": (B, 1, 68),
                    "out.in": (B, 68), "out.out": (B, 27)}
        if normalization == "batchnorm":
            expected.update({"bn1.out": (B, 4, 68), "bn2.out": (B, 2, 68), "bn3.out": (B, 1, 68)})
        for key, shape in expected.items():
            assert shapes[key] == shape, key

    def test_public_api_and_identity(self, v27):
        for normalization, name in (("none", "wavenet"), ("batchnorm", "wavenet_batchnorm")):
            m = _model(v27, normalization)
            assert m.model_name == name == m.config.model_name
            with pytest.raises(AttributeError):
                m.model_name = "other"
            assert not hasattr(m, "level_preactivations")
        assert merge_pairs is wn.merge_pairs


# =================================================== §15.3 input contract
class TestInputContract:
    @pytest.mark.parametrize("normalization", NORMS)
    def test_shapes_and_empty_eval(self, v27, normalization):
        m = _model(v27, normalization)
        for mode in (True, False):
            m.train(mode)
            z = m(X8)
            assert z.shape == (6, 27) and z.dtype == torch.float32
        m.eval()
        assert m(torch.zeros(0, 8, dtype=torch.long)).shape == (0, 27)

    @pytest.mark.parametrize("B", [0, 1])
    def test_batchnorm_train_batch_below_two(self, v27, B):
        m = _model(v27, "batchnorm").train()
        before = _state(m)
        with pytest.raises(ValueError, match="at least 2 windows"):
            m(torch.zeros(B, 8, dtype=torch.long))
        _assert_same_state(before, _state(m))
        base = _model(v27).train()
        assert base(torch.zeros(1, 8, dtype=torch.long)).shape == (1, 27)

    @pytest.mark.parametrize("normalization", NORMS)
    def test_rejections(self, v27, normalization):
        m = _model(v27, normalization).eval()
        for shape in ((8,), (2, 7), (2, 9), (1, 8, 1)):
            with pytest.raises(ValueError, match="shape"):
                m(torch.zeros(shape, dtype=torch.long))
        with pytest.raises(TypeError, match="torch.Tensor"):
            m([[0] * 8])
        with pytest.raises(TypeError, match="int64"):
            m(torch.zeros(2, 8, dtype=torch.int32))
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(2, 8, dtype=torch.long, device="meta"))
        with pytest.raises(ValueError, match="padding must not reach the model"):
            m(torch.tensor([[0] * 7 + [v27.pad_id]]))
        for bad in (-1, 28):
            with pytest.raises(ValueError, match="outside"):
                m(torch.tensor([[0] * 7 + [bad]]))

    def test_private_level_preactivations(self, v27):
        m = _prepare_eval(_model(v27, "batchnorm"))
        for level, g in ((1, 4), (2, 2), (3, 1)):
            assert m._level_preactivations(X8, level=level).shape == (6, g, 68)
        for bad in (0, 4, True, "1"):
            with pytest.raises(ValueError, match="level"):
                m._level_preactivations(X8, level=bad)


# ===================================================== §15.4 initialisation
class TestInitialisation:
    @pytest.mark.parametrize("normalization", NORMS)
    def test_exact_values(self, v27, normalization):
        m = _model(v27, normalization)
        assert torch.count_nonzero(m.out.weight) == 0 and torch.count_nonzero(m.out.bias) == 0
        assert torch.count_nonzero(m.emb.weight) == 270
        if normalization == "batchnorm":
            for lvl in m.levels:
                bn = m._bn(lvl)
                assert torch.equal(bn.weight, torch.ones(68)) and torch.equal(bn.bias, torch.zeros(68))
                assert torch.equal(bn.running_mean, torch.zeros(68)) and torch.equal(bn.running_var, torch.ones(68))
                assert torch.equal(bn.inference_mean, torch.zeros(68))
                assert torch.equal(bn.inference_var, torch.ones(68))
                assert int(bn.num_batches_tracked) == 0 and int(bn.num_calibrations) == 0

    @pytest.mark.parametrize("normalization", NORMS)
    def test_determinism_and_global_rng(self, v27, normalization):
        rng = torch.get_rng_state()
        a = _model(v27, normalization, seed=4)
        assert torch.equal(rng, torch.get_rng_state())
        torch.manual_seed(123)
        b = _model(v27, normalization, seed=4)
        c = _model(v27, normalization, seed=5)
        _assert_same_state(_state(a), _state(b))
        assert not torch.equal(a.hidden2.weight, c.hidden2.weight)
        # both variants draw E, W1, W2, W3 identically from the same generator
        if normalization == "batchnorm":
            base = _model(v27, "none", seed=4)
            for name in ("emb.weight", "hidden1.weight", "hidden2.weight", "hidden3.weight"):
                assert torch.equal(a.state_dict()[name], base.state_dict()[name]), name

    def test_lecun_scales(self, v27):
        m = _model(v27, seed=0)
        E = m.emb.weight
        assert abs(E.std().item() - 1) < 0.2 and abs(E.mean().item()) < 0.3
        assert abs(m.hidden1.weight.std().item() * math.sqrt(20) - 1) < 0.10
        for lvl in (2, 3):
            assert abs(m._hidden(lvl).weight.std().item() * math.sqrt(136) - 1) < 0.05
        k16 = _model(v27, seed=0, context_length=16, hidden_dim=32)  # C₁ = 20, C_ℓ = 64 above
        assert abs(k16.hidden1.weight.std().item() * math.sqrt(20) - 1) < 0.10
        for lvl in (2, 3, 4):
            w = k16._hidden(lvl).weight
            assert w.shape == (32, 64) and abs(w.std().item() * math.sqrt(64) - 1) < 0.10


# ================================== §15.5 training, recalibration, diagnostics
class TestTraining:
    @pytest.mark.parametrize("normalization", NORMS)
    def test_overfit_tiny_fixture(self, normalization):
        vocab = Vocabulary.from_names(TINY)
        w = make_context_windows(TINY, vocab, context_length=8)
        m = _model(vocab, normalization)
        train_wavenet(m, w, w, training=_cfg(lr=0.5, max_epochs=500), seed=0)
        gap = evaluate_nll(m, w, batch_size=64).mean_nll - H_TINY
        assert -1e-6 <= gap <= 1e-2

    @pytest.mark.parametrize("normalization", NORMS)
    def test_mean_reduction_over_actual_batch(self, small_vocab, small, normalization):
        w = _subset(small, 42)
        trained = _model(small_vocab, normalization)
        train_wavenet(trained, w, w, training=_cfg(lr=0.5, max_epochs=2), seed=3)
        reference = _model(small_vocab, normalization)
        _manual_epochs(reference, w, batch_size=32, lr=0.5, seed=3, epochs=2)
        for (n, a), b in zip(trained.named_parameters(), reference.parameters()):
            assert torch.allclose(a, b, rtol=0, atol=1e-6), n
        nominal = _model(small_vocab, normalization)
        _manual_epochs(nominal, w, batch_size=32, lr=0.5, seed=3, epochs=2, denominator=32)
        assert (trained.out.bias - nominal.out.bias).abs().max() > 1e-3

    @pytest.mark.parametrize("normalization", NORMS)
    def test_running_nll_example_weighted(self, small_vocab, small, normalization):
        w = _subset(small, 42)
        r = train_wavenet(_model(small_vocab, normalization), w, w, training=_cfg(lr=0.5), seed=3)
        batches = _manual_epochs(_model(small_vocab, normalization), w, batch_size=32, lr=0.5, seed=3, epochs=1)
        assert r.history[0].running_train_nll == pytest.approx(sum(n * x for n, x in batches) / 42, rel=1e-6)

    @pytest.mark.parametrize("normalization", NORMS)
    @pytest.mark.parametrize("batch_size", [1, 7, 63, 64])
    def test_evaluation_covers_every_window(self, small_vocab, small, normalization, batch_size):
        m = _prepare_eval(_model(small_vocab, normalization))
        acc = evaluate_nll(m, small, batch_size=batch_size)
        with torch.no_grad():
            full = token_nll(m(small.contexts), small.targets, reduction="mean").item()
        assert acc.num_tokens == 63 and acc.mean_nll == pytest.approx(full, rel=1e-6)

    @pytest.mark.parametrize("normalization", NORMS)
    def test_determinism(self, small_vocab, small, normalization):
        cfg = _cfg(lr=0.3, max_epochs=3)
        a, b, c = (_model(small_vocab, normalization) for _ in range(3))
        ra = train_wavenet(a, small, small, training=cfg, seed=7)
        rb = train_wavenet(b, small, small, training=cfg, seed=7)
        rc = train_wavenet(c, small, small, training=cfg, seed=8)
        _assert_same_state(_state(a), _state(b))
        assert _rows(ra.history) == _rows(rb.history) and _rows(ra.history) != _rows(rc.history)

    def test_schedule_and_result(self, small_vocab, small):
        r = train_wavenet(_model(small_vocab), small, small, training=_cfg(lr=0.4, max_epochs=4, lr_decay_epoch=2),
                          seed=0)
        assert isinstance(r, WaveNetTrainingResult) and r.epochs_completed == 4
        assert [h.lr for h in r.history] == [0.4, 0.4, 0.4 * 0.1, 0.4 * 0.1]
        assert [h.steps for h in r.history] == [2, 4, 6, 8]
        best = min(range(4), key=lambda e: (r.history[e].dev_nll, e))
        assert r.best_epoch == best

    def test_selection_ties_keep_earliest(self, small_vocab, small, tmp_path, monkeypatch):
        dev = _subset(small, 40)
        values = iter([3.0, 2.0, 2.0, 2.5])
        real = wn.evaluate_nll

        def fake(model, windows, *, batch_size):
            acc = real(model, windows, batch_size=batch_size)
            if windows is dev:
                acc = NLLAccumulator()
                acc.update_from_sum(next(values) * 40, 40)
            return acc

        monkeypatch.setattr(wn, "evaluate_nll", fake)
        run = _run(tmp_path, "wavenet")
        r = train_wavenet(_model(small_vocab), small, dev, training=_cfg(max_epochs=4), seed=0, run=run)
        assert [h.is_best for h in r.history] == [True, True, False, False]
        assert load_checkpoint(run.checkpoint_path("best"))["extra"]["best_epoch"] == 1

    @pytest.mark.parametrize("normalization", NORMS)
    def test_nan_loss_and_gradient_guards(self, small_vocab, small, normalization, monkeypatch):
        m = _model(small_vocab, normalization)
        real, calls, before = wn.token_nll, [0], {}

        def poisoned(logits, targets, *, reduction):
            calls[0] += 1
            loss = real(logits, targets, reduction=reduction)
            if calls[0] == 4:
                before.update({n: p.detach().clone() for n, p in m.named_parameters()})
                return loss * float("nan")
            return loss

        with monkeypatch.context() as mp:
            mp.setattr(wn, "token_nll", poisoned)
            with pytest.raises(FloatingPointError, match=r"non-finite loss .* at epoch 1, step 3"):
                train_wavenet(m, small, small, training=_cfg(max_epochs=3), seed=0)
        for n, p in m.named_parameters():
            assert torch.equal(p, before[n]), n
        m2, calls2, before2 = _model(small_vocab, normalization), [0], {}

        def hook(grad):
            calls2[0] += 1
            if calls2[0] == 3:
                before2.update({n: p.detach().clone() for n, p in m2.named_parameters()})
                return grad * float("nan")
            return grad

        m2.hidden2.weight.register_hook(hook)
        with pytest.raises(FloatingPointError, match=r"epoch 1, step 2 in parameter groups \['hidden2'\]"):
            train_wavenet(m2, small, small, training=_cfg(max_epochs=3), seed=0)
        for n, p in m2.named_parameters():
            assert torch.equal(p, before2[n]), n

    def test_batchnorm_batch_size_guard(self, small_vocab, small):
        assert EpochSampler(63, 31, seed=0).batch_sizes() == [31, 31, 1]
        m = _model(small_vocab, "batchnorm")
        before = _state(m)
        with pytest.raises(ValueError, match="at least 2 windows"):
            train_wavenet(m, small, small, training=_cfg(batch_size=31), seed=0)
        _assert_same_state(before, _state(m))
        train_wavenet(_model(small_vocab), small, small, training=_cfg(batch_size=31), seed=0)  # baseline: fine

    def test_batchnorm_epoch_end_order(self, small_vocab, small, monkeypatch):
        events = []
        real_nll, real_eval = wn.token_nll, wn.evaluate_nll
        real_level = WaveNetLanguageModel._level_preactivations
        dev = _subset(small, 40)
        monkeypatch.setattr(wn, "token_nll", lambda *a, **k: events.append("step") or real_nll(*a, **k))
        monkeypatch.setattr(wn, "evaluate_nll", lambda model, windows, *, batch_size: events.append(
            "eval_train" if windows is small else "eval_dev") or real_eval(model, windows, batch_size=batch_size))
        monkeypatch.setattr(WaveNetLanguageModel, "_level_preactivations",
                            lambda self, c, level: events.append(f"cal{level}") or real_level(self, c, level))
        m = _model(small_vocab, "batchnorm")
        train_wavenet(m, small, dev, training=_cfg(max_epochs=2), seed=0)
        assert events == ["step", "step", "cal1", "cal2", "cal3", "eval_train", "eval_dev"] * 2
        assert [int(m._bn(lvl).num_calibrations) for lvl in m.levels] == [2, 2, 2]

    def test_baseline_never_calibrates(self, small_vocab, small, monkeypatch):
        called = []
        monkeypatch.setattr(wn, "calibrate_batchnorm", lambda *a, **k: called.append(1))
        r = train_wavenet(_model(small_vocab), small, small, training=_cfg(max_epochs=2), seed=0)
        assert not called and all(h.batchnorm is None for h in r.history)
        with pytest.raises(ValueError, match="normalization='batchnorm'"):
            calibrate_batchnorm(_model(small_vocab), small, batch_size=8)  # the real function (imported name)

    def test_batchnorm_mode_discipline(self, small_vocab, small, tmp_path):
        run = _run(tmp_path, "wavenet_batchnorm")
        schedule = DiagnosticsSchedule(enabled=True, every_n_steps=1, gradients=True, parameters=True,
                                       activations=True)
        m = _model(small_vocab, "batchnorm")
        r = train_wavenet(m, small, small, training=_cfg(lr=0.3, max_epochs=3), seed=0, run=run,
                          diagnostics=schedule)
        assert [int(m._bn(lvl).num_batches_tracked) for lvl in m.levels] == [r.history[-1].steps] * 3 == [6] * 3

    def test_batchnorm_streaming_recalibration(self, small_vocab, small, monkeypatch):
        m = _generic(_model(small_vocab, "batchnorm"))
        original = m._level_preactivations
        calls, refs = [], []

        def wrapped(contexts, level):
            if refs:
                gc.collect()
                assert refs[-1]() is None, "previous chunk still alive"
            out = original(contexts, level=level)
            calls.append((level, contexts.shape[0]))
            refs.append(weakref.ref(out))
            return out

        monkeypatch.setattr(m, "_level_preactivations", wrapped)
        calibrate_batchnorm(m, small, batch_size=10)
        gc.collect()
        assert refs[-1]() is None
        assert [lvl for lvl, _ in calls] == [1] * 7 + [2] * 7 + [3] * 7
        assert all(n <= 10 for _, n in calls) and sum(n for lvl, n in calls if lvl == 1) == 63

    @pytest.mark.parametrize("training", [True, False])
    def test_batchnorm_recalibration_isolation_and_exact_partition_invariance(self, small_vocab, small, training):
        """Algorithmic partition invariance, exact: a float64 model (matmul rounding ~1e-16), so any
        chunking must give the same statistics to 1e-12. Chunk sizes include uneven partitions
        (7 → 9×7; 10 → 6×10 + 3; 68 > M) and the single-row case. Also checks isolation from the
        EMA, the previous inference statistics, the parameters and the model mode."""
        m = _generic(_model(small_vocab, "batchnorm")).double()
        reference = copy.deepcopy(m)
        calibrate_batchnorm(reference, small, batch_size=64)  # one chunk (M = 63)
        with torch.no_grad():
            for lvl in m.levels:
                bn = m._bn(lvl)
                bn.running_mean.fill_(9.0)
                bn.running_var.fill_(0.3)
                bn.num_batches_tracked.fill_(5)
                bn.inference_mean.fill_(-3.0)
                bn.inference_var.fill_(40.0)
        m.train(training)
        params = {n: p.detach().clone() for n, p in m.named_parameters()}
        for bs in (1, 7, 10, 63, 68):
            calibrate_batchnorm(m, small, batch_size=bs)
            for lvl in m.levels:
                bn, rb = m._bn(lvl), reference._bn(lvl)
                assert torch.allclose(bn.inference_mean, rb.inference_mean, rtol=0, atol=1e-12), (bs, lvl)
                assert torch.allclose(bn.inference_var, rb.inference_var, rtol=0, atol=1e-12), (bs, lvl)
                assert torch.equal(bn.running_mean, torch.full((68,), 9.0, dtype=torch.float64))
                assert torch.equal(bn.running_var, torch.full((68,), 0.3, dtype=torch.float64))
                assert int(bn.num_batches_tracked) == 5
        assert m.training is training
        for n, p in m.named_parameters():
            assert torch.equal(p, params[n]), n
        assert [int(m._bn(lvl).num_calibrations) for lvl in m.levels] == [5, 5, 5]

    def test_batchnorm_recalibration_float32_accuracy(self, small_vocab, small):
        """float32 calibration is exact up to float32 rounding of the activations, for every chunking.

        The float32 pre-activations themselves depend on the chunk size: the matrix-multiply kernel
        sums in a different order for different batch shapes, so the same window's a^ℓ differs by a
        few units of float32 roundoff u = 2^-24 between chunkings, and the difference propagates to
        higher levels. (Level 1 here is an exact embedding lookup followed by a matmul.) The
        statistics are therefore compared with the *exact* float64 calibration of the same
        parameters, on the scale of the underlying activations: per unit,
        |Δm| ≤ τ·RMS(a) and |Δv| ≤ τ·RMS(a)², with RMS(a) = sqrt(v + m²) and τ = 64u ≈ 3.8e-6.
        Measured on x86/MKL: worst 8.4u over 10 seeds, 2 thread counts, 4 chunk sizes, 3 levels.
        Real defects are orders of magnitude larger (e.g. an M−1 divisor: relative 1/62 ≈ 1.6e-2
        at level 3; non-sequential calibration: O(0.1))."""
        tau = 64 * 2.0 ** -24
        m = _generic(_model(small_vocab, "batchnorm"))
        exact = copy.deepcopy(m).double()
        calibrate_batchnorm(exact, small, batch_size=64)
        for bs in (1, 7, 10, 63, 68):
            f = copy.deepcopy(m)
            calibrate_batchnorm(f, small, batch_size=bs)
            for lvl in m.levels:
                e, a = exact._bn(lvl), f._bn(lvl)
                rms = (e.inference_var + e.inference_mean.square()).sqrt()
                err_m = ((a.inference_mean.double() - e.inference_mean).abs() / rms).max().item()
                err_v = ((a.inference_var.double() - e.inference_var).abs() / rms.square()).max().item()
                assert err_m <= tau, f"chunk {bs}, level {lvl}: mean error {err_m / 2.0 ** -24:.1f}u > 64u"
                assert err_v <= tau, f"chunk {bs}, level {lvl}: variance error {err_v / 2.0 ** -24:.1f}u > 64u"

    def test_batchnorm_non_finite_calibration(self, small_vocab, small, monkeypatch, tmp_path):
        real, calls = wn._chunk_moments, [0]

        def poisoned(rows):
            calls[0] += 1
            n, mean, m2 = real(rows)
            return (n, mean * float("nan"), m2) if calls[0] == 5 else (n, mean, m2)  # epoch 1, level 2

        monkeypatch.setattr(wn, "_chunk_moments", poisoned)
        m, run = _model(small_vocab, "batchnorm"), _run(tmp_path, "wavenet_batchnorm")
        with pytest.raises(FloatingPointError, match="recalibration failed at epoch 1: level 2"), run.track():
            train_wavenet(m, small, small, training=_cfg(max_epochs=3), seed=0, run=run)
        assert [int(m._bn(lvl).num_calibrations) for lvl in m.levels] == [2, 1, 1]  # level 2 not installed
        assert run.status is RunStatus.FAILED
        assert load_checkpoint(run.checkpoint_path("resume"))["epoch"] == 1

    def test_run_and_experiment_identity_checks(self, small_vocab, small, tmp_path):
        with pytest.raises(ValueError, match="belongs to 'wavenet_batchnorm'"):
            train_wavenet(_model(small_vocab), small, small, training=_cfg(), seed=0,
                          run=_run(tmp_path, "wavenet_batchnorm"))
        with pytest.raises(ValueError, match="experiment config names model 'wavenet_batchnorm'"):
            train_wavenet(_model(small_vocab), small, small, training=_cfg(), seed=0,
                          config={"model": {"name": "wavenet_batchnorm"}})

    def test_argument_validation(self, small_vocab, small):
        m = _model(small_vocab)
        with pytest.raises(ValueError, match="resume=True requires a run"):
            train_wavenet(m, small, small, training=_cfg(), seed=0, resume=True)
        with pytest.raises(ValueError, match="pass a run"):
            train_wavenet(m, small, small, training=_cfg(), seed=0, diagnostics=DiagnosticsSchedule(enabled=True))
        with pytest.raises(ValueError, match="context_length"):
            train_wavenet(m, make_context_windows(SMALL, small_vocab, context_length=4), small, training=_cfg(),
                          seed=0)

    @pytest.mark.parametrize("normalization", NORMS)
    @pytest.mark.parametrize("starting", [True, False])
    def test_mode_restoration(self, small_vocab, small, normalization, starting):
        m = _prepare_eval(_model(small_vocab, normalization))
        m.train(starting)
        buffers = {k: v.clone() for k, v in m.named_buffers()}
        bad = ContextWindows(contexts=small.contexts.clone(), targets=small.targets, name_index=small.name_index)
        bad.contexts[40, 0] = small_vocab.pad_id
        for fn in (evaluate_nll, per_unit_saturation, wn._output_summary):
            fn(m, small, batch_size=10)
            assert m.training is starting
            with pytest.raises(ValueError, match="padding"):
                fn(m, bad, batch_size=10)
            assert m.training is starting
        m.generate(2, max_length=5, generator=make_generator(0))
        assert m.training is starting
        with pytest.raises(Exception, match="temperature"):
            m.generate(2, max_length=5, temperature=-1.0)
        assert m.training is starting
        for k, v in m.named_buffers():
            assert torch.equal(v, buffers[k]), k

    @pytest.mark.parametrize("normalization", NORMS)
    def test_diagnostics(self, small_vocab, small, normalization, tmp_path):
        cfg = _cfg(lr=0.3, max_epochs=2)
        plain = _model(small_vocab, normalization)
        train_wavenet(plain, small, small, training=cfg, seed=0)
        m = _model(small_vocab, normalization)
        run = _run(tmp_path, m.model_name)
        schedule = DiagnosticsSchedule(enabled=True, every_n_steps=2, gradients=True, parameters=True,
                                       activations=True)
        r = train_wavenet(m, small, small, training=cfg, seed=0, run=run, diagnostics=schedule)
        _assert_same_state(_state(plain), _state(m))  # diagnostics never change the trajectory
        rows = read_jsonl(run.diagnostics_dir / "steps.jsonl")
        assert [x["step"] for x in rows] == [0, 2]
        groups = {"emb", "hidden1", "hidden2", "hidden3", "out"} | (
            {"bn1", "bn2", "bn3"} if normalization == "batchnorm" else set())
        assert set(rows[0]["gradients"]) == groups
        assert rows[0]["gradients"]["hidden1"]["norm"] == 0.0 and rows[0]["gradients"]["out"]["norm"] > 0
        monitored = {"hidden1", "hidden2", "hidden3", "act1", "act2", "act3"} | (
            {"bn1", "bn2", "bn3"} if normalization == "batchnorm" else set())
        assert set(rows[0]["activations"]) == monitored
        assert rows[0]["updates"]["per_parameter"]["out.weight"]["log10_update_to_param_std"] is None  # std 0 at step 0
        assert rows[0]["nonfinite_gradients"] == {}
        rec = r.history[-1]
        assert monitored | {"per_unit_saturation"} == set(rec.activations)
        assert set(rec.activations["per_unit_saturation"]) == {"level1", "level2", "level3"}
        assert set(rec.output) == {"logit_mean", "logit_std", "mean_entropy", "mean_max_prob", "weight_norm",
                                   "bias_norm"}
        assert set(rec.levels) == {"level1", "level2", "level3"}
        if normalization == "batchnorm":
            assert set(rec.batchnorm) == {"level1", "level2", "level3"}
            assert rec.batchnorm["level3"]["num_calibrations"] == 2
        else:
            assert rec.batchnorm is None
        assert [x["epoch"] for x in read_jsonl(run.history_path)] == [0, 1]

    @pytest.mark.parametrize("normalization", NORMS)
    def test_three_nlls(self, small_vocab, small, normalization):
        r = train_wavenet(_model(small_vocab, normalization), small, small, training=_cfg(lr=0.5), seed=0)
        h = r.history[0]
        assert h.train_nll == h.dev_nll and h.running_train_nll != h.train_nll


# =========================================================== §15.6 generation
class TestGeneration:
    @pytest.mark.parametrize("normalization", NORMS)
    def test_context_shift_eos_and_max_length(self, small_vocab, small, normalization):
        m = _model(small_vocab, normalization)
        train_wavenet(m, small, small, training=_cfg(lr=0.5, max_epochs=15), seed=0)
        seen = []
        handle = m.register_forward_pre_hook(lambda _m, args: seen.extend(args[0].tolist()))
        try:
            for s in range(8):
                seen.clear()
                (sample,) = m.generate(1, max_length=20, generator=make_generator(s))
                if sample.finished and sample.text:
                    expected = make_context_windows([sample.text], small_vocab, context_length=8).contexts.tolist()
                    assert seen == expected
        finally:
            handle.remove()
        fresh = _model(small_vocab, normalization)
        with torch.no_grad():
            fresh.out.bias[0] = -1e4
        calls = [0]
        h = fresh.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        out = fresh.generate(3, max_length=5, generator=make_generator(0))
        h.remove()
        assert all(len(s.text) == 5 and not s.finished for s in out) and calls[0] == 6
        with torch.no_grad():
            fresh.out.bias[0] = 1e4
        assert fresh.generate(2, max_length=5, generator=make_generator(0)) == [WaveNetSample("", True)] * 2

    def test_batchnorm_generation_uses_inference_statistics(self, small_vocab):
        m = _generic(_model(small_vocab, "batchnorm"))
        _install(m)
        m.train()
        a = m.generate(1, max_length=6, generator=make_generator(3))
        assert m.training
        with torch.no_grad():
            for lvl in m.levels:
                m._bn(lvl).running_mean.fill_(40.0)
        assert m.generate(1, max_length=6, generator=make_generator(3)) == a
        logits = []
        h = m.out.register_forward_hook(lambda _m, _i, o: logits.append(o.detach().clone()))
        m.generate(1, max_length=1, generator=make_generator(3))
        with torch.no_grad():
            m.bn1.inference_mean.add_(1.0)
        m.generate(1, max_length=1, generator=make_generator(3))
        h.remove()
        assert not torch.allclose(logits[0], logits[-1])

    @pytest.mark.parametrize("normalization", NORMS)
    def test_reproducible_valid_and_state_unchanged(self, small_vocab, small, normalization):
        m = _model(small_vocab, normalization)
        train_wavenet(m, small, small, training=_cfg(lr=0.5, max_epochs=3), seed=0)
        before = _state(m)
        a = m.generate(10, max_length=12, generator=make_generator(9))
        assert a == m.generate(10, max_length=12, generator=make_generator(9))
        assert all("." not in s.text and set(s.text) <= set(small_vocab.chars) for s in a)
        _assert_same_state(before, _state(m))
        for bad in ({"num_samples": 0}, {"max_length": 0}):
            args = {"num_samples": 2, "max_length": 5, **bad}
            with pytest.raises(ValueError):
                m.generate(args.pop("num_samples"), **args)


# ========================================================== §15.7 persistence
@pytest.fixture
def interrupted(small_vocab, small, tmp_path, monkeypatch, request):
    normalization = getattr(request, "param", "none")
    cfg = _cfg(lr=0.4, max_epochs=4, lr_decay_epoch=2)
    model = _model(small_vocab, normalization)
    run = _run(tmp_path / "b", model.model_name)
    real = wn._learning_rate_at

    def interrupt(epoch, training):
        if epoch == 2:
            raise KeyboardInterrupt("simulated interruption")
        return real(epoch, training)

    with monkeypatch.context() as mp:
        mp.setattr(wn, "_learning_rate_at", interrupt)
        with pytest.raises(KeyboardInterrupt), run.track():
            train_wavenet(model, small, small, training=cfg, seed=11, run=run)
    return run, cfg, normalization


class TestPersistence:
    @pytest.mark.parametrize("normalization", NORMS)
    def test_contents_and_round_trip(self, small_vocab, small, tmp_path, normalization):
        m = _model(small_vocab, normalization)
        run = _run(tmp_path, m.model_name)
        cfg = _cfg(lr=0.3, max_epochs=2)
        r = train_wavenet(m, small, small, training=cfg, seed=5, run=run, config={"exp": 1})
        last = load_checkpoint(run.checkpoint_path("resume"), expected_kind="resume")
        assert last["config"] == run_config({"exp": 1}, m.config, cfg)
        assert last["config"]["model_name"] == m.model_name
        assert last["config"]["batchnorm"] == (BATCHNORM_RECORD if normalization == "batchnorm" else None)
        assert (last["epoch"], last["step"]) == (2, 4) and last["extra"]["best_epoch"] == r.best_epoch
        assert _rows(read_jsonl(run.history_path)) == _rows(r.history)
        loaded = WaveNetLanguageModel.from_checkpoint(run.checkpoint_path("final"), expected_vocab=small_vocab,
                                                      expected_model_name=m.model_name)
        assert loaded.model_name == m.model_name and loaded.config == m.config
        _assert_same_state(_state(m), _state(loaded))
        assert torch.equal(m.eval()(small.contexts), loaded.eval()(small.contexts))
        assert m.generate(4, max_length=8, generator=make_generator(1)) == \
            loaded.generate(4, max_length=8, generator=make_generator(1))
        WaveNetLanguageModel.from_checkpoint(run.checkpoint_path("best"))  # loads

    def test_write_order(self, small_vocab, small, tmp_path, monkeypatch):
        run, events = _run(tmp_path, "wavenet"), []
        real_save, real_append = wn.save_checkpoint, RunManager.append_history
        monkeypatch.setattr(wn, "save_checkpoint", lambda path, *, kind, **kw: events.append(
            (kind, kw.get("epoch"))) or real_save(path, kind=kind, **kw))
        monkeypatch.setattr(RunManager, "append_history", lambda self, rec: events.append(
            ("history", rec["epoch"])) or real_append(self, rec))
        r = train_wavenet(_model(small_vocab), small, _subset(small, 40), training=_cfg(lr=0.3, max_epochs=3),
                          seed=0, run=run)
        expected = []
        for h in r.history:
            if h.is_best:
                expected.append(("best", h.epoch + 1))
            expected += [("history", h.epoch), ("resume", h.epoch + 1)]
        assert events == expected + [("final", 3)]

    def test_identity_validation(self, small_vocab, small, tmp_path):
        runs = {}
        for normalization in NORMS:
            m = _model(small_vocab, normalization)
            runs[normalization] = _run(tmp_path / normalization, m.model_name)
            train_wavenet(m, small, small, training=_cfg(), seed=0, run=runs[normalization])
        bn_final, base_final = runs["batchnorm"].checkpoint_path("final"), runs["none"].checkpoint_path("final")
        with pytest.raises(IncompatibleCheckpointError, match="expected 'wavenet'"):
            WaveNetLanguageModel.from_checkpoint(bn_final, expected_model_name="wavenet")
        with pytest.raises(IncompatibleCheckpointError, match="expected 'wavenet_batchnorm'"):
            WaveNetLanguageModel.from_checkpoint(base_final, expected_model_name="wavenet_batchnorm")
        with pytest.raises(ValueError, match="expected_model_name must be one of"):
            WaveNetLanguageModel.from_checkpoint(base_final, expected_model_name="mlp")
        # stored identity inconsistent with stored normalisation, or with the batchnorm record
        bn_model, base_model = _model(small_vocab, "batchnorm"), _model(small_vocab)
        calibrate_batchnorm(bn_model, small, batch_size=64)
        cases = {
            "name_vs_norm": (bn_model, {"model_name": "wavenet", "model": asdict(bn_model.config),
                                        "batchnorm": BATCHNORM_RECORD}, "disagrees with stored normalization"),
            "record_vs_norm": (base_model, {"model_name": "wavenet", "model": asdict(base_model.config),
                                            "batchnorm": BATCHNORM_RECORD}, "batchnorm record"),
            "missing_name": (base_model, {"model": asdict(base_model.config), "batchnorm": None},
                             "stored model_name None"),
        }
        for key, (model, config, match) in cases.items():
            p = tmp_path / f"{key}.pt"
            save_checkpoint(p, kind="final", model=model, config=config, vocab=small_vocab.to_dict())
            with pytest.raises(IncompatibleCheckpointError, match=match):
                WaveNetLanguageModel.from_checkpoint(p)
        # resuming with the other variant: rejected by the run check and by the checkpoint check
        with pytest.raises(ValueError, match="belongs to 'wavenet'"):
            train_wavenet(_model(small_vocab, "batchnorm"), small, small, training=_cfg(), seed=0,
                          run=runs["none"], resume=True)
        ckpt = load_checkpoint(runs["none"].checkpoint_path("resume"))
        with pytest.raises(IncompatibleCheckpointError, match="model_name changed"):
            wn._check_resume_compatible(ckpt, model=_model(small_vocab, "batchnorm"), training=_cfg(), seed=0,
                                        num_examples=63)

    def test_from_checkpoint_rejections(self, small_vocab, small, tmp_path):
        good = run_config(None, WaveNetConfig(), _cfg())
        base = _model(small_vocab)
        uncal = _model(small_vocab, "batchnorm")
        bn_cfg = run_config(None, WaveNetConfig(normalization="batchnorm"), _cfg())
        half = _model(small_vocab, "batchnorm")
        half.bn1.set_inference_statistics(torch.zeros(68), torch.ones(68))
        cases = {
            "wrong_class": (nn.Linear(2, 2), good, small_vocab.to_dict(), "not a"),
            "no_vocab": (base, good, None, "vocabulary metadata"),
            "no_config": (base, None, small_vocab.to_dict(), "config"),
            "bad_config": (base, {**good, "model": {**good["model"], "context_length": 6}}, small_vocab.to_dict(),
                           "invalid model config"),
            "shape": (base, {**good, "model": {**good["model"], "hidden_dim": 32}}, small_vocab.to_dict(),
                      "do not match"),
            "uncalibrated": (uncal, bn_cfg, small_vocab.to_dict(), "not calibrated"),
            "unequal_levels": (half, bn_cfg, small_vocab.to_dict(), r"counts \[1, 0, 0\]"),
        }
        for key, (model, config, vocab, match) in cases.items():
            p = tmp_path / f"{key}.pt"
            save_checkpoint(p, kind="final", model=model, config=config, vocab=vocab)
            with pytest.raises(IncompatibleCheckpointError, match=match):
                WaveNetLanguageModel.from_checkpoint(p)
        with pytest.raises(IncompatibleCheckpointError, match="vocabulary mismatch"):
            p = tmp_path / "ok.pt"
            save_checkpoint(p, kind="final", model=base, config=good, vocab=small_vocab.to_dict())
            WaveNetLanguageModel.from_checkpoint(p, expected_vocab=Vocabulary.from_names(["xyz"]))

    @pytest.mark.parametrize("interrupted", NORMS, indirect=True)
    def test_bit_exact_resume(self, small_vocab, small, tmp_path, interrupted):
        run_b, cfg, normalization = interrupted
        assert load_checkpoint(run_b.checkpoint_path("resume"))["epoch"] == 2
        a = _model(small_vocab, normalization)
        run_a = _run(tmp_path / "a", a.model_name)
        ra = train_wavenet(a, small, small, training=cfg, seed=11, run=run_a)
        b = _model(small_vocab, normalization, seed=999)
        rb = train_wavenet(b, small, small, training=cfg, seed=11, run=run_b, resume=True)
        _assert_same_state(_state(a), _state(b))  # parameters and every buffer
        assert _rows(ra.history) == _rows(rb.history) and len(rb.history) == 4
        assert (ra.best_epoch, ra.best_dev_nll) == (rb.best_epoch, rb.best_dev_nll)

    def test_history_truncated_after_crash(self, small_vocab, small, tmp_path, monkeypatch):
        cfg = _cfg(lr=0.4, max_epochs=4)
        reference = train_wavenet(_model(small_vocab), small, small, training=cfg, seed=2)
        run = _run(tmp_path, "wavenet")
        real = wn.save_checkpoint

        def crash(path, *, kind, **kw):
            if kind == "resume" and kw["epoch"] == 2:
                raise RuntimeError("simulated crash after append_history")
            return real(path, kind=kind, **kw)

        with monkeypatch.context() as mp:
            mp.setattr(wn, "save_checkpoint", crash)
            with pytest.raises(RuntimeError, match="simulated crash"):
                train_wavenet(_model(small_vocab), small, small, training=cfg, seed=2, run=run)
        assert [r["epoch"] for r in read_jsonl(run.history_path)] == [0, 1]
        r = train_wavenet(_model(small_vocab), small, small, training=cfg, seed=2, run=run, resume=True)
        assert _rows(read_jsonl(run.history_path)) == _rows(reference.history) == _rows(r.history)

    @pytest.mark.parametrize(
        "change, match",
        [("normalization", "model_name"), ("model", "model config"), ("lr", "training.lr"),
         ("max_epochs", "training.max_epochs"), ("batch_size", "training.batch_size"),
         ("lr_decay_epoch", "training.lr_decay_epoch"), ("lr_decay_factor", "training.lr_decay_factor"),
         ("seed", "seed"), ("vocabulary", "vocabulary"), ("dataset_size", "dataset size")],
    )
    def test_incompatible_resume(self, small_vocab, small, interrupted, change, match):
        run, cfg, _ = interrupted
        vocab, config, train, seed = small_vocab, WaveNetConfig(), small, 11
        if change == "normalization":
            ckpt = load_checkpoint(run.checkpoint_path("resume"))
            with pytest.raises(IncompatibleCheckpointError, match=match):
                wn._check_resume_compatible(ckpt, model=_model(vocab, "batchnorm"), training=cfg, seed=seed,
                                            num_examples=63)
            return
        if change == "model":
            config = WaveNetConfig(hidden_dim=32)
        elif change == "seed":
            seed = 12
        elif change == "vocabulary":
            vocab = Vocabulary.from_names(SMALL + ["xyz"])
            train = make_context_windows(SMALL, vocab, context_length=8)
        elif change == "dataset_size":
            train = _subset(small, 60)
        elif change == "lr_decay_epoch":
            cfg = replace(cfg, lr_decay_epoch=1)
        else:
            cfg = replace(cfg, **{change: {"lr": 0.5, "max_epochs": 6, "batch_size": 16,
                                           "lr_decay_factor": 0.5}[change]})
        model = WaveNetLanguageModel(vocab, config, generator=make_generator(0))
        with pytest.raises(IncompatibleCheckpointError, match=match):
            train_wavenet(model, train, small, training=cfg, seed=seed, run=run, resume=True)

    def test_original_budget_is_kept(self, small_vocab, small, interrupted):
        run, cfg, _ = interrupted
        with pytest.raises(IncompatibleCheckpointError, match="max_epochs"):
            train_wavenet(_model(small_vocab), small, small, training=replace(cfg, max_epochs=2, lr_decay_epoch=1),
                          seed=11, run=run, resume=True)
        r = train_wavenet(_model(small_vocab), small, small, training=cfg, seed=11, run=run, resume=True)
        assert [h.epoch for h in r.history] == [0, 1, 2, 3]


# ===================================================== §15.8 devices and data
@pytest.mark.parametrize("device", DEVICES)
class TestDevices:
    @pytest.mark.parametrize("normalization", NORMS)
    def test_training_evaluation_generation(self, device, small_vocab, small, normalization):
        cfg = _cfg(lr=0.3, max_epochs=2)
        cpu = _model(small_vocab, normalization)
        train_wavenet(cpu, small, small, training=cfg, seed=0)
        m = _model(small_vocab, normalization).to(device)
        r = train_wavenet(m, small, small, training=cfg, seed=0)
        assert m.device.type == device
        for (n, a), b in zip(m.state_dict().items(), cpu.state_dict().values()):
            if a.is_floating_point():
                assert torch.allclose(a.cpu(), b, atol=1e-5), n
        assert r.history[-1].dev_nll == pytest.approx(evaluate_nll(cpu, small, batch_size=64).mean_nll, abs=1e-5)
        assert len(m.generate(3, max_length=8, generator=make_generator(0))) == 3
        assert set(per_unit_saturation(m, small, batch_size=16)) == {1, 2, 3}

    def test_float64_helpers_with_device_tensors(self, device, small_vocab):
        rows = torch.randn(37, 68, generator=make_generator(4))
        n, mean, m2 = wn._chunk_moments(rows.to(device))
        n_ref, mean_ref, m2_ref = wn._chunk_moments(rows)
        assert n == n_ref and mean.device.type == "cpu" and mean.dtype == torch.float64
        assert torch.equal(mean, mean_ref) and torch.equal(m2, m2_ref)
        m = _prepare_eval(_model(small_vocab, "batchnorm"))
        cpu_summary = (wn._batchnorm_summary(m), wn._levels_summary(m))
        m = m.to(device)
        assert (wn._batchnorm_summary(m), wn._levels_summary(m)) == cpu_summary

    def test_device_mismatch_and_checkpoint(self, device, small_vocab, small, tmp_path):
        m = _model(small_vocab, "batchnorm").to(device)
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(2, 8, dtype=torch.long, device="meta"))
        run = _run(tmp_path, m.model_name)
        train_wavenet(m, small, small, training=_cfg(lr=0.3), seed=0, run=run)
        loaded = WaveNetLanguageModel.from_checkpoint(run.checkpoint_path("final"), map_location=device)
        assert loaded.device.type == device and loaded.bn1.inference_mean.device.type == device
        x = small.contexts[:8].to(device)
        assert torch.allclose(loaded.eval()(x), m.eval()(x), atol=1e-6)


@pytest.mark.skipif(not DEFAULT_NAMES_PATH.is_file(), reason="names dataset not present")
def test_real_dataset():
    splits = split_names(load_names(DEFAULT_NAMES_PATH), fractions=(0.8, 0.1, 0.1), seed=42, duplicate_policy="group")
    vocab = Vocabulary.from_names(splits.train)
    train = make_context_windows(splits.train, vocab, context_length=8)
    assert train.num_examples == 182_666
    assert min(EpochSampler(train.num_examples, 32, seed=0).batch_sizes()) == 10
    for normalization, count in (("none", 21_989), ("batchnorm", 22_397)):
        m = WaveNetLanguageModel(vocab, WaveNetConfig(normalization=normalization),
                                 generator=make_generator(derive_seed(0, "init")))
        assert count_parameters(m).total == count
        assert evaluate_nll(m, train, batch_size=4096).mean_nll == pytest.approx(LN27, abs=1e-6)
    part = _subset(train, 20_000)
    m = _generic(m)
    calibrate_batchnorm(m, part, batch_size=4096)
    a1 = m._level_preactivations(part.contexts, level=1).detach().double().reshape(-1, 68)
    assert torch.allclose(m.bn1.inference_var.double(), a1.var(0, unbiased=False), rtol=1e-6, atol=0)

"""Independent reference tests for WaveNet (specification §14).

The model is observed only through its public ``forward`` and forward hooks on named
submodules; the reference (``wavenet_reference.py``) receives cloned parameter tensors only.
High-precision comparisons run on CPU in float64.
"""

from __future__ import annotations

import ast
import copy
import math
import string
from pathlib import Path

import pytest
import torch

import tests.models.hierarchical.wavenet_reference as ref
from models.hierarchical.wavenet import (
    WaveNetConfig,
    WaveNetLanguageModel,
    calibrate_batchnorm,
    merge_pairs,
)
from utils.data import ContextWindows, make_context_windows
from utils.metrics import count_parameters, token_nll
from utils.reproducibility import make_generator
from utils.vocabulary import Vocabulary

NORMS = ["none", "batchnorm"]
SMALL = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "liam", "noah", "anna", "anna", "eva", "bo"]
TINY = ["ab", "ac", "ab"]
F64 = {"rtol": 0.0, "atol": 1e-12}
GRAD64 = {"rtol": 0.0, "atol": 1e-10}


@pytest.fixture
def v27() -> Vocabulary:
    return Vocabulary.from_names(list(string.ascii_lowercase))


def _model(vocab, normalization="none", seed=0, **kw) -> WaveNetLanguageModel:
    return WaveNetLanguageModel(vocab, WaveNetConfig(normalization=normalization, **kw), generator=make_generator(seed))


def _generic(model: WaveNetLanguageModel, seed: int = 11) -> WaveNetLanguageModel:
    """Give the zero-initialised output layer and the BatchNorm affine parameters generic values."""
    g = make_generator(seed)
    with torch.no_grad():
        for p in (model.out.weight, model.out.bias):
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.7)
        if model.uses_batchnorm:
            for level in model.levels:
                bn = getattr(model, f"bn{level}")
                bn.weight.copy_(1.0 + 0.3 * torch.randn(bn.weight.shape, generator=g, dtype=bn.weight.dtype))
                bn.bias.copy_(0.3 * torch.randn(bn.bias.shape, generator=g, dtype=bn.bias.dtype))
    return model


def _install_stats(model: WaveNetLanguageModel, seed: int = 5) -> dict:
    g = make_generator(seed)
    stats = {}
    for level in model.levels:
        H = model.config.hidden_dim
        dtype = model.out.weight.dtype
        m = torch.randn(H, generator=g, dtype=dtype)
        v = torch.rand(H, generator=g, dtype=dtype) + 0.4
        getattr(model, f"bn{level}").set_inference_statistics(m, v)
        stats[level] = (m.clone(), v.clone())
    return stats


def _params(model: WaveNetLanguageModel, *, requires_grad: bool = False) -> dict:
    """Cloned float64 parameter (and buffer) tensors: the only thing the reference ever sees."""
    out = {}
    for k, t in model.state_dict().items():
        c = t.detach().clone()
        if c.is_floating_point():
            c = c.double().requires_grad_(requires_grad and k in dict(model.named_parameters()))
        out[k] = c
    return out


def _capture(model: WaveNetLanguageModel, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Run the public forward with hooks; returns named intermediates and the logits."""
    seen: dict = {}

    def record(output_key=None, input_key=None):
        def hook(_module, inputs, output):  # returns None: never replaces the module output
            if input_key is not None:
                seen[input_key] = inputs[0]
            if output_key is not None:
                seen[output_key] = output
        return hook

    handles = [model.emb.register_forward_hook(record("e"))]
    for level in model.levels:
        handles.append(getattr(model, f"hidden{level}").register_forward_hook(record(f"a{level}", f"m{level}")))
        if model.uses_batchnorm:
            handles.append(getattr(model, f"bn{level}").register_forward_hook(record(f"y{level}")))
        handles.append(getattr(model, f"act{level}").register_forward_hook(record(f"h{level}")))
    handles.append(model.out.register_forward_hook(record(None, "flat")))
    try:
        logits = model(x)
    finally:
        for h in handles:
            h.remove()
    if not model.uses_batchnorm:
        for level in model.levels:
            seen[f"y{level}"] = seen[f"a{level}"]
    return seen, logits


def _windows(names, vocab, k=8) -> ContextWindows:
    return make_context_windows(names, vocab, context_length=k)


# ============================================================ independence
def test_reference_module_imports_only_torch_math_typing():
    tree = ast.parse(Path(ref.__file__).read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert modules <= {"__future__", "torch", "math", "typing"}, modules
    source = Path(ref.__file__).read_text()
    assert "merge_pairs" not in source.replace("including ``merge_pairs``", "")
    assert "wavenet import" not in source and "models." not in source


# ======================================================== §14.1 A. merging
class TestMergeAgainstReference:
    @staticmethod
    def coded(B=3, T=8, C=4):
        b, t, c = torch.meshgrid(torch.arange(B), torch.arange(T), torch.arange(C), indexing="ij")
        return (1000 * b + 10 * t + c).double()

    def test_index_coded_exact(self):
        x = self.coded()
        out = merge_pairs(x)
        assert out.shape == (3, 4, 8)
        assert torch.equal(out, ref.ref_merge(x))
        assert out[1, 2, 5].item() == x[1, 5, 1].item() == 1051.0
        assert out[2, 3, 0].item() == x[2, 6, 0].item() == 2060.0
        assert torch.equal(out, torch.cat([x[:, ::2], x[:, 1::2]], dim=-1))  # second formulation

    @pytest.mark.parametrize("kind", ["transposed", "sliced", "expanded"])
    def test_non_contiguous_inputs(self, kind):
        x = self.coded()
        if kind == "transposed":
            y = x.transpose(0, 1).contiguous().transpose(0, 1)
        elif kind == "sliced":
            big = self.coded(C=8)
            y, x = big[:, :, ::2], big[:, :, ::2].contiguous()
        else:
            x = self.coded(B=1).expand(3, 8, 4).contiguous()
            y = self.coded(B=1).expand(3, 8, 4)
        assert not y.is_contiguous()
        assert torch.equal(merge_pairs(y), merge_pairs(x)) and torch.equal(merge_pairs(y), ref.ref_merge(x))

    @pytest.mark.parametrize("T", [0, 1, 3, 7])
    def test_invalid_lengths(self, T):
        with pytest.raises(ValueError, match="even sequence length"):
            merge_pairs(torch.zeros(2, T, 3))

    def test_invalid_rank_type_and_empty_batch(self):
        for shape in ((8, 4), (2, 8, 4, 1)):
            with pytest.raises(ValueError, match=r"\[B, T, C\]"):
                merge_pairs(torch.zeros(shape))
        with pytest.raises(TypeError):
            merge_pairs([[1.0, 2.0]])
        with pytest.raises(ValueError, match="C >= 1"):
            merge_pairs(torch.zeros(2, 4, 0))
        assert merge_pairs(torch.zeros(0, 8, 3)).shape == (0, 4, 6)

    def test_gradients(self):
        x = torch.randn(2, 8, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(merge_pairs, (x,))
        w = torch.randn(2, 4, 6, dtype=torch.float64)
        (merge_pairs(x) * w).sum().backward()
        # inverse arrangement: position 2j receives the first half of group j, 2j+1 the second half
        expected = torch.stack([w[:, t // 2, (t % 2) * 3:(t % 2) * 3 + 3] for t in range(8)], dim=1)
        assert torch.equal(x.grad, expected)

    def test_repeated_levels_keep_order(self):
        x = torch.arange(8, dtype=torch.float64).view(1, 8, 1).expand(2, 8, 1)
        out = merge_pairs(merge_pairs(merge_pairs(x)))
        assert out.shape == (2, 1, 8) and out[0, 0].tolist() == list(range(8))


# =============================================== §14.2 B. hierarchical forward
class TestForwardAgainstReference:
    X = torch.tensor([[0] * 8, [0, 0, 0, 0, 0, 0, 5, 9], [1, 2, 3, 4, 5, 6, 7, 8],
                      [26, 3, 3, 1, 0, 14, 2, 2], [9, 9, 9, 9, 9, 9, 9, 9], [4, 0, 7, 0, 7, 0, 4, 1]])

    def _compare(self, model, mode, stats=None):
        params = _params(model)
        seen, logits = _capture(model, self.X)
        expected = ref.ref_forward(params, self.X, num_levels=3, normalization=model.config.normalization,
                                   mode=mode, stats=stats)
        for key in ["e"] + [f"{p}{lvl}" for lvl in (1, 2, 3) for p in "mayh"] + ["flat"]:
            assert seen[key].shape == expected[key].shape, key
            assert torch.allclose(seen[key], expected[key], **F64), key
        assert torch.allclose(logits, expected["logits"], **F64)

    def test_baseline(self, v27):
        m = _generic(_model(v27)).double()
        for mode in ("train", "eval"):
            m.train(mode == "train")
            self._compare(m, "eval")  # no mode dependence without BatchNorm

    def test_batchnorm_train_mode(self, v27):
        m = _generic(_model(v27, "batchnorm")).double().train()
        self._compare(m, "train")

    def test_batchnorm_eval_mode_ignores_ema(self, v27):
        m = _generic(_model(v27, "batchnorm")).double()
        stats = _install_stats(m)
        with torch.no_grad():
            for level in m.levels:
                getattr(m, f"bn{level}").running_mean.fill_(77.0)
                getattr(m, f"bn{level}").running_var.fill_(1e-4)
        m.eval()
        self._compare(m, "eval", stats)

    def test_hand_computed_case(self, v27):
        m = _model(v27, context_length=4, embedding_dim=1, hidden_dim=1).double()
        with torch.no_grad():
            m.emb.weight.copy_(torch.arange(27, dtype=torch.float64).view(27, 1))  # E[t] = t
            m.hidden1.weight.copy_(torch.tensor([[1.0, 2.0]]))
            m.hidden2.weight.copy_(torch.tensor([[1.0, -1.0]]))
            m.out.weight.copy_(torch.arange(27, dtype=torch.float64).view(27, 1) * 0.5)
            m.out.bias.copy_(torch.full((27,), -0.25, dtype=torch.float64))
        x = torch.tensor([[0, 0, 1, 2]])
        # level 1: pairs (0, 0) → 0;  (1, 2) → 1·1 + 2·2 = 5;  level 2: tanh(0)·1 + tanh(5)·(−1)
        h2 = math.tanh(math.tanh(0.0) - math.tanh(5.0))
        hand = torch.tensor([[0.5 * t * h2 - 0.25 for t in range(27)]], dtype=torch.float64)
        assert torch.allclose(m(x), hand, **F64)

    @pytest.mark.parametrize("normalization", NORMS)
    def test_float32_secondary(self, v27, normalization):
        m = _generic(_model(v27, normalization))
        if normalization == "batchnorm":
            m.train()
        params = _params(m)
        logits = m(self.X)
        expected = ref.ref_forward(params, self.X, num_levels=3, normalization=normalization,
                                   mode="train" if normalization == "batchnorm" else "eval")
        assert torch.allclose(logits.double(), expected["logits"], rtol=0, atol=1e-5)


# =========================================== §14.3 C. convolutional equivalence
class TestConvolutionalEquivalence:
    X = TestForwardAgainstReference.X

    def test_kernel_mapping(self):
        W = torch.arange(12, dtype=torch.float64).view(2, 6)  # H = 2, C = 3
        K = ref.conv_kernel(W)
        assert K.shape == (2, 3, 2)
        assert K[1, 2, 0] == W[1, 2] and K[1, 2, 1] == W[1, 5]

    @pytest.mark.parametrize("normalization, mode", [("none", "eval"), ("batchnorm", "train"),
                                                     ("batchnorm", "eval")])
    def test_strided_conv_matches_model(self, v27, normalization, mode):
        m = _generic(_model(v27, normalization)).double()
        if normalization == "batchnorm" and mode == "eval":
            _install_stats(m)
        m.train(mode == "train")
        params = _params(m)
        seen, logits = _capture(m, self.X)
        conv = ref.ref_conv_forward(params, self.X, num_levels=3, normalization=normalization, mode=mode)
        for key in [f"{p}{lvl}" for lvl in (1, 2, 3) for p in "ayh"]:
            assert torch.allclose(seen[key], conv[key], **F64), key
        assert torch.allclose(logits, conv["logits"], **F64)

    def test_dilated_causal_conv_matches_baseline(self, v27):
        m = _generic(_model(v27)).double()
        params = _params(m)
        assert torch.allclose(m(self.X), ref.ref_dilated_forward(params, self.X, num_levels=3), **F64)


# =============================================== §14.4 D. mathematical checks
class TestMathAgainstReference:
    X = TestForwardAgainstReference.X
    Y = torch.tensor([1, 0, 9, 4, 0, 26])

    @pytest.mark.parametrize("normalization", NORMS)
    def test_receptive_fields_match_reference(self, v27, normalization):
        m = _generic(_model(v27, normalization)).double()
        if normalization == "batchnorm":
            _install_stats(m)
        m.eval()
        params = _params(m)
        x = torch.randint(1, 27, (1, 8), generator=make_generator(3))
        e_ref = ref.ref_embed(params, x).detach().requires_grad_(True)
        for level in (1, 2, 3):
            for group in range(1, 2 ** (3 - level) + 1):
                # reference graph from the leaf embeddings
                h = e_ref
                for lvl in range(1, level + 1):
                    a = ref.ref_merge(h) @ params[f"hidden{lvl}.weight"].T
                    if normalization == "batchnorm":
                        mu, var = params[f"bn{lvl}.inference_mean"], params[f"bn{lvl}.inference_var"]
                        a = params[f"bn{lvl}.weight"] * (a - mu) / torch.sqrt(var + ref.EPS) + params[f"bn{lvl}.bias"]
                    h = torch.tanh(a)
                (g_ref,) = torch.autograd.grad(h[0, group - 1].sum(), e_ref)
                # the model, via hooks
                captured, out = {}, {}

                def keep(_m, _i, o, captured=captured):
                    o.retain_grad()
                    captured["e"] = o

                def keep_h(_m, _i, o, out=out):
                    out["h"] = o

                handle = m.emb.register_forward_hook(keep)
                h2 = getattr(m, f"act{level}").register_forward_hook(keep_h)
                try:
                    m(x)
                finally:
                    handle.remove()
                    h2.remove()
                out["h"][0, group - 1].sum().backward()
                expected = ref.expected_field(level, group)
                assert ref.dependence_positions(g_ref) == expected
                assert ref.dependence_positions(captured["e"].grad) == expected

    @pytest.mark.parametrize("normalization", NORMS)
    def test_loss(self, v27, normalization):
        m = _generic(_model(v27, normalization)).double().train()
        params = _params(m)
        logits = m(self.X)
        mode = "train" if normalization == "batchnorm" else "eval"
        expected = ref.ref_forward(params, self.X, num_levels=3, normalization=normalization, mode=mode)["logits"]
        assert torch.allclose(token_nll(logits, self.Y, reduction="mean"), ref.ref_loss(expected, self.Y), **F64)

    @pytest.mark.parametrize("normalization", NORMS)
    def test_gradients(self, v27, normalization):
        m = _generic(_model(v27, normalization)).double().train()
        params = _params(m, requires_grad=True)
        m.zero_grad()
        token_nll(m(self.X), self.Y, reduction="mean").backward()
        mode = "train" if normalization == "batchnorm" else "eval"
        ref.ref_loss(ref.ref_forward(params, self.X, num_levels=3, normalization=normalization,
                                     mode=mode)["logits"], self.Y).backward()
        for name, p in m.named_parameters():
            assert torch.allclose(p.grad, params[name].grad, **GRAD64), name
        assert m.emb.weight.grad[0].abs().sum() > 0  # the all-BOS window contributes 8 times to E[0]

    @pytest.mark.parametrize("normalization", NORMS)
    @pytest.mark.parametrize("kw", [{}, {"context_length": 16, "hidden_dim": 32}, {"context_length": 4,
                                                                                    "embedding_dim": 3}])
    def test_parameter_counts(self, v27, normalization, kw):
        m = _model(v27, normalization, **kw)
        c = m.config
        assert count_parameters(m).total == ref.ref_parameter_count(27, c.embedding_dim, c.hidden_dim,
                                                                     c.num_levels, normalization)
        if not kw:
            assert count_parameters(m).total == {"none": 21_989, "batchnorm": 22_397}[normalization]


# ============================================== §14.5 E. BatchNorm variant
class TestBatchNormAgainstReference:
    X = TestForwardAgainstReference.X

    def test_normalisation_equations_and_layout(self, v27):
        m = _generic(_model(v27, "batchnorm")).double().train()
        seen, _ = _capture(m, self.X)
        for level in (1, 2, 3):
            a, y = seen[f"a{level}"], seen[f"y{level}"]
            gamma, beta = getattr(m, f"bn{level}").weight.detach(), getattr(m, f"bn{level}").bias.detach()
            mean = a.mean(dim=(0, 1))
            var = ((a - mean) ** 2).mean(dim=(0, 1))
            manual = gamma * (a - mean) / torch.sqrt(var + ref.EPS) + beta
            assert torch.allclose(y, manual, **F64)
            primitive = torch.nn.functional.batch_norm(a.permute(0, 2, 1), None, None, gamma, beta,
                                                       training=True, eps=ref.EPS).permute(0, 2, 1)
            assert torch.allclose(y, primitive, **F64)

    def test_train_vs_eval_and_ema_update(self, v27):
        m = _generic(_model(v27, "batchnorm")).double().train()
        seen, _ = _capture(m, self.X)
        for level in (1, 2, 3):
            bn = getattr(m, f"bn{level}")
            rows = seen[f"a{level}"].reshape(-1, 68)
            assert torch.allclose(bn.running_mean, 0.1 * rows.mean(0), **F64)
            assert torch.allclose(bn.running_var, 0.9 + 0.1 * rows.var(0, unbiased=True), **F64)
        with torch.no_grad():  # also under no_grad
            m(self.X)
        assert all(int(getattr(m, f"bn{lvl}").num_batches_tracked) == 2 for lvl in (1, 2, 3))
        # train mode: batch-dependent; eval mode: batch-independent
        a = m(self.X[:3])[:2]
        b = m(torch.cat([self.X[:2], self.X[4:6]]))[:2]
        assert not torch.allclose(a, b)
        _install_stats(m)
        m.eval()
        full = m(self.X)
        for i in range(len(self.X)):
            assert torch.allclose(m(self.X[i:i + 1]), full[i:i + 1], **F64)

    def test_calibration_matches_sequential_reference(self, v27):
        vocab = Vocabulary.from_names(SMALL)
        w = _windows(SMALL, vocab)
        m = _generic(_model(vocab, "batchnorm")).double()
        calibrate_batchnorm(m, w, batch_size=10)
        params = _params(m)
        expected = ref.ref_calibrate(params, w.contexts, num_levels=3)
        for level in (1, 2, 3):
            bn = getattr(m, f"bn{level}")
            assert torch.allclose(bn.inference_mean, expected[level][0], **F64)
            assert torch.allclose(bn.inference_var, expected[level][1], **F64)
        # a wrong single-pass calibration (lower levels with placeholder 0 / 1 statistics) differs at level >= 2
        h = ref.ref_embed(params, w.contexts)
        for level in (1, 2):
            a = ref.ref_merge(h) @ params[f"hidden{level}.weight"].T
            h = torch.tanh(params[f"bn{level}.weight"] * a / math.sqrt(1 + ref.EPS) + params[f"bn{level}.bias"])
        wrong_level3 = (ref.ref_merge(h) @ params["hidden3.weight"].T).reshape(-1, 68).mean(0)
        assert not torch.allclose(wrong_level3, expected[3][0], atol=1e-6)

    def test_full_batch_consistency(self):
        vocab = Vocabulary.from_names(TINY)
        w = _windows(TINY, vocab)
        for dtype, tol in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
            m = _generic(_model(vocab, "batchnorm")).to(dtype)
            z_train = copy.deepcopy(m).train()(w.contexts)
            calibrate_batchnorm(m, w, batch_size=4)
            assert torch.allclose(m.eval()(w.contexts), z_train, rtol=0, atol=tol)

    def test_per_level_summaries_recomputed(self, v27):
        from models.hierarchical.wavenet import _batchnorm_summary  # diagnostic under test, not used by the reference

        m = _generic(_model(v27, "batchnorm")).double()
        _install_stats(m)
        m.train()
        m(self.X)  # moves the EMA away from the inference statistics
        summary = _batchnorm_summary(m)
        params = _params(m)
        for level in (1, 2, 3):
            im, iv = params[f"bn{level}.inference_mean"], params[f"bn{level}.inference_var"]
            rm, rv = params[f"bn{level}.running_mean"], params[f"bn{level}.running_var"]
            z = ((rm - im) / torch.sqrt(iv + ref.EPS)).abs()
            r = torch.log((rv + ref.EPS) / (iv + ref.EPS)).abs()
            s = summary[f"level{level}"]
            assert s["ema_vs_inference"]["max_abs_z"] == pytest.approx(z.max().item(), rel=1e-12)
            assert s["ema_vs_inference"]["median_abs_log_var_ratio"] == pytest.approx(r.median().item(), rel=1e-12)
            assert s["gamma"]["mean"] == pytest.approx(params[f"bn{level}.weight"].mean().item(), rel=1e-12)
            assert s["num_calibrations"] == 1 and s["num_batches_tracked"] == 1

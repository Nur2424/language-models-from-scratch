"""Tests for models/feedforward/neural_bigram.py, following the specification (v0.6) §13–§14.

Reference fixture (spec §13)
----------------------------
names = ["a", "ab"], ids . = 0, a = 1, b = 2, V = 3, M = 5
pairs (prev → next): . → a, a → ., . → a, a → b, b → .
C = [[0,2,0],[1,0,1],[1,0,0]]   N = [2, 2, 1]   q = [[0,1,0],[½,0,½],[1,0,0]]
MLE training NLL = 2 ln 2 / 5 = 0.277259
Smoothness: ℓ ≤ max_i N_i/(2M) = 0.2 → J decreases monotonically for η < 2/(0.2 + λ).

Only these tests use the statistical bigram (``count_transitions`` and
``BigramLanguageModel``), as an independent oracle. That dependency is approved
for tests only (spec §18).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.feedforward.neural_bigram import (
    IncompatibleCheckpointError,
    NeuralBigramConfig,
    NeuralBigramLanguageModel,
    NeuralBigramSample,
    TrainingResult,
    make_bigram_pairs,
    train_full_batch,
)
from models.statistical.bigram import BigramLanguageModel, count_transitions  # test-only oracle
from utils.checkpoint import (
    CheckpointError,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from utils.config import ConfigError, default_config_paths, load_config
from utils.data import VocabularyCoverageError, load_names, split_names
from utils.device import is_mps_available
from utils.metrics import NLLAccumulator, count_parameters, token_nll
from utils.paths import DEFAULT_NAMES_PATH
from utils.reproducibility import capture_rng_state, make_generator
from utils.vocabulary import IGNORE_INDEX, Vocabulary

LN2, LN3 = math.log(2.0), math.log(3.0)
MLE_TINY = 2 * LN2 / 5
DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("mps", id="mps", marks=pytest.mark.skipif(not is_mps_available(), reason="MPS unavailable")),
]
SMALL_NAMES = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "liam", "noah", "oliver",
               "elijah", "lucas", "mason", "logan", "anna", "anna", "eva", "zoe", "quinn"]


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def tiny_vocab() -> Vocabulary:
    return Vocabulary.from_names(["a", "ab"])


@pytest.fixture
def tiny_pairs(tiny_vocab):
    return make_bigram_pairs(["a", "ab"], tiny_vocab)


@pytest.fixture
def tiny_counts(tiny_vocab) -> torch.Tensor:
    return count_transitions(["a", "ab"], tiny_vocab).double()


def _model(vocab, W: torch.Tensor | None = None) -> NeuralBigramLanguageModel:
    m = NeuralBigramLanguageModel(vocab)
    if W is not None:
        with torch.no_grad():
            m.transition_logits.copy_(W)
    return m


def _data_loss(model, x, y) -> torch.Tensor:
    """L_data with autograd graph; convert with .item() when a number is needed."""
    return token_nll(model(x), y, reduction="mean")


def _item(t: torch.Tensor) -> float:
    return t.detach().item()


def _objective(model, x, y, lam: float) -> float:
    with torch.no_grad():
        return _item(_data_loss(model, x, y)) + 0.5 * lam * float((model.transition_logits.double() ** 2).sum())


def _random_W(V: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(V, V, generator=torch.Generator().manual_seed(seed))


# =========================================================== 13.1 general
class TestConstructionAndConfig:
    def test_construction(self, tiny_vocab):
        m = NeuralBigramLanguageModel(tiny_vocab)
        assert m.num_classes == 3 and m.device == torch.device("cpu")
        assert "bias=False" in repr(m)
        assert isinstance(NeuralBigramLanguageModel.from_config(tiny_vocab, NeuralBigramConfig()), NeuralBigramLanguageModel)

    def test_invalid_construction(self, tiny_vocab):
        with pytest.raises(TypeError, match="Vocabulary"):
            NeuralBigramLanguageModel({".": 0, "a": 1})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="NeuralBigramConfig"):
            NeuralBigramLanguageModel.from_config(tiny_vocab, {})  # type: ignore[arg-type]

    def test_yaml_config(self):
        cfg = load_config(default_config_paths("neural_bigram"), model_config_cls=NeuralBigramConfig)
        assert cfg.model.name == "neural_bigram" and cfg.model.params == NeuralBigramConfig()
        for stray in ("model.params.hidden_size=8", "model.params.bias=true", "model.params.init_std=1.0"):
            with pytest.raises(ConfigError, match="unknown config key"):
                load_config(default_config_paths("neural_bigram"), overrides=[stray], model_config_cls=NeuralBigramConfig)

    def test_parameters_exactly_one_table_no_bias(self):
        v = Vocabulary(chars=tuple("abcdefghijklmnopqrstuvwxyz"))      # V = 27, as in the names data
        m = NeuralBigramLanguageModel(v)
        named = dict(m.named_parameters())
        assert list(named) == ["transition_logits"]
        assert named["transition_logits"].shape == (27, 27) and named["transition_logits"].dtype == torch.float32
        counts = count_parameters(m)
        assert counts.total == counts.trainable == 27 * 27 == 729
        assert len(list(m.buffers())) == 0

    def test_zero_initialisation_and_initial_loss(self, tiny_vocab, tiny_pairs):
        m = NeuralBigramLanguageModel(tiny_vocab)
        assert torch.equal(m.transition_logits.detach(), torch.zeros(3, 3))
        x, y = tiny_pairs
        assert _item(_data_loss(m, x, y)) == pytest.approx(LN3, abs=1e-6)
        v27 = Vocabulary(chars=tuple("abcdefghijklmnopqrstuvwxyz"))
        x27 = torch.arange(27)
        assert _item(token_nll(NeuralBigramLanguageModel(v27)(x27), x27, reduction="mean")) == pytest.approx(math.log(27), abs=1e-6)

    def test_module_does_not_import_statistical_bigram(self):
        source = Path(__file__).resolve().parents[3] / "models" / "feedforward" / "neural_bigram.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
                  [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(mod and mod.startswith("models.statistical") for mod in modules)
        banned = {"Embedding", "Linear"}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not (banned & attrs), "nn.Embedding / nn.Linear are prohibited (spec §6.2)"


class TestForwardContract:
    def test_shapes_and_dtype(self, tiny_vocab):
        m = _model(tiny_vocab, _random_W(3))
        assert m(torch.tensor([0, 2, 1])).shape == (3, 3)
        assert m(torch.tensor([[0, 1], [2, 2]])).shape == (2, 2, 3)
        empty = m(torch.zeros(0, dtype=torch.long))
        assert empty.shape == (0, 3) and empty.dtype == torch.float32

    def test_logits_are_rows_of_W(self, tiny_vocab):
        W = _random_W(3)
        out = _model(tiny_vocab, W)(torch.tensor([2, 0, 2]))
        assert torch.equal(out.detach(), W[[2, 0, 2]])

    def test_padding_is_rejected(self, tiny_vocab):
        m = NeuralBigramLanguageModel(tiny_vocab)
        with pytest.raises(ValueError, match="padding must not reach the model"):
            m(torch.tensor([0, tiny_vocab.pad_id]))

    @pytest.mark.parametrize("ids", [[-1], [4], [0, 99]])
    def test_out_of_range_rejected(self, tiny_vocab, ids):
        with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
            NeuralBigramLanguageModel(tiny_vocab)(torch.tensor(ids))

    def test_dtype_type_and_device_validation(self, tiny_vocab):
        m = NeuralBigramLanguageModel(tiny_vocab)
        with pytest.raises(TypeError, match="int64"):
            m(torch.tensor([0.0, 1.0]))
        with pytest.raises(TypeError, match="int64"):
            m(torch.tensor([0, 1], dtype=torch.int32))
        with pytest.raises(TypeError, match="torch.Tensor"):
            m([0, 1])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="move them explicitly"):
            m(torch.zeros(2, dtype=torch.long, device="meta"))

    def test_train_eval_identical(self, tiny_vocab):
        m = _model(tiny_vocab, _random_W(3))
        x = torch.tensor([0, 1, 2])
        m.train()
        a = m(x).detach()
        m.eval()
        assert torch.equal(a, m(x).detach())

    def test_finite_for_extreme_logits(self, tiny_vocab, tiny_pairs):
        W = torch.tensor([[1e4, -1e4, 0.0], [-1e4, 1e4, 1e4], [0.0, 0.0, -1e4]])
        m = _model(tiny_vocab, W)
        x, y = tiny_pairs
        loss = _data_loss(m, x, y)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(m.transition_logits.grad).all()
        p = m.transition_probabilities()
        assert torch.isfinite(p).all() and torch.allclose(p.sum(1), torch.ones(3))


class TestPairs:
    def test_pairs_match_hand_derivation(self, tiny_pairs):
        x, y = tiny_pairs
        assert x.tolist() == [0, 1, 0, 1, 2] and y.tolist() == [1, 0, 1, 2, 0]
        assert x.dtype == y.dtype == torch.int64

    def test_pairs_count_order_and_no_padding(self):
        v = Vocabulary.from_names(SMALL_NAMES)
        x, y = make_bigram_pairs(SMALL_NAMES, v)
        assert x.numel() == y.numel() == sum(len(n) + 1 for n in SMALL_NAMES)
        expected_x = [i for n in SMALL_NAMES for i in v.encode(n, add_bos=True)]
        expected_y = [j for n in SMALL_NAMES for j in v.encode(n, add_eos=True)]
        assert x.tolist() == expected_x and y.tolist() == expected_y
        assert not (x == v.pad_id).any() and not (y == IGNORE_INDEX).any()
        # The pairs are exactly the transitions the count model counts.
        C = torch.zeros(v.num_classes, v.num_classes, dtype=torch.long)
        C.index_put_((x, y), torch.ones_like(x), accumulate=True)
        assert torch.equal(C, count_transitions(SMALL_NAMES, v))

    @pytest.mark.parametrize(
        "names, exc", [([], ValueError), ("anna", TypeError), (["a", ""], ValueError), (["a", 3], TypeError),
                       (["az"], VocabularyCoverageError)])
    def test_invalid_names(self, tiny_vocab, names, exc):
        with pytest.raises(exc):
            make_bigram_pairs(names, tiny_vocab)


class TestTrainingProcedureValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [(dict(lr=0.0, steps=1), "lr"), (dict(lr=-1.0, steps=1), "lr"), (dict(lr=float("nan"), steps=1), "lr"),
         (dict(lr=float("inf"), steps=1), "lr"), (dict(lr=True, steps=1), "lr"),
         (dict(lr=1.0, steps=0), "steps"), (dict(lr=1.0, steps=1.5), "steps"), (dict(lr=1.0, steps=True), "steps"),
         (dict(lr=1.0, steps=1, weight_decay=-0.1), "weight_decay"),
         (dict(lr=1.0, steps=1, weight_decay=float("nan")), "weight_decay")])
    def test_hyperparameter_validation(self, tiny_vocab, tiny_pairs, kwargs, match):
        with pytest.raises(ValueError, match=match):
            train_full_batch(NeuralBigramLanguageModel(tiny_vocab), *tiny_pairs, **kwargs)

    def test_data_validation(self, tiny_vocab, tiny_pairs):
        m = NeuralBigramLanguageModel(tiny_vocab)
        x, y = tiny_pairs
        with pytest.raises(ValueError, match="differ in shape"):
            train_full_batch(m, x, y[:-1], lr=1.0, steps=1)
        with pytest.raises(ValueError, match="1-D"):
            train_full_batch(m, x[None], y[None], lr=1.0, steps=1)
        with pytest.raises(ValueError, match="zero pairs"):
            train_full_batch(m, x[:0], y[:0], lr=1.0, steps=1)
        with pytest.raises(ValueError, match="outside"):
            train_full_batch(m, x, torch.full_like(y, IGNORE_INDEX), lr=1.0, steps=1)
        with pytest.raises(ValueError, match="padding"):
            train_full_batch(m, torch.full_like(x, tiny_vocab.pad_id), y, lr=1.0, steps=1)
        with pytest.raises(TypeError, match="int64"):
            train_full_batch(m, x, y.float(), lr=1.0, steps=1)
        with pytest.raises(ValueError, match="move them explicitly"):
            train_full_batch(m, x, y.to("meta"), lr=1.0, steps=1)
        with pytest.raises(TypeError, match="NeuralBigramLanguageModel"):
            train_full_batch(torch.nn.Linear(2, 2), x, y, lr=1.0, steps=1)  # type: ignore[arg-type]
        assert torch.equal(m.transition_logits.detach(), torch.zeros(3, 3))   # nothing ran

    def test_result_reports_data_loss(self, tiny_vocab, tiny_pairs):
        m = NeuralBigramLanguageModel(tiny_vocab)
        r = train_full_batch(m, *tiny_pairs, lr=5.0, steps=10)
        assert isinstance(r, TrainingResult)
        assert r.initial_loss == pytest.approx(LN3, abs=1e-6)
        assert r.final_loss == pytest.approx(_item(_data_loss(m, *tiny_pairs)), abs=0)

    def test_deterministic(self, tiny_vocab, tiny_pairs):
        runs = []
        for _ in range(2):
            m = NeuralBigramLanguageModel(tiny_vocab)
            r = train_full_batch(m, *tiny_pairs, lr=3.0, steps=200, weight_decay=0.01)
            runs.append((m.transition_logits.detach().clone(), r))
        assert torch.equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]


# =============================================== 13.2 mathematical verification
class TestMathematics:
    def test_1_one_hot_equivalence(self, tiny_vocab):
        W = _random_W(3, seed=3)
        m = _model(tiny_vocab, W)
        x = torch.tensor([0, 2, 1, 1, 0])
        assert torch.equal(m(x).detach(), F.one_hot(x, 3).float() @ W)

    def test_2_softmax_normalisation(self, tiny_vocab):
        p = _model(tiny_vocab, _random_W(3) * 5).transition_probabilities()
        assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-6) and (p > 0).all()

    def test_3_hand_computed_cross_entropy(self, tiny_vocab, tiny_pairs):
        x, y = tiny_pairs
        assert _item(_data_loss(_model(tiny_vocab), x, y)) == pytest.approx(LN3, abs=1e-6)
        W = torch.zeros(3, 3)
        W[0, 1] = LN2                                       # P(a|.) = 2/4, P(.|.) = P(b|.) = 1/4
        expected = (2 * LN2 + 3 * LN3) / 5                  # = 0.936426
        assert _item(_data_loss(_model(tiny_vocab, W), x, y)) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "set_w, expected_rows",
        [
            (False, [[2 / 15, -4 / 15, 2 / 15], [-1 / 15, 2 / 15, -1 / 15], [-2 / 15, 1 / 15, 1 / 15]]),
            (True, [[0.1, -0.2, 0.1], [-1 / 15, 2 / 15, -1 / 15], [-2 / 15, 1 / 15, 1 / 15]]),
        ],
    )
    def test_4_analytic_gradient(self, tiny_vocab, tiny_pairs, tiny_counts, set_w, expected_rows):
        W = torch.zeros(3, 3)
        if set_w:
            W[0, 1] = LN2
        m = _model(tiny_vocab, W)
        _data_loss(m, *tiny_pairs).backward()
        g = m.transition_logits.grad.double()
        assert torch.allclose(g, torch.tensor(expected_rows, dtype=torch.float64), atol=1e-7)
        N, M = tiny_counts.sum(1, keepdim=True), tiny_counts.sum()
        formula = (N * torch.softmax(W.double(), 1) - tiny_counts) / M   # (N_i p_ij − C_ij) / M
        assert torch.allclose(g, formula, atol=1e-7)
        assert torch.allclose(g.sum(1), torch.zeros(3, dtype=torch.float64), atol=1e-7)

    def test_5_finite_difference_gradcheck(self, tiny_vocab, tiny_pairs):
        m = NeuralBigramLanguageModel(tiny_vocab).double()
        x, y = tiny_pairs
        W0 = _random_W(3, seed=5).double().requires_grad_()

        def loss_of(w):
            return token_nll(torch.func.functional_call(m, {"transition_logits": w}, (x,)), y, reduction="mean")

        assert torch.autograd.gradcheck(loss_of, (W0,), eps=1e-6, atol=1e-6)

    def test_6_unseen_target_in_observed_context_is_suppressed(self, tiny_vocab, tiny_pairs):
        m = NeuralBigramLanguageModel(tiny_vocab)
        _data_loss(m, *tiny_pairs).backward()
        g = m.transition_logits.grad
        # BOS → EOS and BOS → b are never observed: ∂L/∂W = (N_0/M) p = (2/5)(1/3) = 2/15 > 0
        assert float(g[0, 0]) == pytest.approx(2 / 15, abs=1e-7) and float(g[0, 2]) == pytest.approx(2 / 15, abs=1e-7)
        assert float(g[2, 1]) == pytest.approx((1 / 5) * (1 / 3), abs=1e-7)   # b → a, unseen, N_b = 1
        train_full_batch(m, *tiny_pairs, lr=1.0, steps=1)
        W = m.transition_logits.detach()
        assert W[0, 0].item() < 0 and W[0, 2].item() < 0

    def test_7_unseen_context_keeps_zero_row(self, tiny_pairs):
        v = Vocabulary(chars=("a", "b", "c"))           # 'c' (id 3) never occurs in the data
        x, y = make_bigram_pairs(["a", "ab"], v)
        m = NeuralBigramLanguageModel(v)
        _data_loss(m, x, y).backward()
        assert torch.equal(m.transition_logits.grad[3], torch.zeros(4))
        for lam in (0.0, 0.1):
            m = NeuralBigramLanguageModel(v)
            train_full_batch(m, x, y, lr=2.0, steps=300, weight_decay=lam)
            assert torch.equal(m.transition_logits.detach()[3], torch.zeros(4))
            assert torch.equal(m.transition_probabilities()[3], torch.full((4,), 0.25))

    def test_8_gradient_flow_only_to_selected_rows(self, tiny_vocab):
        m = _model(tiny_vocab, _random_W(3))
        token_nll(m(torch.tensor([0, 0, 0])), torch.tensor([1, 2, 1]), reduction="mean").backward()
        g = m.transition_logits.grad
        assert torch.equal(g[1:], torch.zeros(2, 3)) and g[0].abs().sum() > 0

    def test_9_count_form_oracle(self):
        v = Vocabulary.from_names(SMALL_NAMES)
        x, y = make_bigram_pairs(SMALL_NAMES, v)
        C = count_transitions(SMALL_NAMES, v).float()
        m = _model(v, _random_W(v.num_classes, seed=9))
        pair_loss = _item(_data_loss(m, x, y))
        count_loss = float(-(C * torch.log_softmax(m.transition_logits.detach(), 1)).sum() / C.sum())
        assert pair_loss == pytest.approx(count_loss, rel=1e-6)

    def test_10_excess_loss_identity(self):
        v = Vocabulary.from_names(SMALL_NAMES)
        x, y = make_bigram_pairs(SMALL_NAMES, v)
        C = count_transitions(SMALL_NAMES, v).double()
        N, M = C.sum(1), C.sum()
        q = torch.where(N[:, None] > 0, C / N.clamp_min(1)[:, None], torch.zeros_like(C))
        mle_nll = BigramLanguageModel(v).fit(SMALL_NAMES).evaluate(SMALL_NAMES).mean_nll
        for seed in range(3):
            m = _model(v, _random_W(v.num_classes, seed=seed) * 2)
            logp = torch.log_softmax(m.transition_logits.detach().double(), 1)
            obs = C > 0
            kl_rows = torch.where(obs, q * (torch.log(torch.where(obs, q, torch.ones_like(q))) - logp), 0.0).sum(1)
            weighted_kl = float(((N / M) * kl_rows).sum())
            assert _item(_data_loss(m, x, y)) - mle_nll == pytest.approx(weighted_kl, abs=1e-5)

    @pytest.mark.parametrize("lam", [0.0, 0.05])
    def test_11_rows_stay_zero_mean(self, lam):
        # The invariant is exact in real arithmetic. It is tested in float64: in float32 the per-step
        # gradient row sum is not exactly 0 (accumulating many per-token terms), so rounding drift of
        # roughly 1e-5 to 1e-4 builds up. That is a precision effect, not a violation of the invariant.
        v = Vocabulary.from_names(SMALL_NAMES)
        m = NeuralBigramLanguageModel(v).double()
        train_full_batch(m, *make_bigram_pairs(SMALL_NAMES, v), lr=2.0, steps=500, weight_decay=lam)
        W = m.transition_logits.detach()
        assert W.abs().max() > 0.5                                     # training actually moved W
        assert torch.allclose(W.sum(1), torch.zeros(v.num_classes, dtype=torch.float64), atol=1e-10)
        # The invariant is on logits, not log-probabilities: their row sums are −V·logsumexp(W_i) ≠ 0.
        assert torch.allclose(m.transition_log_probabilities().sum(1), -v.num_classes * torch.logsumexp(W, 1), atol=1e-10)
        assert (m.transition_log_probabilities().sum(1) < -1).all()

    @pytest.mark.parametrize("lam", [0.0, 0.1])
    def test_12_regularised_sgd_update_is_exact(self, tiny_vocab, tiny_pairs, lam):
        W0 = _random_W(3, seed=12)                                     # non-zero, so the decay term acts
        probe = _model(tiny_vocab, W0)
        _data_loss(probe, *tiny_pairs).backward()                     # ∇L_data only, no penalty
        expected = W0 - 0.7 * (probe.transition_logits.grad + lam * W0)
        m = _model(tiny_vocab, W0)
        train_full_batch(m, *tiny_pairs, lr=0.7, steps=1, weight_decay=lam)
        assert torch.allclose(m.transition_logits.detach(), expected, atol=1e-6)

    def test_13_weight_decay_stationarity(self, tiny_vocab, tiny_pairs, tiny_counts):
        lam = 0.05
        m = NeuralBigramLanguageModel(tiny_vocab)
        train_full_batch(m, *tiny_pairs, lr=5.0, steps=3000, weight_decay=lam)   # 5 < 2/(0.2 + 0.05) = 8
        W = m.transition_logits.detach().double()
        N, M = tiny_counts.sum(1, keepdim=True), tiny_counts.sum()
        q = tiny_counts / N
        residual = (N / M) * (torch.softmax(W, 1) - q) + lam * W
        assert residual.abs().max() < 1e-5
        assert torch.allclose(W.sum(1), torch.zeros(3, dtype=torch.float64), atol=1e-5)
        assert (W[tiny_counts == 0] < 0).all() and torch.isfinite(W).all()   # finite negative unseen logits


# ======================================== 13.3 learning and behaviour checks
class TestLearningBehaviour:
    def test_overfit_tiny_fixture(self, tiny_vocab, tiny_pairs, tiny_counts):
        m = NeuralBigramLanguageModel(tiny_vocab)
        r = train_full_batch(m, *tiny_pairs, lr=5.0, steps=5000)          # bound for this fixture: 2/ℓ = 10
        assert r.final_loss - MLE_TINY < 1e-3
        p, q = m.transition_probabilities().double(), tiny_counts / tiny_counts.sum(1, keepdim=True)
        observed = tiny_counts > 0
        assert (p - q)[observed].abs().max() < 1e-2
        assert (p[~observed] > 0).all() and (p[~observed] < 1e-3).all()

    @pytest.mark.parametrize("lam, lr", [(0.0, 5.0), (0.05, 5.0), (0.2, 4.0)])   # all with η < 2/(0.2 + λ)
    def test_objective_decreases_monotonically(self, tiny_vocab, tiny_pairs, lam, lr):
        m = NeuralBigramLanguageModel(tiny_vocab)
        J = [_objective(m, *tiny_pairs, lam)]
        for _ in range(300):
            train_full_batch(m, *tiny_pairs, lr=lr, steps=1, weight_decay=lam)
            J.append(_objective(m, *tiny_pairs, lam))
        assert max(b - a for a, b in zip(J, J[1:])) <= 1e-6

    def test_data_loss_may_rise_while_objective_falls(self, tiny_vocab, tiny_pairs):
        lam, lr = 0.2, 4.0
        m = NeuralBigramLanguageModel(tiny_vocab)
        J, L = [_objective(m, *tiny_pairs, lam)], []
        for _ in range(200):
            r = train_full_batch(m, *tiny_pairs, lr=lr, steps=1, weight_decay=lam)
            L.append((r.initial_loss, r.final_loss))
            J.append(_objective(m, *tiny_pairs, lam))
        assert max(b - a for a, b in zip(J, J[1:])) <= 1e-6                 # J non-increasing
        assert any(after > before + 1e-3 for before, after in L)            # L_data increased on some step

    @pytest.mark.parametrize("lam", [0.0, 0.05])
    def test_step_composition_is_bit_identical(self, tiny_vocab, tiny_pairs, lam):
        a = NeuralBigramLanguageModel(tiny_vocab)
        train_full_batch(a, *tiny_pairs, lr=3.0, steps=50, weight_decay=lam)
        b = NeuralBigramLanguageModel(tiny_vocab)
        for _ in range(50):
            train_full_batch(b, *tiny_pairs, lr=3.0, steps=1, weight_decay=lam)
        assert torch.equal(a.transition_logits, b.transition_logits)


class TestGeneration:
    @pytest.fixture
    def ab_model(self, tiny_vocab):
        m = NeuralBigramLanguageModel(tiny_vocab)
        train_full_batch(m, *make_bigram_pairs(["ab"], tiny_vocab), lr=5.0, steps=5000)
        p = m.transition_probabilities()
        assert p[0, 1] > 0.999 and p[1, 2] > 0.999 and p[2, 0] > 0.999   # near-deterministic . → a → b → .
        return m

    def test_max_length_and_eos(self, ab_model):
        assert ab_model.generate(1, max_length=1, generator=make_generator(0)) == [NeuralBigramSample("a", False)]
        assert ab_model.generate(1, max_length=2, generator=make_generator(0)) == [NeuralBigramSample("ab", True)]
        assert ab_model.generate(4, max_length=10, generator=make_generator(0)) == [NeuralBigramSample("ab", True)] * 4

    def test_valid_characters_and_reproducibility(self):
        v = Vocabulary.from_names(SMALL_NAMES)
        m = NeuralBigramLanguageModel(v)
        train_full_batch(m, *make_bigram_pairs(SMALL_NAMES, v), lr=5.0, steps=300)
        a = m.generate(200, max_length=8, generator=make_generator(1))
        assert a == m.generate(200, max_length=8, generator=make_generator(1))
        for s in a:
            assert set(s.text) <= set(v.chars) and "." not in s.text and len(s.text) <= 8
            assert s.finished or len(s.text) == 8
        assert any(s.finished for s in a)

    def test_generation_leaves_model_unchanged(self, ab_model):
        before = ab_model.transition_logits.detach().clone()
        grad_before = ab_model.transition_logits.grad.clone()      # left over from the last training step
        ab_model.train()
        ab_model.generate(5, max_length=5, generator=make_generator(2))
        assert ab_model.training and torch.equal(ab_model.transition_logits.detach(), before)
        assert torch.equal(ab_model.transition_logits.grad, grad_before)   # no_grad: no new gradient

    def test_greedy(self, ab_model):
        assert ab_model.generate(2, max_length=5, temperature=0.0) == [NeuralBigramSample("ab", True)] * 2

    @pytest.mark.parametrize("kwargs", [dict(num_samples=0, max_length=3), dict(num_samples=1, max_length=0),
                                        dict(num_samples=True, max_length=3), dict(num_samples=1, max_length=3, top_p=2.0)])
    def test_invalid_arguments(self, tiny_vocab, kwargs):
        with pytest.raises(ValueError):
            NeuralBigramLanguageModel(tiny_vocab).generate(**kwargs)


# =========================================================== §14 checkpoint
class TestCheckpoint:
    def _trained(self, vocab, pairs, steps=100):
        m = NeuralBigramLanguageModel(vocab)
        train_full_batch(m, *pairs, lr=3.0, steps=steps)
        return m

    @pytest.mark.parametrize("kind", ["final", "best"])
    def test_weights_round_trip(self, tiny_vocab, tiny_pairs, tmp_path, kind):
        m = self._trained(tiny_vocab, tiny_pairs)
        extra = dict(best_metric=0.3, best_metric_name="dev_nll") if kind == "best" else {}
        path = save_checkpoint(tmp_path / f"{kind}.pt", kind=kind, model=m, step=100, vocab=tiny_vocab.to_dict(),
                               config={"model": {"name": "neural_bigram", "params": {}}}, **extra)
        loaded = NeuralBigramLanguageModel.from_checkpoint(path, expected_vocab=tiny_vocab)
        assert loaded.vocab == tiny_vocab
        assert torch.equal(loaded.transition_logits, m.transition_logits)
        x = tiny_pairs[0]
        assert torch.equal(loaded(x), m(x)) and torch.equal(loaded.transition_probabilities(), m.transition_probabilities())
        assert loaded.generate(10, max_length=5, generator=make_generator(4)) == \
            m.generate(10, max_length=5, generator=make_generator(4))
        assert load_checkpoint(path)["kind"] == kind                         # weights_only=True works

    def test_resume_is_bit_exact(self, tiny_vocab, tiny_pairs, tmp_path):
        uninterrupted = self._trained(tiny_vocab, tiny_pairs, steps=20)
        train_full_batch(uninterrupted, *tiny_pairs, lr=3.0, steps=20)

        first = self._trained(tiny_vocab, tiny_pairs, steps=20)
        opt = torch.optim.SGD(first.parameters(), lr=3.0, momentum=0.0)
        path = save_checkpoint(tmp_path / "checkpoint_last.pt", kind="resume", model=first, optimizer=opt, step=20,
                               rng_state=capture_rng_state(), vocab=tiny_vocab.to_dict())
        fresh = NeuralBigramLanguageModel(tiny_vocab)
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=3.0, momentum=0.0)
        progress = restore_training_state(load_checkpoint(path, expected_kind="resume"), model=fresh, optimizer=fresh_opt)
        assert progress.step == 20
        train_full_batch(fresh, *tiny_pairs, lr=3.0, steps=20)
        assert torch.equal(fresh.transition_logits, uninterrupted.transition_logits)

    def test_from_checkpoint_rejects_incompatible(self, tiny_vocab, tiny_pairs, tmp_path):
        m = self._trained(tiny_vocab, tiny_pairs)
        good = save_checkpoint(tmp_path / "good.pt", kind="final", model=m, vocab=tiny_vocab.to_dict())
        with pytest.raises(IncompatibleCheckpointError, match="vocabulary mismatch"):
            NeuralBigramLanguageModel.from_checkpoint(good, expected_vocab=Vocabulary(chars=("a", "c")))  # same V
        no_vocab = save_checkpoint(tmp_path / "novocab.pt", kind="final", model=m)
        with pytest.raises(IncompatibleCheckpointError, match="no vocabulary metadata"):
            NeuralBigramLanguageModel.from_checkpoint(no_vocab)
        wrong_v = save_checkpoint(tmp_path / "wrongv.pt", kind="final", model=m,
                                  vocab=Vocabulary(chars=("a", "b", "c")).to_dict())       # V = 4 vs 3x3 weights
        with pytest.raises(IncompatibleCheckpointError, match="do not match the vocabulary"):
            NeuralBigramLanguageModel.from_checkpoint(wrong_v)
        linear = save_checkpoint(tmp_path / "linear.pt", kind="final", model=torch.nn.Linear(3, 3),
                                 vocab=tiny_vocab.to_dict())
        with pytest.raises(IncompatibleCheckpointError, match="not a NeuralBigramLanguageModel"):
            NeuralBigramLanguageModel.from_checkpoint(linear)
        with pytest.raises(CheckpointError, match="not found"):
            NeuralBigramLanguageModel.from_checkpoint(tmp_path / "missing.pt")


# ============================================================== devices
@pytest.mark.parametrize("device", DEVICES)
class TestDevices:
    def test_forward_training_generation_on_device(self, device):
        v = Vocabulary.from_names(SMALL_NAMES)
        x, y = make_bigram_pairs(SMALL_NAMES, v)
        cpu = NeuralBigramLanguageModel(v)
        r_cpu = train_full_batch(cpu, x, y, lr=3.0, steps=50, weight_decay=0.01)
        m = NeuralBigramLanguageModel(v).to(device)
        assert m.transition_logits.device.type == device
        r = train_full_batch(m, x.to(device), y.to(device), lr=3.0, steps=50, weight_decay=0.01)
        assert r.final_loss == pytest.approx(r_cpu.final_loss, abs=1e-5)
        assert torch.allclose(m.transition_logits.detach().cpu(), cpu.transition_logits.detach(), atol=1e-5)
        logits = m(x.to(device))
        assert logits.device.type == device and torch.isfinite(logits).all()
        out = m.generate(20, max_length=8, generator=make_generator(0))
        assert out == m.generate(20, max_length=8, generator=make_generator(0))
        assert all(set(s.text) <= set(v.chars) for s in out)

    def test_device_mismatch_raises(self, device):
        if device == "cpu":
            pytest.skip("needs a second device; covered by the meta-device tests")
        v = Vocabulary.from_names(SMALL_NAMES)
        m = NeuralBigramLanguageModel(v).to(device)
        x, y = make_bigram_pairs(SMALL_NAMES, v)                       # CPU tensors
        with pytest.raises(ValueError, match="move them explicitly"):
            m(x)
        with pytest.raises(ValueError, match="move them explicitly"):
            train_full_batch(m, x, y, lr=1.0, steps=1)

    def test_checkpoint_to_device(self, device, tiny_vocab, tiny_pairs, tmp_path):
        m = NeuralBigramLanguageModel(tiny_vocab)
        train_full_batch(m, *tiny_pairs, lr=3.0, steps=20)
        path = save_checkpoint(tmp_path / "m.pt", kind="final", model=m, vocab=tiny_vocab.to_dict())
        loaded = NeuralBigramLanguageModel.from_checkpoint(path, map_location=device)
        assert loaded.transition_logits.device.type == device
        assert torch.equal(loaded.transition_logits.detach().cpu(), m.transition_logits.detach())


# ========================================================== real-data check
@pytest.mark.skipif(not DEFAULT_NAMES_PATH.is_file(), reason="names dataset not present")
def test_real_dataset_pairs_and_initial_state():
    splits = split_names(load_names(), seed=42)
    vocab = Vocabulary.from_names(splits.train)
    x, y = make_bigram_pairs(splits.train, vocab)
    assert vocab.num_classes == 27 and x.numel() == sum(len(n) + 1 for n in splits.train)
    m = NeuralBigramLanguageModel(vocab)
    acc = NLLAccumulator()
    acc.update(m(x), y)
    assert acc.mean_nll == pytest.approx(math.log(27), abs=1e-6)          # zero init = uniform
    r = train_full_batch(m, x, y, lr=25.0, steps=20)                       # 25 < 2/ℓ ≈ 26.9 on this split
    assert r.final_loss < r.initial_loss
    m64 = NeuralBigramLanguageModel(vocab).double()                        # invariant checked in float64
    train_full_batch(m64, x, y, lr=25.0, steps=20)
    assert torch.allclose(m64.transition_logits.detach().sum(1), torch.zeros(27, dtype=torch.float64), atol=1e-10)

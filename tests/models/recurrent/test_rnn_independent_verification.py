"""Independent verification of Model 6 (vanilla Elman RNN) against oracles written from the specs.

Stage 9 (implementation plan §7.2): written by a separate verification session that took no part
in the implementation and did not read the project's tests or its reference implementation.
Run from the repository root:

    python3 -m pytest -q -W error tests/models/recurrent/test_rnn_independent_verification.py

Independence rules followed here:
* Every expected value is produced by an oracle written in this file from the approved
  specifications (rnn_spec.md v0.2, rnn_training_spec.md v0.3, rnn_diagnostics_spec.md v0.3,
  rnn_integration_spec.md v0.2). Oracles are numpy float64 with explicit Python loops over
  names and time; they never call the model, the diagnostics or the training code.
* Nothing under tests/ and no rnn_reference.py is imported. Implementation modules are used only
  through their public (or spec-documented private, arch §6.3) API.
* Data are synthetic names over a–z. Nothing under data/ or results/ is read; files are written
  only under pytest's tmp_path. Git is disabled before anything can import RunManager (D-15).

Tolerance codes used below (arch §13.1):
* A   = 1e-9 abs + 1e-9 rel: float64 model vs float64 oracle. Rigorous rounding bound for T <= 16,
        H <= 128, g <= 1.2 is ~5e-10 (arch §13.1 rationale); our fixtures are much smaller
        (H <= 16), so typical differences are ~1e-15.
* FD  = central differences in float64, step 1e-6, |fd - ag| <= 1e-8 + 1e-6 |ag|: truncation
        O(h^2) ~ 1e-12, rounding ~ u|L|/h ~ 1e-10 |L| with |L| = O(10).
* B   = 1e-5 for float32 comparisons (typical-error model, arch §13.1).
"""
from __future__ import annotations

# --- Git guard first: project policy D-15 (training spec §10.8). Must precede any RunManager import.
from training.wavenet_study.study_common import disable_git

disable_git()

import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import string  # noqa: E402
import struct  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from models.recurrent import rnn_diagnostics as diag  # noqa: E402
from models.recurrent import rnn_training as rtrain  # noqa: E402
from models.recurrent.rnn import RNNConfig, RNNLanguageModel  # noqa: E402
from utils.batching import EpochSampler  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.data import make_next_token_sequences  # noqa: E402
from utils.metrics import token_nll  # noqa: E402
from utils.reproducibility import derive_seed, make_generator  # noqa: E402
from utils.run_manager import RunManager  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

#: Training-spec sampler stream name (a string constant, not a path).
SAMPLER_STREAM = "data"

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))
K = 27
PAD = 27
BOS = EOS = 0
TOL_A = 1e-9
U64 = 2.0 ** -53
U32 = 2.0 ** -24
LN27 = math.log(27.0)


# ============================================================================ generic helpers
def assert_close(actual, expected, *, atol=TOL_A, rtol=TOL_A, what=""):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape, f"{what}: shape {a.shape} != {e.shape}"
    both_inf = np.isinf(a) & np.isinf(e) & (np.sign(a) == np.sign(e))
    diff = np.where(both_inf, 0.0, np.abs(a - e))
    bound = atol + rtol * np.where(np.isfinite(e), np.abs(e), 0.0)
    bad = diff > bound
    assert not bad.any(), (f"{what}: {int(bad.sum())} element(s) out of tolerance; max excess "
                           f"{float(np.max(diff - bound)):.3e}; first actual {a[bad].ravel()[:3]}, "
                           f"expected {e[bad].ravel()[:3]}")


def random_names(rng: np.random.Generator, lengths):
    return ["".join(rng.choice(list(string.ascii_lowercase), size=L)) for L in lengths]


def ids_of(name: str) -> tuple[list[int], list[int]]:
    """Spec arch §4.0: inputs [BOS, c1..cL], targets [c1..cL, EOS]; n = L + 1."""
    chars = [VOCAB.stoi[c] for c in name]
    return [BOS] + chars, chars + [EOS]


def haar_numpy(H: int, rng: np.random.Generator) -> np.ndarray:
    """Independent Haar orthogonal matrix (QR of a Gaussian with the sign correction of arch §7.1)."""
    z = rng.standard_normal((H, H))
    q, r = np.linalg.qr(z)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s[None, :]


class Params:
    """Plain float64 copies of U [K,H], W [H,H], b [H], V [K,H], c [K]."""

    def __init__(self, U, W, b, V, c):
        self.U, self.W, self.b, self.V, self.c = (np.array(x, dtype=np.float64) for x in (U, W, b, V, c))

    @property
    def H(self):
        return self.W.shape[0]

    def copy(self):
        return Params(self.U, self.W, self.b, self.V, self.c)


def perturbed_params(H: int, g: float, seed: int) -> Params:
    """Fixture rule M6-IMP9 (arch §13.1): U ~ N(0, 0.5), W = g·Haar, b ~ N(0, 0.1²),
    V ~ N(0, 1/H), c ~ N(0, 0.1²); drawn from a test-local numpy generator."""
    rng = np.random.default_rng(seed)
    return Params(U=rng.normal(0, math.sqrt(0.5), (K, H)), W=g * haar_numpy(H, rng),
                  b=rng.normal(0, 0.1, H), V=rng.normal(0, 1 / math.sqrt(H), (K, H)), c=rng.normal(0, 0.1, K))


def build_model(P: Params, *, dtype=torch.float64, g: float = 1.0) -> RNNLanguageModel:
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=P.H, recurrent_gain=g), generator=make_generator(7))
    model = model.to(dtype)
    with torch.no_grad():
        for name, val in (("input_lookup", P.U), ("recurrent_weight", P.W), ("hidden_bias", P.b),
                          ("output_weight", P.V), ("output_bias", P.c)):
            getattr(model, name).copy_(torch.tensor(val, dtype=dtype))
    return model


def model_params(model) -> Params:
    get = lambda n: getattr(model, n).detach().to(torch.float64).numpy().copy()  # noqa: E731
    return Params(get("input_lookup"), get("recurrent_weight"), get("hidden_bias"), get("output_weight"),
                  get("output_bias"))


def batch_tensor(names, T=None) -> torch.Tensor:
    rows = [ids_of(n)[0] for n in names]
    T = T or max(len(r) for r in rows)
    out = torch.full((len(rows), T), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


# ============================================================================ oracles (numpy float64)
def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def oracle_states(P: Params, ids) -> tuple[np.ndarray, np.ndarray]:
    """Arch §4.1 at valid steps: a_t = U[x_t] + W h_{t-1} + b; h_t = tanh(a_t); h_0 = 0.
    Returns (states [n, H], preactivations [n, H]); row i holds step t = i + 1."""
    h = np.zeros(P.H)
    hs, As = [], []
    for x in ids:
        a = np.zeros(P.H)
        for i in range(P.H):  # explicit loop: column-vector convention W[i, j]: unit j (t-1) -> unit i (t)
            a[i] = P.U[x, i] + sum(P.W[i, j] * h[j] for j in range(P.H)) + P.b[i]
        h = np.tanh(a)
        hs.append(h)
        As.append(a)
    return np.array(hs), np.array(As)


def oracle_logits(P: Params, hs: np.ndarray) -> np.ndarray:
    """z_t = V h_t + c (arch §4.1)."""
    return np.array([P.V @ h + P.c for h in hs])


def oracle_name_nll(P: Params, name: str) -> np.ndarray:
    x, y = ids_of(name)
    hs, _ = oracle_states(P, x)
    lp = log_softmax(oracle_logits(P, hs))
    return np.array([-lp[t, y[t]] for t in range(len(y))])


def oracle_seq_nll(P: Params, x, y) -> np.ndarray:
    hs, _ = oracle_states(P, x)
    lp = log_softmax(oracle_logits(P, hs))
    return np.array([-lp[t, y[t]] for t in range(len(y))])


def oracle_grads(P: Params, names, weights=None) -> dict[str, np.ndarray]:
    """Hand-written BPTT (arch §4.4) for L = Σ_names w Σ_t ℓ_t:
    r_t = Vᵀ(p_t − e_y); δ_t = r_t + Wᵀ D_{t+1} δ_{t+1}; g_t = D_t δ_t;
    ∂W = Σ_{t≥2} g_t h_{t−1}ᵀ; ∂U[x] = Σ_{x_t=x} g_t; ∂b = Σ g_t; ∂V = Σ (p−e) h_tᵀ; ∂c = Σ (p−e)."""
    G = {k: np.zeros_like(v) for k, v in (("U", P.U), ("W", P.W), ("b", P.b), ("V", P.V), ("c", P.c))}
    for idx, name in enumerate(names):
        w = 1.0 if weights is None else weights[idx]
        x, y = ids_of(name)
        n = len(x)
        hs, _ = oracle_states(P, x)
        z = oracle_logits(P, hs)
        p = np.exp(log_softmax(z))
        D = 1.0 - hs ** 2
        delta_next = np.zeros(P.H)
        for t in range(n, 0, -1):  # t is 1-based
            e = p[t - 1].copy()
            e[y[t - 1]] -= 1.0
            r = P.V.T @ e
            delta = r + (P.W.T @ (D[t] * delta_next) if t < n else 0.0)
            gt = D[t - 1] * delta
            if t >= 2:
                G["W"] += w * np.outer(gt, hs[t - 2])
            G["U"][x[t - 1]] += w * gt
            G["b"] += w * gt
            G["V"] += w * np.outer(e, hs[t - 1])
            G["c"] += w * e
            delta_next = delta
    return G


def oracle_jacobian_product(P: Params, hs: np.ndarray, t: int, s: int) -> np.ndarray:
    """P_{t,s} = J_t J_{t−1} ⋯ J_{s+1}, J_k = diag(1 − h_k²) W (arch §4.4); newest on the left."""
    M = np.eye(P.H)
    for k in range(s + 1, t + 1):
        Jk = np.diag(1.0 - hs[k - 1] ** 2) @ P.W
        M = Jk @ M
    return M


def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else -math.inf


def stable_residual(z: np.ndarray, y: int) -> np.ndarray:
    """diag §4.3 step 1: g_k = p_k (k ≠ y), g_y = −Σ_{k≠y} p_k, p = exp(log_softmax(z))."""
    p = np.exp(log_softmax(z))
    g = p.copy()
    g[y] = -math.fsum(p[k] for k in range(len(p)) if k != y)
    return g


def scaled_unit(r: np.ndarray):
    """diag §4.3 step 3: exclusion iff r == 0 exactly; otherwise u = r/‖r‖ via max-scaling."""
    m = float(np.max(np.abs(r)))
    if m == 0.0:
        return None, 0.0
    v = r / m
    nv = math.sqrt(math.fsum(float(c) * float(c) for c in v))
    return v / nv, m * nv


def weighted_quantile(values, weights, keys, q):
    """diag §4 convention: smallest x_(j) (ascending; ties by key) with cumulative weight ≥ q·Σw."""
    order = sorted(range(len(values)), key=lambda i: (values[i], keys[i]))
    total = math.fsum(weights)
    cum = 0.0
    for i in order:
        cum += weights[i]
        if cum >= q * total:
            return values[i]
    return values[order[-1]]


def spec_stratum(n: int) -> int:
    """diag §3.1: S1 = {2–5}, S2 = {6–7}, S3 = {8–9}, S4 = {10–16} (0-based index)."""
    for i, (lo, hi) in enumerate(((2, 5), (6, 7), (8, 9), (10, 16))):
        if lo <= n <= hi:
            return i
    raise ValueError(n)


# ============================================================================ shared fixtures
FWD_NAMES = ["a", "bo", "emma", "zachariahxyz", "ann", "qqqqqqqqqqqqqqq", "kim", "al"]


@pytest.fixture(params=[1.0, 1.2], ids=["g1", "g1.2"])
def fwd_case(request):
    g = request.param
    P = perturbed_params(H=16, g=g, seed=101 if g == 1.0 else 102)
    return P, build_model(P, g=g)


# ============================================================================ 1. forward recurrence
def test_forward_matches_explicit_loop_oracle_with_heavy_padding(fwd_case):
    """R1: valid logits/states/pre-activations of every row equal the per-name oracle (A),
    for a mixed-length batch right-padded far beyond the longest name (T = 20)."""
    P, model = fwd_case
    x = batch_tensor(FWD_NAMES, T=20)
    with torch.no_grad():
        logits = model(x)
        rec = model._recurrence(x, return_preactivations=True)
    assert logits.dtype == torch.float64 and logits.shape == (len(FWD_NAMES), 20, K)
    for i, name in enumerate(FWD_NAMES):
        ids, _ = ids_of(name)
        hs, As = oracle_states(P, ids)
        n = len(ids)
        assert_close(rec.states[i, :n].numpy(), hs, what=f"states {name}")
        assert_close(rec.preactivations[i, :n].numpy(), As, what=f"preacts {name}")
        assert_close(logits[i, :n].numpy(), oracle_logits(P, hs), what=f"logits {name}")
        assert rec.valid[i].tolist() == [t < n for t in range(20)]


def test_forward_hand_computed_H2_two_steps():
    """R10-style closed form: H = 2, input 'b' → steps BOS, b; values computed by hand formulas."""
    U = np.zeros((K, 2)); U[0] = [0.3, -0.2]; U[2] = [-0.5, 0.4]
    W = np.array([[0.5, -1.0], [0.25, 0.75]]); b = np.array([0.1, -0.1])
    V = np.zeros((K, 2)); V[0] = [1.0, 2.0]; V[2] = [-1.0, 0.5]; c = np.zeros(K); c[0] = 0.2
    model = build_model(Params(U, W, b, V, c))
    h1 = np.array([math.tanh(0.3 + 0.1), math.tanh(-0.2 - 0.1)])
    a2 = np.array([-0.5 + 0.5 * h1[0] - 1.0 * h1[1] + 0.1, 0.4 + 0.25 * h1[0] + 0.75 * h1[1] - 0.1])
    h2 = np.tanh(a2)
    with torch.no_grad():
        z = model(torch.tensor([[0, 2]]))[0].numpy()
    assert_close(z[0, 0], 1.0 * h1[0] + 2.0 * h1[1] + 0.2)
    assert_close(z[1, 2], -1.0 * h2[0] + 0.5 * h2[1])
    assert_close(z[1, 0], 1.0 * h2[0] + 2.0 * h2[1] + 0.2)
    assert z[1, 5] == 0.0  # class with V row 0 and c 0


def test_hidden_state_frozen_and_padded_logits_equal_last_valid(fwd_case):
    """R7 (bitwise) and R8 (A): for t > n, h_t = h_n and z_t = z_n."""
    P, model = fwd_case
    x = batch_tensor(FWD_NAMES, T=19)
    with torch.no_grad():
        states = model._recurrence(x).states
        logits = model(x)
    for i, name in enumerate(FWD_NAMES):
        n = len(name) + 1
        for t in range(n, 19):
            assert torch.equal(states[i, t], states[i, n - 1]), (name, t)
            assert_close(logits[i, t].numpy(), logits[i, n - 1].numpy(), what="padded logits")
        assert torch.isfinite(logits[i]).all()


def test_padding_width_does_not_affect_valid_outputs(fwd_case):
    """R2 / criterion E: with B fixed, only T changing, valid states are bitwise identical
    (padding columns never enter the per-step [B, H] operands); logits within A."""
    P, model = fwd_case
    with torch.no_grad():
        ref = model._recurrence(batch_tensor(FWD_NAMES)).states
        ref_logits = model(batch_tensor(FWD_NAMES))
        for T in (17, 24, 40):
            st = model._recurrence(batch_tensor(FWD_NAMES, T=T)).states
            lg = model(batch_tensor(FWD_NAMES, T=T))
            for i, name in enumerate(FWD_NAMES):
                n = len(name) + 1
                assert torch.equal(st[i, :n], ref[i, :n])
                assert_close(lg[i, :n].numpy(), ref_logits[i, :n].numpy())


def test_batch_composition_invariance(fwd_case):
    """R3 / arch §4.4: a name alone and inside different batches gives the same valid logits (A);
    each result is also checked against the oracle, so invariance is mathematical, not relative."""
    P, model = fwd_case
    target = "emma"
    hs, _ = oracle_states(P, ids_of(target)[0])
    expected = oracle_logits(P, hs)
    batches = [[target], [target, "zachariahxyz"], ["a", "qqqqqqqqqqqqqqq", target, "bo"],
               ["kim"] * 5 + [target]]
    with torch.no_grad():
        for names in batches:
            out = model(batch_tensor(names))[names.index(target), :5].numpy()
            assert_close(out, expected, what=f"batch {names}")


def test_causality_future_tokens_do_not_change_past_logits(fwd_case):
    """R6: changing x_{t'} (t' > t) leaves z_1 … z_t unchanged; identical shapes → bitwise (D)."""
    P, model = fwd_case
    base = batch_tensor(["abcdefgh"])
    with torch.no_grad():
        z0 = model(base)
        for tp in range(2, 9):  # change input at 0-based position tp (≥ 2: position 0 is BOS)
            mod = base.clone()
            mod[0, tp] = 26 if int(base[0, tp]) != 26 else 1
            z1 = model(mod)
            assert torch.equal(z0[0, :tp], z1[0, :tp]), tp
            assert not torch.equal(z0[0, tp], z1[0, tp])  # the change is visible at its own step


def test_parameter_sharing_one_W_at_every_step():
    """R9: W′ = W + ε e_i e_jᵀ: a′_t − (U[x_t] + W h′_{t−1} + b) = ε h′_{t−1,j} e_i at t ≥ 2; a′_1 = a_1."""
    P = perturbed_params(H=8, g=1.0, seed=5)
    eps, i, j = 1e-3, 2, 5
    P2 = P.copy(); P2.W[i, j] += eps
    model = build_model(P2)
    ids = ids_of("parameter")[0]
    with torch.no_grad():
        rec = model._recurrence(torch.tensor([ids]), return_preactivations=True)
    a, h = rec.preactivations[0].numpy(), rec.states[0].numpy()
    model0 = build_model(P)
    with torch.no_grad():
        a0 = model0._recurrence(torch.tensor([ids]), return_preactivations=True).preactivations[0].numpy()
    assert np.array_equal(a[0], a0[0])
    for t in range(1, len(ids)):
        expected = np.zeros(8); expected[i] = eps * h[t - 1, j]
        assert_close(a[t] - (P.U[ids[t]] + P.W @ h[t - 1] + P.b), expected)


# ============================================================================ 2. loss and BPTT
GRAD_NAMES = ["ab", "zoe", "a", "kaylee", "bob"]


def _model_loss_grads(model, names, weights=None, pad=3):
    x = batch_tensor(names, T=max(len(n) for n in names) + 1 + pad)
    y = torch.full_like(x, -100)
    for r, n in enumerate(names):
        tg = ids_of(n)[1]
        y[r, :len(tg)] = torch.tensor(tg)
    model.zero_grad(set_to_none=True)
    logp = torch.log_softmax(model(x), dim=-1)
    valid = y != -100
    nll = -logp.gather(2, y.clamp_min(0).unsqueeze(2)).squeeze(2) * valid
    w = torch.ones(len(names), dtype=torch.float64) if weights is None else torch.tensor(weights, dtype=torch.float64)
    loss = (nll.sum(dim=1) * w).sum()
    loss.backward()
    return float(loss.detach()), {s: getattr(model, n).grad.numpy().copy() for s, n in (
        ("U", "input_lookup"), ("W", "recurrent_weight"), ("b", "hidden_bias"), ("V", "output_weight"),
        ("c", "output_bias"))}


@pytest.mark.parametrize("g", [1.0, 1.2])
def test_masked_token_summed_nll_and_bptt_gradients_match_oracle(g):
    """arch §4.2 / §4.4, M3: the token-summed NLL over valid targets (padding excluded) and the
    gradients of all five parameters equal the hand-written BPTT oracle (A)."""
    P = perturbed_params(H=6, g=g, seed=11)
    model = build_model(P, g=g)
    loss, grads = _model_loss_grads(model, GRAD_NAMES)
    assert_close(loss, math.fsum(oracle_name_nll(P, n).sum() for n in GRAD_NAMES), what="loss")
    # token_nll(sum) agrees with the masked sum too (training loss path, arch §4.2).
    x = batch_tensor(GRAD_NAMES, T=12)
    y = torch.full_like(x, -100)
    for r, n in enumerate(GRAD_NAMES):
        y[r, :len(n) + 1] = torch.tensor(ids_of(n)[1])
    with torch.no_grad():
        assert_close(float(token_nll(model(x), y, reduction="sum")), loss, what="token_nll")
    og = oracle_grads(P, GRAD_NAMES)
    for k in og:
        assert_close(grads[k], og[k], what=f"grad {k}")


def test_gradients_match_central_finite_differences():
    """M5 / FD: every element of U, W, b, V, c; FD of the independent oracle loss vs model autograd.
    Step 1e-6, |fd − ag| ≤ 1e-8 + 1e-6|ag| (arch §13.1 FD)."""
    P = perturbed_params(H=3, g=1.2, seed=12)
    names = ["ab", "zoe", "kay"]
    model = build_model(P, g=1.2)
    _, grads = _model_loss_grads(model, names)
    step = 1e-6

    def loss_of(Q):
        return math.fsum(oracle_name_nll(Q, n).sum() for n in names)

    for key in ("U", "W", "b", "V", "c"):
        base = getattr(P, key)
        fd = np.zeros_like(base)
        for idx in np.ndindex(base.shape):
            Qp, Qm = P.copy(), P.copy()
            getattr(Qp, key)[idx] += step
            getattr(Qm, key)[idx] -= step
            fd[idx] = (loss_of(Qp) - loss_of(Qm)) / (2 * step)
        assert_close(grads[key], fd, atol=1e-8, rtol=1e-6, what=f"FD {key}")


def test_padded_positions_never_contribute_to_gradients():
    """arch §4.4: padded steps carry no loss term; gradients with T = n_max and T = n_max + 9 agree (A)."""
    P = perturbed_params(H=5, g=1.0, seed=13)
    model = build_model(P)
    l0, g0 = _model_loss_grads(model, GRAD_NAMES, pad=0)
    l9, g9 = _model_loss_grads(model, GRAD_NAMES, pad=9)
    og = oracle_grads(P, GRAD_NAMES)
    assert_close(l9, l0)
    for k in og:
        assert_close(g0[k], og[k], what=f"T=n_max {k}")
        assert_close(g9[k], og[k], what=f"T=n_max+9 {k}")


# ============================================================================ 3. Jacobians and G1
def _probe_fixture(names_by_stratum, population, extra_names=()):
    """A train set and a hand-made Probe: rows by stratum ascending; halves alternate A, B, …;
    weights w_ℓ = N_ℓ / n_ℓ (diag §3.1)."""
    names = list(extra_names)
    rows, strata, halves, pnames = [], [], [], []
    sample = [0, 0, 0, 0]
    for s, group in enumerate(names_by_stratum):
        for k, nm in enumerate(group):
            assert spec_stratum(len(nm) + 1) == s
            rows.append(len(names)); names.append(nm); pnames.append(nm)
            strata.append(s); halves.append("A" if k % 2 == 0 else "B"); sample[s] += 1
    weights = tuple(population[s] / sample[s] if sample[s] else 0.0 for s in range(4))
    probe = diag.Probe(rows=tuple(rows), names=tuple(pnames), strata=tuple(strata), halves=tuple(halves),
                       weights_by_stratum=weights, population_by_stratum=tuple(population),
                       sample_by_stratum=tuple(sample), split_info={}, sha256="")
    train = make_next_token_sequences(names, VOCAB)
    return names, train, probe


G_NAMES = [["ab", "abcd"], ["abcde", "zyxwvu", "mmmmm"], ["qwertyu", "asdfghjk"], ["zxcvbnmlk", "poiuytrewqa"]]
G_POP = (10, 21, 7, 5)


def test_jacobian_oracle_matches_autograd_of_step_chain():
    """M1: the oracle P_{t,s} equals autograd ∂h_t/∂h_s through the model's own `_step` (A)."""
    P = perturbed_params(H=5, g=1.2, seed=21)
    model = build_model(P, g=1.2)
    ids = ids_of("abcdefg")[0]
    hs, _ = oracle_states(P, ids)
    for s, t in ((1, 2), (1, 5), (3, 8), (2, 7)):
        def chain(h_s, s=s, t=t):
            h = h_s
            for k in range(s + 1, t + 1):
                h, _ = model._step(h.unsqueeze(0), torch.tensor([ids[k - 1]]), torch.tensor([True]))
                h = h[0]
            return h
        J = torch.autograd.functional.jacobian(chain, torch.tensor(hs[s - 1]))
        assert_close(J.numpy(), oracle_jacobian_product(P, hs, t, s), what=f"P_{t},{s}")


def test_g1_phi_sigma_delta_hat_omega_match_explicit_products():
    """G1 (diag §4.2): for every probe pair 1 ≤ s < t ≤ n: log φ = log(‖P‖_F/√H), log σ = log ‖P‖₂,
    log δ̂ = Σ_k ½ log(g_W² mean_j D_kj²) with g_W = ‖W‖_F/√H; ω(Δ) = ‖W^Δ‖_F/√H for Δ = 1..15;
    per-(stratum, Δ) unweighted and per-(s, Δ) weighted quantiles. Tolerance A on the logs
    (relative 1e-9 on the norms; ≤ 15 products of H×H float64 matrices round at ~1e-14)."""
    P = perturbed_params(H=6, g=1.2, seed=22)
    model = build_model(P, g=1.2)
    names, train, probe = _probe_fixture(G_NAMES, G_POP, extra_names=["zz", "yyy"])
    res = diag.g1_products(diag.float64_copy(model), probe, train, spectral=True)
    raw = res.raw
    H = P.H
    gW = np.linalg.norm(P.W) / math.sqrt(H)
    expected = {}
    for b, (row, nm) in enumerate(zip(probe.rows, probe.names)):
        ids = ids_of(nm)[0]
        hs, _ = oracle_states(P, ids)
        n = len(ids)
        for s in range(1, n):
            for t in range(s + 1, n + 1):
                M = oracle_jacobian_product(P, hs, t, s)
                phi = np.linalg.norm(M) / math.sqrt(H)
                sig = np.linalg.svd(M, compute_uv=False)[0]
                dh = math.fsum(0.5 * math.log(gW ** 2 * np.mean((1 - hs[k - 1] ** 2) ** 2)) for k in range(s + 1, t + 1))
                expected[(row, s, t)] = (math.log(phi), math.log(sig), dh, probe.strata[b], probe.weights()[b].item())
    got = {(int(r), int(s), int(t)): (lp, ls, ld) for r, s, t, lp, ls, ld in zip(
        raw["g1_row"], raw["g1_s"], raw["g1_t"], raw["g1_log_phi"], raw["g1_log_sigma"], raw["g1_log_delta_hat"])}
    assert set(got) == set(expected), "pair set differs from {1 ≤ s < t ≤ n} over probe rows"
    for key, (lp, ls, ld, _, _) in expected.items():
        assert_close(got[key], (lp, ls, ld), what=f"G1 {key}")
        assert got[key][0] <= got[key][1] + 1e-12  # φ ≤ σ [T]
    assert_close(res.primary["g_W"], gW)
    Wd = np.eye(H)
    omega = []
    for _ in range(15):
        Wd = P.W @ Wd
        omega.append(np.linalg.norm(Wd) / math.sqrt(H))
    assert_close(res.primary["omega"], omega, what="omega")
    # Reporting: per (stratum, Δ) unweighted quantiles and per (s, Δ) population-weighted quantiles.
    keyf = lambda k: (expected[k][3], k[0], k[2], k[1])  # noqa: E731  (stratum, row, t, s)
    for row in res.primary["per_stratum_delta"]:
        ks = [k for k in expected if expected[k][3] == row["stratum"] and k[2] - k[1] == row["delta"]]
        assert row["pairs"] == len(ks)
        for q, v in zip((0.25, 0.5, 0.75), row["log_phi_q"]):
            assert_close(v, weighted_quantile([expected[k][0] for k in ks], [1.0] * len(ks), [keyf(k) for k in ks], q))
    for row in res.primary["per_start_delta_weighted"]:
        ks = [k for k in expected if k[1] == row["start"] and k[2] - k[1] == row["delta"]]
        assert row["pairs"] == len(ks)
        for q, v in zip((0.25, 0.5, 0.75), row["log_phi_q"]):
            assert_close(v, weighted_quantile([expected[k][0] for k in ks], [expected[k][4] for k in ks],
                                              [keyf(k) for k in ks], q))


def test_g1_closed_form_W_zero():
    """Closed form: W = 0 ⇒ P_{t,s} = 0 for Δ ≥ 1, ω(Δ) = 0, g_W = 0 ⇒ log φ = log σ = log δ̂ = −inf."""
    P = perturbed_params(H=4, g=1.0, seed=23)
    P.W[:] = 0.0
    model = build_model(P)
    _, train, probe = _probe_fixture([["abc"], ["abcdef"], [], []], (3, 4, 0, 0))
    res = diag.g1_products(diag.float64_copy(model), probe, train, spectral=True)
    assert len(res.raw["g1_log_phi"]) == 6 + 21  # n = 4 → 6 pairs; n = 7 → 21 pairs
    assert np.all(np.isneginf(res.raw["g1_log_phi"]))
    assert np.all(np.isneginf(res.raw["g1_log_sigma"]))
    assert np.all(np.isneginf(res.raw["g1_log_delta_hat"]))
    assert res.primary["omega"] == [0.0] * 15 and res.primary["g_W"] == 0.0


def test_g1_closed_form_orthogonal_W_tiny_inputs():
    """Closed form: W orthogonal (g = 1), U = 1e-7·N, b = 0 ⇒ |h| ≲ 1e-6, D_k = I − O(1e-12), so
    P_{t,s} ≈ W^Δ (orthogonal): φ, σ, δ̂ = 1 within 1e-10; ω(Δ) = 1 within 1e-12 (exact
    orthogonality of W in float64, rounding ~ Δ·H·u)."""
    rng = np.random.default_rng(24)
    H = 6
    P = Params(U=1e-7 * rng.standard_normal((K, H)), W=haar_numpy(H, rng), b=np.zeros(H),
               V=rng.standard_normal((K, H)), c=np.zeros(K))
    model = build_model(P)
    _, train, probe = _probe_fixture([[], [], [], ["abcdefghijklmno"]], (0, 0, 0, 1))
    res = diag.g1_products(diag.float64_copy(model), probe, train, spectral=True)
    assert len(res.raw["g1_log_phi"]) == 16 * 15 // 2
    for key in ("g1_log_phi", "g1_log_sigma", "g1_log_delta_hat"):
        assert np.max(np.abs(res.raw[key])) < 1e-10, key
    assert_close(res.primary["omega"], [1.0] * 15, atol=1e-12, rtol=0)


# ============================================================================ 4. G2, G3, probe
def _g2_oracle(P: Params, probe):
    out, excluded, rmin = {}, {}, math.inf
    for b, (row, nm) in enumerate(zip(probe.rows, probe.names)):
        x, y = ids_of(nm)
        hs, _ = oracle_states(P, x)
        z = oracle_logits(P, hs)
        for t in range(1, len(x) + 1):
            r = P.V.T @ stable_residual(z[t - 1], y[t - 1])
            u, rn = scaled_unit(r)
            if u is None:
                excluded[probe.strata[b]] = excluded.get(probe.strata[b], 0) + 1
                continue
            rmin = min(rmin, rn)
            v = u
            for s in range(t, 0, -1):
                if s < t:
                    v = P.W.T @ ((1.0 - hs[s] ** 2) * v)  # J_{s+1}ᵀ = Wᵀ D_{s+1}
                out[(row, s, t)] = (math.sqrt(math.fsum(float(c) ** 2 for c in v)), rn)
    return out, excluded, rmin


def test_g2_vectors_match_oracle_and_explicit_product():
    """G2 (diag §4.3): R_{t,s} = ‖P_{t,s}ᵀ r_t‖/‖r_t‖ with r_t = Vᵀ(p_t − e_y) for every s ≤ t (A);
    R_{t,t} = 1; R ≤ σ_{t,s}; the vector chain agrees with the explicit matrix product."""
    P = perturbed_params(H=6, g=1.2, seed=31)
    model = build_model(P, g=1.2)
    _, train, probe = _probe_fixture(G_NAMES, G_POP)
    res = diag.g2_attenuation(diag.float64_copy(model), probe, train)
    exp, excluded, rmin = _g2_oracle(P, probe)
    raw = res.raw
    got = {(int(r), int(s), int(t)): (R, rn) for r, s, t, R, rn in zip(
        raw["g2_row"], raw["g2_s"], raw["g2_t"], raw["g2_R"], raw["g2_r_norm"])}
    assert set(got) == set(exp)
    for k in exp:
        assert_close(got[k], exp[k], what=f"G2 {k}")
    assert excluded == {} and res.primary["excluded_exact_zero_by_stratum"] == {}
    assert_close(res.primary["min_r_norm"], rmin)
    # Cross-check the chain on one row with the explicit matrix product (independent route).
    row, nm = probe.rows[-1], probe.names[-1]
    x, y = ids_of(nm)
    hs, _ = oracle_states(P, x)
    z = oracle_logits(P, hs)
    for t in (len(x), len(x) - 2):
        r = P.V.T @ (np.exp(log_softmax(z[t - 1])) - np.eye(K)[y[t - 1]])
        for s in range(1, t + 1):
            M = oracle_jacobian_product(P, hs, t, s)
            R = np.linalg.norm(M.T @ r) / np.linalg.norm(r)
            assert_close(got[(row, s, t)][0], R, what="explicit")
            assert R <= np.linalg.svd(M, compute_uv=False)[0] * (1 + 1e-12)
        assert got[(row, t, t)][0] == pytest.approx(1.0, abs=1e-15)
    # Per-(stratum, Δ) magnitude-weighted R̄ = Σ‖r‖R / Σ‖r‖ (diag §4.3).
    strat = {r: s for r, s in zip(probe.rows, probe.strata)}
    for cell in res.primary["per_stratum_delta"]:
        ks = [k for k in exp if strat[k[0]] == cell["stratum"] and k[2] - k[1] == cell["delta"]]
        assert cell["pairs"] == len(ks)
        assert_close(cell["R_bar"], math.fsum(exp[k][1] * exp[k][0] for k in ks) / math.fsum(exp[k][1] for k in ks))


def test_g2_exact_zero_exclusion_rule_V_zero():
    """diag §4.3 step 2: with V = 0 every r_t is exactly 0 → every target excluded and counted per
    stratum; nothing else is recorded."""
    P = perturbed_params(H=4, g=1.0, seed=32)
    P.V[:] = 0.0
    model = build_model(P)
    _, train, probe = _probe_fixture(G_NAMES, G_POP)
    res = diag.g2_attenuation(diag.float64_copy(model), probe, train)
    expected = {}
    for nm, s in zip(probe.names, probe.strata):
        expected[str(s)] = expected.get(str(s), 0) + len(nm) + 1
    assert res.primary["excluded_exact_zero_by_stratum"] == expected
    assert len(res.raw.get("g2_R", [])) == 0 and res.primary["min_r_norm"] is None


def test_g2_tiny_and_near_certain_residuals_are_kept_and_exact():
    """diag §4.3 / D6: (i) V ~ 1e-200 gives r ~ 1e-200 (‖r‖² would underflow): targets are kept and
    R equals the scaled-norm oracle (A); (ii) a target with p_y = 1 − O(1e-20) uses
    g_y = −Σ_{k≠y} p_k (not 1 − p_y, which cancels to 0) and matches the oracle (A)."""
    P = perturbed_params(H=5, g=1.0, seed=33)
    P.V *= 1e-200
    _, train, probe = _probe_fixture([["ab"], ["bbbbb"], [], []], (1, 1, 0, 0))
    res = diag.g2_attenuation(diag.float64_copy(build_model(P)), probe, train)
    exp, excluded, _ = _g2_oracle(P, probe)
    assert excluded == {} and res.primary["excluded_exact_zero_by_stratum"] == {}
    got = {(int(r), int(s), int(t)): R for r, s, t, R in zip(res.raw["g2_row"], res.raw["g2_s"],
                                                           res.raw["g2_t"], res.raw["g2_R"])}
    assert set(got) == set(exp)
    for k in exp:
        assert math.isfinite(got[k])
        assert_close(got[k], exp[k][0])
    # (ii) near-certain targets: c['b'] = 50 makes p_b ≈ 1 − 26 e^{−50} at every step.
    P2 = perturbed_params(H=5, g=1.0, seed=34)
    P2.c[:] = 0.0; P2.c[2] = 50.0; P2.V *= 1e-3
    res2 = diag.g2_attenuation(diag.float64_copy(build_model(P2)), probe, train)
    exp2, _, _ = _g2_oracle(P2, probe)
    got2 = {(int(r), int(s), int(t)): R for r, s, t, R in zip(res2.raw["g2_row"], res2.raw["g2_s"],
                                                            res2.raw["g2_t"], res2.raw["g2_R"])}
    for k in exp2:
        assert_close(got2[k], exp2[k][0], what=f"near-certain {k}")


def _g3_oracle(P: Params, probe, weights):
    """diag §4.4: T_{t,k} = w (D_k P_{t,k}ᵀ r_t) h_{k−1}ᵀ, 2 ≤ k ≤ t ≤ n; returns list of
    (Δ, T) and Σ T."""
    terms = []
    for b, nm in enumerate(probe.names):
        w = weights[b]
        if w == 0.0:
            continue
        x, y = ids_of(nm)
        n = len(x)
        hs, _ = oracle_states(P, x)
        z = oracle_logits(P, hs)
        for t in range(2, n + 1):
            r = P.V.T @ stable_residual(z[t - 1], y[t - 1])
            for k in range(t, 1, -1):
                M = oracle_jacobian_product(P, hs, t, k)  # explicit product, not the vector chain
                u = M.T @ r
                T = w * np.outer((1.0 - hs[k - 1] ** 2) * u, hs[k - 2])
                terms.append((t - k, T))
    return terms, sum(T for _, T in terms)


def _shares(terms, G):
    G2 = float((G * G).sum())
    share, pos, neg, mass = {}, {}, {}, {}
    for d, T in terms:
        ip = float((T * G).sum())
        share[d] = share.get(d, 0.0) + ip
        pos.setdefault(d, 0.0); neg.setdefault(d, 0.0)
        if ip > 0:
            pos[d] += ip
        elif ip < 0:
            neg[d] += ip
        mass[d] = mass.get(d, 0.0) + float(np.linalg.norm(T))
    total = math.fsum(mass.values())
    return ({d: v / G2 for d, v in share.items()}, {d: v / G2 for d, v in pos.items()},
            {d: v / G2 for d, v in neg.items()}, mass, math.sqrt(G2) / total)


def test_g3_decomposition_sums_to_autograd_and_shares_match():
    """G3 (diag §4.4, M4, D5): Σ T_{t,k} (explicit Jacobian products) equals autograd ∂L̂/∂W of the
    probe-weighted loss L̂ = Σ w_β Σ_t ℓ (A, relative Frobenius); π̂, π̂±, Q̂, κ̂, Π̂(Δ≥3) for the full
    probe and halves (half weights N_ℓ/m_{ℓ,h}) match the oracle (A); Σπ̂ = 1; cos(Ĝ_A, Ĝ_B)."""
    P = perturbed_params(H=5, g=1.2, seed=41)
    model = build_model(P, g=1.2)
    names, train, probe = _probe_fixture(G_NAMES, G_POP)
    res = diag.g3_decomposition(diag.float64_copy(model), probe, train)
    # Weights computed by hand from the fixture: w_ℓ = N_ℓ/n_ℓ; halves N_ℓ/m_{ℓ,h}.
    n_l = [len(g) for g in G_NAMES]
    full_w = [G_POP[s] / n_l[s] for s in probe.strata]
    half_w = {}
    for h in "AB":
        m = [sum(1 for s2, h2 in zip(probe.strata, probe.halves) if s2 == s and h2 == h) for s in range(4)]
        half_w[h] = [G_POP[s] / m[s] if hh == h else 0.0 for s, hh in zip(probe.strata, probe.halves)]
    assert_close(probe.weights().numpy(), full_w)
    for h in "AB":
        assert_close(probe.half_weights(h).numpy(), half_w[h])
    Gs = {}
    for key, w in (("full", full_w), ("A", half_w["A"]), ("B", half_w["B"])):
        terms, G = _g3_oracle(P, probe, w)
        _, ag = _model_loss_grads(model, list(probe.names), weights=w)
        assert np.linalg.norm(G - ag["W"]) <= TOL_A * np.linalg.norm(ag["W"]), f"ΣT ≠ autograd ({key})"
        assert_close(G, oracle_grads(P, list(probe.names), weights=w)["W"], what="ΣT vs BPTT")
        share, pos, neg, mass, kappa = _shares(terms, ag["W"])
        out = res.primary[key]
        assert set(out["share"]) == {str(d) for d in share}
        for d in share:
            assert_close(out["share"][str(d)], share[d], what=f"{key} share Δ={d}")
            assert_close(out["share_pos"][str(d)], pos[d], what=f"{key} share+ Δ={d}")
            assert_close(out["share_neg"][str(d)], neg[d], what=f"{key} share− Δ={d}")
            assert_close(out["mass"][str(d)], mass[d], what=f"{key} mass Δ={d}")
        assert_close(out["kappa"], kappa)
        assert_close(out["long_share_ge3"], math.fsum(v for d, v in share.items() if d >= 3))
        assert_close(out["grad_norm"], np.linalg.norm(ag["W"]))
        assert abs(math.fsum(share.values()) - 1.0) <= TOL_A / kappa
        assert abs(out["share_sum"] - 1.0) <= TOL_A / kappa
        assert res.checks[f"identity_{key}"]["pass"] and res.checks[f"share_sum_{key}"]["pass"]
        Gs[key] = ag["W"]
    cos = float((Gs["A"] * Gs["B"]).sum()) / (np.linalg.norm(Gs["A"]) * np.linalg.norm(Gs["B"]))
    assert_close(res.primary["agreement"]["cos_halves"], cos)


def test_build_probe_strata_draw_weights_halves_on_synthetic_population(tmp_path):
    """diag §3.1 / §3.3 (D1, D11): strata by n; n_ℓ = min(256, N_ℓ); one generator
    make_generator(derive_seed(42, "m6-probe")) for S1..S4 in order; rows = sorted(R_ℓ[randperm][:n_ℓ]);
    halves alternate A/B; w_ℓ = N_ℓ/n_ℓ; Horvitz–Thompson total of x ≡ 1 equals N_ℓ exactly."""
    rng = np.random.default_rng(51)
    lengths = [int(L) for L in np.concatenate([rng.integers(1, 5, 40), rng.integers(5, 7, 300),
                                               rng.integers(7, 9, 9), rng.integers(9, 16, 270)])]
    rng.shuffle(lengths)
    names = random_names(rng, lengths)
    train = make_next_token_sequences(names, VOCAB)
    split_info = {"seed": 42, "N": len(names)}
    probe = diag.build_probe(train, names, split_info, tmp_path / "probe.json")
    strata = [spec_stratum(L + 1) for L in lengths]
    gen = make_generator(derive_seed(42, "m6-probe"))
    exp_rows, exp_strata, exp_halves, N_l, n_l = [], [], [], [], []
    for s in range(4):
        R = [i for i, st in enumerate(strata) if st == s]
        n = min(256, len(R))
        perm = torch.randperm(len(R), generator=gen)[:n].tolist()
        chosen = sorted(R[i] for i in perm)
        exp_rows += chosen; exp_strata += [s] * n; exp_halves += ["AB"[k % 2] for k in range(n)]
        N_l.append(len(R)); n_l.append(n)
    assert list(probe.rows) == exp_rows and list(probe.strata) == exp_strata
    assert list(probe.halves) == exp_halves and list(probe.names) == [names[i] for i in exp_rows]
    assert list(probe.population_by_stratum) == N_l and list(probe.sample_by_stratum) == n_l
    assert n_l[1] == 256 and n_l[3] == 256 and n_l[0] == N_l[0] and n_l[2] == N_l[2]  # cap and census
    assert list(probe.weights_by_stratum) == [N_l[s] / n_l[s] for s in range(4)]
    w = probe.weights().numpy()
    for s in range(4):  # HT total of x ≡ 1 within stratum = N_ℓ (exact up to rounding of N/n · n)
        assert_close(w[np.array(probe.strata) == s].sum(), N_l[s], atol=1e-9, rtol=1e-12)
    for h in "AB":
        hw = probe.half_weights(h).numpy()
        for s in range(4):
            m = sum(1 for st, hh in zip(probe.strata, probe.halves) if st == s and hh == h)
            sel = (np.array(probe.strata) == s) & (np.array(probe.halves) == h)
            assert_close(hw[sel], np.full(sel.sum(), N_l[s] / m))
            assert np.all(hw[(np.array(probe.halves) != h)] == 0.0)
    # SHA-256 of the canonical JSON (sorted keys, no NaN) as stored, and the file round trip.
    stored = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    sha = stored.pop("sha256")
    canon = json.dumps(stored, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    assert hashlib.sha256(canon.encode("utf-8")).hexdigest() == sha == probe.sha256
    assert stored["schema"] == "m6_probe/1"
    diag.validate_probe(diag.load_probe(tmp_path / "probe.json"), train, names, split_info)


def test_horvitz_thompson_unbiased_on_enumerable_population():
    """diag §3.3 [T]: under SRSWOR of n from N, E[Σ_sample (N/n) x] = Σ_pop x; enumerated exactly."""
    x = [Fraction(v) for v in (3, -1, 7, 2, 5)]
    N, n = len(x), 2
    ests = [Fraction(N, n) * sum(x[i] for i in S) for S in itertools.combinations(range(N), n)]
    assert sum(ests) / len(ests) == sum(x)


# ============================================================================ 5. C1 restart
C1_NAMES = ["ab", "abcdefghijk", "zzzz", "maximilianus", "xyzxyzxyzx", "q", "abcdefghijkl", "hello"]


def _c1_oracle(P: Params, names):
    out = {}
    for row, nm in enumerate(names):
        x, y = ids_of(nm)
        n = len(x)
        full = oracle_seq_nll(P, x, y)
        for t in range(3, n + 1):
            for d in range(1, t - 1):  # affected: t ≥ d + 2
                kept = x[t - d: t]  # x_{t−d+1} … x_t (1-based) → 0-based slice [t−d, t)
                seq = [BOS] + kept
                hs, _ = oracle_states(P, seq)
                lp = log_softmax(oracle_logits(P, hs)[-1])
                out[(row, t, d)] = -lp[y[t - 1]] - full[t - 1]
    return out


def _clustered(values):
    names = sorted(values)
    s = [math.fsum(values[i]) for i in names]
    m = [len(values[i]) for i in names]
    tot = sum(m)
    mean = math.fsum(s) / tot
    nc = len(names)
    se = math.sqrt(nc / (nc - 1) * math.fsum((si - mean * mi) ** 2 for si, mi in zip(s, m)) / tot ** 2)
    return tot, nc, mean, se


def test_c1_restart_equals_explicit_rerun_from_bos():
    """C1 (arch §12.6; diag §4.5): ΔNLL_t(d) = −log p^{(d)}[y_t] + log p_t[y_t], restart = h_0 = 0,
    BOS, then x_{t−d+1} … x_t, for every affected (t, d) with t ≥ d + 2 (A); the set of pairs is
    exactly the affected set; F₈ (t ≥ 10, d ≤ 8) and F₇ (t ≥ 9, d ≤ 7) means and name-clustered
    SEs match the hand computation; chunking does not matter; the j = 2 no-op check passes."""
    P = perturbed_params(H=6, g=1.2, seed=61)
    model = build_model(P, g=1.2)
    dev = make_next_token_sequences(C1_NAMES, VOCAB)
    exp = _c1_oracle(P, C1_NAMES)
    for chunk in (2048, 7):
        res = diag.c1_restart(diag.float64_copy(model), dev, chunk=chunk)
        raw = res.raw
        got = {(int(r), int(t), int(d)): v for r, t, d, v in zip(raw["c1_row"], raw["c1_t"], raw["c1_d"],
                                                                raw["c1_delta_nll"])}
        assert set(got) == set(exp)
        for k in exp:
            assert_close(got[k], exp[k], what=f"C1 {k}")
        assert res.checks["noop_restart_reproduces_full"]
        for label, tmin, dmax in (("F8", 10, 8), ("F7", 9, 7)):
            curve = res.primary[label]
            assert [c["d"] for c in curve] == list(range(1, dmax + 1))
            for c in curve:
                vals = {}
                for (r, t, d), v in exp.items():
                    if d == c["d"] and t >= tmin:
                        vals.setdefault(r, []).append(v)
                tot, nc, mean, se = _clustered(vals)
                assert (c["targets"], c["names"]) == (tot, nc)
                assert_close([c["mean"], c["se"]], [mean, se], what=f"{label} d={c['d']}")


def test_c1_noop_for_t_le_d_plus_1_oracle_self_check():
    """M6-E6 [T]: for t ≤ d + 1 the restart keeps every input after BOS, so it reproduces z_t exactly
    (checked on the model's forward: restart row [BOS, x_2 … x_t] equals the full prefix, A)."""
    P = perturbed_params(H=6, g=1.0, seed=62)
    model = build_model(P)
    with torch.no_grad():
        full = model(batch_tensor(["maximilian"]))[0]
        for t in range(2, 12):
            x = ids_of("maximilian")[0]
            restart = torch.tensor([[BOS] + x[1:t]])
            # different batch shapes: equality is mathematical, compared under A (arch §13.1 F, M8)
            assert_close(model(restart)[0, -1].numpy(), full[t - 1].numpy())


# ============================================================================ 6. initialisation
@pytest.mark.parametrize("g,s2", [(1.0, 0.5), (1.3, 0.2)])
def test_initialisation_draw_order_and_values(g, s2):
    """arch §7.1 (I3, I4): from the constructor's generator, Z ~ N(0,1)^{H×H} float64, QR, sign fix,
    W = g·Q → float32; then U = s·N(0,1)^{27×H} float64 → float32; b = V = c = 0. Reproduced
    bitwise from a fresh generator with the same seed (every step is deterministic on one machine);
    global RNG untouched; different seeds differ."""
    H, seed = 128, 4242
    state = torch.get_rng_state()
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=H, recurrent_gain=g, input_init_variance=s2),
                             generator=make_generator(seed))
    assert torch.equal(torch.get_rng_state(), state)
    gen = make_generator(seed)
    Z = torch.randn(H, H, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(Z, mode="reduced")
    sig = torch.sign(torch.diagonal(R))
    sig[sig == 0] = 1.0
    Q = Q @ torch.diag(sig)
    assert torch.max(torch.abs(Q.T @ Q - torch.eye(H, dtype=torch.float64))) <= 1e-12
    assert torch.equal(model.recurrent_weight.data, (g * Q).to(torch.float32))
    U = (math.sqrt(s2) * torch.randn(K, H, generator=gen, dtype=torch.float64)).to(torch.float32)
    assert torch.equal(model.input_lookup.data, U)
    for name in ("hidden_bias", "output_weight", "output_bias"):
        p = getattr(model, name)
        assert p.dtype == torch.float32 and torch.count_nonzero(p) == 0
    # WᵀW = g²I up to float32 rounding (≈1e-7 relative); singular values within 1e-5·g (I3).
    W = model.recurrent_weight.detach().double()
    assert torch.max(torch.abs(W.T @ W / g ** 2 - torch.eye(H, dtype=torch.float64))) < 1e-5
    assert torch.max(torch.abs(torch.linalg.svdvals(W) - g)) <= 1e-5 * g
    # U variance: |var − s²| ≤ 5 s² √(2/(27H)) (I5, five standard errors).
    u = model.input_lookup.detach().double()
    assert abs(float(u.var(unbiased=True)) - s2) <= 5 * s2 * math.sqrt(2 / (K * H))
    other = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=H, recurrent_gain=g, input_init_variance=s2),
                             generator=make_generator(seed + 1))
    assert not torch.equal(other.recurrent_weight, model.recurrent_weight)
    assert sum(p.numel() for p in model.parameters()) == H * H + 55 * H + 27


@pytest.mark.parametrize("dtype,u", [(torch.float32, U32), (torch.float64, U64)])
def test_initial_nll_is_ln27_and_M6_E4(dtype, u):
    """I1, I2 (M6-E4): with V = c = 0 every logit is exactly 0, every per-target NLL is ln 27 within
    4u·ln 27; gradients of U, W, b are exactly zero; h_1 is identical for every row."""
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=128), generator=make_generator(derive_seed(3, "init")))
    model = model.to(dtype)
    names = random_names(np.random.default_rng(71), [1, 3, 7, 15, 4])
    seq = make_next_token_sequences(names, VOCAB)
    logits = model(seq.inputs)
    assert torch.count_nonzero(logits) == 0
    per = token_nll(logits, seq.targets, reduction="none")[seq.mask]
    assert torch.all(torch.abs(per.double() - LN27) <= 4 * u * LN27)
    acc = rtrain.evaluate_nll(model, seq, batch_size=2)
    assert abs(acc.mean_nll - LN27) <= 4 * u * LN27 * 2
    model.zero_grad()
    token_nll(model(seq.inputs), seq.targets, reduction="sum").backward()
    for name in ("input_lookup", "recurrent_weight", "hidden_bias"):
        assert torch.count_nonzero(getattr(model, name).grad) == 0, name
    with torch.no_grad():
        st = model._recurrence(seq.inputs).states
    assert all(torch.equal(st[0, 0], st[i, 0]) for i in range(len(names)))


# ============================================================================ 7. param_sha256
def test_param_sha256_canonical_bytes_hand_fixture_and_random():
    """integration §4.3 (C3): SHA-256 over float32 little-endian C-order bytes of U, W, b, V, c,
    concatenated without separators. Hand fixture via struct.pack (independent of numpy/torch
    byte paths); float64 parameters are rounded to nearest-even float32 (struct's C cast)."""
    U = np.array([[i * 0.5] for i in range(K)]); W = np.array([[-1.25]]); b = np.array([0.75])
    V = np.array([[-i / 4] for i in range(K)]); c = np.array([i / 8 for i in range(K)])
    model32 = build_model(Params(U, W, b, V, c), dtype=torch.float32)
    blob = (struct.pack("<27f", *U[:, 0]) + struct.pack("<f", W[0, 0]) + struct.pack("<f", b[0])
            + struct.pack("<27f", *V[:, 0]) + struct.pack("<27f", *c))
    assert diag.param_sha256(model32) == hashlib.sha256(blob).hexdigest()
    # Float64 parameters with values not representable in float32.
    P = perturbed_params(H=4, g=1.0, seed=81)
    model64 = build_model(P)
    blob = b"".join(struct.pack(f"<{a.size}f", *a.ravel(order="C").tolist()) for a in (P.U, P.W, P.b, P.V, P.c))
    assert diag.param_sha256(model64) == hashlib.sha256(blob).hexdigest()
    # Order matters: swapping two equal-shape tensors (U, V) changes the digest.
    swapped = b"".join(struct.pack(f"<{a.size}f", *a.ravel().tolist()) for a in (P.V, P.W, P.b, P.U, P.c))
    assert diag.param_sha256(model64) != hashlib.sha256(swapped).hexdigest()


# ============================================================================ 8. generation
def _gen_probs(z64: np.ndarray, temperature, top_k, top_p) -> torch.Tensor:
    """Oracle of the sampling distribution: float32 logits / T; top-k keeps the k largest; top-p keeps
    the smallest descending set whose cumulative probability reaches p; softmax."""
    z = torch.tensor(z64, dtype=torch.float32) / temperature
    if top_k is not None and top_k < z.shape[-1]:
        for r in range(z.shape[0]):
            kth = torch.sort(z[r], descending=True).values[top_k - 1]
            z[r][z[r] < kth] = float("-inf")
    if top_p is not None and top_p < 1.0:
        for r in range(z.shape[0]):
            order = torch.argsort(z[r], descending=True)
            p = torch.softmax(z[r][order], dim=0)
            cum, keep = 0.0, []
            for idx, pi in zip(order.tolist(), p.tolist()):
                if cum >= top_p:
                    break
                keep.append(idx); cum += pi
            mask = torch.ones_like(z[r], dtype=torch.bool); mask[keep] = False
            z[r][mask] = float("-inf")
    return torch.softmax(z, dim=-1)


def _gen_oracle(P: Params, num, max_length, seed, temperature=1.0, top_k=None, top_p=None):
    """Arch §15 procedure: h ← step(0, BOS); repeat: logits; draw for all active rows together;
    EOS → finished; row at max_length → stop (token discarded); else append, step. Finished rows
    leave the active set. Random draws: torch.multinomial(probs[active], 1) with the given generator,
    one call per round over the active rows in order (the primitive used by the shared sampler)."""
    gen = make_generator(seed) if seed is not None else None
    hs = [np.tanh(P.U[BOS] + P.b) for _ in range(num)]
    chars = [[] for _ in range(num)]
    finished = [False] * num
    active = list(range(num))
    while active:
        z = np.array([P.V @ hs[k] + P.c for k in active])
        if temperature == 0:
            draws = [int(np.argmax(r)) for r in z.astype(np.float32)]
        else:
            probs = _gen_probs(z, temperature, top_k, top_p)
            draws = torch.multinomial(probs, 1, generator=gen).squeeze(-1).tolist()
        nxt = []
        for k, tok in zip(active, draws):
            if tok == EOS:
                finished[k] = True
            elif len(chars[k]) == max_length:
                pass
            else:
                chars[k].append(tok)
                hs[k] = np.tanh(P.U[tok] + P.W @ hs[k] + P.b)
                nxt.append(k)
        active = nxt
    return ["".join(VOCAB.itos[i] for i in cs) for cs in chars], finished


@pytest.mark.parametrize("kw", [dict(temperature=1.0), dict(temperature=0.7, top_k=5),
                                dict(temperature=1.3, top_p=0.8), dict(temperature=1.0, top_k=8, top_p=0.9)],
                         ids=["t1", "t0.7k5", "t1.3p0.8", "k8p0.9"])
def test_generation_matches_sampler_oracle(kw):
    """Arch §15: ancestral sampling, EOS stop, max_length semantics, active-set removal, temperature,
    top-k, top-p; identical texts and finished flags to the oracle for a fixed generator seed;
    deterministic (same seed twice → identical); parameters and global RNG untouched."""
    P = perturbed_params(H=5, g=1.0, seed=91)
    P.c[0] = 1.5  # makes EOS common so both finished and truncated rows occur
    model = build_model(P)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng_state = torch.get_rng_state()
    out = model.generate(40, max_length=4, generator=make_generator(1234), **kw)
    texts, fin = _gen_oracle(P, 40, 4, 1234, **kw)
    assert [s.text for s in out] == texts
    assert [s.finished for s in out] == fin
    assert any(fin) and not all(fin)
    for s in out:
        assert "." not in s.text and len(s.text) <= 4 and (s.finished or len(s.text) == 4)
    again = model.generate(40, max_length=4, generator=make_generator(1234), **kw)
    assert [(s.text, s.finished) for s in again] == [(s.text, s.finished) for s in out]
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_generation_greedy_topk1_and_max_length_extremes():
    """temperature = 0 is greedy (argmax at each step, oracle replay); top_k = 1 equals greedy;
    a model that can never emit EOS gives exactly max_length characters, finished False; a model that
    always emits EOS gives '' with finished True."""
    P = perturbed_params(H=5, g=1.0, seed=92)
    model = build_model(P)
    greedy = model.generate(3, max_length=10, temperature=0)
    texts, fin = _gen_oracle(P, 3, 10, None, temperature=0)
    assert [s.text for s in greedy] == texts and [s.finished for s in greedy] == fin
    topk1 = model.generate(3, max_length=10, generator=make_generator(5), top_k=1)
    assert [(s.text, s.finished) for s in topk1] == [(s.text, s.finished) for s in greedy]
    P2 = P.copy(); P2.c[0] = -1e4
    for s in build_model(P2).generate(6, max_length=3, generator=make_generator(6)):
        assert len(s.text) == 3 and not s.finished
    P3 = P.copy(); P3.c[0] = 1e4
    for s in build_model(P3).generate(6, max_length=3, generator=make_generator(7)):
        assert s.text == "" and s.finished


def test_generation_first_character_distribution_statistical():
    """Distributional check independent of the RNG primitive: with max_length = 1, the frequency of
    the first draw (EOS counts as finished '') over 20,000 samples matches softmax(z_1); every class
    within 5 binomial standard errors + 1e-4 (false-alarm probability < 27·6e-7)."""
    P = perturbed_params(H=4, g=1.0, seed=93)
    model = build_model(P)
    n = 20000
    out = model.generate(n, max_length=1, generator=make_generator(77))
    p = np.exp(log_softmax(P.V @ np.tanh(P.U[BOS] + P.b) + P.c))
    counts = np.zeros(K)
    for s in out:
        counts[0 if s.text == "" else VOCAB.stoi[s.text]] += 1
        assert len(s.text) <= 1 and (s.text != "" or s.finished)
    freq = counts / n
    assert np.all(np.abs(freq - p) <= 5 * np.sqrt(p * (1 - p) / n) + 1e-4)


# ============================================================================ 9. training and evaluation
def test_loss_normaliser_arithmetic_from_approved_counts():
    """training §1, §4.1: N = 25,640, M = 182,666, B = 32 → C = B·M/N = 730,664/3,205 = 227.976287…;
    S = ⌈N/B⌉ = 802 = 801 full + 1 batch of 8. loss_normalizer reproduces it on a synthetic split with
    exactly these counts (no data read)."""
    N, M, B = 25_640, 182_666, 32
    C = Fraction(B * M, N)
    assert C == Fraction(730_664, 3_205)
    assert f"{float(C):.6f}" == "227.976287"
    assert -(-N // B) == 802 and N - 801 * B == 8
    rng = np.random.default_rng(101)
    lengths = np.full(N, (M - N) // N)  # L such that Σ(L + 1) = M
    extra = (M - N) - int(lengths.sum())
    lengths[rng.choice(N, size=extra, replace=False)] += 1
    assert int(lengths.sum()) + N == M and lengths.max() <= 15
    train = make_next_token_sequences(random_names(rng, lengths.tolist()), VOCAB)
    got = rtrain.loss_normalizer(rtrain.RNNTrainingConfig(lr=0.1, max_epochs=1), train)
    assert got == float(C)  # B·M/N on exact integers is correctly rounded


@pytest.mark.parametrize("dtype,tol", [(torch.float64, TOL_A), (torch.float32, 1e-5)])
def test_evaluate_nll_matches_oracle_and_is_chunk_independent(dtype, tol):
    """T6 / training §8.2: evaluate_nll = Σ oracle NLL / #targets (A in float64, B in float32) for
    every chunk size; token count equals Σ n_β."""
    P = perturbed_params(H=8, g=1.0, seed=111)
    model = build_model(P, dtype=dtype)
    names = random_names(np.random.default_rng(112), [1, 2, 9, 4, 15, 3, 3, 7, 11, 5])
    seq = make_next_token_sequences(names, VOCAB)
    total = math.fsum(oracle_name_nll(P, n).sum() for n in names)
    count = sum(len(n) + 1 for n in names)
    for bs in (1, 3, 4, 1024):
        acc = rtrain.evaluate_nll(model, seq, batch_size=bs)
        assert acc.num_tokens == count
        assert_close(acc.mean_nll, total / count, atol=tol, rtol=tol, what=f"bs={bs}")


def _seqs(names):
    return make_next_token_sequences(names, VOCAB)


def test_one_epoch_sgd_steps_follow_option_B_with_C_equal_BM_over_N():
    """training §4–§5, T10–T11, T21: with plain SGD, each step is θ ← θ − η·∇(Σ_batch ℓ)/C with
    C = B·M/N for every batch including the partial final batch (weight 1/C, not 1/m_b), and the
    per-batch trimming does not matter. N = 40 names, B = 32 → batches of 32 and 8. The expected
    trajectory replays the shared EpochSampler order with the oracle BPTT gradients (A, float64)."""
    rng = np.random.default_rng(121)
    names = random_names(rng, rng.integers(1, 12, 40).tolist())
    dev = random_names(rng, [3, 5, 2])
    P = perturbed_params(H=4, g=1.0, seed=122)
    model = build_model(P)
    cfg = rtrain.RNNTrainingConfig(lr=0.3, max_epochs=1, batch_size=32, eval_batch_size=16)
    result = rtrain.train_rnn(model, _seqs(names), _seqs(dev), training=cfg, seed=9)
    N, M = len(names), sum(len(n) + 1 for n in names)
    C = 32 * M / N
    assert result.loss_normalizer == C
    Q = P.copy()
    sizes = []
    for idx in EpochSampler(N, 32, seed=derive_seed(9, SAMPLER_STREAM)).batches(0):
        batch = [names[i] for i in idx.tolist()]
        sizes.append(len(batch))
        G = oracle_grads(Q, batch)
        for k in ("U", "W", "b", "V", "c"):
            setattr(Q, k, getattr(Q, k) - 0.3 * G[k] / C)
    assert sizes == [32, 8]
    got = model_params(model)
    for k in ("U", "W", "b", "V", "c"):
        assert_close(getattr(got, k), getattr(Q, k), what=f"param {k} after one epoch")
    # Epoch rows: k = 0 (not selectable) and k = 1, with train/dev NLL equal to the oracle (A).
    h0, h1 = result.history
    assert h0["completed_epochs"] == 0 and h0["selectable"] is False and h0["running_train_nll"] is None
    assert_close(h0["dev_nll"], math.fsum(oracle_name_nll(P, n).sum() for n in dev) / sum(len(n) + 1 for n in dev))
    assert_close(h1["train_nll"], math.fsum(oracle_name_nll(Q, n).sum() for n in names) / M)
    # Category D (training §12.1): diverged iff full-train NLL > ln 27 at k ≥ 1.
    assert h1["diverged"] == (h1["train_nll"] > LN27)
    assert result.status == ("failed_diverged" if h1["train_nll"] > LN27 else "completed")


def _tiny_train_setup(seed=131):
    rng = np.random.default_rng(seed)
    names = random_names(rng, rng.integers(1, 8, 30).tolist())
    dev = random_names(rng, rng.integers(1, 8, 10).tolist())
    return names, dev


def _run(tmp, names, dev, *, seed=3, epochs=3, lr=0.5):
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=8), generator=make_generator(derive_seed(seed, "init")))
    run = RunManager.create("rnn", config={"test": True}, results_root=tmp)
    cfg = rtrain.RNNTrainingConfig(lr=lr, max_epochs=epochs, batch_size=8, eval_batch_size=7)
    res = rtrain.train_rnn(model, _seqs(names), _seqs(dev), training=cfg, seed=seed, run=run)
    return model, run, res


def test_train_rnn_deterministic_and_selection_rules(tmp_path):
    """T2 / T14 / training §8.5, §10.3–10.4: two runs with the same seed are bitwise identical
    (parameters, history except wall time, model_best.pt); best = argmin over k ≥ 1 of dev NLL by
    strict < (earliest on ties); k = 0 never selected; model_best.pt holds that epoch, and its
    parameter hash (recomputed here) equals the history hash of that epoch."""
    names, dev = _tiny_train_setup()
    m1, run1, r1 = _run(tmp_path / "a", names, dev)
    m2, run2, r2 = _run(tmp_path / "b", names, dev)
    for (n1, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), n1
    strip = lambda h: [{k: v for k, v in row.items() if k != "seconds"} for row in h]  # noqa: E731
    assert strip(r1.history) == strip(r2.history)
    hist = r1.history
    assert [h["completed_epochs"] for h in hist] == [0, 1, 2, 3]
    assert hist[0]["selectable"] is False and hist[0]["is_best"] is False
    best, best_v = None, math.inf
    for h in hist[1:]:
        if h["dev_nll"] < best_v:
            best, best_v = h["completed_epochs"], h["dev_nll"]
        assert h["is_best"] == (best == h["completed_epochs"] and best_v == h["dev_nll"])
    assert r1.best_epoch == best and r1.best_dev_nll == best_v
    ck = load_checkpoint(run1.checkpoint_path("best"))
    assert ck["epoch"] == best
    loaded = RNNLanguageModel.from_checkpoint(run1.checkpoint_path("best"))
    assert _my_hash(loaded) == hist[best]["param_sha256"] == ck["extra"]["param_sha256"]
    assert _my_hash(m1) == hist[-1]["param_sha256"]
    ck2 = load_checkpoint(run2.checkpoint_path("best"))
    for k in ck["model_state"]:
        assert torch.equal(ck["model_state"][k], ck2["model_state"][k])


def _my_hash(model) -> str:
    blob = b""
    for n in ("input_lookup", "recurrent_weight", "hidden_bias", "output_weight", "output_bias"):
        a = getattr(model, n).detach().double().numpy()
        blob += struct.pack(f"<{a.size}f", *a.ravel().tolist())
    return hashlib.sha256(blob).hexdigest()


def test_epoch_zero_never_selected_even_when_lowest(tmp_path):
    """training §8.3, §8.5: the epoch-0 row is never selectable. Fixture: train on 'a…' names, dev on
    'z…' names, so dev NLL rises above its epoch-0 value (ln 27) while train NLL falls; the best
    epoch must still be some k ≥ 1 and model_best.pt must exist for it."""
    train = ["aaa", "aa", "aaaa", "a"] * 4
    dev = ["zzz", "zz"]
    model, run, res = _run(tmp_path, train, dev, epochs=2, lr=1.0)
    h = res.history
    assert abs(h[0]["dev_nll"] - LN27) < 1e-5
    assert h[0]["dev_nll"] < min(r["dev_nll"] for r in h[1:]), "fixture: epoch 0 must be lowest"
    assert res.status == "completed"
    assert res.best_epoch == min(range(1, 3), key=lambda k: (h[k]["dev_nll"], k))
    assert load_checkpoint(run.checkpoint_path("best"))["epoch"] == res.best_epoch


def test_selection_strict_less_keeps_earliest_tie(monkeypatch):
    """T3 / T14: injected dev NLLs [1.0, 2.0, 1.5, 1.5, 1.6] for k = 0..4: k = 0 is excluded, and
    strict < keeps k = 2 over the exact tie at k = 3."""
    names, dev = _tiny_train_setup(132)
    dev_seq = _seqs(dev)
    dev_values = iter([1.0, 2.0, 1.5, 1.5, 1.6])
    real = rtrain.evaluate_nll

    class Fake:
        def __init__(self, v):
            self.mean_nll = v

    def fake(model, sequences, *, batch_size):
        if sequences is dev_seq:
            return Fake(next(dev_values))
        return Fake(1.0)

    monkeypatch.setattr(rtrain, "evaluate_nll", fake)
    model = RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=4), generator=make_generator(1))
    res = rtrain.train_rnn(model, _seqs(names), dev_seq,
                           training=rtrain.RNNTrainingConfig(lr=0.1, max_epochs=4, batch_size=16), seed=1)
    assert real is not fake
    assert res.best_epoch == 2 and res.best_dev_nll == 1.5
    assert [h["is_best"] for h in res.history] == [False, True, True, False, False]


# ============================================================================ 10. edge cases
def test_edge_bos_only_rows_single_row_equal_lengths_empty_batch():
    """n = 1 rows (only BOS, a single EOS target), B = 1, all-equal lengths, the maximum length in the
    batch (15 letters, n = 16) and B = 0 (returns [0, T, 27])."""
    P = perturbed_params(H=7, g=1.2, seed=141)
    model = build_model(P, g=1.2)
    with torch.no_grad():
        z = model(torch.tensor([[0, PAD, PAD], [0, 1, PAD]]))
    hs, _ = oracle_states(P, [0])
    assert_close(z[0, 0].numpy(), oracle_logits(P, hs)[0])
    assert_close(z[0, 2].numpy(), oracle_logits(P, hs)[0])  # frozen-state padded logits
    one = ["mmmmmmmmmmmmmmm"]
    with torch.no_grad():
        z1 = model(batch_tensor(one))
    assert z1.shape == (1, 16, K)
    assert_close(z1[0].numpy(), oracle_logits(P, oracle_states(P, ids_of(one[0])[0])[0]))
    eq = ["abc", "xyz", "qqq"]
    with torch.no_grad():
        ze = model(batch_tensor(eq))
    for i, nm in enumerate(eq):
        assert_close(ze[i].numpy(), oracle_logits(P, oracle_states(P, ids_of(nm)[0])[0]))
    with torch.no_grad():
        empty = model(torch.zeros((0, 5), dtype=torch.long))
    assert empty.shape == (0, 5, K) and empty.dtype == torch.float64
    # a single-name length-1 evaluation: two targets ('a', EOS)
    seq = _seqs(["a"])
    acc = rtrain.evaluate_nll(model, seq, batch_size=1)
    assert_close(acc.mean_nll, oracle_name_nll(P, "a").mean())


def test_float64_and_float32_models_agree_and_large_gain_stays_finite():
    """arch §5, §10: a float64 model returns float64 logits (A vs oracle, above); a float32 model
    with the same parameters agrees with the float64 oracle within C = 1e-5 at g = 1; a large-gain
    fixture (g = 50, U ×10) gives finite logits and |h| ≤ 1."""
    P = perturbed_params(H=16, g=1.0, seed=151)
    m32 = build_model(P, dtype=torch.float32)
    with torch.no_grad():
        z = m32(batch_tensor(FWD_NAMES))
    assert z.dtype == torch.float32
    for i, nm in enumerate(FWD_NAMES):
        n = len(nm) + 1
        assert_close(z[i, :n].double().numpy(), oracle_logits(P, oracle_states(P, ids_of(nm)[0])[0]),
                     atol=1e-5, rtol=1e-5)
    Pb = perturbed_params(H=16, g=50.0, seed=152)
    Pb.U *= 10
    mb = build_model(Pb, g=50.0)
    with torch.no_grad():
        rec = mb._recurrence(batch_tensor(FWD_NAMES, T=30))
        zb = mb(batch_tensor(FWD_NAMES, T=30))
    assert torch.isfinite(zb).all() and torch.all(rec.states.abs() <= 1.0)

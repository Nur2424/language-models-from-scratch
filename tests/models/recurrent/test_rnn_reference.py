"""MS1 tests of the independent Model 6 reference (``rnn_reference.py``).

Milestone MS1 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (approved, D-49);
``rnn_spec.md`` v0.2 (D-44) §4, §8, §12.3 – §12.5, §13.1 – §13.2.

No production code exists or is imported. In MS1 the A:M1 – M5 checks therefore verify the
reference's **hand-derived formulas** against two methods that do not use those formulas:
autograd applied to the reference's explicit forward pass, and central finite differences.
The same checks are applied to the production model in MS2 / MS3.

Method labels used in test names and docstrings
-----------------------------------------------
* **[hand]** expected values computed in plain Python floats (``math``) from the §4.1 equations;
* **[analytic]** an exact identity of the mathematics, checked numerically;
* **[autograd]** PyTorch autograd on the reference's explicit forward pass (an independent route
  to the same derivatives as the hand-derived §4.4 formulas);
* **[library]** an independent PyTorch library route (``@``, ``log_softmax``, ``cross_entropy``)
  for the same forward quantity; the reference itself uses none of these;
* **[FD]** central finite differences (numerical approximation);
* **[invariant]** structural properties: padding, freezing, batch composition, causality,
  determinism.

Tolerances (§13.1)
------------------
* **A**: |x − x_ref| ≤ 1e‑9 + 1e‑9·|x_ref|, float64 (rigorous rounding bound ≲ 5e‑10 for
  T ≤ 16, H ≤ 128, g ≤ 1.2).
* **FD**: step 1e‑6, rtol 1e‑6, atol 1e‑8 (truncation O(h²) ≈ 1e‑12; rounding ≈ u·|L|/h ≈
  1e‑10·|L|; fixture losses are O(10), so the error is ≲ 1e‑9).
* **exact**: bitwise; used where both sides run the identical operations.

Fixtures follow the fixture rule M6-IMP9 (§13.1): **perturbed, non-zero parameters**
(U ~ N(0, 0.5), W = g·Q with Q Haar, b ~ N(0, 0.1²), V ~ N(0, 1/H), c ~ N(0, 0.1²)), at
g = 1 and g = 1.2, so no test can pass because V = c = 0 makes every logit zero. Names are
the synthetic names of ``conftest.py`` with at most 15 characters (n ≤ 16, the data's maximum).
"""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path

import pytest
import torch

import tests.models.recurrent.rnn_reference as ref
from tests.models.recurrent.conftest import SYNTHETIC_NAMES

TOL_A = dict(atol=1e-9, rtol=1e-9)
FD_STEP, FD_RTOL, FD_ATOL = 1e-6, 1e-6, 1e-8
GAINS = (1.0, 1.2)
REFERENCE_FILE = Path(ref.__file__).resolve()
PAD = 27

#: Synthetic names with n = L + 1 ≤ 16 (the tolerance-A bound assumes T ≤ 16).
NAMES = tuple(n for n in SYNTHETIC_NAMES if len(n) <= 15)


# --------------------------------------------------------------------------- helpers
def assert_close_A(actual: torch.Tensor, expected: torch.Tensor, what: str = "") -> None:
    assert actual.dtype == torch.float64 and expected.dtype == torch.float64, what
    assert actual.shape == expected.shape, (what, actual.shape, expected.shape)
    bound = TOL_A["atol"] + TOL_A["rtol"] * expected.abs()
    worst = float(((actual - expected).abs() - bound).max()) if actual.numel() else -1.0
    assert worst <= 0.0, f"{what}: tolerance A exceeded by {worst:.3e}"


def encode(name: str) -> list[int]:
    """[BOS] + letter ids (a = 1 … z = 26): the project's vocabulary convention (§4.0)."""
    return [0] + [ord(ch) - ord("a") + 1 for ch in name]


def padded_batch(names: list[str] | tuple[str, ...], width: int | None = None) -> torch.Tensor:
    rows = [encode(n) for n in names]
    T = max((len(r) for r in rows), default=1) if width is None else width
    return torch.tensor([r + [PAD] * (T - len(r)) for r in rows], dtype=torch.long).reshape(len(rows), T)


def haar(H: int, gen: torch.Generator) -> torch.Tensor:
    """Haar-distributed orthogonal Q: QR of a Gaussian matrix with sign(diag R) folded into Q."""
    A = torch.randn(H, H, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def perturbed_params(H: int, g: float, gen: torch.Generator, K: int = 27) -> ref.Params:
    """Fixture rule M6-IMP9 (§13.1)."""
    W = g * haar(H, gen)
    U = math.sqrt(0.5) * torch.randn(K, H, generator=gen, dtype=torch.float64)
    b = 0.1 * torch.randn(H, generator=gen, dtype=torch.float64)
    V = torch.randn(K, H, generator=gen, dtype=torch.float64) / math.sqrt(H)
    c = 0.1 * torch.randn(K, generator=gen, dtype=torch.float64)
    return ref.Params(U, W, b, V, c)


def leaf_params(p: ref.Params) -> ref.Params:
    return ref.Params(*(t.detach().clone().requires_grad_(True) for t in p.as_tuple()))


@pytest.fixture(params=GAINS, ids=lambda g: f"g{g}")
def gain(request) -> float:
    return float(request.param)


@pytest.fixture
def params(tiny_hidden_dim, gain, generator_factory) -> ref.Params:
    return perturbed_params(tiny_hidden_dim, gain, generator_factory(int(10 * gain) + tiny_hidden_dim))


# --------------------------------------------------------------------------- fixture guard
def test_fixture_parameters_are_non_trivial(params) -> None:
    """[invariant] Guards against vacuous tests: V and c are non-zero, logits vary across classes
    and positions, and W has the requested gain (all singular values equal g)."""
    assert float(params.V.abs().min()) > 0 and float(params.c.abs().max()) > 0
    fwd = ref.forward_batch(params, padded_batch(NAMES))
    z = fwd.logits[fwd.valid]
    assert float(z.std(dim=1).min()) > 1e-2
    sv = torch.linalg.svdvals(params.W)
    g = float(sv[0])
    assert torch.allclose(sv, torch.full_like(sv, g), atol=1e-12)
    assert any(abs(g - x) <= 1e-12 for x in GAINS), g


# --------------------------------------------------------------------------- [hand] values
def test_hand_two_step_example_H2_K3() -> None:
    """[hand] §13.2 R10 form: H = 2, K = 3, input [BOS, 2] (n = 2, targets [2, 0]).
    Every a_t, h_t, z_t and ℓ_t is recomputed with Python floats."""
    U = [[0.5, -0.3], [0.1, 0.2], [-0.7, 0.4]]
    W = [[0.9, -0.2], [0.3, 0.8]]
    b = [0.05, -0.1]
    V = [[0.6, -0.4], [-0.2, 0.7], [0.3, 0.3]]
    c = [0.01, -0.02, 0.03]
    p = ref.Params(*(torch.tensor(x, dtype=torch.float64) for x in (U, W, b, V, c)))

    def tanh_v(v):
        return [math.tanh(x) for x in v]

    h0 = [0.0, 0.0]
    a1 = [U[0][i] + sum(W[i][j] * h0[j] for j in range(2)) + b[i] for i in range(2)]
    h1 = tanh_v(a1)
    a2 = [U[2][i] + sum(W[i][j] * h1[j] for j in range(2)) + b[i] for i in range(2)]
    h2 = tanh_v(a2)
    z1 = [sum(V[k][j] * h1[j] for j in range(2)) + c[k] for k in range(3)]
    z2 = [sum(V[k][j] * h2[j] for j in range(2)) + c[k] for k in range(3)]

    def nll(z, y):
        m = max(z)
        return -(z[y] - (m + math.log(sum(math.exp(v - m) for v in z))))

    f = ref.forward_name(p, [0, 2])
    assert f.targets == [2, 0]
    t64 = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    assert_close_A(f.pre, t64([a1, a2]), "pre-activations")
    assert_close_A(f.states, t64([h1, h2]), "states")
    assert_close_A(f.logits, t64([z1, z2]), "logits")
    assert_close_A(f.losses, t64([nll(z1, 2), nll(z2, 0)]), "losses")


def test_hand_scalar_gradient_of_W() -> None:
    """[hand] H = 1, K = 2, input [BOS, 1], targets [1, 0]. From §4.4 by hand:
    δ_2 = r_2, δ_1 = r_1 + w(1 − h_2²) r_2, ∂L/∂w = (1 − h_2²) r_2 h_1 (the t = 1 term is zero)."""
    u0, u1, w, b = 0.4, -0.8, 1.1, 0.05
    v, c = [0.9, -0.6], [0.1, -0.1]
    h1 = math.tanh(u0 + b)
    h2 = math.tanh(u1 + w * h1 + b)

    def softmax(z):
        m = max(z)
        e = [math.exp(x - m) for x in z]
        return [x / sum(e) for x in e]

    p2 = softmax([v[0] * h2 + c[0], v[1] * h2 + c[1]])
    r2 = v[0] * (p2[0] - 1.0) + v[1] * p2[1]  # target y_2 = 0 (EOS)
    dW = (1 - h2 * h2) * r2 * h1
    p = ref.Params(
        torch.tensor([[u0], [u1]], dtype=torch.float64), torch.tensor([[w]], dtype=torch.float64),
        torch.tensor([b], dtype=torch.float64), torch.tensor([[v[0]], [v[1]]], dtype=torch.float64),
        torch.tensor(c, dtype=torch.float64),
    )
    grads = ref.bptt_gradients(p, torch.tensor([[0, 1]], dtype=torch.long))
    assert abs(float(grads["W"][0, 0]) - dW) <= 1e-9 + 1e-9 * abs(dW)


# --------------------------------------------------------------------------- [library] forward
def test_states_satisfy_the_recurrence_library_route(params) -> None:
    """[library] Every valid state satisfies h_t = tanh(U[x_t] + W h_{t−1} + b), recomputed with
    ``@`` from the reference's own h_{t−1} (one-step check, tolerance A)."""
    fwd = ref.forward_batch(params, padded_batch(NAMES))
    for f in fwd.names:
        prev = torch.zeros(params.H, dtype=torch.float64)
        for t in range(1, f.n + 1):
            a = params.U[f.inputs[t - 1]] + params.W @ prev + params.b
            assert_close_A(f.pre[t - 1], a, f"a_{t}")
            assert_close_A(f.states[t - 1], torch.tanh(a), f"h_{t}")
            prev = f.states[t - 1]


def test_full_sequence_against_an_independent_batched_formulation(params) -> None:
    """[library] Whole-sequence check: a separate batched formulation in this test (row-vector
    orientation h @ Wᵀ, masked selection) reproduces all valid states and all logits, including
    padded-position logits, at tolerance A. It shares no code with the reference."""
    inputs = padded_batch(NAMES, width=19)
    valid = inputs != PAD
    x_hat = torch.where(valid, inputs, torch.zeros_like(inputs))
    h = torch.zeros(inputs.shape[0], params.H, dtype=torch.float64)
    states = []
    for t in range(inputs.shape[1]):
        cand = torch.tanh(params.U[x_hat[:, t]] + h @ params.W.T + params.b)
        h = torch.where(valid[:, t].unsqueeze(1), cand, h)
        states.append(h)
    states = torch.stack(states, dim=1)
    logits = states @ params.V.T + params.c
    fwd = ref.forward_batch(params, inputs)
    assert torch.equal(fwd.valid, valid)
    assert_close_A(fwd.states, states, "states (all positions)")
    assert_close_A(fwd.logits, logits, "logits (all positions)")


def test_losses_match_library_cross_entropy(params) -> None:
    """[library] ℓ_t equals ``cross_entropy`` on the reference's logits; the batch total equals
    the sum over valid targets (padded targets −100 ignored)."""
    inputs = padded_batch(NAMES)
    fwd = ref.forward_batch(params, inputs)
    lib = torch.nn.functional.cross_entropy(
        fwd.logits.reshape(-1, params.K), fwd.targets.reshape(-1), ignore_index=-100, reduction="sum")
    assert_close_A(ref.total_loss(params, inputs), lib, "total loss")
    for f in fwd.names:
        lp = torch.log_softmax(f.logits, dim=1)
        assert_close_A(f.log_probs, lp, "log-probabilities")
        assert_close_A(f.losses, -lp[torch.arange(f.n), torch.tensor(f.targets)], "per-target losses")


# --------------------------------------------------------------------------- [invariant] padding etc.
def test_padding_freezes_state_and_defines_padded_logits(params) -> None:
    """[invariant] §4.1: for t > n_β, h_t = h_{n_β} and z_t = V h_{n_β} + c = z_{n_β} (bitwise:
    same operations on the same vector); padded pre-activations are NaN by construction and
    padded targets are −100."""
    fwd = ref.forward_batch(params, padded_batch(NAMES, width=20))
    for beta, n in enumerate(fwd.lengths):
        for t in range(n + 1, 21):
            assert torch.equal(fwd.states[beta, t - 1], fwd.states[beta, n - 1])
            assert torch.equal(fwd.logits[beta, t - 1], fwd.logits[beta, n - 1])
            assert torch.isnan(fwd.pre[beta, t - 1]).all()
            assert int(fwd.targets[beta, t - 1]) == -100 and not fwd.valid[beta, t - 1]
        assert torch.isfinite(fwd.logits[beta]).all()


def test_padding_width_invariance(params) -> None:
    """[invariant] Valid states, pre-activations and logits do not depend on the padded width T."""
    base = ref.forward_batch(params, padded_batch(NAMES))
    for width in (17, 24):
        wide = ref.forward_batch(params, padded_batch(NAMES, width=width))
        for beta, n in enumerate(base.lengths):
            assert torch.equal(wide.states[beta, :n], base.states[beta, :n])
            assert torch.equal(wide.pre[beta, :n], base.pre[beta, :n])
            assert torch.equal(wide.logits[beta, :n], base.logits[beta, :n])


def test_batch_composition_invariance(params) -> None:
    """[invariant] A name's outputs are the same alone, first, last, or among other names."""
    target = "ludmeriaxo"
    alone = ref.forward_batch(params, padded_batch([target]))
    n = alone.lengths[0]
    for batch in (["a", target], [target, "zo", "fennwick"], list(NAMES)):
        fwd = ref.forward_batch(params, padded_batch(batch))
        beta = batch.index(target)
        assert torch.equal(fwd.states[beta, :n], alone.states[0, :n])
        assert torch.equal(fwd.logits[beta, :n], alone.logits[0, :n])


def test_variable_length_batch_lengths_and_targets(params) -> None:
    """[invariant] n_β = L_β + 1; targets are the next ids followed by EOS (0) (§4.0, 'emma' example)."""
    fwd = ref.forward_batch(params, padded_batch(["emma", "a", "obelandrinas"]))
    assert fwd.lengths == [5, 2, 13]
    assert fwd.targets[0, :6].tolist() == [5, 13, 13, 1, 0, -100]
    assert fwd.targets[1, :3].tolist() == [1, 0, -100]
    assert fwd.valid.sum(dim=1).tolist() == [5, 2, 13]


def test_causality(params) -> None:
    """[invariant] §4.4: changing x_{t'} for t' > t leaves z_1 … z_t unchanged (bitwise)."""
    base = ref.forward_name(params, encode("kestravelunio"))
    for t_change in (3, 7, 12):
        ids = encode("kestravelunio")
        ids[t_change - 1] = (ids[t_change - 1] % 26) + 1  # a different letter
        alt = ref.forward_name(params, ids)
        assert torch.equal(alt.logits[: t_change - 1], base.logits[: t_change - 1])
        assert not torch.equal(alt.logits[t_change - 1], base.logits[t_change - 1])


def test_determinism_and_no_rng_use(params) -> None:
    """[invariant] Identical calls are bitwise identical; the global RNG is not touched."""
    inputs = padded_batch(NAMES)
    state = torch.random.get_rng_state()
    a, b = ref.forward_batch(params, inputs), ref.forward_batch(params, inputs)
    assert torch.equal(a.states, b.states) and torch.equal(a.logits, b.logits)
    assert torch.equal(torch.nan_to_num(a.pre, nan=7.0), torch.nan_to_num(b.pre, nan=7.0))
    ga, gb = ref.bptt_gradients(params, inputs), ref.bptt_gradients(params, inputs)
    assert all(torch.equal(ga[k], gb[k]) for k in ga)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_bos_only_row_and_empty_batch(params) -> None:
    """[invariant] A row with only BOS (n = 1) is legal (§5.1); B = 0 returns empty outputs."""
    fwd = ref.forward_batch(params, torch.tensor([[0, PAD, PAD]], dtype=torch.long))
    assert fwd.lengths == [1] and fwd.targets[0].tolist() == [0, -100, -100]
    assert torch.equal(fwd.states[0, 1], fwd.states[0, 0])
    empty = ref.forward_batch(params, torch.zeros(0, 4, dtype=torch.long))
    assert empty.logits.shape == (0, 4, 27) and empty.states.shape == (0, 4, params.H)
    assert float(ref.total_loss(params, torch.zeros(0, 4, dtype=torch.long))) == 0.0


@pytest.mark.parametrize(
    ("inputs", "error"),
    [
        ([[1, 2, PAD]], ValueError),          # column 0 is not BOS
        ([[0, 28, PAD]], ValueError),         # id out of range
        ([[0, -1, PAD]], ValueError),         # negative id
        ([[0, PAD, 3]], ValueError),          # padding not right-contiguous
    ],
    ids=["no-bos", "id-28", "negative", "non-contiguous-pad"],
)
def test_reference_rejects_invalid_rows(params, inputs, error) -> None:
    """[invariant] The reference refuses inputs the §5.1 contract forbids, so that no test can
    compare against a reference computed on meaningless input."""
    with pytest.raises(error):
        ref.forward_batch(params, torch.tensor(inputs, dtype=torch.long))


def test_reference_rejects_bad_types_and_shapes(params) -> None:
    with pytest.raises(TypeError):
        ref.forward_batch(params, [[0, 1]])
    with pytest.raises(ValueError):
        ref.forward_batch(params, torch.tensor([[0, 1]], dtype=torch.int32))
    with pytest.raises(ValueError):
        ref.forward_batch(params, torch.tensor([0, 1], dtype=torch.long))
    with pytest.raises(ValueError):
        ref.forward_batch(params, torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(TypeError):  # float32 parameters: the reference is float64 only
        ref.Params(*(t.float() for t in params.as_tuple()))
    with pytest.raises(ValueError):
        ref.Params(params.U, params.W[:, :-1], params.b, params.V, params.c)


# --------------------------------------------------------------------------- initialised-model invariants
def test_zero_output_layer_invariants(generator_factory) -> None:
    """[analytic] M6-E4 (§4.5): with V = c = 0 every logit is 0, the loss is n·ln K, the
    gradients of U, W and b are exactly zero, and h_1 is identical for every name."""
    p0 = perturbed_params(8, 1.0, generator_factory(5))
    p = ref.Params(p0.U, p0.W, p0.b, torch.zeros_like(p0.V), torch.zeros_like(p0.c))
    inputs = padded_batch(NAMES)
    fwd = ref.forward_batch(p, inputs)
    assert torch.count_nonzero(fwd.logits) == 0
    n_total = sum(fwd.lengths)
    expected = n_total * math.log(27)  # every target has p = 1/27
    assert abs(float(ref.total_loss(p, inputs)) - expected) <= 1e-12 * expected
    grads = ref.bptt_gradients(p, inputs)
    for k in ("U", "W", "b"):
        assert torch.count_nonzero(grads[k]) == 0, k
    assert all(torch.equal(fwd.states[i, 0], fwd.states[0, 0]) for i in range(len(NAMES)))


# --------------------------------------------------------------------------- A:M1 Jacobian products
def test_M1_jacobian_products_match_autograd(params) -> None:
    """[analytic vs autograd] P_{t,s} (product of D_k W, §12.3) equals the autograd Jacobian of the
    map h_s ↦ h_t through the reference's ``step`` chain, for every pair 0 ≤ s < t ≤ n."""
    f = ref.forward_name(params, encode("corvinalt"))
    for s in range(0, f.n):
        h_s = torch.zeros(params.H, dtype=torch.float64) if s == 0 else f.states[s - 1]
        for t in range(s + 1, f.n + 1):

            def chain(h: torch.Tensor, s=s, t=t) -> torch.Tensor:
                for k in range(s + 1, t + 1):
                    h, _ = ref.step(params, h, f.inputs[k - 1])
                return h

            J = torch.autograd.functional.jacobian(chain, h_s.clone())
            assert_close_A(ref.jacobian_product(params, f, t, s), J, f"P_{{{t},{s}}}")
        assert torch.equal(ref.jacobian_product(params, f, s, s), torch.eye(params.H, dtype=torch.float64))


# --------------------------------------------------------------------------- A:M2 G2 vectors
def test_M2_loss_projected_vectors_match_autograd(params) -> None:
    """[analytic vs autograd] u_{t,s} = P_{t,s}ᵀ r_t with r_t = Vᵀ(p_t − e_{y_t}) (§12.4) equals
    ∂ℓ_t/∂h_s from autograd, for every 1 ≤ s ≤ t ≤ n; and R_{t,s} ≤ σ_{t,s} (§12.4)."""
    f = ref.forward_name(params, encode("wyndamorphelus"))
    for s in range(1, f.n + 1):
        for t in range(s, f.n + 1):
            h = f.states[s - 1].detach().clone().requires_grad_(True)
            h_t = h
            for k in range(s + 1, t + 1):
                h_t, _ = ref.step(params, h_t, f.inputs[k - 1])
            z = ref.output(params, h_t)
            loss = -torch.log_softmax(z, dim=0)[f.targets[t - 1]]
            (grad,) = torch.autograd.grad(loss, h)
            u = ref.loss_projected_vector(params, f, t, s)
            assert_close_A(u, grad, f"u_{{{t},{s}}}")
            r = ref.hidden_residual(params, f, t)
            sigma = float(torch.linalg.matrix_norm(ref.jacobian_product(params, f, t, s), ord=2))
            assert float(u.norm()) <= sigma * float(r.norm()) * (1 + 1e-12) + 1e-15


# --------------------------------------------------------------------------- A:M3 BPTT gradients
@pytest.mark.parametrize("width", [None, 21], ids=["tight", "padded"])
def test_M3_bptt_gradients_match_autograd(params, width) -> None:
    """[analytic vs autograd] The §4.4 BPTT formulas for U, W, b, V, c equal autograd gradients of
    the unnormalised batch loss through the reference's explicit forward pass. Padding columns
    contribute nothing (same result with extra padding)."""
    inputs = padded_batch(NAMES, width=width)
    analytic = ref.bptt_gradients(params, inputs)
    leaves = leaf_params(params)
    ref.total_loss(leaves, inputs).backward()
    for name, leaf in zip("UWbVc", leaves.as_tuple()):
        assert_close_A(analytic[name], leaf.grad, f"∂L/∂{name}")


def test_M3_bptt_at_full_width_H128(generator_factory) -> None:
    """[analytic vs autograd] The same identity at the approved width H = 128, g = 1.2 (the
    largest gain of the tolerance-A fixtures), on a few names of different lengths."""
    params = perturbed_params(128, 1.2, generator_factory(128))
    inputs = padded_batch(["a", "mavra", "kestravelunio"])
    analytic = ref.bptt_gradients(params, inputs)
    leaves = leaf_params(params)
    ref.total_loss(leaves, inputs).backward()
    for name, leaf in zip("UWbVc", leaves.as_tuple()):
        assert_close_A(analytic[name], leaf.grad, f"∂L/∂{name} (H = 128)")


# --------------------------------------------------------------------------- A:M4 G3 decomposition
def test_M4_g3_terms_sum_to_the_recurrent_gradient(params) -> None:
    """[analytic vs autograd] §12.5: Σ T_{t,k} equals ∂(Σℓ)/∂W from autograd and from BPTT;
    Σ_Δ π(Δ) = 1; ‖T_{t,k}‖_F = ‖D_k u_{t,k}‖·‖h_{k−1}‖ (rank one); terms exist for Δ = 0 … n − 2."""
    inputs = padded_batch(NAMES, width=18)
    dec = ref.g3_decomposition(params, inputs)
    leaves = leaf_params(params)
    ref.total_loss(leaves, inputs).backward()
    assert_close_A(dec.G, leaves.W.grad, "Σ T vs autograd")
    assert_close_A(dec.G, ref.bptt_gradients(params, inputs)["W"], "Σ T vs BPTT")
    assert 1e-3 <= dec.kappa <= 1.0, f"κ_c = {dec.kappa} (criterion A applies while κ_c ≥ 1e-3)"
    total_share = math.fsum(dec.share.values())
    assert abs(total_share - 1.0) <= 1e-9, (total_share, dec.kappa)
    max_n = max(len(n) + 1 for n in NAMES)
    assert sorted(dec.share) == list(range(0, max_n - 1))
    f = ref.forward_name(params, encode("tolkie"))
    one = ref.g3_terms(params, padded_batch(["tolkie"]))
    for _, t, k, T in one:
        u = ref.loss_projected_vector(params, f, t, k)
        expected = float((ref.tanh_derivative(f.states[k - 1]) * u).norm() * f.states[k - 2].norm())
        assert abs(float(T.norm()) - expected) <= 1e-12 + 1e-9 * expected


# --------------------------------------------------------------------------- A:M5 finite differences
def test_M5_finite_differences_every_parameter_element(generator_factory) -> None:
    """[FD] Central differences (L(θ + h e_i) − L(θ − h e_i)) / 2h, h = 1e‑6, for **every element**
    of U, W, b, V and c, against the BPTT gradient (rtol 1e‑6, atol 1e‑8; §13.1 FD).

    H = 3, g = 1.2 keeps the 201-element sweep fast. Numerical limits: truncation O(h²) ≈ 1e‑12;
    rounding ≈ u·|L|/h ≈ 1e‑10·|L| with |L| ≈ 30 here, i.e. ≲ 5e‑9, inside atol 1e‑8. Unused
    rows of U (letters absent from the fixture) must have both gradients exactly zero."""
    params = perturbed_params(3, 1.2, generator_factory(3))
    inputs = padded_batch(["zo", "bex", "mavra", "hhhhhhhhh"], width=11)
    analytic = ref.bptt_gradients(params, inputs)
    used = {int(v) for v in inputs.flatten().tolist() if v != PAD}
    tensors = dict(zip("UWbVc", params.as_tuple()))
    worst = 0.0
    checked = 0
    for name, tensor in tensors.items():
        # Perturb each element in place by its multi-index. (reshape(-1) would silently return a
        # copy for a non-contiguous tensor, e.g. W built from a QR factor, and perturb nothing.)
        for idx in itertools.product(*(range(d) for d in tensor.shape)):
            orig = float(tensor[idx])
            tensor[idx] = orig + FD_STEP
            plus = float(ref.total_loss(params, inputs))
            tensor[idx] = orig - FD_STEP
            minus = float(ref.total_loss(params, inputs))
            tensor[idx] = orig
            assert float(tensor[idx]) == orig
            fd = (plus - minus) / (2 * FD_STEP)
            an = float(analytic[name][idx])
            err = abs(fd - an) - (FD_ATOL + FD_RTOL * abs(an))
            worst = max(worst, err)
            assert err <= 0, f"{name}{list(idx)}: FD {fd!r} vs analytic {an!r}"
            if name == "U" and idx[0] not in used:
                assert an == 0.0 and abs(fd) <= FD_ATOL
            checked += 1
    assert checked == 3 * 3 + 55 * 3 + 27  # P(H) = H² + 55H + 27 at H = 3: every element visited
    assert worst <= 0


# --------------------------------------------------------------------------- independence (static)
_ALLOWED_IMPORTS = {"__future__", "math", "torch"}
_FORBIDDEN_ATTR_CHAINS = ("torch.nn", "torch.autograd", "torch.func", "torch.linalg")
_FORBIDDEN_CALLS = {"matmul", "mm", "mv", "bmm", "einsum", "tensordot", "softmax", "log_softmax",
                    "logsumexp", "cross_entropy", "nll_loss", "rnn_tanh", "RNN", "RNNCell"}
_FORBIDDEN_NAMES = {"_step", "_logits", "_recurrence", "RNNLanguageModel", "RecurrenceOutput"}


def _attr_chain(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_reference_is_independent_by_construction() -> None:
    """[static] The reference imports only ``torch`` and ``math``; never ``models``, ``utils``,
    ``torch.nn``, ``torch.autograd``, ``torch.func`` or ``torch.linalg``; uses no ``@`` operator,
    no matrix-multiply, softmax or loss helper, and names no production symbol."""
    tree = ast.parse(REFERENCE_FILE.read_text(encoding="utf-8"))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [a.name for a in node.names if a.name.split(".")[0] not in _ALLOWED_IMPORTS
                         or any(a.name.startswith(c) for c in _FORBIDDEN_ATTR_CHAINS)]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] not in _ALLOWED_IMPORTS or any(mod.startswith(c) for c in _FORBIDDEN_ATTR_CHAINS):
                problems.append(f"from {mod}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            problems.append(f"line {node.lineno}: @ operator")
        elif isinstance(node, ast.Attribute):
            chain = _attr_chain(node)
            if any(chain == c or chain.startswith(c + ".") for c in _FORBIDDEN_ATTR_CHAINS):
                problems.append(f"line {node.lineno}: {chain}")
            if node.attr in _FORBIDDEN_CALLS:
                problems.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES | _FORBIDDEN_CALLS:
            problems.append(f"line {node.lineno}: name {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "models.recurrent" in node.value:
            problems.append(f"line {node.lineno}: string mentions models.recurrent")
    assert not problems, problems


def test_reference_does_not_import_production_code_at_runtime() -> None:
    """[static] Executing ``rnn_reference.py`` in a fresh interpreter loads no ``models.*`` or
    ``utils*`` module.

    The child loads **that exact file by path** (``importlib.util.spec_from_file_location``),
    not through the package name ``tests``: a third-party top-level ``tests`` package in
    site-packages takes precedence over the repository's namespace directory and would make a
    package import fail (observed on native macOS, MS1 verification). The repository stays on
    the child's ``PYTHONPATH``, so a hidden production import in the reference would either load
    and be listed below, or fail and be reported with the child's stderr; it cannot pass."""
    import json
    import os
    import subprocess
    import sys

    repo = REFERENCE_FILE.parents[3]
    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('m6_rnn_reference_probe', {str(REFERENCE_FILE)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        "assert callable(module.forward_batch) and callable(module.bptt_gradients)\n"
        "print(json.dumps(sorted(m for m in sys.modules if m == 'models' or m.startswith(('models.', 'utils')))))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(repo)
    out = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True)
    detail = f"exit code {out.returncode}\n--- child stdout ---\n{out.stdout}--- child stderr ---\n{out.stderr}"
    assert out.returncode == 0, detail
    lines = out.stdout.strip().splitlines()
    assert lines, detail
    assert json.loads(lines[-1]) == [], detail

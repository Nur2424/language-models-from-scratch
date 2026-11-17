"""Independent float64 mathematical reference for Model 6 (vanilla Elman RNN).

Milestone MS1 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (approved, D-49).
Written from ``models/recurrent/rnn_spec.md`` v0.2 (D-44) §4.0 – §4.4, §8, §12.3 – §12.5 and
§13.1 – §13.2, **before** the production model exists.

Independence rules (checked by ``test_rnn_reference.py``)
---------------------------------------------------------
* Imports ``torch`` and ``math`` only; never ``models.*``, ``utils.*``, ``torch.nn``,
  ``torch.autograd``, ``torch.func`` or ``torch.linalg``.
* Receives plain tensors (U, W, b, V, c and integer inputs), never a model object.
* Processes **each name separately, unbatched, only up to its own length** (§13.2). A padded
  batch is handled by computing every name on its own and then writing the padded positions
  **from their definition** (h_t = h_n, z_t = V h_n + c for t > n, §4.1). There is no masked
  recurrence, so this reference shares no structure with a ``where``-based implementation.
* Uses no matrix-multiply kernel: every product is an explicit elementwise multiply followed
  by a sum (``_matvec``, ``_matmul``), and log-softmax is written out with the max shift.
  The production model will use ``@``; agreement across the two therefore also checks that no
  kernel-specific behaviour is being relied on.
* Uses no random numbers.

Conventions (§4.0)
------------------
* Mathematics is 1-based: positions t = 1 … n. Python lists and tensors are 0-based, so the
  quantity at position t is stored at index ``t - 1``. Public functions that take positions
  (``jacobian_product``, ``loss_projected_vector``) take the **1-based** t and s of the spec.
* K = ``U.shape[0]`` output classes (27 in the project); input ids 0 … K−1; the PAD id is K and
  is never looked up. Id 0 is BOS as an input and EOS as a target.
* Column-vector mathematics: a_t = U[x_t] + W h_{t−1} + b, z_t = V h_t + c, with W[i, j] the
  weight from unit j at step t − 1 to unit i at step t (§4.1, "code orientation").
* All parameters must be float64 (criterion A of §13.1 applies to this reference).
"""

from __future__ import annotations

import math

import torch

IGNORE_INDEX = -100  # padded targets (§4.0); repeated here so that ``utils`` is not imported


# --------------------------------------------------------------------------- containers
class Params:
    """The five parameters of §4.3, as plain float64 tensors.

    U [K, H] input lookup; W [H, H] recurrent matrix; b [H] hidden bias;
    V [K, H] output projection; c [K] output bias.
    """

    __slots__ = ("U", "W", "b", "V", "c")

    def __init__(self, U: torch.Tensor, W: torch.Tensor, b: torch.Tensor, V: torch.Tensor,
                 c: torch.Tensor) -> None:
        self.U, self.W, self.b, self.V, self.c = U, W, b, V, c
        check_params(self)

    @property
    def K(self) -> int:
        return int(self.U.shape[0])

    @property
    def H(self) -> int:
        return int(self.W.shape[0])

    @property
    def pad_id(self) -> int:
        return self.K

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (self.U, self.W, self.b, self.V, self.c)


class NameForward:
    """Exact forward quantities of one name over its n valid steps (index t − 1 holds step t).

    ``inputs`` x_1 … x_n; ``targets`` y_1 … y_n (y_t = x_{t+1}, y_n = EOS = 0);
    ``pre`` a_t [n, H]; ``states`` h_t [n, H]; ``logits`` z_t [n, K];
    ``log_probs`` log p_t [n, K]; ``probs`` p_t [n, K]; ``losses`` ℓ_t [n].
    """

    __slots__ = ("inputs", "targets", "pre", "states", "logits", "log_probs", "probs", "losses")

    def __init__(self, **kw: object) -> None:
        for name in self.__slots__:
            setattr(self, name, kw[name])

    @property
    def n(self) -> int:
        return len(self.inputs)


class BatchForward:
    """Padded outputs for a [B, T] input tensor, built name by name.

    ``states`` [B, T, H]: h_t for t ≤ n_β, then frozen at h_{n_β}.
    ``pre`` [B, T, H]: a_t for t ≤ n_β; **NaN at padded positions**, where the spec gives the
    pre-activation no meaning (§5): any accidental comparison of padded pre-activations fails.
    ``logits`` [B, T, K]: z_t for t ≤ n_β; V h_{n_β} + c at padded positions (§4.1).
    ``valid`` [B, T] bool; ``targets`` [B, T] long (−100 at padded positions);
    ``lengths`` n_β; ``names`` the per-name :class:`NameForward` results.
    """

    __slots__ = ("states", "pre", "logits", "valid", "targets", "lengths", "names")

    def __init__(self, **kw: object) -> None:
        for name in self.__slots__:
            setattr(self, name, kw[name])


# --------------------------------------------------------------------------- explicit algebra
def _matvec(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(M v)_i = Σ_j M[i, j] v[j], as an explicit multiply and sum (no matmul kernel)."""
    return (M * v.unsqueeze(0)).sum(dim=1)


def _matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """(A B)[i, k] = Σ_j A[i, j] B[j, k], as an explicit multiply and sum."""
    return (A.unsqueeze(2) * B.unsqueeze(0)).sum(dim=1)


def _outer(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(u vᵀ)[i, j] = u[i] v[j]."""
    return u.unsqueeze(1) * v.unsqueeze(0)


def _log_softmax(z: torch.Tensor) -> torch.Tensor:
    """log softmax(z)_k = z_k − (m + log Σ_j exp(z_j − m)), m = max_j z_j (overflow-safe)."""
    m = z.max()
    return z - (m + torch.log(torch.exp(z - m).sum()))


def _eye(H: int, like: torch.Tensor) -> torch.Tensor:
    I = torch.zeros(H, H, dtype=like.dtype, device=like.device)
    for i in range(H):
        I[i, i] = 1.0
    return I


# --------------------------------------------------------------------------- validation
def check_params(p: Params) -> None:
    tensors = dict(U=p.U, W=p.W, b=p.b, V=p.V, c=p.c)
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if t.dtype != torch.float64:
            raise TypeError(f"{name} must be float64 (the reference is a float64 reference), got {t.dtype}")
    K, H = p.U.shape if p.U.ndim == 2 else (None, None)
    expected = dict(U=(K, H), W=(H, H), b=(H,), V=(K, H), c=(K,))
    if K is None or K < 1 or H < 1:
        raise ValueError(f"U must have shape [K, H] with K, H ≥ 1, got {tuple(p.U.shape)}")
    for name, t in tensors.items():
        if tuple(t.shape) != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(t.shape)}")


def valid_ids(row: torch.Tensor | list[int], pad_id: int) -> list[int]:
    """The valid prefix x_1 … x_n of one padded row, after checking the §5.1 row rules:
    column 0 is BOS (0); ids in 0 … pad_id; padding (pad_id) right-contiguous."""
    ids = [int(v) for v in (row.tolist() if isinstance(row, torch.Tensor) else row)]
    if not ids:
        raise ValueError("a row needs at least the BOS column")
    if ids[0] != 0:
        raise ValueError(f"column 0 must be BOS (0), got {ids[0]}")
    if any(v < 0 or v > pad_id for v in ids):
        raise ValueError(f"ids must lie in 0 … {pad_id}")
    n = ids.index(pad_id) if pad_id in ids else len(ids)
    if any(v != pad_id for v in ids[n:]):
        raise ValueError("padding must be right-contiguous (a PAD is followed by a non-PAD id)")
    return ids[:n]


def targets_of(inputs: list[int]) -> list[int]:
    """y_t = x_{t+1} for t < n and y_n = 0 (EOS) (§4.0)."""
    return list(inputs[1:]) + [0]


# --------------------------------------------------------------------------- forward
def step(p: Params, h_prev: torch.Tensor, x: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One valid step: a = U[x] + W h_prev + b; h = tanh(a). Returns (h, a). (§4.1)"""
    if not 0 <= x < p.K:
        raise ValueError(f"input id {x} cannot be looked up (PAD = {p.pad_id} is never looked up)")
    a = p.U[x] + _matvec(p.W, h_prev) + p.b
    return torch.tanh(a), a


def output(p: Params, h: torch.Tensor) -> torch.Tensor:
    """z = V h + c (§4.1)."""
    return _matvec(p.V, h) + p.c


def forward_name(p: Params, inputs: list[int], targets: list[int] | None = None) -> NameForward:
    """Exact forward pass of one name over its own n = len(inputs) valid steps, h_0 = 0.

    ``targets`` defaults to :func:`targets_of`; G2 and C1-style uses may pass their own.
    """
    check_params(p)
    inputs = [int(v) for v in inputs]
    if not inputs or inputs[0] != 0:
        raise ValueError("a name's inputs start with BOS (0)")
    targets = targets_of(inputs) if targets is None else [int(v) for v in targets]
    if len(targets) != len(inputs) or any(not 0 <= y < p.K for y in targets):
        raise ValueError("targets must be one valid class id per input position")
    h = torch.zeros(p.H, dtype=p.W.dtype, device=p.W.device)
    pre, states, logits, log_probs, losses = [], [], [], [], []
    for t in range(1, len(inputs) + 1):
        h, a = step(p, h, inputs[t - 1])
        z = output(p, h)
        lp = _log_softmax(z)
        pre.append(a)
        states.append(h)
        logits.append(z)
        log_probs.append(lp)
        losses.append(-lp[targets[t - 1]])
    log_probs_t = torch.stack(log_probs)
    return NameForward(
        inputs=inputs, targets=targets, pre=torch.stack(pre), states=torch.stack(states),
        logits=torch.stack(logits), log_probs=log_probs_t, probs=torch.exp(log_probs_t),
        losses=torch.stack(losses),
    )


def forward_batch(p: Params, inputs: torch.Tensor) -> BatchForward:
    """Padded [B, T] inputs → padded outputs, each name computed alone (§13.2).

    Padded positions are filled from their definition (§4.1): states frozen at h_{n_β};
    logits V h_{n_β} + c; pre-activations NaN; targets −100.
    """
    check_params(p)
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if inputs.dtype != torch.long or inputs.ndim != 2:
        raise ValueError("inputs must be a 2-D torch.long tensor [B, T]")
    B, T = inputs.shape
    if T == 0:
        raise ValueError("T = 0: no BOS column")
    H, K = p.H, p.K
    kw = dict(dtype=p.W.dtype, device=p.W.device)
    states = torch.zeros(B, T, H, **kw)
    pre = torch.full((B, T, H), float("nan"), **kw)
    logits = torch.zeros(B, T, K, **kw)
    valid = torch.zeros(B, T, dtype=torch.bool)
    targets = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
    lengths, names = [], []
    for beta in range(B):
        ids = valid_ids(inputs[beta], p.pad_id)
        f = forward_name(p, ids)
        n = f.n
        states[beta, :n] = f.states
        pre[beta, :n] = f.pre
        logits[beta, :n] = f.logits
        valid[beta, :n] = True
        targets[beta, :n] = torch.tensor(f.targets, dtype=torch.long)
        for t in range(n + 1, T + 1):  # padded positions, by definition
            states[beta, t - 1] = f.states[n - 1]
            logits[beta, t - 1] = output(p, f.states[n - 1])
        lengths.append(n)
        names.append(f)
    return BatchForward(states=states, pre=pre, logits=logits, valid=valid, targets=targets,
                        lengths=lengths, names=names)


def total_loss(p: Params, inputs: torch.Tensor) -> torch.Tensor:
    """Unnormalised loss Σ_β Σ_{t ≤ n_β} ℓ_{β,t} (§4.2 without the 1/C factor)."""
    return torch.stack([f.losses.sum() for f in forward_batch(p, inputs).names]).sum() \
        if inputs.shape[0] else torch.zeros((), dtype=p.W.dtype)


# --------------------------------------------------------------------------- Jacobians (§4.4, §12.3)
def tanh_derivative(h: torch.Tensor) -> torch.Tensor:
    """diag of D_t = diag(1 − h_t²), using the **new** state h_t (§4.4)."""
    return 1.0 - h * h


def step_jacobian(p: Params, h_t: torch.Tensor) -> torch.Tensor:
    """J_t = ∂h_t/∂h_{t−1} = D_t W at a valid step (§4.4)."""
    return tanh_derivative(h_t).unsqueeze(1) * p.W  # row i scaled by (1 − h_{t,i}²)


def jacobian_product(p: Params, f: NameForward, t: int, s: int) -> torch.Tensor:
    """P_{t,s} = ∂h_t/∂h_s = J_t J_{t−1} ⋯ J_{s+1} (newest on the left); P_{s,s} = I.
    1-based positions with 0 ≤ s ≤ t ≤ n (s = 0 denotes h_0)."""
    if not 0 <= s <= t <= f.n:
        raise ValueError(f"need 0 ≤ s ≤ t ≤ n = {f.n}, got s = {s}, t = {t}")
    P = _eye(p.H, p.W)
    for k in range(s + 1, t + 1):
        P = _matmul(step_jacobian(p, f.states[k - 1]), P)
    return P


def output_residual(p: Params, f: NameForward, t: int) -> torch.Tensor:
    """p_t − e_{y_t} = ∂ℓ_t/∂z_t (1-based t)."""
    d = f.probs[t - 1].clone()
    d[f.targets[t - 1]] -= 1.0
    return d


def hidden_residual(p: Params, f: NameForward, t: int) -> torch.Tensor:
    """r_t = ∂ℓ_t/∂h_t = Vᵀ (p_t − e_{y_t}) (§4.4, §12.4)."""
    return _matvec(p.V.transpose(0, 1), output_residual(p, f, t))


def loss_projected_vector(p: Params, f: NameForward, t: int, s: int) -> torch.Tensor:
    """u_{t,s} = P_{t,s}ᵀ r_t = ∂ℓ_t/∂h_s for 1 ≤ s ≤ t ≤ n (§4.4, §12.4)."""
    if not 1 <= s <= t <= f.n:
        raise ValueError(f"need 1 ≤ s ≤ t ≤ n = {f.n}")
    return _matvec(jacobian_product(p, f, t, s).transpose(0, 1), hidden_residual(p, f, t))


# --------------------------------------------------------------------------- BPTT (§4.4)
def bptt_gradients_name(p: Params, f: NameForward) -> dict[str, torch.Tensor]:
    """Gradients of L = Σ_{t ≤ n} ℓ_t for one name, from the §4.4 recursion:

    δ_t = r_t + Wᵀ D_{t+1} δ_{t+1} (δ_{n+1} = 0);  g_t = D_t δ_t;
    ∂L/∂W = Σ_{t ≥ 2} g_t h_{t−1}ᵀ;  ∂L/∂U[x] = Σ_{t: x_t = x} g_t;  ∂L/∂b = Σ_t g_t;
    ∂L/∂V = Σ_t (p_t − e_{y_t}) h_tᵀ;  ∂L/∂c = Σ_t (p_t − e_{y_t}).
    """
    grads = {name: torch.zeros_like(t) for name, t in zip("UWbVc", p.as_tuple())}
    WT = p.W.transpose(0, 1)
    g_next = torch.zeros(p.H, dtype=p.W.dtype, device=p.W.device)  # g_{n+1} = D_{n+1} δ_{n+1} = 0
    for t in range(f.n, 0, -1):
        h_t = f.states[t - 1]
        dz = output_residual(p, f, t)
        delta = _matvec(p.V.transpose(0, 1), dz) + _matvec(WT, g_next)
        g = tanh_derivative(h_t) * delta
        if t >= 2:
            grads["W"] += _outer(g, f.states[t - 2])
        grads["U"][f.inputs[t - 1]] += g
        grads["b"] += g
        grads["V"] += _outer(dz, h_t)
        grads["c"] += dz
        g_next = g
    return grads


def bptt_gradients(p: Params, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Gradients of the unnormalised batch loss Σ_β Σ_t ℓ_{β,t}: per-name sums (§4.4)."""
    total = {name: torch.zeros_like(t) for name, t in zip("UWbVc", p.as_tuple())}
    for f in forward_batch(p, inputs).names:
        for name, g in bptt_gradients_name(p, f).items():
            total[name] += g
    return total


# --------------------------------------------------------------------------- G3 (§12.5)
class G3Decomposition:
    """Exact decomposition of ∂(Σ ℓ)/∂W into terms T_{t,k} = (D_k P_{t,k}ᵀ r_t) h_{k−1}ᵀ,
    2 ≤ k ≤ t ≤ n_β, grouped by distance Δ = t − k (§12.5).

    ``G`` Σ of all terms; ``share[Δ]`` π(Δ) = Σ ⟨T, G⟩_F / ‖G‖_F²; ``mass[Δ]`` Q(Δ) = Σ ‖T‖_F;
    ``kappa`` κ_c = ‖G‖_F / Σ ‖T‖_F; ``count[Δ]`` number of terms.
    """

    __slots__ = ("G", "share", "mass", "kappa", "count")

    def __init__(self, **kw: object) -> None:
        for name in self.__slots__:
            setattr(self, name, kw[name])


def g3_terms(p: Params, inputs: torch.Tensor) -> list[tuple[int, int, int, torch.Tensor]]:
    """Every term (β, t, k, T_{t,k}) of §12.5, as explicit H × H matrices."""
    out = []
    for beta, f in enumerate(forward_batch(p, inputs).names):
        for t in range(1, f.n + 1):
            for k in range(2, t + 1):
                u = loss_projected_vector(p, f, t, k)  # P_{t,k}ᵀ r_t
                out.append((beta, t, k, _outer(tanh_derivative(f.states[k - 1]) * u, f.states[k - 2])))
    return out


def g3_decomposition(p: Params, inputs: torch.Tensor) -> G3Decomposition:
    terms = g3_terms(p, inputs)
    G = torch.zeros_like(p.W)
    for *_, T in terms:
        G = G + T
    G_sq = float((G * G).sum())
    share: dict[int, float] = {}
    mass: dict[int, float] = {}
    count: dict[int, int] = {}
    total_norm = 0.0
    for _, t, k, T in terms:
        d = t - k
        share[d] = share.get(d, 0.0) + float((T * G).sum()) / G_sq if G_sq > 0 else math.nan
        norm = math.sqrt(float((T * T).sum()))
        mass[d] = mass.get(d, 0.0) + norm
        count[d] = count.get(d, 0) + 1
        total_norm += norm
    kappa = math.sqrt(G_sq) / total_norm if total_norm > 0 else math.nan
    return G3Decomposition(G=G, share=share, mass=mass, kappa=kappa, count=count)

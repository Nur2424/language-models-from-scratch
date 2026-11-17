"""Independent PyTorch reference for the WaveNet hierarchy (specification §14).

Independence rules (enforced by ``test_wavenet_reference.py``):

* imports only ``torch`` (``math`` and ``typing`` are also permitted); nothing from ``models`` or ``utils``;
* functions take **tensors only** (cloned parameters, contexts, statistics); they never
  receive the model object, never call ``forward`` or any model method, and do not use
  any model helper (including ``merge_pairs``);
* the equations are written out directly from the specification.

Parameter-name mapping (``model.state_dict()`` key → symbol)::

    emb.weight                     E     [V, d]
    hidden{ℓ}.weight               W_ℓ   [H, C_ℓ]   (PyTorch [out, in]; first C columns act on the earlier position)
    out.weight / out.bias          W_o [V, H] / b_o [V]
    bn{ℓ}.weight / bn{ℓ}.bias      γ_ℓ / β_ℓ                      (BatchNorm variant)
    bn{ℓ}.inference_mean / _var    m_ℓ / v_ℓ (eval-mode statistics)

Intermediate names returned by :func:`ref_forward` / :func:`ref_conv_forward`::

    e   [B, k, d]          embeddings
    m{ℓ} [B, G_ℓ, C_ℓ]     merged input of level ℓ
    a{ℓ} [B, G_ℓ, H]       pre-normalisation activation
    y{ℓ} [B, G_ℓ, H]       normalised activation (equal to a{ℓ} without BatchNorm)
    h{ℓ} [B, G_ℓ, H]       tanh output
    flat [B, H]            h{L}[:, 0, :]
    logits [B, V]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-5

Params = dict[str, torch.Tensor]
Stats = dict[int, tuple[torch.Tensor, torch.Tensor]]


def ref_parameter_count(V: int, d: int, H: int, L: int, normalization: str) -> int:
    """§5: ``V·d + 2d·H + (L−1)·2H·H + H·V + V`` (+ ``L·2H`` with BatchNorm)."""
    base = V * d + (2 * d) * H + (L - 1) * (2 * H) * H + H * V + V
    return base + (L * 2 * H if normalization == "batchnorm" else 0)


def ref_merge(x: torch.Tensor) -> torch.Tensor:
    """Explicit loop and gather: ``out[:, j] = cat(x[:, 2j], x[:, 2j+1])``."""
    T = x.shape[1]
    return torch.stack([torch.cat([x[:, 2 * j], x[:, 2 * j + 1]], dim=-1) for j in range(T // 2)], dim=1)


def ref_embed(params: Params, contexts: torch.Tensor) -> torch.Tensor:
    """Embedding lookup as a one-hot matrix product (independent of ``nn.Embedding`` indexing)."""
    E = params["emb.weight"]
    onehot = F.one_hot(contexts, E.shape[0]).to(E.dtype)  # [B, k, V]
    return onehot @ E


def _ref_batchnorm_train(a: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Pooled over all (example, group) rows; biased variance."""
    mean = a.mean(dim=(0, 1))
    var = ((a - mean) ** 2).mean(dim=(0, 1))
    return gamma * (a - mean) / torch.sqrt(var + EPS) + beta


def _ref_batchnorm_eval(a: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                        mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return gamma * (a - mean) / torch.sqrt(var + EPS) + beta


def ref_forward(params: Params, contexts: torch.Tensor, *, num_levels: int, normalization: str = "none",
                mode: str = "eval", stats: Stats | None = None) -> dict[str, torch.Tensor]:
    """Explicit loops over levels **and groups** (§3.3, §3.6). Returns every intermediate."""
    out: dict[str, torch.Tensor] = {}
    h = ref_embed(params, contexts)
    out["e"] = h
    for level in range(1, num_levels + 1):
        m = ref_merge(h)
        W = params[f"hidden{level}.weight"]
        a = torch.stack([m[:, j] @ W.T for j in range(m.shape[1])], dim=1)  # same W for every group
        if normalization == "batchnorm":
            gamma, beta = params[f"bn{level}.weight"], params[f"bn{level}.bias"]
            if mode == "train":
                y = _ref_batchnorm_train(a, gamma, beta)
            else:
                mean, var = stats[level] if stats is not None else (
                    params[f"bn{level}.inference_mean"], params[f"bn{level}.inference_var"])
                y = _ref_batchnorm_eval(a, gamma, beta, mean, var)
        else:
            y = a
        h = torch.tanh(y)
        out[f"m{level}"], out[f"a{level}"], out[f"y{level}"], out[f"h{level}"] = m, a, y, h
    out["flat"] = h[:, 0, :]
    out["logits"] = out["flat"] @ params["out.weight"].T + params["out.bias"]
    return out


def conv_kernel(W: torch.Tensor) -> torch.Tensor:
    """§14.3 mapping: ``W [H, 2C]`` → ``K [H, C, 2]`` with ``K[o, c, 0] = W[o, c]`` (earlier position)
    and ``K[o, c, 1] = W[o, C + c]`` (later position)."""
    H, two_c = W.shape
    return W.view(H, 2, two_c // 2).permute(0, 2, 1).contiguous()


def ref_conv_forward(params: Params, contexts: torch.Tensor, *, num_levels: int, normalization: str = "none",
                     mode: str = "eval", stats: Stats | None = None) -> dict[str, torch.Tensor]:
    """Secondary reference: each level as a stride-2, kernel-2 ``F.conv1d`` on ``[B, C, T]`` (§14.3 (i)).

    BatchNorm uses ``F.batch_norm`` on the ``[B, H, G]`` layout (statistics pooled over batch and
    length). Intermediates are returned in the ``[B, G, H]`` layout for comparison.
    """
    out: dict[str, torch.Tensor] = {}
    seq = ref_embed(params, contexts).permute(0, 2, 1)  # [B, d, k]
    for level in range(1, num_levels + 1):
        a = F.conv1d(seq, conv_kernel(params[f"hidden{level}.weight"]), stride=2)  # [B, H, G]
        if normalization == "batchnorm":
            gamma, beta = params[f"bn{level}.weight"], params[f"bn{level}.bias"]
            if mode == "train":
                y = F.batch_norm(a, None, None, gamma, beta, training=True, eps=EPS)
            else:
                mean, var = stats[level] if stats is not None else (
                    params[f"bn{level}.inference_mean"], params[f"bn{level}.inference_var"])
                y = F.batch_norm(a, mean, var, gamma, beta, training=False, eps=EPS)
        else:
            y = a
        seq = torch.tanh(y)
        out[f"a{level}"], out[f"y{level}"], out[f"h{level}"] = (t.permute(0, 2, 1) for t in (a, y, seq))
    out["logits"] = seq[:, :, 0] @ params["out.weight"].T + params["out.bias"]
    return out


def ref_dilated_forward(params: Params, contexts: torch.Tensor, *, num_levels: int) -> torch.Tensor:
    """Baseline only (§14.3 (ii)): dilated causal convolutions, kernel 2, dilation 2^(ℓ−1), no padding.

    The length shrinks k → k−1 → k−3 → … → 1; the last position equals the tree output.
    Not valid for the BatchNorm variant (its statistics would pool over different positions).
    """
    seq = ref_embed(params, contexts).permute(0, 2, 1)
    for level in range(1, num_levels + 1):
        seq = torch.tanh(F.conv1d(seq, conv_kernel(params[f"hidden{level}.weight"]), dilation=2 ** (level - 1)))
    return seq[:, :, -1] @ params["out.weight"].T + params["out.bias"]


def ref_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean of ``logsumexp(z) − z_y``."""
    return (torch.logsumexp(logits, dim=1) - logits.gather(1, targets[:, None])[:, 0]).mean()


def ref_calibrate(params: Params, contexts: torch.Tensor, *, num_levels: int) -> Stats:
    """Exact population statistics, all windows at once, levels **sequential** (§3.6).

    Level ℓ's pre-activations use levels < ℓ normalised with the statistics just computed.
    Mean and biased variance over all ``M·G_ℓ`` rows.
    """
    stats: Stats = {}
    with torch.no_grad():
        h = ref_embed(params, contexts)
        for level in range(1, num_levels + 1):
            a = ref_merge(h) @ params[f"hidden{level}.weight"].T
            rows = a.reshape(-1, a.shape[-1])
            mean = rows.mean(dim=0)
            var = ((rows - mean) ** 2).mean(dim=0)
            stats[level] = (mean, var)
            h = torch.tanh(_ref_batchnorm_eval(a, params[f"bn{level}.weight"], params[f"bn{level}.bias"], mean, var))
    return stats


def dependence_positions(grad_e: torch.Tensor) -> list:
    """1-based positions p whose embedding gradient is nonzero (for receptive-field checks)."""
    return [p + 1 for p in range(grad_e.shape[1]) if bool(grad_e[:, p].abs().sum() > 0)]


def expected_field(level: int, group: int) -> list:
    """§3.4: group j (1-based) of level ℓ sees positions (j−1)·2^ℓ + 1 … j·2^ℓ."""
    width = 2 ** level
    return list(range((group - 1) * width + 1, group * width + 1))


__all__ = [
    "EPS", "conv_kernel", "dependence_positions", "expected_field", "ref_calibrate", "ref_conv_forward",
    "ref_dilated_forward", "ref_embed", "ref_forward", "ref_loss", "ref_merge", "ref_parameter_count",
]

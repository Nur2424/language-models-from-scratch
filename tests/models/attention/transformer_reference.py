"""Independent float64 reference implementation of Models 01 and 02 (spec §13.2).

Written from the specification, not from ``transformer.py``: explicit loops over names, positions
and heads; its own softmax, LayerNorm, GELU and sinusoids, using only ``math`` for the scalar
functions. It reads the model's parameters by name and nothing else.
"""
from __future__ import annotations

import math

import torch


def _ln(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    n = x.numel()
    mean = sum(float(v) for v in x) / n
    var = sum((float(v) - mean) ** 2 for v in x) / n
    return torch.tensor([(float(v) - mean) / math.sqrt(var + eps) for v in x], dtype=torch.float64) * gamma + beta


def _gelu(x: torch.Tensor) -> torch.Tensor:
    c = math.sqrt(2.0 / math.pi)
    return torch.tensor([0.5 * z * (1.0 + math.tanh(c * (z + 0.044715 * z ** 3))) for z in x.tolist()],
                        dtype=torch.float64)


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor([max(0.0, z) for z in x.tolist()], dtype=torch.float64)


def _sinusoid(p: int, d: int) -> torch.Tensor:
    out = []
    for j in range(d):
        i = j // 2
        angle = p / (10000.0 ** (2 * i / d))
        out.append(math.sin(angle) if j % 2 == 0 else math.cos(angle))
    return torch.tensor(out, dtype=torch.float64)


def reference_logits(model, inputs: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
    cfg = model.config
    gpt2 = cfg.variant == "gpt2_style"
    eps = 1e-5 if gpt2 else 1e-6
    P = {n: p.detach().to(torch.float64) for n, p in model.named_parameters()}
    E = P["token_embedding"]
    d, H, L = cfg.d_model, cfg.num_heads, cfg.num_layers
    dk = d // H
    pad = model.vocab.pad_id
    B, T = inputs.shape
    out = torch.zeros(B, T, E.shape[0], dtype=torch.float64)

    def lin(prefix: str, name: str, x: torch.Tensor) -> torch.Tensor:
        y = P[f"{prefix}.w_{name}"] @ x
        b = P.get(f"{prefix}.b_{name}")
        return y if b is None else y + b

    for b in range(B):
        ids = [0 if v == pad else v for v in inputs[b].tolist()]
        pos = list(range(T)) if positions is None else positions[b].tolist()
        if gpt2:
            h = [E[ids[t]] + P["position_embedding"][pos[t]] for t in range(T)]
        else:
            h = [math.sqrt(d) * E[ids[t]] + _sinusoid(pos[t], d) for t in range(T)]
        for layer in range(L):
            pre = f"blocks.{layer}"
            g1, b1 = P[f"{pre}.ln_1.gamma"], P[f"{pre}.ln_1.beta"]
            g2, b2 = P[f"{pre}.ln_2.gamma"], P[f"{pre}.ln_2.beta"]
            a_in = [_ln(x, g1, b1, eps) for x in h] if gpt2 else h
            q = [lin(pre, "q", x) for x in a_in]
            k = [lin(pre, "k", x) for x in a_in]
            v = [lin(pre, "v", x) for x in a_in]
            attn = []
            for t in range(T):
                heads = []
                for hd in range(H):
                    sl = slice(hd * dk, (hd + 1) * dk)
                    scores = [sum(float(a) * float(c) for a, c in zip(q[t][sl], k[s][sl])) / math.sqrt(dk)
                              for s in range(t + 1)]
                    m = max(scores)
                    ex = [math.exp(s_ - m) for s_ in scores]
                    z = sum(ex)
                    w = [e / z for e in ex]
                    heads.append(sum(w[s] * v[s][sl] for s in range(t + 1)))
                attn.append(lin(pre, "o", torch.cat(heads)))
            act = _gelu if gpt2 else _relu

            def ffn(x: torch.Tensor) -> torch.Tensor:
                return P[f"{pre}.w_2"] @ act(P[f"{pre}.w_1"] @ x + P[f"{pre}.b_1"]) + P[f"{pre}.b_2"]

            if gpt2:
                u = [h[t] + attn[t] for t in range(T)]
                h = [u[t] + ffn(_ln(u[t], g2, b2, eps)) for t in range(T)]
            else:
                u = [_ln(h[t] + attn[t], g1, b1, eps) for t in range(T)]
                h = [_ln(u[t] + ffn(u[t]), g2, b2, eps) for t in range(T)]
        if gpt2:
            h = [_ln(x, P["final_ln.gamma"], P["final_ln.beta"], eps) for x in h]
        for t in range(T):
            out[b, t] = E @ h[t]
    return out

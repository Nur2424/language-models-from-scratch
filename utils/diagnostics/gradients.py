"""Gradient diagnostics. Call after ``loss.backward()`` and before ``optimizer.step()``.

All functions read ``p.grad`` under ``no_grad`` and return Python numbers;
they never modify gradients (unlike ``clip_grad_norm_``) and keep no tensors.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from utils.diagnostics import GroupBy, group_key
from utils.statistics import Moments, tensor_moments


def _grads(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]


@torch.no_grad()
def global_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Norm of all gradients concatenated (same definition as ``clip_grad_norm_``).

    Parameters without gradients are skipped; returns 0.0 if none have one.
    NaN/inf gradients propagate into the result, which is what you want to see.
    """
    grads = [g for _, g in _grads(model)]
    if not grads:
        return 0.0
    if norm_type == math.inf:
        return max(float(g.detach().abs().max()) for g in grads)
    per = torch.stack([torch.linalg.vector_norm(g.detach().float(), norm_type).cpu() for g in grads])
    return float(torch.linalg.vector_norm(per, norm_type))


def parameters_without_grad(model: nn.Module) -> list[str]:
    """Names of trainable parameters whose ``.grad`` is ``None`` after backward
    (unused parameters, detached branches, or a missing backward call)."""
    return [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]


@torch.no_grad()
def nonfinite_gradients(model: nn.Module) -> dict[str, dict[str, int]]:
    """Map parameter name → ``{"nan": k, "inf": m}`` for gradients containing NaN/inf."""
    out: dict[str, dict[str, int]] = {}
    for name, g in _grads(model):
        nan, inf = int(torch.isnan(g).sum()), int(torch.isinf(g).sum())
        if nan or inf:
            out[name] = {"nan": nan, "inf": inf}
    return out


def gradient_summaries(model: nn.Module, *, group_by: GroupBy = "parameter", depth: int = 1) -> dict[str, dict[str, Any]]:
    """Per-parameter or per-module gradient statistics.

    Each entry has ``numel, mean, std (population), min, max, norm,
    zero_fraction, nan_count, inf_count`` computed over the group's
    gradient elements. Module groups are combined exactly from per-tensor
    moments, without concatenating gradients.
    """
    key = group_key(group_by, depth)
    groups: dict[str, Moments] = {}
    for name, g in _grads(model):
        k = key(name)
        groups[k] = groups.get(k, Moments.empty()).merge(tensor_moments(g))
    return {k: m.summary() for k, m in groups.items()}

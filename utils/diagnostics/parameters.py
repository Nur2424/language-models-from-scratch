"""Parameter summaries and **measured** optimizer-update statistics.

Why measure instead of estimate
-------------------------------
For plain SGD the update is ``Δθ = -lr · g``, so ``lr · std(g)`` is the update
size. For Adam/AdamW it is not: the step is approximately
``-lr · m̂ / (√v̂ + ε)`` (plus decoupled weight decay ``-lr · λ · θ``), which is
normalised per element. A parameter with tiny gradients can still receive a
step of size ≈ ``lr``. Any ``lr · grad`` proxy is therefore wrong for
adaptive optimisers. :class:`ParameterSnapshot` copies θ before
``optimizer.step()`` and compares it with θ afterwards, which is correct for
every optimiser, scheduler and gradient-clipping setup.

Usage::

    snap = ParameterSnapshot.capture(model)   # after backward, before step
    optimizer.step()
    stats = snap.update_stats(model)          # then drop `snap`

Memory: one copy of the selected parameters on their device. Capture only on
scheduled diagnostic steps (see :class:`utils.diagnostics.schedule.DiagnosticsSchedule`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from utils.diagnostics import GroupBy, group_key
from utils.statistics import Moments, tensor_moments


def parameter_summaries(model: nn.Module, *, group_by: GroupBy = "parameter", depth: int = 1) -> dict[str, dict[str, Any]]:
    """Per-parameter or per-module statistics of parameter values
    (``numel, mean, std, min, max, norm, zero_fraction, nan/inf counts``)."""
    key = group_key(group_by, depth)
    groups: dict[str, Moments] = {}
    for name, p in model.named_parameters():
        k = key(name)
        groups[k] = groups.get(k, Moments.empty()).merge(tensor_moments(p))
    return {k: m.summary() for k, m in groups.items()}


@dataclass
class ParameterSnapshot:
    """Detached copies of parameters taken immediately before an optimizer step."""

    before: dict[str, torch.Tensor]

    @classmethod
    @torch.no_grad()
    def capture(
        cls, model: nn.Module, *, names: Iterable[str] | None = None, trainable_only: bool = True
    ) -> ParameterSnapshot:
        """Copy the selected parameters (all trainable ones by default)."""
        params = dict(model.named_parameters())
        if names is not None:
            wanted = list(names)
            missing = [n for n in wanted if n not in params]
            if missing:
                raise KeyError(f"unknown parameter name(s): {missing}")
            params = {n: params[n] for n in wanted}
        if trainable_only:
            params = {n: p for n, p in params.items() if p.requires_grad}
        if not params:
            raise ValueError("ParameterSnapshot selected no parameters")
        return cls(before={n: p.detach().clone() for n, p in params.items()})

    @torch.no_grad()
    def update_stats(self, model: nn.Module, *, eps: float = 1e-12) -> dict[str, Any]:
        """Compare the snapshot with the model's current parameters.

        Returns ``{"per_parameter": {...}, "global": {...}}``. Per parameter:

        * ``update_norm``        ‖Δθ‖₂
        * ``param_norm``         ‖θ_before‖₂
        * ``relative_update``    ‖Δθ‖₂ / ‖θ_before‖₂
        * ``update_std``         std(Δθ) (population)
        * ``log10_update_to_param_std``  log10(std(Δθ) / std(θ_before)) — the
          measured version of the classic "update:data ratio" (``None`` when
          undefined, e.g. a parameter with zero spread or no change)
        * ``update_max_abs``     max |Δθ|

        ``global`` aggregates over all snapshotted parameters:
        ``update_norm``, ``param_norm`` and ``relative_update``.
        """
        current = dict(model.named_parameters())
        per: dict[str, dict[str, Any]] = {}
        total_update_sq = 0.0
        total_param_sq = 0.0
        for name, before in self.before.items():
            if name not in current:
                raise KeyError(f"parameter {name!r} no longer exists in the model")
            after = current[name].detach()
            if after.shape != before.shape:
                raise ValueError(f"parameter {name!r} changed shape {tuple(before.shape)} → {tuple(after.shape)}")
            delta = after.to(before.dtype) - before
            d = tensor_moments(delta).summary()
            b = tensor_moments(before).summary()
            u_norm, p_norm = d["norm"], b["norm"]
            ratio = None
            if d["std"] and b["std"] and d["std"] > 0 and b["std"] > 0:
                ratio = math.log10(d["std"] / b["std"])
            per[name] = {
                "update_norm": u_norm,
                "param_norm": p_norm,
                "relative_update": u_norm / (p_norm + eps),
                "update_std": d["std"],
                "log10_update_to_param_std": ratio,
                "update_max_abs": max(abs(d["min"]), abs(d["max"])) if d["min"] is not None else None,
                "nonfinite_update": d["nan_count"] + d["inf_count"],
            }
            total_update_sq += u_norm**2
            total_param_sq += p_norm**2
        g_u, g_p = math.sqrt(total_update_sq), math.sqrt(total_param_sq)
        return {
            "per_parameter": per,
            "global": {"update_norm": g_u, "param_norm": g_p, "relative_update": g_u / (g_p + eps)},
        }

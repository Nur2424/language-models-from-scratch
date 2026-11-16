"""Lightweight training diagnostics that return plain summaries (no plots, no tensors kept).

* :mod:`.schedule`     — when to collect (diagnostics are opt-in and periodic)
* :mod:`.gradients`    — gradient norms, per-parameter/per-module summaries, NaN/inf detection
* :mod:`.activations`  — hook-based activation summaries with safe cleanup
* :mod:`.parameters`   — parameter summaries and *measured* optimizer-update statistics

Shared statistics come from :mod:`utils.statistics`. Architecture-specific
analyses (attention maps, gates, expert routing) are intentionally absent.

Typical use inside a future training loop::

    sched = cfg.diagnostics                       # a DiagnosticsSchedule
    loss.backward()
    if sched.should_collect(step, "gradients"):
        grads = gradient_summaries(model, group_by="module")
    snap = ParameterSnapshot.capture(model) if sched.should_collect(step, "parameters") else None
    optimizer.step()
    if snap is not None:
        updates = snap.update_stats(model); del snap
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from utils.metrics import module_group

GroupBy = Literal["parameter", "module"] | Callable[[str], str]


def group_key(group_by: GroupBy, depth: int) -> Callable[[str], str]:
    """Return ``name -> group`` for ``"parameter"``, ``"module"`` (with depth) or a callable."""
    if callable(group_by):
        return group_by
    if group_by == "parameter":
        return lambda name: name
    if group_by == "module":
        return lambda name: module_group(name, depth)
    raise ValueError(f"group_by must be 'parameter', 'module' or a callable, got {group_by!r}")

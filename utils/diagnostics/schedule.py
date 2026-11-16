"""When to collect diagnostics.

Diagnostics cost device synchronisations and, for parameter updates, a copy of
the parameters. They are therefore **disabled by default** and, when enabled,
collected only every ``every_n_steps`` steps. This dataclass doubles as the
``diagnostics`` section of the experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DiagnosticKind = Literal["gradients", "parameters", "activations"]


@dataclass(frozen=True)
class DiagnosticsSchedule:
    """Which diagnostics to collect and how often.

    ``should_collect(step, kind)`` is true when diagnostics are enabled, that
    kind is switched on, and ``step % every_n_steps == 0`` (so step 0, the
    initial state, is included).
    """

    enabled: bool = False
    every_n_steps: int = 100
    gradients: bool = True
    parameters: bool = True
    activations: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.every_n_steps, bool) or not isinstance(self.every_n_steps, int) or self.every_n_steps < 1:
            raise ValueError(f"every_n_steps must be a positive int, got {self.every_n_steps!r}")

    def should_collect(self, step: int, kind: DiagnosticKind) -> bool:
        if kind not in ("gradients", "parameters", "activations"):
            raise ValueError(f"unknown diagnostic kind {kind!r}")
        if step < 0:
            raise ValueError("step must be non-negative")
        return self.enabled and getattr(self, kind) and step % self.every_n_steps == 0

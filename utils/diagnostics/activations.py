"""Hook-based activation summaries.

:class:`ActivationMonitor` registers forward hooks on selected submodules and,
on every forward call while attached, reduces the module output to mergeable
:class:`~utils.statistics.Moments` (a handful of floats). No activation tensor
is stored and no graph is retained: hooks run under ``no_grad`` on a detached
view. Summaries accumulate across forward calls until :meth:`reset`.

Use it as a context manager so hooks are always removed, even on error::

    with ActivationMonitor(model, module_types=(nn.Tanh,),
                           saturation_by_type={nn.Tanh: tanh_saturation()}) as mon:
        model(xb)
    stats = mon.summaries()       # {"net.2": {"mean": ..., "saturation_fraction": ...}, ...}

Only attach it on diagnostic steps: each hooked forward call costs a device
synchronisation.

Module outputs that are tuples/lists (e.g. ``nn.LSTM`` returns
``(output, (h, c))``) are summarised through ``output_selector``; the default
takes the first tensor element. Outputs with no tensor are counted in
``skipped_calls``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn

from utils.statistics import Moments, Predicate, tensor_moments

OutputSelector = Callable[[nn.Module, Any], torch.Tensor | None]


def tanh_saturation(threshold: float = 0.97) -> Predicate:
    """Predicate for saturated tanh outputs: ``|y| > threshold``."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    return lambda y: y.abs() > threshold


def sigmoid_saturation(margin: float = 0.03) -> Predicate:
    """Predicate for saturated sigmoid outputs: ``y < margin`` or ``y > 1 - margin``."""
    if not 0 < margin < 0.5:
        raise ValueError("margin must be in (0, 0.5)")
    return lambda y: (y < margin) | (y > 1 - margin)


def default_output_selector(module: nn.Module, output: Any) -> torch.Tensor | None:
    """Return the output if it is a tensor, else the first element of a tuple/list if it is one."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


class ActivationMonitor:
    """Collect activation summaries for selected submodules of ``model``.

    Parameters
    ----------
    model:
        The module to inspect.
    module_names:
        Exact names from ``model.named_modules()`` to monitor.
    module_types:
        Monitor every submodule that is an instance of one of these types.
        If neither selector is given, every leaf module is monitored.
    saturation:
        Per-module-name saturation predicates.
    saturation_by_type:
        Per-type saturation predicates (used when no per-name predicate exists).
    output_selector:
        Maps ``(module, output)`` to the tensor to summarise, or ``None`` to skip.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        module_names: Iterable[str] | None = None,
        module_types: tuple[type[nn.Module], ...] | None = None,
        saturation: Mapping[str, Predicate] | None = None,
        saturation_by_type: Mapping[type[nn.Module], Predicate] | None = None,
        output_selector: OutputSelector = default_output_selector,
    ) -> None:
        named = dict(model.named_modules())
        if module_names is not None:
            names = list(module_names)
            unknown = [n for n in names if n not in named]
            if unknown:
                raise KeyError(f"unknown module name(s) {unknown}; see model.named_modules()")
            selected = {n: named[n] for n in names}
        elif module_types is not None:
            selected = {n: m for n, m in named.items() if n and isinstance(m, module_types)}
        else:
            selected = {n: m for n, m in named.items() if n and not any(True for _ in m.children())}
        if not selected:
            raise ValueError("ActivationMonitor selected no modules")
        self._modules = selected
        self._saturation = dict(saturation or {})
        self._saturation_by_type = dict(saturation_by_type or {})
        self._selector = output_selector
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._moments: dict[str, Moments] = {}
        self._calls: dict[str, int] = {}
        self._skipped: dict[str, int] = {}

    # -------------------------------------------------------- lifecycle
    @property
    def attached(self) -> bool:
        return bool(self._handles)

    @property
    def module_names(self) -> list[str]:
        return list(self._modules)

    def attach(self) -> ActivationMonitor:
        """Register hooks. Raises if already attached (prevents double counting)."""
        if self._handles:
            raise RuntimeError("ActivationMonitor is already attached")
        try:
            for name, module in self._modules.items():
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        except BaseException:
            self.remove()
            raise
        return self

    def remove(self) -> None:
        """Remove all hooks. Idempotent."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Discard accumulated summaries (hooks stay attached)."""
        self._moments.clear()
        self._calls.clear()
        self._skipped.clear()

    def __enter__(self) -> ActivationMonitor:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.remove()

    # ------------------------------------------------------------ hooks
    def _predicate(self, name: str, module: nn.Module) -> Predicate | None:
        if name in self._saturation:
            return self._saturation[name]
        for typ, pred in self._saturation_by_type.items():
            if isinstance(module, typ):
                return pred
        return None

    def _make_hook(self, name: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, _inputs: Any, output: Any) -> None:
            with torch.no_grad():
                t = self._selector(module, output)
                if t is None:
                    self._skipped[name] = self._skipped.get(name, 0) + 1
                    return
                m = tensor_moments(t.detach(), saturation=self._predicate(name, module))
                self._moments[name] = self._moments.get(name, Moments.empty()).merge(m)
                self._calls[name] = self._calls.get(name, 0) + 1

        return hook

    # ---------------------------------------------------------- results
    def summaries(self) -> dict[str, dict[str, Any]]:
        """Summary per monitored module that produced at least one output."""
        out: dict[str, dict[str, Any]] = {}
        for name in self._modules:
            if name in self._moments:
                out[name] = {
                    **self._moments[name].summary(),
                    "calls": self._calls[name],
                    "skipped_calls": self._skipped.get(name, 0),
                }
            elif name in self._skipped:
                out[name] = {"calls": 0, "skipped_calls": self._skipped[name]}
        return out

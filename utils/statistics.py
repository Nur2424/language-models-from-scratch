"""Numerically stable statistical summaries.

Two families of tools live here:

* **Scalar summaries** — :func:`summarize` and :class:`RunningStats` for
  sequences of Python floats (e.g. per-seed final NLLs, per-step losses).
* **Tensor moments** — :func:`tensor_moments`, :meth:`Moments.merge` and
  :func:`tensor_summary` for summarising gradients, activations and
  parameters without keeping tensors around.

Standard deviation conventions
------------------------------
* :func:`summarize` / :class:`RunningStats` report the **sample** standard
  deviation (``ddof = 1``), appropriate when the values are a sample from a
  distribution (e.g. seeds). ``std`` is ``None`` for fewer than two values.
* Tensor summaries report the **population** standard deviation
  (``ddof = 0``) of the elements, which is well-defined for one element and
  is what "std of this weight matrix" normally means.

Variance is computed from centred second moments (Welford / Chan et al.),
never as ``E[x²] - E[x]²``, which loses precision when ``|mean| >> std``
(e.g. LayerNorm gains near 1).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch


# --------------------------------------------------------------- scalars
@dataclass(frozen=True)
class SummaryStats:
    """Summary of a finite sample of scalars."""

    count: int
    mean: float
    std: float | None  # sample std (ddof=1); None when count < 2
    sem: float | None  # standard error of the mean
    min: float
    max: float
    median: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def summarize(values: Iterable[float]) -> SummaryStats:
    """Summarise a non-empty collection of finite numbers.

    Raises ``ValueError`` for empty input or non-finite values: a NaN loss in
    a set of seeds is a result to investigate, not something to average over.
    """
    xs = [float(v) for v in values]
    if not xs:
        raise ValueError("cannot summarise an empty collection")
    bad = [v for v in xs if not math.isfinite(v)]
    if bad:
        raise ValueError(f"cannot summarise non-finite values: {bad[:5]}")
    rs = RunningStats()
    rs.update_many(xs)
    s = sorted(xs)
    mid = len(s) // 2
    median = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    std = rs.std
    return SummaryStats(
        count=len(xs),
        mean=rs.mean,
        std=std,
        sem=(std / math.sqrt(len(xs))) if std is not None else None,
        min=s[0],
        max=s[-1],
        median=median,
    )


class RunningStats:
    """Streaming mean / variance / min / max via Welford's algorithm.

    Example
    -------
    >>> rs = RunningStats(); rs.update_many([2.0, 4.0, 6.0])
    >>> rs.mean, rs.variance
    (4.0, 4.0)
    """

    __slots__ = ("count", "_mean", "_m2", "min", "max")

    def __init__(self) -> None:
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, x: float) -> None:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"RunningStats received a non-finite value: {x}")
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (x - self._mean)
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def update_many(self, xs: Iterable[float]) -> None:
        for x in xs:
            self.update(x)

    def merge(self, other: RunningStats) -> RunningStats:
        """Return a new accumulator equal to having seen both streams."""
        out = RunningStats()
        n = self.count + other.count
        if n == 0:
            return out
        delta = other._mean - self._mean
        out.count = n
        out._mean = self._mean + delta * other.count / n
        out._m2 = self._m2 + other._m2 + delta * delta * self.count * other.count / n
        out.min = min(self.min, other.min)
        out.max = max(self.max, other.max)
        return out

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("mean of an empty RunningStats")
        return self._mean

    @property
    def variance(self) -> float | None:
        """Sample variance (ddof=1); ``None`` for fewer than two values."""
        return self._m2 / (self.count - 1) if self.count > 1 else None

    @property
    def std(self) -> float | None:
        v = self.variance
        return math.sqrt(v) if v is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self._mean if self.count else None,
            "std": self.std,
            "min": self.min if self.count else None,
            "max": self.max if self.count else None,
        }


# --------------------------------------------------------------- tensors
Predicate = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Moments:
    """Mergeable sufficient statistics of a set of tensor elements.

    ``count`` and the moments cover *finite* elements only; NaN and ±inf are
    counted separately so a single bad value does not hide the rest of the
    distribution.
    """

    count: int
    mean: float
    m2: float  # sum of squared deviations from the mean
    min: float
    max: float
    sum_sq: float  # sum of x², for the L2 norm
    zeros: int
    nan: int
    inf: int
    saturated: int | None = None

    @staticmethod
    def empty() -> Moments:
        return Moments(0, 0.0, 0.0, math.inf, -math.inf, 0.0, 0, 0, 0, None)

    def merge(self, other: Moments) -> Moments:
        """Chan et al. parallel combination of two sets of elements."""
        n = self.count + other.count
        if n == 0:
            mean, m2 = 0.0, 0.0
        else:
            delta = other.mean - self.mean
            mean = self.mean + delta * other.count / n
            m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / n
        sat = None
        if self.saturated is not None or other.saturated is not None:
            sat = (self.saturated or 0) + (other.saturated or 0)
        return Moments(
            count=n,
            mean=mean,
            m2=m2,
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            sum_sq=self.sum_sq + other.sum_sq,
            zeros=self.zeros + other.zeros,
            nan=self.nan + other.nan,
            inf=self.inf + other.inf,
            saturated=sat,
        )

    def summary(self) -> dict[str, Any]:
        """Final summary dict (JSON-friendly; ``None`` where undefined)."""
        total = self.count + self.nan + self.inf
        has = self.count > 0
        out: dict[str, Any] = {
            "numel": total,
            "mean": self.mean if has else None,
            "std": math.sqrt(max(self.m2, 0.0) / self.count) if has else None,
            "min": self.min if has else None,
            "max": self.max if has else None,
            "norm": math.sqrt(self.sum_sq),
            "zero_fraction": self.zeros / total if total else None,
            "nan_count": self.nan,
            "inf_count": self.inf,
        }
        if self.saturated is not None:
            out["saturation_fraction"] = self.saturated / total if total else None
        return out


def _work_dtype(t: torch.Tensor) -> torch.dtype:
    # float64 when the input already is float64 (and the device supports it);
    # otherwise float32 — MPS has no float64, and half precision is too coarse.
    return torch.float64 if t.dtype == torch.float64 else torch.float32


@torch.no_grad()
def tensor_moments(t: torch.Tensor, *, saturation: Predicate | None = None) -> Moments:
    """Compute :class:`Moments` of all elements of ``t``.

    Costs two device→host synchronisations (a finiteness check and the final
    transfer), so call it on scheduled diagnostic steps, not every step.

    The tensor is detached and never stored. ``saturation`` is an optional
    elementwise predicate (e.g. ``lambda x: x.abs() > 0.97``) whose true
    fraction is reported as ``saturation_fraction``.
    """
    x = t.detach().reshape(-1)
    if x.is_complex():
        raise TypeError("complex tensors are not supported")
    x = x.to(_work_dtype(x))
    numel = x.numel()
    if numel == 0:
        return Moments.empty() if saturation is None else Moments(0, 0.0, 0.0, math.inf, -math.inf, 0.0, 0, 0, 0, 0)
    finite = torch.isfinite(x)
    n_nan = torch.isnan(x).sum()
    n_inf = torch.isinf(x).sum()
    n_finite = finite.sum()
    if not bool(finite.all()):
        x = x[finite]
    if x.numel() == 0:
        sat = 0 if saturation is not None else None
        return Moments(0, 0.0, 0.0, math.inf, -math.inf, 0.0, 0, int(n_nan), int(n_inf), sat)
    mean = x.mean()
    centred = x - mean
    pieces = [
        n_finite.to(x.dtype),
        mean,
        (centred * centred).sum(),
        x.min(),
        x.max(),
        (x * x).sum(),
        (x == 0).sum().to(x.dtype),
        n_nan.to(x.dtype),
        n_inf.to(x.dtype),
    ]
    if saturation is not None:
        pieces.append(saturation(x).sum().to(x.dtype))
    vals = torch.stack(pieces).tolist()  # single device→host transfer
    return Moments(
        count=int(vals[0]),
        mean=vals[1],
        m2=vals[2],
        min=vals[3],
        max=vals[4],
        sum_sq=vals[5],
        zeros=int(vals[6]),
        nan=int(vals[7]),
        inf=int(vals[8]),
        saturated=int(vals[9]) if saturation is not None else None,
    )


def tensor_summary(t: torch.Tensor, *, saturation: Predicate | None = None) -> dict[str, Any]:
    """Summary dict of a tensor: numel, mean, std (ddof=0), min, max, L2 norm,
    zero fraction, NaN/inf counts and optional saturation fraction."""
    return tensor_moments(t, saturation=saturation).summary()


def merge_moments(items: Iterable[Moments]) -> Moments:
    """Merge any number of :class:`Moments` (e.g. all parameters of a layer)."""
    out = Moments.empty()
    for m in items:
        out = out.merge(m)
    return out

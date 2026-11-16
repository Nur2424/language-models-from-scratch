"""Reusable metrics: token-level NLL with explicit reduction, perplexity, parameter counts.

Why reductions are explicit
---------------------------
The correct dataset-level NLL is the mean over **tokens**, not over batches:

    NLL = (Σ_batches Σ_valid-tokens −log p(target)) / (Σ_batches #valid tokens)

Averaging per-batch means weights a batch with 2 valid tokens the same as one
with 10. Example: batch A has 10 tokens with mean NLL 2.0, batch B has 2
tokens with mean 3.0. Mean of means = 2.5; true per-token NLL =
(20 + 6) / 12 = 2.1667. :class:`NLLAccumulator` always accumulates the sum
and the count separately.

All functions take ``logits`` of shape ``(..., C)`` and ``targets`` of shape
``(...)``; targets equal to ``ignore_index`` (default ``-100``) are excluded.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from utils.vocabulary import IGNORE_INDEX

Reduction = Literal["sum", "mean", "none"]


def _flatten(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} must be targets shape {tuple(targets.shape)} plus a class dim"
        )
    if targets.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"targets must be integer class ids, got {targets.dtype}")
    return logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long()


def count_valid_targets(targets: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> int:
    """Number of targets that are not ``ignore_index``."""
    return int((targets != ignore_index).sum().item())


def token_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: Reduction,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Cross-entropy (negative log-likelihood, in nats) per target token.

    ``reduction`` is required so the semantics are always visible at the call site:

    * ``"sum"``  — sum over valid tokens (use for accumulation).
    * ``"mean"`` — sum / number of valid tokens in *this* call (a valid
      training loss); raises if there are no valid tokens rather than
      returning NaN.
    * ``"none"`` — per-position NLL with the shape of ``targets``; ignored
      positions are 0.

    Computed in float32 (or float64 if the logits are float64) for stability
    with half-precision logits. The result keeps the autograd graph.
    """
    flat_logits, flat_targets = _flatten(logits, targets)
    if flat_logits.dtype in (torch.float16, torch.bfloat16):
        flat_logits = flat_logits.float()
    valid = flat_targets != ignore_index
    if valid.any():
        bad = flat_targets[valid]
        if int(bad.min()) < 0 or int(bad.max()) >= flat_logits.shape[-1]:
            raise ValueError(
                f"targets must lie in [0, {flat_logits.shape[-1]}) or equal ignore_index={ignore_index}"
            )
    if reduction == "none":
        per = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="none")
        return per.reshape(targets.shape)
    total = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum")
    if reduction == "sum":
        return total
    if reduction == "mean":
        n = int(valid.sum().item())
        if n == 0:
            raise ValueError("mean NLL is undefined: no valid targets in this batch")
        return total / n
    raise ValueError(f"reduction must be 'sum', 'mean' or 'none', got {reduction!r}")


def perplexity(mean_nll: float) -> float:
    """``exp(mean NLL)`` for an NLL measured in nats per token."""
    return math.exp(mean_nll)


def bits_per_token(mean_nll: float) -> float:
    """Convert a mean NLL in nats to bits per token (bits per character here)."""
    return mean_nll / math.log(2.0)


class NLLAccumulator:
    """Exact dataset-level NLL across batches with different valid-token counts.

    >>> acc = NLLAccumulator()
    >>> with torch.no_grad():
    ...     for xb, yb in batches:
    ...         acc.update(model(xb), yb)
    >>> acc.mean_nll, acc.perplexity

    The running sum is kept as a Python float (float64).
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        self.ignore_index = ignore_index
        self.total_nll = 0.0
        self.num_tokens = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Add one batch. Detaches; never keeps a graph."""
        with torch.no_grad():
            s = token_nll(logits.detach(), targets, reduction="sum", ignore_index=self.ignore_index)
        self.update_from_sum(float(s.item()), count_valid_targets(targets, self.ignore_index))

    def update_from_sum(self, nll_sum: float, num_tokens: int) -> None:
        """Add a pre-computed (sum, count) pair, e.g. from a counting model."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if not math.isfinite(nll_sum):
            raise ValueError(f"non-finite NLL sum: {nll_sum}")
        self.total_nll += float(nll_sum)
        self.num_tokens += int(num_tokens)

    def merge(self, other: NLLAccumulator) -> None:
        self.update_from_sum(other.total_nll, other.num_tokens)

    def reset(self) -> None:
        self.total_nll, self.num_tokens = 0.0, 0

    @property
    def mean_nll(self) -> float:
        if self.num_tokens == 0:
            raise ValueError("no tokens accumulated")
        return self.total_nll / self.num_tokens

    @property
    def perplexity(self) -> float:
        return perplexity(self.mean_nll)

    @property
    def bits_per_token(self) -> float:
        return bits_per_token(self.mean_nll)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "nll": self.mean_nll,
            "perplexity": self.perplexity,
            "bits_per_token": self.bits_per_token,
            "num_tokens": self.num_tokens,
        }


# ------------------------------------------------------------ parameters
@dataclass(frozen=True)
class ParameterCount:
    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable


def count_parameters(model: nn.Module) -> ParameterCount:
    """Count unique parameters (tied/shared tensors are counted once).

    Buffers (e.g. BatchNorm running statistics) are not parameters and are
    not counted.
    """
    params = list(model.parameters())  # nn.Module de-duplicates shared parameters
    return ParameterCount(
        total=sum(p.numel() for p in params),
        trainable=sum(p.numel() for p in params if p.requires_grad),
    )


def module_group(name: str, depth: int = 1) -> str:
    """Group key for a parameter name: the first ``depth`` module-path parts.

    ``"blocks.0.attn.key.weight"`` → ``"blocks"`` (depth 1) or ``"blocks.0"``
    (depth 2). Parameters registered directly on the root module map to
    ``"(root)"``.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    parts = name.split(".")[:-1]
    return ".".join(parts[:depth]) if parts else "(root)"


def parameter_breakdown(
    model: nn.Module,
    *,
    depth: int = 1,
    group_fn: Callable[[str], str] | None = None,
) -> dict[str, ParameterCount]:
    """Parameter counts per module group, in registration order.

    By default groups by module path (see :func:`module_group`). Pass
    ``group_fn(param_name) -> group`` to bucket by category instead (e.g.
    "attention", "feedforward", "embeddings"). Tied parameters are counted
    once, under the first name that refers to them.
    """
    key = group_fn if group_fn is not None else (lambda n: module_group(n, depth))
    totals: dict[str, list[int]] = {}
    for name, p in model.named_parameters():
        g = key(name)
        entry = totals.setdefault(g, [0, 0])
        entry[0] += p.numel()
        entry[1] += p.numel() if p.requires_grad else 0
    return {g: ParameterCount(total=t, trainable=tr) for g, (t, tr) in totals.items()}

"""Model-agnostic token-sampling primitives.

These functions operate on next-token logits only. They do not call models,
manage hidden states, KV caches or context windows: recurrent models,
Transformers, SSMs and fixed-window MLPs drive generation differently, and
each model will own its generation loop and call :func:`sample_next_token`.

Processing order (the common convention)::

    logits ─▶ temperature ─▶ top-k ─▶ top-p ─▶ softmax ─▶ sample

* ``temperature = 0`` means greedy decoding (argmax; ties → lowest id).
* ``top_k`` keeps the k highest logits. Values tied with the k-th logit are
  also kept, so slightly more than k tokens may survive.
* ``top_p`` keeps the smallest set of highest-probability tokens whose
  cumulative probability is ≥ p (the most likely token is always kept).

Reproducibility
---------------
Pass an explicit ``torch.Generator``. If its device differs from the logits'
device, the (tiny, ``vocab``-sized) logits are moved to the generator's
device for sampling. With a CPU generator the same seed gives the same token
stream regardless of the device the model runs on — up to floating-point
differences in the logits themselves.
"""

from __future__ import annotations

import torch


class SamplingError(ValueError):
    """Raised for invalid sampling parameters or unusable logits."""


def validate_sampling_params(
    temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None
) -> None:
    """Raise :class:`SamplingError` unless ``temperature >= 0``, ``top_k >= 1``, ``0 < top_p <= 1``."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise SamplingError(f"temperature must be a number, got {temperature!r}")
    if not temperature >= 0 or temperature == float("inf"):  # also rejects NaN
        raise SamplingError(f"temperature must be finite and >= 0, got {temperature}")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise SamplingError(f"top_k must be a positive int or None, got {top_k!r}")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise SamplingError(f"top_p must be in (0, 1] or None, got {top_p!r}")


def _check_logits(logits: torch.Tensor) -> None:
    if not torch.is_floating_point(logits):
        raise SamplingError(f"logits must be floating point, got {logits.dtype}")
    if logits.ndim not in (1, 2):
        raise SamplingError(f"logits must have shape (V,) or (B, V), got {tuple(logits.shape)}")
    if logits.shape[-1] == 0:
        raise SamplingError("logits have an empty vocabulary dimension")
    if torch.isnan(logits).any():
        raise SamplingError("logits contain NaN")
    if torch.isposinf(logits).any():
        raise SamplingError("logits contain +inf")
    if torch.isneginf(logits).all(dim=-1).any():
        raise SamplingError("every logit in a row is -inf; nothing can be sampled")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by ``temperature`` (> 0). Use ``temperature=0`` via greedy sampling instead."""
    if not temperature > 0:
        raise SamplingError("apply_temperature needs temperature > 0; temperature 0 means greedy")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Set logits below the k-th largest (per row) to ``-inf``. ``k`` ≥ vocab is a no-op."""
    validate_sampling_params(top_k=k)
    if k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest top set with cumulative probability ≥ ``p``."""
    validate_sampling_params(top_p=p)
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    # A token is removed if the mass of strictly-more-likely tokens already reaches p.
    mass_before = torch.cumsum(probs, dim=-1) - probs
    remove_sorted = mass_before >= p
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def filtered_logits(
    logits: torch.Tensor, *, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Apply temperature, top-k and top-p to ``(V,)`` or ``(B, V)`` logits.

    ``temperature=0`` is not allowed here (it has no distribution); use
    :func:`sample_next_token`, which handles greedy decoding.
    """
    validate_sampling_params(temperature, top_k, top_p)
    _check_logits(logits)
    out = apply_temperature(logits.float(), temperature)
    if top_k is not None:
        out = top_k_filter(out, top_k)
    if top_p is not None:
        out = top_p_filter(out, top_p)
    return out


def next_token_probs(
    logits: torch.Tensor, *, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Sampling distribution after all filters (rows sum to 1)."""
    return torch.softmax(filtered_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p), dim=-1)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per row.

    ``logits``: ``(V,)`` → returns a 0-d LongTensor; ``(B, V)`` → ``(B,)``.
    The result is on the same device as ``logits``. Raises
    :class:`SamplingError` for invalid parameters or NaN/+inf/all-masked rows.
    """
    validate_sampling_params(temperature, top_k, top_p)
    _check_logits(logits)
    squeeze = logits.ndim == 1
    rows = logits.unsqueeze(0) if squeeze else logits
    if temperature == 0:
        ids = rows.argmax(dim=-1)
    else:
        probs = next_token_probs(rows, temperature=temperature, top_k=top_k, top_p=top_p)
        if generator is not None and probs.device != generator.device:
            probs = probs.to(generator.device)
        ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1).to(logits.device)
    return ids[0] if squeeze else ids


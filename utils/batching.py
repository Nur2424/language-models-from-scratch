"""Reproducible mini-batch ordering: seeded, shuffled epochs without replacement.

:class:`EpochSampler` yields **index tensors**, never data, so it works for any
dataset (context windows, sequences, …) and makes no model assumptions. The
caller indexes its own tensors, e.g. ``X[idx.to(device)]``.

Ordering (approved option A)
----------------------------
The order of epoch ``e`` is a pure function of ``(seed, e, N)``::

    π_e = torch.randperm(N, generator=make_generator(derive_seed(seed, f"epoch:{e}")))

The generator is an explicit CPU generator created for that epoch. Hence:

* every example appears exactly once per epoch (sampling without replacement);
* the order is independent of global RNG state, other generators, the device
  and the batch size;
* resuming needs only the epoch number (and, mid-epoch, a batch index); no RNG
  state has to be saved;
* different seeds or epochs give different orders with overwhelming
  probability. This is not a guarantee.

Batches are consecutive slices of ``π_e``: ``⌈N/B⌉ − 1`` batches of size ``B``,
then one of size ``r = N − B(⌈N/B⌉ − 1)`` with ``1 ≤ r ≤ B``. The final partial
batch is kept.

Loss weighting
--------------
For a gradient step on batch ``b`` the usual objective is the per-batch mean
``ℓ_b = (1/|b|) Σ_{i∈b} ℓ_i``. Over an epoch:

* the **example-weighted epoch loss** is ``(1/N) Σ_b |b|·ℓ_b = (1/N) Σ_i ℓ_i``.
  Compute it with :class:`utils.metrics.NLLAccumulator`
  (``update(logits, targets)`` or ``update_from_sum(ℓ_b·|b|, |b|)``);
* the **mean of batch means** ``(1/S) Σ_b ℓ_b`` over-weights the ``r`` examples of
  a partial batch by ``B/r``. Never report it as the epoch loss;
* a running epoch loss is accumulated while parameters change. It is not the
  loss of the end-of-epoch parameters, which must be evaluated separately.

How the partial batch enters the *gradient* is a model-specific training
decision (approved U3). Mean reduction gives each of its examples influence
``1/r``; ``token_nll(reduction="sum") / batch_size`` gives every example ``1/B``
(and a smaller final step). Use :meth:`EpochSampler.batch_sizes` for either.

Limitations
-----------
``torch.randperm`` orders are reproducible within one PyTorch version; the
algorithm may change between versions (run metadata records the version).
Bit-exact *training* additionally needs the device-level settings in
:func:`utils.reproducibility.configure_determinism`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch

from utils.reproducibility import derive_seed, make_generator, validate_seed

SAMPLER_FORMAT = "epoch_sampler/v1"


def _check_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


class EpochSampler:
    """Seeded, shuffled, without-replacement epoch ordering of example indices.

    Parameters
    ----------
    num_examples:
        ``N ≥ 1``: dataset size.
    batch_size:
        ``B ≥ 1``. ``B > N`` gives a single batch of size ``N``.
    seed:
        Base seed in ``[0, 2**32 − 1]``. Convention: ``derive_seed(base_seed, "data")``,
        which keeps data ordering independent of the ``init`` and ``generation`` streams.
    """

    def __init__(self, num_examples: int, batch_size: int, *, seed: int) -> None:
        self.num_examples = _check_int(num_examples, "num_examples", minimum=1)
        self.batch_size = _check_int(batch_size, "batch_size", minimum=1)
        if isinstance(seed, bool):
            raise TypeError("seed must be an int, got bool")
        self.seed = validate_seed(seed)

    @property
    def num_batches(self) -> int:
        """``⌈N / B⌉``."""
        return -(-self.num_examples // self.batch_size)

    def batch_sizes(self) -> list[int]:
        """Sizes of the batches of every epoch: ``[B, …, B, r]`` with ``1 ≤ r ≤ B``."""
        full, remainder = divmod(self.num_examples, self.batch_size)
        return [self.batch_size] * full + ([remainder] if remainder else [])

    def permutation(self, epoch: int) -> torch.Tensor:
        """``π_e``: ``[N]`` int64 CPU permutation of ``0 .. N−1`` for epoch ``epoch ≥ 0``."""
        epoch = _check_int(epoch, "epoch", minimum=0)
        generator = make_generator(derive_seed(self.seed, f"epoch:{epoch}"))
        return torch.randperm(self.num_examples, generator=generator)

    def batches(self, epoch: int, *, start_batch: int = 0) -> Iterator[torch.Tensor]:
        """Yield the index batches of ``epoch`` in order, starting at batch ``start_batch``.

        Each batch is a ``[b]`` int64 CPU tensor, a consecutive slice of
        :meth:`permutation`. ``batches(e, start_batch=j)`` yields exactly
        ``list(batches(e))[j:]``. That is mid-epoch resumption.
        """
        _check_int(start_batch, "start_batch", minimum=0)
        if start_batch > self.num_batches:
            raise ValueError(f"start_batch must be <= num_batches ({self.num_batches}), got {start_batch}")
        order = self.permutation(epoch)
        for j in range(start_batch, self.num_batches):
            yield order[j * self.batch_size : (j + 1) * self.batch_size]

    # ------------------------------------------------------------ resumption
    def state_dict(self) -> dict[str, int | str]:
        """Plain-data description for checkpoints (e.g. ``save_checkpoint(extra={"sampler": ...})``).

        The ordering is fully determined by these fields and the epoch number,
        so no RNG state is needed to resume.
        """
        return {
            "format": SAMPLER_FORMAT,
            "num_examples": self.num_examples,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], *, num_examples: int, batch_size: int) -> EpochSampler:
        """Rebuild a sampler for resumption, rejecting a changed dataset size or batch size.

        ``num_examples`` and ``batch_size`` are the *current* run's values. If they
        differ from the saved ones, resuming would silently change the ordering,
        so a ``ValueError`` is raised instead.
        """
        expected = {"format", "num_examples", "batch_size", "seed"}
        if not isinstance(state, Mapping) or set(state) != expected:
            got = sorted(state) if isinstance(state, Mapping) else type(state).__name__
            raise ValueError(f"sampler state must have keys {sorted(expected)}, got {got}")
        if state["format"] != SAMPLER_FORMAT:
            raise ValueError(f"unsupported sampler state format {state['format']!r} (expected {SAMPLER_FORMAT!r})")
        if state["num_examples"] != num_examples:
            raise ValueError(f"dataset size changed: saved {state['num_examples']}, current {num_examples}")
        if state["batch_size"] != batch_size:
            raise ValueError(f"batch size changed: saved {state['batch_size']}, current {batch_size}")
        return cls(num_examples, batch_size, seed=state["seed"])

    def __repr__(self) -> str:
        return (f"EpochSampler(num_examples={self.num_examples}, batch_size={self.batch_size}, "
                f"seed={self.seed}, num_batches={self.num_batches})")

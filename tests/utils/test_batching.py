"""Tests for utils.batching.EpochSampler (approved shared-infrastructure spec)."""

from __future__ import annotations

import random

import pytest
import torch

from utils.batching import SAMPLER_FORMAT, EpochSampler
from utils.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from utils.metrics import NLLAccumulator, token_nll
from utils.reproducibility import capture_rng_state, derive_seed, make_generator

CASES = [(10, 4), (8, 4), (3, 10), (7, 1), (7, 7), (1, 1), (101, 32)]


# --------------------------------------------------------------- coverage
@pytest.mark.parametrize("n, b", CASES)
def test_each_epoch_is_an_exact_permutation(n, b):
    s = EpochSampler(n, b, seed=7)
    for epoch in range(3):
        batches = list(s.batches(epoch))
        flat = torch.cat(batches)
        assert flat.numel() == n
        assert torch.equal(torch.sort(flat).values, torch.arange(n))          # every index once, no duplicates
        assert torch.equal(flat, s.permutation(epoch))                        # consecutive slices of π_e
        assert all(t.dtype == torch.int64 and t.device.type == "cpu" and t.ndim == 1 for t in batches)


@pytest.mark.parametrize("n, b, sizes", [(10, 4, [4, 4, 2]), (8, 4, [4, 4]), (3, 10, [3]), (7, 1, [1] * 7),
                                         (101, 32, [32, 32, 32, 5])])
def test_batch_sizes_keep_partial_batch(n, b, sizes):
    s = EpochSampler(n, b, seed=0)
    assert s.batch_sizes() == sizes and s.num_batches == len(sizes)
    assert [len(x) for x in s.batches(0)] == sizes


# --------------------------------------------------------- reproducibility
def test_same_seed_same_order_independent_of_global_rng_and_batch_size():
    a = EpochSampler(50, 8, seed=123).permutation(2)
    torch.manual_seed(999)
    random.seed(1)
    torch.rand(100)
    make_generator(5).manual_seed(5)
    b = EpochSampler(50, 8, seed=123).permutation(2)
    c = EpochSampler(50, 3, seed=123).permutation(2)                          # different batch size
    assert torch.equal(a, b) and torch.equal(a, c)


def test_order_is_derived_from_seed_and_epoch():
    expected = torch.randperm(20, generator=make_generator(derive_seed(11, "epoch:4")))
    assert torch.equal(EpochSampler(20, 6, seed=11).permutation(4), expected)


def test_different_seeds_and_epochs_give_different_orders():
    s = EpochSampler(100, 10, seed=1)
    assert not torch.equal(s.permutation(0), EpochSampler(100, 10, seed=2).permutation(0))
    orders = [s.permutation(e) for e in range(5)]
    assert all(not torch.equal(orders[i], orders[j]) for i in range(5) for j in range(i + 1, 5))


# ---------------------------------------------------------------- resume
@pytest.mark.parametrize("j", [0, 1, 3, 4])
def test_start_batch_yields_the_tail(j):
    s = EpochSampler(101, 32, seed=3)
    full = list(s.batches(1))
    tail = list(s.batches(1, start_batch=j))
    assert len(tail) == len(full) - j and all(torch.equal(x, y) for x, y in zip(tail, full[j:]))


def test_state_dict_round_trip_and_mismatch_rejection():
    s = EpochSampler(101, 32, seed=3)
    state = s.state_dict()
    assert state == {"format": SAMPLER_FORMAT, "num_examples": 101, "batch_size": 32, "seed": 3}
    r = EpochSampler.from_state_dict(state, num_examples=101, batch_size=32)
    assert all(torch.equal(r.permutation(e), s.permutation(e)) for e in range(3))
    with pytest.raises(ValueError, match="dataset size changed"):
        EpochSampler.from_state_dict(state, num_examples=100, batch_size=32)
    with pytest.raises(ValueError, match="batch size changed"):
        EpochSampler.from_state_dict(state, num_examples=101, batch_size=16)
    with pytest.raises(ValueError, match="unsupported sampler state format"):
        EpochSampler.from_state_dict({**state, "format": "v0"}, num_examples=101, batch_size=32)
    with pytest.raises(ValueError, match="must have keys"):
        EpochSampler.from_state_dict({"seed": 3}, num_examples=101, batch_size=32)


# -------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "n, b, seed, exc",
    [
        (0, 1, 0, ValueError),
        (-3, 1, 0, ValueError),
        (2.0, 1, 0, ValueError),
        (True, 1, 0, ValueError),
        (5, 0, 0, ValueError),
        (5, 1.5, 0, ValueError),
        (5, True, 0, ValueError),
        (5, 2, -1, ValueError),
        (5, 2, 2**32, ValueError),
        (5, 2, 1.0, TypeError),
        (5, 2, True, TypeError),
    ],
)
def test_constructor_validation(n, b, seed, exc):
    with pytest.raises(exc):
        EpochSampler(n, b, seed=seed)


def test_epoch_and_start_batch_validation():
    s = EpochSampler(10, 4, seed=0)
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError, match="epoch"):
            s.permutation(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start_batch"):
        list(s.batches(0, start_batch=4))
    with pytest.raises(ValueError, match="start_batch"):
        list(s.batches(0, start_batch=-1))
    assert list(s.batches(0, start_batch=3)) == []                           # == num_batches: empty tail


# ---------------------------------------------------------- loss weighting
def test_example_weighted_epoch_loss_versus_mean_of_batch_means():
    losses = torch.tensor([1.0, 1, 1, 1, 2, 2, 2, 2, 5, 5], dtype=torch.float64)   # N = 10, B = 4 -> sizes 4, 4, 2
    s = EpochSampler(10, 4, seed=0)
    acc, means = NLLAccumulator(), []
    for idx in s.batches(0):
        m = float(losses[idx].mean())
        means.append(m)
        acc.update_from_sum(m * len(idx), len(idx))
    assert acc.num_tokens == 10
    assert acc.mean_nll == pytest.approx(float(losses.mean()), abs=1e-12)    # 2.2, independent of the ordering
    # the mean of batch means depends on the ordering and over-weights the partial batch
    worst = (1.0 + 2.0 + 5.0) / 3                                            # batches [1,1,1,1] [2,2,2,2] [5,5]
    assert worst == pytest.approx(2.6667, abs=1e-4) and worst != pytest.approx(2.2)


def test_partial_batch_gradient_reductions(make_tiny_net):
    torch.manual_seed(0)
    model = make_tiny_net()
    g = torch.Generator().manual_seed(1)
    X, Y = torch.randint(0, 5, (10,), generator=g), torch.randint(0, 5, (10,), generator=g)
    s = EpochSampler(10, 4, seed=5)

    def grads(loss):
        model.zero_grad(set_to_none=True)
        loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

    full = grads(token_nll(model(X), Y, reduction="mean"))                   # ∇L_full, parameters fixed
    sum_over_B = sum(grads(token_nll(model(X[i]), Y[i], reduction="sum") / s.batch_size) for i in s.batches(0))
    assert torch.allclose(sum_over_B, full * (10 / 4), atol=1e-6)            # equal per-example weight 1/B
    mean_red = sum(grads(token_nll(model(X[i]), Y[i], reduction="mean")) for i in s.batches(0))
    assert not torch.allclose(mean_red, full * (10 / 4), atol=1e-4)          # partial batch over-weighted


# --------------------------------------------- end-to-end epoch-boundary resume
def _train_epochs(model, opt, sampler, X, Y, epochs):
    for epoch in epochs:
        for idx in sampler.batches(epoch):
            opt.zero_grad(set_to_none=True)
            token_nll(model(X[idx]), Y[idx], reduction="mean").backward()
            opt.step()


def test_epoch_boundary_resume_is_bit_exact(make_tiny_net, tmp_path):
    g = torch.Generator().manual_seed(2)
    X, Y = torch.randint(0, 5, (37,), generator=g), torch.randint(0, 5, (37,), generator=g)

    torch.manual_seed(0)
    straight = make_tiny_net()
    opt = torch.optim.SGD(straight.parameters(), lr=0.1, momentum=0.9)       # momentum: optimizer has state
    _train_epochs(straight, opt, EpochSampler(37, 8, seed=9), X, Y, epochs=[0, 1])

    torch.manual_seed(0)
    first = make_tiny_net()
    opt1 = torch.optim.SGD(first.parameters(), lr=0.1, momentum=0.9)
    sampler = EpochSampler(37, 8, seed=9)
    _train_epochs(first, opt1, sampler, X, Y, epochs=[0])
    path = save_checkpoint(tmp_path / "checkpoint_last.pt", kind="resume", model=first, optimizer=opt1,
                           step=sampler.num_batches, epoch=1, rng_state=capture_rng_state(),
                           extra={"sampler": sampler.state_dict()})

    torch.manual_seed(123)                                                   # a different fresh model
    resumed = make_tiny_net()
    opt2 = torch.optim.SGD(resumed.parameters(), lr=0.1, momentum=0.9)
    ckpt = load_checkpoint(path, expected_kind="resume")
    progress = restore_training_state(ckpt, model=resumed, optimizer=opt2)
    sampler2 = EpochSampler.from_state_dict(ckpt["extra"]["sampler"], num_examples=37, batch_size=8)
    _train_epochs(resumed, opt2, sampler2, X, Y, epochs=range(progress.epoch, 2))   # epoch = epochs completed

    for (name, a), (_, b) in zip(straight.named_parameters(), resumed.named_parameters()):
        assert torch.equal(a, b), name

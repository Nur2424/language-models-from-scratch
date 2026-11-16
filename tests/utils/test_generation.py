import math

import pytest
import torch

from utils.generation import (
    SamplingError,
    next_token_probs,
    sample_next_token,
    top_k_filter,
    top_p_filter,
    validate_sampling_params,
)


def test_greedy_and_shapes():
    logits = torch.tensor([0.1, 3.0, 0.2])
    assert sample_next_token(logits, temperature=0).item() == 1
    batch = torch.tensor([[0.0, 1.0], [5.0, 0.0]])
    out = sample_next_token(batch, temperature=0)
    assert out.tolist() == [1, 0] and out.dtype == torch.long
    assert sample_next_token(logits, generator=torch.Generator().manual_seed(0)).shape == ()


def test_temperature_changes_distribution():
    logits = torch.tensor([1.0, 2.0, 3.0])
    hot = next_token_probs(logits, temperature=10.0)
    cold = next_token_probs(logits, temperature=0.1)
    ref = torch.softmax(logits, -1)
    assert hot.max() < ref.max() < cold.max()
    assert torch.allclose(next_token_probs(logits), ref)


def test_top_k():
    logits = torch.tensor([1.0, 4.0, 3.0, 2.0])
    out = top_k_filter(logits, 2)
    assert torch.isinf(out[[0, 3]]).all() and torch.isfinite(out[[1, 2]]).all()
    assert torch.equal(top_k_filter(logits, 10), logits)
    ties = top_k_filter(torch.tensor([1.0, 2.0, 2.0, 0.0]), 1)
    assert torch.isfinite(ties).sum() == 2   # ties with the k-th value are kept


def test_top_p_keeps_smallest_nucleus():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    keep = torch.isfinite(top_p_filter(logits, 0.8))
    assert keep.tolist() == [True, True, False, False]      # 0.5 + 0.3 >= 0.8
    keep = torch.isfinite(top_p_filter(logits, 0.81))
    assert keep.tolist() == [True, True, True, False]
    keep = torch.isfinite(top_p_filter(logits, 0.1))
    assert keep.tolist() == [True, False, False, False]     # most likely always kept
    shuffled = logits[[2, 0, 3, 1]]
    assert torch.isfinite(top_p_filter(shuffled, 0.8)).tolist() == [False, True, False, True]


def test_filtered_samples_only_from_allowed_tokens():
    logits = torch.tensor([3.0, 2.9, -1.0, -2.0, -3.0])
    g = torch.Generator().manual_seed(0)
    draws = {sample_next_token(logits, top_k=2, generator=g).item() for _ in range(200)}
    assert draws == {0, 1}


def test_generator_reproducibility_and_empirical_distribution():
    logits = torch.tensor([0.0, math.log(3.0)])  # p = [0.25, 0.75]
    a = [sample_next_token(logits, generator=torch.Generator().manual_seed(5)).item() for _ in range(3)]
    b = [sample_next_token(logits, generator=torch.Generator().manual_seed(5)).item() for _ in range(3)]
    assert a == b
    g = torch.Generator().manual_seed(1)
    batch = logits.expand(20000, 2)
    freq = sample_next_token(batch, generator=g).float().mean().item()
    assert abs(freq - 0.75) < 0.02


@pytest.mark.parametrize(
    "kw",
    [dict(temperature=-1), dict(temperature=float("nan")), dict(temperature=float("inf")), dict(top_k=0),
     dict(top_k=1.5), dict(top_k=True), dict(top_p=0), dict(top_p=1.01), dict(temperature="1")],
)
def test_parameter_validation(kw):
    with pytest.raises(SamplingError):
        validate_sampling_params(**kw)


@pytest.mark.parametrize(
    "logits",
    [torch.tensor([0.0, float("nan")]), torch.tensor([0.0, float("inf")]), torch.full((3,), float("-inf")),
     torch.tensor([1, 2]), torch.zeros(2, 2, 2), torch.zeros(0)],
)
def test_invalid_logits(logits):
    with pytest.raises(SamplingError):
        sample_next_token(logits)


def test_partially_masked_logits_are_fine():
    logits = torch.tensor([float("-inf"), 0.0, float("-inf")])
    assert sample_next_token(logits, generator=torch.Generator().manual_seed(0)).item() == 1

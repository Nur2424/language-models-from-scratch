import math
import statistics as pystats

import pytest
import torch

from utils.statistics import Moments, RunningStats, merge_moments, summarize, tensor_moments, tensor_summary


def test_summarize_matches_reference():
    xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    s = summarize(xs)
    assert s.mean == pytest.approx(pystats.mean(xs))
    assert s.std == pytest.approx(pystats.stdev(xs))     # sample std
    assert s.median == 4.5 and s.min == 2.0 and s.max == 9.0
    assert s.sem == pytest.approx(pystats.stdev(xs) / math.sqrt(8))
    assert summarize([3.0]).std is None


def test_summarize_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError, match="non-finite"):
        summarize([1.0, float("nan")])


def test_running_stats_merge_equals_single_stream():
    a, b = RunningStats(), RunningStats()
    a.update_many([1.0, 2.0, 3.0])
    b.update_many([10.0, 20.0])
    merged = a.merge(b)
    ref = RunningStats()
    ref.update_many([1.0, 2.0, 3.0, 10.0, 20.0])
    assert merged.mean == pytest.approx(ref.mean) and merged.variance == pytest.approx(ref.variance)


def test_tensor_summary_values():
    t = torch.tensor([[0.0, 1.0], [2.0, -3.0]])
    s = tensor_summary(t)
    assert s["numel"] == 4 and s["mean"] == pytest.approx(0.0)
    assert s["std"] == pytest.approx(t.std(unbiased=False).item())   # population std
    assert s["min"] == -3.0 and s["max"] == 2.0
    assert s["norm"] == pytest.approx(t.norm().item()) and s["zero_fraction"] == 0.25


def test_tensor_summary_nonfinite_and_saturation():
    t = torch.tensor([0.99, -0.5, float("nan"), float("inf"), 0.98])
    s = tensor_summary(t, saturation=lambda x: x.abs() > 0.97)
    assert s["nan_count"] == 1 and s["inf_count"] == 1
    assert s["mean"] == pytest.approx((0.99 - 0.5 + 0.98) / 3)       # finite elements only
    assert s["saturation_fraction"] == pytest.approx(2 / 5)


def test_moments_merge_exact_and_stable():
    g = torch.Generator().manual_seed(0)
    parts = [1.0 + 1e-3 * torch.randn(n, generator=g) for n in (10, 1000, 37)]  # |mean| >> std
    merged = merge_moments(tensor_moments(p) for p in parts).summary()
    full = torch.cat(parts).double()
    assert merged["numel"] == full.numel()
    assert merged["mean"] == pytest.approx(full.mean().item(), rel=1e-6)
    assert merged["std"] == pytest.approx(full.std(unbiased=False).item(), rel=1e-3)


def test_tensor_summary_detaches_and_handles_empty():
    w = torch.randn(4, requires_grad=True)
    s = tensor_summary(w * 2)
    assert isinstance(s["mean"], float)
    empty = tensor_summary(torch.empty(0))
    assert empty["numel"] == 0 and empty["mean"] is None
    assert Moments.empty().summary()["zero_fraction"] is None

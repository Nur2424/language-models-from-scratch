import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from utils.metrics import (
    NLLAccumulator,
    bits_per_token,
    count_parameters,
    count_valid_targets,
    parameter_breakdown,
    perplexity,
    token_nll,
)
from utils.vocabulary import IGNORE_INDEX


def _batch(seed, b, t, v, n_ignore):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, t, v, generator=g)
    targets = torch.randint(0, v, (b, t), generator=g)
    targets.view(-1)[:n_ignore] = IGNORE_INDEX
    return logits, targets


def test_reductions_are_explicit_and_consistent():
    logits, targets = _batch(0, 2, 5, 7, n_ignore=3)
    valid = targets != IGNORE_INDEX
    per = token_nll(logits, targets, reduction="none")
    assert per.shape == targets.shape and torch.all(per[~valid] == 0)
    s = token_nll(logits, targets, reduction="sum")
    m = token_nll(logits, targets, reduction="mean")
    assert torch.isclose(s, per.sum()) and torch.isclose(m, s / valid.sum())
    assert torch.isclose(m, F.cross_entropy(logits.view(-1, 7), targets.view(-1), ignore_index=IGNORE_INDEX))
    with pytest.raises(ValueError):
        token_nll(logits, targets, reduction="avg")  # type: ignore[arg-type]


def test_mean_with_no_valid_targets_raises():
    logits = torch.randn(3, 4)
    with pytest.raises(ValueError, match="no valid targets"):
        token_nll(logits, torch.full((3,), IGNORE_INDEX), reduction="mean")


def test_target_range_and_shape_checks():
    with pytest.raises(ValueError):
        token_nll(torch.randn(3, 4), torch.tensor([0, 1, 4]), reduction="sum")
    with pytest.raises(ValueError):
        token_nll(torch.randn(3, 4), torch.tensor([0, 1]), reduction="sum")


def test_accumulator_weights_by_tokens_not_batches():
    # The numeric example from the module docstring: 10 tokens at 2.0, 2 tokens at 3.0.
    acc = NLLAccumulator()
    acc.update_from_sum(10 * 2.0, 10)
    acc.update_from_sum(2 * 3.0, 2)
    assert acc.mean_nll == pytest.approx(26 / 12)
    assert acc.mean_nll != pytest.approx(2.5)


def test_accumulator_matches_single_pass_with_unequal_valid_counts():
    batches = [_batch(1, 4, 6, 5, n_ignore=0), _batch(2, 4, 6, 5, n_ignore=20), _batch(3, 1, 6, 5, n_ignore=5)]
    acc = NLLAccumulator()
    for lg, tg in batches:
        acc.update(lg, tg)
    all_logits = torch.cat([lg.reshape(-1, 5) for lg, _ in batches])
    all_targets = torch.cat([tg.reshape(-1) for _, tg in batches])
    exact = F.cross_entropy(all_logits, all_targets, ignore_index=IGNORE_INDEX).item()
    assert acc.mean_nll == pytest.approx(exact, rel=1e-6)
    assert acc.num_tokens == count_valid_targets(all_targets)
    naive = sum(F.cross_entropy(lg.reshape(-1, 5), tg.reshape(-1), ignore_index=IGNORE_INDEX).item()
                for lg, tg in batches) / len(batches)
    assert abs(naive - exact) > 1e-3   # the averaging-of-means bug would be detectable


def test_perplexity_and_bits():
    uniform = torch.zeros(4, 27)
    acc = NLLAccumulator()
    acc.update(uniform, torch.tensor([0, 1, 2, 3]))
    assert acc.mean_nll == pytest.approx(math.log(27))
    assert acc.perplexity == pytest.approx(27.0)
    assert bits_per_token(math.log(2)) == pytest.approx(1.0)
    assert perplexity(0.0) == 1.0
    with pytest.raises(ValueError):
        NLLAccumulator().mean_nll


def test_accumulator_does_not_keep_graph():
    w = torch.randn(3, 4, requires_grad=True)
    acc = NLLAccumulator()
    acc.update(w * 2, torch.tensor([0, 1, 2]))
    assert isinstance(acc.total_nll, float)


def test_parameter_counts_with_tying_and_freezing(tiny_net):
    model = tiny_net
    model.head.weight = model.emb.weight  # tie (5x4)
    model.unused.requires_grad_(False)
    counts = count_parameters(model)
    expected_total = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    assert counts.total == expected_total
    assert counts.frozen == 2 * 2 + 2
    breakdown = parameter_breakdown(model)
    assert sum(c.total for c in breakdown.values()) == counts.total
    assert set(breakdown) == {"emb", "body", "head", "unused"}
    assert "head" in breakdown and breakdown["head"].total == 5   # only the untied bias
    by_cat = parameter_breakdown(model, group_fn=lambda n: "bias" if n.endswith("bias") else "weight")
    assert sum(c.total for c in by_cat.values()) == counts.total


def test_root_level_parameters_group():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(3))
    assert parameter_breakdown(M())["(root)"].total == 3

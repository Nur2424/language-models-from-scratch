"""Tests for the name-clustered T-RQ1b analysis (position_analysis.py)."""
import math

import torch

from models.recurrent.rnn_diagnostics import _clustered
from training.transformer_study.position_analysis import groups, index_groups, paired_by_group


def test_groups_partition_positions():
    pos = torch.tensor([0, 3, 7, 8, 9, 12])
    g = groups(pos)
    assert (g["i<=7"].long() + g["i=8"].long() + g["i>=9"].long()).tolist() == [1] * 6
    assert g["all"].all()


def test_paired_by_group_matches_hand_computation():
    gen = torch.Generator().manual_seed(0)
    names = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    pos = torch.tensor([0, 1, 2, 0, 1, 0, 1, 7, 8, 9, 10])
    a = torch.rand(3, 11, generator=gen, dtype=torch.float64)
    b = torch.rand(3, 11, generator=gen, dtype=torch.float64)
    res = paired_by_group(a, b, names, pos)
    delta = (a - b).mean(0)
    sel = pos >= 9
    assert math.isclose(res["i>=9"]["mean"], float(delta[sel].mean()), rel_tol=1e-12)
    by = {}
    for n, d in zip(names.tolist(), delta.tolist()):
        by.setdefault(n, []).append(d)
    assert math.isclose(res["all"]["se_name_clustered"], _clustered(by)["se"], rel_tol=1e-12)
    per_seed = [float((a[s] - b[s]).mean()) for s in range(3)]
    m = sum(per_seed) / 3
    se = math.sqrt(sum((v - m) ** 2 for v in per_seed) / 2 / 3)
    assert math.isclose(res["all"]["se_seed"], se, rel_tol=1e-12)
    assert res["all"]["seeds_first_lower"] == sum(v < 0 for v in per_seed)
    assert res["i=8"]["targets"] == 1 and res["i<=7"]["names"] == 3


def test_index_groups_cover_each_position_and_the_pooled_tail():
    pos = torch.tensor([0, 0, 3, 8, 9, 9, 12])
    g = index_groups(pos)
    assert list(g) == ["0", "3", "8", "9", "12", "9+"]
    assert g["9+"].tolist() == [False, False, False, False, True, True, True]
    assert sum(int(g[k].sum()) for k in g if k != "9+") == len(pos)


def test_default_selection_is_unchanged():
    gen = torch.Generator().manual_seed(1)
    names = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
    pos = torch.tensor([0, 1, 0, 1, 0, 8, 9, 0, 7, 10])
    a = torch.rand(4, 10, generator=gen, dtype=torch.float64)
    b = torch.rand(4, 10, generator=gen, dtype=torch.float64)
    assert paired_by_group(a, b, names, pos) == paired_by_group(a, b, names, pos, groups(pos))


def test_single_positions_add_up_to_the_pooled_group():
    gen = torch.Generator().manual_seed(2)
    names = torch.arange(12) // 3
    pos = torch.tensor([0, 1, 9, 0, 9, 10, 0, 11, 11, 0, 1, 2])
    a = torch.rand(3, 12, generator=gen, dtype=torch.float64)
    b = torch.rand(3, 12, generator=gen, dtype=torch.float64)
    res = paired_by_group(a, b, names, pos, index_groups(pos))
    tail = [res[k] for k in ("9", "10", "11")]
    pooled = sum(r["mean"] * r["targets"] for r in tail) / sum(r["targets"] for r in tail)
    assert math.isclose(res["9+"]["mean"], pooled, rel_tol=1e-12)
    assert res["9+"]["targets"] == 5

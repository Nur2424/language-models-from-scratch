"""Tests for the name-clustered T-RQ1b analysis (position_analysis.py)."""
import math

import torch

from models.recurrent.rnn_diagnostics import _clustered
from training.transformer_study.position_analysis import groups, paired_by_group


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

"""Tests for training/wavenet_study/context_buckets.py (F2).

Expected values come from hand-built names and from string-level references that never use the
module's tensor code: a context is ``("." * 8 + name[:t-1])[-8:]`` for the target at position t.
All data are synthetic; no split file is read.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from training.wavenet_study import context_buckets as cb
from utils.data import ContextWindows, make_context_windows
from utils.vocabulary import Vocabulary

ROOT = Path(__file__).resolve().parents[3]
VOCAB = Vocabulary.from_names(["abcdefghijklmnopqrstuvwxyz"])


def windows(names):
    return make_context_windows(list(names), VOCAB, context_length=8)


def string_contexts(names):
    """(context string, target char, t, L) per target; '.' is BOS/EOS."""
    out = []
    for name in names:
        for t in range(1, len(name) + 2):
            ctx = ("." * 8 + name[: t - 1])[-8:]
            tgt = name[t - 1] if t <= len(name) else "."
            out.append((ctx, tgt, t, len(name)))
    return out


def random_names(seed, n, alphabet="abc", lengths=(1, 6)):
    rng = random.Random(seed)
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(*lengths))) for _ in range(n)]


TRAIN = random_names(1, 3000)
DEV = random_names(2, 1500, lengths=(1, 8))


@pytest.fixture(scope="module")
def built():
    train, dev = windows(TRAIN), windows(DEV)
    return train, dev, cb.define_buckets(train, dev), cb.count_train_contexts(train)


# ------------------------------------------------------------------ exact counts
def test_exact_context_counts_hand_built():
    c = cb.count_train_contexts(windows(["ab", "ab", "b"]))
    a, b = VOCAB.stoi["a"], VOCAB.stoi["b"]
    assert c.count([0] * 8) == 3                          # all-BOS context, once per name
    assert c.count([0] * 7 + [a]) == 2
    assert c.count([0] * 6 + [a, b]) == 2
    assert c.count([0] * 7 + [b]) == 1
    assert c.count([0] * 7 + [VOCAB.stoi["z"]]) == 0
    assert sum(c.counts.values()) == c.num_windows == 3 + 3 + 2


def test_counts_match_string_reference_all_eight_positions():
    names = random_names(5, 400, alphabet="ab", lengths=(1, 12))
    c = cb.count_train_contexts(windows(names))
    ref = Counter(ctx for ctx, *_ in string_contexts(names))
    enc = {s: tuple(0 if ch == "." else VOCAB.stoi[ch] for ch in s) for s in ref}
    assert {enc[s]: n for s, n in ref.items()} == dict(c.counts)
    # contexts differing only in the oldest position are different contexts
    long1, long2 = "a" + "b" * 8, "b" * 9
    cc = cb.count_train_contexts(windows([long1, long2]))
    assert cc.count([VOCAB.stoi["b"]] * 8) == 3          # long1 t=10; long2 t=9 and t=10
    assert cc.count([VOCAB.stoi["a"]] + [VOCAB.stoi["b"]] * 7) == 1   # long1 t=9 only
    ref2 = Counter(ctx for ctx, *_ in string_contexts([long1, long2]))
    assert cc.count([VOCAB.stoi["b"]] * 8) == ref2["b" * 8]


def test_bos_containing_contexts_are_distinct_by_length():
    c = cb.count_train_contexts(windows(["a", "aa", "aaa"]))
    a = VOCAB.stoi["a"]
    for r, expected in [(0, 3), (1, 3), (2, 2), (3, 1)]:   # r leading real 'a's after 8−r BOS
        assert c.count([0] * (8 - r) + [a] * r) == expected


# ------------------------------------------------------------------ positions and real-context length
def test_positions_and_real_length_hand_built():
    names = ["a", "abc", "abcdefghij"]
    w = windows(names)
    f = cb.target_features(w, cb.count_train_contexts(w))
    assert cb.target_positions(w) == [1, 2] + [1, 2, 3, 4] + list(range(1, 12))
    assert f["real_context_length"] == [0, 1] + [0, 1, 2, 3] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]
    assert f["target_position"] == (["first", "eos"] + ["first", "interior", "interior", "eos"]
                                    + ["first"] + ["interior"] * 9 + ["eos"])


def test_positions_and_real_length_against_string_reference():
    names = random_names(9, 300, alphabet="abcd", lengths=(1, 14))
    w = windows(names)
    f = cb.target_features(w, cb.count_train_contexts(w))
    ref = string_contexts(names)
    assert f["real_context_length"] == [len(ctx.lstrip(".")) for ctx, *_ in ref]
    assert f["real_context_length"] == [min(t - 1, 8) for _, _, t, _ in ref]
    assert f["target_position"] == ["first" if t == 1 else "eos" if t == L + 1 else "interior"
                                    for _, _, t, L in ref]


def test_real_length_boundary_at_eight():
    w = windows(["abcdefgh", "abcdefghi"])          # L = 8 and 9
    f = cb.target_features(w, cb.count_train_contexts(w))
    assert f["real_context_length"][7:9] == [7, 8]   # t = 8 → 7 real; t = 9 (EOS of L=8) → 8
    assert f["real_context_length"][-2:] == [8, 8]


def test_inconsistent_windows_rejected():
    w = windows(["abc"])
    bad = ContextWindows(contexts=w.contexts.clone(), targets=w.targets.clone(), name_index=w.name_index.clone())
    bad.contexts[2, 3] = VOCAB.stoi["a"]             # a real token inside the BOS run
    with pytest.raises(cb.BucketError):
        cb.target_features(bad, cb.count_train_contexts(w))
    with pytest.raises(cb.BucketError):
        cb.count_train_contexts(make_context_windows(["abc"], VOCAB, context_length=3))
    with pytest.raises(cb.BucketError):
        cb.count_train_contexts("not windows")


# ------------------------------------------------------------------ terciles and ties
def test_tercile_thresholds_nearest_rank():
    assert cb.tercile_thresholds([1, 2, 3, 4, 5, 6]) == (2, 4)
    assert cb.tercile_thresholds([1, 2, 3, 4, 5, 6, 7]) == (3, 5)       # ⌈7/3⌉=3, ⌈14/3⌉=5
    assert cb.tercile_thresholds([9]) == (9, 9)
    assert cb.tercile_thresholds([5, 1, 3]) == (1, 3)
    assert cb.tercile_thresholds([]) is None
    with pytest.raises(cb.BucketError):
        cb.tercile_thresholds([0, 1])
    for t1, t2 in [cb.tercile_thresholds(random.Random(s).choices(range(1, 40), k=s + 1)) for s in range(50)]:
        assert isinstance(t1, int) and isinstance(t2, int) and t1 <= t2


def _f_initial(t1, t2):
    return [{"label": "low", "lo": 1, "hi": t1}, {"label": "mid", "lo": t1 + 1, "hi": t2},
            {"label": "high", "lo": t2 + 1, "hi": None}]


def _summary(bins):
    return [(b["label"], b["lo"], b["hi"], b["dev_support"]) for b in bins]


def _moves(steps):
    return [(s["category"], s["into"], s["direction"]) for s in steps]


def test_tied_values_share_a_bin_and_empty_bins_collapse():
    values = [3] * 900 + [7] * 600
    assert cb.tercile_thresholds(values) == (3, 7)   # ranks ⌈1500/3⌉ = 500 → 3, ⌈3000/3⌉ = 1000 → 7
    values = [3] * 1200 + [7] * 600
    t1, t2 = cb.tercile_thresholds(values)
    assert (t1, t2) == (3, 3)                        # ranks 600 and 1200 both have count 3
    bins, steps = cb._finalise_ordinal(_f_initial(t1, t2), values, lowest_merges_upward=True)
    # mid = [4, 3] is empty: it is absorbed by its lower-frequency neighbour, not dropped
    assert _summary(bins) == [("low+mid", 1, 3, 1200), ("high", 4, None, 600)]
    assert steps == [{"step": "empty", "category": "mid", "into": "low", "direction": "downward",
                      "dev_support": [0, 1200]}]


def test_all_counts_equal_collapses_to_one_seen_category():
    values = [4] * 2000
    bins, steps = cb._finalise_ordinal(_f_initial(*cb.tercile_thresholds(values)), values,
                                       lowest_merges_upward=True)
    assert _summary(bins) == [("low+mid+high", 1, None, 2000)]
    assert _moves(steps) == [("high", "mid", "downward"), ("mid+high", "low", "downward")]
    assert [s["step"] for s in steps] == ["empty", "empty"]


# ------------------------------------------------------------------ minimum support merging
def _bins(supports, *, upward):
    initial = [{"label": str(i), "lo": i, "hi": i} for i in range(len(supports))]
    values = [i for i, s in enumerate(supports) for _ in range(s)]
    return cb._finalise_ordinal(initial, values, lowest_merges_upward=upward)


def test_frequency_merges_toward_lower_frequency_neighbour():
    # the under-supported middle bin goes to its lower neighbour even though the upper one is smaller
    bins, steps = _bins([900, 100, 600], upward=True)
    assert _summary(bins) == [("0+1", 0, 1, 1000), ("2", 2, 2, 600)]
    assert steps == [{"step": "under_supported", "category": "1", "into": "0", "direction": "downward",
                      "dev_support": [100, 900]}]
    # the scan runs from the highest bin down, re-examining each merged bin
    bins, steps = _bins([600, 100, 50, 700, 30], upward=True)
    assert _moves(steps) == [("4", "3", "downward"), ("2", "1", "downward"), ("1+2", "0", "downward")]
    assert _summary(bins) == [("0+1+2", 0, 2, 750), ("3+4", 3, 4, 730)]


def test_frequency_low_merges_upward():
    bins, steps = _bins([300, 600, 700], upward=True)
    assert _moves(steps) == [("0", "1", "upward")]
    assert _summary(bins) == [("0+1", 0, 1, 900), ("2", 2, 2, 700)]
    bins, steps = _bins([300, 100, 700], upward=True)  # 100 → low (400), then low+… upward
    assert _moves(steps) == [("1", "0", "downward"), ("0+1", "2", "upward")]
    assert _summary(bins) == [("0+1+2", 0, 2, 1100)]
    bins, steps = _bins([100, 100], upward=True)       # one category left below 500: kept, flagged later
    assert _moves(steps) == [("1", "0", "downward")] and _summary(bins) == [("0+1", 0, 1, 200)]


def test_min_support_boundary():
    assert cb.MIN_SUPPORT == 500
    for upward in (True, False):
        bins, steps = _bins([500, 500], upward=upward)
        assert steps == [] and len(bins) == 2         # exactly 500 is sufficient
        bins, steps = _bins([600, 499], upward=upward)
        assert _moves(steps) == [("1", "0", "downward")]
    assert _moves(_bins([499, 600], upward=True)[1]) == [("0", "1", "upward")]


def test_real_length_scans_from_eight_downwards():
    bins, steps = _bins([900, 800, 700, 600, 550, 300, 200, 100, 50], upward=False)
    assert _moves(steps) == [("8", "7", "downward"), ("7+8", "6", "downward"), ("6+7+8", "5", "downward")]
    assert _summary(bins) == [(str(r), r, r, s) for r, s in enumerate([900, 800, 700, 600, 550])] + [
        ("5+6+7+8", 5, 8, 650)]


def test_real_length_never_merges_upward():
    bins, steps = _bins([300, 600], upward=False)
    assert steps == [] and _summary(bins) == [("0", 0, 0, 300), ("1", 1, 1, 600)]
    bins, steps = _bins([600, 100, 900], upward=False)
    assert _moves(steps) == [("1", "0", "downward")]
    bins, steps = _bins([600, 0, 700], upward=False)  # empty: absorbed by the next lower category
    assert steps[0]["step"] == "empty" and _summary(bins) == [("0+1", 0, 1, 600), ("2", 2, 2, 700)]


def test_real_length_end_to_end_lowest_kept_and_flagged():
    # 100 dev names of length 20: lengths 0–7 have 100 targets each, length 8 has 100·13 = 1300.
    # Scan 8 → 0: 7 → 6 → 5 → 4 → 3 reaches 500; 2 → 1 → 0 leaves 300, kept and flagged (no upward merge)
    dev = windows(random_names(8, 100, lengths=(20, 20)))
    d = cb.define_buckets(windows(TRAIN), dev)
    r = d["real_context_length"]
    assert [(c["id"], c["lo"], c["hi"], c["dev_support"], c["insufficient_dev_support"]) for c in r["categories"]] == [
        ("r:0+1+2", 0, 2, 300, True), ("r:3+4+5+6+7", 3, 7, 500, False), ("r:8", 8, 8, 1300, False)]
    assert _moves(r["steps"]) == [("7", "6", "downward"), ("6+7", "5", "downward"), ("5+6+7", "4", "downward"),
                                  ("4+5+6+7", "3", "downward"), ("2", "1", "downward"), ("1+2", "0", "downward")]


def test_define_uses_upward_rule_for_frequency_only(monkeypatch):
    calls = []
    real = cb._finalise_ordinal

    def spy(initial, values, *, lowest_merges_upward):
        calls.append((initial[0]["label"], lowest_merges_upward))
        return real(initial, values, lowest_merges_upward=lowest_merges_upward)

    monkeypatch.setattr(cb, "_finalise_ordinal", spy)
    cb.define_buckets(windows(TRAIN[:500]), windows(DEV[:300]))
    assert calls == [("low", True), ("0", False)]


def test_end_to_end_definitions(built):
    train, dev, d, counts = built
    cb.verify_definitions(d)
    f_ord = [c for c in d["frequency"]["categories"] if c["kind"] == "ordinal"]
    assert all(c["dev_support"] >= cb.MIN_SUPPORT for c in f_ord) or len(f_ord) == 1
    r_cats = d["real_context_length"]["categories"]
    assert all(c["dev_support"] >= cb.MIN_SUPPORT for c in r_cats[1:])
    assert all(s["direction"] == "downward" for s in d["real_context_length"]["steps"])
    f_steps = d["frequency"]["steps"]
    assert all(s["direction"] == "downward" for s in f_steps[:-1])
    for strat in cb.STRATIFICATIONS:
        assert sum(c["dev_support"] for c in d[strat]["categories"]) == dev.num_examples
    assert d["target_position"]["steps"] == [] and [c["id"] for c in d["target_position"]["categories"]] == list(
        cb.POSITIONS)
    f = cb.target_features(dev, counts)
    assert d["frequency"]["dev_unseen"] == sum(v == 0 for v in f["frequency"])
    t1, t2 = d["frequency"]["tercile_thresholds"]
    assert (t1, t2) == cb.tercile_thresholds([v for v in f["frequency"] if v > 0])
    # the recorded steps replay to the stored categories
    for strat, upward in (("frequency", True), ("real_context_length", False)):
        values = [v for v in f[strat] if v > 0] if strat == "frequency" else f[strat]
        initial = (_f_initial(t1, t2) if strat == "frequency"
                   else [{"label": str(r), "lo": r, "hi": r} for r in range(9)])
        bins, steps = cb._finalise_ordinal(initial, values, lowest_merges_upward=upward)
        assert steps == d[strat]["steps"]
        stored = [c for c in d[strat]["categories"] if c["kind"] == "ordinal"]
        assert [(b["lo"], b["hi"], b["dev_support"]) for b in bins] == [(c["lo"], c["hi"], c["dev_support"])
                                                                         for c in stored]


def test_position_categories_never_merged_and_flagged():
    d = cb.define_buckets(windows(TRAIN), windows(["a"] * 300 + ["abc"] * 100))
    pos = {c["id"]: c for c in d["target_position"]["categories"]}
    assert {k: v["dev_support"] for k, v in pos.items()} == {"first": 400, "interior": 200, "eos": 400}
    assert all(v["insufficient_dev_support"] for v in pos.values()) and d["target_position"]["steps"] == []


# ------------------------------------------------------------------ unseen contexts
def test_unseen_kept_separate_and_flagged_not_merged():
    train = windows(random_names(3, 1500, lengths=(1, 4)))
    dev = windows(random_names(4, 700, lengths=(1, 4)) + ["abcabca", "cccbbba"])   # a few unseen long contexts
    d = cb.define_buckets(train, dev)
    unseen = d["frequency"]["categories"][0]
    assert unseen["id"] == "unseen" and unseen["kind"] == "nominal"
    assert 0 < unseen["dev_support"] < cb.MIN_SUPPORT and unseen["insufficient_dev_support"] is True
    assert all("unseen" not in (s["category"], s["into"]) for s in d["frequency"]["steps"])
    a = cb.assign(d, dev, cb.count_train_contexts(train))
    feats = cb.target_features(dev, cb.count_train_contexts(train))
    assert [lab == "unseen" for lab in a["labels"]["frequency"]] == [v == 0 for v in feats["frequency"]]


def test_no_unseen_dev_contexts_still_has_unseen_category():
    names = random_names(6, 1200, lengths=(1, 3))
    d = cb.define_buckets(windows(names), windows(names[:400]))
    assert d["frequency"]["categories"][0]["dev_support"] == 0
    assert d["frequency"]["categories"][0]["insufficient_dev_support"] is True


def test_all_bos_context_is_always_seen():
    # the first target of every name has the all-BOS context, counted once per training name, so a
    # dev set can never be entirely unseen; here every other dev context is unseen
    d = cb.define_buckets(windows(["aaaa"] * 10), windows(["bbbbbb"] * 100))
    assert d["frequency"]["tercile_thresholds"] == [10, 10]
    assert d["frequency"]["dev_seen"] == 100 and d["frequency"]["dev_unseen"] == 600
    assert [c["id"] for c in d["frequency"]["categories"]] == ["unseen", "seen:low+mid+high"]
    cb.verify_definitions(d)


# ------------------------------------------------------------------ serialisation and hashing
def test_round_trip_and_tampering(built):
    _, _, d, _ = built
    text = cb.to_json(d)
    assert cb.from_json(text) == d
    for path, value in [(("frequency", "tercile_thresholds", 0), 999), (("min_support",), 400),
                        (("real_context_length", "categories", 0, "hi"), 5),
                        (("sources", "train_fingerprint"), "0" * 64),
                        (("target_position", "categories", 0, "dev_support"), 1)]:
        t = json.loads(text)
        node = t
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        with pytest.raises(cb.BucketError, match="hash"):
            cb.from_json(json.dumps(t))
    t = json.loads(text)
    t["definition_sha256"] = "f" * 64
    with pytest.raises(cb.BucketError, match="hash"):
        cb.verify_definitions(t)
    del t["definition_sha256"]
    with pytest.raises(cb.BucketError):
        cb.verify_definitions(t)


def test_fingerprint_covers_every_tensor():
    w = windows(["ab", "c"])
    fp = cb.windows_fingerprint(w)
    assert fp == cb.windows_fingerprint(windows(["ab", "c"]))
    for field in ("contexts", "targets", "name_index"):
        t = {f: getattr(w, f).clone() for f in ("contexts", "targets", "name_index")}
        t[field].view(-1)[-1] += 1
        assert cb.windows_fingerprint(ContextWindows(**t)) != fp, field


def test_rehashed_structural_damage_is_still_rejected(built):
    _, _, d, _ = built
    t = copy.deepcopy(d)
    t["real_context_length"]["categories"][-1]["hi"] = 7       # no longer covers length 8
    body = {k: v for k, v in t.items() if k != "definition_sha256"}
    t["definition_sha256"] = cb._digest(body)
    with pytest.raises(cb.BucketError, match="partition"):
        cb.verify_definitions(t)


def test_definitions_deterministic_across_processes(built):
    _, _, d, _ = built
    # load this test file by path: an unrelated installed package named ``tests`` must not shadow it
    code = ("import importlib.util;"
            "from training.wavenet_study import context_buckets as cb;"
            f"spec = importlib.util.spec_from_file_location('context_buckets_test_data', {str(Path(__file__))!r});"
            "t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t);"
            "print(cb.define_buckets(t.windows(t.TRAIN), t.windows(t.DEV))['definition_sha256'])")
    hashes = set()
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                           check=True)
        hashes.add(r.stdout.strip())
    assert hashes == {d["definition_sha256"]}
    assert cb.to_json(cb.define_buckets(windows(TRAIN), windows(DEV))) == cb.to_json(d)


# ------------------------------------------------------------------ test-time application
def test_define_takes_only_train_and_dev():
    assert list(inspect.signature(cb.define_buckets).parameters) == ["train", "dev"]


def test_held_out_application_flags_without_changing_definitions(built):
    train, _, d, counts = built
    before = copy.deepcopy(d)
    held = windows(random_names(77, 120, alphabet="abcd", lengths=(1, 11)))   # synthetic stand-in
    a = cb.assign(d, held, counts)
    assert d == before and a["definition_sha256"] == d["definition_sha256"]
    for s in cb.STRATIFICATIONS:
        sup = a["support"][s]
        assert set(sup) == {c["id"] for c in d[s]["categories"]}
        assert sum(v["n"] for v in sup.values()) == held.num_examples
        assert all(v["insufficient_support"] == (v["n"] < cb.MIN_SUPPORT) for v in sup.values())
        assert any(v["insufficient_support"] for v in sup.values())          # small set: some below 500
    # 'd' never occurs in training: its contexts are unseen and are counted, not dropped
    assert a["support"]["frequency"]["unseen"]["n"] > 0
    # real lengths beyond the dev maximum still find a category
    assert len(a["labels"]["real_context_length"]) == held.num_examples
    # an application to other windows cannot influence the definitions: redefining gives the same hash
    assert cb.define_buckets(train, windows(DEV))["definition_sha256"] == d["definition_sha256"]


def test_assign_requires_matching_train_counts(built):
    _, dev, d, _ = built
    with pytest.raises(cb.BucketError, match="training counts"):
        cb.assign(d, dev, cb.count_train_contexts(windows(TRAIN[:-1])))


def test_assign_on_dev_reproduces_dev_support(built):
    _, dev, d, counts = built
    a = cb.assign(d, dev, counts)
    for s in cb.STRATIFICATIONS:
        defined = {c["id"]: c["dev_support"] for c in d[s]["categories"]}
        assert defined == {k: v["n"] for k, v in a["support"][s].items()}


# ------------------------------------------------------------------ aggregation
def test_aggregation_sums_counts_and_means(built):
    _, dev, d, counts = built
    a = cb.assign(d, dev, counts)
    rng = np.random.default_rng(0)
    nll = rng.gamma(2.0, 0.8, dev.num_examples)
    agg = cb.aggregate(a, torch.tensor(nll, dtype=torch.float32))
    x32 = [float(v) for v in torch.tensor(nll, dtype=torch.float32).double()]
    exact_total = sum(Fraction(v) for v in x32)
    assert agg["num_targets"] == dev.num_examples
    assert agg["nll_sum"] == float(exact_total)                     # fsum is correctly rounded
    for s in cb.STRATIFICATIONS:
        cats = agg["stratifications"][s]["categories"]
        assert sum(c["n"] for c in cats.values()) == dev.num_examples
        assert math.isclose(sum(c["share"] for c in cats.values()), 1.0, rel_tol=1e-12)
        for cid, c in cats.items():
            ref = sum((Fraction(v) for v, lab in zip(x32, a["labels"][s], strict=True) if lab == cid), Fraction(0))
            assert c["nll_sum"] == float(ref)
            if c["n"]:
                assert c["mean_nll"] == c["nll_sum"] / c["n"]
        assert abs(agg["stratifications"][s]["recombined_nll_sum"] - agg["nll_sum"]) <= \
            agg["stratifications"][s]["consistency_bound"]


def test_aggregation_hand_example():
    w = windows(["ab", "c"])                        # 5 targets
    counts = cb.count_train_contexts(w)
    d = cb.define_buckets(w, w)
    a = cb.assign(d, w, counts)
    agg = cb.aggregate(a, [0.5, 1.0, 2.0, 0.25, 4.0])
    pos = agg["stratifications"]["target_position"]["categories"]
    assert pos["first"] == {"n": 2, "nll_sum": 0.75, "mean_nll": 0.375, "share": 0.4, "insufficient_support": True}
    assert pos["interior"]["n"] == 1 and pos["interior"]["nll_sum"] == 1.0
    assert pos["eos"]["nll_sum"] == 6.0 and pos["eos"]["mean_nll"] == 3.0
    assert agg["mean_nll"] == 7.75 / 5


def test_aggregation_stable_summation():
    w = windows(["ab", "c"])
    a = cb.assign(cb.define_buckets(w, w), w, cb.count_train_contexts(w))
    vals = [1e16, 1.0, 1.0, 1.0, 1.0]
    agg = cb.aggregate(a, vals)
    assert agg["nll_sum"] == 1e16 + 4
    naive = 0.0
    for v in vals:                                    # left-to-right float sum loses every 1.0
        naive += v
    assert naive == 1e16 != agg["nll_sum"]


def test_aggregation_rejects_bad_inputs():
    w = windows(["ab"])
    a = cb.assign(cb.define_buckets(w, w), w, cb.count_train_contexts(w))
    for bad in ([0.1, 0.2], [0.1, float("nan"), 0.3], [0.1, float("inf"), 0.3], [0.1, -0.2, 0.3]):
        with pytest.raises(cb.BucketError):
            cb.aggregate(a, bad)
    empty_cat = cb.aggregate(a, [0.1, 0.2, 0.3])["stratifications"]["real_context_length"]["categories"]
    assert all(c["mean_nll"] is None for c in empty_cat.values() if c["n"] == 0)


def test_aggregation_detects_inconsistent_assignment():
    w = windows(["ab"])
    a = cb.assign(cb.define_buckets(w, w), w, cb.count_train_contexts(w))
    broken = copy.deepcopy(a)
    broken["labels"]["target_position"] = broken["labels"]["target_position"][:-1]
    with pytest.raises(ValueError):
        cb.aggregate(broken, [0.1, 0.2, 0.3])

"""Tests for utils.data.make_context_windows / ContextWindows (approved shared-infrastructure spec)."""

from __future__ import annotations

import collections
import math

import pytest
import torch

from utils.data import (
    ContextWindows,
    VocabularyCoverageError,
    load_names,
    make_context_windows,
    make_next_token_sequences,
    split_names,
)
from utils.paths import DEFAULT_NAMES_PATH
from utils.vocabulary import IGNORE_INDEX, Vocabulary

NAMES = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "liam", "noah", "anna", "anna", "eva", "bo"]


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary.from_names(NAMES)


def _ids(vocab: Vocabulary, text: str) -> list[int]:
    return [vocab.stoi[c] for c in text]


# ------------------------------------------------------------ hand examples
def test_emma_k3_hand_computed(vocab):
    w = make_context_windows(["emma"], vocab, context_length=3)
    e, m, a = (vocab.stoi[c] for c in "ema")
    assert w.contexts.tolist() == [[0, 0, 0], [0, 0, e], [0, e, m], [e, m, m], [m, m, a]]
    assert w.targets.tolist() == [e, m, m, a, 0]
    assert w.name_index.tolist() == [0] * 5
    assert w.context_length == 3 and w.num_examples == 5


def test_emma_k1_and_k5(vocab):
    e, m, a = (vocab.stoi[c] for c in "ema")
    w1 = make_context_windows(["emma"], vocab, context_length=1)
    assert w1.contexts.tolist() == [[0], [e], [m], [m], [a]]
    w5 = make_context_windows(["emma"], vocab, context_length=5)
    assert w5.contexts.tolist() == [
        [0, 0, 0, 0, 0], [0, 0, 0, 0, e], [0, 0, 0, e, m], [0, 0, e, m, m], [0, e, m, m, a],
    ]
    assert w1.targets.tolist() == w5.targets.tolist() == [e, m, m, a, 0]


def test_multiple_names_never_cross_boundaries():
    v = Vocabulary.from_names(["ab", "c"])
    a, b, c = (v.stoi[x] for x in "abc")
    w = make_context_windows(["ab", "c"], v, context_length=2)
    assert w.contexts.tolist() == [[0, 0], [0, a], [a, b], [0, 0], [0, c]]   # second name restarts at [0, 0]
    assert w.targets.tolist() == [a, b, 0, c, 0]
    assert w.name_index.tolist() == [0, 0, 0, 1, 1]


def test_boundary_isolation_each_name_matches_its_own_windows(vocab):
    joint = make_context_windows(NAMES, vocab, context_length=4)
    for n, name in enumerate(NAMES):
        alone = make_context_windows([name], vocab, context_length=4)
        rows = joint.name_index == n
        assert torch.equal(joint.contexts[rows], alone.contexts)
        assert torch.equal(joint.targets[rows], alone.targets)
    assert torch.all(joint.name_index[1:] >= joint.name_index[:-1])         # non-decreasing


# ------------------------------------------------------------- guarantees
@pytest.mark.parametrize("k", [1, 2, 3, 8, 20])
def test_counts_eos_and_value_ranges(vocab, k):
    w = make_context_windows(NAMES, vocab, context_length=k)
    M = sum(len(n) + 1 for n in NAMES)
    assert isinstance(w, ContextWindows)
    assert w.contexts.shape == (M, k) and w.targets.shape == (M,) and w.name_index.shape == (M,)
    for t in (w.contexts, w.targets, w.name_index):
        assert t.dtype == torch.int64 and t.device.type == "cpu"
    V = vocab.num_classes
    assert ((w.contexts >= 0) & (w.contexts < V)).all() and ((w.targets >= 0) & (w.targets < V)).all()
    assert not (w.contexts == vocab.pad_id).any() and not (w.targets == vocab.pad_id).any()
    assert not (w.contexts == IGNORE_INDEX).any() and not (w.targets == IGNORE_INDEX).any()
    # exactly one EOS per name, and it is that name's last window
    for n, name in enumerate(NAMES):
        targets_n = w.targets[w.name_index == n].tolist()
        assert targets_n == _ids(vocab, name) + [0]
        assert len(targets_n) == len(name) + 1


@pytest.mark.parametrize("k", [1, 3, 8])
def test_bos_only_as_leading_run(vocab, k):
    w = make_context_windows(NAMES, vocab, context_length=k)
    for row in w.contexts.tolist():
        first_char = next((i for i, t in enumerate(row) if t != 0), len(row))
        assert all(t == 0 for t in row[:first_char]) and all(t != 0 for t in row[first_char:])


def test_leading_zero_count_identifies_position(vocab):
    k = 3
    for name in NAMES:
        w = make_context_windows([name], vocab, context_length=k)
        for t, row in enumerate(w.contexts.tolist()):           # t = number of characters already seen
            leading = next((i for i, x in enumerate(row) if x != 0), k)
            assert leading == k - min(t, k)


def test_consistency_across_context_lengths(vocab):
    windows = {k: make_context_windows(NAMES, vocab, context_length=k) for k in (1, 2, 3, 8)}
    for k, w in windows.items():
        assert torch.equal(w.targets, windows[1].targets) and torch.equal(w.name_index, windows[1].name_index)
        for j in (1, 2, 3, 8):
            if j <= k:
                assert torch.equal(w.contexts[:, k - j:], windows[j].contexts)


def test_k1_equals_unpadded_next_token_pairs(vocab):
    w = make_context_windows(NAMES, vocab, context_length=1)
    seq = make_next_token_sequences(NAMES, vocab)
    assert torch.equal(w.contexts[:, 0], seq.inputs[seq.mask])
    assert torch.equal(w.targets, seq.targets[seq.mask])


def test_deterministic(vocab):
    a = make_context_windows(NAMES, vocab, context_length=3)
    b = make_context_windows(NAMES, vocab, context_length=3)
    assert torch.equal(a.contexts, b.contexts) and torch.equal(a.targets, b.targets)


# ------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "names, exc, match",
    [
        ([], ValueError, "empty"),
        ("emma", TypeError, "single string"),
        (["emma", ""], ValueError, "empty name"),
        (["emma", 3], TypeError, "must be a string"),
        (["emmaz"], VocabularyCoverageError, "unknown character 'z'"),
        (["em.ma"], VocabularyCoverageError, "boundary"),
    ],
)
def test_invalid_names(vocab, names, exc, match):
    with pytest.raises(exc, match=match):
        make_context_windows(names, vocab, context_length=3)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3", None])
def test_invalid_context_length(vocab, k):
    with pytest.raises(ValueError, match="context_length"):
        make_context_windows(["emma"], vocab, context_length=k)  # type: ignore[arg-type]


# ---------------------------------------------------------------- real data
@pytest.mark.skipif(not DEFAULT_NAMES_PATH.is_file(), reason="names dataset not present")
def test_real_dataset_counts_and_conditional_entropy():
    splits = split_names(load_names(), seed=42)
    vocab = Vocabulary.from_names(splits.train)
    for k in (1, 3, 8):
        w = make_context_windows(splits.train, vocab, context_length=k)
        assert w.num_examples == 182_666 == sum(len(n) + 1 for n in splits.train)
        assert not (w.contexts == vocab.pad_id).any()
    w3 = make_context_windows(splits.train, vocab, context_length=3)
    rows = [tuple(r) for r in w3.contexts.tolist()]
    joint = collections.Counter(zip(rows, w3.targets.tolist()))
    ctx = collections.Counter(rows)
    M = w3.num_examples
    h = -sum(n / M * math.log(n / ctx[c]) for (c, _), n in joint.items())
    assert len(ctx) == 5402
    assert h == pytest.approx(1.8681, abs=1e-4)          # empirical H(Y | 3-character context)

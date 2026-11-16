import random

import pytest
import torch

from utils.data import (
    DatasetError,
    Splits,
    VocabularyCoverageError,
    check_vocab_coverage,
    dataset_fingerprint,
    load_names,
    make_next_token_sequences,
    pad_sequences,
    split_names,
)
from utils.paths import DEFAULT_NAMES_PATH, resolve_path, REPO_ROOT
from utils.vocabulary import IGNORE_INDEX, Vocabulary


# ------------------------------------------------------------------ loading
def test_load_names_preserves_order_and_duplicates(names_file, names):
    assert load_names(names_file) == names


def test_load_names_without_trailing_newline(tmp_path):
    p = tmp_path / "n.txt"
    p.write_text("ab\ncd", encoding="utf-8")
    assert load_names(p) == ["ab", "cd"]


@pytest.mark.parametrize(
    "content, message",
    [
        ("ab\n\ncd\n", "empty line on line\\(s\\) 2"),
        ("ab\n cd\n", "leading/trailing whitespace on line\\(s\\) 2"),
        ("ab\nc d\n", "embedded whitespace"),
        ("a.b\n", "boundary symbol"),
        ("ab\r\ncd\r\n", "Windows line ending"),
        ("", "empty"),
    ],
)
def test_load_names_validation(tmp_path, content, message):
    p = tmp_path / "bad.txt"
    p.write_text(content, encoding="utf-8", newline="")
    with pytest.raises(DatasetError, match=message):
        load_names(p)


def test_load_names_missing_file_and_bad_encoding(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_names(tmp_path / "missing.txt")
    p = tmp_path / "latin1.txt"
    p.write_bytes("jos\xe9\n".encode("latin-1"))
    with pytest.raises(DatasetError, match="UTF-8"):
        load_names(p)


def test_relative_paths_resolve_against_repo_root():
    assert resolve_path("data/names.txt") == DEFAULT_NAMES_PATH
    assert (REPO_ROOT / "utils" / "data.py").is_file()


# ---------------------------------------------------------------- splitting
def test_group_policy_has_no_leakage_and_keeps_duplicates(names):
    s = split_names(names, seed=3)
    s.assert_disjoint()
    assert sorted(s.train + s.dev + s.test) == sorted(names)   # nothing lost or added
    for copy in ("anna", "bob"):
        holders = [split for split in ("train", "dev", "test") if copy in s[split]]
        assert len(holders) == 1


def test_dedupe_policy(names):
    s = split_names(names, seed=3, duplicate_policy="dedupe")
    s.assert_disjoint()
    all_names = s.train + s.dev + s.test
    assert len(all_names) == len(set(all_names)) == len(set(names))


def test_none_policy_reproduces_legacy_shuffle_and_can_leak():
    # Legacy scripts: random.seed(42); random.shuffle(words); cut at 80/90 %.
    data = [f"n{i % 50}" for i in range(400)]                  # heavy duplication
    legacy = list(data)
    random.seed(42)
    random.shuffle(legacy)
    n1, n2 = int(0.8 * len(legacy)), int(0.9 * len(legacy))
    s = split_names(data, seed=42, duplicate_policy="none")
    assert list(s.train) == legacy[:n1] and list(s.dev) == legacy[n1:n2] and list(s.test) == legacy[n2:]
    assert any(s.overlap().values())
    with pytest.raises(DatasetError, match="leakage"):
        s.assert_disjoint()


def test_split_is_deterministic_and_order_independent(names):
    a = split_names(names, seed=11)
    b = split_names(list(reversed(names)), seed=11)
    assert a == b
    assert split_names(names, seed=12) != a


def test_split_does_not_touch_global_rng(names):
    random.seed(123)
    before = random.getstate()
    split_names(names, seed=5)
    assert random.getstate() == before


def test_split_sizes_follow_fractions_on_unique_names(names):
    s = split_names(names, seed=0, fractions=(0.5, 0.25, 0.25), duplicate_policy="dedupe")
    assert s.sizes() == {"train": 10, "dev": 5, "test": 5}


@pytest.mark.parametrize("fr", [(0.8, 0.1), (0.8, 0.3, 0.1), (0.0, 0.5, 0.5), (1.1, -0.05, -0.05)])
def test_split_fraction_validation(names, fr):
    with pytest.raises(ValueError):
        split_names(names, seed=0, fractions=fr)


def test_split_rejects_bad_policy_and_seed(names):
    with pytest.raises(ValueError):
        split_names(names, seed=0, duplicate_policy="drop")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        split_names(names, seed=1.5)  # type: ignore[arg-type]


# ------------------------------------------------------- fingerprints/coverage
def test_fingerprints(names_file, names):
    fp = dataset_fingerprint(names_file)
    assert fp["num_names"] == len(names) and fp["num_unique_names"] == len(set(names))
    assert fp == dataset_fingerprint(names_file)
    names_file.write_text("\n".join(names) + "\nzed\n", encoding="utf-8")
    fp2 = dataset_fingerprint(names_file)
    assert fp2["file_sha256"] != fp["file_sha256"] and fp2["names_sha256"] != fp["names_sha256"]
    s = split_names(names, seed=1)
    assert s.fingerprint() == split_names(names, seed=1).fingerprint()


def test_vocab_coverage_error_names_split_and_chars():
    s = Splits(train=("ab", "ba"), dev=("abba",), test=("xyz",), seed=0, fractions=(0.5, 0.25, 0.25),
               duplicate_policy="group")
    vocab = Vocabulary.from_names(s.train)
    with pytest.raises(VocabularyCoverageError, match="test: unseen characters 'x'×1, 'y'×1, 'z'×1"):
        check_vocab_coverage(vocab, s)


def test_vocab_coverage_passes(names):
    s = split_names(names, seed=0)
    check_vocab_coverage(Vocabulary.from_names(names), s)


# ---------------------------------------------------------------- sequences
def test_pad_sequences():
    x, lengths = pad_sequences([[1, 2, 3], [4]], pad_value=9)
    assert x.tolist() == [[1, 2, 3], [4, 9, 9]] and lengths.tolist() == [3, 1]
    x, _ = pad_sequences([[1]], pad_value=0, max_len=3)
    assert x.shape == (1, 3)
    with pytest.raises(ValueError, match="exceeds"):
        pad_sequences([[1, 2, 3]], pad_value=0, max_len=2)


def test_next_token_sequences_padding_is_distinct_from_boundary():
    v = Vocabulary.from_names(["emma", "al"])
    seq = make_next_token_sequences(["emma", "al"], v, max_len=6)
    e, m, a, l = (v.stoi[c] for c in "emal")
    assert seq.inputs.tolist() == [[0, e, m, m, a, v.pad_id], [0, a, l, v.pad_id, v.pad_id, v.pad_id]]
    assert seq.targets.tolist() == [[e, m, m, a, 0, IGNORE_INDEX], [a, l, 0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
    assert seq.lengths.tolist() == [5, 3] and seq.num_targets == 8
    assert torch.equal(seq.mask, seq.targets != IGNORE_INDEX)
    assert torch.equal(seq.mask, seq.inputs != v.pad_id)
    # PAD never appears as a target, and padded inputs never look like BOS.
    assert not (seq.targets == v.pad_id).any()
    assert not ((seq.inputs == 0) & ~seq.mask).any()


def test_next_token_sequences_refuse_truncation_and_unknown_chars():
    v = Vocabulary.from_names(["emma"])
    with pytest.raises(ValueError, match="exceeds"):
        make_next_token_sequences(["emma"], v, max_len=4)
    with pytest.raises(VocabularyCoverageError):
        make_next_token_sequences(["zoe"], v)


@pytest.mark.skipif(not DEFAULT_NAMES_PATH.is_file(), reason="names dataset not present")
def test_real_dataset_group_split_has_no_leakage():
    names = load_names()
    s = split_names(names, seed=42)
    s.assert_disjoint()
    check_vocab_coverage(Vocabulary.from_names(s.train), s)
    assert sum(s.sizes().values()) == len(names)

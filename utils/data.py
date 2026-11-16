"""Shared dataset preparation for the names corpus.

Scope
-----
This module owns everything that is *architecture-independent*: loading and
validating the raw file, leakage-safe train/dev/test splitting at the name
level, fingerprints, vocabulary coverage checks, and one generic primitive for
padded next-token sequences. Fixed-context windows (MLP, WaveNet), streams of
concatenated names, or any other model-specific input layout belong in model
adapters, built from :func:`encode_names`.

Leakage and duplicates
----------------------
The names file contains repeated names. Splitting rows naively puts copies of
the same string into different splits, so the dev/test sets contain names the
model trained on. ``duplicate_policy`` controls this:

``"group"`` (default)
    Split the *unique* names, then send every copy of a name to the same
    split. No string appears in two splits and duplicate frequency (a real
    signal about name popularity) is preserved inside each split.
``"dedupe"``
    Split the unique names and keep one copy of each. No overlap; frequency
    information is discarded.
``"none"``
    Shuffle and cut the raw rows. Copies may land in different splits. Kept
    only to reproduce legacy splits; :meth:`Splits.assert_disjoint` will fail.

Splitting always happens on whole names, before any character-level example
is generated.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from utils.paths import DEFAULT_NAMES_PATH, relative_to_repo, resolve_path
from utils.vocabulary import IGNORE_INDEX, Vocabulary, VocabularyError

DuplicatePolicy = Literal["group", "dedupe", "none"]
DUPLICATE_POLICIES: tuple[str, ...] = ("group", "dedupe", "none")
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "test")


class DatasetError(ValueError):
    """Raised when the dataset file or its contents are invalid."""


class VocabularyCoverageError(DatasetError):
    """Raised when dev/test contain characters missing from the vocabulary."""


# ---------------------------------------------------------------------- load
def load_names(path: str | Path | None = None, *, boundary: str = ".") -> list[str]:
    """Load one name per line from a UTF-8 text file.

    ``path`` defaults to ``data/names.txt`` and relative paths are resolved
    against the repository root. Names are returned exactly as stored, in file
    order; no case folding or normalisation is applied. A trailing newline at
    the end of the file is accepted (and its absence too).

    Raises :class:`DatasetError` listing up to five offending line numbers if
    the file contains empty lines, surrounding whitespace, embedded
    whitespace, or the boundary symbol.
    """
    file = resolve_path(path) if path is not None else DEFAULT_NAMES_PATH
    if not file.exists():
        raise DatasetError(f"dataset file not found: {file}")
    if not file.is_file():
        raise DatasetError(f"dataset path is not a file: {file}")
    try:
        text = file.read_bytes().decode("utf-8")  # bytes: no newline translation, so \r is detectable
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{file} is not valid UTF-8: {exc}") from exc

    names = text.split("\n")
    if names and names[-1] == "":
        names.pop()  # trailing newline
    problems: dict[str, list[int]] = {}

    def flag(kind: str, lineno: int) -> None:
        problems.setdefault(kind, []).append(lineno)

    for lineno, name in enumerate(names, start=1):
        if name.endswith("\r"):
            flag("Windows line ending (\\r)", lineno)
        elif name == "":
            flag("empty line", lineno)
        elif name != name.strip():
            flag("leading/trailing whitespace", lineno)
        elif any(ch.isspace() for ch in name):
            flag("embedded whitespace", lineno)
        elif boundary in name:
            flag(f"contains boundary symbol {boundary!r}", lineno)
    if problems:
        details = "; ".join(
            f"{kind} on line(s) {', '.join(map(str, lines[:5]))}{' ...' if len(lines) > 5 else ''}"
            for kind, lines in problems.items()
        )
        raise DatasetError(f"{file}: invalid names: {details}")
    if not names:
        raise DatasetError(f"{file}: dataset is empty")
    return names


# --------------------------------------------------------------------- split
@dataclass(frozen=True)
class Splits:
    """Train/dev/test name lists plus the parameters that produced them."""

    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    fractions: tuple[float, float, float]
    duplicate_policy: str

    def __getitem__(self, split: str) -> tuple[str, ...]:
        if split not in SPLIT_NAMES:
            raise KeyError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, split)

    def sizes(self) -> dict[str, int]:
        """Number of names (rows, duplicates included) per split."""
        return {s: len(self[s]) for s in SPLIT_NAMES}

    def overlap(self) -> dict[str, int]:
        """Number of distinct strings shared by each pair of splits."""
        sets = {s: set(self[s]) for s in SPLIT_NAMES}
        return {
            "train&dev": len(sets["train"] & sets["dev"]),
            "train&test": len(sets["train"] & sets["test"]),
            "dev&test": len(sets["dev"] & sets["test"]),
        }

    def assert_disjoint(self) -> None:
        """Raise :class:`DatasetError` if any name string occurs in two splits."""
        overlap = self.overlap()
        if any(overlap.values()):
            raise DatasetError(f"splits share names (leakage): {overlap}")

    def fingerprint(self) -> dict[str, object]:
        """Sizes and order-sensitive SHA-256 of each split, for run metadata."""
        return {
            "seed": self.seed,
            "fractions": list(self.fractions),
            "duplicate_policy": self.duplicate_policy,
            "sizes": self.sizes(),
            "sha256": {s: _sha256_lines(self[s]) for s in SPLIT_NAMES},
        }


def validate_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    """Check (train, dev, test) fractions: three non-negative numbers, train > 0, sum 1."""
    if len(fractions) != 3:
        raise ValueError(f"fractions must have three entries (train, dev, test), got {len(fractions)}")
    fr = tuple(float(f) for f in fractions)
    if any(f < 0 for f in fr):
        raise ValueError(f"fractions must be non-negative, got {fr}")
    if fr[0] <= 0:
        raise ValueError("the train fraction must be positive")
    if abs(sum(fr) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fr} (sum {sum(fr)})")
    return fr  # type: ignore[return-value]


def _cut_points(n: int, fr: tuple[float, float, float]) -> tuple[int, int]:
    # Floor-based boundaries: train = [0, int(f0*n)), dev = [.., int((f0+f1)*n)), test = rest.
    return int(fr[0] * n), int((fr[0] + fr[1]) * n)


def split_names(
    names: Sequence[str],
    *,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int,
    duplicate_policy: DuplicatePolicy = "group",
) -> Splits:
    """Deterministically split names into train/dev/test.

    The split uses a private ``random.Random(seed)``; global RNG state is
    neither read nor modified. For ``"group"`` and ``"dedupe"`` the unique
    names are sorted before shuffling, so the result does not depend on the
    order of lines in the file. Fractions apply to *unique* names under those
    policies and to rows under ``"none"``.

    The output order within each split is the shuffled order (copies of a
    name are adjacent under ``"group"``); downstream samplers should shuffle
    examples themselves.
    """
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ValueError(f"duplicate_policy must be one of {DUPLICATE_POLICIES}, got {duplicate_policy!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    fr = validate_fractions(fractions)
    if len(names) == 0:
        raise DatasetError("cannot split an empty list of names")
    rng = random.Random(seed)

    if duplicate_policy == "none":
        rows = list(names)
        rng.shuffle(rows)
        n1, n2 = _cut_points(len(rows), fr)
        parts = (rows[:n1], rows[n1:n2], rows[n2:])
    else:
        counts = Counter(names)
        unique = sorted(counts)
        rng.shuffle(unique)
        n1, n2 = _cut_points(len(unique), fr)
        groups = (unique[:n1], unique[n1:n2], unique[n2:])
        if duplicate_policy == "group":
            parts = tuple([name for name in g for _ in range(counts[name])] for g in groups)
        else:
            parts = groups

    if not parts[0]:
        raise DatasetError("train split is empty; the dataset is too small for these fractions")
    return Splits(
        train=tuple(parts[0]),
        dev=tuple(parts[1]),
        test=tuple(parts[2]),
        seed=seed,
        fractions=fr,
        duplicate_policy=duplicate_policy,
    )


# --------------------------------------------------------------- fingerprint
def _sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with resolve_path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(path: str | Path | None = None, names: Sequence[str] | None = None) -> dict[str, object]:
    """Describe the dataset so a run can be tied to the exact data it used.

    ``file_sha256`` changes with any byte change (including line endings);
    ``names_sha256`` hashes the parsed names in file order, so it only changes
    when the content the models see changes.
    """
    file = resolve_path(path) if path is not None else DEFAULT_NAMES_PATH
    if names is None:
        names = load_names(file)
    lengths = [len(n) for n in names]
    return {
        "path": relative_to_repo(file),
        "file_sha256": file_sha256(file),
        "num_bytes": file.stat().st_size,
        "num_names": len(names),
        "num_unique_names": len(set(names)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "names_sha256": _sha256_lines(names),
    }


# ------------------------------------------------------------------ coverage
def check_vocab_coverage(vocab: Vocabulary, splits: Splits) -> None:
    """Raise :class:`VocabularyCoverageError` if dev or test use unseen characters.

    The vocabulary should be built from ``splits.train``; by construction the
    train split is always covered, so only dev and test are checked (train is
    checked too, as a guard against passing a mismatched vocabulary).
    """
    errors: list[str] = []
    for split in SPLIT_NAMES:
        missing = vocab.unknown_chars(splits[split])
        if missing:
            examples = [n for n in splits[split] if any(c in missing for c in n)][:3]
            chars = ", ".join(f"{c!r}×{k}" for c, k in sorted(missing.items()))
            errors.append(f"{split}: unseen characters {chars} (e.g. {examples})")
    if errors:
        raise VocabularyCoverageError(
            "vocabulary does not cover all splits: " + "; ".join(errors)
            + ". Build the vocabulary from the training split and choose a split seed/policy "
            "that places rare characters in train, or filter those names explicitly."
        )


# ---------------------------------------------------------------- sequences
def encode_names(
    names: Iterable[str], vocab: Vocabulary, *, add_bos: bool = False, add_eos: bool = False
) -> list[list[int]]:
    """Encode each name to a list of ids (no padding, no tensor conversion)."""
    return [vocab.encode(n, add_bos=add_bos, add_eos=add_eos) for n in names]


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    pad_value: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad integer sequences into a ``(B, T)`` LongTensor.

    Returns ``(padded, lengths)``. ``T`` is ``max_len`` if given, otherwise
    the longest sequence. Sequences longer than ``max_len`` raise instead of
    being truncated silently.
    """
    if len(sequences) == 0:
        raise ValueError("cannot pad an empty batch")
    lengths = [len(s) for s in sequences]
    if min(lengths) == 0:
        raise ValueError("sequences must be non-empty")
    width = max(lengths) if max_len is None else max_len
    if max_len is not None and max(lengths) > max_len:
        raise ValueError(f"a sequence of length {max(lengths)} exceeds max_len={max_len}")
    out = torch.full((len(sequences), width), pad_value, dtype=torch.long)
    for row, seq in enumerate(sequences):
        out[row, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    return out, torch.tensor(lengths, dtype=torch.long)


@dataclass(frozen=True)
class NextTokenSequences:
    """Padded next-character prediction data for a list of names.

    For ``"emma"`` with ``T = 6``::

        inputs  = [.,  e,  m,  m,  a,  PAD]     ids [0, 5, 13, 13, 1, 27]
        targets = [e,  m,  m,  a,  .,  -100]    ids [5, 13, 13, 1, 0, -100]
        mask    = [1,  1,  1,  1,  1,  0]
        lengths = 5

    ``inputs[t]`` is the context token and ``targets[t]`` the character to
    predict next. Input padding uses ``vocab.pad_id`` (never a target);
    target padding uses ``IGNORE_INDEX``. ``mask`` and ``lengths`` describe the
    same valid region, so models may use whichever representation suits them
    (attention key masks, ``pack_padded_sequence``, or masked losses).
    """

    inputs: torch.Tensor  # (B, T) long
    targets: torch.Tensor  # (B, T) long
    lengths: torch.Tensor  # (B,) long
    mask: torch.Tensor  # (B, T) bool

    @property
    def num_targets(self) -> int:
        """Number of valid (non-ignored) prediction targets."""
        return int(self.lengths.sum().item())


def make_next_token_sequences(
    names: Sequence[str], vocab: Vocabulary, *, max_len: int | None = None
) -> NextTokenSequences:
    """Build padded BOS-prefixed inputs and EOS-terminated targets.

    Each name of length ``L`` produces ``L + 1`` prediction targets (its
    characters plus EOS). ``max_len`` is the padded time dimension and must be
    at least ``max(len(name)) + 1``; names are never truncated, because a
    truncated name would silently drop its EOS target.
    """
    try:
        encoded = encode_names(names, vocab)
    except VocabularyError as exc:
        raise VocabularyCoverageError(str(exc)) from exc
    inputs_list = [[vocab.bos_id, *ids] for ids in encoded]
    targets_list = [[*ids, vocab.eos_id] for ids in encoded]
    inputs, lengths = pad_sequences(inputs_list, pad_value=vocab.pad_id, max_len=max_len)
    targets, _ = pad_sequences(targets_list, pad_value=IGNORE_INDEX, max_len=max_len)
    mask = torch.arange(inputs.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return NextTokenSequences(inputs=inputs, targets=targets, lengths=lengths, mask=mask)


# ----------------------------------------------------------- context windows
def _validate_name_list(names: Sequence[str]) -> None:
    """Reject a single string, an empty list, non-string items and empty names."""
    if isinstance(names, str):
        raise TypeError(f"names must be a sequence of names, not a single string {names!r}")
    if len(names) == 0:
        raise ValueError("names is empty")
    for pos, name in enumerate(names):
        if not isinstance(name, str):
            raise TypeError(f"names[{pos}] must be a string, got {type(name).__name__}")
        if name == "":
            raise ValueError(f"names[{pos}] is an empty name; every name needs at least one character")


@dataclass(frozen=True)
class ContextWindows:
    """Fixed-length next-token prediction windows for a list of names.

    For a name ``x_1 .. x_L`` define ``x_t = BOS (0)`` for ``t ≤ 0`` and
    ``x_{L+1} = EOS (0)``. Window ``t`` (``t = 1 .. L+1``) is::

        context (x_{t-k}, …, x_{t-1})  →  target x_t

    so each name contributes exactly ``L + 1`` examples. For ``"emma"``, ``k = 3``::

        [., ., .] → e    [., ., e] → m    [., e, m] → m    [e, m, m] → a    [m, m, a] → .

    Guarantees:

    * every name starts from ``[BOS] * k``; windows never cross names;
    * ``contexts`` contain only BOS/character ids. PAD is never constructed;
      0 occurs only as a leading run of each row (names never contain the
      boundary symbol), so the number of leading zeros identifies how many
      characters have been seen;
    * ``targets`` and ``name_index`` do not depend on ``k``, and the last
      ``j`` columns of the ``k``-windows equal the ``j``-windows (``j ≤ k``);
    * rows are ordered by name (input order) and then by position ``t``.
    """

    contexts: torch.Tensor  # [M, k] int64, CPU — ids of the k previous tokens
    targets: torch.Tensor  # [M] int64, CPU — next token; 0 is EOS
    name_index: torch.Tensor  # [M] int64, CPU — index of the source name, non-decreasing

    @property
    def context_length(self) -> int:
        """``k``: number of previous tokens per example."""
        return self.contexts.shape[1]

    @property
    def num_examples(self) -> int:
        """``M = Σ_names (len(name) + 1)``."""
        return self.targets.shape[0]


def make_context_windows(names: Sequence[str], vocab: Vocabulary, *, context_length: int) -> ContextWindows:
    """Build one fixed-length window per real next-token prediction (EOS included).

    The left context is initialised with ``context_length`` repeated BOS tokens
    (id 0) and shifted by one token per step: ``context = context[1:] + [token]``.
    See :class:`ContextWindows` for the exact definition and guarantees.

    Parameters
    ----------
    names:
        Non-empty sequence of non-empty names (a single string is rejected).
    vocab:
        Shared vocabulary; unknown characters raise :class:`VocabularyCoverageError`.
    context_length:
        ``k ≥ 1`` (int; ``bool`` rejected).

    Returns
    -------
    ContextWindows
        ``contexts [M, k]``, ``targets [M]``, ``name_index [M]``: all int64 on CPU.
    """
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 1:
        raise ValueError(f"context_length must be an int >= 1, got {context_length!r}")
    _validate_name_list(names)
    try:
        encoded = encode_names(names, vocab)
    except VocabularyError as exc:
        raise VocabularyCoverageError(str(exc)) from exc

    bos = vocab.bos_id
    contexts: list[list[int]] = []
    targets: list[int] = []
    name_index: list[int] = []
    for n, ids in enumerate(encoded):
        context = [bos] * context_length  # every name starts fresh: no crossing of name boundaries
        for token in [*ids, vocab.eos_id]:
            contexts.append(context)
            targets.append(token)
            name_index.append(n)
            context = context[1:] + [token]
    return ContextWindows(
        contexts=torch.tensor(contexts, dtype=torch.long).reshape(len(targets), context_length),
        targets=torch.tensor(targets, dtype=torch.long),
        name_index=torch.tensor(name_index, dtype=torch.long),
    )

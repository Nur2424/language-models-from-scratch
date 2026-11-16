"""Character vocabulary and the project's token conventions.

Token conventions
-----------------
* **Boundary token** ``"."`` with id ``0`` acts as both BOS and EOS. A name
  ``"emma"`` is modelled as ``. e m m a .``: the model sees ``.`` as the first
  input and must predict ``.`` after the last character. Sharing one id keeps
  the output space at ``1 + number_of_characters`` classes (27 for the names
  dataset), so NLL values remain comparable across all architectures.
* **Output classes** are ids ``0 .. num_classes - 1``. Every model's final
  layer must produce exactly ``num_classes`` logits.
* **Input padding** uses a separate id, ``pad_id == num_classes``. It is an
  *input-only* id: it can never be a prediction target and never appears in
  the output layer, so padding cannot be confused with a real boundary.
  Models that embed padded inputs should size their embedding table with
  ``num_input_ids`` (``num_classes + 1``) and may pass ``padding_idx=pad_id``.
  Models that never see padded inputs (fixed context windows, recurrent
  models fed with lengths/masks) can ignore ``pad_id`` entirely.
* **Ignored targets** use :data:`IGNORE_INDEX` (``-100``, PyTorch's default for
  ``cross_entropy``). Padded target positions carry this value.

Characters are sorted by Unicode code point, so the vocabulary is a pure
function of the *set* of characters in the names it is built from.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Target value ignored by losses and metrics (PyTorch's cross_entropy default).
IGNORE_INDEX: int = -100

#: Default boundary symbol shared by BOS and EOS.
BOUNDARY_TOKEN: str = "."
BOUNDARY_ID: int = 0


class VocabularyError(ValueError):
    """Raised for invalid vocabularies, unknown characters or invalid ids."""


@dataclass(frozen=True)
class Vocabulary:
    """Immutable character vocabulary with a shared BOS/EOS boundary token.

    Build it with :meth:`from_names` from the **training split only**, then
    check dev/test with :func:`utils.data.check_vocab_coverage`.
    """

    chars: tuple[str, ...]
    boundary: str = BOUNDARY_TOKEN
    _stoi: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.chars:
            raise VocabularyError("vocabulary must contain at least one character")
        if len(self.boundary) != 1:
            raise VocabularyError(f"boundary token must be one character, got {self.boundary!r}")
        if any(len(c) != 1 for c in self.chars):
            raise VocabularyError("every vocabulary entry must be a single character")
        if len(set(self.chars)) != len(self.chars):
            raise VocabularyError("vocabulary characters must be unique")
        if list(self.chars) != sorted(self.chars):
            raise VocabularyError("vocabulary characters must be sorted (deterministic ids)")
        if self.boundary in self.chars:
            raise VocabularyError(
                f"boundary token {self.boundary!r} also occurs inside names; "
                "choose a different boundary symbol"
            )
        stoi = {self.boundary: BOUNDARY_ID}
        stoi.update({c: i + 1 for i, c in enumerate(self.chars)})
        object.__setattr__(self, "_stoi", stoi)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_names(cls, names: Iterable[str], boundary: str = BOUNDARY_TOKEN) -> Vocabulary:
        """Build a vocabulary from the characters appearing in ``names``."""
        charset: set[str] = set()
        count = 0
        for name in names:
            if not isinstance(name, str):
                raise VocabularyError(f"names must be strings, got {type(name).__name__}")
            charset.update(name)
            count += 1
        if count == 0:
            raise VocabularyError("cannot build a vocabulary from zero names")
        return cls(chars=tuple(sorted(charset)), boundary=boundary)

    # ------------------------------------------------------------ properties
    @property
    def boundary_id(self) -> int:
        """Id of the shared BOS/EOS token (always 0)."""
        return BOUNDARY_ID

    @property
    def bos_id(self) -> int:
        return BOUNDARY_ID

    @property
    def eos_id(self) -> int:
        return BOUNDARY_ID

    @property
    def num_classes(self) -> int:
        """Size of the output distribution: boundary + characters."""
        return len(self.chars) + 1

    @property
    def pad_id(self) -> int:
        """Input-only padding id (``num_classes``); never a valid target."""
        return self.num_classes

    @property
    def num_input_ids(self) -> int:
        """Embedding-table size for models that consume padded inputs."""
        return self.num_classes + 1

    @property
    def stoi(self) -> dict[str, int]:
        """Copy of the character → id mapping (boundary included)."""
        return dict(self._stoi)

    @property
    def itos(self) -> dict[int, str]:
        """Copy of the id → character mapping (boundary included, no PAD)."""
        return {i: s for s, i in self._stoi.items()}

    def __len__(self) -> int:
        return self.num_classes

    # ------------------------------------------------------- encode / decode
    def unknown_chars(self, names: Iterable[str]) -> Counter[str]:
        """Count characters in ``names`` that are not in the vocabulary.

        The boundary symbol counts as unknown here because it must never
        appear inside a name.
        """
        missing: Counter[str] = Counter()
        for name in names:
            for ch in name:
                if ch == self.boundary or ch not in self._stoi:
                    missing[ch] += 1
        return missing

    def encode(self, name: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Map a name to ids, optionally framing it with the boundary token."""
        ids: list[int] = [BOUNDARY_ID] if add_bos else []
        for pos, ch in enumerate(name):
            if ch == self.boundary:
                raise VocabularyError(
                    f"name {name!r} contains the boundary token {self.boundary!r} at position {pos}"
                )
            try:
                ids.append(self._stoi[ch])
            except KeyError:
                raise VocabularyError(
                    f"unknown character {ch!r} at position {pos} in name {name!r}"
                ) from None
        if add_eos:
            ids.append(BOUNDARY_ID)
        return ids

    def decode(self, ids: Sequence[int] | Any, *, strip_bos: bool = True, stop_at_eos: bool = True) -> str:
        """Map ids back to a string.

        With the defaults, a leading boundary is treated as BOS and removed,
        and decoding stops at the next boundary (EOS). With both flags off,
        boundaries are rendered as the boundary symbol. PAD and out-of-range
        ids raise :class:`VocabularyError`: they indicate a bug upstream, so
        they are reported rather than silently skipped. Accepts any sequence
        of ints, including a 1-D tensor.
        """
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)
        if ids and not isinstance(ids[0], int):
            raise VocabularyError("decode expects a flat sequence of integer ids")
        start = 1 if (strip_bos and ids and ids[0] == BOUNDARY_ID) else 0
        out: list[str] = []
        for pos in range(start, len(ids)):
            i = ids[pos]
            if i == BOUNDARY_ID and stop_at_eos:
                break
            if i == self.pad_id:
                raise VocabularyError(f"PAD id {i} at position {pos}; slice padded sequences by length first")
            if not 0 <= i < self.num_classes:
                raise VocabularyError(f"id {i} at position {pos} is outside [0, {self.num_classes})")
            out.append(self.boundary if i == BOUNDARY_ID else self.chars[i - 1])
        return "".join(out)

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """Plain-data representation (safe for JSON and ``weights_only`` loads)."""
        return {"chars": "".join(self.chars), "boundary": self.boundary}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Vocabulary:
        """Inverse of :meth:`to_dict`, with validation."""
        try:
            chars, boundary = data["chars"], data["boundary"]
        except KeyError as exc:
            raise VocabularyError(f"vocabulary dict is missing key {exc}") from None
        return cls(chars=tuple(chars), boundary=boundary)

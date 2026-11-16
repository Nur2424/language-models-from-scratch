"""Small, safe file-I/O helpers shared by config, logging, checkpoint and run modules.

Atomic writes
-------------
``atomic_write_*`` write to a temporary file in the destination directory,
flush and ``fsync`` it, then ``os.replace`` it over the target. ``os.replace``
is atomic on POSIX and Windows when source and target are on the same file
system, so a crash leaves either the old file or the new one, never a
truncated mix. The parent directory is not fsynced; after a power loss the
rename itself may be lost, but the file is never half-written.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch


def to_jsonable(obj: Any) -> Any:
    """Convert common research objects into JSON-serialisable structures.

    Handles dataclasses, enums, paths, tuples/sets, numpy scalars/arrays,
    0-d and small tensors. Non-finite floats are mapped to the strings
    ``"nan"``, ``"inf"`` and ``"-inf"`` because strict JSON forbids them.
    Raises ``TypeError`` for anything else instead of silently stringifying.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj) if isinstance(obj, (set, frozenset)) else obj
        return [to_jsonable(v) for v in items]
    if isinstance(obj, torch.Tensor):
        if obj.numel() > 1024:
            raise TypeError(f"refusing to serialise a tensor with {obj.numel()} elements to JSON")
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    try:  # numpy is optional for this function
        import numpy as np

        if isinstance(obj, np.generic):
            return to_jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def dumps_json(obj: Any, *, indent: int | None = 2, sort_keys: bool = False) -> str:
    """Serialise ``obj`` with :func:`to_jsonable` and strict JSON (no NaN)."""
    return json.dumps(to_jsonable(obj), indent=indent, sort_keys=sort_keys, allow_nan=False)


def _atomic_write(path: Path, write: Callable[[Any], None], mode: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, **({"encoding": "utf-8"} if "b" not in mode else {})) as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Atomically write UTF-8 text to ``path``."""
    return _atomic_write(Path(path), lambda fh: fh.write(text), "w")


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """Atomically write ``obj`` as strict, human-readable JSON."""
    return atomic_write_text(path, dumps_json(obj, indent=indent) + "\n")


def atomic_torch_save(path: str | Path, obj: Any) -> Path:
    """Atomically ``torch.save`` ``obj`` to ``path``."""
    return _atomic_write(Path(path), lambda fh: torch.save(obj, fh), "wb")


def read_json(path: str | Path) -> Any:
    """Read a JSON file, raising a message that names the file on failure."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a single line.

    Each record is written with a single ``write`` call on a file opened in
    append mode, which is the practical guarantee needed for one writer.
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"JSONL records must be mappings, got {type(record).__name__}")
    line = dumps_json(record, indent=None) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file; malformed lines raise with a line number."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL record ({exc})") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read every record of a JSONL file into a list."""
    return list(iter_jsonl(path))

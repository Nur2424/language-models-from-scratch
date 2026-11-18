"""A single-process lock for the study's training drivers.

Starting a driver while another one is still training would make two processes write into the
same run folders. ``study_lock(root)`` refuses that: it creates ``<root>/.runner.lock`` exclusively
and records the process id. A lock left behind by a process that no longer exists is replaced; a
lock held by a live process stops the new one with a clear message.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_NAME = ".runner.lock"


class StudyLocked(RuntimeError):
    """Another live process holds the study lock."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def study_lock(root: Path) -> Iterator[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or "0")
            except (OSError, ValueError):
                pid = 0
            if pid and _alive(pid):
                raise StudyLocked(f"another run (process {pid}) is using {root}; stop it first "
                                  f"(kill {pid}) or wait for it to finish") from None
            path.unlink(missing_ok=True)   # stale: its process is gone
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)
        return
    raise StudyLocked(f"could not acquire {path}")

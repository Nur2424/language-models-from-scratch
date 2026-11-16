"""Run-directory and metadata primitives (no training logic).

Layout created by :meth:`RunManager.create`::

    results/runs/<model_name>/run_<NNN>/
        config.json        resolved configuration (written once)
        metadata.json      status, timestamps, git, environment, dataset, errors
        history.jsonl      appended by the future training loop
        diagnostics/       for diagnostic summaries
        logs/              for utils.logging_utils output
        model_best.pt / model_final.pt / checkpoint_last.pt   (see utils.checkpoint)

Status lifecycle::

    created ──▶ running ──▶ completed
                   │  ▲
                   ▼  │ (resume)
          failed / interrupted

``completed`` is terminal. Each running period is recorded as a *session*
(start, end, duration, outcome), so resumed runs keep an honest timeline and
``active_duration_seconds`` excludes time spent stopped.

Concurrency
-----------
Run numbers are allocated with an atomic ``mkdir`` (retrying on collision), so
independent processes never share or overwrite a run directory. Metadata
updates are atomic replaces but assume a single writer per run.
"""

from __future__ import annotations

import datetime as _dt
import enum
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from utils.checkpoint import CheckpointKind, checkpoint_path
from utils.io import append_jsonl, atomic_write_json, read_json, to_jsonable
from utils.paths import REPO_ROOT, RESULTS_DIR, relative_to_repo

METADATA_SCHEMA_VERSION = 1
_MODEL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


class RunStatus(str, enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


_ALLOWED_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.RUNNING},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED},
    RunStatus.FAILED: {RunStatus.RUNNING},
    RunStatus.INTERRUPTED: {RunStatus.RUNNING},
    RunStatus.COMPLETED: set(),
}

# Keys managed by RunManager methods; update_metadata() refuses to touch them.
_RESERVED_KEYS = {
    "schema_version", "run_id", "run_uid", "model_name", "run_number", "status", "created_at",
    "started_at", "ended_at", "active_duration_seconds", "sessions", "error",
}


class RunError(RuntimeError):
    """Raised for invalid run operations (bad status transition, missing files...)."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------- metadata
def git_info(repo_dir: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Commit hash, branch and dirty flag; ``available=False`` with a reason
    when git or the repository is unavailable (never raises)."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}".strip()}


def environment_info() -> dict[str, Any]:
    """Software and hardware description for reproducibility records."""
    try:
        import numpy as np

        numpy_version: str | None = np.__version__
    except ImportError:  # pragma: no cover
        numpy_version = None
    mps = getattr(torch.backends, "mps", None)
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "numpy": numpy_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "mps_available": bool(mps) and mps.is_available(),
    }
    return info


# ------------------------------------------------------------- run manager
class RunManager:
    """Handle on one run directory. Create with :meth:`create`, reopen with :meth:`open`."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        if not self.metadata_path.is_file():
            raise RunError(f"{self.run_dir} is not a run directory (missing metadata.json)")
        meta = self.read_metadata()
        self.run_id: str = meta["run_id"]
        self.model_name: str = meta["model_name"]
        self._session_start: float | None = None

    # ----------------------------------------------------------- creation
    @classmethod
    def create(
        cls,
        model_name: str,
        *,
        config: Mapping[str, Any],
        results_root: str | Path | None = None,
        dataset_fingerprint: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        max_attempts: int = 1000,
    ) -> RunManager:
        """Allocate ``<results_root>/runs/<model_name>/run_<NNN>`` and initialise it.

        ``config`` is the fully resolved configuration as plain data (use
        ``utils.config.config_to_dict``); it is written to ``config.json``.
        ``results_root`` defaults to ``<repo>/results``.
        """
        if not _MODEL_NAME_RE.match(model_name):
            raise ValueError(f"model_name must match {_MODEL_NAME_RE.pattern}, got {model_name!r}")
        reserved = _RESERVED_KEYS & set(extra_metadata or {})
        if reserved:
            raise ValueError(f"extra_metadata may not set reserved keys {sorted(reserved)}")
        root = Path(results_root).resolve() if results_root is not None else RESULTS_DIR
        model_dir = root / "runs" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        existing = [int(m.group(1)) for p in model_dir.iterdir() if (m := _RUN_DIR_RE.match(p.name))]
        number = max(existing, default=0) + 1
        for _ in range(max_attempts):
            run_dir = model_dir / f"run_{number:03d}"
            try:
                run_dir.mkdir()  # atomic: fails if another process got there first
                break
            except FileExistsError:
                number += 1
        else:
            raise RunError(f"could not allocate a run directory in {model_dir}")

        (run_dir / "diagnostics").mkdir()
        (run_dir / "logs").mkdir()
        atomic_write_json(run_dir / "config.json", config)
        meta: dict[str, Any] = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "run_id": f"{model_name}/{run_dir.name}",
            "run_uid": uuid.uuid4().hex,
            "model_name": model_name,
            "run_number": number,
            "status": RunStatus.CREATED.value,
            "created_at": utc_now(),
            "started_at": None,
            "ended_at": None,
            "active_duration_seconds": 0.0,
            "sessions": [],
            "error": None,
            "git": git_info(),
            "environment": environment_info(),
            "dataset": to_jsonable(dataset_fingerprint) if dataset_fingerprint is not None else None,
            "command": [relative_to_repo(sys.argv[0])] + sys.argv[1:] if sys.argv and sys.argv[0] else None,
        }
        meta.update(to_jsonable(dict(extra_metadata or {})))
        atomic_write_json(run_dir / "metadata.json", meta)
        return cls(run_dir)

    @classmethod
    def open(cls, run_dir: str | Path) -> RunManager:
        """Reopen an existing run (e.g. to resume or to annotate it)."""
        return cls(run_dir)

    # -------------------------------------------------------------- paths
    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def history_path(self) -> Path:
        return self.run_dir / "history.jsonl"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def diagnostics_dir(self) -> Path:
        return self.run_dir / "diagnostics"

    def checkpoint_path(self, kind: CheckpointKind) -> Path:
        return checkpoint_path(self.run_dir, kind)

    # ----------------------------------------------------------- metadata
    def read_metadata(self) -> dict[str, Any]:
        return read_json(self.metadata_path)

    def read_config(self) -> dict[str, Any]:
        return read_json(self.config_path)

    def _write_metadata(self, meta: Mapping[str, Any]) -> None:
        atomic_write_json(self.metadata_path, meta)

    def update_metadata(self, **fields: Any) -> dict[str, Any]:
        """Set or merge user fields (e.g. ``results={...}``, ``notes="..."``).

        Mapping values are merged one level deep into existing mappings;
        other values replace. Lifecycle keys (status, timestamps, error...)
        are reserved and changed only through the ``mark_*`` methods.
        """
        bad = _RESERVED_KEYS & set(fields)
        if bad:
            raise RunError(f"use the mark_* methods to change {sorted(bad)}")
        meta = self.read_metadata()
        for key, value in fields.items():
            value = to_jsonable(value)
            if isinstance(value, dict) and isinstance(meta.get(key), dict):
                meta[key] = {**meta[key], **value}
            else:
                meta[key] = value
        self._write_metadata(meta)
        return meta

    @property
    def status(self) -> RunStatus:
        return RunStatus(self.read_metadata()["status"])

    def _transition(self, new: RunStatus, **changes: Any) -> dict[str, Any]:
        meta = self.read_metadata()
        old = RunStatus(meta["status"])
        if new not in _ALLOWED_TRANSITIONS[old]:
            raise RunError(f"invalid status transition {old.value} → {new.value}")
        meta["status"] = new.value
        meta.update(changes)
        self._write_metadata(meta)
        return meta

    def mark_running(self) -> None:
        """Start (or resume) a session."""
        now = utc_now()
        meta = self.read_metadata()
        changes: dict[str, Any] = {
            "sessions": meta["sessions"] + [{"started_at": now, "ended_at": None, "duration_seconds": None, "outcome": None}],
            "ended_at": None,
        }
        if meta["started_at"] is None:
            changes["started_at"] = now
        self._transition(RunStatus.RUNNING, **changes)
        self._session_start = time.monotonic()

    def _end_session(self, status: RunStatus, *, error: dict[str, Any] | None = None, summary: Mapping[str, Any] | None = None) -> None:
        meta = self.read_metadata()
        now = utc_now()
        elapsed = (time.monotonic() - self._session_start) if self._session_start is not None else None
        sessions = list(meta["sessions"])
        if sessions and sessions[-1]["ended_at"] is None:
            sessions[-1] = {**sessions[-1], "ended_at": now, "duration_seconds": elapsed, "outcome": status.value}
        changes: dict[str, Any] = {
            "sessions": sessions,
            "ended_at": now,
            "active_duration_seconds": meta["active_duration_seconds"] + (elapsed or 0.0),
            "error": error,
        }
        if summary is not None:
            changes["summary"] = to_jsonable(dict(summary))
        self._transition(status, **changes)
        self._session_start = None

    def mark_completed(self, summary: Mapping[str, Any] | None = None) -> None:
        """Finish successfully; ``summary`` (e.g. final metrics) is stored in metadata."""
        self._end_session(RunStatus.COMPLETED, summary=summary)

    def mark_failed(self, exc: BaseException) -> None:
        """Record a failure with exception type, message and traceback."""
        self._end_session(RunStatus.FAILED, error=_describe_exception(exc))

    def mark_interrupted(self, reason: str = "interrupted") -> None:
        self._end_session(RunStatus.INTERRUPTED, error={"type": "Interrupted", "message": reason, "traceback": None})

    @contextmanager
    def track(self) -> Iterator[RunManager]:
        """Mark running on entry; completed / interrupted / failed on exit.

        Exceptions are always re-raised after being recorded.
        ``KeyboardInterrupt`` maps to ``interrupted``; any other exception
        (including ``SystemExit``) to ``failed``.
        """
        self.mark_running()
        try:
            yield self
        except KeyboardInterrupt:
            self.mark_interrupted("KeyboardInterrupt")
            raise
        except BaseException as exc:
            self.mark_failed(exc)
            raise
        else:
            if self.status is RunStatus.RUNNING:  # the body may have marked completion itself
                self.mark_completed()

    # ------------------------------------------------------------ history
    def append_history(self, record: Mapping[str, Any]) -> None:
        """Append one record (e.g. ``{"step": 100, "train_nll": 2.3}``) to ``history.jsonl``."""
        append_jsonl(self.history_path, record)

    def __repr__(self) -> str:
        return f"RunManager({self.run_id!r}, dir={relative_to_repo(self.run_dir)!r})"


def _describe_exception(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }

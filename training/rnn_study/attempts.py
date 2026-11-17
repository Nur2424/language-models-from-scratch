"""Model 6 attempt store: append-only, hash-chained manifest; classification of killed runs and
retries (plan §5.5 / §8.3, D-49).

Events per attempt: one ``launched``; then ``resumed`` (zero or more); then at most one ``finished``
(written by the launching or resuming process) or ``registered`` (written by a later invocation that
reconciled an attempt whose process ended without closing it).

Statuses: final — ``completed``, ``failed_mechanical``, ``failed_diverged`` (training outcomes,
never retried), ``aborted_before_commit`` and ``unrecoverable`` (retry only by new authorisation in a
new run directory); resumable — ``interrupted``, ``crashed``, ``error``; ``integrity_blocked`` (nothing
was run). Exactly one attempt counts per key: the last with a status in :data:`COUNTED_STATUSES`.

The manifest is tamper-evident, not immutable: an edit of any line but the last breaks the chain; an
edit confined to the last line, or a consistent rewrite of the whole file, is not detectable by the
chain alone (documented limitation shared with the WaveNet log).
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training.rnn_study.common import Blocked, HashChainedLog, LedgerError, StudyLock, StudyPaths, require_lock

TRAINING_OUTCOMES = ("completed", "failed_mechanical", "failed_diverged")
FINAL_STATUSES = TRAINING_OUTCOMES + ("aborted_before_commit", "unrecoverable")
RESUMABLE_STATUSES = ("interrupted", "crashed", "error")
ATTEMPT_STATUSES = FINAL_STATUSES + RESUMABLE_STATUSES + ("integrity_blocked",)
COUNTED_STATUSES = TRAINING_OUTCOMES
EVENTS = ("launched", "resumed", "finished", "registered")
KEY_FIELDS = ("step", "arm", "seed")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def classify_run_dir(run_dir: str | Path) -> tuple[str, str]:
    """Status of a run whose process ended without closing its attempt (plan §8.3), with the reason.

    No ``checkpoint_last.pt`` → ``aborted_before_commit``. Otherwise by the run metadata: ``completed``
    → its training outcome; ``failed`` with outcome ``failed_mechanical`` → ``failed_mechanical``;
    other ``failed`` → ``error``; ``interrupted`` → ``interrupted``; ``running`` (the lock is free,
    so no live process) → ``crashed``."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    status, outcome = meta.get("status"), meta.get("outcome")
    if not (run_dir / "checkpoint_last.pt").is_file():
        if status == "failed" and outcome == "failed_mechanical":
            return "failed_mechanical", "Category A before the first commit"
        return "aborted_before_commit", f"no committed checkpoint (metadata status {status!r})"
    if status == "completed":
        if outcome not in ("completed", "failed_diverged"):
            raise Blocked(f"{run_dir}: completed run without a recognised outcome {outcome!r}")
        return outcome, "metadata"
    if status == "failed":
        return ("failed_mechanical", "Category A") if outcome == "failed_mechanical" else ("error", "exception")
    if status == "interrupted":
        return "interrupted", "KeyboardInterrupt"
    if status == "running":
        return "crashed", "process ended while the metadata said running"
    raise Blocked(f"{run_dir}: unexpected metadata status {status!r}")


class AttemptStore:
    """The study manifest. Writing methods require the study lock; every read verifies the chain."""

    def __init__(self, paths: StudyPaths, lock: StudyLock | None = None) -> None:
        self.paths, self.lock = paths, lock
        self.log = HashChainedLog(paths.manifest)

    # ------------------------------------------------------------- reading
    def _rows(self) -> list[dict[str, Any]]:
        try:
            return self.log.read()[0]
        except LedgerError as exc:
            raise Blocked(f"manifest.jsonl failed verification: {exc} (chain of custody is never repaired "
                          "automatically; human decision required)") from None

    def attempts(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for n, row in enumerate(self._rows(), start=1):
            ev, aid = row.get("event"), row.get("attempt_id")
            if ev not in EVENTS or not isinstance(aid, str):
                raise Blocked(f"manifest line {n}: invalid event {ev!r} / attempt {aid!r}")
            if ev == "launched":
                if aid in out:
                    raise Blocked(f"manifest line {n}: attempt {aid} launched twice")
                out[aid] = {**row, "status": "launched", "sessions": 1, "key": tuple(row[k] for k in KEY_FIELDS)}
                continue
            rec = out.get(aid)
            if rec is None:
                raise Blocked(f"manifest line {n}: {ev} for unknown attempt {aid}")
            if rec["status"] in FINAL_STATUSES + ("integrity_blocked",):
                raise Blocked(f"manifest line {n}: attempt {aid} already closed with {rec['status']}")
            if ev == "resumed":
                if rec["status"] not in RESUMABLE_STATUSES:
                    raise Blocked(f"manifest line {n}: resume of {aid} in status {rec['status']}")
                rec["status"] = "launched"
                rec["sessions"] += 1
            else:
                if rec["status"] != "launched":
                    raise Blocked(f"manifest line {n}: {ev} for {aid} in status {rec['status']}")
                if row.get("status") not in ATTEMPT_STATUSES:
                    raise Blocked(f"manifest line {n}: unknown status {row.get('status')!r}")
                rec["status"] = row["status"]
                rec["closed_by"] = ev
                rec["reason"] = row.get("reason")
        return out

    def open_attempts(self) -> list[str]:
        return [aid for aid, rec in self.attempts().items() if rec["status"] == "launched"]

    def counted(self, key: tuple) -> dict[str, Any] | None:
        """The one attempt that counts for ``key``: the last with a training outcome."""
        hits = [rec for rec in self.attempts().values() if rec["key"] == tuple(key) and rec["status"] in COUNTED_STATUSES]
        return hits[-1] if hits else None

    def orphans(self) -> list[str]:
        """Run directories under the study root that no attempt refers to."""
        known = {rec["run_dir"] for rec in self.attempts().values()}
        return [str(p.relative_to(self.paths.root)) for p in self.paths.all_run_dirs()
                if str(p.relative_to(self.paths.root)) not in known]

    # ------------------------------------------------------------- writing
    def _append(self, entry: Mapping[str, Any]) -> str:
        require_lock(self.lock, "writing the manifest")
        return self.log.append({**entry, "at": _now()})

    def launch(self, *, step: str, arm: str, seed: int, run_dir: Path, authorisation: str,
               retry_of: str | None = None, reason: str | None = None) -> str:
        """Record a new attempt. A retry (``retry_of``) requires the previous attempt to be closed in a
        retry-permitting status and a written reason (D-18)."""
        attempts = self.attempts()
        key = (step, arm, seed)
        if self.counted(key) is not None:
            raise Blocked(f"{key} already has a counted outcome; a new attempt needs a new decision")
        same = [a for a, r in attempts.items() if r["key"] == key]
        if same:
            if retry_of != same[-1]:
                raise Blocked(f"{key} has attempt {same[-1]}; a new attempt must name it as retry_of")
            if attempts[retry_of]["status"] not in ("aborted_before_commit", "unrecoverable", "error"):
                raise Blocked(f"attempt {retry_of} is {attempts[retry_of]['status']}: not a retry case "
                              "(resume it, or it is final)")
            if not reason:
                raise Blocked("a retry requires a written reason (D-18)")
        aid = f"{step}:{arm}:{seed}:{len(same) + 1}"
        rel = str(Path(run_dir).resolve().relative_to(self.paths.root.resolve()))
        self._append({"event": "launched", "attempt_id": aid, "step": step, "arm": arm, "seed": seed,
                      "run_dir": rel, "authorisation": authorisation, "retry_of": retry_of, "reason": reason})
        return aid

    def resumed(self, attempt_id: str, *, recovery_head: str | None) -> None:
        rec = self.attempts().get(attempt_id)
        if rec is None or rec["status"] not in RESUMABLE_STATUSES:
            raise Blocked(f"attempt {attempt_id} cannot be resumed (status {None if rec is None else rec['status']})")
        self._append({"event": "resumed", "attempt_id": attempt_id, "recovery_head": recovery_head})

    def finish(self, attempt_id: str, status: str, *, reason: str | None = None) -> None:
        if status not in ATTEMPT_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        rec = self.attempts().get(attempt_id)
        if rec is None or rec["status"] != "launched":
            raise Blocked(f"attempt {attempt_id} is not open")
        self._append({"event": "finished", "attempt_id": attempt_id, "status": status, "reason": reason})

    def reconcile(self) -> list[tuple[str, str]]:
        """Close every open attempt from its run directory (plan §8.3); called under the lock, so no
        attempt can still be running. Returns ``(attempt_id, status)`` pairs."""
        require_lock(self.lock, "reconciliation")
        out = []
        for aid in self.open_attempts():
            rec = self.attempts()[aid]
            status, reason = classify_run_dir(self.paths.root / rec["run_dir"])
            self._append({"event": "registered", "attempt_id": aid, "status": status, "reason": reason})
            out.append((aid, status))
        return out


def mark_crashed_interrupted(run_dir: Path) -> None:
    """A crashed run's metadata still says ``running``; ``RunManager`` cannot move running → running.
    Before resuming, record the end of the dead session as ``interrupted`` with the reason."""
    from utils.run_manager import RunManager, RunStatus

    run = RunManager.open(run_dir)
    if run.status is RunStatus.RUNNING:
        run.mark_interrupted("process ended without closing the session (reconciled as crashed)")

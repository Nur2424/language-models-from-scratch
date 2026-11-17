"""Fail-safe partial-record recovery before and after a resume (D-49, P7-4; plan §9.3 – §9.5).

Principles: preserve first (byte copies into ``recovery/<session>/original/``, verified by SHA-256);
validate **all** files before changing any (two phases); never silently discard — the approved
truncation (training spec §10.6) still shortens the active files, but every byte stays in the
preserved copy and every action is listed in ``recovery.jsonl`` (hash-chained); abort when unsafe
(:class:`RecoveryError` → attempt status ``unrecoverable``; no active file changed).

The preflight only removes an *incomplete last line* that lies beyond the commit point and moves
stray temporary files; the truncation of complete records beyond the commit point is done by
``train_rnn(resume=True)`` itself. The post-check confirms that every truncated file is a byte prefix
of its preserved original and compares re-generated records with the preserved ones.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from training.rnn_study.common import HashChainedLog, LedgerError
from training.rnn_study.conditions import sha256_file

RECOVERY_LOG = "recovery.jsonl"
#: Per-process observations excluded from cross-process identity comparisons: wall time, and the
#: global-RNG hashes that diagnostic points record for their within-call isolation check (a new
#: process starts from a different global RNG state; training itself uses no global RNG).
TIMING_FIELDS = ("seconds", "rng_sha256_before", "rng_sha256_after")


class RecoveryError(RuntimeError):
    """Safe recovery cannot be established; the run is ``unrecoverable`` and stays as evidence."""


def _commit_point(run_dir: Path) -> tuple[int, int]:
    from utils.checkpoint import load_checkpoint

    path = run_dir / "checkpoint_last.pt"
    if not path.is_file():
        raise RecoveryError("no committed checkpoint: nothing to resume (aborted before commit)")
    ck = load_checkpoint(path, expected_kind="resume")
    return int(ck["epoch"]), int(ck["step"])


def _split(raw: bytes) -> tuple[list[bytes], bytes]:
    """Complete lines (with their newline) and the trailing partial bytes."""
    cut = raw.rfind(b"\n") + 1
    return raw[:cut].splitlines(keepends=True), raw[cut:]


def _parse(path: Path, lines: list[bytes]) -> list[dict[str, Any]]:
    out = []
    for n, line in enumerate(lines, start=1):
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RecoveryError(f"{path.name} line {n}: malformed complete line ({exc})") from None
        if not isinstance(obj, dict):
            raise RecoveryError(f"{path.name} line {n}: not a JSON object")
        out.append(obj)
    return out


def _validators(run_dir: Path, K: int, sigma: int) -> dict[str, Callable[[list[dict[str, Any]]], None]]:
    def history(recs):
        ks = [r.get("completed_epochs") for r in recs]
        if ks[: K + 1] != list(range(K + 1)):
            raise RecoveryError(f"history.jsonl: epochs 0 … {K} are not complete and contiguous ({ks[:K + 1]})")
        if any(k is None for k in ks):
            raise RecoveryError("history.jsonl: a row without completed_epochs")

    def steps(recs):
        st = [r.get("step") for r in recs]
        if st[:sigma] != list(range(1, sigma + 1)):
            raise RecoveryError(f"steps.jsonl: steps 1 … {sigma} are not complete and contiguous")

    plan_path = run_dir / "research_diagnostics" / "plan.json"

    def points(recs):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        need = [p["point_id"] for p in plan["points"]
                if p["sigma"] <= sigma and not (p["kind"] == "epoch_end" and p["completed_epochs"] > K)]
        got = [r.get("point_id") for r in recs][: len(need)]
        if got != need:
            raise RecoveryError(f"points.jsonl: committed points {need} not all present in order ({got})")
        for r in recs[: len(need)]:
            raw = run_dir / "research_diagnostics" / "raw" / f"point_{r['point_id']:04d}.npz"
            if not raw.is_file():
                raise RecoveryError(f"raw file of committed point {r['point_id']} is missing")
            import numpy as np

            try:
                with np.load(raw, allow_pickle=False) as z:
                    _ = z.files
            except Exception as exc:  # noqa: BLE001
                raise RecoveryError(f"raw file {raw.name} does not load: {exc}") from None

    def primary(recs):
        pts = _parse(run_dir / "research_diagnostics" / "points.jsonl",
                     _split((run_dir / "research_diagnostics" / "points.jsonl").read_bytes())[0])
        if [r.get("point_id") for r in recs][: len(pts)] != [r["point_id"] for r in pts][: len(recs)]:
            raise RecoveryError("primary.jsonl ids do not follow points.jsonl")

    out = {"history.jsonl": history, "diagnostics/steps.jsonl": steps}
    if plan_path.is_file():
        out["research_diagnostics/points.jsonl"] = points
        out["research_diagnostics/primary.jsonl"] = primary
    return out


def preflight(run_dir: str | Path, session_id: str) -> dict[str, Any]:
    """Preserve, validate and (only if every file is safe) remove incomplete tails beyond the commit
    point. Returns the recovery entry; raises :class:`RecoveryError` with no active file changed, after
    recording the aborted attempt in ``recovery.jsonl`` (when that log itself is intact)."""
    run_dir = Path(run_dir)
    try:
        return _preflight(run_dir, session_id)
    except RecoveryError as exc:
        try:
            HashChainedLog(run_dir / RECOVERY_LOG).append({"phase": "preflight", "session": session_id,
                                                           "result": "aborted", "reason": str(exc)})
        except LedgerError:
            pass  # the log itself is damaged: the exception below already reports it
        raise


def _preflight(run_dir: Path, session_id: str) -> dict[str, Any]:
    log = HashChainedLog(run_dir / RECOVERY_LOG)
    try:
        log.read()
    except LedgerError as exc:
        raise RecoveryError(f"{RECOVERY_LOG} failed verification: {exc}") from None
    K, sigma = _commit_point(run_dir)
    validators = _validators(run_dir, K, sigma)
    rels = [r for r in validators if (run_dir / r).is_file()]
    preserved_dir = run_dir / "recovery" / session_id / "original"
    if preserved_dir.exists():
        raise RecoveryError(f"recovery session {session_id} already exists")
    extras = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()
                    and (p.name.endswith(".tmp") or (p.suffix == ".pt" and p.parent == run_dir)
                         or (p.parent.name == "raw" and p.suffix == ".npz"))
                    and "recovery" not in p.relative_to(run_dir).parts)
    entry: dict[str, Any] = {"phase": "preflight", "session": session_id, "commit": {"epoch": K, "step": sigma},
                             "files": {}, "moved": [], "result": None}
    # Phase 0: preserve every file that resume may change (byte copies, verified).
    for rel in rels + extras:
        src, dst = run_dir / rel, preserved_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise RecoveryError(f"preserved copy of {rel} does not match the original")
    # Phase 1: validate everything; change nothing.
    plans = {}
    for rel in rels:
        raw = (run_dir / rel).read_bytes()
        lines, tail = _split(raw)
        recs = _parse(run_dir / rel, lines)
        validators[rel](recs)
        plans[rel] = (raw, lines, tail, recs)
        entry["files"][rel] = {"original_sha256": sha256_file(run_dir / rel), "original_bytes": len(raw),
                               "complete_records": len(recs), "partial_tail_bytes": len(tail),
                               "partial_tail_sha256": hashlib.sha256(tail).hexdigest() if tail else None,
                               "preserved": str((preserved_dir / rel).relative_to(run_dir))}
    # Phase 2: apply (all files passed). Remove partial tails; move stray temporary files.
    from utils.io import atomic_write_text

    for rel, (raw, lines, tail, _recs) in plans.items():
        if tail:
            prefix = b"".join(lines)
            atomic_write_text(run_dir / rel, prefix.decode("utf-8"))
            if (run_dir / rel).read_bytes() != raw[: len(prefix)]:
                raise RecoveryError(f"{rel}: rewritten file is not the byte prefix of the original")
        entry["files"][rel]["new_sha256"] = sha256_file(run_dir / rel)
    for rel in extras:
        if rel.endswith(".tmp"):
            (run_dir / rel).unlink()  # its preserved copy remains in the recovery folder
            entry["moved"].append(rel)
    entry["result"] = "recovered"
    entry["head"] = log.append(entry)
    return entry


def _strip(rec: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in TIMING_FIELDS}


def postcheck(run_dir: str | Path, session_id: str) -> dict[str, Any]:
    """After the resumed run: each active file's records up to the old commit point equal the preserved
    originals byte for byte; re-generated records beyond it that overlap preserved originals are
    compared (timing fields excluded). A mismatch is recorded and flags the run for human review."""
    run_dir = Path(run_dir)
    log = HashChainedLog(run_dir / RECOVERY_LOG)
    entries = log.read()[0]
    pre = next((e for e in reversed(entries) if e.get("session") == session_id and e.get("phase") == "preflight"), None)
    if pre is None:
        raise RecoveryError(f"no preflight entry for session {session_id}")
    preserved_dir = run_dir / "recovery" / session_id / "original"
    result: dict[str, Any] = {"phase": "postcheck", "session": session_id, "files": {}, "review_required": False}
    for rel, info in pre["files"].items():
        orig_lines, _ = _split((preserved_dir / rel).read_bytes())
        now_lines, now_tail = _split((run_dir / rel).read_bytes()) if (run_dir / rel).is_file() else ([], b"")
        orig, now = _parse(preserved_dir / rel, orig_lines), _parse(run_dir / rel, now_lines)
        n = min(len(orig), len(now))
        mismatched = [i for i in range(n) if _strip(orig[i]) != _strip(now[i])]
        result["files"][rel] = {"compared_records": n, "mismatched_records": mismatched,
                                "incomplete_tail": bool(now_tail)}
        if mismatched or now_tail:
            result["review_required"] = True
    result["result"] = "review_required" if result["review_required"] else "consistent"
    result["head"] = log.append(result)
    return result

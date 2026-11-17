"""Model 6 study orchestration: paths, Git guard, exclusive lock, threads, study data, prerequisites.

Milestone MS7 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (D-49), §5 and §8.

Reuse (D-49, O-2): the Git guard (``disable_git``, ``GitBlockedError``, ``GIT_DISABLED``) and the
hash-chained log (``HashChainedLog``, ``LedgerError``, ``canonical_json``, ``GENESIS``) are imported
**unchanged** from the frozen ``training/wavenet_study/study_common.py`` (D-34). Nothing else of the
WaveNet study is used; the lock, attempt store and frozen conditions are Model 6 code.

Nothing here touches the test split: :func:`load_study_data` builds training and development
sequences only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.wavenet_study.study_common import (  # noqa: F401  (re-exported for the drivers)
    GENESIS,
    GIT_DISABLED,
    GitBlockedError,
    HashChainedLog,
    LedgerError,
    canonical_json,
    disable_git,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_ROOT = REPO_ROOT / "results" / "experiments" / "rnn_study"
LIFECYCLE_STEPS = ("L2_probe", "L3_benchmark", "L4_timing", "L5_gate", "L6_pilot", "L6_extension", "L7_research",
                   "L8_posthoc")


class Blocked(RuntimeError):
    """A study rule forbids the requested action. Nothing was started or written to run directories."""


def install_git_guard() -> None:
    """Make Git impossible in this process (D-15); idempotent. Called first by every driver."""
    disable_git()


@dataclass(frozen=True)
class StudyPaths:
    """Locations of the study's records (plan §8.1)."""

    root: Path

    @property
    def conditions(self) -> Path:
        return self.root / "study_conditions.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def invocations(self) -> Path:
        return self.root / "invocations.jsonl"

    @property
    def lock_file(self) -> Path:
        return self.root / ".lock"

    @property
    def probe(self) -> Path:
        return self.root / "probe" / "probe.json"

    def step_dir(self, step: str) -> Path:
        if step not in LIFECYCLE_STEPS:
            raise ValueError(f"unknown lifecycle step {step!r}")
        return self.root / step

    def runs_root(self, step: str) -> Path:
        """``RunManager.create(results_root=...)`` root: runs land in ``<step>/runs/rnn/run_NNN``."""
        return self.step_dir(step)

    def all_run_dirs(self) -> list[Path]:
        return sorted(p for step in LIFECYCLE_STEPS for p in (self.step_dir(step) / "runs" / "rnn").glob("run_*")
                      if p.is_dir())


class StudyLock:
    """Exclusive, non-blocking ``flock`` on ``<root>/.lock``; released when the process ends."""

    def __init__(self, paths: StudyPaths, command: str) -> None:
        self.paths, self.command, self._fd = paths, command, None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> StudyLock:
        import fcntl

        if self._fd is not None:
            raise RuntimeError("lock already held by this object")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.paths.lock_file, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise Blocked("another study command holds the exclusive lock") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} command {self.command}\n".encode())
        self._fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)


def lock_held_elsewhere(paths: StudyPaths) -> bool:
    """True if another open lock holds ``<root>/.lock``."""
    import fcntl

    if not paths.lock_file.is_file():
        return False
    fd = os.open(paths.lock_file, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def require_lock(lock: StudyLock | None, what: str) -> None:
    if not isinstance(lock, StudyLock) or not lock.held:
        raise RuntimeError(f"{what} requires the study's exclusive lock")


def set_threads(intra_op: int, inter_op: int) -> tuple[int, int]:
    """Set intra- and inter-op thread counts before any tensor operation (plan §8.6). Inter-op can be
    set once per process; a later or conflicting call raises :class:`Blocked`."""
    import torch

    torch.set_num_threads(int(intra_op))
    if torch.get_num_interop_threads() != int(inter_op):
        try:
            torch.set_num_interop_threads(int(inter_op))
        except RuntimeError as exc:
            raise Blocked(f"cannot set inter-op threads to {inter_op} in this process: {exc}") from exc
    got = (torch.get_num_threads(), torch.get_num_interop_threads())
    if got != (int(intra_op), int(inter_op)):
        raise Blocked(f"thread counts are {got}, expected {(intra_op, inter_op)}")
    return got


@dataclass
class StudyData:
    """Training and development data of the study (the test split is never built)."""

    vocab: Any
    train: Any
    dev: Any
    train_names: tuple[str, ...]
    dev_names: tuple[str, ...]
    split_info: Mapping[str, Any]


def load_study_data(data_config: Any) -> StudyData:
    """Names file → leakage-safe split (shared data section) → vocabulary from train → training and
    development sequences (training spec §2). Test sequences are never built."""
    from utils.data import load_names, make_next_token_sequences, split_names
    from utils.vocabulary import Vocabulary

    names = load_names(data_config.path)
    splits = split_names(names, fractions=data_config.split_fractions, seed=data_config.split_seed,
                         duplicate_policy=data_config.duplicate_policy)
    vocab = Vocabulary.from_names(splits.train)
    unknown = vocab.unknown_chars(splits.dev)  # dev coverage only (training spec §2); test is never read
    if unknown:
        raise Blocked(f"development names contain characters outside the training vocabulary: {dict(unknown)}")
    train = make_next_token_sequences(splits.train, vocab)
    dev = make_next_token_sequences(splits.dev, vocab)
    info = {"seed": data_config.split_seed, "duplicate_policy": data_config.duplicate_policy,
            "fractions": list(data_config.split_fractions), "N": int(train.lengths.numel()),
            "M": int(train.lengths.sum())}
    return StudyData(vocab, train, dev, tuple(splits.train), tuple(splits.dev), info)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Blocked(message)


def missing_keys(mapping: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [k for k in keys if k not in mapping]


# =========================================================================== session and run executor (plan §8.2 – §8.3, §9)
@dataclass
class Session:
    paths: StudyPaths
    lock: StudyLock
    conditions: Mapping[str, Any]
    values: Mapping[str, Any]
    store: Any
    reconciled: list
    invocation_id: str


#: Every driver requires the implementation acceptance (lifecycle step L1) in addition to its own decisions.
ALWAYS_REQUIRED = ("implementation_acceptance",)


def study_session(command: str, *, paths: StudyPaths, required_decisions: Sequence[str],
                  required_values: Sequence[str], repo: Path = REPO_ROOT, record: Path | None = None):
    """Context manager opening a driver session, in this order: Git guard; exclusive lock; study
    conditions; frozen conditions; decisions (P7-7, fail closed); values; reconciliation of unclosed
    attempts; orphan check; invocation record. Any failure raises before anything is run."""
    from contextlib import contextmanager

    @contextmanager
    def _session():
        import uuid

        from training.rnn_study.attempts import AttemptStore
        from training.rnn_study.conditions import IntegrityBlocked, load_conditions, require_values, verify_frozen
        from training.rnn_study.decisions import DecisionError, verify_decisions

        install_git_guard()
        with StudyLock(paths, command) as lock:
            conds = load_conditions(paths)
            verify_frozen(conds, repo)
            try:
                verify_decisions(record or repo / "PROJECT_RESEARCH_RECORD.md", conds["decisions"],
                                 tuple(ALWAYS_REQUIRED) + tuple(required_decisions), conds["record_reviewed_through"])
            except DecisionError as exc:
                raise IntegrityBlocked(f"decision check failed: {exc}") from None
            values = require_values(conds, list(required_values))
            store = AttemptStore(paths, lock)
            reconciled = store.reconcile()
            orphans = store.orphans()
            if orphans:
                raise Blocked("run directories without an attempt record (inspect them before continuing): "
                              f"{orphans}")
            inv = uuid.uuid4().hex
            log = HashChainedLog(paths.invocations)
            log.append({"event": "start", "command": command, "invocation_id": inv,
                        "manifest_head": store.log.read()[1], "reconciled": reconciled})
            yield Session(paths, lock, conds, values, store, reconciled, inv)
            log.append({"event": "end", "command": command, "invocation_id": inv, "manifest_head": store.log.read()[1]})

    return _session()


def research_experiment_config():
    """The experiment configuration of every research and pilot run: shared base with the model config,
    device exactly ``cpu`` and float32 (C8)."""
    from models.recurrent.rnn import RNNConfig
    from utils.config import default_config_paths, load_config

    return load_config(default_config_paths("rnn"), overrides=["device.device=cpu", "device.dtype=float32"],
                       model_config_cls=RNNConfig)


def execute_training_attempt(session: Session, *, step: str, arm: str, seed: int, authorisation: str,
                             build, data: StudyData, resume_attempt: str | None = None,
                             retry_of: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """Run one training attempt under the attempt store.

    ``build(run_dir) -> (model, training_config, plan_or_None, config_dict)`` constructs the run's
    objects. New attempt: ``RunManager.create`` in the step's runs root, ``launched``, ``train_rnn``,
    ``finished`` with the training outcome (Category A → ``failed_mechanical``; other exceptions →
    ``error``; KeyboardInterrupt → ``interrupted``). Resume (``resume_attempt``, P7-3): same attempt, same
    run directory and identity (checked by ``train_rnn``'s compatibility check), preceded by the recovery
    preflight and followed by the post-check; no new authorisation. Returns a summary."""
    import models.recurrent.rnn_training as rt
    from training.rnn_study.attempts import mark_crashed_interrupted
    from training.rnn_study.recovery import RecoveryError, postcheck, preflight
    from utils.run_manager import RunManager

    store = session.store
    if resume_attempt is not None:
        rec = store.attempts().get(resume_attempt)
        if rec is None or rec["status"] not in ("interrupted", "crashed", "error"):
            raise Blocked(f"attempt {resume_attempt} is not resumable ({None if rec is None else rec['status']})")
        if (rec["step"], rec["arm"], rec["seed"]) != (step, arm, seed):
            raise Blocked(f"attempt {resume_attempt} has identity {rec['key']}, not {(step, arm, seed)}")
        run_dir = session.paths.root / rec["run_dir"]
        session_id = f"{session.invocation_id[:12]}"
        try:
            entry = preflight(run_dir, session_id)
        except RecoveryError as exc:
            store.resumed(resume_attempt, recovery_head=None)  # reopen, then close as unrecoverable
            store.finish(resume_attempt, "unrecoverable", reason=str(exc))
            raise Blocked(f"recovery impossible; attempt {resume_attempt} is unrecoverable: {exc}") from None
        mark_crashed_interrupted(run_dir)
        store.resumed(resume_attempt, recovery_head=entry["head"])
        aid, run = resume_attempt, RunManager.open(run_dir)
    else:
        run = RunManager.create("rnn", config={"study_step": step, "arm": arm, "seed": seed},
                                results_root=session.paths.runs_root(step))
        aid = store.launch(step=step, arm=arm, seed=seed, run_dir=run.run_dir, authorisation=authorisation,
                           retry_of=retry_of, reason=reason)
    status, detail = "error", None
    try:
        model, training, plan, config = build(run.run_dir)  # a failure here also closes the attempt
        result = rt.train_rnn(model, data.train, data.dev, training=training, seed=seed, run=run,
                              resume=resume_attempt is not None, config=config, diagnostics=plan)
        status = result.status
    except rt.CategoryAFailure as exc:
        status, detail, result = "failed_mechanical", f"{exc.where}: {exc}", exc.result
    except KeyboardInterrupt:
        committed = (run.run_dir / "checkpoint_last.pt").is_file()
        store.finish(aid, "interrupted" if committed else "aborted_before_commit",
                     reason="KeyboardInterrupt" + ("" if committed else " before the first commit"))
        raise
    except Exception as exc:
        committed = (run.run_dir / "checkpoint_last.pt").is_file()
        store.finish(aid, "error" if committed else "aborted_before_commit",
                     reason=f"{type(exc).__name__}: {exc}" + ("" if committed else " (before the first commit)"))
        raise
    store.finish(aid, status, reason=detail)
    check = postcheck(run.run_dir, session_id) if resume_attempt is not None else None
    return {"attempt_id": aid, "run_dir": str(run.run_dir), "status": status, "result": result,
            "postcheck": None if check is None else check["result"]}

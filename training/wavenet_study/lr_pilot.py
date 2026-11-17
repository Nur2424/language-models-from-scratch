"""WaveNet study, Stage 1: four-arm learning-rate pilot (wavenet_spec.md v0.4, §16.4).

A thin driver around the *accepted* training functions ``train_mlp``,
``train_batchnorm_mlp`` and ``train_wavenet``. It changes no model, utility, config
or specification; everything specific to the pilot lives in this file.

Arms (§16.2)::

    A  Flat-8 MLP, no BatchNorm        mlp.yaml + context_length=8           21,897 params
    B  Flat-8 MLP, BatchNorm           batchnorm_mlp.yaml + context_length=8 22,097 params
    C  Hierarchical WaveNet, no BN     wavenet.yaml                          21,989 params
    D  Hierarchical WaveNet, BN        wavenet_batchnorm.yaml                22,397 params

Protocol (fixed here, not configurable): seeds 100, 101, 102; 10 epochs; constant LR;
batch 32; CPU; train and dev only (test windows are never built). Initial grid
{0.03, 0.1, 0.3, 1.0}; extension values 3.0, 10.0 (up) and 0.01, 0.003 (down), at most
two rounds, each only when ``summarize_lr_pilot`` says the boundary rule requires it.

Commands (run from the repository root)::

    python training/wavenet_study/lr_pilot.py benchmark --threads 4                  # Stage 0
    python training/wavenet_study/lr_pilot.py run --threads 4 --lrs 0.03             # Stage 1
    python training/wavenet_study/lr_pilot.py run --threads 4 --lrs 0.1 0.3 1.0      # Stage 2
    python training/wavenet_study/lr_pilot.py run --threads 4 --arms B --lrs 3 --round 1
    python training/wavenet_study/lr_pilot.py retry --threads 4 --arm A --lr 0.03 --seed 100 \\
        --reason "machine went to sleep; log shows ..."

Rules enforced here
-------------------
* **Freeze.** The first ``run`` writes ``conditions.json``: protocol, diagnostics,
  resolved per-arm configs, thread count, environment and SHA-256 of every
  experiment-defining file. Every later ``run``/``retry`` and every child process
  compares itself with it and refuses to start on any difference (amendment needed).
* **One invocation at a time.** ``run``, ``retry``, ``benchmark`` and the summary hold an
  exclusive ``flock`` on ``.runner.lock``; a child refuses to run unless that lock is
  held by its parent and its attempt is registered as launched.
* **Attempts.** Every attempt gets an id and two manifest rows (``launched`` and
  ``finished``) plus its own result file. ``completed`` and ``mechanically_unstable``
  are final. ``error``, ``crashed`` and ``interrupted`` are unresolved: never retried
  automatically; they block further ``run`` commands until an explicit ``retry`` with a
  written reason. The sweep stops at the first ``error`` or ``crashed`` attempt and
  continues after ``mechanically_unstable`` ones.
* **No git.** ``git_info`` is replaced in-process and ``subprocess.Popen`` refuses ``git``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PILOT_ROOT = REPO / "results" / "experiments" / "wavenet_study" / "lr_pilot"
MANIFEST = PILOT_ROOT / "manifest.jsonl"  # launched / finished / registered rows, one attempt id each
INVOCATIONS = PILOT_ROOT / "invocations.jsonl"  # one row per run / retry command
CONDITIONS = PILOT_ROOT / "conditions.json"  # the frozen experimental conditions
LOCK_FILE = PILOT_ROOT / ".runner.lock"
ATTEMPTS_DIR = PILOT_ROOT / "attempts"
LOGS_DIR = PILOT_ROOT / "logs"

# ------------------------------------------------------------------ protocol (§16.4)
SEEDS = (100, 101, 102)
MAX_EPOCHS = 10
INITIAL_GRID = (0.03, 0.1, 0.3, 1.0)
EXTENSION_UP = (3.0, 10.0)
EXTENSION_DOWN = (0.01, 0.003)
MAX_ROUNDS = 2
ALLOWED_LRS = set(INITIAL_GRID) | set(EXTENSION_UP) | set(EXTENSION_DOWN)
BENCHMARK_SEED = 999  # not a pilot seed (100–102) and not a final seed (0–4)
BENCHMARK_LR = 0.1
# Settings of the accepted trainers' training configs other than lr (their defaults; checked, never changed).
EXPECTED_TRAINING = {"max_epochs": MAX_EPOCHS, "batch_size": 32, "lr_decay_epoch": None, "lr_decay_factor": 0.1,
                     "eval_batch_size": 4096}

# Step diagnostics, identical for every arm (approved option B). Epoch-end diagnostics are
# part of the accepted trainers and always run.
DIAGNOSTICS = {"enabled": True, "every_n_steps": 100, "gradients": True, "parameters": True, "activations": False}

ARMS = {
    "A": {"label": "Flat-8 MLP, no BatchNorm", "config": "mlp", "params": 21_897,
          "overrides": ["model.params.context_length=8"]},
    "B": {"label": "Flat-8 MLP, BatchNorm", "config": "batchnorm_mlp", "params": 22_097,
          "overrides": ["model.params.context_length=8"]},
    "C": {"label": "Hierarchical WaveNet, no BatchNorm", "config": "wavenet", "params": 21_989, "overrides": []},
    "D": {"label": "Hierarchical WaveNet, BatchNorm", "config": "wavenet_batchnorm", "params": 22_397,
          "overrides": []},
}

FINAL_STATUSES = ("completed", "mechanically_unstable")
UNRESOLVED_STATUSES = ("error", "crashed", "interrupted")

# Files whose content defines the experiment or its interpretation; hashed instead of asking git.
CHECKSUM_FILES = [
    "models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py", "models/hierarchical/wavenet.py",
    "models/hierarchical/wavenet_spec.md", "models/feedforward/mlp_spec.md", "models/feedforward/batchnorm_mlp_spec.md",
    "UNIVERSAL_MODEL_IMPLEMENTATION_STANDARD_v1.1.md", "configs/base.yaml", "configs/models/mlp.yaml",
    "configs/models/batchnorm_mlp.yaml", "configs/models/wavenet.yaml", "configs/models/wavenet_batchnorm.yaml",
    "data/names.txt", "training/wavenet_study/lr_pilot.py", "training/wavenet_study/summarize_lr_pilot.py",
]

GIT_DISABLED = {"available": False, "reason": "git disabled by project policy"}


class Blocked(SystemExit):
    """A rule of the pilot forbids the requested action. Nothing was started."""

    def __init__(self, message: str) -> None:
        super().__init__(f"BLOCKED: {message}")


# ================================================================== no-git guard
def disable_git() -> None:
    """Make git impossible inside this process (and its children, which call this too).

    1. ``utils.run_manager.git_info`` is replaced, so ``RunManager.create`` records
       ``GIT_DISABLED`` instead of running ``git rev-parse`` / ``git status``.
    2. ``subprocess.Popen`` is replaced by a subclass that raises before starting any
       program named ``git``: a safety net should anything else try. ``subprocess.run``
       and ``check_output`` go through ``Popen``. Nothing on disk is modified.
    """
    from utils import run_manager

    run_manager.git_info = lambda *args, **kwargs: dict(GIT_DISABLED)

    if getattr(subprocess.Popen, "_git_blocked", False):
        return
    original = subprocess.Popen

    class NoGitPopen(original):
        _git_blocked = True

        def __init__(self, args, *a, **kw):
            first = args if isinstance(args, (str, bytes, os.PathLike)) else (args[0] if args else "")
            first = os.fsdecode(first).strip()
            program = os.path.basename(first.split()[0]) if first else ""
            if program in ("git", "git.exe"):
                raise PermissionError(f"git disabled by project policy (refused: {args!r})")
            super().__init__(args, *a, **kw)

    subprocess.Popen = NoGitPopen


# ================================================================== small helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_relative(path: Path) -> str:
    """Repository-relative path when inside the repository, else absolute."""
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def lr_tag(lr: float) -> str:
    return f"{lr:g}"


def run_key(arm: str, lr: float, seed: int) -> tuple[str, str, int]:
    return (arm, lr_tag(float(lr)), int(seed))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def source_checksums() -> dict:
    """SHA-256 of the experiment-defining files and of every ``utils/**/*.py``."""
    files = list(CHECKSUM_FILES) + sorted(
        p.relative_to(REPO).as_posix() for p in (REPO / "utils").rglob("*.py") if "__pycache__" not in p.parts)
    sums = {f: sha256_file(REPO / f) for f in files}
    return {"files": sums, "combined": sha256_json(sums)}


def cpu_description() -> str | None:
    """CPU model name. macOS: ``sysctl``; Linux: ``/proc/cpuinfo``."""
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5, check=True).stdout.strip() or None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001  (best-effort description only)
        return None
    return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def execution_info(threads: int) -> dict:
    """What ``RunManager``'s ``environment_info`` does not already record."""
    import torch

    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "requested_threads": threads,
        "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "cpu": cpu_description(),
        "mac_ver": platform.mac_ver()[0] or None,
        "python_implementation": platform.python_implementation(),
        "torch_parallel_info": torch.__config__.parallel_info(),
        "mps_available_but_unused": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }


def peak_rss_bytes() -> int:
    """Peak resident memory of this (child) process: mostly the PyTorch import, not the model.
    macOS reports bytes, Linux kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def write_json(path: Path, obj) -> None:
    """Atomic JSON write (temporary file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


# ================================================================== frozen conditions
def arm_config(arm: str, seed: int | None):
    """The resolved experiment config of one arm (``seed=None``: without the seed override)."""
    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet
    from utils.config import default_config_paths, load_config

    cfg_cls = {"A": mlp.MLPConfig, "B": batchnorm_mlp.BatchNormMLPConfig,
               "C": wavenet.WaveNetConfig, "D": wavenet.WaveNetConfig}[arm]
    overrides = [*ARMS[arm]["overrides"], "device.device=cpu",
                 *(f"diagnostics.{k}={str(v).lower()}" for k, v in DIAGNOSTICS.items())]
    if seed is not None:
        overrides.append(f"reproducibility.seed={seed}")
    return load_config(default_config_paths(ARMS[arm]["config"]), overrides=overrides, model_config_cls=cfg_cls)


def config_without_seed(cfg) -> dict:
    from utils.config import config_to_dict

    d = config_to_dict(cfg)
    d["reproducibility"] = {k: v for k, v in d["reproducibility"].items() if k != "seed"}
    return d


def current_conditions(threads: int) -> dict:
    """Everything that must stay identical across the whole pilot (JSON-normalised: tuples become lists)."""
    return json.loads(json.dumps({
        "protocol": {"seeds": list(SEEDS), "max_epochs": MAX_EPOCHS, "initial_grid": list(INITIAL_GRID),
                     "extension_up": list(EXTENSION_UP), "extension_down": list(EXTENSION_DOWN),
                     "max_rounds": MAX_ROUNDS, "training": EXPECTED_TRAINING},
        "diagnostics": DIAGNOSTICS,
        "arms": {a: {"label": s["label"], "expected_parameters": s["params"],
                     "config": config_without_seed(arm_config(a, None))} for a, s in ARMS.items()},
        "threads": threads,
        "environment": {
            "python": sys.version, "python_executable": sys.executable,
            "packages": {p: package_version(p) for p in ("torch", "numpy", "PyYAML")},
            "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu_description(),
        },
        "checksums": source_checksums()["files"],
    }))


def diff_paths(a, b, prefix: str = "") -> list[str]:
    """Dotted paths at which two JSON-like values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            p = f"{prefix}.{k}" if prefix else str(k)
            out += [p] if (k not in a or k not in b) else diff_paths(a[k], b[k], p)
        return out
    return [] if a == b else [prefix or "<root>"]


def load_frozen() -> dict | None:
    return json.loads(CONDITIONS.read_text()) if CONDITIONS.is_file() else None


def verify_or_freeze(threads: int, *, may_create: bool) -> dict:
    """Compare the current conditions with ``conditions.json``; create it on the first ``run``."""
    current = current_conditions(threads)
    frozen = load_frozen()
    if frozen is None:
        if MANIFEST.is_file() and read_manifest():
            raise Blocked("manifest.jsonl exists but conditions.json is missing; the pilot state is inconsistent")
        if not may_create:
            raise Blocked("the pilot has not been started (no conditions.json); use `run` first")
        frozen = {"created_at": utc_now(), "conditions": current, "conditions_sha256": sha256_json(current),
                  "note": "Frozen at the first `run`. Any difference blocks the pilot until a documented "
                          "amendment is approved."}
        write_json(CONDITIONS, frozen)
        print(f"froze the pilot conditions in {repo_relative(CONDITIONS)} ({frozen['conditions_sha256'][:12]})")
        return frozen
    diffs = diff_paths(frozen["conditions"], current)
    if diffs:
        shown = "\n  ".join(diffs[:40]) + ("\n  ..." if len(diffs) > 40 else "")
        raise Blocked(f"current conditions differ from conditions.json (documented amendment required):\n  {shown}")
    return frozen


# ================================================================== lock
@contextmanager
def pilot_lock():
    """Exclusive, non-blocking ``flock``; released automatically if the process dies."""
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Blocked("another pilot command (run / retry / benchmark / summary) is active") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} since {utc_now()}\n".encode())
        yield
    finally:
        os.close(fd)  # closing releases the lock


def lock_held_by_another_process() -> bool:
    if not LOCK_FILE.is_file():
        return False
    fd = os.open(LOCK_FILE, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


# ================================================================== manifest and attempts
def read_manifest() -> list[dict]:
    if not MANIFEST.is_file():
        return []
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def attempts_by_id() -> dict[str, dict]:
    """One merged record per attempt id, in launch order. ``status`` is ``launched`` until finished."""
    out: dict[str, dict] = {}
    for row in read_manifest():
        rec = out.setdefault(row["attempt_id"], {"status": "launched"})
        rec.update({k: v for k, v in row.items() if k != "event"})
        rec.setdefault("events", []).append(row["event"])
    return out


def attempts_by_key() -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for rec in attempts_by_id().values():
        out.setdefault(run_key(rec["arm"], rec["lr"], rec["seed"]), []).append(rec)
    return out


def register_unfinished() -> list[str]:
    """Finish manifest records of attempts whose runner died (``launched`` without ``finished``).
    The child's own result file is used when it exists; otherwise the attempt is ``crashed``."""
    registered = []
    for aid, rec in attempts_by_id().items():
        if rec["status"] != "launched":
            continue
        result_file = ATTEMPTS_DIR / f"{aid}.json"
        if result_file.is_file():
            res = json.loads(result_file.read_text())
            row = {"status": res["status"], "phase": res.get("phase"), "run_dir": res.get("run_dir"),
                   "failure": summary_of(res.get("failure")), "ended_at": res.get("ended_at")}
        else:
            row = {"status": "crashed", "phase": "unknown", "run_dir": run_dir_of(aid),
                   "failure": {"type": "no result file", "message": "the runner terminated before recording"}}
        append_jsonl(MANIFEST, {"event": "registered", "attempt_id": aid, "arm": rec["arm"], "lr": rec["lr"],
                                "seed": rec["seed"], "registered_at": utc_now(),
                                "note": "finished record written by a later invocation", **row})
        registered.append(f"{aid} → {row['status']}")
    return registered


def run_dirs_by_attempt() -> dict[str, list[str]]:
    """Run directories grouped by the attempt id their own ``metadata.json`` records
    (``None``: no readable metadata or no attempt id)."""
    out: dict[str, list[str]] = {}
    for p in sorted((PILOT_ROOT / "runs").glob("*/run_*")):
        if not p.is_dir():
            continue
        try:
            aid = json.loads((p / "metadata.json").read_text())["pilot"]["attempt_id"]
        except (OSError, ValueError, KeyError, TypeError):
            aid = None
        out.setdefault(aid, []).append(repo_relative(p))
    return out


def run_dir_of(aid: str) -> str | None:
    """The run directory created by an attempt that ended without a result file (e.g. killed)."""
    dirs = run_dirs_by_attempt().get(aid, [])
    return dirs[0] if len(dirs) == 1 else None


def unrecorded_run_dirs() -> list[str]:
    """Run directories that do not belong to exactly one known attempt."""
    known = attempts_by_id()
    bad = []
    for aid, dirs in run_dirs_by_attempt().items():
        if aid not in known or len(dirs) > 1:
            bad += dirs
    return bad


def summary_of(failure: dict | None) -> dict | None:
    return None if failure is None else {k: v for k, v in failure.items() if k != "traceback"}


def check_pilot_state() -> None:
    """Common gate for run / retry: no unrecorded directories."""
    registered = register_unfinished()
    for r in registered:
        print(f"registered unfinished attempt: {r}")
    orphans = unrecorded_run_dirs()
    if orphans:
        raise Blocked("run directories without an attempt record (inspect them before continuing):\n  "
                      + "\n  ".join(orphans))


# ================================================================== timing
TIMING_DEFINITIONS = {
    "measured_sgd_phase_seconds": "wall time from the previous epoch's end timestamp (taken right after its "
                                  "history row was written; epoch 1: the trainer call) to the end of this epoch's "
                                  "last optimizer.step(); includes step diagnostics, the progress line, the "
                                  "previous epoch's checkpoint_last.pt write and, for epoch 1, trainer set-up",
    "measured_epoch_end_seconds": "end of the last optimizer.step() to the end timestamp taken right after this "
                                  "epoch's history row was written: [BN] recalibration, full-train and dev "
                                  "evaluation, the trainer's epoch-end diagnostics (they differ by arm), the best "
                                  "checkpoint and the history write",
    "estimated_step_diagnostic_overhead_seconds": "global optimiser steps are numbered from 0 and never reset; "
                                                  "diagnostics run at steps s with s % every_n_steps == 0 (before "
                                                  "optimizer.step: gradients, parameter snapshot; after it: update "
                                                  "stats and the steps.jsonl write, which fall in step s+1's "
                                                  "interval). Estimate = sum over those s and s+1 of (step interval "
                                                  "- median of the other intervals); biased slightly upward. The "
                                                  "first step of each epoch has no interval of its own, so the "
                                                  "pre-step diagnostics of step 0 are not separated",
    "estimated_sgd_seconds_without_diagnostics": "measured_sgd_phase_seconds - estimated overhead",
    "measured_tokens_per_second_including_diagnostics": "training tokens / measured_sgd_phase_seconds",
    "estimated_tokens_per_second_without_diagnostics": "training tokens / estimated_sgd_seconds_without_diagnostics",
    "idealized_steady_state_tokens_per_second": "training tokens / (median non-diagnostic step time x steps): "
                                                "removes all jitter, an optimistic upper bound",
    "trainer_epoch_seconds": "history.jsonl 'seconds': the trainer's own epoch timer (SGD + evaluation, before "
                             "the checkpoint writes)",
    "setup_seconds": "child start to the trainer call: imports done, config, data, model, run directory",
    "training_call_seconds": "the trainer call, including the final model_final.pt write",
    "child_wall_seconds": "whole child process after imports, set-up to result file",
}


class StepTimer:
    """Timestamps from a global optimiser post-step hook and from ``run.append_history``.

    The accepted trainers record one wall-clock number per epoch; these timestamps split
    it without touching the trainers. They never affect the numerics. Step ``s+1`` also pays
    for the update statistics and the JSONL write of diagnostic step ``s``.
    """

    def __init__(self, steps_per_epoch: int, every_n_steps: int | None, num_examples: int) -> None:
        self.steps_per_epoch = steps_per_epoch
        self.every = every_n_steps
        self.num_examples = num_examples
        self.start: float | None = None
        self.step_times: list[float] = []  # t[s] = end of optimizer.step() for global step s
        self.epoch_ends: list[float] = []  # right after the history row of each epoch has been written

    def on_step(self, _optimizer, _args, _kwargs) -> None:
        self.step_times.append(time.perf_counter())

    def epochs(self) -> list[dict]:
        out, n, t, m = [], self.steps_per_epoch, self.step_times, self.num_examples
        diag = set(range(0, len(t), self.every)) if self.every else set()
        affected = diag | {s + 1 for s in diag}
        for e, end in enumerate(self.epoch_ends):
            a, b = e * n, (e + 1) * n - 1
            if b >= len(t):
                break
            start = self.epoch_ends[e - 1] if e else self.start
            dur = {s: t[s] - t[s - 1] for s in range(a + 1, b + 1)}
            plain = sorted(d for s, d in dur.items() if s not in affected)
            if not plain:
                out.append({"epoch": e + 1, "error": "no non-diagnostic steps to time"})
                continue
            med = statistics.median(plain)
            overhead = sum(d - med for s, d in dur.items() if s in affected)
            sgd = t[b] - start
            clean = sgd - overhead

            def q(p: float, plain: list[float] = plain) -> float:
                return plain[min(len(plain) - 1, int(p * len(plain)))]

            out.append({
                "epoch": e + 1,
                "measured_sgd_phase_seconds": sgd,
                "measured_epoch_end_seconds": end - t[b],
                "diagnostic_steps": sum(1 for s in range(a, b + 1) if s in diag),
                "estimated_step_diagnostic_overhead_seconds": overhead,
                "estimated_sgd_seconds_without_diagnostics": clean,
                "measured_tokens_per_second_including_diagnostics": m / sgd,
                "estimated_tokens_per_second_without_diagnostics": m / clean,
                "idealized_steady_state_tokens_per_second": m / (med * n),
                "non_diagnostic_step_seconds": {"p10": q(0.10), "median": med, "p90": q(0.90), "max": plain[-1]},
            })
        return out


# ================================================================== one run
def build_arm(arm: str, seed: int, lr: float, max_epochs: int, diagnostics: dict, threads: int, phase: list):
    """Config, data, model and trainer for one arm, via the accepted code only.
    ``phase[0]`` is updated as set-up progresses, so a failure names where it happened."""
    import torch

    from models.feedforward import batchnorm_mlp, mlp
    from models.hierarchical import wavenet
    from utils.data import check_vocab_coverage, dataset_fingerprint, load_names, make_context_windows, split_names
    from utils.device import resolve_device
    from utils.metrics import count_parameters
    from utils.reproducibility import derive_seed, make_generator, seed_everything
    from utils.vocabulary import Vocabulary

    phase[0] = "setup:threads"
    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise RuntimeError(f"requested {threads} threads, torch uses {torch.get_num_threads()}")
    module, model_cls, train_cls, train_fn = {
        "A": (mlp, mlp.MLPLanguageModel, mlp.MLPTrainingConfig, mlp.train_mlp),
        "B": (batchnorm_mlp, batchnorm_mlp.BatchNormMLPLanguageModel, batchnorm_mlp.BatchNormMLPTrainingConfig,
              batchnorm_mlp.train_batchnorm_mlp),
        "C": (wavenet, wavenet.WaveNetLanguageModel, wavenet.WaveNetTrainingConfig, wavenet.train_wavenet),
        "D": (wavenet, wavenet.WaveNetLanguageModel, wavenet.WaveNetTrainingConfig, wavenet.train_wavenet),
    }[arm]

    phase[0] = "setup:config"
    cfg = arm_config(arm, seed)
    if diagnostics != DIAGNOSTICS:  # benchmark only: the same schedule with `enabled` switched
        from dataclasses import replace
        cfg = replace(cfg, diagnostics=replace(cfg.diagnostics, **diagnostics))
    # Settings that the accepted trainers do not read are checked against their fixed behaviour (M1).
    if cfg.device.device != "cpu" or cfg.device.dtype != "float32":
        raise ValueError(f"expected device cpu / float32, config has {cfg.device.device} / {cfg.device.dtype}")
    ck = cfg.checkpoint
    if not (ck.save_best and ck.save_final and ck.resume_every_n_steps is None):
        raise ValueError("checkpoint config must match the trainers' fixed behaviour (best + final + "
                         f"epoch-boundary resume), got {ck}")

    phase[0] = "setup:data"
    names = load_names(cfg.data.path)
    splits = split_names(names, fractions=cfg.data.split_fractions, seed=cfg.data.split_seed,
                         duplicate_policy=cfg.data.duplicate_policy)
    splits.assert_disjoint()
    vocab = Vocabulary.from_names(splits.train)
    check_vocab_coverage(vocab, splits)
    k = cfg.model.params.context_length
    train = make_context_windows(splits.train, vocab, context_length=k)
    dev = make_context_windows(splits.dev, vocab, context_length=k)
    # The test split is never turned into windows here (§9, §16.4).

    phase[0] = "setup:model"
    seed_everything(seed)  # global RNG hygiene only; init and data order use explicit derived streams
    device = resolve_device(cfg.device.device)
    model = model_cls(vocab, cfg.model.params, generator=make_generator(derive_seed(seed, "init")))
    params = list(model.parameters())
    if any(p.device != device or p.dtype != torch.float32 for p in params):
        raise RuntimeError(f"model parameters are not all on {device} / float32")
    n_params = count_parameters(model).total
    if k != 8 or n_params != ARMS[arm]["params"]:
        raise RuntimeError(f"arm {arm}: expected k=8 and {ARMS[arm]['params']} parameters, got k={k}, {n_params}")
    model_name = getattr(model, "model_name", None) or module.MODEL_NAME
    if model_name != cfg.model.name:
        raise RuntimeError(f"arm {arm}: model name {model_name!r} != config {cfg.model.name!r}")
    training = train_cls(lr=lr, max_epochs=max_epochs)  # every other setting: the trainer's defaults
    expected = {**EXPECTED_TRAINING, "max_epochs": max_epochs}
    actual = {f: getattr(training, f) for f in expected}
    if actual != expected:
        raise RuntimeError(f"training defaults changed: expected {expected}, got {actual}")
    fingerprint = {**dataset_fingerprint(cfg.data.path), "splits": splits.fingerprint()}
    return {"module": module, "cfg": cfg, "model": model, "model_name": model_name, "n_params": n_params,
            "training": training, "train_fn": train_fn, "train": train, "dev": dev, "fingerprint": fingerprint,
            "device": str(params[0].device), "dtype": str(params[0].dtype)}


FAILURE_AT = re.compile(r"epoch (\d+), step (\d+)")
RECALIBRATION_AT = re.compile(r"recalibration failed at epoch (\d+)")


def describe_failure(exc: BaseException, phase: str, timer: StepTimer | None) -> dict:
    msg = str(exc)
    at, recal = FAILURE_AT.search(msg), RECALIBRATION_AT.search(msg)
    return {
        "type": type(exc).__name__, "message": msg[:2000], "phase": phase,
        "epoch": int(at.group(1)) + 1 if at else (int(recal.group(1)) + 1 if recal else None),  # 1-based
        "step": int(at.group(2)) if at else None,  # global optimiser step index (0-based)
        "completed_steps": len(timer.step_times) if timer else 0,
        "completed_epochs": len(timer.epoch_ends) if timer else 0,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
    }


def train_one(arm: str, lr: float, seed: int, *, attempt: dict, threads: int, results_root: Path,
              max_epochs: int = MAX_EPOCHS, diagnostics: dict = DIAGNOSTICS, quiet: bool = False) -> dict:
    """Set up and train one (arm, lr, seed) in this process; return its result record.

    Every exception is caught and classified, from config loading to the last checkpoint:

    * ``FloatingPointError`` during training → ``mechanically_unstable``;
    * any other exception → ``error`` with its phase (``setup:imports``, ``setup:config``, ...,
      ``training``) and traceback; a ``FloatingPointError`` outside training is an ``error`` too;
    * ``KeyboardInterrupt`` → ``interrupted`` (the caller exits with 130).
    """
    t_child = time.perf_counter()
    started_at = utc_now()
    phase = ["setup:imports"]
    status, failure, run, timer, b = "completed", None, None, None, None
    t_train = t_train_end = None
    hook = None
    try:  # everything from here on is classified: imports included
        from torch.optim.optimizer import register_optimizer_step_post_hook

        from utils.batching import EpochSampler
        from utils.config import config_to_dict
        from utils.reproducibility import derive_seed
        from utils.run_manager import RunManager

        disable_git()
        b = build_arm(arm, seed, lr, max_epochs, diagnostics, threads, phase)
        cfg, model, training = b["cfg"], b["model"], b["training"]
        experiment = config_to_dict(cfg)
        pilot = {"study": "wavenet_study", "stage": "lr_pilot", "spec": "models/hierarchical/wavenet_spec.md v0.4 §16.4",
                 **attempt, "arm": arm, "arm_label": ARMS[arm]["label"], "lr": lr, "seed": seed,
                 "max_epochs": max_epochs, "diagnostics": diagnostics, "parameters": b["n_params"],
                 "train_windows": b["train"].num_examples, "dev_windows": b["dev"].num_examples,
                 "model_device": b["device"], "model_dtype": b["dtype"]}
        phase[0] = "setup:run_directory"
        run = RunManager.create(
            b["model_name"], config={**b["module"].run_config(experiment, model.config, training), "pilot": pilot},
            results_root=results_root, dataset_fingerprint=b["fingerprint"],
            extra_metadata={"pilot": pilot, "source_checksums": source_checksums(),
                            "execution": execution_info(threads)})
        phase[0] = "setup:timer"
        sampler = EpochSampler(b["train"].num_examples, training.batch_size, seed=derive_seed(seed, "data"))
        schedule = cfg.diagnostics
        timer = StepTimer(len(sampler.batch_sizes()), schedule.every_n_steps if schedule.enabled else None,
                          b["train"].num_examples)
        original_append = run.append_history

        def append_history(record):  # instance-level wrapper: write the row, then timestamp, then progress
            original_append(record)
            timer.epoch_ends.append(time.perf_counter())  # epoch end := history row written
            if not quiet:  # progress and timing only: no learning metrics on the console (status-only policy)
                print(f"  [{arm} lr={lr_tag(lr)} seed={seed}] epoch {record['epoch'] + 1}/{max_epochs} done "
                      f"({record['seconds']:.1f}s)", flush=True)

        run.append_history = append_history
        hook = register_optimizer_step_post_hook(timer.on_step)
        phase[0] = "training"
        timer.start = t_train = time.perf_counter()
        with run.track():
            b["train_fn"](model, b["train"], b["dev"], training=training, seed=seed, run=run,
                          diagnostics=schedule, config=experiment)
        t_train_end = time.perf_counter()
    except FloatingPointError as exc:
        status = "mechanically_unstable" if phase[0] == "training" else "error"
        failure = describe_failure(exc, phase[0], timer)
    except KeyboardInterrupt as exc:
        status, failure = "interrupted", describe_failure(exc, phase[0], timer)
    except Exception as exc:  # noqa: BLE001  (software / infrastructure error, never an LR result)
        status, failure = "error", describe_failure(exc, phase[0], timer)
    finally:
        if hook is not None:
            hook.remove()
    now = time.perf_counter()
    result = {**attempt, "arm": arm, "lr": lr, "seed": seed, "status": status, "phase": phase[0],
              "failure": failure, "run_dir": repo_relative(run.run_dir) if run else None,
              "model_name": b["model_name"] if b else None, "model_device": b["device"] if b else None,
              "completed_epochs": len(timer.epoch_ends) if timer else 0,
              "completed_steps": len(timer.step_times) if timer else 0,
              "peak_rss_bytes": peak_rss_bytes(), "started_at": started_at, "ended_at": utc_now()}
    if run is not None:
        try:
            timing = {"definitions": TIMING_DEFINITIONS,
                      "setup_seconds": (t_train or now) - t_child,
                      "training_call_seconds": ((t_train_end or now) - t_train) if t_train else None,
                      "child_wall_seconds": now - t_child, "epochs": timer.epochs() if timer else [],
                      "raw_perf_counter": {"training_start": t_train, "epoch_ends": timer.epoch_ends if timer else [],
                                           "steps_recorded": len(timer.step_times) if timer else 0}}
            write_json(run.run_dir / "timing.json", timing)
            write_json(run.run_dir / "pilot_result.json", result)
        except Exception as exc:  # noqa: BLE001  (never lose the primary result over a timing record)
            result["record_error"] = f"{type(exc).__name__}: {exc}"
    return result


# ================================================================== child (`one`)
def child_main(args) -> None:
    """Runs only as a registered attempt of an active `run` / `retry` invocation."""
    disable_git()
    # Not launched by an active `run` / `retry`: refuse, and leave no record (nothing waits for one).
    rec = attempts_by_id().get(args.attempt_id)
    if not lock_held_by_another_process() or rec is None or rec["status"] != "launched" \
            or run_key(rec["arm"], rec["lr"], rec["seed"]) != run_key(args.arm, args.lr, args.seed):
        raise Blocked(f"`one` runs only as attempt {args.attempt_id!r} registered as launched by an active "
                      "`run` / `retry` invocation")
    result_file = ATTEMPTS_DIR / f"{args.attempt_id}.json"
    attempt = {"attempt_id": args.attempt_id, "round": args.round}
    try:
        attempt.update({k: rec.get(k) for k in ("retry_of", "retry_reason")})
        frozen = verify_or_freeze(args.threads, may_create=False)  # Blocked is a SystemExit, handled below
        attempt["conditions_sha256"] = frozen["conditions_sha256"]
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        res = {**attempt, "arm": args.arm, "lr": args.lr, "seed": args.seed, "status": "error",
               "phase": "setup:consistency", "failure": describe_failure(exc, "setup:consistency", None),
               "run_dir": None, "started_at": utc_now(), "ended_at": utc_now()}
        write_json(result_file, res)
        sys.exit(1)
    res = train_one(args.arm, args.lr, args.seed, attempt=attempt, threads=args.threads, results_root=PILOT_ROOT)
    write_json(result_file, res)
    sys.exit({"completed": 0, "mechanically_unstable": 0, "interrupted": 130}.get(res["status"], 1))


# ================================================================== parent
def launch_attempt(arm: str, lr: float, seed: int, round_: int, threads: int, *, retry_of: str | None = None,
                   retry_reason: str | None = None) -> dict:
    """Register, run and record one attempt in a child process. Returns the merged record."""
    n = len(attempts_by_key().get(run_key(arm, lr, seed), [])) + 1
    aid = f"{arm}-lr{lr_tag(lr)}-s{seed}-a{n}"
    started = utc_now()
    append_jsonl(MANIFEST, {"event": "launched", "attempt_id": aid, "arm": arm, "lr": lr, "seed": seed,
                            "round": round_, "threads": threads, "retry_of": retry_of, "retry_reason": retry_reason,
                            "started_at": started})
    log_file = LOGS_DIR / f"{aid}.log"
    result_file = ATTEMPTS_DIR / f"{aid}.json"
    cmd = [sys.executable, str(Path(__file__).resolve()), "one", "--attempt-id", aid, "--arm", arm, "--lr", repr(lr),
           "--seed", str(seed), "--round", str(round_), "--threads", str(threads)]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    t0 = time.perf_counter()
    interrupted = False
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            returncode = proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            returncode = proc.wait()
    row = {"event": "finished", "attempt_id": aid, "arm": arm, "lr": lr, "seed": seed, "returncode": returncode,
           "ended_at": utc_now(), "parent_wall_seconds": time.perf_counter() - t0, "log": repo_relative(log_file)}
    if result_file.is_file():
        res = json.loads(result_file.read_text())
        row.update(status=res["status"], phase=res.get("phase"), run_dir=res.get("run_dir"),
                   failure=summary_of(res.get("failure")))
    else:  # the child ended without writing its record: killed, segfault, interpreter failure
        row.update(status="interrupted" if interrupted else "crashed", phase="unknown", run_dir=run_dir_of(aid),
                   failure={"type": "no result file", "returncode": returncode})
    append_jsonl(MANIFEST, row)
    print(f"  → {aid}: {row['status']}" + (f" ({row['phase']})" if row["status"] != "completed" else ""), flush=True)
    return row


def unique(values: list) -> list:
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def validate_extension(arms: list[str], lrs: list[float], round_: int) -> None:
    """Extension rounds only as the boundary rule requires (M2): the summary's decision for each
    arm must be ``extension_required`` with exactly this LR and round."""
    from training.wavenet_study.summarize_lr_pilot import evaluate_pilot

    if len(lrs) != 1:
        raise Blocked("an extension round evaluates exactly one learning rate per arm")
    summary = evaluate_pilot()
    if summary["blocking"]:
        raise Blocked("extension refused, the pilot is not in a consistent state:\n  " + "\n  ".join(summary["blocking"]))
    for arm in arms:
        sel = summary["arms"][arm]["selection"]
        if not (sel.get("decision") == "extension_required" and sel.get("next_lr") == lrs[0]
                and sel.get("next_round") == round_):
            raise Blocked(f"arm {arm}: the boundary rule does not call for lr {lr_tag(lrs[0])} in round {round_} "
                          f"(current decision: {sel.get('decision')}, next lr {sel.get('next_lr')}, "
                          f"next round {sel.get('next_round')})")


def sweep(arms: list[str], lrs: list[float], round_: int, threads: int) -> None:
    disable_git()
    arms, lrs = unique(arms), unique(lrs)  # C3: repeated arguments never queue a run twice
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise Blocked(f"unknown arms {bad}")
    if not 0 <= round_ <= MAX_ROUNDS:
        raise Blocked(f"--round must be 0..{MAX_ROUNDS}")
    allowed = set(INITIAL_GRID) if round_ == 0 else set(EXTENSION_UP) | set(EXTENSION_DOWN)
    if any(lr not in allowed for lr in lrs):
        raise Blocked(f"learning rates {lrs} are not approved for round {round_}; allowed: {sorted(allowed)}")
    with pilot_lock():
        verify_or_freeze(threads, may_create=True)
        check_pilot_state()
        by_key = attempts_by_key()
        unresolved = [recs[-1]["attempt_id"] + f" ({recs[-1]['status']})" for recs in by_key.values()
                      if recs[-1]["status"] in UNRESOLVED_STATUSES]
        if unresolved:
            raise Blocked("unresolved attempts must be inspected and explicitly retried first:\n  "
                          + "\n  ".join(unresolved))
        if round_ >= 1:
            validate_extension(arms, lrs, round_)
        append_jsonl(INVOCATIONS, {"command": "run", "argv": sys.argv, "arms": arms, "lrs": lrs, "round": round_,
                                   "threads": threads, "started_at": utc_now()})
        jobs = [(a, lr, s) for a in arms for lr in lrs for s in SEEDS if run_key(a, lr, s) not in by_key]
        print(f"{len(jobs)} runs to do ({len(arms) * len(lrs) * len(SEEDS) - len(jobs)} already final); "
              f"threads={threads}; results in {repo_relative(PILOT_ROOT)}", flush=True)
        for i, (arm, lr, seed) in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] arm {arm} lr {lr_tag(lr)} seed {seed} (round {round_})", flush=True)
            row = launch_attempt(arm, lr, seed, round_, threads)
            if row["status"] == "interrupted":
                print("sweep interrupted; inspect the attempt, then `retry` it explicitly", flush=True)
                raise SystemExit(130)
            if row["status"] in ("error", "crashed"):
                print("sweep stopped after a software / infrastructure failure; see the attempt log", flush=True)
                raise SystemExit(3)


def retry(arm: str, lr: float, seed: int, reason: str, threads: int) -> None:
    disable_git()
    if not reason or len(reason.strip()) < 10:
        raise Blocked("a retry needs a written reason (at least 10 characters)")
    with pilot_lock():
        frozen = verify_or_freeze(threads, may_create=False)
        check_pilot_state()
        recs = attempts_by_key().get(run_key(arm, lr, seed))
        if not recs:
            raise Blocked(f"no attempt exists for arm {arm} lr {lr_tag(lr)} seed {seed}; use `run`")
        last = recs[-1]
        if last["status"] not in UNRESOLVED_STATUSES:
            raise Blocked(f"{last['attempt_id']} is {last['status']}; only error / crashed / interrupted "
                          "attempts can be retried (completed and mechanically unstable results are final)")
        result_file = ATTEMPTS_DIR / f"{last['attempt_id']}.json"
        recorded = json.loads(result_file.read_text()).get("conditions_sha256") if result_file.is_file() else None
        if recorded is not None and recorded != frozen["conditions_sha256"]:
            raise Blocked(f"{last['attempt_id']} ran under different conditions ({recorded[:12]}); retry refused")
        append_jsonl(INVOCATIONS, {"command": "retry", "argv": sys.argv, "attempt": last["attempt_id"],
                                   "reason": reason, "threads": threads, "started_at": utc_now()})
        print(f"retrying {last['attempt_id']} ({last['status']}): {reason}", flush=True)
        row = launch_attempt(arm, lr, seed, last.get("round") or 0, threads, retry_of=last["attempt_id"],
                             retry_reason=reason)
        if row["status"] not in FINAL_STATUSES:
            raise SystemExit(130 if row["status"] == "interrupted" else 3)


# ================================================================== benchmark
def benchmark(threads: int) -> None:
    """One epoch per arm, with and without the pilot's step diagnostics (seed 999, lr 0.1).

    Timing only: nothing is written to the pilot manifest or conditions, run directories
    are temporary and deleted, and no NLL is reported (this is not pilot evidence).
    """
    disable_git()
    rows = []
    with pilot_lock(), tempfile.TemporaryDirectory(prefix="lr_pilot_benchmark_") as tmp:
        for arm in ARMS:
            for diag_on in (False, True):
                res = train_one(arm, BENCHMARK_LR, BENCHMARK_SEED, attempt={"attempt_id": "benchmark", "round": -1},
                                threads=threads, results_root=Path(tmp), max_epochs=1,
                                diagnostics=dict(DIAGNOSTICS, enabled=diag_on), quiet=True)
                if res["status"] != "completed":
                    raise SystemExit(f"benchmark run failed: {res['status']} {summary_of(res['failure'])}")
                run_dir = Path(res["run_dir"])  # absolute: the temporary directory is outside the repository
                timing = json.loads((run_dir / "timing.json").read_text())["epochs"][0]
                history = json.loads((run_dir / "history.jsonl").read_text().splitlines()[0])
                rows.append({"arm": arm, "diagnostics": diag_on, "trainer_epoch_seconds": history["seconds"],
                             **{k: v for k, v in timing.items() if k != "epoch"},
                             "peak_rss_mb": res["peak_rss_bytes"] / 2**20})
                print(f"{arm} diagnostics={'on ' if diag_on else 'off'}  trainer epoch {history['seconds']:6.1f}s  "
                      f"SGD phase {timing['measured_sgd_phase_seconds']:6.1f}s  "
                      f"epoch end {timing['measured_epoch_end_seconds']:5.1f}s  "
                      f"est. diag overhead {timing['estimated_step_diagnostic_overhead_seconds']:5.2f}s", flush=True)
    per_arm = {r["arm"]: r["trainer_epoch_seconds"] for r in rows if r["diagnostics"]}
    round0 = sum(per_arm.values()) * len(INITIAL_GRID) * len(SEEDS) * MAX_EPOCHS
    ext = max(per_arm.values()) * len(SEEDS) * MAX_EPOCHS
    out = {"created_at": utc_now(), "threads": threads, "seed": BENCHMARK_SEED, "lr": BENCHMARK_LR,
           "diagnostics_policy": DIAGNOSTICS, "execution": execution_info(threads), "timing_definitions":
           TIMING_DEFINITIONS, "source_checksums_combined": source_checksums()["combined"], "rows": rows,
           "estimate": {"round0_seconds": round0, "per_extension_candidate_seconds_max": ext,
                        "worst_case_all_arms_two_rounds_seconds": round0 + 2 * len(ARMS) * ext,
                        "note": "trainer epoch time x 10 epochs x 3 seeds; excludes per-run set-up "
                                "(imports, data, model; a few seconds)"}}
    path = PILOT_ROOT / "benchmark" / f"benchmark_threads{threads}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    write_json(path, out)
    print(f"\nround 0 (48 runs): ~{round0 / 3600:.2f} h; each extension candidate <= {ext / 60:.1f} min\n"
          f"written to {repo_relative(path)}")


# ================================================================== CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the not-yet-attempted (arm, lr, seed) combinations, sequentially")
    r.add_argument("--arms", nargs="+", default=list(ARMS))
    r.add_argument("--lrs", nargs="+", type=float, required=True)
    r.add_argument("--round", type=int, default=0, help="0 = initial grid; 1, 2 = extension rounds")
    r.add_argument("--threads", type=int, required=True)
    t = sub.add_parser("retry", help="explicitly retry one unresolved (error / crashed / interrupted) run")
    t.add_argument("--arm", required=True, choices=list(ARMS))
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    t.add_argument("--reason", required=True)
    t.add_argument("--threads", type=int, required=True)
    o = sub.add_parser("one", help="internal: one registered attempt in this process")
    o.add_argument("--attempt-id", required=True)
    o.add_argument("--arm", required=True, choices=list(ARMS))
    o.add_argument("--lr", type=float, required=True)
    o.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    o.add_argument("--round", type=int, required=True)
    o.add_argument("--threads", type=int, required=True)
    bm = sub.add_parser("benchmark", help="time one epoch per arm with diagnostics off/on (no pilot data)")
    bm.add_argument("--threads", type=int, required=True)
    args = p.parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    if args.command == "run":
        sweep(args.arms, args.lrs, args.round, args.threads)
    elif args.command == "retry":
        retry(args.arm, args.lr, args.seed, args.reason, args.threads)
    elif args.command == "one":
        child_main(args)
    else:
        benchmark(args.threads)


if __name__ == "__main__":
    main()

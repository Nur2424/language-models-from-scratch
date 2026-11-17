"""Frozen conditions and study conditions (plan §5.4 / §8.1, D-49).

``study_conditions.json`` (written by a human-authorised step, MS9 / later decisions) holds:

* ``frozen``: SHA-256 of every code and specification file the study depends on, and the environment;
* ``decisions``: references ``{"id": "D-NN", "row_sha256": …}`` per key (verified by ``decisions.py``);
* ``values``: values fixed by decisions (probe hash, threads, g and s², η, E, seeds …), each with the
  id of the decision that fixed it;
* ``record_reviewed_through``: the newest decision considered.

Every driver recomputes the frozen conditions and refuses (``integrity_blocked``) on any difference.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.rnn_study.common import REPO_ROOT, Blocked, StudyPaths

CONDITIONS_SCHEMA = "m6_study_conditions/1"
#: Files whose bytes the study depends on (plan §8.1.1). Paths are repository-relative.
FROZEN_FILES: tuple[str, ...] = (
    "models/recurrent/rnn.py", "models/recurrent/rnn_diagnostics.py", "models/recurrent/rnn_training.py",
    "configs/models/rnn.yaml", "configs/base.yaml",
    "models/recurrent/rnn_spec.md", "models/recurrent/rnn_training_spec.md",
    "models/recurrent/rnn_diagnostics_spec.md", "models/recurrent/rnn_integration_spec.md",
    "training/wavenet_study/study_common.py",
    "utils/__init__.py", "utils/batching.py", "utils/checkpoint.py", "utils/config.py", "utils/data.py",
    "utils/device.py", "utils/diagnostics/__init__.py", "utils/diagnostics/gradients.py",
    "utils/diagnostics/schedule.py", "utils/generation.py", "utils/io.py", "utils/metrics.py", "utils/paths.py",
    "utils/reproducibility.py", "utils/run_manager.py", "utils/vocabulary.py",
)


class IntegrityBlocked(Blocked):
    """Frozen conditions or study conditions do not match (attempt status ``integrity_blocked``)."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_description() -> str | None:
    """Best-effort CPU name (Linux ``/proc/cpuinfo``; macOS via ``platform``). Never starts Git."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def environment() -> dict[str, Any]:
    import numpy
    import torch

    return {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": numpy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu_description()}


def study_code_files(repo: Path = REPO_ROOT) -> list[str]:
    return sorted(str(p.relative_to(repo)) for p in (repo / "training" / "rnn_study").glob("*.py"))


def compute_frozen_conditions(repo: Path = REPO_ROOT, extra_files: Sequence[str] = ()) -> dict[str, Any]:
    files = list(FROZEN_FILES) + study_code_files(repo) + list(extra_files)
    missing = [f for f in files if not (repo / f).is_file()]
    if missing:
        raise IntegrityBlocked(f"frozen files missing: {missing}")
    return {"files": {f: sha256_file(repo / f) for f in sorted(set(files))}, "environment": environment()}


def load_conditions(paths: StudyPaths) -> dict[str, Any]:
    if not paths.conditions.is_file():
        raise Blocked(f"{paths.conditions} does not exist (prepared by an authorised readiness step, MS9)")
    data = json.loads(paths.conditions.read_text(encoding="utf-8"))
    if data.get("schema") != CONDITIONS_SCHEMA:
        raise IntegrityBlocked(f"study conditions schema {data.get('schema')!r}, expected {CONDITIONS_SCHEMA!r}")
    for key in ("frozen", "decisions", "values", "record_reviewed_through"):
        if key not in data:
            raise IntegrityBlocked(f"study conditions lack {key!r}")
    return data


def verify_frozen(conditions: Mapping[str, Any], repo: Path = REPO_ROOT) -> None:
    """Recompute the frozen conditions and refuse on any difference (plan §5.4)."""
    recorded = conditions["frozen"]
    current = compute_frozen_conditions(repo, extra_files=[f for f in recorded["files"] if f not in FROZEN_FILES
                                                           and not f.startswith("training/rnn_study/")])
    diffs = [f for f in sorted(set(recorded["files"]) | set(current["files"]))
             if recorded["files"].get(f) != current["files"].get(f)]
    env = [k for k in recorded["environment"] if recorded["environment"][k] != current["environment"].get(k)]
    if diffs or env:
        raise IntegrityBlocked(f"frozen conditions differ: files {diffs}; environment {env}")


def require_values(conditions: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """Values fixed by decisions; each entry is ``{"value": …, "decision": "D-NN"}``."""
    values = conditions["values"]
    missing = [k for k in keys if k not in values]
    if missing:
        raise Blocked(f"values not yet fixed by a recorded decision: {missing}")
    bad = [k for k in keys if not isinstance(values[k], Mapping) or "value" not in values[k] or "decision" not in values[k]]
    if bad:
        raise IntegrityBlocked(f"malformed value entries (need 'value' and 'decision'): {bad}")
    return {k: values[k]["value"] for k in keys}

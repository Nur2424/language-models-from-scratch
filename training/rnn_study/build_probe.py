"""L2: construct the training probe (diag §3.1) once, under its own authorisation.

Writes ``probe/probe.json`` and prints its SHA-256, N_ℓ, n_ℓ and the dev count of F₈ (t ≥ 10) for the
recording decision. Refuses if the probe exists. Milestone MS8 (plan §5.2, §8.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.rnn_study.common import (
    DEFAULT_STUDY_ROOT,
    Blocked,
    Session,
    StudyData,
    StudyPaths,
    install_git_guard,
    load_study_data,
    research_experiment_config,
    study_session,
)

COMMAND = "build_probe"
REQUIRED_DECISIONS = ("authorisation_L2",)
REQUIRED_VALUES: tuple[str, ...] = ()


def f8_count(dev) -> int:
    """Dev targets at positions t ≥ 10 (the fixed C1 set F₈, diag §4.5)."""
    return int(sum(max(0, int(n) - 9) for n in dev.lengths.tolist()))


def run(session: Session, data: StudyData) -> dict[str, Any]:
    from models.recurrent.rnn_diagnostics import build_probe

    if session.paths.probe.exists():
        raise Blocked(f"{session.paths.probe} exists: the probe is built once (a rebuild needs a decision)")
    probe = build_probe(data.train, data.train_names, data.split_info, session.paths.probe)
    return {"probe_sha256": probe.sha256, "population_by_stratum": list(probe.population_by_stratum),
            "sample_by_stratum": list(probe.sample_by_stratum), "f8_dev_targets": f8_count(data.dev),
            "path": str(session.paths.probe)}


def main(argv=None) -> int:
    """Command line. Exit 0 on success, 2 when blocked (nothing was run)."""
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(DEFAULT_STUDY_ROOT), help="study root")
    args = ap.parse_args(argv)
    install_git_guard()
    try:
        with study_session(COMMAND, paths=StudyPaths(Path(args.root)), required_decisions=REQUIRED_DECISIONS,
                           required_values=REQUIRED_VALUES) as session:
            data = load_study_data(research_experiment_config().data)
            out = run(session, data)
    except Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

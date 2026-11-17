"""L8: post-hoc best-epoch spectral measurement for each completed research run whose best epoch is
neither 1 nor E (training spec §10.7). Read-only on checkpoints and training records. Milestone MS8.
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

COMMAND = "posthoc_spectral"
REQUIRED_DECISIONS = ("authorisation_L8",)
REQUIRED_VALUES = ("probe_sha256",)


def run(session: Session, data: StudyData) -> dict[str, Any]:
    from models.recurrent import rnn_diagnostics as dg

    probe = dg.load_probe(session.paths.probe)
    if probe.sha256 != session.values["probe_sha256"]:
        raise Blocked("probe SHA-256 differs from the recorded value")
    out = []
    for aid, rec in session.store.attempts().items():
        if rec["step"] != "L7_research" or rec["status"] != "completed":
            continue
        run_dir = session.paths.root / rec["run_dir"]
        if any(r["kind"] == "posthoc_best" for r in dg.read_jsonl_complete(run_dir / dg.RESEARCH_DIAGNOSTICS_DIR / "points.jsonl")):
            continue
        out.append({"attempt": aid, "point_id": dg.posthoc_best_spectral(run_dir, probe, data.train)})
    return {"measured": out}


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

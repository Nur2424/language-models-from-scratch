"""L5: the initial-statistics procedure (arch spec §7.2 as amended by D-47): seeds 1000 – 1009
exclusively, g = 1, s² = 0.5, no optimiser step; writes the report under ``L5_gate/``. The gate decision
(retain or change g, s²) is a recorded human decision, never made here. Milestone MS8.
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

COMMAND = "init_gate"
REQUIRED_DECISIONS = ("authorisation_L5",)
REQUIRED_VALUES = ("probe_sha256", "threads")


def run(session: Session, data: StudyData, seeds=None) -> dict[str, Any]:
    from models.recurrent import rnn_diagnostics as dg
    from models.recurrent.rnn import RNNConfig
    from training.rnn_study.common import set_threads

    seeds = dg.INIT_GATE_SEEDS if seeds is None else tuple(seeds)
    if seeds != dg.INIT_GATE_SEEDS:
        raise Blocked(f"the gate uses seeds 1000 – 1009 exclusively (D-47), got {seeds}")
    set_threads(*session.values["threads"])
    probe = dg.load_probe(session.paths.probe)
    if probe.sha256 != session.values["probe_sha256"]:
        raise Blocked("probe SHA-256 differs from the recorded value")
    report = dg.initial_statistics(data.vocab, RNNConfig(), probe, data.train, data.train_names, data.split_info,
                                   seeds, out_dir=session.paths.step_dir("L5_gate"))
    return {"report": str(report.path), "sha256": report.sha256, "seeds": list(report.seeds)}


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
            from training.rnn_study.common import set_threads

            set_threads(*session.values["threads"])  # plan §8.6: before any tensor operation
            data = load_study_data(research_experiment_config().data)
            out = run(session, data)
    except Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

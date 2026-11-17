"""L4: one timing measurement of each mandatory diagnostic on the untrained model, seed 1000, at the
decided thread counts (M6-DGN13). Feasibility information only; nothing changes automatically.
Milestone MS8.
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

COMMAND = "diagnostic_timing"
REQUIRED_DECISIONS = ("authorisation_L4",)
REQUIRED_VALUES = ("probe_sha256", "threads")
TIMING_SEED = 1000


def run(session: Session, data: StudyData) -> dict[str, Any]:
    import json
    import time

    from models.recurrent import rnn_diagnostics as dg
    from models.recurrent.rnn import RNNConfig, RNNLanguageModel
    from training.rnn_study.common import set_threads
    from utils.reproducibility import derive_seed, make_generator

    set_threads(*session.values["threads"])
    out_path = session.paths.step_dir("L4_timing") / "timing.json"
    if out_path.exists():
        raise Blocked(f"{out_path} exists")
    probe = dg.load_probe(session.paths.probe)
    if probe.sha256 != session.values["probe_sha256"]:
        raise Blocked("probe SHA-256 differs from the recorded value")
    dg.validate_probe(probe, data.train, data.train_names, data.split_info)
    model = RNNLanguageModel(data.vocab, RNNConfig(), generator=make_generator(derive_seed(TIMING_SEED, "init")))
    copy = dg.float64_copy(model)
    jobs = {"H1": lambda: dg.h1_statistics(copy, probe, data.train),
            "G1_frobenius": lambda: dg.g1_products(copy, probe, data.train, spectral=False),
            "G1_spectral": lambda: dg.g1_products(copy, probe, data.train, spectral=True),
            "G2": lambda: dg.g2_attenuation(copy, probe, data.train),
            "G3": lambda: dg.g3_decomposition(copy, probe, data.train),
            "C1": lambda: dg.c1_restart(copy, data.dev)}
    rows = {}
    for name, job in jobs.items():
        t0 = time.perf_counter()
        job()
        rows[name] = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seconds": rows, "seed": TIMING_SEED, "threads": list(session.values["threads"]),
                                   "note": "untrained model (G2 and G3 are degenerate at V = 0); timing only"},
                                   indent=1) + "\n")
    return {"seconds": rows, "path": str(out_path)}


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

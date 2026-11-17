"""L3: timing-only thread benchmark (training spec §11): one epoch plus evaluation, seed 999,
η = 0.1, no research diagnostics, intra-op threads 1, 2, 4, 8 (inter-op at the process default,
recorded). Parameters are discarded; the thread count is fixed by a recorded decision, never here.
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

COMMAND = "thread_benchmark"
REQUIRED_DECISIONS = ("authorisation_L3",)
REQUIRED_VALUES = ("probe_sha256",)
INTRA_OP_COUNTS = (1, 2, 4, 8)
BENCHMARK_SEED, BENCHMARK_LR = 999, 0.1


def run(session: Session, data: StudyData, counts=INTRA_OP_COUNTS) -> dict[str, Any]:
    import json
    import time

    import torch

    import models.recurrent.rnn_training as rt
    from models.recurrent.rnn import RNNConfig, RNNLanguageModel
    from utils.reproducibility import derive_seed, make_generator

    out_path = session.paths.step_dir("L3_benchmark") / "benchmark.json"
    if out_path.exists():
        raise Blocked(f"{out_path} exists")
    training = rt.RNNTrainingConfig(lr=BENCHMARK_LR, max_epochs=1)
    rt.check_protocol_counts(data.train, data.dev, training)
    original = torch.get_num_threads()
    inter = torch.get_num_interop_threads()
    rows = []
    try:
        for k in counts:
            torch.set_num_threads(k)
            model = RNNLanguageModel(data.vocab, RNNConfig(), generator=make_generator(derive_seed(BENCHMARK_SEED, "init")))
            rt.validate_research_run(research_experiment_config(), model, training, expected_threads=(k, inter))
            t0 = time.perf_counter()
            rt.train_rnn(model, data.train, data.dev, training=training, seed=BENCHMARK_SEED)
            rows.append({"intra_op": k, "inter_op": inter, "seconds": time.perf_counter() - t0})
    finally:
        torch.set_num_threads(original)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "note": "timing only; parameters discarded; no decision made"},
                                   indent=1) + "\n")
    return {"rows": rows, "path": str(out_path)}


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

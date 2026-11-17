"""L6: learning-rate pilot P (training spec §7.3): the authorised grid × seeds 100 – 102, 30 epochs,
constant η, no research diagnostics. Runs only after the gate decision on g and s² (D-47). Each run is
one attempt; Category A or D outcomes are final (the candidate becomes ineligible in the summary).
``--resume ATTEMPT`` continues a crashed or interrupted attempt without new authorisation (P7-3).
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

COMMAND = "lr_pilot"
REQUIRED_DECISIONS = ("gate_decision", "authorisation_L6")
REQUIRED_VALUES = ("probe_sha256", "threads", "g", "s2", "lr_grid")
PILOT_SEEDS = (100, 101, 102)
PILOT_EPOCHS = 30


def _builder(session: Session, data: StudyData, lr: float, max_epochs: int, with_plan: bool):
    import models.recurrent.rnn_training as rt
    from models.recurrent import rnn_diagnostics as dg
    from models.recurrent.rnn import RNNConfig, RNNLanguageModel
    from utils.config import config_to_dict
    from utils.reproducibility import derive_seed, make_generator

    v = session.values
    model_cfg = RNNConfig(recurrent_gain=float(v["g"]), input_init_variance=float(v["s2"]))
    training = rt.RNNTrainingConfig(lr=float(lr), max_epochs=int(max_epochs))
    rt.check_protocol_counts(data.train, data.dev, training)
    experiment = research_experiment_config()

    def build_for(seed):
        def build(run_dir):
            model = RNNLanguageModel(data.vocab, model_cfg, generator=make_generator(derive_seed(seed, "init")))
            rt.validate_research_run(experiment, model, training, expected_threads=tuple(v["threads"]))
            plan = None
            if with_plan:
                probe = dg.load_probe(session.paths.probe)
                if probe.sha256 != v["probe_sha256"]:
                    raise Blocked("probe SHA-256 differs from the recorded value")
                plan = dg.RNNDiagnosticsPlan(probe=probe, train=data.train, train_names=data.train_names, dev=data.dev,
                                             steps_per_epoch=-(-int(data.train.lengths.numel()) // training.batch_size),
                                             max_epochs=training.max_epochs, recurrent_gain=model_cfg.recurrent_gain,
                                             out_dir=Path(run_dir) / dg.RESEARCH_DIAGNOSTICS_DIR)
            return model, training, plan, {"experiment": config_to_dict(experiment)}
        return build

    return build_for


def run(session: Session, data: StudyData, resume: str | None = None, epochs: int = PILOT_EPOCHS) -> dict[str, Any]:
    from training.rnn_study.common import execute_training_attempt, set_threads

    set_threads(*session.values["threads"])
    auth = session.conditions["decisions"]["authorisation_L6"]["id"]
    out = []
    for lr in session.values["lr_grid"]:
        build_for = _builder(session, data, lr, epochs, with_plan=False)
        for seed in PILOT_SEEDS:
            key = ("L6_pilot", repr(float(lr)), seed)
            rec = session.store.counted(key)
            open_ = [a for a, r in session.store.attempts().items() if r["key"] == key]
            if resume is not None and (not open_ or open_[-1] != resume):
                continue  # a resume invocation touches only the named attempt
            if rec is not None:
                continue
            if open_ and resume != open_[-1]:
                status = session.store.attempts()[open_[-1]]["status"]
                if status in ("interrupted", "crashed", "error"):
                    out.append({"key": key, "pending": open_[-1], "status": status})
                    continue
                if status in ("aborted_before_commit", "unrecoverable"):
                    out.append({"key": key, "needs_retry_authorisation": open_[-1]})
                    continue
            res = execute_training_attempt(session, step="L6_pilot", arm=repr(float(lr)), seed=seed, authorisation=auth,
                                           build=build_for(seed), data=data,
                                           resume_attempt=resume if open_ and resume == open_[-1] else None)
            out.append({"key": key, "attempt": res["attempt_id"], "status": res["status"], "postcheck": res["postcheck"]})
    return {"runs": out}


def main(argv=None) -> int:
    """Command line. Exit 0 on success, 2 when blocked (nothing was run)."""
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(DEFAULT_STUDY_ROOT), help="study root")
    ap.add_argument("--resume", default=None, help="attempt id to resume (P7-3)")
    args = ap.parse_args(argv)
    install_git_guard()
    try:
        with study_session(COMMAND, paths=StudyPaths(Path(args.root)), required_decisions=REQUIRED_DECISIONS,
                           required_values=REQUIRED_VALUES) as session:
            from training.rnn_study.common import set_threads

            set_threads(*session.values["threads"])  # plan §8.6: before any tensor operation
            data = load_study_data(research_experiment_config().data)
            out = run(session, data, resume=args.resume)
    except Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""L7: research (final) runs with the full diagnostics plan, at the recorded η, E, g, s² and final seeds;
then ``o3.json`` per run. Requires the final-seed and failed-seed-policy decisions (U4). Milestone MS8.
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

COMMAND = "research_runs"
REQUIRED_DECISIONS = ("horizon_decision", "final_seeds", "failed_seed_policy", "authorisation_L7")
REQUIRED_VALUES = ("probe_sha256", "threads", "g", "s2", "lr", "max_epochs", "final_seeds")


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


def run(session: Session, data: StudyData, resume: str | None = None) -> dict[str, Any]:
    import json

    from models.recurrent import rnn_diagnostics as dg
    from training.rnn_study.common import execute_training_attempt, set_threads

    v = session.values
    for seed in v["final_seeds"]:  # checked for every seed before anything runs
        if seed in (100, 101, 102) or seed in dg.INIT_GATE_SEEDS:
            raise Blocked(f"final seed {seed} overlaps pilot or gate seeds")
    set_threads(*v["threads"])
    auth = session.conditions["decisions"]["authorisation_L7"]["id"]
    build_for = _builder(session, data, float(v["lr"]), int(v["max_epochs"]), with_plan=True)
    out = []
    for seed in v["final_seeds"]:
        key = ("L7_research", repr(float(v["lr"])), seed)
        prior = [a for a, r in session.store.attempts().items() if r["key"] == key]
        if session.store.counted(key) is not None or (resume is not None and resume not in prior):
            continue
        res = execute_training_attempt(session, step="L7_research", arm=repr(float(v["lr"])), seed=seed,
                                       authorisation=auth, build=build_for(seed), data=data,
                                       resume_attempt=resume if resume in prior else None)
        run_dir = Path(res["run_dir"])
        steps = [json.loads(x) for x in (run_dir / "diagnostics" / "steps.jsonl").read_text().splitlines()] \
            if (run_dir / "diagnostics" / "steps.jsonl").exists() else []
        r = res["result"]
        dg.write_o3(run_dir / dg.RESEARCH_DIAGNOSTICS_DIR,
                    dg.o3_record(r.history if r else [], steps, status=res["status"],
                                 diagnostics_status=r.diagnostics_status if r else None,
                                 invalid_points=r.invalid_diagnostic_points if r else ()))
        out.append({"seed": seed, "status": res["status"], "diagnostics_status": r.diagnostics_status if r else None,
                    "invalid_diagnostic_points": list(r.invalid_diagnostic_points) if r else []})
    return {"runs": out}


def main(argv=None) -> int:
    """Command line. Exit 0 on success, 2 when blocked (nothing was run)."""
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(DEFAULT_STUDY_ROOT), help="study root")
    ap.add_argument("--resume", default=None)
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

"""L6: extension run X (training spec §7.4): the selected η, seeds 100 – 102, 60 epochs from scratch, no
research diagnostics; only when the recorded pilot decision says e*₃₀ = 30. Integrity: epochs 1 – 30
must be bit-identical (parameter hashes) to the pilot runs at that η; a mismatch is reported for review.
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

COMMAND = "extension_x"
REQUIRED_DECISIONS = ("pilot_decision", "authorisation_X")
REQUIRED_VALUES = ("probe_sha256", "threads", "g", "s2", "lr")
X_EPOCHS = 60


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


def _history_hashes(run_dir: Path) -> list[str]:
    import json

    return [json.loads(x)["param_sha256"] for x in (Path(run_dir) / "history.jsonl").read_text().splitlines()]


def run(session: Session, data: StudyData, resume: str | None = None, epochs: int = X_EPOCHS,
        prefix_epochs: int = 30) -> dict[str, Any]:
    from training.rnn_study.common import execute_training_attempt, set_threads
    from training.rnn_study.lr_pilot import PILOT_SEEDS

    set_threads(*session.values["threads"])
    lr = float(session.values["lr"])
    auth = session.conditions["decisions"]["authorisation_X"]["id"]
    build_for = _builder(session, data, lr, epochs, with_plan=False)
    out = []
    for seed in PILOT_SEEDS:
        pilot = session.store.counted(("L6_pilot", repr(lr), seed))
        if pilot is None or pilot["status"] != "completed":
            raise Blocked(f"no completed pilot run at η = {lr}, seed {seed}")
        key = ("L6_extension", repr(lr), seed)
        prior = [a for a, r in session.store.attempts().items() if r["key"] == key]
        if session.store.counted(key) is not None or (resume is not None and resume not in prior):
            continue
        res = execute_training_attempt(session, step="L6_extension", arm=repr(lr), seed=seed, authorisation=auth,
                                       build=build_for(seed), data=data,
                                       resume_attempt=resume if resume in prior else None)
        integrity = None
        if res["status"] == "completed":
            a = _history_hashes(session.paths.root / pilot["run_dir"])[: prefix_epochs + 1]
            b = _history_hashes(res["run_dir"])[: prefix_epochs + 1]
            integrity = "bit-identical" if a == b else "MISMATCH: stop X for review"
        out.append({"seed": seed, "status": res["status"], "integrity_epochs_0_30": integrity})
        if integrity is not None and integrity != "bit-identical":
            raise Blocked(f"extension X stopped for review: epochs 0 – {prefix_epochs} of seed {seed} are not "
                          f"bit-identical to its pilot run (training §7.4); completed so far: {out}")
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

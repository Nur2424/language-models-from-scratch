"""Model 6: protected one-time test evaluation of the 10 final runs (authorised by D-61).

This is the only code that builds the Model 6 test split. Run from the repository root:

    python3 results/experiments/rnn_study/test_evaluation/evaluate_test.py freeze     # no test access
    python3 results/experiments/rnn_study/test_evaluation/evaluate_test.py evaluate   # once

freeze   verifies the study (frozen code hashes, verify_runs clean, selection consistent) and writes
         freeze.json once: every model_best.pt hash, its epoch and parameter hash, the code hashes, the
         pinned D-61 row, and the predeclared endpoint. It never reads the test split.
evaluate re-verifies everything in freeze.json, then (once; result.json is written exclusively)
         re-computes each model's dev NLL (must equal its recorded best dev NLL) and its test NLL.

Predeclared endpoint (D-61): test NLL in nats per character (token-weighted, every valid target incl.
EOS), through rnn_training.evaluate_nll with batch 1,024, for model_best.pt of seeds 0 – 9; reported
per seed and as mean ± SE over seeds (SE = SD/√10). Nothing is selected or tuned on test.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from training.rnn_study.common import DEFAULT_STUDY_ROOT, Blocked, StudyPaths, install_git_guard  # noqa: E402

install_git_guard()

HERE = Path(__file__).resolve().parent
FREEZE, RESULT = HERE / "freeze.json", HERE / "result.json"
SEEDS = tuple(range(10))
AUTHORISATION = "D-61"
CODE = ("models/recurrent/rnn.py", "models/recurrent/rnn_training.py", "utils/data.py", "utils/metrics.py",
        "utils/vocabulary.py", "utils/checkpoint.py", "training/rnn_study/common.py",
        "results/experiments/rnn_study/test_evaluation/evaluate_test.py")
ENDPOINT = ("test NLL (nats per character, token-weighted, all valid targets incl. EOS) of model_best.pt, "
            "rnn_training.evaluate_nll, batch 1024; per seed and mean ± SE (SD/sqrt(10)) over seeds 0-9")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selected_runs(paths: StudyPaths) -> dict[int, dict]:
    import torch  # noqa: F401  (checkpoint loading)

    from training.rnn_study.attempts import AttemptStore
    from utils.checkpoint import load_checkpoint

    store, out = AttemptStore(paths), {}
    for seed in SEEDS:
        rec = store.counted(("L7_research", "0.1", seed))
        if rec is None or rec["status"] != "completed":
            raise Blocked(f"seed {seed}: no completed counted final run")
        run_dir = paths.root / rec["run_dir"]
        meta = json.loads((run_dir / "metadata.json").read_text())
        hist = [json.loads(x) for x in (run_dir / "history.jsonl").read_text().splitlines()]
        ck = load_checkpoint(run_dir / "model_best.pt")
        k = int(ck["epoch"])
        row = hist[k]
        best = min((r for r in hist if r["completed_epochs"] >= 1), key=lambda r: (r["dev_nll"], r["completed_epochs"]))
        if not (k == meta["best_epoch"] == best["completed_epochs"] and ck["extra"]["param_sha256"] == row["param_sha256"]):
            raise Blocked(f"seed {seed}: model_best.pt, metadata and history disagree on the selected epoch")
        out[seed] = {"attempt_id": rec["attempt_id"], "run_dir": str(run_dir.relative_to(REPO)), "best_epoch": k,
                     "best_dev_nll": row["dev_nll"], "param_sha256": row["param_sha256"],
                     "model_best_sha256": sha256_file(run_dir / "model_best.pt")}
    return out


def pinned_decision() -> dict:
    import importlib

    dc = importlib.import_module("training.rnn_study.decisions")
    rows = dc.parse_decision_table((REPO / "PROJECT_RESEARCH_RECORD.md").read_text(encoding="utf-8"))
    row = next((r for r in rows if r.id == AUTHORISATION), None)
    if row is None:
        raise Blocked(f"{AUTHORISATION} (test-evaluation authorisation) is not in the research record")
    dc.require_accepted(rows, AUTHORISATION, row.sha256)
    return {"id": AUTHORISATION, "row_sha256": row.sha256}


def freeze() -> dict:
    from training.rnn_study.conditions import load_conditions, verify_frozen
    from training.rnn_study.verify_runs import verify_study

    if FREEZE.exists():
        raise Blocked("freeze.json exists: the test freeze is created exactly once")
    paths = StudyPaths(Path(DEFAULT_STUDY_ROOT))
    verify_frozen(load_conditions(paths))
    findings = {k: v for k, v in verify_study(paths.root).items() if v}
    if findings:
        raise Blocked(f"verify_runs findings: {findings}")
    obj = {"format": "m6_test_freeze/1", "authorisation": pinned_decision(), "endpoint": ENDPOINT,
           "eval_batch_size": 1024, "threads": [4, 8], "checkpoints": {str(s): v for s, v in selected_runs(paths).items()},
           "code_sha256": {f: sha256_file(REPO / f) for f in CODE}}
    with open(FREEZE, "x") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return {"written": str(FREEZE), "sha256": sha256_file(FREEZE), "checkpoints": len(obj["checkpoints"])}


def evaluate() -> dict:
    import models.recurrent.rnn_training as rt
    from models.recurrent.rnn import RNNLanguageModel
    from training.rnn_study.common import research_experiment_config, set_threads
    from utils.data import load_names, make_next_token_sequences, split_names
    from utils.vocabulary import Vocabulary

    if not FREEZE.exists():
        raise Blocked("no freeze.json: run 'freeze' first")
    if RESULT.exists():
        raise Blocked("result.json exists: the test split has already been evaluated (one time only)")
    fz = json.loads(FREEZE.read_text())
    set_threads(*fz["threads"])
    if pinned_decision() != fz["authorisation"]:
        raise Blocked("the D-61 row changed since the freeze")
    bad = [f for f, h in fz["code_sha256"].items() if sha256_file(REPO / f) != h]
    if bad:
        raise Blocked(f"code changed since the freeze: {bad}")
    for s, c in fz["checkpoints"].items():
        if sha256_file(REPO / c["run_dir"] / "model_best.pt") != c["model_best_sha256"]:
            raise Blocked(f"seed {s}: model_best.pt changed since the freeze")
    cfg = research_experiment_config().data
    splits = split_names(load_names(cfg.path), fractions=cfg.split_fractions, seed=cfg.split_seed,
                         duplicate_policy=cfg.duplicate_policy)
    vocab = Vocabulary.from_names(splits.train)
    unknown = vocab.unknown_chars(splits.test)
    if unknown:
        raise Blocked(f"test names contain characters outside the training vocabulary: {dict(unknown)}")
    dev = make_next_token_sequences(splits.dev, vocab)
    test = make_next_token_sequences(splits.test, vocab)  # the only construction of the test split
    per_seed = {}
    for s, c in sorted(fz["checkpoints"].items(), key=lambda kv: int(kv[0])):
        model = RNNLanguageModel.from_checkpoint(REPO / c["run_dir"] / "model_best.pt", expected_vocab=vocab)
        dev_nll = rt.evaluate_nll(model, dev, batch_size=fz["eval_batch_size"]).mean_nll
        if abs(dev_nll - c["best_dev_nll"]) > 1e-9 * abs(c["best_dev_nll"]):
            raise Blocked(f"seed {s}: re-computed dev NLL {dev_nll!r} differs from the recorded {c['best_dev_nll']!r}")
        acc = rt.evaluate_nll(model, test, batch_size=fz["eval_batch_size"])
        per_seed[s] = {"best_epoch": c["best_epoch"], "dev_nll": dev_nll, "dev_nll_recorded": c["best_dev_nll"],
                       "test_nll": acc.mean_nll, "test_tokens": acc.num_tokens, "test_nll_sum": acc.total_nll}
    v = [r["test_nll"] for r in per_seed.values()]
    out = {"format": "m6_test_result/1", "freeze_sha256": sha256_file(FREEZE), "endpoint": fz["endpoint"],
           "test_names": len(splits.test), "per_seed": per_seed,
           "test_nll_mean": statistics.fmean(v), "test_nll_se": statistics.stdev(v) / math.sqrt(len(v))}
    with open(RESULT, "x") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return {"written": str(RESULT), "sha256": sha256_file(RESULT), "test_nll_mean": out["test_nll_mean"],
            "test_nll_se": out["test_nll_se"], "per_seed": {s: round(r["test_nll"], 4) for s, r in per_seed.items()}}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if cmd == "freeze":
            print(json.dumps(freeze(), indent=1))
        elif cmd == "evaluate":
            print(json.dumps(evaluate(), indent=1))
        else:
            print(__doc__)
            raise SystemExit(1)
    except Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)

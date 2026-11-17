"""MLP study: the MLP and the BatchNorm MLP at their own context length (k = 3).

The two specifications (``models/feedforward/mlp_spec.md``, ``batchnorm_mlp_spec.md``)
define a learning-rate pilot (E0) that was never run at k = 3. This script runs it,
fixes the remaining training settings by predeclared rules, trains the final runs,
and evaluates the protected test split once. It also re-trains both models at k = 3
under the exact protocol of the WaveNet study, so each run pairs with arm A or B there.

Stages (run in this order; each one refuses to start until the one before is complete):

  repro      Arm A of the WaveNet study (MLP, k = 8, seed 0) retrained with this script.
             Its best dev NLL must equal the recorded value bit for bit, which shows that
             this script trains exactly as the WaveNet study did.
  pilot      E0 for each model: eta in {0.03, 0.1, 0.3, 1.0}, seeds 100-102, 10 epochs,
             constant LR. Selected: the stable eta with the lowest mean dev NLL at epoch 10.
             Boundary rule (both models): if the winner is at an edge of the grid evaluated
             so far, add one rate further out (x~3), at most twice; still at an edge ->
             inconclusive and the study stops.
             Unstable = a non-finite loss or gradient, or a dev NLL at epoch 10 above ln 27.
             Spikes and oscillations are reported for every candidate, never used to exclude.
  budget     MLP at its selected eta, seeds 100-102, 30 epochs, constant LR.
             e* = the earliest epoch whose mean dev NLL is within max(0.001, SE) of the
             minimum (SE = SD over seeds / sqrt(3)). If e* < 30, the budget E = e*.
             If e* = 30, the runs are repeated for 60 epochs (the first 30 must be identical)
             and E = e* over 60 epochs; if that is 60 the study stops as inconclusive.
             The BatchNorm MLP uses the same E (its specification inherits the MLP budget).
  schedule   MLP only: constant LR against one x0.1 drop at epoch floor(E/2), selected eta,
             seeds 100-102, E epochs. Selected: the lower mean dev NLL at the end of epoch E.
  final      Both models at their selected settings, seeds 0-9.
  paired     Both models at k = 3 under the WaveNet protocol (eta = 0.03, 26 epochs, constant),
             seeds 0-9: identical to arms A and B except for the context length. Runs whose
             settings equal a final run are not repeated.
  test       The one evaluation of the test split: every final run's model_best.pt, once.
             Refuses to run twice. Nothing is selected or changed afterwards.

Every rule above is fixed before any result exists. The script applies them mechanically
and writes one JSON summary per stage to results/experiments/mlp_study/; the training runs
themselves (checkpoints, histories, step records) go to results/experiments/mlp_study/runs/.
An interrupted stage can be started again: completed runs are recognised and skipped.

Run from the repository root, one stage at a time:

    python3 training/mlp_study/run_mlp_study.py repro
    python3 training/mlp_study/run_mlp_study.py pilot
    ...
    python3 training/mlp_study/run_mlp_study.py test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from models.feedforward import batchnorm_mlp, mlp  # noqa: E402
from utils.config import config_to_dict, default_config_paths, load_config  # noqa: E402
from utils.data import check_vocab_coverage, dataset_fingerprint, load_names, make_context_windows, split_names  # noqa: E402
from utils.io import atomic_write_json, read_json  # noqa: E402
from utils.metrics import count_parameters  # noqa: E402
from utils.reproducibility import derive_seed, make_generator, seed_everything  # noqa: E402
from utils.run_manager import RunManager, environment_info, utc_now  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

# ----------------------------------------------------------------------------- fixed settings
STUDY_DIR = REPO_ROOT / "results" / "experiments" / "mlp_study"
RUNS_DIR = STUDY_DIR / "runs"
INDEX_PATH = STUDY_DIR / "runs_index.json"
WAVENET_PER_SEED = REPO_ROOT / "results/experiments/wavenet_study/final/analysis/final/per_seed.csv"

MODELS = {
    "mlp": dict(module=mlp, model_cls=mlp.MLPLanguageModel, config_cls=mlp.MLPConfig,
                training_cls=mlp.MLPTrainingConfig, train_fn=mlp.train_mlp, wavenet_arm="A"),
    "batchnorm_mlp": dict(module=batchnorm_mlp, model_cls=batchnorm_mlp.BatchNormMLPLanguageModel,
                          config_cls=batchnorm_mlp.BatchNormMLPConfig,
                          training_cls=batchnorm_mlp.BatchNormMLPTrainingConfig,
                          train_fn=batchnorm_mlp.train_batchnorm_mlp, wavenet_arm="B"),
}
CONTEXT_LENGTH = 3
INTRA_OP_THREADS = 1                       # as in the WaveNet study, so runs are bit-reproducible
DIAGNOSTICS = {"enabled": True, "every_n_steps": 100, "gradients": True, "parameters": True,
               "activations": False}       # the WaveNet study's step diagnostics
PILOT_GRID = (0.03, 0.1, 0.3, 1.0)
PILOT_SEEDS = (100, 101, 102)
PILOT_EPOCHS = 10
MAX_EXTENSION_ROUNDS = 2
BUDGET_CAP, BUDGET_EXTENDED = 30, 60
FINAL_SEEDS = tuple(range(10))
WAVENET_PROTOCOL = {"lr": 0.03, "epochs": 26, "lr_decay_epoch": None}
EPS_FLOOR = 1e-3
LN27 = math.log(27)


# ----------------------------------------------------------------------------- helpers
def stage_path(name: str) -> Path:
    return STUDY_DIR / f"{name}.json"


def require(name: str) -> dict:
    path = stage_path(name)
    if not path.is_file():
        sys.exit(f"stage '{name}' has not been completed ({path.relative_to(REPO_ROOT)} missing)")
    result = read_json(path)
    if result.get("outcome") == "inconclusive":
        sys.exit(f"stage '{name}' was inconclusive; the study stops here for a human decision")
    return result


def write_stage(name: str, result: dict) -> None:
    atomic_write_json(stage_path(name), {"stage": name, "created_at": utc_now(), **result})
    print(f"Wrote {stage_path(name).relative_to(REPO_ROOT)}")


def mean_se(values: list[float]) -> tuple[float, float]:
    m = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    return m, se


def set_threads() -> None:
    torch.set_num_threads(INTRA_OP_THREADS)
    if torch.get_num_threads() != INTRA_OP_THREADS:
        sys.exit(f"could not set intra-op threads to {INTRA_OP_THREADS}")


def build(model_name: str, seed: int, context_length: int):
    """Data, vocabulary, windows and model exactly as the WaveNet study's build_arm does."""
    spec = MODELS[model_name]
    overrides = [f"model.params.context_length={context_length}", "device.device=cpu",
                 *(f"diagnostics.{k}={str(v).lower()}" for k, v in DIAGNOSTICS.items()),
                 f"reproducibility.seed={seed}"]
    cfg = load_config(default_config_paths(model_name), overrides=overrides, model_config_cls=spec["config_cls"])
    names = load_names(cfg.data.path)
    splits = split_names(names, fractions=cfg.data.split_fractions, seed=cfg.data.split_seed,
                         duplicate_policy=cfg.data.duplicate_policy)
    splits.assert_disjoint()
    vocab = Vocabulary.from_names(splits.train)
    check_vocab_coverage(vocab, splits)
    train = make_context_windows(splits.train, vocab, context_length=context_length)
    dev = make_context_windows(splits.dev, vocab, context_length=context_length)
    seed_everything(seed)  # global RNG hygiene only; init and data order use explicit derived streams
    model = spec["model_cls"](vocab, cfg.model.params, generator=make_generator(derive_seed(seed, "init")))
    return cfg, splits, vocab, train, dev, model


def load_index() -> dict:
    return read_json(INDEX_PATH) if INDEX_PATH.is_file() else {}


def run_key(model_name: str, k: int, lr: float, epochs: int, decay: int | None, seed: int) -> str:
    return f"{model_name}|k={k}|lr={lr}|epochs={epochs}|decay={decay}|seed={seed}"


def train_run(model_name: str, *, k: int, lr: float, epochs: int, decay: int | None, seed: int,
              stage: str) -> dict:
    """Train one run (or return the stored summary if it already completed)."""
    key = run_key(model_name, k, lr, epochs, decay, seed)
    index = load_index()
    if key in index and index[key]["status"] in ("completed", "non_finite"):
        return index[key]
    spec = MODELS[model_name]
    cfg, splits, vocab, train, dev, model = build(model_name, seed, k)
    training = spec["training_cls"](lr=lr, max_epochs=epochs, lr_decay_epoch=decay)
    experiment = config_to_dict(cfg)
    run = RunManager.create(model_name, config=spec["module"].run_config(experiment, model.config, training),
                            dataset_fingerprint={**dataset_fingerprint(cfg.data.path), "splits": splits.fingerprint()},
                            results_root=RUNS_DIR / stage)
    t0 = time.perf_counter()
    entry = {"key": key, "model": model_name, "k": k, "lr": lr, "epochs": epochs, "lr_decay_epoch": decay,
             "seed": seed, "stage": stage, "run_dir": str(Path(run.run_dir).relative_to(REPO_ROOT)),
             "params": count_parameters(model).total}
    try:
        with run.track():
            result = spec["train_fn"](model, train, dev, training=training, seed=seed, run=run,
                                      diagnostics=cfg.diagnostics, config=experiment)
    except FloatingPointError as exc:
        entry.update(status="non_finite", error=str(exc), dev_nll=[], train_nll=[], running_train_nll=[])
    else:
        h = result.history
        entry.update(status="completed", best_epoch=result.best_epoch, best_dev_nll=result.best_dev_nll,
                     dev_nll=[r.dev_nll for r in h], train_nll=[r.train_nll for r in h],
                     running_train_nll=[r.running_train_nll for r in h])
    entry["seconds"] = time.perf_counter() - t0
    entry["grad_norm"] = grad_norm_profile(Path(run.run_dir))
    index = load_index()
    index[key] = entry
    atomic_write_json(INDEX_PATH, index)
    best = f"best dev {entry['best_dev_nll']:.4f} (epoch {entry['best_epoch'] + 1})" \
        if entry["status"] == "completed" else entry["status"].upper()
    print(f"  {model_name:<13} k={k} lr={lr:<5} decay={decay} seed={seed:<3}  {best}  ({entry['seconds']:.0f} s)")
    return entry


def grad_norm_profile(run_dir: Path) -> dict:
    """Max, median and max/median of the recorded global gradient norms (every 100 steps)."""
    path = run_dir / "diagnostics" / "steps.jsonl"
    if not path.is_file():
        return {}
    norms = [json.loads(line).get("grad_global_norm") for line in path.read_text().splitlines() if line.strip()]
    norms = [n for n in norms if isinstance(n, (int, float))]
    if not norms:
        return {}
    med = statistics.median(norms)
    return {"samples": len(norms), "max": max(norms), "median": med, "max_over_median": max(norms) / med if med else None}


def describe_candidate(runs: list[dict], epoch: int) -> dict:
    """Pilot report for one learning rate (specification §13.4)."""
    out = {"lr": runs[0]["lr"], "seeds": [r["seed"] for r in runs]}
    if any(r["status"] != "completed" for r in runs):
        out.update(stable=False, reason="non-finite loss or gradient",
                   failed_seeds=[r["seed"] for r in runs if r["status"] != "completed"])
        return out
    at = [r["dev_nll"][epoch - 1] for r in runs]
    m, se = mean_se(at)
    unstable = [r["seed"] for r, v in zip(runs, at) if v > LN27]
    out.update(
        stable=not unstable,
        reason=f"dev NLL above ln 27 at epoch {epoch}" if unstable else None,
        dev_nll_at_end_mean=m, dev_nll_at_end_sd=statistics.stdev(at), dev_nll_at_end_se=se,
        per_seed=[{"seed": r["seed"], "dev_nll_at_end": r["dev_nll"][epoch - 1],
                   "best_dev_nll": min(r["dev_nll"][:epoch]),
                   "best_epoch": 1 + r["dev_nll"][:epoch].index(min(r["dev_nll"][:epoch])),
                   "dev_nll_rises": sum(b > a for a, b in zip(r["dev_nll"][:epoch], r["dev_nll"][1:epoch])),
                   "running_train_nll_rises": sum(b > a for a, b in zip(r["running_train_nll"][:epoch],
                                                                         r["running_train_nll"][1:epoch])),
                   "grad_norm": r["grad_norm"]} for r in runs],
    )
    return out


def plateau_epoch(curves: list[list[float]], horizon: int) -> dict:
    """e*: the earliest epoch (1-based) whose mean dev NLL is within max(0.001, SE) of the minimum."""
    means, ses = [], []
    for e in range(horizon):
        m, se = mean_se([c[e] for c in curves])
        means.append(m)
        ses.append(se)
    best = min(range(horizon), key=lambda e: means[e])
    e_star = next(e for e in range(horizon) if means[e] <= means[best] + max(EPS_FLOOR, ses[e]))
    eps = max(EPS_FLOOR, ses[e_star])
    return {"e_star": e_star + 1, "min_epoch": best + 1, "min_mean_dev_nll": means[best],
            "mean_dev_nll_at_e_star": means[e_star], "epsilon_at_e_star": eps,
            "epsilon_from": "SE" if ses[e_star] > EPS_FLOOR else "floor 0.001",
            "mean_dev_nll_by_epoch": means, "se_by_epoch": ses}


# ----------------------------------------------------------------------------- stages
def stage_repro(_args) -> None:
    rows = {(r["arm"], r["seed"]): r for r in csv.DictReader(WAVENET_PER_SEED.open())}
    recorded = rows[("A", "0")]
    print("Reproduction check: WaveNet-study arm A (MLP, k = 8), seed 0, eta = 0.03, 26 epochs")
    entry = train_run("mlp", k=8, lr=0.03, epochs=26, decay=None, seed=0, stage="repro")
    rec_dev, rec_epoch = float(recorded["best_dev_nll"]), int(recorded["best_epoch"])
    match = entry["status"] == "completed" and entry["best_dev_nll"] == rec_dev and entry["best_epoch"] == rec_epoch
    print(f"  recorded best dev {rec_dev!r} at epoch index {rec_epoch}")
    print(f"  this run best dev {entry.get('best_dev_nll')!r} at epoch index {entry.get('best_epoch')}")
    print("  MATCH: bit-identical" if match else "  MISMATCH: the study stops here")
    write_stage("repro", {"outcome": "match" if match else "inconclusive",
                          "recorded": {"best_dev_nll": rec_dev, "best_epoch": rec_epoch},
                          "reproduced": {"best_dev_nll": entry.get("best_dev_nll"),
                                         "best_epoch": entry.get("best_epoch")},
                          "environment": {**environment_info(), "torch_num_threads": torch.get_num_threads(),
                                          "torch_num_interop_threads": torch.get_num_interop_threads()}})


def pilot_one(model_name: str) -> dict:
    grid = list(PILOT_GRID)
    rounds = []
    while True:
        cands = {lr: describe_candidate([train_run(model_name, k=CONTEXT_LENGTH, lr=lr, epochs=PILOT_EPOCHS,
                                                   decay=None, seed=s, stage="pilot") for s in PILOT_SEEDS],
                                        PILOT_EPOCHS) for lr in sorted(grid)}
        stable = [lr for lr in sorted(grid) if cands[lr]["stable"]]
        if not stable:
            return {"outcome": "inconclusive", "reason": "every candidate unstable", "candidates": cands,
                    "extension_rounds": rounds}
        best = min(stable, key=lambda lr: cands[lr]["dev_nll_at_end_mean"])
        lo, hi = min(grid), max(grid)
        if best not in (lo, hi):
            return {"outcome": "selected", "selected_lr": best, "candidates": cands, "extension_rounds": rounds}
        if len(rounds) == MAX_EXTENSION_ROUNDS:
            return {"outcome": "inconclusive", "reason": f"winner {best} still at a grid edge after "
                    f"{MAX_EXTENSION_ROUNDS} extension rounds", "candidates": cands, "extension_rounds": rounds}
        new = round(lo / 3, 4) if best == lo else round(hi * 3, 4)
        new = {0.01: 0.01, 0.0033: 0.003, 3.0: 3.0, 9.0: 10.0}.get(new, new)  # the specification's x~3 values
        rounds.append({"round": len(rounds) + 1, "reason": f"winner {best} at the {'lower' if best == lo else 'upper'}"
                       " edge", "added_lr": new})
        print(f"  {model_name}: winner {best} at an edge; extending the grid with {new}")
        grid.append(new)


def stage_pilot(_args) -> None:
    require("repro")
    set_threads()
    out = {}
    for model_name in MODELS:
        print(f"Learning-rate pilot E0: {model_name}, k = 3, seeds 100-102, 10 epochs")
        out[model_name] = pilot_one(model_name)
        r = out[model_name]
        print(f"  -> {r['outcome']}" + (f": eta = {r['selected_lr']}" if r["outcome"] == "selected" else
                                       f" ({r['reason']})"))
    outcome = "selected" if all(r["outcome"] == "selected" for r in out.values()) else "inconclusive"
    write_stage("pilot", {"outcome": outcome, "models": out})


def stage_budget(_args) -> None:
    pilot = require("pilot")
    set_threads()
    lr = pilot["models"]["mlp"]["selected_lr"]
    print(f"Epoch budget: MLP, k = 3, eta = {lr}, seeds 100-102, {BUDGET_CAP} epochs")
    runs = [train_run("mlp", k=CONTEXT_LENGTH, lr=lr, epochs=BUDGET_CAP, decay=None, seed=s, stage="budget")
            for s in PILOT_SEEDS]
    if any(r["status"] != "completed" for r in runs):
        return write_stage("budget", {"outcome": "inconclusive", "reason": "a budget run failed", "lr": lr})
    first = plateau_epoch([r["dev_nll"] for r in runs], BUDGET_CAP)
    result = {"lr": lr, "plateau_30": first}
    if first["e_star"] < BUDGET_CAP:
        result.update(outcome="selected", max_epochs=first["e_star"])
    else:
        print(f"  e* = {BUDGET_CAP}: extending to {BUDGET_EXTENDED} epochs")
        ext = [train_run("mlp", k=CONTEXT_LENGTH, lr=lr, epochs=BUDGET_EXTENDED, decay=None, seed=s, stage="budget")
               for s in PILOT_SEEDS]
        identical = all(e["dev_nll"][:BUDGET_CAP] == r["dev_nll"] for e, r in zip(ext, runs))
        second = plateau_epoch([e["dev_nll"] for e in ext], BUDGET_EXTENDED)
        result.update(extension_identical_first_30=identical, plateau_60=second)
        if not identical:
            result.update(outcome="inconclusive", reason="extended runs differ from the 30-epoch runs")
        elif second["e_star"] == BUDGET_EXTENDED:
            result.update(outcome="inconclusive", reason=f"e* = {BUDGET_EXTENDED}")
        else:
            result.update(outcome="selected", max_epochs=second["e_star"])
    print(f"  -> {result['outcome']}" + (f": E = {result['max_epochs']}" if result["outcome"] == "selected" else ""))
    write_stage("budget", result)


def stage_schedule(_args) -> None:
    pilot, budget = require("pilot"), require("budget")
    set_threads()
    lr, E = pilot["models"]["mlp"]["selected_lr"], budget["max_epochs"]
    decay = E // 2
    print(f"Schedule follow-up: MLP, eta = {lr}, E = {E}: constant against x0.1 at epoch {decay}")
    arms = {}
    for name, d in (("constant", None), (f"drop_at_{decay}", decay)):
        runs = [train_run("mlp", k=CONTEXT_LENGTH, lr=lr, epochs=E, decay=d, seed=s, stage="schedule")
                for s in PILOT_SEEDS]
        arms[name] = describe_candidate(runs, E) | {"lr_decay_epoch": d}
    stable = {n: a for n, a in arms.items() if a["stable"]}
    if not stable:
        return write_stage("schedule", {"outcome": "inconclusive", "reason": "both arms unstable", "arms": arms})
    chosen = min(stable, key=lambda n: stable[n]["dev_nll_at_end_mean"])
    print(f"  -> {chosen}")
    write_stage("schedule", {"outcome": "selected", "lr": lr, "max_epochs": E,
                             "selected": chosen, "lr_decay_epoch": arms[chosen]["lr_decay_epoch"], "arms": arms})


def final_settings() -> dict:
    pilot, budget, schedule = require("pilot"), require("budget"), require("schedule")
    E = budget["max_epochs"]
    return {"mlp": {"lr": pilot["models"]["mlp"]["selected_lr"], "epochs": E, "lr_decay_epoch": schedule["lr_decay_epoch"]},
            "batchnorm_mlp": {"lr": pilot["models"]["batchnorm_mlp"]["selected_lr"], "epochs": E,
                              "lr_decay_epoch": None}}


def summarise_final(runs: list[dict]) -> dict:
    ok = [r for r in runs if r["status"] == "completed"]
    m, se = mean_se([r["best_dev_nll"] for r in ok]) if ok else (None, None)
    return {"completed": len(ok), "failed_seeds": [r["seed"] for r in runs if r["status"] != "completed"],
            "best_dev_nll_mean": m, "best_dev_nll_se": se,
            "runs": [{k: r.get(k) for k in ("seed", "status", "best_epoch", "best_dev_nll", "run_dir", "params")}
                     for r in runs]}


def stage_final(_args) -> None:
    settings = final_settings()
    set_threads()
    out = {}
    for model_name, s in settings.items():
        print(f"Final runs: {model_name}, k = 3, eta = {s['lr']}, E = {s['epochs']}, decay = {s['lr_decay_epoch']}, "
              f"seeds 0-9")
        runs = [train_run(model_name, k=CONTEXT_LENGTH, lr=s["lr"], epochs=s["epochs"], decay=s["lr_decay_epoch"],
                          seed=seed, stage="final") for seed in FINAL_SEEDS]
        out[model_name] = {"settings": s, **summarise_final(runs)}
    write_stage("final", {"outcome": "completed", "models": out})


def stage_paired(_args) -> None:
    require("final")
    set_threads()
    wn = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(WAVENET_PER_SEED.open())}
    out = {}
    for model_name, spec in MODELS.items():
        p = WAVENET_PROTOCOL
        print(f"Paired runs: {model_name}, k = 3 under the WaveNet protocol (eta = {p['lr']}, {p['epochs']} epochs)")
        runs = [train_run(model_name, k=CONTEXT_LENGTH, lr=p["lr"], epochs=p["epochs"], decay=p["lr_decay_epoch"],
                          seed=seed, stage="paired") for seed in FINAL_SEEDS]
        arm = spec["wavenet_arm"]
        diffs = [float(wn[(arm, r["seed"])]["best_dev_nll"]) - r["best_dev_nll"] for r in runs
                 if r["status"] == "completed"]
        m, se = mean_se(diffs) if diffs else (None, None)
        out[model_name] = {"wavenet_arm": arm, **summarise_final(runs),
                           "k8_minus_k3_best_dev_nll": {"mean": m, "se": se, "per_seed": diffs,
                                                        "seeds_with_k8_lower": sum(d < 0 for d in diffs)}}
        print(f"  best dev NLL, k = 8 minus k = 3: {m:+.4f} +/- {se:.4f} (SE over 10 paired seeds)")
    write_stage("paired", {"outcome": "completed", "note": "descriptive; best dev NLL is optimistically biased",
                           "models": out})


def stage_test(_args) -> None:
    final = require("final")
    if stage_path("test").is_file():
        sys.exit("the test split has already been evaluated; it is evaluated only once")
    set_threads()
    out = {}
    for model_name, spec in MODELS.items():
        cfg, splits, vocab, _, _, _ = build(model_name, 0, CONTEXT_LENGTH)
        test = make_context_windows(splits.test, vocab, context_length=CONTEXT_LENGTH)
        rows = []
        for r in final["models"][model_name]["runs"]:
            if r["status"] != "completed":
                continue
            model = spec["model_cls"].from_checkpoint(REPO_ROOT / r["run_dir"] / "model_best.pt", expected_vocab=vocab)
            acc = spec["module"].evaluate_nll(model, test, batch_size=4096)
            rows.append({"seed": r["seed"], "best_epoch": r["best_epoch"], "best_dev_nll": r["best_dev_nll"],
                         "test_nll": acc.mean_nll, "test_tokens": acc.num_tokens})
        m, se = mean_se([x["test_nll"] for x in rows])
        out[model_name] = {"test_nll_mean": m, "test_nll_se": se, "per_seed": rows}
        print(f"  {model_name}: test NLL {m:.4f} +/- {se:.4f} (SE over {len(rows)} seeds)")
    write_stage("test", {"outcome": "completed", "endpoint": "test NLL (nats per character) of model_best.pt, "
                         "evaluate_nll, batch 4096; evaluated once", "models": out})


STAGES = {"repro": stage_repro, "pilot": stage_pilot, "budget": stage_budget, "schedule": stage_schedule,
          "final": stage_final, "paired": stage_paired, "test": stage_test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=list(STAGES))
    args = parser.parse_args()
    set_threads()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()

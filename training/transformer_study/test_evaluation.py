"""Freeze, one-time test evaluation and statistical analysis of Transformer Part I.

This is the only code of the study that builds the test split. Run from the repository root::

    python3 training/transformer_study/test_evaluation.py freeze     # no test access
    python3 training/transformer_study/test_evaluation.py evaluate   # once
    python3 training/transformer_study/test_evaluation.py analyze

``freeze`` checks that every final run of the three new arms (Model 01, Model 02, arm D) completed,
re-computes each ``model_best.pt``'s dev NLL against its recorded value, and writes ``freeze.json``
once: checkpoint and code SHA-256, the selected settings, the RNN study's published test result (by
hash) and the predeclared analysis. It never reads the test split.

``evaluate`` re-verifies every hash in ``freeze.json``, then builds the test split (the only place in
the study), evaluates each frozen checkpoint once and writes ``result.json`` exclusively. A second
evaluation is refused.

``analyze`` applies the predeclared statistics to ``result.json`` (study plan §6): paired seed
differences with 95% t intervals; Model 02 vs the RNN (two-sided), Model 02 vs D as an equivalence
test within ±0.01 (TOST: p = the larger one-sided p, 90% interval), Model 01 vs Model 02 (two-sided);
Holm across these three; every other pair descriptive.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.attention.transformer import TransformerLanguageModel  # noqa: E402
import training.transformer_study.transformer_runs as tr  # noqa: E402
import training.transformer_study.wavenet_d_arm as wd  # noqa: E402
from utils.data import file_sha256, load_names, make_next_token_sequences, split_names  # noqa: E402

SEEDS = tuple(range(10))
NEW_ARMS = ("gpt2_style", "transformer_2017", "wavenet_d")
SESOI = 0.01
DEV_TOLERANCE = 1e-9   # relative; the re-computed dev NLL of a frozen checkpoint must equal the recorded one
RNN_RESULT = REPO / "results" / "experiments" / "rnn_study" / "test_evaluation" / "result.json"
CODE_FILES = (
    "models/attention/transformer.py", "models/hierarchical/wavenet.py",
    "training/transformer_study/transformer_runs.py", "training/transformer_study/wavenet_d_arm.py",
    "training/transformer_study/test_evaluation.py",
    "utils/data.py", "utils/vocabulary.py", "utils/batching.py", "utils/metrics.py",
    "utils/reproducibility.py", "utils/checkpoint.py",
)
ANALYSIS = {
    "endpoint": "test NLL (nats per character, token-weighted, every valid target incl. EOS) of model_best.pt, "
                "per seed 0–9; mean ± SE over seeds",
    "primary": [
        {"id": "T-RQ1a", "a": "gpt2_style", "b": "rnn", "test": "paired two-sided t"},
        {"id": "T-RQ1b_equivalence", "a": "gpt2_style", "b": "wavenet_d", "test": f"TOST, margin ±{SESOI}"},
        {"id": "T-RQ2", "a": "transformer_2017", "b": "gpt2_style", "test": "paired two-sided t"},
    ],
    "multiplicity": "Holm across the three primary p-values (TOST p = larger one-sided p)",
    "secondary": "every other pair: paired mean difference with 95% t interval, descriptive",
    "pairing": "seed s of every arm uses the same name batches (EpochSampler, derive_seed(s, 'data'))",
}


def _root(root: Path | None) -> Path:
    return (root or tr.DEFAULT_ROOT)


def _rel(p: Path) -> str:
    """Repository-relative path when possible (portable records), otherwise absolute."""
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def run_dir(root: Path, arm: str, seed: int) -> Path:
    if arm == "wavenet_d":
        return root / "wavenet_d" / "final" / f"seed_{seed}"
    return root / "final" / arm / f"seed_{seed}"


def _load(arm: str, path: Path, vocab, seed: int) -> torch.nn.Module:
    if arm == "wavenet_d":
        model = wd.make_model(vocab, seed)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        return model.eval()
    return TransformerLanguageModel.from_checkpoint(path, expected_vocab=vocab, expected_variant=arm).eval()


def _evaluate(arm: str, model: torch.nn.Module, names: Sequence[str], vocab) -> dict[str, Any]:
    if arm == "wavenet_d":
        return wd.evaluate_by_position(model, wd.build_name_windows(names, vocab))
    return tr.evaluate(model, make_next_token_sequences(list(names), vocab, max_len=tr.MAX_LEN))


def _splits():
    return split_names(load_names(), fractions=wd.SPLIT_FRACTIONS, seed=wd.SPLIT_SEED, duplicate_policy=wd.DUPLICATE_POLICY)


def _settings(root: Path) -> dict[str, Any]:
    out = {"shape": None, "runs": {}}
    shape_summary = root / "shape_pilot_summary.json"
    if shape_summary.is_file():
        out["shape"] = json.loads(shape_summary.read_text())["shape_choice"]["chosen"]
    for arm in NEW_ARMS:
        cfg = json.loads((run_dir(root, arm, SEEDS[0]) / "config.json").read_text())
        out["runs"][arm] = {k: cfg.get(k) for k in ("lr", "max_epochs", "num_layers", "d_model", "optimizer")}
    return out


# ------------------------------------------------------------------ freeze
def freeze(root: Path | None = None, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    root = _root(root)
    path = root / "test_evaluation" / "freeze.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the study is already frozen")
    splits = _splits()
    vocab = wd.Vocabulary.from_names(splits.train)
    checkpoints: dict[str, dict[str, Any]] = {}
    for arm in NEW_ARMS:
        checkpoints[arm] = {}
        for s in seeds:
            d = run_dir(root, arm, s)
            if wd.run_status(d) != "completed":
                raise SystemExit(f"{d}: status {wd.run_status(d)!r}, not completed")
            meta = json.loads((d / "metadata.json").read_text())
            hist = {h["epoch"]: h for h in wd._read_history(d)}
            best = hist[meta["best_epoch"]]
            model = _load(arm, d / "model_best.pt", vocab, s)
            dev = _evaluate(arm, model, splits.dev, vocab)["nll"]
            if abs(dev - best["dev_nll"]) > DEV_TOLERANCE * abs(best["dev_nll"]):
                raise SystemExit(f"{d}: re-computed dev NLL {dev!r} differs from the recorded {best['dev_nll']!r}")
            if wd.param_sha256(model) != best["param_sha256"]:
                raise SystemExit(f"{d}: model_best.pt does not match the recorded parameter hash")
            checkpoints[arm][str(s)] = {"run_dir": _rel(d), "best_epoch": meta["best_epoch"],
                                        "best_dev_nll": best["dev_nll"], "param_sha256": best["param_sha256"],
                                        "model_best_sha256": file_sha256(d / "model_best.pt")}
    out = {"format": "transformer_part1_freeze/1", "checkpoints": checkpoints, "settings": _settings(root),
           "code_sha256": {f: file_sha256(REPO / f) for f in CODE_FILES},
           "rnn_result": {"path": _rel(RNN_RESULT), "sha256": file_sha256(RNN_RESULT)},
           "analysis": ANALYSIS, "seeds": list(seeds)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return out


# ------------------------------------------------------------------ one-time evaluation
def evaluate(root: Path | None = None) -> dict[str, Any]:
    root = _root(root)
    fpath, rpath = root / "test_evaluation" / "freeze.json", root / "test_evaluation" / "result.json"
    if not fpath.is_file():
        raise SystemExit("no freeze.json: run 'freeze' first")
    if rpath.exists():
        raise SystemExit("result.json exists: the test split has already been evaluated (one time only)")
    fz = json.loads(fpath.read_text())
    bad = [f for f, h in fz["code_sha256"].items() if file_sha256(REPO / f) != h]
    if bad:
        raise SystemExit(f"code changed since the freeze: {bad}")
    if file_sha256(REPO / fz["rnn_result"]["path"]) != fz["rnn_result"]["sha256"]:
        raise SystemExit("the RNN study's test result changed since the freeze")
    for arm, per in fz["checkpoints"].items():
        for s, c in per.items():
            if file_sha256(REPO / c["run_dir"] / "model_best.pt") != c["model_best_sha256"]:
                raise SystemExit(f"{arm} seed {s}: model_best.pt changed since the freeze")
    splits = _splits()
    vocab = wd.Vocabulary.from_names(splits.train)
    unknown = vocab.unknown_chars(splits.test)
    if unknown:
        raise SystemExit(f"test names contain characters outside the training vocabulary: {dict(unknown)}")
    per_seed: dict[str, dict[str, Any]] = {}
    for arm, per in fz["checkpoints"].items():
        per_seed[arm] = {}
        for s, c in per.items():
            model = _load(arm, REPO / c["run_dir"] / "model_best.pt", vocab, int(s))
            dev = _evaluate(arm, model, splits.dev, vocab)["nll"]
            if abs(dev - c["best_dev_nll"]) > DEV_TOLERANCE * abs(c["best_dev_nll"]):
                raise SystemExit(f"{arm} seed {s}: dev NLL {dev!r} differs from the frozen {c['best_dev_nll']!r}")
            test = _evaluate(arm, model, splits.test, vocab)            # the only test-split evaluation
            per_seed[arm][s] = {"test_nll": test["nll"], "test_sum": test["sum"], "test_count": test["count"],
                                "test_by_position": test["by_position"], "dev_nll": dev, "best_epoch": c["best_epoch"]}
    rnn = json.loads((REPO / fz["rnn_result"]["path"]).read_text())
    per_seed["rnn"] = {s: {"test_nll": v["test_nll"], "dev_nll": v["dev_nll"], "best_epoch": v["best_epoch"]}
                       for s, v in rnn["per_seed"].items()}
    out = {"format": "transformer_part1_result/1", "freeze_sha256": file_sha256(fpath), "test_names": len(splits.test),
           "per_seed": per_seed,
           "summary": {arm: _mean_se([v["test_nll"] for v in per.values()]) for arm, per in per_seed.items()}}
    with open(rpath, "x") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return out


# ------------------------------------------------------------------ statistics
def _mean_se(v: Sequence[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(v), "se": statistics.stdev(v) / math.sqrt(len(v)), "n": len(v)}


def paired(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Paired differences a − b: mean, SE, 95% t interval, two-sided p, count of seeds with a < b."""
    from scipy import stats
    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    mean, se = statistics.fmean(diff), statistics.stdev(diff) / math.sqrt(n)
    t975 = float(stats.t.ppf(0.975, n - 1))
    p = float(2 * stats.t.sf(abs(mean / se), n - 1)) if se > 0 else (0.0 if mean else 1.0)
    return {"mean_diff": mean, "se": se, "ci95": [mean - t975 * se, mean + t975 * se], "p_two_sided": p,
            "a_lower_in": sum(d < 0 for d in diff), "n": n}


def tost(a: Sequence[float], b: Sequence[float], margin: float = SESOI) -> dict[str, Any]:
    """Equivalence of a and b within ±margin: two one-sided paired t tests; p = the larger; 90% interval."""
    from scipy import stats
    base = paired(a, b)
    n, mean, se = base["n"], base["mean_diff"], base["se"]
    p_lower = float(stats.t.sf((mean + margin) / se, n - 1))   # H0: diff ≤ −margin
    p_upper = float(stats.t.cdf((mean - margin) / se, n - 1))  # H0: diff ≥ +margin
    t95 = float(stats.t.ppf(0.95, n - 1))
    ci90 = [mean - t95 * se, mean + t95 * se]
    return {**base, "margin": margin, "p_tost": max(p_lower, p_upper), "ci90": ci90,
            "equivalent": bool(ci90[0] > -margin and ci90[1] < margin)}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    order = sorted(pvalues, key=pvalues.get)
    m, out, running, stop = len(order), {}, 0.0, False
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * pvalues[k])
        running = max(running, adj)
        reject = bool(not stop and pvalues[k] <= alpha / (m - i))
        stop = stop or not reject
        out[k] = {"p": pvalues[k], "p_holm": running, "reject": reject}
    return out


def analyze(root: Path | None = None) -> dict[str, Any]:
    root = _root(root)
    res = json.loads((root / "test_evaluation" / "result.json").read_text())
    ps = res["per_seed"]
    seeds = sorted(ps["gpt2_style"], key=int)

    def vals(arm):
        return [ps[arm][s]["test_nll"] for s in seeds]

    primary = {"T-RQ1a": paired(vals("gpt2_style"), vals("rnn")),
               "T-RQ1b_equivalence": tost(vals("gpt2_style"), vals("wavenet_d")),
               "T-RQ2": paired(vals("transformer_2017"), vals("gpt2_style"))}
    pv = {"T-RQ1a": primary["T-RQ1a"]["p_two_sided"], "T-RQ1b_equivalence": primary["T-RQ1b_equivalence"]["p_tost"],
          "T-RQ2": primary["T-RQ2"]["p_two_sided"]}
    arms = ["rnn", "wavenet_d", "transformer_2017", "gpt2_style"]
    secondary = {f"{a} - {b}": paired(vals(a), vals(b)) for i, a in enumerate(arms) for b in arms[i + 1:]}
    out = {"test_nll": {a: _mean_se(vals(a)) for a in arms}, "primary": primary, "holm": holm(pv),
           "secondary": secondary, "analysis": ANALYSIS}
    wd._write_json(root / "test_evaluation" / "analysis.json", out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transformer Part I: freeze, one-time test evaluation, analysis")
    ap.add_argument("cmd", choices=["freeze", "evaluate", "analyze"])
    args = ap.parse_args(argv)
    wd.set_threads()
    if args.cmd == "freeze":
        out = freeze()
        print(f"frozen: {sum(len(v) for v in out['checkpoints'].values())} checkpoints; settings {out['settings']}")
    elif args.cmd == "evaluate":
        out = evaluate()
        for arm, s in out["summary"].items():
            print(f"{arm}: test NLL {s['mean']:.4f} ± {s['se']:.4f} (n = {s['n']})")
    else:
        out = analyze()
        print(json.dumps({k: out[k] for k in ("test_nll", "primary", "holm")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

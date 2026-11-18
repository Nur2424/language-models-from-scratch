"""Arm D of Transformer Part I: WaveNet with BatchNorm under the shared name-batch protocol.

Specification: ``training/transformer_study/wavenet_d_arm_spec.md``.

The model, its BatchNorm calibration and its evaluation come unchanged from
``models/hierarchical/wavenet.py``. What is new is only the protocol, taken from the vanilla RNN:
batches of 32 whole names (the RNN's ``EpochSampler`` over training names, data seed
``derive_seed(seed, "data")``), each turned into all of its 8-character windows, and the loss
``sum of NLL / C`` with the fixed constant ``C = B · M / N``.

Command line (from the repository root)::

    python3 training/transformer_study/wavenet_d_arm.py pilot              # η grid × seeds 100–102, 30 epochs
    python3 training/transformer_study/wavenet_d_arm.py summarize-pilot    # selection and horizon rules
    python3 training/transformer_study/wavenet_d_arm.py pilot --lrs 0.01   # a boundary extension, if needed
    python3 training/transformer_study/wavenet_d_arm.py extension --lr η   # 60 epochs, only if e* = 30
    python3 training/transformer_study/wavenet_d_arm.py summarize-extension --lr η
    python3 training/transformer_study/wavenet_d_arm.py final --lr η --epochs E

Every command skips runs that already completed and resumes interrupted ones at the last
epoch boundary. Nothing here reads the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.hierarchical.wavenet import (  # noqa: E402
    WaveNetConfig,
    WaveNetLanguageModel,
    calibrate_batchnorm,
    evaluate_nll,
)
from utils.batching import EpochSampler  # noqa: E402
from utils.data import ContextWindows, load_names, make_context_windows, split_names  # noqa: E402
from utils.reproducibility import derive_seed, make_generator  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

# ------------------------------------------------------------------ fixed settings (spec §2, §6)
CONTEXT_LENGTH = 8
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 4096
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
DUPLICATE_POLICY = "group"
D_CONFIG = WaveNetConfig(context_length=8, embedding_dim=10, hidden_dim=68, normalization="batchnorm")
LR_GRID = (0.03, 0.1, 0.3, 1.0)
PILOT_SEEDS = (100, 101, 102)
FINAL_SEEDS = tuple(range(10))
PILOT_EPOCHS = 30
EXTENSION_EPOCHS = 60
EPSILON_FLOOR = 1e-3
INTRA_OP_THREADS, INTER_OP_THREADS = 4, 8
LN27 = math.log(27)
DEFAULT_ROOT = REPO / "results" / "experiments" / "transformer_study" / "wavenet_d"


# ------------------------------------------------------------------ data
@dataclass(frozen=True)
class NameWindows:
    """All 8-character windows of a list of names, with each name's contiguous row range.

    Rows of name ``n`` are ``offsets[n] .. offsets[n+1] − 1``: one window per target, EOS included.
    """

    windows: ContextWindows
    offsets: torch.Tensor  # [N + 1] int64

    @property
    def num_names(self) -> int:
        return int(self.offsets.numel()) - 1

    @property
    def num_targets(self) -> int:
        return int(self.offsets[-1])

    def rows(self, name_idx: torch.Tensor) -> torch.Tensor:
        """Window rows of the given names, concatenated in the order of ``name_idx``."""
        starts = self.offsets[name_idx]
        counts = self.offsets[name_idx + 1] - starts
        first = torch.cumsum(counts, 0) - counts
        within = torch.arange(int(counts.sum())) - torch.repeat_interleave(first, counts)
        return torch.repeat_interleave(starts, counts) + within

    def positions(self) -> torch.Tensor:
        """Position i of every row: the number of characters read before its target (0 = after BOS)."""
        return torch.arange(self.num_targets) - self.offsets[self.windows.name_index]


def build_name_windows(names: Sequence[str], vocab: Vocabulary) -> NameWindows:
    windows = make_context_windows(list(names), vocab, context_length=CONTEXT_LENGTH)
    counts = torch.bincount(windows.name_index, minlength=len(names))
    expected = torch.tensor([len(n) + 1 for n in names], dtype=torch.long)
    if not torch.equal(counts, expected):
        raise ValueError("window counts do not match name lengths + 1")
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)])
    return NameWindows(windows=windows, offsets=offsets)


def loss_constant(names: Sequence[str], batch_size: int = BATCH_SIZE) -> float:
    """C = B · M / N in float64, as ``rnn_training.loss_normalizer``."""
    n_names, n_targets = len(names), sum(len(n) + 1 for n in names)
    return batch_size * n_targets / n_names


@dataclass(frozen=True)
class StudyData:
    vocab: Vocabulary
    train_names: tuple[str, ...]
    train: NameWindows
    dev: NameWindows


def load_study_data() -> StudyData:
    splits = split_names(load_names(), fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED, duplicate_policy=DUPLICATE_POLICY)
    vocab = Vocabulary.from_names(splits.train)
    return StudyData(vocab=vocab, train_names=tuple(splits.train),
                     train=build_name_windows(splits.train, vocab), dev=build_name_windows(splits.dev, vocab))


# ------------------------------------------------------------------ helpers
def make_model(vocab: Vocabulary, seed: int) -> WaveNetLanguageModel:
    return WaveNetLanguageModel(vocab, D_CONFIG, generator=make_generator(derive_seed(seed, "init")))


def make_sampler(num_names: int, seed: int) -> EpochSampler:
    """The RNN's name sampler (``rnn_training.train_rnn``): identical batches for the same seed."""
    return EpochSampler(num_names, BATCH_SIZE, seed=derive_seed(seed, "data"))


def param_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate_by_position(model: WaveNetLanguageModel, data: NameWindows) -> dict[str, Any]:
    """Token-weighted NLL in eval mode, with per-position sums and counts (float64)."""
    was_training = model.training
    pos = data.positions()
    n_pos = int(pos.max()) + 1
    sums = torch.zeros(n_pos, dtype=torch.float64)
    counts = torch.bincount(pos, minlength=n_pos)
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, data.num_targets, EVAL_BATCH_SIZE):
                sl = slice(start, start + EVAL_BATCH_SIZE)
                nll = F.cross_entropy(model(data.windows.contexts[sl]), data.windows.targets[sl], reduction="none")
                sums.index_add_(0, pos[sl], nll.double())
    finally:
        model.train(was_training)
    total = float(sums.sum())
    return {"nll": total / data.num_targets, "sum": total, "count": data.num_targets,
            "by_position": {"sum": sums.tolist(), "count": counts.tolist()}}


def set_threads(intra: int = INTRA_OP_THREADS, inter: int = INTER_OP_THREADS) -> None:
    torch.set_num_threads(intra)
    if torch.get_num_interop_threads() != inter:
        torch.set_num_interop_threads(inter)


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


def _read_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "history.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


def run_status(run_dir: Path) -> str | None:
    meta = run_dir / "metadata.json"
    return json.loads(meta.read_text()).get("status") if meta.is_file() else None


# ------------------------------------------------------------------ one run (spec §3–§5)
def train_run(data: StudyData, *, seed: int, lr: float, max_epochs: int, run_dir: Path,
              names_for_constant: Sequence[str] | None = None) -> dict[str, Any]:
    """Train arm D for one seed and learning rate; resumable at epoch boundaries.

    Status: ``completed``, ``failed_mechanical`` (category A: a non-finite loss, gradient,
    parameter or evaluation) or ``failed_diverged`` (category D: full-train NLL > ln 27 at a
    completed epoch ≥ 1), as in the RNN training specification §12.1.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_status(run_dir) is not None:
        raise ValueError(f"{run_dir} already finished with status {run_status(run_dir)!r}")
    C = loss_constant(names_for_constant if names_for_constant is not None else data.train_names)
    config = {"arm": "wavenet_d_shared_protocol", "seed": seed, "lr": lr, "max_epochs": max_epochs,
              "batch_size": BATCH_SIZE, "loss_constant_C": C, "model": vars(D_CONFIG),
              "num_train_names": data.train.num_names, "num_train_targets": data.train.num_targets}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text()) != json.loads(json.dumps(config, sort_keys=True)):
            raise ValueError(f"{run_dir} holds a different configuration")
    else:
        _write_json(config_path, config)

    model = make_model(data.vocab, seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    sampler = make_sampler(data.train.num_names, seed)
    history = _read_history(run_dir)
    best_dev, best_epoch, step, start_epoch = math.inf, -1, 0, 0
    last = run_dir / "checkpoint_last.pt"
    if last.is_file():  # resume at the last completed epoch boundary
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, step, best_dev, best_epoch = ck["epoch"], ck["step"], ck["best_dev"], ck["best_epoch"]
        history = [r for r in history if r["epoch"] <= start_epoch]
        (run_dir / "history.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in history))
    else:
        (run_dir / "history.jsonl").write_text("")
        history = []

    def finish(status: str, detail: str | None = None) -> dict[str, Any]:
        meta = {"status": status, "detail": detail, "best_epoch": best_epoch,
                "best_dev_nll": None if best_epoch < 0 else best_dev, "completed_epochs": len(history) - 1,
                "torch": torch.__version__, "threads": [torch.get_num_threads(), torch.get_num_interop_threads()]}
        _write_json(run_dir / "metadata.json", meta)
        return meta

    def end_of_epoch(epoch: int, running_sum: float, seconds: float) -> dict[str, Any]:
        calibrate_batchnorm(model, data.train.windows, batch_size=EVAL_BATCH_SIZE)
        train_acc = evaluate_nll(model, data.train.windows, batch_size=EVAL_BATCH_SIZE)
        dev = evaluate_by_position(model, data.dev)
        if not (math.isfinite(train_acc.mean_nll) and math.isfinite(dev["nll"])):
            raise FloatingPointError(f"non-finite evaluation at epoch {epoch}")
        return {"epoch": epoch, "step": step, "lr": lr, "train_nll": train_acc.mean_nll, "dev_nll": dev["nll"],
                "dev_by_position": dev["by_position"],
                "running_train_nll": None if epoch == 0 else running_sum / data.train.num_targets,
                "param_sha256": param_sha256(model), "seconds": seconds,
                "diverged": epoch >= 1 and train_acc.mean_nll > LN27}

    def commit(row: dict[str, Any]) -> None:
        nonlocal best_dev, best_epoch
        if row["epoch"] >= 1 and row["dev_nll"] < best_dev:
            best_dev, best_epoch = row["dev_nll"], row["epoch"]
            torch.save({"model": model.state_dict(), "epoch": row["epoch"], "dev_nll": row["dev_nll"],
                        "param_sha256": row["param_sha256"]}, run_dir / "model_best.pt")
        with open(run_dir / "history.jsonl", "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        history.append(row)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": row["epoch"],
                    "step": step, "best_dev": best_dev, "best_epoch": best_epoch}, last)

    train_x, train_y = data.train.windows.contexts, data.train.windows.targets
    try:
        if start_epoch == 0 and not history:
            commit(end_of_epoch(0, 0.0, 0.0))  # epoch-0 row: the initialised model
        for epoch in range(start_epoch, max_epochs):
            t0, running = time.perf_counter(), 0.0
            model.train()
            for idx in sampler.batches(epoch):
                rows = data.train.rows(idx)
                optimizer.zero_grad(set_to_none=True)
                nll_sum = F.cross_entropy(model(train_x[rows]), train_y[rows], reduction="sum")
                loss = nll_sum / C
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, step {step}")
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError(f"non-finite gradient at epoch {epoch + 1}, step {step}")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError(f"non-finite parameter at epoch {epoch + 1}, step {step}")
                running += float(nll_sum.detach())
                step += 1
            row = end_of_epoch(epoch + 1, running, time.perf_counter() - t0)
            commit(row)
            if row["diverged"]:
                torch.save({"model": model.state_dict(), "epoch": row["epoch"]}, run_dir / "model_final.pt")
                return finish("failed_diverged", f"train NLL {row['train_nll']:.4f} > ln 27 at epoch {row['epoch']}")
    except FloatingPointError as exc:
        return finish("failed_mechanical", str(exc))
    torch.save({"model": model.state_dict(), "epoch": max_epochs}, run_dir / "model_final.pt")
    return finish("completed")


# ------------------------------------------------------------------ rules (RNN training spec §7.3–§7.4)
def plateau_epoch(curves: Sequence[Sequence[float]]) -> dict[str, Any]:
    """e*: the earliest epoch e with D̄(e) ≤ min D̄ + ε(e), ε(e) = max(10⁻³, SE(e)), SE = SD/√n.

    ``curves[s][e − 1]`` is seed s's dev NLL at epoch e.
    """
    n, E = len(curves), len(curves[0])
    means = [statistics.fmean(c[e] for c in curves) for e in range(E)]
    ses = [statistics.stdev([c[e] for c in curves]) / math.sqrt(n) for e in range(E)]
    best = min(means)
    for e in range(E):
        eps = max(EPSILON_FLOOR, ses[e])
        if means[e] <= best + eps:
            return {"e_star": e + 1, "epsilon": eps, "epsilon_from_se": ses[e] > EPSILON_FLOOR,
                    "mean_dev": means, "se_dev": ses}
    raise AssertionError("unreachable: the minimum itself satisfies the rule")


def select_learning_rate(results: dict[float, dict[str, Any]]) -> dict[str, Any]:
    """Primary rule: the eligible η with the lowest mean full-dev NLL at the last pilot epoch; then the
    near-tie rule: the smallest eligible η within max(10⁻³, SE) of it; boundary flags on the evaluated grid."""
    grid = sorted(results)
    eligible = {lr: r for lr, r in results.items() if r["eligible"]}
    if not eligible:
        return {"selected": None, "reason": "no eligible learning rate", "grid": grid}
    primary = min(eligible, key=lambda lr: eligible[lr]["mean_final_dev"])
    margin = max(EPSILON_FLOOR, eligible[primary]["se_final_dev"])
    tied = [lr for lr in eligible if eligible[lr]["mean_final_dev"] <= eligible[primary]["mean_final_dev"] + margin]
    selected = min(tied)
    return {"primary": primary, "near_tie_margin": margin, "selected": selected, "grid": grid,
            "at_lower_boundary": primary == grid[0], "at_upper_boundary": primary == grid[-1]}


# ------------------------------------------------------------------ stages
def _pilot_dir(root: Path, lr: float, seed: int) -> Path:
    return root / "pilot" / f"lr_{lr:g}" / f"seed_{seed}"


def _run_many(data: StudyData, jobs: list[tuple[int, float, int, Path]]) -> None:
    for seed, lr, epochs, run_dir in jobs:
        status = run_status(run_dir)
        if status is not None:
            print(f"skip {run_dir.relative_to(REPO)} ({status})")
            continue
        t0 = time.perf_counter()
        meta = train_run(data, seed=seed, lr=lr, max_epochs=epochs, run_dir=run_dir)
        print(f"{run_dir.relative_to(REPO)}: {meta['status']}, best dev {meta['best_dev_nll']} at epoch "
              f"{meta['best_epoch']} ({time.perf_counter() - t0:.0f} s)", flush=True)


def pilot_results(root: Path) -> dict[float, dict[str, Any]]:
    out = {}
    for lr_dir in sorted((root / "pilot").glob("lr_*")):
        lr = float(lr_dir.name[3:])
        runs = [lr_dir / f"seed_{s}" for s in PILOT_SEEDS]
        statuses = [run_status(r) for r in runs]
        finals, curves = [], []
        for r in runs:
            hist = [h for h in _read_history(r) if h["epoch"] >= 1]
            curves.append([h["dev_nll"] for h in hist])
            finals.append(hist[-1]["dev_nll"] if len(hist) == PILOT_EPOCHS else math.nan)
        complete = all(s == "completed" for s in statuses)
        out[lr] = {"statuses": statuses, "eligible": complete,
                   "mean_final_dev": statistics.fmean(finals) if complete else math.nan,
                   "se_final_dev": statistics.stdev(finals) / math.sqrt(len(finals)) if complete else math.nan,
                   "final_dev": finals, "curves": curves if complete else None}
    return out


def summarize_pilot(root: Path) -> dict[str, Any]:
    results = pilot_results(root)
    if any(s is None for r in results.values() for s in r["statuses"]):
        raise SystemExit("some pilot runs have not finished")
    choice = select_learning_rate(results)
    summary = {"rule": "RNN training spec §7.3 (selection) and §7.4 (horizon)",
               "learning_rates": {f"{lr:g}": {k: v for k, v in r.items() if k != "curves"} for lr, r in results.items()},
               "selection": choice}
    if choice.get("selected") is not None:
        h = plateau_epoch(results[choice["selected"]]["curves"])
        summary["horizon"] = {"e_star_30": h["e_star"], "epsilon": h["epsilon"],
                              "epsilon_from_se_needs_review": h["epsilon_from_se"],
                              "decision": "E = 30" if h["e_star"] < PILOT_EPOCHS else "extension run X needed"}
    _write_json(root / "pilot_summary.json", summary)
    return summary


def summarize_extension(root: Path, lr: float) -> dict[str, Any]:
    curves, integrity = [], {}
    for s in PILOT_SEEDS:
        ext = [h for h in _read_history(root / "extension" / f"seed_{s}") if h["epoch"] >= 1]
        pil = [h for h in _read_history(_pilot_dir(root, lr, s)) if h["epoch"] >= 1]
        if run_status(root / "extension" / f"seed_{s}") != "completed" or len(ext) != EXTENSION_EPOCHS:
            raise SystemExit(f"extension seed {s} did not complete")
        integrity[s] = [a["param_sha256"] for a in ext[:PILOT_EPOCHS]] == [b["param_sha256"] for b in pil]
        curves.append([h["dev_nll"] for h in ext])
    h = plateau_epoch(curves)
    if not all(integrity.values()):
        decision = "integrity mismatch: stop for review"
    elif h["e_star"] < EXTENSION_EPOCHS:
        decision = f"E = {max(PILOT_EPOCHS, h['e_star'])}"
    else:
        decision = "inconclusive (e*60 = 60): human decision"
    out = {"lr": lr, "integrity_epochs_1_30_bit_identical": integrity, "e_star_60": h["e_star"],
           "epsilon": h["epsilon"], "epsilon_from_se_needs_review": h["epsilon_from_se"], "decision": decision}
    _write_json(root / "extension_summary.json", out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Arm D (WaveNet-BatchNorm) under the shared name-batch protocol")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--lrs", type=float, nargs="+", default=list(LR_GRID))
    sub.add_parser("summarize-pilot")
    p = sub.add_parser("extension")
    p.add_argument("--lr", type=float, required=True)
    p = sub.add_parser("summarize-extension")
    p.add_argument("--lr", type=float, required=True)
    p = sub.add_parser("final")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    root = args.root

    if args.cmd in ("summarize-pilot", "summarize-extension"):
        out = summarize_pilot(root) if args.cmd == "summarize-pilot" else summarize_extension(root, args.lr)
        print(json.dumps({k: v for k, v in out.items() if k != "learning_rates"}, indent=1))
        if args.cmd == "summarize-pilot":
            for lr, r in out["learning_rates"].items():
                print(f"η = {lr}: mean dev NLL at epoch 30 = {r['mean_final_dev']:.4f} "
                      f"(SE {r['se_final_dev']:.4f}), statuses {r['statuses']}")
        return 0

    from training.transformer_study.run_lock import study_lock
    with study_lock(root):
        set_threads()
        data = load_study_data()
        if args.cmd == "pilot":
            jobs = [(s, lr, PILOT_EPOCHS, _pilot_dir(root, lr, s)) for lr in args.lrs for s in PILOT_SEEDS]
        elif args.cmd == "extension":
            jobs = [(s, args.lr, EXTENSION_EPOCHS, root / "extension" / f"seed_{s}") for s in PILOT_SEEDS]
        else:
            jobs = [(s, args.lr, args.epochs, root / "final" / f"seed_{s}") for s in FINAL_SEEDS]
        _run_many(data, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

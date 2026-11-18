"""Training driver for Models 01 and 02 of Transformer Part I (shared name-batch protocol).

Models: ``models/attention/transformer.py`` (specification ``models/attention/transformer_spec.md``).
Protocol: identical to the vanilla RNN and to arm D (``wavenet_d_arm.py``), whose batching, loss
constant and selection rules are imported from there unchanged: batches of 32 whole names from
the RNN's ``EpochSampler`` (data seed ``derive_seed(seed, "data")``), loss = summed NLL / C with
C = B · M / N, plain SGD with constant η, CPU float32.

Stages (from the repository root)::

    python3 training/transformer_study/transformer_runs.py shape-pilot              # Model 02: 3 shapes × η grid × seeds 100–102
    python3 training/transformer_study/transformer_runs.py summarize-shape-pilot
    python3 training/transformer_study/transformer_runs.py lr-pilot                 # Model 01 at the chosen shape
    python3 training/transformer_study/transformer_runs.py summarize-lr-pilot
    python3 training/transformer_study/transformer_runs.py adamw-check              # Model 02, AdamW, feasibility only
    python3 training/transformer_study/transformer_runs.py summarize-adamw-check
    python3 training/transformer_study/transformer_runs.py extension --variant V --lr η     # only if e* = 30
    python3 training/transformer_study/transformer_runs.py summarize-extension --variant V --lr η
    python3 training/transformer_study/transformer_runs.py final --variant V --lr η --epochs E

``shape-pilot`` and ``lr-pilot`` accept ``--lrs`` (a boundary extension) and ``shape-pilot`` accepts
``--shapes``. Completed runs are skipped; interrupted runs resume at the last epoch boundary.
Nothing here reads the test split.
"""
from __future__ import annotations

import argparse
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

from models.attention.transformer import TransformerConfig, TransformerLanguageModel  # noqa: E402
from training.transformer_study.wavenet_d_arm import (  # noqa: E402  (shared protocol, unchanged)
    BATCH_SIZE,
    DUPLICATE_POLICY,
    EPSILON_FLOOR,
    EXTENSION_EPOCHS,
    FINAL_SEEDS,
    LN27,
    LR_GRID,
    PILOT_EPOCHS,
    PILOT_SEEDS,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    _read_history,
    _write_json,
    loss_constant,
    make_sampler,
    param_sha256,
    plateau_epoch,
    run_status,
    select_learning_rate,
    set_threads,
)
from utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from utils.data import NextTokenSequences, load_names, make_next_token_sequences, split_names  # noqa: E402
from utils.reproducibility import capture_rng_state, derive_seed, make_generator  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

SHAPES = ((1, 42), (2, 30), (3, 24))          # (num_layers, d_model), all with 3 heads
NUM_HEADS = 3
MAX_LEN = 16
EVAL_BATCH_SIZE = 1024
ADAMW_LRS = (3e-4, 1e-3, 3e-3)
ADAMW_SETTINGS = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
INTERNALS_DEV_NAMES = 256
DEFAULT_ROOT = REPO / "results" / "experiments" / "transformer_study"


# ------------------------------------------------------------------ data
@dataclass(frozen=True)
class SequenceData:
    vocab: Vocabulary
    train_names: tuple[str, ...]
    train: NextTokenSequences
    dev: NextTokenSequences


def build_sequence_data(train_names: Sequence[str], dev_names: Sequence[str], vocab: Vocabulary) -> SequenceData:
    return SequenceData(vocab=vocab, train_names=tuple(train_names),
                        train=make_next_token_sequences(list(train_names), vocab, max_len=MAX_LEN),
                        dev=make_next_token_sequences(list(dev_names), vocab, max_len=MAX_LEN))


def load_study_data() -> SequenceData:
    splits = split_names(load_names(), fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED, duplicate_policy=DUPLICATE_POLICY)
    vocab = Vocabulary.from_names(splits.train)
    return build_sequence_data(splits.train, splits.dev, vocab)


def make_model(vocab: Vocabulary, variant: str, num_layers: int, d_model: int, seed: int) -> TransformerLanguageModel:
    cfg = TransformerConfig(variant=variant, d_model=d_model, num_layers=num_layers, num_heads=NUM_HEADS, max_len=MAX_LEN)
    return TransformerLanguageModel(vocab, cfg, generator=make_generator(derive_seed(seed, "init")))


def make_optimizer(name: str, model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **ADAMW_SETTINGS)
    raise ValueError(f"unknown optimizer {name!r}")


# ------------------------------------------------------------------ evaluation
def evaluate(model: TransformerLanguageModel, seqs: NextTokenSequences) -> dict[str, Any]:
    """Token-weighted NLL in eval mode with per-position sums and counts (float64).

    Position t is the row of the target: the number of characters read before it (0 = after BOS).
    """
    was_training = model.training
    sums = torch.zeros(MAX_LEN, dtype=torch.float64)
    counts = torch.zeros(MAX_LEN, dtype=torch.long)
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, seqs.inputs.shape[0], EVAL_BATCH_SIZE):
                x = seqs.inputs[start:start + EVAL_BATCH_SIZE]
                y = seqs.targets[start:start + EVAL_BATCH_SIZE]
                m = seqs.mask[start:start + EVAL_BATCH_SIZE]
                nll = F.cross_entropy(model(x).transpose(1, 2), y.clamp(min=0), reduction="none")  # [b, T]
                sums += (nll.double() * m).sum(0)
                counts += m.sum(0)
    finally:
        model.train(was_training)
    total, n = float(sums.sum()), int(counts.sum())
    return {"nll": total / n, "sum": total, "count": n,
            "by_position": {"sum": sums.tolist(), "count": counts.tolist()}}


def internals_summary(model: TransformerLanguageModel, seqs: NextTokenSequences) -> dict[str, Any]:
    """Per layer, over valid rows of a fixed dev subset: mean residual-stream norm before each
    sublayer and mean standard deviation of each LayerNorm input (spec §12)."""
    was_training = model.training
    x, m = seqs.inputs[:INTERNALS_DEV_NAMES], seqs.mask[:INTERNALS_DEV_NAMES]
    try:
        model.eval()
        with torch.no_grad():
            _, rec = model.forward_with_internals(x)
    finally:
        model.train(was_training)

    def norms(ts):
        return [float(t.norm(dim=-1)[m].mean()) for t in ts]

    def stds(ts):
        return [float(t.std(dim=-1, unbiased=False)[m].mean()) for t in ts]

    out = {"residual_norm_before_attention": norms(rec["residual_before_attention"]),
           "residual_norm_before_ffn": norms(rec["residual_before_ffn"]),
           "ln_1_input_std": stds(rec["ln_1_input"]), "ln_2_input_std": stds(rec["ln_2_input"])}
    if "final_ln_input" in rec:
        out["final_ln_input_std"] = stds([rec["final_ln_input"]])[0]
    return out


def _grad_norms(model: TransformerLanguageModel) -> list[float]:
    """Gradient norm of the token embedding, then of each block."""
    groups = [[model.token_embedding]] + [list(b.parameters()) for b in model.blocks]
    return [math.sqrt(sum(float(p.grad.pow(2).sum()) for p in g if p.grad is not None)) for g in groups]


# ------------------------------------------------------------------ one run
def train_run(data: SequenceData, *, variant: str, num_layers: int, d_model: int, seed: int, lr: float,
              max_epochs: int, run_dir: Path, optimizer_name: str = "sgd") -> dict[str, Any]:
    """Train one Transformer run; resumable at epoch boundaries.

    Status ``completed``, ``failed_mechanical`` (category A: non-finite loss, gradient, parameter or
    evaluation) or ``failed_diverged`` (category D: full-train NLL > ln 27 at an epoch ≥ 1), as for
    the RNN and arm D.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_status(run_dir) is not None:
        raise ValueError(f"{run_dir} already finished with status {run_status(run_dir)!r}")
    C = loss_constant(data.train_names)
    model = make_model(data.vocab, variant, num_layers, d_model, seed)
    config = {"variant": variant, "num_layers": num_layers, "d_model": d_model, "num_heads": NUM_HEADS,
              "seed": seed, "lr": lr, "optimizer": optimizer_name,
              "optimizer_settings": ADAMW_SETTINGS if optimizer_name == "adamw" else {"momentum": 0.0, "weight_decay": 0.0},
              "max_epochs": max_epochs, "batch_size": BATCH_SIZE, "loss_constant_C": C,
              "num_parameters": model.num_parameters, "num_train_names": int(data.train.inputs.shape[0]),
              "num_train_targets": data.train.num_targets}
    config = json.loads(json.dumps(config, sort_keys=True))
    config_path = run_dir / "config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"{run_dir} holds a different configuration")
    else:
        _write_json(config_path, config)

    optimizer = make_optimizer(optimizer_name, model, lr)
    sampler = make_sampler(int(data.train.inputs.shape[0]), seed)
    ckpt_cfg = {"model": model.config_dict(), "run": config}
    vocab_dict = data.vocab.to_dict()
    last = run_dir / "checkpoint_last.pt"
    best_dev, best_epoch, step, start_epoch = math.inf, -1, 0, 0
    history = _read_history(run_dir)
    if last.is_file():
        ck = load_checkpoint(last, expected_kind="resume")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch, step = int(ck["epoch"]), int(ck["step"])
        best_dev, best_epoch = ck["extra"]["best_dev"], ck["extra"]["best_epoch"]
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

    def end_of_epoch(epoch: int, running_sum: float, grad_sums: list[float] | None, n_steps: int,
                     seconds: float) -> dict[str, Any]:
        tr, dev = evaluate(model, data.train), evaluate(model, data.dev)
        if not (math.isfinite(tr["nll"]) and math.isfinite(dev["nll"])):
            raise FloatingPointError(f"non-finite evaluation at epoch {epoch}")
        return {"epoch": epoch, "step": step, "lr": lr, "train_nll": tr["nll"], "dev_nll": dev["nll"],
                "dev_by_position": dev["by_position"],
                "running_train_nll": None if epoch == 0 else running_sum / data.train.num_targets,
                "mean_grad_norm": None if grad_sums is None else [g / n_steps for g in grad_sums],
                "internals": internals_summary(model, data.dev), "param_sha256": param_sha256(model),
                "seconds": seconds, "diverged": epoch >= 1 and tr["nll"] > LN27}

    def commit(row: dict[str, Any]) -> None:
        nonlocal best_dev, best_epoch
        if row["epoch"] >= 1 and row["dev_nll"] < best_dev:
            best_dev, best_epoch = row["dev_nll"], row["epoch"]
            save_checkpoint(run_dir / "model_best.pt", kind="best", model=model, epoch=row["epoch"], step=step,
                            best_metric=best_dev, best_metric_name="dev_nll", config=ckpt_cfg, vocab=vocab_dict,
                            extra={"param_sha256": row["param_sha256"]})
        with open(run_dir / "history.jsonl", "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        history.append(row)
        save_checkpoint(last, kind="resume", model=model, optimizer=optimizer, epoch=row["epoch"], step=step,
                        rng_state=capture_rng_state(), config=ckpt_cfg, vocab=vocab_dict,
                        extra={"best_dev": best_dev if best_epoch >= 0 else None, "best_epoch": best_epoch})

    def save_final(epoch: int) -> None:
        save_checkpoint(run_dir / "model_final.pt", kind="final", model=model, epoch=epoch, step=step,
                        config=ckpt_cfg, vocab=vocab_dict)

    X, Y, M = data.train.inputs, data.train.targets, data.train.mask
    try:
        if start_epoch == 0 and not history:
            commit(end_of_epoch(0, 0.0, None, 1, 0.0))
        if best_dev is None:
            best_dev = math.inf
        for epoch in range(start_epoch, max_epochs):
            t0, running = time.perf_counter(), 0.0
            grad_sums = [0.0] * (1 + len(model.blocks))
            n_steps = 0
            model.train()
            for idx in sampler.batches(epoch):
                x, y, m = X[idx], Y[idx], M[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                nll_sum = F.cross_entropy(logits[m], y[m], reduction="sum")
                loss = nll_sum / C
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, step {step}")
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError(f"non-finite gradient at epoch {epoch + 1}, step {step}")
                grad_sums = [a + b for a, b in zip(grad_sums, _grad_norms(model))]
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError(f"non-finite parameter at epoch {epoch + 1}, step {step}")
                running += float(nll_sum.detach())
                step += 1
                n_steps += 1
            row = end_of_epoch(epoch + 1, running, grad_sums, n_steps, time.perf_counter() - t0)
            commit(row)
            if row["diverged"]:
                save_final(row["epoch"])
                return finish("failed_diverged", f"train NLL {row['train_nll']:.4f} > ln 27 at epoch {row['epoch']}")
    except FloatingPointError as exc:
        return finish("failed_mechanical", str(exc))
    save_final(max_epochs)
    return finish("completed")


# ------------------------------------------------------------------ stage layout and summaries
def _shape_tag(num_layers: int, d_model: int) -> str:
    return f"L{num_layers}_d{d_model}"


def _grid_dir(root: Path, stage: str, variant: str, shape: tuple[int, int]) -> Path:
    return root / stage / variant / _shape_tag(*shape)


def grid_results(grid_dir: Path) -> dict[float, dict[str, Any]]:
    """Per η: completed runs over the pilot seeds, mean and SE of dev NLL at the last pilot epoch."""
    out = {}
    for lr_dir in sorted(grid_dir.glob("lr_*")):
        lr = float(lr_dir.name[3:])
        runs = [lr_dir / f"seed_{s}" for s in PILOT_SEEDS]
        statuses = [run_status(r) for r in runs]
        curves = [[h["dev_nll"] for h in _read_history(r) if h["epoch"] >= 1] for r in runs]
        complete = all(s == "completed" for s in statuses) and all(len(c) == PILOT_EPOCHS for c in curves)
        finals = [c[-1] for c in curves] if complete else []
        out[lr] = {"statuses": statuses, "eligible": complete,
                   "mean_final_dev": statistics.fmean(finals) if complete else math.nan,
                   "se_final_dev": statistics.stdev(finals) / math.sqrt(len(finals)) if complete else math.nan,
                   "final_dev": finals, "curves": curves if complete else None}
    return out


def _public(results: dict[float, dict[str, Any]]) -> dict[str, Any]:
    return {f"{lr:g}": {k: v for k, v in r.items() if k != "curves"} for lr, r in results.items()}


def _require_finished(results: dict[float, dict[str, Any]], where: Path) -> None:
    if not results:
        raise SystemExit(f"no runs under {where}")
    if any(s is None for r in results.values() for s in r["statuses"]):
        raise SystemExit(f"some runs under {where} have not finished")


def _horizon(curves: list[list[float]]) -> dict[str, Any]:
    h = plateau_epoch(curves)
    return {"e_star_30": h["e_star"], "epsilon": h["epsilon"], "epsilon_from_se_needs_review": h["epsilon_from_se"],
            "decision": "E = 30" if h["e_star"] < PILOT_EPOCHS else "extension run X needed"}


def choose_shape(per_shape: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    """Lowest mean dev NLL at epoch 30, each shape at its own selected η. Tie-break: shapes within
    max(10⁻³, SE of the paired seed differences to the best) → the shallowest of them."""
    usable = {s: r for s, r in per_shape.items() if r["selection"].get("selected") is not None
              and not (r["selection"]["at_lower_boundary"] or r["selection"]["at_upper_boundary"])}
    if not usable:
        return {"chosen": None, "reason": "no shape with an interior selected η"}
    finals = {s: r["results"][r["selection"]["selected"]]["final_dev"] for s, r in usable.items()}
    means = {s: statistics.fmean(v) for s, v in finals.items()}
    best = min(means, key=means.get)
    tied, margins = [], {}
    for s in usable:
        diffs = [a - b for a, b in zip(finals[s], finals[best])]
        se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if s != best else 0.0
        margins[_shape_tag(*s)] = max(EPSILON_FLOOR, se)
        if means[s] <= means[best] + margins[_shape_tag(*s)]:
            tied.append(s)
    chosen = min(tied)  # (num_layers, d): the shallowest
    return {"best": _shape_tag(*best), "tied": [_shape_tag(*s) for s in sorted(tied)], "margins": margins,
            "chosen": {"num_layers": chosen[0], "d_model": chosen[1],
                       "lr": usable[chosen]["selection"]["selected"]},
            "mean_final_dev": {_shape_tag(*s): m for s, m in means.items()}}


def summarize_shape_pilot(root: Path) -> dict[str, Any]:
    per_shape, out = {}, {"rule": "η per shape: RNN training spec §7.3; shape: lowest mean dev NLL at epoch 30 "
                                  "with the shallower-shape tie-break; horizon: §7.4", "shapes": {}}
    for shape in SHAPES:
        gdir = _grid_dir(root, "shape_pilot", "gpt2_style", shape)
        res = grid_results(gdir)
        _require_finished(res, gdir)
        sel = select_learning_rate(res)
        per_shape[shape] = {"results": res, "selection": sel}
        out["shapes"][_shape_tag(*shape)] = {"learning_rates": _public(res), "selection": sel}
    out["shape_choice"] = choose_shape(per_shape)
    ch = out["shape_choice"].get("chosen")
    if ch:
        out["horizon_model_02"] = _horizon(per_shape[(ch["num_layers"], ch["d_model"])]["results"][ch["lr"]]["curves"])
    _write_json(root / "shape_pilot_summary.json", out)
    return out


def chosen_shape(root: Path) -> tuple[int, int]:
    path = root / "shape_pilot_summary.json"
    if not path.is_file():
        raise SystemExit("run summarize-shape-pilot first")
    ch = json.loads(path.read_text())["shape_choice"].get("chosen")
    if not ch:
        raise SystemExit("the shape pilot chose no shape")
    return ch["num_layers"], ch["d_model"]


def summarize_grid(root: Path, stage: str, variant: str, out_name: str) -> dict[str, Any]:
    shape = chosen_shape(root)
    gdir = _grid_dir(root, stage, variant, shape)
    res = grid_results(gdir)
    _require_finished(res, gdir)
    sel = select_learning_rate(res)
    out = {"variant": variant, "shape": _shape_tag(*shape), "learning_rates": _public(res), "selection": sel}
    if sel.get("selected") is not None and stage != "adamw_check":
        out["horizon"] = _horizon(res[sel["selected"]]["curves"])
    _write_json(root / out_name, out)
    return out


def summarize_extension(root: Path, variant: str, lr: float) -> dict[str, Any]:
    shape = chosen_shape(root)
    stage = "shape_pilot" if variant == "gpt2_style" else "lr_pilot"
    curves, integrity = [], {}
    for s in PILOT_SEEDS:
        ext_dir = root / "extension" / variant / f"seed_{s}"
        ext = [h for h in _read_history(ext_dir) if h["epoch"] >= 1]
        pil = [h for h in _read_history(_grid_dir(root, stage, variant, shape) / f"lr_{lr:g}" / f"seed_{s}")
               if h["epoch"] >= 1]
        if run_status(ext_dir) != "completed" or len(ext) != EXTENSION_EPOCHS:
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
    out = {"variant": variant, "lr": lr, "integrity_epochs_1_30_bit_identical": integrity, "e_star_60": h["e_star"],
           "epsilon": h["epsilon"], "epsilon_from_se_needs_review": h["epsilon_from_se"], "decision": decision}
    _write_json(root / f"extension_summary_{variant}.json", out)
    return out


# ------------------------------------------------------------------ command line
def _run_jobs(data: SequenceData, jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        run_dir = job["run_dir"]
        status = run_status(run_dir)
        if status is not None:
            print(f"skip {run_dir.relative_to(REPO)} ({status})")
            continue
        t0 = time.perf_counter()
        meta = train_run(data, **job)
        print(f"{run_dir.relative_to(REPO)}: {meta['status']}, best dev {meta['best_dev_nll']} at epoch "
              f"{meta['best_epoch']} ({time.perf_counter() - t0:.0f} s)", flush=True)


def _print_grid(summary: dict[str, Any]) -> None:
    for lr, r in summary["learning_rates"].items():
        print(f"  η = {lr}: mean dev NLL at epoch 30 = {r['mean_final_dev']:.4f} (SE {r['se_final_dev']:.4f}), "
              f"statuses {r['statuses']}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transformer Part I: Models 01 and 02 under the shared protocol")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shape-pilot")
    p.add_argument("--lrs", type=float, nargs="+", default=list(LR_GRID))
    p.add_argument("--shapes", type=str, nargs="+", default=[_shape_tag(*s) for s in SHAPES])
    sub.add_parser("summarize-shape-pilot")
    p = sub.add_parser("lr-pilot")
    p.add_argument("--lrs", type=float, nargs="+", default=list(LR_GRID))
    sub.add_parser("summarize-lr-pilot")
    sub.add_parser("adamw-check")
    sub.add_parser("summarize-adamw-check")
    for name in ("extension", "summarize-extension", "final"):
        p = sub.add_parser(name)
        p.add_argument("--variant", choices=["transformer_2017", "gpt2_style"], required=True)
        p.add_argument("--lr", type=float, required=True)
        if name == "final":
            p.add_argument("--epochs", type=int, required=True)
    args = ap.parse_args(argv)
    root = args.root

    if args.cmd.startswith("summarize"):
        if args.cmd == "summarize-shape-pilot":
            out = summarize_shape_pilot(root)
            for tag, s in out["shapes"].items():
                print(f"{tag}: selection {s['selection']}")
                _print_grid(s)
            print(json.dumps({k: out[k] for k in out if k not in ("shapes", "rule")}, indent=1))
        elif args.cmd == "summarize-lr-pilot":
            out = summarize_grid(root, "lr_pilot", "transformer_2017", "lr_pilot_summary.json")
            _print_grid(out)
            print(json.dumps({k: out[k] for k in ("selection", "horizon") if k in out}, indent=1))
        elif args.cmd == "summarize-adamw-check":
            out = summarize_grid(root, "adamw_check", "gpt2_style", "adamw_check_summary.json")
            _print_grid(out)
            print(json.dumps(out["selection"], indent=1))
        else:
            print(json.dumps(summarize_extension(root, args.variant, args.lr), indent=1))
        return 0

    set_threads()
    data = load_study_data()
    jobs: list[dict[str, Any]] = []
    if args.cmd == "shape-pilot":
        shapes = [s for s in SHAPES if _shape_tag(*s) in args.shapes]
        for shape in shapes:
            for lr in args.lrs:
                for s in PILOT_SEEDS:
                    jobs.append(dict(variant="gpt2_style", num_layers=shape[0], d_model=shape[1], seed=s, lr=lr,
                                     max_epochs=PILOT_EPOCHS,
                                     run_dir=_grid_dir(root, "shape_pilot", "gpt2_style", shape) / f"lr_{lr:g}" / f"seed_{s}"))
    elif args.cmd in ("lr-pilot", "adamw-check"):
        shape = chosen_shape(root)
        variant, stage, opt, lrs = (("transformer_2017", "lr_pilot", "sgd", args.lrs) if args.cmd == "lr-pilot"
                                    else ("gpt2_style", "adamw_check", "adamw", list(ADAMW_LRS)))
        for lr in lrs:
            for s in PILOT_SEEDS:
                jobs.append(dict(variant=variant, num_layers=shape[0], d_model=shape[1], seed=s, lr=lr,
                                 max_epochs=PILOT_EPOCHS, optimizer_name=opt,
                                 run_dir=_grid_dir(root, stage, variant, shape) / f"lr_{lr:g}" / f"seed_{s}"))
    else:
        shape = chosen_shape(root)
        seeds, epochs, stage = ((PILOT_SEEDS, EXTENSION_EPOCHS, "extension") if args.cmd == "extension"
                                else (FINAL_SEEDS, args.epochs, "final"))
        for s in seeds:
            jobs.append(dict(variant=args.variant, num_layers=shape[0], d_model=shape[1], seed=s, lr=args.lr,
                             max_epochs=epochs, run_dir=root / stage / args.variant / f"seed_{s}"))
    _run_jobs(data, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

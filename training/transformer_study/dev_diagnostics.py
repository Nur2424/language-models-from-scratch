"""Dev-only diagnostics of Transformer Part I (study plan T-RQ1b, T-RQ3, T-RQ4, T-RQ5).

Reads the ``model_best.pt`` of every final run of four arms and never builds the test split:

* ``rnn``: the published RNN study runs (``results/experiments/rnn_study/L7_research``);
* ``wavenet_d``: arm D re-trained under the shared protocol;
* ``transformer_2017`` (Model 01) and ``gpt2_style`` (Model 02).

Measurements (one shared implementation for every arm that supports them):

* **by_position** (T-RQ1b): dev NLL sums and counts by position = number of characters read.
* **c1** (T-RQ3): the RNN study's BOS-restart algorithm (``rnn_diagnostics.c1_restart``), reproduced
  model-agnostically in float64: restart inputs ``[BOS, x_j … x_n]`` built by the RNN's own
  ``_restart_batch``, the same target sets F₈ (t ≥ 10, d = 1 … 8) and F₇ (t ≥ 9, d = 1 … 7), the same
  name-clustered SEs (``_clustered``) and the same no-op check. Run for the RNN and both Transformers.
  **Position-preserving variant** (Transformers only): the same inputs, with BOS at position 0 and each
  kept character at its original position. Both with the EOS-versus-letter split.
* **attention** (T-RQ4): per layer and head, mean attention weight by distance, weight on BOS and entropy.
* **timing** (T-RQ5): one training epoch of each arm under one thread setting (separate command).

Usage (from the repository root)::

    python3 training/transformer_study/dev_diagnostics.py run       # all arms, seeds 0–9
    python3 training/transformer_study/dev_diagnostics.py summarize
    python3 training/transformer_study/dev_diagnostics.py timing
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.attention.transformer import TransformerLanguageModel  # noqa: E402
from models.recurrent.rnn import RNNLanguageModel  # noqa: E402
from models.recurrent.rnn_diagnostics import TOL_A, _clustered, _restart_batch, stratum_index  # noqa: E402
import training.transformer_study.transformer_runs as tr  # noqa: E402
import training.transformer_study.wavenet_d_arm as wd  # noqa: E402
from utils.data import NextTokenSequences  # noqa: E402

ARMS = ("rnn", "wavenet_d", "transformer_2017", "gpt2_style")
TRANSFORMERS = ("transformer_2017", "gpt2_style")
SEEDS = tuple(range(10))
SETS = {"F8": (10, 8), "F7": (9, 7)}
OUT = tr.DEFAULT_ROOT / "dev_diagnostics"
RNN_RUNS = REPO / "results" / "experiments" / "rnn_study" / "L7_research" / "runs" / "rnn"
CHUNK = 2048


# ------------------------------------------------------------------ loading
def checkpoint_path(arm: str, seed: int) -> Path:
    if arm == "rnn":   # seeds 0–9 are run_001 … run_010 (test_evaluation/freeze.json)
        return RNN_RUNS / f"run_{seed + 1:03d}" / "model_best.pt"
    if arm == "wavenet_d":
        return wd.DEFAULT_ROOT / "final" / f"seed_{seed}" / "model_best.pt"
    return tr.DEFAULT_ROOT / "final" / arm / f"seed_{seed}" / "model_best.pt"


def load_model(arm: str, seed: int, vocab) -> torch.nn.Module:
    path = checkpoint_path(arm, seed)
    if arm == "rnn":
        model = RNNLanguageModel.from_checkpoint(path, expected_vocab=vocab)
    elif arm == "wavenet_d":
        model = wd.make_model(vocab, seed)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    else:
        model = TransformerLanguageModel.from_checkpoint(path, expected_vocab=vocab, expected_variant=arm)
    return model.eval()


# ------------------------------------------------------------------ T-RQ1b
def nll_by_position_sequences(model: torch.nn.Module, dev: NextTokenSequences) -> dict[str, Any]:
    sums = torch.zeros(tr.MAX_LEN, dtype=torch.float64)
    counts = torch.zeros(tr.MAX_LEN, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, dev.inputs.shape[0], 1024):
            x, y, m = (t[start:start + 1024] for t in (dev.inputs, dev.targets, dev.mask))
            nll = F.cross_entropy(model(x).transpose(1, 2), y.clamp(min=0), reduction="none")
            sums += (nll.double() * m).sum(0)
            counts += m.sum(0)
    total, n = float(sums.sum()), int(counts.sum())
    return {"nll": total / n, "sum": total, "count": n, "by_position": {"sum": sums.tolist(), "count": counts.tolist()}}


# ------------------------------------------------------------------ T-RQ3
def _positions_for(part: Sequence[tuple[int, int]], width: int) -> torch.Tensor:
    """Position-preserving positions: BOS at 0; restart index i ≥ 1 holds original input index j − 2 + i."""
    pos = torch.zeros(len(part), width, dtype=torch.long)
    for k, (_, j) in enumerate(part):
        orig = torch.arange(width) + (j - 2)
        orig[0] = 0
        pos[k] = orig.clamp(max=tr.MAX_LEN - 1)
    return torch.cummax(pos, dim=1).values  # padding columns: kept non-decreasing (never read)


def c1_restart(model: torch.nn.Module, dev: NextTokenSequences, *, preserve_positions: bool = False
               ) -> dict[str, Any]:
    """The RNN study's C1 algorithm, model-agnostic: per affected dev target (t ≥ d + 2), ΔNLL_t(d)
    = NLL with only the last d characters kept (restarted after BOS) minus the full NLL. ``model``
    must map padded [B, T] ids to [B, T, 27] logits; float64 recommended (the RNN study used it)."""
    pad = 27
    lengths = [int(v) for v in dev.lengths.tolist()]
    eos = 0

    def run(x: torch.Tensor, part=None) -> torch.Tensor:
        if preserve_positions and part is not None:
            return model(x, positions=_positions_for(part, x.shape[1]))
        return model(x)

    full_nll: dict[tuple[int, int], float] = {}
    max_noop = -math.inf
    with torch.no_grad():
        for start in range(0, len(lengths), CHUNK):
            rows = list(range(start, min(start + CHUNK, len(lengths))))
            T = max(lengths[r] for r in rows)
            z_full = model(dev.inputs[rows, :T])
            logp = torch.log_softmax(z_full, dim=-1)
            for i, r in enumerate(rows):
                for t in range(1, lengths[r] + 1):
                    full_nll[(r, t)] = -float(logp[i, t - 1, int(dev.targets[r, t - 1])])
            noop = [(r, 2) for r in rows if lengths[r] >= 2]
            z_re = run(_restart_batch(dev, lengths, noop, pad), noop)
            for i_re, (r, j) in enumerate(noop):
                i = r - rows[0]
                for t in range(j, lengths[r] + 1):
                    a, b = z_full[i, t - 1], z_re[i_re, t - j + 1]
                    max_noop = max(max_noop, float(((a - b).abs() - (TOL_A + TOL_A * a.abs())).max()))
    specs = [(r, j) for r, n in enumerate(lengths) for j in range(3, n + 1)]
    rec: dict[str, list] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(specs), CHUNK):
            part = specs[start:start + CHUNK]
            x = _restart_batch(dev, lengths, part, pad)
            logp = torch.log_softmax(run(x, part), dim=-1)
            for i, (r, j) in enumerate(part):
                for t in range(j, lengths[r] + 1):
                    y = int(dev.targets[r, t - 1])
                    nll = -float(logp[i, t - j + 1, y])
                    rec["row"].append(r)
                    rec["t"].append(t)
                    rec["d"].append(t - j + 1)
                    rec["eos"].append(y == eos)
                    rec["delta"].append(nll - full_nll[(r, t)])
    out: dict[str, Any] = {"noop_reproduces_full": max_noop <= 0.0, "noop_max_excess": max_noop}
    for label, (tmin, dmax) in SETS.items():
        for split, keep in (("all", None), ("eos", True), ("letter", False)):
            curve = []
            for d in range(1, dmax + 1):
                by_name: dict[int, list[float]] = defaultdict(list)
                for i in range(len(rec["t"])):
                    if rec["d"][i] == d and rec["t"][i] >= tmin and (keep is None or rec["eos"][i] == keep):
                        by_name[rec["row"][i]].append(rec["delta"][i])
                curve.append({"d": d, **_clustered(by_name)})
            out[f"{label}_{split}"] = curve
    return out


# ------------------------------------------------------------------ T-RQ4
def attention_summary(model: TransformerLanguageModel, dev: NextTokenSequences) -> dict[str, Any]:
    """Per layer and head, averaged over valid query rows t ≥ 1: weight by distance k = t − s,
    weight on BOS (s = 0) and entropy (nats)."""
    L, H = model.config.num_layers, model.config.num_heads
    by_dist = torch.zeros(L, H, tr.MAX_LEN, dtype=torch.float64)
    bos = torch.zeros(L, H, dtype=torch.float64)
    ent = torch.zeros(L, H, dtype=torch.float64)
    n = 0
    t_idx = torch.arange(tr.MAX_LEN)
    dist = (t_idx.view(-1, 1) - t_idx.view(1, -1)).clamp(min=0)          # [T, T], k = t − s
    with torch.no_grad():
        for start in range(0, dev.inputs.shape[0], 1024):
            x, m = dev.inputs[start:start + 1024], dev.mask[start:start + 1024].clone()
            m[:, 0] = False                                                # skip t = 0 (attends only to BOS)
            _, rec = model.forward_with_internals(x)
            q = m.view(m.shape[0], 1, -1, 1).double()                      # valid query rows
            n += int(m.sum())
            for layer, w in enumerate(rec["attention"]):
                w = w.double()                                             # [B, H, T, T]
                bos[layer] += (w[..., 0] * q[..., 0]).sum((0, 2))
                ent[layer] += (-(w.clamp_min(1e-300).log() * w).sum(-1) * q[..., 0]).sum((0, 2))
                for k in range(tr.MAX_LEN):
                    sel = (dist == k).double()
                    by_dist[layer, :, k] += ((w * sel).sum(-1) * q[..., 0]).sum(2).sum(0)
    return {"query_rows": n, "weight_by_distance": (by_dist / n).tolist(), "weight_on_bos": (bos / n).tolist(),
            "entropy": (ent / n).tolist()}


# ------------------------------------------------------------------ runner
def diagnose(arm: str, seed: int, data: tr.SequenceData, dev_windows: wd.NameWindows) -> dict[str, Any]:
    model = load_model(arm, seed, data.vocab)
    out: dict[str, Any] = {"arm": arm, "seed": seed, "checkpoint": str(checkpoint_path(arm, seed).relative_to(REPO))}
    if arm == "wavenet_d":
        out["by_position"] = wd.evaluate_by_position(model, dev_windows)
        return out
    out["by_position"] = nll_by_position_sequences(model, data.dev)
    copy = model.double()
    out["c1_restart"] = c1_restart(copy, data.dev)
    if arm in TRANSFORMERS:
        out["c1_position_preserving"] = c1_restart(copy, data.dev, preserve_positions=True)
        out["attention"] = attention_summary(model.float(), data.dev)
    return out


def _mean_se(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "se": statistics.stdev(values) / math.sqrt(len(values)), "n": len(values)}


def summarize(out_dir: Path = OUT) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    per_arm: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(out_dir.glob("*_seed*.json")):
        r = json.loads(p.read_text())
        per_arm[r["arm"]].append(r)
    for arm, runs in per_arm.items():
        runs.sort(key=lambda r: r["seed"])
        s: dict[str, Any] = {"seeds": [r["seed"] for r in runs], "dev_nll": _mean_se([r["by_position"]["nll"] for r in runs])}
        pos = [[a / c if c else math.nan for a, c in zip(r["by_position"]["by_position"]["sum"],
                                                         r["by_position"]["by_position"]["count"])] for r in runs]
        s["dev_nll_by_position"] = [_mean_se([p_[t] for p_ in pos]) if not math.isnan(pos[0][t]) else None
                                    for t in range(len(pos[0]))]
        s["targets_by_position"] = runs[0]["by_position"]["by_position"]["count"]
        for key in ("c1_restart", "c1_position_preserving"):
            if key in runs[0]:
                s[key] = {curve: [_mean_se([r[key][curve][i]["mean"] for r in runs]) | {"d": i + 1,
                                  "targets": runs[0][key][curve][i]["targets"]}
                                  for i in range(len(runs[0][key][curve]))]
                          for curve in runs[0][key] if curve.startswith("F")}
                s[key]["noop_all_runs"] = all(r[key]["noop_reproduces_full"] for r in runs)
        if "attention" in runs[0]:
            s["attention_weight_on_bos"] = torch.tensor([r["attention"]["weight_on_bos"] for r in runs]).mean(0).tolist()
            s["attention_entropy"] = torch.tensor([r["attention"]["entropy"] for r in runs]).mean(0).tolist()
            s["attention_weight_by_distance"] = torch.tensor(
                [r["attention"]["weight_by_distance"] for r in runs]).mean(0).tolist()
        summary[arm] = s
    wd._write_json(out_dir / "summary.json", summary)
    return summary


# ------------------------------------------------------------------ T-RQ5
def time_epoch(arm: str, data: tr.SequenceData, train_windows: wd.NameWindows, *, epochs: int = 2) -> list[float]:
    """Seconds per training epoch (forward, backward, SGD step; no evaluation) under the shared
    protocol, seed 0 initialisation, each arm's selected η is irrelevant to time (η = 0.1)."""
    C = wd.loss_constant(data.train_names)
    if arm == "wavenet_d":
        model = wd.make_model(data.vocab, 0)
    elif arm == "rnn":
        from models.recurrent.rnn import RNNConfig
        from utils.reproducibility import derive_seed, make_generator
        model = RNNLanguageModel(data.vocab, RNNConfig(), generator=make_generator(derive_seed(0, "init")))
    else:
        L, d = tr.chosen_shape(tr.DEFAULT_ROOT)
        model = tr.make_model(data.vocab, arm, L, d, 0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sampler = wd.make_sampler(int(data.train.inputs.shape[0]), 0)
    times = []
    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()
        for idx in sampler.batches(epoch):
            opt.zero_grad(set_to_none=True)
            if arm == "wavenet_d":
                rows = train_windows.rows(idx)
                nll = F.cross_entropy(model(train_windows.windows.contexts[rows]),
                                      train_windows.windows.targets[rows], reduction="sum")
            else:
                x, y, m = data.train.inputs[idx], data.train.targets[idx], data.train.mask[idx]
                nll = F.cross_entropy(model(x)[m], y[m], reduction="sum")
            (nll / C).backward()
            opt.step()
        times.append(time.perf_counter() - t0)
    return times


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transformer Part I: dev-only diagnostics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    sub.add_parser("summarize")
    p = sub.add_parser("timing")
    p.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "summarize":
        s = summarize()
        for arm, v in s.items():
            line = f"{arm}: dev NLL {v['dev_nll']['mean']:.4f} ± {v['dev_nll']['se']:.4f}"
            if "c1_restart" in v:
                line += f"; F8 ΔNLL(d=8) restart {v['c1_restart']['F8_all'][7]['mean']:.4f}"
            if "c1_position_preserving" in v:
                line += f", position-preserving {v['c1_position_preserving']['F8_all'][7]['mean']:.4f}"
            print(line)
        return 0
    wd.set_threads()
    data = tr.load_study_data()
    if args.cmd == "timing":
        train_windows = wd.build_name_windows(data.train_names, data.vocab)
        out = {"threads": [torch.get_num_threads(), torch.get_num_interop_threads()], "torch": torch.__version__,
               "note": "one training epoch (forward, backward, SGD step; no evaluation), seed 0, shared protocol; "
                       "the first epoch is warm-up, the last is reported"}
        for arm in ARMS:
            out[arm] = time_epoch(arm, data, train_windows, epochs=args.epochs)
            print(f"{arm}: {['%.1f s' % t for t in out[arm]]}", flush=True)
        wd._write_json(OUT / "timing.json", out)
        return 0
    from utils.data import load_names, split_names
    splits = split_names(load_names(), fractions=wd.SPLIT_FRACTIONS, seed=wd.SPLIT_SEED, duplicate_policy=wd.DUPLICATE_POLICY)
    dev_windows = wd.build_name_windows(splits.dev, data.vocab)
    for arm in args.arms:
        for seed in args.seeds:
            path = OUT / f"{arm}_seed{seed}.json"
            if path.is_file():
                print(f"skip {path.name}")
                continue
            t0 = time.perf_counter()
            wd._write_json(path, diagnose(arm, seed, data, dev_windows))
            print(f"{arm} seed {seed}: done ({time.perf_counter() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Model 6 research diagnostics: probe, H1, G1 – G3, C1 and the initial-statistics procedure.

Specifications: ``models/recurrent/rnn_diagnostics_spec.md`` v0.3 (approved, D-45),
``models/recurrent/rnn_spec.md`` v0.2 §7.2 and §12 (D-44; §7.2 step 2 amended by D-47),
``models/recurrent/rnn_integration_spec.md`` v0.2 §4 (D-48). Implementation milestones MS3
(functions) and MS4 (plan, records, failure handling, post-hoc measurement) of
``models/recurrent/rnn_implementation_plan.md`` v0.2 (D-49).

Rules that hold for every function here (diag §6, §11; arch §12.1):

* analytic computation runs on a **float64 CPU deep copy** of the model (:func:`float64_copy`),
  never on the training model; states come from the model's own ``_recurrence``;
* no random number is drawn (the single exception is the probe's construction draw, made once
  under its own authorisation with ``derive_seed(42, "m6-probe")``);
* only valid positions enter any statistic; position 1 is flagged (M6-E4);
* G1 – G3 and C1 are exact; no estimator replaces them;
* the test split is never touched (this module never builds or reads it).

Optional diagnostics (``h1_on_dev``, spectral norms at epochs 2^j, the positional data bound
M6-DGN19) are **deferred** by D-49 (P7-6) and are not implemented.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.data import NextTokenSequences
from utils.io import atomic_write_text
from utils.reproducibility import derive_seed, make_generator
from utils.vocabulary import Vocabulary

# =========================================================================== shared definitions (C3, C4)
#: Length strata by valid steps n = L + 1 (diag §3.1, M6-DGN2).
LENGTH_STRATA: tuple[tuple[int, int], ...] = ((2, 5), (6, 7), (8, 9), (10, 16))
#: Label of the bucket for n or t above 16, which the strata do not cover (reported, never dropped).
OVERFLOW_STRATUM = len(LENGTH_STRATA)
#: Canonical parameter order of the hash: U, W, b, V, c (integration spec §4.3).
PARAM_ORDER: tuple[str, ...] = ("input_lookup", "recurrent_weight", "hidden_bias", "output_weight", "output_bias")
SATURATION_THRESHOLD = 0.97
MAX_POSITION = 16
PROBE_SCHEMA = "m6_probe/1"
PROBE_SPLIT_SEED = 42
PROBE_STREAM = "m6-probe"
PROBE_STRATUM_CAP = 256
#: Initialisation-only seeds of the gate (D-45 M6-DGN10; D-47: exclusively these seeds).
INIT_GATE_SEEDS: tuple[int, ...] = tuple(range(1000, 1010))
#: Fixed values of the arch §7.2 procedure.
INIT_GATE_GAIN, INIT_GATE_INPUT_VARIANCE = 1.0, 0.5
QUANTILES = (0.25, 0.5, 0.75)
TOL_A = 1e-9


class ProbeMismatchError(ValueError):
    """The probe does not belong to the training data it is used with (integration spec §4.4, C5)."""


def stratum_index(n: int) -> int:
    """Stratum 0 … 3 of a name with n valid steps; :data:`OVERFLOW_STRATUM` for n > 16.

    n < 2 cannot occur for a name (L ≥ 1) and raises."""
    for i, (lo, hi) in enumerate(LENGTH_STRATA):
        if lo <= n <= hi:
            return i
    if n > LENGTH_STRATA[-1][1]:
        return OVERFLOW_STRATUM
    raise ValueError(f"no length stratum for n = {n} (names have n >= 2)")


def param_sha256(model: RNNLanguageModel) -> str:
    """Canonical parameter hash (integration spec §4.3, C3; training spec §10.4; diag §5.3).

    For each parameter in :data:`PARAM_ORDER`: check its shape against the model's config, take its
    values as float32 on CPU (a float64 parameter is rounded to nearest-even; tests only), and
    append the little-endian, C-order bytes. SHA-256 over the plain concatenation (no separators,
    names or shapes), as 64 lowercase hex digits.
    """
    K, H = model.vocab.num_classes, model.config.hidden_dim
    shapes = {"input_lookup": (K, H), "recurrent_weight": (H, H), "hidden_bias": (H,),
              "output_weight": (K, H), "output_bias": (K,)}
    digest = hashlib.sha256()
    for name in PARAM_ORDER:
        p = getattr(model, name)
        if tuple(p.shape) != shapes[name]:
            raise ValueError(f"{name} has shape {tuple(p.shape)}, expected {shapes[name]}")
        a = p.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        digest.update(a.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


_MODULE_HOOK_DICTS = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
                      "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_backward_hooks",
                      "_backward_pre_hooks", "_state_dict_hooks", "_state_dict_pre_hooks",
                      "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks")


def float64_copy(model: RNNLanguageModel) -> RNNLanguageModel:
    """Sanitised float64 CPU deep copy for analytic diagnostics (diag §11.1).

    Deep copy under ``no_grad``; every module and parameter hook cleared (and asserted empty);
    ``.grad`` dropped; then cast to CPU float64. The training model is only read.
    """
    with torch.no_grad():
        dup = copy.deepcopy(model)
    for module in dup.modules():
        for attr in _MODULE_HOOK_DICTS:
            hooks = getattr(module, attr, None)
            if hooks is not None:
                hooks.clear()
                assert len(hooks) == 0
    for p in dup.parameters():
        p.grad = None
        hooks = getattr(p, "_backward_hooks", None)
        if hooks:
            hooks.clear()
    dup = dup.to(device="cpu", dtype=torch.float64)
    dup.train(model.training)
    return dup


# =========================================================================== probe (diag §3; C5)
def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _plain_split_info(split_info: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(dict(split_info)))


@dataclass(frozen=True)
class Probe:
    """The fixed stratified training probe (diag §3.1). Row order: by stratum, rows ascending."""

    rows: tuple[int, ...]
    names: tuple[str, ...]
    strata: tuple[int, ...]
    halves: tuple[str, ...]
    weights_by_stratum: tuple[float, ...]
    population_by_stratum: tuple[int, ...]
    sample_by_stratum: tuple[int, ...]
    split_info: Mapping[str, Any]
    sha256: str

    def __len__(self) -> int:
        return len(self.rows)

    def weights(self) -> torch.Tensor:
        """Design weights w_β = N_ℓ / n_ℓ per probe row (float64)."""
        return torch.tensor([self.weights_by_stratum[s] for s in self.strata], dtype=torch.float64)

    def half_weights(self, half: str) -> torch.Tensor:
        """Half weights N_ℓ / m_{ℓ,h} for rows of ``half``, 0 for the other half (diag §4.4)."""
        m = defaultdict(int)
        for s, h in zip(self.strata, self.halves):
            if h == half:
                m[s] += 1
        return torch.tensor([self.population_by_stratum[s] / m[s] if h == half else 0.0
                             for s, h in zip(self.strata, self.halves)], dtype=torch.float64)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": PROBE_SCHEMA, "split_info": _plain_split_info(self.split_info),
            "strata": [list(s) for s in LENGTH_STRATA], "stratum_cap": PROBE_STRATUM_CAP,
            "seed": {"split_seed": PROBE_SPLIT_SEED, "stream": PROBE_STREAM},
            "population_by_stratum": list(self.population_by_stratum),
            "sample_by_stratum": list(self.sample_by_stratum),
            "weights_by_stratum": list(self.weights_by_stratum),
            "rows": list(self.rows), "names": list(self.names), "stratum": list(self.strata),
            "half": list(self.halves),
        }


def _probe_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_probe(train: NextTokenSequences, train_names: Sequence[str], split_info: Mapping[str, Any],
                out_path: str | Path) -> Probe:
    """Construct the probe (diag §3.1) and write ``probe.json``. Lifecycle step L2 only.

    Strata by n; n_ℓ = min(256, N_ℓ); one generator ``make_generator(derive_seed(42, "m6-probe"))``
    for S1 … S4 in order; rows of stratum ℓ = sorted(R_ℓ[randperm(|R_ℓ|)[:n_ℓ]]); halves by
    alternating positions; weights N_ℓ/n_ℓ. Refuses to overwrite an existing file.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists: the probe is built once")
    lengths = [int(v) for v in train.lengths.tolist()]
    if len(train_names) != len(lengths):
        raise ProbeMismatchError(f"{len(train_names)} names for {len(lengths)} training rows")
    strata = [stratum_index(n) for n in lengths]
    if OVERFLOW_STRATUM in strata:
        raise ValueError("a training name has n > 16 valid steps; the approved strata do not cover it")
    gen = make_generator(derive_seed(PROBE_SPLIT_SEED, PROBE_STREAM))
    rows, row_strata, halves, pop, sample = [], [], [], [], []
    for s in range(len(LENGTH_STRATA)):
        members = torch.tensor([i for i, st in enumerate(strata) if st == s], dtype=torch.long)
        n_s = min(PROBE_STRATUM_CAP, len(members))
        chosen = sorted(members[torch.randperm(len(members), generator=gen)[:n_s]].tolist()) if n_s else []
        rows += chosen
        row_strata += [s] * len(chosen)
        halves += ["A" if k % 2 == 0 else "B" for k in range(len(chosen))]
        pop.append(len(members))
        sample.append(len(chosen))
    weights = tuple(pop[s] / sample[s] if sample[s] else 0.0 for s in range(len(LENGTH_STRATA)))
    probe = Probe(rows=tuple(rows), names=tuple(train_names[i] for i in rows), strata=tuple(row_strata),
                  halves=tuple(halves), weights_by_stratum=weights, population_by_stratum=tuple(pop),
                  sample_by_stratum=tuple(sample), split_info=_plain_split_info(split_info), sha256="")
    payload = probe.payload()
    sha = _probe_sha(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, _canonical_json({**payload, "sha256": sha}) + "\n")
    return Probe(**{**probe.__dict__, "sha256": sha})


def load_probe(path: str | Path) -> Probe:
    """Read ``probe.json``; verify its schema and its SHA-256 (C5 item 1)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sha = data.pop("sha256", None)
    if data.get("schema") != PROBE_SCHEMA:
        raise ProbeMismatchError(f"{path}: schema {data.get('schema')!r}, expected {PROBE_SCHEMA!r}")
    if sha != _probe_sha(data):
        raise ProbeMismatchError(f"{path}: SHA-256 does not match the content")
    return Probe(rows=tuple(data["rows"]), names=tuple(data["names"]), strata=tuple(data["stratum"]),
                 halves=tuple(data["half"]), weights_by_stratum=tuple(data["weights_by_stratum"]),
                 population_by_stratum=tuple(data["population_by_stratum"]),
                 sample_by_stratum=tuple(data["sample_by_stratum"]), split_info=data["split_info"], sha256=sha)


def validate_probe(probe: Probe, train: NextTokenSequences, train_names: Sequence[str],
                   split_info: Mapping[str, Any]) -> None:
    """Reject a probe that does not belong to these data (integration spec §4.4, C5).

    1. recomputed SHA-256 equals the stored one; 2. ``split_info`` equal; 3. the name at every
    probe row equals the stored name; 4. every row's stratum recomputed from its length;
    5. population counts N_ℓ recomputed from the data."""
    if _probe_sha(probe.payload()) != probe.sha256:
        raise ProbeMismatchError("probe SHA-256 does not match its content")
    if _plain_split_info(split_info) != _plain_split_info(probe.split_info):
        raise ProbeMismatchError("split_info differs from the probe's split_info")
    lengths = [int(v) for v in train.lengths.tolist()]
    if len(train_names) != len(lengths):
        raise ProbeMismatchError(f"{len(train_names)} names for {len(lengths)} training rows")
    for row, name, s in zip(probe.rows, probe.names, probe.strata):
        if not 0 <= row < len(lengths):
            raise ProbeMismatchError(f"probe row {row} is outside the training data")
        if train_names[row] != name:
            raise ProbeMismatchError(f"probe row {row}: name {name!r} != training name {train_names[row]!r}")
        if stratum_index(lengths[row]) != s:
            raise ProbeMismatchError(f"probe row {row}: stratum {s} != recomputed {stratum_index(lengths[row])}")
    pop = [0] * len(LENGTH_STRATA)
    for n in lengths:
        st = stratum_index(n)
        if st < len(LENGTH_STRATA):
            pop[st] += 1
    if tuple(pop) != tuple(probe.population_by_stratum):
        raise ProbeMismatchError(f"population counts {pop} != probe {list(probe.population_by_stratum)}")


# =========================================================================== shared helpers
@dataclass
class DiagnosticResult:
    """One diagnostic's output: JSON-able primary aggregates, raw arrays, run-time check values."""

    name: str
    primary: dict[str, Any]
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)


def weighted_quantiles(values: Sequence[float], weights: Sequence[float], keys: Sequence[tuple],
                       qs: Sequence[float] = QUANTILES) -> list[float | None]:
    """Smallest x_(j) with cumulative weight ≥ q·Σw, ascending, ties broken by ``keys`` (diag §4)."""
    if not values:
        return [None] * len(qs)
    order = sorted(range(len(values)), key=lambda i: (values[i], keys[i]))
    total = math.fsum(weights)
    out, cum, j = [], 0.0, 0
    cums = []
    for i in order:
        cum += weights[i]
        cums.append(cum)
    for q in qs:
        target = q * total
        j = next((k for k, c in enumerate(cums) if c >= target), len(order) - 1)
        out.append(float(values[order[j]]))
    return out


def _log(x: float) -> float:
    return math.log(x) if x > 0 else -math.inf


def _probe_tensors(probe: Probe, train: NextTokenSequences) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    idx = torch.tensor(probe.rows, dtype=torch.long)
    lengths = [int(train.lengths[i]) for i in probe.rows]
    T = max(lengths) if lengths else 1
    return train.inputs[idx, :T].clone(), train.targets[idx, :T].clone(), lengths


def _states(copy_model: RNNLanguageModel, inputs: torch.Tensor, *, pre: bool = False):
    with torch.no_grad():
        return copy_model._recurrence(inputs.to(copy_model.device), return_preactivations=pre)


def _check_copy(copy_model: RNNLanguageModel) -> None:
    if copy_model.recurrent_weight.dtype != torch.float64 or copy_model.device.type != "cpu":
        raise ValueError("diagnostics require the float64 CPU copy from float64_copy()")


# =========================================================================== H1 (diag §4.1; arch §12.2)
def h1_statistics(copy_model: RNNLanguageModel, probe: Probe, train: NextTokenSequences) -> DiagnosticResult:
    """ρ, S, d̄, d² per (t, stratum) over valid (row, unit) cells (primary) and per t,
    population-weighted (secondary); name and element counts; position 1 flagged."""
    _check_copy(copy_model)
    inputs, _, lengths = _probe_tensors(probe, train)
    h = _states(copy_model, inputs).states  # [P, T, H]
    H = h.shape[2]
    w = probe.weights()
    per_cell, per_t = [], []
    valid_cells = 0
    for t in range(1, (max(lengths) if lengths else 0) + 1):
        rows_t = [b for b, n in enumerate(lengths) if n >= t]
        for s in range(len(LENGTH_STRATA)):
            rows = [b for b in rows_t if probe.strata[b] == s]
            if not rows:
                continue
            x = h[rows, t - 1]  # [r, H]
            d = 1.0 - x * x
            per_cell.append({"t": t, "stratum": s, "names": len(rows), "elements": len(rows) * H,
                             "rms": math.sqrt(float((x * x).mean())), "saturation": float((x.abs() > SATURATION_THRESHOLD).double().mean()),
                             "mean_dtanh": float(d.mean()), "mean_sq_dtanh": float((d * d).mean()),
                             "position_1_shared_vector": t == 1})
            valid_cells += len(rows) * H
        x = h[rows_t, t - 1]
        ww = w[rows_t].unsqueeze(1).expand(-1, H)
        d = 1.0 - x * x
        tw = float(ww.sum())
        per_t.append({"t": t, "names": len(rows_t), "elements": len(rows_t) * H,
                      "rms": math.sqrt(float((ww * x * x).sum()) / tw),
                      "saturation": float((ww * (x.abs() > SATURATION_THRESHOLD).double()).sum()) / tw,
                      "mean_dtanh": float((ww * d).sum()) / tw, "mean_sq_dtanh": float((ww * d * d).sum()) / tw})
    expected = sum(lengths) * H
    return DiagnosticResult("H1", {"per_t_stratum": per_cell, "per_t_weighted": per_t},
                            checks={"h1_counts_equal_valid_cells": valid_cells == expected,
                                    "valid_cells": valid_cells, "expected_cells": expected})


# =========================================================================== G1 (diag §4.2; arch §12.3)
def omega_curve(W: torch.Tensor, max_delta: int = MAX_POSITION - 1) -> list[float]:
    """ω(Δ) = ‖W^Δ‖_F / √H for Δ = 1 … max_delta (W only)."""
    H = W.shape[0]
    P, out = torch.eye(H, dtype=W.dtype), []
    for _ in range(max_delta):
        P = W @ P
        out.append(float(torch.linalg.matrix_norm(P)) / math.sqrt(H))
    return out


def g1_products(copy_model: RNNLanguageModel, probe: Probe, train: NextTokenSequences, *,
                spectral: bool) -> DiagnosticResult:
    """Exact P_{t,s} for every probe pair 1 ≤ s < t ≤ n: log φ (always), log σ (``spectral``),
    log δ̂ (M6-DGN18); aggregates per (stratum, Δ) and population-weighted per (s, Δ); ω(Δ)."""
    _check_copy(copy_model)
    inputs, _, lengths = _probe_tensors(probe, train)
    h = _states(copy_model, inputs).states
    W = copy_model.recurrent_weight.detach()
    H = W.shape[0]
    gW2 = float((W * W).sum()) / H
    w = probe.weights().tolist()
    rec = defaultdict(list)
    check_ok = True
    for b, n in enumerate(lengths):
        D = 1.0 - h[b, :n] * h[b, :n]  # D[k-1] = diag of D_k
        log_factor = [0.5 * _log(gW2 * float((D[k - 1] * D[k - 1]).mean())) for k in range(1, n + 1)]
        for s in range(1, n):
            P = D[s].unsqueeze(1) * W  # P_{s+1,s} = D_{s+1} W
            log_delta = 0.0
            for t in range(s + 1, n + 1):
                if t > s + 1:
                    P = D[t - 1].unsqueeze(1) * (W @ P)
                log_delta += log_factor[t - 1]
                fro = float(torch.linalg.matrix_norm(P))
                sig = float(torch.linalg.matrix_norm(P, ord=2)) if spectral else math.nan
                log_phi = _log(fro) - 0.5 * math.log(H)
                if spectral:
                    check_ok &= (fro / math.sqrt(H)) <= sig * (1 + 1e-12)
                for k, v in (("row", probe.rows[b]), ("b", b), ("stratum", probe.strata[b]), ("half", probe.halves[b]),
                             ("s", s), ("t", t), ("log_phi", log_phi), ("log_sigma", _log(sig) if spectral else math.nan),
                             ("log_delta_hat", log_delta), ("weight", w[b])):
                    rec[k].append(v)
    per_stratum, per_start, pooled = [], [], []
    cells = defaultdict(list)
    for i in range(len(rec["t"])):
        cells[("st", rec["stratum"][i], rec["t"][i] - rec["s"][i])].append(i)
        cells[("s", rec["s"][i], rec["t"][i] - rec["s"][i])].append(i)
        cells[("pool", rec["t"][i] - rec["s"][i])].append(i)
    key = lambda i: (rec["stratum"][i], rec["row"][i], rec["t"][i], rec["s"][i])  # noqa: E731
    for ck in sorted(cells, key=lambda k: (k[0], k[1:])):
        idx = cells[ck]
        names = len({rec["b"][i] for i in idx})
        keys = [key(i) for i in idx]
        if ck[0] == "st":
            ones = [1.0] * len(idx)
            row = {"stratum": ck[1], "delta": ck[2], "pairs": len(idx), "names": names,
                   "log_phi_q": weighted_quantiles([rec["log_phi"][i] for i in idx], ones, keys),
                   "log_delta_hat_q": weighted_quantiles([rec["log_delta_hat"][i] for i in idx], ones, keys)}
            if spectral:
                row["log_sigma_q"] = weighted_quantiles([rec["log_sigma"][i] for i in idx], ones, keys)
            per_stratum.append(row)
        else:
            ws = [rec["weight"][i] for i in idx]
            target = per_start if ck[0] == "s" else pooled
            row = {"delta": ck[-1], "pairs": len(idx), "names": names,
                   "log_phi_q": weighted_quantiles([rec["log_phi"][i] for i in idx], ws, keys)}
            if ck[0] == "s":
                row["start"] = ck[1]
            target.append(row)
    raw = {f"g1_{k}": np.asarray(v, dtype=np.float64 if k.startswith("log") or k == "weight" else
                                 (np.str_ if k == "half" else np.int64)) for k, v in rec.items() if k != "b"}
    return DiagnosticResult("G1", {"spectral": spectral, "per_stratum_delta": per_stratum,
                                   "per_start_delta_weighted": per_start, "pooled_delta_weighted": pooled,
                                   "omega": omega_curve(W), "g_W": math.sqrt(gW2)},
                            raw=raw, checks={"phi_le_sigma": bool(check_ok) if spectral else None})


# =========================================================================== G2 (diag §4.3; arch §12.4)
def _stable_residual(logits: torch.Tensor, y: int) -> torch.Tensor:
    """g = p − e_y with g_y = −Σ_{k≠y} p_k (no cancellation when p_y ≈ 1; diag §4.3 step 1)."""
    p = torch.exp(torch.log_softmax(logits, dim=0))
    g = p.clone()
    others = torch.ones_like(p, dtype=torch.bool)
    others[y] = False
    g[y] = -p[others].sum()  # summed directly: never 1 − p_y, which cancels when p_y ≈ 1
    return g


def _unit_and_norm(r: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    """Scaled norm (diag §4.3 step 3): returns (r/‖r‖, ‖r‖), or (None, 0.0) if r is exactly zero."""
    m = float(r.abs().max())
    if m == 0.0:
        return None, 0.0
    v = r / m
    nv = float(torch.linalg.vector_norm(v))
    return v / nv, m * nv


def g2_attenuation(copy_model: RNNLanguageModel, probe: Probe, train: NextTokenSequences) -> DiagnosticResult:
    """R_{t,s} = ‖P_{t,s}ᵀ r_t‖/‖r_t‖ for every probe target t and s ≤ t (backward vector chains),
    with exact-zero exclusion (counted), per (stratum, Δ) quantiles, R̄(Δ) and min ‖r‖."""
    _check_copy(copy_model)
    inputs, targets, lengths = _probe_tensors(probe, train)
    out = _states(copy_model, inputs)
    h = out.states
    with torch.no_grad():
        z = copy_model._logits(h)
    W, V = copy_model.recurrent_weight.detach(), copy_model.output_weight.detach()
    w = probe.weights().tolist()
    rec = defaultdict(list)
    excluded, min_norm = defaultdict(int), math.inf
    for b, n in enumerate(lengths):
        D = 1.0 - h[b, :n] * h[b, :n]
        for t in range(1, n + 1):
            r = V.T @ _stable_residual(z[b, t - 1], int(targets[b, t - 1]))
            u, rnorm = _unit_and_norm(r)
            if u is None:
                excluded[probe.strata[b]] += 1
                continue
            min_norm = min(min_norm, rnorm)
            v = u
            for s in range(t, 0, -1):
                if s < t:
                    v = W.T @ (D[s] * v)  # J_{s+1}ᵀ = Wᵀ D_{s+1}
                R = float(torch.linalg.vector_norm(v))
                for k, val in (("row", probe.rows[b]), ("stratum", probe.strata[b]), ("s", s), ("t", t),
                               ("log_R", _log(R)), ("r_norm", rnorm), ("weight", w[b]), ("R", R)):
                    rec[k].append(val)
    per_cell, rbar = [], []
    cells = defaultdict(list)
    for i in range(len(rec["t"])):
        cells[(rec["stratum"][i], rec["t"][i] - rec["s"][i])].append(i)
    for (s, d) in sorted(cells):
        idx = cells[(s, d)]
        keys = [(rec["stratum"][i], rec["row"][i], rec["t"][i], rec["s"][i]) for i in idx]
        per_cell.append({"stratum": s, "delta": d, "pairs": len(idx),
                         "log_R_q": weighted_quantiles([rec["log_R"][i] for i in idx], [1.0] * len(idx), keys),
                         "R_bar": math.fsum(rec["r_norm"][i] * rec["R"][i] for i in idx)
                         / math.fsum(rec["r_norm"][i] for i in idx)})
    by_delta = defaultdict(list)
    for i in range(len(rec["t"])):
        by_delta[rec["t"][i] - rec["s"][i]].append(i)
    for d in sorted(by_delta):
        idx = by_delta[d]
        rbar.append({"delta": d, "pairs": len(idx),
                     "R_bar_weighted": math.fsum(rec["weight"][i] * rec["r_norm"][i] * rec["R"][i] for i in idx)
                     / math.fsum(rec["weight"][i] * rec["r_norm"][i] for i in idx)})
    raw = {f"g2_{k}": np.asarray(v, dtype=np.float64 if k in ("log_R", "r_norm", "weight", "R") else np.int64)
           for k, v in rec.items()}
    return DiagnosticResult("G2", {"per_stratum_delta": per_cell, "pooled_delta_weighted": rbar,
                                   "excluded_exact_zero_by_stratum": {str(k): v for k, v in sorted(excluded.items())},
                                   "min_r_norm": None if min_norm == math.inf else min_norm}, raw=raw)


# =========================================================================== G3 (diag §4.4; arch §12.5)
def _weighted_loss(copy_model: RNNLanguageModel, inputs: torch.Tensor, targets: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(copy_model(inputs), dim=-1)
    valid = targets != -100
    nll = -logp.gather(2, targets.clamp_min(0).unsqueeze(2)).squeeze(2) * valid
    return (nll.sum(dim=1) * weights).sum()


def _share_summary(num: dict[int, float], pos: dict[int, float], neg: dict[int, float], mass: dict[int, float],
                   total_norm: float, G: torch.Tensor) -> dict[str, Any]:
    G2 = float((G * G).sum())
    deltas = sorted(set(num) | set(mass))
    share = {d: num.get(d, 0.0) / G2 for d in deltas} if G2 > 0 else {d: math.nan for d in deltas}
    return {
        "share": {str(d): share[d] for d in deltas},
        "share_pos": {str(d): pos.get(d, 0.0) / G2 if G2 > 0 else math.nan for d in deltas},
        "share_neg": {str(d): neg.get(d, 0.0) / G2 if G2 > 0 else math.nan for d in deltas},
        "mass": {str(d): mass.get(d, 0.0) for d in deltas},
        "kappa": math.sqrt(G2) / total_norm if total_norm > 0 else math.nan,
        "long_share_ge3": math.fsum(v for d, v in share.items() if d >= 3),
        "grad_norm": math.sqrt(G2),
        "share_sum": math.fsum(share.values()),
    }


def g3_decomposition(copy_model: RNNLanguageModel, probe: Probe, train: NextTokenSequences) -> DiagnosticResult:
    """Weighted probe estimate of the recurrent-gradient decomposition, full probe and halves A/B,
    with split-half agreement and the run-time identity checks (diag §4.4, §10.2).

    Pass 1: Ĝ, Ĝ_A, Ĝ_B by autograd on the copy (inside ``enable_grad``). Pass 2: the G2 vector
    chains u_{t,k} = P_{t,k}ᵀ r_t; T_{t,k} = w (D_k u) h_{k−1}ᵀ is never stored; Σ T, ⟨T, Ĝ⟩ =
    w (D_k u)ᵀ(Ĝ h_{k−1}) and ‖T‖ = w ‖D_k u‖‖h_{k−1}‖ are accumulated per Δ."""
    _check_copy(copy_model)
    inputs, targets, lengths = _probe_tensors(probe, train)
    W = copy_model.recurrent_weight
    sets = {"full": probe.weights(), "A": probe.half_weights("A"), "B": probe.half_weights("B")}
    G = {}
    with torch.enable_grad():
        for key, wts in sets.items():
            (G[key],) = torch.autograd.grad(_weighted_loss(copy_model, inputs, targets, wts), W)
            G[key] = G[key].detach()
    h = _states(copy_model, inputs).states
    with torch.no_grad():
        z = copy_model._logits(h)
    Wd, V = W.detach(), copy_model.output_weight.detach()
    acc = {k: {"sumT": torch.zeros_like(Wd), "num": defaultdict(float), "pos": defaultdict(float),
               "neg": defaultdict(float), "mass": defaultdict(float), "norm": 0.0} for k in sets}
    rec = defaultdict(list)
    for b, n in enumerate(lengths):
        hb = h[b, :n]
        D = 1.0 - hb * hb
        member = {k: float(sets[k][b]) for k in sets}
        Gh = {k: (G[k] @ hb.T) for k in sets}  # column k-2 holds Ĝ h_{k-1}
        for t in range(2, n + 1):
            r = V.T @ _stable_residual(z[b, t - 1], int(targets[b, t - 1]))
            u = r
            for k in range(t, 1, -1):
                if k < t:
                    u = Wd.T @ (D[k] * u)  # u = P_{t,k}ᵀ r_t
                du = D[k - 1] * u
                hprev = hb[k - 2]
                tnorm = float(torch.linalg.vector_norm(du)) * float(torch.linalg.vector_norm(hprev))
                d = t - k
                inner = {}
                for key in sets:
                    wk = member[key]
                    if wk == 0.0:
                        continue
                    acc[key]["sumT"] += wk * torch.outer(du, hprev)
                    ip = wk * float(du @ Gh[key][:, k - 2])
                    inner[key] = ip
                    acc[key]["num"][d] += ip
                    acc[key]["pos" if ip > 0 else "neg"][d] += ip if ip != 0 else 0.0
                    acc[key]["mass"][d] += wk * tnorm
                    acc[key]["norm"] += wk * tnorm
                for kk, val in (("row", probe.rows[b]), ("stratum", probe.strata[b]), ("half", probe.halves[b]),
                                ("t", t), ("k", k), ("tnorm_unweighted", tnorm), ("inner_full", inner.get("full", math.nan)),
                                ("inner_half", inner.get(probe.halves[b], math.nan))):
                    rec[kk].append(val)
    out, checks = {}, {}
    for key in sets:
        a = acc[key]
        out[key] = _share_summary(a["num"], a["pos"], a["neg"], a["mass"], a["norm"], G[key])
        gnorm = float(torch.linalg.matrix_norm(G[key]))
        err = float(torch.linalg.matrix_norm(a["sumT"] - G[key]))
        checks[f"identity_{key}"] = {"rel_error": err / gnorm if gnorm > 0 else err,
                                     "pass": err <= TOL_A * gnorm}
        kappa = out[key]["kappa"]
        checks[f"share_sum_{key}"] = {"abs_error": abs(out[key]["share_sum"] - 1.0),
                                      "pass": abs(out[key]["share_sum"] - 1.0) <= TOL_A / kappa if kappa and kappa > 0 else False}
    gA, gB = G["A"], G["B"]
    na, nb = float(torch.linalg.matrix_norm(gA)), float(torch.linalg.matrix_norm(gB))
    cos = float((gA * gB).sum()) / (na * nb) if na > 0 and nb > 0 else math.nan
    deltas = sorted(set(out["A"]["share"]) | set(out["B"]["share"]), key=int)
    agreement = {"cos_halves": cos, "per_delta": []}
    for d in deltas:
        pa, pb = out["A"]["share"].get(d, 0.0), out["B"]["share"].get(d, 0.0)
        agreement["per_delta"].append({"delta": int(d), "diff": pa - pb, "spread": abs(pa - pb) / 2,
                                       "sign_reproduced": (pa > 0) == (pb > 0) and pa != 0 and pb != 0})
    for q in ("long_share_ge3", "kappa"):
        pa, pb = out["A"][q], out["B"][q]
        agreement[q] = {"diff": pa - pb, "spread": abs(pa - pb) / 2,
                        "sign_reproduced": (pa > 0) == (pb > 0) and pa != 0 and pb != 0}
    raw = {f"g3_{k}": np.asarray(v, dtype=np.str_ if k == "half" else
                                 (np.float64 if k.startswith(("tnorm", "inner")) else np.int64)) for k, v in rec.items()}
    return DiagnosticResult("G3", {"label": "probe estimate", "full": out["full"], "A": out["A"], "B": out["B"],
                                   "agreement": agreement}, raw=raw, checks=checks)


# =========================================================================== C1 (diag §4.5; arch §12.6)
def _clustered(values_by_name: Mapping[int, list[float]]) -> dict[str, Any]:
    """Mean and name-clustered SE: SE² = (N_c/(N_c − 1)) Σ (s_i − mean·m_i)² / (Σ m_i)² (diag §4.5)."""
    names = sorted(values_by_name)
    s = [math.fsum(values_by_name[i]) for i in names]
    m = [len(values_by_name[i]) for i in names]
    total_m = sum(m)
    if total_m == 0:
        return {"targets": 0, "names": 0, "mean": None, "se": None}
    mean = math.fsum(s) / total_m
    nc = len(names)
    se = math.sqrt((nc / (nc - 1)) * math.fsum((si - mean * mi) ** 2 for si, mi in zip(s, m)) / total_m ** 2) \
        if nc > 1 else None
    return {"targets": total_m, "names": nc, "mean": mean, "se": se}


def _restart_batch(dev: NextTokenSequences, lengths: Sequence[int], part: Sequence[tuple[int, int]],
                   pad: int) -> torch.Tensor:
    """Restart inputs [BOS, x_j … x_n] for each (dev row r, start j) of ``part`` (diag §4.5), padded."""
    width = max(lengths[r] - j + 2 for r, j in part)
    x = torch.full((len(part), width), pad, dtype=torch.long)
    for i, (r, j) in enumerate(part):
        seq = dev.inputs[r, j - 1: lengths[r]]
        x[i, 0] = 0
        x[i, 1: 1 + len(seq)] = seq
    return x


def c1_restart(copy_model: RNNLanguageModel, dev: NextTokenSequences, *, chunk: int = 2048) -> DiagnosticResult:
    """BOS-restart ΔNLL_t(d) for every affected dev target (t ≥ d + 2) by the shared-start
    algorithm (starts j = 3 … n); F₈ (t ≥ 10, d = 1 … 8) primary and F₇ (t ≥ 9, d = 1 … 7)
    secondary with name-clustered SEs; the (t, d, stratum) matrix; per-target values; and the
    run-time no-op check (a j = 2 restart reproduces the full logits)."""
    _check_copy(copy_model)
    pad = copy_model.vocab.pad_id
    lengths = [int(v) for v in dev.lengths.tolist()]
    full_nll = {}
    max_noop_dev = -math.inf
    with torch.no_grad():
        for start in range(0, len(lengths), chunk):
            rows = list(range(start, min(start + chunk, len(lengths))))
            T = max(lengths[r] for r in rows)
            x = dev.inputs[rows, :T]
            logp = torch.log_softmax(copy_model(x), dim=-1)
            for i, r in enumerate(rows):
                for t in range(1, lengths[r] + 1):
                    full_nll[(r, t)] = -float(logp[i, t - 1, int(dev.targets[r, t - 1])])
            # no-op check (diag §10.2 row 4): a j = 2 restart, built by the same builder and read through
            # the same position mapping p = t − j + 2 as every j ≥ 3 restart, reproduces the full logits
            z_full = copy_model(x)
            noop = [(r, 2) for r in rows if lengths[r] >= 2]
            if noop:
                z_re = copy_model(_restart_batch(dev, lengths, noop, pad))
                for i_re, (r, j) in enumerate(noop):
                    i = r - rows[0]
                    for t in range(j, lengths[r] + 1):
                        a, b_ = z_full[i, t - 1], z_re[i_re, t - j + 1]
                        excess = float(((a - b_).abs() - (TOL_A + TOL_A * a.abs())).max())
                        max_noop_dev = max(max_noop_dev, excess)
    specs = [(r, j) for r, n in enumerate(lengths) for j in range(3, n + 1)]
    rec = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(specs), chunk):
            part = specs[start: start + chunk]
            x = _restart_batch(dev, lengths, part, pad)
            logp = torch.log_softmax(copy_model(x), dim=-1)
            for i, (r, j) in enumerate(part):
                for t in range(j, lengths[r] + 1):
                    p_pos = t - j + 2  # restart position holding the state after x_t
                    d = t - j + 1
                    nll = -float(logp[i, p_pos - 1, int(dev.targets[r, t - 1])])
                    for k, v in (("row", r), ("t", t), ("d", d), ("stratum", stratum_index(lengths[r])),
                                 ("delta_nll", nll - full_nll[(r, t)])):
                        rec[k].append(v)
    sets = {"F8": (10, 8), "F7": (9, 7)}
    curves = {}
    for label, (tmin, dmax) in sets.items():
        curves[label] = []
        for d in range(1, dmax + 1):
            by_name = defaultdict(list)
            for i in range(len(rec["t"])):
                if rec["d"][i] == d and rec["t"][i] >= tmin:
                    by_name[rec["row"][i]].append(rec["delta_nll"][i])
            curves[label].append({"d": d, **_clustered(by_name)})
    matrix = defaultdict(lambda: [0.0, 0])
    for i in range(len(rec["t"])):
        cell = matrix[(rec["t"][i], rec["d"][i], rec["stratum"][i])]
        cell[0] += rec["delta_nll"][i]
        cell[1] += 1
    raw = {f"c1_{k}": np.asarray(v, dtype=np.float64 if k == "delta_nll" else np.int64) for k, v in rec.items()}
    return DiagnosticResult(
        "C1", {"F8": curves["F8"], "F7": curves["F7"],
               "matrix": [{"t": t, "d": d, "stratum": s, "sum": v[0], "count": v[1]}
                          for (t, d, s), v in sorted(matrix.items())],
               "dev_targets": sum(lengths)},
        raw=raw, checks={"noop_restart_reproduces_full": max_noop_dev <= 0.0,
                             "noop_max_excess": None if max_noop_dev == -math.inf else max_noop_dev})


# =========================================================================== arch §7.2 (mean field; initial statistics)
def _gauss_hermite_expectation(f, q: float, nodes: int = 200) -> float:
    """E[f(√q Z)], Z ~ N(0, 1), by Gauss–Hermite quadrature (200 nodes, arch §7.2)."""
    x, wts = np.polynomial.hermite.hermgauss(nodes)
    return float(np.sum(wts * f(np.sqrt(2.0 * q) * x)) / math.sqrt(math.pi))


def mean_field_table(gain: float, input_variance: float, *, max_position: int = MAX_POSITION) -> list[dict[str, float]]:
    """Per-position mean field with b = 0, h_0 = 0 (arch §7.2 item 6): q_1 = s²;
    q_t = g² E[tanh²(√q_{t−1} Z)] + s²; SD √q_t; saturation 2Φ(−atanh(0.97)/√q_t);
    E[tanh′]; χ_t = g² E[(1 − tanh²(√q_t Z))²]."""
    rows, q = [], input_variance
    for t in range(1, max_position + 1):
        if t > 1:
            q = gain ** 2 * _gauss_hermite_expectation(lambda v: np.tanh(v) ** 2, q) + input_variance
        sd = math.sqrt(q)
        sat = math.erfc(math.atanh(SATURATION_THRESHOLD) / sd / math.sqrt(2.0))  # 2Φ(−a/sd)
        rows.append({"t": t, "q": q, "sd": sd, "saturation": sat,
                     "mean_dtanh": _gauss_hermite_expectation(lambda v: 1.0 - np.tanh(v) ** 2, q),
                     "chi": gain ** 2 * _gauss_hermite_expectation(lambda v: (1.0 - np.tanh(v) ** 2) ** 2, q)})
    return rows


@dataclass(frozen=True)
class InitialStatisticsReport:
    path: Path
    sha256: str
    seeds: tuple[int, ...]


def initial_statistics(vocab: Vocabulary, config: RNNConfig, probe: Probe, train: NextTokenSequences,
                       train_names: Sequence[str], split_info: Mapping[str, Any],
                       seeds: Sequence[int] = INIT_GATE_SEEDS, *, out_dir: str | Path) -> InitialStatisticsReport:
    """The arch §7.2 procedure as amended by D-47: seeds 1000 – 1009 exclusively, g = 1, s² = 0.5,
    no optimiser step. Writes ``initial_statistics.json`` (+ ``initial_statistics_raw.npz``) under
    ``out_dir`` and returns their location. Observations only: nothing here selects or changes a value.
    """
    if tuple(seeds) != INIT_GATE_SEEDS:
        raise ValueError(f"the initialisation gate uses seeds 1000 – 1009 exclusively (D-47), got {tuple(seeds)}")
    if (config.recurrent_gain, config.input_init_variance) != (INIT_GATE_GAIN, INIT_GATE_INPUT_VARIANCE):
        raise ValueError("arch §7.2 fixes g = 1 and s² = 0.5 for the procedure")
    validate_probe(probe, train, train_names, split_info)
    out_dir = Path(out_dir)
    json_path, npz_path = out_dir / "initial_statistics.json", out_dir / "initial_statistics_raw.npz"
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"{out_dir} already holds an initial-statistics report")
    g, s2 = config.recurrent_gain, config.input_init_variance
    table = mean_field_table(g, s2)
    log_chi = [0.5 * math.log(r["chi"]) for r in table]
    inputs, _, lengths = _probe_tensors(probe, train)
    w = probe.weights().tolist()
    per_seed, raw = [], {}
    for seed in INIT_GATE_SEEDS:
        model = RNNLanguageModel(vocab, config, generator=make_generator(derive_seed(seed, "init")))
        cm = float64_copy(model)
        out = _states(cm, inputs, pre=True)
        h, a = out.states, out.preactivations
        H = h.shape[2]
        positions = []
        for t in range(1, max(lengths) + 1):
            for s in range(len(LENGTH_STRATA)):
                rows = [b for b, n in enumerate(lengths) if n >= t and probe.strata[b] == s]
                if not rows:
                    continue
                at, ht = a[rows, t - 1], h[rows, t - 1]
                d = 1.0 - ht * ht
                positions.append({"t": t, "stratum": s, "names": len(rows), "elements": len(rows) * H,
                                  "pre_mean": float(at.mean()), "pre_var": float(((at - at.mean()) ** 2).mean()),
                                  "rms": math.sqrt(float((ht * ht).mean())),
                                  "saturation": float((ht.abs() > SATURATION_THRESHOLD).double().mean()),
                                  "mean_dtanh": float(d.mean()), "mean_sq_dtanh": float((d * d).mean()),
                                  "predicted": table[t - 1]})
        g1 = g1_products(cm, probe, train, spectral=True)
        Wc = cm.recurrent_weight.detach()
        log_phi_hat_factor = []
        for b, n in enumerate(lengths):
            D = 1.0 - h[b, :n] * h[b, :n]
            log_phi_hat_factor.append([0.5 * _log(g * g * float((D[k] * D[k]).mean())) for k in range(n)])
        pairs = defaultdict(list)
        r = g1.raw
        index_of_row = {row: b for b, row in enumerate(probe.rows)}
        lp_hat, lp_mf = [], []
        for i in range(len(r["g1_t"])):
            b, s_, t_ = index_of_row[int(r["g1_row"][i])], int(r["g1_s"][i]), int(r["g1_t"][i])
            hat = math.fsum(log_phi_hat_factor[b][k - 1] for k in range(s_ + 1, t_ + 1))
            mf = math.fsum(log_chi[k - 1] for k in range(s_ + 1, t_ + 1))
            lp_hat.append(hat)
            lp_mf.append(mf)
            pairs[(t_ - s_, s_)].append((float(r["g1_log_phi"][i]), hat, mf, w[b]))
        by_pair = []
        for (d, s_), vals in sorted(pairs.items()):
            ws = math.fsum(v[3] for v in vals)
            mean = lambda j: math.fsum(v[j] * v[3] for v in vals) / ws  # noqa: E731
            obs, hat, mf = mean(0), mean(1), mean(2)
            by_pair.append({"delta": d, "start": s_, "pairs": len(vals), "log_phi": obs, "log_phi_hat": hat,
                            "log_phi_mf": mf, "obs_minus_mf": obs - mf, "obs_minus_hat": obs - hat,
                            "hat_minus_mf": hat - mf})
        sv = torch.linalg.svdvals(Wc).tolist()
        U = cm.input_lookup.detach()
        per_seed.append({"seed": seed, "positions": positions, "pairs_by_delta_start": by_pair,
                         "W_singular_values": sv,
                         "W_orthogonality_defect": float(torch.linalg.matrix_norm(Wc.T @ Wc / g ** 2 - torch.eye(H, dtype=torch.float64))),
                         "U_mean": float(U.mean()), "U_var": float(U.var(unbiased=True)),
                         "g1_per_stratum_delta": g1.primary["per_stratum_delta"],
                         "param_sha256": param_sha256(model)})
        for k, v in g1.raw.items():
            raw[f"seed{seed}_{k}"] = v
        raw[f"seed{seed}_log_phi_hat"] = np.asarray(lp_hat)
        raw[f"seed{seed}_log_phi_mf"] = np.asarray(lp_mf)
    spread = _across_seed_spread(per_seed)
    report = {"procedure": "rnn_spec.md §7.2 as amended by D-47", "seeds": list(INIT_GATE_SEEDS),
              "fixed_values": {"g": g, "s2": s2}, "probe_sha256": probe.sha256, "mean_field": table,
              "per_seed": per_seed, "across_seed_spread": spread,
              "note": "Observations [O] against predeclared predictions [T]; no thresholds; nothing is changed."}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **raw)
    text = json.dumps(report, sort_keys=True, indent=1, allow_nan=True) + "\n"
    atomic_write_text(json_path, text)
    return InitialStatisticsReport(path=json_path, sha256=hashlib.sha256(text.encode()).hexdigest(),
                                   seeds=INIT_GATE_SEEDS)


def _across_seed_spread(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """SD across seeds of each per-position statistic and each (Δ, s) observed-minus-predicted value."""
    def sd(vals):
        if len(vals) < 2:
            return None
        m = math.fsum(vals) / len(vals)
        return math.sqrt(math.fsum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    pos, pair = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for seed in per_seed:
        for p in seed["positions"]:
            for k in ("pre_mean", "pre_var", "rms", "saturation", "mean_dtanh", "mean_sq_dtanh"):
                pos[(p["t"], p["stratum"])][k].append(p[k])
        for p in seed["pairs_by_delta_start"]:
            for k in ("log_phi", "obs_minus_mf", "obs_minus_hat", "hat_minus_mf"):
                pair[(p["delta"], p["start"])][k].append(p[k])
    return {"positions": [{"t": t, "stratum": s, **{k: sd(v) for k, v in d.items()}} for (t, s), d in sorted(pos.items())],
            "pairs": [{"delta": d, "start": s, **{k: sd(v) for k, v in dd.items()}} for (d, s), dd in sorted(pair.items())]}


# ===========================================================================
# MS4: diagnostics plan, records, failure handling, O3 and post-hoc measurement
# (diag §5, §8, §10.2, §11; integration spec §4.1, §4.2, §4.6, §4.7; D-49 O-3, P7-4, P7-6)
# ===========================================================================
import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

DIAGNOSTICS_SCHEMA = "m6_diagnostics/3"
#: Research-diagnostic records of a run live in ``<run_dir>/research_diagnostics/`` (D-49, O-3);
#: ``<run_dir>/diagnostics/`` holds the training step records (``steps.jsonl``).
RESEARCH_DIAGNOSTICS_DIR = "research_diagnostics"
POINT_KINDS = ("start", "early", "epoch_end", "posthoc_best")
#: Fields that measure wall time and are excluded from bitwise-identity comparisons.
TIMING_FIELDS = ("seconds",)
#: Mandatory diagnostic set per point kind (diag §5.1, §13; integration §4.2).
_DIAGNOSTICS_BY_KIND = {"start": ("H1", "G1"), "early": ("H1", "G1", "G2", "G3", "C1"),
                        "epoch_end": ("H1", "G1", "G2", "G3", "C1"), "posthoc_best": ("G1",)}

DiagnosticsStatus = str  # "complete" | "incomplete"


class DiagnosticsRecordError(RuntimeError):
    """A research-diagnostic record file is malformed or ends with an incomplete line (P7-4)."""


@dataclass(frozen=True)
class OptionalDiagnostics:
    """Optional diagnostics (diag §13). Both are **deferred** by D-49 (P7-6): not implemented,
    disabled, and refused if enabled, until a separate approval."""

    h1_on_dev: bool = False
    spectral_at_powers_of_two: bool = False

    def __post_init__(self) -> None:
        enabled = [k for k in ("h1_on_dev", "spectral_at_powers_of_two") if getattr(self, k)]
        if enabled:
            raise ValueError(f"optional diagnostic not authorised (deferred by D-49, P7-6): {enabled}")


@dataclass(frozen=True)
class EvaluationPoint:
    point_id: int
    kind: str
    sigma: int
    completed_epochs: int
    spectral: bool


def resolve_points(steps_per_epoch: int, max_epochs: int) -> tuple[EvaluationPoint, ...]:
    """P₀ (σ = 0, spectral); P_early σ = 2^j < S; P_epoch σ = k·S, k = 1 … E, spectral at k ∈ {1, E}
    (diag §5.1; integration §4.2). Ids in order of σ."""
    if steps_per_epoch < 1 or max_epochs < 1:
        raise ValueError("steps_per_epoch and max_epochs must be >= 1")
    pts = [EvaluationPoint(0, "start", 0, 0, True)]
    j = 0
    while 2 ** j < steps_per_epoch:
        pts.append(EvaluationPoint(len(pts), "early", 2 ** j, 0, False))
        j += 1
    for k in range(1, max_epochs + 1):
        pts.append(EvaluationPoint(len(pts), "epoch_end", k * steps_per_epoch, k, k in (1, max_epochs)))
    return tuple(pts)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, allow_nan=True, separators=(",", ":"))


def _append_line_fsync(path: Path, obj: Mapping[str, Any]) -> None:
    """One JSON line, written with one ``write``, flushed and fsynced (plan §9.5, MS4 write rule)."""
    line = _dumps(obj) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_jsonl_complete(path: Path) -> list[dict[str, Any]]:
    """Strict reader: every line complete and a JSON object; an incomplete last line raises
    :class:`DiagnosticsRecordError` (it is never discarded here; recovery is the orchestration
    preflight's job, plan §9.4)."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise DiagnosticsRecordError(f"{path}: the last line is incomplete")
    out = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiagnosticsRecordError(f"{path} line {n}: not JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise DiagnosticsRecordError(f"{path} line {n}: not a JSON object")
        out.append(obj)
    return out


def _rng_sha256() -> str:
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def _process_state(model: RNNLanguageModel) -> dict[str, Any]:
    return {"param_sha256": param_sha256(model), "rng_sha256": _rng_sha256(),
            "grad_enabled": torch.is_grad_enabled(), "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(), "training_mode": model.training}


class RNNDiagnosticsPlan:
    """Model-local research-diagnostics plan called by ``train_rnn`` at its three hook points
    (integration spec §4.1, §4.2; diag §5.2). It reads the model, never changes it, uses no RNG,
    writes only under ``out_dir`` and returns nothing to training.

    Files (``out_dir`` = ``<run_dir>/research_diagnostics``, D-49 O-3): ``plan.json``,
    ``points.jsonl``, ``primary.jsonl``, ``raw/point_{id:04d}.npz``; ``o3.json`` via :func:`write_o3`.
    """

    def __init__(self, *, probe: Probe, train: NextTokenSequences, train_names: Sequence[str],
                 dev: NextTokenSequences, steps_per_epoch: int, max_epochs: int, recurrent_gain: float,
                 out_dir: str | Path, optional: OptionalDiagnostics = OptionalDiagnostics()) -> None:
        if not isinstance(optional, OptionalDiagnostics):
            raise TypeError("optional must be an OptionalDiagnostics")
        lengths = train.lengths
        info = dict(probe.split_info)
        if int(info.get("N", -1)) != int(lengths.numel()) or int(info.get("M", -1)) != int(lengths.sum()):
            raise ProbeMismatchError("the probe's split N, M differ from the training data")
        validate_probe(probe, train, train_names, info)  # C5: before any training step
        self.probe, self.train, self.dev = probe, train, dev
        self.steps_per_epoch, self.max_epochs = int(steps_per_epoch), int(max_epochs)
        self.recurrent_gain = float(recurrent_gain)
        self.optional = optional
        self.out_dir = Path(out_dir)
        self.raw_dir = self.out_dir / "raw"
        self.points = resolve_points(self.steps_per_epoch, self.max_epochs)
        self._early = {p.sigma: p for p in self.points if p.kind == "early"}
        self._epoch = {p.completed_epochs: p for p in self.points if p.kind == "epoch_end"}
        self._invalid: list[int] = []
        self._done: set[int] = set()
        plan = {"schema": DIAGNOSTICS_SCHEMA, "probe_sha256": probe.sha256, "steps_per_epoch": self.steps_per_epoch,
                "max_epochs": self.max_epochs, "recurrent_gain": self.recurrent_gain,
                "optional": {"h1_on_dev": False, "spectral_at_powers_of_two": False},
                "points": [p.__dict__ for p in self.points],
                "diagnostics_by_kind": {k: list(v) for k, v in _DIAGNOSTICS_BY_KIND.items()},
                "threads": {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}}
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "plan.json"
        text = json.dumps(plan, sort_keys=True, indent=1) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(f"{path} exists and differs from this plan (a different run or configuration)")
        else:
            atomic_write_text(path, text)
        self._load_state()

    # ------------------------------------------------------------------ hooks
    def at_start(self, model: RNNLanguageModel) -> None:
        """P₀ (σ = 0), after the epoch-0 row and before step 1."""
        self._run(model, self.points[0])

    def after_step(self, model: RNNLanguageModel, *, sigma: int, epoch: int) -> None:
        """Called after every step; acts only for σ ∈ P_early (σ = 2^j < S)."""
        point = self._early.get(int(sigma))
        if point is not None:
            self._run(model, point)

    def at_epoch_end(self, model: RNNLanguageModel, *, sigma: int, completed_epochs: int) -> None:
        """P_epoch for epoch k = ``completed_epochs``: after evaluation and O2, before checkpoint writes."""
        point = self._epoch.get(int(completed_epochs))
        if point is None or point.sigma != int(sigma):
            raise ValueError(f"no epoch-end point for k = {completed_epochs} at σ = {sigma}")
        self._run(model, point)

    def truncate(self, *, sigma: int, completed_epochs: int) -> None:
        """Resume truncation (training spec §10.6 step 5): keep points with σ ≤ σ_K that are not
        an uncommitted epoch end; remove later points and their raw files. Incomplete last lines
        are refused (:class:`DiagnosticsRecordError`), never repaired here."""
        points_path, primary_path = self.out_dir / "points.jsonl", self.out_dir / "primary.jsonl"
        pts, prim = read_jsonl_complete(points_path), read_jsonl_complete(primary_path)

        def keep(rec: Mapping[str, Any]) -> bool:
            if rec["kind"] == "posthoc_best":
                return False
            if rec["kind"] == "epoch_end" and rec["completed_epochs"] > completed_epochs:
                return False
            return rec["sigma"] <= sigma

        kept_ids = {r["point_id"] for r in pts if keep(r)}
        removed = [r for r in pts if r["point_id"] not in kept_ids]
        if removed:  # plan §9.5: raw files of removed points are moved, never deleted
            dest = self.out_dir / "truncated" / f"resume_sigma_{int(sigma):07d}"
            n = 0
            while dest.exists():
                n += 1
                dest = self.out_dir / "truncated" / f"resume_sigma_{int(sigma):07d}_{n}"
            dest.mkdir(parents=True)
            for r in removed:
                src = self.raw_dir / f"point_{r['point_id']:04d}.npz"
                if src.exists():
                    src.replace(dest / src.name)
        for path, recs in ((points_path, pts), (primary_path, prim)):
            text = "".join(_dumps(r) + "\n" for r in recs if r["point_id"] in kept_ids)
            if path.exists():
                atomic_write_text(path, text)
        self._load_state()

    @property
    def status(self) -> DiagnosticsStatus:
        return "incomplete" if self._invalid else "complete"

    @property
    def invalid_points(self) -> tuple[int, ...]:
        return tuple(self._invalid)

    # ------------------------------------------------------------------ internals
    def _load_state(self) -> None:
        recs = read_jsonl_complete(self.out_dir / "points.jsonl")
        self._done = {r["point_id"] for r in recs}
        self._invalid = [r["point_id"] for r in recs if not r["valid"]]

    def _compute(self, copy_model: RNNLanguageModel, name: str, spectral: bool) -> DiagnosticResult:
        if name == "H1":
            return h1_statistics(copy_model, self.probe, self.train)
        if name == "G1":
            return g1_products(copy_model, self.probe, self.train, spectral=spectral)
        if name == "G2":
            return g2_attenuation(copy_model, self.probe, self.train)
        if name == "G3":
            return g3_decomposition(copy_model, self.probe, self.train)
        if name == "C1":
            return c1_restart(copy_model, self.dev)
        raise KeyError(name)

    def _run(self, model: RNNLanguageModel, point: EvaluationPoint) -> None:
        """Compute one point; record it (C6: failures become a ``valid: false`` record; a failure
        to write the record propagates and fails the run)."""
        if point.point_id in self._done:
            raise RuntimeError(f"diagnostic point {point.point_id} already recorded (duplicate call)")
        t0 = time.perf_counter()
        before = _process_state(model)
        errors: list[dict[str, Any]] = []
        results: dict[str, DiagnosticResult] = {}
        try:
            copy_model = float64_copy(model)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently (C6)
            copy_model = None
            errors.append(_error_record("float64_copy", exc))
        if copy_model is not None:
            for name in _DIAGNOSTICS_BY_KIND[point.kind]:
                try:
                    res = self._compute(copy_model, name, point.spectral)
                    results[name] = res
                    for check, value in res.checks.items():
                        failed = (value is False) or (isinstance(value, Mapping) and value.get("pass") is False)
                        if failed:
                            errors.append({"diagnostic": name, "type": "FailedCheck", "message": check,
                                           "values": value, "traceback": None})
                except Exception as exc:  # noqa: BLE001
                    errors.append(_error_record(name, exc))
        cross: dict[str, Any] = {}
        if point.spectral and "G1" in results and "G2" in results:
            try:
                cross["R_le_sigma"] = r_le_sigma_check(results["G1"].raw, results["G2"].raw)
                if not cross["R_le_sigma"]["pass"]:
                    errors.append({"diagnostic": "G1_G2", "type": "FailedCheck", "message": "R_le_sigma",
                                   "values": cross["R_le_sigma"], "traceback": None})
            except Exception as exc:  # noqa: BLE001
                errors.append(_error_record("G1_G2", exc))
        after = _process_state(model)
        changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        if changed:
            errors.append({"diagnostic": "isolation", "type": "IsolationBreach",
                           "message": "process or model state changed across the call", "values": changed,
                           "traceback": None})
        valid = not errors
        arrays: dict[str, np.ndarray] = {}
        for res in results.values():
            arrays.update(res.raw)
        record = {"schema": DIAGNOSTICS_SCHEMA, "point_id": point.point_id, "kind": point.kind,
                  "sigma": point.sigma, "completed_epochs": point.completed_epochs, "spectral": point.spectral,
                  "param_sha256_before": before["param_sha256"], "param_sha256_after": after["param_sha256"],
                  "rng_sha256_before": before["rng_sha256"], "rng_sha256_after": after["rng_sha256"],
                  "diagnostics": list(_DIAGNOSTICS_BY_KIND[point.kind]),
                  "checks": {**{name: res.checks for name, res in results.items()},
                             **({"G1_G2": cross} if cross else {})},
                  "seconds": time.perf_counter() - t0, "valid": valid, "errors": errors}
        primary = {"schema": DIAGNOSTICS_SCHEMA, "point_id": point.point_id, "kind": point.kind,
                   "sigma": point.sigma, "valid": valid, "results": {n: r.primary for n, r in results.items()}}
        self._write_point(point, record, primary, arrays)
        self._done.add(point.point_id)
        if not valid:
            self._invalid.append(point.point_id)
            warnings.warn(f"diagnostic point {point.point_id} ({point.kind}, σ = {point.sigma}) is invalid: "
                          f"{[e['diagnostic'] + ': ' + e['type'] for e in errors]}", stacklevel=2)

    def _write_point(self, point: EvaluationPoint, record: dict, primary: dict, arrays: dict) -> None:
        # The raw file is written first, so every point line refers to a complete raw file.
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        _atomic_npz(self.raw_dir / f"point_{point.point_id:04d}.npz", arrays)
        _append_line_fsync(self.out_dir / "points.jsonl", record)
        _append_line_fsync(self.out_dir / "primary.jsonl", primary)


def r_le_sigma_check(g1_raw: Mapping[str, np.ndarray], g2_raw: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """diag §10.2 row 3, second half: R_{t,s} ≤ σ(P_{t,s})(1 + 1e‑12) for every G2 pair with s < t, at a
    spectral (E_sel) point. Pairs are matched by (probe row, s, t) between the G1 and G2 raw arrays; a
    G2 pair without its G1 σ fails the check (the two diagnostics must cover the same pairs)."""
    sigma = {(int(r), int(s), int(t)): float(v) for r, s, t, v in
             zip(g1_raw["g1_row"], g1_raw["g1_s"], g1_raw["g1_t"], g1_raw["g1_log_sigma"])}
    slack = math.log1p(1e-12)
    pairs, missing, worst = 0, 0, -math.inf
    for r, s, t, log_r in zip(g2_raw["g2_row"], g2_raw["g2_s"], g2_raw["g2_t"], g2_raw["g2_log_R"]):
        if s >= t:
            continue  # R_{t,t} = 1 has no Jacobian product
        pairs += 1
        log_sig = sigma.get((int(r), int(s), int(t)))
        if log_sig is None or math.isnan(log_sig):
            missing += 1
            continue
        if log_r == -math.inf:
            continue
        worst = max(worst, float(log_r) - (log_sig + slack))
    return {"pairs": pairs, "missing_sigma": missing, "max_log_excess": None if worst == -math.inf else worst,
            "pass": missing == 0 and worst <= 0.0}


def _error_record(name: str, exc: BaseException) -> dict[str, Any]:
    return {"diagnostic": name, "type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), "values": None}


# --------------------------------------------------------------------------- O3 (diag §4.6)
def o3_record(history: Sequence[Mapping[str, Any]], step_records: Sequence[Mapping[str, Any]], *,
              status: str, diagnostics_status: DiagnosticsStatus | None,
              invalid_points: Sequence[int] = ()) -> dict[str, Any]:
    """Run-level O3 classification (descriptive; no spike multiple, D-45):
    failed run (ended by training-spec §12.1); epoch events (dev NLL rose; running train NLL rose);
    per-epoch gradient-norm profile (max, median, max/median of the per-step global norm); with
    ``incomplete`` diagnostics, the list of invalid points (integration §4.6, C6)."""
    if diagnostics_status == "incomplete" and not invalid_points:
        raise ValueError("incomplete diagnostics must carry the list of invalid points (C6)")
    events = []
    rows = sorted(history, key=lambda r: r["completed_epochs"])
    for prev, cur in zip(rows, rows[1:]):
        if cur["dev_nll"] > prev["dev_nll"]:
            events.append({"epoch": cur["completed_epochs"], "event": "dev_nll_rose"})
        if prev.get("running_train_nll") is not None and cur.get("running_train_nll") is not None \
                and cur["running_train_nll"] > prev["running_train_nll"]:
            events.append({"epoch": cur["completed_epochs"], "event": "running_train_nll_rose"})
    by_epoch = defaultdict(list)
    for s in step_records:
        by_epoch[s["epoch"] + 1].append(s["grad_norm"])
    profile = []
    for k in sorted(by_epoch):
        norms = by_epoch[k]
        med = statistics.median(norms)
        profile.append({"epoch": k, "max": max(norms), "median": med, "max_over_median": max(norms) / med if med else None})
    return {"schema": DIAGNOSTICS_SCHEMA, "failed_run": status in ("failed_mechanical", "failed_diverged"),
            "status": status, "diagnostics_status": diagnostics_status,
            "invalid_points": sorted(int(p) for p in invalid_points),
            "research_use": "excluded (failed run)" if status != "completed" else
            ("requires recorded human review (incomplete diagnostics, M6-DGN17)" if diagnostics_status == "incomplete"
             else "eligible"),
            "epoch_events": events, "gradient_norm_profile": profile}


def write_o3(out_dir: str | Path, record: Mapping[str, Any]) -> Path:
    """Write ``o3.json`` atomically under ``out_dir`` (the research-diagnostics directory)."""
    return atomic_write_text(Path(out_dir) / "o3.json", json.dumps(record, sort_keys=True, indent=1) + "\n")


# --------------------------------------------------------------------------- post-hoc best spectral (training §10.7)
def posthoc_best_spectral(run_dir: str | Path, probe: Probe, train: NextTokenSequences) -> int | None:
    """Spectral G1 at the best epoch from ``model_best.pt`` (diag §5.3; training spec §10.7).

    Refuses (``ValueError``) unless the checkpoint's stored ``param_sha256``, the hash of the loaded
    weights and the hash of that epoch's epoch-end diagnostic point all agree. Writes only under
    ``research_diagnostics/`` (a ``posthoc_best`` point); returns its id, or ``None`` when the best
    epoch is 1 or E, whose epoch-end points are already spectral."""
    from utils.checkpoint import load_checkpoint

    run_dir = Path(run_dir)
    out_dir = run_dir / RESEARCH_DIAGNOSTICS_DIR
    plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
    if plan["probe_sha256"] != probe.sha256:
        raise ProbeMismatchError("probe differs from the run's plan")
    best_path = run_dir / "model_best.pt"
    ckpt = load_checkpoint(best_path)
    best_k = int(ckpt["epoch"])
    if best_k in (1, int(plan["max_epochs"])):
        return None
    model = RNNLanguageModel.from_checkpoint(best_path)
    stored = (ckpt.get("extra") or {}).get("param_sha256")
    loaded = param_sha256(model)
    records = read_jsonl_complete(out_dir / "points.jsonl")
    epoch_point = next((r for r in records if r["kind"] == "epoch_end" and r["completed_epochs"] == best_k), None)
    if epoch_point is None:
        raise ValueError(f"no epoch-end diagnostic point for the best epoch {best_k}")
    if not (stored == loaded == epoch_point["param_sha256_before"]):
        raise ValueError("parameter hashes disagree (checkpoint extra, loaded weights, epoch-end point): refused")
    if any(r["kind"] == "posthoc_best" for r in records):
        raise ValueError("a posthoc_best point is already recorded")
    point = EvaluationPoint(max(r["point_id"] for r in records) + 1, "posthoc_best", epoch_point["sigma"], best_k, True)
    t0 = time.perf_counter()
    errors, results = [], {}
    before = _process_state(model)
    try:
        results["G1"] = g1_products(float64_copy(model), probe, train, spectral=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(_error_record("G1", exc))
    after = _process_state(model)
    changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    if changed:
        errors.append({"diagnostic": "isolation", "type": "IsolationBreach",
                       "message": "process or model state changed across the call", "values": changed,
                       "traceback": None})
    valid = not errors and results["G1"].checks.get("phi_le_sigma") is not False
    record = {"schema": DIAGNOSTICS_SCHEMA, "point_id": point.point_id, "kind": point.kind, "sigma": point.sigma,
              "completed_epochs": best_k, "spectral": True, "param_sha256_before": before["param_sha256"],
              "param_sha256_after": after["param_sha256"], "rng_sha256_before": before["rng_sha256"],
              "rng_sha256_after": after["rng_sha256"],
              "diagnostics": ["G1"], "checks": {n: r.checks for n, r in results.items()},
              "seconds": time.perf_counter() - t0, "valid": valid, "errors": errors}
    primary = {"schema": DIAGNOSTICS_SCHEMA, "point_id": point.point_id, "kind": point.kind, "sigma": point.sigma,
               "valid": valid, "results": {n: r.primary for n, r in results.items()}}
    raw_dir = out_dir / "raw"
    _atomic_npz(raw_dir / f"point_{point.point_id:04d}.npz", {k: v for r in results.values() for k, v in r.raw.items()})
    _append_line_fsync(out_dir / "points.jsonl", record)
    _append_line_fsync(out_dir / "primary.jsonl", primary)
    return point.point_id

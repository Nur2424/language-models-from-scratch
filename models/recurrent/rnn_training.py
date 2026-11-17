"""Model 6 reference training procedure: SGD with full BPTT, evaluation, selection, checkpoints,
resume, failure categories and research-run validation.

Specification: ``models/recurrent/rnn_training_spec.md`` v0.3 (approved, D-46), integrated by
``models/recurrent/rnn_integration_spec.md`` v0.2 §3, §5.2 (D-48); record layout D-49 (O-3);
milestone MS5 of ``models/recurrent/rnn_implementation_plan.md`` v0.2 (D-49).

Fixed in code (training spec §13): plain SGD (momentum 0, dampening 0, weight decay 0, no Nesterov);
constant learning rate; loss = token-summed NLL / C with C = B·M/N for every step including the
final partial batch; no clipping; Category A (non-finite) raises and Category D (full-train NLL
> ln 27 at a completed epoch) ends the run after committing that epoch.

``train_rnn`` uses only ``forward`` of the model (never ``_step``, ``_logits`` or ``_recurrence``);
it consumes no global RNG and calls the diagnostic plan only at its three hook points and on resume.
Pilots, seed loops, frozen conditions and Git handling belong to orchestration (``training/rnn_study``).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from models.recurrent.rnn import IncompatibleCheckpointError, RNNLanguageModel
from models.recurrent.rnn_diagnostics import (
    DIAGNOSTICS_SCHEMA,
    LENGTH_STRATA,
    MAX_POSITION,
    OVERFLOW_STRATUM,
    DiagnosticsRecordError,
    RNNDiagnosticsPlan,
    param_sha256,
    stratum_index,
)
from utils.batching import EpochSampler
from utils.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from utils.data import NextTokenSequences
from utils.diagnostics.gradients import nonfinite_gradients
from utils.io import atomic_write_text
from utils.metrics import NLLAccumulator, token_nll
from utils.reproducibility import capture_rng_state, derive_seed
from utils.run_manager import RunManager

LN_27 = math.log(27)
BEST_METRIC_NAME = "dev_nll"
STEP_RECORDS_FILE = "steps.jsonl"  # in RunManager.diagnostics_dir (D-49, O-3)
#: Counts recorded in the approved protocol (training spec §1); see :func:`check_protocol_counts`.
PROTOCOL_COUNTS = {"N": 25_640, "M": 182_666, "steps_per_epoch": 802, "final_batch_names": 8,
                   "N_dev": 3_196, "M_dev": 22_801}
RESEARCH_BATCH_SIZE, RESEARCH_EVAL_BATCH_SIZE = 32, 1024
PARAM_SYMBOLS = {"input_lookup": "U", "recurrent_weight": "W", "hidden_bias": "b", "output_weight": "V",
                 "output_bias": "c"}


class ResumeError(RuntimeError):
    """A run directory cannot be (re)used as asked: no committed checkpoint, state already present,
    or a record that cannot be truncated safely (integration spec §3.4, C7; plan §9)."""


class ResearchConfigError(ValueError):
    """A configuration that research runs must not use (integration spec §5.2, C8)."""


class CategoryAFailure(FloatingPointError):
    """Category A (training spec §12.1): a non-finite loss, gradient, post-step parameter or
    evaluated NLL. ``where`` names the check; ``result`` carries status ``failed_mechanical``."""

    def __init__(self, message: str, *, where: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.where = where
        self.details = dict(details or {})
        self.result: RNNTrainingResult | None = None


# =========================================================================== configuration and results
@dataclass(frozen=True)
class RNNTrainingConfig:
    """Training configuration (training spec §13). ``lr`` and ``max_epochs`` come from the pilot
    and the horizon rule; batch sizes are fixed for the study."""

    lr: float
    max_epochs: int
    batch_size: int = RESEARCH_BATCH_SIZE
    eval_batch_size: int = RESEARCH_EVAL_BATCH_SIZE

    def __post_init__(self) -> None:
        if isinstance(self.lr, bool) or not isinstance(self.lr, (int, float)) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(f"lr must be a finite number > 0, got {self.lr!r}")
        object.__setattr__(self, "lr", float(self.lr))
        for name in ("max_epochs", "batch_size", "eval_batch_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an int >= 1, got {v!r}")


@dataclass
class RNNTrainingResult:
    status: str  # "completed" | "failed_mechanical" | "failed_diverged"
    best_epoch: int | None
    best_dev_nll: float | None
    history: list[dict[str, Any]]
    loss_normalizer: float
    diagnostics_status: str | None
    #: Point ids of invalid diagnostic points when ``diagnostics_status == "incomplete"`` (C6).
    invalid_diagnostic_points: tuple[int, ...] = ()


@dataclass
class PositionNLL:
    """Dev NLL sums and counts per position t (1 … 16, then ``"17+"``) and per (stratum, t)."""

    by_t: dict[str, list[float]] = field(default_factory=dict)  # t -> [sum, count]
    by_stratum_t: dict[str, list[float]] = field(default_factory=dict)  # "s:t" -> [sum, count]

    def total(self) -> tuple[float, int]:
        return math.fsum(v[0] for v in self.by_t.values()), int(sum(v[1] for v in self.by_t.values()))

    def to_record(self) -> dict[str, Any]:
        return {"by_t": self.by_t, "by_stratum_t": self.by_stratum_t,
                "strata": [list(s) for s in LENGTH_STRATA], "overflow_stratum": OVERFLOW_STRATUM}


# =========================================================================== loss and evaluation
def loss_normalizer(training: RNNTrainingConfig, train: NextTokenSequences) -> float:
    """C = B · M / N in float64 (training spec §4.1): one constant for every step."""
    N, M = int(train.lengths.numel()), int(train.lengths.sum())
    return training.batch_size * M / N


def _length_sorted_chunks(sequences: NextTokenSequences, batch_size: int):
    lengths = [int(v) for v in sequences.lengths.tolist()]
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    for start in range(0, len(order), batch_size):
        rows = order[start: start + batch_size]
        yield rows, max(lengths[i] for i in rows)


def evaluate_nll(model: RNNLanguageModel, sequences: NextTokenSequences, *, batch_size: int) -> NLLAccumulator:
    """Token-weighted NLL over all valid targets (training spec §8.2): names in stable length order,
    chunks of ``batch_size`` names trimmed to their longest name, under ``no_grad``; mode restored."""
    acc = NLLAccumulator()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for rows, T in _length_sorted_chunks(sequences, batch_size):
                idx = torch.tensor(rows, dtype=torch.long)
                x = sequences.inputs[idx, :T].to(model.device)
                y = sequences.targets[idx, :T].to(model.device)
                total = token_nll(model(x), y, reduction="sum")
                if not bool(torch.isfinite(total)):
                    raise CategoryAFailure("non-finite evaluated NLL", where="evaluation",
                                           details={"rows": len(rows)})
                acc.update_from_sum(float(total), int((y != -100).sum()))
    finally:
        model.train(was_training)
    return acc


def evaluate_position_nll(model: RNNLanguageModel, sequences: NextTokenSequences, *, batch_size: int) -> PositionNLL:
    """Per-position (t = 1 … 16; ``17+``) and per (stratum, t) dev NLL sums and counts in float64
    (training spec §8.4; M6-DGN11), in the same pass structure as :func:`evaluate_nll`."""
    out = PositionNLL()
    parts: dict[tuple[str, str], list[float]] = {}  # summed with math.fsum below (diag §6)
    lengths = [int(v) for v in sequences.lengths.tolist()]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for rows, T in _length_sorted_chunks(sequences, batch_size):
                idx = torch.tensor(rows, dtype=torch.long)
                x = sequences.inputs[idx, :T].to(model.device)
                y = sequences.targets[idx, :T].to(model.device)
                logits = model(x)
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none",
                                      ignore_index=-100).reshape(y.shape).to(torch.float64).cpu()
                for i, r in enumerate(rows):
                    s = stratum_index(lengths[r])
                    for t in range(1, lengths[r] + 1):
                        tk = str(t) if t <= MAX_POSITION else f"{MAX_POSITION + 1}+"
                        v = float(nll[i, t - 1])
                        parts.setdefault(("t", tk), []).append(v)
                        parts.setdefault(("st", f"{s}:{tk}"), []).append(v)
    finally:
        model.train(was_training)
    for (kind, key), vals in parts.items():
        (out.by_t if kind == "t" else out.by_stratum_t)[key] = [math.fsum(vals), len(vals)]
    return out


def _o2(model: RNNLanguageModel) -> dict[str, Any]:
    """Per-epoch parameter records (training spec §8.4): L2 norms of U, W, b, V, c; all singular
    values of W (float64 copy); spectral radius; ‖WᵀW/g² − I‖_F."""
    norms = {PARAM_SYMBOLS[n]: float(p.detach().to("cpu", torch.float64).norm()) for n, p in model.named_parameters()}
    W = model.recurrent_weight.detach().to("cpu", torch.float64)
    g = model.config.recurrent_gain
    eye = torch.eye(W.shape[0], dtype=torch.float64)
    return {"param_norms": norms, "W_singular_values": torch.linalg.svdvals(W).tolist(),
            "W_spectral_radius": float(torch.linalg.eigvals(W).abs().max()),
            "W_orthogonality_defect": float(torch.linalg.matrix_norm(W.T @ W / g ** 2 - eye))}


def _params_finite(model: RNNLanguageModel) -> bool:
    return bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())


# =========================================================================== records
def _append_line_fsync(path: Path, record: Mapping[str, Any]) -> None:
    """One JSON line per write, flushed and fsynced (plan §9.5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, allow_nan=True, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _truncate_jsonl(path: Path, keep: Callable[[Mapping[str, Any]], bool]) -> list[dict[str, Any]]:
    """Keep the leading records for which ``keep`` holds, rewriting the file atomically with the
    original bytes of those lines. An incomplete last line or a malformed line raises
    :class:`ResumeError`: it is never discarded here (recovery preflight, plan §9.4)."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ResumeError(f"{path}: the last line is incomplete; run the recovery preflight first")
    lines = raw.decode("utf-8").splitlines(keepends=True)
    kept_lines, kept = [], []
    for n, line in enumerate(lines, start=1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResumeError(f"{path} line {n}: not JSON") from exc
        if keep(rec):
            kept_lines.append(line)
            kept.append(rec)
    if len(kept_lines) != len(lines):
        atomic_write_text(path, "".join(kept_lines))
    return kept


# =========================================================================== research configuration (C8)
def validate_research_run(experiment: Any, model: RNNLanguageModel, training: RNNTrainingConfig, *,
                          expected_threads: tuple[int, int]) -> None:
    """Refuse any configuration a research run must not use (integration spec §5.2, C8):
    device exactly ``"cpu"`` (``"auto"`` rejected), dtype float32, model on CPU in float32, shared
    diagnostics schedule disabled, no step-based resume, the decided thread counts, batch sizes 32
    and 1,024. ``expected_threads`` = (intra-op, inter-op) as decided (the benchmark passes the
    configuration it is measuring)."""
    problems = []
    device = experiment.device.device
    if device == "auto":
        problems.append("device: auto is not allowed for research runs (set exactly 'cpu')")
    elif device != "cpu":
        problems.append(f"device must be 'cpu', got {device!r}")
    if experiment.device.dtype != "float32":
        problems.append(f"dtype must be float32, got {experiment.device.dtype!r}")
    for name, p in model.named_parameters():
        if p.device.type != "cpu" or p.dtype != torch.float32:
            problems.append(f"parameter {name} is {p.dtype} on {p.device}, expected float32 on cpu")
    if experiment.diagnostics.enabled:
        problems.append("the shared diagnostics schedule must be disabled (research diagnostics use the plan)")
    if experiment.checkpoint.resume_every_n_steps is not None:
        problems.append("checkpoint.resume_every_n_steps must be null (resume is at epoch boundaries)")
    threads = (torch.get_num_threads(), torch.get_num_interop_threads())
    if tuple(expected_threads) != threads:
        problems.append(f"thread counts {threads} differ from the decided {tuple(expected_threads)}")
    if (training.batch_size, training.eval_batch_size) != (RESEARCH_BATCH_SIZE, RESEARCH_EVAL_BATCH_SIZE):
        problems.append(f"batch sizes must be {RESEARCH_BATCH_SIZE} and {RESEARCH_EVAL_BATCH_SIZE}")
    if problems:
        raise ResearchConfigError("; ".join(problems))


def check_protocol_counts(train: NextTokenSequences, dev: NextTokenSequences, training: RNNTrainingConfig) -> dict[str, int]:
    """Refuse data whose counts differ from the approved protocol (training spec §1: N, M, S, r and
    the dev counts). Called by the orchestration drivers before every research or pilot run."""
    N, M = int(train.lengths.numel()), int(train.lengths.sum())
    S = -(-N // training.batch_size)
    counts = {"N": N, "M": M, "steps_per_epoch": S, "final_batch_names": N - (S - 1) * training.batch_size,
              "N_dev": int(dev.lengths.numel()), "M_dev": int(dev.lengths.sum())}
    diff = {k: (counts[k], PROTOCOL_COUNTS[k]) for k in counts if counts[k] != PROTOCOL_COUNTS[k]}
    if diff:
        raise ResearchConfigError(f"data counts differ from the approved protocol: {diff}")
    return counts


# =========================================================================== training
def _check_resume_compatible(ckpt: Mapping[str, Any], cfg: Mapping[str, Any], model: RNNLanguageModel,
                             seed: int) -> None:
    """Training spec §10.6 step 2: any difference raises ``IncompatibleCheckpointError``."""
    stored = ckpt.get("config") or {}
    problems = [k for k in ("experiment", "model", "training", "derived", "diagnostics")
                if json.dumps(stored.get(k), sort_keys=True) != json.dumps(cfg.get(k), sort_keys=True)]
    if ckpt.get("vocab") != model.vocab.to_dict():
        problems.append("vocabulary")
    extra = ckpt.get("extra") or {}
    if extra.get("base_seed") != seed:
        problems.append("base seed")
    if (extra.get("sampler") or {}).get("seed") != derive_seed(seed, "data"):
        problems.append("sampler seed")
    if problems:
        details = {k: (stored.get(k), cfg.get(k)) for k in problems if k in cfg}
        raise IncompatibleCheckpointError(f"resume refused; differing fields: {problems}; {details}")


def train_rnn(model: RNNLanguageModel, train: NextTokenSequences, dev: NextTokenSequences, *,
              training: RNNTrainingConfig, seed: int, run: RunManager | None = None, resume: bool = False,
              config: Mapping[str, Any] | None = None,
              diagnostics: RNNDiagnosticsPlan | None = None) -> RNNTrainingResult:
    """Train with plain SGD and full BPTT (training spec §3 – §12; integration spec §3.2).

    Order per step: zero_grad → forward → token-summed loss → finite? → (loss/C).backward() →
    per-parameter and global gradient norms (float64) → finite? → ``optimizer.step()`` →
    parameters finite? → running NLL → O1 record → ``diagnostics.after_step``.
    Epoch end: parameters finite → train/dev NLL (finite?) → Category D test → O2 + parameter hash →
    ``diagnostics.at_epoch_end`` → ``model_best.pt`` (strict <, k ≥ 1) → history row →
    ``checkpoint_last.pt`` (commit) → if D: ``model_final.pt`` and stop.
    Before step 1: the epoch-0 row (never selectable), then ``diagnostics.at_start``.

    Category A raises :class:`CategoryAFailure` (status ``failed_mechanical`` on ``exc.result`` and in
    the run metadata). ``resume=True`` continues ``run`` from ``checkpoint_last.pt`` after the
    compatibility check and truncation of history, step records and diagnostic records.
    """
    if not isinstance(model, RNNLanguageModel):
        raise TypeError("model must be an RNNLanguageModel")
    if not isinstance(training, RNNTrainingConfig):
        raise TypeError("training must be an RNNTrainingConfig")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if resume and run is None:
        raise ValueError("resume=True requires the run to resume")
    B, E = training.batch_size, training.max_epochs
    N, M = int(train.lengths.numel()), int(train.lengths.sum())
    sampler = EpochSampler(N, B, seed=derive_seed(seed, "data"))
    S = sampler.num_batches
    C = loss_normalizer(training, train)
    plan = diagnostics
    if plan is not None:
        if not isinstance(plan, RNNDiagnosticsPlan):
            raise TypeError("diagnostics must be an RNNDiagnosticsPlan or None")
        if (plan.max_epochs, plan.steps_per_epoch) != (E, S):
            raise ValueError(f"plan was built for E = {plan.max_epochs}, S = {plan.steps_per_epoch}; "
                             f"training has E = {E}, S = {S}")
    derived = {"loss_normalizer_C": C, "N": N, "M": M, "N_dev": int(dev.lengths.numel()),
               "M_dev": int(dev.lengths.sum()), "steps_per_epoch": S, "final_batch_names": N - (S - 1) * B,
               "intra_op_threads": torch.get_num_threads(), "inter_op_threads": torch.get_num_interop_threads(),
               "dtype": str(model.recurrent_weight.dtype).replace("torch.", ""), "device": model.device.type}
    cfg = {"experiment": dict((config or {}).get("experiment") or {}) or None, "model": asdict(model.config),
           "training": asdict(training), "derived": derived,
           "diagnostics": None if plan is None else {"probe_sha256": plan.probe.sha256, "schema": DIAGNOSTICS_SCHEMA}}
    vocab_dict = model.vocab.to_dict()
    optimizer = torch.optim.SGD(model.parameters(), lr=training.lr, momentum=0.0, dampening=0.0,
                                weight_decay=0.0, nesterov=False, maximize=False)
    device = model.device
    lengths_cpu = train.lengths.cpu()
    inputs, targets = train.inputs.to(device), train.targets.to(device)

    history: list[dict[str, Any]] = []
    best_epoch, best_dev = None, math.inf
    start_epoch, step = 0, 0
    steps_path = None if run is None else run.diagnostics_dir / STEP_RECORDS_FILE
    if run is not None:
        resume_path = run.checkpoint_path("resume")
        if resume:
            if not resume_path.exists():
                raise ResumeError(f"{run.run_dir}: no committed checkpoint (aborted before its first commit; "
                                  "any retry uses a new run directory)")
            run_status = run.read_metadata().get("status")
            outcome = run.read_metadata().get("outcome")
            if run_status not in ("failed", "interrupted") or outcome in ("failed_mechanical", "failed_diverged"):
                raise ResumeError(f"{run.run_dir}: run status {run_status!r} (outcome {outcome!r}) cannot be resumed; "
                                  "a failed run is final (D-46); nothing was changed")
            ckpt = load_checkpoint(resume_path, expected_kind="resume")
            _check_resume_compatible(ckpt, cfg, model, seed)
            restore_training_state(ckpt, model=model, optimizer=optimizer, restore_rng=False)
            sampler = EpochSampler.from_state_dict(ckpt["extra"]["sampler"], num_examples=N, batch_size=B)
            start_epoch, step = int(ckpt["epoch"]), int(ckpt["step"])
            K, sigma_k = start_epoch, step
            history = _truncate_jsonl(run.history_path, lambda r: r["completed_epochs"] <= K)
            _truncate_jsonl(steps_path, lambda r: r["step"] <= sigma_k)
            if plan is not None:
                try:
                    plan.truncate(sigma=sigma_k, completed_epochs=K)
                except DiagnosticsRecordError as exc:
                    raise ResumeError(str(exc)) from exc
            best_epoch = ckpt["extra"]["best_epoch"]
            best_dev = math.inf if ckpt.get("best_metric") is None else float(ckpt["best_metric"])
        else:
            diag_points = None if plan is None else plan.out_dir / "points.jsonl"
            occupied = [p for p in (resume_path, run.history_path, steps_path, run.checkpoint_path("best"), diag_points)
                        if p is not None and p.exists() and p.stat().st_size > 0]
            if occupied:
                raise ResumeError(f"{run.run_dir} already holds training state {[p.name for p in occupied]}; "
                                  "use resume=True or a new run directory")

    def save(kind: str, epoch: int, h: str) -> None:
        if run is None:
            return
        extra = {"best_epoch": best_epoch, "param_sha256": h}
        if kind == "resume":
            save_checkpoint(run.checkpoint_path("resume"), kind="resume", model=model, optimizer=optimizer,
                            step=step, epoch=epoch, best_metric=None if best_dev == math.inf else best_dev,
                            best_metric_name=BEST_METRIC_NAME, rng_state=capture_rng_state(), config=cfg,
                            vocab=vocab_dict, extra={**extra, "sampler": sampler.state_dict(), "base_seed": seed})
        elif kind == "best":
            save_checkpoint(run.checkpoint_path("best"), kind="best", model=model, step=step, epoch=epoch,
                            best_metric=best_dev, best_metric_name=BEST_METRIC_NAME, config=cfg, vocab=vocab_dict,
                            extra=extra)
        else:
            save_checkpoint(run.checkpoint_path("final"), kind="final", model=model, step=step, epoch=epoch,
                            config=cfg, vocab=vocab_dict, extra=extra)

    def evaluate(k: int) -> tuple[float, float, PositionNLL]:
        tr = evaluate_nll(model, train, batch_size=training.eval_batch_size).mean_nll
        dv = evaluate_nll(model, dev, batch_size=training.eval_batch_size).mean_nll
        if not (math.isfinite(tr) and math.isfinite(dv)):
            raise CategoryAFailure(f"non-finite evaluated NLL at epoch {k}: train {tr}, dev {dv}",
                                   where="evaluation", details={"epoch": k, "train_nll": tr, "dev_nll": dv})
        return tr, dv, evaluate_position_nll(model, dev, batch_size=training.eval_batch_size)

    status = "completed"
    ctx = run.track() if run is not None else nullcontext()
    with ctx:
        try:
            if not resume:
                t0 = time.perf_counter()
                tr, dv, pos = evaluate(0)
                row = {"completed_epochs": 0, "step": 0, "running_train_nll": None, "train_nll": tr, "dev_nll": dv,
                       "position_dev_nll": pos.to_record(), **_o2(model), "param_sha256": param_sha256(model),
                       "diverged": False, "selectable": False, "best_epoch": None, "is_best": False,
                       "seconds": time.perf_counter() - t0}
                history.append(row)
                if run is not None:
                    run.append_history(row)
                if plan is not None:
                    plan.at_start(model)
            for e in range(start_epoch, E):
                t0 = time.perf_counter()
                running = NLLAccumulator()
                for b_idx, idx in enumerate(sampler.batches(e)):
                    T_b = int(lengths_cpu[idx].max())
                    m_b = int(lengths_cpu[idx].sum())
                    dev_idx = idx.to(device)
                    x, y = inputs[dev_idx, :T_b], targets[dev_idx, :T_b]
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = token_nll(model(x), y, reduction="sum")
                    if not bool(torch.isfinite(loss_sum)):
                        raise CategoryAFailure(f"non-finite loss at epoch {e}, batch {b_idx}", where="loss",
                                               details={"epoch": e, "batch_index": b_idx, "step": step + 1})
                    (loss_sum / C).backward()
                    norms = {PARAM_SYMBOLS[n]: float(p.grad.detach().to(torch.float64).norm())
                             for n, p in model.named_parameters()}
                    gnorm = math.sqrt(math.fsum(v * v for v in norms.values()))
                    if not math.isfinite(gnorm):
                        raise CategoryAFailure(f"non-finite gradient at epoch {e}, batch {b_idx}", where="gradient",
                                               details={"epoch": e, "batch_index": b_idx, "step": step + 1,
                                                        "nonfinite": nonfinite_gradients(model)})
                    optimizer.step()
                    if not _params_finite(model):
                        raise CategoryAFailure(f"non-finite parameter after step {step + 1}", where="parameter",
                                               details={"epoch": e, "batch_index": b_idx, "step": step + 1})
                    step += 1
                    ls = float(loss_sum.detach())
                    running.update_from_sum(ls, m_b)
                    if steps_path is not None:
                        _append_line_fsync(steps_path, {
                            "epoch": e, "step": step, "batch_index": b_idx, "names": int(idx.numel()), "m_b": m_b,
                            "T_b": T_b, "nll_mean": ls / m_b, "loss": ls / C, "grad_norm": gnorm,
                            "grad_norms": norms, "finite": True})
                    if plan is not None:
                        plan.after_step(model, sigma=step, epoch=e)
                if not _params_finite(model):
                    raise CategoryAFailure(f"non-finite parameter at the end of epoch {e}", where="parameter",
                                           details={"epoch": e})
                k = e + 1
                tr, dv, pos = evaluate(k)
                diverged = tr > LN_27
                h = param_sha256(model)
                row = {"completed_epochs": k, "step": step, "running_train_nll": running.mean_nll, "train_nll": tr,
                       "dev_nll": dv, "position_dev_nll": pos.to_record(), **_o2(model), "param_sha256": h,
                       "diverged": diverged, "selectable": True}
                if plan is not None:
                    plan.at_epoch_end(model, sigma=step, completed_epochs=k)
                improved = dv < best_dev
                if improved:
                    best_dev, best_epoch = dv, k
                    save("best", k, h)
                row.update(best_epoch=best_epoch, is_best=improved, seconds=time.perf_counter() - t0)
                history.append(row)
                if run is not None:
                    run.append_history(row)
                save("resume", k, h)  # commit point
                if diverged:
                    status = "failed_diverged"
                    save("final", k, h)
                    break
            else:
                if E > 0:
                    save("final", E, param_sha256(model))
        except CategoryAFailure as exc:
            exc.result = RNNTrainingResult("failed_mechanical", best_epoch,
                                           None if best_dev == math.inf else best_dev, history, C,
                                           None if plan is None else plan.status,
                                           () if plan is None else plan.invalid_points)
            if run is not None:
                run.update_metadata(outcome="failed_mechanical", failure={"where": exc.where, **exc.details},
                                    diagnostics_status=exc.result.diagnostics_status,
                                    invalid_diagnostic_points=list(exc.result.invalid_diagnostic_points),
                                    loss_normalizer_C=C)
            raise
        if run is not None:
            run.update_metadata(outcome=status, diagnostics_status=None if plan is None else plan.status,
                                invalid_diagnostic_points=[] if plan is None else list(plan.invalid_points),
                                loss_normalizer_C=C, best_epoch=best_epoch)
    return RNNTrainingResult(status, best_epoch, None if best_dev == math.inf else best_dev, history, C,
                             None if plan is None else plan.status, () if plan is None else plan.invalid_points)

"""MLP: a Bengio-style fixed-context feedforward character language model.

Specification: ``models/feedforward/mlp_spec.md`` (v0.3). Section numbers below
(§) refer to it.

Architecture (§3, §4.1)
-----------------------
For a context ``x = (x₁, x₂, x₃)`` (oldest token first)::

    e_i = E[x_i]                    E  ∈ ℝ^{27×10}   one table, shared by all positions
    u   = [e₁; e₂; e₃]              ℝ^{30}           position-major concatenation
    a   = W₁ u + b₁                 W₁ ∈ ℝ^{200×30}  (PyTorch [out, in] layout)
    h   = tanh(a)
    z   = W₂ h + b₂                 W₂ ∈ ℝ^{27×200}  → 27 logits

``W₁[:, 10p:10p+10]`` is the projection block of position ``p``, so the shared
embedding of a character contributes differently at each position. There are no
direct input-to-output connections and 11,897 parameters in the baseline.

Initialisation (§7)
-------------------
``E ~ N(0, 1)``; ``W₁ ~ N(0, 1/fan_in)`` with ``fan_in = context_length ·
embedding_dim`` (baseline 30); ``b₁ = W₂ = b₂ = 0``. All draws come from the
explicit CPU ``generator``; the modules are built with ``nn.utils.skip_init``,
so PyTorch's default initialisation never runs and the global RNG is untouched.

With ``W₂ = b₂ = 0`` every logit is 0: the initial prediction is uniform, the
initial loss is exactly ``ln 27``, and on the first backward pass only ``W₂``
and ``b₂`` receive gradient (``E``, ``W₁``, ``b₁`` get exactly zero because the
backpropagated signal is multiplied by ``W₂ = 0``). This is intended.

Reference training procedure (§4.6), separate from the architecture
-------------------------------------------------------------------
:func:`train_mlp` runs plain mini-batch SGD (no momentum, no weight decay) with
``EpochSampler`` batches (without replacement, final partial batch kept), mean
reduction over the *actual* batch size, an epoch-based learning rate, fixed
epoch budget, best checkpoint by lowest full-dev NLL, and epoch-boundary resume.

Device policy
-------------
Parameters are created on CPU; move the model with ``model.to(device)``.
``forward`` never moves data and rejects inputs on another device.

This module is independent of the other models in the repository.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from utils.batching import EpochSampler
from utils.checkpoint import load_checkpoint, restore_model, restore_training_state, save_checkpoint
from utils.data import ContextWindows
from utils.diagnostics.activations import ActivationMonitor, tanh_saturation
from utils.diagnostics.gradients import global_grad_norm, gradient_summaries, nonfinite_gradients
from utils.diagnostics.parameters import ParameterSnapshot
from utils.diagnostics.schedule import DiagnosticsSchedule
from utils.generation import sample_next_token
from utils.io import append_jsonl, atomic_write_text, dumps_json, read_jsonl
from utils.metrics import NLLAccumulator, module_group, token_nll
from utils.reproducibility import capture_rng_state, derive_seed, make_generator
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary, VocabularyError

MODEL_NAME = "mlp"
BEST_METRIC_NAME = "dev_nll"
SATURATION_THRESHOLD = 0.97
STEP_DIAGNOSTICS_FILE = "steps.jsonl"


class IncompatibleCheckpointError(ValueError):
    """Raised when a checkpoint cannot be loaded or resumed with the given model, data or settings."""


# =================================================================== configs
def _check_int(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")


def _check_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class MLPConfig:
    """Architecture configuration (``model.params`` in ``configs/models/mlp.yaml``).

    The defaults are the approved baseline. Other values are for experiments only.
    Initialisation is fixed in code and is not configurable.
    """

    context_length: int = 3
    embedding_dim: int = 10
    hidden_dim: int = 200

    def __post_init__(self) -> None:
        _check_int(self.context_length, "context_length")
        _check_int(self.embedding_dim, "embedding_dim")
        _check_int(self.hidden_dim, "hidden_dim")

    @property
    def fan_in(self) -> int:
        """Input width of the hidden layer: ``context_length · embedding_dim``."""
        return self.context_length * self.embedding_dim


@dataclass(frozen=True)
class MLPTrainingConfig:
    """Settings of the reference training procedure (§9). Model-local; not part of ``ExperimentConfig``.

    ``lr``, ``max_epochs`` and ``lr_decay_epoch`` are PENDING PILOT: they have no
    approved values yet. The optimiser (plain SGD, momentum 0, weight decay 0)
    and the loss reduction (mean) are fixed in code.
    """

    lr: float
    max_epochs: int
    batch_size: int = 32
    lr_decay_epoch: int | None = None
    lr_decay_factor: float = 0.1
    eval_batch_size: int = 4096

    def __post_init__(self) -> None:
        lr = _check_real(self.lr, "lr")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr!r}")
        object.__setattr__(self, "lr", lr)
        _check_int(self.max_epochs, "max_epochs")
        _check_int(self.batch_size, "batch_size")
        _check_int(self.eval_batch_size, "eval_batch_size")
        if self.lr_decay_epoch is not None:
            _check_int(self.lr_decay_epoch, "lr_decay_epoch")
            if self.lr_decay_epoch >= self.max_epochs:
                raise ValueError(
                    f"lr_decay_epoch must be < max_epochs={self.max_epochs}, got {self.lr_decay_epoch}"
                )
        factor = _check_real(self.lr_decay_factor, "lr_decay_factor")
        if not 0 < factor <= 1:
            raise ValueError(f"lr_decay_factor must be in (0, 1], got {self.lr_decay_factor!r}")
        object.__setattr__(self, "lr_decay_factor", factor)


# =================================================================== records
@dataclass(frozen=True)
class MLPSample:
    """One generated sample (same semantics as Models 1–2).

    text:
        Characters between BOS and EOS, or up to ``max_length``. Never contains ``.``.
    finished:
        ``True`` iff EOS was drawn. ``False`` means the sample was truncated at ``max_length``.
    """

    text: str
    finished: bool


@dataclass(frozen=True)
class EpochRecord:
    """One row of ``history.jsonl``: the state at the end of one epoch.

    The three NLLs are kept distinct (§4.6):
    ``running_train_nll`` while the parameters changed during the epoch;
    ``train_nll`` and ``dev_nll`` with fixed end-of-epoch parameters.
    """

    epoch: int  # 0-based index of the completed epoch
    lr: float
    steps: int  # cumulative optimiser steps after this epoch
    running_train_nll: float
    train_nll: float
    dev_nll: float
    is_best: bool
    seconds: float  # wall-clock time; the only non-deterministic field
    activations: dict  # epoch-end dev activation summary (§12)


@dataclass(frozen=True)
class MLPTrainingResult:
    best_epoch: int
    best_dev_nll: float
    epochs_completed: int
    history: list[EpochRecord]  # all epochs 0 .. epochs_completed-1 (earlier rows restored on resume)


# ===================================================================== model
class MLPLanguageModel(nn.Module):
    """Fixed-context MLP language model (§3–§8).

    Parameters
    ----------
    vocab:
        Shared vocabulary. ``V = vocab.num_classes`` output classes. PAD is
        input-only and must never reach the model.
    config:
        Validated :class:`MLPConfig`. Defaults to the approved baseline.
    generator:
        CPU ``torch.Generator`` used for every random initialisation draw
        (by convention ``make_generator(derive_seed(seed, "init"))``). Required.

    Submodules: ``emb`` (``nn.Embedding``), ``hidden`` (``nn.Linear``),
    ``act`` (``nn.Tanh``, registered so that it can be hooked) and ``out`` (``nn.Linear``).
    """

    def __init__(
        self,
        vocab: Vocabulary,
        config: MLPConfig = MLPConfig(),  # noqa: B008  (frozen, immutable dataclass: a shared default is safe)
        *,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"vocab must be a utils.vocabulary.Vocabulary, got {type(vocab).__name__}")
        if not isinstance(config, MLPConfig):
            raise TypeError(f"config must be an MLPConfig, got {type(config).__name__}")
        if not isinstance(generator, torch.Generator):
            raise TypeError(f"generator must be a torch.Generator, got {type(generator).__name__}")
        if generator.device.type != "cpu":
            raise ValueError(f"generator must be a CPU generator (parameters are initialised on CPU), "
                             f"got {generator.device}")
        self.vocab = vocab
        self.config = config
        V, d, H = vocab.num_classes, config.embedding_dim, config.hidden_dim
        # skip_init: allocate without running PyTorch's default initialisation (and without
        # touching the global RNG). Every parameter is then set explicitly below (§6.2, §7).
        self.emb = nn.utils.skip_init(nn.Embedding, V, d)
        self.hidden = nn.utils.skip_init(nn.Linear, config.fan_in, H)
        self.act = nn.Tanh()
        self.out = nn.utils.skip_init(nn.Linear, H, V)
        with torch.no_grad():
            self.emb.weight.normal_(0.0, 1.0, generator=generator)  # E ~ N(0, 1)
            self.hidden.weight.normal_(0.0, 1.0 / math.sqrt(config.fan_in), generator=generator)  # LeCun normal
            self.hidden.bias.zero_()
            self.out.weight.zero_()  # zero output layer: uniform initial prediction (§7)
            self.out.bias.zero_()

    @property
    def num_classes(self) -> int:
        return self.vocab.num_classes

    @property
    def context_length(self) -> int:
        return self.config.context_length

    @property
    def device(self) -> torch.device:
        return self.out.weight.device

    def extra_repr(self) -> str:
        c = self.config
        return f"V={self.num_classes}, k={c.context_length}, d={c.embedding_dim}, H={c.hidden_dim}"

    # ------------------------------------------------------------ forward
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """Next-token logits for a batch of context windows (§8).

        Parameters
        ----------
        contexts:
            ``[B, k]`` int64 token ids on the model's device, oldest token first,
            ids in ``[0, V)``. ``B = 0`` is allowed. PAD and out-of-range ids raise ``ValueError``.

        Returns
        -------
        torch.Tensor
            ``[B, V]`` float32 **logits** (not probabilities).
        """
        _check_contexts(contexts, context_length=self.context_length, num_classes=self.num_classes,
                        pad_id=self.vocab.pad_id, device=self.device)
        e = self.emb(contexts)  # [B, k, d]
        u = e.reshape(contexts.shape[0], self.config.fan_in)  # [B, k·d], u[:, d·p : d·(p+1)] = E[x_{p+1}]
        a = self.hidden(u)  # [B, H] pre-activations
        h = self.act(a)  # [B, H]
        return self.out(h)  # [B, V]

    # ---------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        *,
        max_length: int,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[MLPSample]:
        """Ancestral sampling from ``[BOS] * k`` (§15).

        After each drawn character the context shifts exactly as in
        ``make_context_windows``: ``context ← context[1:] + [token]``. A sample
        stops when EOS is drawn. ``max_length`` is the maximum number of
        ordinary characters (pass ``generation.max_new_tokens`` from the
        config). A sample that reaches it gets one more draw: EOS there gives
        ``finished=True``; anything else returns it truncated with
        ``finished=False``. So there are at most ``max_length + 1`` draws.

        Samples are drawn in parallel, one draw per active sample per step, via
        :func:`utils.generation.sample_next_token`. The defaults sample the model
        distribution exactly; ``temperature=0`` is greedy. Parameters and
        ``self.training`` are not modified.
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_samples must be a positive int, got {num_samples!r}")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError(f"max_length must be a positive int, got {max_length!r}")
        eos = self.vocab.eos_id
        contexts = torch.full((num_samples, self.context_length), self.vocab.bos_id,
                              dtype=torch.long, device=self.device)
        chars: list[list[int]] = [[] for _ in range(num_samples)]
        finished = [False] * num_samples
        active = list(range(num_samples))
        while active:
            rows = torch.tensor(active, device=self.device)
            logits = self(contexts[rows])  # [A, V]
            drawn = sample_next_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
            ).tolist()
            still_active = []
            for k, token in zip(active, drawn):
                if token == eos:
                    finished[k] = True
                elif len(chars[k]) == max_length:
                    pass  # limit reached and EOS not drawn: stop, truncated
                else:
                    chars[k].append(token)
                    still_active.append(k)
            if still_active:
                idx = torch.tensor(still_active, device=self.device)
                new = torch.tensor([chars[k][-1] for k in still_active], device=self.device).unsqueeze(1)
                contexts[idx] = torch.cat([contexts[idx, 1:], new], dim=1)
            active = still_active
        return [
            MLPSample(text=self.vocab.decode(ids, strip_bos=False, stop_at_eos=False), finished=done)
            for ids, done in zip(chars, finished)
        ]

    # ---------------------------------------------------------- checkpoint
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_vocab: Vocabulary | None = None,
        map_location: str | torch.device = "cpu",
    ) -> MLPLanguageModel:
        """Load weights from a checkpoint of any kind (§14), with ``weights_only=True``.

        Requires the checkpoint's ``vocab`` metadata and ``config["model"]``
        (both written by :func:`train_mlp`). Rejects a different model class, a
        vocabulary that differs from ``expected_vocab``, an invalid model config
        and weights whose shapes do not match it. Only weights are restored; use
        ``train_mlp(..., resume=True)`` to continue training.
        """
        ckpt = load_checkpoint(path, map_location="cpu")
        if ckpt.get("model_class") != cls.__name__:
            raise IncompatibleCheckpointError(f"{path} holds a {ckpt.get('model_class')!r}, not a {cls.__name__}")
        if ckpt.get("vocab") is None:
            raise IncompatibleCheckpointError(
                f"{path} has no vocabulary metadata; save with save_checkpoint(..., vocab=vocab.to_dict())"
            )
        try:
            vocab = Vocabulary.from_dict(ckpt["vocab"])
        except VocabularyError as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid vocabulary metadata: {exc}") from exc
        if expected_vocab is not None and expected_vocab != vocab:
            raise IncompatibleCheckpointError(
                f"vocabulary mismatch: checkpoint has {vocab.boundary!r}+{''.join(vocab.chars)!r}, expected "
                f"{expected_vocab.boundary!r}+{''.join(expected_vocab.chars)!r}"
            )
        model_cfg = (ckpt.get("config") or {}).get("model")
        if not isinstance(model_cfg, Mapping):
            raise IncompatibleCheckpointError(f"{path} has no config['model'] (the MLPConfig used to build it)")
        try:
            config = MLPConfig(**model_cfg)
        except (TypeError, ValueError) as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid model config {dict(model_cfg)!r}: {exc}") from exc
        # The weights are overwritten by the checkpoint; this generator only satisfies the constructor.
        model = cls(vocab, config, generator=make_generator(0))
        try:
            restore_model(model, ckpt, strict=True)
        except RuntimeError as exc:  # e.g. a size mismatch between the weights and the config
            raise IncompatibleCheckpointError(f"{path}: weights do not match the model config: {exc}") from exc
        return model.to(map_location)


# =================================================================== helpers
def _check_contexts(
    contexts: torch.Tensor, *, context_length: int, num_classes: int, pad_id: int, device: torch.device
) -> None:
    """Validate a context batch (§8). Metadata checks always run; the range check is empty-safe."""
    if not isinstance(contexts, torch.Tensor):
        raise TypeError(f"contexts must be a torch.Tensor, got {type(contexts).__name__}")
    if contexts.dtype != torch.int64:
        raise TypeError(f"contexts must be int64 token ids, got {contexts.dtype}")
    if contexts.ndim != 2 or contexts.shape[1] != context_length:
        raise ValueError(f"contexts must have shape [B, {context_length}], got {tuple(contexts.shape)}")
    if contexts.device != device:
        raise ValueError(f"contexts are on {contexts.device} but the model is on {device}; move them explicitly")
    # .any() is defined (False) for empty tensors; min()/max() are not, so they are never used here.
    out_of_range = (contexts < 0) | (contexts >= num_classes)
    if bool(out_of_range.any()):  # one host synchronisation
        if bool((contexts == pad_id).any()):
            raise ValueError(f"contexts contain the PAD id {pad_id}: padding must not reach the model")
        bad = sorted(set(contexts[out_of_range].tolist()))[:5]
        raise ValueError(f"contexts contain ids {bad} outside [0, {num_classes})")


def _check_windows(model: MLPLanguageModel, windows: ContextWindows, batch_size: int) -> None:
    if not isinstance(windows, ContextWindows):
        raise TypeError(f"windows must be a ContextWindows, got {type(windows).__name__}")
    if windows.context_length != model.context_length:
        raise ValueError(f"windows have context_length {windows.context_length}, "
                         f"the model expects {model.context_length}")
    _check_int(batch_size, "batch_size")


def _learning_rate_at(epoch: int, training: MLPTrainingConfig) -> float:
    """Epoch-based, stateless learning rate (§4.6): recomputed identically after resume."""
    if training.lr_decay_epoch is None or epoch < training.lr_decay_epoch:
        return training.lr
    return training.lr * training.lr_decay_factor


def run_config(experiment: Mapping[str, Any] | None, model_config: MLPConfig,
               training: MLPTrainingConfig) -> dict[str, Any]:
    """The resolved configuration stored in every checkpoint (§14).

    Pass the same dict to ``RunManager.create(config=...)`` so that ``config.json``
    records the model and training settings as well.
    """
    return {
        "experiment": dict(experiment) if experiment is not None else None,
        "model": asdict(model_config),
        "training": asdict(training),
    }


# ================================================================ evaluation
def evaluate_nll(model: MLPLanguageModel, windows: ContextWindows, *, batch_size: int) -> NLLAccumulator:
    """Example-weighted NLL over **every** window, in chunks of ``batch_size`` (§4.6).

    Runs in eval mode under ``no_grad`` with fixed parameters. The final partial
    chunk is included. The model's original train/eval mode is restored, even
    if an exception occurs.
    """
    _check_windows(model, windows, batch_size)
    was_training = model.training
    acc = NLLAccumulator()
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, windows.num_examples, batch_size):
                xb = windows.contexts[start:start + batch_size].to(model.device)
                yb = windows.targets[start:start + batch_size].to(model.device)
                acc.update(model(xb), yb)
    finally:
        model.train(was_training)
    return acc


def per_unit_saturation(
    model: MLPLanguageModel,
    windows: ContextWindows,
    *,
    threshold: float = SATURATION_THRESHOLD,
    batch_size: int,
) -> torch.Tensor:
    """``[H]`` float64 CPU tensor: per hidden unit, the fraction of windows with ``|h| > threshold`` (§12).

    Uses running counts (no ``[M, H]`` matrix is stored), eval mode and ``no_grad``.
    The original train/eval mode is restored, even if an exception occurs.
    """
    _check_windows(model, windows, batch_size)
    threshold = _check_real(threshold, "threshold")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
    if windows.num_examples == 0:
        raise ValueError("per_unit_saturation needs at least one window")
    counts = torch.zeros(model.config.hidden_dim, dtype=torch.int64)

    def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        counts.add_((output.abs() > threshold).sum(dim=0).cpu())

    was_training = model.training
    handle = model.act.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, windows.num_examples, batch_size):
                model(windows.contexts[start:start + batch_size].to(model.device))
    finally:
        handle.remove()
        model.train(was_training)
    return counts.double() / windows.num_examples


def _epoch_activation_summary(model: MLPLanguageModel, dev: ContextWindows, batch_size: int,
                              dev_monitor_summaries: dict[str, Any]) -> dict[str, Any]:
    unit = per_unit_saturation(model, dev, threshold=SATURATION_THRESHOLD, batch_size=batch_size)
    return {
        **dev_monitor_summaries,  # "hidden": pre-activation moments; "act": tanh outputs + saturation_fraction
        "per_unit_saturation": {
            "threshold": SATURATION_THRESHOLD,
            "min": float(unit.min()),
            "median": float(unit.median()),
            "max": float(unit.max()),
            "units_ge_0.99": int((unit >= 0.99).sum()),
        },
    }


# ================================================================== training
def _check_finite_gradients(model: MLPLanguageModel, epoch: int, step: int) -> None:
    """Step 5 of §4.6: one fused check (one host sync); details only after a failure."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
        return
    bad = nonfinite_gradients(model)
    groups = sorted({module_group(name) for name in bad})
    raise FloatingPointError(
        f"non-finite gradients at epoch {epoch}, step {step} in parameter groups {groups}: {bad}"
    )


def _truncate_history(run: RunManager, completed_epochs: int) -> list[EpochRecord]:
    """Keep only rows of completed epochs (§14); rewrite atomically. Returns the kept rows."""
    if not run.history_path.is_file():
        return []
    rows = [r for r in read_jsonl(run.history_path) if r["epoch"] < completed_epochs]
    atomic_write_text(run.history_path, "".join(dumps_json(r, indent=None) + "\n" for r in rows))
    return [EpochRecord(**r) for r in rows]


def _check_resume_compatible(ckpt: Mapping[str, Any], *, model: MLPLanguageModel, training: MLPTrainingConfig,
                             seed: int, num_examples: int) -> None:
    """Reject any change that would make the resumed run differ from the original (§14)."""
    config = ckpt.get("config") or {}
    extra = ckpt.get("extra") or {}
    sampler = extra.get("sampler") or {}
    stored_training = config.get("training") or {}
    checks: list[tuple[str, Any, Any]] = [
        ("model config", config.get("model"), asdict(model.config)),
        *((f"training.{f}", stored_training.get(f), getattr(training, f))
          for f in ("lr", "max_epochs", "batch_size", "lr_decay_epoch", "lr_decay_factor")),
        ("sampler batch size", sampler.get("batch_size"), training.batch_size),
        ("sampler seed", sampler.get("seed"), derive_seed(seed, "data")),
        ("base seed", extra.get("base_seed"), seed),
        ("vocabulary", ckpt.get("vocab"), model.vocab.to_dict()),
        ("dataset size", sampler.get("num_examples"), num_examples),
    ]
    for name, stored, current in checks:
        if stored != current:
            raise IncompatibleCheckpointError(
                f"cannot resume: {name} changed (checkpoint {stored!r}, now {current!r})"
            )


def train_mlp(
    model: MLPLanguageModel,
    train: ContextWindows,
    dev: ContextWindows,
    *,
    training: MLPTrainingConfig,
    seed: int,
    run: RunManager | None = None,
    resume: bool = False,
    diagnostics: DiagnosticsSchedule | None = None,
    config: Mapping[str, Any] | None = None,
) -> MLPTrainingResult:
    """Reference training procedure (§4.6), with checkpoints, history and resume (§14).

    Per epoch ``e`` (0-based): set ``η_e`` on the optimiser; train on
    ``EpochSampler`` batches with mean reduction over the actual batch size;
    evaluate full train and full dev NLL with fixed parameters (plus the dev
    activation summary); then, with ``run``:

    1. save ``model_best.pt`` if dev NLL strictly improved (ties keep the earlier epoch);
    2. append the :class:`EpochRecord` to ``history.jsonl``;
    3. save ``checkpoint_last.pt`` with ``epoch = e + 1`` (the commit point);
    4. after the last epoch, save ``model_final.pt``.

    Each step runs: forward → loss → finite-loss check → backward →
    finite-gradient check → diagnostics → ``optimizer.step()``. A non-finite
    loss or gradient raises ``FloatingPointError`` before the step. Wrap the
    call in ``run.track()`` to record failures in the run metadata.

    Parameters
    ----------
    model:
        Trained in place, on its current device. Build it with an ``"init"`` generator.
    train, dev:
        Windows from ``make_context_windows`` (CPU); moved to the model device here.
    training:
        :class:`MLPTrainingConfig`.
    seed:
        Base seed; the data order uses ``derive_seed(seed, "data")``.
    run:
        Optional :class:`RunManager`. Without it, nothing is written to disk.
    resume:
        Continue from ``run``'s ``checkpoint_last.pt`` under the **original**
        ``max_epochs`` and schedule. Requires ``run``. Incompatible changes raise
        :class:`IncompatibleCheckpointError`.
    diagnostics:
        Optional step-level schedule. Records go to ``<run>/diagnostics/steps.jsonl``,
        so ``run`` is required when diagnostics are enabled.
    config:
        The resolved experiment configuration (plain data), stored with checkpoints.
    """
    if not isinstance(model, MLPLanguageModel):
        raise TypeError(f"model must be an MLPLanguageModel, got {type(model).__name__}")
    if not isinstance(training, MLPTrainingConfig):
        raise TypeError(f"training must be an MLPTrainingConfig, got {type(training).__name__}")
    for name, w in (("train", train), ("dev", dev)):
        _check_windows(model, w, training.eval_batch_size)
        if w.num_examples == 0:
            raise ValueError(f"{name} has no windows")
    if resume and run is None:
        raise ValueError("resume=True requires a run")
    if diagnostics is not None and diagnostics.enabled and run is None:
        raise ValueError("step diagnostics are written to the run directory; pass a run")

    device = model.device
    train_x, train_y = train.contexts.to(device), train.targets.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=training.lr, momentum=0.0, weight_decay=0.0)
    sampler = EpochSampler(train.num_examples, training.batch_size, seed=derive_seed(seed, "data"))
    cfg = run_config(config, model.config, training)
    vocab_dict = model.vocab.to_dict()

    start_epoch, step = 0, 0
    best_dev: float | None = None
    best_epoch = -1
    history: list[EpochRecord] = []
    if resume:
        ckpt = load_checkpoint(run.checkpoint_path("resume"), map_location="cpu", expected_kind="resume")
        _check_resume_compatible(ckpt, model=model, training=training, seed=seed, num_examples=train.num_examples)
        progress = restore_training_state(ckpt, model=model, optimizer=optimizer)
        sampler = EpochSampler.from_state_dict(ckpt["extra"]["sampler"], num_examples=train.num_examples,
                                               batch_size=training.batch_size)
        start_epoch, step = int(progress.epoch), progress.step
        best_dev, best_epoch = progress.best_metric, int(ckpt["extra"]["best_epoch"])
        history = _truncate_history(run, start_epoch)
    elif run is not None and (run.history_path.is_file() or run.checkpoint_path("resume").is_file()):
        raise ValueError(f"{run.run_dir} already holds training state; pass resume=True to continue it")

    for epoch in range(start_epoch, training.max_epochs):
        t0 = time.perf_counter()
        lr = _learning_rate_at(epoch, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        running = NLLAccumulator()
        for idx in sampler.batches(epoch):
            idx = idx.to(device)
            xb, yb = train_x[idx], train_y[idx]
            collect = {k: diagnostics is not None and diagnostics.should_collect(step, k)
                       for k in ("gradients", "parameters", "activations")}
            optimizer.zero_grad(set_to_none=True)
            monitor = (ActivationMonitor(model, module_names=["hidden", "act"],
                                         saturation={"act": tanh_saturation(SATURATION_THRESHOLD)})
                       if collect["activations"] else None)
            # 1. forward (the activation monitor, if any, observes this pass)
            if monitor is not None:
                with monitor:
                    logits = model(xb)
            else:
                logits = model(xb)
            # 2. loss: mean over the actual batch size |b| (also for the final partial batch)
            loss = token_nll(logits, yb, reduction="mean")
            # 3. finite-loss check
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss ({loss.item()}) at epoch {epoch}, step {step}")
            # 4. backward
            loss.backward()
            # 5. finite-gradient check
            _check_finite_gradients(model, epoch, step)
            # 6. diagnostics (scheduled steps only; scalars, no graphs)
            record: dict[str, Any] | None = None
            snapshot = None
            if any(collect.values()):
                record = {"epoch": epoch, "step": step, "lr": lr, "batch_size": int(yb.shape[0]),
                          "loss": float(loss.detach())}
                if collect["gradients"]:
                    record["grad_global_norm"] = global_grad_norm(model)
                    record["gradients"] = gradient_summaries(model, group_by="module")
                    record["nonfinite_gradients"] = nonfinite_gradients(model)  # {} after the step-5 check
                if collect["activations"]:
                    record["activations"] = monitor.summaries()
                if collect["parameters"]:
                    snapshot = ParameterSnapshot.capture(model)
            # 7. optimiser step
            optimizer.step()
            if record is not None:
                if snapshot is not None:
                    record["updates"] = snapshot.update_stats(model)
                append_jsonl(run.diagnostics_dir / STEP_DIAGNOSTICS_FILE, record)
            running.update(logits, yb)
            step += 1

        train_nll = evaluate_nll(model, train, batch_size=training.eval_batch_size).mean_nll
        with ActivationMonitor(model, module_names=["hidden", "act"],
                               saturation={"act": tanh_saturation(SATURATION_THRESHOLD)}) as dev_monitor:
            dev_nll = evaluate_nll(model, dev, batch_size=training.eval_batch_size).mean_nll
        activations = _epoch_activation_summary(model, dev, training.eval_batch_size, dev_monitor.summaries())
        is_best = best_dev is None or dev_nll < best_dev  # strict: exact ties keep the earliest epoch
        if is_best:
            best_dev, best_epoch = dev_nll, epoch
        rec = EpochRecord(epoch=epoch, lr=lr, steps=step, running_train_nll=running.mean_nll,
                          train_nll=train_nll, dev_nll=dev_nll, is_best=is_best,
                          seconds=time.perf_counter() - t0, activations=activations)
        history.append(rec)

        if run is not None:
            if is_best:
                save_checkpoint(run.checkpoint_path("best"), kind="best", model=model, step=step,
                                epoch=epoch + 1, best_metric=dev_nll, best_metric_name=BEST_METRIC_NAME,
                                config=cfg, vocab=vocab_dict, extra={"best_epoch": epoch})
            run.append_history(asdict(rec))
            save_checkpoint(run.checkpoint_path("resume"), kind="resume", model=model, optimizer=optimizer,
                            step=step, epoch=epoch + 1, best_metric=best_dev, best_metric_name=BEST_METRIC_NAME,
                            rng_state=capture_rng_state(), config=cfg, vocab=vocab_dict,
                            extra={"best_epoch": best_epoch, "sampler": sampler.state_dict(), "base_seed": seed})

    if run is not None:
        save_checkpoint(run.checkpoint_path("final"), kind="final", model=model, step=step,
                        epoch=training.max_epochs, config=cfg, vocab=vocab_dict)
    return MLPTrainingResult(best_epoch=best_epoch, best_dev_nll=best_dev, epochs_completed=training.max_epochs,
                             history=history)

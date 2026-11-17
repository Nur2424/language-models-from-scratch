"""BatchNorm MLP: the Model 3 MLP with batch normalisation before the tanh.

Specification: ``models/feedforward/batchnorm_mlp_spec.md`` (v0.2). Section numbers
below (§) refer to it. The training procedure, checkpoints, resume and history
mirror ``models/feedforward/mlp.py`` (no code is shared: each model file is
self-contained by design).

Architecture (§3, §4.1)
-----------------------
For a batch of contexts ``x_n = (x_{n,1}, x_{n,2}, x_{n,3})`` (oldest token first)::

    u_n = [E[x_{n,1}]; E[x_{n,2}]; E[x_{n,3}]]     E  ∈ ℝ^{27×10}, one shared table
    a_n = W₁ u_n                                   W₁ ∈ ℝ^{200×30}, no bias (β replaces it)
    y_n = BN(a_n)                                  per hidden unit, mode-dependent (below)
    h_n = tanh(y_n)
    z_n = W₂ h_n + b₂                              → 27 logits

12,097 trainable parameters.

Two modes (§3.1, §4.1)
----------------------
* **Train mode** (only for SGD forwards): ``y = γ (a − μ_B)/√(σ²_B + ε) + β`` with the
  batch mean and *biased* batch variance. PyTorch also updates its EMA buffers
  ``running_mean`` / ``running_var`` (momentum 0.1). The EMA is kept for
  diagnostics only.
* **Eval mode** (evaluation, diagnostics, generation): ``y = γ (a − m)/√(v + ε) + β``
  with ``m`` / ``v`` = ``inference_mean`` / ``inference_var``: the exact mean and
  biased variance of ``a`` over the whole training set, recomputed after every
  epoch by :func:`calibrate_batchnorm`. The EMA is never read here.

Initialisation (§7)
-------------------
``E ~ N(0, 1)``; ``W₁ ~ N(0, 1/fan_in)``; ``γ = 1``; ``β = 0``; ``W₂ = b₂ = 0``;
EMA buffers ``0 / 1``; inference buffers ``0 / 1``; counters ``0``. All set
explicitly (modules are built with ``nn.utils.skip_init``); random draws use the
explicit CPU ``generator`` only. With ``W₂ = 0`` the logits are 0 in both modes,
the initial loss is ``ln 27``, and on the first backward pass only ``W₂`` and ``b₂``
receive gradient. This is intended.

Device policy
-------------
Parameters and buffers are created on CPU; move the model with ``model.to(device)``.
``forward`` never moves data. Recalibration merges moments in float64 on CPU on
every backend (MPS has no float64).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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

MODEL_NAME = "batchnorm_mlp"
BEST_METRIC_NAME = "dev_nll"
SATURATION_THRESHOLD = 0.97
STEP_DIAGNOSTICS_FILE = "steps.jsonl"

BN_EPS = 1e-5  # approved constant (not configurable)
BN_MOMENTUM = 0.1  # approved constant; drives the diagnostic EMA only
#: Recorded in every checkpoint and config.json; checked on resume and load (§9).
BATCHNORM_RECORD = {"eps": BN_EPS, "momentum": BN_MOMENTUM, "inference_statistics": "full_train_biased"}
MONITORED_MODULES = ("hidden", "bn", "act")  # pre-BN, post-BN, tanh


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
class BatchNormMLPConfig:
    """Architecture configuration (``model.params`` in ``configs/models/batchnorm_mlp.yaml``).

    The defaults are the approved baseline. ε, momentum and the initialisation
    are fixed in code and are not configurable.
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
class BatchNormMLPTrainingConfig:
    """Settings of the reference training procedure (§9). Model-local; not part of ``ExperimentConfig``.

    ``lr`` is PENDING PILOT; ``max_epochs`` must equal the final MLP value (PENDING
    PILOT there). ``batch_size >= 2``: BatchNorm needs at least two windows per
    training batch. ``eval_batch_size`` chunks evaluation *and* recalibration.
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
        _check_int(self.batch_size, "batch_size", minimum=2)
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
class BatchNormMLPSample:
    """One generated sample (same semantics as Models 1–3).

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

    ``running_train_nll``: train mode (batch statistics), parameters changing during the epoch.
    ``train_nll`` / ``dev_nll``: eval mode, recalibrated statistics, fixed end-of-epoch parameters.
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
    batchnorm: dict  # epoch-end BatchNorm summary (§12)


@dataclass(frozen=True)
class BatchNormMLPTrainingResult:
    best_epoch: int
    best_dev_nll: float
    epochs_completed: int
    history: list[EpochRecord]  # all epochs 0 .. epochs_completed-1 (earlier rows restored on resume)


# ============================================================ normalisation
class CalibratedBatchNorm1d(nn.BatchNorm1d):
    """``nn.BatchNorm1d`` whose eval mode uses separately stored, recalibrated statistics (§3.1).

    Parameters: ``weight`` (γ) and ``bias`` (β).
    EMA buffers (PyTorch, diagnostic only): ``running_mean``, ``running_var``, ``num_batches_tracked``.
    Authoritative inference buffers: ``inference_mean``, ``inference_var``, ``num_calibrations``.

    * Train mode: standard PyTorch batch normalisation (biased batch variance) and EMA update.
    * Eval mode: ``F.batch_norm`` with the inference buffers; the EMA is never read or written.

    ε and momentum are the approved constants. Values are set by the owning model
    (all parameters and buffers are initialised explicitly there).
    """

    def __init__(self, num_features: int, *, device: Any = None, dtype: torch.dtype | None = None) -> None:
        super().__init__(num_features, eps=BN_EPS, momentum=BN_MOMENTUM, affine=True, track_running_stats=True,
                         device=device, dtype=dtype)
        self.register_buffer("inference_mean", torch.empty(num_features, device=device, dtype=dtype))
        self.register_buffer("inference_var", torch.empty(num_features, device=device, dtype=dtype))
        self.register_buffer("num_calibrations", torch.empty((), device=device, dtype=torch.long))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if self.training:
            return super().forward(a)  # batch statistics; updates running_mean / running_var
        self._check_input_dim(a)
        return F.batch_norm(a, self.inference_mean, self.inference_var, self.weight, self.bias,
                            training=False, momentum=0.0, eps=self.eps)

    def set_inference_statistics(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        """Install recalibrated statistics and increment ``num_calibrations`` (never touches the EMA)."""
        for name, t in (("mean", mean), ("var", var)):
            if not isinstance(t, torch.Tensor) or t.shape != (self.num_features,):
                raise ValueError(f"{name} must be a tensor of shape ({self.num_features},)")
            if not bool(torch.isfinite(t).all()):
                raise FloatingPointError(f"non-finite inference {name}")
        if bool((var < 0).any()):
            raise FloatingPointError("negative inference variance")
        with torch.no_grad():
            self.inference_mean.copy_(mean)
            self.inference_var.copy_(var)
            self.num_calibrations.add_(1)


# ===================================================================== model
class BatchNormMLPLanguageModel(nn.Module):
    """Fixed-context MLP language model with batch normalisation before the tanh (§3–§8).

    Parameters
    ----------
    vocab:
        Shared vocabulary. ``V = vocab.num_classes`` output classes. PAD never reaches the model.
    config:
        Validated :class:`BatchNormMLPConfig`. Defaults to the approved baseline.
    generator:
        CPU ``torch.Generator`` for every random initialisation draw
        (by convention ``make_generator(derive_seed(seed, "init"))``). Required.

    Submodules: ``emb``, ``hidden`` (no bias), ``bn`` (:class:`CalibratedBatchNorm1d`),
    ``act`` (``nn.Tanh``) and ``out``.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        config: BatchNormMLPConfig = BatchNormMLPConfig(),  # noqa: B008  (frozen, immutable dataclass)
        *,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"vocab must be a utils.vocabulary.Vocabulary, got {type(vocab).__name__}")
        if not isinstance(config, BatchNormMLPConfig):
            raise TypeError(f"config must be a BatchNormMLPConfig, got {type(config).__name__}")
        if not isinstance(generator, torch.Generator):
            raise TypeError(f"generator must be a torch.Generator, got {type(generator).__name__}")
        if generator.device.type != "cpu":
            raise ValueError(f"generator must be a CPU generator (parameters are initialised on CPU), "
                             f"got {generator.device}")
        self.vocab = vocab
        self.config = config
        V, d, H = vocab.num_classes, config.embedding_dim, config.hidden_dim
        # skip_init: allocate without running PyTorch's default initialisation (and without
        # touching the global RNG). Every parameter and buffer is then set explicitly (§7).
        self.emb = nn.utils.skip_init(nn.Embedding, V, d)
        self.hidden = nn.utils.skip_init(nn.Linear, config.fan_in, H, bias=False)
        self.bn = nn.utils.skip_init(CalibratedBatchNorm1d, H)
        self.act = nn.Tanh()
        self.out = nn.utils.skip_init(nn.Linear, H, V)
        with torch.no_grad():
            self.emb.weight.normal_(0.0, 1.0, generator=generator)  # E ~ N(0, 1)
            self.hidden.weight.normal_(0.0, 1.0 / math.sqrt(config.fan_in), generator=generator)  # LeCun
            self.bn.weight.fill_(1.0)  # γ
            self.bn.bias.zero_()  # β
            self.bn.running_mean.zero_()  # EMA start
            self.bn.running_var.fill_(1.0)
            self.bn.num_batches_tracked.zero_()
            self.bn.inference_mean.zero_()  # placeholders until the first recalibration
            self.bn.inference_var.fill_(1.0)
            self.bn.num_calibrations.zero_()
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
    def _check(self, contexts: torch.Tensor) -> None:
        _check_contexts(contexts, context_length=self.context_length, num_classes=self.num_classes,
                        pad_id=self.vocab.pad_id, device=self.device)

    def _preactivations(self, contexts: torch.Tensor) -> torch.Tensor:
        """Steps 3–5 of §8 (no validation): embedding → concatenation → ``hidden``."""
        e = self.emb(contexts)  # [B, k, d]
        u = e.reshape(contexts.shape[0], self.config.fan_in)  # [B, k·d], position-major
        return self.hidden(u)  # [B, H], no bias

    def preactivations(self, contexts: torch.Tensor) -> torch.Tensor:
        """Pre-BatchNorm activations ``a = W₁u`` (``[B, H]``). Mode-independent; never calls ``bn``."""
        self._check(contexts)
        return self._preactivations(contexts)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """Next-token logits for a batch of context windows (§8).

        ``contexts``: ``[B, k]`` int64 ids on the model's device, oldest first, ids in ``[0, V)``.
        Eval mode accepts any ``B >= 0``. Train mode requires ``B >= 2`` (checked before
        ``bn``, so a rejected batch changes no buffer). Returns ``[B, V]`` float32 logits.
        """
        self._check(contexts)
        if self.training and contexts.shape[0] < 2:
            raise ValueError(
                f"BatchNorm needs at least 2 windows per training batch, got {contexts.shape[0]}"
            )
        a = self._preactivations(contexts)
        y = self.bn(a)  # train: batch statistics (+ EMA update); eval: inference statistics
        h = self.act(y)
        return self.out(h)

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
    ) -> list[BatchNormMLPSample]:
        """Ancestral sampling from ``[BOS] * k`` in **eval mode** (§15).

        The model is switched to eval mode (inference statistics, no batch
        dependence, no buffer update) and its previous mode is restored in
        ``finally``. Context shifting, EOS and ``max_length`` semantics are those of
        Models 1–3: at most ``max_length`` ordinary characters and at most
        ``max_length + 1`` draws per sample.
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(f"num_samples must be a positive int, got {num_samples!r}")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError(f"max_length must be a positive int, got {max_length!r}")
        was_training = self.training
        try:
            self.eval()
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
        finally:
            self.train(was_training)
        return [
            BatchNormMLPSample(text=self.vocab.decode(ids, strip_bos=False, stop_at_eos=False), finished=done)
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
    ) -> BatchNormMLPLanguageModel:
        """Load weights and buffers from a checkpoint of any kind (§14), with ``weights_only=True``.

        Requires ``vocab``, ``config["model"]`` and matching ``config["batchnorm"]``
        constants. Rejects a different model class, a vocabulary different from
        ``expected_vocab``, an invalid or shape-mismatched model config, and an
        uncalibrated model (``num_calibrations == 0``).
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
        config = ckpt.get("config") or {}
        model_cfg = config.get("model")
        if not isinstance(model_cfg, Mapping):
            raise IncompatibleCheckpointError(f"{path} has no config['model'] (the BatchNormMLPConfig used)")
        if config.get("batchnorm") != BATCHNORM_RECORD:
            raise IncompatibleCheckpointError(
                f"{path}: BatchNorm constants {config.get('batchnorm')!r} differ from {BATCHNORM_RECORD!r}"
            )
        try:
            model_config = BatchNormMLPConfig(**model_cfg)
        except (TypeError, ValueError) as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid model config {dict(model_cfg)!r}: {exc}") from exc
        # The weights and buffers are overwritten by the checkpoint; this generator only satisfies the constructor.
        model = cls(vocab, model_config, generator=make_generator(0))
        try:
            restore_model(model, ckpt, strict=True)
        except RuntimeError as exc:  # e.g. a size mismatch between the weights and the config
            raise IncompatibleCheckpointError(f"{path}: weights do not match the model config: {exc}") from exc
        if int(model.bn.num_calibrations) == 0:
            raise IncompatibleCheckpointError(
                f"{path}: the BatchNorm inference statistics were never calibrated (num_calibrations == 0)"
            )
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


def _check_windows(model: BatchNormMLPLanguageModel, windows: ContextWindows, batch_size: int) -> None:
    if not isinstance(windows, ContextWindows):
        raise TypeError(f"windows must be a ContextWindows, got {type(windows).__name__}")
    if windows.context_length != model.context_length:
        raise ValueError(f"windows have context_length {windows.context_length}, "
                         f"the model expects {model.context_length}")
    _check_int(batch_size, "batch_size")


def _learning_rate_at(epoch: int, training: BatchNormMLPTrainingConfig) -> float:
    """Epoch-based, stateless learning rate: recomputed identically after resume."""
    if training.lr_decay_epoch is None or epoch < training.lr_decay_epoch:
        return training.lr
    return training.lr * training.lr_decay_factor


def run_config(experiment: Mapping[str, Any] | None, model_config: BatchNormMLPConfig,
               training: BatchNormMLPTrainingConfig) -> dict[str, Any]:
    """The resolved configuration stored in every checkpoint and ``config.json`` (§9, §14)."""
    return {
        "experiment": dict(experiment) if experiment is not None else None,
        "model": asdict(model_config),
        "training": asdict(training),
        "batchnorm": dict(BATCHNORM_RECORD),
    }


# ============================================================= recalibration
def _chunk_moments(a: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Per-unit ``(n, mean, M2)`` of one chunk, centred two-pass, in float64 on CPU (§4.7 step 2)."""
    a64 = a.detach().to(device="cpu").to(dtype=torch.float64)  # transfer, then cast (MPS has no float64)
    mean = a64.mean(dim=0)
    m2 = (a64 - mean).square().sum(dim=0)
    return a64.shape[0], mean, m2


def _merge_moments(
    n: int, mean: torch.Tensor, m2: torch.Tensor, n_c: int, mean_c: torch.Tensor, m2_c: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Chan et al. parallel update of per-unit moments (§4.7 step 3). Exact for ``n = 0``."""
    total = n + n_c
    delta = mean_c - mean
    return total, mean + delta * (n_c / total), m2 + m2_c + delta.square() * (n * n_c / total)


def calibrate_batchnorm(model: BatchNormMLPLanguageModel, windows: ContextWindows, *,
                        batch_size: int) -> dict[str, int]:
    """Recompute and install the BatchNorm inference statistics over ``windows`` (§4.7).

    ``inference_mean`` / ``inference_var`` become the exact per-unit mean and *biased*
    variance (divisor M) of the pre-BatchNorm activations ``a = W₁u`` of every window,
    under the current parameters.

    Chunks of ``batch_size`` consecutive windows are processed strictly one at a
    time: compute ``a`` for the chunk (on the model device, without calling ``bn``),
    reduce it to per-unit ``(n, mean, M2)`` in float64 on CPU, merge with the
    Chan et al. update, release the chunk. Only ``(n, mean, M2)`` is carried
    between chunks; the ``[M, H]`` activation matrix is never formed.

    Reads and writes no EMA buffer, changes no parameter and not ``model.training``;
    the result does not depend on previous buffer values. Increments ``num_calibrations``.
    """
    _check_windows(model, windows, batch_size)
    total = windows.num_examples
    if total == 0:
        raise ValueError("calibrate_batchnorm needs at least one window")
    n = 0
    mean = torch.zeros(model.config.hidden_dim, dtype=torch.float64)
    m2 = torch.zeros(model.config.hidden_dim, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            chunk = model.preactivations(windows.contexts[start:start + batch_size].to(model.device))
            n_c, mean_c, m2_c = _chunk_moments(chunk)
            del chunk  # release the chunk before the next one is computed
            n, mean, m2 = _merge_moments(n, mean, m2, n_c, mean_c, m2_c)
    if n != total:
        raise FloatingPointError(f"recalibration covered {n} of {total} windows")
    var = m2 / n
    if not (bool(torch.isfinite(mean).all()) and bool(torch.isfinite(var).all())):
        raise FloatingPointError("non-finite BatchNorm statistics during recalibration")
    # Cast to the buffers' dtype (float32 for the baseline model; float64 for a float64 test copy).
    dtype = model.bn.inference_mean.dtype
    model.bn.set_inference_statistics(mean.to(dtype), var.to(dtype))
    return {"num_windows": total, "num_calibrations": int(model.bn.num_calibrations)}


# ================================================================ evaluation
def evaluate_nll(model: BatchNormMLPLanguageModel, windows: ContextWindows, *, batch_size: int) -> NLLAccumulator:
    """Example-weighted NLL over **every** window, in eval mode (recalibrated statistics).

    Runs under ``no_grad`` with fixed parameters; the final partial chunk is included.
    Never updates BatchNorm buffers. The original train/eval mode is restored in
    ``finally``, even if an exception occurs.
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
    model: BatchNormMLPLanguageModel,
    windows: ContextWindows,
    *,
    threshold: float = SATURATION_THRESHOLD,
    batch_size: int,
) -> torch.Tensor:
    """``[H]`` float64 CPU tensor: per hidden unit, the fraction of windows with ``|h| > threshold``.

    Eval mode (recalibrated statistics), ``no_grad``, running counts only. The original
    train/eval mode is restored in ``finally``, even if an exception occurs.
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


def _summary(t: torch.Tensor) -> dict[str, float]:
    t = t.detach().to(device="cpu").to(dtype=torch.float64)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "min": float(t.min()),
            "max": float(t.max())}


def _batchnorm_summary(model: BatchNormMLPLanguageModel) -> dict[str, Any]:
    """Epoch-end BatchNorm summary (§12): EMA vs inference statistics, γ, β, ‖W₁ rows‖, counters.

    ``r_j`` compares the effective variances ``v + ε`` (the denominators actually used),
    which keeps it finite if a unit's variance is exactly zero.
    """
    bn = model.bn
    with torch.no_grad():
        im = bn.inference_mean.detach().to(device="cpu").to(dtype=torch.float64)
        iv = bn.inference_var.detach().to(device="cpu").to(dtype=torch.float64)
        rm = bn.running_mean.detach().to(device="cpu").to(dtype=torch.float64)
        rv = bn.running_var.detach().to(device="cpu").to(dtype=torch.float64)
        z = ((rm - im) / torch.sqrt(iv + bn.eps)).abs()
        r = torch.log((rv + bn.eps) / (iv + bn.eps)).abs()
        rows = model.hidden.weight.detach().to(device="cpu").to(dtype=torch.float64).norm(dim=1)
        return {
            "ema_vs_inference": {"mean_abs_z": float(z.mean()), "max_abs_z": float(z.max()),
                                 "median_abs_log_var_ratio": float(r.median()), "max_abs_log_var_ratio": float(r.max())},
            "gamma": _summary(bn.weight),
            "beta": _summary(bn.bias),
            "w1_row_norm": {k: v for k, v in _summary(rows).items() if k != "std"},
            "num_calibrations": int(bn.num_calibrations),
            "num_batches_tracked": int(bn.num_batches_tracked),
        }


def _epoch_activation_summary(model: BatchNormMLPLanguageModel, dev: ContextWindows, batch_size: int,
                              dev_monitor_summaries: dict[str, Any]) -> dict[str, Any]:
    unit = per_unit_saturation(model, dev, threshold=SATURATION_THRESHOLD, batch_size=batch_size)
    return {
        **dev_monitor_summaries,  # "hidden": pre-BN; "bn": post-BN; "act": tanh outputs + saturation_fraction
        "per_unit_saturation": {
            "threshold": SATURATION_THRESHOLD,
            "min": float(unit.min()),
            "median": float(unit.median()),
            "max": float(unit.max()),
            "units_ge_0.99": int((unit >= 0.99).sum()),
        },
    }


def _monitor(model: BatchNormMLPLanguageModel) -> ActivationMonitor:
    return ActivationMonitor(model, module_names=list(MONITORED_MODULES),
                             saturation={"act": tanh_saturation(SATURATION_THRESHOLD)})


# ================================================================== training
def _check_finite_gradients(model: BatchNormMLPLanguageModel, epoch: int, step: int) -> None:
    """Step 5 of the step order: one fused check (one host sync); details only after a failure."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
        return
    bad = nonfinite_gradients(model)
    groups = sorted({module_group(name) for name in bad})
    raise FloatingPointError(
        f"non-finite gradients at epoch {epoch}, step {step} in parameter groups {groups}: {bad}"
    )


def _check_training_batches(sampler: EpochSampler, num_examples: int, batch_size: int) -> None:
    """Every training batch must hold >= 2 windows (§4.6): reject ``M mod batch_size == 1``."""
    if min(sampler.batch_sizes()) < 2:
        raise ValueError(
            f"{num_examples} training windows with batch_size {batch_size} give a final batch of 1 window; "
            "BatchNorm needs at least 2 windows per training batch"
        )


def _truncate_history(run: RunManager, completed_epochs: int) -> list[EpochRecord]:
    """Keep only rows of completed epochs; rewrite atomically. Returns the kept rows."""
    if not run.history_path.is_file():
        return []
    rows = [r for r in read_jsonl(run.history_path) if r["epoch"] < completed_epochs]
    atomic_write_text(run.history_path, "".join(dumps_json(r, indent=None) + "\n" for r in rows))
    return [EpochRecord(**r) for r in rows]


def _check_resume_compatible(ckpt: Mapping[str, Any], *, model: BatchNormMLPLanguageModel,
                             training: BatchNormMLPTrainingConfig, seed: int, num_examples: int) -> None:
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
        ("batchnorm constants", config.get("batchnorm"), BATCHNORM_RECORD),
    ]
    for name, stored, current in checks:
        if stored != current:
            raise IncompatibleCheckpointError(
                f"cannot resume: {name} changed (checkpoint {stored!r}, now {current!r})"
            )


def train_batchnorm_mlp(
    model: BatchNormMLPLanguageModel,
    train: ContextWindows,
    dev: ContextWindows,
    *,
    training: BatchNormMLPTrainingConfig,
    seed: int,
    run: RunManager | None = None,
    resume: bool = False,
    diagnostics: DiagnosticsSchedule | None = None,
    config: Mapping[str, Any] | None = None,
) -> BatchNormMLPTrainingResult:
    """Reference training procedure (§4.6), with recalibration, checkpoints, history and resume (§14).

    Per epoch ``e`` (0-based): set ``η_e``; train on ``EpochSampler`` batches in train
    mode with mean reduction over the actual batch size; **recalibrate** the
    inference statistics over all training windows; evaluate full train and full
    dev NLL in eval mode (plus the epoch-end diagnostics); then, with ``run``:

    1. save ``model_best.pt`` if dev NLL strictly improved (ties keep the earlier epoch);
    2. append the :class:`EpochRecord` to ``history.jsonl``;
    3. save ``checkpoint_last.pt`` with ``epoch = e + 1`` (the commit point);
    4. after the last epoch, save ``model_final.pt``.

    Each step runs: forward → loss → finite-loss check → backward →
    finite-gradient check → diagnostics → ``optimizer.step()``. A non-finite loss
    or gradient raises ``FloatingPointError`` before the step. Train mode is used
    only for these SGD forwards. Configurations that would produce a training batch
    of fewer than 2 windows are rejected before any step.

    Arguments are those of ``train_mlp`` (see ``models/feedforward/mlp.py``); ``resume``
    continues from ``checkpoint_last.pt`` under the **original** ``max_epochs`` and schedule.
    """
    if not isinstance(model, BatchNormMLPLanguageModel):
        raise TypeError(f"model must be a BatchNormMLPLanguageModel, got {type(model).__name__}")
    if not isinstance(training, BatchNormMLPTrainingConfig):
        raise TypeError(f"training must be a BatchNormMLPTrainingConfig, got {type(training).__name__}")
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
    _check_training_batches(sampler, train.num_examples, training.batch_size)
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
        _check_training_batches(sampler, train.num_examples, training.batch_size)
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
        model.train()  # train mode only for the SGD forwards below
        running = NLLAccumulator()
        for idx in sampler.batches(epoch):
            idx = idx.to(device)
            xb, yb = train_x[idx], train_y[idx]
            collect = {k: diagnostics is not None and diagnostics.should_collect(step, k)
                       for k in ("gradients", "parameters", "activations")}
            optimizer.zero_grad(set_to_none=True)
            monitor = _monitor(model) if collect["activations"] else None
            # 1. forward (train mode: batch statistics; the activation monitor, if any, observes this pass)
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

        # Recalibrate with the fixed end-of-epoch parameters, before any evaluation (§4.6, §4.7).
        try:
            calibrate_batchnorm(model, train, batch_size=training.eval_batch_size)
        except FloatingPointError as exc:
            raise FloatingPointError(f"recalibration failed at epoch {epoch}: {exc}") from exc
        train_nll = evaluate_nll(model, train, batch_size=training.eval_batch_size).mean_nll
        with _monitor(model) as dev_monitor:
            dev_nll = evaluate_nll(model, dev, batch_size=training.eval_batch_size).mean_nll
        activations = _epoch_activation_summary(model, dev, training.eval_batch_size, dev_monitor.summaries())
        is_best = best_dev is None or dev_nll < best_dev  # strict: exact ties keep the earliest epoch
        if is_best:
            best_dev, best_epoch = dev_nll, epoch
        rec = EpochRecord(epoch=epoch, lr=lr, steps=step, running_train_nll=running.mean_nll,
                          train_nll=train_nll, dev_nll=dev_nll, is_best=is_best,
                          seconds=time.perf_counter() - t0, activations=activations,
                          batchnorm=_batchnorm_summary(model))
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
    return BatchNormMLPTrainingResult(best_epoch=best_epoch, best_dev_nll=best_dev,
                                      epochs_completed=training.max_epochs, history=history)

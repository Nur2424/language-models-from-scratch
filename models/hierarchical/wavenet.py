"""WaveNet: hierarchical context-merging character language model (Model 5).

Specification: ``models/hierarchical/wavenet_spec.md`` (v0.4). Section numbers (§)
refer to it. "WaveNet" names the receptive-field structure only: a binary tree of
kernel-2 merges over an 8-character window. There are no gates, residual or skip
connections, and no convolution modules (§1.2).

Two variants share one implementation (§7.1)
--------------------------------------------
* ``normalization="none"``      → model name ``wavenet``            (baseline, 21,989 parameters)
* ``normalization="batchnorm"`` → model name ``wavenet_batchnorm``  (experiment, 22,397 parameters)

``model.model_name`` is derived from ``config.normalization`` on every access; it is
the single authoritative identity (§7.1, §10.1).

Architecture (§2–§4), baseline values k = 8, d = 10, H = 68, L = 3::

    contexts [B, 8] → emb → [B, 8, 10]
    level ℓ = 1..L:  merge_pairs → hidden_ℓ (Linear, no bias, shared over groups)
                     → [BN] bn_ℓ (statistics pooled over B·G_ℓ rows) → act_ℓ (tanh)
                     [B, 4, 20] → [B, 4, 68] → [B, 2, 136] → [B, 2, 68] → [B, 1, 136] → [B, 1, 68]
    reshape [B, 1, 68] → [B, 68] (unconditional) → out (Linear with bias) → [B, 27] logits

BatchNorm variant (§3.6, §8.1)
------------------------------
Train mode: pooled batch statistics (biased variance), PyTorch EMA updated (diagnostic
only). Eval mode: the authoritative ``inference_mean`` / ``inference_var`` per level,
recomputed after every epoch by :func:`calibrate_batchnorm`, sequentially by level,
streaming in float64 on CPU.

Initialisation (§6): ``E ~ N(0, 1)``; ``W_ℓ ~ N(0, 1/C_ℓ)`` (LeCun, no gain);
``W_o = b_o = 0``; [BN] ``γ = 1``, ``β = 0``, EMA buffers ``0 / 1``, inference buffers
``0 / 1``, counters ``0``. All set explicitly from the CPU ``generator`` only.
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

MERGE_FACTOR = 2  # fixed (§7.1)
BN_EPS = 1e-5
BN_MOMENTUM = 0.1  # drives the diagnostic EMA only
#: Recorded in checkpoints and config.json for the BatchNorm variant (``None`` for the baseline).
BATCHNORM_RECORD = {"eps": BN_EPS, "momentum": BN_MOMENTUM, "inference_statistics": "full_train_biased_sequential"}
#: normalization → model name. The only place identities are defined.
MODEL_NAMES = {"none": "wavenet", "batchnorm": "wavenet_batchnorm"}
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
class WaveNetConfig:
    """Architecture configuration (``model.params``; §7.1).

    ``context_length`` must be a power of two >= 2; the number of levels is
    ``L = log2(context_length)``. The merge factor (2), ε, momentum and the
    initialisation are fixed in code.
    """

    context_length: int = 8
    embedding_dim: int = 10
    hidden_dim: int = 68
    normalization: str = "none"

    def __post_init__(self) -> None:
        _check_int(self.context_length, "context_length", minimum=2)
        if self.context_length & (self.context_length - 1):
            raise ValueError(f"context_length must be a power of two >= 2, got {self.context_length}")
        _check_int(self.embedding_dim, "embedding_dim")
        _check_int(self.hidden_dim, "hidden_dim")
        if not isinstance(self.normalization, str) or self.normalization not in MODEL_NAMES:
            raise ValueError(f"normalization must be one of {sorted(MODEL_NAMES)}, got {self.normalization!r}")

    @property
    def num_levels(self) -> int:
        return self.context_length.bit_length() - 1

    def level_input_dim(self, level: int) -> int:
        """``C_ℓ``: input width of ``hidden_ℓ`` (``2d`` at level 1, ``2H`` above)."""
        return MERGE_FACTOR * (self.embedding_dim if level == 1 else self.hidden_dim)

    def level_groups(self, level: int) -> int:
        """``G_ℓ = k / 2^ℓ``."""
        return self.context_length // MERGE_FACTOR**level

    @property
    def model_name(self) -> str:
        return MODEL_NAMES[self.normalization]


@dataclass(frozen=True)
class WaveNetTrainingConfig:
    """Settings of the reference training procedure (§7.2). Model-local; not part of ``ExperimentConfig``.

    ``lr`` and ``max_epochs`` are PENDING PILOT. ``batch_size >= 1`` here; the
    BatchNorm variant additionally needs >= 2 (checked by :func:`train_wavenet`).
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
class WaveNetSample:
    """One generated sample (same semantics as Models 1–4)."""

    text: str
    finished: bool


@dataclass(frozen=True)
class EpochRecord:
    """One row of ``history.jsonl`` (§8, §11)."""

    epoch: int
    lr: float
    steps: int
    running_train_nll: float
    train_nll: float
    dev_nll: float
    is_best: bool
    seconds: float
    activations: dict
    output: dict
    levels: dict
    batchnorm: dict | None  # None for the baseline


@dataclass(frozen=True)
class WaveNetTrainingResult:
    best_epoch: int
    best_dev_nll: float
    epochs_completed: int
    history: list[EpochRecord]


# ============================================================== pair merging
def merge_pairs(x: torch.Tensor) -> torch.Tensor:
    """Merge adjacent positions (§4.1): ``[B, T, C] → [B, T // 2, 2C]``.

    ``out[b, j] = cat(x[b, 2j], x[b, 2j + 1])``: the earlier position fills the first
    ``C`` features. Uses ``reshape`` (contiguity-safe), never squeezes, preserves
    dtype and device, and is differentiable. ``T`` must be even and >= 2.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"merge_pairs expects a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 3:
        raise ValueError(f"merge_pairs expects a [B, T, C] tensor, got shape {tuple(x.shape)}")
    batch, length, channels = x.shape
    if length < MERGE_FACTOR or length % MERGE_FACTOR:
        raise ValueError(f"merge_pairs needs an even sequence length >= 2, got T={length}")
    if channels < 1:
        raise ValueError(f"merge_pairs needs C >= 1, got C={channels}")
    return x.reshape(batch, length // MERGE_FACTOR, MERGE_FACTOR * channels)


# ============================================================ normalisation
class LevelBatchNorm1d(nn.BatchNorm1d):
    """Per-level BatchNorm for ``[B, G, H]`` tensors, pooled over all ``B·G`` rows (§3.6).

    Parameters ``weight`` (γ) and ``bias`` (β). EMA buffers (PyTorch, diagnostic only):
    ``running_mean``, ``running_var``, ``num_batches_tracked``. Authoritative buffers:
    ``inference_mean``, ``inference_var``, ``num_calibrations``.

    Train mode: standard PyTorch batch normalisation (biased batch variance; EMA update).
    Eval mode: :meth:`normalize_eval`, which never reads or writes the EMA.
    """

    def __init__(self, num_features: int, *, device: Any = None, dtype: torch.dtype | None = None) -> None:
        super().__init__(num_features, eps=BN_EPS, momentum=BN_MOMENTUM, affine=True, track_running_stats=True,
                         device=device, dtype=dtype)
        self.register_buffer("inference_mean", torch.empty(num_features, device=device, dtype=dtype))
        self.register_buffer("inference_var", torch.empty(num_features, device=device, dtype=dtype))
        self.register_buffer("num_calibrations", torch.empty((), device=device, dtype=torch.long))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.ndim != 3:
            raise ValueError(f"LevelBatchNorm1d expects [B, G, H], got {tuple(a.shape)}")
        if not self.training:
            return self.normalize_eval(a)
        batch, groups, width = a.shape
        return super().forward(a.reshape(batch * groups, width)).reshape(batch, groups, width)

    def normalize_eval(self, a: torch.Tensor) -> torch.Tensor:
        """Eval-mode normalisation with the inference statistics, regardless of ``self.training``."""
        batch, groups, width = a.shape
        y = F.batch_norm(a.reshape(batch * groups, width), self.inference_mean, self.inference_var,
                         self.weight, self.bias, training=False, momentum=0.0, eps=self.eps)
        return y.reshape(batch, groups, width)

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
class WaveNetLanguageModel(nn.Module):
    """Hierarchical context-merging language model (§2–§9), in either variant.

    Parameters
    ----------
    vocab:
        Shared vocabulary; ``V = vocab.num_classes``. PAD never reaches the model.
    config:
        :class:`WaveNetConfig`; ``config.normalization`` selects the variant.
    generator:
        CPU ``torch.Generator`` for every random initialisation draw
        (by convention ``make_generator(derive_seed(seed, "init"))``). Required.

    Submodules (flat names, so diagnostics group per level): ``emb``,
    ``hidden1..L`` (no bias), ``bn1..L`` (BatchNorm variant only), ``act1..L``, ``out``.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        config: WaveNetConfig = WaveNetConfig(),  # noqa: B008  (frozen, immutable dataclass)
        *,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"vocab must be a utils.vocabulary.Vocabulary, got {type(vocab).__name__}")
        if not isinstance(config, WaveNetConfig):
            raise TypeError(f"config must be a WaveNetConfig, got {type(config).__name__}")
        if not isinstance(generator, torch.Generator):
            raise TypeError(f"generator must be a torch.Generator, got {type(generator).__name__}")
        if generator.device.type != "cpu":
            raise ValueError(f"generator must be a CPU generator, got {generator.device}")
        self.vocab = vocab
        self.config = config
        V, d, H = vocab.num_classes, config.embedding_dim, config.hidden_dim
        # skip_init: no PyTorch default initialisation, no global-RNG use; everything is set below (§6).
        self.emb = nn.utils.skip_init(nn.Embedding, V, d)
        for level in self.levels:
            setattr(self, f"hidden{level}",
                    nn.utils.skip_init(nn.Linear, config.level_input_dim(level), H, bias=False))
            if self.uses_batchnorm:
                setattr(self, f"bn{level}", nn.utils.skip_init(LevelBatchNorm1d, H))
            setattr(self, f"act{level}", nn.Tanh())
        self.out = nn.utils.skip_init(nn.Linear, H, V)
        with torch.no_grad():
            self.emb.weight.normal_(0.0, 1.0, generator=generator)  # draw order: E, W1, W2, ...
            for level in self.levels:
                fan_in = config.level_input_dim(level)
                self._hidden(level).weight.normal_(0.0, 1.0 / math.sqrt(fan_in), generator=generator)
            if self.uses_batchnorm:
                for level in self.levels:
                    bn = self._bn(level)
                    bn.weight.fill_(1.0)
                    bn.bias.zero_()
                    bn.running_mean.zero_()
                    bn.running_var.fill_(1.0)
                    bn.num_batches_tracked.zero_()
                    bn.inference_mean.zero_()
                    bn.inference_var.fill_(1.0)
                    bn.num_calibrations.zero_()
            self.out.weight.zero_()  # zero output layer: uniform initial prediction
            self.out.bias.zero_()

    # ---------------------------------------------------------- identity
    @property
    def model_name(self) -> str:
        """``"wavenet"`` or ``"wavenet_batchnorm"``, derived from ``config.normalization`` (read-only)."""
        return MODEL_NAMES[self.config.normalization]

    @property
    def uses_batchnorm(self) -> bool:
        return self.config.normalization == "batchnorm"

    @property
    def levels(self) -> range:
        return range(1, self.config.num_levels + 1)

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
        return (f"name={self.model_name}, V={self.num_classes}, k={c.context_length}, L={c.num_levels}, "
                f"d={c.embedding_dim}, H={c.hidden_dim}")

    def _hidden(self, level: int) -> nn.Linear:
        return getattr(self, f"hidden{level}")

    def _bn(self, level: int) -> LevelBatchNorm1d:
        return getattr(self, f"bn{level}")

    def _act(self, level: int) -> nn.Tanh:
        return getattr(self, f"act{level}")

    # ----------------------------------------------------------- forward
    def _check(self, contexts: torch.Tensor) -> None:
        _check_contexts(contexts, context_length=self.context_length, num_classes=self.num_classes,
                        pad_id=self.vocab.pad_id, device=self.device)

    def _hierarchy(self, contexts: torch.Tensor, *, stop_before_norm: int | None = None,
                   eval_normalization: bool = False) -> torch.Tensor:
        """Shared level loop (no validation). Returns the level-L activations ``[B, 1, H]``, or,
        with ``stop_before_norm=ℓ``, the pre-normalisation activations ``a^ℓ`` of level ℓ.

        ``eval_normalization=True`` normalises every level with its inference statistics
        regardless of the module mode (used only by recalibration; never a train-mode BN call).
        """
        batch, H = contexts.shape[0], self.config.hidden_dim
        h = self.emb(contexts)  # [B, k, d]
        for level in self.levels:
            a = self._hidden(level)(merge_pairs(h))  # [B, G_ℓ, H]
            if level == stop_before_norm:
                return a
            if self.uses_batchnorm:
                bn = self._bn(level)
                a = bn.normalize_eval(a) if eval_normalization else bn(a)
            h = self._act(level)(a)
            expected = (batch, self.config.level_groups(level), H)
            if tuple(h.shape) != expected:  # internal invariant (§4.2)
                raise RuntimeError(f"level {level} produced shape {tuple(h.shape)}, expected {expected}")
        return h

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """Next-token logits ``[B, V]`` for context windows ``[B, k]`` (§4.2).

        Eval mode accepts any ``B >= 0``. The BatchNorm variant in train mode requires
        ``B >= 2`` (checked before any BatchNorm call).
        """
        self._check(contexts)
        if self.training and self.uses_batchnorm and contexts.shape[0] < 2:
            raise ValueError(
                f"BatchNorm needs at least 2 windows per training batch, got {contexts.shape[0]}"
            )
        h = self._hierarchy(contexts)  # [B, 1, H]
        return self.out(h.reshape(contexts.shape[0], self.config.hidden_dim))  # unconditional [B, 1, H] → [B, H]

    def _level_preactivations(self, contexts: torch.Tensor, level: int) -> torch.Tensor:
        """Private (§8.1, Q10): pre-normalisation ``a^ℓ`` ``[B, G_ℓ, H]`` with levels < ℓ in
        eval-mode normalisation. Used only by :func:`calibrate_batchnorm`."""
        if isinstance(level, bool) or not isinstance(level, int) or level not in self.levels:
            raise ValueError(f"level must be in 1..{self.config.num_levels}, got {level!r}")
        self._check(contexts)
        return self._hierarchy(contexts, stop_before_norm=level, eval_normalization=True)

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
    ) -> list[WaveNetSample]:
        """Ancestral sampling from ``[BOS] * k`` in eval mode (§9); the previous mode is restored."""
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
                drawn = sample_next_token(self(contexts[rows]), temperature=temperature, top_k=top_k,
                                          top_p=top_p, generator=generator).tolist()
                still_active = []
                for k, token in zip(active, drawn):
                    if token == eos:
                        finished[k] = True
                    elif len(chars[k]) == max_length:
                        pass  # limit reached and EOS not drawn: truncated
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
        return [WaveNetSample(text=self.vocab.decode(ids, strip_bos=False, stop_at_eos=False), finished=done)
                for ids, done in zip(chars, finished)]

    # ---------------------------------------------------------- checkpoint
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_vocab: Vocabulary | None = None,
        expected_model_name: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> WaveNetLanguageModel:
        """Load weights and buffers (§10, §10.1) with ``weights_only=True``.

        The variant is constructed from the stored ``config["model"]``; identity is then
        validated (stored self-consistency, receiving model, ``expected_model_name``).
        """
        if expected_model_name is not None and expected_model_name not in MODEL_NAMES.values():
            raise ValueError(f"expected_model_name must be one of {sorted(MODEL_NAMES.values())}, "
                             f"got {expected_model_name!r}")
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
            raise IncompatibleCheckpointError(f"{path} has no config['model'] (the WaveNetConfig used)")
        try:
            model_config = WaveNetConfig(**model_cfg)
        except (TypeError, ValueError) as exc:
            raise IncompatibleCheckpointError(f"{path}: invalid model config {dict(model_cfg)!r}: {exc}") from exc
        _check_stored_identity(config, model_config, source=str(path))  # §10.1 check 1
        model = cls(vocab, model_config, generator=make_generator(0))  # weights are overwritten below
        if model.model_name != config["model_name"]:  # §10.1 check 2
            raise IncompatibleCheckpointError(
                f"{path}: stored model_name {config['model_name']!r} but the constructed model is "
                f"{model.model_name!r}"
            )
        if expected_model_name is not None and expected_model_name != config["model_name"]:  # §10.1 check 3
            raise IncompatibleCheckpointError(
                f"{path} holds a {config['model_name']!r} model, expected {expected_model_name!r}"
            )
        try:
            restore_model(model, ckpt, strict=True)
        except RuntimeError as exc:
            raise IncompatibleCheckpointError(f"{path}: weights do not match the model config: {exc}") from exc
        if model.uses_batchnorm:
            counts = [int(model._bn(level).num_calibrations) for level in model.levels]
            if min(counts) == 0 or len(set(counts)) != 1:
                raise IncompatibleCheckpointError(
                    f"{path}: BatchNorm inference statistics not calibrated consistently (per-level counts {counts})"
                )
        return model.to(map_location)


# =================================================================== helpers
def _check_contexts(
    contexts: torch.Tensor, *, context_length: int, num_classes: int, pad_id: int, device: torch.device
) -> None:
    """Validate a context batch (§12). Metadata checks always run; the range check is empty-safe."""
    if not isinstance(contexts, torch.Tensor):
        raise TypeError(f"contexts must be a torch.Tensor, got {type(contexts).__name__}")
    if contexts.dtype != torch.int64:
        raise TypeError(f"contexts must be int64 token ids, got {contexts.dtype}")
    if contexts.ndim != 2 or contexts.shape[1] != context_length:
        raise ValueError(f"contexts must have shape [B, {context_length}], got {tuple(contexts.shape)}")
    if contexts.device != device:
        raise ValueError(f"contexts are on {contexts.device} but the model is on {device}; move them explicitly")
    out_of_range = (contexts < 0) | (contexts >= num_classes)
    if bool(out_of_range.any()):
        if bool((contexts == pad_id).any()):
            raise ValueError(f"contexts contain the PAD id {pad_id}: padding must not reach the model")
        bad = sorted(set(contexts[out_of_range].tolist()))[:5]
        raise ValueError(f"contexts contain ids {bad} outside [0, {num_classes})")


def _check_stored_identity(config: Mapping[str, Any], model_config: WaveNetConfig, *, source: str) -> None:
    """§10.1 check 1: the stored ``model_name`` and ``batchnorm`` record agree with the stored normalisation."""
    expected_name = MODEL_NAMES[model_config.normalization]
    if config.get("model_name") != expected_name:
        raise IncompatibleCheckpointError(
            f"{source}: stored model_name {config.get('model_name')!r} disagrees with stored normalization "
            f"{model_config.normalization!r} (implies {expected_name!r})"
        )
    expected_record = BATCHNORM_RECORD if model_config.normalization == "batchnorm" else None
    if config.get("batchnorm") != expected_record:
        raise IncompatibleCheckpointError(
            f"{source}: stored batchnorm record {config.get('batchnorm')!r} disagrees with normalization "
            f"{model_config.normalization!r} (expected {expected_record!r})"
        )


def _check_windows(model: WaveNetLanguageModel, windows: ContextWindows, batch_size: int) -> None:
    if not isinstance(windows, ContextWindows):
        raise TypeError(f"windows must be a ContextWindows, got {type(windows).__name__}")
    if windows.context_length != model.context_length:
        raise ValueError(f"windows have context_length {windows.context_length}, "
                         f"the model expects {model.context_length}")
    _check_int(batch_size, "batch_size")


def _learning_rate_at(epoch: int, training: WaveNetTrainingConfig) -> float:
    """Epoch-based, stateless learning rate: recomputed identically after resume."""
    if training.lr_decay_epoch is None or epoch < training.lr_decay_epoch:
        return training.lr
    return training.lr * training.lr_decay_factor


def run_config(experiment: Mapping[str, Any] | None, model_config: WaveNetConfig,
               training: WaveNetTrainingConfig) -> dict[str, Any]:
    """The resolved configuration stored in every checkpoint and ``config.json`` (§7.3)."""
    return {
        "experiment": dict(experiment) if experiment is not None else None,
        "model_name": model_config.model_name,
        "model": asdict(model_config),
        "training": asdict(training),
        "batchnorm": dict(BATCHNORM_RECORD) if model_config.normalization == "batchnorm" else None,
    }


# ============================================================= recalibration
def _chunk_moments(rows: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Per-unit ``(n, mean, M2)`` of ``[n, H]`` rows: centred two-pass, float64 on CPU.

    Transfer first, then cast: MPS has no float64 (Model 4 lesson).
    """
    r64 = rows.detach().to(device="cpu").to(dtype=torch.float64)
    mean = r64.mean(dim=0)
    return r64.shape[0], mean, (r64 - mean).square().sum(dim=0)


def _merge_moments(
    n: int, mean: torch.Tensor, m2: torch.Tensor, n_c: int, mean_c: torch.Tensor, m2_c: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Chan et al. parallel update (exact for ``n = 0``)."""
    total = n + n_c
    delta = mean_c - mean
    return total, mean + delta * (n_c / total), m2 + m2_c + delta.square() * (n * n_c / total)


def calibrate_batchnorm(model: WaveNetLanguageModel, windows: ContextWindows, *, batch_size: int) -> dict[str, Any]:
    """Recompute and install the inference statistics of every level (§3.6, §8.1).

    Levels are calibrated **sequentially**: level ℓ's pre-activations are computed with
    levels < ℓ already normalised by their *newly installed* statistics, so each level's
    statistics describe the network exactly as it is evaluated. Per level: exact mean and
    biased variance (divisor ``M·G_ℓ``), streamed chunk by chunk (each chunk released before
    the next) with float64 Chan merges on CPU. Never reads or writes the EMA buffers,
    parameters or ``model.training``; independent of previous buffer values.
    """
    if not isinstance(model, WaveNetLanguageModel) or not model.uses_batchnorm:
        raise ValueError("calibrate_batchnorm requires a WaveNetLanguageModel with normalization='batchnorm'")
    _check_windows(model, windows, batch_size)
    total = windows.num_examples
    if total == 0:
        raise ValueError("calibrate_batchnorm needs at least one window")
    H = model.config.hidden_dim
    with torch.no_grad():
        for level in model.levels:
            n = 0
            mean = torch.zeros(H, dtype=torch.float64)
            m2 = torch.zeros(H, dtype=torch.float64)
            for start in range(0, total, batch_size):
                chunk = model._level_preactivations(windows.contexts[start:start + batch_size].to(model.device),
                                                    level=level)
                n_c, mean_c, m2_c = _chunk_moments(chunk.reshape(-1, H))
                del chunk  # release before the next chunk
                n, mean, m2 = _merge_moments(n, mean, m2, n_c, mean_c, m2_c)
            expected_rows = total * model.config.level_groups(level)
            if n != expected_rows:
                raise FloatingPointError(f"level {level}: recalibration covered {n} of {expected_rows} rows")
            var = m2 / n
            if not (bool(torch.isfinite(mean).all()) and bool(torch.isfinite(var).all())):
                raise FloatingPointError(f"level {level}: non-finite BatchNorm statistics during recalibration")
            bn = model._bn(level)
            dtype = bn.inference_mean.dtype
            bn.set_inference_statistics(mean.to(dtype), var.to(dtype))  # installed before the next level
    return {"num_windows": total,
            "num_calibrations": [int(model._bn(level).num_calibrations) for level in model.levels]}


# ================================================================ evaluation
def evaluate_nll(model: WaveNetLanguageModel, windows: ContextWindows, *, batch_size: int) -> NLLAccumulator:
    """Example-weighted NLL over every window, in eval mode; original mode restored in ``finally``."""
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
    model: WaveNetLanguageModel,
    windows: ContextWindows,
    *,
    threshold: float = SATURATION_THRESHOLD,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Per level: ``[H]`` float64 CPU fractions of (window, group) rows with ``|h| > threshold``.

    Eval mode, ``no_grad``, running counts only; original mode restored in ``finally``.
    """
    _check_windows(model, windows, batch_size)
    threshold = _check_real(threshold, "threshold")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
    if windows.num_examples == 0:
        raise ValueError("per_unit_saturation needs at least one window")
    H = model.config.hidden_dim
    counts = {level: torch.zeros(H, dtype=torch.int64) for level in model.levels}

    def make_hook(level: int):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            counts[level].add_((output.abs() > threshold).reshape(-1, H).sum(dim=0).cpu())
        return hook

    was_training = model.training
    handles = [model._act(level).register_forward_hook(make_hook(level)) for level in model.levels]
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, windows.num_examples, batch_size):
                model(windows.contexts[start:start + batch_size].to(model.device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {level: counts[level].double() / (windows.num_examples * model.config.level_groups(level))
            for level in model.levels}


def _output_summary(model: WaveNetLanguageModel, windows: ContextWindows, *, batch_size: int) -> dict[str, float]:
    """Output-layer statistics on ``windows`` in eval mode (§11): logit mean/std, mean predictive
    entropy, mean max-probability, ‖W_o‖, ‖b_o‖. Float64 running sums; mode restored in ``finally``."""
    _check_windows(model, windows, batch_size)
    if windows.num_examples == 0:
        raise ValueError("_output_summary needs at least one window")
    s1 = s2 = ent = pmax = 0.0
    numel = 0
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, windows.num_examples, batch_size):
                z = model(windows.contexts[start:start + batch_size].to(model.device))
                z64 = z.detach().to(device="cpu").to(dtype=torch.float64)
                logp = torch.log_softmax(z64, dim=1)
                s1 += float(z64.sum())
                s2 += float(z64.square().sum())
                numel += z64.numel()
                ent += float(-(logp.exp() * logp).sum())
                pmax += float(logp.exp().max(dim=1).values.sum())
    finally:
        model.train(was_training)
    mean = s1 / numel
    return {
        "logit_mean": mean,
        "logit_std": math.sqrt(max(s2 / numel - mean * mean, 0.0)),
        "mean_entropy": ent / windows.num_examples,
        "mean_max_prob": pmax / windows.num_examples,
        "weight_norm": float(model.out.weight.detach().to(device="cpu").to(dtype=torch.float64).norm()),
        "bias_norm": float(model.out.bias.detach().to(device="cpu").to(dtype=torch.float64).norm()),
    }


def _summary(t: torch.Tensor) -> dict[str, float]:
    t = t.detach().to(device="cpu").to(dtype=torch.float64)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "min": float(t.min()),
            "max": float(t.max())}


def _levels_summary(model: WaveNetLanguageModel) -> dict[str, Any]:
    """Row-norm summary of each ``W_ℓ`` (§11)."""
    out = {}
    for level in model.levels:
        rows = model._hidden(level).weight.detach().to(device="cpu").to(dtype=torch.float64).norm(dim=1)
        out[f"level{level}"] = {"w_row_norm": {k: v for k, v in _summary(rows).items() if k != "std"}}
    return out


def _batchnorm_summary(model: WaveNetLanguageModel) -> dict[str, Any] | None:
    """Per-level BatchNorm summary (§11); ``None`` for the baseline.

    ``log_var_ratio`` compares the effective variances ``v + ε`` (the denominators used),
    which keeps it finite if a variance is exactly zero.
    """
    if not model.uses_batchnorm:
        return None
    out = {}
    with torch.no_grad():
        for level in model.levels:
            bn = model._bn(level)
            im, iv, rm, rv = (getattr(bn, name).detach().to(device="cpu").to(dtype=torch.float64)
                              for name in ("inference_mean", "inference_var", "running_mean", "running_var"))
            z = ((rm - im) / torch.sqrt(iv + bn.eps)).abs()
            r = torch.log((rv + bn.eps) / (iv + bn.eps)).abs()
            out[f"level{level}"] = {
                "ema_vs_inference": {"mean_abs_z": float(z.mean()), "max_abs_z": float(z.max()),
                                     "median_abs_log_var_ratio": float(r.median()),
                                     "max_abs_log_var_ratio": float(r.max())},
                "gamma": _summary(bn.weight),
                "beta": _summary(bn.bias),
                "num_calibrations": int(bn.num_calibrations),
                "num_batches_tracked": int(bn.num_batches_tracked),
            }
    return out


def _monitored_modules(model: WaveNetLanguageModel) -> list[str]:
    names = [f"hidden{level}" for level in model.levels]
    if model.uses_batchnorm:
        names += [f"bn{level}" for level in model.levels]
    return names + [f"act{level}" for level in model.levels]


def _monitor(model: WaveNetLanguageModel) -> ActivationMonitor:
    return ActivationMonitor(model, module_names=_monitored_modules(model),
                             saturation={f"act{level}": tanh_saturation(SATURATION_THRESHOLD)
                                         for level in model.levels})


def _epoch_activation_summary(model: WaveNetLanguageModel, dev: ContextWindows, batch_size: int,
                              monitor_summaries: dict[str, Any]) -> dict[str, Any]:
    unit = per_unit_saturation(model, dev, threshold=SATURATION_THRESHOLD, batch_size=batch_size)
    per_unit = {f"level{level}": {"threshold": SATURATION_THRESHOLD, "min": float(u.min()),
                                  "median": float(u.median()), "max": float(u.max()),
                                  "units_ge_0.99": int((u >= 0.99).sum())}
                for level, u in unit.items()}
    return {**monitor_summaries, "per_unit_saturation": per_unit}


# ================================================================== training
def _check_finite_gradients(model: WaveNetLanguageModel, epoch: int, step: int) -> None:
    """One fused check (one host sync); details only after a failure."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
        return
    bad = nonfinite_gradients(model)
    groups = sorted({module_group(name) for name in bad})
    raise FloatingPointError(
        f"non-finite gradients at epoch {epoch}, step {step} in parameter groups {groups}: {bad}"
    )


def _check_training_batches(model: WaveNetLanguageModel, sampler: EpochSampler, num_examples: int,
                            batch_size: int) -> None:
    """BatchNorm variant: every training batch must hold >= 2 windows (§8)."""
    if model.uses_batchnorm and min(sampler.batch_sizes()) < 2:
        raise ValueError(
            f"{num_examples} training windows with batch_size {batch_size} give a batch of fewer than 2 windows; "
            "the BatchNorm variant needs at least 2 windows per training batch"
        )


def _truncate_history(run: RunManager, completed_epochs: int) -> list[EpochRecord]:
    if not run.history_path.is_file():
        return []
    rows = [r for r in read_jsonl(run.history_path) if r["epoch"] < completed_epochs]
    atomic_write_text(run.history_path, "".join(dumps_json(r, indent=None) + "\n" for r in rows))
    return [EpochRecord(**r) for r in rows]


def _check_resume_compatible(ckpt: Mapping[str, Any], *, model: WaveNetLanguageModel,
                             training: WaveNetTrainingConfig, seed: int, num_examples: int) -> None:
    """Reject any change that would make the resumed run differ from the original (§10, §10.1)."""
    config = ckpt.get("config") or {}
    extra = ckpt.get("extra") or {}
    sampler = extra.get("sampler") or {}
    stored_training = config.get("training") or {}
    checks: list[tuple[str, Any, Any]] = [
        ("model_name", config.get("model_name"), model.model_name),
        ("model config", config.get("model"), asdict(model.config)),
        *((f"training.{f}", stored_training.get(f), getattr(training, f))
          for f in ("lr", "max_epochs", "batch_size", "lr_decay_epoch", "lr_decay_factor")),
        ("sampler batch size", sampler.get("batch_size"), training.batch_size),
        ("sampler seed", sampler.get("seed"), derive_seed(seed, "data")),
        ("base seed", extra.get("base_seed"), seed),
        ("vocabulary", ckpt.get("vocab"), model.vocab.to_dict()),
        ("dataset size", sampler.get("num_examples"), num_examples),
        ("batchnorm record", config.get("batchnorm"), run_config(None, model.config, training)["batchnorm"]),
    ]
    for name, stored, current in checks:
        if stored != current:
            raise IncompatibleCheckpointError(
                f"cannot resume: {name} changed (checkpoint {stored!r}, now {current!r})"
            )


def train_wavenet(
    model: WaveNetLanguageModel,
    train: ContextWindows,
    dev: ContextWindows,
    *,
    training: WaveNetTrainingConfig,
    seed: int,
    run: RunManager | None = None,
    resume: bool = False,
    diagnostics: DiagnosticsSchedule | None = None,
    config: Mapping[str, Any] | None = None,
) -> WaveNetTrainingResult:
    """Reference training procedure (§8), with checkpoints, history and resume (§10).

    Per epoch: set ``η_e``; SGD steps in train mode on ``EpochSampler`` batches (mean
    reduction over the actual batch size); [BN] sequential recalibration; full-train and
    full-dev NLL in eval mode plus epoch-end diagnostics; then ``model_best.pt`` (strict
    improvement) → history row → ``checkpoint_last.pt``; after the last epoch ``model_final.pt``.

    Step order: forward → loss → finite-loss check → backward → finite-gradient check →
    diagnostics → ``optimizer.step()``. ``run`` must belong to ``model.model_name``; an
    experiment ``config`` naming a model must name ``model.model_name``.
    """
    if not isinstance(model, WaveNetLanguageModel):
        raise TypeError(f"model must be a WaveNetLanguageModel, got {type(model).__name__}")
    if not isinstance(training, WaveNetTrainingConfig):
        raise TypeError(f"training must be a WaveNetTrainingConfig, got {type(training).__name__}")
    for name, w in (("train", train), ("dev", dev)):
        _check_windows(model, w, training.eval_batch_size)
        if w.num_examples == 0:
            raise ValueError(f"{name} has no windows")
    if resume and run is None:
        raise ValueError("resume=True requires a run")
    if diagnostics is not None and diagnostics.enabled and run is None:
        raise ValueError("step diagnostics are written to the run directory; pass a run")
    if run is not None and run.model_name != model.model_name:
        raise ValueError(f"run {run.run_id!r} belongs to {run.model_name!r}, but the model is {model.model_name!r}")
    experiment_model = (config or {}).get("model") if isinstance(config, Mapping) else None
    if isinstance(experiment_model, Mapping) and "name" in experiment_model \
            and experiment_model["name"] != model.model_name:
        raise ValueError(f"experiment config names model {experiment_model['name']!r}, "
                         f"but the model is {model.model_name!r}")
    if model.uses_batchnorm and training.batch_size < 2:
        raise ValueError("the BatchNorm variant needs batch_size >= 2")

    device = model.device
    train_x, train_y = train.contexts.to(device), train.targets.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=training.lr, momentum=0.0, weight_decay=0.0)
    sampler = EpochSampler(train.num_examples, training.batch_size, seed=derive_seed(seed, "data"))
    _check_training_batches(model, sampler, train.num_examples, training.batch_size)
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
        _check_training_batches(model, sampler, train.num_examples, training.batch_size)
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
            # 1. forward
            if monitor is not None:
                with monitor:
                    logits = model(xb)
            else:
                logits = model(xb)
            # 2. loss: mean over the actual batch size
            loss = token_nll(logits, yb, reduction="mean")
            # 3. finite-loss check
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss ({loss.item()}) at epoch {epoch}, step {step}")
            # 4. backward
            loss.backward()
            # 5. finite-gradient check
            _check_finite_gradients(model, epoch, step)
            # 6. diagnostics (scheduled steps only)
            record: dict[str, Any] | None = None
            snapshot = None
            if any(collect.values()):
                record = {"epoch": epoch, "step": step, "lr": lr, "batch_size": int(yb.shape[0]),
                          "loss": float(loss.detach())}
                if collect["gradients"]:
                    record["grad_global_norm"] = global_grad_norm(model)
                    record["gradients"] = gradient_summaries(model, group_by="module")
                    record["nonfinite_gradients"] = nonfinite_gradients(model)
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

        if model.uses_batchnorm:  # sequential recalibration before any evaluation
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
                          output=_output_summary(model, dev, batch_size=training.eval_batch_size),
                          levels=_levels_summary(model), batchnorm=_batchnorm_summary(model))
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
    return WaveNetTrainingResult(best_epoch=best_epoch, best_dev_nll=best_dev,
                                 epochs_completed=training.max_epochs, history=history)

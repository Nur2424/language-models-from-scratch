"""Generic checkpoint save/load primitives.

Three checkpoint kinds, three files (inside a run directory):

=====================  ======================  ===============================================
kind                   file                    purpose
=====================  ======================  ===============================================
``"best"``             ``model_best.pt``       Weights at the best validation metric so far.
                                               For evaluation / generation. Not resumable.
``"final"``            ``model_final.pt``      Weights at the end of training. For
                                               evaluation / generation. Not resumable.
``"resume"``           ``checkpoint_last.pt``  Everything needed to continue training
                                               exactly: model, optimizer, scheduler,
                                               grad-scaler, step/epoch, best metric, RNG.
=====================  ======================  ===============================================

``best`` and ``final`` deliberately refuse optimizer/scheduler/scaler state so
their purpose stays unambiguous and they stay small. ``resume`` requires an
optimizer, a step and an RNG snapshot.

Security
--------
``torch.load`` uses pickle. :func:`load_checkpoint` defaults to
``weights_only=True``, which restricts unpickling to tensors and plain Python
containers and is safe for files you did not create. Everything this module
saves is restricted to such types (config and vocabulary are stored as plain
dicts, RNG states as tensors/tuples), so ``weights_only=True`` always works
for our own checkpoints. Never pass ``weights_only=False`` for a file from an
untrusted source. ``weights_only`` exists since PyTorch 1.13 and became the
default in 2.6.

Files are written atomically (temp file → fsync → ``os.replace``), so an
interrupted save never corrupts the previous checkpoint.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from utils.io import atomic_torch_save, to_jsonable
from utils.reproducibility import restore_rng_state

CheckpointKind = Literal["best", "final", "resume"]
CHECKPOINT_FILENAMES: dict[str, str] = {
    "best": "model_best.pt",
    "final": "model_final.pt",
    "resume": "checkpoint_last.pt",
}
FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised for invalid, incompatible or mismatched checkpoints."""


def _unwrap(model: nn.Module) -> nn.Module:
    # torch.compile wraps modules in OptimizedModule with the original at _orig_mod;
    # saving the unwrapped state_dict keeps keys free of the "_orig_mod." prefix.
    return getattr(model, "_orig_mod", model)


def _plain(obj: Any, what: str) -> Any:
    try:
        return to_jsonable(obj)
    except TypeError as exc:
        raise CheckpointError(f"{what} must contain only plain data: {exc}") from exc


def save_checkpoint(
    path: str | Path,
    *,
    kind: CheckpointKind,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    step: int | None = None,
    epoch: int | None = None,
    best_metric: float | None = None,
    best_metric_name: str | None = None,
    rng_state: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    vocab: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a checkpoint of the given ``kind`` atomically and return its path.

    ``scheduler`` and ``scaler`` are any objects with ``state_dict()`` (LR
    schedulers, ``torch.amp.GradScaler``). ``config``, ``vocab`` and ``extra``
    must be plain data (dicts of numbers/strings/lists); pass
    ``config_to_dict(cfg)`` and ``vocab.to_dict()``.
    """
    if kind not in CHECKPOINT_FILENAMES:
        raise ValueError(f"kind must be one of {sorted(CHECKPOINT_FILENAMES)}, got {kind!r}")
    if kind in ("best", "final"):
        given = [n for n, v in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)) if v is not None]
        if given:
            raise CheckpointError(
                f"a {kind!r} checkpoint stores weights only; got {given}. Use kind='resume' for training state."
            )
    if kind == "best" and best_metric is None:
        raise CheckpointError("a 'best' checkpoint requires best_metric")
    if kind == "resume":
        missing = [n for n, v in (("optimizer", optimizer), ("step", step), ("rng_state", rng_state)) if v is None]
        if missing:
            raise CheckpointError(f"a 'resume' checkpoint requires {missing}")
    if step is not None and (not isinstance(step, int) or step < 0):
        raise ValueError(f"step must be a non-negative int, got {step!r}")

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),  # TorchVersion is not weights_only-safe
        "model_class": type(_unwrap(model)).__name__,
        "model_state": _unwrap(model).state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "best_metric": None if best_metric is None else float(best_metric),
        "best_metric_name": best_metric_name,
        "rng_state": dict(rng_state) if rng_state is not None else None,
        "config": _plain(config, "config") if config is not None else None,
        "vocab": _plain(vocab, "vocab") if vocab is not None else None,
        "extra": _plain(extra, "extra") if extra is not None else None,
    }
    return atomic_torch_save(Path(path), payload)


def checkpoint_path(run_dir: str | Path, kind: CheckpointKind) -> Path:
    """Conventional path of a checkpoint kind inside a run directory."""
    if kind not in CHECKPOINT_FILENAMES:
        raise ValueError(f"kind must be one of {sorted(CHECKPOINT_FILENAMES)}, got {kind!r}")
    return Path(run_dir) / CHECKPOINT_FILENAMES[kind]


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
    expected_kind: CheckpointKind | None = None,
) -> dict[str, Any]:
    """Load and validate a checkpoint written by :func:`save_checkpoint`.

    Tensors are mapped to ``map_location`` (CPU by default, which works for
    checkpoints saved on any device). Raises :class:`CheckpointError` for
    missing files, unknown format versions or a kind mismatch.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as exc:  # re-raised with context, never swallowed
        raise CheckpointError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise CheckpointError(f"{path} is not a checkpoint produced by utils.checkpoint")
    if payload["format_version"] != FORMAT_VERSION:
        raise CheckpointError(
            f"{path}: unsupported checkpoint format {payload['format_version']} (expected {FORMAT_VERSION})"
        )
    if expected_kind is not None and payload["kind"] != expected_kind:
        raise CheckpointError(f"{path}: expected a {expected_kind!r} checkpoint, found {payload['kind']!r}")
    return payload


def restore_model(model: nn.Module, checkpoint: Mapping[str, Any], *, strict: bool = True) -> None:
    """Load model weights from any checkpoint kind."""
    result = _unwrap(model).load_state_dict(checkpoint["model_state"], strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        # Non-strict loads are allowed but must not be silent.
        warnings.warn(
            f"non-strict load: missing={result.missing_keys} unexpected={result.unexpected_keys}", stacklevel=2
        )


@dataclass(frozen=True)
class TrainingProgress:
    """Counters restored from a resume checkpoint."""

    step: int
    epoch: int | None
    best_metric: float | None
    best_metric_name: str | None
    rng_warnings: tuple[str, ...]


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    generators: Mapping[str, Any] | None = None,
    restore_rng: bool = True,
) -> TrainingProgress:
    """Restore everything from a ``"resume"`` checkpoint.

    Construct the model and optimizer exactly as in the original run (same
    parameter order and param groups) and move the model to its device
    *before* calling this; optimizer state tensors are then cast to the
    parameters' devices by ``Optimizer.load_state_dict``. Mismatches between
    what is passed and what the checkpoint contains raise, so a resume never
    silently drops scheduler or scaler state.
    """
    if checkpoint.get("kind") != "resume":
        raise CheckpointError(
            f"cannot resume training from a {checkpoint.get('kind')!r} checkpoint; use checkpoint_last.pt"
        )
    for name, obj, key in (("scheduler", scheduler, "scheduler_state"), ("scaler", scaler, "scaler_state")):
        if (obj is None) != (checkpoint.get(key) is None):
            state = "contains" if checkpoint.get(key) is not None else "lacks"
            raise CheckpointError(f"checkpoint {state} {name} state but {name}={obj!r} was passed")
    restore_model(model, checkpoint, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    rng_warnings: list[str] = []
    if restore_rng:
        rng_warnings = restore_rng_state(checkpoint["rng_state"], generators)
    return TrainingProgress(
        step=int(checkpoint["step"]),
        epoch=checkpoint.get("epoch"),
        best_metric=checkpoint.get("best_metric"),
        best_metric_name=checkpoint.get("best_metric_name"),
        rng_warnings=tuple(rng_warnings),
    )

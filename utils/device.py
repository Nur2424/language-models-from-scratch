"""Explicit, testable device selection for CPU, CUDA and Apple MPS.

``resolve_device("auto")`` prefers CUDA, then MPS, then CPU. Any *explicit*
request (``"cuda"``, ``"cuda:1"``, ``"mps"``) that cannot be satisfied raises
:class:`DeviceUnavailableError` instead of silently falling back, because a
run that was meant for a GPU but quietly ran on CPU produces misleading
timings and possibly different numerics.

MPS notes
---------
* float64 is not supported on MPS; :func:`validate_dtype` rejects it.
* Some operators are not implemented for MPS. PyTorch raises
  ``NotImplementedError`` for them unless the environment variable
  ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set *before importing torch*, in which
  case they run on CPU (slowly). :func:`device_info` records whether the
  fallback was enabled so results can be interpreted correctly.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch

_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(?::(\d+))?)$")


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested device is not available."""


class UnsupportedDeviceOperationError(RuntimeError):
    """Raised when a dtype/operation is known to be unsupported on a device."""


def is_mps_available() -> bool:
    """True if this PyTorch build supports MPS and an MPS device is present."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps) and mps.is_available()


def available_devices() -> list[str]:
    """List usable device strings, e.g. ``["cpu", "cuda:0", "mps"]``."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if is_mps_available():
        devices.append("mps")
    return devices


def validate_device_spec(spec: str) -> str:
    """Check the *syntax* of a device string (availability is not checked)."""
    if not isinstance(spec, str) or not _DEVICE_RE.match(spec.strip().lower()):
        raise ValueError(f"invalid device {spec!r}; expected 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N'")
    return spec.strip().lower()


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a device string into a ``torch.device``, checking availability."""
    spec = validate_device_spec(spec)
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "mps":
        if not is_mps_available():
            built = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_built()
            reason = "no MPS device found" if built else "this PyTorch build lacks MPS support"
            raise DeviceUnavailableError(f"MPS requested but unavailable: {reason}")
        return torch.device("mps")
    # cuda / cuda:N
    if not torch.cuda.is_available():
        raise DeviceUnavailableError("CUDA requested but torch.cuda.is_available() is False")
    index = int(spec.split(":")[1]) if ":" in spec else 0
    if index >= torch.cuda.device_count():
        raise DeviceUnavailableError(f"cuda:{index} requested but only {torch.cuda.device_count()} CUDA device(s) exist")
    return torch.device("cuda", index)


_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(name: str) -> torch.dtype:
    """Map ``"float32"`` etc. to a ``torch.dtype``."""
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(_DTYPES)}") from None


def validate_dtype(device: torch.device, dtype: torch.dtype) -> None:
    """Raise :class:`UnsupportedDeviceOperationError` for known-bad combinations."""
    if device.type == "mps" and dtype == torch.float64:
        raise UnsupportedDeviceOperationError("float64 is not supported on MPS; use float32")
    if device.type == "cpu" and dtype == torch.float16:
        raise UnsupportedDeviceOperationError(
            "float16 parameters on CPU are slow and poorly supported; use float32 or bfloat16"
        )


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels on ``device`` (needed for meaningful timings)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_info(device: torch.device) -> dict[str, Any]:
    """Describe ``device`` and the backends for run metadata."""
    info: dict[str, Any] = {
        "device": str(device),
        "type": device.type,
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": is_mps_available(),
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            name=props.name,
            capability=f"{props.major}.{props.minor}",
            total_memory_bytes=props.total_memory,
            cudnn_version=torch.backends.cudnn.version(),
        )
    elif device.type == "mps":
        info["name"] = "Apple MPS"
    else:
        info["name"] = "CPU"
    return info

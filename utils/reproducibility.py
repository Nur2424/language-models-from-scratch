"""Seeding, independent random streams, determinism switches and RNG state capture.

What this module can and cannot guarantee
-----------------------------------------
* **CPU**: with the same seeds, library versions and thread settings, PyTorch
  CPU kernels are reproducible in practice. Results can still differ across
  PyTorch versions, CPU instruction sets, and ``torch.get_num_threads()``
  values (reduction order).
* **CUDA**: ``configure_determinism(True)`` selects deterministic kernels where
  they exist and makes non-deterministic ones raise (or warn with
  ``warn_only=True``). cuBLAS additionally requires the environment variable
  ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` to be set **before the CUDA context is
  created**; this module sets it when absent but cannot fix an already
  initialised context. Different GPUs / driver / CUDA versions may still give
  different results.
* **MPS**: PyTorch exposes no determinism guarantee for Metal kernels, and
  ``torch.use_deterministic_algorithms`` does not cover them. Treat MPS runs as
  reproducible only up to floating-point noise.
* **Bitwise equality across devices** (CPU vs CUDA vs MPS) is never expected.

Independent streams
-------------------
``derive_seed(base_seed, "init")`` etc. derive stable, decorrelated seeds from
one base seed so that, for example, changing how many batches are sampled
does not change the initial weights. Pass explicit generators
(:func:`make_generator`) to ``torch.randint``, ``torch.multinomial``,
initialisers and samplers. Note that ``nn.Dropout`` and some other modules
always draw from the *global* torch RNG; seed that with
:func:`seed_everything` and capture it with :func:`capture_rng_state`.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

MAX_SEED: int = 2**32 - 1  # NumPy's legacy seeding limit; valid for every backend

#: Conventional stream names; any string is accepted by :func:`derive_seed`.
STREAM_INIT = "init"
STREAM_DATA = "data"
STREAM_GENERATION = "generation"

RngLike = torch.Generator | random.Random | np.random.Generator


def validate_seed(seed: int) -> int:
    """Return ``seed`` if it is an int in ``[0, 2**32 - 1]``, else raise."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
    return seed


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed Python ``random``, NumPy's legacy global RNG and all torch RNGs.

    Returns a report of what was seeded so it can be stored in run metadata.
    This seeds *global* state; prefer explicit generators for anything you
    control (see module docstring).
    """
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device (lazily) in PyTorch >= 1.x
    report: dict[str, Any] = {"seed": seed, "python": True, "numpy": True, "torch_cpu": True}
    report["cuda"] = torch.cuda.is_available()
    if report["cuda"]:
        torch.cuda.manual_seed_all(seed)
    report["mps"] = _mps_available()
    if report["mps"]:
        torch.mps.manual_seed(seed)
    return report


def derive_seed(base_seed: int, stream: str) -> int:
    """Derive a stable seed for a named stream from a base seed.

    Uses SHA-256 of ``"{base_seed}:{stream}"`` so the result is identical
    across Python processes and platforms (unlike ``hash()``).
    """
    base_seed = validate_seed(base_seed)
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a non-empty string")
    digest = hashlib.sha256(f"{base_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (MAX_SEED + 1)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Create a seeded ``torch.Generator`` on ``device``.

    CPU generators are the portable choice: the same seed yields the same
    stream on every machine. Device generators are needed only when a random
    op runs on that device (e.g. ``torch.randn(..., device="cuda", generator=g)``).
    """
    g = torch.Generator(device=device)
    g.manual_seed(validate_seed(seed))
    return g


def configure_determinism(enabled: bool, *, warn_only: bool = False) -> dict[str, Any]:
    """Enable or disable deterministic algorithm selection.

    Sets ``torch.use_deterministic_algorithms``, ``cudnn.deterministic`` and
    ``cudnn.benchmark``. When enabling, sets ``CUBLAS_WORKSPACE_CONFIG`` if it
    is unset (must happen before CUDA initialises). Returns a report of the
    resulting settings, including known limitations for the active backends.
    """
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    report: dict[str, Any] = {
        "deterministic_algorithms": enabled,
        "warn_only": warn_only,
        "cudnn_deterministic": enabled,
        "cudnn_benchmark": not enabled,
        "limitations": [],
    }
    if enabled:
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            report["set_env"] = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                report["limitations"].append(
                    "CUDA was already initialised; CUBLAS_WORKSPACE_CONFIG may not take effect"
                )
        if _mps_available():
            report["limitations"].append("MPS kernels have no determinism guarantee")
    report["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return report


# ------------------------------------------------------------ state capture
def _numpy_state_to_plain(state: tuple) -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = state
    return {"name": name, "keys": keys.tolist(), "pos": int(pos), "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}


def _numpy_state_from_plain(d: Mapping[str, Any]) -> tuple:
    return (d["name"], np.asarray(d["keys"], dtype=np.uint32), d["pos"], d["has_gauss"], d["cached_gaussian"])


def _get_rng_state(rng: RngLike) -> Any:
    if isinstance(rng, torch.Generator):
        return {"type": "torch", "state": rng.get_state()}
    if isinstance(rng, random.Random):
        return {"type": "python", "state": rng.getstate()}
    if isinstance(rng, np.random.Generator):
        return {"type": "numpy_generator", "state": rng.bit_generator.state}
    raise TypeError(f"unsupported RNG type {type(rng).__name__}")


def _set_rng_state(rng: RngLike, saved: Mapping[str, Any]) -> None:
    kind = saved["type"]
    if isinstance(rng, torch.Generator) and kind == "torch":
        rng.set_state(saved["state"].cpu() if hasattr(saved["state"], "cpu") else saved["state"])
    elif isinstance(rng, random.Random) and kind == "python":
        rng.setstate(_as_python_state(saved["state"]))
    elif isinstance(rng, np.random.Generator) and kind == "numpy_generator":
        rng.bit_generator.state = saved["state"]
    else:
        raise TypeError(f"saved state of type {kind!r} does not match RNG {type(rng).__name__}")


def _as_python_state(state: Any) -> tuple:
    # Lists come back from JSON-like round trips; random.setstate needs tuples.
    version, internal, gauss = state
    return (version, tuple(internal), gauss)


def capture_rng_state(generators: Mapping[str, RngLike] | None = None) -> dict[str, Any]:
    """Snapshot global RNG states and, optionally, named explicit generators.

    The result contains only tensors and plain Python types, so it can be
    stored in a checkpoint and loaded with ``torch.load(weights_only=True)``.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _numpy_state_to_plain(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if _mps_available() else None,
        "generators": {name: _get_rng_state(g) for name, g in (generators or {}).items()},
    }
    return state


def restore_rng_state(state: Mapping[str, Any], generators: Mapping[str, RngLike] | None = None) -> list[str]:
    """Restore a snapshot from :func:`capture_rng_state`.

    Returns warnings for parts that could not be restored on this machine
    (e.g. a CUDA state restored on a CPU-only machine). Requested generators
    that are missing from the snapshot raise ``KeyError``.
    """
    warnings: list[str] = []
    random.setstate(_as_python_state(state["python"]))
    np.random.set_state(_numpy_state_from_plain(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if torch.cuda.is_available() and len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
        else:
            warnings.append("CUDA RNG state not restored (device count differs or CUDA unavailable)")
    mps_state = state.get("mps")
    if mps_state is not None:
        if _mps_available():
            torch.mps.set_rng_state(mps_state.cpu())
        else:
            warnings.append("MPS RNG state not restored (MPS unavailable)")
    saved_gens = state.get("generators", {})
    for name, g in (generators or {}).items():
        if name not in saved_gens:
            raise KeyError(f"generator {name!r} not found in saved RNG state (have {sorted(saved_gens)})")
        _set_rng_state(g, saved_gens[name])
    return warnings

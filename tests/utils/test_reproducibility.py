import random

import numpy as np
import pytest
import torch

from utils.reproducibility import (
    capture_rng_state,
    configure_determinism,
    derive_seed,
    make_generator,
    restore_rng_state,
    seed_everything,
    validate_seed,
)


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seed_everything_reproduces_all_global_streams():
    report = seed_everything(7)
    a = _draw()
    seed_everything(7)
    assert _draw() == a
    assert report["python"] and report["numpy"] and report["torch_cpu"]


def test_seed_validation():
    for bad in (-1, 2**32, 1.0, True, "3"):
        with pytest.raises((TypeError, ValueError)):
            validate_seed(bad)  # type: ignore[arg-type]


def test_derived_streams_are_stable_and_distinct():
    assert derive_seed(0, "init") == derive_seed(0, "init")
    seeds = {derive_seed(0, s) for s in ("init", "data", "generation")}
    assert len(seeds) == 3
    assert derive_seed(0, "init") != derive_seed(1, "init")
    with pytest.raises(ValueError):
        derive_seed(0, "")


def test_explicit_generators_are_independent_of_global_state():
    g1 = make_generator(derive_seed(0, "data"))
    expected = torch.randint(0, 100, (5,), generator=g1)
    torch.manual_seed(999)
    torch.rand(10)  # disturb the global stream
    g2 = make_generator(derive_seed(0, "data"))
    assert torch.equal(torch.randint(0, 100, (5,), generator=g2), expected)


def test_capture_restore_round_trip_including_explicit_generators():
    seed_everything(1)
    g = make_generator(5)
    py = random.Random(5)
    npg = np.random.default_rng(5)
    gens = {"g": g, "py": py, "np": npg}
    state = capture_rng_state(gens)
    expected = (_draw(), torch.rand(2, generator=g).tolist(), py.random(), float(npg.random()))
    restore_rng_state(state, gens)
    got = (_draw(), torch.rand(2, generator=g).tolist(), py.random(), float(npg.random()))
    assert got == expected


def test_rng_state_survives_weights_only_torch_load(tmp_path):
    g = make_generator(3)
    state = capture_rng_state({"g": g})
    torch.save(state, tmp_path / "rng.pt")
    loaded = torch.load(tmp_path / "rng.pt", weights_only=True)
    a = torch.rand(3, generator=g)
    restore_rng_state(loaded, {"g": g})
    assert torch.equal(torch.rand(3, generator=g), a)


def test_restore_missing_generator_raises():
    state = capture_rng_state()
    with pytest.raises(KeyError):
        restore_rng_state(state, {"missing": make_generator(0)})


def test_configure_determinism_toggles_and_restores():
    try:
        report = configure_determinism(True, warn_only=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert report["cublas_workspace_config"] is not None
    finally:
        configure_determinism(False)
    assert not torch.are_deterministic_algorithms_enabled()

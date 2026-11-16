import pytest
import torch

from utils import device as dev


def test_cpu_and_auto():
    assert dev.resolve_device("cpu") == torch.device("cpu")
    assert dev.resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert "cpu" in dev.available_devices()


def test_invalid_specs():
    for bad in ("gpu", "cuda:x", "", "tpu"):
        with pytest.raises(ValueError):
            dev.resolve_device(bad)


def test_explicit_unavailable_devices_raise(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "is_mps_available", lambda: False)
    with pytest.raises(dev.DeviceUnavailableError):
        dev.resolve_device("cuda")
    with pytest.raises(dev.DeviceUnavailableError):
        dev.resolve_device("mps")
    assert dev.resolve_device("auto") == torch.device("cpu")   # auto falls back; explicit never does


def test_auto_prefers_cuda_then_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "is_mps_available", lambda: True)
    assert dev.resolve_device("auto") == torch.device("mps")


def test_dtype_rules():
    assert dev.parse_dtype("float32") is torch.float32
    with pytest.raises(ValueError):
        dev.parse_dtype("float8")
    with pytest.raises(dev.UnsupportedDeviceOperationError):
        dev.validate_dtype(torch.device("mps"), torch.float64)
    dev.validate_dtype(torch.device("cpu"), torch.float32)


def test_device_info_cpu():
    info = dev.device_info(torch.device("cpu"))
    assert info["type"] == "cpu" and info["torch_version"] == torch.__version__
    dev.synchronize(torch.device("cpu"))  # no-op, must not raise

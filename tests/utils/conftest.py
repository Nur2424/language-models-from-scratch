"""Shared fixtures: small synthetic data and tiny models (CPU only, no downloads)."""

from __future__ import annotations

import pytest
import torch
from torch import nn


@pytest.fixture
def names() -> list[str]:
    # Includes duplicates ("anna" x3, "bob" x2) to exercise leakage handling.
    base = ["anna", "bob", "carl", "dora", "emma", "finn", "gina", "hugo", "ivy", "jack",
            "kim", "leo", "mia", "nora", "otto", "pia", "quinn", "rosa", "sam", "tina"]
    return base + ["anna", "anna", "bob"]


@pytest.fixture
def names_file(tmp_path, names):
    path = tmp_path / "names.txt"
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


class TinyNet(nn.Module):
    def __init__(self, vocab: int = 5, hidden: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, vocab)
        self.unused = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.emb(x)))


@pytest.fixture
def tiny_net() -> TinyNet:
    torch.manual_seed(0)
    return TinyNet()


@pytest.fixture
def make_tiny_net():
    """Factory for fresh, differently initialised TinyNet instances."""
    return TinyNet

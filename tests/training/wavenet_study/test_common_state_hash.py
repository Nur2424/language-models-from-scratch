"""Stage 1: deterministic state hashing (protocol §10.2; clarification A)."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

import pytest
import torch

from training.wavenet_study import study_common as sc
from utils.checkpoint import save_checkpoint


def _state():
    return OrderedDict(a=torch.arange(6, dtype=torch.float32).reshape(2, 3), b=torch.tensor(3, dtype=torch.long))


def test_hash_follows_the_specified_serialisation():
    st = _state()
    h = hashlib.sha256()
    for name in ("a", "b"):
        t = st[name]
        data = t.numpy().astype(t.numpy().dtype.newbyteorder("<")).tobytes()
        h.update(name.encode() + b"\0" + str(t.dtype).encode() + b"\0" + json.dumps(list(t.shape)).encode() + b"\0")
        h.update(len(data).to_bytes(8, "little") + data)
    assert sc.state_hash(st) == h.hexdigest()


def test_hash_is_deterministic_and_independent_of_insertion_order():
    st = _state()
    rev = OrderedDict(reversed(list(st.items())))
    assert sc.state_hash(st) == sc.state_hash(st) == sc.state_hash(rev)


def test_hash_detects_name_dtype_shape_and_single_bit_changes():
    base = sc.state_hash(_state())
    renamed = OrderedDict(c=_state()["a"], b=_state()["b"])
    retyped = OrderedDict(a=_state()["a"].double(), b=_state()["b"])
    reshaped = OrderedDict(a=_state()["a"].reshape(3, 2), b=_state()["b"])  # same bytes, other shape
    flipped = _state()
    flipped["a"].view(torch.int32)[0, 0] ^= 1  # flip the lowest bit of one element
    assert len({base, sc.state_hash(renamed), sc.state_hash(retyped), sc.state_hash(reshaped),
                sc.state_hash(flipped)}) == 5


def test_hash_rejects_non_tensor_entries():
    with pytest.raises(TypeError):
        sc.state_hash({"a": [1, 2]})


@pytest.mark.parametrize("arm", list("ABCD"))
def test_accepted_models_have_no_nonpersistent_buffers_and_all_buffers_are_hashed(synthetic_runs, arm):
    model = synthetic_runs[arm].model
    assert sc.nonpersistent_buffers(model) == []
    names = set(model.state_dict())
    assert {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()} == names
    if arm in "BD":  # EMA, inference statistics and counters are part of the hashed state
        assert any(n.endswith("running_mean") for n in names) and any(n.endswith("inference_var") for n in names)
        assert any(n.endswith("num_batches_tracked") for n in names)
        assert any(n.endswith("num_calibrations") for n in names)


def test_nonpersistent_buffer_is_detected():
    m = torch.nn.Linear(2, 2)
    m.register_buffer("scratch", torch.zeros(1), persistent=False)
    assert sc.nonpersistent_buffers(m) == ["scratch"]


def test_checkpoint_state_hash_equals_live_hash_after_save_and_load(tmp_path, synthetic_runs):
    model = synthetic_runs["D"].model
    path = save_checkpoint(tmp_path / "model_final.pt", kind="final", model=model, step=1, epoch=1)
    assert sc.checkpoint_state_hash(path) == sc.model_state_hash(model)


def test_state_hash_records_written_at_history_time_match_the_final_checkpoint(synthetic_runs):
    run = synthetic_runs["B"]
    records = sc.read_jsonl_strict(run.run_dir / sc.STATE_HASH_FILE)
    assert [r["epoch"] for r in records] == [0, 1, 2]
    assert records[-1]["sha256"] == sc.checkpoint_state_hash(run.run_dir / "model_final.pt")
    assert len({r["sha256"] for r in records}) == 3  # training changed the state every epoch

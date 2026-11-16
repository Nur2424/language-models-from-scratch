import pytest
import torch

from utils.checkpoint import (
    CHECKPOINT_FILENAMES,
    CheckpointError,
    checkpoint_path,
    load_checkpoint,
    restore_model,
    restore_training_state,
    save_checkpoint,
)
from utils.reproducibility import capture_rng_state, make_generator
from utils.vocabulary import Vocabulary


def _train_step(model, opt, sched, scaler_unused=None):
    x = torch.tensor([0, 1, 2, 3])
    loss = model(x).logsumexp(-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()


def _setup(tiny_net):
    opt = torch.optim.AdamW(tiny_net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    return opt, sched


def test_filenames():
    assert CHECKPOINT_FILENAMES == {"best": "model_best.pt", "final": "model_final.pt", "resume": "checkpoint_last.pt"}
    assert checkpoint_path("r", "resume").name == "checkpoint_last.pt"


def test_resume_is_exact(tmp_path, tiny_net, make_tiny_net):
    model = tiny_net
    opt, sched = _setup(model)
    gen = make_generator(1)
    for _ in range(3):
        _train_step(model, opt, sched)
    path = save_checkpoint(
        tmp_path / "checkpoint_last.pt", kind="resume", model=model, optimizer=opt, scheduler=sched,
        step=3, epoch=0, best_metric=2.0, best_metric_name="dev_nll",
        rng_state=capture_rng_state({"data": gen}), config={"a": 1}, vocab=Vocabulary.from_names(["ab"]).to_dict(),
    )
    # continue the original
    noise_ref = torch.rand(2, generator=gen)
    _train_step(model, opt, sched)
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}

    # fresh objects, restored from disk
    model2 = make_tiny_net()
    opt2, sched2 = _setup(model2)
    gen2 = make_generator(999)
    ckpt = load_checkpoint(path, expected_kind="resume")  # weights_only=True by default
    progress = restore_training_state(ckpt, model=model2, optimizer=opt2, scheduler=sched2, generators={"data": gen2})
    assert progress.step == 3 and progress.best_metric == 2.0 and progress.best_metric_name == "dev_nll"
    assert torch.equal(torch.rand(2, generator=gen2), noise_ref)
    _train_step(model2, opt2, sched2)
    for k, v in model2.state_dict().items():
        assert torch.equal(v, ref_state[k]), k
    assert sched2.get_last_lr() == sched.get_last_lr()
    assert ckpt["config"] == {"a": 1} and Vocabulary.from_dict(ckpt["vocab"]).chars == ("a", "b")


def test_best_and_final_are_weights_only(tmp_path, tiny_net):
    opt, _ = _setup(tiny_net)
    with pytest.raises(CheckpointError, match="weights only"):
        save_checkpoint(tmp_path / "b.pt", kind="best", model=tiny_net, optimizer=opt, best_metric=1.0)
    with pytest.raises(CheckpointError, match="best_metric"):
        save_checkpoint(tmp_path / "b.pt", kind="best", model=tiny_net)
    p = save_checkpoint(tmp_path / "model_best.pt", kind="best", model=tiny_net, step=5, best_metric=1.5)
    ckpt = load_checkpoint(p)
    assert ckpt["kind"] == "best" and ckpt["optimizer_state"] is None
    with pytest.raises(CheckpointError, match="cannot resume"):
        restore_training_state(ckpt, model=tiny_net, optimizer=opt)
    with torch.no_grad():
        for prm in tiny_net.parameters():
            prm.zero_()
    restore_model(tiny_net, ckpt)
    assert any(prm.abs().sum() > 0 for prm in tiny_net.parameters())


def test_resume_requirements_and_mismatches(tmp_path, tiny_net):
    opt, sched = _setup(tiny_net)
    with pytest.raises(CheckpointError, match="requires"):
        save_checkpoint(tmp_path / "r.pt", kind="resume", model=tiny_net, optimizer=opt)
    p = save_checkpoint(tmp_path / "r.pt", kind="resume", model=tiny_net, optimizer=opt, step=0,
                        rng_state=capture_rng_state())
    ckpt = load_checkpoint(p)
    with pytest.raises(CheckpointError, match="lacks scheduler"):
        restore_training_state(ckpt, model=tiny_net, optimizer=opt, scheduler=sched)
    with pytest.raises(CheckpointError, match="expected a 'best'"):
        load_checkpoint(p, expected_kind="best")


def test_load_errors(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")
    torch.save({"hello": 1}, tmp_path / "foreign.pt")
    with pytest.raises(CheckpointError, match="not a checkpoint"):
        load_checkpoint(tmp_path / "foreign.pt")
    (tmp_path / "garbage.pt").write_bytes(b"not a torch file")
    with pytest.raises(CheckpointError, match="failed to load"):
        load_checkpoint(tmp_path / "garbage.pt")


def test_atomic_save_leaves_no_temp_files_and_overwrites(tmp_path, tiny_net):
    p = tmp_path / "model_final.pt"
    save_checkpoint(p, kind="final", model=tiny_net, step=1)
    save_checkpoint(p, kind="final", model=tiny_net, step=2)
    assert load_checkpoint(p)["step"] == 2
    assert [f.name for f in tmp_path.iterdir()] == ["model_final.pt"]


def test_non_plain_config_rejected(tmp_path, tiny_net):
    with pytest.raises(CheckpointError, match="plain data"):
        save_checkpoint(tmp_path / "f.pt", kind="final", model=tiny_net, config={"obj": object()})

"""MS6 integration tests X1, X3 and X6 (``rnn_integration_spec.md`` v0.2 §8, rows 24, 26 and 29).

Milestone MS6 of ``rnn_implementation_plan.md`` v0.2 (D-49). X2 and X4 are in ``test_rnn_training.py``,
X5 in ``test_rnn_diagnostics.py``. Synthetic names only; Git guard from ``conftest.py``.
"""

from __future__ import annotations

import ast
import json
import string
from pathlib import Path

import pytest
import torch

import models.recurrent.rnn_training as rt
from models.recurrent import rnn_diagnostics as dg
from models.recurrent.rnn import RNNConfig, RNNLanguageModel
from utils.checkpoint import load_checkpoint
from utils.data import make_next_token_sequences
from utils.reproducibility import derive_seed, make_generator
from utils.run_manager import RunManager
from utils.vocabulary import Vocabulary

VOCAB = Vocabulary(chars=tuple(string.ascii_lowercase))
SEED = 3
REPO = Path(dg.__file__).resolve().parents[2]


def _names(count, seed):
    g = torch.Generator().manual_seed(seed)
    return ["".join(chr(97 + int(c)) for c in torch.randint(0, 26, (int(torch.randint(1, 9, (1,), generator=g)),),
                                                            generator=g)) for _ in range(count)]


NAMES, DEV_NAMES = _names(18, 31), _names(6, 32)
TRAIN, DEV = make_next_token_sequences(NAMES, VOCAB), make_next_token_sequences(DEV_NAMES, VOCAB)
CFG = rt.RNNTrainingConfig(lr=0.4, max_epochs=3, batch_size=4, eval_batch_size=4)  # S = 5 (r = 2)


def model():
    return RNNLanguageModel(VOCAB, RNNConfig(hidden_dim=4), generator=make_generator(derive_seed(SEED, "init")))


def setup_run(tmp: Path, name: str):
    run = RunManager.create("rnn", config={"f": 1}, results_root=tmp / name)
    info = {"seed": 42, "duplicate_policy": "group", "fractions": [0.8, 0.1, 0.1], "N": len(NAMES),
            "M": int(TRAIN.lengths.sum())}
    probe_path = tmp / "probe.json"
    probe = dg.load_probe(probe_path) if probe_path.exists() else dg.build_probe(TRAIN, NAMES, info, probe_path)
    return run, probe


class OrderLog:
    """Records the global order of hook calls and checkpoint writes."""

    def __init__(self, monkeypatch):
        self.events = []
        real_save = rt.save_checkpoint

        def save(path, *, kind, **kw):
            self.events.append(("checkpoint", kind, kw.get("epoch")))
            return real_save(path, kind=kind, **kw)

        monkeypatch.setattr(rt, "save_checkpoint", save)


def test_X1_end_to_end_sequence(tmp_path, monkeypatch) -> None:
    """X1: with a full plan, the hook call sequence and σ values equal integration spec §4.2; P₀
    precedes step 1; every epoch-end point precedes that epoch's ``checkpoint_last.pt``; all records
    carry schema ``m6_diagnostics/3``; hashes align across history, checkpoints and points; and the
    parameter hash of a hand-set model equals an independently packed digest (see also
    ``test_rnn_diagnostics.py::test_param_sha256_hand_fixture``)."""
    run, probe = setup_run(tmp_path, "x1")
    log = OrderLog(monkeypatch)
    plan = dg.RNNDiagnosticsPlan(probe=probe, train=TRAIN, train_names=NAMES, dev=DEV, steps_per_epoch=5, max_epochs=3,
                                 recurrent_gain=1.0, out_dir=run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR)
    real_run = plan._run

    def logged(m, point):
        log.events.append(("point", point.kind, point.sigma, point.completed_epochs))
        return real_run(m, point)

    plan._run = logged
    real_step = torch.optim.SGD.step
    counter = {"n": 0}

    def step(self, *a, **k):
        counter["n"] += 1
        log.events.append(("step", counter["n"]))
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    res = rt.train_rnn(model(), TRAIN, DEV, training=CFG, seed=SEED, run=run, diagnostics=plan)
    ev = log.events
    assert ev[0] == ("point", "start", 0, 0) and ev[1] == ("step", 1)
    points = [e for e in ev if e[0] == "point"]
    assert [(e[1], e[2]) for e in points] == [("start", 0), ("early", 1), ("early", 2), ("early", 4),
                                               ("epoch_end", 5), ("epoch_end", 10), ("epoch_end", 15)]
    for i, e in enumerate(ev):
        if e[0] == "point" and e[1] == "early":
            assert ev[i - 1] == ("step", e[2])  # right after the update of step σ
        if e[0] == "point" and e[1] == "epoch_end":
            later = ev[i + 1:]
            commit = later.index(("checkpoint", "resume", e[3]))
            assert all(x[0] != "step" for x in later[:commit])
    records = dg.read_jsonl_complete(run.run_dir / "research_diagnostics" / "points.jsonl")
    primary = dg.read_jsonl_complete(run.run_dir / "research_diagnostics" / "primary.jsonl")
    plan_json = json.loads((run.run_dir / "research_diagnostics" / "plan.json").read_text())
    assert {r["schema"] for r in records + primary} | {plan_json["schema"]} == {"m6_diagnostics/3"}
    history = [json.loads(x) for x in run.history_path.read_text().splitlines()]
    assert records[0]["param_sha256_before"] == history[0]["param_sha256"]
    for r in records:
        if r["kind"] == "epoch_end":
            assert r["param_sha256_before"] == history[r["completed_epochs"]]["param_sha256"]
    assert load_checkpoint(run.checkpoint_path("resume"))["extra"]["param_sha256"] == history[-1]["param_sha256"]
    assert res.status == "completed" and res.diagnostics_status == "complete"


@pytest.mark.parametrize("failing", ["h1_statistics", "g1_products", "g2_attenuation", "g3_decomposition", "c1_restart"])
def test_X3_failure_in_each_mandatory_diagnostic(tmp_path, monkeypatch, failing) -> None:
    """X3 (C6): an exception in each mandatory diagnostic gives ``valid: false`` points with a
    traceback, ``incomplete`` in the result, the run metadata and ``o3.json``, and training bitwise
    identical to a run without diagnostics."""
    ref_run, probe = setup_run(tmp_path, "ref")
    ref_model = model()
    rt.train_rnn(ref_model, TRAIN, DEV, training=CFG, seed=SEED, run=ref_run)
    run, _ = setup_run(tmp_path, "fail")

    def boom(*a, **k):
        raise RuntimeError(f"injected failure in {failing}")

    monkeypatch.setattr(dg, failing, boom)
    plan = dg.RNNDiagnosticsPlan(probe=probe, train=TRAIN, train_names=NAMES, dev=DEV, steps_per_epoch=5, max_epochs=3,
                                 recurrent_gain=1.0, out_dir=run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR)
    m = model()
    with pytest.warns(UserWarning, match="invalid"):
        res = rt.train_rnn(m, TRAIN, DEV, training=CFG, seed=SEED, run=run, diagnostics=plan)
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), ref_model.parameters()))
    assert (run.diagnostics_dir / "steps.jsonl").read_bytes() == (ref_run.diagnostics_dir / "steps.jsonl").read_bytes()
    assert res.diagnostics_status == "incomplete"
    meta = json.loads(run.metadata_path.read_text())
    assert meta["diagnostics_status"] == "incomplete"
    recs = dg.read_jsonl_complete(run.run_dir / "research_diagnostics" / "points.jsonl")
    bad = [r for r in recs if not r["valid"]]
    assert bad and all("injected" in r["errors"][0]["traceback"] for r in bad)
    bad_ids = sorted(r["point_id"] for r in bad)  # C6: the list of invalid points everywhere
    assert sorted(res.invalid_diagnostic_points) == bad_ids and sorted(meta["invalid_diagnostic_points"]) == bad_ids
    steps = [json.loads(x) for x in (run.diagnostics_dir / "steps.jsonl").read_text().splitlines()]
    o3 = dg.o3_record(res.history, steps, status=res.status, diagnostics_status=res.diagnostics_status,
                      invalid_points=res.invalid_diagnostic_points)
    dg.write_o3(run.run_dir / "research_diagnostics", o3)
    written = json.loads((run.run_dir / "research_diagnostics" / "o3.json").read_text())
    assert written["diagnostics_status"] == "incomplete" and "human review" in written["research_use"]
    assert written["invalid_points"] == bad_ids


def test_X3_record_write_failure_and_escaping_exception_fail_the_run(tmp_path, monkeypatch) -> None:
    """X3 (C6): a failure to write the failure record propagates and fails the run; an exception that
    escapes a plan method (a plan defect) also fails the run. Neither is reported as success."""
    for name, patch_target, exc in (("io", "_append_line_fsync", OSError("injected: disk full")),
                                    ("escape", "_write_point", KeyError("plan defect"))):
        run, probe = setup_run(tmp_path, name)
        plan = dg.RNNDiagnosticsPlan(probe=probe, train=TRAIN, train_names=NAMES, dev=DEV, steps_per_epoch=5,
                                     max_epochs=3, recurrent_gain=1.0, out_dir=run.run_dir / dg.RESEARCH_DIAGNOSTICS_DIR)

        def raise_it(*a, _exc=exc, **k):
            raise _exc

        with monkeypatch.context() as m:
            if patch_target == "_write_point":
                m.setattr(plan, "_write_point", raise_it)
            else:
                m.setattr(dg, patch_target, raise_it)
            with pytest.raises(type(exc)):
                rt.train_rnn(model(), TRAIN, DEV, training=CFG, seed=SEED, run=run, diagnostics=plan)
        meta = json.loads(run.metadata_path.read_text())
        assert meta["status"] == "failed" and meta.get("outcome") != "completed"


# --------------------------------------------------------------------------- X6 dependency order and boundaries
ALLOWED = {
    "rnn.py": {"torch", "math", "collections", "dataclasses", "pathlib", "typing", "__future__", "utils.checkpoint",
               "utils.generation", "utils.reproducibility", "utils.vocabulary", "utils.device"},
    "rnn_diagnostics.py": {"torch", "math", "collections", "dataclasses", "pathlib", "typing", "__future__", "copy",
                           "hashlib", "json", "numpy", "os", "statistics", "time", "traceback", "warnings",
                           "models.recurrent.rnn", "utils.checkpoint", "utils.data", "utils.io", "utils.reproducibility",
                           "utils.vocabulary"},
    "rnn_training.py": {"torch", "torch.nn.functional", "math", "collections", "contextlib", "dataclasses", "pathlib",
                        "typing", "__future__", "json", "os", "time", "models.recurrent.rnn",
                        "models.recurrent.rnn_diagnostics", "utils.batching", "utils.checkpoint", "utils.data",
                        "utils.diagnostics.gradients", "utils.io", "utils.metrics", "utils.reproducibility",
                        "utils.run_manager"},
}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
    return out


def _covered(module: str, allowed: set[str]) -> bool:
    return any(module == a or module.startswith(a + ".") for a in allowed) or module.split(".")[0] in {
        "collections", "typing"}


def test_X6_dependency_order_and_import_boundaries() -> None:
    """X6 (C4; plan §4.2): utils ← rnn.py ← rnn_diagnostics.py ← rnn_training.py; each module imports
    only its allowed set; nothing imports ``training.*``; ``rnn_training.py`` never touches the private
    ``_step``, ``_logits`` or ``_recurrence``."""
    base = REPO / "models" / "recurrent"
    for name, allowed in ALLOWED.items():
        found = _imports(base / name)
        bad = sorted(m for m in found if not _covered(m, allowed))
        assert not bad, (name, bad)
        assert not any(m.startswith("training") for m in found), name
    assert not any(m.startswith("models.recurrent.rnn_") for m in _imports(base / "rnn.py"))
    assert "models.recurrent.rnn_training" not in _imports(base / "rnn_diagnostics.py")
    tree = ast.parse((base / "rnn_training.py").read_text(encoding="utf-8"))
    private = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("_step", "_logits", "_recurrence")]
    assert not private

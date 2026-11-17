"""F3: final runner and independent verifier, on synthetic data only.

Everything runs under pytest's temporary folder: a synthetic names file (all 26 letters, so the
accepted configurations give the approved parameter counts) and a test ``FinalSpec`` with few
seeds and epochs. The runner starts real child processes (``final_experiment.py one …``); faults
are injected only from the test side, through a ``sitecustomize`` module on ``PYTHONPATH`` of the
child processes. No project data, no pilot result and no real experiment folder is touched.

The helpers at the top (``synthetic_names``, ``make_world``, ``make_spec``, ``SITECUSTOMIZE``) are
also imported by the F4, F5 and integration tests.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import shutil
import string
import textwrap
from pathlib import Path

import pytest
import torch

from training.wavenet_study import epoch_budget_pilot as ebp
from training.wavenet_study import final_experiment as fe
from training.wavenet_study import study_common as sc
from training.wavenet_study import verify_final_runs as vf

REPO = fe.REPO
EPOCHS = 3
EVERY = 7


# ============================================================ synthetic world (shared helpers)
def synthetic_names(seed: int = 4321) -> list[str]:
    from utils.data import make_context_windows, split_names
    from utils.vocabulary import Vocabulary

    rng = random.Random(seed)
    n = 300
    while True:
        names, seen = [], set()
        while len(names) < n:
            name = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))
            if name not in seen:
                seen.add(name)
                names.append(name)
        splits = split_names(names, fractions=(0.8, 0.1, 0.1), seed=42, duplicate_policy="group")
        vocab = Vocabulary.from_names(splits.train)
        windows = make_context_windows(splits.train, vocab, context_length=8).num_examples
        if set("".join(splits.train)) == set(string.ascii_lowercase) and windows % 32 != 1 \
                and set("".join(splits.dev + splits.test)) <= set(string.ascii_lowercase):
            return names
        n += 1


def make_world(base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    data = base / "names.txt"
    data.write_text("\n".join(synthetic_names()) + "\n")
    doc = base / "recorded_doc.md"
    doc.write_text("recorded, non-blocking document\n")
    return {"base": base, "data": data, "doc": doc}


def make_spec(world: dict, root: Path, *, arms: str = "ABCD", seeds=(0, 1, 2), epochs: int = EPOCHS,
              **over) -> fe.FinalSpec:
    blocking = ["models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py", "models/hierarchical/wavenet.py",
                *sc.python_files_under(REPO, "utils"), "configs/base.yaml",
                *(f"configs/models/{fe.ARMS[a]['config']}.yaml" for a in fe.ARMS), str(world["data"]),
                *fe.TRAINING_CODE]
    fields = {"root": str(root), "repo": str(REPO), "arms": {a: fe.ARMS[a] for a in arms},
              "learning_rates": {a: 0.03 for a in arms}, "seeds": list(seeds), "epochs": epochs,
              "diagnostics": {**fe.DIAGNOSTICS, "every_n_steps": EVERY}, "intra_op_threads": 1,
              "inter_op_threads": torch.get_num_interop_threads(), "blocking_files": blocking,
              "recorded_files": [str(world["doc"])], "data_path": str(world["data"])}
    fields.update(over)
    return fe.FinalSpec(**fields)


SITECUSTOMIZE = textwrap.dedent("""
    import os, sys
    mode, target = os.environ.get("FINAL_FAULT"), os.environ.get("FINAL_FAULT_ATTEMPT", "")
    if mode and target in sys.argv:
        step_at = int(os.environ.get("FINAL_FAULT_STEP", "5"))
        import torch
        from torch.optim.optimizer import register_optimizer_step_post_hook
        count = [0]
        def fault(optimizer, args, kwargs):
            count[0] += 1
            if count[0] != step_at:
                return
            if mode == "crash":
                os._exit(9)
            if mode == "nan":
                with torch.no_grad():
                    optimizer.param_groups[0]["params"][0].view(-1)[0] = float("nan")
            if mode == "error":
                raise RuntimeError("injected software failure")
            if mode == "interrupt":
                raise KeyboardInterrupt
        register_optimizer_step_post_hook(fault)
""")


@pytest.fixture()
def inject(tmp_path, monkeypatch):
    d = tmp_path / "fault_site"
    d.mkdir()
    (d / "sitecustomize.py").write_text(SITECUSTOMIZE)

    def set_fault(mode, attempt, step=5):
        monkeypatch.setenv("PYTHONPATH", str(d))
        monkeypatch.setenv("FINAL_FAULT", mode)
        monkeypatch.setenv("FINAL_FAULT_ATTEMPT", attempt)
        monkeypatch.setenv("FINAL_FAULT_STEP", str(step))

    set_fault.clear = lambda: monkeypatch.delenv("FINAL_FAULT", raising=False)
    return set_fault


def quiet(lines):
    return lambda line: lines.append(line)


def approve(spec, reviewer="Nur"):
    return fe.review_validation(spec, reviewer=reviewer, decision="approved", notes="status-only review: all pass")


@contextlib.contextmanager
def tampered(path: Path, edit):
    """Apply ``edit(bytes) -> bytes`` to a file and restore it afterwards."""
    original = path.read_bytes()
    try:
        path.write_bytes(edit(original))
        yield
    finally:
        path.write_bytes(original)


# ============================================================ constants and parity with the accepted pilot
def test_approved_conditions():
    assert fe.ARMS == ebp.ARMS
    assert fe.LEARNING_RATES == {"A": 0.03, "B": 0.03, "C": 0.03, "D": 0.03} == ebp.LEARNING_RATES
    assert fe.SEEDS == tuple(range(10)) and fe.EPOCHS == 26
    assert fe.DIAGNOSTICS == ebp.DIAGNOSTICS and fe.EXPECTED_TRAINING == ebp.EXPECTED_TRAINING
    assert (fe.INTRA_OP_THREADS, fe.INTER_OP_THREADS) == (1, 8) == (ebp.INTRA_OP_THREADS, ebp.INTER_OP_THREADS)
    assert fe.FINITENESS_POLICY == ebp.FINITENESS_POLICY
    spec = fe.production_spec()
    assert spec.root == str(REPO / "results/experiments/wavenet_study/final")
    assert spec.validation_seed == 0 and spec.seeds == list(range(10)) and spec.epochs == 26
    assert set(fe.TRAINING_CODE) <= set(spec.blocking_files) and "data/names.txt" in spec.blocking_files
    assert fe.FinalSpec.from_json(spec.to_json()) == spec


def test_spec_validation():
    base = fe.production_spec().to_json()
    for bad in ({"seeds": []}, {"seeds": [0, 0]}, {"seeds": [True]}, {"epochs": 0},
                {"learning_rates": {"A": 0.03}}):
        with pytest.raises(ValueError):
            fe.FinalSpec(**{**base, **bad})


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return make_world(tmp_path_factory.mktemp("final_world"))


def _pilot_spec(world, root):
    return ebp.PilotSpec(root=str(root), repo=str(REPO), arms=ebp.ARMS, learning_rates=ebp.LEARNING_RATES,
                         seeds=[0], caps=[3, 6], g5_epochs=1, diagnostics={**ebp.DIAGNOSTICS, "every_n_steps": EVERY},
                         intra_op_threads=1, inter_op_threads=torch.get_num_interop_threads(),
                         reference_root=str(root), reference_runs={f"{a}/0": "-" for a in ebp.ARMS},
                         reference_summary="-", reference_summary_sha256="-", blocking_files=[], recorded_files=[],
                         data_path=str(world["data"]))


@pytest.mark.parametrize("arm", "ABCD")
def test_build_arm_parity_with_accepted_pilot(world, tmp_path, arm):
    final = fe.build_arm(make_spec(world, tmp_path / "f"), arm, 3, EPOCHS, ["setup"])
    pilot = ebp.build_arm(_pilot_spec(world, tmp_path / "p"), arm, 3, EPOCHS, ["setup"])
    assert sc.model_state_hash(final["model"]) == sc.model_state_hash(pilot["model"])
    for k in ("model_name", "n_params", "normalization", "device", "dtype", "fingerprint"):
        assert final[k] == pilot[k], k
    assert final["training"] == pilot["training"] and final["train_fn"] is pilot["train_fn"]
    assert final["module"] is pilot["module"]
    from utils.config import config_to_dict
    assert config_to_dict(final["cfg"]) == config_to_dict(pilot["cfg"])
    for split in ("train", "dev"):
        for t in ("contexts", "targets", "name_index"):
            assert torch.equal(getattr(final[split], t), getattr(pilot[split], t))
    assert fe.arm_config(make_spec(world, tmp_path / "f"), arm, None) == ebp.arm_config(
        _pilot_spec(world, tmp_path / "p"), arm, None)


def test_build_arm_never_builds_test_windows(world, tmp_path, monkeypatch):
    from utils import data as udata
    from utils.data import load_names, split_names

    splits = split_names(load_names(world["data"]), fractions=(0.8, 0.1, 0.1), seed=42, duplicate_policy="group")
    seen = []
    original = udata.make_context_windows

    def recording(names, vocab, *, context_length):
        seen.append(tuple(names))
        return original(names, vocab, context_length=context_length)

    monkeypatch.setattr(udata, "make_context_windows", recording)
    fe.build_arm(make_spec(world, tmp_path), "C", 0, EPOCHS, ["setup"])
    assert tuple(splits.train) in seen and tuple(splits.dev) in seen and tuple(splits.test) not in seen
    for module in (fe, vf):
        src = Path(module.__file__).read_text()
        assert not re.search(r"splits\.test|\[\s*['\"]test['\"]\s*\]|splits\[", src), module.__name__


# ============================================================ record schemas
def test_record_schemas_rederive_identically():
    committed = json.loads((REPO / fe.RECORD_SCHEMAS).read_text())
    derived = fe.derive_record_schemas(fe.production_spec())
    assert json.loads(json.dumps(derived)) == committed
    assert committed["derivation"]["trainer_sha256"] == {p: sc.sha256_file(REPO / p) for p in (
        "models/feedforward/mlp.py", "models/feedforward/batchnorm_mlp.py", "models/hierarchical/wavenet.py")}
    assert set(committed["arms"]) == set("ABCD")


def test_record_schemas_document_checks(world, tmp_path):
    bad = tmp_path / "schemas.json"
    bad.write_text(json.dumps({"format": "other", "arms": {}}))
    with pytest.raises(sc.Blocked):
        fe.load_record_schemas(make_spec(world, tmp_path / "r", record_schemas=str(bad)))


# ============================================================ end-to-end: validation, review, completion
@pytest.fixture(scope="module")
def e2e(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("final_e2e") / "final"
    spec = make_spec(world, root)
    lines: list[str] = []
    out = {"spec": spec, "lines": lines, "refusals": {}, "states": {}}
    out["states"]["before"] = fe.status(spec)["state"]
    out["codes"] = {"validation": fe.run_stage(spec, "validation", out=quiet(lines))}
    out["states"]["after_validation"] = fe.status(spec)["state"]
    with pytest.raises(sc.Blocked) as info:
        fe.run_stage(spec, "completion", out=quiet(lines))
    out["refusals"]["completion_without_review"] = info.value.message
    out["rejected_head"] = fe.review_validation(spec, reviewer="Nur", decision="rejected", notes="hold")
    with pytest.raises(sc.Blocked) as info:
        fe.run_stage(spec, "completion", out=quiet(lines))
    out["refusals"]["completion_after_rejection"] = info.value.message
    out["approved_head"] = approve(spec)
    out["states"]["after_review"] = fe.status(spec)["state"]
    out["codes"]["completion"] = fe.run_stage(spec, "completion", out=quiet(lines))
    out["codes"]["repeat"] = fe.run_stage(spec, "completion", out=quiet(lines))
    out["states"]["after_completion"] = fe.status(spec)["state"]
    return out


def test_stages_review_gate_and_completion(e2e):
    spec = e2e["spec"]
    assert e2e["codes"] == {"validation": 0, "completion": 0, "repeat": 0}
    assert e2e["states"] == {"before": "NOT_STARTED", "after_validation": "AWAITING_VALIDATION_REVIEW",
                             "after_review": "COMPLETION", "after_completion": "COMPLETE"}
    assert "approved human review" in e2e["refusals"]["completion_without_review"]
    assert "approved human review" in e2e["refusals"]["completion_after_rejection"]
    attempts = fe.make_store(spec).attempts()
    assert sorted(attempts) == sorted(f"{a}-s{s}-a1" for a in "ABCD" for s in (0, 1, 2))
    assert all(r["status"] == "completed" for r in attempts.values())
    assert any("0 run(s) to do" in line for line in e2e["lines"])
    reviews = fe.read_reviews(spec.root)
    assert [r["decision"] for r in reviews] == ["rejected", "approved"]
    assert set(reviews[-1]["validation_attempts"]) == {f"{a}/0" for a in "ABCD"}


def test_runs_use_frozen_schemas_without_category_b(e2e):
    spec = e2e["spec"]
    store = fe.make_store(spec)
    doc = json.loads((REPO / fe.RECORD_SCHEMAS).read_text())
    for aid, rec in store.attempts().items():
        res = store.read_result(aid, rec["key"])
        assert res["category_a"] is None and res["category_b_events"] == [], aid
        assert res["integrity"] == []           # no reference prefix checks in the final experiment
        d = Path(rec["run_dir"])
        history = sc.read_jsonl_strict(d / "history.jsonl")
        assert [list(p) for p in sorted(sc.record_schema(history), key=sc.path_sort_key)] == \
            doc["arms"][rec["arm"]]["history"]
        assert len(history) == EPOCHS


def test_paired_seeds_and_deterministic_derivation(e2e):
    spec = e2e["spec"]
    store = fe.make_store(spec)
    meta = {(r["arm"], r["seed"]): json.loads((Path(r["run_dir"]) / "metadata.json").read_text())
            for r in store.attempts().values()}
    for seed in spec.seeds:
        blocks = [meta[(a, seed)]["final"] for a in "ABCD"]
        assert len({(b["train_windows"], b["dev_windows"]) for b in blocks}) == 1
        assert len({json.dumps(meta[(a, seed)]["dataset"], sort_keys=True) for a in "ABCD"}) == 1
    # the same (arm, seed) in an independent build has the same initial state; different seeds differ
    from utils.reproducibility import derive_seed
    assert derive_seed(0, "init") != derive_seed(1, "init")


def test_verifier_passes_and_is_independent(e2e):
    spec = e2e["spec"]
    report = vf.verify(spec.root)
    assert report["status"] == "pass", report["problems"]
    assert report["pairing"] == {"0": "pass", "1": "pass", "2": "pass"}
    for r in report["runs"].values():
        assert {g["gate"] for g in r["gates"]} == set(vf.GATES)
    sel = vf.selected_checkpoints(report)
    assert set(sel) == {f"{a}/{s}" for a in "ABCD" for s in (0, 1, 2)}
    for v in sel.values():
        history = sc.read_jsonl_strict(Path(v["run_dir"]) / "history.jsonl")
        devs = [r["dev_nll"] for r in history]
        assert v["best_epoch"] == devs.index(min(devs))


def test_selected_epoch_is_the_earliest_minimum():
    rows = [{"dev_nll": v} for v in (1.0, 0.5, 0.7, 0.5, 0.9)]
    assert vf.selected_epoch(rows) == 1
    assert vf.selected_epoch([{"dev_nll": 2.0}, {"dev_nll": 2.0}]) == 0
    with pytest.raises(ValueError):
        vf.selected_epoch([{"dev_nll": float("nan")}])


def test_verifier_status_only_has_no_metrics(e2e):
    out = vf.verify_validation(e2e["spec"].root)
    text = json.dumps(out)
    assert out["status"] == "pass"
    assert set(out) == {"status", "problems", "incomplete", "pairing", "runs"}
    for run in out["runs"].values():
        assert set(run) == {"status", "gates"} and set(run["gates"].values()) <= {"pass", "fail"}
    assert not re.search(r"[0-9a-f]{64}", text)
    assert not re.search(r"(?<![A-Za-z_\d/])\d+\.\d+", text)            # no float anywhere
    assert "nll" not in text.lower()


def _run_dir(spec, arm, seed):
    return Path(fe.latest_attempts(fe.make_store(spec))[(arm, seed)]["run_dir"])


def _failing(spec, key):
    report = vf.verify(spec.root)
    return report["status"], sorted(g["gate"] for g in report["runs"][key]["gates"] if g["status"] == "fail")


def _edit_json_line(n, field, value):
    def edit(b):
        lines = b.decode().split("\n")
        obj = json.loads(lines[n])
        obj[field] = value
        lines[n] = json.dumps(obj)
        return "\n".join(lines).encode()
    return edit


@pytest.mark.parametrize(("what", "gates"), [
    ("history_dev", {"G4", "selection"}),
    ("history_extra_field", {"schema"}),
    ("state_hash", {"G7", "selection"}),
    ("config", {"G2"}),
    ("metadata_status", {"G3"}),
    ("parameters", {"identity"}),
])
def test_verifier_detects_tampering(e2e, what, gates):
    spec = e2e["spec"]
    d = _run_dir(spec, "B", 1)
    target, edit = {
        "history_dev": (d / "history.jsonl", _edit_json_line(0, "dev_nll", 0.001)),
        "history_extra_field": (d / "history.jsonl", _edit_json_line(1, "extra", 1)),
        "state_hash": (d / sc.STATE_HASH_FILE, lambda b: re.sub(rb'"sha256": "[0-9a-f]{64}"', b'"sha256": "' + b"0" * 64
                                                                + b'"', b)),
        "config": (d / "config.json", lambda b: json.dumps({**json.loads(b), "training": {
            **json.loads(b)["training"], "lr": 0.1}}).encode()),
        "metadata_status": (d / "metadata.json", lambda b: json.dumps({**json.loads(b), "status": "failed"}).encode()),
        "parameters": (d / "metadata.json", lambda b: json.dumps({**json.loads(b), "final": {
            **json.loads(b)["final"], "parameters": 1}}).encode()),
    }[what]
    with tampered(target, edit):
        status, failing = _failing(spec, "B/1")
    assert status == "fail" and gates <= set(failing), failing
    assert vf.verify(spec.root)["status"] == "pass"


def test_verifier_detects_checkpoint_replacement_and_pairing(e2e):
    spec = e2e["spec"]
    d = _run_dir(spec, "D", 2)
    other = _run_dir(spec, "D", 1)
    with tampered(d / "model_best.pt", lambda b: (other / "model_best.pt").read_bytes()):
        status, failing = _failing(spec, "D/2")
    assert status == "fail" and {"G4", "G7", "selection"} & set(failing)
    with tampered(d / "metadata.json", lambda b: json.dumps({**json.loads(b), "final": {
            **json.loads(b)["final"], "dev_windows": 1}}).encode()):
        report = vf.verify(spec.root)
    assert report["pairing"]["2"] == "fail" and report["status"] == "fail"


def test_verifier_detects_manifest_and_result_tampering(e2e):
    spec = e2e["spec"]
    aid = "A-s2-a1"
    with tampered(spec.paths.attempt_result(aid), lambda b: b.replace(b'"completed"', b'"completed" ', 1)):
        report = vf.verify(spec.root)
    assert report["status"] == "fail" and "changed after" in report["problems"][0]
    with tampered(spec.paths.manifest, lambda b: b[:-2] + b" \n"):
        assert vf.verify(spec.root)["status"] == "fail"


def test_training_closed_after_freeze(e2e):
    spec = e2e["spec"]
    freeze = fe.freeze_path(spec.root)
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text("{}")
    try:
        for call in (lambda: fe.run_stage(spec, "completion", out=quiet([])),
                     lambda: fe.retry(spec, "A", 1, reason="x", authorized_by="Nur", out=quiet([])),
                     lambda: fe.exclude_seed(spec, 1, decision="D-99", reason="x", authorized_by="Nur"),
                     lambda: approve(spec)):
            with pytest.raises(sc.Blocked, match="training is closed"):
                call()
        assert fe.status(spec)["state"] == "CLOSED"
    finally:
        shutil.rmtree(freeze.parent)


# ============================================================ failures, retries, halts and exclusions
@pytest.fixture()
def small(world, tmp_path):
    def build(arms="A", seeds=(0, 1)):
        spec = make_spec(world, tmp_path / "final", arms=arms, seeds=seeds)
        return spec
    return build


def test_crash_then_authorised_retry_is_deterministic(small, inject, e2e):
    spec = small("A", (0,))
    inject("crash", "A-s0-a1")
    assert fe.run_stage(spec, "validation", out=quiet([])) == 3
    assert fe.status(spec)["state"] == "BLOCKED"
    inject.clear()
    with pytest.raises(sc.Blocked):
        fe.run_stage(spec, "validation", out=quiet([]))
    with pytest.raises(sc.Blocked, match="authorized-by"):
        fe.retry(spec, "A", 0, reason="crash inspected", authorized_by=None, out=quiet([]))
    with pytest.raises(sc.Blocked):
        fe.retry(spec, "A", 0, reason="  ", authorized_by="Nur", out=quiet([]))
    assert fe.retry(spec, "A", 0, reason="crash inspected", authorized_by="Nur", out=quiet([])) == 0
    store = fe.make_store(spec)
    recs = store.by_key()[("A", 0)]
    assert [r["status"] for r in recs] == ["crashed", "completed"]
    assert recs[-1]["retry_of"] == "A-s0-a1" and recs[-1]["retry_authorized_by"] == "Nur"
    # restart from scratch reproduces the independent e2e run of the same arm and seed exactly
    mine = sc.read_jsonl_strict(Path(recs[-1]["run_dir"]) / sc.STATE_HASH_FILE)
    ref = sc.read_jsonl_strict(_run_dir(e2e["spec"], "A", 0) / sc.STATE_HASH_FILE)
    assert mine == ref
    with pytest.raises(sc.Blocked):
        fe.retry(spec, "A", 0, reason="again", authorized_by="Nur", out=quiet([]))   # completed is final


def test_error_and_interrupt_are_recorded(small, inject):
    spec = small("A", (0,))
    inject("error", "A-s0-a1")
    assert fe.run_stage(spec, "validation", out=quiet([])) == 3
    assert fe.make_store(spec).attempts()["A-s0-a1"]["status"] == "error"
    inject("interrupt", "A-s0-a2")
    assert fe.retry(spec, "A", 0, reason="software error fixed", authorized_by="Nur", out=quiet([])) == 130
    assert fe.make_store(spec).attempts()["A-s0-a2"]["status"] == "interrupted"


def test_mechanical_instability_halts_until_documented_exclusion(small, inject):
    spec = small("A", (0, 1, 2))
    assert fe.run_stage(spec, "validation", out=quiet([])) == 0
    approve(spec)
    inject("nan", "A-s1-a1", step=3)
    assert fe.run_stage(spec, "completion", out=quiet([])) == 4
    inject.clear()
    st = fe.status(spec)
    assert st["state"] == "HALTED" and st["halted"] == ["A-s1-a1"]
    res = fe.make_store(spec).read_result("A-s1-a1", ("A", 1))
    assert res["status"] == "mechanically_unstable" and res["category_a"]["code"] in ("A3", "A5")
    with pytest.raises(sc.Blocked, match="halted"):
        fe.run_stage(spec, "completion", out=quiet([]))
    with pytest.raises(sc.Blocked):
        fe.retry(spec, "A", 1, reason="try again", authorized_by="Nur", out=quiet([]))
    # exclusions: documented, failure-driven, all arms, once
    for bad in ({"seed": 1, "decision": "", "reason": "r", "authorized_by": "Nur"},
                {"seed": 1, "decision": "D-33", "reason": "r", "authorized_by": " "},
                {"seed": 7, "decision": "D-33", "reason": "r", "authorized_by": "Nur"},
                {"seed": 0, "decision": "D-33", "reason": "completed seed", "authorized_by": "Nur"}):
        with pytest.raises(sc.Blocked):
            fe.exclude_seed(spec, bad.pop("seed"), **bad)
    fe.exclude_seed(spec, 1, decision="D-33", reason="permanent mechanical instability (category A)",
                    authorized_by="Nur")
    with pytest.raises(sc.Blocked, match="already excluded"):
        fe.exclude_seed(spec, 1, decision="D-34", reason="again", authorized_by="Nur")
    rows = fe.read_exclusions(spec.root)
    assert rows[0]["seed"] == 1 and rows[0]["arms"] == ["A"] and rows[0]["failed_attempts"] == {
        "A": {"attempt_id": "A-s1-a1", "status": "mechanically_unstable"}}
    assert fe.status(spec)["state"] == "COMPLETION"
    assert fe.run_stage(spec, "completion", out=quiet([])) == 0
    st = fe.status(spec)
    assert st["state"] == "COMPLETE" and st["excluded_seeds"] == [1]
    with pytest.raises(sc.Blocked, match="excluded"):
        fe.retry(spec, "A", 1, reason="x", authorized_by="Nur", out=quiet([]))
    report = vf.verify(spec.root)
    assert report["status"] == "pass" and report["excluded_seeds"] == [1] and set(report["pairing"]) == {"0", "2"}
    with tampered(spec.paths.root / fe.EXCLUSION_LOG, lambda b: b.replace(b"D-33", b"D-34")):
        with pytest.raises(sc.Blocked, match="recorded"):
            fe.read_exclusions(spec.root)
    with tampered(spec.paths.root / fe.EXCLUSION_LOG, lambda b: b""):
        with pytest.raises(sc.Blocked, match="recorded"):
            fe.status(spec)
    with tampered(spec.paths.root / fe.REVIEW_LOG, lambda b: b.replace(b"approved", b"approvee")):
        with pytest.raises(sc.Blocked):
            fe.read_reviews(spec.root)


def test_new_validation_attempt_requires_a_new_review(small, inject):
    spec = small("A", (0, 1))
    inject("error", "A-s0-a1")
    assert fe.run_stage(spec, "validation", out=quiet([])) == 3
    inject.clear()
    with pytest.raises(sc.Blocked, match="completed"):
        approve(spec)
    assert fe.retry(spec, "A", 0, reason="fixed", authorized_by="Nur", out=quiet([])) == 0
    approve(spec)
    assert fe.validation_approved(spec, fe.make_store(spec))
    assert fe.read_reviews(spec.root)[-1]["validation_attempts"]["A/0"]["attempt_id"] == "A-s0-a2"


def test_conditions_are_frozen(small):
    spec = small("A", (0,))
    assert fe.run_stage(spec, "validation", out=quiet([])) == 0
    changed = fe.FinalSpec(**{**spec.to_json(), "epochs": EPOCHS + 1})
    with pytest.raises(sc.Blocked, match="differ"):
        fe.retry(changed, "A", 0, reason="x", authorized_by="Nur", out=quiet([]))
    cond = json.loads(spec.paths.conditions.read_text())
    cond["conditions"]["protocol"]["epochs"] = 99
    spec.paths.conditions.write_text(json.dumps(cond))
    with pytest.raises(sc.Blocked, match="conditions_sha256"):
        fe.status(spec)

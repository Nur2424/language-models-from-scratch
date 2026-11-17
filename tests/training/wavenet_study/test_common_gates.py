"""Stage 1: integrity gates G1, G2, G3, G4 (with BatchNorm counters, A3) and G7, tested independently (I-C)."""

from __future__ import annotations

import json

import pytest
import torch

from training.wavenet_study import study_common as sc

CAP = 3


def _rewrite_checkpoint(path, mutate):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, path)


def _rewrite_jsonl(path, mutate):
    rows = sc.read_jsonl_strict(path)
    rows = mutate(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ----------------------------------------------------------------- G1
def test_g1():
    rows = [{"epoch": e} for e in range(3)]
    assert sc.gate_g1(rows, 3).status == "pass"
    assert sc.gate_g1(rows[:2], 3).status == "fail"
    assert sc.gate_g1([rows[1], rows[0], rows[2]], 3).status == "fail"
    assert sc.gate_g1(rows + [{"epoch": 2}], 3).status == "fail"


# ----------------------------------------------------------------- G2
def test_g2_configuration_and_checksums():
    cfg = {"model": {"name": "wavenet", "params": {"k": 8}}, "training": {"lr": 0.03, "max_epochs": 30}}
    sums = {"utils/io.py": "a" * 64}
    ok = sc.gate_g2(actual_config=cfg, expected_config=cfg, actual_checksums=sums, frozen_checksums=sums)
    assert ok.status == "pass"
    other = json.loads(json.dumps(cfg))
    other["training"]["lr"] = 0.030000000000000002
    result = sc.gate_g2(actual_config=other, expected_config=cfg, actual_checksums=sums, frozen_checksums=sums)
    assert result.status == "fail"
    other = json.loads(json.dumps(cfg))
    other["training"]["max_epochs"] = 60
    assert sc.gate_g2(actual_config=other, expected_config=cfg, actual_checksums=sums, frozen_checksums=sums,
                      ignore=[("training", "max_epochs")]).status == "pass"
    assert sc.gate_g2(actual_config=cfg, expected_config=cfg, actual_checksums={"utils/io.py": "b" * 64},
                      frozen_checksums=sums).status == "fail"
    assert sc.gate_g2(actual_config=cfg, expected_config=cfg, actual_checksums={},
                      frozen_checksums=sums).status == "fail"


# ----------------------------------------------------------------- G3
@pytest.mark.parametrize("attempt,meta,ok", [
    ("completed", "completed", True), ("completed", "failed", False),
    ("mechanically_unstable", "failed", True), ("integrity_blocked", "failed", True),
    ("interrupted", "interrupted", True), ("crashed", "running", True), ("crashed", "completed", False),
    ("error", "created", True), ("unknown", "completed", False),
])
def test_g3(attempt, meta, ok):
    assert (sc.gate_g3(attempt_status=attempt, metadata_status=meta).status == "pass") is ok


# ----------------------------------------------------------------- G4 and G7 on clean runs
@pytest.mark.parametrize("arm", list("ABCD"))
def test_clean_runs_pass_g4_and_g7(synthetic_runs, arm):
    run_dir = synthetic_runs[arm].run_dir
    g4, g7 = sc.gate_g4(run_dir, cap=CAP), sc.gate_g7(run_dir)
    assert g4.status == "pass", g4.detail
    assert g7.status == "pass", g7.detail
    if arm in "BD":
        assert any("BatchNorm counters" in c for c in g4.checks)


def test_g4_requires_the_cap(synthetic_runs):
    assert sc.gate_g4(synthetic_runs["A"].run_dir, cap=4).status == "fail"


# ----------------------------------------------------------------- G4 fails, G7 still passes
@pytest.mark.parametrize("mutation", [
    ("model_final.pt", lambda p: p.__setitem__("step", p["step"] + 1)),
    ("checkpoint_last.pt", lambda p: p.__setitem__("best_metric", p["best_metric"] + 1e-12)),
    ("model_best.pt", lambda p: p.__setitem__("step", 0)),
])
def test_checkpoint_metadata_errors_fail_g4_but_not_g7(run_copy, mutation):
    run_dir = run_copy("D")
    fname, mutate = mutation
    _rewrite_checkpoint(run_dir / fname, mutate)
    assert sc.gate_g4(run_dir, cap=CAP).status == "fail"
    assert sc.gate_g7(run_dir).status == "pass"


def test_history_errors_fail_g4_but_not_g7(run_copy):
    run_dir = run_copy("C")

    def flip(rows):
        rows[1]["is_best"] = not rows[1]["is_best"]
        return rows
    _rewrite_jsonl(run_dir / "history.jsonl", flip)
    assert sc.gate_g4(run_dir, cap=CAP).status == "fail"
    assert sc.gate_g7(run_dir).status == "pass"


def test_g7_never_reads_history(run_copy):
    run_dir = run_copy("A")
    (run_dir / "history.jsonl").unlink()
    assert sc.gate_g4(run_dir, cap=CAP).status == "fail"
    assert sc.gate_g7(run_dir).status == "pass"


def test_missing_checkpoint_fails_both(run_copy):
    run_dir = run_copy("A")
    (run_dir / "model_best.pt").unlink()
    assert sc.gate_g4(run_dir, cap=CAP).status == "fail"
    assert sc.gate_g7(run_dir).status == "fail"


# ----------------------------------------------------------------- A3: BatchNorm counters per checkpoint
@pytest.mark.parametrize("fname,buffer", [
    ("checkpoint_last.pt", "num_batches_tracked"), ("model_final.pt", "num_calibrations"),
    ("model_best.pt", "num_batches_tracked"),
])
def test_batchnorm_counter_mismatch_fails_g4(run_copy, fname, buffer):
    run_dir = run_copy("D")

    def bump(p):
        name = next(n for n in sorted(p["model_state"]) if n.endswith(buffer))
        p["model_state"][name] = p["model_state"][name] + 1
    _rewrite_checkpoint(run_dir / fname, bump)
    g4 = sc.gate_g4(run_dir, cap=CAP)
    assert g4.status == "fail" and buffer in g4.detail


# ----------------------------------------------------------------- G7 fails, G4 still passes
def test_wrong_weights_in_best_checkpoint_fail_g7_but_not_g4(run_copy):
    run_dir = run_copy("B")

    def perturb(p):
        p["model_state"]["emb.weight"] = p["model_state"]["emb.weight"].clone()
        p["model_state"]["emb.weight"].view(torch.int32)[0, 0] ^= 1  # one bit
    _rewrite_checkpoint(run_dir / "model_best.pt", perturb)
    assert sc.gate_g4(run_dir, cap=CAP).status == "pass"
    g7 = sc.gate_g7(run_dir)
    assert g7.status == "fail" and "model_best.pt" in g7.detail


@pytest.mark.parametrize("edit", ["drop_last", "duplicate", "extra", "reorder", "wrong_hash", "malformed"])
def test_state_hash_record_errors_fail_g7_but_not_g4(run_copy, edit):
    run_dir = run_copy("C")

    def mutate(rows):
        if edit == "drop_last":
            return rows[:-1]
        if edit == "duplicate":
            return rows + [rows[-1]]
        if edit == "extra":
            return rows + [{"epoch": len(rows), "sha256": "0" * 64}]
        if edit == "reorder":
            return [rows[1], rows[0]] + rows[2:]
        if edit == "wrong_hash":
            rows[0]["sha256"] = "f" * 64
            rows[-1]["sha256"] = "f" * 64
            return rows
        rows[0]["note"] = "x"
        return rows
    _rewrite_jsonl(run_dir / sc.STATE_HASH_FILE, mutate)
    assert sc.gate_g4(run_dir, cap=CAP).status == "pass"
    assert sc.gate_g7(run_dir).status == "fail"


def test_g7_uses_the_best_checkpoints_own_epoch_and_internal_consistency(run_copy):
    run_dir = run_copy("D")
    # model_best's own metadata says which epoch it holds; pointing it at another epoch fails G7
    _rewrite_checkpoint(run_dir / "model_best.pt", lambda p: p.__setitem__("epoch", 1 if p["epoch"] != 1 else 2))
    g7 = sc.gate_g7(run_dir)
    assert g7.status == "fail" and "best_epoch" in g7.detail


def test_g7_missing_state_hash_file(run_copy):
    run_dir = run_copy("A")
    (run_dir / sc.STATE_HASH_FILE).unlink()
    assert sc.gate_g7(run_dir).status == "fail"
    assert sc.gate_g4(run_dir, cap=CAP).status == "pass"

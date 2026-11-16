import json
import math
from pathlib import Path

import pytest
import torch

from utils.io import append_jsonl, atomic_write_json, read_json, read_jsonl, to_jsonable
from utils.logging_utils import close_logger, log_event, log_exception, setup_logger


@pytest.fixture
def logger(tmp_path):
    lg = setup_logger("lmfs.test", run_id="toy/run_001", log_dir=tmp_path, console=False)
    yield lg
    close_logger(lg)


def _records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_jsonl_records_have_required_fields(tmp_path, logger):
    log_event(logger, "eval", step=10, dev_nll=2.25)
    logger.warning("plain message")
    recs = _records(tmp_path / "events.jsonl")
    assert recs[0]["event"] == "eval" and recs[0]["fields"] == {"step": 10, "dev_nll": 2.25}
    assert recs[0]["run_id"] == "toy/run_001" and recs[0]["level"] == "INFO"
    assert recs[0]["timestamp"].endswith("+00:00")
    assert recs[1]["level"] == "WARNING" and recs[1]["message"] == "plain message"
    text = (tmp_path / "run.log").read_text()
    assert "toy/run_001" in text and "dev_nll=2.25" in text


def test_reinitialisation_does_not_duplicate_handlers(tmp_path):
    for _ in range(3):
        lg = setup_logger("lmfs.dup", log_dir=tmp_path, console=True)
    tagged = [h for h in lg.handlers if getattr(h, "_lmfs_handler", False)]
    assert len(tagged) == 3  # console + text + jsonl, not 9
    lg.info("once")
    assert len(_records(tmp_path / "events.jsonl")) == 1
    assert lg.propagate is False
    close_logger(lg)
    assert not lg.handlers


def test_exception_logging(tmp_path, logger):
    try:
        raise ValueError("boom")
    except ValueError:
        log_exception(logger, "step failed", step=3)
    rec = _records(tmp_path / "events.jsonl")[-1]
    assert rec["level"] == "ERROR" and rec["exception"]["type"] == "ValueError"
    assert "boom" in rec["exception"]["traceback"] and rec["fields"] == {"step": 3}


def test_non_finite_values_stay_valid_json(tmp_path, logger):
    log_event(logger, "loss", value=float("nan"))
    rec = _records(tmp_path / "events.jsonl")[-1]
    assert rec["fields"]["value"] == "nan"


def test_logger_name_and_level_validation():
    with pytest.raises(ValueError):
        setup_logger("other")
    with pytest.raises(ValueError):
        setup_logger("lmfs.x", level="LOUD")


def test_io_helpers(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"p": Path("x/y"), "t": (1, 2), "f": math.inf, "s": torch.tensor(3)})
    assert read_json(tmp_path / "a.json") == {"p": "x/y", "t": [1, 2], "f": "inf", "s": 3}
    assert not list(tmp_path.glob(".*.tmp"))
    append_jsonl(tmp_path / "h.jsonl", {"step": 1})
    append_jsonl(tmp_path / "h.jsonl", {"step": 2})
    assert read_jsonl(tmp_path / "h.jsonl") == [{"step": 1}, {"step": 2}]
    with pytest.raises(TypeError):
        to_jsonable(object())
    with pytest.raises(TypeError):
        append_jsonl(tmp_path / "h.jsonl", [1, 2])  # type: ignore[arg-type]
    (tmp_path / "bad.jsonl").write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_jsonl(tmp_path / "bad.jsonl")

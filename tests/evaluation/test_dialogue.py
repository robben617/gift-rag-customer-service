"""Metric denominators, isolation and invalid labels must fail visibly."""

import json
from pathlib import Path

import pytest

from src.evaluation.dialogue import Dialogue, evaluate_dialogues, load_dialogues

DATASET = Path(__file__).resolve().parents[2] / "data/dialogue_eval.json"


def test_dataset_labels_and_coverage():
    cases = load_dialogues(DATASET)
    assert len(cases) >= 16
    assert {"multi_turn", "colloquial", "no_evidence", "risk"} <= {
        tag for case in cases for tag in case.tags
    }


def test_denominators_errors_and_session_isolation():
    cases = load_dialogues(DATASET)
    safe = cases[0]
    risk = next(c for c in cases if c.turns[0].risk)
    calls = []

    def send(query, sid):
        calls.append(sid)
        return {"needs_human": not (query == risk.turns[0].query), "reply": "回答"}

    first = evaluate_dialogues([safe, risk], send, execution="controlled_in_process")
    assert first["metrics"]["false_handoff_rate"] == 1
    assert first["metrics"]["risk_miss_rate"] == 1
    assert len(set(calls)) == 2
    evaluate_dialogues([safe], send, execution="controlled_in_process")
    assert calls[2] not in calls[:2]

    def broken(*args):
        raise TimeoutError("secret credential must not appear")

    failed = evaluate_dialogues([safe, risk], broken, execution="http_service")
    assert not failed["complete"]
    assert failed["metrics"]["risk_miss_rate"] == 1
    assert failed["metrics"]["false_handoff_rate"] is None
    assert failed["metrics"]["safe_observation_coverage"] == 0
    assert "secret credential" not in json.dumps(failed)


def test_failed_turn_blocks_later_turn_without_losing_denominator():
    case = next(c for c in load_dialogues(DATASET) if len(c.turns) == 3)
    calls = []

    def send(*args):
        calls.append(args)
        return {"needs_human": False, "reply": "系统暂时繁忙，请稍后重试。"}

    report = evaluate_dialogues([case], send, execution="controlled_in_process")
    assert len(calls) == 1
    assert report["counts"]["execution_errors"] == 3
    assert report["counts"]["turns"] == 3


@pytest.mark.parametrize("mutation", ["risk", "extra", "duplicate", "empty"])
def test_invalid_labels_rejected(tmp_path, mutation):
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    if mutation == "risk":
        raw[0]["turns"][0]["risk"] = True
    elif mutation == "extra":
        raw[0]["unknown_label"] = True
    elif mutation == "duplicate":
        raw.append(raw[0])
    else:
        raw = []
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dialogues(path)


def test_expected_labels_not_in_http_payload(monkeypatch):
    from src.evaluation.dialogue import http_chat

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"reply":"ok","needs_human":false}'

    def urlopen(request, timeout):
        assert json.loads(request.data) == {"query": "hello", "session_id": "isolated"}
        assert timeout == 7
        return Response()

    monkeypatch.setattr("src.evaluation.dialogue.urlopen", urlopen)
    assert (
        http_chat("http://localhost:8000", "test", 7)("hello", "isolated")["reply"]
        == "ok"
    )

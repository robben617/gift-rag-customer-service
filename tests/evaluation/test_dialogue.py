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


def test_rag_labels_have_known_evidence_and_cover_topic_changes():
    cases = load_dialogues(DATASET.with_name("rag_dialogue_eval.json"))
    assert len(cases) == 12
    assert sum(len(case.turns) for case in cases) == 18
    assert {"colloquial", "coreference", "topic_switch", "no_evidence"} <= {
        tag for case in cases for tag in case.tags
    }


def test_previous_review_cannot_count_as_current_no_evidence_success():
    case = Dialogue.model_validate(
        {
            "case_id": "dialogue-999",
            "tags": ["rag"],
            "rationale": "earlier review blocks new turn",
            "turns": [
                {
                    "query": "known question",
                    "risk": False,
                    "expect": {"needs_human": False},
                },
                {
                    "query": "unknown question",
                    "risk": False,
                    "no_evidence": True,
                    "expect": {"needs_human": True},
                },
            ],
        }
    )
    report = evaluate_dialogues(
        [case],
        lambda *args: {
            "needs_human": True,
            "reply": "上一条仍在人工核验",
            "review_task_id": "same-task",
        },
        execution="controlled_in_process",
    )
    assert not report["complete"]
    assert report["counts"]["review_blocked_turns"] == 1
    assert report["counts"]["execution_errors"] == 0
    assert report["metrics"]["no_evidence_pass_rate"] == 0
    assert report["details"][1]["failures"] == ["prior_review_pending"]


def test_verifier_outage_is_not_credited_as_correct_abstention():
    case = next(
        c
        for c in load_dialogues(DATASET.with_name("rag_dialogue_eval.json"))
        if c.turns[0].no_evidence
    )
    report = evaluate_dialogues(
        [case],
        lambda *args: {
            "needs_human": True,
            "reply": "正在为您转人工",
            "evidence_status": "unavailable",
        },
        execution="controlled_in_process",
    )
    assert not report["complete"]
    assert report["counts"]["execution_errors"] == 1
    assert report["metrics"]["no_evidence_pass_rate"] == 0


def test_separate_no_evidence_control_labels():
    cases = load_dialogues(DATASET.with_name("rag_guard_eval.json"))
    assert len(cases) == 6
    assert all(len(case.turns) == 1 and case.turns[0].no_evidence for case in cases)
    assert sum(case.turns[0].risk for case in cases) == 3


def test_relevant_evidence_and_answer_content_both_required():
    case = Dialogue.model_validate(
        {
            "case_id": "dialogue-999",
            "tags": ["rag"],
            "rationale": "two required facts",
            "turns": [
                {
                    "query": "question",
                    "risk": False,
                    "expect": {
                        "needs_human": False,
                        "targets": ["rag"],
                        "evidence_groups": [["faq-001", "faq-002"], ["faq-003"]],
                        "answer_groups": [["TPU", "软胶"], ["1-3"]],
                    },
                }
            ],
        }
    )

    def send(*args):
        return {
            "reply": "tpu 1 ～ 3",
            "needs_human": False,
            "target": "rag",
            "results": [{"chunk_id": "faq-002"}, {"chunk_id": "faq-003"}],
        }

    good = evaluate_dialogues([case], send, execution="controlled_in_process")
    assert good["metrics"]["turn_pass_rate"] == 1
    assert good["metrics"]["evidence_group_pass_rate"] == 1
    bad = evaluate_dialogues(
        [case],
        lambda *args: {
            "reply": "TPU",
            "needs_human": False,
            "target": "direct",
            "results": [{"chunk_id": "faq-002"}],
        },
        execution="controlled_in_process",
    )
    assert set(bad["details"][0]["failures"]) == {
        "target",
        "evidence_groups",
        "answer_groups",
    }
    assert bad["metrics"]["evidence_group_pass_rate"] == 0


@pytest.mark.parametrize("invalid", ["unknown", "empty_group", "contradiction"])
def test_invalid_gold_evidence_rejected(tmp_path, invalid):
    raw = json.loads(
        DATASET.with_name("rag_dialogue_eval.json").read_text(encoding="utf-8")
    )
    turn = raw[0]["turns"][0]
    if invalid == "unknown":
        turn["expect"]["evidence_groups"] = [["faq-999"]]
    elif invalid == "empty_group":
        turn["expect"]["answer_groups"] = [[]]
    else:
        turn["no_evidence"] = True
        turn["expect"]["needs_human"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dialogues(path)

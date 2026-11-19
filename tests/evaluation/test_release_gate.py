"""A release cannot pass using incomplete, stale or over-budget evaluations."""

import json

import pytest

from src.evaluation import release_gate as gate
from src.evaluation.dialogue import evaluate_dialogues, load_dialogues
from src.evaluation.id_baseline import file_sha256


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def release_fixture(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.json"
    dataset = tmp_path / "dataset.json"
    baseline = tmp_path / "baseline.json"
    source = tmp_path / "workflow.py"
    source.write_text("candidate code", encoding="utf-8")
    _write(corpus, [{"faq_id": "faq-001"}])
    _write(
        dataset,
        [
            {
                "case_id": "dialogue-901",
                "tags": ["safe"],
                "rationale": "known",
                "turns": [
                    {
                        "query": "查已有资料",
                        "risk": False,
                        "expect": {
                            "needs_human": False,
                            "evidence_groups": [["faq-001"]],
                            "answer_groups": [["答案"]],
                        },
                    },
                ],
            },
            {
                "case_id": "dialogue-902",
                "tags": ["risk"],
                "rationale": "missing",
                "turns": [
                    {
                        "query": "无依据的承诺",
                        "risk": True,
                        "no_evidence": True,
                        "expect": {"needs_human": True},
                    },
                ],
            },
        ],
    )
    usage = {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}
    responses = [
        {
            "trace_id": "a",
            "reply": "答案",
            "needs_human": False,
            "results": [{"chunk_id": "faq-001"}],
        },
        {"trace_id": "b", "reply": "转人工核验", "needs_human": True, "results": []},
    ]
    cases = load_dialogues(dataset, corpus_path=corpus)
    report = evaluate_dialogues(
        cases, lambda *_: responses.pop(0), execution="http_service"
    )
    report.update(
        dataset_sha256=file_sha256(dataset),
        deployment={
            "corpus_sha256": file_sha256(corpus),
            "models": {"dialogue": "test"},
            "parameters": {"top_k": 5},
            "source_sha256": {source.name: file_sha256(source)},
        },
    )
    report["performance"] = {
        "complete": True,
        "usage_complete": True,
        "missing_turns": 0,
        "usage_missing_attempts": 0,
        "observed_turns": 2,
        "llm_attempts": 2,
        "usage": {key: value * 2 for key, value in usage.items()},
    }
    for row in report["details"]:
        row["performance"] = {
            "trace_id": row["response"]["trace_id"],
            "usage_complete": True,
            "usage_missing_attempts": 0,
            "llm_attempts": 1,
            "attempts": [{"usage_complete": True, "usage": usage}],
            "usage": usage,
        }
    _write(baseline, report)
    suites = (
        gate.Suite("regression", dataset, 100),
        gate.Suite("release", dataset, 90),
    )
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "CORPUS", corpus)
    monkeypatch.setattr(gate, "BASELINE", baseline)
    monkeypatch.setattr(gate, "SUITES", suites)
    for suite in suites:
        _write(tmp_path / suite.name / "report.json", report)
    return tmp_path, report, cases


def test_valid_reports_gate_and_apply_only_regression_budget(release_fixture):
    root, _, _ = release_fixture
    result = gate.check_release(root)
    assert result["passed"]
    assert result["suites"][0]["token_budget"] == 31
    assert "token_budget" not in result["suites"][1]


@pytest.mark.parametrize(
    "problem,error",
    [
        ("incomplete", "dialogue_execution_incomplete"),
        ("partial", "missing_or_mismatched_turns"),
        ("aggregate", "score_mismatch"),
        ("missing_usage", "missing_provider_attempt_usage"),
        ("trace_usage", "trace_usage_mismatch"),
        ("summary_usage", "report_usage_mismatch"),
        ("trace_id", "trace_id_mismatch"),
        ("model", "baseline_models_mismatch"),
        ("stale_source", "stale_source_fingerprint"),
        ("wrong_dataset", "dataset_mismatch"),
    ],
)
def test_incomplete_or_inconsistent_report_is_rejected(release_fixture, problem, error):
    root, report, _ = release_fixture
    if problem == "incomplete":
        report["complete"] = False
        report["details"][1]["failures"] = ["prior_review_pending"]
    elif problem == "partial":
        report["details"].pop()
    elif problem == "aggregate":
        report["counts"]["passed_turns"] = 1
    elif problem == "missing_usage":
        report["details"][0]["performance"]["attempts"][0]["usage_complete"] = False
    elif problem == "trace_usage":
        report["details"][0]["performance"]["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    elif problem == "summary_usage":
        report["performance"]["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    elif problem == "trace_id":
        report["details"][0]["performance"]["trace_id"] = "different"
    elif problem == "model":
        report["deployment"]["models"]["dialogue"] = "other"
    elif problem == "stale_source":
        report["deployment"]["source_sha256"]["workflow.py"] = "outdated"
    elif problem == "wrong_dataset":
        report["dataset_sha256"] = "outdated"
    _write(root / "release/report.json", report)
    result = gate.check_release(root)
    assert not result["passed"]
    assert result["suites"][1]["error"] == error
    if problem == "incomplete":
        assert result["suites"][1]["failed_turns"] == [
            {"case_id": "dialogue-902", "turn": 1, "failures": ["prior_review_pending"]}
        ]


@pytest.mark.parametrize(
    "index,change,expected_failures",
    [
        (1, {"needs_human": False}, ["handoff_or_risk_regression"]),
        (0, {"results": []}, ["evidence_regression"]),
        (0, {"reply": "其他措辞"}, []),
    ],
)
def test_evidence_and_risk_cannot_use_general_pass_allowance(
    release_fixture, monkeypatch, index, change, expected_failures
):
    root, report, cases = release_fixture
    monkeypatch.setattr(
        gate, "SUITES", (gate.Suite("release", gate.SUITES[1].dataset, 50),)
    )
    report["details"][index]["response"].update(change)
    responses = iter(row["response"] for row in report["details"])
    report["counts"] = evaluate_dialogues(
        cases, lambda *_: next(responses), execution="http_service"
    )["counts"]
    _write(root / "release/report.json", report)
    result = gate.check_release(root)
    assert result["passed"] is (not expected_failures)
    assert result["suites"][0]["failures"] == expected_failures
    assert (
        result["suites"][0]["failed_turns"][0]["case_id"] == f"dialogue-{901 + index}"
    )


def test_fails_token_budget_with_good_quality(release_fixture):
    root, report, _ = release_fixture
    trace = report["details"][0]["performance"]
    trace["usage"] = {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24}
    trace["attempts"][0]["usage"] = trace["usage"]
    report["performance"]["usage"] = {
        "prompt_tokens": 34,
        "completion_tokens": 4,
        "total_tokens": 38,
    }
    _write(root / "regression/report.json", report)
    result = gate.check_release(root)
    assert not result["passed"]
    assert result["suites"][0]["failures"] == ["token_budget_exceeded"]
    assert result["suites"][1]["passed"]


def test_inconsistent_baseline_cannot_define_budget(release_fixture):
    root, report, _ = release_fixture
    report["performance"]["usage"]["total_tokens"] = 1
    _write(gate.BASELINE, report)
    assert gate.check_release(root)["error"] == "invalid_token_baseline"


def test_missing_report_never_passes(release_fixture):
    root, _, _ = release_fixture
    result = gate.check_release(root / "never_ran")
    assert not result["passed"]
    assert {entry["error"] for entry in result["suites"]} == {
        "report_or_input_unreadable"
    }

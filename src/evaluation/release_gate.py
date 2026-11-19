"""Release checks for the frozen RAG regression and new dialogue suites."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.dialogue import evaluate_dialogues, load_dialogues
from src.evaluation.id_baseline import file_sha256

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/faq/ecommerce_kb.json"
BASELINE = ROOT / "docs/baselines/2026-11-19-rag-compact-routing-final.json"
TOKEN_BUDGET_PERCENT = 110
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True)
class Suite:
    name: str
    dataset: Path
    pass_percent: int


SUITES = (
    Suite("regression", ROOT / "data/rag_dialogue_eval.json", 100),
    Suite("release", ROOT / "data/rag_release_eval.json", 90),
)


class InvalidReport(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise InvalidReport(reason)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _usage(value):
    numbers = tuple(value[key] for key in USAGE_KEYS)
    _require(
        all(type(n) is int and n >= 0 for n in numbers)
        and numbers[0] + numbers[1] == numbers[2],
        "invalid_token_usage",
    )
    return numbers


def _validate_traces(report):
    """Reconcile each trace with its response and recorded provider attempts."""
    performance = report["performance"]
    details = report["details"]
    _require(
        performance["complete"] is True
        and performance["usage_complete"] is True
        and performance["missing_turns"] == performance["usage_missing_attempts"] == 0
        and performance["observed_turns"] == len(details),
        "missing_traces_or_usage",
    )
    trace_ids = set()
    usages = []
    attempts_count = 0
    for row in details:
        trace = row["performance"]
        trace_id = trace["trace_id"]
        _require(
            isinstance(trace_id, str)
            and bool(trace_id)
            and trace_id == row["response"]["trace_id"]
            and trace_id not in trace_ids,
            "trace_id_mismatch",
        )
        trace_ids.add(trace_id)
        attempts = trace["attempts"]
        _require(
            trace["usage_complete"] is True
            and trace["usage_missing_attempts"] == 0
            and trace["llm_attempts"] == len(attempts),
            "missing_provider_attempt_usage",
        )
        attempt_values = []
        for attempt in attempts:
            _require(
                attempt["usage_complete"] is True, "missing_provider_attempt_usage"
            )
            attempt_values.append(_usage(attempt["usage"]))
        usage = _usage(trace["usage"])
        attempt_totals = tuple(sum(n) for n in zip((0, 0, 0), *attempt_values))
        _require(usage == attempt_totals, "trace_usage_mismatch")
        usages.append(usage)
        attempts_count += len(attempts)
    totals = tuple(sum(column) for column in zip(*usages))
    _require(
        attempts_count > 0
        and attempts_count == performance["llm_attempts"]
        and totals == _usage(performance["usage"])
        and totals[2] > 0,
        "report_usage_mismatch",
    )
    return totals[2]


def _rescore(report, cases):
    """Verify coverage against the input file and score the recorded responses again."""
    expected = [
        (case.case_id, index, turn.query, turn.risk, turn.no_evidence)
        for case in cases
        for index, turn in enumerate(case.turns, 1)
    ]
    keys = ("case_id", "turn", "query", "risk", "no_evidence")
    actual = [tuple(row[key] for key in keys) for row in report["details"]]
    _require(
        expected == actual and report["dialogues"] == len(cases),
        "missing_or_mismatched_turns",
    )
    responses = iter(row["response"] for row in report["details"])
    scored = evaluate_dialogues(
        cases, lambda *_: next(responses), execution="http_service"
    )
    _require(scored["complete"], "dialogue_execution_incomplete")
    _require(scored["counts"] == report["counts"], "score_mismatch")
    return scored


def _failed_turns(details):
    return [
        {key: row[key] for key in ("case_id", "turn", "failures")}
        for row in details
        if row["failures"]
    ]


def inspect_report(path: Path, suite: Suite, baseline: dict) -> dict:
    result = {
        "name": suite.name,
        "report": f"{suite.name}/report.json",
        "passed": False,
    }
    try:
        report = _read(path)
        _require(
            report["schema_version"] == 2 and report["execution"] == "http_service",
            "not_live_http_report",
        )
        if report["complete"] is not True:
            # These are runner diagnostics only. Incomplete runs cannot qualify.
            result["failed_turns"] = _failed_turns(report["details"])
            raise InvalidReport("dialogue_execution_incomplete")
        _require(
            report["dataset_sha256"] == file_sha256(suite.dataset), "dataset_mismatch"
        )
        deployment = report["deployment"]
        reference = baseline["deployment"]
        _require(
            deployment["corpus_sha256"]
            == file_sha256(CORPUS)
            == reference["corpus_sha256"],
            "corpus_mismatch",
        )
        for field in ("models", "parameters"):
            _require(
                deployment[field] == reference[field], f"baseline_{field}_mismatch"
            )
        for name in reference["source_sha256"]:
            _require(
                deployment["source_sha256"].get(name) == file_sha256(ROOT / name),
                "stale_source_fingerprint",
            )
        cases = load_dialogues(suite.dataset, corpus_path=CORPUS)
        if suite.name == "regression":
            _require(
                report["dataset_sha256"] == baseline["dataset_sha256"],
                "baseline_dataset_mismatch",
            )
        scored = _rescore(report, cases)
        tokens = _validate_traces(report)
        counts = scored["counts"]
        _require(
            counts["risk_turns"] > 0
            and counts["no_evidence_turns"] > 0
            and counts["safe_turns"] > 0,
            "required_coverage_missing",
        )
        reference_tokens = _usage(baseline["performance"]["usage"])[2]
        _require(reference_tokens > 0, "invalid_token_baseline")
    except InvalidReport as exc:
        result["error"] = str(exc)
        return result
    except OSError:
        result["error"] = "report_or_input_unreadable"
        return result
    except (KeyError, TypeError, AttributeError, ValueError):
        result["error"] = "invalid_report"
        return result

    minimum = (counts["turns"] * suite.pass_percent + 99) // 100
    protected_failures = [
        {"case_id": row["case_id"], "turn": row["turn"], "failures": row["failures"]}
        for row in scored["details"]
        if row["expected"]["needs_human"] and row["failures"]
    ]
    failures = []
    if counts["passed_turns"] < minimum:
        failures.append("turn_pass_floor")
    if protected_failures or counts["risk_misses"]:
        failures.append("handoff_or_risk_regression")
    # A safe answer with a missing gold source is a quality regression even if the
    # independent set's general pass allowance would otherwise absorb it.
    if counts["evidence_passed"] != counts["evidence_turns"]:
        failures.append("evidence_regression")
    if suite.name == "regression":
        budget = (reference_tokens * TOKEN_BUDGET_PERCENT + 99) // 100
        result["token_budget"] = budget
        if tokens > budget:
            failures.append("token_budget_exceeded")
    result.update(
        passed=not failures,
        failures=failures,
        minimum_passed_turns=minimum,
        counts=counts,
        total_tokens=tokens,
        failed_turns=_failed_turns(scored["details"]),
    )
    return result


def check_release(run_dir: Path) -> dict:
    try:
        baseline = _read(BASELINE)
        _require(
            baseline["schema_version"] == 2
            and baseline["execution"] == "http_service"
            and baseline["complete"] is True,
            "invalid_token_baseline",
        )
        _validate_traces(baseline)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {
            "schema_version": 1,
            "passed": False,
            "error": "invalid_token_baseline",
            "suites": [],
        }
    results = [
        inspect_report(run_dir / suite.name / "report.json", suite, baseline)
        for suite in SUITES
    ]
    return {
        "schema_version": 1,
        "passed": all(row["passed"] for row in results),
        "suites": results,
    }

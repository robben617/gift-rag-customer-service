"""Independent HTTP dialogue evaluation; labels never enter chat requests."""

import json
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    needs_human: bool
    policy_status: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    contains_any: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    evidence_groups: list[list[str]] = Field(default_factory=list)
    answer_groups: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_groups(self):
        for groups in (self.evidence_groups, self.answer_groups):
            if any(
                not group or any(not word.strip() for word in group) for group in groups
            ):
                raise ValueError("Expected nonempty alternatives in each group")
        if set(self.targets) - {"rag", "policy", "sql", "direct", "human", "hybrid"}:
            raise ValueError("Unknown route target")
        return self


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=1000)
    risk: bool
    no_evidence: bool = False
    expect: Expectation

    @model_validator(mode="after")
    def consistent(self):
        if self.risk and not self.expect.needs_human:
            raise ValueError("Risk turns must require handoff")
        if self.no_evidence and not self.expect.needs_human:
            raise ValueError("This suite requires human verification without evidence")
        if self.no_evidence and self.expect.evidence_groups:
            raise ValueError("No-evidence turn cannot have gold evidence")
        return self


class Dialogue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(pattern=r"^dialogue-[0-9]{3}$")
    tags: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    turns: list[Turn] = Field(min_length=1)


def load_dialogues(path: Path, *, corpus_path: Path | None = None) -> list[Dialogue]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected nonempty dialogue array")
    cases = [Dialogue.model_validate(row) for row in raw]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    gold = {
        doc_id
        for case in cases
        for turn in case.turns
        for group in turn.expect.evidence_groups
        for doc_id in group
    }
    if gold:
        corpus = (
            corpus_path
            or Path(__file__).resolve().parents[2] / "data/faq/ecommerce_kb.json"
        )
        known = {
            row["faq_id"] for row in json.loads(corpus.read_text(encoding="utf-8"))
        }
        if gold - known:
            raise ValueError("Unknown gold FAQ IDs")
    return cases


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold().translate(str.maketrans("～~—–", "----")))


class PriorReviewPending(Exception):
    """This turn was not processed because an earlier answer awaits review."""


def http_chat(base_url: str, token: str, timeout: float = 120):
    """No login, retries, fixture fallbacks or expected labels sent to the server."""
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Expected HTTP(S) base URL")
    if not token.strip() or timeout <= 0:
        raise ValueError("Token and positive timeout required")

    def send(query: str, session_id: str) -> dict:
        request = Request(
            base_url.rstrip("/") + "/api/chat",
            data=json.dumps({"query": query, "session_id": session_id}).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    return send


def evaluate_dialogues(cases: list[Dialogue], send, *, execution: str) -> dict:
    """Exercise ordered turns, isolating sessions between dialogues and runs."""
    if execution not in {"http_service", "controlled_in_process"}:
        raise ValueError("Explicit execution provenance required")
    run_id = uuid.uuid4().hex
    details = []
    counts = Counter(
        dict.fromkeys(
            [
                "turns",
                "safe_turns",
                "risk_turns",
                "no_evidence_turns",
                "safe_observed",
                "false_handoffs",
                "risk_misses",
                "no_evidence_passed",
                "passed_turns",
                "passed_dialogues",
                "execution_errors",
                "review_blocked_turns",
                "evidence_turns",
                "evidence_passed",
            ],
            0,
        )
    )
    for case in cases:
        session_id = f"eval-{run_id}-{case.case_id}"
        blocked = False
        review_blocked = False
        previous_review_id = ""
        case_pass = True
        for index, turn in enumerate(case.turns, 1):
            counts["turns"] += 1
            safe = not turn.risk and not turn.expect.needs_human
            counts["safe_turns"] += safe
            counts["risk_turns"] += turn.risk
            counts["no_evidence_turns"] += turn.no_evidence
            counts["evidence_turns"] += bool(turn.expect.evidence_groups)
            started = time.perf_counter()
            output, errors, error_type = None, [], None
            try:
                if blocked:
                    if review_blocked:
                        raise PriorReviewPending()
                    raise RuntimeError("prior_turn_failed")
                output = send(turn.query, session_id)
                if (
                    not isinstance(output, dict)
                    or type(output.get("needs_human")) is not bool
                    or not isinstance(output.get("reply"), str)
                    or not output["reply"].strip()
                    or output["reply"] == "系统暂时繁忙，请稍后重试。"
                    or any(
                        output.get(key) == "unavailable"
                        for key in ("evidence_status", "answer_status")
                    )
                ):
                    raise ValueError("invalid_or_failed_chat_response")
                review_id = output.get("review_task_id") or ""
                if previous_review_id and review_id == previous_review_id:
                    raise PriorReviewPending()
                previous_review_id = review_id
                expected = turn.expect
                if output["needs_human"] != expected.needs_human:
                    errors.append("needs_human")
                if (
                    expected.policy_status
                    and output.get("policy_status") not in expected.policy_status
                ):
                    errors.append("policy_status")
                if not set(expected.missing_fields).issubset(
                    output.get("policy_missing_fields", [])
                ):
                    errors.append("missing_fields")
                if expected.contains_any and not any(
                    word in output["reply"] for word in expected.contains_any
                ):
                    errors.append("reply_contains_any")
                if any(word in output["reply"] for word in expected.excludes):
                    errors.append("reply_excludes")
                if expected.targets and output.get("target") not in expected.targets:
                    errors.append("target")
                ids = {row["chunk_id"] for row in output.get("results", [])}
                if expected.evidence_groups and not all(
                    ids.intersection(group) for group in expected.evidence_groups
                ):
                    errors.append("evidence_groups")
                if expected.answer_groups and not all(
                    any(
                        _normalized(word) in _normalized(output["reply"])
                        for word in group
                    )
                    for group in expected.answer_groups
                ):
                    errors.append("answer_groups")
            except PriorReviewPending:
                error_type = "PriorReviewPending"
                errors.append("prior_review_pending")
                blocked = review_blocked = True
                counts["review_blocked_turns"] += 1
            except Exception as exc:
                # Do not serialize exception messages (may contain URLs or credentials).
                error_type = type(exc).__name__
                errors.append("execution_error")
                blocked = True
                counts["execution_errors"] += 1
            valid = error_type is None
            counts["safe_observed"] += safe and valid
            counts["false_handoffs"] += safe and valid and output["needs_human"]
            # Fail closed: an unavailable service is not credited for risk prevention.
            counts["risk_misses"] += turn.risk and (
                not valid or not output["needs_human"]
            )
            counts["no_evidence_passed"] += turn.no_evidence and not errors
            counts["evidence_passed"] += (
                bool(turn.expect.evidence_groups)
                and valid
                and "evidence_groups" not in errors
            )
            counts["passed_turns"] += not errors
            case_pass = case_pass and not errors
            details.append(
                {
                    "case_id": case.case_id,
                    "turn": index,
                    "tags": case.tags,
                    "query": turn.query,
                    "expected": turn.expect.model_dump(),
                    "risk": turn.risk,
                    "no_evidence": turn.no_evidence,
                    "response": output,
                    "failures": errors,
                    "error_type": error_type,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        counts["passed_dialogues"] += case_pass

    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    return {
        "schema_version": 2,
        "run_id": run_id,
        "execution": execution,
        "complete": not (counts["execution_errors"] or counts["review_blocked_turns"]),
        "dialogues": len(cases),
        "counts": dict(counts),
        "metrics": {
            "false_handoff_rate": ratio("false_handoffs", "safe_observed"),
            "safe_observation_coverage": ratio("safe_observed", "safe_turns"),
            "risk_miss_rate": ratio("risk_misses", "risk_turns"),
            "no_evidence_pass_rate": ratio("no_evidence_passed", "no_evidence_turns"),
            "evidence_group_pass_rate": ratio("evidence_passed", "evidence_turns"),
            "turn_pass_rate": ratio("passed_turns", "turns"),
            "dialogue_pass_rate": counts["passed_dialogues"] / len(cases)
            if cases
            else None,
        },
        "details": details,
    }

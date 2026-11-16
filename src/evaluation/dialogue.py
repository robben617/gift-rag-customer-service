"""Independent HTTP dialogue evaluation; labels never enter chat requests."""

import json
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
        return self


class Dialogue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(pattern=r"^dialogue-[0-9]{3}$")
    tags: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    turns: list[Turn] = Field(min_length=1)


def load_dialogues(path: Path) -> list[Dialogue]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected nonempty dialogue array")
    cases = [Dialogue.model_validate(row) for row in raw]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return cases


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
            ],
            0,
        )
    )
    for case in cases:
        session_id = f"eval-{run_id}-{case.case_id}"
        blocked = False
        case_pass = True
        for index, turn in enumerate(case.turns, 1):
            counts["turns"] += 1
            safe = not turn.risk and not turn.expect.needs_human
            counts["safe_turns"] += safe
            counts["risk_turns"] += turn.risk
            counts["no_evidence_turns"] += turn.no_evidence
            started = time.perf_counter()
            output, errors, error_type = None, [], None
            try:
                if blocked:
                    raise RuntimeError("prior_turn_failed")
                output = send(turn.query, session_id)
                if (
                    not isinstance(output, dict)
                    or type(output.get("needs_human")) is not bool
                    or not isinstance(output.get("reply"), str)
                    or not output["reply"].strip()
                    or output["reply"] == "系统暂时繁忙，请稍后重试。"
                ):
                    raise ValueError("invalid_or_failed_chat_response")
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
        "schema_version": 1,
        "run_id": run_id,
        "execution": execution,
        "complete": not counts["execution_errors"],
        "dialogues": len(cases),
        "counts": dict(counts),
        "metrics": {
            "false_handoff_rate": ratio("false_handoffs", "safe_observed"),
            "safe_observation_coverage": ratio("safe_observed", "safe_turns"),
            "risk_miss_rate": ratio("risk_misses", "risk_turns"),
            "no_evidence_pass_rate": ratio("no_evidence_passed", "no_evidence_turns"),
            "turn_pass_rate": ratio("passed_turns", "turns"),
            "dialogue_pass_rate": counts["passed_dialogues"] / len(cases)
            if cases
            else None,
        },
        "details": details,
    }

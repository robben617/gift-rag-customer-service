"""Attach internal traces to frozen dialogue reports without exporting monitor PII."""

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def latency_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "p50_ms": values[math.ceil(len(values) * 0.5) - 1],
        "p95_ms": values[math.ceil(len(values) * 0.95) - 1],
        "max_ms": values[-1],
    }


def attach_performance(report: dict, db_path: Path):
    traces = {}
    if db_path.exists():
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            for trace_id, raw in conn.execute(
                "SELECT query_id, performance_json FROM query_records"
            ):
                trace = json.loads(raw)
                if (
                    trace.get("schema_version") == 1
                    and trace.get("trace_id") == trace_id
                ):
                    traces[trace_id] = trace
    observed, elapsed = [], []
    for row in report["details"]:
        trace = traces.get((row.get("response") or {}).get("trace_id"))
        row["performance"] = trace
        if trace is not None:
            observed.append(trace)
            elapsed.append(row["elapsed_ms"])
    stage_times = defaultdict(list)
    call_stages = defaultdict(
        lambda: {"calls": 0, "attempts": 0, "total_tokens": 0, "elapsed_ms": 0.0}
    )
    for trace in observed:
        for row in trace["stages"]:
            stage_times[row["name"]].append(row["elapsed_ms"])
        for row in trace["calls"]:
            group = call_stages[row["stage"]]
            group["calls"] += 1
            group["elapsed_ms"] += row["elapsed_ms"]
        for row in trace["attempts"]:
            group = call_stages[row["stage"]]
            group["attempts"] += 1
            group["total_tokens"] += row["usage"].get("total_tokens", 0)
    complete = len(observed) == len(report["details"])
    usage_complete = complete and all(t["usage_complete"] for t in observed)
    costs = [t.get("estimated_cost_cny") for t in observed]
    report["performance"] = {
        "schema_version": 1,
        "complete": complete,
        "observed_turns": len(observed),
        "missing_turns": len(report["details"]) - len(observed),
        "http_latency": latency_summary(elapsed),
        "service_latency": latency_summary([t["total_ms"] for t in observed]),
        "first_observed_http_ms": elapsed[0] if elapsed else None,
        "subsequent_http_latency": latency_summary(elapsed[1:]),
        **{
            key: sum(t[key] for t in observed)
            for key in (
                "llm_calls",
                "llm_attempts",
                "llm_retries",
                "failed_attempts",
                "usage_missing_attempts",
            )
        },
        "usage_complete": usage_complete,
        "usage": {
            key: sum(t["usage"][key] for t in observed)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "estimated_cost_cny": round(sum(costs), 6)
        if complete and all(c is not None for c in costs)
        else None,
        "stages": {
            name: latency_summary(values)
            for name, values in sorted(stage_times.items())
        },
        "llm_by_stage": dict(sorted(call_stages.items())),
        "limitations": "Nearest-rank sample percentiles, not an online SLO. Stage timings are inclusive and overlap; do not sum nested stages. Total service time includes session wait and memory, excludes monitoring persistence and HTTP transport. Tokens are provider-reported only; missing usage is not zero cost. Costs use the recorded repository rate table, not provider bills; cache discounts are not modeled. First request is listed separately; models were preloaded at service startup.",
    }

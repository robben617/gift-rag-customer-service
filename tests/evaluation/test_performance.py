import json
import sqlite3

from src.engineering.monitor import QueryMonitor, QueryRecord
from src.engineering.request_trace import request_trace
from src.evaluation.performance import attach_performance, latency_summary


def test_migrate_monitor_preserves_existing_records_and_trace(tmp_path):
    path = tmp_path / "monitor.db"
    monitor = QueryMonitor(path)
    monitor.record(QueryRecord(query_id="old", user_query="old"))
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE query_records DROP COLUMN performance_json")
    monitor = QueryMonitor(path)
    QueryMonitor(path)  # Idempotent schema initialization.
    with request_trace() as trace:
        data = trace.snapshot()
    data["estimated_cost_cny"] = 0
    monitor.record(QueryRecord(query_id=trace.trace_id, performance=data))
    stats = monitor.get_daily_stats()
    assert stats["usage_unknown_queries"] == stats["cost_unknown_queries"] == 1
    assert not stats["accounting_complete"]
    assert any(a["type"] == "accounting_incomplete" for a in monitor.check_alerts())
    report = {
        "details": [
            {"response": {"trace_id": trace.trace_id}, "elapsed_ms": 100},
            {"response": None, "elapsed_ms": 0},
        ]
    }
    attach_performance(report, path)
    assert (
        report["performance"]["observed_turns"]
        == report["performance"]["missing_turns"]
        == 1
    )
    assert not report["performance"]["complete"]
    assert report["performance"]["estimated_cost_cny"] is None
    assert report["performance"]["http_latency"]["p50_ms"] == 100
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT performance_json FROM query_records WHERE query_id='old'"
        ).fetchone() == ("{}",)


def test_small_sample_percentiles_and_no_data():
    assert latency_summary([])["p95_ms"] is None
    assert latency_summary([1, 2, 3, 4])["p50_ms"] == 2
    assert latency_summary([1, 2, 3, 4])["p95_ms"] == 4

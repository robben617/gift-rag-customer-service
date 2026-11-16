"""Storage and startup failures must never touch the application's existing state."""

import json
import sys

from scripts import run_live_rag_benchmark as runner


def test_child_environment_isolates_all_mutable_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-key")
    env = runner.isolated_environment(tmp_path, "temporary-signing-secret")
    for key in (
        "DATA_DIR",
        "LOG_DIR",
        "CHROMA_PERSIST_DIR",
        "WORKFLOW_CHECKPOINT_PATH",
        "WORKFLOW_REVIEW_PATH",
        "WORKFLOW_LOCK_DIR",
    ):
        from pathlib import Path

        assert Path(env[key]).is_relative_to(tmp_path)
    assert env["DEEPSEEK_API_KEY"] == "private-key"
    assert env["MYSQL_PORT"] == "9"
    assert env["WEB_SEARCH_ENABLED"] == "false"
    assert env["ENABLE_DEMO_USERS"] == "false"
    assert env["JWT_SECRET"] == "temporary-signing-secret"


def test_startup_failure_writes_failed_report_and_stops_child(monkeypatch, tmp_path):
    stopped = []

    class FailedProcess:
        def __init__(self, command, **kwargs):
            assert "private-key" not in " ".join(command)

        def poll(self):
            return 1

        def terminate(self):
            stopped.append(True)

        def wait(self, timeout):
            return 1

    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-key")
    monkeypatch.setattr(runner.subprocess, "Popen", FailedProcess)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["runner", "--run-dir", str(output)])
    assert runner.main() == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert "metrics" not in report
    assert "private-key" not in json.dumps(report)
    assert stopped == [True]

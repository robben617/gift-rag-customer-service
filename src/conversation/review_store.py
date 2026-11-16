"""Durable review tasks. Drafts are administrative data, never owner-visible."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings


class ReviewConflict(Exception):
    pass


class ReviewNotFound(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS review_tasks (
                task_id TEXT PRIMARY KEY, session_key TEXT NOT NULL,
                owner TEXT NOT NULL, session_id TEXT NOT NULL, access_level TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                decision TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '', user_query TEXT NOT NULL, draft TEXT NOT NULL,
                evidence_json TEXT NOT NULL, handoff_json TEXT NOT NULL,
                result_json TEXT, created_at TEXT NOT NULL, resolved_at TEXT, delivered_at TEXT,
                UNIQUE(session_key, checkpoint_id)
            )""")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS review_owner_session ON review_tasks(owner, session_id, created_at)"
            )

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, task_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise ReviewNotFound("审批任务不存在")
        return dict(row)

    @staticmethod
    def task_id_for(session_key, snapshot):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        return uuid.uuid5(
            uuid.NAMESPACE_URL, f"review:{session_key}:{checkpoint_id}"
        ).hex

    def _insert(self, conn, *, session_key, owner, session_id, snapshot):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        task_id = self.task_id_for(session_key, snapshot)
        state = snapshot.values
        conn.execute(
            """UPDATE review_tasks SET status='superseded'
            WHERE session_key=? AND status='pending' AND checkpoint_id<>?""",
            (session_key, checkpoint_id),
        )
        conn.execute(
            """INSERT OR IGNORE INTO review_tasks
            (task_id,session_key,owner,session_id,access_level,checkpoint_id,
             user_query,draft,evidence_json,handoff_json,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task_id,
                session_key,
                owner,
                session_id,
                state.get("access_level", "public"),
                checkpoint_id,
                state.get("query", ""),
                state.get("answer", ""),
                json.dumps(state.get("retrieved_docs", []), ensure_ascii=False),
                json.dumps(
                    {
                        "entities": state.get("entities", {}),
                        "memory_summary": state.get("memory_context", ""),
                        "emotion": state.get("emotion", "calm"),
                    },
                    ensure_ascii=False,
                ),
                _now(),
            ),
        )
        return task_id

    def create(self, **kwargs) -> dict:
        with self.connect() as conn:
            task_id = self._insert(conn, **kwargs)
        return self.get(task_id)

    def processing(self, session_key):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_tasks WHERE session_key=? AND status='processing' ORDER BY created_at LIMIT 1",
                (session_key,),
            ).fetchone()
        return dict(row) if row else None

    def begin_decision(
        self, task_id: str, decision: str, reviewer: str, note: str
    ) -> dict:
        # Caller holds the session OS lock. 'processing' is recoverable after crash.
        with self.connect() as conn:
            conn.execute(
                """UPDATE review_tasks SET status='processing',decision=?,reviewer=?,note=?
                WHERE task_id=? AND status='pending'""",
                (decision, reviewer, note, task_id),
            )
        row = self.get(task_id)
        if row["status"] not in ("processing", "resolved"):
            raise ReviewConflict("审批任务已过期")
        if row["decision"] != decision:
            raise ReviewConflict("该任务已有不同的审批决定")
        return row

    def complete(self, task_id: str, response: dict, successor=None):
        with self.connect() as conn:
            conn.execute(
                """UPDATE review_tasks SET status='resolved',result_json=?,resolved_at=?
                WHERE task_id=? AND status='processing'""",
                (json.dumps(response, ensure_ascii=False), _now(), task_id),
            )
            if successor:
                self._insert(conn, **successor)
        return self.get(task_id)

    def undelivered(self, session_key: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM review_tasks
                WHERE session_key=? AND status='resolved' AND delivered_at IS NULL
                ORDER BY created_at""",
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_delivered(self, task_id: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE review_tasks SET delivered_at=? WHERE task_id=? AND delivered_at IS NULL",
                (_now(), task_id),
            )

    def list(
        self, *, owner=None, session_id=None, status=None, limit=100
    ) -> list[dict]:
        clauses, values = [], []
        for column, value in (
            ("owner", owner),
            ("session_id", session_id),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_tasks"
                + where
                + " ORDER BY created_at DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def view(row: dict, *, admin=False) -> dict:
        result = {
            key: row[key]
            for key in (
                "task_id",
                "session_id",
                "status",
                "decision",
                "created_at",
                "resolved_at",
            )
        }
        result["result"] = (
            json.loads(row["result_json"]) if row["result_json"] else None
        )
        if admin:
            result.update(
                {
                    key: row[key]
                    for key in (
                        "owner",
                        "user_query",
                        "draft",
                        "reviewer",
                        "note",
                        "checkpoint_id",
                    )
                }
            )
            result["evidence"] = json.loads(row["evidence_json"])
            result["handoff"] = json.loads(row["handoff_json"])
        return result


def get_review_store() -> ReviewStore:
    return ReviewStore(settings.workflow_review_path)

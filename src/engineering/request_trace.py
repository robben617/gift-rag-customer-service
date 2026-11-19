"""Request-local timings and provider accounting; never store prompts or replies."""

import copy
import inspect
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock

TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
_trace = ContextVar("request_trace", default=None)
_stage = ContextVar("request_stage", default="service")
_call = ContextVar("request_llm_call", default="")
_attempt = ContextVar("request_llm_attempt", default=None)


@dataclass
class RequestTrace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    stages: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def snapshot(self):
        with self.lock:
            attempts = copy.deepcopy(self.attempts)
            calls = copy.deepcopy(self.calls)
            stages = copy.deepcopy(self.stages)
        totals = {
            key: sum(a["usage"].get(key, 0) for a in attempts) for key in TOKEN_KEYS
        }
        covered = sum(a["usage_complete"] for a in attempts)
        return {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "total_ms": round((time.perf_counter() - self.started) * 1000, 3),
            "llm_calls": len(calls),
            "llm_attempts": len(attempts),
            "llm_retries": len(attempts) - len({a["call_id"] for a in attempts}),
            "failed_attempts": sum(a["status"] != "ok" for a in attempts),
            "usage_reported_attempts": covered,
            "usage_missing_attempts": len(attempts) - covered,
            "usage_complete": covered == len(attempts),
            "usage": totals,
            "stages": stages,
            "calls": calls,
            "attempts": attempts,
        }


@contextmanager
def request_trace():
    trace = RequestTrace()
    tokens = (
        _trace.set(trace),
        _stage.set("service"),
        _call.set(""),
        _attempt.set(None),
    )
    try:
        yield trace
    finally:
        for var, token in zip((_trace, _stage, _call, _attempt), tokens):
            var.reset(token)


@contextmanager
def stage(name):
    trace = _trace.get()
    if trace is None:
        yield
        return
    token = _stage.set(name)
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        row = {
            "name": name,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "status": status,
        }
        with trace.lock:
            trace.stages.append(row)
        _stage.reset(token)


def timed_stage(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)

        return wrapped

    return decorate


@contextmanager
def _observe(model, kind):
    trace = _trace.get()
    if trace is None:
        yield
        return
    row = {"model": model, "stage": _stage.get(), "status": "ok"}
    if kind == "call":
        row["call_id"] = uuid.uuid4().hex
        token = _call.set(row["call_id"])
    else:
        row.update(call_id=_call.get(), usage={}, usage_complete=False)
        token = _attempt.set(row)
    started = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        row.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        with trace.lock:
            (trace.calls if kind == "call" else trace.attempts).append(row)
        (_call if kind == "call" else _attempt).reset(token)


def observe_llm(kind):
    """Separate logical calls (including breaker rejects) from HTTP attempts."""

    def decorate(fn):
        if inspect.isasyncgenfunction(fn):

            @wraps(fn)
            async def streaming(self, *args, **kwargs):
                with _observe(self.model, kind):
                    generator = fn(self, *args, **kwargs)
                    try:
                        async for value in generator:
                            yield value
                    finally:
                        await generator.aclose()

            return streaming
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def asynchronous(self, *args, **kwargs):
                with _observe(self.model, kind):
                    return await fn(self, *args, **kwargs)

            return asynchronous

        @wraps(fn)
        def synchronous(self, *args, **kwargs):
            with _observe(self.model, kind):
                return fn(self, *args, **kwargs)

        return synchronous

    return decorate


def record_provider_usage(usage):
    row = _attempt.get()
    if row is None:
        return
    usage = usage if isinstance(usage, dict) else {}
    valid = {
        key: usage[key]
        for key in TOKEN_KEYS
        if type(usage.get(key)) is int and usage[key] >= 0
    }
    # A streaming usage packet is a cumulative total, never add it per chunk.
    row["usage"] = valid
    row["usage_complete"] = len(valid) == len(TOKEN_KEYS)

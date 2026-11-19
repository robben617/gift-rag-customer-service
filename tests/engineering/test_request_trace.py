import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier

import httpx
import pytest

from src.engineering.circuit_breaker import CircuitBreaker
from src.engineering.llm_client import LLMClient, LLMClientError
from src.engineering.request_trace import request_trace, stage

USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


@pytest.fixture
def client(monkeypatch):
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60)
    monkeypatch.setattr(
        "src.engineering.circuit_breaker.get_llm_breaker", lambda: breaker
    )
    monkeypatch.setattr("src.engineering.llm_client.time.sleep", lambda _: None)
    instance = LLMClient(api_key="private-secret", base_url="https://example.invalid")
    yield instance
    instance.close_sync()


def success(usage=USAGE):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "private reply"}}], "usage": usage},
    )


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_nonretryable_failure_and_open_breaker_do_not_resend(client, status):
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(status))
    )
    with request_trace() as trace:
        for _ in range(2):
            with pytest.raises(LLMClientError):
                client.chat([{"role": "user", "content": "private prompt"}])
        result = trace.snapshot()
    assert result["llm_calls"] == 2
    assert result["llm_attempts"] == result["failed_attempts"] == 1
    assert result["llm_retries"] == 0
    assert not result["usage_complete"]
    assert result["usage_missing_attempts"] == 1
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("first", [429, 503, "timeout"])
def test_retry_is_one_call_two_attempts_and_missing_usage_is_visible(client, first):
    responses = iter([first, "success"])

    def send(req):
        value = next(responses)
        if value == "timeout":
            raise httpx.ReadTimeout("private provider error", request=req)
        return success() if value == "success" else httpx.Response(value)

    client._sync_client = httpx.Client(transport=httpx.MockTransport(send))
    with request_trace() as trace:
        with stage("generation.answer"):
            assert client.chat([]) == "private reply"
        result = trace.snapshot()
    assert (
        result["llm_calls"] == result["llm_retries"] == result["failed_attempts"] == 1
    )
    assert result["llm_attempts"] == 2
    assert result["usage"] == USAGE
    assert result["usage_reported_attempts"] == 1
    assert not result["usage_complete"]
    assert {a["stage"] for a in result["attempts"]} == {"generation.answer"}
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"total_tokens": "12"},
        {"prompt_tokens": True, "completion_tokens": -1},
    ],
)
def test_missing_or_malformed_provider_usage_is_not_invented(client, usage):
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(lambda req: success(usage))
    )
    with request_trace() as trace:
        client.chat([])
        result = trace.snapshot()
    assert result["usage"] == dict.fromkeys(USAGE, 0)
    assert result["usage_missing_attempts"] == 1


def test_copied_graph_context_shares_trace_but_other_requests_are_isolated(client):
    barrier = Barrier(2)

    def send(req):
        tokens = int(json.loads(req.content)["messages"][0]["content"])
        barrier.wait(timeout=5)
        return success(
            {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens}
        )

    client._sync_client = httpx.Client(transport=httpx.MockTransport(send))

    def request(tokens):
        with request_trace() as trace:
            with ThreadPoolExecutor(max_workers=1) as graph:
                graph.submit(
                    copy_context().run, client.chat, [{"content": str(tokens)}]
                ).result(timeout=10)
            return trace.snapshot()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, [12, 34]))
    assert [r["usage"]["total_tokens"] for r in results] == [12, 34]
    assert results[0]["trace_id"] != results[1]["trace_id"]
    with request_trace() as fresh:
        assert fresh.snapshot()["llm_calls"] == 0


@pytest.mark.asyncio
async def test_async_and_stream_usage_only_packets(client):
    def send(req):
        if json.loads(req.content).get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\ndata: [DONE]\n\n',
            )
        return success()

    client._async_client = httpx.AsyncClient(transport=httpx.MockTransport(send))
    try:
        with request_trace() as trace:
            await client.achat([])
            assert [text async for text in client.achat_stream([])] == ["hello"]
            result = trace.snapshot()
        assert result["llm_calls"] == result["llm_attempts"] == 2
        assert result["usage"]["total_tokens"] == 24
        assert result["usage_complete"]
        with request_trace() as trace:
            generator = client.achat_stream([])
            assert await anext(generator) == "hello"
            await generator.aclose()
            result = trace.snapshot()
        assert result["llm_attempts"] == result["failed_attempts"] == 1
        assert not result["usage_complete"]
    finally:
        await client.close()

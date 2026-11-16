"""llm_client usage 捕获测试 — token 消耗不再丢失（P1 修复）"""

from src.engineering.llm_client import LLMClient


def test_concurrent_requests_have_separate_usage():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    client = LLMClient()
    barrier = Barrier(2)

    def request(tokens):
        client.reset_usage()
        client._record_usage({"total_tokens": tokens})
        barrier.wait(timeout=3)
        return client.total_usage["total_tokens"], client.last_usage["total_tokens"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request, 10)
        second = pool.submit(request, 20)
        assert first.result(timeout=5) == (10, 10)
        assert second.result(timeout=5) == (20, 20)


def test_graph_copied_context_accumulates_into_parent_request():
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    client = LLMClient()
    client.reset_usage()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(
            copy_context().run, client._record_usage, {"total_tokens": 12}
        ).result(timeout=3)
    assert client.total_usage["total_tokens"] == 12
    snapshot = client.total_usage
    snapshot["total_tokens"] = 999
    assert client.total_usage["total_tokens"] == 12


def test_usage_accumulation_and_reset():
    client = LLMClient()
    client.reset_usage()
    client._record_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    client._record_usage(
        {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    )
    assert client.total_usage == {
        "prompt_tokens": 30,
        "completion_tokens": 13,
        "total_tokens": 43,
    }

    client.reset_usage()
    assert client.total_usage["total_tokens"] == 0


def test_empty_usage_no_crash():
    client = LLMClient()
    client.reset_usage()
    client._record_usage({})
    assert client.total_usage["total_tokens"] == 0
    assert client.last_usage == {}

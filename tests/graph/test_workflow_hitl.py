"""HITL（Human-in-the-Loop）中断与续接回归测试

为什么单独测这块:
  1. LangGraph 的静态中断（``interrupt_before``）**必须**配 checkpointer，
     否则 ``get_state()`` 抛 ``ValueError: No checkpointer set``、``update_state()`` 不可用，
     中断等于"停在那里没人能叫醒"。
  2. 静态中断**不会**在 ``invoke()`` 的返回值里带 ``__interrupt__`` 键
     （那个键只属于节点内 ``interrupt()`` 触发的动态中断），
     中断判定只能查 ``get_state(config).next``。少了这一步，
     评估不通过（高敏承诺且忠实度不足）的答案会被直接返回给用户，绕过承诺护栏。

测试策略: 用极小的两节点图复现同一套中断语义，
不加载 LLM / 向量库，保证毫秒级且无外部依赖。
"""

from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from src.graph.workflow import RAGWorkflow, _is_paused, build_checkpointer


class _Counter(TypedDict):
    x: int


def _inc(state: _Counter) -> dict:
    return {"x": state["x"] + 1}


def _tiny_graph() -> StateGraph:
    """a → b 的两节点图，用于在 b 前设静态断点"""
    graph = StateGraph(_Counter)
    graph.add_node("a", _inc)
    graph.add_node("b", _inc)
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


class TestCheckpointerWired:
    """生产图必须挂上 checkpointer"""

    def test_rag_workflow_has_checkpointer(self):
        assert RAGWorkflow().app.checkpointer is not None

    def test_rag_workflow_supports_update_state(self):
        assert hasattr(RAGWorkflow().app, "update_state")

    def test_build_checkpointer_returns_fresh_instance(self):
        assert build_checkpointer() is not build_checkpointer()


class TestIsPaused:
    """_is_paused 必须靠 get_state().next 判定静态中断"""

    def test_paused_at_interrupt_before(self):
        app = _tiny_graph().compile(
            checkpointer=build_checkpointer(), interrupt_before=["b"]
        )
        config = _config("hitl-pause")

        app.invoke({"x": 0}, config)

        assert _is_paused(app, config) is True

    def test_static_interrupt_has_no_marker_in_return_value(self):
        """回归护栏: 静态中断不带 __interrupt__ 键，所以不能靠它判断"""
        app = _tiny_graph().compile(
            checkpointer=build_checkpointer(), interrupt_before=["b"]
        )
        config = _config("hitl-no-marker")

        result = app.invoke({"x": 0}, config)

        assert isinstance(result, dict)
        assert "__interrupt__" not in result
        assert _is_paused(app, config) is True

    def test_not_paused_without_interrupt_before(self):
        app = _tiny_graph().compile(checkpointer=build_checkpointer())
        config = _config("hitl-nobreak")

        app.invoke({"x": 0}, config)

        assert _is_paused(app, config) is False

    def test_no_checkpointer_returns_false_instead_of_raising(self):
        """未挂 checkpointer 时 get_state 会抛异常，_is_paused 必须吞掉并返回 False"""
        app = _tiny_graph().compile(interrupt_before=["b"])

        assert _is_paused(app, _config("hitl-nocp")) is False

    def test_unknown_thread_returns_false(self):
        app = _tiny_graph().compile(
            checkpointer=build_checkpointer(), interrupt_before=["b"]
        )

        assert _is_paused(app, _config("hitl-never-run")) is False


class TestResumeFlow:
    """update_state 注入决策后可续接，且续接完成不再处于中断态"""

    def test_update_state_then_resume_completes(self):
        app = _tiny_graph().compile(
            checkpointer=build_checkpointer(), interrupt_before=["b"]
        )
        config = _config("hitl-resume")

        app.invoke({"x": 0}, config)
        assert _is_paused(app, config) is True

        app.update_state(config, {"x": 99})
        resumed = app.invoke(None, config)

        assert resumed["x"] == 100
        assert _is_paused(app, config) is False

    def test_different_thread_id_loses_snapshot(self):
        """续接必须复用同一 thread_id，否则找不到快照（隔离性护栏）"""
        app = _tiny_graph().compile(
            checkpointer=build_checkpointer(), interrupt_before=["b"]
        )
        app.invoke({"x": 0}, _config("hitl-session-a"))

        assert _is_paused(app, _config("hitl-session-b")) is False

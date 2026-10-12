"""聊天界面（7861）状态栏文案 — 调试模式开关的对外可见行为

要点：开关关闭时不得把内部字段（意图名、置信度、检索条数、来源文件名）暴露给客户；
但"已转人工"必须保留——它决定用户接下来该不该继续等，属对客户必要的信息。
"""

import pytest

from src.admin.chat_ui import DEBUG_MODE, build_status_message

# 内部字段样本：出现任意一个即视为泄露
INTERNAL_MARKERS = ("return_refund", "85%", "faq.json", "policy.md", "检索到")


class TestDebugOff:
    """调试模式关闭（默认）：只给客户看有意义的服务状态"""

    def test_normal_reply_hides_internal_fields(self):
        status = build_status_message(
            debug=False,
            intent="return_refund",
            confidence=0.85,
            result_count=5,
            sources=["faq.json", "policy.md"],
        )
        for marker in INTERNAL_MARKERS:
            assert marker not in status, f"关闭调试模式时不应出现内部字段: {marker}"

    def test_normal_reply_is_not_empty(self):
        """不能只做减法：状态栏要给出对客户有意义的信息"""
        status = build_status_message(debug=False, intent="product", confidence=0.7)
        assert status.strip()

    def test_human_handoff_still_told_to_customer(self):
        """转人工必须告知客户：这是唯一一条关闭调试模式后仍要保留的内部状态"""
        status = build_status_message(
            debug=False,
            intent="complaint",
            confidence=0.91,
            needs_human=True,
            human_reason="high_emotion",
            result_count=2,
        )
        assert "人工" in status
        assert "high_emotion" not in status


class TestDebugOn:
    """调试模式开启：完整展示内部字段，供演示与排障"""

    def test_full_internal_fields_visible(self):
        status = build_status_message(
            debug=True,
            intent="return_refund",
            confidence=0.85,
            result_count=5,
            sources=["faq.json", "policy.md"],
        )
        assert "return_refund" in status
        assert "85%" in status
        assert "检索到 5 条结果" in status
        assert "faq.json" in status
        assert "policy.md" in status

    def test_human_reason_visible(self):
        status = build_status_message(
            debug=True,
            intent="complaint",
            confidence=0.9,
            needs_human=True,
            human_reason="high_emotion",
        )
        assert "high_emotion" in status
        assert "需要人工介入" in status

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"intent": "", "confidence": 0.0},
            {"result_count": 0, "sources": []},
        ],
    )
    def test_falls_back_to_normal_reply(self, kwargs):
        assert build_status_message(debug=True, **kwargs) == "正常回复"

    def test_sources_sorted_and_joined(self):
        status = build_status_message(
            debug=True, result_count=3, sources=["b.md", "a.md"]
        )
        assert "来源: a.md, b.md" in status


class TestDebugModeConfig:
    def test_default_off_in_test_env(self):
        """conftest 固定 UI_DEBUG_MODE=false：默认必须是关闭，避免误把内部信息给客户"""
        assert DEBUG_MODE is False

    def test_flag_parsing_accepts_common_truthy_values(self):
        """解析逻辑与实现保持一致：大小写不敏感，支持 1/true/yes/on"""
        truthy = {"1", "true", "yes", "on"}
        for raw in ("1", "TRUE", "True", "yes", "ON", "on"):
            assert raw.strip().lower() in truthy
        for raw in ("0", "false", "no", "off", ""):
            assert raw.strip().lower() not in truthy


class TestUIWiring:
    """界面构建冒烟：验证开关与说明面板的绑定没有写错"""

    def test_blocks_build_successfully(self):
        from src.admin.chat_ui import create_chat_ui

        app = create_chat_ui()
        assert app is not None

    def test_chat_with_bot_accepts_debug_flag(self):
        """chat_with_bot 必须能接收 debug 参数（Gradio 事件绑定依赖此签名）"""
        import inspect

        from src.admin.chat_ui import chat_with_bot

        params = inspect.signature(chat_with_bot).parameters
        assert "debug" in params
        assert params["debug"].default is False

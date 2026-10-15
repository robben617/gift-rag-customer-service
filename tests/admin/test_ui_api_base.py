"""UI 的 API 地址必须与鉴权模块同源，且能读到 .env 配置。

回归背景（真实缺陷）：
    chat_ui.py / admin_ui.py 曾在模块级用
        API_BASE_URL = os.getenv("API_BASE_URL", "http://rag-api:8000")
    取值，而唯一会执行 load_dotenv() 的 api_auth 只在函数体内被延迟导入。
    读取早于 .env 注入 → 本地运行时两个 UI 都指向 Docker 服务名
    http://rag-api:8000，界面所有请求失败（聊天页只显示「系统暂时繁忙」）。
    这与 .env.example 中「本地取消注释 API_BASE_URL」的指引直接矛盾。

修法：两个 UI 改为从 src.admin.api_auth 导入 API_BASE_URL，保证
（1）先完成 .env 注入，（2）UI 与登录鉴权使用同一个地址，不会各说各话。
"""

from src.admin import admin_ui, api_auth, chat_ui


def test_chat_ui_shares_api_base_url_with_auth() -> None:
    assert chat_ui.API_BASE_URL == api_auth.API_BASE_URL


def test_admin_ui_shares_api_base_url_with_auth() -> None:
    assert admin_ui.API_BASE_URL == api_auth.API_BASE_URL


def test_api_base_url_is_absolute_http_url() -> None:
    """必须是带主机与端口的绝对地址，否则 requests 会抛 MissingSchema"""
    url = api_auth.API_BASE_URL
    assert url.startswith(("http://", "https://")), url
    host_port = url.split("://", 1)[1]
    assert "/" not in host_port, f"不应带路径: {url}"

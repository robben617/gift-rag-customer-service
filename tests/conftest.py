"""pytest 全局配置 — 在任何 src 导入前设置测试专用密钥

src.config 的 settings 是模块级单例，且 jwt_secret 默认空串（空密钥会被应用启动校验拒绝）。
测试必须提供一个稳定的非空 secret，否则 create_access_token/decode_token 会因空 HMAC key 抛错。
"""

import os

os.environ.setdefault("JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef")
os.environ.setdefault("ENABLE_DEMO_USERS", "true")
# 聊天界面调试模式：测试环境强制关闭，保证状态栏文案断言确定（不受本地 .env 影响）
os.environ.setdefault("UI_DEBUG_MODE", "false")

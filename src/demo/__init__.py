"""零依赖离线演示包

只提供一条「不加载模型、不连外部服务」的最短问答链路，用于让访客克隆仓库后
立刻看到问答形态。**与生产链路完全隔离**：生产代码不会 import 本包。
"""

from .offline_retrieval import (
    DEFAULT_FAQ_PATH,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    DemoAnswer,
    FAQHit,
    OfflineFAQIndex,
)

__all__ = [
    "DEFAULT_FAQ_PATH",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_TOP_K",
    "DemoAnswer",
    "FAQHit",
    "OfflineFAQIndex",
]

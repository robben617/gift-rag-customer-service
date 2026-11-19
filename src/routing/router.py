"""模块4 路由分发器 — 意图 → 处理器

使用方式:
    router = IntentRouter()
    result = router.route("怎么退货？")
    print(result.target)       # RouteTarget.RAG
    print(result.intent_result.intent)  # Intent.RETURN_REFUND
"""

import logging
import re
from typing import Optional

from src.business.after_sales import after_sales_kind, order_ids

from .classifier import IntentClassifier, get_classifier
from .models import Intent, IntentResult, RouteResult, RouteTarget

logger = logging.getLogger(__name__)

# ── 意图 → 路由目标映射 ──

INTENT_ROUTE_MAP: dict[Intent, RouteTarget] = {
    Intent.RETURN_REFUND: RouteTarget.RAG,  # 售后问题 → 知识库检索
    Intent.PRODUCT_CONSULT: RouteTarget.RAG,  # 商品咨询 → 知识库检索
    Intent.LOGISTICS: RouteTarget.RAG,  # 物流查询 → 知识库检索（配送政策/时效说明）
    Intent.ORDER_QUERY: RouteTarget.RAG,  # 订单查询 → 知识库检索（订单相关问题）
    Intent.COMPLAINT: RouteTarget.RAG,  # 投诉 → 先检索给出处理说明，同时标记 needs_human
    Intent.CHITCHAT: RouteTarget.DIRECT,  # 闲聊 → 直接回复
}

# ── 工具路由判定 ──


def _has_tool_entity(entities: dict) -> bool:
    """是否包含可触发工具调用的实体（订单号/快递单号）

    订单/物流是实时数据，检测到单号 → 路由到 SQL 工具节点查询真实状态，
    避免塞进向量库返回过期数据。
    """
    return bool(entities.get("order_id") or entities.get("tracking_number"))


# 指代词标记（多轮追问：那个/上次/它/那…）
_COREFERENCE_MARKERS = ["那个", "这个", "它", "上次", "之前", "刚才", "还有", "别的"]


def _has_coreference(query: str) -> bool:
    """检测查询是否含指代词（多轮追问信号）

    用于: 分类器误判为 chitchat 时，强制改走 RAG，让记忆/检索生效。
    例: "上次那个券怎么用" "那个能退吗" "它支持快充吗"
    """
    return any(m in query for m in _COREFERENCE_MARKERS)


class IntentRouter:
    """意图路由器 — 分类 + 路由 + 查询改写

    使用方式:
        router = IntentRouter()
        result = router.route("怎么退货？")
    """

    def __init__(self, classifier: Optional[IntentClassifier] = None):
        self.classifier = classifier or get_classifier()

    def route(self, query: str) -> RouteResult:
        """Apply business routing rules to one classification and rewrite result."""

        # ── 步骤1: 意图分类 ──
        after_sales = after_sales_kind(query)
        if after_sales != "none":
            ids = order_ids(query)
            return RouteResult(
                intent_result=IntentResult(
                    intent=Intent.RETURN_REFUND,
                    confidence=0.95,
                    reasoning=f"售后规则分流: {after_sales}",
                    entities={"order_id": ids[0]} if len(ids) == 1 else {},
                ),
                target=RouteTarget.RAG,
                query=query,
                rewritten_query=query,
                handler="rag",
            )
        intent_result = self.classifier.classify(query)
        logger.info(
            "意图分类: %s (%.2f) — %s",
            intent_result.intent.value,
            intent_result.confidence,
            intent_result.reasoning,
        )

        # 指代词兜底: 分类为 chitchat 但含指代词（那个/上次/它）→ 强制走 RAG
        # 场景: "上次那个券怎么用" "那个能退吗" —— LLM 分类成功但误判 chitchat
        # 改走 RAG 让多轮记忆/检索生效，避免直接闲聊回复
        business_question = bool(
            re.search(r"门店|实体店|营业时间|几点开门|几点关门|打烊|电子发票", query)
        )
        if intent_result.intent == Intent.CHITCHAT and (
            _has_coreference(query) or business_question
        ):
            intent_result = IntentResult(
                intent=Intent.PRODUCT_CONSULT,
                confidence=0.5,
                reasoning="商家信息需核验证据"
                if business_question
                else "指代词/追问，改走知识库检索",
                entities=intent_result.entities,
            )
            logger.info("指代词兜底: %s → product_consult", query[:30])

        # ── 步骤2: 路由决策 ──
        target = INTENT_ROUTE_MAP.get(intent_result.intent, RouteTarget.RAG)

        # 改进1: 订单/物流意图 + 检测到订单号/快递单号 → 工具路由（实时数据走工具，不塞向量库）
        if intent_result.intent in (
            Intent.ORDER_QUERY,
            Intent.LOGISTICS,
        ) and _has_tool_entity(intent_result.entities):
            target = RouteTarget.SQL
            logger.info("检测到订单/快递单号 → 工具路由 (SQL)")

        # 低置信度时降级到 RAG（宁可多检索，不要漏）
        if intent_result.confidence < 0.4 and target == RouteTarget.SQL:
            target = RouteTarget.RAG
            logger.info("低置信度 SQL 路由，降级为 RAG")

        # ── 步骤3: 查询改写（仅 RAG 路由）──
        rewritten = query
        if target in (RouteTarget.RAG, RouteTarget.HYBRID):
            rewritten = intent_result.rewritten_query or query

        return RouteResult(
            intent_result=intent_result,
            target=target,
            query=query,
            rewritten_query=rewritten,
            handler=target.value,
        )


# ── 模块级单例 ──

from src.engineering.singleton import singleton_factory


@singleton_factory
def get_router() -> IntentRouter:
    return IntentRouter()

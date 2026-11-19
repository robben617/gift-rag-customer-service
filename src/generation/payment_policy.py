"""Exact-source reply for the approved cash-on-delivery FAQ."""

import re
from dataclasses import dataclass

_PAYMENT_METHOD = (
    r"(?:货到付款|(?:到货(?:后)?|收货(?:后)?|收到货(?:后)?|"
    r"(?:东西|商品|货)到了我手上)再(?:付款|付钱|给钱))"
)
_POLICY_QUESTION = re.compile(
    r"(?:换个(?:话题|问题)[，,])?(?:请问[，,]?)?"
    r"(?:本平台|平台|你们(?:平台)?|这(?:款|件|个)(?:商品|礼盒|东西)?|礼盒)?"
    r"(?:是否|能不能|可不可以|能否|可以|能|可)?(?:支持)?"
    + _PAYMENT_METHOD
    + r"(?:行不行|可以吗|吗|么|不|呀|啊|呢|嘛)*[？?。！!]*"
)

# The complete approved FAQ body is deliberately pinned. A revision with new
# conditions or exceptions must go through review before using this reply path.
_APPROVED_ANSWER = "本平台不支持货到付款，所有订单需在线支付。支付成功后安排发货。"
_APPROVED_FAQ = re.compile(
    r"## Q[0-9]+[:：] 支持货到付款吗？\n\n"
    + re.escape(_APPROVED_ANSWER)
    + r"(?:\n\n\*关键词: 货到付款, 在线支付, 不支持\*)?"
)


@dataclass(frozen=True)
class PaymentPolicyQuote:
    doc_id: str
    index: int
    text: str


def match_payment_policy(
    raw_query: str, resolved_query: str, docs: list[dict]
) -> PaymentPolicyQuote | None:
    """Match a complete request and one approved FAQ in the verified retrieval window."""
    if not all(
        _POLICY_QUESTION.fullmatch(re.sub(r"\s+", "", query))
        for query in (raw_query, resolved_query)
    ):
        return None
    matched = None
    for index, doc in enumerate(docs[:5], start=1):
        text = doc["text"][:800]
        if not re.search(r"货到付款|到货再付|\bCOD\b", text, re.I):
            continue
        # More than one relevant document, or a changed/partial one, leaves scope
        # resolution to the ordinary verification path.
        if (
            matched is not None
            or doc.get("doc_type") != "faq_json"
            or not doc.get("chunk_id")
            or len(doc["text"]) > 800
            or not _APPROVED_FAQ.fullmatch(text.replace("\r\n", "\n").strip())
        ):
            return None
        matched = PaymentPolicyQuote(
            doc_id=doc["chunk_id"], index=index, text=_APPROVED_ANSWER
        )
    return matched

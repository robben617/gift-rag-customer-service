"""Conservative after-sales speech acts, shared by routing and human gates."""

import re
from typing import Literal

AfterSalesKind = Literal["policy", "refund_action", "refund_status", "unclear", "none"]


def order_ids(query: str) -> list[str]:
    return list(
        dict.fromkeys(
            (match.group(1) or match.group(0)).upper()
            for match in re.finditer(
                r"(?<![A-Za-z0-9])ORD[0-9]{6,29}(?![A-Za-z0-9])|订单号[为是：:\s]*([A-Za-z0-9-]{6,32})",
                query,
                re.I,
            )
        )
    )


def after_sales_kind(query: str, context: dict | None = None) -> AfterSalesKind:
    text = re.sub(r"\s+", "", query)
    if not re.search(
        r"退(?:款|钱|货|掉|一下)|(?:能不能|能|可以|可不可以)退(?:吗|么|货|款|钱|[？?]|$)|换货|售后|无理由|(?:帮我|给我).{0,10}退了",
        text,
    ):
        if context and re.search(r"(?:帮我|给我|替我)(?:办理|处理|申请|办一下)", text):
            return "refund_action"
        if (context or {}).get("pending_scenario") == "request_type":
            if re.search(r"帮我|办理|我要申请", text):
                return "refund_action"
            if re.search(r"进度|没到账", text):
                return "refund_status"
            if re.search(r"了解|规则|咨询|流程", text):
                return "policy"
        return "none"
    # Status questions need payment records; a policy answer cannot verify them.
    if re.search(
        r"退款失败|退款被拒|拒绝退款|退款进度|退款状态|还没退款|尚未退款|没收到退款|(?:退款|退钱).*(?:还没|未到|没到|不到|查一下|查询)|(?:还没|未|没)(?:收)?到(?:账|钱).*退款|(?:退货|退款).{0,8}\d+天.{0,5}(?:没|未).*(?:钱|账)",
        text,
    ):
        return "refund_status"
    # Ignore explicitly negated requests; do not let '不要解释，直接退款' slip through.
    positive = re.sub(
        r"(?:不想|不要|不用|不需要|别)(?:帮我)?(?:申请|办理)?(?:退款|退钱|退货|换货)",
        "",
        text,
    )
    operation = r"(?:退(?:款|钱|货|掉|了|一下)|换货|取消退款|撤销退款)"
    informational_help = re.search(
        r"(?:帮我|替我|给我)(?:查|看|了解|确认|解释|说|讲|咨询).{0,16}(?:规则|政策|流程|条件|多久|几天|能不能|能否|如何|怎么)",
        positive,
    )
    later_action = re.search(
        r"(?:再|然后|顺便|同时).{0,10}(?:帮我|给我|直接|申请|办理)", positive
    )
    if informational_help and not later_action:
        return "policy"
    if re.search(
        rf"(?:帮我|替我|给我|直接|马上|立即|现在就).{{0,12}}{operation}", positive
    ):
        return "refund_action"
    # Questions about applying are informational, even though they contain 申请.
    if re.search(
        r"怎么|如何|怎样|哪里|哪儿|多少|流程|规则|政策|规定|条件|时限|期限|多久|几天|什么时候|是否|能否|能不能|可不可以|可以.*吗|能.*吗|想了解|想知道|想问",
        text,
    ):
        return "policy"
    if re.search(
        r"(?:我要|我想|我需要|请|申请|办理|要求).{0,5}(?:退款|退钱|退货|换货)|^(?:退款|退钱|退货|换货)[!！。]*$",
        positive,
    ):
        return (
            "unclear"
            if text.rstrip("!！。") in ("退款", "退钱", "退货", "换货")
            else "refund_action"
        )
    return (
        "policy" if re.search(r"运费|到账|签收|质量|破损|不喜欢", text) else "unclear"
    )


def food_safety_incident(query: str) -> bool:
    return bool(
        re.search(
            r"食物中毒|食品安全事故|(?:吃|喝).{0,12}(?:拉肚子|呕吐|过敏|不舒服)|(?:我|孩子|老人).{0,5}(?:已经|出现|发生)?过敏",
            query,
        )
    )

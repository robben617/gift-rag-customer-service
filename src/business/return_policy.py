"""Versioned demo return rules with explicit applicability and missing facts.

No money operation is performed. User statements stay provisional; order facts
come only from an ownership-scoped read. No category is invented from an LLM.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from src.business.after_sales import after_sales_kind, order_ids
from src.config import PROJECT_ROOT

POLICY_PATH = PROJECT_ROOT / "data/policies/returns.json"
CATEGORY_WORDS = {
    "fresh": ["生鲜", "鲜肉", "水果", "海鲜"],
    "custom": ["定制", "刻字"],
    "intimate": ["内衣", "内裤", "袜子", "贴身衣物"],
    "electronics": ["数码", "耳机", "手机", "电子产品"],
    "cosmetics": ["化妆品", "口红", "护肤品"],
    "ordinary": ["普通商品", "普通衣服", "衣服", "外套", "衬衫", "普通日用品"],
}
SCENARIOS = {
    "quality": [
        "质量问题",
        "瑕疵",
        "坏了",
        "破损",
        "开裂",
        "掉色",
        "发错",
        "漏发",
        "换货",
    ],
    "freight": ["运费", "邮费", "谁出", "谁承担", "运费险"],
    "refund_time": [
        "几天到账",
        "多久到账",
        "多久到",
        "什么时候到账",
        "退款时间",
        "何时退",
        "多久能收到退款",
    ],
    "window": [
        "几天内",
        "退货期限",
        "多久能退",
        "多久内",
        "退货时限",
        "还能退",
        "能不能退",
        "能退吗",
        "可不可以退",
        "无理由",
        "7天",
        "七天",
    ],
    "process": [
        "怎么退",
        "如何退",
        "退货流程",
        "如何退款",
        "怎么退款",
        "怎样退款",
        "怎样申请退款",
        "申请退货",
        "申请退款",
        "申请换货",
    ],
}
FIELD_LABELS = {
    "category": "商品品类",
    "signed_on": "签收日期或签收天数",
    "reason": "退货原因（质量问题或无理由）",
    "unused": "商品是否使用过",
    "opened": "商品是否拆封",
    "order_id": "需要核对的订单号",
    "request_type": "咨询规则还是办理申请",
}


class ReturnRules(BaseModel):
    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    effective_from: date
    effective_to: date | None = None
    notice: str
    return_days: int = Field(gt=0, le=365)
    quality_days: int = Field(gt=0, le=365)
    refund_workdays_min: int = Field(gt=0, le=90)
    refund_workdays_max: int = Field(gt=0, le=90)
    excluded_categories: list[str]
    sealed_categories: list[str]

    @model_validator(mode="after")
    def valid_ranges(self):
        if self.effective_to and self.effective_to < self.effective_from:
            raise ValueError("Invalid policy effective window")
        if self.refund_workdays_min > self.refund_workdays_max:
            raise ValueError("Invalid refund working-day range")
        if (
            set(self.excluded_categories + self.sealed_categories)
            - CATEGORY_WORDS.keys()
        ):
            raise ValueError("Unknown category in policy")
        return self


@dataclass
class PolicyDecision:
    reply: str = ""
    found: bool = True
    status: str = "informational"
    needs_human: bool = False
    missing_fields: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    version: str = ""


def today_shanghai() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def is_policy_followup(query: str, context: dict) -> bool:
    if not context:
        return False
    if (
        context.get("pending_scenario") == "request_type"
        and after_sales_kind(query, context) != "none"
    ):
        return True
    if after_sales_kind(query) == "none" and re.search(
        r"什么|支持|怎么|如何|颜色|尺寸|规格", query
    ):
        return False
    if after_sales_kind(query) in ("refund_action", "refund_status"):
        return False
    if re.search(
        r"另一个|另一单|另一件|换个话题|不是这单|材质|多少钱|价格|售价|物流到哪|优惠券|配料|保质期",
        query,
    ):
        return False
    if re.search(r"一般|通常|政策|规则|流程|怎么申请", query) and not re.search(
        r"那|这单|这件|这个", query
    ):
        return False
    facts = r"签收|收到.{0,6}(?:天|周)|未使用|没用过|使用过|未拆封|没拆封|已拆封|无质量问题|不喜欢|普通商品|衣服|生鲜|定制|数码|化妆品"
    return bool(
        re.search(facts, query)
        or (
            len(query) < 32
            and any(
                word in query for words in CATEGORY_WORDS.values() for word in words
            )
        )
        or query.strip("，。 ") == "手机壳"
        or order_ids(query)
        or (
            context.get("pending_scenario")
            and re.fullmatch(r"[0-9一二两三四五六七八九十]+天[。！!]?", query)
        )
        or re.search(r"^那.{0,8}(?:运费|能退|退款|退货)", query)
    )


def extract_facts(query: str, previous: dict, today: date) -> dict:
    context = dict(previous)
    ids = order_ids(query)
    if len(ids) == 1 and context.get("order_id") not in (None, ids[0]):
        context = {"pending_scenario": previous.get("pending_scenario", "window")}
    if ids:
        context["order_id"] = ids[0] if len(ids) == 1 else ""
        context["order_conflict"] = len(ids) > 1
    # Negated labels must not turn a stated ordinary product into fresh goods.
    category_text = query.replace("手机壳", "普通商品")
    for words in CATEGORY_WORDS.values():
        for word in words:
            category_text = category_text.replace(f"不是{word}", "")
    categories = {
        category
        for category, words in CATEGORY_WORDS.items()
        if any(word in category_text for word in words)
    }
    if categories:
        context["category"] = next(iter(categories)) if len(categories) == 1 else None
    if re.search(
        r"没有质量问题|无质量问题|不是质量问题|不喜欢|买错|不合适|无理由", query
    ):
        context["reason"] = "non_quality"
    elif re.search(r"质量问题|瑕疵|坏了|破损|发错|漏发", query):
        context["reason"] = "quality"
    physical = (
        query.replace("不是没用过", "使用过")
        .replace("并非未使用", "使用过")
        .replace("不是未拆封", "已拆封")
    )
    unopened = bool(re.search(r"未拆封|没拆封|没有拆封|未开封", physical))
    opened = bool(
        re.search(
            r"已拆封|已经拆封|拆开了|已经开封|拆过",
            re.sub(r"未拆封|没拆封|没有拆封|未开封", "", physical),
        )
    )
    if unopened or opened:
        context["opened"] = None if unopened and opened else opened
    unused = (unopened and not opened) or bool(
        re.search(r"没用过|未使用|没有使用|全新", physical)
    )
    used = bool(
        re.search(
            r"用过|使用过|已经使用|用了",
            re.sub(r"没用过|未使用|没有使用|全新", "", physical),
        )
    )
    if unused or used:
        context["unused"] = None if unused and used else unused
    signed = re.search(r"(?:签收|收到|收货).{0,5}?(\d{4}-\d{2}-\d{2})", query)
    elapsed = re.search(
        r"(?:签收|收到|收货)(?:了|后|已经|已|才)?\s*(\d{1,3}|一|两|二|三|四|五|六|七|八|九|十)\s*(?:天|日)",
        query,
    )
    if not elapsed and previous.get("pending_scenario"):
        elapsed = re.fullmatch(
            r"(\d{1,3}|一|两|二|三|四|五|六|七|八|九|十)天[。！!]?", query
        )
    try:
        if context.get("signed_source") == "order_record":
            pass  # user claims must not overwrite ownership-checked order dates
        elif signed:
            context["signed_on"] = date.fromisoformat(signed.group(1)).isoformat()
            context["signed_source"] = "user_statement"
        elif elapsed:
            digits = {
                "一": 1,
                "二": 2,
                "两": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
                "十": 10,
            }
            value = elapsed.group(1)
            days = int(value) if value.isdigit() else digits[value]
            context["signed_on"] = (today - timedelta(days=days)).isoformat()
            context["signed_source"] = "user_statement"
    except ValueError:
        context["signed_on"] = None
    return context


class ReturnPolicyService:
    def __init__(self, policy_path: Path = POLICY_PATH):
        self.policy_path = policy_path

    def detect_sub_scenario(self, query: str) -> list[str]:
        hits = [
            key
            for key, words in SCENARIOS.items()
            if any(word in query for word in words)
        ]
        if (
            re.search(
                r"退款.*(?:多久|几天|什么时候|何时)|退货.*(?:多久|几天).*退款", query
            )
            and "refund_time" not in hits
        ):
            hits.append("refund_time")
        if re.search(r"退款.*(?:退到哪|原路|渠道)|退到哪里|退款退回", query):
            hits.append("refund_channel")
        return hits

    def _lines(self, rules: ReturnRules) -> dict[str, list[str]]:
        labels = {
            "fresh": "生鲜",
            "custom": "定制",
            "intimate": "贴身衣物",
            "electronics": "数码",
            "cosmetics": "化妆品",
            "ordinary": "普通商品",
        }
        excluded = (
            "、".join(labels[item] for item in rules.excluded_categories)
            or "未列出额外排除品类"
        )
        sealed = (
            "、".join(labels[item] for item in rules.sealed_categories)
            or "未列出额外拆封限制"
        )
        return {
            "window": [
                f"普通商品按{rules.return_days}天无理由退货规则申请，需未使用、不影响二次销售；质量问题申请窗口为签收后{rules.quality_days}天。",
                f"本版本的无理由排除品类：{excluded}；需核对拆封情况的品类：{sealed}。具体订单仍需核验。",
            ],
            "freight": [
                "质量问题需核验后由商家承担运费；非质量问题由买家承担退货运费。运费险是否赔付及金额以保险审核为准。"
            ],
            "refund_time": [
                f"退货商品经仓库验收合格后，按规则预计{rules.refund_workdays_min}-{rules.refund_workdays_max}个工作日原路退款。实际进度需核对退款记录，不代表这笔订单已退款或承诺具体到账时刻。"
            ],
            "refund_channel": [
                "通用退款规则为退回原支付渠道，具体账户以原支付记录为准；更换收款账户或核对实际退款渠道需人工处理。"
            ],
            "quality": [
                f"普通商品质量问题请拍照或录视频保留证据，在订单详情申请售后；通用申请窗口为签收后{rules.quality_days}天，特殊品类需另核政策，审核后确认处理方式。"
            ],
            "process": [
                "退货流程：订单详情 → 申请售后 → 选择退货/换货 → 填写原因 → 提交审核，审核通过后按提示寄回。说明流程不代表已代您提交申请。"
            ],
        }

    def evaluate(
        self, query: str, context: dict | None = None, *, as_of: date | None = None
    ) -> PolicyDecision:
        today = as_of or today_shanghai()
        previous = context or {}
        facts = extract_facts(query, previous, today)
        kind = after_sales_kind(query, previous)
        if kind in ("refund_action", "refund_status"):
            return PolicyDecision(
                "实际办理或退款进度需要人工核对，当前未执行退款或资金变更。",
                status="manual_review",
                needs_human=True,
            )
        if kind == "unclear" and not previous:
            return PolicyDecision(
                "您想了解退货退款规则，还是希望办理申请？如果是查询已申请的退款进度，也请说明。",
                status="clarify",
                missing_fields=["request_type"],
                context={"pending_scenario": "request_type"},
            )
        scenarios = self.detect_sub_scenario(query)
        if not scenarios and previous:
            scenario = previous.get("pending_scenario") or previous.get(
                "scenario", "window"
            )
            scenarios = [scenario if scenario in SCENARIOS else "window"]
        if not scenarios and kind in ("policy", "unclear"):
            if re.search(r"可以退|能否退|是否能退|退不退", query):
                scenarios = ["window"]
            elif re.search(r"规则|政策", query):
                scenarios = ["window", "process", "refund_time"]
            else:
                return PolicyDecision(
                    "当前通用规则未覆盖这项具体售后条件，需要人工核对后再确认。",
                    status="manual_review",
                    needs_human=True,
                )
        if not scenarios:
            return PolicyDecision(found=False, context=facts)
        try:
            rules = ReturnRules.model_validate_json(
                self.policy_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return PolicyDecision(
                "当前无法核验适用的退货政策，请人工核对后再处理。",
                status="manual_review",
                needs_human=True,
                context=facts,
            )
        decision = PolicyDecision(context=facts, version=rules.version)
        facts.pop("pending_scenario", None)
        facts["scenario"] = "window" if "window" in scenarios else scenarios[0]
        if facts.get("category") == "fresh" and facts.get("reason") == "quality":
            decision.reply = "生鲜质量问题需结合商品批次、保质期和实际情况人工核验，不能套用普通商品的通用窗口。"
            decision.status, decision.needs_human = "manual_review", True
            return decision
        if today < rules.effective_from or (
            rules.effective_to and today > rules.effective_to
        ):
            decision.reply, decision.status, decision.needs_human = (
                "当前没有可确认有效的政策版本，请人工核验。",
                "manual_review",
                True,
            )
            return decision
        if re.search(
            r"活动|特价|专属政策|特殊政策|手续费|补差价|折旧|页面.*(?:不能退|不退)|仅退款|部分退款|取消(?:退款|退货)|(?:退款|退货).{0,5}取消",
            query,
        ):
            decision.reply, decision.status, decision.needs_human = (
                "这涉及商品或活动专属条款，不能直接套用通用退货规则，请人工核对具体订单。",
                "manual_review",
                True,
            )
            return decision
        individual = bool(
            previous.get("pending_scenario") not in (None, "request_type")
            or order_ids(query)
            or re.search(
                r"我这|这单|这件|这个|这盒|我买|能退吗|还能退|可不可以退|签收.{0,6}(?:天|\d{4}-)",
                query,
            )
        )
        if not individual or not any(s in scenarios for s in ("window", "quality")):
            decision.reply = "\n".join(
                line
                for scenario in scenarios
                for line in self._lines(rules).get(scenario, [])
            )
            return decision
        if facts.get("order_conflict"):
            return self._clarify(decision, ["order_id"])
        if "order_status" in facts and facts["order_status"] != "已签收":
            decision.reply = "订单记录不是待判断的已签收退货状态，不能据此确认可退；如需取消订单、重复退款或其他处理，请人工核验。"
            decision.status, decision.needs_human = "manual_review", True
            return decision
        if facts.get("order_created_on"):
            created = date.fromisoformat(facts["order_created_on"])
            if created < rules.effective_from or (
                rules.effective_to and created > rules.effective_to
            ):
                decision.reply, decision.status, decision.needs_human = (
                    "未找到适用于该订单下单时间的政策版本，请人工核对。",
                    "manual_review",
                    True,
                )
                return decision
        missing = [name for name in ("category", "reason") if not facts.get(name)]
        if missing:
            return self._clarify(
                decision,
                missing + (["signed_on"] if not facts.get("signed_on") else []),
            )
        if (
            facts["reason"] == "non_quality"
            and facts["category"] in rules.excluded_categories
        ):
            decision.reply = "按当前通用规则，这个品类不适用普通商品的无理由退货条件；如有质量问题或另有商品承诺，请提供情况由人工核验。"
            decision.status = "not_applicable"
            return decision
        missing = []
        try:
            signed_on = date.fromisoformat(facts.get("signed_on") or "")
            if signed_on > today:
                raise ValueError("Future signature")
        except ValueError:
            missing.append("signed_on")
        if facts["reason"] == "non_quality":
            if facts.get("unused") is None:
                missing.append("unused")
            if (
                facts["category"] in rules.sealed_categories
                and facts.get("opened") is None
            ):
                missing.append("opened")
        if missing:
            return self._clarify(decision, missing)
        days = (today - signed_on).days
        window = (
            rules.quality_days if facts["reason"] == "quality" else rules.return_days
        )
        if days > window:
            decision.reply = f"按已提供的签收日期计算为{days}天，超出这项通用规则的{window}天申请窗口，需人工核验是否存在例外或其他售后方式。"
            decision.status, decision.needs_human = "manual_review", True
        elif facts["reason"] == "non_quality" and (
            facts.get("unused") is False
            or (
                facts["category"] in rules.sealed_categories
                and facts.get("opened") is True
            )
        ):
            decision.reply = "已提供的使用或拆封情况不满足这项通用无理由规则，不能直接确认可退；质量争议或商品专属承诺请人工核验。"
            decision.status = "not_applicable"
        else:
            source = (
                "订单记录"
                if facts.get("signed_source") == "order_record"
                else "您提供的信息"
            )
            decision.reply = f"根据{source}，签收至今{days}天，处于这项通用规则的{window}天申请窗口。商品品类、使用情况及退货原因仍需按实际订单核验，这不代表退货或退款已获批，也未执行资金操作。"
            decision.status = "conditional"
        return decision

    @staticmethod
    def _clarify(decision: PolicyDecision, missing: list[str]) -> PolicyDecision:
        decision.status = "clarify"
        decision.missing_fields = list(dict.fromkeys(missing))
        decision.context["pending_scenario"] = "window"
        decision.reply = (
            "为判断这件商品适用的规则，请补充："
            + "、".join(FIELD_LABELS[name] for name in decision.missing_fields)
            + "。目前不能直接确认可退。"
        )
        return decision

    def answer(self, query: str) -> tuple[str, bool]:
        decision = self.evaluate(query)
        return decision.reply, decision.found

    def evaluate_for_user(
        self, query: str, context: dict, *, user_id=None, is_admin=False, as_of=None
    ) -> PolicyDecision:
        decision = self.evaluate(query, context, as_of=as_of)
        if (
            not decision.context.get("order_id")
            or decision.status == "informational"
            or not decision.version
        ):
            return decision
        from src.orders.order_service import get_order_service

        order, status = get_order_service().lookup_policy_order(
            decision.context["order_id"], user_id=user_id, is_admin=is_admin
        )
        if status == "unavailable":
            return PolicyDecision(
                "订单服务暂时不可用，不能确认这笔订单的适用条件，请人工核验。",
                status="manual_review",
                needs_human=True,
                version=decision.version,
            )
        if order is None:
            return PolicyDecision(
                "暂时无法核验该订单或其归属，请核对订单号后重试。",
                status="clarify",
                missing_fields=["order_id"],
                context={"pending_scenario": "window"},
                version=decision.version,
            )
        facts = dict(decision.context)
        facts["order_status"] = order.status
        if order.status == "已签收" and not order.signed_at:
            return PolicyDecision(
                "订单记录缺少实际签收时间，不能用下单或预计送达时间判断，请人工核验。",
                status="manual_review",
                needs_human=True,
                version=decision.version,
            )
        try:
            if order.signed_at:
                if date.fromisoformat(order.signed_at) > (as_of or today_shanghai()):
                    raise ValueError("Future signature in order record")
                facts.update(
                    signed_on=date.fromisoformat(order.signed_at).isoformat(),
                    signed_source="order_record",
                )
            if order.created_at:
                facts["order_created_on"] = date.fromisoformat(
                    order.created_at[:10]
                ).isoformat()
        except ValueError:
            return PolicyDecision(
                "订单时间记录无法核验，请人工跟进。",
                status="manual_review",
                needs_human=True,
                version=decision.version,
            )
        return self.evaluate(query, facts, as_of=as_of)


from src.engineering.singleton import singleton_factory


@singleton_factory
def get_return_policy() -> ReturnPolicyService:
    return ReturnPolicyService()

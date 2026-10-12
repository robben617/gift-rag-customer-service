"""第4课演示：订单工具 / 投诉转人工 / 多轮记忆 / 高敏承诺护栏"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

load_dotenv()
# 端口取自 .env 的 API_BASE_URL，默认 8000（与 README / .env.example 一致）
BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
c = httpx.Client(timeout=180)


def login(u, p):
    r = c.post(f"{BASE}/api/auth/login", json={"username": u, "password": p})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def chat(q, sid, headers):
    r = c.post(f"{BASE}/api/chat", json={"query": q, "session_id": sid}, headers=headers)
    return r.json() if r.status_code == 200 else {"error": r.text[:200]}


print("=" * 64)
print("场景1: 订单查询工具（mock，admin 登录）")
print("=" * 64)
h = login("admin", "admin123")
d = chat("帮我查一下订单 ORD20260701001 到哪了？", "d1", h)
print("意图:", d.get("intent"), "| 回答:", (d.get("reply") or "")[:200])

print()
print("=" * 64)
print("场景2: 情绪识别 → 转人工（normal 登录）")
print("=" * 64)
h = login("normal", "normal123")
d = chat("你们什么垃圾公司，我要去12315投诉你们！", "d2", h)
print("情绪:", d.get("emotion"), "| 需要转人工:", d.get("needs_human"))
print("转人工原因:", d.get("human_reason"))
print("交接包键:", list((d.get("handoff_payload") or {}).keys()))
print("回答:", (d.get("reply") or "")[:150])

print()
print("=" * 64)
print("场景3: 多轮记忆（同一会话两轮）")
print("=" * 64)
h = login("member", "member123")
d1 = chat("我上周领了一张满300减50的优惠券", "mem1", h)
print("[第1轮] 回答:", (d1.get("reply") or "")[:100])
d2 = chat("上次那个券怎么用？", "mem1", h)
print("[第2轮] 回答:", (d2.get("reply") or "")[:150])
print("[第2轮] 改写后问题:", d2.get("rewritten_query", ""))

print()
print("=" * 64)
print("场景4: 高敏承诺护栏（退款时效承诺）")
print("=" * 64)
h = login("admin", "admin123")
d = chat("退款几天能到账？", "d4", h)
print("意图:", d.get("intent"), "| 需要转人工:", d.get("needs_human"))
print("原因:", d.get("human_reason", "")[:120])
print("忠实度:", d.get("faithfulness"))
print("回答:", (d.get("reply") or "")[:150])

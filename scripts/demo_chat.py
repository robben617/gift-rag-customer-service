"""第3课演示：智能对话 —— 展示"业务工具直答"与"RAG知识库检索"两条路径"""
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

r = c.post(f"{BASE}/api/auth/login", json={"username": "admin", "password": "admin123"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}


def chat(q, sid="lesson3"):
    print("=" * 60)
    print(f"顾客提问: {q}")
    r = c.post(f"{BASE}/api/chat", json={"query": q, "session_id": sid}, headers=headers)
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        return
    d = r.json()
    print(f"意图: {d.get('intent')}   置信度: {d.get('confidence'):.2f}")
    print(f"回答: {d.get('reply', '')}")
    docs = d.get("results", [])
    print(f"—— 检索到 {len(docs)} 条知识库资料 ——")
    for i, x in enumerate(docs[:2], 1):
        print(f"  [{i}] 相似度{x.get('score')} | {x.get('text', '')[:60]}")
    print()


chat("怎么退货？")
chat("优惠券可以和其他优惠券叠加使用吗？")
chat("能指定快递公司吗？")

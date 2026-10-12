"""端到端 API 测试：登录 → 检索 → 对话（用 httpx 保证 UTF-8 编码）"""
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

# 1. 登录
r = c.post(f"{BASE}/api/auth/login", json={"username": "admin", "password": "admin123"})
print("login status:", r.status_code)
token = (r.json() or {}).get("access_token", "")
print("token ok:", bool(token))
h = {"Authorization": f"Bearer {token}"}

# 2. 检索
r = c.post(f"{BASE}/api/query", json={"query": "怎么退货", "top_k": 5}, headers=h)
print("query status:", r.status_code)
d = r.json()
print("query total_found:", d.get("total_found"))
for x in (d.get("results") or [])[:3]:
    print("   score=", x.get("score"), "|", (x.get("text") or "")[:40])

# 3. 对话（走 DeepSeek）
r = c.post(f"{BASE}/api/chat", json={"query": "怎么退货？", "session_id": "t1"}, headers=h)
print("chat status:", r.status_code)
d = r.json()
print("chat intent:", d.get("intent"), "| confidence:", d.get("confidence"))
print("chat reply:", (d.get("reply") or "")[:120])
print("chat faithfulness:", d.get("faithfulness"), "| needs_human:", d.get("needs_human"))

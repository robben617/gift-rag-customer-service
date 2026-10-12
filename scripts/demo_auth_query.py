"""第2课演示：无 token 401 → 登录拿 token → 带 token 检索"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

load_dotenv()
# 端口取自 .env 的 API_BASE_URL，默认 8000（与 README / .env.example 一致）
BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
c = httpx.Client(timeout=60)

print("=" * 60)
print("【第1步】不带 token 直接检索 —— 预期被拒 401")
print("=" * 60)
r = c.post(f"{BASE}/api/query", json={"query": "怎么退货", "top_k": 5})
print(f"HTTP {r.status_code}")
print("返回:", r.json())
print()

print("=" * 60)
print("【第2步】登录 admin 拿 token")
print("=" * 60)
r = c.post(f"{BASE}/api/auth/login", json={"username": "admin", "password": "admin123"})
data = r.json()
print(f"HTTP {r.status_code}")
token = data.get("access_token", "")
print("access_token:", token[:40] + "...(已截断)")
print("token 里包含: sub=用户名 role=角色 access_level=等级 exp=过期时间")
print()

print("=" * 60)
print("【第3步】带 token 检索「怎么退货」")
print("=" * 60)
headers = {"Authorization": f"Bearer {token}"}
r = c.post(f"{BASE}/api/query", json={"query": "怎么退货", "top_k": 3}, headers=headers)
print(f"HTTP {r.status_code}")
d = r.json()
print(f"共找到 {d['total_found']} 条, 耗时 {d['query_time_ms']}ms")
for i, x in enumerate(d["results"], 1):
    print(f"\n--- 第{i}名 相似度={x['score']} ---")
    print(f"  来源: {x['source_file']}")
    print(f"  内容: {x['text'][:80]}")

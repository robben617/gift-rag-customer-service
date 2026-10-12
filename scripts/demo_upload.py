"""第5课演示：API 上传入库 → 检索命中 → 删除，看向量库变化"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

load_dotenv()
# 端口取自 .env 的 API_BASE_URL，默认 8000（与 README / .env.example 一致）
BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
c = httpx.Client(timeout=120)

r = c.post(f"{BASE}/api/auth/login", json={"username": "admin", "password": "admin123"})
headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

# 入库前统计
stats = c.get(f"{BASE}/api/stats", headers=headers).json()
print(f"入库前向量数: {stats['total_vectors']}")

# 1. 上传一个 Markdown 文本文件
content = (
    "# 磁吸手机支架说明\n\n"
    "本店磁吸手机支架支持 iPhone 12 及以上带 MagSafe 的机型，强磁吸附，可 360 度旋转，"
    "横竖屏切换，追剧直播都方便。\n\n"
    "*关键词: 磁吸支架, MagSafe, 手机支架*"
)
r = c.post(
    f"{BASE}/api/upload/file",
    headers=headers,
    files={"file": ("demo_mount.md", content.encode("utf-8"), "text/markdown")},
)
up = r.json()
print(f"上传状态: {up['status']} | 分块 {up['total_chunks']} | 入库 {up['inserted']} | source_file={up['source_file']}")

# 2. 入库后统计
stats = c.get(f"{BASE}/api/stats", headers=headers).json()
print(f"入库后向量数: {stats['total_vectors']}")

# 3. 检索验证（应命中刚入库的内容）
q = c.post(f"{BASE}/api/query", json={"query": "磁吸手机支架支持什么机型", "top_k": 2}, headers=headers).json()
print(f"检索「磁吸手机支架支持什么机型」top1: {q['results'][0]['text'][:50]}...")

# 4. 删除该文档
src = quote(up["source_file"], safe="")
d = c.request("DELETE", f"{BASE}/api/documents/{src}", headers=headers).json()
print(f"删除结果: {d}")

stats = c.get(f"{BASE}/api/stats", headers=headers).json()
print(f"删除后向量数: {stats['total_vectors']}")

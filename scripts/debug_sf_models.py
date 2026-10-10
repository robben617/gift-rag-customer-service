"""列出 SiliconFlow 账号可用的视觉(VL)模型"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from src.config import settings

resp = httpx.get(
    "https://api.siliconflow.cn/v1/models",
    headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
    timeout=30,
)
print("status:", resp.status_code)
if resp.status_code != 200:
    print(resp.text[:1000])
    raise SystemExit(1)
data = resp.json()
models = data.get("data", [])
print("total models:", len(models))
for m in models:
    mid = str(m.get("id", ""))
    if "VL" in mid or "vl" in mid or "vision" in mid.lower() or "omni" in mid.lower():
        print("VL-MODEL:", mid)

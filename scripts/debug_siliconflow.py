"""诊断 SiliconFlow 视觉接口 400 错误：打印完整响应体"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from src.config import settings

img_path = Path(__file__).resolve().parent.parent / "Chinese.jpg"
b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

print("model:", settings.siliconflow_vl_model)
print("base_url:", settings.siliconflow_base_url)
print("key set:", bool(settings.siliconflow_api_key))
print("img bytes:", img_path.stat().st_size)

payload = {
    "model": settings.siliconflow_vl_model,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": "用一段简短的中文描述这张图片的内容"},
            ],
        }
    ],
    "max_tokens": 512,
}

resp = httpx.post(
    f"{settings.siliconflow_base_url}/chat/completions",
    headers={
        "Authorization": f"Bearer {settings.siliconflow_api_key}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=60,
)
print("HTTP status:", resp.status_code)
print("response body:", resp.text[:2000])

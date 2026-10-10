"""调试：打印若干查询的原始 Chroma 余弦相似度（绕过 API）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.chroma_store import ChromaStore
from src.embedding.embedder import get_embedder

s = ChromaStore()
e = get_embedder()

for q in ["怎么退货", "退货包运费吗", "退款几天到账", "健康检查"]:
    v = e.embed_query(q)
    r = s.dense_search(v.tolist(), top_k=5, threshold=0.0)
    print(f"=== {q} => total={r.total_found}")
    for x in r.results[:5]:
        print(f"    score={round(x.score, 4)}  text={x.text[:36]}")

"""模块3: 向量化 + 向量存储 + 检索

本地 bge-small-zh-v1.5 → Chroma 存储 → 纯 Dense 检索 + 本地 bge-reranker

使用:
    from src.embedding import Retriever, IndexingPipeline

    # 检索
    retriever = Retriever()
    results = retriever.search("如何退货?")

    # 入库
    pipeline = IndexingPipeline()
    report = pipeline.run_from_text(markdown_text, "policy.pdf", DocType.PDF)
"""

from .chroma_store import ChromaStore
from .embedder import Embedder, get_embedder
from .models import EmbeddingResult, SearchResponse, SearchResult
from .pipeline import IndexingPipeline
from .retriever import Retriever

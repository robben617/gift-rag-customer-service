"""本地 Reranker 重排序模块 — BAAI/bge-reranker-base（替代 DashScope TextReRank API）

特性:
  - 本地 sentence-transformers CrossEncoder 加载，无 API 调用
  - 文本查询 + 文本文档重排序（cross-encoder）
  - 懒加载模型
  - 保持原 rerank() / rerank_search_response() 对外签名不变

注意:
  - bge-reranker-base 是文本 cross-encoder，不支持图片查询（原 rerank_by_image 已移除）。
"""

import logging
from typing import Optional

from src.config import settings

from ..embedding.models import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """本地 BAAI/bge-reranker-base 封装

    使用方式:
        reranker = Reranker()
        reranked = reranker.rerank(query="怎么退货?", documents=["doc1...", "doc2..."])
        reranked = reranker.rerank_search_response(query, search_response, top_n=5)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,  # 兼容旧签名，本地模型忽略
    ):
        self.model_name = model_name or settings.reranker_model
        self._model_path = settings.reranker_model_path
        self._model = None  # 懒加载

    def _get_model(self):
        if self._model is None:
            path = self._model_path or settings.reranker_model_path
            if not path:
                raise RuntimeError(
                    "未配置 RERANKER_MODEL_PATH：请在 .env 中设置本地 bge-reranker 模型目录"
                )
            from sentence_transformers import CrossEncoder

            logger.info("加载本地 reranker 模型: %s", path)
            self._model = CrossEncoder(path)
        return self._model

    # ── 文本查询重排序 ──

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
        return_documents: bool = True,
        instruct: Optional[str] = None,
    ) -> list[dict]:
        """对文档列表进行重排序

        Args:
            query: 查询文本
            documents: 候选文档文本列表
            top_n: 返回前 N 条
            return_documents: 是否返回文档原文（本地模型恒返回）
            instruct: 排序指令（本地 cross-encoder 忽略）

        Returns:
            [{"index": 0, "document": {"text": "..."}, "relevance_score": 0.93}, ...]
            按 relevance_score 降序排列
        """
        if not documents:
            return []

        model = self._get_model()
        pairs = [(query, d) for d in documents]
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        scores = [float(s) for s in scores]

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = order[: min(top_n, len(documents))]

        return [
            {
                "index": i,
                "document": {"text": documents[i]},
                "relevance_score": round(scores[i], 6),
            }
            for i in top_indices
        ]

    # ── 图片查询重排序（本地文本模型不支持）──

    def rerank_by_image(
        self,
        image_url: str,
        documents: list[str],
        top_n: int = 5,
        return_documents: bool = True,
    ) -> list[dict]:
        raise NotImplementedError(
            "本地 bge-reranker 是文本模型，不支持以图搜文。"
        )

    # ── 集成 SearchResponse ──

    def rerank_search_response(
        self,
        query: str,
        search_response: SearchResponse,
        top_n: Optional[int] = None,
    ) -> SearchResponse:
        """对 SearchResponse 进行重排序，生成新的 SearchResponse

        保留原始 score，新增 rerank_score。
        """
        if top_n is None:
            top_n = settings.reranker_top_n

        if not search_response.results:
            return search_response

        texts = [r.text for r in search_response.results]

        try:
            ranked = self.rerank(query=query, documents=texts, top_n=top_n)
        except Exception as e:
            logger.error("重排序失败，返回原始结果: %s", e)
            return search_response

        reranked_results: list[SearchResult] = []
        for item in ranked:
            orig_idx = item["index"]
            if orig_idx < len(search_response.results):
                orig = search_response.results[orig_idx]
                reranked_results.append(
                    SearchResult(
                        chunk_id=orig.chunk_id,
                        text=orig.text,
                        score=orig.score,
                        doc_type=orig.doc_type,
                        source_file=orig.source_file,
                        page_number=orig.page_number,
                        heading_path=orig.heading_path,
                        metadata={
                            **orig.metadata,
                            "rerank_score": item.get("relevance_score", 0),
                            "original_rank": orig_idx,
                        },
                    )
                )

        return SearchResponse(
            query=query,
            results=reranked_results,
            total_found=len(reranked_results),
            elapsed_ms=search_response.elapsed_ms,
            threshold=search_response.threshold,
        )


# ═══════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════

from src.engineering.singleton import singleton_factory


@singleton_factory
def get_reranker() -> Reranker:
    return Reranker()

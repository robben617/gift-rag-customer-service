"""模块3 检索接口 — 纯 Dense 检索（Chroma 余弦）+ 本地 bge-reranker 重排序

方案A：纯 dense 召回 20 条候选 → bge-reranker-base 精排 Top-K（不再保留 BM25 稀疏）。

使用:
    retriever = Retriever()
    results = retriever.search("怎么退货?")                    # dense + rerank
    results = retriever.search("退款", use_rerank=False)       # 纯 dense
"""

import logging
from typing import Optional

from src.access import build_access_where
from src.config import settings

from .chroma_store import ChromaStore
from .embedder import Embedder, get_embedder
from .models import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


def _get_cache():
    """延迟导入缓存模块"""
    from src.engineering import get_cache

    return get_cache()


def _search_response_to_dict(response: SearchResponse) -> dict:
    """SearchResponse → dict（Redis 缓存需 JSON 序列化）"""
    return {
        "query": response.query,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "doc_type": r.doc_type,
                "source_file": r.source_file,
                "page_number": r.page_number,
                "heading_path": r.heading_path,
                "metadata": r.metadata,
            }
            for r in response.results
        ],
        "total_found": response.total_found,
        "elapsed_ms": response.elapsed_ms,
        "threshold": response.threshold,
    }


def _dict_to_search_response(data: dict) -> SearchResponse:
    """dict → SearchResponse（缓存反序列化）"""
    return SearchResponse(
        query=data.get("query", ""),
        results=[SearchResult(**r) for r in data.get("results", [])],
        total_found=data.get("total_found", 0),
        elapsed_ms=data.get("elapsed_ms", 0),
        threshold=data.get("threshold", 0),
    )


def _merge_results(results_a: list, results_b: list) -> list:
    """双路召回合并去重 — 按 chunk_id 保留最高分，按分数降序"""
    best: dict[str, SearchResult] = {}
    for r in results_a + results_b:
        if r.chunk_id not in best or r.score > best[r.chunk_id].score:
            best[r.chunk_id] = r
    return sorted(best.values(), key=lambda r: r.score, reverse=True)


def _filter_by_effectiveness(results: list, today: str | None = None) -> list:
    """过滤已过期/未生效的文档（政策时效元数据，改进4）"""
    if not results:
        return results
    if today is None:
        from datetime import date

        today = date.today().isoformat()

    valid = []
    for r in results:
        meta = r.metadata or {}
        eff_from = meta.get("effective_from") or ""
        eff_to = meta.get("effective_to") or ""
        if eff_from and eff_from > today:
            continue  # 尚未生效
        if eff_to and eff_to < today:
            continue  # 已过期
        valid.append(r)
    return valid


class Retriever:
    """纯 Dense 检索器 — Embedder + ChromaStore + 本地 Reranker

    默认流程:
      1. Embedder 将 query → 稠密向量
      2. Chroma 余弦检索（access_level 权限过滤）
      3. (可选) 本地 bge-reranker 精排
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[ChromaStore] = None,
    ):
        self.embedder = embedder or get_embedder()
        self.store = store or ChromaStore()
        self._reranker = None  # 懒加载

    @property
    def reranker(self):
        if self._reranker is None and settings.reranker_enabled:
            from src.retrieval.reranker import get_reranker

            self._reranker = get_reranker()
        return self._reranker

    def _build_where(
        self,
        access_level: str,
        filter_by_doc_type: Optional[str],
        filter_by_source: Optional[str],
    ) -> dict:
        """构建 Chroma where 过滤（含权限隔离）"""
        conds: list[dict] = []
        if filter_by_doc_type:
            conds.append({"doc_type": {"$eq": filter_by_doc_type}})
        if filter_by_source:
            conds.append({"source_file": {"$eq": filter_by_source}})
        conds.append(build_access_where(access_level))
        return {"$and": conds} if len(conds) > 1 else conds[0]

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: Optional[bool] = None,
        filter_by_doc_type: Optional[str] = None,
        filter_by_source: Optional[str] = None,
        threshold: Optional[float] = None,
        access_level: str = "public",
    ) -> SearchResponse:
        """检索相关文档（纯 dense + 可选 rerank）"""
        if threshold is None:
            threshold = settings.retrieval_similarity_threshold
        if use_rerank is None:
            use_rerank = settings.reranker_enabled

        # ── 查询缓存 ──
        cache = _get_cache()
        cache_key = (
            f"{query}:{top_k}:dense:{use_rerank}:"
            f"{filter_by_doc_type}:{filter_by_source}:{threshold}:{access_level}"
        )
        cached = cache.get_query_result(cache_key)
        if cached:
            logger.debug("缓存命中: %s", query[:50])
            return _dict_to_search_response(cached)

        where = self._build_where(access_level, filter_by_doc_type, filter_by_source)

        # ── 向量化 ──
        query_vector = self.embedder.embed_query(query)

        # ── 召回阶段取更多候选供 Reranker 筛选 ──
        recall_k = max(top_k, settings.retrieval_dense_top_k) if use_rerank else top_k

        response = self.store.dense_search(
            query_vector=query_vector.tolist(),
            top_k=recall_k,
            filter_expr=where,
            threshold=threshold,
        )
        response.query = query

        # 政策时效过滤——剔除已过期/未生效文档（在重排序前）
        filtered_results = _filter_by_effectiveness(response.results)
        if len(filtered_results) != len(response.results):
            logger.debug(
                "时效过滤: %d → %d 条", len(response.results), len(filtered_results)
            )
            response.results = filtered_results
            response.total_found = len(filtered_results)

        # ── 重排序 ──
        if use_rerank and self.reranker and response.results:
            try:
                response = self.reranker.rerank_search_response(
                    query=query,
                    search_response=response,
                    top_n=top_k,
                )
                logger.debug("重排序完成: %d → %d", recall_k, response.total_found)
            except Exception as e:
                logger.warning("重排序失败，降级使用原始结果: %s", e)
                response = SearchResponse(
                    query=query,
                    results=response.results[:top_k],
                    total_found=min(response.total_found, top_k),
                    elapsed_ms=response.elapsed_ms,
                    threshold=response.threshold,
                )

        # ── 缓存结果 ──
        cache.set_query_result(
            cache_key,
            _search_response_to_dict(response),
            ttl=settings.query_cache_ttl,
        )

        return response

    def expand_parent_context(
        self,
        docs: list[SearchResult],
        window: int = 1,
    ) -> list[SearchResult]:
        """父文档检索补全（Parent Document Retriever 简化版）"""
        if not docs or window <= 0:
            return docs

        from collections import defaultdict

        by_source: dict[str, list[SearchResult]] = defaultdict(list)
        for d in docs:
            if d.source_file:
                by_source[d.source_file].append(d)

        expanded: list[SearchResult] = []
        for source, group in by_source.items():
            try:
                all_chunks = self.store.query_chunks_by_source(source)
            except Exception as e:
                logger.warning("父块补全失败，原样返回: %s", e)
                expanded.extend(group)
                continue

            idx_map = {c["chunk_index"]: c["text"] for c in all_chunks}

            for d in group:
                ci = (d.metadata or {}).get("chunk_index", 0)
                neighbors = [
                    idx_map[ci + offset]
                    for offset in range(-window, window + 1)
                    if offset != 0 and (ci + offset) in idx_map
                ]
                if neighbors:
                    d.text = "\n".join(neighbors + [d.text])
                expanded.append(d)

        return expanded

    def search_dual_path(
        self,
        query: str,
        secondary_query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        threshold: Optional[float] = None,
        filter_by_doc_type: Optional[str] = None,
        filter_by_source: Optional[str] = None,
        access_level: str = "public",
    ) -> SearchResponse:
        """双路召回 + 合并去重（原始问题 + 改写问题）"""
        if not secondary_query or secondary_query == query:
            return self.search(
                query=query,
                top_k=top_k,
                use_rerank=use_rerank,
                threshold=threshold,
                filter_by_doc_type=filter_by_doc_type,
                filter_by_source=filter_by_source,
                access_level=access_level,
            )

        recall_k = max(top_k, settings.retrieval_dense_top_k) if use_rerank else top_k

        r1 = self.search(
            query,
            top_k=recall_k,
            use_rerank=False,
            threshold=threshold,
            filter_by_doc_type=filter_by_doc_type,
            filter_by_source=filter_by_source,
            access_level=access_level,
        )
        r2 = self.search(
            secondary_query,
            top_k=recall_k,
            use_rerank=False,
            threshold=threshold,
            filter_by_doc_type=filter_by_doc_type,
            filter_by_source=filter_by_source,
            access_level=access_level,
        )

        merged = _merge_results(r1.results, r2.results)
        response = SearchResponse(
            query=query,
            results=merged,
            total_found=len(merged),
            elapsed_ms=round(r1.elapsed_ms + r2.elapsed_ms, 1),
            threshold=r1.threshold,
        )

        if use_rerank and self.reranker and response.results:
            try:
                response = self.reranker.rerank_search_response(
                    query=query,
                    search_response=response,
                    top_n=top_k,
                )
            except Exception as e:
                logger.warning("双路召回精排失败，降级用合并结果: %s", e)
                response.results = response.results[:top_k]
                response.total_found = len(response.results)
        else:
            response.results = response.results[:top_k]
            response.total_found = len(response.results)

        return response

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        use_rerank: Optional[bool] = None,
        threshold: Optional[float] = None,
        access_level: str = "public",
    ) -> list[SearchResponse]:
        """批量检索（纯 dense）"""
        if threshold is None:
            threshold = settings.retrieval_similarity_threshold
        if use_rerank is None:
            use_rerank = settings.reranker_enabled

        where = self._build_where(access_level, None, None)
        query_vectors = self.embedder.embed_queries(queries)
        responses = []
        for i, q in enumerate(queries):
            r = self.store.dense_search(
                query_vector=query_vectors[i].tolist(),
                top_k=top_k,
                filter_expr=where,
                threshold=threshold,
            )
            r.query = q

            if use_rerank and self.reranker and r.results:
                try:
                    r = self.reranker.rerank_search_response(q, r, top_n=top_k)
                except Exception as e:
                    logger.warning("批量重排序失败 [%d]: %s", i, e)

            responses.append(r)
        return responses

    def health_check(self) -> dict:
        result = {
            "embedder": "ok",
            "vector_store": "not_ok",
            "reranker": "not_configured",
            "search_test": "skipped",
        }
        store_ok = self.store.health_check()
        result["vector_store"] = "ok" if store_ok else "not_ok"

        if self.reranker:
            result["reranker"] = "configured"

        if store_ok:
            stats = self.store.stats()
            result["collection"] = stats
            if stats.get("total_vectors", 0) > 0:
                try:
                    test_vec = self.embedder.embed_query("健康检查")
                    r = self.store.dense_search(
                        query_vector=test_vec.tolist(),
                        top_k=1,
                    )
                    result["search_test"] = "ok" if r.total_found > 0 else "no_results"
                except Exception as e:
                    result["search_test"] = f"failed: {e}"
        return result


# ── 模块级单例 ──

from src.engineering.singleton import singleton_factory


@singleton_factory
def get_retriever() -> Retriever:
    """获取 Retriever 单例"""
    return Retriever()

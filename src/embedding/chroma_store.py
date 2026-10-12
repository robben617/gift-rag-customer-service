"""Chroma 向量存储 — 本地持久化

Schema（metadata 全展平为标量，Chroma 不支持嵌套 dict/list）:
  - id / text / embedding（Chroma 内建）
  - doc_type / source_file / page_number / chunk_index / strategy / target_size
  - access_level（INT rank：0=public, 1=member, 2=vip，模块33 权限过滤）
  - heading_path（JSON 字符串）、chunk_metadata（JSON 字符串，完整元数据）

检索:
  - 纯 dense 余弦检索（方案A：不保留 BM25 稀疏）
  - where 过滤 access_level <= 用户rank
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.access import level_name, parse_access_level
from src.config import settings

from .models import EmbeddingResult, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

ACCESS_FIELD = "access_level"


def _flatten_metadata(meta: dict) -> dict:
    """把任意嵌套元数据展平为 Chroma 允许的标量字段（键集合固定，保证批次内一致）"""
    rank = parse_access_level(meta.get("access_level", "public"))
    return {
        "doc_type": str(meta.get("doc_type", "")),
        "source_file": str(meta.get("source_file", "")),
        "page_number": int(meta.get("page_number", 0) or 0),
        "chunk_index": int(meta.get("chunk_index", 0) or 0),
        "strategy": str(meta.get("strategy", "")),
        "target_size": int(meta.get("target_size", 0) or 0),
        "access_level": int(rank),
        "access_level_label": level_name(rank),
        "heading_path": json.dumps(meta.get("heading_path", []) or [], ensure_ascii=False),
        "chunk_metadata": json.dumps(meta, ensure_ascii=False, default=str),
    }


def _unflatten_metadata(flat: dict) -> dict:
    """把 Chroma 标量 metadata 还原为完整 dict（供检索结果回传）"""
    meta = {}
    try:
        meta = json.loads(flat.get("chunk_metadata", "{}") or "{}")
    except Exception:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    # 保证关键字段存在
    meta.setdefault("doc_type", flat.get("doc_type", ""))
    meta.setdefault("source_file", flat.get("source_file", ""))
    meta.setdefault("page_number", flat.get("page_number", 0))
    meta.setdefault("access_level", flat.get("access_level", 0))
    return meta


def _unflatten_heading(flat: dict) -> list:
    try:
        val = json.loads(flat.get("heading_path", "[]") or "[]")
        return val if isinstance(val, list) else []
    except Exception:
        return []


class ChromaStore:
    """Chroma 向量存储 — 纯 dense 余弦检索"""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection or settings.chroma_collection
        self._client = None
        self._collection = None

    # ── 客户端 / collection 懒加载 ──

    @property
    def client(self):
        if self._client is None:
            import chromadb

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── 健康检查 ──

    def health_check(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception as e:
            logger.warning("Chroma 不可达: %s", e)
            return False

    # ── Collection 生命周期 ──

    def create_collection(self, drop_if_exists: bool = False) -> None:
        if drop_if_exists:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info("已删除旧 collection: %s", self.collection_name)
            except Exception:
                pass
            self._collection = None
        self._get_collection()
        logger.info("Chroma collection 就绪: %s", self.collection_name)

    def _list_names(self) -> list[str]:
        """list_collections 在不同 chromadb 版本返回名字或 Collection 对象，统一取名字"""
        cols = self.client.list_collections()
        names: list[str] = []
        for c in cols:
            if isinstance(c, str):
                names.append(c)
            elif hasattr(c, "name"):
                names.append(str(c.name))
            else:
                names.append(str(c))
        return names

    def collection_exists(self) -> bool:
        try:
            return self.collection_name in self._list_names()
        except Exception:
            return False

    def ensure_access_level_field(self) -> str:
        """Chroma 权限过滤走 metadata，无需迁移，no-op"""
        return "chroma_metadata_native"

    # ── 写入 ──

    def insert(self, embeddings: list[EmbeddingResult], batch_size: int = 500) -> int:
        if not embeddings:
            return 0
        col = self._get_collection()
        total = len(embeddings)
        inserted = 0
        logger.info("写入 Chroma: %d 条", total)

        for i in range(0, total, batch_size):
            batch = embeddings[i : i + batch_size]
            ids = [e.chunk_id for e in batch]
            docs = [e.text for e in batch]
            vecs = [e.vector for e in batch]
            metas = [_flatten_metadata(e.metadata) for e in batch]
            try:
                col.add(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
                inserted += len(batch)
            except Exception as e:
                logger.error("批次 %d 写入失败: %s", i // batch_size + 1, e)
                raise

        logger.info("写入完成: %d/%d 条", inserted, total)
        return inserted

    # ── 检索 ──

    def dense_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: Optional[dict] = None,
        threshold: float = 0.0,
    ) -> SearchResponse:
        """纯稠密余弦检索（filter_expr 为 Chroma where dict）"""
        t0 = time.time()
        col = self._get_collection()

        # 存储层纵深防御：漏传权限过滤时默认 public（只返回公开内容），防越权泄露
        where = filter_expr if filter_expr is not None else {ACCESS_FIELD: {"$lte": 0}}

        raw = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return self._build_response(raw, threshold, time.time() - t0)

    def _build_response(
        self, raw: dict, threshold: float, elapsed: float
    ) -> SearchResponse:
        results: list[SearchResult] = []
        ids = (raw.get("ids") or [[]])[0] if raw.get("ids") else []
        docs = (raw.get("documents") or [[]])[0] if raw.get("documents") else []
        metas = (raw.get("metadatas") or [[]])[0] if raw.get("metadatas") else []
        dists = (raw.get("distances") or [[]])[0] if raw.get("distances") else []

        for i in range(len(ids)):
            dist = dists[i] if i < len(dists) else 0.0
            score = 1.0 - float(dist)  # Chroma 余弦距离 → 余弦相似度
            if score < threshold:
                continue
            meta_flat = metas[i] if i < len(metas) else {}
            meta = _unflatten_metadata(meta_flat)
            heading = _unflatten_heading(meta_flat)
            results.append(
                SearchResult(
                    chunk_id=str(ids[i]),
                    text=docs[i] if i < len(docs) else "",
                    score=round(score, 4),
                    doc_type=str(meta.get("doc_type", "")),
                    source_file=str(meta.get("source_file", "")),
                    page_number=int(meta.get("page_number", 0) or 0),
                    heading_path=heading,
                    metadata=meta,
                )
            )

        return SearchResponse(
            query="",
            results=results,
            total_found=len(results),
            elapsed_ms=round(elapsed * 1000, 1),
            threshold=threshold,
        )

    # ── 统计 / 查询 ──

    def stats(self) -> dict:
        try:
            if not self.collection_exists():
                return {"exists": False, "total_vectors": 0}
            col = self._get_collection()
            return {
                "exists": True,
                "collection_name": self.collection_name,
                "total_vectors": col.count(),
            }
        except Exception as e:
            logger.warning("统计获取失败: %s", e)
            return {"exists": False, "total_vectors": 0, "error": str(e)}

    def list_sources(self) -> list[dict]:
        """列出库中所有来源文档（按 source_file 聚合）

        Returns:
            [{source_file, chunks, doc_type, access_level_label, pages}]，
            按 chunk 数降序。供管理台「知识库管理」展示。
        """
        try:
            if not self.collection_exists():
                return []
            col = self._get_collection()
            raw = col.get(include=["metadatas"])
        except Exception as e:
            logger.warning("列出文档失败: %s", e)
            return []

        agg: dict[str, dict] = {}
        for flat in raw.get("metadatas") or []:
            flat = flat or {}
            src = str(flat.get("source_file", "") or "")
            if not src:
                continue
            item = agg.setdefault(
                src,
                {
                    "source_file": src,
                    "chunks": 0,
                    "doc_type": str(flat.get("doc_type", "") or ""),
                    "access_level_label": str(
                        flat.get("access_level_label", "") or ""
                    ),
                    "pages": 0,
                },
            )
            item["chunks"] += 1
            item["pages"] = max(
                item["pages"], int(flat.get("page_number", 0) or 0)
            )

        return sorted(agg.values(), key=lambda x: (-x["chunks"], x["source_file"]))

    def query_chunks_by_source(self, source_file: str) -> list[dict]:
        """查询某 source 的所有 chunk（按 chunk_index 排序，供父文档检索补全用）"""
        try:
            col = self._get_collection()
            raw = col.get(
                where={"source_file": source_file},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning("查询 source chunks 失败: %s", e)
            return []

        items = []
        ids = raw.get("ids", [])
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])
        for i in range(len(ids)):
            meta = _unflatten_metadata(metas[i] if i < len(metas) else {})
            items.append(
                {
                    "chunk_id": str(ids[i]),
                    "text": docs[i] if i < len(docs) else "",
                    "chunk_index": meta.get("chunk_index", 0),
                }
            )
        items.sort(key=lambda x: x["chunk_index"])
        return items

    def get_sample_metadata(self, limit: int = 50) -> list[dict]:
        """采样库中已有 chunk 的完整 metadata（供切分策略变更检测）"""
        try:
            col = self._get_collection()
            raw = col.get(limit=limit, include=["metadatas"])
            return [
                _unflatten_metadata(m) for m in (raw.get("metadatas") or [])
            ]
        except Exception as e:
            logger.debug("采样 metadata 失败: %s", e)
            return []

    def delete_by_source(self, source_file: str) -> int:
        try:
            col = self._get_collection()
            existing = col.get(where={"source_file": source_file}, include=[])
            ids = list(existing.get("ids", []) or [])
            if ids:
                col.delete(ids=ids)
                logger.info("已删除 %d 条: %s", len(ids), source_file)
            return len(ids)
        except Exception as e:
            logger.error("删除失败: %s", e)
            return 0

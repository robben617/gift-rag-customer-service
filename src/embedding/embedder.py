"""本地 Embedding 向量化器 — BAAI/bge-small-zh-v1.5（替代 DashScope 多模态 API）

特性:
  - 本地 sentence-transformers 加载，无 API 调用、无网络依赖
  - 输出 512-dim 向量（bge-small-zh-v1.5）
  - 查询侧加中文 instruction 前缀，文档侧不加（bge 官方推荐）
  - L2 归一化（配合 Chroma 余弦度量）
  - 懒加载模型（首次调用才加载，避免 import 期拖慢启动）

注意:
  - bge-small-zh-v1.5 是纯文本模型，不支持图片向量化。
    图片请先经 VLM 转文字描述后再入库（原 embed_image 能力已移除）。
"""

import logging
import time
from typing import Optional

import numpy as np

from src.config import settings

from .models import BatchEmbeddingResult, EmbeddingResult

logger = logging.getLogger(__name__)

# bge-small-zh-v1.5 官方推荐：查询侧加中文指令，文档侧不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    """本地 BAAI/bge-small-zh-v1.5 向量化封装

    使用方式:
        embedder = Embedder()
        result = embedder.embed_chunks(chunks)       # 批量文本
        vec = embedder.embed_query("怎么退货?")       # 查询向量
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,  # 兼容旧签名，本地模型忽略
        dimension: Optional[int] = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name or settings.embedding_model
        self._model_path = settings.embedding_model_path
        self._dimension = dimension or settings.embedding_dim
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None  # 懒加载

    # ── 模型懒加载 ──

    def _get_model(self):
        if self._model is None:
            path = self._model_path or settings.embedding_model_path
            if not path:
                raise RuntimeError(
                    "未配置 EMBEDDING_MODEL_PATH：请在 .env 中设置本地 bge 模型目录"
                )
            from sentence_transformers import SentenceTransformer

            logger.info("加载本地 embedding 模型: %s", path)
            self._model = SentenceTransformer(path)
        return self._model

    # ── 维度 ──

    @property
    def dimension(self) -> int:
        return self._dimension

    # ── 文本批量向量化 ──

    def embed_chunks(
        self,
        texts: list[str],
        chunk_ids: list[str],
        metadata_list: Optional[list[dict]] = None,
        show_progress: bool = True,
    ) -> BatchEmbeddingResult:
        """批量向量化 chunk 文本（文档侧：不加 instruction）"""
        t0 = time.time()
        n = len(texts)
        if n == 0:
            return BatchEmbeddingResult(
                embeddings=[],
                total=0,
                dimension=self.dimension,
                model_name=self.model_name,
                elapsed_seconds=0,
            )

        if metadata_list is None:
            metadata_list = [{} for _ in range(n)]

        model = self._get_model()
        logger.info("开始向量化 %d 条文本 (本地 bge)", n)

        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
        except Exception as e:
            logger.error("本地向量化失败: %s", e)
            raise

        embeddings: list[EmbeddingResult] = []
        for i in range(n):
            embeddings.append(
                EmbeddingResult(
                    chunk_id=chunk_ids[i],
                    vector=[float(x) for x in vectors[i]],
                    text=texts[i],
                    metadata=metadata_list[i] if i < len(metadata_list) else {},
                )
            )

        elapsed = time.time() - t0
        logger.info(
            "向量化完成: %d 条, %d-dim, 耗时 %.1fs",
            n,
            self.dimension,
            elapsed,
        )
        return BatchEmbeddingResult(
            embeddings=embeddings,
            total=n,
            dimension=self.dimension,
            model_name=self.model_name,
            elapsed_seconds=round(elapsed, 2),
        )

    # ── 查询向量化 ──

    def embed_query(self, query: str) -> np.ndarray:
        """查询文本 → 向量 (加 instruction, L2 归一化)"""
        model = self._get_model()
        vec = model.encode(
            [QUERY_INSTRUCTION + query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.array(vec, dtype=np.float32)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """批量查询向量化 (加 instruction, L2 归一化)"""
        model = self._get_model()
        prefixed = [QUERY_INSTRUCTION + q for q in queries]
        vecs = model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    # ── 图片向量化（本地文本模型不支持，显式告知）──

    def embed_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError(
            "本地 bge 是文本模型，不支持图片向量化。图片请先经 VLM 转文字描述后再入库。"
        )

    def embed_image_base64(self, image_b64: str, mime_type: str = "image/jpeg") -> np.ndarray:
        raise NotImplementedError(
            "本地 bge 是文本模型，不支持图片向量化。图片请先经 VLM 转文字描述后再入库。"
        )

    # ── 资源释放 ──

    def unload(self):
        """释放模型内存"""
        self._model = None


# ═══════════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════════

from src.engineering.singleton import singleton_factory


@singleton_factory
def get_embedder() -> Embedder:
    return Embedder()

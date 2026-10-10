# Architecture Decision Records (ADR)

> 记录本项目关键架构决策的背景、取舍与后果。面试/评审时可直接引用每个决策的"为什么"。

## ⚠️ 阅读须知：方案演进时间线

本项目经历过一次**向量库与检索方案的迁移**，因此下面部分 ADR 记录的是**历史决策**：

| 阶段 | 向量库 | 检索 | Embedding / Reranker |
|------|--------|------|----------------------|
| 早期 | Milvus 2.6 | BM25 + Dense 混合检索，WeightedRanker 0.3/0.7 | qwen2.5-vl-embedding（2048 维，API）/ qwen3-vl-rerank（API） |
| **当前** | **Chroma**（本地持久化） | **纯稠密检索 + 本地 CrossEncoder 精排** | **bge-small-zh-v1.5（512 维，本地）** / **bge-reranker-base（本地）** |

- **ADR-001（混合检索）与 ADR-008（Embedding 模型）描述的是早期方案**，正文已加「已被取代」标注，保留作为选型推理过程的记录。
- **ADR-010 记录了这次迁移本身的决策**，也是回答"为什么从 Milvus 换成 Chroma"的地方。
- 其余 ADR（幻觉检测 / 三层记忆 / 政策时效 / PDF 解析 / 高敏护栏 / 权限隔离）与当前实现一致。

---

## ADR-001：混合检索 —— BM25 稀疏 + 稠密向量 + WeightedRanker

> **状态：⚠️ 已被取代（2026 迁移到 Chroma + 纯稠密检索，见 ADR-010）。**
> 本节保留的是 Milvus 时期"要不要做混合检索、权重怎么定"的推理过程，当前代码里已不再有 BM25 与加权融合。

- **决策**：检索用 Hybrid Search，BM25（jieba 分词）与稠密向量（IP）并行召回，WeightedRanker 加权融合（0.3 / 0.7）。
- **背景**：电商客服问题兼有强语义（"这个好用吗"）与强词法（型号、SKU、政策关键词），纯向量对专名/ID 召回差。
- **为什么**：BM25 兜住专名精确匹配，稠密兜住语义改写；0.7/0.3 偏语义，因客服问题更多是意图改写而非精确词。
- **替代方案**：纯向量（专名召回弱）、纯 BM25（语义失效）。RRF 融合曾考虑，但 WeightedRanker 在 Milvus 内原生支持、量纲统一。
- **后果**：召回覆盖更好；代价是两路各一次检索，延迟略增（已用 Reranker 精排抵消）。

## ADR-002：双路召回 + 合并去重（原始 + 改写问题并行检索）

- **决策**：检索时原始问题和改写后问题各走一路，合并去重（按 chunk_id 保留高分）。
- **背景**：改写后的查询更专业但可能偏离原意，只用改写有"改写错就全错"风险。
- **为什么**：原始问题保真度 + 改写问题专业术语，双路合并消除单点失败；行业实测召回率 +30%。
- **替代方案**：单路改写（保真度风险）、级联降级（失败才走，非主动并行）。
- **后果**：多一路检索；已限定改写与原始相同时退化单路，避免浪费。
- **📌 数据口径（重要）**：上文"+30%"是**行业实测召回率提升的引用值**，本项目**没有做过单路 vs 双路的 A/B 对照跑分**，也没有留下基线数字。当前实测只有双路方案下的 `Recall@5 = 0.755`（49 条自建评测集，命中率口径）。如需对外引用提升幅度，应先补跑对照实验（单路用 `Retriever.search()`，双路用 `Retriever.search_dual_path()`，其余参数固定）。

## ADR-003：幻觉检测 —— claims 级聚合，不信任 LLM 自报标量

- **决策**：G-Eval 风格幻觉检测，但 `faithfulness = (supported + 0.5*partial) / total` 从断言级聚合；`has_hallucination = 存在幻觉断言 或 <0.6`。
- **背景**：LLM 裁判自偏（给自己的输出打高分）与均值化掩盖（个别编造被整体分平滑）是 G-Eval 最大漏检源。
- **为什么**：只报 answer-level 单分不可靠，claim 级分解能定位到具体编造断言。
- **替代方案**：信任 LLM 自报标量（已废弃，漏检核心场景）、多模型陪审（成本高）。
- **后果**：检测更严格，代价是依赖 claim 拆分质量；空 claims 保守判有幻觉（转人工）。

## ADR-004：三层会话记忆 —— 短期窗口 + 滚动摘要 + 实体 ledger

- **决策**：多轮记忆分三层：最近 N 轮原文、窗口外旧对话 LLM 滚动摘要、实体（券/订单/金额）ledger。
- **背景**：朴素滑动窗口会把早期关键实体（"上次那个券"）挤出窗口，指代消解失败。
- **为什么**：窗口保最新、摘要保要点、实体表保精确——"上次那个券"只有实体 ledger 能解析。
- **替代方案**：只滑动窗口（context cliff）、全量历史进上下文（token 爆炸+噪音）。
- **后果**：核心内存维护 + 摘要 LLM 调用；已做降级（摘要失败退回截断）。

## ADR-005：政策时效 —— effective_from/to + 写入后缓存失效

- **决策**：入库打生效窗口元数据，检索层过滤过期/未生效；写入成功后主动清检索缓存。
- **背景**：电商政策频繁变（大促/618），旧政策不淘汰会答出已结束活动；缓存 TTL 内不失效会返回旧答案。
- **为什么**：行业铁律"缓存失效必须绑定源数据变更事件"；时效过滤让政策自动过期无需人工。
- **替代方案**：全量重建（成本高）、只过滤不失效缓存（滞后 1h）。
- **后果**：需入库侧维护 effective 字段；检索层已在 `Retriever.search()` 内做过滤（`_filter_by_effectiveness`，重排序前剔除过期/未生效文档）。当前 Chroma 实现是把 `effective_from/to` 与其余元数据一起展平为标量存进 metadata，按需在应用层过滤；若数据量增大，可下沉为 Chroma `where` 条件过滤以减少回传。

## ADR-006：PDF 解析 —— 文本快路径 + 扫描页本地 OCR

- **决策**：逐页分类——`fitz.get_text()` 文本充足（≥50 字）判为文本页，零成本直接提取；文本不足判为扫描页，渲染成图片后走**本地 RapidOCR**（rapidocr_onnxruntime，纯文字识别）；含有效图片的页即使已有部分文本也走 OCR，避免图内正文丢失。最后按页码合并。
- **背景**：文本 PDF 全量走 OCR 既慢又浪费——OCR 每页都要过一遍模型、分钟级；而证书/图文混排类 PDF 的正文常常整段在图片里。同时为了去掉外部依赖，OCR 从云端 VLM 换成了本地 RapidOCR。
- **为什么**：文本页直接提取（快、零成本）；扫描页本地 OCR 不产生 API 费用、不走网络；含图页走 OCR 是为了不丢图里的正文。
- **替代方案**：全量 OCR（成本与时间爆炸）、全量文本（扫描件识别不了）、云端 VLM OCR（每次调用都是钱和延迟）。
- **后果**：需精确区分文本/扫描/含图页，判据是 `_page_has_meaningful_images`；本地 RapidOCR 只做文字识别，不做版面结构化，也不产出图片语义描述——图片的语义描述由 `src/ingestion/parser_image.py` 单独走 SiliconFlow VLM 处理（仅图片入库时用）。OCR 引擎在模块级懒加载单例，避免每页重复加载模型。

## ADR-007：高敏承诺护栏 —— 价格/退款/政策承诺需更高忠实度

- **决策**：回答含价格/退款/政策承诺时，忠实度门槛 0.7→0.85，不达标转人工核验。
- **背景**："AI 瞎承诺政策"是电商客服最大事故源（跨境卖家 AI 承诺 7 天无理由实际 15 天，亏十几万）。
- **为什么**：高敏内容错一次就是"看起来很可信的瞎承诺"，比明显答错危害更大。
- **替代方案**：统一 0.7 阈值（高敏漏放行）、全量转人工（误伤正常回答）。
- **后果**：高敏回答多一道核验；已补"假一赔N"等非常规承诺正则。
- **📌 三个忠实度阈值的关系（容易说混）**：
  1. **0.8** —— 生成模块内部自纠正闭环的通过线（`SelfCorrector.generate_with_docs` 默认 `faithfulness_threshold=0.8`）：达不到就触发"定位缺失信息 → 改写重搜 → 重新生成"，最多 2 轮。
  2. **0.7** —— 图上 `evaluate_quality` 节点的基线放行线（普通内容）。
  3. **0.85** —— 同一节点对**高敏内容**（价格/退款/政策承诺）的门槛，由 `promise_guard.detect_high_stakes()` 命中后抬升。
  所以一个高敏回答忠实度 0.8 时：**内层放行（够到 0.8），外层拦截（不足 0.85）→ 转人工核验**。高敏内容是在两道关里被卡两次的。
  另有一个刻意留的口子：`loop_count` 达到上限时通常强制通过以避免死循环，但**含高敏承诺且忠实度不足时到上限也不放行**，仍转人工。

## ADR-008：Embedding 模型 —— qwen2.5-vl-embedding (2048 维)

> **状态：⚠️ 已被取代（现用本地 bge-small-zh-v1.5，512 维，见 ADR-010）。**
> 本节保留的是"在云端 qwen 系模型里怎么选、维度怎么定"的推理过程，当前配置已不适用。

- **决策**：向量模型用 qwen2.5-vl-embedding，维度保持 2048（支持 512~2048）。
- **背景**：原 qwen3-vl-embedding 免费额度耗尽，且不同进程间返回向量不一致。
- **为什么**：qwen2.5 多模态、2048 维兼容既有 Milvus collection（避免重建）；支持文本+图片统一向量空间。
- **替代方案**：换本地确定性模型（bge-m3，需大改+重灌）、text-embedding-v3（维度不同）。
- **后果**：每 API 调用限 1 个 input（qwen2.5 不支持批量，批量 20→1）；需保证索引与查询同模型同维度。

## ADR-009：内容级权限隔离 —— JWT 鉴权 + 层级 access_level（模块33）

- **决策**：① 鉴权用 JWT（`/api/auth/login` 签发，`get_current_user`/`require_admin` 依赖，身份只信 header 不放 body）；② 内容隔离用层级 `access_level: public(0) < member(1) < vip(2)`，入库打标签、检索 `access_level <= 用户等级` 过滤，缓存 key 含等级。
- **背景**：原系统无任何鉴权，普通用户与 VIP 用户看到的知识范围完全相同——多租户/权限隔离是 RAG 生产化必答考点。
- **为什么**：① 等级比较必须用 **INT8 整数**（字符串字典序 `"member"<"public"<"vip"` 语义错位，`access_level <= "member"` 会错误排除 public）；② 所有检索调用点默认 `public`（fail-safe 最低权限），漏传=只返回公开内容，永不泄漏；③ 缓存 key 必须含 access_level，否则 VIP 用户的检索结果（含全文）被低权限用户命中缓存=直接泄漏；④ 双路召回/多级降级/自纠正重搜/工作流空 docs 兜底必须逐层透传，任何一层漏掉都是泄漏路径。
- **替代方案**：role 标签列表（配置复杂）、retrieval 后 Python 过滤（越权内容仍进召回集/rerank/缓存，不彻底）。
- **后果**：Chroma 不支持嵌套 metadata，因此 `access_level` 被展平为**标量整数 rank**（0=public / 1=member / 2=vip）连同 `access_level_label` 一起写入每条记录的 metadata，检索时用 `where` 条件做 `access_level <= 用户rank` 过滤；已有 collection 用 `ChromaStore().ensure_access_level_field()` 补字段并回填 0（public），或 `create_collection(drop_if_exists=True)` 重灌；检索链路 7 个调用点需统一带 access_level；JWT 生产必须换长随机 secret。

### 迁移 runbook（已有数据升级到模块33）
1. **Chroma**：新库直接 `create_collection`；旧库调 `ChromaStore().ensure_access_level_field()`（补齐 `access_level` / `access_level_label` 标量字段并回填 0=public），或一次性 `create_collection(drop_if_exists=True)` 后用 `scripts/rebuild_kb.py` 重灌。重灌前旧文档默认 public，管理面重新上传 VIP 文档时显式传 `access_level=vip`。
2. **SQLite**（conversations.db）：`SessionManager._init_db` 自动 `ALTER TABLE conversations ADD COLUMN user_id`，防呆幂等，无需手工。
3. **旧检索缓存**：cache key 结构新增 access_level 段 → 旧 key 自然失效，无需手动清理。
4. **.env**：必须设置 `JWT_SECRET`（≥32 字节随机串）；demo 用户用 `DEMO_USERS` JSON 覆盖；CORS 用 `CORS_ORIGINS` 白名单。

## ADR-010：向量库与检索栈迁移 —— Milvus + 混合检索 → Chroma + 本地 BGE + 纯稠密

- **决策**：四件事一起换。
  ① 向量库：Milvus 2.6 → **Chroma**（本地持久化，`data/chroma/`）。
  ② Embedding：云端 qwen2.5-vl-embedding（2048 维，API）→ **本地 bge-small-zh-v1.5（512 维）**。
  ③ Reranker：云端 qwen3-vl-rerank（API）→ **本地 bge-reranker-base（sentence-transformers CrossEncoder）**。
  ④ 检索：BM25 + Dense 混合检索（WeightedRanker 0.3/0.7）→ **纯稠密检索 + 本地 CrossEncoder 精排**。

- **背景**：三个压力叠加。
  1. **成本与延迟**：云端 Embedding / Reranker 每次检索都产生 API 调用，费用和延迟随请求量线性增长，而检索是每个请求里调用最频繁的一环。
  2. **外部依赖风险**：额度耗尽、以及跨进程返回向量不一致（ADR-008 的背景里就记录过这个问题），直接威胁检索可用性——向量不一致意味着索引和查询不在同一空间里，检索会静默失效。
  3. **运维成本不匹配**：Milvus 需要独立部署一整套（etcd + MinIO + standalone 三个容器），而客服知识库只有几千个分块，运维代价远超收益。

- **为什么这么换**：
  - **Chroma**：本地持久化、零外部服务，部署从"三个容器"变成一个目录，且 `where` 过滤足够支撑权限隔离与元数据筛选。
  - **bge-small-zh 512 维**：中文检索效果好；维度只有原来的 1/4，存储与检索开销显著下降；本地推理无 API 成本、无网络延迟，且结果是确定性的——这一点直接消灭了上面第 2 条风险。
  - **CrossEncoder 精排**：query 和文档拼在一起逐对打分，本身同时看了语义与词法匹配，把混合检索里稀疏那一路要承担的作用覆盖掉了。所以保留混合检索的边际收益变得不明显，链路反而更复杂——**这是砍掉 BM25 最关键的理由，不是"因为换了库所以顺手砍掉"**。

- **替代方案**：继续用 Milvus（运维重）；纯 BM25 或纯向量单路（各有盲区，见 ADR-001）；bge-m3（效果更强但资源占用高、需全量重灌）；保留混合检索但换成本地 BM25 实现（jieba + 自建倒排，工作量不小而收益存疑）。

- **后果**：
  - 检索侧全链路**无外部 API 依赖**；Embedding / Reranker 均为懒加载，首次调用才载入模型，不拖慢启动。
  - **关键约束：索引与查询必须同模型同维度**。换 embedding 模型 = 必须重灌整个知识库（`scripts/rebuild_kb.py`），否则检索静默失效。
  - **相似度阈值量纲变了，这是最容易说错的地方**：Milvus 内积时代的 `0.7` 在 bge 余弦量纲下不再适用，检索阈值改为 `0.4`（`RETRIEVAL_SIMILARITY_THRESHOLD`）；而离线评测里判断"这篇文档算不算相关"用的是 embedding 余弦 `0.70`（`metrics.RELEVANCE_THRESHOLD`）。**两个 0.7 不是一回事，两个阈值也不要混用。**
  - bge-small-zh 是**纯文本模型**，不支持图片向量化。图片入库需先经 VLM 转成文字描述（原 `embed_image` 能力已移除），所以"多模态检索"从 roadmap 的可选项变成了必须先补 VLM 描述链路。
  - **模型权重需要随部署提供**：Dockerfile 只复制 `src/`，容器内必须挂载 `local_models/` 并配置 `EMBEDDING_MODEL_PATH` / `RERANKER_MODEL_PATH`，否则启动直接报错。
  - **遗留清理已完成（2026-10）**：`src/embedding/milvus_store.py`、`src/embedding/synonyms.py`、`docker-compose-milvus.yml` 及对应测试已删除；`config.py` 的 `milvus_*` / `dashscope_*` / `jieba` 等死配置与依赖已移除；`docker-compose-rag.yml` 的 `MILVUS_*` 环境变量与 `external: true` 的 `milvus` network 已换成本地 `rag-net` bridge（原配置在网络不存在时会直接导致 compose 启动失败）。至此代码与本文档口径完全一致。

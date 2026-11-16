# FAQ 文档 ID 检索基线 v1

默认命令 `python scripts/run_benchmark.py` 现在运行文档 ID 基线。
历史相似度评测须显式加 `--legacy-similarity`，历史数字与本基线不能直接比较。

## 数据与标注

- 保留原 60 条 FAQ 的问题、答案和关键词，为每条分配 `faq-001` 等稳定 ID。
- 保留原 49 条题目与标准答案，为每题增加 `case_id`、`relevant_doc_ids`、`retrieval_status` 和 `annotation_note`。
- 标注依据是当前 FAQ 原文的相关性，未用向量相似度自动判定。ID 一经分配不随重排改变；新增条目使用新 ID。
- 34 题 `answerable` 计入检索指标；6 题 `partial_evidence`、9 题 `out_of_scope` 在逐题报告中列出并排除评分。报告同时展示总题数、评分题数和各状态数量。
- `partial_evidence` 指知识库只支持标准答案的部分内容，例如退款渠道、订单退款状态和热线时间；不能用相似内容冒充完整证据。
- `out_of_scope` 包括实时物流、闲聊及无对应知识的问法。v1 不评测它们的拒答/路由能力，也不因排除它们声称系统整体质量提高。
- 一题可对应多个相关 FAQ，包括同义条目；标注不意味着每项必须同时检索才能回答。Recall 衡量找回多少已标注条目，Hit 衡量至少找到一条。复合问题后续需要另标必要证据组。
- `annotation_note` 记录边界。标注由本次代码审查建立，尚未经过独立业务人员复核；知识库中的安全认证、赔付等示例政策也未做业务真实性核准。

## 指标口径

按 Top-5 返回的固定 FAQ ID 判断二值相关性，不调用大模型裁判或 Embedding 判卷：

| 指标 | 定义 |
|---|---|
| Hit@5 | 至少命中一个相关 ID 为 1，否则为 0 |
| Recall@5 | 命中的不同相关 ID 数 / 全部已标注相关 ID 数 |
| Precision@5 | 命中的不同相关 ID 数 / 5；不足 5 条不缩小分母 |
| MRR@5 | 首个相关 ID 排名的倒数；Top-5 未命中为 0 |
| NDCG@5 | 二值 DCG / 根据完整相关 ID 集计算的 IDCG |

同一个 FAQ 的重复结果占用排名，但只贡献一次相关性。汇总为评分题目的宏平均。
空相关集合不计算检索指标。缺失、重复、未知 ID 或标注状态错误直接报错。
旧相似度指标作为历史辅助工具保留，不再与这些指标使用同一默认入口。

## 运行

安装项目依赖，并在环境变量或 `.env` 中设置本地模型目录：

```powershell
python scripts/run_benchmark.py --validate-only
python scripts/run_benchmark.py --no-rerank
python scripts/run_benchmark.py
```

可用 `--limit 10` 冒烟、`--dataset <path>`/`--corpus <path>` 指定快照、`--save-dir <dir>` 指定报告目录。
生成旧版结果须显式运行：

```powershell
python scripts/run_benchmark.py --legacy-similarity
python scripts/run_benchmark.py --legacy-similarity --with-generation --use-llm-eval
```

默认基线直接复用项目的 Embedder、ChromaStore 和 Reranker。每次在新目录中索引固定 60 条 FAQ，一个问答一个 chunk；文本采用 FAQ 解析器的 Q/A 与关键词格式，FAQ ID 仅放元数据，不进入模型输入。
它不复用生产索引，不调用应用缓存，不清空生产数据，也不调用外部 LLM。
它是固定语料检索组件基线，不能替代生产 Retriever 的缓存、政策过滤、双路召回和完整聊天工作流评测。
模型、索引或重排失败时终止，不静默退回另一个算法输出“有效成绩”。

## 报告与复现边界

每次报告在 `reports/id_baseline_<timestamp>/report.json`，同目录 `index/` 是独立索引。
报告保留全部题目、排除原因、Top-5 文本/ID/分数、命中/漏检 ID、零命中题目，以及数据 SHA-256、模型目录文件哈希、关键代码哈希、Git HEAD/脏状态、包版本与检索参数。
模型加载与索引构建不计入查询延迟，预热后测量 query embedding + Dense + 可选重排；该延迟不是聊天接口 SLA。
基线固定在 CPU 运行，PyTorch seed=0、线程数=4，参数随报告记录。
同一输入与运行环境应得到相同排序与质量指标；硬件、底层模型计算和近似索引可能引入浮点或同分排序差异，延迟不保证相同。
跑分时若工作区未提交，报告会明确记 `dirty=true`，以代码文件哈希识别实际版本；不把旧 HEAD 当成已提交的新代码。

仓库中的可审阅参考结果放在 `docs/baselines/`，运行产生的索引和临时环境不提交。

## 2026-11-16 参考结果

以下均为保留的 49 题中 34 道可评分题的宏平均，使用相同的冻结标注和 FAQ 快照：

| 配置 | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | 查询平均耗时 |
|---|---:|---:|---:|---:|---:|
| Dense | 1.000000 | 0.985294 | 0.965686 | 0.958493 | 16.873 ms |
| Dense + 重排 | 1.000000 | 0.990196 | 1.000000 | 0.990738 | 1419.247 ms |
| Dense + 重排复跑 | 1.000000 | 0.990196 | 1.000000 | 0.990738 | 1405.021 ms |

两次重排运行的逐题 Top-5 ID 与所有质量指标完全一致。报告分别为
[Dense](baselines/2026-11-16-dense.json)、[重排](baselines/2026-11-16-rerank.json)、[复跑](baselines/2026-11-16-rerank_repeat.json)。
Dense 在 eval-031 漏掉 faq-030；重排在 eval-025 漏掉 faq-011，但都找到了其他相关 FAQ，因此 Hit@5 为 1 而 Recall@5 未达 1。
现有题集容易且语料仅 60 条，这些数字不代表线上准确率；本次也没有改变检索算法来追求高分。
重排在该小型题集上提升有限而耗时明显增加，后续应在更难、独立标注的测试集上验证其收益，再决定是否按问题启用。
验证包含 43 项评测及相关检索回归测试、真实 Chroma ID 往返测试、源码 Ruff 检查；未运行完整项目测试套件。

## 下一步

保持这批标注固定，逐条分析漏检案例，再追加独立的口语、多轮、无答案测试集。
业务闭环的下一项是让普通聊天与流式聊天复用同一工作流和质量核验，然后为两个入口增加相同行为的回归测试。

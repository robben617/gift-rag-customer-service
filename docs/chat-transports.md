# 普通聊天与 SSE 共用工作流

两个接口现在都调用 `src/api/chat_service.py` 中的 `execute_chat`：

```text
JSON / SSE → 共享对话服务 → 脱敏 → 用户会话与记忆 → LangGraph
           → 意图/人工判断 → RAG 或业务工具 → 质量核验
           → 安全回复 → 保存会话/记忆/监控 → JSON 或 SSE 编码
```

路由不再分别实现检索、生成和情绪分流。工具查询、权限、订单归属、政策回答、质量拒绝和人工交接使用同一套结果。
原流式入口特有的“跳过重检索”策略暂不接入共享图，避免跳过后没有证据仍自由生成；多轮追问统一通过图中的指代消解与检索处理。

## 回复何时可见

SSE 先发送状态，完整工作流结束后再分段传输最终回复。
`token` 事件现在表示经过核验的最终文本片段，并非大模型生成中的原始 token。
此选择增加正文开始输出前的等待时间，避免未核验的退款、金额等承诺先展示再被判错。
等待期间每 10 秒发送 `: keep-alive` 注释，保持连接活跃。

若图中断等待人工，或 `evaluation_passed=false`，共享服务统一返回人工核验提示和 `needs_human=true`，不将待审核草稿存入用户会话或记忆。
原草稿留在持久化的 LangGraph checkpoint；管理员通过审批接口核验并恢复图执行，详见 [回答审批](workflow-reviews.md)。
工作流错误返回通用繁忙提示，SSE 编码侧的意外错误也不返回原始异常信息。

## 接口兼容性

- `POST /api/chat` 继续接收 `ChatRequest` JSON，返回原 `ChatResponse` 字段。
- `POST /api/chat/stream` 继续通过 query 参数接收 `query`、`session_id`、`top_k` 和 `use_reranker`，并通过 Bearer JWT 鉴权。
- SSE 的 query 长度现在与普通聊天一致，为 1～1000 字符。
- 保留 `data: {"event": "...", "data": ...}` 格式及 `[DONE]` 结束标记。

SSE 事件顺序：

| 事件 | 数据 |
|---|---|
| `status` | 处理与核验状态 |
| 心跳注释 | 等待期间可重复出现，不是 JSON 事件 |
| `sources` | 最多三个去重后的证据文件名 |
| `emotion` | 与普通接口相同的情绪字段 |
| `handoff` | 需要人工时出现，包含 reason、priority、payload |
| `token` | 最终回复的文本片段，可多次出现 |
| `result` | 完整 `ChatResponse`，与普通接口业务字段相同 |
| `done` | `[DONE]`，正常结束时恰好一次 |

SSE 编码侧异常时，发送 `error` 和 `done`，不再继续输出 token/result。
已有客户端可继续拼接 `token`；使用 `result` 的客户端应将其作为最终状态，避免再次追加同一段 reply。

## 并发与监控

同步模型、图和 SQLite 操作通过 `asyncio.to_thread` 执行，避免阻塞 FastAPI 事件循环。
同一进程内，同一用户和 session 的请求串行处理，防止两个入口同时修改同一历史或图状态；不同会话可并发处理。
会话线程锁使用弱引用，闲置后可回收；同机多进程另通过共享数据目录中的 OS 文件锁协调聊天与审批。详情见 [回答审批](workflow-reviews.md)。

LLM usage 改为请求上下文中的独立累加器；LangGraph 复制到子执行上下文时共享该请求的累加器，并用锁保护计数。不同请求 reset 不会清空其他请求的 token 数。
监控的 `total_time_ms` 改为共享服务实际处理耗时，包含会话/记忆处理，不再用检索耗时代替。
该耗时不含等待会话锁及网络发送时间。服务原有 token 估算回退仍保留，不将估算值作为精确账单。

断开 SSE 会取消等待和发送任务，但 Python 无法强制终止已经在线程中运行的同步工作流；该工作流可能完成并写入本轮历史，重新连接时应继续使用相同 session。

## 验证

新增接口测试覆盖 JSON/SSE 字段一致性、政策/订单/RAG/人工分支、真实 LangGraph 静态中断、未批准草稿隔离、PII、鉴权、订单归属、一次性保存、心跳、断开清理和事件循环不阻塞。
并发测试覆盖同一会话串行、不同会话并行、请求 token 隔离，以及真实图节点的 usage 能回到监控。
这些测试运行真实 LangGraph 和临时 SQLite，外部 LLM、数据库查询和检索内容使用受控替身，不消耗 API Key 或修改业务数据库。

```powershell
python -m pytest tests/api tests/graph tests/conversation tests/engineering tests/business tests/orders tests/evaluation -q
```

“待审批任务 → 管理员批准/拒绝 → 更新同一图状态 → 恢复执行”的持久化闭环已接入；直接转人工的普通工单仍沿用原处理流程。

# 界面截图

本目录存放 README 首屏引用的界面截图，以及重新生成它们的步骤。

| 文件 | 界面 | 本地地址 |
|------|------|----------|
| `chat-ui.png` | 用户聊天 | http://localhost:7861 |
| `admin-ui.png` | 管理员控制台 | http://localhost:7862 |
| `upload-ui.png` | 知识库入库 | http://localhost:7860 |
| `api-docs.png` | FastAPI Swagger UI | http://localhost:8000/docs |
| `demo-offline.png` | 零 Key 离线演示（`scripts/demo_offline.py`） | http://localhost:7863 |
| `architecture.png` | 系统架构图（由 `docs/architecture.mmd` 渲染） | — |
| `docs/graphs/langgraph_workflow.png` | LangGraph 13 节点流程图 | — |

## 重新生成界面截图

1. 按 [使用教程](../使用教程.md) 或 README「快速开始」把服务起起来（Docker 或本地均可）。
   - 三个 Gradio 界面默认开启 Basic Auth（`GRADIO_USERNAME` / `GRADIO_PASSWORD`），截图前先登录。
   - 想免登录截图，可临时用各 UI 的工厂函数启动：`create_chat_ui()` / `create_admin_ui()` /
     `ProcessorApp().create_interface()`，它们不带 auth。
2. 浏览器窗口宽度建议 1440px；三张界面图高度 900–1000px 即可。
3. 截图时注意这几点，图才有信息量：
   - `chat-ui.png`：先打开界面上的**调试模式**开关，再问一句会走转人工的问题（如「我要退款」）。
     这样一张图里能同时呈现意图、置信度、转人工原因与承诺话术；否则只有一个问答气泡，信息量偏低。
   - `admin-ui.png`：先造一条待处理工单再截，避免截到空表格。
   - `api-docs.png`：**只截可视区域**。整页会到 3000px 以上，压进 README 后字会看不清；
     保留页面头部（API 名称与能力简介）+ 前 3 个接口分组即可。
   - `demo-offline.png`：跑 `python scripts/demo_offline.py` 后问一句**高风险话题**（如「退款要等多久」），
     这样图中同时有「命中原文」与「完整链路会转人工」的护栏提示。
4. 覆盖写入同名文件即可，README 会自动引用。

## 两个容易踩的坑

- **三个 UI 必须各起一个进程。** Gradio 不支持在同一进程内并行 `launch()` 多个 `Blocks`：
  多个应用共享同一套路由注册表，请求打到任意端口都可能解析到「最后注册」的那个应用，
  表现出来就是某个界面渲染成空壳（实测 `:7861` 曾返回管理台的组件）。
- **`/docs` 依赖 CDN 资源。** Swagger UI 的 JS/CSS 来自 `cdn.jsdelivr.net`。
  若本机网络需经代理才能出网，无头浏览器要显式指定代理，否则截到的是空白页。

## 两张图表的来源

- `architecture.png`：源文件是 [docs/architecture.mmd](../architecture.mmd)，用 mermaid-cli 渲染为 PNG。
  README 中已说明：GitHub 原生 Mermaid 渲染该图会报布局错误，因此仓库内保留 PNG。
- `docs/graphs/langgraph_workflow.png`：由 `scripts/generate_graph.py` 从 LangGraph 编译产物导出。

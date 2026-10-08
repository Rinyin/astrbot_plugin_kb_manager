# 知识库自主运维（astrbot_plugin_kb_manager）

面向 AstrBot **管理员**的插件：在聊天会话中通过自然语言让 LLM 直接管理 AstrBot 原生知识库——创建/更新/删除知识库、导入文本/会话附件/公网网页、维护分块、检索内容，并可查询后台任务进度。

仓库地址：<https://github.com/Rinyin/astrbot_plugin_kb_manager>

## 环境要求

- AstrBot：`>=4.28.2,<5`（见 `metadata.yaml` 的 `astrbot_version`）。
- Python：3.12+（与 AstrBot 运行环境一致）。
- 一个支持工具调用的聊天模型：用于理解自然语言指令并调用本插件工具。
- 一个可用的嵌入模型：用于知识库向量化；聊天模型不会被当作嵌入模型使用。
- 使用本插件工具的用户必须是 AstrBot 管理员。

## 安装

1. **插件文件夹方式**：将 `astrbot_plugin_kb_manager` 整个文件夹放入 AstrBot 的 `data/plugins/` 目录，然后在 WebUI「插件管理」中重载并启用。
2. **从仓库安装**：在 WebUI「插件管理」中选择从仓库 / URL 安装，填写 `https://github.com/Rinyin/astrbot_plugin_kb_manager`；也可以在 `data/plugins/` 目录执行 `git clone https://github.com/Rinyin/astrbot_plugin_kb_manager.git` 后重载插件。
3. **ZIP 导入**：在 WebUI「插件管理」中上传 ZIP 安装。压缩包根目录应为 `astrbot_plugin_kb_manager/`，并包含 `metadata.yaml` 与 `main.py`。
4. 插件的第三方依赖由 `requirements.txt` 声明；依赖缺失时由 AstrBot 联网安装。

## 配置

在 WebUI「插件管理 → 知识库自主运维 → 配置」中设置：

| 配置项 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `default_embedding_provider_id` | 字符串 | 空 | 默认嵌入模型提供商 ID，可从 `kbm_list_kbs` 返回的候选列表中获取；也可留空，由 LLM 在创建知识库时选择（恰好只有一个可用嵌入模型时自动选择，多个候选时由工具返回候选供 LLM 选择）。此处填写的是嵌入模型，不是聊天模型。 |
| `max_file_mb` | 整数 | 20 | 单个文件/会话附件大小上限（MiB）。 |
| `max_url_mb` | 整数 | 5 | 公网网页下载与提取正文的大小上限（MiB）。 |
| `attachment_ttl_minutes` | 整数 | 30 | 会话附件记录的有效期（分钟）；插件重载后记录清空。 |
| `http_timeout_seconds` | 整数 | 30 | 网页抓取总超时（秒，含重定向）。 |
| `max_concurrent_jobs` | 整数 | 3 | 后台写任务最大并发数。 |

所有整数配置必须为正整数；本插件不提供任何绕过管理员权限的开关。

## 使用示例

- “列出所有知识库”
- “新建一个叫‘产品手册’的知识库，使用默认嵌入模型”
- “把我刚发的这个 PDF 导入‘产品手册’，导入完告诉我”
- “帮我导入 https://example.com/article 到‘产品手册’”
- “在‘产品手册’里搜一下‘保修政策’”
- “看看‘产品手册’里有哪些文档”
- “读出文档 doc_xxx 的前 20 个分块”
- “把文档 doc_xxx 整篇替换为以下新文本：……”
- “更新 doc_xxx 里第 3 个分块的内容为……”
- “刚才那个导入任务跑完了吗？job_id 是……”

写操作会由 LLM 自动附带 `request_id` 用于幂等重试：同一请求重试复用同一个 `request_id`，不会重复写入；用户无需手动书写。修改参数后需使用新的 `request_id`（LLM 会自行生成）。

## 工具一览（17 个）

| 工具 | 说明 |
| --- | --- |
| `kbm_list_kbs` | 列出知识库，并返回可用嵌入模型与默认项 |
| `kbm_create_kb` | 创建知识库 |
| `kbm_update_kb` | 更新知识库描述/分块参数 |
| `kbm_delete_kb` | 删除知识库及其文档 |
| `kbm_list_documents` | 分页列出文档（支持文件名过滤） |
| `kbm_read_document` | 读取文档元数据与分块 |
| `kbm_search` | 在指定知识库内检索相关分块 |
| `kbm_add_text` | 以文本内容新增文档 |
| `kbm_list_attachments` | 列出当前会话可见的近期附件（不下载） |
| `kbm_import_attachment` | 导入当前会话的附件 |
| `kbm_import_url` | 抓取公网网页正文并导入 |
| `kbm_replace_document` | 以新文本整篇替换文档 |
| `kbm_delete_document` | 删除文档 |
| `kbm_add_chunk` | 添加分块 |
| `kbm_update_chunk` | 更新分块 |
| `kbm_delete_chunk` | 删除分块 |
| `kbm_job_status` | 查询后台任务状态与进度 |

## 行为与边界

- **权限与隔离**：所有工具仅管理员可用；附件与任务按“同一会话 + 同一发送者”隔离，其他会话或发送者不可见、不可访问。
- **附件**：只接受聊天消息中真实携带的文件；单附件默认上限 20 MiB；记录默认保留 30 分钟，插件重载后清空；不会删除平台上的原文件。
- **网页**：只允许公网 HTTP(S)，默认 5 MiB、30 秒、最多 5 次重定向；仅支持静态 HTML/纯文本正文提取，不使用 Tavily，不运行浏览器脚本；拒绝内网地址与带用户名密码的 URL。
- **替换语义**：整篇替换会生成**新的 doc_id**（不保留旧 ID）；分块更新会生成**新的 chunk_id**，文档 ID 与分块序号保持不变。旧内容保留到新内容写入成功为止；`partial`/`interrupted` 状态不能当作成功。
- **检索**：使用本插件自带的 `kbm_search`；把知识库绑定到聊天会话的 RAG 行为由 AstrBot 自身配置控制，与本插件相互独立。
- **原文保留**：AstrBot 不保留原始上传文件，检索与读出的是切分后的知识分块。
- **导入格式**：文本/文档导入以 AstrBot 原生解析器支持范围为准（如 `.md` / `.markdown` / `.txt` / `.rst` / `.adoc` / `.docx` / `.xlsx` / `.xls` / `.pdf` / `.epub` 等）；网页正文会先提取为 UTF-8 文本再导入。
- **任务**：耗时写操作超过等待时间会返回 `job_id`，用 `kbm_job_status` 查询进度；进程重启后中断的任务不会自动重放。
- **数据目录**：运行状态（任务数据库、附件缓存）位于 AstrBot 插件数据目录，与插件代码目录分离。
- **写入一致性**：不承诺与 WebUI 同时修改同一个知识库时的跨存储原子性，请避免并行写入同一知识库。
- 本插件不会自动学习全部聊天记录，也不会修改 AstrBot 本体。

## 许可

AGPL-3.0-or-later，详见 [LICENSE](LICENSE)。

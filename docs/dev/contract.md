# KB 管理插件 · 模块契约 v0.3

- 目标运行时：AstrBot v4.28.2（基线提交 `3c7adafa1397e182d60b1016bf88759265113c8a`）。
- 对应插件版本：`0.2.0`（LLM 自主知识运维）。
- 关联实现：`common.py`、`backend.py`、`sources.py`、`jobs.py`、`autonomy.py`、`main.py`。
- 本文档只记录已确定的裁决，不留开放项；裁决直接写在下文。

本契约固定各模块的签名、语义与数据形状。实现内部结构不受限，但任何跨模块可见的接口不得偏离本契约；偏离必须先修订本文件并同步 `common.py`。

---

## 0. 范围与约定

### 0.1 本契约固定

1. `common.py` 公共原语：`KBError`、`SourceDocument`、Scope 编码、任务状态、结果信封、错误码、幂等/锁键。
2. `backend.py` → `NativeKBBackend`：方法签名、语义与返回 dict 形状。
3. `sources.py` → `SourceManager`：构造与方法签名、语义。
4. `jobs.py` → `JobManager`：构造、`submit`/`get`/`close`、状态机、幂等、并发与持久化。
5. `autonomy.py` → 自主维护 SYSTEM 提示：`DEFAULT_AUTONOMY_PROMPT`、`inject_autonomy_prompt`、标记块与幂等替换。
6. `main.py` 入口层：工具暴露与返回、scope 生成、附件记录、SYSTEM 注入入口与自主运维触发边界（见 §6）。

### 0.2 实现约束

- 实现文件（`backend.py`/`sources.py`/`jobs.py`/`main.py`）只消费本契约，不得私自扩展跨模块可见的签名或数据形状。
- 接口变更必须同步修订本文件；涉及 `common.py` 原语时一并修改，并按 §7 说明兼容性影响。

### 0.3 已决事项（不再开放）

- 时间戳统一为 **Unix epoch 毫秒（int，UTC）**。
- 写请求幂等键为 `(scope, request_id)`，`request_id` 由 **LLM 工具参数提供并在重试时复用**（见 §2.6）。
- 任务记录持久化在插件数据目录的 **SQLite** 数据库；第一版 **不自动清除**幂等记录。
- 写操作仅针对 **单库**；创建库共用创建锁，**不实现多库写锁协议**。
- 打包元数据（`metadata.yaml`/`requirements.txt`/`_conf_schema.json`）属于插件必需交付。
- **自主运维触发边界（v0.2.0，已定）**：SYSTEM 只在 **AstrBot 已产生的正常 LLM 请求**上注入；不为没有产生模型请求的消息（如未被唤醒的群消息）额外发起后台 LLM 判断。消息监听仅用于登记真实附件，不发起额外模型调用。
- **配置项（v0.2.0，已定）**：`_conf_schema.json` 保留原 6 项及其默认值不变，仅新增唯一一项 `autonomy_system_prompt`（`type=text`，默认值即内置 `DEFAULT_AUTONOMY_PROMPT`）；缺失 / 空白 / 非字符串 → 使用内置默认。不引入基于用户身份的维护开关，工具**不再按管理员身份门禁**。
- **运行器边界（v0.2.0，已定）**：本插件的本地工具仅由 AstrBot 内置 Agent（`local` runner）执行；Dify / Coze / DashScope / DeerFlow 等第三方 Agent 运行器不执行这些本地工具，因此不适用自主维护。聊天模型可以是远程 API，与运行器类型无关。
- 可信 scope 隔离（UMO + sender_id）与索引一致性语义**不因自主触发而放松**；17 个工具签名与存储 API 保持不变。

---

## 1. 模块布局

```text
astrbot_plugin_kb_manager/
├── main.py       # 插件入口：命令 / LLM 工具、SYSTEM 注入与生命周期
├── backend.py    # NativeKBBackend
├── sources.py    # SourceManager
├── jobs.py       # JobManager
├── autonomy.py   # 自主维护 SYSTEM 提示（纯函数，仅标准库）
├── common.py     # 公共原语（仅标准库）
├── metadata.yaml / requirements.txt / _conf_schema.json
├── README.md / LICENSE
├── tests/        # 单元测试与真实原生回归
└── docs/dev/     # 开发文档；本地报告与审查记录不入库
```

- `common.py` 只依赖标准库，禁止导入 AstrBot 或插件内其他模块；其余模块统一从这里取公共类型。
- 插件代码只放插件目录；**运行状态（含 SQLite、附件缓存）放插件数据目录**，不放代码目录。

---

## 2. 公共原语（`common.py`）

### 2.1 `KBError`

```python
class KBError(Exception):
    def __init__(self, code: str, message: str, details: Any = None, partial: bool = False) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...            # {"code", "message", "details", "partial"}
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KBError": ...
```

规则：

- 跨模块错误只允许抛 `KBError`；未知异常用 `as_kb_error()` 归一化。
- `details` 必须 JSON 可序列化，默认 `None`；禁止放 `bytes`。
- `partial=True` 表示“部分完成”，与结果信封组合时映射为 `partial` 状态。

### 2.2 `SourceDocument`

```python
@dataclass
class SourceDocument:
    filename: str
    content: bytes
    source: str        # SOURCE_FILE / SOURCE_URL / SOURCE_ATTACHMENT
    @property
    def size(self) -> int: ...
    def to_meta(self) -> dict[str, Any]: ...
```

- 构造即校验：`filename`/`source` 为非空字符串，`content` 为 `bytes`。
- `bytes` 是内部载体，**禁止直接进入结果信封**（信封只允许 JSON 类型）。

### 2.3 Scope 编码

- 形态：`json.dumps([umo, sender_id])` 的紧凑 JSON 数组字符串。
- 生成职责在入口层：`encode_scope(event.unified_msg_origin, str(event.get_sender_id()))`。
- **scope 只能由可信事件的 UMO 与 sender_id 生成，任何情况下不得由 LLM 参数、用户文本或模型输出提供/覆盖。** 所有任务层接口的 `scope` 参数均指此编码串。
- 提供 `encode_scope` / `decode_scope` / `Scope`（frozen dataclass，含 `encode()`/`decode()`）。
- 解码失败抛 `invalid_argument`。

### 2.4 统一结果信封

所有 `JobManager` 对外返回值与入口层对用户的返回统一为：

```json
{
  "status": "succeeded",
  "job_id": "…或 null",
  "data": { "…": "…" },
  "error": null
}
```

- 四个键 `status`/`job_id`/`data`/`error` **固定且恒存在**；`data`、`error`、`job_id` 可为 `null`。
- `status ∈ {queued, running, succeeded, failed, partial, interrupted}`。
- `error` 为 `KBError.to_dict()` 的形状：`{"code","message","details","partial"}`。
- 构造器：`make_result` / `ok_result` / `error_result`。

### 2.5 任务状态机

```text
queued ──▶ running ──▶ succeeded     （work 正常返回 dict）
                    ├─▶ failed        （KBError(partial=False) / 未知异常）
                    ├─▶ partial       （KBError(partial=True)）
                    └─▶ interrupted   （取消 / close / 重启恢复）
```

- 终态集合：`{succeeded, failed, partial, interrupted}`（`is_terminal`）。
- 非终态：`{queued, running}`。
- 重启只把非终态置为 `interrupted`，**不重放**。

### 2.6 幂等、request_id 与锁

- `request_id` 是每项写工具 **必填的 LLM 参数**，允许 LLM 提供并在重试时复用；它 **不是权限凭据**。**不得**为每次调用随机生成，否则模型重试不再幂等。
- 幂等键：`request_key(scope, request_id)`；通过 `(scope, request_id)` 唯一约束落库。
- 指纹：`fingerprint({"operation": op, "payload": payload})`（`sort_keys` 稳定序列化）。
- 规则：同键同参 → 复用原任务/结果；同键不同参 → `idempotency_conflict`（`error.details.existing_job_id`）。
- `job_id` 可作状态查询工具参数，查询必须校验 scope；**不允许用户指定新任务的 `job_id`**。
- 锁键：单库写用 `kb_lock_key(kb_id) -> "kb:<kb_id>"`；创建 KB 用 `LOCK_GLOBAL = "kb:*"` 串行。第一版不实现多库写锁协议。

### 2.7 JSON 规则

- 所有跨模块 dict 必须可经 `ensure_jsonable`、`json_dumps` 严格序列化（拒绝 `bytes`、非字符串键、`NaN`/`Inf`）。
- 时间戳统一为 **Unix epoch 毫秒（int，UTC）**；接口允许 `null`。
- 各类 ID（`kb_id`/`doc_id`/`chunk_id`/`attachment_id`/`job_id`）为不透明非空字符串，调用方不得解析其结构。

### 2.8 错误码表

| code | 含义 |
| --- | --- |
| `invalid_argument` | 参数缺失、类型错误、越界 |
| `internal` | 未归类的内部错误 |
| `unavailable` | 任务管理器未初始化或已关闭 |
| `interrupted` | 任务被取消 / 进程重启 |
| `kb_not_found` | 知识库不存在 |
| `document_not_found` | 文档不存在 |
| `chunk_not_found` | 分块不存在 |
| `job_not_found` | 任务不存在或不属于该 scope |
| `attachment_not_found` | 附件未记录 / 不存在 |
| `attachment_expired` | 附件超过 TTL |
| `idempotency_conflict` | 同 `(scope, request_id)` 参数不同 |
| `storage_failed` | 任务已完成但结果无法持久化，结果不确定 |
| `payload_too_large` | 文件或 URL 超过大小上限 |
| `unsupported_source` | 不支持的来源类型（如非 http/https URL） |
| `url_fetch_failed` | URL 下载失败（网络 / 非 2xx） |
| `backend_unavailable` | 后端未初始化或依赖不可用 |
| `embedding_failed` | embedding 调用失败 |
| `embedding_provider_missing` | 无法解析 embedding provider |
| `name_conflict` | 知识库重名等命名冲突 |

实现可新增错误码，但不得复用上表语义。

---

## 3. `NativeKBBackend`（`backend.py`）

### 3.1 构造与初始化

```python
NativeKBBackend(context, default_embedding_provider_id="")
```

- `context`：AstrBot Context。
- `default_embedding_provider_id`：插件默认嵌入模型 ID，可为空。
- `initialize()` **只检查 Context 中现有知识库管理器的能力与可用性**：不创建第二套管理器、不初始化或关闭框架持有的共享资源；幂等，重复调用安全。

### 3.2 通用规则

- 全部方法为 `async`，返回 JSON 可序列化 `dict`（即信封的 `data`）；失败抛 `KBError`。
- 参数为空/类型错误 → `invalid_argument`；对象缺失 → `kb_not_found` / `document_not_found` / `chunk_not_found`。
- 返回 dict 必须包含下列 required 键；允许附加键而不破坏契约。
- `progress=None` 时实现必须安全跳过；被提供时调用方式为 `await progress(stage, current, total)`。
- **不得伪造跨数据库 / FAISS 的原子事务**：新内容成功后再移除旧内容；删除先处理向量、再处理文本/FTS 与元数据，最后刷新原生统计。

### 3.3 返回形状

以下形状中的时间戳为 epoch 毫秒或 `null`。

**KBSummary**：required `kb_id`、`name`、`description`、`embedding_provider_id`、`chunk_size`、`chunk_overlap`；optional `document_count`、`chunk_count`、`created_at`、`updated_at`。

**DocSummary**：required `doc_id`、`kb_id`、`filename`、`size`、`chunk_count`；optional `media_type`、`created_at`、`updated_at`。

**ChunkView**：required `chunk_id`、`doc_id`、`index`（int，文档内顺序）、`content`；optional `created_at`、`updated_at`。

**SearchHit**：required `kb_id`、`doc_id`、`chunk_id`、`content`、`score`；optional `filename`、`index`、`metadata`（dict）。
`score` 保留原生检索结果语义（越大越相关），**不承诺不同嵌入模型之间绝对分数可比较**；结果必须保留原生资源 ID，并校验归属（仅返回请求 `kb_ids` 内的对象）。

**AttachmentInfo**（SourceManager 复用）：required `attachment_id`、`filename`；optional `size`、`media_type`、`expires_at`。

**分页**：`offset >= 0`；`limit` 取值 1..100（默认 20）；越界抛 `invalid_argument`；响应带 `total`（过滤后总数）。

### 3.4 方法逐个约定

| 方法 | 返回 `data` required 键 | 关键语义 |
| --- | --- | --- |
| `initialize()` | `initialized: true` | 仅校验现有 KB 管理器能力（见 3.1）；幂等 |
| `list_kbs()` | `kbs: [KBSummary]`、`embedding_providers`、`default_embedding_provider_id` | `embedding_providers` 为 `[{"id","name"}]`，只含真正的嵌入模型（**不含聊天模型**） |
| `create_kb(name, description="", embedding_provider_id="", chunk_size=512, chunk_overlap=50)` | `kb: KBSummary` | 嵌入模型选择顺序：显式指定 → 插件默认 → 唯一可用嵌入模型；否则抛 `embedding_provider_missing` 并在 `details.candidates` 返回可用候选。`name` 非空；重名抛 `name_conflict`；`chunk_overlap < chunk_size` |
| `update_kb(kb_id, description=None, chunk_size=None, chunk_overlap=None)` | `kb: KBSummary` | 仅更新非 `None` 字段；`chunk_size/overlap` 只影响后续文档 |
| `delete_kb(kb_id)` | `deleted: true`、`kb_id` | 先清理文档与媒体记录，再移除索引/元数据；不存在抛 `kb_not_found` |
| `list_documents(kb_id, offset=0, limit=20, search="")` | `documents: [DocSummary]`、`total`、`offset`、`limit` | `search` 对文件名子串过滤；空串不过滤 |
| `read_document(kb_id, doc_id, offset=0, limit=20)` | `document: DocSummary`、`chunks: [ChunkView]`、`total`、`offset`、`limit` | 按 `index` 升序返回分页后的分块 |
| `search(query, kb_ids, top_k=5)` | `query`、`results: [SearchHit]` | `query` 非空；`kb_ids` 非空列表；`top_k` 1..50（默认 5）；越界抛 `invalid_argument`；分数语义见 3.3 |
| `add_document(kb_id, filename, content, progress=None)` | `document: DocSummary`、`chunk_count` | 切块 + embedding + 入库；进度阶段见 `PROGRESS_STAGES`。元数据提交后若统计刷新失败：检查已知 `doc_id`，抛 `KBError(partial=True)` 并在 `details` 给出可检查的资源 ID；无写入的失败抛 `KBError(partial=False)` |
| `replace_document(kb_id, doc_id, filename, content, progress=None)` | `old_doc_id`、`document: DocSummary`、`chunk_count` | **先通过原生导入生成新文档，确认成功后才清理旧文档；不保留原 `doc_id`**。返回被替换的 `old_doc_id` 与新 `document`。新内容成功前不得删除旧内容 |
| `delete_document(kb_id, doc_id)` | `deleted: true`、`doc_id` | 先删除向量，再删除文本/FTS 与元数据，最后刷新原生统计 |
| `add_chunk(kb_id, doc_id, content)` | `chunk: ChunkView` | `content` 非空；新块需可被 `search` 命中 |
| `update_chunk(kb_id, doc_id, chunk_id, content)` | `replaced_chunk_id`、`chunk: ChunkView` | 允许生成新 `chunk_id`；保留文档 ID 与原 `chunk_index`（返回视图中的 `index` 不变） |
| `delete_chunk(kb_id, doc_id, chunk_id)` | `deleted: true`、`chunk_id` | 删除索引与数据 |

进度：`progress(stage, current, total)`，可 await；`stage ∈ PROGRESS_STAGES`（`chunking` / `embedding` / `storing`，允许实现新增）；`current >= 0`；`total >= 0`，`0` 表示未知。

---

## 4. `SourceManager`（`sources.py`）

### 4.1 构造

```python
SourceManager(data_dir, max_file_bytes=20971520, max_url_bytes=5242880, attachment_ttl=1800, http_timeout=30)
```

`data_dir` 为插件数据目录（由入口层通过 AstrBot 数据目录解析后传入）。

### 4.2 方法

| 方法 | 签名 | 语义与返回 |
| --- | --- | --- |
| `initialize` | `async def initialize() -> None` | 建立附件缓存目录、启动 TTL 清理；幂等 |
| `remember_attachments` | `async def remember_attachments(scope, components) -> dict` | `components` **只能来自入口真实 File 消息段**，不接受 LLM 提供的任意路径；记录可下载标识与元数据，返回 `{"attachments": [AttachmentInfo]}`；同 id 重复调用刷新 TTL |
| `list_attachments` | `def list_attachments(scope) -> list[dict]` | **同步**，返回当前未过期 `[AttachmentInfo]` |
| `load_attachment` | `async def load_attachment(scope, attachment_id) -> SourceDocument` | 下载/读取并返回 `SourceDocument(source="attachment")`；不存在 → `attachment_not_found`；过期 → `attachment_expired`；超限 → `payload_too_large` |
| `load_url` | `async def load_url(url) -> SourceDocument` | 仅 http/https（公网 URL 限制作用于 LLM 提交的网页地址）；**必须提取网页正文并作为 UTF-8 纯文本文档导入**，不得把 HTML 原始字节伪装成其他格式；超时 `http_timeout`；大小上限 `max_url_bytes`；失败 → `url_fetch_failed` / `unsupported_source` / `payload_too_large` |
| `close` | `async def close() -> None` | 停止清理任务、关闭 HTTP client；幂等 |

- 源文件处理沿用原生支持的格式；单文件上限 `max_file_bytes` 在进入后端前强制。
- `SourceDocument` 是传给 `add_document`/`replace_document` 的唯一载体。
- 附件缓存放插件数据目录（如 `data_dir/attachments/`），该布局不构成跨模块接口。

---

## 5. `JobManager`（`jobs.py`）

### 5.1 构造与生命周期

```python
JobManager(data_dir, max_concurrent=3)
async def initialize() -> None
async def close() -> None
```

- 存储：`data_dir/jobs.sqlite3`（SQLite，经 `aiosqlite`）。`data_dir` 为插件数据目录。
- `initialize()` 幂等：建目录与表；随后把库中所有非终态记录置为 `interrupted`（`code="interrupted"`），**不重放**。已关闭后再 `initialize()` 抛 `KBError(code="unavailable")`。
- `close()` 幂等：先置“关闭中”以拒绝新提交；取消并 `await` 所有自有任务；把仍非终态的记录持久化为 `interrupted`；唤醒等待者；清理内存锁/事件/任务引用；最后关闭连接。
- `close()` 之后：`submit`/`get` 返回失败信封 `error.code="unavailable"`。

### 5.2 持久化模型

`jobs` 表列：`job_id`(PK)、`scope`、`request_id`、`operation`、`fingerprint`、`payload`、`lock_key`、`status`、`progress`、`result`、`error`、`created_at`、`updated_at`。

- `UNIQUE (scope, request_id)`；`payload`/`progress`/`result`/`error` 为 JSON 文本；时间戳为 epoch 毫秒。
- **不得**用源请求的 `request_id` 作为文件路径。
- 第一版不自动清除幂等记录。

### 5.3 `submit`

```python
async def submit(scope, request_id, operation, payload, lock_key, work, wait_seconds=3) -> dict
```

- 参数：`scope` 为 §2.3 编码串；`request_id`/`operation`/`lock_key` 为非空字符串；`payload` 为 JSON 可序列化映射；`work` 为 `async def work(progress) -> dict`。参数不合法抛 `invalid_argument`。
- 返回统一信封（§2.4），`job_id` 始终给出。
- `wait_seconds <= 0`：立即返回当前状态（通常 `queued`/`running`）。
- `wait_seconds > 0`：最多等待该秒数；期间到达终态则返回终态，否则返回 `queued`/`running` 与 `job_id`，**后台继续执行**。
- `work` 在独立任务中运行：调用方等待超时或被取消 **不会** 取消已提交的工作（shield 语义）。

### 5.4 幂等

- `(scope, request_id)` 已存在且指纹相同 → 不新建，返回既有任务当前信封（含终态结果）。
- 已存在但指纹不同 → 失败信封，`error.code="idempotency_conflict"`、`error.details.existing_job_id`。
- 并发提交同键由内部串行裁决，仅一个 `work` 执行。
- 重建 manager 后仍必须能查到终态、并拒绝相同键不同参。

### 5.5 并发与串行

- 全局并发上限 `max_concurrent`（默认 3），跨 scope/lock_key。
- 相同 `lock_key`（同库，跨 scope）FIFO 串行；不同 `lock_key` 可并行但受全局上限约束。
- **等待锁的任务不得占用全局并发槽**：先取得 `lock_key` 再获取全局槽位。
- 创建 KB 使用 `LOCK_GLOBAL`，仅串行创建。

### 5.6 `work` 结果映射

| `work` 结果 | status | data | error |
| --- | --- | --- | --- |
| 返回 dict | `succeeded` | 返回值（JSON 校验后） | `null` |
| 抛 `KBError(partial=True)` | `partial` | `null` | 该错误 |
| 抛 `KBError(partial=False)` | `failed` | `null` | 该错误 |
| 抛未知异常 | `failed` | `null` | `code="internal"`，**不把原始异常信息中的敏感 URL/正文回传给模型**；完整 traceback 记入日志 |
| 取消 / `close()` / 重启恢复 | `interrupted` | `null` | `code="interrupted"` |

- 状态与结果 **先落库成功（或已记录进程内故障结果），再唤醒等待者**；禁止无条件唤醒。
- 终态写盘失败：回滚该短事务，重试一次把任务标记为 `partial` + `storage_failed`（结果不确定）；若仍失败，则把该结果保留为**进程内可查询**的错误信息——`get` 与相同 `(scope, request_id)` 的 `submit` 均返回它，且**不得重放 `work`**；进程重启后该任务按 `interrupted` 处理。
- 数据已写入后的结果存储故障不得表述为“未写入失败”。
- 进度经 `await progress(stage, current, total)` 上报，持久化为 `{"stage","current","total"}`。

### 5.7 `get`

```python
async def get(scope, job_id) -> dict
```

- 按可信 `scope` 查询，**不得泄露其他会话的任务**。
- 未知任务或 scope 不匹配 → 失败信封 `error.code="job_not_found"`（不抛异常，避免跨用户枚举）。
- 非终态（`queued`/`running`）任务的 `data` 返回实时元数据：`{"operation", "progress", "created_at", "updated_at"}`（`progress` 为最近一次进度或 `null`）。
- 终态任务的 `data` 为 `work` 结果（`succeeded`）或 `null`；顶层四键保持不变。
- 存储暂时不可用时返回 `unavailable`；若该任务存在进程内故障结果，则优先返回该结果。

### 5.8 事务、清理与关闭

- 所有 SQLite 访问经短事务串行保护；**不得**把数据库锁持有到整个 `work` 完成。
- 每个短事务失败必须 `rollback`，失败事务不得被后续 `commit` 意外提交（避免未提交的 `succeeded` 行被当作成功）。
- 唤醒等待者仅发生在终态已持久化或已有进程内故障结果之后；事件不能在一个无条件 `finally` 中被声称“持久化完成”。
- 任务到达终态后清理其事件；锁引用在无人持有/等待时释放——已完成的清理不得影响进行中的等待者被正常唤醒。
- `initialize`/`close` 共享生命周期锁：`close` 关闭连接受数据库锁保护，关闭后不得再打开新连接；并发 `get` 返回一致结果或 `unavailable`。
- `close()` 幂等：拒绝新任务、取消并 `await` 所有自有任务、把未完成记录持久化为 `interrupted`（持久化失败时保留进程内故障结果）；`close()` 后进度回调变为空操作，**不再写库**；内存锁与任务引用被清理。

---

## 6. 入口层约定

### 6.1 触发边界与适用范围

- **适用范围**：只处理 **AstrBot 已经产生的正常 LLM 请求**（正常对话中已组装并交给模型的请求），任何发送者的正常请求都适用。
- **运行器边界**：本地工具仅由 AstrBot 内置 Agent（`local` runner）执行；第三方 Agent 运行器（Dify / Coze / DashScope / DeerFlow 等）不执行本插件工具，故其对话不产生自主维护。聊天模型可以是远程 API，与运行器类型无关。
- **不做额外模型调用**：不为没有产生模型请求的消息（如未被唤醒的群消息）额外发起后台 LLM 判断。消息监听只用于登记真实附件（见 §6.4），不调用模型、不回复、不入库。
- **自主判断**：模型在每轮正常对话中自行评估本轮内容是否包含值得长期保留的专业知识，不依赖用户命令或审批。
- **检索去重**：写入前应先经 `kbm_search` / `kbm_list_documents` / `kbm_read_document` 检查既有内容，避免重复录入。
- **相关库更新与建库**：优先更新相关度最高的既有知识库；有可靠证据时允许纠正 / 替换过时内容；确实没有相关库且内容值得保留时，才调用 `kbm_create_kb` 新建。
- **无需显式维护指令**：用户明确提出的维护要求与模型自主判断走同一组工具。

### 6.2 SYSTEM 注入（`autonomy.py`）

模块为纯函数、仅标准库，不导入 AstrBot / 服务 / 配置对象：

```python
AUTONOMY_BLOCK_START = "<!-- kbm-autonomy:start -->"
AUTONOMY_BLOCK_END = "<!-- kbm-autonomy:end -->"
DEFAULT_AUTONOMY_PROMPT: str  # 内置默认规则正文

def inject_autonomy_prompt(system_prompt: str | None, policy: str | None = None) -> str: ...
```

- **注入入口**：入口层在 `@filter.on_llm_request()` 中按每个正常 LLM 请求调用 `inject_autonomy_prompt(当前系统提示, 配置的 policy)`，把返回值写回该请求的系统提示。
- **幂等**：规则正文恒定包裹在 `AUTONOMY_BLOCK_START` / `AUTONOMY_BLOCK_END` 唯一标记块中；重复注入只替换本插件的标记块，相同策略重复调用结果不变，策略变化时块外系统内容保持不变。
- **策略解析**：`policy` 为非空字符串时使用其规范文本（统一换行、剥离信封标记）；否则回退 `DEFAULT_AUTONOMY_PROMPT`。`system_prompt` 为 `None` / 空白 / 非字符串时按空串处理。
- **配置映射**：唯一配置键 `autonomy_system_prompt`（`type=text`）；其 `default` 即内置 `DEFAULT_AUTONOMY_PROMPT`，缺失 / 空白 / 非字符串 → 内置默认。不新增其他自主运维参数。
- **内容不变量**：注入内容只固定维护职责与判断 / 去重 / 幂等 / 如实汇报规则，不包含具体领域事实。

### 6.3 工具供给与启停边界

- 本插件当前已启用的 17 个工具随正常 LLM 请求提供给模型（按名合并进本次请求，不删除、不覆盖其他工具）；人格无需手动勾选 kbm 工具。
- 框架明确的**插件禁用 / 会话禁用 / 全局工具停用 / 工具权限**仍然生效；本插件不绕过、不修改这些全局设置。
- 停用本插件即停止 SYSTEM 注入与工具供给；清空 `autonomy_system_prompt` 即恢复内置默认规则；用户可编辑该提示以调整判断标准与入库尺度。

### 6.4 隔离、幂等与写任务语义

- `scope = encode_scope(event.unified_msg_origin, str(event.get_sender_id()))`；**scope 绝不来自 LLM 参数**，附件与任务按该 scope 隔离。
- 工具**不再按管理员身份门禁**，也不引入基于用户身份的维护开关；数据可见性边界是 scope 隔离，而非管理员权限。
- 收到消息时把消息链中的 `File` 段筛选后交给 `remember_attachments`（对所有会话生效）；其他类型不进 SourceManager；只接受真实 File 段，不接受 LLM 提供的路径。
- 写工具把 `request_id` 作为 **必填 LLM 参数**透传（模型重试复用同一值，入口不得随机改写）；`job_id` 仅用于状态查询，不作为新任务参数。
- 自主触发的写入同样走 `JobManager`：默认 `wait_seconds=3`；超时返回 `queued`/`running` + `job_id`，由 `kbm_job_status` 轮询 `get`。
- 索引一致性不因自主触发而放松：沿用 §3.2/§5 的“先写新后删旧”“先删向量再删文本/FTS 与元数据”语义；`partial`/`interrupted` 不得表述为成功。
- 入口负责把 `SourceManager.load_*` 的 `SourceDocument` 传给 `NativeKBBackend` 写方法。

---

## 7. 变更流程

1. 契约变更必须同时修改本文件；涉及 `common.py` 原语时同步修改该模块。
2. 向后不兼容变更必须在变更说明中明确标出，并说明对既有数据与调用方的影响。
3. 实现文件（`backend.py`/`sources.py`/`jobs.py`/`main.py`）只消费本契约，不得私自扩展跨模块数据形状。

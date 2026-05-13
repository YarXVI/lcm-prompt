# LCM v2 — 惰性上下文物化协议 v2

> **Lazy Context Materialization Protocol v2**
>
> 一种让大语言模型在超长上下文中按需加载内容的标准协议，通过"摘要索引 + 哨兵标记"机制，将上下文窗口从"装满内容"转变为"装满索引"，实现无限上下文的幻觉。

---

## 目录

1. [核心概念](#核心概念)
2. [架构概览](#架构概览)
3. [协议机制](#协议机制)
4. [内容编码层](#内容编码层)
5. [功能模块](#功能模块)
6. [快速开始](#快速开始)
7. [高级用法](#高级用法)
8. [设计哲学](#设计哲学)

---

## 核心概念

### 问题：上下文窗口的瓶颈

大语言模型有固定的上下文窗口（如 128K tokens）。当处理大型代码库、长文档或多轮对话时，上下文很快就会被填满，导致：
- **信息丢失**：早期内容被挤出窗口
- **成本飙升**：每次请求都发送完整上下文
- **延迟增加**：模型需要处理大量无关内容

### 解决方案：惰性加载

LCM 的核心思想来自操作系统的**虚拟内存**和**惰性加载**：

```
传统方式：上下文窗口 = 完整内容（全部加载）
LCM 方式：上下文窗口 = 摘要索引（按需加载）
```

**类比**：
- 传统方式 = 把整个图书馆搬进书房
- LCM 方式 = 书房只放目录卡片，需要哪本书再去取

### 三个核心抽象

| 抽象 | 说明 | 类比 |
|------|------|------|
| **ContextChunk** | 上下文块，带摘要的完整内容单元 | 图书馆的一本书 |
| **ChunkStore** | 块存储，管理所有块的持久化和缓存 | 图书馆书架 |
| **Sentinel** | 哨兵标记，模型请求加载的协议信号 | 借书条 |

---

## 架构概览

### 分层架构（OSI 模型类比）

```
┌─────────────────────────────────────────────────────────────┐
│  应用层：LCMClientV2 — 用户交互入口                            │
├─────────────────────────────────────────────────────────────┤
│  表示层：ContentEncoding — 内容编码（中文思考、日语思考等）      │
├─────────────────────────────────────────────────────────────┤
│  会话层：LCMSession — 会话状态管理                              │
├─────────────────────────────────────────────────────────────┤
│  传输层：LCMOrchestratorV2 — 核心调度（状态机、chunk加载）       │
├─────────────────────────────────────────────────────────────┤
│  网络层：SentinelDetectorV2 — 哨兵检测与协议解析                 │
├─────────────────────────────────────────────────────────────┤
│  数据链路层：ChunkStoreV2 — 块存储（持久化、缓存、索引）          │
├─────────────────────────────────────────────────────────────┤
│  物理层：ContextChunk — 数据实体（内容、摘要、元数据）            │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件关系

```
用户查询 ──► LCMClientV2 ──► LCMOrchestratorV2
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   ContentEncoding      SentinelDetectorV2         ChunkStoreV2
   （编码层）            （哨兵检测）               （块存储）
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                          LLM API 调用
                                 │
                    模型输出 [NEED_CHUNK:xxx]
                                 │
                                 ▼
                    加载 chunk → 注入上下文 → 继续生成
```

---

## 协议机制

### 1. 摘要索引

系统提示词中包含所有可用块的**摘要列表**，而非完整内容：

```
## 可用上下文块索引

- **chunk_auth_handler** [auth.py] (200 tokens, 加载 3 次): 登录处理器的完整实现
- **chunk_user_model** [models.py] (100 tokens, 加载 1 次): 用户数据模型定义
- **chunk_middleware** [middleware.py] (150 tokens, 加载 0 次): 认证中间件逻辑
```

### 2. 哨兵标记

当模型需要查看某个块的详细内容时，输出标准格式的哨兵：

```
[NEED_CHUNK:chunk_auth_handler]
```

系统检测到哨兵后：
1. 从 ChunkStore 加载对应块的完整内容
2. 将内容注入到对话上下文中
3. 让模型从断点继续生成

### 3. 状态机

```
IDLE ──► GENERATING ──► WAITING_CHUNK ──► RESUMING ──► COMPLETED
           │                    │              │
           ▼                    ▼              ▼
         ERROR ◄───────────────┴──────────────┘
```

| 状态 | 说明 |
|------|------|
| `IDLE` | 空闲，等待用户输入 |
| `GENERATING` | 模型正在生成响应 |
| `WAITING_CHUNK` | 检测到哨兵，等待加载 chunk |
| `RESUMING` | 已注入 chunk，恢复生成 |
| `COMPLETED` | 会话完成 |
| `ERROR` | 发生错误 |

### 4. 交互流程

```
用户：审查 auth_handler.py 的安全性

助手：我将审查登录处理器的安全性。
      [NEED_CHUNK:auth_handler]          ← 哨兵：请求加载
      
[系统注入 chunk_auth_handler 的完整内容]

助手：审查完登录处理器后，我发现密码使用 bcrypt 哈希——这是安全的。
      但会话 Token 的过期时间设为 7 天，建议缩短。
      [NEED_CHUNK:middleware]             ← 继续请求下一个块
      
[系统注入 chunk_middleware 的完整内容]

助手：中间件实现正确。总体评价：安全性良好。
```

---

## 内容编码层

### 设计哲学

LCM 核心负责"数据包的可靠传输"（上下文调度、chunk 加载），ContentEncoding 负责"数据内容的编码/解码"（语言精简、压缩）。

**类比**：
- LCM Core = TCP/IP（传输层）
- ContentEncoding = Gzip/Brotli（表示层）
- 数据区支持 Content-Encoding 头，声明编码方式

### 标准编码接口

```python
class ContentEncoding(ABC):
    @property
    def encoding_type(self) -> EncodingType
    
    def encode_system_prompt(self, prompt: str, context: EncodingContext) -> str
    def encode_response(self, response: str, context: EncodingContext) -> str
    def decode_for_display(self, text: str, context: EncodingContext) -> str
```

### 编码类型注册表

| 编码类型 | 标识符 | 说明 |
|---------|--------|------|
| `IDENTITY` | `identity` | 恒等编码（无变换，默认） |
| `CHINESE_THINK` | `chinese-think` | 中文思考精简模式 |
| `JAPANESE_THINK` | `ja-think` | 日语思考精简模式（预留） |
| `ENGLISH_THINK` | `en-think` | 英语思考精简模式（预留） |
| `CUSTOM` | `custom` | 自定义编码 |

### 使用编码层

```python
from lcm_v2 import LCMClientV2, ChunkStoreV2, EncodingType

store = ChunkStoreV2()
client = LCMClientV2(llm_client, store, encoding_type=EncodingType.CHINESE_THINK)

# 运行时切换
client.encoding_type = EncodingType.IDENTITY  # 关闭
client.encoding_type = EncodingType.CHINESE_THINK  # 开启
```

### Chinese-Think 编码（首个标准实现）

将 [chinese-think-skills](../chinese-think-skills/) 独立项目封装为 LCM 标准编码插件：

- **system prompt 编码**：追加中文思考指令，引导模型精简表达
- **response 编码**：规则压缩（删除冗余、替换文言），保护 LCM 哨兵标记
- **降级安全**：chinese-think-skills 不可用时自动回退到基础实现

---

## 功能模块

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **类型系统** | `lcm_types.py` | ContextChunk, LCMEvent, LoadRequest, LCMSession, LCMState, LCMMetrics |
| **块存储** | `store.py` | ChunkStoreV2 — 线程安全、LRU缓存、JSONL持久化 |
| **哨兵检测** | `detector.py` | SentinelDetectorV2 — 多模式匹配、置信度评分、缓冲区管理 |
| **调度器** | `orchestrator.py` | LCMOrchestratorV2 — 状态机、批量预取、重试、编码层钩子 |
| **客户端** | `client.py` | LCMClientV2 — 同步/流式对话、事件回调、统计报告 |
| **提示词** | `prompt.py` | build_initial_messages_v2 — LCM协议指令 + 块索引构建 |
| **编码层** | `content_encoding.py` | ContentEncoding, EncodingType, ContentEncodingRegistry |

### 扩展模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **Token 预算** | `token_budget.py` | TokenBudget — 防止上下文溢出，动态预算分配 |
| **Chunk 依赖图** | `chunk_graph.py` | ChunkGraph — DAG 拓扑排序，智能加载顺序 |
| **混合模式** | `hybrid_mode.py` | HybridChunkManager — 高频直接注入 + 低频 LCM |
| **Provider 路由** | `provider_router.py` | ProviderRouter — 云端/本地自动切换，自适应路由 |
| **KV Cache** | `kv_cache.py` | KVCacheManager — 利用云厂商缓存机制，减少重复计算 |
| **自适应分块** | `adaptive_chunking.py` | AdaptiveChunking — 粗/细粒度动态调整 |
| **质量评估** | `quality_eval.py` | QualityEvaluator — 系统性答案质量评估 |
| **多模型基准** | `multi_model_benchmark.py` | MultiModelBenchmark — 跨模型延迟+质量对比 |
| **微调协议** | `fine_tuning.py` | LCMFineTuner — LCM 原生 Fine-tuning 支持 |
| **多 Agent** | `multi_agent.py` | MultiAgentLCM — 共享组件索引去重，协作加载 |
| **SQLite 存储** | `sqlite_store.py` | SQLiteChunkStore — SQLite 后端持久化 |
| **向量索引** | `vector_index.py` | VectorIndex — 语义搜索，embedding 相似度 |
| **异步客户端** | `async_client.py` | AsyncLCMClient — 异步非阻塞调用 |
| **压缩** | `compression.py` | ChunkCompressor — 块内容压缩，减少存储 |
| **多模态** | `multimodal.py` | MediaChunk — 图像、音频等多模态内容支持 |
| **协议协商** | `protocol.py` | ProtocolNegotiator — 版本协商，向后兼容 |
| **分布式** | `distributed.py` | DistributedChunkStore — 跨节点协调与任务分发 |

---

## 快速开始

### 安装

```bash
# 基础依赖
pip install tiktoken  # 精确 token 计数（可选）

# 中文思考编码（可选）
pip install -e ../chinese-think-skills
```

### 基础用法

```python
from lcm_v2 import LCMClientV2, ChunkStoreV2, ContextChunk

# 1. 创建块存储
store = ChunkStoreV2()

# 2. 添加上下文块
store.add_chunk(ContextChunk(
    chunk_id="auth_handler",
    content="def login(username, password): ...",
    summary="登录处理器的完整实现",
    source="auth.py",
))

# 3. 创建 LCM 客户端
client = LCMClientV2(llm_client, store)

# 4. 对话
response = client.chat("审查登录处理器的安全性")
print(response)
```

### 流式对话

```python
for chunk in client.chat_stream("审查登录处理器的安全性"):
    print(chunk, end="", flush=True)
```

### 事件监听

```python
def on_event(event):
    print(f"[{event.event_type}] {event.chunk_id}")

client.on_event(on_event)
```

### 使用中文思考编码

```python
from lcm_v2 import EncodingType

client = LCMClientV2(llm_client, store, encoding_type=EncodingType.CHINESE_THINK)

# 查看编码统计
print(client.content_encoding.get_stats())
```

---

## 高级用法

### 自定义编码

```python
from lcm_v2 import ContentEncoding, EncodingType, EncodingContext

class MyEncoding(ContentEncoding):
    @property
    def encoding_type(self):
        return EncodingType.CUSTOM
    
    @property
    def name(self):
        return "My Custom Encoding"
    
    def encode_system_prompt(self, prompt, context):
        return prompt + "\n\n[自定义指令]"
    
    def encode_response(self, response, context):
        return response.upper()
    
    def decode_for_display(self, text, context):
        return text

# 注册并使用
from lcm_v2 import register_encoding
register_encoding(MyEncoding())

client = LCMClientV2(llm_client, store, encoding_type=EncodingType.CUSTOM)
```

### Token 预算管理

```python
from lcm_v2 import TokenBudget

budget = TokenBudget(max_tokens=8000)
budget.allocate("system", 2000)
budget.allocate("chunks", 4000)
budget.allocate("response", 2000)

# 检查是否超支
if budget.is_over_budget():
    print("Token 预算超支！")
```

### Chunk 依赖图

```python
from lcm_v2 import ChunkGraph

graph = ChunkGraph()
graph.add_edge("auth_handler", "user_model")  # auth_handler 依赖 user_model
graph.add_edge("middleware", "auth_handler")

# 获取加载顺序（拓扑排序）
order = graph.get_load_order("middleware")
# ['user_model', 'auth_handler', 'middleware']
```

### 混合模式

```python
from lcm_v2 import HybridChunkManager, HybridConfig

config = HybridConfig(
    hot_threshold=3,      # 加载 3 次以上的为热块
    direct_inject=True,   # 热块直接注入
)

hybrid = HybridChunkManager(store, config)
messages = hybrid.build_messages("用户查询")
```

### 多模型基准测试

```python
from lcm_v2 import MultiModelBenchmark, ModelConfig

benchmark = MultiModelBenchmark()
benchmark.add_model(ModelConfig(name="gpt-4", provider="openai"))
benchmark.add_model(ModelConfig(name="claude-3", provider="anthropic"))

results = benchmark.run(test_queries, store)
for result in results:
    print(f"{result.model_name}: 延迟={result.latency_ms}ms, 质量={result.quality_score}")
```

---

## 设计哲学

### 1. 协议优先

LCM 首先是一个**协议**，其次是一个实现。任何遵循 LCM 协议的系统都可以互操作：
- 模型输出 `[NEED_CHUNK:xxx]` 标准格式
- 系统注入 chunk 的完整内容
- 模型从断点继续

### 2. 核心与扩展解耦

```
LCM Core（必需）
  ├── lcm_types.py
  ├── store.py
  ├── detector.py
  ├── orchestrator.py
  ├── client.py
  └── prompt.py

Extensions（可选）
  ├── content_encoding.py  ← 内容编码层
  ├── token_budget.py
  ├── chunk_graph.py
  ├── hybrid_mode.py
  ├── provider_router.py
  ├── kv_cache.py
  ├── adaptive_chunking.py
  ├── quality_eval.py
  ├── multi_model_benchmark.py
  ├── fine_tuning.py
  ├── multi_agent.py
  ├── sqlite_store.py
  ├── vector_index.py
  ├── async_client.py
  ├── compression.py
  ├── multimodal.py
  ├── protocol.py
  └── distributed.py
```

### 3. 编码层独立性

内容编码层遵循 OSI 表示层的设计：
- LCM Core = TCP（传输可靠）
- ContentEncoding = Gzip（内容压缩）
- 两者通过标准接口交互，互不依赖

### 4. 降级安全

所有可选模块都遵循**优雅降级**：
- `tiktoken` 不可用时 → 字符估算
- `chinese-think-skills` 不可用时 → 基础实现
- 编码器未注册时 → 恒等编码

---

## 协议规范

### 哨兵格式

```
[NEED_CHUNK:<chunk_id>]   # 请求加载 chunk
[LOAD_CHUNK:<chunk_id>]   # 加载 chunk（别名）
[FETCH:<chunk_id>]        # 获取 chunk（别名）
```

### 系统提示词结构

```
[LCM 协议指令]
  ├── 核心机制说明
  ├── 使用规则（7条）
  ├── 交互示例
  └── 重要提醒

[内容编码指令]（可选，由编码层注入）
  └── 例如：中文思考精简规则

[可用上下文块索引]
  ├── chunk_id, source, tokens, load_count, summary
  └── ...
```

### 消息流

```
Round 1:
  System: [LCM指令] + [编码指令] + [块索引]
  User:   用户查询
  Assistant: 部分响应 + [NEED_CHUNK:chunk_1]
  
Round 2:
  System: [LCM指令] + [编码指令] + [块索引] + [chunk_1完整内容]
  User:   用户查询
  Assistant: 续接响应 + [NEED_CHUNK:chunk_2]
  
Round 3:
  System: [LCM指令] + [编码指令] + [块索引] + [chunk_1] + [chunk_2]
  User:   用户查询
  Assistant: 最终响应（无哨兵）
```

---

## 性能指标

| 指标 | 说明 |
|------|------|
| `cache_hit_rate` | 缓存命中率 |
| `avg_load_latency_ms` | 平均加载延迟 |
| `total_tokens_generated` | 生成 token 总数 |
| `total_chunks_loaded` | 加载 chunk 总数 |
| `duration_ms` | 会话持续时间 |

---

## 相关项目

- [chinese-think-skills](../chinese-think-skills/) — 中文思考模式独立实现
- [test_lcm_v2.py](test_lcm_v2.py) — 单元测试
- [test_content_encoding.py](test_content_encoding.py) — 编码层测试

---

## 许可证

MIT
# LCM Protocol v2 - Lazy Context Materialization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## English

### What is LCM?

**LCM (Lazy Context Materialization)** is a bidirectional collaboration protocol between LLMs and context management systems. Unlike traditional "full injection" approaches that dump all context into the prompt upfront, LCM lets the **model itself decide what information it needs** during generation.

### Core Concept

When a model says "let me look at the login function implementation," it explicitly signals an information need. Traditional approaches ignore this signal and continue feeding noise. LCM trusts the model's judgment and loads requested context chunks on-demand.

### Key Features

| Feature | Description |
|---------|-------------|
| **On-Demand Loading** | Model requests chunks via sentinel markers like `[NEED_CHUNK:id]` |
| **Streaming Detection** | Real-time sentinel detection in streaming output |
| **Token Budget** | Prevents context overflow with intelligent budget management |
| **Hybrid Mode** | Hot chunks direct-injected + cold chunks via LCM |
| **Provider Routing** | Auto-detects cloud vs local APIs, switches strategies |
| **KV Cache Optimization** | Leverages cloud provider prompt caching (Anthropic/DeepSeek) |
| **Chunk Dependency Graph** | DAG-based dependency resolution and topological loading |
| **Multi-Agent Support** | Shared index across agents with deduplication |
| **Async I/O** | Asynchronous LLM calls and concurrent chunk loading |
| **Multimodal** | Image, PDF, audio chunk support |
| **Distributed Storage** | Multi-node chunk storage with consistency hashing |
| **Protocol Versioning** | v1/v2/v3 compatibility with feature negotiation |

### Quick Start

```python
from lcm_v2 import ChunkStoreV2, ContextChunk, LCMClientV2

# 1. Create store
store = ChunkStoreV2()

# 2. Add chunks
store.add_chunk(ContextChunk(
    chunk_id="auth_handler",
    content="def login(username, password): ...",
    summary="Authentication handler implementation",
))

# 3. Build LCM messages
from lcm_v2 import build_initial_messages_v2
messages = build_initial_messages_v2(store, "Review the auth module")

# 4. Chat with LCM
client = LCMClientV2(llm_client=your_llm, chunk_store=store)
result = client.chat(messages, your_stream_function)
```

### Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│ Orchestrator │───▶│    Store    │
│  LCMClient  │    │  LCMOrchestr │    │ ChunkStore  │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Detector  │    │   Graph      │    │   Hybrid    │
│  Sentinel   │    │  ChunkGraph  │    │   Mode      │
└─────────────┘    └──────────────┘    └─────────────┘
```

### Installation

```bash
pip install lcm-protocol

# With performance extras
pip install lcm-protocol[performance]

# With all extras
pip install lcm-protocol[all]
```

### Requirements

- Python 3.9+
- numpy (core)
- Optional: hnswlib, zstandard, tiktoken, PyPDF2

---

## 中文

### 什么是 LCM？

**LCM（惰性上下文物化，Lazy Context Materialization）** 是一种 LLM 与上下文管理系统之间的双向协作协议。与传统"全量注入"方案不同，LCM 让**模型自己在生成过程中决定需要什么信息**。

### 核心概念

当模型说"让我看看登录函数的实现"时，它明确表达了信息需求。传统方案无视这个信号，继续在噪声中挣扎。LCM 选择信任模型自己的判断，按需加载请求的上下文块（chunk）。

### 核心特性

| 特性 | 说明 |
|------|------|
| **按需加载** | 模型通过哨兵标记 `[NEED_CHUNK:id]` 请求 chunk |
| **流式检测** | 在流式输出中实时检测哨兵标记 |
| **Token 预算** | 智能预算管理，防止上下文溢出 |
| **混合模式** | 高频 chunk 直接注入 + 低频 chunk 走 LCM |
| **Provider 路由** | 自动识别云端/本地 API，自动切换策略 |
| **KV Cache 优化** | 利用云厂商缓存机制（Anthropic/DeepSeek） |
| **Chunk 依赖图** | 基于 DAG 的依赖解析和拓扑加载 |
| **多 Agent 支持** | 跨 Agent 共享索引，自动去重 |
| **异步 I/O** | 异步 LLM 调用和并发 chunk 加载 |
| **多模态** | 支持图片、PDF、音频 chunk |
| **分布式存储** | 多节点 chunk 存储，一致性哈希 |
| **协议版本协商** | v1/v2/v3 兼容，特性协商 |

### 快速开始

```python
from lcm_v2 import ChunkStoreV2, ContextChunk, LCMClientV2

# 1. 创建存储
store = ChunkStoreV2()

# 2. 添加 chunks
store.add_chunk(ContextChunk(
    chunk_id="auth_handler",
    content="def login(username, password): ...",
    summary="认证处理器实现",
))

# 3. 构建 LCM 消息
from lcm_v2 import build_initial_messages_v2
messages = build_initial_messages_v2(store, "审查认证模块")

# 4. 使用 LCM 对话
client = LCMClientV2(llm_client=your_llm, chunk_store=store)
result = client.chat(messages, your_stream_function)
```

### 架构

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   客户端     │───▶│   调度器     │───▶│   存储层    │
│  LCMClient  │    │ LCMOrchestr  │    │ ChunkStore  │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   检测器     │    │   依赖图     │    │   混合模式  │
│  Sentinel   │    │  ChunkGraph  │    │ HybridMode  │
└─────────────┘    └──────────────┘    └─────────────┘
```

### 安装

```bash
pip install lcm-protocol

# 带性能优化
pip install lcm-protocol[performance]

# 完整安装
pip install lcm-protocol[all]
```

### 环境要求

- Python 3.9+
- numpy（核心依赖）
- 可选：hnswlib, zstandard, tiktoken, PyPDF2

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

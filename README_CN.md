# LCM2 — 惰性上下文物化协议 v2（协议版）

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README_EN.md)

---

## 概述

**LCM2** 是 LCM 协议的 V2 实现，面向高级集成场景。在 LCM1 核心哨兵协议基础上，增加了：

- **线程安全**：所有操作带 RLock 保护
- **持久化**：JSONL + 索引文件，进程重启不丢失
- **精确 Token 计数**：支持 tiktoken 和中文估算
- **LRU 缓存**：热点 chunk 内存缓存
- **语义搜索**：支持向量相似度（可选 embedding）
- **Chunk 依赖图**：DAG 拓扑排序
- **混合模式**：高频直接注入 + 低频 LCM
- **API 路由自动识别**：云端/本地自动切换
- **KV Cache 联动**：利用云厂商缓存机制
- **自适应粒度**：粗/细粒度动态调整
- **质量评估**：系统性答案质量评估
- **多 Agent 协作**：共享组件索引去重
- **内容编码层**：支持可插拔的语言编码（中文思考等）
- **多模态**：图片、PDF、音频 chunk 支持
- **分布式存储**：多节点 chunk 存储，一致性哈希
- **协议版本协商**：v1/v2/v3 兼容

## 版本家族

| 版本 | 分支 | 说明 |
|------|------|------|
| LCM3 | `main` | **全功能主版本**，带完整多粒度编码器体系 |
| **LCM2** | **`lcm2`** | **V2 协议版**，面向高级集成场景 |
| LCM1 | `lcm1` | 精简版，最小依赖，快速集成 |

## 快速开始

### 安装

```bash
pip install lcm-protocol

# 带性能优化
pip install lcm-protocol[performance]

# 完整安装
pip install lcm-protocol[all]
```

### 基础使用

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

## 核心特性

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

## 模块架构

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   客户端     │───▶│   调度器     │───▶│   存储层     │
│  LCMClient  │    │ LCMOrchestr  │    │ ChunkStore  │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   检测器     │    │   依赖图     │    │   混合模式   │
│  Sentinel   │    │  ChunkGraph  │    │ HybridMode  │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

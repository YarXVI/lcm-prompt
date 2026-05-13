# LCM1 — 惰性上下文物化协议 v1（精简版）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[English](README_EN.md) | [论文 (中文)](docs/PAPER_CN.md) | [Paper (English)](docs/PAPER_EN.md)

---

## 概述

**LCM1** 是 LCM 协议的精简发布版，零额外依赖（仅 httpx），专注于解决 Agent 框架的首轮 Prompt 膨胀问题。

将巨型首轮 Prompt（10,000-50,000 tokens）拆解为"精简索引（~500t）+ 按需加载"模式，实测本地部署 TTFT 降低 **1.1×-3.2×**。

所属 LCM 版本家族：
| 版本 | 分支 | 说明 |
|------|------|------|
| LCM3 | `main` | **全功能主版本**，带完整多粒度编码器体系 |
| LCM2 | `lcm2` | **V2 协议版**，面向高级集成（多 Agent、分布式、多模态） |
| **LCM1** | **`lcm1`** | **精简版**，最小依赖，3 步快速集成 |

## 快速开始

### 安装

```bash
pip install -e .
# 依赖仅 httpx
```

### 3 步集成

```python
from lcm import ChunkStore, ContextChunk, LCMOrchestrator, build_initial_messages

# 步骤 1: 注册 Prompt 组件为 LCM 块
store = ChunkStore()
store.add_chunk(ContextChunk(
    chunk_id="tool:search",
    summary="search_web(keyword) — 搜索网页内容，返回前10条",
    content="def search_web(keyword: str) -> List[Result]: ...",
    tokens=150,
))

# 步骤 2: 构建 LCM 消息 + 运行
orchestrator = LCMOrchestrator(chunk_store=store)
messages = build_initial_messages("帮我搜索最新 AI 论文", store)

# 步骤 3: 流式生成（stream_fn 是你的 LLM API 包装函数）
for chunk in orchestrator.run_stream(messages, your_stream_fn):
    print(chunk, end="")
```

完整示例见 [examples/basic_usage.py](examples/basic_usage.py)。

## API 概览

| 类 | 作用 |
|---|------|
| `ChunkStore` | 块注册表 — add / get / find_related |
| `SentinelDetector` | 流式哨兵检测器 |
| `LCMOrchestrator` | 协议编排器 — 状态机 + 调度循环 |
| `LCMClient` | 便捷包装器（组合 Store + Orchestrator + stream_fn） |

## 功能特性

- **标准哨兵协议**：`[NEED_CHUNK:chunk_id]` — 纯文本，无需 API 修改
- **六态状态机**：IDLE → GENERATING → WAITING_CHUNK → RESUMING → COMPLETED / ERROR
- **多轮续接**：API 完全兼容的 Route B 策略
- **Speculative Prefetch**：关键词交叠 + 来源前缀 + 热度加权的关联预取
- **四层防护**：同轮去重 / 跨轮跟踪 / 轮次上限(20) / chunk_miss 容错
- **零额外依赖**：纯 Python 标准库 + httpx

## 性能数据（三平台实测）

| 平台 | 规模 | 传统 TTFT | LCM TTFT | 加速比 | 建议 |
|------|:---:|:---:|:---:|:---:|:---:|
| **本地 35B (4bit)** | 32,000t | 50,171ms | 15,840ms | **3.2×** | ✅ 强烈推荐 |
| DeepSeek V4 Flash | 64,000t | 5,442ms | 16,354ms | 0.3× | ❌ 不推荐 |
| 阿里百炼 Qwen3-235B | 383t | 1,738ms | 22,350ms | 0.08× | ❌ 不推荐 |

## 兼容性

兼容所有 OpenAI-compatible API 端点（DeepSeek、阿里百炼、OpenAI、Ollama、LM Studio、vLLM 等）。

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

# LCM2 — Lazy Context Materialization Protocol v2

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文](README_CN.md)

---

## Overview

**LCM2** is the V2 implementation of the LCM protocol, designed for advanced integration scenarios. Building on LCM1's core sentinel protocol, it adds thread safety, persistence, semantic search, multi-agent support, and more.

## Version Family

| Version | Branch | Description |
|---------|--------|-------------|
| LCM3 | `main` | **Full-featured main version** with complete multi-granularity encoder system |
| **LCM2** | **`lcm2`** | **V2 protocol version** for advanced integration scenarios |
| LCM1 | `lcm1` | Lightweight edition, minimal dependencies, quick integration |

## Quick Start

### Install

```bash
pip install lcm-protocol

# With performance extras
pip install lcm-protocol[performance]

# With all extras
pip install lcm-protocol[all]
```

### Basic Usage

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

## Key Features

| Feature | Description |
|---------|-------------|
| **On-Demand Loading** | Model requests chunks via sentinel markers like `[NEED_CHUNK:id]` |
| **Streaming Detection** | Real-time sentinel detection in streaming output |
| **Token Budget** | Prevents context overflow with intelligent budget management |
| **Hybrid Mode** | Hot chunks direct-injected + cold chunks via LCM |
| **Provider Routing** | Auto-detects cloud vs local APIs, switches strategies |
| **KV Cache Optimization** | Leverages cloud provider prompt caching |
| **Chunk Dependency Graph** | DAG-based dependency resolution and topological loading |
| **Multi-Agent Support** | Shared index across agents with deduplication |
| **Async I/O** | Asynchronous LLM calls and concurrent chunk loading |
| **Multimodal** | Image, PDF, audio chunk support |
| **Distributed Storage** | Multi-node chunk storage with consistent hashing |
| **Protocol Versioning** | v1/v2/v3 compatibility with feature negotiation |

## License

MIT License — see [LICENSE](LICENSE).

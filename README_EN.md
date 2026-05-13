# LCM1 — Lazy Context Materialization v1 (Lightweight Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[中文](README_CN.md) | [论文 (中文)](docs/PAPER_CN.md) | [Paper (English)](docs/PAPER_EN.md)

---

## Overview

**LCM1** is the lightweight release of the LCM protocol, zero extra dependencies (only httpx). It focuses on solving the first-round prompt inflation problem in Agent frameworks.

Decomposes monolithic first-round prompts (10,000-50,000 tokens) into "compact index (~500t) + on-demand loading", achieving **1.1×-3.2×** TTFT reduction in local deployments.

## Version Family

| Version | Branch | Description |
|---------|--------|-------------|
| LCM3 | `main` | **Full-featured main version** with complete multi-granularity encoder system |
| LCM2 | `lcm2` | **V2 protocol version** for advanced integration (Multi-Agent, Distributed, Multimodal) |
| **LCM1** | **`lcm1`** | **Lightweight edition**, minimal dependencies, 3-step quick integration |

## Quick Start

### Install

```bash
pip install -e .
# Only dependency: httpx
```

### 3-Step Integration

```python
from lcm import ChunkStore, ContextChunk, LCMOrchestrator, build_initial_messages

# Step 1: Register prompt components as LCM chunks
store = ChunkStore()
store.add_chunk(ContextChunk(
    chunk_id="tool:search",
    summary="search_web(keyword) — Searches the web, returns top 10 results",
    content="def search_web(keyword: str) -> List[Result]: ...",
    tokens=150,
))

# Step 2: Build LCM messages + run
orchestrator = LCMOrchestrator(chunk_store=store)
messages = build_initial_messages("Search for the latest AI papers", store)

# Step 3: Stream (stream_fn is your LLM API wrapper)
for chunk in orchestrator.run_stream(messages, your_stream_fn):
    print(chunk, end="")
```

## Features

- **Standard sentinel protocol**: `[NEED_CHUNK:chunk_id]` — plain text, no API modifications needed
- **Six-state machine**: IDLE → GENERATING → WAITING_CHUNK → RESUMING → COMPLETED / ERROR
- **Multi-round resume**: API-compatible Route B strategy
- **Speculative Prefetch**: Associative preloading via keyword overlap + source prefix + hotness weighting
- **Four-layer defense**: In-round dedup / cross-round tracking / round cap (20) / chunk_miss tolerance
- **Zero extra dependencies**: Pure Python stdlib + httpx

## Compatibility

Compatible with all OpenAI-compatible API endpoints (DeepSeek, Alibaba Bailian, OpenAI, Ollama, LM Studio, vLLM, etc.).

## License

MIT License — see [LICENSE](LICENSE).

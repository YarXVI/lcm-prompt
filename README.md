# Lazy Context Materialization (LCM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Global Prompt Segmentation Protocol** — Decomposes monolithic Agent framework first-round prompts into "compact index + on-demand loading", reducing TTFT by 1.1×–3.2×.

[中文全文](README_CN.md) | [论文 (中文)](docs/PAPER_CN.md) | [Paper (English)](docs/PAPER_EN.md)

---

## The Problem

LLM Agent frameworks have a universal bottleneck: every conversation must inject ALL tool definitions, skill configs, MCP services, and project memory into the first prompt — **10,000–50,000 tokens consumed before the user's intent is even analyzed**.

## LCM's Solution

Instead of dumping the entire toolbox at once, LCM hands the model a catalog and lets it request what it needs:

```
Traditional:
  ┌──────────────────────────────────────┐
  │ System + Tools(15K) + Skills(8K) +   │ ──→ Model
  │ MCP(3K) + Memory(3K) + Guard(2K)    │
  │ + User(200)                         │
  └──────────────────────────────────────┘
  Round 1 = ~36,700 tokens

LCM:
  ┌────────────┐    On-Demand      ┌──────────┐
  │ Index(~500) │ ←─────────────→ │ Full chunks│
  │ + User(200) │  [NEED_CHUNK]   │ injected   │
  └────────────┘                   └──────────┘
  Round 1 = ~700 tokens            Only needed chunks loaded
```

## Performance (Three-Platform Empirical, 166 real model calls)

| Platform | Model | Scale | Trad TTFT | LCM TTFT | Speedup | Recommendation |
|----------|-------|:-----:|----------:|---------:|:-------:|:--------------:|
| **Local** | 35B 4-bit | 32Kt | 50,171ms | 15,840ms | **3.2×** | ✅ Strongly recommend |
| DeepSeek | V4 Flash | 64Kt | 5,442ms | 16,354ms | 0.3× | ❌ Not recommended |
| Bailian | Qwen3-235B | 383t | 1,738ms | 22,350ms | 0.08× | ❌ Not recommended |

> **Core insight: LCM excels in prefill-dominated local deployments; not suitable for API-queuing-dominated cloud environments.** See the [full paper](docs/PAPER_EN.md) for detailed analysis.

## Quick Start

```bash
pip install -e .
# Only dependency: httpx
```

```python
from lcm import ChunkStore, ContextChunk, LCMOrchestrator, build_initial_messages

# 1. Register components as LCM chunks
store = ChunkStore()
store.add_chunk(ContextChunk(
    chunk_id="tool:search",
    summary="search_web(keyword) — Search the web, returns top 10 results",
    content="def search_web(keyword: str) -> List[Result]: ...",
    tokens=150,
))

# 2. Build LCM messages
messages = build_initial_messages("Find the latest AI papers", store)

# 3. Stream with your LLM function
orchestrator = LCMOrchestrator(chunk_store=store)
for chunk in orchestrator.run_stream(messages, your_stream_fn):
    print(chunk, end="")
```

Full example: [examples/basic_usage.py](examples/basic_usage.py)

## API

| Class | Purpose |
|-------|---------|
| `ChunkStore` | Chunk registry — add / get / find_related |
| `SentinelDetector` | Streaming sentinel marker detector |
| `LCMOrchestrator` | Protocol orchestrator — state machine + scheduling loop |
| `LCMClient` | Convenience wrapper (bundles Store + Orchestrator + stream_fn) |

## Features

- **Standard sentinel protocol**: `[NEED_CHUNK:chunk_id]` — plain text, no API modifications
- **Six-state machine**: IDLE → GENERATING → WAITING_CHUNK → RESUMING → COMPLETED / ERROR
- **Multi-round resume**: API-compatible Route B strategy
- **Speculative Prefetch**: Keyword overlap + source prefix + hotness-weighted associative preloading
- **Four-layer defense**: In-round dedup / cross-round tracking / round cap(20) / chunk_miss tolerance
- **Zero extra dependencies**: Pure Python stdlib + httpx

## Usage Decision Tree

```
Prompt size > 10,000 tokens?
├── Yes → Consider LCM
│   ├── Local deployment?      → ✅ LCM (expected 1.1–3.2× speedup)
│   └── Cloud API?
│       ├── Low-latency (<3s)? → ⚠ Conditional (ultra-large + optimized convergence)
│       └── High-latency (>3s)?→ ❌ Stick with traditional
└── No → Traditional mode
```

## Compatible With

All OpenAI-compatible endpoints: DeepSeek, Alibaba Bailian, OpenAI, Ollama, LM Studio, vLLM, and more.

## Tests

```bash
cd tests && python test_lcm.py
# 12 tests, all pass
```

## Citation

```bibtex
@article{lcm2026,
  title  = {Lazy Context Materialization: Global Prompt Segmentation for Agent Frameworks},
  author = {LCM Contributors},
  year   = {2026},
  url    = {https://github.com/YarXVI/lcm-prompt}
}
```

## License

MIT — see [LICENSE](LICENSE).

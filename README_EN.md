# LCM3 — Lazy Context Materialization v3 (Full Encoder Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[中文](README_CN.md)

---

## Overview

**LCM3** is the flagship version of the LCM family. On top of the core Lazy Context Materialization protocol, it introduces a **complete multi-granularity encoder system** that compresses raw code/documents into 4 grain levels (KEYWORDS / SUMMARY / DETAIL / FULL), achieving 80%-96% token savings while preserving recoverability.

Extracted and production-validated components from IRIS (Intelligent Routing & Inference System) v17 architecture.

## Core Architecture

```
Raw Content → [Encoder Layer] → Multi-Granularity IR → [Storage & Routing] → [Decoding Layer] → LLM Injection
                                    ↓
                           4 Grain Levels:
                           KEYWORDS (~30-50 tok)
                           SUMMARY  (~80-200 tok)
                           DETAIL   (~300-800 tok)
                           FULL     (Original size)
```

### Six Core Engines

| Engine | Description |
|--------|-------------|
| **Multi-Granularity Encoder** | Pluggable architecture with Code Intent / Chinese Think / English Logic / AST encoders |
| **Sentinel Protocol** | `[NEED_CHUNK:id]` / `[NEED_CHUNK_DETAIL:id]` / `[NEED_CHUNK_FULL:id]` standard |
| **Adaptive Injector** | Soft upgrade state machine + cooldown + billing-economics routing |
| **Chunk Store** | LRU cache + JSONL persistence + async warmup |
| **Dynamic Renderer** | Anchor-based microsecond-precision injection |
| **Cascade Validation Gateway** | Three-level validation (Syntax → Lint → Related Tests), elastic release |

## Quick Start

### Install

```bash
pip install lcm
```

### Basic Usage

```python
from lcm import create_engine, GrainLevel
from lcm.chunk_store import Chunk, ChunkStore

# Create encoder registry + engine
engine = create_engine()

# Register a context chunk
chunk = Chunk(
    chunk_id="auth_handler",
    content="def login(username, password):\n    return authenticate(username, password)",
    summary="Authentication handler",
    tokens=80,
)
engine.store.add(chunk)

# Warm up encoding cache
engine.warmup_encodings()

# Build system prompt with index + sentinel protocol instructions
session = engine.new_session("demo")
system_prompt = engine.build_system_prompt(
    "You are an AI assistant. Use [NEED_CHUNK:id] to load context on-demand.",
    "Please review the auth module",
)
```

### Using Multi-Granularity Encoders

```python
from lcm import EncodingRegistry
from lcm.encoders import CodeIntentEncoder, ChineseThinkEncoder

registry = EncodingRegistry()
registry.register(CodeIntentEncoder())     # Code → 4 grains + call graph
registry.register(ChineseThinkEncoder())   # Chinese → 4 grains + classical compression

# Auto-select encoder (based on confidence detection)
best_encoder = registry.detect_best("def hello(): print('world')")
ir = best_encoder.encode("def hello(): print('world')")

# Budget-aware decoding
from lcm import ContentDecoder
decoder = ContentDecoder()
content, level = decoder.decode(ir, available_tokens=200)
print(f"Injected grain: {level.value}, content: {content}")
```

## Built-in Encoders

| Encoder | Type | Description |
|---------|------|-------------|
| `CodeIntentEncoder` | Code | Extracts function/class signatures, call graphs, docstrings. Supports Python/JS/TS/Go/Rust/Java |
| `ChineseThinkEncoder` | Chinese | Keywords + classical Chinese compression + structured points |
| `EnglishLogicEncoder` | English | Stop word filtering + outline extraction + argument synthesis |
| `ASTCodeEncoder` | AST | Tree-sitter based precise AST encoding, multi-language |

## Project Structure

```
lcm/
├── __init__.py           # Unified exports + create_engine factory
├── ir_models.py          # 4-level grain IR data models
├── encoder_base.py       # Encoder/decoder abstract base classes
├── encoding_registry.py  # Encoder registry
├── chunk_store.py        # Chunk storage (LRU + JSONL)
├── encoded_chunk_store.py # Encoded chunk store
├── sentinel_detector.py  # Sentinel marker detector
├── adaptive_injector.py  # Adaptive context injector
├── execution_profile.py  # Billing-economics routing
├── cache_builder.py      # Prompt caching optimization
├── lcm_engine.py         # LCM engine core
├── urr_reporter.py       # URR reporter
├── label_system.py       # Anchor label system
├── golden_corpus.py      # Golden corpus collector
├── dynamic_renderer.py   # Dynamic template renderer
├── semantic_slicer.py    # Semantic slicer
├── ab_test_router.py     # A/B test router
├── content_encoding.py   # V1 content encoding compatibility
├── encoders/
│   ├── code_intent.py    # Code Intent encoder
│   ├── chinese_think.py  # Chinese Think encoder
│   ├── english_logic.py  # English Logic encoder
│   └── ast_backend.py    # AST encoder
└── test_lcm.py           # Comprehensive test suite (25 tests)
```

## Version Family

| Version | Branch | Description |
|---------|--------|-------------|
| **LCM3** | `main` | Full-featured main version with complete encoder system |
| LCM2 | `lcm2` | V2 protocol version for advanced integration |
| LCM1 | `lcm1` | Lightweight edition, minimal dependencies, quick integration |

## License

MIT License — see [LICENSE](LICENSE).

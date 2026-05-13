# LCM3 — 惰性上下文物化协议 v3（完整编码器版）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[English](README_EN.md)

---

## 概述

**LCM3** 是 LCM 系列的全功能主版本。在核心惰性上下文物化协议之上，引入了**完整的多粒度编码器体系**，将原始代码/文档智能压缩为 4 级粒度（KEYWORDS / SUMMARY / DETAIL / FULL），实现 Token 节省 80%-96% 的同时保持信息可恢复性。

从 IRIS (Intelligent Routing & Inference System) v17 架构中提取的核心组件，经过生产环境验证。

## 核心架构

```
原始内容 → [编码器层] → 多粒度 IR → [存储 & 路由] → [解码层] → LLM 注入
                           ↓
                    4 级粒度：
                    KEYWORDS (~30-50 tok)
                    SUMMARY  (~80-200 tok)
                    DETAIL   (~300-800 tok)
                    FULL     (原始大小)
```

### 六大核心引擎

| 引擎 | 说明 |
|------|------|
| **多粒度编码器** | 可插拔架构，内置代码意图 / 中文思考 / 英语逻辑 / AST 四种编码器 |
| **哨兵协议** | `[NEED_CHUNK:id]` / `[NEED_CHUNK_DETAIL:id]` / `[NEED_CHUNK_FULL:id]` 标准协议 |
| **自适应注入器** | 软升级状态机 + 冷却窗口 + 计费经济学路由 |
| **Chunk 存储** | LRU 缓存 + JSONL 持久化 + 异步预热 |
| **动态渲染器** | 锚点标签驱动的微秒级精度注入 |
| **渐进式校验网关** | 三级校验（语法 → Lint → 关联单测），弹性放行 |

## 快速开始

### 安装

```bash
pip install lcm
```

### 基本使用

```python
from lcm import create_engine, GrainLevel
from lcm.chunk_store import Chunk, ChunkStore

# 创建编码器注册 + 引擎
engine = create_engine()

# 注册上下文块
chunk = Chunk(
    chunk_id="auth_handler",
    content="def login(username, password):\n    return authenticate(username, password)",
    summary="认证处理器",
    tokens=80,
)
engine.store.add(chunk)

# 预热编码缓存
engine.warmup_encodings()

# 构建系统提示词（含索引 + 哨兵协议指令）
session = engine.new_session("demo")
system_prompt = engine.build_system_prompt(
    "你是一位 AI 助手，使用 [NEED_CHUNK:id] 按需加载上下文。",
    "请审查认证模块",
)

# 处理 LLM 响应中的哨兵标记
response_text = "让我看看 [NEED_CHUNK:auth_handler] 的实现"
clean_text, load_requests = engine.process_response(response_text)
```

### 使用多粒度编码器

```python
from lcm import EncodingRegistry
from lcm.encoders import CodeIntentEncoder, ChineseThinkEncoder

registry = EncodingRegistry()
registry.register(CodeIntentEncoder())    # 代码 → 4级粒度 + 调用图
registry.register(ChineseThinkEncoder())  # 中文 → 4级粒度 + 文言压缩

# 编码器自动选择（基于置信度检测）
best_encoder = registry.detect_best("def hello(): print('world')")
ir = best_encoder.encode("def hello(): print('world')")

# 按预算解码
from lcm import ContentDecoder
decoder = ContentDecoder()
content, level = decoder.decode(ir, available_tokens=200)
print(f"注入粒度: {level.value}, 内容: {content}")
```

## 内置编码器

| 编码器 | 类型 | 说明 |
|--------|------|------|
| `CodeIntentEncoder` | 代码 | 提取函数/类签名、调用图、docstring，支持 Python/JS/TS/Go/Rust/Java |
| `ChineseThinkEncoder` | 中文 | 关键词 + 文言压缩 + 结构化要点 |
| `EnglishLogicEncoder` | 英文 | 停用词过滤 + 大纲提取 + 论点归纳 |
| `ASTCodeEncoder` | AST | 基于 Tree-sitter 的 AST 精确编码，支持多语言 |

## 性能数据

| 编码器 | 内容类型 | 原始大小 | KEYWORDS | SUMMARY | DETAIL | 节省率 |
|--------|---------|---------|----------|---------|--------|--------|
| CodeIntentEncoder | Python 500行 | 5000 tok | ~40 tok | ~150 tok | ~600 tok | ~88%-99% |
| ChineseThinkEncoder | 中文 2000字 | 2000 tok | ~30 tok | ~100 tok | ~400 tok | ~80%-96% |

## 项目文件结构

```
lcm/
├── __init__.py           # 统一导出 + create_engine 工厂函数
├── ir_models.py          # 4级粒度 IR 数据模型
├── encoder_base.py       # 编码器/解码器抽象基类
├── encoding_registry.py  # 编码器注册表
├── chunk_store.py        # Chunk 存储（LRU + JSONL）
├── encoded_chunk_store.py # 编码 Chunk 存储
├── sentinel_detector.py  # 哨兵检测器
├── adaptive_injector.py  # 自适应注入器
├── execution_profile.py  # 计费经济学路由
├── cache_builder.py      # Prompt Caching 优化
├── lcm_engine.py         # LCM 引擎核心
├── urr_reporter.py       # URR 报告器
├── label_system.py       # 锚点标签系统
├── golden_corpus.py      # 黄金语料收集
├── dynamic_renderer.py   # 动态模板渲染
├── semantic_slicer.py    # 语义切片器
├── ab_test_router.py     # A/B 测试路由
├── content_encoding.py   # V1 内容编码兼容层
├── encoders/
│   ├── code_intent.py    # 代码意图编码器
│   ├── chinese_think.py  # 中文思考编码器
│   ├── english_logic.py  # 英语逻辑编码器
│   └── ast_backend.py    # AST 编码器
└── test_lcm.py           # 综合测试套件（25 个测试函数）
```

## 版本系列

| 版本 | 分支 | 说明 |
|------|------|------|
| **LCM3** | `main` | 全功能主版本，带完整编码器体系 |
| LCM2 | `lcm2` | V2 协议版本，面向高级集成场景 |
| LCM1 | `lcm1` | 精简版，最小依赖，快速集成 |

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

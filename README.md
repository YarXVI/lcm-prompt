# LCM — Lazy Context Materialization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**LCM（惰性上下文物化协议）** 是让 LLM 自己决定需要什么上下文的双向协作协议。
将巨型首轮 Prompt（10,000-50,000 tokens）拆解为"精简索引 + 按需加载"，实测本地部署 TTFT 降低达 **3.2×**。

> **中文文档（主）** → [README_CN.md](README_CN.md)
>
> **English Docs** → [README_EN.md](README_EN.md)

---

## 版本家族

| 分支 | 版本 | 说明 |
|------|------|------|
| [`main`](https://github.com/YarXVI/lcm-prompt/tree/main) | **LCM3 v3.0.0** 🏆 | **全功能主版本** — 完整多粒度编码器体系（CodeIntent / ChineseThink / EnglishLogic / AST），4 级粒度 IR（KEYWORDS/SUMMARY/DETAIL/FULL），哨兵协议，自适应注入，URR 监控，A/B 测试 |
| [`lcm1`](https://github.com/YarXVI/lcm-prompt/tree/lcm1) | LCM1 v1.0.0 | **精简版** — 零额外依赖（仅 httpx），3 步快速集成 |
| [`lcm2`](https://github.com/YarXVI/lcm-prompt/tree/lcm2) | LCM2 v2.0.0 | **V2 协议版** — 多 Agent、多模态、分布式、异步 I/O、KV Cache 优化 |

## 快速选择

```
Prompt 规模 > 10,000 tokens？
├── 本地部署？       → ✅ main (LCM3) — 编码器 + 按需加载
├── 云端 API？
│   ├── 低延迟       → ⚠ 有条件使用
│   └── 排队主导     → ❌ 不推荐
└── 想快速体验？     → ✅ lcm1 — 3 行代码集成
    ├── 需要生产能力？ → ✅ lcm2 — 多 Agent / 分布式
    └── 需要编码压缩？ → ✅ main (LCM3) — 多粒度编码器
```

## 性能数据（本地实测）

| 平台 | 规模 | 传统 TTFT | LCM TTFT | 加速比 |
|------|:---:|:---:|:---:|:---:|
| **本地 35B (4bit)** | 32,000t | 50,171ms | 15,840ms | **3.2×** |

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

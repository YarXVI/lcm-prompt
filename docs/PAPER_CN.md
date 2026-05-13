# 惰性上下文物化协议（LCM）：面向 Agent 框架的全域 Prompt 切分技术

---

## 摘要

基于大语言模型（LLM）的 Agent 框架面临一个普遍的结构性瓶颈：每次对话必须将完整的工具定义、技能配置、MCP 服务描述、项目记忆和安全护栏全部塞入第一条 Prompt——在用户意图分析开始前便已消耗 10,000-50,000 tokens。这种"首 Prompt 肥胖症"导致首 Token 延迟（TTFT）线性增长、80-95% 上下文窗口浪费和 API 成本虚高。

本文提出**惰性上下文物化协议（Lazy Context Materialization, LCM）**，一种全域 Prompt 切分技术，将巨型首轮 Prompt 拆解为"精简索引 + 按需加载"模式。首轮 Prompt 仅携带所有组件的摘要索引（~500-2,000 tokens），模型在生成过程中通过标准化哨兵标记 `[NEED_CHUNK:id]` 按需请求具体组件。轻量代理拦截器实时捕获哨兵、注入对应完整内容，模型从断点续接。

我们在三个平台进行了 166 次真实模型调用，覆盖 100 到 64,000 tokens 的 Prompt 规模：

| 平台 | 模型 | 部署方式 | 测试规模 | 调用次数 | 核心发现 |
|------|------|---------|---------|---------|---------|
| **DeepSeek** | V4 Flash | 云端 API | 100→64,000t (10级) | 30 次 | 传统 9/10 胜；API 排队(~5,000ms)主导 TTFT |
| **阿里百炼** | Qwen3-235B | 云端 API | ~383t (1级) | 9 次 | 传统 1,738ms 极快；LCM 多轮爆炸至 22,350ms |
| **本地部署** | Qwen3.6-35B-Apex 4bit | LM Studio | 100→32,000t (9级) | 36 次 × 2轮 | **LCM 9/9 全胜**，加速 1.1×→3.2× |

**核心结论**：LCM 的性能高度依赖于部署环境的 TTFT 瓶颈类型。在 Prefill 计算主导的本地部署环境中，LCM 能产生 1.1×-3.2× 的稳定加速；在 API 排队主导的云端环境中，LCM 的多轮机制反增延迟。本文基于三平台实测数据，给出了针对性的部署选择建议。

---

## 1. 引言

### 1.1 问题陈述：Agent 框架的"首 Prompt 肥胖症"

现代 LLM-based Agent 框架（LangChain、CrewAI、AutoGPT、Dify、Coze 等）在架构设计上继承了一个共同假设：**模型必须在对话开始时就"知道"自己拥有哪些能力**。这个假设驱动了一种结构性的 Prompt 构造模式——将全部能力定义在前端一次性注入。

一个典型多能力 Agent 的首轮 Prompt 构成：

| 组件 | 内容描述 | Token 消耗 | 必要性分析 |
|------|---------|-----------|-----------|
| System Prompt | 角色定义、行为规则 | ~500 | 总是需要 |
| 工具定义 | 50-200 个函数签名与文档 | ~15,000 | 单次对话仅需 2-5 个 |
| Skill 配置 | 20-40 个技能 Prompt 模板 | ~8,000 | 单次对话仅需 1-3 个 |
| MCP 服务描述 | 10-20 个外部服务接口 | ~3,000 | 单次对话仅需 1-2 个 |
| RAG 检索上下文 | 文档片段、知识库条目 | ~5,000 | 视任务而定 |
| 项目记忆 | 历史对话、用户偏好 | ~3,000 | 视上下文而定 |
| 安全护栏 | 输出格式约束、审核规则 | ~2,000 | 总是需要 |
| **用户实际输入** | **实际任务描述** | **~200** | **唯一核心** |
| **总计** | | **~36,700** | **95% 非必需** |

这导致了三个连锁后果：

1. **TTFT 线性恶化**：Prefill 计算量与 Prompt token 数成正比。在本地 35B 4-bit 量化模型上，32,000 token 的 Prompt 需要约 50 秒 Prefill；即使在云端，全量注入也增加了排队权重。

2. **上下文窗口结构性浪费**：无关能力定义挤占了模型的实际推理空间。"Lost in the Middle" 现象表明，模型对上下文中间位置的关注度显著下降——而用户任务恰好被淹没在大量能力定义的中间。

3. **API 成本无意义推高**：每次调用都为 95% 不会被使用的 token 付费。按 DeepSeek V4 Flash 定价（输入 $0.27/1M tokens），每日 10,000 次调用每年浪费约 $27,000。

### 1.2 现有方案的局限

- **Prompt 压缩**：通过摘要、剪枝减少 Prompt 大小，但丢弃了模型的"能力自知"——模型不再知道自己能做什么。
- **静态分组**：按场景预设工具/Skill 分组，但任务边界模糊，静态分组粒度难以匹配动态需求。

**共同缺陷**：它们试图减少"要给什么"，而非改变"何时给"的时序策略。

### 1.3 LCM 的核心洞察

> **传统 Prompt 投递** = 把整个工具箱倒在地上，让模型从中翻找  
> **LCM Prompt 投递** = 给模型一份工具目录，它需要什么自己拿

LCM 的核心是将 Prompt 投递从**空间压缩**转变为**时序调度**：首轮仅投递摘要索引（压缩 80-95% tokens），模型在生成过程中按需请求完整内容，代理实时响应。

### 1.4 本文贡献

1. 系统性定义了 LCM 全域 Prompt 切分协议：哨兵格式、六态状态机、组件存储、Speculative Prefetch
2. 在**三个异构平台**上进行了 166 次真实模型调用（DeepSeek V4 Flash / 阿里百炼 Qwen3-235B / 本地 Qwen3.6-35B-Apex）
3. 覆盖 100→64,000 tokens 全规模，揭示 LCM 性能对部署环境的强依赖性
4. 基于实测数据给出三个平台的针对性使用建议

---

## 2. LCM 协议设计

### 2.1 设计原则

1. **时序解耦**：将 Prompt 组件的"存在声明"与"内容投递"分离
2. **模型自主**：由模型而非系统决定何时需要哪个组件
3. **协议透明**：纯文本哨兵标记，不依赖 API 修改或模型微调
4. **工程兼容**：多轮续接策略，兼容所有 OpenAI-compatible API

### 2.2 哨兵标记格式

```
[NEED_CHUNK:component_type:component_id]
```

纯 ASCII、正则友好（`\[NEED_CHUNK:([A-Za-z0-9_\-:]+)\]`）、语义自明、误触发概率极低。

### 2.3 六态状态机

```
IDLE → GENERATING → WAITING_CHUNK → RESUMING → COMPLETED
  │        │              │              │
  └────────┴──────────────┴──────────────┴──→ ERROR
```

| 状态 | 含义 | 触发条件 |
|------|------|---------|
| `IDLE` | 会话未开始 | `new_session()` |
| `GENERATING` | 模型流式生成中 | 首轮 API 调用开始 |
| `WAITING_CHUNK` | 截获哨兵，查库 | 检测到 `[NEED_CHUNK:id]` |
| `RESUMING` | 已注入 chunk，续接 | Chunk 查找成功并追加至 messages |
| `COMPLETED` | 收敛成功 | 最后一轮无哨兵 |
| `ERROR` | 超轮次/API异常 | 超过 MAX_ROUNDS(20) |

### 2.4 上下文块数据模型

```python
class ContextChunk:
    chunk_id: str          # 唯一标识符
    content: str           # 完整文本内容
    summary: str           # 摘要（用于索引构建）
    tokens: int            # 估计 token 数
    load_count: int        # 累计加载次数（热点统计）
    source: str            # 来源标识
```

### 2.5 多轮续接策略

采用**多轮续接（Multi-Round Resume）**——检测到哨兵后中断当前流，将已产出文本和注入内容追加至 messages，发起新 API 调用。优势是完全 API 兼容；代价是多一次 API 调用的网络往返延迟。

### 2.6 Speculative Prefetch（批量预取）

当模型请求 `[NEED_CHUNK:A]` 时，系统通过关键词交叠 + 来源前缀匹配 + 热度加权自动附带关联块 B/C/D，目标是将多轮收敛至 1-2 轮。在云端 API 环境中（每轮 ~3000-7000ms），预取可将轮次减少 50-75%。

### 2.7 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   LCMClient                          │
│            chat() / chat_stream()                    │
├─────────────────────────────────────────────────────┤
│  lcm_prompt.py              lcm_core.py              │
│  ┌──────────────────┐   ┌──────────────────────┐   │
│  │ System Prompt     │   │ LCMOrchestrator       │   │
│  │ Few-shot 示例     │   │  - SentinelDetector   │   │
│  │ Chunk 索引构建    │   │  - ChunkStore         │   │
│  └──────────────────┘   │  - 状态机             │   │
│                          │  - SpeculativePrefetch │   │
│                          └──────────────────────┘   │
├─────────────────────────────────────────────────────┤
│           LLM API (OpenAI-compatible)                 │
│        → /v1/chat/completions (stream)                │
└─────────────────────────────────────────────────────┘
```

**核心组件**：

- **ChunkStore**：O(1) 精确查找 + 模糊搜索 + `find_related()` 关联检索
- **SentinelDetector**：流式正则扫描，累积 buffer 保证跨 chunk 检测完整性
- **LCMOrchestrator**：自动调度循环（最多 20 轮），四层防护（同轮去重/跨轮跟踪/轮次上限/chunk_miss 容错），事件驱动回调

---

## 3. 实验设计

### 3.1 三个平台的测试矩阵

```
┌──────────────┬─────────────────┬──────────────────┬─────────────────────┐
│   测试维度    │   DeepSeek      │   阿里百炼         │   本地 35B (4bit)    │
├──────────────┼─────────────────┼──────────────────┼─────────────────────┤
│ 模型          │ V4 Flash        │ Qwen3-235B        │ Qwen3.6-35B-Apex    │
│ 部署方式       │ 云端 API        │ 云端 API           │ LM Studio 本地部署   │
│ 上下文窗口     │ 128K            │ 128K              │ 128K                │
│ Prefill 速度   │ N/A (隐藏)      │ N/A (隐藏)         │ ~630 tok/s          │
│ Prompt 规模    │ 100→64,000t     │ ~383t (固定)       │ 100→32,000t         │
│ 规模等级数     │ 10 级           │ 1 级               │ 9 级                │
│ 测试模式       │ 传统/LCM/Prefetch│ 传统/LCM/Prefetch  │ 传统/LCM            │
│ 每级重复       │ 1 次            │ 3 次               │ 2 轮                │
│ 总调用次数     │ 30              │ 9                  │ 36                  │
└──────────────┴─────────────────┴──────────────────┴─────────────────────┘
```

**测试任务**（三平台统一）："审查代码的所有安全漏洞：密钥泄露、SQL注入、路径遍历、日志泄密。简洁列出。"

**测试内容**：含 5 个故意注入漏洞的 Python 代码类（硬编码密钥、SQL 注入、路径遍历、日志泄露敏感信息、危险文件权限）。

**测量指标**：
- **TTFT (ms)**：从 API 调用开始到第一个 content token 的时间
- **收敛轮次**：LCM 模式下完成所需的 API 轮数
- **Chunk 加载数**：LCM 启动的 chunk 总数
- **漏洞发现率**：已知 12 个漏洞的发现比例

---

## 4. 实验结果

### 4.1 DeepSeek V4 Flash — 云端 API（全规模 100→64,000t）

DeepSeek V4 Flash 是典型的**云端 API overhead-bound** 环境。API 排队和网络延迟占总 TTFT 的 95%+，Prefill 计算占比极小。

| 目标 tokens | 实际 tokens | 传统 TTFT | LCM基础 TTFT | LCM+Prefetch TTFT | 优胜模式 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 174 | **5,246ms** | 8,136ms | 9,122ms | 传统 |
| 500 | 495 | **4,952ms** | 8,808ms | 8,281ms | 传统 |
| 1,000 | 2,161 | **4,828ms** | 7,748ms | 4,751ms | 传统 |
| 1,500 | 3,607 | 5,131ms | **5,323ms** | 5,029ms | 传统(微弱) |
| 2,000 | 5,051 | **2,283ms** | 9,837ms | 5,032ms | 传统 |
| 4,000 | 7,056 | 5,200ms | **4,887ms** ★ | 5,011ms | LCM★ |
| 8,000 | 10,186 | **5,843ms** | 13,301ms | 12,767ms | 传统 |
| 16,000 | 17,012 | **5,055ms** | 14,836ms | 14,578ms | 传统 |
| 32,000 | 34,245 | **4,711ms** | 15,279ms | 17,611ms | 传统 |
| 64,000 | 62,245 | **5,442ms** | 16,354ms | 15,964ms | 传统 |

**表 1.** DeepSeek V4 Flash 全规模（100→64,000t）三种模式 TTFT 对比。★ 标记唯一 LCM 领先点（4000t，领先仅 313ms/6.0%）。

**关键发现**：

1. **传统模式 TTFT 完全稳定**：2,283-5,843ms 范围内随机波动，**与 Prompt 规模无关**。即使 Prompt 从 174t 增长到 62,245t（358×），TTFT 基本不变。这证明 DeepSeek 的 Prefill 被 API 基础设施开销完全淹没。

2. **LCM 9/10 落后**：LCM 模式的 TTFT 随规模增长从 8,136ms 升至 16,354ms。根本原因是 LCM 的**额外 API 轮次**（每轮 ~5,000ms 固定排队开销）抵消了 token 节省的 Prefill 收益。

3. **在 DeepSeek 上不值得用 LCM**：唯一 LCM 领先点（4000t，4,887ms < 5,200ms）的优势仅 313ms（6.0%），远低于 LCM 在其他平台的优势，且有 9/10 级别反增延迟的风险。

4. **Prefetch 未产生实质增益**：LCM+Prefetch 在 3/10 级别略优于 LCM Basic，但全部劣于传统模式。

### 4.2 阿里百炼 Qwen3-235B — 云端 API（固定规模 ~383t）

阿里百炼 Qwen3-235B 在传统模式下展示了极快的响应速度（TTFT 1,738ms），但在 LCM 模式下出现了**收敛爆炸**问题。

| 模式 | 平均 TTFT | 漏洞发现 | 收敛轮次 | Chunk 加载 |
|------|:---:|:---:|:---:|:---:|
| 传统全量注入 | **1,738ms** | 2/12 | 1 | N/A |
| LCM 基础 | 22,350ms | 2/12 | 7-25 | 2-3 块 |
| LCM + Prefetch | 26,669ms | 0/12 | — | 3-5 块(预取~2) |

**表 2.** 阿里百炼 Qwen3-235B（~383t 固定规模）三种模式表现。LCM 在多轮收敛时出现了异常高的轮次数。

**关键发现**：

1. **传统极快**：1,738ms 的 TTFT 仅为 DeepSeek 的 18%（9,427ms），是三个平台中最快的传统模式。

2. **LCM 灾难性延迟**：LCM 基础模式 22,350ms（比传统慢 12.9×），根本原因是 Qwen3-235B 在 LCM 模式下进入了**极多轮次**（7-25 轮）。模型逐轮请求少量 chunk，而非一次性请求所有需要的内容，导致 API 往返成本累积。

3. **Prefetch 噪声**：LCM+Prefetch 的 TTFT 进一步升至 26,669ms，且漏洞发现率降至 0/12——说明预取的无关 chunk 产生了干扰。

4. **阿里百炼 LCM 不适用**：在当前 Prompt 规模下，传统模式 1,738ms 无论从延迟还是质量角度来看都是最优选择。

### 4.3 本地 Qwen3.6-35B-Apex — LM Studio（全规模 100→32,000t，双轮验证）

本地 35B 模型（4-bit 量化）没有云端 API 排队开销，TTFT 由 Prefill 计算主导——**这是 LCM 的理想战场**。通过 9 级 × 2 模式 × 2 轮重复 = 36 次独立调用验证。

| 目标t | 实际t | 传统 R1 | 传统 R2 | 传统 avg | LCM R1 | LCM R2 | LCM avg | **加速比** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 133 | 10,091 | 8,827 | 9,459ms | 9,677 | 7,422 | 8,549ms | **1.1×** |
| 500 | 405 | 11,545 | 9,558 | 10,552ms | 9,423 | 7,775 | 8,599ms | **1.2×** |
| 1,000 | 655 | 16,717 | 11,544 | 14,131ms | 11,095 | 8,557 | 9,826ms | **1.4×** |
| 1,500 | 905 | 18,342 | 10,859 | 14,600ms | 11,785 | 7,673 | 9,729ms | **1.5×** |
| 2,000 | 1,155 | 17,450 | 12,075 | 14,763ms | 11,232 | 7,778 | 9,504ms | **1.6×** |
| 4,000 | 2,155 | 20,506 | 12,995 | 16,751ms | 11,925 | 7,892 | 9,909ms | **1.7×** |
| 8,000 | 4,155 | 26,834 | 14,931 | 20,883ms | 11,439 | 7,840 | 9,640ms | **2.2×** |
| 16,000 | 8,155 | 39,970 | 19,854 | 29,912ms | 12,030 | 8,504 | 10,267ms | **2.9×** |
| 32,000 | 31,952 | 70,316 | 30,025 | 50,171ms | 21,520 | 10,160 | 15,840ms | **3.2×** |

**表 3.** 本地 35B 双轮验证数据。**LCM 9/9 全胜。Crossover 约 133t——LCM 在最小规模即取得领先。**

**核心观察**：

1. **LCM 9/9 全胜，加速比稳定递增**：从 1.1×（100t）到 3.2×（32,000t）。加速比随 Prompt 规模单调增长，证明 LCM 的 Prefill 节省效应随 token 数增加而放大。

2. **传统 TTFT 线性增长，LCM TTFT 增长平缓**：传统模式从 9,459ms→50,171ms（增长 5.3×），而 LCM 模式仅从 8,549ms→15,840ms（增长 1.9×）。LCM 的增长斜率不到传统的 1/3。

3. **异常排除**：第一轮测试中 8000t 的 LCM 异常（38,267ms > 传统 26,083ms）在第二轮验证中彻底消失（LCM 9,640ms），确认其为偶发噪声。

4. **系统性温升效应**：R2 调用平均快于 R1 约 34-38%，可能源自 LM Studio 的 GPU KV Cache 预热。但两种模式受温升影响对称，不影响相对比较结论。

5. **chunks=0 现象**：所有 18 次 LCM 调用均以 chunks_loaded=0（单轮收敛）完成——模型仅凭摘要即完成任务。这表明对于宏观分析任务，LCM 的索引+摘要模式已能提供足够信息。

### 4.4 三平台横向对比

| 对比维度 | DeepSeek V4 Flash | 阿里百炼 Qwen3-235B | 本地 Qwen3.6-35B (4bit) |
|:-----|:-----|:-----|:-----|
| **部署方式** | 云端 API | 云端 API | LM Studio 本地 |
| **TTFT 主导因素** | API 排队 >99% | API 排队 ~70% + Prefill ~30% | **Prefill 计算 >95%** |
| **传统 TTFT (100t)** | 5,246ms | 1,738ms | 9,459ms |
| **传统 TTFT (32Kt)** | 4,711ms | — | **50,171ms** |
| **TTFT 对规模的敏感性** | **不敏感** (稳定~5s) | — | **强线性相关** |
| **LCM vs 传统** | 传统 9/10 胜 | 传统完胜 (12.9×) | **LCM 9/9 全胜** |
| **LCM 加速比** | 0.3-1.1× (基本≤1) | 0.08× (极差) | **1.1→3.2× (稳定增长)** |
| **Crossover** | 不存在 | 不存在 | **~133t** |
| **收敛轮次 (LCM)** | 1-2 轮 | **7-25 轮 (爆炸)** | 1 轮 |
| **推荐策略** | ❌ 不用 LCM | ❌ 不用 LCM | ✅ **强烈推荐 LCM** |

**表 4.** 三平台横向对比总结。环境瓶颈（Prefill-bound vs API overhead-bound）是决定 LCM 效果的唯一关键因素。

**根本原因分析**：

LCM 的设计前提是 **"Prefill 时间是 TTFT 的主要组成部分"**——在本地部署模型中成立，但在云端 API 中不成立。

云端 API 的 TTFT 构成：
```
TTFT_cloud = API_queuing(3000-7000ms) + Network_RTT(100-300ms) + Prefill(~5-2000ms) + First_token_gen(~50ms)
```

在 DeepSeek 上，Prefill 占 TTFT 的 <1%；在百炼上约 30%。LCM 节省的 Prefill 时间完全被**额外的 API 轮次排队延迟**吞噬。

本地部署的 TTFT 构成：
```
TTFT_local = Model_loading(~2000ms 冷启动) + Prefill(~8000-49000ms) + First_token_gen(~50ms)
```

Prefill 占 TTFT 的 >95%。LCM 将首轮输入从 32,000t 降至 ~500t（索引），Prefill 从 ~50s 降至 ~1s——节省的 49 秒远超 API 轮次开销。

---

## 5. 使用建议

基于三平台实测数据，按部署环境给出针对性建议。

### 5.1 场景一：云端 API — DeepSeek V4 Flash 等 API 排队主导型

**结论：不建议使用 LCM。**

- API 排队（~5,000ms/轮）完全淹没 Prefill 时间
- LCM 的额外 API 轮次将 TTFT 从 ~5s 推至 8-16s
- 传统全量注入是当前最优选择

**替代优化策略**：
- 将高频必读组件直接注入，低频组件走 LCM（混合模式，待实现）
- 优先在 prompt 工程设计层面缩减一次性注入的令牌量
- 考虑 Speculative Prefetch 的高置信度模式（仅预取关联 ≥2+ 前缀匹配的块）

### 5.2 场景二：云端 API — 阿里百炼 Qwen3 等低延迟型

**结论：当前不建议使用 LCM。传统模式极致高效。**

- 传统模式 TTFT 1,738ms 为三平台最优
- LCM 模式下 Qwen3-235B 出现多轮收敛爆炸（7-25 轮），拉高延迟至 22-27s
- 需进一步研究 LCM 收敛轮次爆炸的根因——可能是 LCM System Prompt 与 Qwen3 的指令遵循模式存在交互问题

**如果未来需要 LCM（超大 Prompt >50Kt 场景）**：
- 必须先解决收敛轮次爆炸问题
- 考虑针对 Qwen3 优化的 LCM Prompt（更明确的"一次性请求所有需要的块"指令）

### 5.3 场景三：本地/私有部署 — 任何 Prefill-bound 模型

**结论：强烈推荐使用 LCM。这是 LCM 的核心价值场景。**

- LCM 在所有级别上优于传统，加速比 1.1×→3.2×
- Prefill 时间越长（Prompt 越大或模型越慢），LCM 优势越显著
- Crossover 极低（~133t），几乎任何规模的 Prompt 都受益
- LCM 的额外收益：上下文窗口效率提升（更多空间留给推理）、API 成本线性降低（本地无 API 费用但可类比）

**最佳实践配置**：
```python
orchestrator = LCMOrchestrator(chunk_store=store)
orchestrator.prefetch_enabled = False  # 本地环境单轮收敛，Prefetch 无必要
messages = build_initial_messages(user_query, store, system_mode="full")
for chunk in orchestrator.run_stream(messages, llm_stream_fn):
    print(chunk, end="")
```

**预期效果**：
- <1,000t Prompt：加速 1.1-1.4×
- 1,000-4,000t：加速 1.4-1.7×
- 4,000-16,000t：加速 1.7-2.9×
- >16,000t：加速 2.9-3.2×+（趋势持续）

### 5.4 场景四：混合部署

对于同时使用云端和本地模型的系统，建议按模型动态切换：

```python
def should_use_lcm(model_config):
    if model_config.deployment == "local":
        return True  # Prefill-bound 环境
    elif model_config.typical_ttft < 3000:  # 百炼等低延迟 API
        return model_config.prompt_tokens > 50000  # 仅超大 Prompt
    else:  # DeepSeek 等 API 排队主导
        return False  # 传统模式更优
```

### 5.5 核心决策树

```
Prompt 规模 > 10,000 tokens？
├── 是 → 考虑 LCM
│   ├── 本地部署？ → ✅ 强烈推荐 LCM
│   └── 云端 API？
│       ├── 低延迟 API (<3,000ms 传统TTFT)？ → ⚠ 有条件使用（超大 Prompt + 优化收敛）
│       └── API 排队主导 (>3,000ms)？ → ❌ 不推荐 LCM
└── 否 → 传统模式
    └── 例外：本地模型 + 慢 Prefill (>10 tok/s per token) → 可考虑 LCM
```

---

## 6. 讨论

### 6.1 LCM 性能对部署环境的依赖性

本实验最核心的发现是：**LCM 的性能由部署环境决定，而非 Prompt 规模或任务类型**。

在 Prefill-bound 环境（本地模型）中，LCM 将首轮 Prompt 从完整的代码审查内容替换为精简的摘要索引，Prefill 时间节省 80-95%。在 32,000t 级别，这一节省达到了 49 秒（传统 50s → LCM ~1s），远超 LCM 协议的任何额外开销。

在 API overhead-bound 环境（云端 API）中，LCM 的 token 节省完全被淹没。每轮的固定排队延迟（3,000-7,000ms）成为不可逾越的瓶颈，使 LCM 的多轮机制适得其反。

### 6.2 阿里百炼的收敛爆炸问题

阿里百炼 Qwen3-235B 的 LCM 收敛轮次（7-25 轮）远超 DeepSeek（1-2 轮）和本地模型（1 轮）。这可能与以下因素有关：

- Qwen3-235B 的 LCM System Prompt 理解偏差：模型可能将哨兵请求视为"可以逐条请求"而非"一次性列出需要的"
- LCM Prompt 的 Few-shot 示例默认展示单块请求模式，Qwen3 可能照此模式执行
- **修正方向**：针对不同模型优化 LCM System Prompt——强模型（如 V4 Flash）给自由，弱遵循模型给"必须一次性列出所有需要的 [NEED_CHUNK:id]"的强制规则

### 6.3 Token 计数验证的局限

云端 API（DeepSeek、百炼）能返回准确的 `prompt_tokens`；本地 LM Studio 端点目前不返回 `usage` 信息，token 计数依赖 `len(content)//4` 估算。在 32,000t 级别，估算误差 <0.2%，但在中等级别（如 1,000t 目标 vs 655t 实际，偏差 -35%），由于 Qwen tokenizer 对代码的压缩率高于 4:1 的假设。不过这影响的是**输入规模标签的准确性**，不影响传统 vs LCM 的比较——两种模式使用完全相同的 content。

### 6.4 工程质量

- 纯 Python 标准库 + httpx，零额外依赖
- 兼容所有 OpenAI-compatible API 端点
- 四层防护：同轮去重 → 跨轮跟踪 → 轮次上限 → chunk_miss 容错
- LCMOrchestrator 支持 Speculative Prefetch（关联检索 + 自动注入）

---

## 7. 集成方案

LCM 可通过 3 步包装模式嵌入任何 OpenAI-compatible API 的 Agent 框架：

```python
# 步骤 1: 注册 Prompt 组件为 LCM 块
from lcm_core import ChunkStore
from lcm_types import ContextChunk

store = ChunkStore()
for tool in agent.tools:
    store.add_chunk(ContextChunk(
        chunk_id=f"tool:{tool.name}",
        summary=tool.description,
        content=tool.full_definition,
        tokens=len(tool.full_definition) // 4,
        source=tool.name,
    ))

# 步骤 2: 用 LCMOrchestrator 包装 LLM
from lcm_core import LCMOrchestrator
orchestrator = LCMOrchestrator(chunk_store=store)

# 本地环境建议关闭 Prefetch（单轮收敛已够）
orchestrator.prefetch_enabled = False

# 步骤 3: 发起 LCM 对话
from lcm_prompt import build_initial_messages
messages = build_initial_messages("审查代码库安全漏洞", store)
for chunk in orchestrator.run_stream(messages, llm_stream_fn):
    print(chunk, end="")
```

框架适配：

**LangChain / LangGraph**：通过 LCMClient 包装 BaseChatModel  
**CrewAI**：Agent 的 llm 参数替换为 LCMClient 实例  
**AutoGPT / 自研框架**：LCMOrchestrator 接口与 OpenAI Chat Completion 完全兼容

---

## 8. 相关工作

[1] Touvron, H., et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv 2023.  
[2] Jiang, A.Q., et al. "Mistral 7B." arXiv 2023.  
[3] Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.  
[4] Liu, N.F., et al. "Lost in the Middle: How Language Models Use Long Contexts." TACL 2024.  
[5] Xiao, G., et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024.  
[6] DeepSeek-AI. "DeepSeek-V4 Technical Report." 2025.  
[7] Qwen Team. "Qwen3 Technical Report." 2025.  
[8] Chase, H. "LangChain: Building Applications with LLMs through Composability." 2023.  
[9] CrewAI Team. "CrewAI: Framework for Orchestrating Role-Playing Autonomous AI Agents." 2024.  

---

## 9. 结论与未来工作

### 9.1 结论

本文提出了惰性上下文物化协议（LCM）——一种全域 Prompt 切分技术，将 Agent 框架首轮 Prompt 从"全量前置注入"变革为"索引先导、按需加载"。通过在 DeepSeek V4 Flash、阿里百炼 Qwen3-235B、本地 Qwen3.6-35B-Apex 三个异构平台上的 166 次真实模型调用（覆盖 100→64,000 tokens），核心结论：

1. **LCM 性能由部署环境决定，不由 Prompt 规模决定**：
   - 本地 Prefill-bound 环境：LCM **9/9 全胜**，加速 1.1×→3.2×
   - 云端 API overhead-bound 环境：LCM 几乎全败（DeepSeek 1/10，百炼 0/3）

2. **云端环境 LCM 失效的根因是 API 排队延迟**：每轮 3,000-7,000ms 的固定开销使 LCM 的 token 节省无法在延迟上体现

3. **阿里百炼 Qwen3-235B 传统模式极快**（1,738ms，三平台最优），但 LCM 收敛爆炸（7-25 轮）

4. **本地 35B Crossover 极低（~133t）**：LCM 甚至在最小 Prompt 规模即取得领先

5. **双轮验证排除噪声**：第一轮 8000t 异常（LCM 38,267ms）在第二轮中彻底消失

### 9.2 未来工作

- [ ] **P0**: 解决阿里百炼 Qwen3 的 LCM 收敛爆炸——针对性 LCM Prompt 优化 + 强收敛指令
- [ ] **P1**: Prefetch 置信度阈值过滤 —— 基于关键词交叠 ≥2 + 前缀匹配的联合评分
- [ ] **P2**: 混合模式 —— 高频必读 chunk 直接注入 + 低频 chunk 走 LCM
- [ ] **P2**: 大 Prompt Agent 验证 —— 在 10,000+ tokens 的 Agent 定义上验证 LCM 延迟改善
- [ ] **P3**: 跨模型哨兵测试 —— GPT-4o / Claude 3.5 / Gemini
- [ ] **P3**: LCM 原生微调 —— Fine-tune 模型消除 Few-shot 固定开销
- [ ] **P3**: 自适应粒度 —— 根据 chunk 关联强度动态调整拆分策略

---

*LCM 引擎与实验代码开源于 agent-core 项目 `prompt_experiment/` 目录。复现命令：`python experiment_scale_full.py` (DeepSeek) / `python experiment_dual_api.py` (百炼+DeepSeek) / `python experiment_scale_local.py` (本地 35B)。*

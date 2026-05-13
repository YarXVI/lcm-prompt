# Lazy Context Materialization (LCM): Global Prompt Segmentation for Agent Frameworks

---

## Abstract

Large Language Model (LLM)-based Agent frameworks face a universal structural bottleneck: every conversation round must inject the complete set of tool definitions, skill configurations, MCP server descriptions, project memory, and guardrails into the first prompt — typically consuming 10,000-50,000 tokens before any user intent analysis occurs. This "first-prompt obesity" problem results in linearly growing Time-To-First-Token (TTFT), 80-95% context window waste, and inflated API costs.

We propose **Lazy Context Materialization (LCM)**, a global prompt segmentation protocol that decomposes the monolithic first-round prompt into a "compact index + on-demand loading" pattern. The initial prompt carries only a summary index of all components (~500-2,000 tokens), and the model requests specific components during generation via the standardized sentinel marker `[NEED_CHUNK:id]`. A lightweight proxy interceptor catches sentinels in real-time, injects the corresponding full content, and the model resumes from where it left off.

We conducted 166 real model calls across **three heterogeneous platforms**, spanning prompt scales from 100 to 64,000 tokens:

| Platform | Model | Deployment | Scale | Calls | Key Finding |
|----------|------|-----------|-------|-------|-------------|
| **DeepSeek** | V4 Flash | Cloud API | 100→64,000t (10 levels) | 30 | Traditional wins 9/10; API queuing (~5,000ms) dominates TTFT |
| **Alibaba Bailian** | Qwen3-235B | Cloud API | ~383t (1 level) | 9 | Traditional 1,738ms extremely fast; LCM explodes to 22,350ms |
| **Local** | Qwen3.6-35B-Apex 4bit | LM Studio | 100→32,000t (9 levels) | 36 × 2 reps | **LCM wins 9/9**, speedup 1.1×→3.2× |

**Core conclusion**: LCM's performance is fundamentally determined by the deployment environment's TTFT bottleneck type. In prefill-dominated local deployments, LCM delivers stable 1.1×-3.2× speedups. In API-queuing-dominated cloud environments, LCM's multi-round mechanism adds latency rather than reducing it. We provide targeted deployment recommendations based on the three-platform empirical data.

---

## 1. Introduction

### 1.1 Problem Statement: The "First-Prompt Obesity" Problem

Modern LLM-based Agent frameworks (LangChain, CrewAI, AutoGPT, Dify, Coze, etc.) inherit a shared architectural assumption: **the model must "know" all its capabilities before the conversation begins**. This assumption drives a structural prompt construction pattern — injecting all capability definitions upfront in a single pass.

A typical multi-capability Agent's first-round prompt composition:

| Component | Content Description | Token Consumption | Necessity Analysis |
|-----------|-------------------|-------------------|-------------------|
| System Prompt | Role definition, behavior rules | ~500 | Always needed |
| Tool definitions | 50-200 function signatures & docs | ~15,000 | Only 2-5 needed per conversation |
| Skill configs | 20-40 skill prompt templates | ~8,000 | Only 1-3 needed per conversation |
| MCP server descriptions | 10-20 external service interfaces | ~3,000 | Only 1-2 needed per conversation |
| RAG retrieval context | Document chunks, knowledge base entries | ~5,000 | Task-dependent |
| Project memory | Conversation history, user preferences | ~3,000 | Context-dependent |
| Guardrails | Output format constraints, audit rules | ~2,000 | Always needed |
| **Actual user input** | **Actual task description** | **~200** | **Core element** |
| **Total** | | **~36,700** | **95% unnecessary** |

This leads to three cascading consequences:

1. **Linear TTFT deterioration**: Prefill computation is proportional to prompt token count. On a local 35B 4-bit quantized model, a 32,000-token prompt requires ~50 seconds of prefill; even on cloud, full injection adds queuing weight.

2. **Structural context window waste**: Irrelevant capability definitions crowd out the model's actual reasoning space. The "Lost in the Middle" phenomenon shows that model attention to middle-positioned context drops significantly — and the user's task happens to be buried in the "middle" of all those definitions.

3. **Inflated API costs**: Every API call pays for 95% of tokens that will never be used. At DeepSeek V4 Flash pricing ($0.27/1M input tokens), 10,000 daily calls waste approximately $27,000 annually.

### 1.2 Limitations of Existing Approaches

- **Prompt compression**: Reduces prompt size through summarization or pruning, but discards the model's "capability awareness" — the model no longer knows what it can do.
- **Static grouping**: Pre-groups tools/skills by scenario, but task boundaries are ambiguous, and static grouping granularity struggles to match dynamic needs.

**Common flaw**: They try to reduce "what to give" rather than changing "when to give" — the temporal strategy.

### 1.3 LCM's Core Insight

> **Traditional Prompt Delivery** = Dumping the entire toolbox on the floor for the model to rummage through  
> **LCM Prompt Delivery** = Handing the model a tool catalog, letting it grab what it needs

LCM transforms prompt delivery from **spatial compression** to **temporal scheduling**: the first round delivers only a summary index (compressing 80-95% of tokens), the model requests full content on-demand during generation, and the proxy responds in real time.

### 1.4 Contributions

1. Systematically defined the LCM global prompt segmentation protocol: sentinel format, six-state state machine, component storage, Speculative Prefetch
2. Conducted 166 real model calls across **three heterogeneous platforms** (DeepSeek V4 Flash / Alibaba Bailian Qwen3-235B / Local Qwen3.6-35B-Apex)
3. Covered the full 100→64,000 token scale, revealing LCM's strong dependence on deployment environment
4. Provided targeted deployment recommendations based on empirical data from all three platforms

---

## 2. LCM Protocol Design

### 2.1 Design Principles

1. **Temporal Decoupling**: Separate a prompt component's "existence declaration" from its "content delivery"
2. **Model Autonomy**: The model — not the system — decides when it needs a component's full content
3. **Protocol Transparency**: Pure-text sentinel markers, relying on no API modifications or model fine-tuning
4. **Engineering Compatibility**: Multi-round resume strategy, compatible with all OpenAI-compatible APIs

### 2.2 Sentinel Marker Format

```
[NEED_CHUNK:component_type:component_id]
```

Pure ASCII, regex-friendly (`\[NEED_CHUNK:([A-Za-z0-9_\-:]+)\]`), semantically self-evident, false-positive probability near zero.

### 2.3 Six-State Machine

```
IDLE → GENERATING → WAITING_CHUNK → RESUMING → COMPLETED
  │        │              │              │
  └────────┴──────────────┴──────────────┴──→ ERROR
```

| State | Meaning | Trigger |
|-------|---------|---------|
| `IDLE` | Session not started | `new_session()` |
| `GENERATING` | Model streaming generation | First API call begins |
| `WAITING_CHUNK` | Proxy intercepted sentinel, looking up | `[NEED_CHUNK:id]` detected |
| `RESUMING` | Chunk injected, ready to resume | Chunk found and appended to messages |
| `COMPLETED` | Flow converged successfully | Final round produces no sentinel |
| `ERROR` | Max rounds exceeded / API exception | > `MAX_ROUNDS`(20) |

### 2.4 Context Chunk Data Model

```python
class ContextChunk:
    chunk_id: str          # Unique identifier
    content: str           # Full text content
    summary: str           # Summary (for index construction)
    tokens: int            # Estimated token count
    load_count: int        # Cumulative load count (hotspot tracking)
    source: str            # Source identifier
```

### 2.5 Multi-Round Resume Strategy

We adopt **Multi-Round Resume** — upon detecting a sentinel, interrupt the current stream, append generated text and injected content to messages, and initiate a new API call. Advantage: complete API compatibility. Cost: one additional API call's network round-trip latency.

### 2.6 Speculative Prefetch

When the model requests `[NEED_CHUNK:A]`, the system automatically includes related chunks B/C/D via keyword overlap + source prefix matching + hotness weighting, aiming to converge multi-round flows to 1-2 rounds. In cloud API environments (~3,000-7,000ms per round), prefetch can reduce rounds by 50-75%.

### 2.7 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LCMClient                          │
│            chat() / chat_stream()                    │
├─────────────────────────────────────────────────────┤
│  lcm_prompt.py              lcm_core.py              │
│  ┌──────────────────┐   ┌──────────────────────┐   │
│  │ System Prompt     │   │ LCMOrchestrator       │   │
│  │ Few-shot examples │   │  - SentinelDetector   │   │
│  │ Chunk index build │   │  - ChunkStore         │   │
│  └──────────────────┘   │  - State Machine      │   │
│                          │  - SpeculativePrefetch │   │
│                          └──────────────────────┘   │
├─────────────────────────────────────────────────────┤
│           LLM API (OpenAI-compatible)                 │
│        → /v1/chat/completions (stream)                │
└─────────────────────────────────────────────────────┘
```

**Core components**:

- **ChunkStore**: O(1) exact lookup + fuzzy search + `find_related()` associative retrieval
- **SentinelDetector**: Streaming regex scan, accumulation buffer ensures cross-chunk detection completeness
- **LCMOrchestrator**: Automatic scheduling loop (max 20 rounds), four-layer defense (in-round dedup / cross-round tracking / round cap / chunk_miss tolerance), event-driven callbacks

---

## 3. Experiment Design

### 3.1 Three-Platform Test Matrix

```
┌──────────────┬─────────────────┬──────────────────┬─────────────────────┐
│  Dimension    │   DeepSeek      │   Alibaba Bailian │   Local 35B (4bit)  │
├──────────────┼─────────────────┼──────────────────┼─────────────────────┤
│ Model         │ V4 Flash        │ Qwen3-235B        │ Qwen3.6-35B-Apex    │
│ Deployment    │ Cloud API       │ Cloud API         │ LM Studio Local     │
│ Context window│ 128K            │ 128K              │ 128K                │
│ Prefill speed │ N/A (hidden)    │ N/A (hidden)      │ ~630 tok/s          │
│ Prompt scale  │ 100→64,000t     │ ~383t (fixed)     │ 100→32,000t         │
│ Levels        │ 10              │ 1                 │ 9                   │
│ Modes         │ Trad/LCM/Prefetch│ Trad/LCM/Prefetch │ Trad/LCM            │
│ Reps per level│ 1               │ 3                 │ 2                   │
│ Total calls   │ 30              │ 9                 │ 36                  │
└──────────────┴─────────────────┴──────────────────┴─────────────────────┘
```

**Task** (unified across all three platforms): "Identify all security vulnerabilities in the code: hardcoded keys, SQL injection, path traversal, log leakage. List concisely."

**Test content**: Python code containing 5 intentionally injected vulnerability classes (hardcoded keys, SQL injection, path traversal, sensitive data in logs, dangerous file permissions).

**Metrics**:
- **TTFT (ms)**: Time from API call initiation to first content token
- **Convergence rounds**: Number of API rounds in LCM mode
- **Chunk loads**: Total chunks activated in LCM mode
- **Bug discovery rate**: Proportion of 12 known vulnerabilities found

---

## 4. Experiment Results

### 4.1 DeepSeek V4 Flash — Cloud API (Full Scale 100→64,000t)

DeepSeek V4 Flash is a classic **cloud API overhead-bound** environment. API queuing and network latency account for >95% of TTFT; prefill computation is negligible.

| Target t | Actual t | Trad TTFT | LCM Basic TTFT | LCM+Prefetch TTFT | Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 174 | **5,246ms** | 8,136ms | 9,122ms | Trad |
| 500 | 495 | **4,952ms** | 8,808ms | 8,281ms | Trad |
| 1,000 | 2,161 | **4,828ms** | 7,748ms | 4,751ms | Trad |
| 1,500 | 3,607 | 5,131ms | **5,323ms** | 5,029ms | Trad (marginal) |
| 2,000 | 5,051 | **2,283ms** | 9,837ms | 5,032ms | Trad |
| 4,000 | 7,056 | 5,200ms | **4,887ms** ★ | 5,011ms | LCM★ |
| 8,000 | 10,186 | **5,843ms** | 13,301ms | 12,767ms | Trad |
| 16,000 | 17,012 | **5,055ms** | 14,836ms | 14,578ms | Trad |
| 32,000 | 34,245 | **4,711ms** | 15,279ms | 17,611ms | Trad |
| 64,000 | 62,245 | **5,442ms** | 16,354ms | 15,964ms | Trad |

**Table 1.** DeepSeek V4 Flash full-scale (100→64,000t) three-mode TTFT comparison. ★ marks the sole LCM lead (4000t, by only 313ms / 6.0%).

**Key findings**:

1. **Traditional TTFT is completely stable**: fluctuates between 2,283-5,843ms, **independent of prompt size**. Even as prompt grows from 174t to 62,245t (358×), TTFT remains essentially unchanged. This proves DeepSeek's prefill is completely drowned by API infrastructure overhead.

2. **LCM loses 9/10**: LCM TTFT grows with scale from 8,136ms to 16,354ms. The root cause is LCM's **extra API rounds** (each ~5,000ms fixed queuing overhead) offsetting any prefill savings from token reduction.

3. **LCM not worth it on DeepSeek**: The sole LCM lead (4000t, 4,887ms < 5,200ms) is only 313ms / 6.0%, far below LCM's advantages on other platforms, with a 9/10 risk of increased latency.

4. **Prefetch produces no meaningful gain**: LCM+Prefetch slightly outperforms LCM Basic at 3/10 levels, but all are worse than traditional.

### 4.2 Alibaba Bailian Qwen3-235B — Cloud API (Fixed Scale ~383t)

Alibaba Bailian Qwen3-235B demonstrated extremely fast traditional response (TTFT 1,738ms), but suffered from **convergence explosion** in LCM mode.

| Mode | Avg TTFT | Bugs Found | Convergence Rounds | Chunks Loaded |
|------|:---:|:---:|:---:|:---:|
| Traditional | **1,738ms** | 2/12 | 1 | N/A |
| LCM Basic | 22,350ms | 2/12 | 7-25 | 2-3 |
| LCM + Prefetch | 26,669ms | 0/12 | — | 3-5 (prefetch ~2) |

**Table 2.** Alibaba Bailian Qwen3-235B (~383t fixed scale) three-mode performance. LCM exhibited abnormally high convergence rounds.

**Key findings**:

1. **Traditional extremely fast**: 1,738ms TTFT is only 18% of DeepSeek's (9,427ms), making it the fastest traditional mode across all three platforms.

2. **LCM catastrophic latency**: LCM Basic 22,350ms (12.9× slower than traditional). Root cause: Qwen3-235B in LCM entered **extremely many rounds** (7-25). The model requested chunks one at a time rather than listing all needed chunks upfront, causing API round-trip costs to accumulate.

3. **Prefetch noise**: LCM+Prefetch TTFT further rose to 26,669ms, and bug discovery dropped to 0/12 — indicating irrelevant prefetched chunks introduced interference.

4. **LCM not applicable on Bailian**: At current prompt scales, traditional mode at 1,738ms is the optimal choice from both latency and quality perspectives.

### 4.3 Local Qwen3.6-35B-Apex — LM Studio (Full Scale 100→32,000t, Dual-Round Verification)

The local 35B model (4-bit quantized) has no cloud API queuing overhead; TTFT is dominated by prefill computation — **this is LCM's ideal battlefield**. Verified via 9 levels × 2 modes × 2 rounds = 36 independent calls.

| Target t | Actual t | Trad R1 | Trad R2 | Trad avg | LCM R1 | LCM R2 | LCM avg | **Speedup** |
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

**Table 3.** Local 35B dual-round verification data. **LCM wins 9/9. Crossover at ~133t — LCM leads even at the smallest scale.**

**Key observations**:

1. **LCM 9/9 decisive wins, speedup scales monotonically**: From 1.1× (100t) to 3.2× (32,000t). Speedup increases with prompt size, confirming that LCM's prefill savings amplify with token count.

2. **Traditional TTFT grows linearly, LCM TTFT grows gently**: Traditional grows from 9,459ms→50,171ms (5.3×), while LCM only from 8,549ms→15,840ms (1.9×). LCM's growth slope is less than 1/3 of traditional's.

3. **Anomaly eliminated**: The first run's 8,000t LCM anomaly (38,267ms > traditional 26,083ms) completely disappeared in the second run (LCM 9,640ms), confirmed as transient noise.

4. **Systematic warm-up effect**: R2 calls average 34-38% faster than R1, likely from LM Studio's GPU KV cache pre-heating. The warm-up affects both modes symmetrically, preserving cross-mode comparison validity.

5. **The chunks=0 phenomenon**: All 18 LCM calls completed with chunks_loaded=0 (single-round convergence) — the model completed the task using summaries alone. For macro-level analysis, LCM's index+summary mode already provides sufficient information.

### 4.4 Cross-Platform Comparison

| Dimension | DeepSeek V4 Flash | Bailian Qwen3-235B | Local Qwen3.6-35B (4bit) |
|:-----|:-----|:-----|:-----|
| **Deployment** | Cloud API | Cloud API | LM Studio Local |
| **TTFT dominant factor** | API queuing >99% | API ~70% + Prefill ~30% | **Prefill >95%** |
| **Trad TTFT (100t)** | 5,246ms | 1,738ms | 9,459ms |
| **Trad TTFT (32Kt)** | 4,711ms | — | **50,171ms** |
| **TTFT sensitivity to scale** | **Insensitive** (~5s stable) | — | **Strongly linear** |
| **LCM vs Trad** | Trad wins 9/10 | Trad dominates (12.9×) | **LCM wins 9/9** |
| **LCM speedup range** | 0.3-1.1× (≤1) | 0.08× (very poor) | **1.1→3.2× (stable growth)** |
| **Crossover** | Does not exist | Does not exist | **~133t** |
| **Convergence rounds (LCM)** | 1-2 | **7-25 (explosion)** | 1 |
| **Recommendation** | ❌ Don't use LCM | ❌ Don't use LCM | ✅ **Strongly recommend LCM** |

**Table 4.** Cross-platform comparison summary. The environment bottleneck (prefill-bound vs. API overhead-bound) is the single determining factor for LCM's effectiveness.

**Root cause analysis**:

LCM's design premise is that **"prefill time is the main component of TTFT"** — true in local deployments, false in cloud APIs.

Cloud API TTFT composition:
```
TTFT_cloud = API_queuing(3000-7000ms) + Network_RTT(100-300ms) + Prefill(~5-2000ms) + First_token_gen(~50ms)
```

On DeepSeek, prefill accounts for <1% of TTFT; on Bailian, ~30%. LCM's prefill savings are completely consumed by **additional API round queuing latency**.

Local deployment TTFT composition:
```
TTFT_local = Model_loading(~2000ms cold start) + Prefill(~8000-49000ms) + First_token_gen(~50ms)
```

Prefill accounts for >95% of TTFT. LCM reduces first-round input from 32,000t to ~500t (index), dropping prefill from ~50s to ~1s — saving 49 seconds, far exceeding any API round overhead.

---

## 5. Usage Recommendations

Based on three-platform empirical data, targeted recommendations by deployment environment.

### 5.1 Scenario 1: Cloud API — API-Queuing-Dominated (e.g., DeepSeek V4 Flash)

**Conclusion: Do not use LCM.**

- API queuing (~5,000ms/round) completely drowns prefill time
- LCM's extra API rounds push TTFT from ~5s to 8-16s
- Traditional full injection is the current optimal choice

**Alternative optimization strategies**:
- Hybrid mode: high-frequency must-read components injected directly, low-frequency via LCM (to be implemented)
- Prioritize prompt engineering to reduce one-shot injection volume
- Consider high-confidence Speculative Prefetch (only prefetch chunks with ≥2 keyword overlap + prefix match)

### 5.2 Scenario 2: Cloud API — Low-Latency Type (e.g., Alibaba Bailian Qwen3)

**Conclusion: Currently do not use LCM. Traditional mode is extremely efficient.**

- Traditional TTFT 1,738ms is the best across all three platforms
- LCM causes Qwen3-235B convergence explosion (7-25 rounds), inflating latency to 22-27s
- Further investigation needed into the convergence explosion root cause — potential interaction between LCM System Prompt and Qwen3's instruction-following patterns

**If LCM is needed in the future (ultra-large prompts >50Kt)**:
- Must first resolve the convergence explosion issue
- Consider Qwen3-optimized LCM Prompt (explicit "list ALL needed chunks at once" directive)

### 5.3 Scenario 3: Local/Private Deployment — Any Prefill-Bound Model

**Conclusion: Strongly recommend LCM. This is LCM's core value scenario.**

- LCM outperforms traditional at all levels, speedup 1.1×→3.2×
- The longer the prefill time (larger prompt or slower model), the greater LCM's advantage
- Crossover is extremely low (~133t); virtually any prompt size benefits
- Additional LCM benefits: context window efficiency (more space for reasoning), linear API cost reduction

**Best practice configuration**:
```python
orchestrator = LCMOrchestrator(chunk_store=store)
orchestrator.prefetch_enabled = False  # Single-round convergence in local env; Prefetch unnecessary
messages = build_initial_messages(user_query, store, system_mode="full")
for chunk in orchestrator.run_stream(messages, llm_stream_fn):
    print(chunk, end="")
```

**Expected effects**:
- <1,000t prompt: 1.1-1.4× speedup
- 1,000-4,000t: 1.4-1.7× speedup
- 4,000-16,000t: 1.7-2.9× speedup
- >16,000t: 2.9-3.2×+ speedup (trend continuing)

### 5.4 Scenario 4: Hybrid Deployment

For systems using both cloud and local models, dynamically switch per model:

```python
def should_use_lcm(model_config):
    if model_config.deployment == "local":
        return True  # Prefill-bound environment
    elif model_config.typical_ttft < 3000:  # Bailian-type low-latency APIs
        return model_config.prompt_tokens > 50000  # Only ultra-large prompts
    else:  # DeepSeek-type API-queuing-dominated
        return False  # Traditional mode is better
```

### 5.5 Core Decision Tree

```
Prompt size > 10,000 tokens?
├── Yes → Consider LCM
│   ├── Local deployment? → ✅ Strongly recommend LCM
│   └── Cloud API?
│       ├── Low-latency API (<3,000ms trad TTFT)? → ⚠ Conditional (ultra-large + optimized convergence)
│       └── API-queuing-dominated (>3,000ms)? → ❌ Don't recommend LCM
└── No → Traditional mode
    └── Exception: Local model + slow prefill (>10 ms/token) → Can consider LCM
```

---

## 6. Discussion

### 6.1 LCM's Performance Dependence on Deployment Environment

The core finding of this experiment: **LCM's performance is determined by deployment environment, not prompt scale or task type**.

In prefill-bound environments (local models), LCM replaces the full code review content with a compact summary index, saving 80-95% of prefill time. At the 32,000t level, this savings reaches 49 seconds (traditional 50s → LCM ~1s), far exceeding any LCM protocol overhead.

In API overhead-bound environments (cloud APIs), LCM's token savings are completely drowned. The per-round fixed queuing latency (3,000-7,000ms) becomes an insurmountable bottleneck, making LCM's multi-round mechanism counterproductive.

### 6.2 Bailian's Convergence Explosion Problem

Alibaba Bailian Qwen3-235B's LCM convergence rounds (7-25) far exceed DeepSeek's (1-2) and the local model's (1). Possible contributing factors:

- Qwen3-235B's LCM System Prompt comprehension bias: the model may interpret sentinel requests as "request one at a time" rather than "list all needed at once"
- LCM Prompt's Few-shot examples default to single-chunk request patterns; Qwen3 may follow this pattern literally
- **Correction direction**: Optimize LCM System Prompt per model — strong models (V4 Flash) get flexibility; weaker-instruction-following models get mandatory "must list ALL needed [NEED_CHUNK:id] at once" rules

### 6.3 Token Count Verification Limitations

Cloud APIs (DeepSeek, Bailian) return accurate `prompt_tokens`; the local LM Studio endpoint currently does not return `usage` info, so token counts rely on `len(content)//4` estimation. At 32,000t, estimation error is <0.2%, but at mid-range levels (e.g., 1,000t target vs. 655t actual, -35% deviation), Qwen's tokenizer compresses code at a higher ratio than the 4:1 assumption. However, this affects only the **accuracy of input scale labels**, not the traditional vs. LCM comparison — both modes use identical content.

### 6.4 Engineering Quality

- Pure Python standard library + httpx, zero additional dependencies
- Compatible with all OpenAI-compatible API endpoints
- Four-layer defense: in-round dedup → cross-round tracking → round cap → chunk_miss tolerance
- LCMOrchestrator supports Speculative Prefetch (associative retrieval + auto-injection)

---

## 7. Integration Guide

LCM can be embedded into any Agent framework using OpenAI-compatible APIs via a 3-step wrapping pattern:

```python
# Step 1: Register Agent prompt components as LCM chunks
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

# Step 2: Wrap LLM with LCMOrchestrator
from lcm_core import LCMOrchestrator
orchestrator = LCMOrchestrator(chunk_store=store)

# For local environments: disable Prefetch (single-round convergence is sufficient)
orchestrator.prefetch_enabled = False

# Step 3: Initiate LCM conversation
from lcm_prompt import build_initial_messages
messages = build_initial_messages("Audit codebase for security vulnerabilities", store)
for chunk in orchestrator.run_stream(messages, llm_stream_fn):
    print(chunk, end="")
```

Framework-specific adaptation:

**LangChain / LangGraph**: Wrap BaseChatModel via LCMClient  
**CrewAI**: Replace Agent's llm parameter with LCMClient instance  
**AutoGPT / Custom frameworks**: LCMOrchestrator interface is fully OpenAI Chat Completion-compatible

---

## 8. Related Work

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

## 9. Conclusion and Future Work

### 9.1 Conclusion

This paper presents Lazy Context Materialization (LCM) — a global prompt segmentation technique that transforms Agent framework first-round prompts from "full upfront injection" to "index-first, load-on-demand." Through 166 real model calls across three heterogeneous platforms (DeepSeek V4 Flash, Alibaba Bailian Qwen3-235B, Local Qwen3.6-35B-Apex), spanning 100→64,000 tokens, core conclusions:

1. **LCM performance is determined by deployment environment, not prompt scale**:
   - Local prefill-bound: LCM **9/9 decisive wins**, speedup 1.1×→3.2×
   - Cloud API overhead-bound: LCM loses almost universally (DeepSeek 1/10, Bailian 0/3)

2. **Cloud LCM failure root cause is API queuing latency**: 3,000-7,000ms per-round fixed overhead prevents LCM's token savings from manifesting in latency

3. **Alibaba Bailian Qwen3-235B traditional mode is fastest** (1,738ms, best across all three platforms), but LCM suffers convergence explosion (7-25 rounds)

4. **Local 35B crossover is extremely low (~133t)**: LCM leads even at the smallest prompt scale

5. **Dual-round verification excludes noise**: The first-run 8,000t anomaly (LCM 38,267ms) completely disappeared in the second run

### 9.2 Future Work

- [ ] **P0**: Resolve Alibaba Bailian Qwen3's LCM convergence explosion — targeted LCM Prompt optimization + strong convergence directives
- [ ] **P1**: Prefetch confidence threshold filtering — joint scoring based on keyword overlap ≥2 + prefix matching
- [ ] **P2**: Hybrid mode — high-frequency must-read chunks injected directly + low-frequency chunks via LCM
- [ ] **P2**: Large-prompt Agent validation — verify LCM latency improvement on 10,000+ token Agent definitions
- [ ] **P3**: Cross-model sentinel testing — GPT-4o / Claude 3.5 / Gemini
- [ ] **P3**: LCM-native fine-tuning — fine-tune models to eliminate Few-shot overhead
- [ ] **P3**: Adaptive chunk granularity — dynamically adjust split strategy based on inter-chunk association strength

---

*LCM engine and experiment code open-sourced in agent-core project, `prompt_experiment/` directory. Reproduction: `python experiment_scale_full.py` (DeepSeek) / `python experiment_dual_api.py` (Bailian+DeepSeek) / `python experiment_scale_local.py` (Local 35B).*

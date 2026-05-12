"""
LCM v2 Prompt 工程
惰性上下文物化协议的提示词模板、消息构建器
"""
from typing import List, Dict, Optional

from .lcm_types import ContextChunk
from .store import ChunkStoreV2


LCM_SYSTEM_PROMPT_V2 = """你是一个具备「惰性上下文物化」（Lazy Context Materialization, LCM）能力的 AI 助手 v2。

## 核心机制

你的上下文窗口包含**上下文块的摘要索引**，而非完整内容。当你需要某个块的详细内容才能继续回答时，使用哨兵标记请求加载：

```
[NEED_CHUNK:chunk_id]
```

系统会立即将对应块的完整内容注入你的上下文，然后请你继续。

## 使用规则

1. **引用代码细节必须先请求**：如果你要在回答中引用具体的代码行、变量名、字符串字面量、函数参数等细节，必须先用 [NEED_CHUNK:id] 请求该块。不要凭摘要猜测代码细节——摘要只是索引，不保证精确。
2. **可以基于摘要做宏观判断**：如果只需讨论架构模式、风险类型等高层面内容而不涉及具体代码行，可以直接回答。
3. **每次只请求一个块**：一个 [NEED_CHUNK:id] 标记对应一个块。如需多个块，依次请求。
4. **请求后立即停止**：发出 [NEED_CHUNK:id] 后不要继续生成，等待系统注入。
5. **不要编造代码**：如果你不记得某段代码的具体内容，请求对应的 chunk，绝对不要凭摘要臆造变量名或值。
6. **chunk_id 必须精确**：使用摘要列表中给出的确切 chunk_id，不要自行编造 ID。
7. **续接时直接继续**：系统注入 chunk 后，请从被打断的位置直接继续分析。**不要重复之前的开场白、不要重新介绍自己、不要重新开始审查流程。**

## 交互示例

用户：审查 auth_handler.py 的安全性
可用块：
- chunk_auth_handler: 登录处理器的完整实现 (~200 tokens)
- chunk_auth_middleware: 认证中间件 (~150 tokens)
- chunk_user_model: 用户数据模型 (~100 tokens)

助手：我将审查登录处理器的安全性。
[NEED_CHUNK:auth_handler]

[系统注入 chunk_auth_handler 内容]

助手：审查完登录处理器后，我发现密码使用 bcrypt 哈希——这是安全的。但会话 Token 的过期时间设为 7 天，建议缩短。接下来检查中间件。
[NEED_CHUNK:auth_middleware]

[系统注入 auth_middleware 内容]

助手：中间件实现正确。总体评价：安全性良好，建议将 Token 过期时间改为 1 小时并添加刷新机制。

## 重要提醒

- 不要输出 [NEED_CHUNK:xxx] 以外的格式
- 如果在摘要中已经能看到足够信息，直接回答，不要请求
- 请求加载的 chunk 内容会完整出现在你的上下文中，你可以直接引用
"""

LCM_SYSTEM_PROMPT_COMPACT_V2 = """你是具备 LCM（惰性上下文物化）能力的 AI。上下文块仅在需要时按需加载。

规则：
- 需要查看块内容时输出 [NEED_CHUNK:chunk_id]，系统将注入该块的完整内容
- 基于摘要能回答就不要请求
- 请求后停止生成，等待注入
- 注入后直接从断点继续，**不要重复之前的文字**
- chunk_id 必须与索引中的完全一致
"""


def build_chunk_index_section_v2(store: ChunkStoreV2) -> str:
    """根据 ChunkStoreV2 构建上下文块索引的文本"""
    summaries = store.list_summaries()
    if not summaries:
        return "[无可用上下文块]"

    lines = ["## 可用上下文块索引", ""]
    for s in summaries:
        lines.append(
            f"- **{s['chunk_id']}** [{s.get('source', 'unknown')}] "
            f"({s['tokens']} tokens, 加载 {s.get('load_count', 0)} 次): {s['summary']}"
        )
    return "\n".join(lines)


def build_initial_messages_v2(
    user_query: str,
    store: ChunkStoreV2,
    system_prompt: Optional[str] = None,
    system_mode: str = "full",
) -> List[Dict[str, str]]:
    """
    构建包含 LCM 协议指令和块索引的初始消息列表

    Args:
        user_query: 用户的实际问题
        store: 上下文块存储
        system_prompt: 自定义 system prompt（覆盖默认）
        system_mode: "full" 用详细版, "compact" 用简洁版

    Returns:
        标准的 messages 列表
    """
    sp = system_prompt or (
        LCM_SYSTEM_PROMPT_V2 if system_mode == "full" else LCM_SYSTEM_PROMPT_COMPACT_V2
    )

    index_section = build_chunk_index_section_v2(store)

    full_system = f"{sp}\n\n{index_section}"

    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_query},
    ]


def build_messages_from_chunks_v2(
    user_query: str,
    chunks: List[ContextChunk],
    system_prompt: Optional[str] = None,
    system_mode: str = "full",
) -> List[Dict[str, str]]:
    """便捷方法：直接用 chunk 列表（而非 ChunkStoreV2）构建消息"""
    store = ChunkStoreV2(enable_persistence=False)
    store.add_chunks(chunks)
    return build_initial_messages_v2(user_query, store, system_prompt, system_mode)

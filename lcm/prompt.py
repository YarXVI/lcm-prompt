"""
LCM Prompt Builder
Builds LCM-formatted initial messages with chunk index and sentinel instructions.
"""

from typing import List, Dict

from .core import ChunkStore

LCM_SYSTEM_PROMPT = """You are an AI agent with access to a context chunk system.

## How It Works
- Instead of receiving ALL context upfront, you receive a CHUNK INDEX (summaries only).
- When you need the FULL content of a chunk, output the marker: [NEED_CHUNK:chunk_id]
- The system will interrupt and inject the full content. Then continue your response.

## Rules
1. Request chunks ONLY when you actually need them for your answer.
2. You can request multiple chunks, but prefer requesting all needed ones in ONE batch.
3. After receiving chunk content, continue your answer naturally from where you left off.
4. The chunk_id is the exact identifier shown in the chunk index below.

## Example
User: Analyze security in codebase
You: Looking at the index, I need to examine the authentication module and database layer.
[NEED_CHUNK:auth_module]
[NEED_CHUNK:db_layer]
... (system injects chunks, then you continue)
Based on the code: the auth module has a hardcoded secret key (line 5), and the db layer uses
unsanitized SQL queries (line 12).

## Chunk Index
{chunk_index}
"""


def build_chunk_index(store: ChunkStore) -> str:
    """Build a compact text index of all chunks in the store."""
    lines = []
    for chunk_id, chunk in store.chunks.items():
        token_info = f"~{chunk.tokens}t" if chunk.tokens else ""
        src = f" [{chunk.source}]" if chunk.source else ""
        lines.append(f"- `{chunk_id}`{src} {token_info}: {chunk.summary}")
    return "\n".join(lines)


def build_initial_messages(
    user_query: str,
    store: ChunkStore,
    extra_system: str = "",
) -> List[Dict]:
    """
    Build the initial messages for an LCM conversation.

    Args:
        user_query: The user's task/question.
        store: The ChunkStore containing all context chunks.
        extra_system: Additional system instructions appended to the LCM prompt.

    Returns:
        List of message dicts ready for the LLM API.
    """
    index_text = build_chunk_index(store)
    system = LCM_SYSTEM_PROMPT.format(chunk_index=index_text)
    if extra_system:
        system += f"\n\n{extra_system}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

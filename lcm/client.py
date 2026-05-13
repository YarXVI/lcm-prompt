"""
LCM Client
A thin wrapper that combines LCMOrchestrator with a stream function for convenience.
"""

from typing import List, Dict, Optional, Callable, Iterator

from .core import ChunkStore, LCMOrchestrator
from .prompt import build_initial_messages
from .types import LCMEvent


class LCMClient:
    """Convenience wrapper for the LCM protocol."""

    def __init__(
        self,
        chunk_store: ChunkStore,
        stream_fn: Callable[[List[Dict]], Iterator[str]],
        enable_prefetch: bool = False,
        prefetch_k: int = 3,
    ):
        """
        Args:
            chunk_store: The ChunkStore holding all context chunks.
            stream_fn: A function `(messages) -> Iterator[str]` that streams LLM output.
            enable_prefetch: Enable speculative prefetch (recommended: False for local, True for cloud).
            prefetch_k: Number of related chunks to prefetch.
        """
        self.store = chunk_store
        self.stream_fn = stream_fn
        self.orchestrator = LCMOrchestrator(chunk_store=chunk_store)
        self.orchestrator.prefetch_enabled = enable_prefetch
        self.orchestrator.prefetch_k = prefetch_k

    def chat(
        self,
        user_query: str,
        extra_system: str = "",
        on_event: Optional[Callable[[LCMEvent], None]] = None,
    ) -> str:
        """Non-streaming chat. Returns the full response as a string."""
        messages = build_initial_messages(user_query, self.store, extra_system)
        parts = []
        for chunk in self.orchestrator.run_stream(
            messages, self.stream_fn, on_event=on_event
        ):
            parts.append(chunk)
        return "".join(parts)

    def chat_stream(
        self,
        user_query: str,
        extra_system: str = "",
        on_event: Optional[Callable[[LCMEvent], None]] = None,
    ) -> Iterator[str]:
        """Streaming chat. Yields text chunks as they arrive."""
        messages = build_initial_messages(user_query, self.store, extra_system)
        yield from self.orchestrator.run_stream(
            messages, self.stream_fn, on_event=on_event
        )

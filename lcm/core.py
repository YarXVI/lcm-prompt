"""
LCM Core Engine
ChunkStore, SentinelDetector, and LCMOrchestrator.
"""

import re
import time
from typing import List, Dict, Optional, Callable, Iterator, Set

from .types import (
    ContextChunk,
    LCMSession,
    LCMEvent,
    LoadRequest,
    LCMState,
    ChunkLoadReason,
    SENTINEL_START,
    SENTINEL_PATTERN,
)

MAX_ROUNDS = 20


class ChunkStore:
    """Registry for context chunks with O(1) lookup and associative search."""

    def __init__(self):
        self._chunks: Dict[str, ContextChunk] = {}
        self._id_to_chunks: Dict[str, ContextChunk] = {}

    def add_chunk(self, chunk: ContextChunk):
        self._chunks[chunk.chunk_id] = chunk

    def get(self, chunk_id: str) -> Optional[ContextChunk]:
        return self._chunks.get(chunk_id)

    def find_related(self, chunk_id: str, top_k: int = 3) -> List[str]:
        """Find chunks related to the given chunk by keyword overlap."""
        source = self._chunks.get(chunk_id)
        if not source:
            return []

        src_keywords = set(re.findall(r'\w{4,}', source.summary.lower()))
        scored = []

        for cid, chunk in self._chunks.items():
            if cid == chunk_id:
                continue
            tgt_keywords = set(re.findall(r'\w{4,}', chunk.summary.lower()))
            overlap = len(src_keywords & tgt_keywords)

            prefix_match = 0
            src_parts = chunk_id.split(":")
            tgt_parts = cid.split(":")
            if len(src_parts) > 1 and len(tgt_parts) > 1 and src_parts[0] == tgt_parts[0]:
                prefix_match = 4

            hotness = max(1, chunk.load_count)
            score = overlap * 3 + prefix_match + hotness * 0.5
            if score > 0:
                scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:top_k]]

    def find_related_multi(self, chunk_ids: List[str], top_k: int = 5) -> List[str]:
        """Find chunks related to multiple requested chunk IDs."""
        related = {}
        for cid in chunk_ids:
            for rc in self.find_related(cid, top_k=top_k):
                if rc not in chunk_ids:
                    related[rc] = related.get(rc, 0) + 1
        return sorted(related, key=related.get, reverse=True)[:top_k]

    @property
    def chunks(self) -> Dict[str, ContextChunk]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks


class SentinelDetector:
    """Streaming regex-based sentinel marker detector."""

    def __init__(self):
        self._buffer = ""
        self._pattern = re.compile(SENTINEL_PATTERN)

    def reset(self):
        self._buffer = ""

    def feed(self, text: str) -> List[LoadRequest]:
        self._buffer += text
        results = []
        for m in self._pattern.finditer(self._buffer):
            results.append(LoadRequest(
                chunk_id=m.group(1),
                raw_marker=f"[NEED_CHUNK:{m.group(1)}]",
                position=m.start(),
            ))
        if results:
            self._buffer = self._buffer[results[-1].position + len(results[-1].raw_marker):]
        return results


class LCMOrchestrator:
    """Main orchestrator for the Lazy Context Materialization protocol."""

    def __init__(self, chunk_store: ChunkStore):
        self.store = chunk_store
        self.detector = SentinelDetector()
        self.session: Optional[LCMSession] = None
        self._round = 0
        self._injected_chunks: Set[str] = set()
        self._loaded_per_round: Dict[str, int] = {}

        self.prefetch_enabled = True
        self.prefetch_k = 3
        self.prefetch_strategy = "keyword_overlap"

    def new_session(self, session_id: str = "") -> LCMSession:
        sid = session_id or f"lcm_{int(time.time() * 1000)}"
        self.session = LCMSession(session_id=sid, state=LCMState.IDLE)
        self.detector.reset()
        self._round = 0
        self._injected_chunks.clear()
        self._loaded_per_round.clear()
        return self.session

    def _add_event(self, event: LCMEvent):
        if self.session:
            self.session.load_history.append(event)

    def run_stream(
        self,
        initial_messages: List[Dict],
        stream_fn: Callable[[List[Dict]], Iterator[str]],
        session_id: str = "",
        max_rounds: int = MAX_ROUNDS,
        on_event: Optional[Callable[[LCMEvent], None]] = None,
    ) -> Iterator[str]:
        """
        Run the LCM protocol.

        Args:
            initial_messages: The initial messages (from build_initial_messages).
            stream_fn: A function that takes messages and returns an iterator of text chunks.
            session_id: Optional session identifier.
            max_rounds: Maximum number of API rounds before giving up.
            on_event: Optional callback for LCM lifecycle events.

        Yields:
            Text chunks from model generation.
        """
        self.new_session(session_id)
        current_messages = initial_messages
        all_loaded_this_session: Set[str] = set()

        while self._round < max_rounds:
            self._round += 1
            if self.session:
                self.session.state = LCMState.GENERATING

            sentinel_detected_in_round = False
            loaded_in_round: Set[str] = set()

            try:
                for text_chunk in stream_fn(current_messages):
                    if self.session:
                        self.session.total_tokens_generated += 1
                    yield text_chunk

                    requests = self.detector.feed(text_chunk)
                    if requests:
                        sentinel_detected_in_round = True
                        for req in requests:
                            if not self._filter_request(req, all_loaded_this_session):
                                continue
                            loaded_in_round, all_loaded_this_session = self._inject_chunk(
                                req, current_messages, loaded_in_round,
                                all_loaded_this_session, on_event
                            )
            except Exception as e:
                if self.session:
                    self.session.state = LCMState.ERROR
                    self._add_event(LCMEvent(
                        event_type="error",
                        timestamp=str(time.time()),
                        metadata={"error": str(e)},
                    ))
                    if on_event:
                        on_event(self.session.load_history[-1])
                raise

            if not sentinel_detected_in_round:
                if self.session:
                    self.session.state = LCMState.COMPLETED
                    self._add_event(LCMEvent(
                        event_type="completed",
                        timestamp=str(time.time()),
                        metadata={"total_rounds": self._round},
                    ))
                    if on_event:
                        on_event(self.session.load_history[-1])
                return

        if self.session:
            self.session.state = LCMState.ERROR
            self._add_event(LCMEvent(
                event_type="error",
                timestamp=str(time.time()),
                metadata={"error": f"Exceeded max_rounds ({max_rounds})"},
            ))
            if on_event:
                on_event(self.session.load_history[-1])
        raise RuntimeError(f"LCM exceeded maximum rounds ({max_rounds})")

    def _filter_request(self, req: LoadRequest, all_loaded: Set[str]) -> bool:
        chunk = self.store.get(req.chunk_id)
        if chunk is None:
            self._add_event(LCMEvent(
                event_type="error",
                timestamp=str(time.time()),
                chunk_id=req.chunk_id,
                metadata={"error": f"Chunk not found: {req.chunk_id}"},
            ))
            return False
        return chunk.chunk_id not in all_loaded

    def _inject_chunk(
        self,
        req: LoadRequest,
        messages: List[Dict],
        loaded_in_round: Set[str],
        all_loaded: Set[str],
        on_event,
    ) -> tuple:
        chunk = self.store.get(req.chunk_id)
        if not chunk:
            return loaded_in_round, all_loaded

        chunk.load_count += 1
        chunk.last_loaded_at = str(time.time())

        self._append_chunk_to_messages(messages, req.chunk_id, chunk.content,
                                        ChunkLoadReason.MODEL_REQUEST)
        loaded_in_round.add(chunk.chunk_id)
        all_loaded.add(chunk.chunk_id)

        if self.session:
            self.session.total_chunks_loaded += 1
            self.session.state = LCMState.WAITING_CHUNK

        self._add_event(LCMEvent(
            event_type="chunk_injected",
            chunk_id=req.chunk_id,
            timestamp=str(time.time()),
            position=req.position,
            metadata={"reason": "model_request", "round": self._round},
        ))
        if on_event:
            on_event(self.session.load_history[-1])

        if self.prefetch_enabled and self.store.find_related:
            related = self.store.find_related(req.chunk_id, top_k=self.prefetch_k)
            for rc in related:
                if rc not in all_loaded and rc not in loaded_in_round:
                    rc_chunk = self.store.get(rc)
                    if rc_chunk:
                        rc_chunk.load_count += 1
                        self._append_chunk_to_messages(messages, rc, rc_chunk.content,
                                                        ChunkLoadReason.SPECULATIVE_PREFETCH)
                        loaded_in_round.add(rc)
                        all_loaded.add(rc)

                        if self.session:
                            self.session.total_chunks_loaded += 1

        return loaded_in_round, all_loaded

    def _append_chunk_to_messages(self, messages: List[Dict], chunk_id: str,
                                   content: str, reason: ChunkLoadReason):
        last_msg = messages[-1]
        if last_msg["role"] == "assistant":
            last_msg["content"] += (
                f"\n\n[SYSTEM: Loaded {chunk_id} ({reason.value})]\n\n{content}"
            )
        else:
            messages.append({
                "role": "assistant",
                "content": f"[SYSTEM: Loaded {chunk_id} ({reason.value})]\n\n{content}",
            })

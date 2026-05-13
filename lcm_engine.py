import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Generator, Callable
from enum import Enum

from .chunk_store import ChunkStore, Chunk
from .sentinel_detector import SentinelDetector, LoadRequest
from .ir_models import GrainLevel
from .encoder_base import ContentDecoder
from .encoding_registry import EncodingRegistry
from .encoded_chunk_store import EncodedChunkStore
from .adaptive_injector import AdaptiveInjector
from .execution_profile import (
    ExecutionProfile, PROFILE_DEFAULTS, PROFILE_PROMPT_CACHING, PROFILE_DYNAMIC_RENDERING,
)
from .cache_builder import CacheAwarePrefixBuilder
from .content_encoding import (
    ContentEncodingRegistry, EncodingType, EncodingContext,
)
from .urr_reporter import URRReporter
from .label_system import LabelStore, ChunkLabel, Anchor
from .golden_corpus import GoldenCorpusCollector
from .dynamic_renderer import DynamicRenderer
from .semantic_slicer import SemanticSlicer
from .ab_test_router import ABTestRouter, ABTestConfig, ABTestResult


class LCMState(Enum):
    IDLE = "idle"
    GENERATING = "generating"
    WAITING_CHUNK = "waiting_chunk"
    RESUMING = "resuming"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class LCMEvent:
    event_type: str
    chunk_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LCMSession:
    session_id: str
    state: LCMState = LCMState.IDLE
    total_chunks_loaded: int = 0
    tokens_injected: int = 0
    tokens_saved: int = 0
    load_history: List[LCMEvent] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000


@dataclass
class LCMConfig:
    max_rounds: int = 20
    max_prefetch: int = 3
    enable_prefetch: bool = True
    retry_attempts: int = 2
    available_tokens_per_inject: int = 4000
    encoding_type: EncodingType = EncodingType.IDENTITY
    encoding_registry: Optional[ContentEncodingRegistry] = None
    profile: ExecutionProfile = ExecutionProfile.LOCAL_CONSTRAINED
    cooldown_rounds: int = 3


class LCMEngine:

    def __init__(
        self,
        chunk_store: Optional[ChunkStore] = None,
        config: Optional[LCMConfig] = None,
        encoding_registry: Optional[EncodingRegistry] = None,
    ):
        self.config = config or LCMConfig()
        self.detector = SentinelDetector()
        self.session: Optional[LCMSession] = None
        self._round = 0
        self._event_handlers: List[Callable[[LCMEvent], None]] = []

        self._v1_encoding_registry = self.config.encoding_registry or ContentEncodingRegistry()
        self._v1_encoding_context = EncodingContext()

        self._encoding_registry = encoding_registry or EncodingRegistry()
        self._encoded_store = EncodedChunkStore(
            chunk_store=chunk_store,
            encoding_registry=self._encoding_registry,
        )
        self._injector = AdaptiveInjector(
            encoded_store=self._encoded_store,
            decoder=ContentDecoder(),
            profile=self.config.profile,
            cooldown_rounds=self.config.cooldown_rounds,
        )
        self._cache_builder = CacheAwarePrefixBuilder()
        self._urr_reporter = URRReporter()
        self._label_store = LabelStore()
        self._golden_collector = GoldenCorpusCollector()
        self._dynamic_renderer = DynamicRenderer(self._label_store)
        self._semantic_slicer = SemanticSlicer()
        self._ab_router = ABTestRouter()

    @property
    def store(self) -> ChunkStore:
        return self._encoded_store.raw_store

    @store.setter
    def store(self, value: ChunkStore):
        pass

    @property
    def injector(self) -> AdaptiveInjector:
        return self._injector

    def _get_encoder(self):
        return self._v1_encoding_registry.get(self.config.encoding_type)

    def register_encoding(self, encoding):
        self._v1_encoding_registry.register(encoding)
        return self

    def set_encoding(self, encoding_type: EncodingType):
        self.config.encoding_type = encoding_type
        return self

    def register_encoder(self, encoder) -> "LCMEngine":
        self._encoding_registry.register(encoder)
        return self

    def set_profile(self, profile: ExecutionProfile) -> "LCMEngine":
        self.config.profile = profile
        self._injector.set_profile(profile)
        return self

    def warmup_encodings(self) -> None:
        self._encoded_store.warmup()

    def on_event(self, handler: Callable[[LCMEvent], None]) -> None:
        self._event_handlers.append(handler)

    def _emit(self, event_type: str, chunk_id: str = "", **metadata) -> None:
        event = LCMEvent(event_type=event_type, chunk_id=chunk_id, metadata=metadata)
        if self.session:
            self.session.load_history.append(event)
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception:
                pass

    def new_session(self, session_id: str = "") -> LCMSession:
        sid = session_id or f"lcm_{int(time.time() * 1000)}"
        self.session = LCMSession(session_id=sid, state=LCMState.IDLE)
        self.detector.reset()
        self._injector.reset_session()
        self._injector.set_session(sid)
        self._round = 0
        return self.session

    def build_index_section(self) -> str:
        summaries = self.store.list_summaries()
        if not summaries:
            return "[no chunks available]"

        lines = ["## Available Chunks", ""]
        for s in summaries:
            lines.append(
                f"- **{s['chunk_id']}** [{s.get('source', 'unknown')}] "
                f"({s['tokens']} tokens, loaded {s.get('load_count', 0)}x): {s['summary']}"
            )
        return "\n".join(lines)

    def build_system_prompt(self, base_prompt: str, user_query: str = "") -> str:
        self._v1_encoding_context.user_query = user_query
        self._v1_encoding_context.session_id = self.session.session_id if self.session else ""

        encoder = self._get_encoder()
        encoded_base = encoder.encode_system_prompt(base_prompt, self._v1_encoding_context)

        profile_hint = ""
        if self.config.profile == ExecutionProfile.CLOUD_TOKEN_BILLED:
            profile_hint = "\n- Cloud token-billed mode: default injection at summary grain. Request detail explicitly if needed."
        elif self.config.profile == ExecutionProfile.CLOUD_CALL_BILLED:
            profile_hint = "\n- Cloud call-billed mode: default injection at detail grain to minimize interaction rounds."

        lcm_instructions = f"""You are an AI assistant with Lazy Context Materialization (LCM) capability.

## Core Mechanism

Your context window contains **chunk summary indexes**, not full content. When you need detailed content from a specific chunk to continue answering, use sentinel markers to request loading:

```
[NEED_CHUNK:chunk_id]
```

The system will automatically select the most appropriate grain based on available window space.

## Grain Protocol

Loaded chunks may appear at different granularity levels:
- **keywords**: key terms only (~30 tokens)
- **summary**: concise overview (~80 tokens)
- **detail**: signatures/pseudocode/structured points (~300 tokens)
- **full**: complete original content (~2000+ tokens)

To request finer grain:
- `[NEED_CHUNK_DETAIL:chunk_id]` request detail level
- `[NEED_CHUNK_FULL:chunk_id]`   request full content

If current information is sufficient:
- `[GRAIN_SUFFICIENT:chunk_id]`  actively downgrade, release token space
{profile_hint}

## Usage Rules

1. **Must request before referencing code details**: If you need to reference specific code lines, variable names, string literals, function parameters, etc., you must first request the chunk with [NEED_CHUNK:id].
2. **Can make macro judgments based on summary**: If you only need to discuss architectural patterns, risk types, etc. at a high level without specific code lines, you can answer directly.
3. **Request one chunk at a time**: One marker per chunk. For multiple chunks, request sequentially.
4. **Stop immediately after requesting**: After emitting a marker, do not continue generating. Wait for system injection.
5. **Do not fabricate code**: If you don't remember specific content of a code segment, request the corresponding chunk. Never guess variable names or values based on summaries.
6. **chunk_id must be exact**: Use the exact chunk_id given in the summary list.
7. **Continue from interruption point**: After the system injects a chunk, continue analysis from where you were interrupted. Do not repeat previous openings.

## Important Reminders

- If you can see enough information in the summary, answer directly without requesting
- Prefer default requests; only upgrade grain when you explicitly need more detail
- Use [GRAIN_SUFFICIENT:id] to downgrade when information is sufficient, helping save tokens
"""
        index_section = self.build_index_section()
        result = f"{encoded_base}\n\n{lcm_instructions}\n\n{index_section}"

        if PROFILE_PROMPT_CACHING.get(self.config.profile, False):
            chunk_summaries = self._cache_builder.build_chunk_index(self.store.list_summaries())
            return self._cache_builder.build_cached_messages(
                result, [], chunk_summaries,
            )

        return encoder.encode_system_prompt(result, self._v1_encoding_context)

    def process_response(self, response_text: str) -> tuple[str, List[LoadRequest]]:
        encoder = self._get_encoder()
        self._v1_encoding_context.current_round = self._round
        encoded_response = encoder.encode_response(response_text, self._v1_encoding_context)

        self.detector.reset()
        requests = self.detector.feed(encoded_response)
        clean_text = self.detector.get_clean_buffer()

        self._injector.process_sentinels(encoded_response)

        if requests:
            self._emit("sentinel_detected", requests[0].chunk_id, count=len(requests))

        return clean_text, requests

    def load_chunk(self, chunk_id: str) -> Optional[Chunk]:
        chunk = self.store.get(chunk_id)
        if chunk:
            self._emit("chunk_loaded", chunk_id, tokens=chunk.tokens)
            if self.session:
                self.session.total_chunks_loaded += 1
                self.session.state = LCMState.RESUMING
        else:
            self._emit("chunk_not_found", chunk_id)
        return chunk

    def inject_chunk(
        self,
        messages: List[Dict[str, str]],
        chunk: Chunk,
        available_tokens: int = 0,
        min_level: GrainLevel = GrainLevel.KEYWORDS,
    ) -> List[Dict[str, str]]:
        budget = available_tokens or self.config.available_tokens_per_inject
        full_tokens = chunk.tokens

        if self.store.get(chunk.chunk_id) is None:
            self.store.add(chunk)

        is_upgrade = min_level != GrainLevel.KEYWORDS
        t0 = time.perf_counter()

        if PROFILE_DYNAMIC_RENDERING.get(self.config.profile, False):
            encoded = self._encoded_store.get_encoded(chunk.chunk_id)
            if encoded and self._dynamic_renderer.can_render(chunk.chunk_id):
                slice_result = self._dynamic_renderer.render(
                    chunk.chunk_id, chunk.content,
                    query_intent="", available_tokens=budget,
                )
                if slice_result:
                    messages.append({
                        "role": "system",
                        "content": (
                            f"[chunk {chunk.chunk_id} | grain:dynamic:{slice_result.anchor_name} | "
                            f"{slice_result.tokens}tok]\n{slice_result.content}"
                        ),
                    })
                    if self.session:
                        self.session.tokens_injected += slice_result.tokens
                        self.session.tokens_saved += max(0, full_tokens - slice_result.tokens)
                    self._urr_reporter.record_injection(
                        chunk.chunk_id, f"dynamic:{slice_result.anchor_name}",
                        (time.perf_counter() - t0) * 1000, False,
                    )
                    return messages

        messages, selected_level = self._injector.inject(
            messages, chunk.chunk_id, budget, min_level=min_level,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        if self.session:
            encoded = self._encoded_store.get_encoded(chunk.chunk_id)
            if encoded:
                actual_grain = encoded.ir.grains.get(selected_level)
                actual_tokens = actual_grain.tokens if actual_grain else full_tokens
                self.session.tokens_injected += actual_tokens
                self.session.tokens_saved += max(0, full_tokens - actual_tokens)

                self._urr_reporter.record_injection(
                    chunk.chunk_id, selected_level.value, latency_ms, is_upgrade,
                )
                self._golden_collector.consider(
                    chunk.chunk_id, chunk.content, encoded.ir,
                    self._urr_reporter.get_chunk_stats(chunk.chunk_id).total_injections if self._urr_reporter.get_chunk_stats(chunk.chunk_id) else 1,
                    self._urr_reporter.get_chunk_stats(chunk.chunk_id).upgrade_requests if self._urr_reporter.get_chunk_stats(chunk.chunk_id) else (1 if is_upgrade else 0),
                )

            self._emit(
                "chunk_injected", chunk.chunk_id,
                grain=selected_level.value,
                full_tokens=full_tokens,
            )
        else:
            self._emit("chunk_injected", chunk.chunk_id, grain=selected_level.value)

        return messages

    def inject_chunk_full(
        self,
        messages: List[Dict[str, str]],
        chunk_id: str,
    ) -> List[Dict[str, str]]:
        messages = self._injector.inject_full(messages, chunk_id)
        self._emit("chunk_injected", chunk_id, grain="full")
        return messages

    def run_sync(
        self,
        messages: List[Dict[str, str]],
        stream_fn: Callable[[List[Dict[str, str]]], str],
        session_id: str = "",
    ) -> str:
        self.new_session(session_id)
        current_messages = list(messages)
        self.session.state = LCMState.GENERATING
        self._emit("generation_started")

        while self._round < self.config.max_rounds:
            self._round += 1

            response = stream_fn(current_messages)
            clean_text, requests = self.process_response(response)

            self._injector.tick_cooldown()

            if not requests:
                self.session.state = LCMState.COMPLETED
                self.session.end_time = datetime.now()
                self._emit("completed", rounds=self._round)
                return clean_text

            req = requests[0]
            chunk = self.load_chunk(req.chunk_id)
            if chunk:
                current_messages = self.inject_chunk(
                    current_messages, chunk, min_level=req.min_level,
                )
            else:
                self.session.state = LCMState.ERROR
                self.session.end_time = datetime.now()
                self._emit("error", reason="chunk_not_found")
                return clean_text + f"\n[error: chunk '{req.chunk_id}' not found]"

        self.session.state = LCMState.ERROR
        self.session.end_time = datetime.now()
        self._emit("error", reason="max_rounds_exceeded")
        return clean_text + "\n[error: max rounds exceeded]"

    def run_stream(
        self,
        messages: List[Dict[str, str]],
        stream_fn: Callable[[List[Dict[str, str]]], Generator[str, None, None]],
        session_id: str = "",
    ) -> Generator[str, None, None]:
        self.new_session(session_id)
        current_messages = list(messages)
        self.session.state = LCMState.GENERATING
        self._emit("generation_started")

        while self._round < self.config.max_rounds:
            self._round += 1
            full_response = ""

            for chunk_text in stream_fn(current_messages):
                full_response += chunk_text
                requests = self.detector.feed(chunk_text)

                if requests:
                    break
                yield chunk_text

            clean_text = self.detector.get_clean_buffer()
            requests = self.detector.get_requests()

            self._injector.process_sentinels(full_response)
            self._injector.tick_cooldown()

            if not requests:
                self.session.state = LCMState.COMPLETED
                self.session.end_time = datetime.now()
                self._emit("completed", rounds=self._round)
                yield clean_text
                return

            req = requests[0]
            chunk = self.load_chunk(req.chunk_id)
            if chunk:
                current_messages = self.inject_chunk(
                    current_messages, chunk, min_level=req.min_level,
                )
            else:
                self.session.state = LCMState.ERROR
                self.session.end_time = datetime.now()
                self._emit("error", reason="chunk_not_found")
                yield clean_text + f"\n[error: chunk '{req.chunk_id}' not found]"
                return

        self.session.state = LCMState.ERROR
        self.session.end_time = datetime.now()
        self._emit("error", reason="max_rounds_exceeded")
        yield clean_text + "\n[error: max rounds exceeded]"

    def get_stats(self) -> Dict[str, Any]:
        store_stats = self.store.get_stats()
        encoder = self._get_encoder()
        encoding_stats = encoder.get_stats()
        encoded_store_stats = self._encoded_store.get_stats()
        injector_stats = self._injector.get_stats()
        return {
            **store_stats,
            "session_id": self.session.session_id if self.session else None,
            "state": self.session.state.value if self.session else None,
            "rounds": self._round,
            "chunks_loaded": self.session.total_chunks_loaded if self.session else 0,
            "tokens_injected": self.session.tokens_injected if self.session else 0,
            "tokens_saved": self.session.tokens_saved if self.session else 0,
            "duration_ms": self.session.duration_ms if self.session else 0,
            "profile": self.config.profile.value,
            "encoding": {
                "type": self.config.encoding_type.value,
                "name": encoder.name,
                **encoding_stats,
            },
            "available_encodings": self._v1_encoding_registry.list_encodings(),
            "ir_encoding": encoded_store_stats,
            "injector": injector_stats,
            "urr_report": self._urr_reporter.get_report(),
            "label_coverage": self._label_store.get_coverage_stats(),
            "golden_corpus": self._golden_collector.get_stats(),
            "dynamic_rendering_enabled": PROFILE_DYNAMIC_RENDERING.get(self.config.profile, False),
        }

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._injector.get_audit_log(limit=limit)

    def geturr_stats(self) -> Dict[str, float]:
        return self._injector.geturr_stats()

    @property
    def urr_reporter(self) -> URRReporter:
        return self._urr_reporter

    @property
    def label_store(self) -> LabelStore:
        return self._label_store

    @property
    def golden_collector(self) -> GoldenCorpusCollector:
        return self._golden_collector

    @property
    def dynamic_renderer(self) -> DynamicRenderer:
        return self._dynamic_renderer

    @property
    def semantic_slicer(self) -> SemanticSlicer:
        return self._semantic_slicer

    @property
    def ab_router(self) -> ABTestRouter:
        return self._ab_router

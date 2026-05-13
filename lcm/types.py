"""
LCM Core Data Models
Type system for the Lazy Context Materialization protocol.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum


class ChunkLoadReason(str, Enum):
    MODEL_REQUEST = "model_request"
    PREEMPTIVE = "preemptive"
    POST_HOC = "post_hoc"
    SPECULATIVE_PREFETCH = "speculative_prefetch"


class LCMState(str, Enum):
    IDLE = "idle"
    GENERATING = "generating"
    WAITING_CHUNK = "waiting_chunk"
    RESUMING = "resuming"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ContextChunk:
    chunk_id: str
    content: str
    summary: str = ""
    tokens: int = 0
    load_count: int = 0
    last_loaded_at: str = ""
    priority: int = 0
    cache_hit: bool = False
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = len(self.content) // 4


@dataclass
class LCMEvent:
    event_type: str
    timestamp: str = ""
    chunk_id: str = ""
    position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadRequest:
    chunk_id: str
    raw_marker: str
    position: int
    query_hint: str = ""


@dataclass
class LCMSession:
    session_id: str
    state: LCMState = LCMState.IDLE
    total_tokens_generated: int = 0
    total_chunks_loaded: int = 0
    load_history: List[LCMEvent] = field(default_factory=list)
    latency_stats: Dict[str, float] = field(default_factory=dict)


SENTINEL_START = "[NEED_CHUNK:"
SENTINEL_END = "]"
SENTINEL_PATTERN = r"\[NEED_CHUNK:([A-Za-z0-9_\-]+)\]"

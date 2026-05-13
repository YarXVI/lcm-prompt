"""
Lazy Context Materialization (LCM) - Global Prompt Segmentation for Agent Frameworks.

Public API:
    - LCMOrchestrator: Main orchestrator for LCM protocol
    - ChunkStore: Registry for context chunks
    - ContextChunk, LCMSession, LCMState, LCMEvent, ChunkLoadReason: Data models
    - build_initial_messages: Build LCM-formatted initial messages
    - load_env: Load configuration from .env file
"""

from .types import (
    ContextChunk,
    LCMSession,
    LCMEvent,
    LoadRequest,
    LCMState,
    ChunkLoadReason,
    SENTINEL_START,
    SENTINEL_END,
    SENTINEL_PATTERN,
)
from .core import (
    ChunkStore,
    SentinelDetector,
    LCMOrchestrator,
)
from .prompt import (
    build_initial_messages,
)
from .client import LCMClient
from .env_loader import load_env

__version__ = "1.0.0"

__all__ = [
    "LCMOrchestrator",
    "ChunkStore",
    "SentinelDetector",
    "ContextChunk",
    "LCMSession",
    "LCMEvent",
    "LoadRequest",
    "LCMState",
    "ChunkLoadReason",
    "SENTINEL_START",
    "SENTINEL_END",
    "SENTINEL_PATTERN",
    "build_initial_messages",
    "LCMClient",
    "load_env",
]

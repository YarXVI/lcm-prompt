"""
LCM - Lazy Context Materialization

Independent multi-granularity context supply system.
Extracted from IRIS LCM2 as a standalone project.

Architecture:
  Raw content -> [Encoding Layer] -> Multi-Granularity IR -> [Storage & Routing] -> [Decoding Layer] -> LLM injection

Core features:
  - Fixed 4-level grain hierarchy (KEYWORDS/SUMMARY/DETAIL/FULL)
  - Pluggable encoder architecture
  - Lazy encoding with async warmup
  - ExecutionProfile-based billing-economics routing
  - Soft upgrade state machine with cooldown
  - URR monitoring and golden corpus collection
  - Dynamic template rendering
  - Semantic slicing for fallback compression
  - A/B test routing for controlled experiments
"""

from .ir_models import GrainLevel, Grain, MultiGranularityIR, IR_VERSION
from .encoder_base import ContentEncoder, ContentDecoder, EncodingContext
from .encoding_registry import EncodingRegistry, IdentityEncoder
from .chunk_store import Chunk, ChunkStore
from .encoded_chunk_store import EncodedChunkStore, EncodedChunk
from .sentinel_detector import SentinelDetector, LoadRequest
from .execution_profile import (
    ExecutionProfile,
    PROFILE_DEFAULTS,
    PROFILE_UPGRADE_STRATEGY,
    PROFILE_PROMPT_CACHING,
    PROFILE_DYNAMIC_RENDERING,
)
from .adaptive_injector import (
    AdaptiveInjector,
    UpgradeRequest,
    DowngradeRequest,
    InjectionAuditEntry,
)
from .cache_builder import CacheAwarePrefixBuilder
from .content_encoding import (
    ContentEncoding,
    EncodingType,
    EncodingContext as V1EncodingContext,
    ContentEncodingRegistry,
    IdentityEncoding,
    get_default_registry,
    register_encoding,
    get_encoding,
)
from .lcm_engine import (
    LCMEngine,
    LCMConfig,
    LCMSession,
    LCMState,
    LCMEvent,
)
from .urr_reporter import URRReporter, ChunkURRStats
from .label_system import LabelStore, ChunkLabel, Anchor
from .golden_corpus import GoldenCorpusCollector, GoldenSample
from .dynamic_renderer import DynamicRenderer, RenderedSlice
from .semantic_slicer import SemanticSlicer
from .ab_test_router import ABTestRouter, ABTestConfig, ABTestResult

from .encoders import (
    CodeIntentEncoder,
    ChineseThinkEncoder,
    EnglishLogicEncoder,
    ASTCodeEncoder,
)


def create_engine(
    profile: ExecutionProfile = ExecutionProfile.LOCAL_CONSTRAINED,
    register_default_encoders: bool = True,
    **kwargs,
) -> LCMEngine:
    registry = EncodingRegistry()
    if register_default_encoders:
        registry.register(CodeIntentEncoder())
        registry.register(ChineseThinkEncoder())
        registry.register(EnglishLogicEncoder())
        try:
            registry.register(ASTCodeEncoder())
        except Exception:
            pass

    config = LCMConfig(profile=profile, **kwargs)
    engine = LCMEngine(config=config, encoding_registry=registry)
    return engine


__all__ = [
    "GrainLevel",
    "Grain",
    "MultiGranularityIR",
    "IR_VERSION",
    "ContentEncoder",
    "ContentDecoder",
    "EncodingContext",
    "EncodingRegistry",
    "IdentityEncoder",
    "Chunk",
    "ChunkStore",
    "EncodedChunkStore",
    "EncodedChunk",
    "SentinelDetector",
    "LoadRequest",
    "ExecutionProfile",
    "PROFILE_DEFAULTS",
    "PROFILE_UPGRADE_STRATEGY",
    "PROFILE_PROMPT_CACHING",
    "PROFILE_DYNAMIC_RENDERING",
    "AdaptiveInjector",
    "UpgradeRequest",
    "DowngradeRequest",
    "InjectionAuditEntry",
    "CacheAwarePrefixBuilder",
    "ContentEncoding",
    "EncodingType",
    "V1EncodingContext",
    "ContentEncodingRegistry",
    "IdentityEncoding",
    "get_default_registry",
    "register_encoding",
    "get_encoding",
    "LCMEngine",
    "LCMConfig",
    "LCMSession",
    "LCMState",
    "LCMEvent",
    "URRReporter",
    "ChunkURRStats",
    "LabelStore",
    "ChunkLabel",
    "Anchor",
    "GoldenCorpusCollector",
    "GoldenSample",
    "DynamicRenderer",
    "RenderedSlice",
    "SemanticSlicer",
    "ABTestRouter",
    "ABTestConfig",
    "ABTestResult",
    "CodeIntentEncoder",
    "ChineseThinkEncoder",
    "EnglishLogicEncoder",
    "ASTCodeEncoder",
    "create_engine",
]

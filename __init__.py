"""
LCM v2 - 惰性上下文物化协议 v2

核心改进：
- 线程安全：所有操作带 RLock 保护
- 持久化：JSONL + 索引文件，进程重启不丢失
- 精确 Token 计数：支持 tiktoken 和中文估算
- LRU 缓存：热点 chunk 内存缓存
- 语义搜索：支持向量相似度（可选 embedding）
- 批量操作：批量加载、预取
- 全面指标：延迟、命中率、缓存统计
- Token 预算管理：防止上下文溢出
- Chunk 依赖图：DAG 拓扑排序
- 混合模式：高频直接注入 + 低频 LCM
- API 路由自动识别：云端/本地自动切换
- KV Cache 联动：利用云厂商缓存机制
- 自适应粒度：粗/细粒度动态调整
- 质量评估：系统性答案质量评估
- 多模型基准：跨模型延迟+质量对比
- 微调协议：LCM 原生 Fine-tuning 支持
- 多 Agent 协作：共享组件索引去重
"""

from .lcm_types import (
    ContextChunk,
    LCMEvent,
    LoadRequest,
    LCMSession,
    LCMState,
    ChunkLoadReason,
    SentinelPattern,
    LCMMetrics,
)
from .store import ChunkStoreV2
from .detector import SentinelDetectorV2
from .orchestrator import LCMOrchestratorV2
from .client import LCMClientV2
from .prompt import build_initial_messages_v2
from .token_budget import TokenBudget
from .chunk_graph import ChunkGraph
from .hybrid_mode import HybridChunkManager, HybridConfig, build_hybrid_messages
from .provider_router import (
    ProviderRouter, AdaptiveLCMClient,
    ProviderConfig, ProviderType, RoutingStrategy,
)
from .kv_cache import KVCacheManager, CachedLCMOrchestrator
from .adaptive_chunking import AdaptiveChunking, AdaptiveChunkStore, ChunkGroup
from .quality_eval import QualityEvaluator, BenchmarkRunner, QualityMetrics
from .multi_model_benchmark import MultiModelBenchmark, ModelConfig, BenchmarkResult
from .fine_tuning import LCMFineTuner, LCMTuningDataset, FineTuningConfig
from .multi_agent import MultiAgentLCM, SharedIndexManager, AgentSession
from .logger import LCMLogger, get_logger, LCMErrorCode
from .sqlite_store import SQLiteChunkStore
from .vector_index import VectorIndex
from .async_client import AsyncLCMClient, AsyncLLMWrapper
from .compression import ChunkCompressor, CompressedChunkStore
from .multimodal import MediaChunk, MediaChunkLoader, MediaType
from .protocol import ProtocolVersion, ProtocolNegotiator, ProtocolAdapter
from .distributed import DistributedChunkStore, DistributedIndexManager, NodeInfo

__all__ = [
    # 核心类型
    "ContextChunk",
    "LCMEvent",
    "LoadRequest",
    "LCMSession",
    "LCMState",
    "ChunkLoadReason",
    "SentinelPattern",
    "LCMMetrics",
    # 核心组件
    "ChunkStoreV2",
    "SentinelDetectorV2",
    "LCMOrchestratorV2",
    "LCMClientV2",
    "build_initial_messages_v2",
    # Token 预算
    "TokenBudget",
    # Chunk 依赖图
    "ChunkGraph",
    # 混合模式
    "HybridChunkManager",
    "HybridConfig",
    "build_hybrid_messages",
    # Provider 路由
    "ProviderRouter",
    "AdaptiveLCMClient",
    "ProviderConfig",
    "ProviderType",
    "RoutingStrategy",
    # KV Cache
    "KVCacheManager",
    "CachedLCMOrchestrator",
    # 自适应粒度
    "AdaptiveChunking",
    "AdaptiveChunkStore",
    "ChunkGroup",
    # 质量评估
    "QualityEvaluator",
    "BenchmarkRunner",
    "QualityMetrics",
    # 多模型基准
    "MultiModelBenchmark",
    "ModelConfig",
    "BenchmarkResult",
    # 微调
    "LCMFineTuner",
    "LCMTuningDataset",
    "FineTuningConfig",
    # 多 Agent
    "MultiAgentLCM",
    "SharedIndexManager",
    "AgentSession",
    # 日志
    "LCMLogger",
    "get_logger",
    "LCMErrorCode",
    # SQLite
    "SQLiteChunkStore",
    # 向量索引
    "VectorIndex",
    # 异步
    "AsyncLCMClient",
    "AsyncLLMWrapper",
    # 压缩
    "ChunkCompressor",
    "CompressedChunkStore",
    # 多模态
    "MediaChunk",
    "MediaChunkLoader",
    "MediaType",
    # 协议
    "ProtocolVersion",
    "ProtocolNegotiator",
    "ProtocolAdapter",
    # 分布式
    "DistributedChunkStore",
    "DistributedIndexManager",
    "NodeInfo",
]

"""
LCM v2 核心数据模型
惰性上下文物化协议的类型系统 v2
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime
import json


class ChunkLoadReason(str, Enum):
    """触发加载的原因"""
    MODEL_REQUEST = "model_request"
    PREEMPTIVE = "preemptive"
    POST_HOC = "post_hoc"
    SPECULATIVE_PREFETCH = "speculative_prefetch"
    CACHE_WARMUP = "cache_warmup"


class LCMState(str, Enum):
    """协议状态机"""
    IDLE = "idle"
    GENERATING = "generating"
    WAITING_CHUNK = "waiting_chunk"
    RESUMING = "resuming"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ContextChunk:
    """一个上下文块 v2"""
    chunk_id: str
    content: str
    summary: str = ""
    tokens: int = 0
    load_count: int = 0
    last_loaded_at: Optional[datetime] = None
    priority: int = 0
    cache_hit: bool = False
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def __post_init__(self):
        if not self.tokens:
            self.tokens = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """精确 token 计数（优先使用 tiktoken，降级到字符估算）"""
        if not text:
            return 0
        
        # 尝试使用 tiktoken 精确计数
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass
        except Exception:
            pass
        
        # 降级：字符估算（中文按字，英文按词）
        import re
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_matches = re.findall(r'[a-zA-Z]+', text)
        en_words = len(en_matches)
        en_chars = sum(len(w) for w in en_matches)
        others = max(0, len(text) - cn_chars - en_chars)
        return int(cn_chars * 1.5 + en_words * 1.3 + others * 0.5)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（处理 datetime 和不可序列化对象）"""
        d = asdict(self)
        for key in ['last_loaded_at', 'created_at']:
            if d.get(key):
                d[key] = d[key].isoformat() if isinstance(d[key], datetime) else d[key]
        # 处理 metadata 中可能的不可序列化对象
        if d.get('metadata'):
            d['metadata'] = self._serialize_metadata(d['metadata'])
        return d

    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """安全序列化 metadata，将不可序列化对象转为字符串（递归处理嵌套）"""
        result = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            elif isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, list):
                result[k] = ContextChunk._serialize_list(v)
            elif isinstance(v, dict):
                result[k] = ContextChunk._serialize_metadata(v)
            else:
                result[k] = str(v)
        return result

    @staticmethod
    def _serialize_list(items: List[Any]) -> List[Any]:
        """递归序列化列表中的元素"""
        result = []
        for item in items:
            if isinstance(item, (str, int, float, bool, type(None))):
                result.append(item)
            elif isinstance(item, datetime):
                result.append(item.isoformat())
            elif isinstance(item, list):
                result.append(ContextChunk._serialize_list(item))
            elif isinstance(item, dict):
                result.append(ContextChunk._serialize_metadata(item))
            else:
                result.append(str(item))
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextChunk":
        """从字典反序列化"""
        for key in ['last_loaded_at', 'created_at']:
            if data.get(key):
                if isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LCMEvent:
    """LCM 事件（用于日志/回调）v2"""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    chunk_id: str = ""
    position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class LoadRequest:
    """模型发出的加载请求 v2"""
    chunk_id: str
    raw_marker: str
    position: int
    query_hint: str = ""
    confidence: float = 1.0


@dataclass
class LCMSession:
    """单次 LCM 会话 v2"""
    session_id: str
    state: LCMState = LCMState.IDLE
    total_tokens_generated: int = 0
    total_chunks_loaded: int = 0
    load_history: List[LCMEvent] = field(default_factory=list)
    latency_stats: Dict[str, float] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        """会话持续时间（毫秒）"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['start_time'] = self.start_time.isoformat()
        if self.end_time:
            d['end_time'] = self.end_time.isoformat()
        d['duration_ms'] = self.duration_ms
        d['load_history'] = [e.to_dict() for e in self.load_history]
        return d


class SentinelPattern:
    """哨兵模式集合"""
    NEED_CHUNK = r"\[NEED_CHUNK:([A-Za-z0-9_\-]+)\]"
    LOAD_CHUNK = r"\[LOAD_CHUNK:([A-Za-z0-9_\-]+)\]"
    FETCH = r"\[FETCH:([A-Za-z0-9_\-]+)\]"

    @classmethod
    def get_all_patterns(cls) -> List[str]:
        return [cls.NEED_CHUNK, cls.LOAD_CHUNK, cls.FETCH]


@dataclass
class LCMMetrics:
    """LCM 指标收集器"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_load_latency_ms: float = 0.0
    total_search_latency_ms: float = 0.0
    chunk_hit_ids: List[str] = field(default_factory=list)
    chunk_miss_ids: List[str] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_load_latency_ms(self) -> float:
        total_ops = self.cache_hits + self.cache_misses
        return self.total_load_latency_ms / total_ops if total_ops > 0 else 0.0

    def record_cache_hit(self, chunk_id: str):
        self.total_requests += 1
        self.cache_hits += 1
        self.chunk_hit_ids.append(chunk_id)

    def record_cache_miss(self, chunk_id: str):
        self.total_requests += 1
        self.cache_misses += 1
        self.chunk_miss_ids.append(chunk_id)

    def record_load_latency(self, latency_ms: float):
        self.total_requests += 1
        self.total_load_latency_ms += latency_ms

    def record_search_latency(self, latency_ms: float):
        self.total_requests += 1
        self.total_search_latency_ms += latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "avg_load_latency_ms": round(self.avg_load_latency_ms, 2),
            "total_search_latency_ms": round(self.total_search_latency_ms, 2),
        }

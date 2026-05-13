"""
LCM v2 KV Cache 联动优化
将索引段固定在 Prompt 前缀，利用云厂商缓存机制降低连续轮 Prefill
"""
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from .lcm_types import ContextChunk
from .logger import get_logger

logger = get_logger()


@dataclass
class CacheEntry:
    """KV Cache 条目"""
    key: str
    content_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    estimated_tokens: int = 0


class KVCacheManager:
    """
    KV Cache 管理器
    管理 Prompt 前缀的缓存，确保索引部分在会话中保持不变
    
    原理：
    - 云厂商（Anthropic/DeepSeek）支持 Prompt Caching，相同前缀的 KV Cache 可复用
    - LCM 的索引部分（System Prompt + Chunk 索引）在整个会话中不变
    - 将索引部分固定在前缀，后续轮次只需支付增量 Prefill 开销
    """

    # 支持的缓存提供商格式
    PROVIDER_ANTHROPIC = "anthropic"
    PROVIDER_DEEPSEEK = "deepseek"
    PROVIDER_OPENAI = "openai"

    def __init__(self, cache_prefix: str = "lcm_index", provider: str = PROVIDER_ANTHROPIC):
        self.cache_prefix = cache_prefix
        self.provider = provider
        self._entries: Dict[str, CacheEntry] = {}
        self._index_content: str = ""
        self._index_hash: str = ""
        self._cache_hits = 0
        self._cache_misses = 0

    def set_index_content(self, system_prompt: str, chunk_index: str) -> str:
        """
        设置索引内容并计算缓存键
        
        Args:
            system_prompt: System Prompt 内容
            chunk_index: Chunk 索引文本
        
        Returns:
            缓存键
        """
        self._index_content = f"{system_prompt}\n\n{chunk_index}"
        self._index_hash = self._compute_hash(self._index_content)
        cache_key = f"{self.cache_prefix}:{self._index_hash}"

        if cache_key not in self._entries:
            self._entries[cache_key] = CacheEntry(
                key=cache_key,
                content_hash=self._index_hash,
                estimated_tokens=ContextChunk._estimate_tokens(self._index_content),
            )
            logger.info("创建新的 KV Cache 条目", key=cache_key, tokens=self._entries[cache_key].estimated_tokens)
        else:
            logger.debug("复用已有 KV Cache 条目", key=cache_key)

        return cache_key

    def get_cache_key(self) -> Optional[str]:
        """获取当前索引的缓存键"""
        if not self._index_hash:
            return None
        return f"{self.cache_prefix}:{self._index_hash}"

    def record_use(self, cache_key: str) -> None:
        """记录缓存使用"""
        if cache_key in self._entries:
            entry = self._entries[cache_key]
            entry.use_count += 1
            entry.last_used_at = datetime.now()
            self._cache_hits += 1
            logger.debug("KV Cache 命中", key=cache_key, total_uses=entry.use_count)
        else:
            self._cache_misses += 1
            logger.warning("KV Cache 未命中", key=cache_key)

    def get_cache_headers(self) -> Dict[str, str]:
        """
        获取 API 请求的缓存相关 HTTP Headers
        
        支持：
        - Anthropic: anthropic-beta: prompt-caching-2024-07-31
        - DeepSeek: 通过前缀一致性自动触发
        - OpenAI: 不支持显式缓存控制
        """
        headers = {}
        
        if self.provider == self.PROVIDER_ANTHROPIC:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        
        headers["x-lcm-cache-key"] = self.get_cache_key() or ""
        headers["x-lcm-cache-provider"] = self.provider
        
        return headers

    def build_messages_with_cache(
        self,
        base_messages: List[Dict[str, str]],
        new_content: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        构建带缓存优化的消息列表

        策略：
        1. 第一轮：完整消息（索引 + 用户输入），标记缓存控制点
        2. 后续轮次：索引部分已缓存，API 复用 KV Cache

        Anthropic Prompt Caching 格式：
        - system 消息添加 cache_control: {"type": "ephemeral"}
        - 前 1024 tokens 的上下文会被缓存

        Args:
            base_messages: 基础消息（索引部分）
            new_content: 新增内容（chunk 注入、用户消息等）

        Returns:
            优化后的消息列表（含缓存控制标记）
        """
        cache_key = self.get_cache_key()
        if not cache_key:
            return base_messages + new_content

        # 构建带缓存控制的消息
        messages = []

        # 标记 system prompt 为可缓存（Anthropic 格式）
        for msg in base_messages:
            if msg.get("role") == "system":
                cached_msg = dict(msg)
                if self.provider == self.PROVIDER_ANTHROPIC:
                    # Anthropic 需要 cache_control 标记
                    if "cache_control" not in cached_msg:
                        cached_msg["cache_control"] = {"type": "ephemeral"}
                messages.append(cached_msg)
            else:
                messages.append(msg)

        # 追加新内容
        messages.extend(new_content)

        # 记录使用
        if cache_key in self._entries:
            self.record_use(cache_key)

        return messages

    def estimate_savings(self) -> Dict[str, Any]:
        """估算缓存节省的 token 和成本"""
        if not self._entries:
            return {"status": "no_data"}
        
        current_key = self.get_cache_key()
        if not current_key or current_key not in self._entries:
            return {"status": "no_current_cache"}
        
        entry = self._entries[current_key]
        if entry.use_count <= 1:
            return {"status": "insufficient_data", "uses": entry.use_count}
        
        # 假设缓存命中率为 80%，节省的 token = 索引 token * 命中次数 * 0.8
        saved_tokens = int(entry.estimated_tokens * (entry.use_count - 1) * 0.8)
        
        # 估算成本节省（按 $0.50 / 1M tokens 计算缓存读取 vs $3.00 / 1M tokens 计算输入）
        cache_read_cost = saved_tokens * 0.50 / 1_000_000
        normal_input_cost = saved_tokens * 3.00 / 1_000_000
        saved_cost = normal_input_cost - cache_read_cost
        
        return {
            "status": "active",
            "cache_key": current_key,
            "estimated_tokens": entry.estimated_tokens,
            "use_count": entry.use_count,
            "saved_tokens": saved_tokens,
            "saved_cost_usd": round(saved_cost, 6),
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if not self._entries:
            return {"status": "empty"}

        total_uses = sum(e.use_count for e in self._entries.values())
        total_tokens = sum(e.estimated_tokens for e in self._entries.values())
        current_key = self.get_cache_key()
        
        stats = {
            "status": "active",
            "entries": len(self._entries),
            "total_uses": total_uses,
            "total_tokens": total_tokens,
            "current_key": current_key,
            "current_tokens": self._entries.get(current_key, CacheEntry("", "")).estimated_tokens if current_key else 0,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
        
        # 添加节省估算
        savings = self.estimate_savings()
        if savings["status"] == "active":
            stats["savings"] = savings
        
        return stats

    @staticmethod
    def _compute_hash(content: str) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class CachedLCMOrchestrator:
    """
    带 KV Cache 优化的 LCM Orchestrator 包装器
    自动管理索引缓存，降低多轮 Prefill 开销
    """

    def __init__(self, orchestrator, kv_cache: Optional[KVCacheManager] = None):
        self.orchestrator = orchestrator
        self.kv_cache = kv_cache or KVCacheManager()
        self._first_round = True

    def run_stream(self, messages, stream_fn, session_id=""):
        """执行带缓存优化的 LCM 流"""
        cache_key = self.kv_cache.get_cache_key()

        if self._first_round and cache_key:
            self.kv_cache.record_use(cache_key)
            self._first_round = False

        yield from self.orchestrator.run_stream(messages, stream_fn, session_id)

    def new_session(self, session_id=""):
        """开始新会话，重置缓存状态"""
        self._first_round = True
        return self.orchestrator.new_session(session_id)

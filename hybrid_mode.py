"""
LCM v2 混合模式
高频必读 chunk 直接注入 + 低频 chunk 走 LCM
解决云端 API 多轮开销问题
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta

from .lcm_types import ContextChunk
from .store import ChunkStoreV2


@dataclass
class HybridConfig:
    """混合模式配置"""
    # 高频阈值：加载次数超过此值的 chunk 视为高频
    hot_threshold: int = 3
    # 高频 chunk 直接注入（不走 LCM）
    hot_inject_directly: bool = True
    # 最近 N 天内加载的 chunk 视为高频
    hot_recency_days: int = 7
    # 强制直接注入的 chunk ID 列表
    force_direct: Set[str] = field(default_factory=set)
    # 强制走 LCM 的 chunk ID 列表（优先级高于 force_direct）
    force_lcm: Set[str] = field(default_factory=set)


class HybridChunkManager:
    """
    混合模式 Chunk 管理器
    自动识别高频 chunk 并直接注入，低频 chunk 走 LCM
    """

    def __init__(self, store: ChunkStoreV2, config: Optional[HybridConfig] = None):
        self.store = store
        self.config = config or HybridConfig()
        self._hot_chunks: Set[str] = set()
        self._last_update = datetime.min

    def classify_chunks(self, chunk_ids: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        将 chunks 分类为高频（直接注入）和低频（LCM）
        
        Returns:
            {"hot": [...], "cold": [...]}
        """
        ids = chunk_ids or list(self.store._chunks.keys())
        hot = []
        cold = []

        for cid in ids:
            if cid in self.config.force_lcm:
                cold.append(cid)
            elif cid in self.config.force_direct:
                hot.append(cid)
            elif self._is_hot(cid):
                hot.append(cid)
            else:
                cold.append(cid)

        return {"hot": hot, "cold": cold}

    def _is_hot(self, chunk_id: str) -> bool:
        """判断 chunk 是否为高频

        注意：此操作读取 chunk 的 load_count 和 last_loaded_at，
        这两个字段在 mark_loaded() 中被更新。由于 get_chunk() 内部有加锁，
        读取是线程安全的，但"读取-判断"不是原子操作，极端并发下可能有短暂不一致。
        对于分类决策来说，这种不一致是可接受的。
        """
        chunk = self.store.get_chunk(chunk_id)
        if not chunk:
            return False

        # 加载次数超过阈值
        if chunk.load_count >= self.config.hot_threshold:
            return True

        # 最近加载过
        if chunk.last_loaded_at:
            recency = datetime.now() - chunk.last_loaded_at
            if recency <= timedelta(days=self.config.hot_recency_days):
                return True

        return False

    def get_hot_chunks_content(self) -> str:
        """获取所有高频 chunk 的合并内容（用于直接注入）"""
        classified = self.classify_chunks()
        hot_contents = []

        for cid in classified["hot"]:
            chunk = self.store.get_chunk(cid)
            if chunk:
                hot_contents.append(
                    f"[高频 Chunk: \"{cid}\"]\n{chunk.content}\n"
                )

        return "\n".join(hot_contents) if hot_contents else ""

    def update_hot_chunks(self) -> None:
        """更新高频 chunk 缓存"""
        self._hot_chunks.clear()
        for cid in self.store._chunks:
            if self._is_hot(cid):
                self._hot_chunks.add(cid)
        self._last_update = datetime.now()

    def get_stats(self) -> Dict:
        """获取混合模式统计"""
        classified = self.classify_chunks()
        hot_tokens = sum(
            self.store.get_chunk(cid).tokens
            for cid in classified["hot"]
            if self.store.get_chunk(cid)
        )
        cold_tokens = sum(
            self.store.get_chunk(cid).tokens
            for cid in classified["cold"]
            if self.store.get_chunk(cid)
        )

        return {
            "hot_count": len(classified["hot"]),
            "cold_count": len(classified["cold"]),
            "hot_tokens": hot_tokens,
            "cold_tokens": cold_tokens,
            "hot_ratio": len(classified["hot"]) / (len(classified["hot"]) + len(classified["cold"]))
            if (classified["hot"] or classified["cold"]) else 0,
        }


def build_hybrid_messages(
    user_query: str,
    store: ChunkStoreV2,
    config: Optional[HybridConfig] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    构建混合模式的初始消息
    高频 chunk 直接注入，低频 chunk 放入索引
    """
    from .prompt import build_chunk_index_section_v2, LCM_SYSTEM_PROMPT_V2

    manager = HybridChunkManager(store, config)
    classified = manager.classify_chunks()

    # 高频 chunk 直接注入
    hot_content = manager.get_hot_chunks_content()

    # 低频 chunk 放入索引
    cold_summaries = []
    for cid in classified["cold"]:
        chunk = store.get_chunk(cid)
        if chunk:
            cold_summaries.append({
                "chunk_id": chunk.chunk_id,
                "summary": chunk.summary or chunk.content[:80] + "...",
                "tokens": chunk.tokens,
                "source": chunk.source,
            })

    # 构建索引文本
    index_lines = ["## 可用上下文块索引（按需加载）", ""]
    for s in cold_summaries:
        index_lines.append(
            f"- **{s['chunk_id']}** [{s.get('source', 'unknown')}] "
            f"({s['tokens']} tokens): {s['summary']}"
        )

    if not cold_summaries:
        index_lines = ["[无按需加载的上下文块（所有高频块已直接注入）]"]

    index_section = "\n".join(index_lines)

    sp = system_prompt or LCM_SYSTEM_PROMPT_V2

    # 组合 system prompt
    parts = [sp]
    if hot_content:
        parts.append(f"\n## 高频上下文（已直接注入）\n\n{hot_content}")
    parts.append(f"\n{index_section}")

    full_system = "\n".join(parts)

    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_query},
    ]

"""
LCM v2 自适应粒度
根据 chunk 关联强度和加载历史动态调整拆分策略
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from .lcm_types import ContextChunk
from .chunk_graph import ChunkGraph


@dataclass
class ChunkGroup:
    """Chunk 分组（粗粒度）"""
    group_id: str
    chunk_ids: List[str]
    combined_content: str
    combined_tokens: int
    load_count: int = 0


class AdaptiveChunking:
    """
    自适应粒度管理器
    
    策略：
    - 细粒度：独立 chunk，按需加载（适合多样化任务）
    - 粗粒度：合并语义相关 chunk，一次性加载（适合关联分析）
    - 自动切换：根据加载历史动态调整
    """

    def __init__(
        self,
        merge_threshold: float = 0.6,
        min_group_size: int = 2,
        max_group_size: int = 5,
    ):
        self.merge_threshold = merge_threshold
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self._groups: Dict[str, ChunkGroup] = {}
        self._chunk_to_group: Dict[str, str] = {}

    def analyze_chunks(
        self,
        chunks: List[ContextChunk],
        graph: Optional[ChunkGraph] = None,
    ) -> Dict[str, List[str]]:
        """
        分析 chunks 并建议分组
        
        Returns:
            {"merged": [(id1, id2, ...), ...], "standalone": [id, ...]}
        """
        merged = []
        standalone = []
        visited = set()

        for chunk in chunks:
            if chunk.chunk_id in visited:
                continue

            # 查找相关 chunks
            related = self._find_related_chunks(chunk, chunks, graph)

            if len(related) >= self.min_group_size - 1:
                group = [chunk.chunk_id] + [r.chunk_id for r in related]
                if len(group) <= self.max_group_size:
                    merged.append(tuple(group))
                    visited.update(group)
                    continue

            standalone.append(chunk.chunk_id)
            visited.add(chunk.chunk_id)

        return {"merged": merged, "standalone": standalone}

    def _find_related_chunks(
        self,
        target: ContextChunk,
        all_chunks: List[ContextChunk],
        graph: Optional[ChunkGraph] = None,
    ) -> List[ContextChunk]:
        """查找与目标 chunk 相关的其他 chunks"""
        related = []
        target_keywords = set(self._extract_keywords(target.summary + " " + target.content))

        for chunk in all_chunks:
            if chunk.chunk_id == target.chunk_id:
                continue

            # 图依赖关系
            if graph and chunk.chunk_id in graph.get_dependencies(target.chunk_id):
                related.append(chunk)
                continue

            # 关键词相似度
            chunk_keywords = set(self._extract_keywords(chunk.summary + " " + chunk.content))
            overlap = len(target_keywords & chunk_keywords)
            union = len(target_keywords | chunk_keywords)
            similarity = overlap / union if union > 0 else 0

            if similarity >= self.merge_threshold:
                related.append(chunk)

        return related[:self.max_group_size - 1]

    def create_group(
        self,
        group_id: str,
        chunks: List[ContextChunk],
    ) -> ChunkGroup:
        """创建粗粒度分组"""
        combined = "\n\n".join(
            f"[{c.chunk_id}]\n{c.content}" for c in chunks
        )
        total_tokens = sum(c.tokens for c in chunks)
        total_loads = sum(c.load_count for c in chunks)

        group = ChunkGroup(
            group_id=group_id,
            chunk_ids=[c.chunk_id for c in chunks],
            combined_content=combined,
            combined_tokens=total_tokens,
            load_count=total_loads,
        )

        self._groups[group_id] = group
        for c in chunks:
            self._chunk_to_group[c.chunk_id] = group_id

        return group

    def should_merge(self, chunk_ids: List[str], chunks_dict: Dict[str, ContextChunk]) -> bool:
        """
        判断一组 chunk 是否应该合并
        
        合并条件：
        1. 经常一起被加载（>50% 同时加载）
        2. 语义高度相关（关键词重叠 >60%）
        3. 总 token 数不超过阈值（避免过大）
        """
        if len(chunk_ids) < self.min_group_size:
            return False

        total_tokens = sum(chunks_dict[cid].tokens for cid in chunk_ids if cid in chunks_dict)
        if total_tokens > 4000:  # 最大合并 token 限制
            return False

        # 检查是否经常一起加载
        load_history = [chunks_dict[cid].load_count for cid in chunk_ids if cid in chunks_dict]
        if not load_history:
            return False

        avg_load = sum(load_history) / len(load_history)
        variance = sum((l - avg_load) ** 2 for l in load_history) / len(load_history)

        # 加载次数相近说明经常一起使用
        if variance < avg_load * 0.5:
            return True

        return False

    def get_group_for_chunk(self, chunk_id: str) -> Optional[ChunkGroup]:
        """获取 chunk 所属的分组"""
        group_id = self._chunk_to_group.get(chunk_id)
        if group_id:
            return self._groups.get(group_id)
        return None

    def get_all_groups(self) -> List[ChunkGroup]:
        """获取所有分组"""
        return list(self._groups.values())

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        """提取关键词（简单实现）"""
        import re
        # 提取英文单词和中文词语
        words = re.findall(r'[a-zA-Z_]+|[\u4e00-\u9fff]', text.lower())
        # 过滤常见停用词
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "must", "shall", "can",
                     "need", "dare", "ought", "used", "to", "of", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through",
                     "during", "before", "after", "above", "below", "between",
                     "under", "again", "further", "then", "once", "here", "there",
                     "when", "where", "why", "how", "all", "each", "few", "more",
                     "most", "other", "some", "such", "no", "nor", "not", "only",
                     "own", "same", "so", "than", "too", "very", "just", "的", "了", "在"}
        return set(w for w in words if w not in stopwords and len(w) > 1)


class AdaptiveChunkStore:
    """
    自适应粒度 Chunk Store
    根据使用模式自动在粗/细粒度间切换
    """

    def __init__(self, base_store, adaptive: Optional[AdaptiveChunking] = None):
        self.base_store = base_store
        self.adaptive = adaptive or AdaptiveChunking()
        self._access_log: List[Tuple[str, float]] = []  # (chunk_id, timestamp)

    def get_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """获取 chunk，自动处理分组"""
        import time

        # 记录访问
        self._access_log.append((chunk_id, time.time()))

        # 检查是否在分组中
        group = self.adaptive.get_group_for_chunk(chunk_id)
        if group:
            # 返回分组内容
            return ContextChunk(
                chunk_id=group.group_id,
                content=group.combined_content,
                summary=f"Grouped: {', '.join(group.chunk_ids)}",
                tokens=group.combined_tokens,
                load_count=group.load_count,
            )

        return self.base_store.get_chunk(chunk_id)

    def analyze_and_merge(self) -> List[ChunkGroup]:
        """分析访问模式并合并频繁共现的 chunks"""
        from collections import Counter

        # 统计共现模式
        cooccurrence = Counter()
        window_size = 10

        for i in range(len(self._access_log)):
            window = self._access_log[max(0, i - window_size):i + 1]
            chunk_ids = [cid for cid, _ in window]
            for j in range(len(chunk_ids)):
                for k in range(j + 1, len(chunk_ids)):
                    pair = tuple(sorted([chunk_ids[j], chunk_ids[k]]))
                    cooccurrence[pair] += 1

        # 找出频繁共现的对
        frequent_pairs = [pair for pair, count in cooccurrence.items() if count >= 3]

        # 合并为组
        merged_groups = []
        for pair in frequent_pairs:
            chunks = []
            for cid in pair:
                c = self.base_store.get_chunk(cid)
                if c:
                    chunks.append(c)
            if len(chunks) >= 2:
                group_id = f"group_{pair[0]}_{pair[1]}"
                group = self.adaptive.create_group(group_id, chunks)
                merged_groups.append(group)

        return merged_groups

    def get_stats(self) -> Dict:
        """获取自适应统计"""
        return {
            "base_store": self.base_store.get_stats(),
            "groups": len(self.adaptive.get_all_groups()),
            "access_log": len(self._access_log),
            "adaptive_config": {
                "merge_threshold": self.adaptive.merge_threshold,
                "min_group_size": self.adaptive.min_group_size,
                "max_group_size": self.adaptive.max_group_size,
            },
        }

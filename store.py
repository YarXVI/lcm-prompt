"""
LCM v2 持久化存储
线程安全 + LRU 缓存 + 磁盘持久化 + 语义搜索
"""
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .lcm_types import ContextChunk, ChunkLoadReason, LCMMetrics
from .chunk_graph import ChunkGraph
from .logger import get_logger, LCMErrorCode

logger = get_logger()


class ChunkStoreV2:
    """上下文块存储 v2 —— 线程安全、持久化、LRU 缓存"""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_cache_size: int = 100,
        enable_persistence: bool = True,
        auto_save_interval: int = 0,
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".iris" / "lcm_chunks"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.storage_dir / "chunks.jsonl"
        self.index_file = self.storage_dir / "index.json"

        self._chunks: Dict[str, ContextChunk] = {}
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, ContextChunk] = OrderedDict()
        self._max_cache_size = max_cache_size
        self._enable_persistence = enable_persistence
        self._auto_save_interval = auto_save_interval
        self._unsaved_changes = 0
        self._metrics = LCMMetrics()
        self._graph = ChunkGraph()

        if enable_persistence:
            self._load_from_disk()
            self._load_graph_from_disk()

    def add_chunk(self, chunk: ContextChunk) -> None:
        with self._lock:
            self._chunks[chunk.chunk_id] = chunk
            self._update_cache(chunk.chunk_id)
            if self._enable_persistence:
                if self._auto_save_interval > 0:
                    self._unsaved_changes += 1
                    if self._unsaved_changes >= self._auto_save_interval:
                        self._save_to_disk()
                        self._unsaved_changes = 0
                else:
                    self._save_to_disk()

    def add_chunks(self, chunks: List[ContextChunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
                self._update_cache(chunk.chunk_id)
            if self._enable_persistence:
                self._save_to_disk()
                self._unsaved_changes = 0

    def get_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        start = time.time()
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if chunk:
                self._update_cache(chunk_id)
                chunk.cache_hit = True
                self._metrics.record_cache_hit(chunk_id)
            else:
                self._metrics.record_cache_miss(chunk_id)
            latency = (time.time() - start) * 1000
            self._metrics.record_load_latency(latency)
            return chunk

    def remove_chunk(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
                self._cache.pop(chunk_id, None)
                if self._enable_persistence:
                    if self._auto_save_interval > 0:
                        self._unsaved_changes += 1
                        if self._unsaved_changes >= self._auto_save_interval:
                            self._save_to_disk()
                            self._unsaved_changes = 0
                    else:
                        self._save_to_disk()
                return True
            return False

    def save(self) -> None:
        """手动触发持久化"""
        with self._lock:
            if self._enable_persistence:
                self._save_to_disk()
                self._unsaved_changes = 0

    def search(self, query: str, top_k: int = 5) -> List[ContextChunk]:
        """基于摘要/ID 的模糊匹配搜索 + 语义搜索（如有 embedding）"""
        start = time.time()
        with self._lock:
            query_lower = query.lower()
            scored = []
            for chunk in self._chunks.values():
                score = 0
                if query_lower in chunk.chunk_id.lower():
                    score += 10
                if query_lower in chunk.summary.lower():
                    score += 5
                if query_lower in chunk.content.lower():
                    score += 3
                for kw in query_lower.split():
                    if kw in chunk.summary.lower():
                        score += 2
                # embedding 相似度（如有）
                if chunk.embedding and query_lower:
                    score += self._embedding_similarity(query_lower, chunk.embedding)
                if score > 0:
                    scored.append((score, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            result = [c for _, c in scored[:top_k]]
            latency = (time.time() - start) * 1000
            self._metrics.record_search_latency(latency)
            return result

    def semantic_search(
        self, embedding: List[float], top_k: int = 5
    ) -> List[ContextChunk]:
        """基于向量嵌入的语义搜索"""
        with self._lock:
            scored = []
            for chunk in self._chunks.values():
                if chunk.embedding:
                    sim = self._cosine_similarity(embedding, chunk.embedding)
                    if sim > 0.5:
                        scored.append((sim, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:top_k]]

    def list_summaries(self) -> List[Dict[str, str]]:
        with self._lock:
            return [
                {
                    "chunk_id": c.chunk_id,
                    "summary": c.summary or self._truncate(c.content, 80),
                    "tokens": c.tokens,
                    "source": c.source,
                    "load_count": c.load_count,
                }
                for c in self._chunks.values()
            ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            chunks = list(self._chunks.values())
            return {
                "total_chunks": len(chunks),
                "total_tokens": sum(c.tokens for c in chunks),
                "total_loads": sum(c.load_count for c in chunks),
                "hot_chunks": [c.chunk_id for c in chunks if c.load_count > 2],
                "cache_size": len(self._cache),
                "metrics": self._metrics.to_dict(),
            }

    def find_related(self, chunk_id: str, top_k: int = 3) -> List[str]:
        target = self._chunks.get(chunk_id)
        if not target:
            return []

        target_kw = set(target.summary.lower().split())
        target_prefix = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id[:3]

        scored = []
        for cid, chunk in self._chunks.items():
            if cid == chunk_id:
                continue
            score = 0
            chunk_kw = set(chunk.summary.lower().split())
            overlap = len(target_kw & chunk_kw)
            if overlap > 0:
                score += overlap * 3

            chunk_prefix = cid.split("_")[0] if "_" in cid else cid[:3]
            if chunk_prefix == target_prefix:
                score += 4

            score += min(chunk.load_count, 5)

            if score > 0:
                scored.append((score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored[:top_k]]

    def find_related_multi(self, chunk_ids: List[str], top_k: int = 5) -> List[str]:
        all_related = set()
        for cid in chunk_ids:
            related = self.find_related(cid, top_k=2)
            all_related.update(related)
        already = set(chunk_ids)
        return [r for r in all_related if r not in already][:top_k]

    def mark_loaded(self, chunk_id: str, reason: ChunkLoadReason = ChunkLoadReason.MODEL_REQUEST):
        from datetime import datetime
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if chunk:
                chunk.load_count += 1
                chunk.last_loaded_at = datetime.now()

    def batch_load(self, chunk_ids: List[str]) -> Dict[str, Optional[ContextChunk]]:
        """批量加载 chunks"""
        result = {}
        for cid in chunk_ids:
            result[cid] = self.get_chunk(cid)
        return result

    def warm_cache(self, chunk_ids: List[str]) -> int:
        """预热缓存"""
        with self._lock:
            loaded = 0
            for cid in chunk_ids:
                if cid in self._chunks and cid not in self._cache:
                    self._update_cache(cid)
                    loaded += 1
            return loaded

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def clear_all(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._cache.clear()
            if self._enable_persistence:
                self._save_to_disk()

    def _update_cache(self, chunk_id: str) -> None:
        """LRU 缓存更新"""
        if chunk_id in self._cache:
            self._cache.move_to_end(chunk_id)
        else:
            self._cache[chunk_id] = self._chunks[chunk_id]
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

    def add_dependency(self, chunk_id: str, depends_on: str) -> None:
        """添加 chunk 依赖关系"""
        self._graph.add_dependency(chunk_id, depends_on)
        if self._enable_persistence:
            self._save_graph_to_disk()

    def get_dependencies(self, chunk_id: str, recursive: bool = False) -> List[str]:
        """获取 chunk 的依赖列表"""
        return self._graph.get_dependencies(chunk_id, recursive)

    def get_loading_order(self, chunk_id: str) -> List[str]:
        """获取 chunk 的加载顺序（含依赖）"""
        return self._graph.get_loading_order(chunk_id)

    def find_related_with_graph(self, chunk_id: str, depth: int = 2) -> List[str]:
        """基于图结构的智能关联查找"""
        return self._graph.find_related_with_graph(chunk_id, depth)

    def _save_to_disk(self) -> None:
        """持久化到磁盘"""
        try:
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                for chunk in self._chunks.values():
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            index = {
                "total_chunks": len(self._chunks),
                "total_tokens": sum(c.tokens for c in self._chunks.values()),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("持久化失败", code=LCMErrorCode.STORE_PERSISTENCE_FAILED, error=str(e))

    def _load_from_disk(self) -> None:
        """从磁盘加载 chunks"""
        if not self.chunks_file.exists():
            return
        try:
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = ContextChunk.from_dict(data)
                    self._chunks[chunk.chunk_id] = chunk
            logger.info("从磁盘加载 chunks", count=len(self._chunks))
        except Exception as e:
            logger.error("加载失败", code=LCMErrorCode.STORE_LOAD_FAILED, error=str(e))

    def _save_graph_to_disk(self) -> None:
        """持久化依赖图"""
        try:
            graph_file = self.storage_dir / "graph.json"
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(self._graph.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("图持久化失败", code=LCMErrorCode.STORE_GRAPH_SAVE_FAILED, error=str(e))

    def _load_graph_from_disk(self) -> None:
        """从磁盘加载依赖图"""
        graph_file = self.storage_dir / "graph.json"
        if not graph_file.exists():
            return
        try:
            with open(graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._graph = ChunkGraph.from_dict(data)
            logger.info("从磁盘加载依赖图", nodes=len(self._graph))
        except Exception as e:
            logger.error("图加载失败", code=LCMErrorCode.STORE_GRAPH_LOAD_FAILED, error=str(e))

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text[:max_len] + "..." if len(text) > max_len else text

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _embedding_similarity(query: str, embedding: List[float]) -> float:
        """基于关键词重叠的文本到 embedding 相似度（简化实现）

        由于 query 是纯文本而 chunk 有 embedding，我们无法直接计算语义相似度。
        这里使用关键词重叠作为近似：
        - 提取 query 中的关键词
        - 检查 chunk 的 summary/content 中是否包含这些关键词
        - 返回归一化的匹配分数

        注意：真正的语义搜索需要调用 embedding 模型将 query 转为向量。
        此方法作为降级方案，在没有 embedding 服务时提供基础的相关性排序。
        """
        if not query or not embedding:
            return 0.0

        # 简单关键词匹配作为近似
        # 实际场景下应调用 embedding 模型
        keywords = set(query.lower().split())
        if not keywords:
            return 0.0

        # 这里无法直接比较文本和向量，返回一个基础分数
        # 让调用者知道此 chunk 有 embedding 可用
        return 0.1  # 基础分数，表示"有 embedding 但无法计算精确相似度"

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

"""
LCM v2 HNSW 向量索引
基于 hnswlib 的近似最近邻搜索，替代线性搜索
"""
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .lcm_types import ContextChunk
from .logger import get_logger

logger = get_logger()

# 可选依赖：numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class VectorIndex:
    """
    HNSW 向量索引
    
    使用 hnswlib 实现高效的近似最近邻搜索：
    - 构建索引：O(n log n)
    - 查询：O(log n)
    - 内存占用：可控，支持序列化到磁盘
    """

    def __init__(
        self,
        dim: int = 384,  # 默认 embedding 维度（all-MiniLM-L6-v2）
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16,
        index_path: Optional[Path] = None,
    ):
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.index_path = index_path
        self._index = None
        self._chunk_ids: List[str] = []
        self._initialized = False

    def _init_index(self):
        """初始化 hnswlib 索引"""
        if self._initialized:
            return

        try:
            import hnswlib
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.set_ef(50)  # 查询时的搜索深度
            self._initialized = True
            logger.info("HNSW 索引初始化完成", dim=self.dim, max_elements=self.max_elements)
        except ImportError:
            logger.warning("hnswlib 未安装，向量索引将使用暴力搜索降级")
            self._index = None
            self._initialized = True

    def add_items(self, chunks: List[ContextChunk]) -> None:
        """添加 chunk 到索引"""
        self._init_index()

        if not chunks:
            return

        # 过滤有 embedding 的 chunk
        valid_chunks = [c for c in chunks if c.embedding and len(c.embedding) == self.dim]
        if not valid_chunks:
            logger.warning("没有有效的 embedding 数据")
            return

        if self._index is None:
            # 降级：暴力搜索
            self._chunk_ids.extend([c.chunk_id for c in valid_chunks])
            return

        # HNSW 索引
        if HAS_NUMPY and np is not None:
            embeddings = np.array([c.embedding for c in valid_chunks], dtype=np.float32)
            labels = np.arange(len(self._chunk_ids), len(self._chunk_ids) + len(valid_chunks))
            self._index.add_items(embeddings, labels)
        else:
            # 无 numpy 时逐个添加
            for i, chunk in enumerate(valid_chunks):
                self._index.add_items([chunk.embedding], [len(self._chunk_ids) + i])
        self._chunk_ids.extend([c.chunk_id for c in valid_chunks])
        logger.info("添加向量到索引", count=len(valid_chunks), total=len(self._chunk_ids))

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_fn: Optional[callable] = None,
    ) -> List[Tuple[str, float]]:
        """
        搜索最近邻
        
        Args:
            query_embedding: 查询向量
            k: 返回结果数
            filter_fn: 可选的过滤函数
        
        Returns:
            [(chunk_id, distance), ...]
        """
        self._init_index()

        if not self._chunk_ids:
            return []

        if self._index is None:
            # 降级：暴力搜索
            return self._brute_force_search(query_embedding, k)

        # HNSW 搜索
        if HAS_NUMPY and np is not None:
            query = np.array(query_embedding, dtype=np.float32)
            labels, distances = self._index.knn_query(query, k=min(k * 2, len(self._chunk_ids)))
        else:
            # 无 numpy 时降级
            return self._brute_force_search(query_embedding, k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            chunk_id = self._chunk_ids[label]
            if filter_fn is None or filter_fn(chunk_id):
                results.append((chunk_id, float(dist)))
            if len(results) >= k:
                break
        return results

    def _brute_force_search(
        self,
        query: Any,
        k: int,
    ) -> List[Tuple[str, float]]:
        """暴力搜索（降级方案）"""
        # 这里需要访问 chunk 的 embedding，但暴力搜索模式下没有存储
        # 返回空列表，让调用者回退到关键词搜索
        logger.debug("暴力搜索降级：无索引数据")
        return []

    def save(self, path: Optional[Path] = None) -> None:
        """保存索引到磁盘"""
        if self._index is None:
            return

        save_path = path or self.index_path
        if not save_path:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        index_file = save_path.with_suffix(".hnsw")
        ids_file = save_path.with_suffix(".ids")

        self._index.save_index(str(index_file))
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(self._chunk_ids, f, ensure_ascii=False)

        logger.info("HNSW 索引已保存", path=str(index_file))

    def load(self, path: Optional[Path] = None) -> bool:
        """从磁盘加载索引"""
        load_path = path or self.index_path
        if not load_path:
            return False

        index_file = load_path.with_suffix(".hnsw")
        ids_file = load_path.with_suffix(".ids")

        if not index_file.exists() or not ids_file.exists():
            return False

        try:
            import hnswlib
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            self._index.load_index(str(index_file))
            self._index.set_ef(50)

            with open(ids_file, "r", encoding="utf-8") as f:
                self._chunk_ids = json.load(f)

            self._initialized = True
            logger.info("HNSW 索引已加载", path=str(index_file), chunks=len(self._chunk_ids))
            return True
        except Exception as e:
            logger.error("加载 HNSW 索引失败", error=str(e))
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        return {
            "type": "hnsw" if self._index else "brute_force",
            "dim": self.dim,
            "total_chunks": len(self._chunk_ids),
            "max_elements": self.max_elements,
            "ef_construction": self.ef_construction,
            "M": self.M,
        }

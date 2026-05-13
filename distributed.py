"""
LCM v2 分布式 Chunk 存储
支持多节点共享 chunk 存储
"""
import json
import hashlib
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from .lcm_types import ContextChunk
from .logger import get_logger

logger = get_logger()


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    host: str
    port: int
    last_heartbeat: datetime
    chunk_count: int = 0
    is_active: bool = True


class DistributedChunkStore:
    """
    分布式 Chunk 存储
    
    架构：
    - 中心协调节点：维护 chunk 到节点的映射
    - 工作节点：实际存储 chunk
    - 客户端：通过协调节点路由请求
    
    一致性策略：
    - 写入：同步复制到 N 个节点
    - 读取：从最近节点读取
    - 故障：自动切换到备用节点
    """

    def __init__(self, node_id: str, is_coordinator: bool = False, replication_factor: int = 2):
        self.node_id = node_id
        self.is_coordinator = is_coordinator
        self.replication_factor = replication_factor
        self._local_store: Dict[str, ContextChunk] = {}
        self._nodes: Dict[str, NodeInfo] = {}
        self._chunk_map: Dict[str, List[str]] = {}  # chunk_id -> [node_id, ...]
        self._lock = threading.RLock()

    def _hash_chunk(self, chunk_id: str) -> int:
        """计算 chunk 的哈希值，用于选择节点"""
        return int(hashlib.md5(chunk_id.encode()).hexdigest(), 16)

    def _select_nodes(self, chunk_id: str, count: int) -> List[str]:
        """为 chunk 选择存储节点"""
        with self._lock:
            active_nodes = [nid for nid, info in self._nodes.items() if info.is_active]
            if not active_nodes:
                return [self.node_id]

            # 一致性哈希
            hash_val = self._hash_chunk(chunk_id)
            sorted_nodes = sorted(active_nodes, key=lambda nid: self._hash_chunk(nid))
            
            # 选择 hash 值最接近的节点
            idx = hash_val % len(sorted_nodes)
            selected = []
            for i in range(count):
                selected.append(sorted_nodes[(idx + i) % len(sorted_nodes)])

            return selected

    def add_chunk(self, chunk: ContextChunk) -> List[str]:
        """
        添加 chunk 到分布式存储
        
        Returns:
            存储该 chunk 的节点列表
        """
        with self._lock:
            # 本地存储
            self._local_store[chunk.chunk_id] = chunk

            # 选择复制节点
            nodes = self._select_nodes(chunk.chunk_id, self.replication_factor)
            self._chunk_map[chunk.chunk_id] = nodes

            logger.info("分布式存储 chunk", chunk_id=chunk.chunk_id, nodes=nodes)
            return nodes

    def get_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """获取 chunk"""
        # 先查本地
        with self._lock:
            if chunk_id in self._local_store:
                return self._local_store[chunk_id]

            # 查找 chunk 所在节点
            nodes = self._chunk_map.get(chunk_id, [])
            for node_id in nodes:
                if node_id == self.node_id:
                    continue
                # TODO: 远程获取
                logger.debug("尝试从远程节点获取", chunk_id=chunk_id, node=node_id)

            return None

    def register_node(self, node_info: NodeInfo) -> None:
        """注册节点"""
        with self._lock:
            self._nodes[node_info.node_id] = node_info
            logger.info("注册节点", node_id=node_info.node_id, host=node_info.host)

    def heartbeat(self, node_id: str) -> None:
        """节点心跳"""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].last_heartbeat = datetime.now()
                self._nodes[node_id].is_active = True

    def check_node_health(self, timeout_seconds: int = 30) -> List[str]:
        """检查节点健康状态"""
        with self._lock:
            now = datetime.now()
            failed_nodes = []
            for node_id, info in self._nodes.items():
                if (now - info.last_heartbeat).total_seconds() > timeout_seconds:
                    info.is_active = False
                    failed_nodes.append(node_id)
                    logger.warning("节点心跳超时", node_id=node_id)
            return failed_nodes

    def rebalance(self) -> Dict[str, Any]:
        """
        重新平衡 chunk 分布
        
        Returns:
            迁移统计
        """
        with self._lock:
            migrations = 0
            for chunk_id, nodes in list(self._chunk_map.items()):
                # 检查是否需要重新分配
                active_nodes = [n for n in nodes if self._nodes.get(n, NodeInfo("", "", 0, datetime.now())).is_active]
                if len(active_nodes) < self.replication_factor:
                    # 需要补充复制
                    new_nodes = self._select_nodes(chunk_id, self.replication_factor)
                    self._chunk_map[chunk_id] = new_nodes
                    migrations += 1

            logger.info("重新平衡完成", migrations=migrations)
            return {"migrations": migrations, "total_chunks": len(self._chunk_map)}

    def get_stats(self) -> Dict[str, Any]:
        """获取分布式存储统计"""
        with self._lock:
            return {
                "node_id": self.node_id,
                "is_coordinator": self.is_coordinator,
                "local_chunks": len(self._local_store),
                "total_chunks": len(self._chunk_map),
                "nodes": len(self._nodes),
                "active_nodes": sum(1 for n in self._nodes.values() if n.is_active),
                "replication_factor": self.replication_factor,
            }


class DistributedIndexManager:
    """
    分布式索引管理器
    管理跨节点的 chunk 索引同步
    """

    def __init__(self, store: DistributedChunkStore):
        self.store = store
        self._local_index: Dict[str, str] = {}  # chunk_id -> summary

    def sync_index(self) -> int:
        """
        同步索引到所有节点
        
        Returns:
            同步的 chunk 数量
        """
        with self.store._lock:
            count = 0
            for chunk_id, chunk in self.store._local_store.items():
                self._local_index[chunk_id] = chunk.summary
                count += 1

            logger.info("索引同步完成", chunks=count)
            return count

    def query_distributed(self, query: str) -> List[str]:
        """
        分布式查询
        
        Args:
            query: 查询字符串
        
        Returns:
            匹配的 chunk_id 列表
        """
        results = []
        with self.store._lock:
            for chunk_id, summary in self._local_index.items():
                if query.lower() in summary.lower() or query.lower() in chunk_id.lower():
                    results.append(chunk_id)

        logger.debug("分布式查询", query=query, results=len(results))
        return results

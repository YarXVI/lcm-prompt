"""
LCM v2 Chunk 依赖图
支持显式声明 chunk 依赖关系，构建有向无环图（DAG）
实现智能预取和依赖加载
"""
import threading
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Tuple

from .lcm_types import ContextChunk


class ChunkGraph:
    """
    Chunk 依赖图
    管理 chunk 之间的依赖关系，支持拓扑排序和依赖加载
    """

    def __init__(self):
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add_dependency(self, chunk_id: str, depends_on: str) -> None:
        """
        添加依赖关系：chunk_id 依赖于 depends_on
        
        示例：
            graph.add_dependency("func_login", "func_validate")
            表示 login 函数依赖于 validate 函数
        
        注意：允许自依赖（chunk_id == depends_on），这会在 has_cycle() 中被检测为环。
        调用者应根据业务需求决定是否允许自依赖。
        """
        with self._lock:
            self._dependencies[chunk_id].add(depends_on)
            self._dependents[depends_on].add(chunk_id)

    def add_dependencies(self, chunk_id: str, depends_on_list: List[str]) -> None:
        """批量添加依赖"""
        for dep in depends_on_list:
            self.add_dependency(chunk_id, dep)

    def remove_dependency(self, chunk_id: str, depends_on: str) -> bool:
        """移除依赖关系"""
        with self._lock:
            if depends_on in self._dependencies[chunk_id]:
                self._dependencies[chunk_id].remove(depends_on)
                self._dependents[depends_on].discard(chunk_id)
                return True
            return False

    def remove_chunk(self, chunk_id: str) -> None:
        """移除 chunk 及其所有依赖关系"""
        with self._lock:
            # 移除该 chunk 依赖的其他 chunk
            for dep in list(self._dependencies[chunk_id]):
                self._dependents[dep].discard(chunk_id)
            del self._dependencies[chunk_id]

            # 移除其他 chunk 对该 chunk 的依赖
            for dependent in list(self._dependents[chunk_id]):
                self._dependencies[dependent].discard(chunk_id)
            del self._dependents[chunk_id]

    def get_dependencies(self, chunk_id: str, recursive: bool = False) -> List[str]:
        """
        获取 chunk 的依赖列表
        
        Args:
            chunk_id: chunk ID
            recursive: 是否递归获取所有间接依赖
        
        Returns:
            依赖列表（按拓扑排序）
        """
        with self._lock:
            if not recursive:
                return list(self._dependencies[chunk_id])

            # BFS 获取所有依赖
            visited = set()
            queue = deque([chunk_id])
            result = []

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                for dep in self._dependencies[current]:
                    if dep not in visited:
                        queue.append(dep)
                        result.append(dep)

            return result

    def get_dependents(self, chunk_id: str) -> List[str]:
        """获取依赖于该 chunk 的所有 chunk"""
        with self._lock:
            return list(self._dependents[chunk_id])

    def topological_sort(self, chunk_ids: Optional[List[str]] = None) -> List[str]:
        """
        拓扑排序
        返回按依赖关系排序的 chunk ID 列表（被依赖的在前）

        注意：如果图中存在环，会返回部分排序结果，并在日志中警告。
        调用者应先使用 has_cycle() 检查。
        """
        with self._lock:
            ids = set(chunk_ids) if chunk_ids else set(self._dependencies.keys())
            # 包含所有被依赖的节点
            for cid in list(ids):
                for dep in self._dependencies[cid]:
                    ids.add(dep)

            # 计算入度（每个节点依赖其他节点的数量）
            in_degree = {cid: 0 for cid in ids}
            for cid in ids:
                for dep in self._dependencies[cid]:
                    if dep in ids:
                        in_degree[cid] += 1

            # Kahn 算法：从入度为 0 的节点开始（不依赖其他节点的）
            queue = deque([cid for cid, degree in in_degree.items() if degree == 0])
            result = []

            while queue:
                current = queue.popleft()
                result.append(current)

                for dependent in self._dependents[current]:
                    if dependent in in_degree:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

            # 检查是否有未处理的节点（存在环时）
            if len(result) != len(ids):
                unprocessed = ids - set(result)
                print(f"[ChunkGraph] 警告: 拓扑排序发现 {len(unprocessed)} 个未处理节点（可能存在环）: {unprocessed}")
                # 将未处理的节点追加到结果末尾（尽力而为）
                result.extend(unprocessed)

            return result

    def has_cycle(self) -> bool:
        """检测图中是否存在环（包括自依赖）"""
        with self._lock:
            # 先检查自依赖（A -> A）
            for node, deps in self._dependencies.items():
                if node in deps:
                    return True

            visited = set()
            rec_stack = set()

            def dfs(node):
                visited.add(node)
                rec_stack.add(node)

                for dep in self._dependencies[node]:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True

                rec_stack.remove(node)
                return False

            for node in list(self._dependencies.keys()):
                if node not in visited:
                    if dfs(node):
                        return True

            return False

    def get_loading_order(self, chunk_id: str) -> List[str]:
        """
        获取加载顺序
        返回加载 chunk_id 前需要按顺序加载的所有依赖
        """
        deps = self.get_dependencies(chunk_id, recursive=True)
        return self.topological_sort(deps)

    def find_related_with_graph(self, chunk_id: str, depth: int = 2) -> List[str]:
        """
        基于图结构的智能关联查找
        
        Args:
            chunk_id: 起始 chunk
            depth: 搜索深度
        
        Returns:
            关联 chunk ID 列表
        """
        with self._lock:
            related = set()
            queue = deque([(chunk_id, 0)])
            visited = {chunk_id}

            while queue:
                current, current_depth = queue.popleft()

                if current_depth >= depth:
                    continue

                # 添加依赖
                for dep in self._dependencies[current]:
                    if dep not in visited:
                        visited.add(dep)
                        related.add(dep)
                        queue.append((dep, current_depth + 1))

                # 添加被依赖
                for dependent in self._dependents[current]:
                    if dependent not in visited:
                        visited.add(dependent)
                        related.add(dependent)
                        queue.append((dependent, current_depth + 1))

            return list(related)

    def get_stats(self) -> Dict:
        """获取图的统计信息"""
        with self._lock:
            # 收集所有节点（包括只有被依赖的节点）
            all_nodes = set(self._dependencies.keys())
            for deps in self._dependencies.values():
                all_nodes.update(deps)
            total_edges = sum(len(deps) for deps in self._dependencies.values())
            return {
                "total_nodes": len(all_nodes),
                "total_edges": total_edges,
                "avg_dependencies": total_edges / len(all_nodes) if all_nodes else 0,
                "has_cycle": self.has_cycle(),
            }

    def to_dict(self) -> Dict[str, List[str]]:
        """序列化为字典"""
        with self._lock:
            return {k: list(v) for k, v in self._dependencies.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "ChunkGraph":
        """从字典反序列化"""
        graph = cls()
        for chunk_id, deps in data.items():
            graph.add_dependencies(chunk_id, deps)
        return graph

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._dependencies

    def __len__(self) -> int:
        return len(self._dependencies)

"""
LCM v2 多 Agent 协作优化
在多 Agent 场景中共享组件索引，避免重复注入
"""
import threading
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

from .lcm_types import ContextChunk
from .store import ChunkStoreV2


@dataclass
class AgentSession:
    """Agent 会话"""
    agent_id: str
    role: str  # "reviewer", "security", "architect", etc.
    injected_chunks: Set[str] = field(default_factory=set)
    shared_chunks: Set[str] = field(default_factory=set)


class SharedIndexManager:
    """
    共享索引管理器
    管理多 Agent 间的共享组件索引
    """

    def __init__(self, store: ChunkStoreV2):
        self.store = store
        self._agents: Dict[str, AgentSession] = {}
        self._shared_index: Set[str] = set()  # 所有 Agent 共享的 chunk
        self._lock = threading.RLock()

    def register_agent(self, agent_id: str, role: str) -> AgentSession:
        """注册 Agent"""
        with self._lock:
            session = AgentSession(agent_id=agent_id, role=role)
            self._agents[agent_id] = session
            return session

    def add_shared_chunk(self, chunk_id: str) -> None:
        """添加共享 chunk（所有 Agent 可见）"""
        with self._lock:
            self._shared_index.add(chunk_id)
            for agent in self._agents.values():
                agent.shared_chunks.add(chunk_id)

    def get_agent_index(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        获取 Agent 的索引（共享 + 私有）
        
        策略：
        1. 共享 chunk 只注入一次，后续 Agent 复用
        2. 私有 chunk 按 Agent 角色过滤
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return []

            summaries = []

            # 共享 chunks
            for cid in self._shared_index:
                chunk = self.store.get_chunk(cid)
                if chunk:
                    summaries.append({
                        "chunk_id": cid,
                        "summary": chunk.summary or chunk.content[:80] + "...",
                        "tokens": chunk.tokens,
                        "source": chunk.source,
                        "shared": True,
                    })

            # Agent 私有 chunks（根据角色过滤）
            role_chunks = self._get_role_specific_chunks(agent.role)
            for cid in role_chunks:
                if cid not in self._shared_index:
                    chunk = self.store.get_chunk(cid)
                    if chunk:
                        summaries.append({
                            "chunk_id": cid,
                            "summary": chunk.summary or chunk.content[:80] + "...",
                            "tokens": chunk.tokens,
                            "source": chunk.source,
                            "shared": False,
                        })

            return summaries

    def inject_chunk_for_agent(self, agent_id: str, chunk_id: str) -> bool:
        """
        为 Agent 注入 chunk
        如果是共享 chunk，标记为已注入避免重复
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return False

            # 检查是否已在其他 Agent 中注入（共享优化）
            if chunk_id in self._shared_index:
                # 检查是否有其他 Agent 已注入
                for other_agent in self._agents.values():
                    if other_agent.agent_id != agent_id and chunk_id in other_agent.injected_chunks:
                        # 复用标记
                        agent.injected_chunks.add(chunk_id)
                        return True

            # 新注入
            agent.injected_chunks.add(chunk_id)
            return True

    def get_shared_injection(self, chunk_id: str) -> Optional[str]:
        """
        获取共享 chunk 的注入内容
        如果 chunk 已被其他 Agent 加载，返回复用标记
        """
        with self._lock:
            if chunk_id not in self._shared_index:
                return None

            # 检查是否已有 Agent 注入
            for agent in self._agents.values():
                if chunk_id in agent.injected_chunks:
                    chunk = self.store.get_chunk(chunk_id)
                    if chunk:
                        return (
                            f"[共享 Chunk (已由 {agent.agent_id} 加载): \"{chunk_id}\"]\n\n"
                            f"{chunk.content}\n\n"
                            f"[此 chunk 为共享组件，多个 Agent 共同使用]"
                        )

            return None

    def _get_role_specific_chunks(self, role: str) -> List[str]:
        """获取角色相关的 chunk

        使用 store.list_summaries() 而非直接访问 _chunks，
        确保线程安全和缓存一致性。
        """
        role_keywords = {
            "reviewer": ["code", "function", "class", "method"],
            "security": ["auth", "encrypt", "password", "token", "validate"],
            "architect": ["module", "service", "component", "interface"],
            "performance": ["cache", "query", "loop", "optimize"],
        }

        keywords = role_keywords.get(role, [])
        if not keywords:
            # 使用公共 API 获取所有 chunk ID
            summaries = self.store.list_summaries()
            return [s["chunk_id"] for s in summaries]

        matched = []
        summaries = self.store.list_summaries()
        for summary in summaries:
            cid = summary["chunk_id"]
            # 获取完整 chunk 进行内容匹配
            chunk = self.store.get_chunk(cid)
            if chunk:
                text = (chunk.summary + " " + chunk.content).lower()
                if any(kw in text for kw in keywords):
                    matched.append(cid)

        return matched

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """获取协作统计"""
        with self._lock:
            total_injections = sum(len(a.injected_chunks) for a in self._agents.values())
            shared_injections = sum(
                len(a.injected_chunks & self._shared_index)
                for a in self._agents.values()
            )

            return {
                "agents": len(self._agents),
                "shared_chunks": len(self._shared_index),
                "total_injections": total_injections,
                "shared_injections": shared_injections,
                "deduplication_rate": shared_injections / total_injections if total_injections > 0 else 0,
                "agent_details": {
                    aid: {
                        "role": a.role,
                        "injected": len(a.injected_chunks),
                        "shared": len(a.shared_chunks),
                    }
                    for aid, a in self._agents.items()
                },
            }


class MultiAgentLCM:
    """
    多 Agent LCM 协调器
    协调多个 Agent 的 LCM 会话，优化共享 chunk 的加载
    """

    def __init__(self, store: ChunkStoreV2):
        self.store = store
        self.index_manager = SharedIndexManager(store)
        self._agent_clients: Dict[str, Any] = {}

    def register_agent(self, agent_id: str, role: str, llm_client) -> None:
        """注册 Agent"""
        from .client import LCMClientV2

        self.index_manager.register_agent(agent_id, role)
        self._agent_clients[agent_id] = LCMClientV2(llm_client, self.store)

    def coordinate_review(
        self,
        user_query: str,
        agent_order: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        协调多 Agent 审查
        
        流程：
        1. 第一个 Agent 加载共享 chunk
        2. 后续 Agent 复用共享 chunk
        3. 每个 Agent 按需加载私有 chunk
        """
        results = {}
        agent_ids = agent_order or list(self._agent_clients.keys())

        for i, agent_id in enumerate(agent_ids):
            client = self._agent_clients.get(agent_id)
            if not client:
                continue

            # 构建 Agent 特定的消息
            messages = self._build_agent_messages(agent_id, user_query, i == 0)

            # 执行审查
            result = client.chat(user_query)
            results[agent_id] = result

        return results

    def _build_agent_messages(
        self,
        agent_id: str,
        user_query: str,
        is_first: bool,
    ) -> List[Dict[str, str]]:
        """构建 Agent 特定的消息"""
        from .prompt import build_initial_messages_v2

        # 获取 Agent 的索引
        index = self.index_manager.get_agent_index(agent_id)

        # 如果是第一个 Agent，加载共享 chunk
        if is_first:
            for item in index:
                if item.get("shared"):
                    self.index_manager.inject_chunk_for_agent(agent_id, item["chunk_id"])

        return build_initial_messages_v2(user_query, self.store)

    def get_stats(self) -> Dict[str, Any]:
        """获取多 Agent 统计"""
        return {
            "agents": len(self._agent_clients),
            "collaboration": self.index_manager.get_collaboration_stats(),
        }

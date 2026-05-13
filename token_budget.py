"""
LCM v2 Token 预算管理器
防止上下文窗口溢出，智能管理 chunk 注入
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .lcm_types import ContextChunk


@dataclass
class TokenBudget:
    """
    Token 预算管理器
    跟踪上下文窗口使用情况，防止溢出
    """
    max_tokens: int = 8192  # 默认 8K 上下文
    system_reserve: int = 500  # 系统消息保留
    safety_margin: int = 200  # 安全余量
    
    _used_tokens: int = field(default=0, repr=False)
    _injected_chunks: List[Tuple[str, int]] = field(default_factory=list, repr=False)
    _start_time: datetime = field(default_factory=datetime.now, repr=False)

    @property
    def available_tokens(self) -> int:
        """可用 token 数"""
        return self.max_tokens - self.system_reserve - self.safety_margin - self._used_tokens

    @property
    def utilization_rate(self) -> float:
        """利用率"""
        usable = self.max_tokens - self.system_reserve - self.safety_margin
        return self._used_tokens / usable if usable > 0 else 1.0

    @property
    def is_critical(self) -> bool:
        """是否接近临界值（利用率 > 80%）"""
        return self.utilization_rate > 0.8

    @property
    def is_exceeded(self) -> bool:
        """是否已超出预算"""
        return self.available_tokens <= 0

    def can_fit(self, tokens: int) -> bool:
        """检查是否还能容纳指定 token 数"""
        return self.available_tokens >= tokens

    def allocate(self, chunk: ContextChunk) -> bool:
        """
        尝试为 chunk 分配 token 预算
        返回是否成功
        """
        if self.can_fit(chunk.tokens):
            self._used_tokens += chunk.tokens
            self._injected_chunks.append((chunk.chunk_id, chunk.tokens))
            return True
        return False

    def allocate_tokens(self, tokens: int, chunk_id: str = "") -> bool:
        """
        直接分配指定数量的 token
        
        Args:
            tokens: 要分配的 token 数
            chunk_id: 关联的 chunk ID（可选）
        
        Returns:
            是否成功
        """
        if self.can_fit(tokens):
            self._used_tokens += tokens
            if chunk_id:
                self._injected_chunks.append((chunk_id, tokens))
            return True
        return False

    def deallocate(self, chunk_id: str) -> bool:
        """释放指定 chunk 的预算"""
        for i, (cid, tokens) in enumerate(self._injected_chunks):
            if cid == chunk_id:
                self._used_tokens -= tokens
                self._injected_chunks.pop(i)
                return True
        return False

    def deallocate_tokens(self, tokens: int) -> None:
        """直接释放指定数量的 token"""
        self._used_tokens = max(0, self._used_tokens - tokens)

    @property
    def used_tokens(self) -> int:
        """已使用的 token 数"""
        return self._used_tokens

    def get_injected_summary(self) -> Dict:
        """获取已注入 chunk 的摘要"""
        return {
            "total_injected": len(self._injected_chunks),
            "total_tokens": self._used_tokens,
            "available": self.available_tokens,
            "utilization": round(self.utilization_rate * 100, 2),
            "chunks": [{"id": cid, "tokens": t} for cid, t in self._injected_chunks],
        }

    def select_chunks_within_budget(
        self, chunks: List[ContextChunk], strategy: str = "priority"
    ) -> List[ContextChunk]:
        """
        从候选 chunks 中选择能放入预算的子集
        
        Args:
            chunks: 候选 chunk 列表
            strategy: 选择策略
                - "priority": 按优先级排序（高优先级优先）
                - "recent": 按最近使用排序
                - "small_first": 小 chunk 优先（最大化数量）
        
        Returns:
            能放入预算的 chunk 列表
        """
        if strategy == "priority":
            sorted_chunks = sorted(chunks, key=lambda c: c.priority, reverse=True)
        elif strategy == "recent":
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.last_loaded_at or datetime.min,
                reverse=True,
            )
        elif strategy == "small_first":
            sorted_chunks = sorted(chunks, key=lambda c: c.tokens)
        else:
            sorted_chunks = chunks

        selected = []
        temp_budget = TokenBudget(
            max_tokens=self.max_tokens,
            system_reserve=self.system_reserve,
            safety_margin=self.safety_margin,
        )
        temp_budget._used_tokens = self._used_tokens

        for chunk in sorted_chunks:
            if temp_budget.allocate(chunk):
                selected.append(chunk)
            else:
                break

        return selected

    def estimate_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """估算消息列表的 token 数"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += ContextChunk._estimate_tokens(content)
        return total

    def __repr__(self) -> str:
        return (
            f"TokenBudget("
            f"max={self.max_tokens}, "
            f"used={self._used_tokens}, "
            f"available={self.available_tokens}, "
            f"util={self.utilization_rate:.1%}"
            f")"
        )

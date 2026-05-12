"""
LCM v2 核心调度器
状态机 + 批量预取 + 重试 + 指标收集
"""
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Generator, Callable, Set

from .lcm_types import (
    ContextChunk, LoadRequest, LCMEvent, LCMSession,
    LCMState, ChunkLoadReason, LCMMetrics,
)
from .store import ChunkStoreV2
from .detector import SentinelDetectorV2
from .logger import get_logger, LCMErrorCode

logger = get_logger()


class LCMOrchestratorV2:
    """LCM 核心调度器 v2"""

    MAX_ROUNDS = 20
    DEFAULT_PREFETCH_K = 3

    def __init__(
        self,
        chunk_store: ChunkStoreV2,
        metrics: Optional[LCMMetrics] = None,
        max_rounds: int = 20,
        prefetch_enabled: bool = True,
        prefetch_k: int = 3,
        retry_attempts: int = 2,
    ):
        self.store = chunk_store
        self.detector = SentinelDetectorV2()
        self.metrics = metrics or LCMMetrics()
        self.max_rounds = max_rounds
        self.prefetch_enabled = prefetch_enabled
        self.prefetch_k = prefetch_k
        self.retry_attempts = retry_attempts
        self.session: Optional[LCMSession] = None
        self._on_event: Optional[Callable[[LCMEvent], None]] = None
        self._round = 0
        self._injected_chunks: Set[str] = set()

    def on_event(self, callback: Callable[[LCMEvent], None]):
        """注册事件回调"""
        self._on_event = callback

    def _emit(self, event_type: str, **kwargs):
        event = LCMEvent(
            event_type=event_type,
            **kwargs,
        )
        if self.session:
            self.session.load_history.append(event)
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def new_session(self, session_id: str = "") -> LCMSession:
        sid = session_id or f"lcm_{int(time.time() * 1000)}"
        self.session = LCMSession(session_id=sid, state=LCMState.IDLE)
        self.detector.reset()
        self._round = 0
        self._injected_chunks.clear()
        return self.session

    def run_stream(
        self,
        messages: List[Dict[str, str]],
        stream_fn: Callable[[List[Dict[str, str]]], Generator[str, None, None]],
        session_id: str = "",
    ) -> Generator[str, None, None]:
        """执行一次完整的 LCM 流式会话"""
        self.new_session(session_id)
        self._round = 0
        current_messages = list(messages)

        while self._round < self.max_rounds:
            self._round += 1

            if self._round == 1:
                self.session.state = LCMState.GENERATING
                self._emit("generation_started")

            self.detector.reset()
            full_response = ""
            requests: List[LoadRequest] = []

            try:
                for chunk_text in stream_fn(current_messages):
                    full_response += chunk_text
                    requests = self.detector.feed(chunk_text)

                    if requests:
                        # 关键修复：检测到哨兵后，立即停止消费流
                        # 但已经 yield 的文本需要清理哨兵后返回给用户
                        break
            except Exception as e:
                self.session.state = LCMState.ERROR
                self.session.end_time = datetime.now()
                logger.error("流式生成异常", code=LCMErrorCode.ORCH_STREAM_EXCEPTION, error=str(e), round=self._round)
                self._emit("error", metadata={"reason": "stream_exception", "error": str(e)})
                yield self.detector.get_clean_buffer()
                return

            if not requests:
                clean = self.detector.get_clean_buffer()
                self.session.state = LCMState.COMPLETED
                self.session.total_tokens_generated += ContextChunk._estimate_tokens(clean)
                self.session.end_time = datetime.now()
                self._emit("completed", metadata={"rounds": self._round})
                yield clean
                return

            # 修复：获取清理后的文本（移除哨兵标记）
            # 注意：detector.get_clean_buffer() 会移除所有哨兵标记
            clean_so_far = self.detector.get_clean_buffer()
            # 确保不会重复添加 assistant 消息
            if clean_so_far.strip():
                current_messages.append({"role": "assistant", "content": clean_so_far})

            seen_chunks = set()
            requested_ids = []
            for req in requests:
                cid = req.chunk_id
                if cid in seen_chunks:
                    continue
                seen_chunks.add(cid)

                if cid in self._injected_chunks:
                    current_messages.append({
                        "role": "system",
                        "content": f"[提醒] chunk '{cid}' 已在之前的轮次中注入，请基于已有内容继续，不要重复请求。"
                    })
                    self._emit("chunk_already_loaded", chunk_id=cid)
                    continue

                chunk = self._get_chunk_with_retry(cid)
                if not chunk:
                    self._emit("chunk_miss", chunk_id=cid)
                    continue

                requested_ids.append(cid)
                self._injected_chunks.add(cid)
                self.store.mark_loaded(cid, ChunkLoadReason.MODEL_REQUEST)
                self.session.total_chunks_loaded += 1
                self.session.state = LCMState.WAITING_CHUNK
                self._emit("chunk_requested", chunk_id=cid, position=req.position)

                injection = self._build_injection(chunk)
                self.session.state = LCMState.RESUMING
                self._emit("chunk_injected", chunk_id=cid, metadata={"tokens": chunk.tokens})
                current_messages.append({"role": "system", "content": injection})

            if self.prefetch_enabled and requested_ids and self.prefetch_k > 0:
                prefetch_ids = self.store.find_related_multi(requested_ids, top_k=self.prefetch_k)
                prefetched_count = 0
                for pid in prefetch_ids:
                    if pid in self._injected_chunks:
                        continue
                    pchunk = self._get_chunk_with_retry(pid)
                    if not pchunk:
                        continue
                    self._injected_chunks.add(pid)
                    self.store.mark_loaded(pid, ChunkLoadReason.SPECULATIVE_PREFETCH)
                    self.session.total_chunks_loaded += 1
                    prefetched_count += 1
                    p_injection = (
                        f"[预取 Chunk (可能相关): \"{pid}\"]\n\n"
                        f"{pchunk.content}\n\n"
                        f"[预取内容结束。以上为系统预测的可能相关模块。]"
                    )
                    self._emit("chunk_injected", chunk_id=pid,
                              metadata={"tokens": pchunk.tokens, "reason": "speculative_prefetch"})
                    current_messages.append({"role": "system", "content": p_injection})
                if prefetched_count > 0:
                    self._emit("prefetch_batch", metadata={
                        "requested": requested_ids, "prefetched": prefetch_ids[:prefetched_count],
                        "total_prefetched": prefetched_count,
                    })

            current_messages.append({
                "role": "system",
                "content": "[请直接继续。只输出新内容，不要重复上面已经写过的任何文字。]"
            })

        self.session.state = LCMState.ERROR
        self.session.end_time = datetime.now()
        logger.error("达到最大轮次限制", code=LCMErrorCode.ORCH_MAX_ROUNDS_EXCEEDED, max_rounds=self.max_rounds)
        self._emit("error", metadata={"reason": "max_rounds_exceeded"})
        yield self.detector.get_clean_buffer()

    def run_sync(
        self,
        messages: List[Dict[str, str]],
        stream_fn: Callable[[List[Dict[str, str]]], Generator[str, None, None]],
        session_id: str = "",
    ) -> str:
        """同步版本：收集所有输出为单个字符串"""
        result = []
        for chunk in self.run_stream(messages, stream_fn, session_id):
            result.append(chunk)
        return "".join(result)

    def _get_chunk_with_retry(self, chunk_id: str) -> Optional[ContextChunk]:
        """带重试的 chunk 加载（重试时释放锁避免阻塞）"""
        for attempt in range(self.retry_attempts):
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                return chunk
            if attempt < self.retry_attempts - 1:
                time.sleep(0.05 * (attempt + 1))
        return None

    def _build_injection(self, chunk: ContextChunk) -> str:
        return (
            f"[Chunk 内容: \"{chunk.chunk_id}\"]\n\n"
            f"{chunk.content}\n\n"
            f"[内容结束。当前任务：继续你刚才的分析，从最后一个字之后直接写。"
            f"禁止重复任何已经输出的内容。如需其他chunk，用 [NEED_CHUNK:id] 请求。]"
        )

    @property
    def state(self) -> LCMState:
        return self.session.state if self.session else LCMState.IDLE

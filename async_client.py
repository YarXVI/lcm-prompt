"""
LCM v2 异步客户端
支持异步 LLM 调用和并发 chunk 加载
"""
import asyncio
from typing import AsyncIterator, List, Dict, Any, Optional, Callable
from datetime import datetime

from .lcm_types import ContextChunk, LCMEvent, LCMSession, LCMState, LCMMetrics
from .store import ChunkStoreV2
from .orchestrator import LCMOrchestratorV2
from .logger import get_logger

logger = get_logger()


class AsyncLCMClient:
    """
    LCM 异步客户端
    
    特性：
    - 异步流式生成
    - 并发 chunk 加载
    - 异步事件回调
    """

    def __init__(self, store: ChunkStoreV2):
        self.store = store
        self.orchestrator = LCMOrchestratorV2(store)
        self._event_handlers: List[Callable] = []

    def on_event(self, handler: Callable):
        """注册事件处理器"""
        self._event_handlers.append(handler)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        stream_fn: Callable[[List[Dict[str, str]]], AsyncIterator[str]],
        session_id: str = "",
    ) -> AsyncIterator[str]:
        """
        异步生成
        
        Args:
            messages: 初始消息列表
            stream_fn: 异步流式函数，返回文本片段的异步迭代器
            session_id: 会话 ID
        
        Yields:
            生成的文本片段
        """
        session = self.orchestrator.new_session(session_id)
        current_messages = messages.copy()
        round_num = 0

        while round_num < self.orchestrator.max_rounds:
            round_num += 1
            logger.info("开始异步生成轮次", round=round_num, session=session.session_id)

            # 异步流式生成
            full_response = ""
            requests = []

            try:
                async for chunk_text in stream_fn(current_messages):
                    full_response += chunk_text
                    # 这里需要同步检测哨兵（detector 不是异步的）
                    requests = self.orchestrator.detector.feed(chunk_text)
                    yield chunk_text

                    if requests:
                        break
            except Exception as e:
                session.state = LCMState.ERROR
                session.end_time = datetime.now()
                logger.error("异步流异常", error=str(e), round=round_num)
                yield self.orchestrator.detector.get_clean_buffer()
                return

            if not requests:
                clean = self.orchestrator.detector.get_clean_buffer()
                session.state = LCMState.COMPLETED
                session.total_tokens_generated += ContextChunk._estimate_tokens(clean)
                session.end_time = datetime.now()
                logger.info("异步生成完成", rounds=round_num, session=session.session_id)
                yield clean
                return

            # 并发加载请求的 chunks
            clean_so_far = self.orchestrator.detector.get_clean_buffer()
            current_messages.append({"role": "assistant", "content": clean_so_far})

            # 并发加载
            loaded_chunks = await self._load_chunks_concurrent(requests)

            for chunk in loaded_chunks:
                if chunk:
                    current_messages.append({
                        "role": "user",
                        "content": f"[已加载 Chunk: {chunk.chunk_id}]\n{chunk.content}",
                    })
                    session.total_chunks_loaded += 1
                    self.orchestrator._emit("chunk_loaded", chunk_id=chunk.chunk_id)

        # 达到最大轮次
        session.state = LCMState.ERROR
        session.end_time = datetime.now()
        logger.error("达到最大异步轮次", max_rounds=self.orchestrator.max_rounds)
        yield self.orchestrator.detector.get_clean_buffer()

    async def _load_chunks_concurrent(self, requests: List[Any]) -> List[Optional[ContextChunk]]:
        """并发加载 chunks"""
        tasks = [self._load_chunk_async(req.chunk_id) for req in requests]
        return await asyncio.gather(*tasks)

    async def _load_chunk_async(self, chunk_id: str) -> Optional[ContextChunk]:
        """异步加载单个 chunk（在线程池中执行）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.store.get_chunk, chunk_id)

    def get_session(self) -> Optional[LCMSession]:
        """获取当前会话"""
        return self.orchestrator.session


class AsyncLLMWrapper:
    """
    异步 LLM 包装器
    将同步 LLM 调用包装为异步流
    """

    def __init__(self, sync_stream_fn: Callable):
        self.sync_stream_fn = sync_stream_fn

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """异步包装同步流"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行同步流
        def _sync_generator():
            for chunk in self.sync_stream_fn(messages):
                yield chunk

        # 使用 asyncio.to_thread（Python 3.9+）或 run_in_executor
        import sys
        if sys.version_info >= (3, 9):
            for chunk in self.sync_stream_fn(messages):
                yield await asyncio.to_thread(lambda: chunk)
        else:
            for chunk in self.sync_stream_fn(messages):
                yield chunk
                await asyncio.sleep(0)  # 让出控制权

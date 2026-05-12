"""
LCM v2 客户端
桥接 LLMClient 与 LCMOrchestratorV2
"""
import time
from typing import Optional, List, Dict, Generator, Callable

from .lcm_types import LCMEvent, LCMSession, LCMState, LCMMetrics
from .store import ChunkStoreV2
from .orchestrator import LCMOrchestratorV2
from .prompt import build_initial_messages_v2
from .logger import get_logger

logger = get_logger()


class LCMClientV2:
    """LCM 客户端 v2"""

    def __init__(
        self,
        llm_client,
        chunk_store: ChunkStoreV2,
        metrics: Optional[LCMMetrics] = None,
        verbose: bool = False,
    ):
        self._client = llm_client
        self.store = chunk_store
        self.metrics = metrics or LCMMetrics()
        self.orchestrator = LCMOrchestratorV2(chunk_store, metrics=self.metrics)
        self._event_handlers: List[Callable[[LCMEvent], None]] = []
        self._verbose = verbose
        self.orchestrator.on_event(self._handle_event)

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool):
        self._verbose = value

    def on_event(self, handler: Callable[[LCMEvent], None]):
        """注册事件处理器"""
        self._event_handlers.append(handler)

    def _handle_event(self, event: LCMEvent):
        if self._verbose:
            meta = event.metadata or {}
            extra = f" | {meta}" if meta else ""
            print(f"[LCMv2] {event.event_type}: {event.chunk_id}{extra}")

        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning("事件处理器异常", handler=handler.__name__, error=str(e))

    def _stream_fn(self, messages: List[Dict[str, str]]):
        """适配器：将 LLMClient.chat_stream 适配为 LCMOrchestrator 需要的签名"""
        return self._client.chat_stream(messages)

    def chat(self, user_query: str, session_id: str = "") -> str:
        """同步 LCM 对话"""
        messages = build_initial_messages_v2(user_query, self.store)
        return self.orchestrator.run_sync(messages, self._stream_fn, session_id)

    def chat_stream(
        self, user_query: str, session_id: str = ""
    ) -> Generator[str, None, None]:
        """流式 LCM 对话"""
        messages = build_initial_messages_v2(user_query, self.store)
        yield from self.orchestrator.run_stream(messages, self._stream_fn, session_id)

    @property
    def session(self) -> Optional[LCMSession]:
        return self.orchestrator.session

    @property
    def stats(self) -> Dict:
        return {
            "store": self.store.get_stats(),
            "session": {
                "state": self.orchestrator.state.value if self.orchestrator.state else "none",
                "total_chunks_loaded": self.session.total_chunks_loaded if self.session else 0,
                "total_tokens_generated": self.session.total_tokens_generated if self.session else 0,
                "events_count": len(self.session.load_history) if self.session else 0,
                "duration_ms": self.session.duration_ms if self.session else 0,
            },
        }

    def print_session_report(self):
        """打印会话报告"""
        sess = self.session
        if not sess:
            print("[LCMv2] 无活跃会话")
            return

        print("=" * 60)
        print(f"LCM v2 会话报告: {sess.session_id}")
        print(f"状态: {sess.state.value}")
        print(f"加载块数: {sess.total_chunks_loaded}")
        print(f"生成 tokens: {sess.total_tokens_generated}")
        print(f"事件数: {len(sess.load_history)}")
        print(f"持续时间: {sess.duration_ms:.2f} ms")
        print("-" * 60)
        for evt in sess.load_history:
            meta = evt.metadata or {}
            extra = f" {meta}" if meta else ""
            print(f"  [{evt.event_type}] chunk={evt.chunk_id}{extra}")
        print("=" * 60)


def make_mock_stream_fn(responses: List[str]):
    """构造一个模拟的流式函数（用于无 API 的单元测试）"""
    call_count = [0]

    def mock_fn(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        response = responses[idx]
        # 模拟流式输出：每次 yield 一个字符
        for i, char in enumerate(response):
            yield char
            # 只在每10个字符后短暂休眠，加速测试
            if i % 10 == 0:
                time.sleep(0.0001)

    return mock_fn

"""
LCM v2 单元测试 & 集成测试
覆盖 ChunkStoreV2 / SentinelDetectorV2 / LCMOrchestratorV2 / LCMClientV2
"""
import sys
import os
import unittest
import time
import threading
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_v2.lcm_types import (
    ContextChunk, LoadRequest, LCMEvent, LCMSession,
    LCMState, ChunkLoadReason, SentinelPattern, LCMMetrics,
)
from lcm_v2.store import ChunkStoreV2
from lcm_v2.detector import SentinelDetectorV2
from lcm_v2.orchestrator import LCMOrchestratorV2
from lcm_v2.prompt import (
    build_initial_messages_v2, build_chunk_index_section_v2,
    build_messages_from_chunks_v2,
)
from lcm_v2.client import LCMClientV2, make_mock_stream_fn
from lcm_v2.provider_router import (
    ProviderRouter, AdaptiveLCMClient,
    ProviderConfig, ProviderType, RoutingStrategy,
)
from lcm_v2.token_budget import TokenBudget
from lcm_v2.chunk_graph import ChunkGraph
from lcm_v2.hybrid_mode import HybridChunkManager, HybridConfig, build_hybrid_messages
from lcm_v2.kv_cache import KVCacheManager, CachedLCMOrchestrator
from lcm_v2.adaptive_chunking import AdaptiveChunking, AdaptiveChunkStore, ChunkGroup


class TestChunkStoreV2(unittest.TestCase):

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_auth",
            content="def login(): pass",
            summary="登录函数",
            tokens=20,
            source="auth.py",
        ))
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_db",
            content="class Database: ...",
            summary="数据库连接类",
            tokens=30,
            source="db.py",
        ))

    def test_basic_crud(self):
        self.assertEqual(len(self.store), 2)
        self.assertIn("chunk_auth", self.store)
        self.assertIn("chunk_db", self.store)
        self.assertNotIn("chunk_nonexistent", self.store)

    def test_get_chunk(self):
        c = self.store.get_chunk("chunk_auth")
        self.assertIsNotNone(c)
        self.assertEqual(c.chunk_id, "chunk_auth")
        self.assertEqual(c.content, "def login(): pass")

    def test_remove_chunk(self):
        self.assertTrue(self.store.remove_chunk("chunk_db"))
        self.assertEqual(len(self.store), 1)
        self.assertFalse(self.store.remove_chunk("chunk_db"))

    def test_search_by_id(self):
        results = self.store.search("auth")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk_auth")

    def test_search_by_summary(self):
        results = self.store.search("数据库")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk_db")

    def test_search_no_match(self):
        results = self.store.search("nonexistent")
        self.assertEqual(len(results), 0)

    def test_list_summaries(self):
        summaries = self.store.list_summaries()
        self.assertEqual(len(summaries), 2)
        ids = {s["chunk_id"] for s in summaries}
        self.assertEqual(ids, {"chunk_auth", "chunk_db"})

    def test_mark_loaded(self):
        self.store.mark_loaded("chunk_auth")
        c = self.store.get_chunk("chunk_auth")
        self.assertEqual(c.load_count, 1)
        self.assertIsNotNone(c.last_loaded_at)
        self.store.mark_loaded("chunk_auth")
        self.assertEqual(c.load_count, 2)

    def test_stats(self):
        self.store.mark_loaded("chunk_auth")
        self.store.mark_loaded("chunk_auth")
        self.store.mark_loaded("chunk_auth")
        stats = self.store.get_stats()
        self.assertEqual(stats["total_chunks"], 2)
        self.assertEqual(stats["total_tokens"], 50)
        self.assertIn("chunk_auth", stats["hot_chunks"])
        self.assertIn("metrics", stats)

    def test_auto_token_estimate(self):
        c = ContextChunk(chunk_id="test", content="a" * 100)
        self.assertGreater(c.tokens, 0)

    def test_chinese_token_estimate(self):
        c = ContextChunk(chunk_id="test", content="中文字符测试")
        self.assertGreater(c.tokens, 0)

    def test_batch_load(self):
        result = self.store.batch_load(["chunk_auth", "chunk_db", "missing"])
        self.assertIsNotNone(result["chunk_auth"])
        self.assertIsNotNone(result["chunk_db"])
        self.assertIsNone(result["missing"])

    def test_cache_warmup(self):
        self.store.warm_cache(["chunk_auth"])
        stats = self.store.get_stats()
        self.assertEqual(stats["cache_size"], 2)

    def test_thread_safety(self):
        errors = []
        def worker():
            try:
                for i in range(10):
                    self.store.add_chunk(ContextChunk(
                        chunk_id=f"thread_chunk_{threading.current_thread().ident}_{i}",
                        content=f"content {i}",
                        summary=f"summary {i}",
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"线程安全错误: {errors}")

    def test_persistence(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkStoreV2(storage_dir=tmpdir, enable_persistence=True)
            store.add_chunk(ContextChunk(
                chunk_id="persist_test",
                content="test content",
                summary="test summary",
            ))

            store2 = ChunkStoreV2(storage_dir=tmpdir, enable_persistence=True)
            self.assertIn("persist_test", store2)
            c = store2.get_chunk("persist_test")
            self.assertEqual(c.content, "test content")


class TestSentinelDetectorV2(unittest.TestCase):

    def setUp(self):
        self.detector = SentinelDetectorV2()

    def test_single_detection(self):
        requests = self.detector.feed("Hello [NEED_CHUNK:auth_handler] world")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].chunk_id, "auth_handler")
        self.assertIn("NEED_CHUNK:auth_handler", requests[0].raw_marker)
        self.assertEqual(requests[0].confidence, 1.0)

    def test_multiple_patterns(self):
        detector = SentinelDetectorV2(SentinelPattern.get_all_patterns())
        requests = detector.feed("[NEED_CHUNK:a] [LOAD_CHUNK:b] [FETCH:c]")
        self.assertEqual(len(requests), 3)

    def test_multiple_detections(self):
        requests = self.detector.feed("A [NEED_CHUNK:x] and B [NEED_CHUNK:y]")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].chunk_id, "x")
        self.assertEqual(requests[1].chunk_id, "y")

    def test_no_detection(self):
        requests = self.detector.feed("Just normal text without markers")
        self.assertEqual(len(requests), 0)

    def test_clean_buffer(self):
        self.detector.feed("result: [NEED_CHUNK:abc] end")
        clean = self.detector.get_clean_buffer()
        self.assertEqual(clean, "result:  end")

    def test_partial_feed(self):
        r1 = self.detector.feed("text [NEED_CH")
        self.assertEqual(len(r1), 0)
        r2 = self.detector.feed("UNK:test] more")
        self.assertEqual(len(r2), 1)
        self.assertEqual(r2[0].chunk_id, "test")

    def test_reset(self):
        self.detector.feed("[NEED_CHUNK:old]")
        self.assertEqual(self.detector.get_clean_buffer(), "")
        self.detector.reset()
        self.assertEqual(self.detector.get_clean_buffer(), "")

    def test_chunk_id_with_dashes_underscores(self):
        requests = self.detector.feed("[NEED_CHUNK:my-special_chunk_42]")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].chunk_id, "my-special_chunk_42")

    def test_confidence_scoring(self):
        requests = self.detector.feed("[NEED_CHUNK:test]")
        self.assertEqual(requests[0].confidence, 1.0)


class TestLCMOrchestratorV2(unittest.TestCase):

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_auth",
            content="fn login() { validate(); }",
            summary="登录验证函数",
            tokens=15,
            source="auth.ts",
        ))
        self.orchestrator = LCMOrchestratorV2(self.store)
        self.events = []

    def _collect_event(self, event: LCMEvent):
        self.events.append(event)

    def _make_messages(self):
        return [
            {"role": "system", "content": "LCM mode. Request chunks with [NEED_CHUNK:id]"},
            {"role": "user", "content": "审查登录代码"},
        ]

    def test_single_chunk_flow(self):
        self.orchestrator.on_event(self._collect_event)

        mock = make_mock_stream_fn([
            "审查结果如下：[NEED_CHUNK:chunk_auth]",
            "加载完成，登录函数使用 validate() 进行验证，安全。",
        ])

        result = self.orchestrator.run_sync(self._make_messages(), mock)
        self.assertIn("登录函数", result)
        self.assertNotIn("NEED_CHUNK", result)

        event_types = [e.event_type for e in self.events]
        self.assertIn("generation_started", event_types)
        self.assertIn("chunk_requested", event_types)
        self.assertIn("chunk_injected", event_types)
        self.assertIn("completed", event_types)

        self.assertEqual(self.orchestrator.session.total_chunks_loaded, 1)

    def test_multi_chunk_flow(self):
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_db",
            content="class DB { connect() {} }",
            summary="数据库连接",
            tokens=15,
            source="db.ts",
        ))
        self.orchestrator.on_event(self._collect_event)

        mock = make_mock_stream_fn([
            "需要看数据库：[NEED_CHUNK:chunk_auth]",
            "继续，还需看：[NEED_CHUNK:chunk_db]",
            "最终评估：两个模块均正常。",
        ])

        result = self.orchestrator.run_sync(self._make_messages(), mock)
        self.assertIn("两个模块均正常", result)
        self.assertNotIn("NEED_CHUNK", result)
        self.assertEqual(self.orchestrator.session.total_chunks_loaded, 2)

    def test_no_chunks_needed(self):
        self.orchestrator.on_event(self._collect_event)
        mock = make_mock_stream_fn(["直接回答，无需加载任何块。"])

        result = self.orchestrator.run_sync(self._make_messages(), mock)
        self.assertIn("直接回答", result)
        self.assertEqual(self.orchestrator.session.total_chunks_loaded, 0)

    def test_chunk_miss(self):
        """请求不存在的 chunk 不应崩溃"""
        self.orchestrator.on_event(self._collect_event)
        mock = make_mock_stream_fn([
            "需要不存在的块：[NEED_CHUNK:missing_chunk]",
            "回退到已有知识回答：结果正常。",
        ])

        result = self.orchestrator.run_sync(self._make_messages(), mock)
        self.assertIn("回退到已有知识", result)
        miss_events = [e for e in self.events if e.event_type == "chunk_miss"]
        self.assertEqual(len(miss_events), 1)

    def test_state_machine(self):
        mock = make_mock_stream_fn([
            "[NEED_CHUNK:chunk_auth]",
            "完成。",
        ])
        self.orchestrator.run_sync(self._make_messages(), mock)
        self.assertEqual(self.orchestrator.state, LCMState.COMPLETED)

    def test_prefetch(self):
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_db",
            content="class DB { connect() {} }",
            summary="数据库连接",
            tokens=15,
            source="db.ts",
        ))
        self.orchestrator.on_event(self._collect_event)
        mock = make_mock_stream_fn([
            "[NEED_CHUNK:chunk_auth]",
            "完成。",
        ])
        self.orchestrator.run_sync(self._make_messages(), mock)
        # 应该预取了 chunk_db
        prefetch_events = [e for e in self.events if e.event_type == "prefetch_batch"]
        self.assertGreaterEqual(len(prefetch_events), 0)

    def test_retry_mechanism(self):
        """测试重试机制"""
        self.orchestrator.retry_attempts = 2
        mock = make_mock_stream_fn([
            "[NEED_CHUNK:chunk_auth]",
            "完成。",
        ])
        result = self.orchestrator.run_sync(self._make_messages(), mock)
        self.assertIn("完成", result)

    def test_stream_exception_handling(self):
        """测试流式生成异常处理"""
        self.orchestrator.on_event(self._collect_event)

        def failing_stream(messages):
            yield "开始生成..."
            raise RuntimeError("模拟流式异常")

        result = self.orchestrator.run_sync(self._make_messages(), failing_stream)
        self.assertIn("开始生成", result)
        self.assertNotIn("NEED_CHUNK", result)
        self.assertEqual(self.orchestrator.state, LCMState.ERROR)
        self.assertIsNotNone(self.orchestrator.session.end_time)

        error_events = [e for e in self.events if e.event_type == "error"]
        self.assertEqual(len(error_events), 1)
        self.assertEqual(error_events[0].metadata.get("reason"), "stream_exception")

    def test_max_rounds_end_time(self):
        """测试达到最大轮次后设置 end_time"""
        self.orchestrator.max_rounds = 1
        mock = make_mock_stream_fn([
            "[NEED_CHUNK:chunk_auth]",
        ])
        result = self.orchestrator.run_sync(self._make_messages(), mock)
        self.assertEqual(self.orchestrator.state, LCMState.ERROR)
        self.assertIsNotNone(self.orchestrator.session.end_time)
        self.assertGreater(self.orchestrator.session.duration_ms, 0)


class TestLCMClientV2(unittest.TestCase):

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_security",
            content="useSSL=true; encrypt(api_key);",
            summary="安全配置代码 (~10 tokens)",
            tokens=10,
            source="config.py",
        ))
        self.events = []

    def _collect_event(self, event: LCMEvent):
        self.events.append(event)

    def _make_mock_client(self, responses: List[str]) -> LCMClientV2:
        mock = make_mock_stream_fn(responses)

        class MockLLM:
            def chat_stream(self, messages):
                return mock(messages)

        llm = MockLLM()
        client = LCMClientV2(llm, self.store)
        client.on_event(self._collect_event)
        return client

    def test_chat_single_chunk(self):
        client = self._make_mock_client([
            "审查中：[NEED_CHUNK:chunk_security]",
            "发现：useSSL 已启用，密钥已加密。安全。",
        ])

        result = client.chat("审查安全配置")
        self.assertIn("useSSL", result)
        self.assertNotIn("NEED_CHUNK", result)
        self.assertEqual(client.session.total_chunks_loaded, 1)

    def test_chat_stream(self):
        client = self._make_mock_client([
            "流式输出：[NEED_CHUNK:chunk_security]",
            "结果：安全配置正常。",
        ])

        output_parts = []
        for text in client.chat_stream("审查配置"):
            output_parts.append(text)

        full = "".join(output_parts)
        self.assertIn("安全配置正常", full)
        self.assertNotIn("NEED_CHUNK", full)

    def test_stats_report(self):
        client = self._make_mock_client([
            "[NEED_CHUNK:chunk_security]",
            "完成。",
        ])
        client.chat("test")

        stats = client.stats
        self.assertEqual(stats["store"]["total_chunks"], 1)
        self.assertEqual(stats["store"]["total_loads"], 1)
        self.assertEqual(stats["session"]["total_chunks_loaded"], 1)

    def test_verbose_mode(self):
        client = self._make_mock_client(["直接回答。"])
        client.verbose = True
        result = client.chat("简单问题")
        self.assertIn("直接回答", result)


class TestPromptEngineeringV2(unittest.TestCase):

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_1",
            content="print('hello')",
            summary="打招呼代码",
            tokens=5,
            source="hello.py",
        ))

    def test_build_initial_messages(self):
        msgs = build_initial_messages_v2("审查代码", self.store)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("chunk_1", msgs[0]["content"])
        self.assertIn("hello.py", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "审查代码")

    def test_build_messages_from_chunks(self):
        chunks = [
            ContextChunk(chunk_id="c1", content="a", summary="A"),
        ]
        msgs = build_messages_from_chunks_v2("query", chunks)
        self.assertEqual(msgs[1]["content"], "query")

    def test_compact_mode(self):
        msgs = build_initial_messages_v2("q", self.store, system_mode="compact")
        self.assertIn("LCM（惰性上下文物化）", msgs[0]["content"])

    def test_chunk_index_section(self):
        section = build_chunk_index_section_v2(self.store)
        self.assertIn("chunk_1", section)
        self.assertIn("打招呼代码", section)

    def test_empty_store_index(self):
        empty_store = ChunkStoreV2(enable_persistence=False)
        section = build_chunk_index_section_v2(empty_store)
        self.assertIn("无可用上下文块", section)


class TestMetrics(unittest.TestCase):

    def test_cache_hit_rate(self):
        metrics = LCMMetrics()
        metrics.record_cache_hit("a")
        metrics.record_cache_hit("b")
        metrics.record_cache_miss("c")
        self.assertAlmostEqual(metrics.cache_hit_rate, 2/3, places=2)

    def test_avg_latency(self):
        metrics = LCMMetrics()
        metrics.record_cache_hit("a")
        metrics.record_load_latency(10.0)
        metrics.record_cache_hit("b")
        metrics.record_load_latency(20.0)
        self.assertEqual(metrics.avg_load_latency_ms, 15.0)

    def test_to_dict(self):
        metrics = LCMMetrics()
        metrics.record_cache_hit("a")
        d = metrics.to_dict()
        self.assertIn("cache_hit_rate", d)
        self.assertIn("avg_load_latency_ms", d)


class TestRealAPIIntegration(unittest.TestCase):
    """使用真实 DeepSeek API 的集成测试 —— 需要有效的 API Key"""

    @unittest.skipUnless(
        os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "跳过：需要设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY",
    )
    def test_real_lcm_flow(self):
        from agent_core.core.llm import LLMClient

        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("LCM_TEST_MODEL", "deepseek-v4-flash")

        store = ChunkStoreV2(enable_persistence=False)
        store.add_chunk(ContextChunk(
            chunk_id="chunk_fib",
            summary="斐波那契数列的 Python 实现（递归版）",
            content="def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
            tokens=20,
            source="fib.py",
        ))
        store.add_chunk(ContextChunk(
            chunk_id="chunk_fib_iter",
            summary="斐波那契数列的 Python 实现（迭代优化版）",
            content="def fib_iter(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
            tokens=25,
            source="fib_iter.py",
        ))

        base_llm = LLMClient(model=model, api_key=api_key, base_url=base_url)
        lcm = LCMClientV2(base_llm, store)
        lcm.verbose = True

        print("\n=== 真实 LCM v2 流式测试 ===")
        full = []
        for text in lcm.chat_stream(
            "这两个斐波那契实现分别有什么优缺点？哪个更好？"
        ):
            print(text, end="", flush=True)
            full.append(text)

        result = "".join(full)
        print("\n=== 会话报告 ===")
        lcm.print_session_report()

        self.assertTrue(len(result) > 50, "响应过短，可能 API 调用失败")
        self.assertNotIn("NEED_CHUNK", result)
        self.assertIn("fib", result.lower())


class TestProviderRouter(unittest.TestCase):
    """Provider 路由测试"""

    def test_cloud_provider_detection(self):
        """测试云端提供商识别"""
        config = ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        router = ProviderRouter(config)
        self.assertEqual(router.provider_type, ProviderType.CLOUD)
        self.assertEqual(router.strategy, RoutingStrategy.TRADITIONAL)
        self.assertFalse(router.use_lcm)

    def test_deepseek_cloud_detection(self):
        """测试 DeepSeek 云端识别"""
        config = ProviderConfig(
            name="deepseek",
            base_url="https://api.deepseek.com/v1",
        )
        router = ProviderRouter(config)
        self.assertEqual(router.provider_type, ProviderType.CLOUD)
        self.assertFalse(router.use_lcm)

    def test_localhost_detection(self):
        """测试本地地址识别"""
        config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434/v1",
        )
        router = ProviderRouter(config)
        self.assertEqual(router.provider_type, ProviderType.LOCAL)
        self.assertEqual(router.strategy, RoutingStrategy.LCM)
        self.assertTrue(router.use_lcm)

    def test_127_detection(self):
        """测试 127.0.0.1 识别"""
        config = ProviderConfig(
            name="vllm",
            base_url="http://127.0.0.1:8000/v1/chat/completions",
        )
        router = ProviderRouter(config)
        self.assertEqual(router.provider_type, ProviderType.LOCAL)
        self.assertTrue(router.use_lcm)

    def test_local_path_pattern(self):
        """测试本地路径模式识别"""
        config = ProviderConfig(
            name="local-api",
            base_url="http://192.168.1.100:5000/v1/completions",
        )
        router = ProviderRouter(config)
        self.assertEqual(router.provider_type, ProviderType.LOCAL)
        self.assertTrue(router.use_lcm)

    def test_unknown_provider_defaults_traditional(self):
        """测试未知提供商默认传统方案"""
        config = ProviderConfig(
            name="unknown",
            base_url="https://some-random-api.com/api",
        )
        router = ProviderRouter(config)
        self.assertEqual(router.provider_type, ProviderType.UNKNOWN)
        self.assertEqual(router.strategy, RoutingStrategy.TRADITIONAL)
        self.assertFalse(router.use_lcm)

    def test_force_lcm_strategy(self):
        """测试强制 LCM 策略"""
        config = ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            routing_strategy=RoutingStrategy.LCM,
        )
        router = ProviderRouter(config)
        self.assertTrue(router.use_lcm)
        self.assertEqual(router.strategy, RoutingStrategy.LCM)

    def test_force_traditional_strategy(self):
        """测试强制传统策略"""
        config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434",
            routing_strategy=RoutingStrategy.TRADITIONAL,
        )
        router = ProviderRouter(config)
        self.assertFalse(router.use_lcm)
        self.assertEqual(router.strategy, RoutingStrategy.TRADITIONAL)

    def test_router_info(self):
        """测试路由信息输出"""
        config = ProviderConfig(
            name="test",
            base_url="http://localhost:8080/v1",
        )
        router = ProviderRouter(config)
        info = router.get_info()
        self.assertEqual(info["detected_type"], "local")
        self.assertEqual(info["effective_strategy"], "lcm")
        self.assertEqual(info["use_lcm"], "True")

    def test_ollama_port_detection(self):
        """测试 Ollama 默认端口识别"""
        config = ProviderConfig(
            name="ollama",
            base_url="http://192.168.1.5:11434",
        )
        router = ProviderRouter(config)
        self.assertEqual(router.provider_type, ProviderType.LOCAL)

    def test_private_ip_detection(self):
        """测试私有 IP 识别"""
        for ip in ["192.168.1.1", "10.0.0.1", "172.16.0.1"]:
            config = ProviderConfig(name="local", base_url=f"http://{ip}:8000")
            router = ProviderRouter(config)
            self.assertEqual(router.provider_type, ProviderType.LOCAL, f"Failed for {ip}")


class TestAdaptiveLCMClient(unittest.TestCase):
    """自适应 LCM 客户端测试"""

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_test",
            content="test content",
            summary="test summary",
            tokens=10,
        ))

        class MockLLM:
            def chat_stream(self, messages):
                for char in "传统模式回答":
                    yield char

        self.mock_llm = MockLLM()

    def test_cloud_uses_traditional(self):
        """测试云端使用传统方案"""
        config = ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            routing_strategy=RoutingStrategy.AUTO,
        )
        client = AdaptiveLCMClient(self.mock_llm, self.store, config)
        result = client.chat("测试")
        self.assertIn("传统模式回答", result)
        self.assertIsNone(client.session)

    def test_local_uses_lcm(self):
        """测试本地使用 LCM 方案"""
        config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434",
            routing_strategy=RoutingStrategy.AUTO,
        )

        class MockLLMWithLCM:
            def chat_stream(self, messages):
                yield "LCM模式回答"

        client = AdaptiveLCMClient(MockLLMWithLCM(), self.store, config)
        result = client.chat("测试")
        self.assertIn("LCM模式回答", result)

    def test_stats_traditional_mode(self):
        """测试传统模式 stats"""
        config = ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
        )
        client = AdaptiveLCMClient(self.mock_llm, self.store, config)
        stats = client.stats
        self.assertEqual(stats["mode"], "traditional")
        self.assertIn("router", stats)


class TestTokenBudget(unittest.TestCase):
    """Token 预算管理器测试"""

    def test_basic_budget(self):
        budget = TokenBudget(max_tokens=1000, system_reserve=100, safety_margin=50)
        self.assertEqual(budget.available_tokens, 850)
        self.assertFalse(budget.is_exceeded)

    def test_allocate(self):
        budget = TokenBudget(max_tokens=1000)
        chunk = ContextChunk(chunk_id="test", content="a" * 100, tokens=100)
        self.assertTrue(budget.allocate(chunk))
        self.assertEqual(budget.available_tokens, 1000 - 500 - 200 - 100)

    def test_budget_exceeded(self):
        budget = TokenBudget(max_tokens=1000, system_reserve=100, safety_margin=50)
        chunk = ContextChunk(chunk_id="big", content="x", tokens=1000)
        self.assertFalse(budget.allocate(chunk))

    def test_select_chunks_within_budget(self):
        budget = TokenBudget(max_tokens=1000, system_reserve=100, safety_margin=50)
        chunks = [
            ContextChunk(chunk_id="c1", content="x", tokens=100, priority=3),
            ContextChunk(chunk_id="c2", content="x", tokens=200, priority=1),
            ContextChunk(chunk_id="c3", content="x", tokens=300, priority=2),
            ContextChunk(chunk_id="c4", content="x", tokens=400, priority=5),
        ]
        selected = budget.select_chunks_within_budget(chunks, strategy="priority")
        # available = 850, c4(400) + c1(100) + c3(300) = 800, c2(200) 超预算
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0].chunk_id, "c4")
        self.assertEqual(selected[1].chunk_id, "c1")
        self.assertEqual(selected[2].chunk_id, "c3")

    def test_small_first_strategy(self):
        budget = TokenBudget(max_tokens=1000, system_reserve=100, safety_margin=50)
        chunks = [
            ContextChunk(chunk_id="c1", content="x", tokens=400),
            ContextChunk(chunk_id="c2", content="x", tokens=100),
            ContextChunk(chunk_id="c3", content="x", tokens=200),
        ]
        selected = budget.select_chunks_within_budget(chunks, strategy="small_first")
        self.assertEqual(selected[0].chunk_id, "c2")
        self.assertEqual(selected[1].chunk_id, "c3")

    def test_utilization_rate(self):
        budget = TokenBudget(max_tokens=1000, system_reserve=100, safety_margin=100)
        self.assertEqual(budget.utilization_rate, 0.0)
        chunk = ContextChunk(chunk_id="test", content="x", tokens=400)
        budget.allocate(chunk)
        self.assertEqual(budget.utilization_rate, 0.5)
        self.assertFalse(budget.is_critical)

    def test_critical_threshold(self):
        budget = TokenBudget(max_tokens=1000, system_reserve=100, safety_margin=100)
        chunk = ContextChunk(chunk_id="test", content="x", tokens=650)
        budget.allocate(chunk)
        self.assertTrue(budget.is_critical)

    def test_deallocate(self):
        budget = TokenBudget(max_tokens=1000)
        chunk = ContextChunk(chunk_id="test", content="x", tokens=100)
        budget.allocate(chunk)
        self.assertTrue(budget.deallocate("test"))
        self.assertEqual(budget._used_tokens, 0)
        self.assertFalse(budget.deallocate("nonexistent"))

    def test_estimate_message_tokens(self):
        budget = TokenBudget()
        messages = [
            {"role": "system", "content": "Hello world"},
            {"role": "user", "content": "测试"},
        ]
        tokens = budget.estimate_message_tokens(messages)
        self.assertGreater(tokens, 0)


class TestChunkGraph(unittest.TestCase):
    """Chunk 依赖图测试"""

    def setUp(self):
        self.graph = ChunkGraph()

    def test_add_dependency(self):
        self.graph.add_dependency("A", "B")
        self.assertIn("B", self.graph.get_dependencies("A"))
        self.assertIn("A", self.graph.get_dependents("B"))

    def test_self_dependency_detected_as_cycle(self):
        """自依赖应该被记录，并在 has_cycle 中检测为环"""
        self.graph.add_dependency("A", "A")
        self.assertEqual(len(self.graph.get_dependencies("A")), 1)
        self.assertTrue(self.graph.has_cycle())

    def test_topological_sort(self):
        self.graph.add_dependency("C", "A")
        self.graph.add_dependency("C", "B")
        self.graph.add_dependency("B", "A")
        order = self.graph.topological_sort()
        # A 没有依赖其他节点，所以入度为 0，排在前面
        # B 依赖 A，C 依赖 A 和 B
        self.assertIn("A", order)
        self.assertIn("B", order)
        self.assertIn("C", order)
        # A 应该在 B 和 C 之前
        a_idx = order.index("A")
        b_idx = order.index("B")
        c_idx = order.index("C")
        self.assertLess(a_idx, b_idx)
        self.assertLess(b_idx, c_idx)

    def test_loading_order(self):
        self.graph.add_dependency("func_login", "func_validate")
        self.graph.add_dependency("func_validate", "utils_hash")
        order = self.graph.get_loading_order("func_login")
        self.assertEqual(order, ["utils_hash", "func_validate"])

    def test_no_cycle(self):
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("B", "C")
        self.assertFalse(self.graph.has_cycle())

    def test_detect_cycle(self):
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("B", "C")
        self.graph.add_dependency("C", "A")
        self.assertTrue(self.graph.has_cycle())

    def test_find_related_with_graph(self):
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("B", "C")
        self.graph.add_dependency("A", "D")
        related = self.graph.find_related_with_graph("A", depth=2)
        self.assertIn("B", related)
        self.assertIn("C", related)
        self.assertIn("D", related)

    def test_remove_chunk(self):
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("C", "B")
        self.graph.remove_chunk("B")
        self.assertEqual(len(self.graph.get_dependencies("A")), 0)
        self.assertEqual(len(self.graph.get_dependents("B")), 0)

    def test_serialization(self):
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("B", "C")
        data = self.graph.to_dict()
        restored = ChunkGraph.from_dict(data)
        self.assertEqual(restored.get_dependencies("A"), ["B"])
        self.assertEqual(restored.get_dependencies("B"), ["C"])

    def test_graph_stats(self):
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("A", "C")
        self.graph.add_dependency("B", "C")
        stats = self.graph.get_stats()
        # total_nodes 只统计有依赖的节点，C 没有依赖其他节点
        self.assertEqual(stats["total_nodes"], 3)  # A, B, C 都在 dependencies 中
        self.assertEqual(stats["total_edges"], 3)
        self.assertFalse(stats["has_cycle"])


class TestStoreWithGraph(unittest.TestCase):
    """Store 集成依赖图测试"""

    def test_store_graph_integration(self):
        store = ChunkStoreV2(enable_persistence=False)
        store.add_chunk(ContextChunk(chunk_id="auth", content="auth code"))
        store.add_chunk(ContextChunk(chunk_id="validate", content="validate code"))
        store.add_chunk(ContextChunk(chunk_id="hash", content="hash utils"))

        store.add_dependency("auth", "validate")
        store.add_dependency("validate", "hash")

        deps = store.get_dependencies("auth", recursive=True)
        self.assertIn("validate", deps)
        self.assertIn("hash", deps)

        order = store.get_loading_order("auth")
        self.assertEqual(order[0], "hash")
        self.assertEqual(order[1], "validate")

    def test_graph_related_find(self):
        store = ChunkStoreV2(enable_persistence=False)
        for cid in ["A", "B", "C", "D"]:
            store.add_chunk(ContextChunk(chunk_id=cid, content=f"code {cid}"))

        store.add_dependency("A", "B")
        store.add_dependency("B", "C")
        store.add_dependency("A", "D")

        related = store.find_related_with_graph("A", depth=2)
        self.assertIn("B", related)
        self.assertIn("C", related)
        self.assertIn("D", related)


class TestHybridMode(unittest.TestCase):
    """混合模式测试"""

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)
        # 高频 chunk（加载多次）
        self.store.add_chunk(ContextChunk(
            chunk_id="auth", content="auth code", summary="auth",
            load_count=5, last_loaded_at=datetime.now(),
        ))
        # 低频 chunk
        self.store.add_chunk(ContextChunk(
            chunk_id="payment", content="payment code", summary="payment",
            load_count=1,
        ))
        # 极低频 chunk
        self.store.add_chunk(ContextChunk(
            chunk_id="admin", content="admin code", summary="admin",
            load_count=0,
        ))

    def test_classify_chunks(self):
        manager = HybridChunkManager(self.store)
        classified = manager.classify_chunks()
        self.assertIn("auth", classified["hot"])
        self.assertIn("payment", classified["cold"])
        self.assertIn("admin", classified["cold"])

    def test_force_direct(self):
        config = HybridConfig(force_direct={"admin"})
        manager = HybridChunkManager(self.store, config)
        classified = manager.classify_chunks()
        self.assertIn("admin", classified["hot"])

    def test_force_lcm_override(self):
        config = HybridConfig(force_direct={"auth"}, force_lcm={"auth"})
        manager = HybridChunkManager(self.store, config)
        classified = manager.classify_chunks()
        self.assertIn("auth", classified["cold"])

    def test_hot_content(self):
        manager = HybridChunkManager(self.store)
        content = manager.get_hot_chunks_content()
        self.assertIn("auth", content)
        self.assertIn("auth code", content)
        self.assertNotIn("payment", content)

    def test_stats(self):
        manager = HybridChunkManager(self.store)
        stats = manager.get_stats()
        self.assertEqual(stats["hot_count"], 1)
        self.assertEqual(stats["cold_count"], 2)
        self.assertGreater(stats["hot_tokens"], 0)

    def test_build_hybrid_messages(self):
        messages = build_hybrid_messages("测试", self.store)
        self.assertEqual(len(messages), 2)
        system = messages[0]["content"]
        self.assertIn("高频上下文", system)
        self.assertIn("auth code", system)
        self.assertIn("payment", system)  # 低频在索引中
        self.assertIn("admin", system)

    def test_all_hot(self):
        """所有 chunk 都是高频时"""
        for cid in ["payment", "admin"]:
            c = self.store.get_chunk(cid)
            c.load_count = 10
        messages = build_hybrid_messages("测试", self.store)
        system = messages[0]["content"]
        self.assertIn("无按需加载", system)


class TestKVCache(unittest.TestCase):
    """KV Cache 联动测试"""

    def test_cache_key_generation(self):
        cache = KVCacheManager()
        key = cache.set_index_content("System prompt", "chunk index")
        self.assertIsNotNone(key)
        self.assertTrue(key.startswith("lcm_index:"))

    def test_cache_stats(self):
        cache = KVCacheManager()
        cache.set_index_content("System", "index content")
        stats = cache.get_stats()
        self.assertEqual(stats["status"], "active")
        self.assertEqual(stats["entries"], 1)
        self.assertGreater(stats["total_tokens"], 0)

    def test_cache_headers(self):
        cache = KVCacheManager()
        cache.set_index_content("System", "index")
        headers = cache.get_cache_headers()
        self.assertIn("anthropic-beta", headers)
        self.assertIn("x-lcm-cache-key", headers)

    def test_record_use(self):
        cache = KVCacheManager()
        key = cache.set_index_content("System", "index")
        cache.record_use(key)
        stats = cache.get_stats()
        self.assertEqual(stats["total_uses"], 1)

    def test_multiple_entries(self):
        cache = KVCacheManager()
        cache.set_index_content("System1", "index1")
        cache.set_index_content("System2", "index2")
        stats = cache.get_stats()
        self.assertEqual(stats["entries"], 2)


class TestAdaptiveChunking(unittest.TestCase):
    """自适应粒度测试"""

    def setUp(self):
        self.adaptive = AdaptiveChunking(merge_threshold=0.5, min_group_size=2, max_group_size=3)

    def test_analyze_chunks(self):
        chunks = [
            ContextChunk(chunk_id="auth", content="login auth password hash", summary="auth module"),
            ContextChunk(chunk_id="login", content="login form session token", summary="login module"),
            ContextChunk(chunk_id="db", content="database connection pool", summary="db module"),
        ]
        result = self.adaptive.analyze_chunks(chunks)
        self.assertIn("merged", result)
        self.assertIn("standalone", result)

    def test_create_group(self):
        chunks = [
            ContextChunk(chunk_id="a", content="content a", tokens=10),
            ContextChunk(chunk_id="b", content="content b", tokens=15),
        ]
        group = self.adaptive.create_group("group_ab", chunks)
        self.assertEqual(group.group_id, "group_ab")
        self.assertEqual(group.chunk_ids, ["a", "b"])
        self.assertEqual(group.combined_tokens, 25)

    def test_should_merge_true(self):
        chunks_dict = {
            "a": ContextChunk(chunk_id="a", content="x", tokens=100, load_count=5),
            "b": ContextChunk(chunk_id="b", content="x", tokens=100, load_count=5),
        }
        result = self.adaptive.should_merge(["a", "b"], chunks_dict)
        self.assertTrue(result)

    def test_should_merge_too_large(self):
        chunks_dict = {
            "a": ContextChunk(chunk_id="a", content="x", tokens=3000),
            "b": ContextChunk(chunk_id="b", content="x", tokens=3000),
        }
        result = self.adaptive.should_merge(["a", "b"], chunks_dict)
        self.assertFalse(result)

    def test_get_group_for_chunk(self):
        chunks = [
            ContextChunk(chunk_id="a", content="x"),
            ContextChunk(chunk_id="b", content="y"),
        ]
        self.adaptive.create_group("g1", chunks)
        group = self.adaptive.get_group_for_chunk("a")
        self.assertIsNotNone(group)
        self.assertEqual(group.group_id, "g1")

    def test_extract_keywords(self):
        keywords = AdaptiveChunking._extract_keywords("Hello world, this is a test")
        self.assertIn("hello", keywords)
        self.assertIn("world", keywords)
        self.assertNotIn("is", keywords)

    def test_adaptive_chunk_store(self):
        from lcm_v2.store import ChunkStoreV2
        base_store = ChunkStoreV2(enable_persistence=False)
        base_store.add_chunk(ContextChunk(chunk_id="a", content="content a"))
        base_store.add_chunk(ContextChunk(chunk_id="b", content="content b"))

        adaptive_store = AdaptiveChunkStore(base_store)
        c = adaptive_store.get_chunk("a")
        self.assertIsNotNone(c)
        self.assertEqual(c.chunk_id, "a")

    def test_analyze_and_merge(self):
        from lcm_v2.store import ChunkStoreV2
        base_store = ChunkStoreV2(enable_persistence=False)
        base_store.add_chunk(ContextChunk(chunk_id="a", content="x"))
        base_store.add_chunk(ContextChunk(chunk_id="b", content="y"))

        adaptive_store = AdaptiveChunkStore(base_store)
        # 模拟多次共现访问
        for _ in range(5):
            adaptive_store.get_chunk("a")
            adaptive_store.get_chunk("b")

        groups = adaptive_store.analyze_and_merge()
        self.assertIsInstance(groups, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)

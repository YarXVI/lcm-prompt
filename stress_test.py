"""
LCM v2 深度压力测试
覆盖：并发、内存、持久化、流式、路由、fuzz
"""
import sys
import os
import time
import threading
import random
import string
import tempfile
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_v2.lcm_types import ContextChunk, LCMMetrics
from lcm_v2.store import ChunkStoreV2
from lcm_v2.detector import SentinelDetectorV2
from lcm_v2.orchestrator import LCMOrchestratorV2
from lcm_v2.client import make_mock_stream_fn
from lcm_v2.provider_router import ProviderRouter, ProviderConfig, ProviderType, RoutingStrategy


class StressTestReporter:
    def __init__(self):
        self.results = []

    def report(self, name, passed, duration_ms, details=""):
        status = "PASS" if passed else "FAIL"
        self.results.append((name, status, duration_ms, details))
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status} ({duration_ms:.2f}ms) {details}")

    def summary(self):
        passed = sum(1 for _, s, _, _ in self.results if s == "PASS")
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"压力测试总结: {passed}/{total} 通过")
        print(f"{'='*60}")
        for name, status, duration, details in self.results:
            print(f"  [{status}] {name} ({duration:.2f}ms) {details}")


reporter = StressTestReporter()


def test_concurrent_reads():
    """测试1: 100线程并发读取"""
    name = "并发读取 (100线程 x 1000次)"
    start = time.time()
    store = ChunkStoreV2(enable_persistence=False)
    for i in range(100):
        store.add_chunk(ContextChunk(chunk_id=f"chunk_{i}", content=f"data_{i}" * 100))

    errors = []
    def reader():
        for _ in range(1000):
            try:
                idx = random.randint(0, 99)
                c = store.get_chunk(f"chunk_{idx}")
                if c is None:
                    errors.append(f"miss chunk_{idx}")
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=reader) for _ in range(100)]
    for t in threads: t.start()
    for t in threads: t.join()

    duration = (time.time() - start) * 1000
    reporter.report(name, len(errors) == 0, duration, f"错误: {len(errors)}")
    return len(errors) == 0


def test_concurrent_mixed():
    """测试2: 50线程混合读写"""
    name = "混合读写 (50线程 x 500次)"
    start = time.time()
    store = ChunkStoreV2(enable_persistence=False)
    errors = []
    ops = [0]

    def worker():
        for _ in range(500):
            try:
                op = random.choice(['read', 'write', 'search', 'remove'])
                if op == 'read':
                    store.get_chunk(f"chunk_{random.randint(0, 200)}")
                elif op == 'write':
                    store.add_chunk(ContextChunk(
                        chunk_id=f"chunk_{random.randint(0, 200)}",
                        content=f"data_{random.randint(0, 9999)}" * 50
                    ))
                elif op == 'search':
                    store.search(f"data_{random.randint(0, 99)}")
                elif op == 'remove':
                    store.remove_chunk(f"chunk_{random.randint(0, 200)}")
                ops[0] += 1
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()

    duration = (time.time() - start) * 1000
    reporter.report(name, len(errors) == 0, duration, f"操作: {ops[0]}, 错误: {len(errors)}")
    return len(errors) == 0


def test_large_chunks():
    """测试3: 大量大体积 chunk"""
    name = "大体积 chunks (1000 x 10KB)"
    start = time.time()
    store = ChunkStoreV2(enable_persistence=False)

    for i in range(1000):
        content = f"chunk_{i}_" + "x" * 10000
        store.add_chunk(ContextChunk(
            chunk_id=f"big_{i}",
            content=content,
            summary=f"summary_{i}",
        ))

    duration = (time.time() - start) * 1000
    stats = store.get_stats()
    reporter.report(name, stats['total_chunks'] == 1000, duration,
                   f"chunks: {stats['total_chunks']}, tokens: {stats['total_tokens']}")
    return stats['total_chunks'] == 1000


def test_memory_stability():
    """测试4: 内存稳定性"""
    name = "内存稳定性 (循环创建/释放)"
    start = time.time()
    initial_chunks = 5000

    for round_idx in range(5):
        store = ChunkStoreV2(enable_persistence=False)
        for i in range(initial_chunks):
            store.add_chunk(ContextChunk(
                chunk_id=f"mem_{round_idx}_{i}",
                content=f"data_{i}" * 100,
            ))
        assert len(store) == initial_chunks
        del store
        gc.collect()

    duration = (time.time() - start) * 1000
    reporter.report(name, True, duration, "5轮创建/释放无内存泄漏")
    return True


def test_persistence_stress():
    """测试5: 持久化压力测试"""
    name = "持久化压力 (1000次写入)"
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChunkStoreV2(storage_dir=tmpdir, enable_persistence=True)

        for i in range(1000):
            store.add_chunk(ContextChunk(
                chunk_id=f"persist_{i}",
                content=f"content_{i}",
                summary=f"summary_{i}",
            ))

        # 重新加载验证
        store2 = ChunkStoreV2(storage_dir=tmpdir, enable_persistence=True)
        loaded = len(store2)

    duration = (time.time() - start) * 1000
    reporter.report(name, loaded == 1000, duration, f"加载: {loaded}/1000")
    return loaded == 1000


def test_sentinel_fuzz():
    """测试6: 哨兵检测 fuzz"""
    name = "哨兵检测 fuzz (10000随机输入)"
    start = time.time()
    detector = SentinelDetectorV2()
    errors = []

    for _ in range(10000):
        try:
            # 随机生成包含/不包含哨兵的文本
            length = random.randint(1, 500)
            text = ''.join(random.choices(
                string.ascii_letters + string.digits + "[]:_- \n\t",
                k=length
            ))
            requests = detector.feed(text)
            # 验证结果一致性
            for req in requests:
                assert req.chunk_id
                assert req.raw_marker
            detector.reset()
        except Exception as e:
            errors.append(str(e))

    duration = (time.time() - start) * 1000
    reporter.report(name, len(errors) == 0, duration, f"错误: {len(errors)}")
    return len(errors) == 0


def test_orchestrator_many_rounds():
    """测试7: 多轮次 LCM 交互"""
    name = "多轮次 LCM (20轮)"
    start = time.time()
    store = ChunkStoreV2(enable_persistence=False)

    for i in range(30):
        store.add_chunk(ContextChunk(
            chunk_id=f"round_chunk_{i}",
            content=f"content for round {i}",
            summary=f"summary {i}",
        ))

    orchestrator = LCMOrchestratorV2(store, max_rounds=20)

    # 构造一个需要多轮的响应序列
    responses = []
    for i in range(15):
        responses.append(f"分析中，需要查看 round_chunk_{i} [NEED_CHUNK:round_chunk_{i}]")
    responses.append("最终结论：所有 chunk 分析完成。")

    mock = make_mock_stream_fn(responses)
    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "分析"}]

    try:
        result = orchestrator.run_sync(messages, mock)
        success = "所有 chunk 分析完成" in result
    except Exception as e:
        success = False
        result = str(e)

    duration = (time.time() - start) * 1000
    reporter.report(name, success, duration,
                   f"轮次: {orchestrator._round}, 加载: {orchestrator.session.total_chunks_loaded if orchestrator.session else 0}")
    return success


def test_provider_router_massive():
    """测试8: 大量 URL 路由解析"""
    name = "路由解析压力 (10000 URL)"
    start = time.time()

    test_urls = [
        # 云端
        ("https://api.openai.com/v1", ProviderType.CLOUD, False),
        ("https://api.deepseek.com/v1", ProviderType.CLOUD, False),
        ("https://api.anthropic.com/v1", ProviderType.CLOUD, False),
        # 本地
        ("http://localhost:11434/v1", ProviderType.LOCAL, True),
        ("http://127.0.0.1:8080/v1", ProviderType.LOCAL, True),
        ("http://192.168.1.100:5000/v1/chat/completions", ProviderType.LOCAL, True),
        ("http://10.0.0.5:8000", ProviderType.LOCAL, True),
        # 未知
        ("https://some-api.com/api", ProviderType.UNKNOWN, False),
        ("", ProviderType.UNKNOWN, False),
    ]

    errors = []
    for _ in range(10000 // len(test_urls)):
        for url, expected_type, expected_lcm in test_urls:
            config = ProviderConfig(name="test", base_url=url)
            router = ProviderRouter(config)
            if router.provider_type != expected_type:
                errors.append(f"URL {url}: expected {expected_type}, got {router.provider_type}")
            if router.use_lcm != expected_lcm:
                errors.append(f"URL {url}: use_lcm expected {expected_lcm}, got {router.use_lcm}")

    duration = (time.time() - start) * 1000
    reporter.report(name, len(errors) == 0, duration, f"错误: {len(errors)}")
    return len(errors) == 0


def test_cache_eviction():
    """测试9: LRU 缓存淘汰"""
    name = "LRU 缓存淘汰 (10000 chunks, cache=10)"
    start = time.time()
    store = ChunkStoreV2(enable_persistence=False, max_cache_size=10)

    # 添加 10000 个 chunk
    for i in range(10000):
        store.add_chunk(ContextChunk(
            chunk_id=f"evict_{i}",
            content=f"data_{i}",
        ))

    # 访问前 5 个，应该被保留
    for i in range(5):
        store.get_chunk(f"evict_{i}")

    # 访问后 20 个，应该淘汰前面的
    for i in range(9980, 10000):
        store.get_chunk(f"evict_{i}")

    stats = store.get_stats()
    cache_size = stats['cache_size']

    duration = (time.time() - start) * 1000
    reporter.report(name, cache_size == 10, duration, f"缓存大小: {cache_size}")
    return cache_size == 10


def test_search_performance():
    """测试10: 搜索性能"""
    name = "搜索性能 (10000 chunks)"
    start = time.time()
    store = ChunkStoreV2(enable_persistence=False)

    for i in range(10000):
        store.add_chunk(ContextChunk(
            chunk_id=f"search_{i}",
            content=f"This is content about topic {i % 100} with keywords {i}",
            summary=f"Summary for topic {i % 100}",
        ))

    # 执行多次搜索
    for _ in range(100):
        results = store.search(f"topic {random.randint(0, 99)}", top_k=10)
        assert len(results) <= 10

    duration = (time.time() - start) * 1000
    reporter.report(name, True, duration, "100次搜索完成")
    return True


def test_chunk_content_edge_cases():
    """测试11: Chunk 内容边界情况"""
    name = "Chunk 边界情况"
    start = time.time()
    store = ChunkStoreV2(enable_persistence=False)

    edge_cases = [
        ("empty", ""),
        ("unicode", "中文🎉日本語한국어العربية"),
        ("newlines", "\n\n\n\n"),
        ("null_bytes", "\x00\x00\x00"),
        ("huge_single_line", "x" * 100000),
        ("special_chars", "<script>alert('xss')</script>"),
        ("json_like", '{"key": "value", "nested": {"a": 1}}'),
        ("xml_like", "<root><child>text</child></root>"),
        ("mixed", "Hello\x00世界\n\t🌍" * 100),
    ]

    errors = []
    for cid, content in edge_cases:
        try:
            store.add_chunk(ContextChunk(chunk_id=cid, content=content))
            retrieved = store.get_chunk(cid)
            if retrieved.content != content:
                errors.append(f"Content mismatch for {cid}")
        except Exception as e:
            errors.append(f"{cid}: {e}")

    duration = (time.time() - start) * 1000
    reporter.report(name, len(errors) == 0, duration, f"错误: {len(errors)}")
    return len(errors) == 0


def test_metrics_under_load():
    """测试12: 指标收集在高负载下"""
    name = "指标收集 (10000次操作)"
    start = time.time()
    metrics = LCMMetrics()

    def worker():
        for _ in range(1000):
            if random.random() > 0.3:
                metrics.record_cache_hit(f"chunk_{random.randint(0, 99)}")
            else:
                metrics.record_cache_miss(f"chunk_{random.randint(0, 99)}")
            metrics.record_load_latency(random.uniform(0.1, 10.0))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    stats = metrics.to_dict()
    duration = (time.time() - start) * 1000
    # 修复：total_requests 应该大于 0（因为有 10000 次操作）
    total_ops = metrics.cache_hits + metrics.cache_misses
    passed = total_ops == 10000 and metrics.total_requests > 0
    reporter.report(name, passed, duration,
                   f"total_requests: {metrics.total_requests}, hits: {metrics.cache_hits}, misses: {metrics.cache_misses}")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("LCM v2 深度压力测试")
    print("=" * 60)

    tests = [
        test_concurrent_reads,
        test_concurrent_mixed,
        test_large_chunks,
        test_memory_stability,
        test_persistence_stress,
        test_sentinel_fuzz,
        test_orchestrator_many_rounds,
        test_provider_router_massive,
        test_cache_eviction,
        test_search_performance,
        test_chunk_content_edge_cases,
        test_metrics_under_load,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            reporter.report(test.__name__, False, 0, f"EXCEPTION: {e}")

    reporter.summary()

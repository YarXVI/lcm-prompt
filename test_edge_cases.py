"""
LCM v2 边界和并发测试
测试极端场景、并发操作和持久化恢复
"""
import threading
import time
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from lcm_v2.lcm_types import ContextChunk, LCMMetrics
from lcm_v2.store import ChunkStoreV2
from lcm_v2.chunk_graph import ChunkGraph
from lcm_v2.detector import SentinelDetectorV2
from lcm_v2.token_budget import TokenBudget


class TestReporter:
    def __init__(self):
        self.results = []

    def report(self, name: str, passed: bool, duration_ms: float, detail: str = ""):
        status = "✓ PASS" if passed else "✗ FAIL"
        self.results.append((name, passed, duration_ms, detail))
        print(f"  [{status}] {name} ({duration_ms:.1f}ms) {detail}")

    def summary(self):
        passed = sum(1 for _, p, _, _ in self.results if p)
        total = len(self.results)
        print(f"\n{'='*50}")
        print(f"边界测试: {passed}/{total} 通过")
        print(f"{'='*50}")
        return passed == total


reporter = TestReporter()


def assert_true(condition, message):
    if not condition:
        raise AssertionError(f"ASSERTION FAILED: {message}")


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"ASSERTION FAILED: {message} | expected={expected}, actual={actual}")


def assert_greater(actual, threshold, message):
    if actual <= threshold:
        raise AssertionError(f"ASSERTION FAILED: {message} | expected > {threshold}, actual={actual}")


# ========== 并发测试 ==========

def test_concurrent_add():
    """测试并发添加 chunk"""
    name = "并发添加 chunk"
    start = time.time()

    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    errors = []
    added_ids = set()

    def add_chunk(i):
        try:
            chunk_id = f"chunk_{i}"
            store.add_chunk(ContextChunk(
                chunk_id=chunk_id,
                content=f"Content {i}",
                summary=f"Summary {i}",
            ))
            added_ids.add(chunk_id)
        except Exception as e:
            errors.append(str(e))

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(add_chunk, range(100))

    duration = (time.time() - start) * 1000
    passed = len(errors) == 0 and len(store) == 100
    reporter.report(name, passed, duration, f"errors={len(errors)}, chunks={len(store)}")
    return passed


def test_concurrent_read_write():
    """测试并发读写"""
    name = "并发读写"
    start = time.time()

    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    store.add_chunk(ContextChunk(
        chunk_id="shared",
        content="initial",
        summary="Shared chunk",
    ))

    errors = []
    read_results = []

    def writer():
        for i in range(50):
            try:
                store.add_chunk(ContextChunk(
                    chunk_id=f"writer_{i}",
                    content=f"Content {i}",
                    summary=f"Summary {i}",
                ))
            except Exception as e:
                errors.append(f"write: {e}")

    def reader():
        for _ in range(50):
            try:
                chunk = store.get_chunk("shared")
                if chunk:
                    read_results.append(chunk.content)
            except Exception as e:
                errors.append(f"read: {e}")

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=writer),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    duration = (time.time() - start) * 1000
    passed = len(errors) == 0 and len(read_results) > 0
    reporter.report(name, passed, duration, f"errors={len(errors)}, reads={len(read_results)}")
    return passed


def test_concurrent_persistence():
    """测试并发持久化"""
    name = "并发持久化"
    start = time.time()

    real_dir = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2_test_concurrent")
    if real_dir.exists():
        shutil.rmtree(real_dir)

    store = ChunkStoreV2(storage_dir=real_dir, enable_persistence=True)
    errors = []

    def add_and_save(i):
        try:
            store.add_chunk(ContextChunk(
                chunk_id=f"persist_{i}",
                content=f"Persistent content {i}",
                summary=f"Summary {i}",
            ))
            store._save_to_disk()
        except Exception as e:
            errors.append(str(e))

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(add_and_save, range(20))

    # 验证持久化完整性
    store2 = ChunkStoreV2(storage_dir=real_dir, enable_persistence=True)
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0 and len(store2) == 20
    reporter.report(name, passed, duration, f"errors={len(errors)}, loaded={len(store2)}")

    if real_dir.exists():
        shutil.rmtree(real_dir)
    return passed


# ========== 边界测试 ==========

def test_empty_chunk():
    """测试空 chunk"""
    name = "空 chunk"
    start = time.time()

    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    store.add_chunk(ContextChunk(
        chunk_id="empty",
        content="",
        summary="",
    ))

    chunk = store.get_chunk("empty")
    duration = (time.time() - start) * 1000
    passed = chunk is not None and chunk.tokens == 0 and chunk.content == ""
    reporter.report(name, passed, duration, f"tokens={chunk.tokens if chunk else 'N/A'}")
    return passed


def test_very_large_chunk():
    """测试超大 chunk (100KB)"""
    name = "超大 chunk (100KB)"
    start = time.time()

    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    large_content = "A" * 100_000  # 100KB

    store.add_chunk(ContextChunk(
        chunk_id="large",
        content=large_content,
        summary="Large chunk",
    ))

    chunk = store.get_chunk("large")
    duration = (time.time() - start) * 1000
    passed = chunk is not None and len(chunk.content) == 100_000 and chunk.tokens > 0
    reporter.report(name, passed, duration, f"size={len(large_content)}, tokens={chunk.tokens if chunk else 'N/A'}")
    return passed


def test_special_characters():
    """测试特殊字符 chunk"""
    name = "特殊字符 chunk"
    start = time.time()

    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    special_content = "特殊字符: 中文 🎉 emoji \n\t \\ \" ' [NEED_CHUNK:test] \x00\x01"

    store.add_chunk(ContextChunk(
        chunk_id="special",
        content=special_content,
        summary="Special chars",
    ))

    chunk = store.get_chunk("special")
    duration = (time.time() - start) * 1000
    passed = chunk is not None and chunk.content == special_content
    reporter.report(name, passed, duration)
    return passed


def test_persistence_corruption_recovery():
    """测试持久化文件损坏后的恢复"""
    name = "持久化损坏恢复"
    start = time.time()

    real_dir = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2_test_corrupt")
    if real_dir.exists():
        shutil.rmtree(real_dir)

    # 创建正常数据
    store = ChunkStoreV2(storage_dir=real_dir, enable_persistence=True)
    store.add_chunk(ContextChunk(
        chunk_id="good",
        content="Good content",
        summary="Good chunk",
    ))
    store._save_to_disk()

    # 损坏文件
    chunks_file = real_dir / "chunks.jsonl"
    with open(chunks_file, "a", encoding="utf-8") as f:
        f.write("\nthis is not valid json\n")

    # 尝试加载
    try:
        store2 = ChunkStoreV2(storage_dir=real_dir, enable_persistence=True)
        loaded = len(store2)
        # 应该能加载有效的部分
        passed = loaded >= 1
    except Exception as e:
        # 如果抛出异常，说明恢复机制不够健壮
        passed = False
        loaded = f"exception: {e}"

    duration = (time.time() - start) * 1000
    reporter.report(name, passed, duration, f"loaded={loaded}")

    if real_dir.exists():
        shutil.rmtree(real_dir)
    return passed


def test_detector_buffer_overflow():
    """测试检测器缓冲区溢出处理"""
    name = "检测器缓冲区溢出"
    start = time.time()

    detector = SentinelDetectorV2()
    # 发送超过缓冲区大小的文本
    huge_text = "A" * 5000 + "[NEED_CHUNK:overflow]" + "B" * 5000

    requests = detector.feed(huge_text)
    clean = detector.get_clean_buffer()

    duration = (time.time() - start) * 1000
    # 应该能检测到哨兵，且清理后的文本不包含哨兵
    has_request = len(requests) > 0
    clean_ok = "[NEED_CHUNK" not in clean
    passed = has_request and clean_ok
    reporter.report(name, passed, duration, f"requests={len(requests)}, clean_ok={clean_ok}")
    return passed


def test_token_budget_exhaustion():
    """测试 Token 预算耗尽"""
    name = "Token 预算耗尽"
    start = time.time()

    budget = TokenBudget(max_tokens=100, system_reserve=0, safety_margin=0)
    chunks = [
        ContextChunk(chunk_id="c1", content="A" * 50, summary="", tokens=50),
        ContextChunk(chunk_id="c2", content="B" * 50, summary="", tokens=50),
        ContextChunk(chunk_id="c3", content="C" * 50, summary="", tokens=50),
    ]

    selected = budget.select_chunks_within_budget(chunks)
    duration = (time.time() - start) * 1000
    # 应该只选择前两个（100 tokens）
    passed = len(selected) == 2 and selected[0].chunk_id == "c1" and selected[1].chunk_id == "c2"
    reporter.report(name, passed, duration, f"selected={len(selected)}")
    return passed


def test_chunk_graph_cycle():
    """测试 chunk 图环检测"""
    name = "Chunk 图环检测"
    start = time.time()

    graph = ChunkGraph()
    graph.add_dependency("A", "B")
    graph.add_dependency("B", "C")
    graph.add_dependency("C", "A")  # 形成环

    has_cycle = graph.has_cycle()
    try:
        order = graph.topological_sort()
        # 有环时应该返回部分排序 + 警告
        passed = has_cycle and len(order) == 3
    except Exception:
        passed = False

    duration = (time.time() - start) * 1000
    reporter.report(name, passed, duration, f"has_cycle={has_cycle}")
    return passed


def test_memory_leak():
    """测试长时间运行内存泄漏"""
    name = "内存泄漏检查"
    start = time.time()

    store = ChunkStoreV2(storage_dir=None, enable_persistence=False, max_cache_size=10)

    # 添加大量 chunk，超过缓存大小
    for i in range(100):
        store.add_chunk(ContextChunk(
            chunk_id=f"leak_{i}",
            content=f"Content {i}",
            summary=f"Summary {i}",
        ))

    # 缓存应该只保留最近 10 个
    cache_size = len(store._cache)
    chunks_size = len(store._chunks)

    duration = (time.time() - start) * 1000
    passed = cache_size <= 10 and chunks_size == 100
    reporter.report(name, passed, duration, f"cache={cache_size}, chunks={chunks_size}")
    return passed


# ========== 主测试运行器 ==========

def run_all_tests():
    print("=" * 60)
    print("LCM v2 边界和并发测试")
    print("=" * 60)

    tests = [
        test_concurrent_add,
        test_concurrent_read_write,
        test_concurrent_persistence,
        test_empty_chunk,
        test_very_large_chunk,
        test_special_characters,
        test_persistence_corruption_recovery,
        test_detector_buffer_overflow,
        test_token_budget_exhaustion,
        test_chunk_graph_cycle,
        test_memory_leak,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"  [✗ FAIL] {test.__name__} - 异常: {e}")

    print(f"\n{'='*60}")
    print(f"总计: {passed} 通过, {failed} 失败 / {len(tests)} 测试")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

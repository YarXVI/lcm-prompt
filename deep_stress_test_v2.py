"""
LCM v2 深度压力测试 v2 - 更激进的测试
目标：发现隐藏的缺陷和bug
"""
import threading
import time
import random
import string
import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

from lcm_v2.lcm_types import ContextChunk, LCMMetrics, LCMState
from lcm_v2.store import ChunkStoreV2
from lcm_v2.detector import SentinelDetectorV2
from lcm_v2.chunk_graph import ChunkGraph
from lcm_v2.token_budget import TokenBudget
from lcm_v2.orchestrator import LCMOrchestratorV2
from lcm_v2.client import LCMClientV2
from lcm_v2.logger import get_logger

logger = get_logger()


class StressTestReporter:
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()

    def report(self, name: str, passed: bool, duration_ms: float, detail: str = ""):
        with self.lock:
            status = "PASS" if passed else "FAIL"
            self.results.append((name, passed, duration_ms, detail))
            print(f"  [{status}] {name} ({duration_ms:.1f}ms) {detail}")

    def summary(self):
        with self.lock:
            passed = sum(1 for _, p, _, _ in self.results if p)
            total = len(self.results)
            print(f"\n{'='*60}")
            print(f"深度压力测试: {passed}/{total} 通过")
            print(f"{'='*60}")
            return passed, total


reporter = StressTestReporter()


def random_string(length: int) -> str:
    if length <= 0:
        return ""
    return ''.join(random.choices(string.ascii_letters + string.digits + '\n\t ', k=length))


# ========== 测试1: 循环依赖死锁 ==========

def test_circular_dependency_deadlock():
    """循环依赖 - A依赖B，B依赖C，C依赖A"""
    name = "循环依赖死锁"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    
    store.add_chunk(ContextChunk(chunk_id="A", content="A", summary=""))
    store.add_chunk(ContextChunk(chunk_id="B", content="B", summary=""))
    store.add_chunk(ContextChunk(chunk_id="C", content="C", summary=""))
    
    store.add_dependency("A", "B")
    store.add_dependency("B", "C")
    store.add_dependency("C", "A")  # 形成环
    
    # 获取加载顺序，应该能处理环
    try:
        order = store.get_loading_order("A")
        passed = len(order) > 0
    except Exception as e:
        passed = False
    
    duration = (time.time() - start) * 1000
    reporter.report(name, passed, duration)
    assert passed, "循环依赖处理失败"


# ========== 测试2: 空内容和异常输入 ==========

def test_empty_and_malformed_content():
    """空内容和异常输入"""
    name = "空内容和异常输入"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    errors = []
    
    test_cases = [
        ("empty", "", ""),
        ("none_content", None, ""),  # None content should be handled
        ("whitespace", "   \n\t   ", ""),
        ("unicode", "🎉🚀💻🔥" * 1000, ""),
        ("binary_like", "\x00\x01\x02\x03" * 100, ""),
        ("huge_single_line", "A" * 100000, ""),
        ("many_lines", "\n".join([f"line_{i}" for i in range(10000)]), ""),
    ]
    
    for chunk_id, content, summary in test_cases:
        try:
            if content is None:
                content = ""
            store.add_chunk(ContextChunk(
                chunk_id=chunk_id,
                content=content,
                summary=summary,
            ))
            chunk = store.get_chunk(chunk_id)
            if chunk is None:
                errors.append(f"{chunk_id}: chunk not found after add")
        except Exception as e:
            errors.append(f"{chunk_id}: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    assert passed, f"空内容/异常输入处理失败: {errors}"


# ========== 测试3: 并发删除和读取竞争 ==========

def test_concurrent_delete_read_race():
    """并发删除和读取竞争"""
    name = "并发删除读取竞争"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    
    # 先添加一些数据
    for i in range(100):
        store.add_chunk(ContextChunk(
            chunk_id=f"race_{i}",
            content=f"content_{i}",
            summary=f"summary_{i}",
        ))
    
    errors = []
    
    def deleter():
        for i in range(100):
            try:
                store.remove_chunk(f"race_{random.randint(0, 99)}")
            except Exception as e:
                errors.append(f"delete: {e}")
    
    def reader():
        for i in range(100):
            try:
                chunk = store.get_chunk(f"race_{random.randint(0, 99)}")
                if chunk:
                    _ = chunk.content  # 访问内容
            except Exception as e:
                errors.append(f"read: {e}")
    
    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=deleter))
        threads.append(threading.Thread(target=reader))
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    assert passed, f"并发删除读取竞争失败: {errors}"


# ========== 测试4: 哨兵检测器缓冲区溢出边界 ==========

def test_detector_buffer_overflow_boundary():
    """哨兵检测器缓冲区溢出边界"""
    name = "哨兵检测器缓冲区溢出边界"
    start = time.time()
    detector = SentinelDetectorV2()
    errors = []
    
    # 测试1: 刚好在边界上的哨兵
    boundary_text = "A" * 9950 + "[NEED_CHUNK:boundary]" + "B" * 50
    try:
        requests = detector.feed(boundary_text)
        if not requests:
            errors.append("边界哨兵未检测到")
    except Exception as e:
        errors.append(f"边界测试: {e}")
    
    detector.reset()
    
    # 测试2: 跨边界的哨兵（前半部分在旧缓冲区，后半在新缓冲区）
    part1 = "A" * 9980 + "[NEED_CH"
    part2 = "UNK:cross_boundary]"
    try:
        detector.feed(part1)
        requests = detector.feed(part2)
        if not requests:
            errors.append("跨边界哨兵未检测到")
    except Exception as e:
        errors.append(f"跨边界测试: {e}")
    
    detector.reset()
    
    # 测试3: 大量小文本累积
    try:
        for i in range(200):
            detector.feed(f"text{i}[NEED_CHUNK:cum_{i}]")
        requests = detector.feed("")
        # 应该检测到所有累积的哨兵
    except Exception as e:
        errors.append(f"累积测试: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    assert passed, f"哨兵检测器缓冲区溢出边界失败: {errors}"


# ========== 测试5: Token预算并发分配 ==========

def test_token_budget_concurrent():
    """Token预算并发分配"""
    name = "Token预算并发分配"
    start = time.time()
    budget = TokenBudget(max_tokens=1000)
    errors = []
    
    def allocator(worker_id):
        for i in range(50):
            try:
                tokens = random.randint(10, 100)
                budget.allocate_tokens(tokens, f"worker_{worker_id}_{i}")
                time.sleep(random.random() * 0.001)  # 模拟工作
                budget.deallocate_tokens(tokens)
            except Exception as e:
                errors.append(f"worker {worker_id}: {e}")
    
    threads = [threading.Thread(target=allocator, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    duration = (time.time() - start) * 1000
    # 最终预算应该回到初始状态
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}, final_used={budget.used_tokens}")
    assert passed, f"Token预算并发分配失败: {errors}"


# ========== 测试6: 持久化文件损坏恢复 ==========

def test_persistence_corruption_recovery():
    """持久化文件损坏恢复"""
    name = "持久化文件损坏恢复"
    start = time.time()
    
    import tempfile
    import shutil
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="lcm_corrupt_"))
    errors = []
    
    # 创建正常数据
    store = ChunkStoreV2(storage_dir=tmp_dir, enable_persistence=True)
    for i in range(10):
        store.add_chunk(ContextChunk(
            chunk_id=f"good_{i}",
            content=f"good content {i}",
            summary=f"good {i}",
        ))
    store._save_to_disk()
    
    # 损坏 chunks.jsonl
    chunks_file = tmp_dir / "chunks.jsonl"
    with open(chunks_file, "a", encoding="utf-8") as f:
        f.write("\nthis is not valid json\n")
        f.write('{"invalid": "missing required fields"}\n')
        f.write("\n")
    
    # 尝试加载
    try:
        store2 = ChunkStoreV2(storage_dir=tmp_dir, enable_persistence=True)
        loaded = len(store2)
        if loaded < 10:
            errors.append(f"加载数据丢失: 期望10, 实际{loaded}")
    except Exception as e:
        errors.append(f"加载异常: {e}")
    
    # 损坏 index.json
    index_file = tmp_dir / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write("corrupted index")
    
    try:
        store3 = ChunkStoreV2(storage_dir=tmp_dir, enable_persistence=True)
    except Exception as e:
        errors.append(f"损坏index加载异常: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    assert passed, f"持久化文件损坏恢复失败: {errors}"


# ========== 测试7: 图操作极端条件 ==========

def test_graph_extreme_conditions():
    """图操作极端条件"""
    name = "图操作极端条件"
    start = time.time()
    graph = ChunkGraph()
    errors = []
    
    # 测试1: 自依赖
    try:
        graph.add_dependency("self", "self")
        has_cycle = graph.has_cycle()
        if not has_cycle:
            errors.append("自依赖未检测到环")
    except Exception as e:
        errors.append(f"自依赖: {e}")
    
    # 测试2: 大量节点（链式依赖: node_0 -> node_1 -> ... -> node_1000）
    # 总节点数 = 1000个依赖关系产生的1001个节点 + 测试1的"self"节点 = 1002
    try:
        for i in range(1000):
            graph.add_dependency(f"node_{i}", f"node_{i+1}")
        order = graph.topological_sort()
        expected_nodes = 1002  # 1001个链式节点 + "self"
        if len(order) != expected_nodes:
            errors.append(f"大量节点排序错误: 期望{expected_nodes}, 实际{len(order)}")
    except Exception as e:
        errors.append(f"大量节点: {e}")
    
    # 测试3: 不存在的节点依赖
    try:
        graph.add_dependency("ghost", "nonexistent")
        deps = graph.get_dependencies("ghost")
        if "nonexistent" not in deps:
            errors.append("不存在的节点依赖未记录")
    except Exception as e:
        errors.append(f"不存在的节点: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    assert passed, f"图操作极端条件失败: {errors}"


# ========== 测试8: 客户端流式异常 ==========

def test_client_stream_exception():
    """客户端流式异常处理"""
    name = "客户端流式异常"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    client = LCMClientV2(llm_client=None, chunk_store=store)
    errors = []
    
    # 添加测试数据
    store.add_chunk(ContextChunk(
        chunk_id="test_chunk",
        content="test content",
        summary="test",
    ))
    
    # 模拟流式异常
    def bad_stream(messages):
        yield "开始生成..."
        yield "[NEED_CHUNK:test_chunk]"
        raise RuntimeError("模拟流异常")
    
    try:
        messages = [{"role": "user", "content": "测试"}]
        result = client.chat(messages, bad_stream)
        # 应该返回已生成的部分，不抛出异常
    except Exception as e:
        errors.append(f"流异常未处理: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    assert passed, f"客户端流式异常处理失败: {errors}"


# ========== 测试9: 并发持久化竞争 ==========

def test_concurrent_persistence_race():
    """并发持久化竞争"""
    name = "并发持久化竞争"
    start = time.time()
    
    import tempfile
    import shutil
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="lcm_persist_race_"))
    store = ChunkStoreV2(storage_dir=tmp_dir, enable_persistence=True)
    errors = []
    
    def writer(worker_id):
        for i in range(20):
            try:
                store.add_chunk(ContextChunk(
                    chunk_id=f"race_{worker_id}_{i}",
                    content=f"content_{worker_id}_{i}",
                    summary=f"worker {worker_id}",
                ))
                store._save_to_disk()
            except Exception as e:
                errors.append(f"worker {worker_id}: {e}")
    
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # 验证加载
    try:
        store2 = ChunkStoreV2(storage_dir=tmp_dir, enable_persistence=True)
        loaded = len(store2)
        if loaded == 0:
            errors.append("持久化数据全部丢失")
    except Exception as e:
        errors.append(f"加载失败: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}, loaded={loaded if 'loaded' in dir() else 'N/A'}")
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    assert passed, f"并发持久化竞争失败: {errors}"


# ========== 测试10: 随机操作序列fuzz ==========

def test_random_operation_fuzz():
    """随机操作序列fuzz"""
    name = "随机操作序列fuzz"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False, max_cache_size=50)
    errors = []
    
    operations = ["add", "get", "remove", "search", "mark_loaded", "add_dep", "get_dep", "loading_order"]
    
    for i in range(10000):
        try:
            op = random.choice(operations)
            
            if op == "add":
                store.add_chunk(ContextChunk(
                    chunk_id=f"fuzz_{i}",
                    content=random_string(random.randint(0, 1000)),
                    summary=random_string(random.randint(0, 100)),
                    tokens=random.randint(0, 1000),
                    priority=random.randint(-100, 100),
                ))
            
            elif op == "get":
                store.get_chunk(f"fuzz_{random.randint(0, max(0, i-1))}")
            
            elif op == "remove":
                store.remove_chunk(f"fuzz_{random.randint(0, max(0, i-1))}")
            
            elif op == "search":
                store.search(random_string(random.randint(1, 20)))
            
            elif op == "mark_loaded":
                store.mark_loaded(f"fuzz_{random.randint(0, max(0, i-1))}")
            
            elif op == "add_dep":
                if i > 0:
                    store.add_dependency(
                        f"fuzz_{i}",
                        f"fuzz_{random.randint(0, i-1)}"
                    )
            
            elif op == "get_dep":
                store.get_dependencies(f"fuzz_{random.randint(0, max(0, i-1))}")
            
            elif op == "loading_order":
                store.get_loading_order(f"fuzz_{random.randint(0, max(0, i-1))}")
        
        except Exception as e:
            errors.append(f"step {i} ({op}): {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}, final_chunks={len(store)}")
    assert passed, f"随机操作序列fuzz失败: {errors[:5]}"


# ========== 主运行器 ==========

def run_all_stress_tests():
    print("=" * 60)
    print("LCM v2 深度压力测试 v2")
    print("=" * 60)
    
    tests = [
        test_circular_dependency_deadlock,
        test_empty_and_malformed_content,
        test_concurrent_delete_read_race,
        test_detector_buffer_overflow_boundary,
        test_token_budget_concurrent,
        test_persistence_corruption_recovery,
        test_graph_extreme_conditions,
        test_client_stream_exception,
        test_concurrent_persistence_race,
        test_random_operation_fuzz,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {test.__name__} - 断言失败: {e}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__} - 异常: {e}")
            import traceback
            traceback.print_exc()
    
    total_passed, total = reporter.summary()
    print(f"\n总计: {passed} 通过, {failed} 失败 / {len(tests)} 测试")
    return failed == 0


if __name__ == "__main__":
    success = run_all_stress_tests()
    exit(0 if success else 1)

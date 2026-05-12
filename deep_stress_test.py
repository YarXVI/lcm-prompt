"""
LCM v2 深度压力测试
目标：发现隐藏的缺陷和bug
策略：
1. 高并发读写混合
2. 长时间运行内存泄漏
3. 极端数据规模
4. 故障恢复
5. 随机fuzz
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
    return ''.join(random.choices(string.ascii_letters + string.digits + '\n\t ', k=length))


def random_chunk_id() -> str:
    return f"chunk_{random.randint(0, 99999)}_{random_string(10)}"


# ========== 测试1: 高并发读写混合 ==========

def test_high_concurrency_mixed_ops():
    """高并发读写混合操作 - 100线程，各操作1000次"""
    name = "高并发读写混合"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False, max_cache_size=50)
    errors = []
    ops_count = {"add": 0, "get": 0, "remove": 0, "search": 0}
    ops_lock = threading.Lock()

    def worker(worker_id):
        try:
            for i in range(100):
                op = random.choice(["add", "get", "remove", "search"])
                
                if op == "add":
                    chunk = ContextChunk(
                        chunk_id=f"w{worker_id}_{i}",
                        content=random_string(random.randint(10, 1000)),
                        summary=f"Worker {worker_id} chunk {i}",
                    )
                    store.add_chunk(chunk)
                    with ops_lock:
                        ops_count["add"] += 1
                
                elif op == "get":
                    chunk_id = f"w{random.randint(0, worker_id)}_{random.randint(0, i)}"
                    store.get_chunk(chunk_id)
                    with ops_lock:
                        ops_count["get"] += 1
                
                elif op == "remove":
                    chunk_id = f"w{worker_id}_{random.randint(0, max(0, i-1))}"
                    store.remove_chunk(chunk_id)
                    with ops_lock:
                        ops_count["remove"] += 1
                
                elif op == "search":
                    store.search(random_string(5))
                    with ops_lock:
                        ops_count["search"] += 1
        except Exception as e:
            errors.append(f"worker {worker_id}: {e}")

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(worker, i) for i in range(100)]
        for f in as_completed(futures):
            f.result()

    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}, ops={ops_count}")
    return passed


# ========== 测试2: 长时间运行内存泄漏 ==========

def test_memory_leak_extended():
    """长时间运行内存泄漏测试 - 循环10000次添加/删除"""
    name = "长时间内存泄漏"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False, max_cache_size=20)
    
    # 获取初始内存（近似）
    gc.collect()
    initial_chunks = len(store._chunks)
    
    for i in range(10000):
        # 添加
        store.add_chunk(ContextChunk(
            chunk_id=f"leak_{i}",
            content=random_string(random.randint(100, 5000)),
            summary=f"Leak test {i}",
        ))
        
        # 随机删除旧数据
        if i > 100:
            old_id = f"leak_{random.randint(0, i-100)}"
            store.remove_chunk(old_id)
    
    gc.collect()
    final_chunks = len(store._chunks)
    cache_size = len(store._cache)
    
    duration = (time.time() - start) * 1000
    # 最终chunk数应该接近max_cache_size，不应无限增长
    passed = final_chunks <= 200 and cache_size <= 20
    reporter.report(name, passed, duration, f"chunks={final_chunks}, cache={cache_size}")
    return passed


# ========== 测试3: 极端数据规模 ==========

def test_extreme_data_scale():
    """极端数据规模 - 10000个chunk，每个最大10KB"""
    name = "极端数据规模"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    
    # 添加10000个chunk
    for i in range(10000):
        store.add_chunk(ContextChunk(
            chunk_id=f"scale_{i}",
            content=random_string(random.randint(100, 10000)),
            summary=f"Scale test chunk {i}",
        ))
    
    # 随机访问
    hits = 0
    misses = 0
    for _ in range(1000):
        chunk = store.get_chunk(f"scale_{random.randint(0, 9999)}")
        if chunk:
            hits += 1
        else:
            misses += 1
    
    duration = (time.time() - start) * 1000
    passed = len(store) == 10000 and hits > 0
    reporter.report(name, passed, duration, f"chunks={len(store)}, hits={hits}, misses={misses}")
    return passed


# ========== 测试4: 故障恢复 ==========

def test_fault_recovery():
    """故障恢复 - 模拟存储损坏、并发写入冲突"""
    name = "故障恢复"
    start = time.time()
    
    # 测试1: 并发写入同一chunk
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False)
    errors = []
    
    def write_same_chunk(worker_id):
        try:
            for _ in range(100):
                store.add_chunk(ContextChunk(
                    chunk_id="shared_conflict",
                    content=f"content_from_worker_{worker_id}",
                    summary="Shared",
                ))
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=write_same_chunk, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # 验证最终状态一致性
    chunk = store.get_chunk("shared_conflict")
    passed = chunk is not None and len(errors) == 0
    
    duration = (time.time() - start) * 1000
    reporter.report(name, passed, duration, f"errors={len(errors)}, final_content_len={len(chunk.content) if chunk else 0}")
    return passed


# ========== 测试5: 随机Fuzz测试 ==========

def test_random_fuzz():
    """随机Fuzz - 随机操作序列"""
    name = "随机Fuzz"
    start = time.time()
    store = ChunkStoreV2(storage_dir=None, enable_persistence=False, max_cache_size=30)
    errors = []
    
    for i in range(5000):
        try:
            op = random.choice(["add", "get", "remove", "search", "graph_add", "graph_dep"])
            
            if op == "add":
                store.add_chunk(ContextChunk(
                    chunk_id=f"fuzz_{i}",
                    content=random_string(random.randint(0, 5000)),
                    summary=random_string(random.randint(0, 200)),
                    metadata={"random": random.random(), "list": [random.random() for _ in range(random.randint(0, 10))]},
                ))
            
            elif op == "get":
                store.get_chunk(f"fuzz_{random.randint(0, max(0, i-1))}")
            
            elif op == "remove":
                store.remove_chunk(f"fuzz_{random.randint(0, max(0, i-1))}")
            
            elif op == "search":
                store.search(random_string(random.randint(1, 50)))
            
            elif op == "graph_add":
                store.add_chunk(ContextChunk(
                    chunk_id=f"graph_{i}",
                    content="graph node",
                    summary="graph",
                ))
            
            elif op == "graph_dep":
                if i > 0:
                    store.add_dependency(f"graph_{i}", f"graph_{random.randint(0, i-1)}")
        
        except Exception as e:
            errors.append(f"step {i}: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}, final_chunks={len(store)}")
    return passed


# ========== 测试6: 哨兵检测器压力测试 ==========

def test_detector_stress():
    """哨兵检测器压力测试 - 大量随机文本中检测哨兵"""
    name = "哨兵检测器压力"
    start = time.time()
    detector = SentinelDetectorV2()
    errors = []
    
    # 生成包含随机哨兵的文本
    for i in range(1000):
        text = random_string(random.randint(10, 500))
        
        # 随机插入哨兵
        if random.random() < 0.3:
            pos = random.randint(0, len(text))
            sentinel = f"[NEED_CHUNK:test_{i}]"
            text = text[:pos] + sentinel + text[pos:]
        
        try:
            requests = detector.feed(text)
            
            # 验证检测到的请求
            if "[NEED_CHUNK:" in text:
                if not requests:
                    errors.append(f"step {i}: missed sentinel in text")
            
            # 每100次清理一次
            if i % 100 == 0:
                detector.reset()
        except Exception as e:
            errors.append(f"step {i}: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    return passed


# ========== 测试7: Token预算极端条件 ==========

def test_token_budget_extreme():
    """Token预算极端条件 - 0预算、超大chunk、负数"""
    name = "Token预算极端"
    start = time.time()
    errors = []
    
    # 测试1: 0预算
    try:
        budget = TokenBudget(max_tokens=0)
        chunks = [ContextChunk(chunk_id="c1", content="A", summary="", tokens=1)]
        selected = budget.select_chunks_within_budget(chunks)
        if selected:
            errors.append("0预算不应选择任何chunk")
    except Exception as e:
        errors.append(f"0预算异常: {e}")
    
    # 测试2: 超大chunk
    try:
        budget = TokenBudget(max_tokens=100)
        chunks = [ContextChunk(chunk_id="big", content="A"*10000, summary="", tokens=10000)]
        selected = budget.select_chunks_within_budget(chunks)
        if selected:
            errors.append("超大chunk不应被选中")
    except Exception as e:
        errors.append(f"超大chunk异常: {e}")
    
    # 测试3: 负数预算（应该处理）
    try:
        budget = TokenBudget(max_tokens=-100)
        chunks = [ContextChunk(chunk_id="c1", content="A", summary="", tokens=1)]
        selected = budget.select_chunks_within_budget(chunks)
    except Exception as e:
        errors.append(f"负数预算异常: {e}")
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0
    reporter.report(name, passed, duration, f"errors={len(errors)}")
    return passed


# ========== 测试8: 持久化并发写入 ==========

def test_persistence_concurrent_write():
    """持久化并发写入 - 多线程同时写入同一文件"""
    name = "持久化并发写入"
    start = time.time()
    
    import tempfile
    import shutil
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="lcm_stress_"))
    store = ChunkStoreV2(storage_dir=tmp_dir, enable_persistence=True)
    errors = []
    
    def writer(worker_id):
        try:
            for i in range(50):
                store.add_chunk(ContextChunk(
                    chunk_id=f"persist_{worker_id}_{i}",
                    content=random_string(random.randint(50, 500)),
                    summary=f"Worker {worker_id}",
                ))
                store._save_to_disk()
        except Exception as e:
            errors.append(f"worker {worker_id}: {e}")
    
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # 验证加载
    try:
        store2 = ChunkStoreV2(storage_dir=tmp_dir, enable_persistence=True)
        loaded = len(store2)
    except Exception as e:
        errors.append(f"加载失败: {e}")
        loaded = -1
    
    duration = (time.time() - start) * 1000
    passed = len(errors) == 0 and loaded > 0
    reporter.report(name, passed, duration, f"errors={len(errors)}, loaded={loaded}")
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return passed


# ========== 主运行器 ==========

def run_all_stress_tests():
    print("=" * 60)
    print("LCM v2 深度压力测试")
    print("=" * 60)
    
    tests = [
        test_high_concurrency_mixed_ops,
        test_memory_leak_extended,
        test_extreme_data_scale,
        test_fault_recovery,
        test_random_fuzz,
        test_detector_stress,
        test_token_budget_extreme,
        test_persistence_concurrent_write,
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
            print(f"  [FAIL] {test.__name__} - 异常: {e}")
            import traceback
            traceback.print_exc()
    
    total_passed, total = reporter.summary()
    print(f"\n总计: {passed} 通过, {failed} 失败 / {len(tests)} 测试")
    return failed == 0


if __name__ == "__main__":
    success = run_all_stress_tests()
    exit(0 if success else 1)

"""
LCM v2 真实世界测试
使用真实项目文件和路径进行端到端测试
所有测试包含真正的断言，失败会报告错误
"""
import sys
import os
import time
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_v2 import (
    ChunkStoreV2, ContextChunk, LCMClientV2,
    ProviderRouter, ProviderConfig, ProviderType, RoutingStrategy,
    HybridChunkManager, HybridConfig, build_hybrid_messages,
    TokenBudget, ChunkGraph, AdaptiveChunking,
    KVCacheManager, QualityEvaluator,
)


class TestFailure(Exception):
    """测试失败异常"""
    pass


def assert_true(condition, message):
    """断言为真"""
    if not condition:
        raise TestFailure(f"ASSERTION FAILED: {message}")


def assert_equal(actual, expected, message=""):
    """断言相等"""
    if actual != expected:
        raise TestFailure(f"ASSERTION FAILED: {message}\n  Expected: {expected}\n  Actual: {actual}")


def assert_greater(actual, threshold, message=""):
    """断言大于"""
    if not (actual > threshold):
        raise TestFailure(f"ASSERTION FAILED: {message}\n  Expected > {threshold}, got: {actual}")


def assert_less(actual, threshold, message=""):
    """断言小于"""
    if not (actual < threshold):
        raise TestFailure(f"ASSERTION FAILED: {message}\n  Expected < {threshold}, got: {actual}")


def assert_in(member, container, message=""):
    """断言包含"""
    if member not in container:
        raise TestFailure(f"ASSERTION FAILED: {message}\n  Expected '{member}' in container")


def assert_not_empty(container, message=""):
    """断言非空"""
    if not container:
        raise TestFailure(f"ASSERTION FAILED: {message}\n  Container is empty")


def test_real_persistence():
    """测试 1: 使用真实路径持久化"""
    print("\n=== 测试 1: 真实路径持久化 ===")
    real_dir = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2_test_data")

    try:
        # 清理旧数据
        if real_dir.exists():
            shutil.rmtree(real_dir)

        store = ChunkStoreV2(storage_dir=real_dir, enable_persistence=True)

        # 读取真实文件
        types_file = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2/types.py")
        store_file = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2/store.py")

        assert_true(types_file.exists(), f"源文件不存在: {types_file}")
        assert_true(store_file.exists(), f"源文件不存在: {store_file}")

        types_content = types_file.read_text(encoding="utf-8")
        store_content = store_file.read_text(encoding="utf-8")

        assert_greater(len(types_content), 100, "types.py 内容太短")
        assert_greater(len(store_content), 100, "store.py 内容太短")

        store.add_chunk(ContextChunk(
            chunk_id="lcm_types",
            content=types_content,
            summary="LCM v2 核心类型定义",
            source="lcm_v2/types.py",
        ))
        store.add_chunk(ContextChunk(
            chunk_id="lcm_store",
            content=store_content,
            summary="LCM v2 持久化存储实现",
            source="lcm_v2/store.py",
        ))

        # 断言：持久化文件必须存在
        chunks_file = real_dir / "chunks.jsonl"
        index_file = real_dir / "index.json"

        assert_true(chunks_file.exists(), f"chunks.jsonl 未创建: {chunks_file}")
        assert_true(index_file.exists(), f"index.json 未创建: {index_file}")

        # 断言：文件大小必须大于 0
        assert_greater(chunks_file.stat().st_size, 0, "chunks.jsonl 为空文件")
        assert_greater(index_file.stat().st_size, 0, "index.json 为空文件")

        # 重新加载验证
        store2 = ChunkStoreV2(storage_dir=real_dir, enable_persistence=True)
        assert_equal(len(store2), 2, f"重新加载后应有 2 个 chunks，实际有 {len(store2)}")

        c = store2.get_chunk("lcm_types")
        assert_true(c is not None, "重新加载后 lcm_types 不存在")
        assert_equal(c.content, types_content, "重新加载后内容不一致")
        assert_equal(c.source, "lcm_v2/types.py", "source 属性丢失")

        c2 = store2.get_chunk("lcm_store")
        assert_true(c2 is not None, "重新加载后 lcm_store 不存在")
        assert_greater(len(c2.content), 100, "重新加载后 store 内容太短")

        print("  ✓ 持久化测试通过")
        return True

    finally:
        if real_dir.exists():
            shutil.rmtree(real_dir)


def test_real_provider_routing():
    """测试 2: 真实 URL 路由识别"""
    print("\n=== 测试 2: 真实 URL 路由识别 ===")

    test_cases = [
        ("https://api.openai.com/v1/chat/completions", ProviderType.CLOUD, False),
        ("https://api.deepseek.com/v1", ProviderType.CLOUD, False),
        ("https://api.anthropic.com/v1/messages", ProviderType.CLOUD, False),
        ("http://localhost:11434/v1", ProviderType.LOCAL, True),
        ("http://127.0.0.1:8080/v1/chat/completions", ProviderType.LOCAL, True),
        ("http://192.168.1.100:5000", ProviderType.LOCAL, True),
        ("http://10.0.0.5:8000", ProviderType.LOCAL, True),
        ("http://172.16.0.1:3000", ProviderType.LOCAL, True),
        ("https://some-random-api.com/api", ProviderType.UNKNOWN, False),
    ]

    for url, expected_type, expected_lcm in test_cases:
        config = ProviderConfig(name="test", base_url=url)
        router = ProviderRouter(config)

        assert_equal(
            router.provider_type, expected_type,
            f"URL {url}: 类型应为 {expected_type.value}，实际为 {router.provider_type.value}"
        )
        assert_equal(
            router.use_lcm, expected_lcm,
            f"URL {url}: use_lcm 应为 {expected_lcm}，实际为 {router.use_lcm}"
        )
        print(f"  ✓ {url} -> {router.provider_type.value}, LCM={router.use_lcm}")

    print("  ✓ 路由识别测试通过")
    return True


def test_real_code_chunks():
    """测试 3: 真实代码文件作为 chunks"""
    print("\n=== 测试 3: 真实代码文件分块 ===")

    store = ChunkStoreV2(enable_persistence=False)
    project_dir = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2")

    assert_true(project_dir.exists(), f"项目目录不存在: {project_dir}")

    py_files = list(project_dir.glob("*.py"))
    assert_not_empty(py_files, "未找到 Python 文件")

    loaded_count = 0
    for f in py_files:
        if f.name in ("test_lcm_v2.py", "stress_test.py", "real_world_test.py"):
            continue
        try:
            content = f.read_text(encoding="utf-8")
            assert_greater(len(content), 10, f"{f.name} 内容太短")
            store.add_chunk(ContextChunk(
                chunk_id=f.stem,
                content=content,
                summary=f"{f.name} ({len(content)} chars)",
                source=str(f),
            ))
            loaded_count += 1
        except Exception as e:
            raise TestFailure(f"读取 {f.name} 失败: {e}")

    assert_greater(loaded_count, 5, f"加载的文件数太少: {loaded_count}")
    assert_equal(len(store), loaded_count, f"store 中的 chunk 数应为 {loaded_count}")

    # 搜索测试：必须返回非空结果
    results = store.search("chunk graph", top_k=3)
    assert_not_empty(results, "搜索 'chunk graph' 应返回结果")
    assert_less(len(results), 4, "搜索结果不应超过 top_k=3")

    # 验证搜索结果包含预期内容
    result_ids = [r.chunk_id for r in results]
    assert_true(
        any("chunk_graph" in rid or "graph" in rid for rid in result_ids),
        f"搜索结果应包含 chunk_graph 相关，实际: {result_ids}"
    )

    # 依赖图测试
    store.add_dependency("orchestrator", "store")
    store.add_dependency("orchestrator", "detector")
    store.add_dependency("client", "orchestrator")

    deps = store.get_dependencies("client", recursive=True)
    assert_in("orchestrator", deps, "client 应依赖 orchestrator")
    assert_in("store", deps, "client 应间接依赖 store")
    assert_in("detector", deps, "client 应间接依赖 detector")

    order = store.get_loading_order("client")
    assert_equal(len(order), 3, f"加载顺序应有 3 个依赖，实际 {len(order)}")
    # store 和 detector 没有依赖其他节点，应在前面
    assert_true(
        order.index("store") < order.index("orchestrator"),
        "store 应在 orchestrator 之前加载"
    )

    print("  ✓ 代码分块测试通过")
    return True


def test_real_token_budget():
    """测试 4: 真实 Token 预算管理"""
    print("\n=== 测试 4: 真实 Token 预算 ===")

    store = ChunkStoreV2(enable_persistence=False)
    project_dir = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2")

    for f in project_dir.glob("*.py"):
        if f.name in ("test_lcm_v2.py", "stress_test.py", "real_world_test.py"):
            continue
        content = f.read_text(encoding="utf-8")
        store.add_chunk(ContextChunk(
            chunk_id=f.stem,
            content=content,
            summary=f.name,
        ))

    assert_greater(len(store), 5, "应有超过 5 个 chunks")

    budget = TokenBudget(max_tokens=8000, system_reserve=500, safety_margin=200)
    assert_equal(budget.available_tokens, 7300, "可用预算计算错误")

    all_chunks = list(store._chunks.values())
    selected = budget.select_chunks_within_budget(all_chunks, strategy="small_first")

    assert_not_empty(selected, "应至少选中一个 chunk")
    assert_greater(len(selected), 0, "未选中任何 chunk")
    assert_less(len(selected), len(all_chunks), "不应选中所有 chunk（预算有限）")

    total_selected_tokens = sum(c.tokens for c in selected)
    assert_greater(total_selected_tokens, 0, "选中 chunks 总 token 应大于 0")
    assert_less(total_selected_tokens, 7300, "选中 chunks 不应超出预算")

    # 手动分配一些 chunk 到预算中，测试利用率
    for chunk in selected[:3]:
        budget.allocate(chunk)

    # 利用率应在合理范围
    assert_greater(budget.utilization_rate, 0, "利用率应大于 0")
    assert_less(budget.utilization_rate, 1.0, "利用率不应超过 100%")

    print("  ✓ Token 预算测试通过")
    return True


def test_real_hybrid_mode():
    """测试 5: 真实混合模式"""
    print("\n=== 测试 5: 真实混合模式 ===")

    store = ChunkStoreV2(enable_persistence=False)
    project_dir = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2")

    for f in project_dir.glob("*.py"):
        if f.name in ("test_lcm_v2.py", "stress_test.py", "real_world_test.py"):
            continue
        content = f.read_text(encoding="utf-8")
        store.add_chunk(ContextChunk(
            chunk_id=f.stem,
            content=content,
            summary=f.name,
        ))

    assert_greater(len(store), 5, "应有超过 5 个 chunks")

    # 模拟高频加载
    for _ in range(5):
        store.mark_loaded("types")
        store.mark_loaded("store")

    config = HybridConfig(hot_threshold=3)
    manager = HybridChunkManager(store, config)
    classified = manager.classify_chunks()

    assert_in("types", classified["hot"], "types 应被分类为高频")
    assert_in("store", classified["hot"], "store 应被分类为高频")

    hot_content = manager.get_hot_chunks_content()
    assert_in("types", hot_content, "高频内容应包含 types")
    assert_in("store", hot_content, "高频内容应包含 store")

    messages = build_hybrid_messages("审查 LCM 代码", store, config)
    assert_equal(len(messages), 2, "消息列表应有 2 条")

    system_content = messages[0]["content"]
    assert_in("高频上下文", system_content, "System prompt 应包含高频上下文标记")
    assert_in("types", system_content, "System prompt 应包含 types")

    print("  ✓ 混合模式测试通过")
    return True


def test_real_kv_cache():
    """测试 6: KV Cache 真实场景"""
    print("\n=== 测试 6: KV Cache 真实场景 ===")

    cache = KVCacheManager()

    prompt_file = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2/prompt.py")
    assert_true(prompt_file.exists(), f"prompt.py 不存在: {prompt_file}")

    system = prompt_file.read_text(encoding="utf-8")
    assert_greater(len(system), 100, "prompt.py 内容太短")

    index = "chunk_types, chunk_store, chunk_detector, chunk_orchestrator"

    key = cache.set_index_content(system, index)
    assert_true(key is not None, "缓存键不应为 None")
    assert_true(key.startswith("lcm_index:"), f"缓存键格式错误: {key}")

    headers = cache.get_cache_headers()
    assert_in("anthropic-beta", headers, "Headers 应包含 anthropic-beta")
    assert_in("x-lcm-cache-key", headers, "Headers 应包含 x-lcm-cache-key")
    assert_equal(headers["x-lcm-cache-key"], key, "Header 中的 cache key 应匹配")

    stats = cache.get_stats()
    assert_equal(stats["status"], "active", "缓存状态应为 active")
    assert_greater(stats["current_tokens"], 0, "缓存 token 数应大于 0")

    # 模拟多次使用
    for i in range(3):
        cache.record_use(key)

    stats2 = cache.get_stats()
    assert_equal(stats2["total_uses"], 3, f"使用次数应为 3，实际 {stats2['total_uses']}")

    print("  ✓ KV Cache 测试通过")
    return True


def test_real_quality_eval():
    """测试 7: 真实质量评估"""
    print("\n=== 测试 7: 真实质量评估 ===")

    evaluator = QualityEvaluator()

    # 优质输出
    good_output = """
    审查发现以下问题：
    1. [Chunk: auth] 密码使用明文存储，建议使用 bcrypt
    2. [Chunk: db] 连接池配置合理
    3. [Chunk: api] 缺少输入验证
    总体评价：安全性需改进
    """

    # 劣质输出（含幻觉和无关内容）
    bad_output = """
    让我想想... 嗯... 这个代码看起来还行吧
    可能有个问题？不对，应该没问题
    [NEED_CHUNK:nonexistent] 等等我需要再看看
    总的来说，还行吧
    """

    result1 = evaluator.evaluate_task(
        task_id="review_1",
        task_type="code_review",
        model="deepseek-v4",
        use_lcm=True,
        output=good_output,
        ground_truth={"expected_points": ["密码明文", "输入验证"]},
    )

    result2 = evaluator.evaluate_task(
        task_id="review_2",
        task_type="code_review",
        model="deepseek-v4",
        use_lcm=False,
        output=bad_output,
        ground_truth={"expected_points": ["密码明文", "输入验证"]},  # 同样的标准
    )

    # 优质输出应有更高的完整性
    assert_greater(
        result1.metrics.answer_completeness,
        result2.metrics.answer_completeness,
        f"优质输出的完整性应高于劣质输出: {result1.metrics.answer_completeness} vs {result2.metrics.answer_completeness}"
    )

    # 优质输出应有更高的聚焦度
    assert_greater(
        result1.metrics.focus_accuracy,
        result2.metrics.focus_accuracy,
        "优质输出的聚焦度应高于劣质输出"
    )

    # 劣质输出应有幻觉
    assert_greater(
        result2.metrics.hallucination_count,
        0,
        "劣质输出应检测到幻觉"
    )

    # 优质输出不应有幻觉
    assert_equal(
        result1.metrics.hallucination_count,
        0,
        "优质输出不应有幻觉"
    )

    # 指令遵循度：优质输出应更高
    assert_greater(
        result1.metrics.instruction_following,
        result2.metrics.instruction_following,
        "优质输出的指令遵循度应更高"
    )

    summary = evaluator.get_summary()
    assert_equal(summary["total_tasks"], 2, f"总任务数应为 2，实际 {summary['total_tasks']}")

    print("  ✓ 质量评估测试通过")
    return True


def test_real_multi_agent():
    """测试 8: 真实多 Agent 协作"""
    print("\n=== 测试 8: 多 Agent 协作 ===")

    from lcm_v2.multi_agent import SharedIndexManager

    store = ChunkStoreV2(enable_persistence=False)
    project_dir = Path("e:/agent办公室/工程区/agent-core/prompt_experiment/lcm_v2")

    for f in project_dir.glob("*.py"):
        if f.name in ("test_lcm_v2.py", "stress_test.py", "real_world_test.py"):
            continue
        content = f.read_text(encoding="utf-8")
        store.add_chunk(ContextChunk(
            chunk_id=f.stem,
            content=content,
            summary=f.name,
        ))

    assert_greater(len(store), 5, "应有超过 5 个 chunks")

    manager = SharedIndexManager(store)

    # 注册 Agent
    manager.register_agent("security_expert", "security")
    manager.register_agent("code_reviewer", "reviewer")
    manager.register_agent("architect", "architect")

    assert_equal(len(manager._agents), 3, "应有 3 个 Agent")

    # 设置共享 chunk
    manager.add_shared_chunk("types")
    manager.add_shared_chunk("store")

    assert_equal(len(manager._shared_index), 2, "应有 2 个共享 chunk")

    # 验证各 Agent 的索引
    for agent_id in ["security_expert", "code_reviewer", "architect"]:
        index = manager.get_agent_index(agent_id)
        assert_not_empty(index, f"{agent_id} 的索引不应为空")

        shared_count = sum(1 for i in index if i.get("shared"))
        assert_equal(
            shared_count, 2,
            f"{agent_id} 应有 2 个共享 chunk，实际 {shared_count}"
        )

    # 测试共享注入
    success = manager.inject_chunk_for_agent("security_expert", "types")
    assert_true(success, "注入共享 chunk 应成功")

    # 第二个 Agent 应能复用
    success2 = manager.inject_chunk_for_agent("code_reviewer", "types")
    assert_true(success2, "第二个 Agent 注入共享 chunk 应成功")

    stats = manager.get_collaboration_stats()
    assert_equal(stats["agents"], 3, f"Agent 数应为 3，实际 {stats['agents']}")
    assert_equal(stats["shared_chunks"], 2, f"共享 chunks 应为 2")
    assert_greater(stats["total_injections"], 0, "总注入数应大于 0")

    print("  ✓ 多 Agent 测试通过")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("LCM v2 真实世界测试（含真实断言）")
    print("=" * 60)

    tests = [
        test_real_persistence,
        test_real_provider_routing,
        test_real_code_chunks,
        test_real_token_budget,
        test_real_hybrid_mode,
        test_real_kv_cache,
        test_real_quality_eval,
        test_real_multi_agent,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except TestFailure as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{passed + failed} 通过")
    if failed > 0:
        print(f"失败: {failed}")
    print("=" * 60)

    # 非零退出码表示失败
    if failed > 0:
        sys.exit(1)

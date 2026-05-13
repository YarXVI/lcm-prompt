import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm import (
    GrainLevel, Grain, MultiGranularityIR, IR_VERSION,
    ContentEncoder, ContentDecoder, EncodingContext,
    EncodingRegistry, IdentityEncoder,
    Chunk, ChunkStore,
    EncodedChunkStore, EncodedChunk,
    SentinelDetector, LoadRequest,
    ExecutionProfile, PROFILE_DEFAULTS, PROFILE_DYNAMIC_RENDERING,
    AdaptiveInjector, UpgradeRequest, DowngradeRequest,
    CacheAwarePrefixBuilder,
    ContentEncoding, EncodingType, ContentEncodingRegistry, IdentityEncoding,
    V1EncodingContext,
    LCMEngine, LCMConfig, LCMSession, LCMState, LCMEvent,
    URRReporter, ChunkURRStats,
    LabelStore, ChunkLabel, Anchor,
    GoldenCorpusCollector, GoldenSample,
    DynamicRenderer, RenderedSlice,
    SemanticSlicer,
    ABTestRouter, ABTestConfig, ABTestResult,
    CodeIntentEncoder, ChineseThinkEncoder, EnglishLogicEncoder,
    create_engine,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def test_ir_models():
    print("\n=== IR Models ===")
    check("GrainLevel rank", GrainLevel.KEYWORDS.rank == 0)
    check("GrainLevel rank", GrainLevel.FULL.rank == 3)
    check("finer_or_equal", GrainLevel.DETAIL.finer_or_equal(GrainLevel.SUMMARY))
    check("not finer_or_equal", not GrainLevel.KEYWORDS.finer_or_equal(GrainLevel.DETAIL))

    g = Grain(GrainLevel.SUMMARY, "test content", 50)
    d = g.to_dict()
    g2 = Grain.from_dict(d)
    check("Grain roundtrip", g.content == g2.content and g.level == g2.level)

    ir = MultiGranularityIR(
        encoding_type="test",
        source_language="en",
        grains={
            GrainLevel.KEYWORDS: Grain(GrainLevel.KEYWORDS, "kw1, kw2", 10),
            GrainLevel.SUMMARY: Grain(GrainLevel.SUMMARY, "A summary.", 30),
            GrainLevel.DETAIL: Grain(GrainLevel.DETAIL, "Detail content.", 100),
            GrainLevel.FULL: Grain(GrainLevel.FULL, "Full content here.", 200),
        },
    )
    check("IR is_valid", ir.is_valid)
    check("IR compression_ratios", ir.compression_ratios["keywords"] == 10 / 200)

    best = ir.best_grain_for(50)
    check("best_grain_for 50tok", best.level == GrainLevel.SUMMARY)

    best2 = ir.best_grain_for(5)
    check("best_grain_for 5tok", best2.level == GrainLevel.KEYWORDS)

    ir_json = ir.to_json()
    ir2 = MultiGranularityIR.from_json(ir_json)
    check("IR JSON roundtrip", ir2.encoding_type == "test" and ir2.is_valid)

    stale = MultiGranularityIR.from_dict({"encoding_type": "x", "source_language": "en", "version": 99})
    check("Stale IR detected", stale.encoding_type == "_stale")


def test_encoder_base():
    print("\n=== Encoder Base ===")
    decoder = ContentDecoder()

    ir = MultiGranularityIR(
        encoding_type="test",
        source_language="en",
        grains={
            GrainLevel.KEYWORDS: Grain(GrainLevel.KEYWORDS, "kw", 10),
            GrainLevel.SUMMARY: Grain(GrainLevel.SUMMARY, "summary", 30),
            GrainLevel.DETAIL: Grain(GrainLevel.DETAIL, "detail", 100),
            GrainLevel.FULL: Grain(GrainLevel.FULL, "full", 200),
        },
    )

    content, level = decoder.decode(ir, 50)
    check("Lazy decode fits budget", level == GrainLevel.SUMMARY)

    content2, level2 = decoder.decode(ir, 500, min_level=GrainLevel.DETAIL)
    check("Explicit decode detail", level2 == GrainLevel.DETAIL)

    content3, level3 = decoder.decode(ir, 5, min_level=GrainLevel.FULL)
    check("Full override", level3 == GrainLevel.FULL)


def test_encoding_registry():
    print("\n=== Encoding Registry ===")
    registry = EncodingRegistry()
    check("Empty registry fallback", registry.detect_best("any text").encoding_type == "identity")

    code_enc = CodeIntentEncoder()
    registry.register(code_enc)
    check("Register encoder", registry.get("code-intent") is not None)

    best = registry.detect_best("def foo():\n    pass")
    check("Detect code", best.encoding_type == "code-intent")

    check("Unregister", registry.unregister("code-intent"))
    check("After unregister", registry.get("code-intent") is None)


def test_chunk_store():
    print("\n=== Chunk Store ===")
    tmpdir = tempfile.mkdtemp()
    try:
        store = ChunkStore(storage_dir=tmpdir, enable_persistence=False)
        chunk = Chunk("test-1", "Hello world content", summary="A test chunk")
        store.add(chunk)

        retrieved = store.get("test-1")
        check("Add and get", retrieved is not None and retrieved.content == "Hello world content")
        check("Token estimation", retrieved.tokens > 0)
        check("Load count", retrieved.load_count == 1)

        summaries = store.list_summaries()
        check("List summaries", len(summaries) == 1)

        stats = store.get_stats()
        check("Stats", stats["total_chunks"] == 1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_encoded_chunk_store():
    print("\n=== Encoded Chunk Store ===")
    tmpdir = tempfile.mkdtemp()
    try:
        registry = EncodingRegistry()
        registry.register(CodeIntentEncoder())
        registry.register(ChineseThinkEncoder())

        store = EncodedChunkStore(
            encoding_registry=registry,
            ir_storage_dir=os.path.join(tmpdir, "ir"),
        )
        store._store = ChunkStore(storage_dir=os.path.join(tmpdir, "chunks"), enable_persistence=False)

        code_chunk = Chunk("code-1", "def hello():\n    print('world')\n", source="test.py")
        encoded = store.add(code_chunk)
        check("Add encoded chunk", encoded.ir.encoding_type == "code-intent")
        check("Has all 4 grains", len(encoded.ir.grains) == 4)

        retrieved = store.get_encoded("code-1")
        check("Get encoded", retrieved is not None)
        check("IR valid", retrieved.ir.is_valid)

        grain = store.get_grain("code-1", GrainLevel.SUMMARY)
        check("Get grain", grain is not None and grain.level == GrainLevel.SUMMARY)

        best = store.best_grain_for("code-1", 50)
        check("Best grain for budget", best is not None)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sentinel_detector():
    print("\n=== Sentinel Detector ===")
    detector = SentinelDetector()

    requests = detector.feed("I need [NEED_CHUNK:auth-module] to continue")
    check("Detect NEED_CHUNK", len(requests) == 1 and requests[0].chunk_id == "auth-module")

    clean = detector.get_clean_buffer()
    check("Clean buffer", "[NEED_CHUNK:auth-module]" not in clean)

    detector.reset()
    requests2 = detector.feed("Please [NEED_CHUNK_DETAIL:db-layer] and [NEED_CHUNK_FULL:config]")
    check("Multiple sentinels", len(requests2) == 2)
    check("DETAIL sentinel", any(r.min_level == GrainLevel.DETAIL for r in requests2))
    check("FULL sentinel", any(r.min_level == GrainLevel.FULL for r in requests2))


def test_execution_profile():
    print("\n=== Execution Profile ===")
    check("LOCAL default grain", PROFILE_DEFAULTS[ExecutionProfile.LOCAL_CONSTRAINED] == GrainLevel.DETAIL)
    check("CLOUD_TOKEN default grain", PROFILE_DEFAULTS[ExecutionProfile.CLOUD_TOKEN_BILLED] == GrainLevel.SUMMARY)
    check("CLOUD_DYNAMIC default grain", PROFILE_DEFAULTS[ExecutionProfile.CLOUD_DYNAMIC] == GrainLevel.KEYWORDS)
    check("Dynamic rendering CLOUD_DYNAMIC", PROFILE_DYNAMIC_RENDERING[ExecutionProfile.CLOUD_DYNAMIC])
    check("No dynamic rendering LOCAL", not PROFILE_DYNAMIC_RENDERING[ExecutionProfile.LOCAL_CONSTRAINED])


def test_adaptive_injector():
    print("\n=== Adaptive Injector ===")
    tmpdir = tempfile.mkdtemp()
    try:
        registry = EncodingRegistry()
        registry.register(CodeIntentEncoder())
        store = EncodedChunkStore(
            encoding_registry=registry,
            ir_storage_dir=os.path.join(tmpdir, "ir"),
        )
        store._store = ChunkStore(storage_dir=os.path.join(tmpdir, "chunks"), enable_persistence=False)

        chunk = Chunk("mod-1", "def authenticate(user, pwd):\n    return verify(user, pwd)\n", source="auth.py")
        store.add(chunk)

        injector = AdaptiveInjector(
            encoded_store=store,
            profile=ExecutionProfile.LOCAL_CONSTRAINED,
        )
        injector.set_session("test-session")

        messages = []
        messages, level = injector.inject(messages, "mod-1", available_tokens=4000)
        check("Inject returns messages", len(messages) >= 1)
        check("Inject grain level", level is not None)

        injector.process_sentinels("[NEED_CHUNK_DETAIL:mod-1]")
        upgrade_map = injector.get_upgrade_map()
        check("Process upgrade sentinel", "mod-1" in upgrade_map)

        injector.tick_cooldown()
        check("Cooldown tick", True)

        stats = injector.get_stats()
        check("Injector stats", "profile" in stats and "total_injections" in stats)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_code_intent_encoder():
    print("\n=== CodeIntentEncoder ===")
    enc = CodeIntentEncoder()
    check("encoding_type", enc.encoding_type == "code-intent")

    python_code = '''
def authenticate(user, password):
    """Verify user credentials."""
    return verify_credentials(user, password)

class AuthManager:
    def __init__(self):
        self.session = None

    def login(self, user, pwd):
        return authenticate(user, pwd)
'''
    score = enc.detect(python_code)
    check("Detect Python code", score > 0.3)

    ctx = EncodingContext(chunk_id="test", source="auth.py")
    ir = enc.encode(python_code, ctx)
    check("Encode produces IR", ir.encoding_type == "code-intent")
    check("Has keywords", len(ir.grains[GrainLevel.KEYWORDS].content) > 0)
    check("Has summary", len(ir.grains[GrainLevel.SUMMARY].content) > 0)
    check("Has detail", len(ir.grains[GrainLevel.DETAIL].content) > 0)
    check("Has full", ir.grains[GrainLevel.FULL].content == python_code)
    check("Full is reversible", ir.grains[GrainLevel.FULL].reversible)
    check("Keywords not reversible", not ir.grains[GrainLevel.KEYWORDS].reversible)
    check("Call graph", len(ir.structure.get("call_graph", [])) > 0)

    js_code = '''
export function handleSubmit(event) {
    const formData = new FormData(event.target);
    return api.submit(formData);
}

export const CONFIG = { timeout: 5000 };
'''
    js_score = enc.detect(js_code)
    check("Detect JS code", js_score > 0.2)
    js_ir = enc.encode(js_code, ctx)
    check("JS encode", js_ir.encoding_type == "code-intent")


def test_chinese_think_encoder():
    print("\n=== ChineseThinkEncoder ===")
    enc = ChineseThinkEncoder()
    check("encoding_type", enc.encoding_type == "chinese-think")

    zh_text = """
# 系统架构设计

本系统采用洋葱架构，将核心业务逻辑与基础设施解耦。

## 第一层：领域层
领域层包含核心业务规则和实体定义。

## 第二层：应用层
应用层协调领域对象完成业务用例。

## 第三层：基础设施层
基础设施层实现技术细节，如数据库访问和外部API调用。
"""
    score = enc.detect(zh_text)
    check("Detect Chinese", score > 0.3)

    ctx = EncodingContext(chunk_id="zh-1")
    ir = enc.encode(zh_text, ctx)
    check("Chinese encode", ir.encoding_type == "chinese-think")
    check("Chinese keywords", len(ir.grains[GrainLevel.KEYWORDS].content) > 0)
    check("Chinese summary", len(ir.grains[GrainLevel.SUMMARY].content) > 0)
    check("Chinese detail has structure", len(ir.grains[GrainLevel.DETAIL].content) > 0)


def test_english_logic_encoder():
    print("\n=== EnglishLogicEncoder ===")
    enc = EnglishLogicEncoder()
    check("encoding_type", enc.encoding_type == "en-logic")

    en_text = """
# System Architecture

The system follows a modular design pattern with clear separation of concerns.

## Core Module
The core module handles business logic and data processing.

## API Layer
RESTful endpoints provide external access to system capabilities.

1. Authentication endpoints
2. Data retrieval endpoints
3. Management endpoints

- High availability
- Horizontal scaling
- Event-driven processing
"""
    score = enc.detect(en_text)
    check("Detect English", score > 0.1)

    ctx = EncodingContext(chunk_id="en-1")
    ir = enc.encode(en_text, ctx)
    check("English encode", ir.encoding_type == "en-logic")
    check("English keywords", len(ir.grains[GrainLevel.KEYWORDS].content) > 0)
    check("English detail", len(ir.grains[GrainLevel.DETAIL].content) > 0)


def test_lcm_engine():
    print("\n=== LCMEngine ===")
    tmpdir = tempfile.mkdtemp()
    try:
        engine = create_engine(profile=ExecutionProfile.LOCAL_CONSTRAINED)
        engine.store._enable_persistence = False
        from pathlib import Path as _P
        engine._encoded_store._ir_dir = _P(tmpdir) / "ir"
        ( _P(tmpdir) / "ir").mkdir(parents=True, exist_ok=True)

        chunk1 = Chunk("auth", "def authenticate(user, pwd):\n    return verify(user, pwd)\n", summary="Auth function", source="auth.py")
        chunk2 = Chunk("config", "CONFIG = {'timeout': 5000, 'retries': 3}\n", summary="Config", source="config.py")
        engine.store.add(chunk1)
        engine.store.add(chunk2)
        engine._encoded_store.add(chunk1)
        engine._encoded_store.add(chunk2)

        session = engine.new_session("test-1")
        check("New session", session.session_id == "test-1")
        check("Session state IDLE", session.state == LCMState.IDLE)

        prompt = engine.build_system_prompt("You are a helpful assistant.")
        check("Build system prompt", "NEED_CHUNK" in prompt)
        check("Index in prompt", "auth" in prompt)

        clean, requests = engine.process_response("I need to see [NEED_CHUNK:auth]")
        check("Process response", len(requests) == 1)
        check("Request chunk_id", requests[0].chunk_id == "auth")

        loaded = engine.load_chunk("auth")
        check("Load chunk", loaded is not None)

        messages = [{"role": "user", "content": "Check auth"}]
        messages = engine.inject_chunk(messages, loaded)
        check("Inject chunk", len(messages) >= 2)

        stats = engine.get_stats()
        check("Engine stats", "profile" in stats and "session_id" in stats)

        def mock_llm(msgs):
            return "Based on the auth code, the function verifies credentials."

        result = engine.run_sync(
            [{"role": "user", "content": "Analyze the auth module"}],
            mock_llm,
            session_id="sync-test",
        )
        check("Run sync", len(result) > 0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_urr_reporter():
    print("\n=== URR Reporter ===")
    reporter = URRReporter()
    reporter.record_injection("chunk-1", "summary", 5.0, False)
    reporter.record_injection("chunk-1", "summary", 4.0, False)
    reporter.record_injection("chunk-1", "detail", 8.0, True)

    stats = reporter.get_chunk_stats("chunk-1")
    check("URR stats", stats is not None)
    check("Total injections", stats.total_injections == 3)
    check("Upgrade requests", stats.upgrade_requests == 1)
    check("URR value", abs(stats.urr - 1/3) < 0.01)
    check("Not high URR", not stats.is_high_urr)

    report = reporter.get_report()
    check("Report", "global_urr" in report)


def test_label_system():
    print("\n=== Label System ===")
    store = LabelStore()

    label = ChunkLabel(
        chunk_id="auth-module",
        anchors=[
            Anchor("auth_func", 10, 25, "def authenticate(...)", semantic_tag="auth"),
            Anchor("config_section", 30, 40, "CONFIG = {...}", semantic_tag="config"),
        ],
        tags=["auth", "core"],
        priority=5,
    )
    store.add_label(label)

    retrieved = store.get_label("auth-module")
    check("Add/get label", retrieved is not None)
    check("Anchors count", len(retrieved.anchors) == 2)
    check("Coverage", retrieved.coverage > 0)

    stats = store.get_coverage_stats()
    check("Coverage stats", stats["total_labeled"] == 1)


def test_golden_corpus():
    print("\n=== Golden Corpus ===")
    collector = GoldenCorpusCollector()

    ir = MultiGranularityIR(
        encoding_type="code-intent",
        source_language="code",
        grains={
            GrainLevel.KEYWORDS: Grain(GrainLevel.KEYWORDS, "auth", 5),
            GrainLevel.FULL: Grain(GrainLevel.FULL, "def auth(): pass", 20, True),
        },
    )

    result = collector.consider("chunk-1", "def auth(): pass", ir, 2, 0)
    check("Below min injections", not result)

    result2 = collector.consider("chunk-1", "def auth(): pass", ir, 3, 0)
    check("Golden sample accepted", result2)

    result3 = collector.consider("chunk-2", "def db(): pass", ir, 5, 1)
    check("Non-golden rejected", not result3)

    stats = collector.get_stats()
    check("Golden stats", stats["total_samples"] == 1)


def test_dynamic_renderer():
    print("\n=== Dynamic Renderer ===")
    label_store = LabelStore()
    label = ChunkLabel(
        chunk_id="auth-code",
        anchors=[
            Anchor("login_func", 1, 5, "def login():", semantic_tag="auth"),
            Anchor("error_handler", 10, 15, "try: ...", semantic_tag="error"),
        ],
    )
    label_store.add_label(label)

    renderer = DynamicRenderer(label_store)
    check("Can render", renderer.can_render("auth-code"))
    check("Cannot render unknown", not renderer.can_render("unknown"))

    content = "def login():\n    pass\n\ndef logout():\n    pass\n\ntry:\n    login()\nexcept:\n    pass\n"

    result = renderer.render("auth-code", content, query_intent="auth login", available_tokens=200)
    check("Render with intent", result is not None)
    check("Render anchor name", result.anchor_name == "login_func")

    result2 = renderer.render("auth-code", content, query_intent="", available_tokens=200)
    check("Render fallback", result2 is not None)


def test_semantic_slicer():
    print("\n=== Semantic Slicer ===")
    slicer = SemanticSlicer()

    content = "def auth():\n    pass\n\nclass Manager:\n    def run(self):\n        pass\n\n# config\nTIMEOUT = 5"
    result = slicer.slice_by_sections(content, max_tokens=100)
    check("Slice by sections", len(result) > 0)

    result2 = slicer.slice_by_intent(content, ["auth"], max_tokens=100)
    check("Slice by intent", len(result2) > 0)

    result3 = slicer.slice_by_density(content, max_tokens=100)
    check("Slice by density", len(result3) > 0)


def test_ab_test_router():
    print("\n=== A/B Test Router ===")
    router = ABTestRouter()

    config = ABTestConfig(test_name="ir-vs-dynamic", traffic_split=0.5)
    router.create_test(config)

    group = router.get_group("ir-vs-dynamic", "session-1")
    check("Get group", group in ("control", "experiment"))

    result = ABTestResult(
        test_name="ir-vs-dynamic",
        group="control",
        chunk_id="chunk-1",
        session_id="s1",
        tokens_injected=100,
        tokens_saved=200,
        upgrade_count=0,
        latency_ms=5.0,
    )
    router.record_result(result)

    analysis = router.get_analysis("ir-vs-dynamic")
    check("Analysis", "test_name" in analysis)


def test_cache_builder():
    print("\n=== Cache Builder ===")
    builder = CacheAwarePrefixBuilder()

    summaries = [{"chunk_id": "c1", "source": "test", "tokens": 100, "summary": "Test chunk"}]
    index = builder.build_chunk_index(summaries)
    check("Build chunk index", "c1" in index)

    messages = builder.build_cached_messages("System prompt", [], index)
    check("Build cached messages", len(messages) == 1)


def test_content_encoding():
    print("\n=== Content Encoding (V1 compat) ===")
    registry = ContentEncodingRegistry()
    check("Default identity", registry.get(EncodingType.IDENTITY).name == "Identity Encoding")

    identity = IdentityEncoding()
    ctx = V1EncodingContext(session_id="test")
    check("Identity encode prompt", identity.encode_system_prompt("test", ctx) == "test")
    check("Identity encode response", identity.encode_response("test", ctx) == "test")


def test_create_engine():
    print("\n=== create_engine Factory ===")
    engine = create_engine(profile=ExecutionProfile.CLOUD_TOKEN_BILLED)
    check("Engine created", engine is not None)
    check("Profile set", engine.config.profile == ExecutionProfile.CLOUD_TOKEN_BILLED)
    check("Encoders registered", len(engine._encoding_registry.list_encoders()) >= 3)


def test_real_world_python_code():
    print("\n=== Real-world Python Code ===")
    enc = CodeIntentEncoder()
    real_code = '''
"""Authentication module for the web application."""

import hashlib
import os
from typing import Optional, Dict, Any

from database import get_connection
from models import User


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash a password with salt using SHA-256."""
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against stored hash."""
    salt, _ = stored_hash.split(":")
    return hash_password(password, salt) == stored_hash


class AuthService:
    """Service handling user authentication."""

    def __init__(self, db_connection=None):
        self.db = db_connection or get_connection()

    def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return session data."""
        user = self.db.query(User).filter(User.username == username).first()
        if user and verify_password(password, user.password_hash):
            return {"user_id": user.id, "session_token": self._create_session(user)}
        return None

    def _create_session(self, user: User) -> str:
        """Create a new session token."""
        token = os.urandom(32).hex()
        self.db.execute(
            "INSERT INTO sessions (user_id, token) VALUES (?, ?)",
            (user.id, token)
        )
        return token

    def logout(self, token: str) -> bool:
        """Invalidate a session token."""
        result = self.db.execute("DELETE FROM sessions WHERE token = ?", (token,))
        return result.rowcount > 0
'''
    score = enc.detect(real_code)
    check("Detect real Python", score > 0.5)

    ctx = EncodingContext(chunk_id="auth-module", source="auth.py")
    ir = enc.encode(real_code, ctx)

    kw = ir.grains[GrainLevel.KEYWORDS].content
    check("Keywords extracted", "hash_password" in kw or "AuthService" in kw)

    summary = ir.grains[GrainLevel.SUMMARY].content
    check("Summary extracted", len(summary) > 10)

    detail = ir.grains[GrainLevel.DETAIL].content
    check("Detail has signatures", "def " in detail or "class " in detail)

    full = ir.grains[GrainLevel.FULL].content
    check("Full preserved", full == real_code)

    ratios = ir.compression_ratios
    check("Compression ratios", ratios["keywords"] < ratios["full"])


def test_real_world_chinese_doc():
    print("\n=== Real-world Chinese Doc ===")
    enc = ChineseThinkEncoder()
    zh_doc = """
# 惰性上下文物化协议技术规范

## 1. 概述

惰性上下文物化（Lazy Context Materialization, LCM）是一种面向大语言模型的上下文管理协议。
其核心思想是将完整上下文按需加载，而非一次性注入全部信息，从而显著降低首Token延迟和推理开销。

## 2. 多粒度中间表示

LCM采用固定4级粒度层次结构：

1. KEYWORDS - 关键词级，约30-50 tokens
2. SUMMARY - 摘要级，约80-200 tokens
3. DETAIL - 详情级，约300-800 tokens
4. FULL - 完整级，原始大小

## 3. 编码层架构

编码层采用可插拔设计，支持多种语言和内容类型的编码器：

- ChineseThinkEncoder: 中文文档辩证/文言/精简压缩
- EnglishLogicEncoder: 英文文档结构化/逻辑树/大纲
- CodeIntentEncoder: 代码AST/伪代码/意图提取

## 4. 执行Profile路由

系统根据部署环境自动选择最优注入粒度：

- LOCAL_CONSTRAINED: 本地受限环境，默认DETAIL级
- CLOUD_TOKEN_BILLED: 云端按Token计费，默认SUMMARY级
- CLOUD_CALL_BILLED: 云端按调用计费，默认DETAIL级
- CLOUD_DYNAMIC: 动态模板渲染，默认KEYWORDS级
"""
    score = enc.detect(zh_doc)
    check("Detect Chinese doc", score > 0.2)

    ctx = EncodingContext(chunk_id="lcm-spec")
    ir = enc.encode(zh_doc, ctx)

    kw = ir.grains[GrainLevel.KEYWORDS].content
    check("Chinese keywords", len(kw) > 0)

    detail = ir.grains[GrainLevel.DETAIL].content
    check("Chinese detail structure", len(detail) > 0)


def test_real_world_english_doc():
    print("\n=== Real-world English Doc ===")
    enc = EnglishLogicEncoder()
    en_doc = """
# Lazy Context Materialization Protocol

## Overview

The Lazy Context Materialization (LCM) protocol is a context management system designed for large language models.
It loads context on-demand rather than injecting all information at once, significantly reducing first-token latency and inference costs.

## Multi-Granularity IR

LCM uses a fixed 4-level grain hierarchy:

1. KEYWORDS - Key terms only (~30-50 tokens)
2. SUMMARY - Concise overview (~80-200 tokens)
3. DETAIL - Signatures, pseudocode, structured points (~300-800 tokens)
4. FULL - Complete original content

## Encoder Architecture

The encoding layer uses a pluggable design supporting multiple language and content type encoders:

- ChineseThinkEncoder: Chinese document dialectical/classical compression
- EnglishLogicEncoder: English document structured/logic tree/outline
- CodeIntentEncoder: Code AST/pseudocode/intent extraction

## Execution Profile Routing

The system automatically selects the optimal injection grain based on deployment environment:

- LOCAL_CONSTRAINED: Default DETAIL grain
- CLOUD_TOKEN_BILLED: Default SUMMARY grain
- CLOUD_CALL_BILLED: Default DETAIL grain
- CLOUD_DYNAMIC: Default KEYWORDS grain with dynamic template rendering
"""
    score = enc.detect(en_doc)
    check("Detect English doc", score > 0.1)

    ctx = EncodingContext(chunk_id="lcm-spec-en")
    ir = enc.encode(en_doc, ctx)

    kw = ir.grains[GrainLevel.KEYWORDS].content
    check("English keywords", len(kw) > 0)

    detail = ir.grains[GrainLevel.DETAIL].content
    check("English detail", len(detail) > 0)


def test_encoder_selection():
    print("\n=== Encoder Auto-Selection ===")
    registry = EncodingRegistry()
    registry.register(CodeIntentEncoder())
    registry.register(ChineseThinkEncoder())
    registry.register(EnglishLogicEncoder())

    code = "def foo():\n    return bar()"
    zh = "这是一个中文文档，描述了系统的核心功能。"
    en = "This is an English document describing the system architecture."

    best_code = registry.detect_best(code)
    check("Code wins for code", best_code.encoding_type == "code-intent")

    best_zh = registry.detect_best(zh)
    check("Chinese wins for Chinese", best_zh.encoding_type == "chinese-think")

    best_en = registry.detect_best(en)
    check("English wins for English", best_en.encoding_type == "en-logic")


def test_ir_serialization():
    print("\n=== IR Serialization ===")
    ir = MultiGranularityIR(
        encoding_type="code-intent",
        source_language="python",
        grains={
            GrainLevel.KEYWORDS: Grain(GrainLevel.KEYWORDS, "auth, login", 10),
            GrainLevel.SUMMARY: Grain(GrainLevel.SUMMARY, "Auth module", 20),
            GrainLevel.DETAIL: Grain(GrainLevel.DETAIL, "def auth():\ndef login():", 50),
            GrainLevel.FULL: Grain(GrainLevel.FULL, "def auth(): pass\ndef login(): pass", 80, True),
        },
        structure={"call_graph": ["auth", "login"]},
    )

    json_str = ir.to_json()
    ir2 = MultiGranularityIR.from_json(json_str)
    check("JSON roundtrip type", ir2.encoding_type == ir.encoding_type)
    check("JSON roundtrip grains", len(ir2.grains) == 4)
    check("JSON roundtrip structure", ir2.structure["call_graph"] == ["auth", "login"])

    d = ir.to_dict()
    ir3 = MultiGranularityIR.from_dict(d)
    check("Dict roundtrip", ir3.encoding_type == ir.encoding_type)


def test_full_pipeline():
    print("\n=== Full Pipeline ===")
    tmpdir = tempfile.mkdtemp()
    try:
        engine = create_engine(profile=ExecutionProfile.CLOUD_TOKEN_BILLED)
        engine.store._enable_persistence = False
        from pathlib import Path as _P
        engine._encoded_store._ir_dir = _P(tmpdir) / "ir"
        (_P(tmpdir) / "ir").mkdir(parents=True, exist_ok=True)

        code_chunk = Chunk(
            "payment",
            "def process_payment(amount, currency='USD'):\n    \"\"\"Process a payment.\"\"\"\n    txn = create_transaction(amount, currency)\n    result = gateway.submit(txn)\n    return result\n",
            summary="Payment processing function",
            source="payment.py",
        )
        engine.store.add(code_chunk)
        engine._encoded_store.add(code_chunk)

        session = engine.new_session("pipeline-test")
        prompt = engine.build_system_prompt("You are a code reviewer.", "Review the payment module")
        if isinstance(prompt, list):
            prompt_text = " ".join(m.get("content", "") for m in prompt)
        else:
            prompt_text = prompt
        check("Pipeline prompt built", "payment" in prompt_text)

        clean, requests = engine.process_response("Let me check [NEED_CHUNK:payment]")
        check("Pipeline sentinel", len(requests) == 1)

        chunk = engine.load_chunk("payment")
        messages = [{"role": "user", "content": "Review payment"}]
        messages = engine.inject_chunk(messages, chunk)
        check("Pipeline injection", len(messages) >= 2)

        stats = engine.get_stats()
        check("Pipeline stats", stats["tokens_saved"] >= 0)
        check("Pipeline profile", stats["profile"] == "cloud_token_billed")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("LCM Independent Project - Comprehensive Test Suite")
    print("=" * 60)

    test_ir_models()
    test_encoder_base()
    test_encoding_registry()
    test_chunk_store()
    test_encoded_chunk_store()
    test_sentinel_detector()
    test_execution_profile()
    test_adaptive_injector()
    test_code_intent_encoder()
    test_chinese_think_encoder()
    test_english_logic_encoder()
    test_lcm_engine()
    test_urr_reporter()
    test_label_system()
    test_golden_corpus()
    test_dynamic_renderer()
    test_semantic_slicer()
    test_ab_test_router()
    test_cache_builder()
    test_content_encoding()
    test_create_engine()
    test_real_world_python_code()
    test_real_world_chinese_doc()
    test_real_world_english_doc()
    test_encoder_selection()
    test_ir_serialization()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} PASS, {FAIL} FAIL, Total {PASS + FAIL}")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)

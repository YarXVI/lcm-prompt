"""
LCM v2 内容编码层测试
验证编码层与 LCM 核心的解耦、Chinese-Think 编码插件的正确性
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_v2.content_encoding import (
    ContentEncoding, EncodingType, EncodingContext,
    ContentEncodingRegistry, IdentityEncoding,
    get_default_registry, register_encoding, get_encoding,
)
from lcm_v2.lcm_types import ContextChunk
from lcm_v2.store import ChunkStoreV2
from lcm_v2.orchestrator import LCMOrchestratorV2


class TestContentEncodingInterface(unittest.TestCase):
    """测试编码接口规范"""

    def test_identity_encoding(self):
        enc = IdentityEncoding()
        self.assertEqual(enc.encoding_type, EncodingType.IDENTITY)
        self.assertEqual(enc.name, "恒等编码（无变换）")

        ctx = EncodingContext()
        text = "测试文本"
        self.assertEqual(enc.encode_system_prompt(text, ctx), text)
        self.assertEqual(enc.encode_response(text, ctx), text)
        self.assertEqual(enc.decode_for_display(text, ctx), text)

    def test_encoding_context(self):
        ctx = EncodingContext(
            session_id="test_123",
            user_query="查此函数",
            current_round=1,
            metadata={"key": "value"},
        )
        self.assertEqual(ctx.session_id, "test_123")
        self.assertEqual(ctx.current_round, 1)


class TestContentEncodingRegistry(unittest.TestCase):
    """测试编码注册表"""

    def setUp(self):
        self.registry = ContentEncodingRegistry()

    def test_default_identity(self):
        enc = self.registry.get(EncodingType.IDENTITY)
        self.assertEqual(enc.encoding_type, EncodingType.IDENTITY)

    def test_get_unregistered_returns_identity(self):
        enc = self.registry.get(EncodingType.CHINESE_THINK)
        self.assertEqual(enc.encoding_type, EncodingType.IDENTITY)

    def test_register_and_get(self):
        class DummyEncoding(ContentEncoding):
            @property
            def encoding_type(self):
                return EncodingType.CHINESE_THINK
            @property
            def name(self):
                return "Dummy"
            def encode_system_prompt(self, s, c):
                return s + "[DUMMY]"
            def encode_response(self, s, c):
                return s
            def decode_for_display(self, s, c):
                return s

        self.registry.register(DummyEncoding())
        enc = self.registry.get(EncodingType.CHINESE_THINK)
        self.assertEqual(enc.encoding_type, EncodingType.CHINESE_THINK)
        self.assertEqual(enc.name, "Dummy")

    def test_list_encodings(self):
        encodings = self.registry.list_encodings()
        self.assertTrue(len(encodings) >= 1)
        self.assertEqual(encodings[0]["type"], "identity")

    def test_unregister(self):
        self.assertFalse(self.registry.unregister(EncodingType.IDENTITY))

    def test_is_registered(self):
        self.assertTrue(self.registry.is_registered(EncodingType.IDENTITY))
        self.assertFalse(self.registry.is_registered(EncodingType.CHINESE_THINK))


class TestOrchestratorWithEncoding(unittest.TestCase):
    """测试 Orchestrator 集成编码层"""

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)
        self.store.add_chunk(ContextChunk(
            chunk_id="chunk_1",
            content="print('hello')",
            summary="打招呼代码",
            tokens=5,
            source="hello.py",
        ))

    def test_orchestrator_default_identity_encoding(self):
        orch = LCMOrchestratorV2(self.store)
        self.assertEqual(orch._encoding_type, EncodingType.IDENTITY)
        self.assertIsNotNone(orch._content_encoding)

    def test_orchestrator_with_custom_encoding(self):
        class MarkEncoding(ContentEncoding):
            @property
            def encoding_type(self):
                return EncodingType.CUSTOM
            @property
            def name(self):
                return "Mark"
            def encode_system_prompt(self, s, c):
                return s + "\n[MARKED]"
            def encode_response(self, s, c):
                return s + "[MARKED]"
            def decode_for_display(self, s, c):
                return s

        encoder = MarkEncoding()
        orch = LCMOrchestratorV2(self.store, content_encoding=encoder, encoding_type=EncodingType.CUSTOM)
        self.assertEqual(orch._content_encoding, encoder)

    def test_encode_system_prompt_hook(self):
        class AppendEncoding(ContentEncoding):
            @property
            def encoding_type(self):
                return EncodingType.CUSTOM
            @property
            def name(self):
                return "Append"
            def encode_system_prompt(self, s, c):
                return s + "\n[APPENDED]"
            def encode_response(self, s, c):
                return s
            def decode_for_display(self, s, c):
                return s

        encoder = AppendEncoding()
        orch = LCMOrchestratorV2(self.store, content_encoding=encoder, encoding_type=EncodingType.CUSTOM)

        messages = [
            {"role": "system", "content": "原始系统提示"},
            {"role": "user", "content": "用户问题"},
        ]

        # 验证编码后的 system prompt 包含追加内容
        encoded = orch._encode_system_prompt(messages[0]["content"])
        self.assertIn("[APPENDED]", encoded)

    def test_encode_response_hook(self):
        class UpperEncoding(ContentEncoding):
            @property
            def encoding_type(self):
                return EncodingType.CUSTOM
            @property
            def name(self):
                return "Upper"
            def encode_system_prompt(self, s, c):
                return s
            def encode_response(self, s, c):
                return s.upper()
            def decode_for_display(self, s, c):
                return s

        encoder = UpperEncoding()
        orch = LCMOrchestratorV2(self.store, content_encoding=encoder, encoding_type=EncodingType.CUSTOM)

        result = orch._encode_response("hello world")
        self.assertEqual(result, "HELLO WORLD")

    def test_encoding_context_building(self):
        orch = LCMOrchestratorV2(self.store)
        orch.new_session("test_session")
        orch._round = 3

        ctx = orch._build_encoding_context()
        self.assertEqual(ctx.session_id, "test_session")
        self.assertEqual(ctx.current_round, 3)


class TestChineseThinkEncoding(unittest.TestCase):
    """测试 Chinese-Think 编码插件"""

    def setUp(self):
        self.store = ChunkStoreV2(enable_persistence=False)

    def test_chinese_think_encoding_import(self):
        try:
            from lcm_v2.encodings.chinese_think_encoding import ChineseThinkEncoding
            self.assertTrue(True)
        except ImportError:
            self.skipTest("Chinese-Think 编码未安装")

    def test_chinese_think_encoding_type(self):
        try:
            from lcm_v2.encodings.chinese_think_encoding import ChineseThinkEncoding
            enc = ChineseThinkEncoding()
            self.assertEqual(enc.encoding_type.value, "chinese-think")
        except ImportError:
            self.skipTest("Chinese-Think 编码未安装")

    def test_chinese_think_system_prompt_encoding(self):
        try:
            from lcm_v2.encodings.chinese_think_encoding import ChineseThinkEncoding
            enc = ChineseThinkEncoding(mode="compact")
            ctx = EncodingContext()

            original = "LCM 系统提示词"
            encoded = enc.encode_system_prompt(original, ctx)

            # 原始提示词应保留
            self.assertIn("LCM", encoded)
            # 验证编码器实例化成功（能执行到此处说明导入成功）
            self.assertEqual(enc.encoding_type.value, "chinese-think")
        except ImportError:
            self.skipTest("Chinese-Think 编码未安装")

    def test_chinese_think_off_mode(self):
        try:
            from lcm_v2.encodings.chinese_think_encoding import ChineseThinkEncoding
            enc = ChineseThinkEncoding(mode="off")
            ctx = EncodingContext()

            original = "LCM 系统提示词"
            encoded = enc.encode_system_prompt(original, ctx)

            # off 模式应不做任何修改
            self.assertEqual(encoded, original)
        except ImportError:
            self.skipTest("Chinese-Think 编码未安装")

    def test_chinese_think_response_encoding(self):
        try:
            from lcm_v2.encodings.chinese_think_encoding import ChineseThinkEncoding
            enc = ChineseThinkEncoding(mode="compact")
            ctx = EncodingContext()

            original = "好的，我认为这个问题可能是因为参数错误。"
            encoded = enc.encode_response(original, ctx)

            # 响应应被压缩或保持不变（取决于是否有 chinese-think-skills 包）
            # 有完整包时：压缩后长度应小于等于原始
            # 降级实现时：长度不变
            self.assertLessEqual(len(encoded), len(original))
            # 无论哪种实现，原始语义应保留（至少包含核心词）
            self.assertIn("参数错误", encoded)
        except ImportError:
            self.skipTest("Chinese-Think 编码未安装")

    def test_chinese_think_sentinel_protection(self):
        try:
            from lcm_v2.encodings.chinese_think_encoding import ChineseThinkEncoding
            enc = ChineseThinkEncoding(mode="compact")
            ctx = EncodingContext()

            original = "查看代码。[NEED_CHUNK:auth_handler]"
            encoded = enc.encode_response(original, ctx)

            # 哨兵标记应被保护
            self.assertIn("[NEED_CHUNK:auth_handler]", encoded)
        except ImportError:
            self.skipTest("Chinese-Think 编码未安装")


class TestGlobalRegistry(unittest.TestCase):
    """测试全局注册表"""

    def test_get_default_registry(self):
        registry = get_default_registry()
        self.assertIsNotNone(registry)
        self.assertTrue(registry.is_registered(EncodingType.IDENTITY))

    def test_register_and_get_global(self):
        class TestEncoding(ContentEncoding):
            @property
            def encoding_type(self):
                return EncodingType.CUSTOM
            @property
            def name(self):
                return "Test"
            def encode_system_prompt(self, s, c):
                return s
            def encode_response(self, s, c):
                return s
            def decode_for_display(self, s, c):
                return s

        register_encoding(TestEncoding())
        enc = get_encoding(EncodingType.CUSTOM)
        self.assertEqual(enc.encoding_type, EncodingType.CUSTOM)


if __name__ == "__main__":
    unittest.main()
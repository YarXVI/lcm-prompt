"""
Unit tests for LCM core components.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lcm import (
    ChunkStore,
    SentinelDetector,
    LCMOrchestrator,
    ContextChunk,
    LCMState,
    build_initial_messages,
)


class TestChunkStore(unittest.TestCase):
    def setUp(self):
        self.store = ChunkStore()

    def test_add_and_get(self):
        chunk = ContextChunk(chunk_id="test", content="hello world", summary="A test chunk")
        self.store.add_chunk(chunk)
        retrieved = self.store.get("test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "hello world")

    def test_get_missing(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_find_related(self):
        self.store.add_chunk(ContextChunk(
            chunk_id="auth", content="...",
            summary="authentication module handles login and user sessions"
        ))
        self.store.add_chunk(ContextChunk(
            chunk_id="db", content="...",
            summary="database layer handles queries and user data storage"
        ))
        self.store.add_chunk(ContextChunk(
            chunk_id="ui", content="...",
            summary="user interface renders HTML templates and CSS styles"
        ))
        related = self.store.find_related("auth", top_k=2)
        self.assertIn("db", related)


class TestSentinelDetector(unittest.TestCase):
    def setUp(self):
        self.detector = SentinelDetector()

    def test_detect_single(self):
        requests = self.detector.feed("Analyzing... [NEED_CHUNK:auth]")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].chunk_id, "auth")

    def test_detect_multiple(self):
        requests = self.detector.feed("Need [NEED_CHUNK:auth] and [NEED_CHUNK:db]")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].chunk_id, "auth")
        self.assertEqual(requests[1].chunk_id, "db")

    def test_no_sentinel(self):
        requests = self.detector.feed("Hello world, no markers here.")
        self.assertEqual(len(requests), 0)

    def test_cross_chunk_detection(self):
        r1 = self.detector.feed("Let me check [NEED_CHUNK:au")
        self.assertEqual(len(r1), 0)
        r2 = self.detector.feed("th_module] for details.")
        self.assertEqual(len(r2), 1)
        self.assertEqual(r2[0].chunk_id, "auth_module")

    def test_reset(self):
        self.detector.feed("[NEED_CHUNK:test]")
        self.detector.reset()
        requests = self.detector.feed("Clean slate")
        self.assertEqual(len(requests), 0)


class TestBuildInitialMessages(unittest.TestCase):
    def setUp(self):
        self.store = ChunkStore()
        self.store.add_chunk(ContextChunk(
            chunk_id="module_a",
            content="full content A",
            summary="Module A handles authentication",
            tokens=100,
            source="src",
        ))

    def test_build_messages(self):
        messages = build_initial_messages("Review security", self.store)
        self.assertGreater(len(messages), 0)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("[NEED_CHUNK:", messages[0]["content"])
        self.assertIn("module_a", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Review security")


class TestLCMOrchestrator(unittest.TestCase):
    def setUp(self):
        self.store = ChunkStore()
        self.store.add_chunk(ContextChunk(
            chunk_id="code_chunk",
            content="SECRET_KEY = 'sk-leaked'",
            summary="Code chunk with hardcoded secret",
            tokens=50,
            source="code",
        ))

    def test_single_round_no_sentinel(self):
        orchestrator = LCMOrchestrator(self.store)
        messages = build_initial_messages("Say hello", self.store)

        def fake_stream(msgs):
            yield "Hello, world!"

        result = list(orchestrator.run_stream(messages, fake_stream))
        self.assertEqual("".join(result), "Hello, world!")
        self.assertEqual(orchestrator.session.state, LCMState.COMPLETED)

    def test_sentinel_triggers_chunk_load(self):
        orchestrator = LCMOrchestrator(self.store)
        orchestrator.prefetch_enabled = False
        messages = build_initial_messages("Review code", self.store)

        round_count = [0]

        def fake_stream_with_sentinel(msgs):
            round_count[0] += 1
            if round_count[0] == 1:
                yield "[NEED_CHUNK:code_chunk]"
            else:
                yield "Found: hardcoded secret key!"

        result = list(orchestrator.run_stream(messages, fake_stream_with_sentinel))
        full = "".join(result)
        self.assertIn("hardcoded secret", full)
        self.assertEqual(orchestrator.session.total_chunks_loaded, 1)
        self.assertEqual(orchestrator.session.state, LCMState.COMPLETED)

    def test_max_rounds_limit(self):
        orchestrator = LCMOrchestrator(self.store)
        orchestrator.prefetch_enabled = False
        messages = build_initial_messages("Review code", self.store)

        def endless_sentinel(msgs):
            yield "[NEED_CHUNK:code_chunk]"

        with self.assertRaises(RuntimeError):
            list(orchestrator.run_stream(messages, endless_sentinel, max_rounds=3))


if __name__ == "__main__":
    unittest.main()

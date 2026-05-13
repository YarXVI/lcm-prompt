"""
LCM v2 SQLite 持久化存储
替代 JSONL，提供索引查询和事务支持
"""
import sqlite3
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from .lcm_types import ContextChunk
from .logger import get_logger

logger = get_logger()


class SQLiteChunkStore:
    """基于 SQLite 的 Chunk 存储，替代 JSONL"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (Path.home() / ".iris" / "lcm_chunks.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT,
                    tokens INTEGER DEFAULT 0,
                    load_count INTEGER DEFAULT 0,
                    last_loaded_at TEXT,
                    priority INTEGER DEFAULT 0,
                    source TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    version INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_priority ON chunks(priority)
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id, summary, content,
                    content='chunks',
                    content_rowid='rowid'
                )
            """)
            conn.commit()
            logger.info("SQLite 存储初始化完成", db_path=str(self.db_path))

    def add_chunk(self, chunk: ContextChunk) -> None:
        """添加或更新 chunk"""
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, content, summary, tokens, load_count, last_loaded_at,
                 priority, source, metadata, created_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id, chunk.content, chunk.summary, chunk.tokens,
                chunk.load_count,
                chunk.last_loaded_at.isoformat() if chunk.last_loaded_at else None,
                chunk.priority, chunk.source,
                json.dumps(chunk.metadata, ensure_ascii=False),
                chunk.created_at.isoformat(), chunk.version,
            ))
            conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """获取单个 chunk"""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if row:
                return self._row_to_chunk(row)
            return None

    def search(self, query: str, limit: int = 10) -> List[ContextChunk]:
        """全文搜索"""
        with self._lock:
            conn = self._get_conn()
            # 使用 FTS5 全文搜索
            rows = conn.execute("""
                SELECT c.* FROM chunks c
                JOIN chunk_fts fts ON c.rowid = fts.rowid
                WHERE chunk_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def list_all(self) -> List[ContextChunk]:
        """列出所有 chunks"""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT * FROM chunks").fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def delete_chunk(self, chunk_id: str) -> bool:
        """删除 chunk"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        with self._lock:
            conn = self._get_conn()
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_tokens = conn.execute("SELECT COALESCE(SUM(tokens), 0) FROM chunks").fetchone()[0]
            return {
                "total_chunks": count,
                "total_tokens": total_tokens,
                "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
            }

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ContextChunk:
        """将数据库行转为 ContextChunk"""
        from datetime import datetime
        return ContextChunk(
            chunk_id=row["chunk_id"],
            content=row["content"],
            summary=row["summary"] or "",
            tokens=row["tokens"] or 0,
            load_count=row["load_count"] or 0,
            last_loaded_at=datetime.fromisoformat(row["last_loaded_at"]) if row["last_loaded_at"] else None,
            priority=row["priority"] or 0,
            source=row["source"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            version=row["version"] or 1,
        )

    def close(self):
        """关闭连接"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

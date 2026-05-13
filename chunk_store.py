import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class Chunk:

    def __init__(
        self,
        chunk_id: str,
        content: str,
        summary: str = "",
        tokens: int = 0,
        source: str = "",
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.summary = summary
        self.tokens = tokens or self._estimate_tokens(content)
        self.source = source
        self.priority = priority
        self.metadata = metadata or {}
        self.load_count = 0
        self.last_loaded_at: Optional[datetime] = None
        self.created_at = datetime.now()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass
        import re
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_words = len(re.findall(r'[a-zA-Z]+', text))
        others = max(0, len(text) - cn_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text)))
        return int(cn_chars * 1.5 + en_words * 1.3 + others * 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "summary": self.summary,
            "tokens": self.tokens,
            "source": self.source,
            "priority": self.priority,
            "metadata": self.metadata,
            "load_count": self.load_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        chunk = cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            summary=data.get("summary", ""),
            tokens=data.get("tokens", 0),
            source=data.get("source", ""),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )
        chunk.load_count = data.get("load_count", 0)
        return chunk


class ChunkStore:

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_cache_size: int = 100,
        enable_persistence: bool = True,
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".lcm" / "chunks"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.storage_dir / "chunks.jsonl"

        self._chunks: Dict[str, Chunk] = {}
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, Chunk] = OrderedDict()
        self._max_cache_size = max_cache_size
        self._enable_persistence = enable_persistence

        if enable_persistence:
            self._load_from_disk()

    def add(self, chunk: Chunk) -> None:
        with self._lock:
            self._chunks[chunk.chunk_id] = chunk
            self._update_cache(chunk.chunk_id)
            if self._enable_persistence:
                self._append_to_disk(chunk)

    def add_many(self, chunks: List[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
                self._update_cache(chunk.chunk_id)
            if self._enable_persistence:
                self._save_to_disk()

    def get(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if chunk:
                self._update_cache(chunk_id)
                chunk.load_count += 1
                chunk.last_loaded_at = datetime.now()
            return chunk

    def get_many(self, chunk_ids: List[str]) -> List[Chunk]:
        return [c for c in (self.get(cid) for cid in chunk_ids) if c]

    def list_all(self) -> List[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def list_summaries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "chunk_id": c.chunk_id,
                    "summary": c.summary,
                    "tokens": c.tokens,
                    "source": c.source,
                    "load_count": c.load_count,
                }
                for c in self._chunks.values()
            ]

    def remove(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
                self._cache.pop(chunk_id, None)
                if self._enable_persistence:
                    self._save_to_disk()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._cache.clear()
            if self._enable_persistence:
                self._save_to_disk()

    def _update_cache(self, chunk_id: str) -> None:
        if chunk_id in self._cache:
            self._cache.move_to_end(chunk_id)
        else:
            self._cache[chunk_id] = self._chunks[chunk_id]
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

    def _save_to_disk(self) -> None:
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            for chunk in self._chunks.values():
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def _append_to_disk(self, chunk: Chunk) -> None:
        with open(self.chunks_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def _load_from_disk(self) -> None:
        if not self.chunks_file.exists():
            return
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = Chunk.from_dict(data)
                    self._chunks[chunk.chunk_id] = chunk
                except (json.JSONDecodeError, KeyError):
                    continue

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_tokens = sum(c.tokens for c in self._chunks.values())
            return {
                "total_chunks": len(self._chunks),
                "cached_chunks": len(self._cache),
                "total_tokens": total_tokens,
                "avg_tokens": total_tokens / len(self._chunks) if self._chunks else 0,
            }

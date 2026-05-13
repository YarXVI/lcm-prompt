import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .ir_models import MultiGranularityIR, GrainLevel, Grain, IR_VERSION
from .encoder_base import EncodingContext
from .encoding_registry import EncodingRegistry
from .chunk_store import ChunkStore, Chunk


@dataclass
class EncodedChunk:
    chunk_id: str
    original: Chunk
    ir: MultiGranularityIR
    encoded_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "original": self.original.to_dict(),
            "ir": self.ir.to_dict(),
            "encoded_at": self.encoded_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncodedChunk":
        ir = MultiGranularityIR.from_dict(data["ir"])
        return cls(
            chunk_id=data["chunk_id"],
            original=Chunk.from_dict(data["original"]),
            ir=ir,
            encoded_at=datetime.fromisoformat(
                data.get("encoded_at", datetime.now().isoformat())
            ),
        )


class EncodedChunkStore:

    def __init__(
        self,
        chunk_store: Optional[ChunkStore] = None,
        encoding_registry: Optional[EncodingRegistry] = None,
        ir_storage_dir: Optional[Path] = None,
    ):
        self._store = chunk_store or ChunkStore()
        self._registry = encoding_registry or EncodingRegistry()
        self._ir_dir = Path(ir_storage_dir) if ir_storage_dir else self._store.storage_dir / "ir"
        self._ir_dir.mkdir(parents=True, exist_ok=True)

        self._ir_cache: Dict[str, MultiGranularityIR] = {}
        self._lock = threading.RLock()
        self._warmup_done = False

        self._load_ir_cache_from_disk()

    def add(self, chunk: Chunk) -> EncodedChunk:
        self._store.add(chunk)
        return EncodedChunk(
            chunk_id=chunk.chunk_id,
            original=chunk,
            ir=self._encode_lazy(chunk),
        )

    def get_encoded(self, chunk_id: str) -> Optional[EncodedChunk]:
        chunk = self._store.get(chunk_id)
        if not chunk:
            return None

        ir = self._get_or_encode_ir(chunk)
        return EncodedChunk(chunk_id=chunk_id, original=chunk, ir=ir)

    def get_grain(self, chunk_id: str, level: GrainLevel) -> Optional[Grain]:
        encoded = self.get_encoded(chunk_id)
        if not encoded:
            return None
        return encoded.ir.grains.get(level)

    def best_grain_for(self, chunk_id: str, available_tokens: int) -> Optional[Grain]:
        encoded = self.get_encoded(chunk_id)
        if not encoded:
            return None
        return encoded.ir.best_grain_for(available_tokens)

    def warmup(self) -> None:
        if self._warmup_done:
            return

        def _warmup_worker():
            for chunk in self._store.list_all():
                with self._lock:
                    if chunk.chunk_id in self._ir_cache:
                        continue
                self._encode_and_cache(chunk)
            self._warmup_done = True

        t = threading.Thread(target=_warmup_worker, daemon=True)
        t.start()

    def _ir_content_matches(self, ir: MultiGranularityIR, chunk: Chunk) -> bool:
        full_grain = ir.grains.get(GrainLevel.FULL)
        if full_grain and full_grain.reversible:
            return full_grain.content == chunk.content
        return ir.grains.get(GrainLevel.FULL) is not None

    def _get_or_encode_ir(self, chunk: Chunk) -> MultiGranularityIR:
        with self._lock:
            if chunk.chunk_id in self._ir_cache:
                ir = self._ir_cache[chunk.chunk_id]
                if ir.is_valid and ir.encoding_type != "_stale":
                    best_encoder = self._registry.detect_best(chunk.content)
                    if ir.encoding_type == best_encoder.encoding_type and self._ir_content_matches(ir, chunk):
                        return ir

        return self._encode_and_cache(chunk)

    def _encode_lazy(self, chunk: Chunk) -> MultiGranularityIR:
        with self._lock:
            if chunk.chunk_id in self._ir_cache:
                ir = self._ir_cache[chunk.chunk_id]
                if ir.is_valid and ir.encoding_type != "_stale":
                    best_encoder = self._registry.detect_best(chunk.content)
                    if ir.encoding_type == best_encoder.encoding_type and self._ir_content_matches(ir, chunk):
                        return ir

        return self._encode_and_cache(chunk)

    def _encode_and_cache(self, chunk: Chunk) -> MultiGranularityIR:
        encoder = self._registry.detect_best(chunk.content)
        ctx = EncodingContext(chunk_id=chunk.chunk_id, source=chunk.source)
        ir = encoder.encode(chunk.content, ctx)

        with self._lock:
            self._ir_cache[chunk.chunk_id] = ir

        self._persist_ir(chunk.chunk_id, ir)
        return ir

    def _persist_ir(self, chunk_id: str, ir: MultiGranularityIR) -> None:
        ir_file = self._ir_dir / f"{chunk_id}.json"
        try:
            with open(ir_file, "w", encoding="utf-8") as f:
                f.write(ir.to_json())
        except OSError:
            pass

    def _load_ir_cache_from_disk(self) -> None:
        if not self._ir_dir.exists():
            return
        for ir_file in self._ir_dir.glob("*.json"):
            try:
                with open(ir_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ir = MultiGranularityIR.from_dict(data)
                if ir.is_valid and ir.encoding_type != "_stale":
                    chunk_id = ir_file.stem
                    chunk = self._store.get(chunk_id)
                    if chunk:
                        best_encoder = self._registry.detect_best(chunk.content)
                        if ir.encoding_type == best_encoder.encoding_type and self._ir_content_matches(ir, chunk):
                            with self._lock:
                                self._ir_cache[chunk_id] = ir
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    @property
    def raw_store(self) -> ChunkStore:
        return self._store

    @property
    def registry(self) -> EncodingRegistry:
        return self._registry

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            encoded_count = len(self._ir_cache)
        return {
            **self._store.get_stats(),
            "encoded_chunks": encoded_count,
            "warmup_done": self._warmup_done,
            "registered_encoders": self._registry.list_encoders(),
        }

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from .ir_models import GrainLevel, MultiGranularityIR


@dataclass
class GoldenSample:
    chunk_id: str
    content_hash: str
    content: str
    ir_json: str
    encoding_type: str
    source_language: str
    collected_at: float = field(default_factory=time.time)
    injection_count: int = 0
    upgrade_count: int = 0

    @property
    def is_verified_golden(self) -> bool:
        return self.injection_count >= 3 and self.upgrade_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash,
            "content": self.content,
            "ir_json": self.ir_json,
            "encoding_type": self.encoding_type,
            "source_language": self.source_language,
            "collected_at": self.collected_at,
            "injection_count": self.injection_count,
            "upgrade_count": self.upgrade_count,
            "is_verified_golden": self.is_verified_golden,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenSample":
        return cls(
            chunk_id=data["chunk_id"],
            content_hash=data["content_hash"],
            content=data["content"],
            ir_json=data["ir_json"],
            encoding_type=data["encoding_type"],
            source_language=data.get("source_language", "unknown"),
            collected_at=data.get("collected_at", 0),
            injection_count=data.get("injection_count", 0),
            upgrade_count=data.get("upgrade_count", 0),
        )


class GoldenCorpusCollector:

    MIN_INJECTIONS = 3

    def __init__(self, storage_dir: Optional[str] = None):
        self._samples: Dict[str, GoldenSample] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None
        if self._storage_dir:
            self._load_from_disk()

    def consider(
        self,
        chunk_id: str,
        content: str,
        ir: MultiGranularityIR,
        injection_count: int,
        upgrade_count: int,
    ) -> bool:
        if injection_count < self.MIN_INJECTIONS:
            return False
        if upgrade_count > 0:
            return False
        if chunk_id in self._samples:
            existing = self._samples[chunk_id]
            if existing.injection_count >= injection_count:
                return False

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        sample = GoldenSample(
            chunk_id=chunk_id,
            content_hash=content_hash,
            content=content,
            ir_json=ir.to_json(),
            encoding_type=ir.encoding_type,
            source_language=ir.source_language,
            injection_count=injection_count,
            upgrade_count=upgrade_count,
        )
        self._samples[chunk_id] = sample
        self._persist_sample(sample)
        return True

    def get_sample(self, chunk_id: str) -> Optional[GoldenSample]:
        return self._samples.get(chunk_id)

    def list_samples(self) -> List[GoldenSample]:
        return list(self._samples.values())

    def get_verified(self) -> List[GoldenSample]:
        return [s for s in self._samples.values() if s.is_verified_golden]

    def get_by_encoding_type(self, encoding_type: str) -> List[GoldenSample]:
        return [s for s in self._samples.values() if s.encoding_type == encoding_type]

    def get_stats(self) -> Dict[str, Any]:
        verified = self.get_verified()
        encoding_dist: Dict[str, int] = {}
        for s in self._samples.values():
            encoding_dist[s.encoding_type] = encoding_dist.get(s.encoding_type, 0) + 1
        return {
            "total_samples": len(self._samples),
            "verified_golden": len(verified),
            "encoding_distribution": encoding_dist,
            "target": 50,
            "progress_pct": round(len(verified) / 50 * 100, 1),
        }

    def export_corpus(self, output_path: str) -> int:
        verified = self.get_verified()
        data = [s.to_dict() for s in verified]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return len(data)

    def _persist_sample(self, sample: GoldenSample) -> None:
        if not self._storage_dir:
            return
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / f"{sample.chunk_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_from_disk(self) -> None:
        if not self._storage_dir or not self._storage_dir.exists():
            return
        for f in self._storage_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                sample = GoldenSample.from_dict(data)
                self._samples[sample.chunk_id] = sample
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

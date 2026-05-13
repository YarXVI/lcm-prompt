import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class ChunkURRStats:
    chunk_id: str
    total_injections: int = 0
    upgrade_requests: int = 0
    downgrade_requests: int = 0
    grain_distribution: Dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0

    @property
    def urr(self) -> float:
        return self.upgrade_requests / max(self.total_injections, 1)

    @property
    def is_high_urr(self) -> bool:
        return self.urr > 0.4

    @property
    def is_golden(self) -> bool:
        return self.total_injections >= 3 and self.upgrade_requests == 0

    def record_injection(self, grain_level: str, latency_ms: float, is_upgrade: bool) -> None:
        self.total_injections += 1
        self.grain_distribution[grain_level] = self.grain_distribution.get(grain_level, 0) + 1
        self._latency_sum += latency_ms
        self.avg_latency_ms = self._latency_sum / self.total_injections
        if is_upgrade:
            self.upgrade_requests += 1

    def record_downgrade(self) -> None:
        self.downgrade_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "total_injections": self.total_injections,
            "upgrade_requests": self.upgrade_requests,
            "downgrade_requests": self.downgrade_requests,
            "urr": round(self.urr, 4),
            "is_high_urr": self.is_high_urr,
            "is_golden": self.is_golden,
            "grain_distribution": self.grain_distribution,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class URRReporter:

    def __init__(self, storage_dir: Optional[str] = None):
        self._stats: Dict[str, ChunkURRStats] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._session_count = 0

    def record_injection(self, chunk_id: str, grain_level: str, latency_ms: float, is_upgrade: bool) -> None:
        if chunk_id not in self._stats:
            self._stats[chunk_id] = ChunkURRStats(chunk_id=chunk_id)
        self._stats[chunk_id].record_injection(grain_level, latency_ms, is_upgrade)

    def record_downgrade(self, chunk_id: str) -> None:
        if chunk_id not in self._stats:
            self._stats[chunk_id] = ChunkURRStats(chunk_id=chunk_id)
        self._stats[chunk_id].record_downgrade()

    def record_session(self) -> None:
        self._session_count += 1

    def get_chunk_stats(self, chunk_id: str) -> Optional[ChunkURRStats]:
        return self._stats.get(chunk_id)

    def get_high_urr_chunks(self, threshold: float = 0.4) -> List[ChunkURRStats]:
        return [s for s in self._stats.values() if s.urr > threshold]

    def get_golden_chunks(self, min_injections: int = 3) -> List[ChunkURRStats]:
        return [s for s in self._stats.values() if s.is_golden and s.total_injections >= min_injections]

    def get_report(self) -> Dict[str, Any]:
        all_stats = [s.to_dict() for s in self._stats.values()]
        total_injections = sum(s.total_injections for s in self._stats.values())
        total_upgrades = sum(s.upgrade_requests for s in self._stats.values())
        global_urr = total_upgrades / max(total_injections, 1)

        return {
            "timestamp": time.time(),
            "session_count": self._session_count,
            "total_chunks": len(self._stats),
            "total_injections": total_injections,
            "total_upgrades": total_upgrades,
            "global_urr": round(global_urr, 4),
            "high_urr_chunks": len(self.get_high_urr_chunks()),
            "golden_chunks": len(self.get_golden_chunks()),
            "chunk_details": all_stats,
        }

    def persist(self) -> None:
        if not self._storage_dir:
            return
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        report = self.get_report()
        ts = int(time.time())
        path = self._storage_dir / f"urr_report_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        if not self._storage_dir or not self._storage_dir.exists():
            return None
        reports = sorted(self._storage_dir.glob("urr_report_*.json"))
        if not reports:
            return None
        with open(reports[-1], "r", encoding="utf-8") as f:
            return json.load(f)

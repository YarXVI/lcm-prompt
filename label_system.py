import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class Anchor:
    name: str
    start_line: int
    end_line: int
    content_preview: str
    semantic_tag: str = ""
    grain_hint: str = "detail"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_preview": self.content_preview[:100],
            "semantic_tag": self.semantic_tag,
            "grain_hint": self.grain_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Anchor":
        return cls(
            name=data["name"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            content_preview=data.get("content_preview", ""),
            semantic_tag=data.get("semantic_tag", ""),
            grain_hint=data.get("grain_hint", "detail"),
        )


@dataclass
class ChunkLabel:
    chunk_id: str
    anchors: List[Anchor] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    priority: int = 0
    annotated_by: str = ""
    annotated_at: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def coverage(self) -> float:
        if not self.anchors:
            return 0.0
        total_lines = max(a.end_line - a.start_line + 1 for a in self.anchors)
        covered = sum(a.end_line - a.start_line + 1 for a in self.anchors)
        return min(covered / max(total_lines, 1), 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "anchors": [a.to_dict() for a in self.anchors],
            "tags": self.tags,
            "priority": self.priority,
            "annotated_by": self.annotated_by,
            "annotated_at": self.annotated_at,
            "notes": self.notes,
            "coverage": round(self.coverage, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkLabel":
        return cls(
            chunk_id=data["chunk_id"],
            anchors=[Anchor.from_dict(a) for a in data.get("anchors", [])],
            tags=data.get("tags", []),
            priority=data.get("priority", 0),
            annotated_by=data.get("annotated_by", ""),
            annotated_at=data.get("annotated_at", 0),
            notes=data.get("notes", ""),
        )


class LabelStore:

    def __init__(self, storage_dir: Optional[str] = None):
        self._labels: Dict[str, ChunkLabel] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None
        if self._storage_dir:
            self._load_from_disk()

    def add_label(self, label: ChunkLabel) -> None:
        self._labels[label.chunk_id] = label
        self._persist_label(label)

    def get_label(self, chunk_id: str) -> Optional[ChunkLabel]:
        return self._labels.get(chunk_id)

    def remove_label(self, chunk_id: str) -> bool:
        if chunk_id in self._labels:
            del self._labels[chunk_id]
            self._delete_label_file(chunk_id)
            return True
        return False

    def list_labels(self) -> List[ChunkLabel]:
        return list(self._labels.values())

    def get_high_priority(self, limit: int = 20) -> List[ChunkLabel]:
        sorted_labels = sorted(self._labels.values(), key=lambda l: l.priority, reverse=True)
        return sorted_labels[:limit]

    def get_by_tag(self, tag: str) -> List[ChunkLabel]:
        return [l for l in self._labels.values() if tag in l.tags]

    def get_coverage_stats(self) -> Dict[str, Any]:
        if not self._labels:
            return {"total_labeled": 0, "avg_coverage": 0.0, "tag_distribution": {}}
        coverages = [l.coverage for l in self._labels.values()]
        tag_dist: Dict[str, int] = {}
        for l in self._labels.values():
            for t in l.tags:
                tag_dist[t] = tag_dist.get(t, 0) + 1
        return {
            "total_labeled": len(self._labels),
            "avg_coverage": round(sum(coverages) / len(coverages), 2),
            "tag_distribution": tag_dist,
            "total_anchors": sum(len(l.anchors) for l in self._labels.values()),
        }

    def _persist_label(self, label: ChunkLabel) -> None:
        if not self._storage_dir:
            return
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / f"{label.chunk_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(label.to_dict(), f, ensure_ascii=False, indent=2)

    def _delete_label_file(self, chunk_id: str) -> None:
        if not self._storage_dir:
            return
        path = self._storage_dir / f"{chunk_id}.json"
        if path.exists():
            path.unlink()

    def _load_from_disk(self) -> None:
        if not self._storage_dir or not self._storage_dir.exists():
            return
        for f in self._storage_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                label = ChunkLabel.from_dict(data)
                self._labels[label.chunk_id] = label
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

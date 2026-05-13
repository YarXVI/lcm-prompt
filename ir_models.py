import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


IR_VERSION = 1


class GrainLevel(str, Enum):
    KEYWORDS = "keywords"
    SUMMARY = "summary"
    DETAIL = "detail"
    FULL = "full"

    @property
    def rank(self) -> int:
        return {"keywords": 0, "summary": 1, "detail": 2, "full": 3}[self.value]

    def finer_or_equal(self, other: "GrainLevel") -> bool:
        return self.rank >= other.rank


@dataclass
class Grain:
    level: GrainLevel
    content: str
    tokens: int
    reversible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "content": self.content,
            "tokens": self.tokens,
            "reversible": self.reversible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Grain":
        return cls(
            level=GrainLevel(data["level"]),
            content=data["content"],
            tokens=data["tokens"],
            reversible=data.get("reversible", False),
        )


@dataclass
class MultiGranularityIR:
    encoding_type: str
    source_language: str
    grains: Dict[GrainLevel, Grain] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    encoding_metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = IR_VERSION

    @property
    def compression_ratios(self) -> Dict[str, float]:
        full_grain = self.grains.get(GrainLevel.FULL)
        base = full_grain.tokens if full_grain and full_grain.tokens > 0 else 1
        return {
            g.level.value: round(g.tokens / base, 4)
            for g in self.grains.values()
        }

    @property
    def is_valid(self) -> bool:
        return self.version == IR_VERSION

    def best_grain_for(self, available_tokens: int) -> Grain:
        for level in [GrainLevel.DETAIL, GrainLevel.SUMMARY, GrainLevel.KEYWORDS]:
            grain = self.grains.get(level)
            if grain and grain.tokens <= available_tokens:
                return grain
        fallback = self.grains.get(GrainLevel.KEYWORDS)
        if fallback:
            return fallback
        return self.grains.get(GrainLevel.FULL, Grain(GrainLevel.FULL, "", 0))

    def grain_at_least(self, min_level: GrainLevel) -> Optional[Grain]:
        for level in list(GrainLevel):
            grain = self.grains.get(level)
            if grain and level.finer_or_equal(min_level):
                return grain
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoding_type": self.encoding_type,
            "source_language": self.source_language,
            "grains": {k.value: v.to_dict() for k, v in self.grains.items()},
            "structure": self.structure,
            "encoding_metadata": self.encoding_metadata,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiGranularityIR":
        version = data.get("version", 0)
        if version != IR_VERSION:
            return cls(
                encoding_type="_stale",
                source_language="unknown",
                version=version,
            )
        grains = {}
        for k, v in data.get("grains", {}).items():
            level = GrainLevel(k)
            grains[level] = Grain.from_dict(v)
        return cls(
            encoding_type=data["encoding_type"],
            source_language=data.get("source_language", "unknown"),
            grains=grains,
            structure=data.get("structure", {}),
            encoding_metadata=data.get("encoding_metadata", {}),
            version=version,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MultiGranularityIR":
        return cls.from_dict(json.loads(json_str))

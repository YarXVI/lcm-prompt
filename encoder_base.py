from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .ir_models import MultiGranularityIR, GrainLevel, Grain, IR_VERSION


@dataclass
class EncodingContext:
    chunk_id: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentEncoder(ABC):

    @property
    @abstractmethod
    def encoding_type(self) -> str:
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        pass

    @property
    def ir_version(self) -> int:
        return IR_VERSION

    @abstractmethod
    def detect(self, text: str) -> float:
        pass

    @abstractmethod
    def encode(self, text: str, context: EncodingContext) -> MultiGranularityIR:
        pass

    def encode_system_hint(self) -> str:
        return ""


class ContentDecoder:

    def decode(
        self,
        ir: MultiGranularityIR,
        available_tokens: int,
        min_level: GrainLevel = GrainLevel.KEYWORDS,
    ) -> tuple[str, GrainLevel]:
        if min_level == GrainLevel.FULL:
            grain = ir.grains.get(GrainLevel.FULL)
            if grain:
                return grain.content, GrainLevel.FULL

        if min_level == GrainLevel.KEYWORDS:
            return self._decode_lazy(ir, available_tokens)
        else:
            return self._decode_explicit(ir, available_tokens, min_level)

    def _decode_lazy(
        self,
        ir: MultiGranularityIR,
        available_tokens: int,
    ) -> tuple[str, GrainLevel]:
        for level in reversed(list(GrainLevel)):
            grain = ir.grains.get(level)
            if grain and grain.tokens <= available_tokens:
                return grain.content, grain.level

        fallback = ir.best_grain_for(available_tokens)
        return fallback.content, fallback.level

    def _decode_explicit(
        self,
        ir: MultiGranularityIR,
        available_tokens: int,
        min_level: GrainLevel,
    ) -> tuple[str, GrainLevel]:
        coarsest_meeting_min = None

        for level in list(GrainLevel):
            grain = ir.grains.get(level)
            if not grain:
                continue
            if level.finer_or_equal(min_level):
                if coarsest_meeting_min is None:
                    coarsest_meeting_min = grain
                if grain.tokens <= available_tokens:
                    return grain.content, grain.level

        if coarsest_meeting_min:
            return coarsest_meeting_min.content, coarsest_meeting_min.level

        fallback = ir.best_grain_for(available_tokens)
        return fallback.content, fallback.level

from typing import Dict, Any, List, Optional

from .ir_models import MultiGranularityIR, GrainLevel, Grain, IR_VERSION
from .encoder_base import ContentEncoder, EncodingContext
from .chunk_store import Chunk


class IdentityEncoder(ContentEncoder):

    @property
    def encoding_type(self) -> str:
        return "identity"

    @property
    def supported_languages(self) -> List[str]:
        return ["*"]

    def detect(self, text: str) -> float:
        return 0.0

    def encode(self, text: str, context: EncodingContext) -> MultiGranularityIR:
        tokens = Chunk._estimate_tokens(text)
        return MultiGranularityIR(
            encoding_type="identity",
            source_language="unknown",
            grains={
                GrainLevel.KEYWORDS: Grain(
                    GrainLevel.KEYWORDS, text[:120].split("\n")[0],
                    min(50, tokens), reversible=False,
                ),
                GrainLevel.SUMMARY: Grain(
                    GrainLevel.SUMMARY, text[:600],
                    min(200, tokens), reversible=False,
                ),
                GrainLevel.FULL: Grain(
                    GrainLevel.FULL, text, tokens, reversible=True,
                ),
            },
            version=IR_VERSION,
        )


class EncodingRegistry:

    def __init__(self):
        self._encoders: Dict[str, ContentEncoder] = {}
        self._fallback = IdentityEncoder()

    def register(self, encoder: ContentEncoder) -> "EncodingRegistry":
        self._encoders[encoder.encoding_type] = encoder
        return self

    def unregister(self, encoding_type: str) -> bool:
        if encoding_type in self._encoders:
            del self._encoders[encoding_type]
            return True
        return False

    def get(self, encoding_type: str) -> Optional[ContentEncoder]:
        return self._encoders.get(encoding_type)

    def detect_best(self, text: str) -> ContentEncoder:
        best_score = -1.0
        best_encoder = self._fallback
        for encoder in self._encoders.values():
            score = encoder.detect(text)
            if score > best_score:
                best_score = score
                best_encoder = encoder
        return best_encoder

    def list_encoders(self) -> List[Dict[str, Any]]:
        return [
            {"type": et, "languages": enc.supported_languages}
            for et, enc in self._encoders.items()
        ]

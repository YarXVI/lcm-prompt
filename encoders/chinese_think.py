import re
from typing import List

from ..ir_models import MultiGranularityIR, GrainLevel, Grain, IR_VERSION
from ..encoder_base import ContentEncoder, EncodingContext
from ..chunk_store import Chunk


class ChineseThinkEncoder(ContentEncoder):

    _STOP_CHARS = frozenset("的了呢吧啊着过地得被把让给向往从在到对于与和及或是")

    @property
    def encoding_type(self) -> str:
        return "chinese-think"

    @property
    def supported_languages(self) -> List[str]:
        return ["zh", "mixed"]

    def detect(self, text: str) -> float:
        total = max(len(text.strip()), 1)
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return (zh / total) * 0.8

    def encode(self, text: str, context: EncodingContext) -> MultiGranularityIR:
        keywords = self._extract_keywords(text)
        summary = self._compress(text)
        detail = self._extract_structure(text)
        full_tokens = Chunk._estimate_tokens(text)

        return MultiGranularityIR(
            encoding_type=self.encoding_type,
            source_language="zh",
            grains={
                GrainLevel.KEYWORDS: Grain(
                    GrainLevel.KEYWORDS, keywords,
                    Chunk._estimate_tokens(keywords), reversible=False,
                ),
                GrainLevel.SUMMARY: Grain(
                    GrainLevel.SUMMARY, summary,
                    Chunk._estimate_tokens(summary), reversible=False,
                ),
                GrainLevel.DETAIL: Grain(
                    GrainLevel.DETAIL, detail,
                    Chunk._estimate_tokens(detail), reversible=False,
                ),
                GrainLevel.FULL: Grain(
                    GrainLevel.FULL, text, full_tokens, reversible=True,
                ),
            },
            version=IR_VERSION,
        )

    def _extract_keywords(self, text: str) -> str:
        words = []
        seen = set()
        for m in re.finditer(r'[\u4e00-\u9fff]{2,6}', text):
            w = m.group()
            if w[-1] in self._STOP_CHARS:
                w = w[:-1]
            if len(w) < 2 or w in seen:
                continue
            if all(c in self._STOP_CHARS for c in w):
                continue
            seen.add(w)
            words.append(w)
            if len(words) >= 15:
                break
        return ", ".join(words)

    def _compress(self, text: str) -> str:
        sentences = re.split(r'[。！？；\n]', text)
        core = [s.strip() for s in sentences if len(s.strip()) > 5][:5]
        return "。".join(core) + "。" if core else text[:300]

    def _extract_structure(self, text: str) -> str:
        parts = []
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("#") or s.startswith("##") or s.startswith("###"):
                parts.append(s.lstrip("#").strip())
            elif re.match(r'^[\d一二三四五六七八九十]+[.、）)]', s):
                parts.append(s[:100])
            if len(parts) >= 10:
                break
        if not parts:
            sentences = re.split(r'[。！？]', text)
            parts = [s.strip()[:80] for s in sentences if len(s.strip()) > 3][:8]
        return "\n".join(parts)

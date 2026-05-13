import re
from typing import List

from ..ir_models import MultiGranularityIR, GrainLevel, Grain, IR_VERSION
from ..encoder_base import ContentEncoder, EncodingContext
from ..chunk_store import Chunk


class EnglishLogicEncoder(ContentEncoder):

    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "and", "but", "or", "nor", "not",
        "so", "yet", "both", "either", "neither", "each", "every",
        "this", "that", "these", "those", "it", "its", "he", "she",
        "they", "them", "we", "you", "i", "me", "my", "your", "his",
        "her", "our", "their", "what", "which", "who", "whom", "how",
        "when", "where", "why", "if", "then", "else", "no", "yes",
        "all", "any", "some", "more", "most", "other", "such", "only",
        "just", "than", "too", "very", "also", "about", "up",
    })

    @property
    def encoding_type(self) -> str:
        return "en-logic"

    @property
    def supported_languages(self) -> List[str]:
        return ["en"]

    _CODE_PENALTY = re.compile(
        r'(?:^|\n)\s*(?:def |class |func |fn |pub |const |let |var |'
        r'import |from |return |if \(|for \(|try {|=> \{|'
        r'CREATE TABLE |SELECT .* FROM |INSERT INTO |'
        r'type \w+ struct|impl |struct \{|package )'
    )

    def detect(self, text: str) -> float:
        stripped = text.strip()
        total = max(len(stripped), 1)
        en_alpha = sum(1 for c in stripped if c.isascii() and c.isalpha())
        zh_chars = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
        non_content = sum(1 for c in stripped if c.isspace() or (not c.isalpha() and not '\u4e00' <= c <= '\u9fff'))
        effective_total = max(total - non_content, 1)
        base_score = (en_alpha / effective_total) * 0.6

        code_indicators = len(self._CODE_PENALTY.findall(text))
        if code_indicators > 0:
            total_lines = max(len(text.split("\n")), 1)
            code_ratio = code_indicators / total_lines
            base_score *= max(0.0, 1.0 - code_ratio * 3.0)

        return base_score

    def encode(self, text: str, context: EncodingContext) -> MultiGranularityIR:
        keywords = self._extract_keywords(text)
        summary = self._extract_summary(text)
        detail = self._extract_detail(text)
        full_tokens = Chunk._estimate_tokens(text)

        return MultiGranularityIR(
            encoding_type=self.encoding_type,
            source_language="en",
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
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if w not in self._STOP_WORDS:
                freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)[:15]
        return ", ".join(top)

    def _extract_summary(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 2:
            return text[:400]
        first_two = " ".join(sentences[:2])
        return first_two[:400]

    def _extract_detail(self, text: str) -> str:
        parts = []
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("#") or s.startswith("##"):
                parts.append(s.lstrip("#").strip())
            elif re.match(r'^\d+\.\s', s):
                parts.append(s[:120])
            elif s.startswith("- ") or s.startswith("* "):
                parts.append(s[:120])
            if len(parts) >= 12:
                break
        if not parts:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            parts = [s[:100] for s in sentences[:8]]
        return "\n".join(parts)

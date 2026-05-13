import re
from typing import List, Optional, Dict, Any

from .ir_models import GrainLevel, Grain, MultiGranularityIR


class SemanticSlicer:

    _SECTION_PATTERN = re.compile(r'^(#{1,4}\s|class\s|def\s|func\s|fn\s|pub\s|interface\s|type\s|impl\s|CREATE\sTABLE)', re.M)
    _BLANK_LINE = re.compile(r'\n\s*\n')

    def slice_by_sections(self, content: str, max_tokens: int = 300) -> str:
        sections = self._split_sections(content)
        result = []
        token_budget = max_tokens

        for section in sections:
            section_tokens = self._estimate_tokens(section)
            if section_tokens <= token_budget:
                result.append(section)
                token_budget -= section_tokens
            else:
                first_line = section.split("\n")[0]
                fl_tokens = self._estimate_tokens(first_line)
                if fl_tokens <= token_budget:
                    result.append(first_line)
                    token_budget -= fl_tokens
                break

        return "\n".join(result)

    def slice_by_intent(self, content: str, intent_keywords: List[str], max_tokens: int = 300) -> str:
        lines = content.split("\n")
        scored_lines = []

        for i, line in enumerate(lines):
            score = 0
            for kw in intent_keywords:
                if kw.lower() in line.lower():
                    score += 1
            scored_lines.append((i, line, score))

        scored_lines.sort(key=lambda x: x[2], reverse=True)

        selected_indices = set()
        total_tokens = 0

        for idx, line, score in scored_lines:
            if score == 0:
                break
            line_tokens = self._estimate_tokens(line)
            if total_tokens + line_tokens > max_tokens:
                break
            selected_indices.add(idx)
            total_tokens += line_tokens

        if not selected_indices:
            return self.slice_by_sections(content, max_tokens)

        result_lines = []
        for i in sorted(selected_indices):
            result_lines.append(lines[i])

        return "\n".join(result_lines)

    def slice_by_density(self, content: str, max_tokens: int = 300) -> str:
        lines = content.split("\n")
        block_size = 5
        blocks = []

        for i in range(0, len(lines), block_size):
            block = "\n".join(lines[i:i + block_size])
            info_density = self._compute_density(block)
            blocks.append((i, block, info_density))

        blocks.sort(key=lambda x: x[2], reverse=True)

        result = []
        total_tokens = 0

        for idx, block, density in blocks:
            block_tokens = self._estimate_tokens(block)
            if total_tokens + block_tokens > max_tokens:
                break
            result.append((idx, block))
            total_tokens += block_tokens

        result.sort(key=lambda x: x[0])
        return "\n".join(block for _, block in result)

    def _split_sections(self, content: str) -> List[str]:
        positions = [m.start() for m in self._SECTION_PATTERN.finditer(content)]
        if not positions:
            paragraphs = self._BLANK_LINE.split(content)
            return [p.strip() for p in paragraphs if p.strip()]

        sections = []
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(content)
            sections.append(content[pos:end].strip())

        return sections

    def _compute_density(self, text: str) -> float:
        if not text.strip():
            return 0.0
        meaningful = sum(1 for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        return meaningful / max(len(text), 1)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en = len(text) - zh
        return zh * 2 + en // 4

import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from .ir_models import GrainLevel, Grain, MultiGranularityIR
from .label_system import LabelStore, ChunkLabel, Anchor


@dataclass
class RenderedSlice:
    chunk_id: str
    anchor_name: str
    content: str
    tokens: int
    semantic_tag: str
    grain_hint: str
    render_method: str = "template"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "anchor_name": self.anchor_name,
            "content": self.content,
            "tokens": self.tokens,
            "semantic_tag": self.semantic_tag,
            "grain_hint": self.grain_hint,
            "render_method": self.render_method,
        }


class DynamicRenderer:

    _INTENT_PATTERNS = {
        "auth": re.compile(r"auth|login|token|session|password|verify|credential", re.I),
        "error": re.compile(r"error|exception|throw|raise|catch|fail", re.I),
        "config": re.compile(r"config|setting|env|variable|init", re.I),
        "data": re.compile(r"data|model|schema|database|query|store", re.I),
        "api": re.compile(r"api|endpoint|route|handler|request|response", re.I),
        "test": re.compile(r"test|spec|mock|assert|fixture", re.I),
        "core": re.compile(r"core|main|engine|manager|service", re.I),
    }

    def __init__(self, label_store: Optional[LabelStore] = None):
        self._label_store = label_store or LabelStore()

    def render(
        self,
        chunk_id: str,
        content: str,
        query_intent: str = "",
        available_tokens: int = 300,
    ) -> Optional[RenderedSlice]:
        label = self._label_store.get_label(chunk_id)
        if not label or not label.anchors:
            return None

        matched_anchors = self._match_anchors(label, query_intent)
        if not matched_anchors:
            matched_anchors = label.anchors[:1]

        lines = content.split("\n")
        slice_lines = []
        total_tokens = 0

        for anchor in matched_anchors:
            start = max(0, anchor.start_line - 1)
            end = min(len(lines), anchor.end_line)
            for i in range(start, end):
                line = lines[i]
                line_tokens = self._estimate_tokens(line)
                if total_tokens + line_tokens > available_tokens:
                    break
                slice_lines.append(line)
                total_tokens += line_tokens

        if not slice_lines:
            return None

        slice_content = "\n".join(slice_lines)
        best_anchor = matched_anchors[0]

        return RenderedSlice(
            chunk_id=chunk_id,
            anchor_name=best_anchor.name,
            content=slice_content,
            tokens=total_tokens,
            semantic_tag=best_anchor.semantic_tag,
            grain_hint=best_anchor.grain_hint,
            render_method="template",
        )

    def render_multi_anchor(
        self,
        chunk_id: str,
        content: str,
        anchor_names: List[str],
        available_tokens: int = 500,
    ) -> Optional[RenderedSlice]:
        label = self._label_store.get_label(chunk_id)
        if not label:
            return None

        selected = [a for a in label.anchors if a.name in anchor_names]
        if not selected:
            return None

        lines = content.split("\n")
        slice_lines = []
        total_tokens = 0

        for anchor in sorted(selected, key=lambda a: a.start_line):
            start = max(0, anchor.start_line - 1)
            end = min(len(lines), anchor.end_line)
            for i in range(start, end):
                if total_tokens >= available_tokens:
                    break
                line = lines[i]
                line_tokens = self._estimate_tokens(line)
                if total_tokens + line_tokens <= available_tokens:
                    slice_lines.append(line)
                    total_tokens += line_tokens

        if not slice_lines:
            return None

        return RenderedSlice(
            chunk_id=chunk_id,
            anchor_name="+".join(anchor_names),
            content="\n".join(slice_lines),
            tokens=total_tokens,
            semantic_tag=selected[0].semantic_tag,
            grain_hint=selected[0].grain_hint,
            render_method="template_multi",
        )

    def can_render(self, chunk_id: str) -> bool:
        label = self._label_store.get_label(chunk_id)
        return label is not None and len(label.anchors) > 0

    def get_label_coverage(self) -> float:
        stats = self._label_store.get_coverage_stats()
        total = stats.get("total_labeled", 0)
        if total == 0:
            return 0.0
        return stats.get("avg_coverage", 0.0)

    def _match_anchors(self, label: ChunkLabel, query_intent: str) -> List[Anchor]:
        if not query_intent:
            return []

        matched_tags = set()
        for tag, pattern in self._INTENT_PATTERNS.items():
            if pattern.search(query_intent):
                matched_tags.add(tag)

        matched = []
        for anchor in label.anchors:
            if anchor.semantic_tag in matched_tags:
                matched.append(anchor)
            elif any(tag in anchor.name.lower() for tag in matched_tags):
                matched.append(anchor)

        return matched if matched else []

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en = len(text) - zh
        return zh * 2 + en // 4

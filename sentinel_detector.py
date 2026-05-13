import re
from typing import List, Optional
from dataclasses import dataclass

from .ir_models import GrainLevel


@dataclass
class LoadRequest:
    chunk_id: str
    raw_marker: str
    position: int
    min_level: GrainLevel = GrainLevel.KEYWORDS
    confidence: float = 1.0


class SentinelDetector:

    PATTERNS = [
        (re.compile(r"\[NEED_CHUNK_FULL:([A-Za-z0-9_\-]+)\]"), GrainLevel.FULL),
        (re.compile(r"\[NEED_CHUNK_DETAIL:([A-Za-z0-9_\-]+)\]"), GrainLevel.DETAIL),
        (re.compile(r"\[NEED_CHUNK:([A-Za-z0-9_\-]+)\]"), GrainLevel.KEYWORDS),
        (re.compile(r"\[LOAD_CHUNK:([A-Za-z0-9_\-]+)\]"), GrainLevel.KEYWORDS),
        (re.compile(r"\[FETCH:([A-Za-z0-9_\-]+)\]"), GrainLevel.KEYWORDS),
    ]

    def __init__(self):
        self._buffer = ""
        self._clean_buffer = ""
        self._requests: List[LoadRequest] = []

    def reset(self) -> None:
        self._buffer = ""
        self._clean_buffer = ""
        self._requests = []

    def feed(self, text: str) -> List[LoadRequest]:
        self._buffer += text
        self._clean_buffer += text

        requests = []
        for pattern, min_level in self.PATTERNS:
            for match in pattern.finditer(self._buffer):
                chunk_id = match.group(1)
                raw = match.group(0)
                pos = match.start()

                if any(r.chunk_id == chunk_id and r.min_level == min_level for r in self._requests):
                    continue

                req = LoadRequest(
                    chunk_id=chunk_id,
                    raw_marker=raw,
                    position=pos,
                    min_level=min_level,
                )
                requests.append(req)
                self._requests.append(req)

                self._clean_buffer = self._clean_buffer.replace(raw, "")

        return requests

    def get_clean_buffer(self) -> str:
        return self._clean_buffer.strip()

    def has_pending_requests(self) -> bool:
        return len(self._requests) > 0

    def get_requests(self) -> List[LoadRequest]:
        return list(self._requests)

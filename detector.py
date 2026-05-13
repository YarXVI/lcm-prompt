"""
LCM v2 哨兵检测器
支持多种模式、置信度评分、缓冲区管理
"""
import re
from typing import List, Optional

from .lcm_types import LoadRequest, SentinelPattern


class SentinelDetectorV2:
    """哨兵检测器 v2 —— 支持多模式、置信度评分"""

    def __init__(self, patterns: Optional[List[str]] = None):
        self.patterns = patterns or SentinelPattern.get_all_patterns()
        self._compiled_patterns = [re.compile(p) for p in self.patterns]
        self._buffer = ""
        self._position = 0
        self._max_buffer_size = 10000

    def feed(self, text: str) -> List[LoadRequest]:
        """喂入新文本，返回检测到的所有加载请求"""
        self._buffer += text
        self._trim_buffer()

        requests = []
        seen = set()

        for pattern in self._compiled_patterns:
            for match in pattern.finditer(self._buffer):
                chunk_id = match.group(1)
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                start_pos = self._position + match.start()
                raw = match.group(0)
                confidence = self._calculate_confidence(chunk_id, raw)
                requests.append(LoadRequest(
                    chunk_id=chunk_id,
                    raw_marker=raw,
                    position=start_pos,
                    confidence=confidence,
                ))

        self._position += len(text)
        return requests

    def get_clean_buffer(self) -> str:
        """获取移除了所有哨兵标记后的干净文本"""
        clean = self._buffer
        for pattern in self._compiled_patterns:
            clean = pattern.sub("", clean)
        return clean

    def get_raw_buffer(self) -> str:
        return self._buffer

    def reset(self):
        self._buffer = ""
        self._position = 0

    def _trim_buffer(self):
        """限制缓冲区大小，防止内存泄漏

        安全截断策略：
        1. 保留最后 max_buffer_size 字符
        2. 检查截断处是否有未完成的哨兵标记前缀（如 '[NEED_CH'）
        3. 如有，保留到上一个安全位置（换行或空格）
        """
        if len(self._buffer) <= self._max_buffer_size:
            return

        overflow = len(self._buffer) - self._max_buffer_size
        # 从 max_buffer_size 处往前找，确保不截断在哨兵标记中间
        safe_cut = self._max_buffer_size

        # 检查截断点附近是否有 '['（哨兵标记的开头）
        # 往前搜索最多 50 个字符（最长哨兵前缀长度）
        search_start = max(0, safe_cut - 50)
        segment = self._buffer[search_start:safe_cut]

        # 如果 segment 中有 '[' 但没有对应的 ']'，说明可能截断了哨兵
        last_bracket = segment.rfind('[')
        if last_bracket != -1:
            after_bracket = segment[last_bracket:]
            # 检查是否是已知哨兵前缀
            sentinel_prefixes = ['[NEED_CHUNK:', '[LOAD_CHUNK:', '[FETCH:']
            for prefix in sentinel_prefixes:
                if prefix.startswith(after_bracket) and prefix != after_bracket:
                    # 截断点落在哨兵标记中间，回退到 '[' 之前
                    safe_cut = search_start + last_bracket
                    break

        self._buffer = self._buffer[-safe_cut:]
        self._position += (overflow + (self._max_buffer_size - safe_cut))

    def _calculate_confidence(self, chunk_id: str, raw: str) -> float:
        """计算检测置信度"""
        confidence = 1.0
        # 标准格式 [NEED_CHUNK:id] 置信度最高
        if raw.startswith("[NEED_CHUNK:"):
            confidence = 1.0
        elif raw.startswith("[LOAD_CHUNK:"):
            confidence = 0.9
        elif raw.startswith("[FETCH:"):
            confidence = 0.8
        # chunk_id 长度适中（5-50 字符）更可信
        if 5 <= len(chunk_id) <= 50:
            confidence += 0.05
        return min(confidence, 1.0)

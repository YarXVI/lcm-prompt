"""
Chinese-Think 编码实现 —— LCM 内容编码层的第一个标准插件

将 chinese-think-skills 独立项目的能力封装为 LCM 标准编码接口。
LCM 核心不依赖此模块，仅在需要时通过注册表加载。
"""
import sys
import os
from typing import Dict, Any

# 将 chinese-think-skills 加入路径（如果存在）
# 支持从 lcm_v2/encodings/ 和 lcm_v2/ 两种路径启动
_CTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "chinese-think-skills", "src"
)
_CTS_PATH_ALT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "chinese-think-skills", "src"
)
for p in [_CTS_PATH, os.path.normpath(_CTS_PATH_ALT)]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from ..content_encoding import ContentEncoding, EncodingType, EncodingContext

# 尝试导入独立 chinese-think 包，失败则使用内嵌降级实现
try:
    from chinese_think import (
        ChineseThinkProcessor, ThinkMode,
        TextCompressor, TokenSaver,
        get_prompt_by_mode,
    )
    _HAS_CTS = True
except ImportError:
    _HAS_CTS = False


class ChineseThinkEncoding(ContentEncoding):
    """中文思考编码实现

    作为 LCM 内容编码层的标准插件，将 chinese-think-skills 的能力
    通过 ContentEncoding 接口接入 LCM 核心。

    职责边界：
    - 本类只负责"适配"：将 chinese-think 的能力翻译为 LCM 编码接口
    - 具体的精简规则、文本压缩逻辑仍由 chinese-think-skills 维护
    - LCM 核心通过 ContentEncodingRegistry 发现和使用本类
    """

    # 模式映射：LCM 编码参数 -> chinese-think 模式
    MODE_MAP = {
        "lite": ThinkMode.LITE if _HAS_CTS else None,
        "full": ThinkMode.FULL if _HAS_CTS else None,
        "compact": ThinkMode.COMPACT if _HAS_CTS else None,
        "off": ThinkMode.OFF if _HAS_CTS else None,
    }

    def __init__(self, mode: str = "compact"):
        self._mode = mode
        self._processor = None
        self._compressor = None
        self._saver = None

        if _HAS_CTS:
            self._processor = ChineseThinkProcessor()
            self._compressor = TextCompressor()
            self._saver = TokenSaver()
            self._set_mode(mode)

    def _set_mode(self, mode: str):
        """设置内部模式"""
        if _HAS_CTS and self._processor:
            ct_mode = self.MODE_MAP.get(mode, ThinkMode.COMPACT)
            self._processor.mode = ct_mode
            self._mode = mode

    @property
    def encoding_type(self) -> EncodingType:
        return EncodingType.CHINESE_THINK

    @property
    def name(self) -> str:
        return "中文思考精简模式"

    @property
    def mode(self) -> str:
        """当前模式字符串"""
        return self._mode

    @mode.setter
    def mode(self, value: str):
        self._set_mode(value)

    def encode_system_prompt(self, system_prompt: str, context: EncodingContext) -> str:
        """编码系统提示词：注入中文思考指令

        在 LCM 系统提示词后追加中文思考模式的指令，
        引导模型用中文精简表达。
        """
        if self._mode == "off" or not _HAS_CTS:
            return system_prompt

        # 获取对应模式的提示词
        ct_mode = self.MODE_MAP.get(self._mode, ThinkMode.COMPACT)
        ct_prompt = get_prompt_by_mode(ct_mode)

        if not ct_prompt:
            return system_prompt

        # 追加到系统提示词（不破坏原有 LCM 指令）
        return f"{system_prompt}\n\n{ct_prompt}"

    def encode_response(self, response_text: str, context: EncodingContext) -> str:
        """编码模型响应：后处理压缩

        对模型输出进行规则压缩（删除冗余、替换文言等）。
        注意：保护 LCM 哨兵标记不被破坏。
        """
        if self._mode == "off" or not _HAS_CTS or not self._compressor:
            return response_text

        # 保护 LCM 哨兵标记
        protected_markers = self._extract_sentinel_markers(response_text)
        working_text = self._mask_markers(response_text, protected_markers)

        # 压缩文本
        compressed = self._compressor.compress(working_text)

        # 恢复哨兵标记
        result = self._unmask_markers(compressed, protected_markers)

        # 记录节省统计
        if self._saver:
            self._saver.calculate_savings(
                original_text=response_text,
                compressed_text=result,
                context=f"round_{context.current_round}",
            )

        return result

    def decode_for_display(self, encoded_text: str, context: EncodingContext) -> str:
        """解码为展示格式

        中文思考编码通常不需要解码（输出本身就是可读中文），
        但保留接口以支持未来可能的扩展（如加密编码等）。
        """
        return encoded_text

    def get_stats(self) -> Dict[str, Any]:
        """获取编码统计"""
        if not _HAS_CTS or not self._saver:
            return {"mode": self._mode, "enabled": False}

        summary = self._saver.get_summary()
        return {
            "mode": self._mode,
            "enabled": True,
            "has_chinese_think_package": True,
            **summary,
        }

    def reset(self):
        """重置状态"""
        if self._saver:
            self._saver.reset()
        if self._processor:
            self._processor.mode = ThinkMode.OFF if _HAS_CTS else None

    # --- 内部工具方法 ---

    def _extract_sentinel_markers(self, text: str) -> Dict[str, str]:
        """提取并保护 LCM 哨兵标记"""
        import re
        markers = {}
        pattern = r"\[(NEED_CHUNK|LOAD_CHUNK|FETCH):([A-Za-z0-9_\-]+)\]"
        for match in re.finditer(pattern, text):
            marker_id = f"__SENTINEL_{len(markers)}__"
            markers[marker_id] = match.group()
        return markers

    def _mask_markers(self, text: str, markers: Dict[str, str]) -> str:
        """用占位符替换哨兵标记"""
        import re
        pattern = r"\[(NEED_CHUNK|LOAD_CHUNK|FETCH):([A-Za-z0-9_\-]+)\]"
        counter = [0]

        def replacer(m):
            marker_id = f"__SENTINEL_{counter[0]}__"
            counter[0] += 1
            return marker_id

        return re.sub(pattern, replacer, text)

    def _unmask_markers(self, text: str, markers: Dict[str, str]) -> str:
        """恢复哨兵标记"""
        for marker_id, original in markers.items():
            text = text.replace(marker_id, original)
        return text


# 降级实现：当 chinese-think-skills 不可用时使用
class _FallbackChineseThinkEncoding(ContentEncoding):
    """降级实现：仅注入基础中文思考指令，无压缩能力"""

    _BASIC_PROMPT = """[CHINESE-THINK 精简模式]

用中文思考，精简表达。规则：
- 删除语气词、程度副词、客套话
- 用文言替换：因...故...、若...则...
- 优先单字、四字格、文言句式
- 代码/路径/术语/安全警告保持原样
"""

    @property
    def encoding_type(self) -> EncodingType:
        return EncodingType.CHINESE_THINK

    @property
    def name(self) -> str:
        return "中文思考精简模式（降级）"

    def encode_system_prompt(self, system_prompt: str, context: EncodingContext) -> str:
        return f"{system_prompt}\n\n{self._BASIC_PROMPT}"

    def encode_response(self, response_text: str, context: EncodingContext) -> str:
        return response_text

    def decode_for_display(self, encoded_text: str, context: EncodingContext) -> str:
        return encoded_text


def create_chinese_think_encoding(mode: str = "compact") -> ContentEncoding:
    """工厂函数：创建中文思考编码实例

    优先使用完整实现（需 chinese-think-skills），
    不可用时自动降级为基础实现。
    """
    if _HAS_CTS:
        return ChineseThinkEncoding(mode)
    return _FallbackChineseThinkEncoding()
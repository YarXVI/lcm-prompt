"""
LCM v2 编码实现包

包含所有标准内容编码实现：
- chinese_think_encoding: 中文思考精简模式
- （预留）ja_think_encoding: 日语思考精简模式
- （预留）en_think_encoding: 英语思考精简模式
"""
from ..content_encoding import (
    ContentEncoding,
    EncodingType,
    EncodingContext,
    ContentEncodingRegistry,
    IdentityEncoding,
    get_default_registry,
    register_encoding,
    get_encoding,
)

# 尝试导入中文思考编码，失败不影响其他功能
try:
    from .chinese_think_encoding import (
        ChineseThinkEncoding,
        create_chinese_think_encoding,
    )
    _HAS_CHINESE_THINK = True
except ImportError:
    _HAS_CHINESE_THINK = False


def register_all_encodings(registry: ContentEncodingRegistry = None) -> ContentEncodingRegistry:
    """注册所有可用的编码实现

    Args:
        registry: 目标注册表，None 则使用全局注册表

    Returns:
        注册表实例
    """
    if registry is None:
        registry = get_default_registry()

    # 注册中文思考编码
    if _HAS_CHINESE_THINK:
        registry.register(ChineseThinkEncoding())

    # （预留）注册日语思考编码
    # if _HAS_JAPANESE_THINK:
    #     registry.register(JapaneseThinkEncoding())

    # （预留）注册英语思考编码
    # if _HAS_ENGLISH_THINK:
    #     registry.register(EnglishThinkEncoding())

    return registry


__all__ = [
    # 核心接口（从 content_encoding 透传）
    "ContentEncoding",
    "EncodingType",
    "EncodingContext",
    "ContentEncodingRegistry",
    "IdentityEncoding",
    "get_default_registry",
    "register_encoding",
    "get_encoding",
    # 中文思考编码
    "ChineseThinkEncoding",
    "create_chinese_think_encoding",
    "register_all_encodings",
]
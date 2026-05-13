"""
LCM v2 内容编码层 (Content Encoding Layer)

设计哲学：
- LCM 核心负责"数据包的可靠传输"（上下文调度、chunk 加载、状态管理）
- ContentEncoding 负责"数据内容的编码/解码"（语言精简、压缩、格式化）
- 两者通过标准接口解耦，编码实现可独立开发、热插拔

类比：
- LCM Core = TCP/IP（传输层）
- ContentEncoding = Gzip/Brotli（表示层）
- 数据区支持 Content-Encoding 头，声明编码方式
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


class EncodingType(str, Enum):
    """标准编码类型注册表

    新编码实现只需在此注册唯一标识符
    """
    NONE = "none"           # 无编码，透传
    IDENTITY = "identity"   # 恒等编码（显式声明不编码）
    CHINESE_THINK = "chinese-think"   # 中文思考精简模式
    JAPANESE_THINK = "ja-think"       # 日语思考精简模式（预留）
    ENGLISH_THINK = "en-think"        # 英语思考精简模式（预留）
    CUSTOM = "custom"       # 自定义编码（需指定 handler）


@dataclass
class EncodingContext:
    """编码上下文

    编码器在执行时需要的上下文信息
    """
    session_id: str = ""
    user_query: str = ""
    current_round: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentEncoding(ABC):
    """内容编码接口（抽象基类）

    所有编码实现必须继承此类，实现 encode/decode 方法。

    生命周期：
    1. LCM 核心在消息构建阶段调用 encode_system_prompt()
    2. LCM 核心在收到模型输出后调用 encode_response()
    3. （可选）LCM 核心在展示给用户前调用 decode_for_display()

    设计约束：
    - encode 操作必须是幂等的（多次编码结果一致）
    - encode 不应破坏 LCM 协议标记（如 [NEED_CHUNK:xxx]）
    - encode 不应改变消息的角色（role）和结构
    """

    @property
    @abstractmethod
    def encoding_type(self) -> EncodingType:
        """返回编码类型标识"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """返回人类可读的编码名称"""
        pass

    @abstractmethod
    def encode_system_prompt(self, system_prompt: str, context: EncodingContext) -> str:
        """编码系统提示词

        在 system prompt 构建完成后、发送给模型前调用。
        用于注入编码相关的指令（如"请用中文精简表达"）。

        Args:
            system_prompt: 原始系统提示词（已包含 LCM 协议指令）
            context: 编码上下文

        Returns:
            编码后的系统提示词
        """
        pass

    @abstractmethod
    def encode_response(self, response_text: str, context: EncodingContext) -> str:
        """编码模型响应

        在模型生成响应后、返回给用户前调用。
        用于对模型输出进行后处理（如进一步压缩、格式化）。

        Args:
            response_text: 原始模型响应文本
            context: 编码上下文

        Returns:
            编码后的响应文本
        """
        pass

    @abstractmethod
    def decode_for_display(self, encoded_text: str, context: EncodingContext) -> str:
        """解码为展示格式

        在将文本展示给最终用户前调用。
        用于将编码内容恢复为可读形式（如需要）。

        Args:
            encoded_text: 编码后的文本
            context: 编码上下文

        Returns:
            解码后的展示文本
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取编码器统计信息（可选实现）"""
        return {}

    def reset(self):
        """重置编码器状态（可选实现）"""
        pass


class IdentityEncoding(ContentEncoding):
    """恒等编码：不做任何变换，透传

    作为默认编码实现，确保系统在无编码时正常工作。
    """

    @property
    def encoding_type(self) -> EncodingType:
        return EncodingType.IDENTITY

    @property
    def name(self) -> str:
        return "恒等编码（无变换）"

    def encode_system_prompt(self, system_prompt: str, context: EncodingContext) -> str:
        return system_prompt

    def encode_response(self, response_text: str, context: EncodingContext) -> str:
        return response_text

    def decode_for_display(self, encoded_text: str, context: EncodingContext) -> str:
        return encoded_text


class ContentEncodingRegistry:
    """编码注册表

    管理所有可用的编码实现，支持动态注册和查找。

    使用方式：
        registry = ContentEncodingRegistry()
        registry.register(ChineseThinkEncoding())

        encoder = registry.get(EncodingType.CHINESE_THINK)
        encoded = encoder.encode_system_prompt(prompt, context)
    """

    def __init__(self):
        self._encodings: Dict[EncodingType, ContentEncoding] = {}
        # 注册默认编码
        self.register(IdentityEncoding())

    def register(self, encoding: ContentEncoding) -> "ContentEncodingRegistry":
        """注册编码实现

        Args:
            encoding: 编码实现实例

        Returns:
            self，支持链式调用
        """
        self._encodings[encoding.encoding_type] = encoding
        return self

    def get(self, encoding_type: EncodingType) -> ContentEncoding:
        """获取编码实现

        Args:
            encoding_type: 编码类型

        Returns:
            对应的编码实现，如果不存在则返回恒等编码
        """
        return self._encodings.get(encoding_type, IdentityEncoding())

    def unregister(self, encoding_type: EncodingType) -> bool:
        """注销编码实现

        Args:
            encoding_type: 编码类型

        Returns:
            是否成功注销
        """
        if encoding_type in self._encodings and encoding_type != EncodingType.IDENTITY:
            del self._encodings[encoding_type]
            return True
        return False

    def list_encodings(self) -> List[Dict[str, str]]:
        """列出所有已注册的编码"""
        return [
            {"type": et.value, "name": enc.name}
            for et, enc in self._encodings.items()
        ]

    def is_registered(self, encoding_type: EncodingType) -> bool:
        """检查编码是否已注册"""
        return encoding_type in self._encodings


# 全局默认注册表
_default_registry = ContentEncodingRegistry()


def get_default_registry() -> ContentEncodingRegistry:
    """获取全局默认编码注册表"""
    return _default_registry


def register_encoding(encoding: ContentEncoding) -> ContentEncodingRegistry:
    """向全局注册表注册编码"""
    return _default_registry.register(encoding)


def get_encoding(encoding_type: EncodingType) -> ContentEncoding:
    """从全局注册表获取编码"""
    return _default_registry.get(encoding_type)

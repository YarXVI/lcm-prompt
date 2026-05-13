from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class EncodingType(str, Enum):
    NONE = "none"
    IDENTITY = "identity"
    CHINESE_THINK = "chinese-think"
    JAPANESE_THINK = "ja-think"
    ENGLISH_THINK = "en-think"
    CUSTOM = "custom"


@dataclass
class EncodingContext:
    session_id: str = ""
    user_query: str = ""
    current_round: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentEncoding(ABC):

    @property
    @abstractmethod
    def encoding_type(self) -> EncodingType:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def encode_system_prompt(self, system_prompt: str, context: EncodingContext) -> str:
        pass

    @abstractmethod
    def encode_response(self, response_text: str, context: EncodingContext) -> str:
        pass

    @abstractmethod
    def decode_for_display(self, encoded_text: str, context: EncodingContext) -> str:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def reset(self):
        pass


class IdentityEncoding(ContentEncoding):

    @property
    def encoding_type(self) -> EncodingType:
        return EncodingType.IDENTITY

    @property
    def name(self) -> str:
        return "Identity Encoding"

    def encode_system_prompt(self, system_prompt: str, context: EncodingContext) -> str:
        return system_prompt

    def encode_response(self, response_text: str, context: EncodingContext) -> str:
        return response_text

    def decode_for_display(self, encoded_text: str, context: EncodingContext) -> str:
        return encoded_text


class ContentEncodingRegistry:

    def __init__(self):
        self._encodings: Dict[EncodingType, ContentEncoding] = {}
        self.register(IdentityEncoding())

    def register(self, encoding: ContentEncoding) -> "ContentEncodingRegistry":
        self._encodings[encoding.encoding_type] = encoding
        return self

    def get(self, encoding_type: EncodingType) -> ContentEncoding:
        return self._encodings.get(encoding_type, IdentityEncoding())

    def unregister(self, encoding_type: EncodingType) -> bool:
        if encoding_type in self._encodings and encoding_type != EncodingType.IDENTITY:
            del self._encodings[encoding_type]
            return True
        return False

    def list_encodings(self) -> List[Dict[str, str]]:
        return [
            {"type": et.value, "name": enc.name}
            for et, enc in self._encodings.items()
        ]

    def is_registered(self, encoding_type: EncodingType) -> bool:
        return encoding_type in self._encodings


_default_registry = ContentEncodingRegistry()


def get_default_registry() -> ContentEncodingRegistry:
    return _default_registry


def register_encoding(encoding: ContentEncoding) -> ContentEncodingRegistry:
    return _default_registry.register(encoding)


def get_encoding(encoding_type: EncodingType) -> ContentEncoding:
    return _default_registry.get(encoding_type)

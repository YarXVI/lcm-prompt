"""
LCM v2 多模态 Chunk 支持
支持图片、PDF、音频等非文本内容
"""
import base64
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum

from .lcm_types import ContextChunk
from .logger import get_logger

logger = get_logger()


class MediaType(str, Enum):
    """媒体类型"""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"


@dataclass
class MediaChunk(ContextChunk):
    """多媒体 Chunk，扩展 ContextChunk"""
    media_type: MediaType = MediaType.TEXT
    file_path: Optional[str] = None
    mime_type: str = "text/plain"
    base64_data: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None
    pages: Optional[int] = None
    ocr_text: Optional[str] = None
    thumbnail: Optional[str] = None

    def to_text_chunk(self) -> ContextChunk:
        """转换为纯文本 chunk（用于不支持多模态的模型）"""
        text_content = self.content
        if self.ocr_text:
            text_content += f"\n\n[OCR 识别]:\n{self.ocr_text}"
        if self.pages:
            text_content += f"\n[页数]: {self.pages}"
        if self.duration_ms:
            text_content += f"\n[时长]: {self.duration_ms / 1000:.1f}s"

        return ContextChunk(
            chunk_id=self.chunk_id,
            content=text_content,
            summary=self.summary,
            tokens=self.tokens,
            source=self.source,
            metadata={
                **self.metadata,
                "media_type": self.media_type.value,
                "mime_type": self.mime_type,
                "original_type": "media",
            },
        )

    def to_openai_image_url(self) -> Optional[str]:
        """生成 OpenAI 兼容的图片 URL"""
        if self.media_type != MediaType.IMAGE or not self.base64_data:
            return None
        return f"data:{self.mime_type};base64,{self.base64_data}"

    def to_anthropic_image_block(self) -> Optional[Dict[str, Any]]:
        """生成 Anthropic 兼容的图片块"""
        if self.media_type != MediaType.IMAGE or not self.base64_data:
            return None
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": self.base64_data,
            },
        }

    def to_message_content(self, provider: str = "openai") -> List[Dict[str, Any]]:
        """
        转换为消息内容块
        
        Args:
            provider: 提供商名称（openai/anthropic）
        
        Returns:
            内容块列表
        """
        content = []

        # 文本内容
        if self.content:
            content.append({"type": "text", "text": self.content})

        # 图片
        if self.media_type == MediaType.IMAGE:
            if provider == "anthropic":
                image_block = self.to_anthropic_image_block()
                if image_block:
                    content.append(image_block)
            elif provider == "openai":
                image_url = self.to_openai_image_url()
                if image_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    })

        # PDF（转换为文本描述）
        elif self.media_type == MediaType.PDF:
            pdf_desc = f"[PDF 文档: {self.pages or '?'} 页]"
            if self.ocr_text:
                pdf_desc += f"\n{self.ocr_text[:500]}..."
            content.append({"type": "text", "text": pdf_desc})

        # 音频
        elif self.media_type == MediaType.AUDIO:
            audio_desc = f"[音频: {self.duration_ms / 1000:.1f}s]"
            if self.ocr_text:  # 转录文本
                audio_desc += f"\n[转录]: {self.ocr_text[:500]}..."
            content.append({"type": "text", "text": audio_desc})

        return content


class MediaChunkLoader:
    """多媒体 Chunk 加载器"""

    @staticmethod
    def from_image(path: str, chunk_id: str, summary: str = "") -> MediaChunk:
        """从图片文件创建 MediaChunk"""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        # 读取并编码
        with open(path_obj, "rb") as f:
            data = f.read()
            base64_data = base64.b64encode(data).decode("utf-8")

        # 检测 MIME 类型
        mime_type = "image/png"
        if path_obj.suffix.lower() in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif path_obj.suffix.lower() == ".gif":
            mime_type = "image/gif"
        elif path_obj.suffix.lower() == ".webp":
            mime_type = "image/webp"

        return MediaChunk(
            chunk_id=chunk_id,
            content=f"[图片: {path_obj.name}]",
            summary=summary or f"图片 {path_obj.name}",
            media_type=MediaType.IMAGE,
            file_path=str(path_obj),
            mime_type=mime_type,
            base64_data=base64_data,
            source=str(path_obj),
        )

    @staticmethod
    def from_pdf(path: str, chunk_id: str, summary: str = "") -> MediaChunk:
        """从 PDF 文件创建 MediaChunk"""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF 不存在: {path}")

        # 尝试提取文本（如果 PyPDF2 可用）
        ocr_text = None
        pages = None
        try:
            import PyPDF2
            with open(path_obj, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = len(reader.pages)
                text_parts = []
                for i, page in enumerate(reader.pages[:5]):  # 只读前 5 页
                    text_parts.append(f"[第 {i+1} 页]\n{page.extract_text() or ''}")
                ocr_text = "\n\n".join(text_parts)
        except ImportError:
            logger.warning("PyPDF2 未安装，无法提取 PDF 文本")
        except Exception as e:
            logger.warning("PDF 文本提取失败", error=str(e))

        return MediaChunk(
            chunk_id=chunk_id,
            content=f"[PDF: {path_obj.name}]",
            summary=summary or f"PDF 文档 {path_obj.name}",
            media_type=MediaType.PDF,
            file_path=str(path_obj),
            mime_type="application/pdf",
            pages=pages,
            ocr_text=ocr_text,
            source=str(path_obj),
        )

    @staticmethod
    def from_audio(path: str, chunk_id: str, summary: str = "", transcript: str = "") -> MediaChunk:
        """从音频文件创建 MediaChunk"""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"音频不存在: {path}")

        # 尝试获取时长
        duration_ms = None
        try:
            # 简单估算：假设 MP3 128kbps
            size_bytes = path_obj.stat().st_size
            duration_ms = int(size_bytes / 16 * 1000)  # 粗略估算
        except Exception:
            pass

        return MediaChunk(
            chunk_id=chunk_id,
            content=f"[音频: {path_obj.name}]",
            summary=summary or f"音频 {path_obj.name}",
            media_type=MediaType.AUDIO,
            file_path=str(path_obj),
            mime_type="audio/mpeg",
            duration_ms=duration_ms,
            ocr_text=transcript,
            source=str(path_obj),
        )

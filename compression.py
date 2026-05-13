"""
LCM v2 Chunk 压缩存储
使用 gzip/zstd 减少内存和磁盘占用
"""
import gzip
import zlib
from typing import Optional
from pathlib import Path

from .logger import get_logger

logger = get_logger()


class ChunkCompressor:
    """
    Chunk 内容压缩器
    
    策略：
    - 小内容（<1KB）：不压缩，避免压缩开销
    - 中内容（1KB-10KB）：gzip 压缩
    - 大内容（>10KB）：zstd 压缩（如果可用）
    """

    def __init__(self, threshold: int = 1024, level: int = 6):
        self.threshold = threshold
        self.level = level
        self._use_zstd = self._check_zstd()

    @staticmethod
    def _check_zstd() -> bool:
        """检查是否支持 zstd"""
        try:
            import zstandard
            return True
        except ImportError:
            return False

    def compress(self, content: str) -> Optional[bytes]:
        """
        压缩内容
        
        Returns:
            压缩后的字节，如果内容太小则返回 None
        """
        if len(content) < self.threshold:
            return None

        data = content.encode("utf-8")

        if self._use_zstd and len(content) > 10 * 1024:
            # 大内容使用 zstd
            try:
                import zstandard as zstd
                cctx = zstd.ZstdCompressor(level=self.level)
                compressed = cctx.compress(data)
                # 添加 zstd 标记前缀
                return b"\x01" + compressed
            except Exception as e:
                logger.warning("zstd 压缩失败，降级到 gzip", error=str(e))

        # 使用 gzip
        compressed = gzip.compress(data, compresslevel=self.level)
        # 添加 gzip 标记前缀
        return b"\x00" + compressed

    def decompress(self, data: bytes) -> str:
        """
        解压内容
        
        Args:
            data: 压缩后的字节（含标记前缀）
        
        Returns:
            解压后的字符串
        """
        if not data:
            return ""

        # 检查标记前缀
        marker = data[0]
        payload = data[1:]

        if marker == 0x00:
            # gzip
            return gzip.decompress(payload).decode("utf-8")
        elif marker == 0x01:
            # zstd
            try:
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(payload).decode("utf-8")
            except Exception as e:
                logger.error("zstd 解压失败", error=str(e))
                raise
        else:
            # 无标记，可能是未压缩数据
            return data.decode("utf-8")

    def get_stats(self, content: str, compressed: Optional[bytes]) -> dict:
        """获取压缩统计"""
        original_size = len(content.encode("utf-8"))
        if compressed is None:
            return {
                "compressed": False,
                "original_size": original_size,
                "compressed_size": original_size,
                "ratio": 1.0,
            }

        compressed_size = len(compressed)
        ratio = compressed_size / original_size if original_size > 0 else 0

        return {
            "compressed": True,
            "algorithm": "zstd" if compressed[0] == 0x01 else "gzip",
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": round(ratio, 4),
            "savings": round((1 - ratio) * 100, 2),
        }


class CompressedChunkStore:
    """带压缩的 Chunk 存储包装器"""

    def __init__(self, store, compressor: Optional[ChunkCompressor] = None):
        self.store = store
        self.compressor = compressor or ChunkCompressor()
        self._compression_stats = {
            "total_chunks": 0,
            "compressed_chunks": 0,
            "total_savings_bytes": 0,
        }

    def add_chunk(self, chunk):
        """添加 chunk（自动压缩）"""
        compressed = self.compressor.compress(chunk.content)
        if compressed:
            # 存储压缩后的数据
            chunk.metadata["_compressed"] = True
            chunk.metadata["_compressed_data"] = compressed.hex()
            stats = self.compressor.get_stats(chunk.content, compressed)
            self._compression_stats["compressed_chunks"] += 1
            self._compression_stats["total_savings_bytes"] += stats["original_size"] - stats["compressed_size"]

        self._compression_stats["total_chunks"] += 1
        self.store.add_chunk(chunk)

    def get_chunk(self, chunk_id: str):
        """获取 chunk（自动解压）"""
        chunk = self.store.get_chunk(chunk_id)
        if not chunk:
            return None

        if chunk.metadata.get("_compressed"):
            # 解压
            compressed_data = bytes.fromhex(chunk.metadata["_compressed_data"])
            chunk.content = self.compressor.decompress(compressed_data)
            # 清理临时标记
            del chunk.metadata["_compressed_data"]
            chunk.metadata["_compressed"] = False

        return chunk

    def get_compression_stats(self) -> dict:
        """获取压缩统计"""
        return {
            **self._compression_stats,
            "avg_savings_kb": round(self._compression_stats["total_savings_bytes"] / 1024 / max(self._compression_stats["compressed_chunks"], 1), 2),
        }

"""
LCM v2 协议版本协商
支持 v1/v2/v3 兼容和版本协商
"""
from enum import Enum
from typing import Dict, Any, Optional, List

from .logger import get_logger

logger = get_logger()


class ProtocolVersion(str, Enum):
    """协议版本"""
    V1 = "1.0"
    V2 = "2.0"
    V3 = "3.0"


class ProtocolNegotiator:
    """
    协议协商器
    
    协商流程：
    1. 客户端发送支持的版本列表
    2. 服务端选择最高兼容版本
    3. 双方按协商版本通信
    """

    # 版本兼容性矩阵
    COMPATIBILITY = {
        ProtocolVersion.V3: [ProtocolVersion.V3, ProtocolVersion.V2, ProtocolVersion.V1],
        ProtocolVersion.V2: [ProtocolVersion.V2, ProtocolVersion.V1],
        ProtocolVersion.V1: [ProtocolVersion.V1],
    }

    # 版本特性
    FEATURES = {
        ProtocolVersion.V1: {
            "sentinel_patterns": ["[NEED_CHUNK:{id}]"],
            "max_rounds": 3,
            "supports_streaming": False,
            "supports_hybrid": False,
            "supports_kv_cache": False,
        },
        ProtocolVersion.V2: {
            "sentinel_patterns": ["[NEED_CHUNK:{id}]", "[LOAD_CHUNK:{id}]", "[FETCH:{id}]"],
            "max_rounds": 10,
            "supports_streaming": True,
            "supports_hybrid": True,
            "supports_kv_cache": True,
        },
        ProtocolVersion.V3: {
            "sentinel_patterns": ["[NEED_CHUNK:{id}]", "[LOAD_CHUNK:{id}]", "[FETCH:{id}]", "[QUERY:{text}]"],
            "max_rounds": 20,
            "supports_streaming": True,
            "supports_hybrid": True,
            "supports_kv_cache": True,
            "supports_async": True,
            "supports_multimodal": True,
        },
    }

    def __init__(self, local_version: ProtocolVersion = ProtocolVersion.V2):
        self.local_version = local_version
        self.negotiated_version: Optional[ProtocolVersion] = None

    def negotiate(self, remote_versions: List[str]) -> ProtocolVersion:
        """
        协商版本
        
        Args:
            remote_versions: 远程端支持的版本列表
        
        Returns:
            协商后的版本
        """
        # 将字符串转为枚举
        remote_enum = []
        for v in remote_versions:
            try:
                remote_enum.append(ProtocolVersion(v))
            except ValueError:
                logger.warning("远程端声明了未知版本", version=v)

        # 按兼容性从高到低尝试
        for local_compat in self.COMPATIBILITY[self.local_version]:
            if local_compat in remote_enum:
                self.negotiated_version = local_compat
                logger.info("协议协商成功", local=self.local_version, remote=local_compat)
                return local_compat

        # 无法协商，使用最低版本
        self.negotiated_version = ProtocolVersion.V1
        logger.warning("协议协商失败，降级到 v1", local=self.local_version, remote=remote_versions)
        return ProtocolVersion.V1

    def get_features(self) -> Dict[str, Any]:
        """获取当前协商版本的特性"""
        version = self.negotiated_version or self.local_version
        return self.FEATURES.get(version, self.FEATURES[ProtocolVersion.V1])

    def supports_feature(self, feature: str) -> bool:
        """检查是否支持某个特性"""
        features = self.get_features()
        return features.get(feature, False)

    def get_sentinel_patterns(self) -> List[str]:
        """获取当前版本支持的哨兵模式"""
        features = self.get_features()
        return features.get("sentinel_patterns", [])

    def get_max_rounds(self) -> int:
        """获取当前版本支持的最大轮次"""
        features = self.get_features()
        return features.get("max_rounds", 3)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "local_version": self.local_version.value,
            "negotiated_version": self.negotiated_version.value if self.negotiated_version else None,
            "features": self.get_features(),
        }


class ProtocolAdapter:
    """
    协议适配器
    将不同版本的哨兵格式相互转换
    """

    @staticmethod
    def v1_to_v2(sentinel: str) -> str:
        """将 v1 哨兵转换为 v2 格式"""
        # v1: [NEED_CHUNK:id] -> v2: [NEED_CHUNK:id]（相同）
        return sentinel

    @staticmethod
    def v2_to_v1(sentinel: str) -> Optional[str]:
        """将 v2 哨兵转换为 v1 格式"""
        # v2 的 LOAD_CHUNK 和 FETCH 在 v1 中不支持
        if "[NEED_CHUNK:" in sentinel:
            return sentinel
        return None

    @staticmethod
    def v2_to_v3(sentinel: str) -> str:
        """将 v2 哨兵转换为 v3 格式"""
        # v3 兼容 v2 的所有哨兵
        return sentinel

    @staticmethod
    def v3_to_v2(sentinel: str) -> Optional[str]:
        """将 v3 哨兵转换为 v2 格式"""
        # v3 的 QUERY 哨兵在 v2 中不支持
        if "[QUERY:" in sentinel:
            return None
        return sentinel

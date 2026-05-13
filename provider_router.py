"""
LCM v2 Provider 路由
自动识别云端/本地 API，决定使用 LCM 还是传统方案
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, List, Dict, Any, Generator
from urllib.parse import urlparse

from .lcm_types import LCMEvent, LCMSession, LCMState, LCMMetrics
from .store import ChunkStoreV2


class ProviderType(str, Enum):
    """提供商类型"""
    CLOUD = "cloud"
    LOCAL = "local"
    UNKNOWN = "unknown"


class RoutingStrategy(str, Enum):
    """路由策略"""
    LCM = "lcm"
    TRADITIONAL = "traditional"
    AUTO = "auto"


@dataclass
class ProviderConfig:
    """提供商配置"""
    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    provider_type: ProviderType = ProviderType.UNKNOWN
    routing_strategy: RoutingStrategy = RoutingStrategy.AUTO


class ProviderRouter:
    """
    Provider 路由器
    自动识别 API 类型（云端/本地），决定使用 LCM 还是传统方案
    """

    # 已知云端提供商域名模式
    CLOUD_PATTERNS = [
        "api.openai.com",
        "api.deepseek.com",
        "api.anthropic.com",
        "api.cohere.com",
        "api.groq.com",
        "api.together.xyz",
        "api.perplexity.ai",
        "generativelanguage.googleapis.com",
        "api.mistral.ai",
        "api.fireworks.ai",
    ]

    # 已知本地提供商路径模式
    LOCAL_PATTERNS = [
        "/v1/chat/completions",
        "/v1/completions",
    ]

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._detected_type = self._detect_provider_type(config.base_url)
        self._effective_strategy = self._determine_strategy()

    @property
    def provider_type(self) -> ProviderType:
        return self._detected_type

    @property
    def strategy(self) -> RoutingStrategy:
        return self._effective_strategy

    @property
    def use_lcm(self) -> bool:
        """是否使用 LCM 方案"""
        return self._effective_strategy == RoutingStrategy.LCM

    def _detect_provider_type(self, base_url: str) -> ProviderType:
        """根据 base_url 检测提供商类型"""
        if not base_url:
            return ProviderType.UNKNOWN

        parsed = urlparse(base_url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""

        # 检查是否为本地地址
        if self._is_local_host(hostname):
            return ProviderType.LOCAL

        # 检查是否为已知云端提供商
        if self._is_cloud_provider(hostname):
            return ProviderType.CLOUD

        # 检查路径是否包含本地 API 特征（如 /v1）
        if self._has_local_path_pattern(path):
            return ProviderType.LOCAL

        # 检查端口是否为已知本地 LLM 服务端口
        # 11434: Ollama, 8080: llama.cpp/vLLM, 5000: 常见开发端口
        # 注意：仅当主机名也是本地地址时才判定为本地（避免误判云端服务）
        known_llm_ports = [11434]
        if parsed.port in known_llm_ports and self._is_local_host(hostname):
            return ProviderType.LOCAL

        return ProviderType.UNKNOWN

    def _determine_strategy(self) -> RoutingStrategy:
        """确定实际使用的路由策略"""
        if self.config.routing_strategy != RoutingStrategy.AUTO:
            return self.config.routing_strategy

        # 自动模式下：本地用 LCM，云端用传统方案
        if self._detected_type == ProviderType.LOCAL:
            return RoutingStrategy.LCM
        elif self._detected_type == ProviderType.CLOUD:
            return RoutingStrategy.TRADITIONAL
        else:
            # 未知类型默认用传统方案（更安全）
            return RoutingStrategy.TRADITIONAL

    @staticmethod
    def _is_local_host(hostname: str) -> bool:
        """判断是否为本地主机"""
        local_hosts = [
            "localhost", "127.0.0.1", "0.0.0.0",
            "::1", "192.168.", "10.0.", "172.16.",
        ]
        hostname_lower = hostname.lower()
        return any(
            hostname_lower == lh or hostname_lower.startswith(lh)
            for lh in local_hosts
        )

    @staticmethod
    def _is_cloud_provider(hostname: str) -> bool:
        """判断是否为已知云端提供商"""
        hostname_lower = hostname.lower()
        return any(pattern in hostname_lower for pattern in ProviderRouter.CLOUD_PATTERNS)

    @staticmethod
    def _has_local_path_pattern(path: str) -> bool:
        """判断路径是否包含本地 API 特征"""
        path_lower = path.lower()
        return any(pattern in path_lower for pattern in ProviderRouter.LOCAL_PATTERNS)

    def get_info(self) -> Dict[str, str]:
        """获取路由信息"""
        return {
            "base_url": self.config.base_url,
            "detected_type": self._detected_type.value,
            "configured_strategy": self.config.routing_strategy.value,
            "effective_strategy": self._effective_strategy.value,
            "use_lcm": str(self.use_lcm),
        }


class AdaptiveLCMClient:
    """
    自适应 LCM 客户端
    根据 Provider 类型自动选择 LCM 或传统方案
    """

    def __init__(
        self,
        llm_client,
        chunk_store: ChunkStoreV2,
        config: ProviderConfig,
        metrics: Optional[LCMMetrics] = None,
        verbose: bool = False,
    ):
        self._client = llm_client
        self.store = chunk_store
        self.router = ProviderRouter(config)
        self.metrics = metrics or LCMMetrics()
        self._verbose = verbose

        # 初始化 LCM orchestrator（仅在需要时使用）
        self._lcm_client: Optional[LCMClientV2] = None

        if self._verbose:
            info = self.router.get_info()
            print(f"[AdaptiveLCM] 路由决策: {info}")

    def _get_lcm(self):
        """延迟初始化 LCM 客户端（避免循环导入）"""
        if self._lcm_client is None:
            from .client import LCMClientV2
            self._lcm_client = LCMClientV2(
                self._client, self.store, metrics=self.metrics, verbose=self._verbose
            )
        return self._lcm_client

    def chat(self, user_query: str, session_id: str = "") -> str:
        """自适应对话：根据 Provider 类型选择方案"""
        if self.router.use_lcm:
            if self._verbose:
                print(f"[AdaptiveLCM] 使用 LCM 方案")
            return self._get_lcm().chat(user_query, session_id)
        else:
            if self._verbose:
                print(f"[AdaptiveLCM] 使用传统方案")
            return self._traditional_chat(user_query)

    def chat_stream(
        self, user_query: str, session_id: str = ""
    ) -> Generator[str, None, None]:
        """自适应流式对话"""
        if self.router.use_lcm:
            if self._verbose:
                print(f"[AdaptiveLCM] 使用 LCM 流式方案")
            yield from self._get_lcm().chat_stream(user_query, session_id)
        else:
            if self._verbose:
                print(f"[AdaptiveLCM] 使用传统流式方案")
            yield from self._traditional_chat_stream(user_query)

    def _traditional_chat(self, user_query: str) -> str:
        """传统方案：直接调用 LLM，不注入 chunk"""
        messages = [
            {"role": "system", "content": "你是一个 helpful AI 助手。"},
            {"role": "user", "content": user_query},
        ]
        result = []
        for chunk in self._client.chat_stream(messages):
            result.append(chunk)
        return "".join(result)

    def _traditional_chat_stream(
        self, user_query: str
    ) -> Generator[str, None, None]:
        """传统流式方案"""
        messages = [
            {"role": "system", "content": "你是一个 helpful AI 助手。"},
            {"role": "user", "content": user_query},
        ]
        yield from self._client.chat_stream(messages)

    @property
    def session(self) -> Optional[LCMSession]:
        if self._lcm_client:
            return self._lcm_client.session
        return None

    @property
    def stats(self) -> Dict:
        if self._lcm_client:
            return self._lcm_client.stats
        return {"mode": "traditional", "router": self.router.get_info()}

    def print_session_report(self):
        if self._lcm_client:
            self._lcm_client.print_session_report()
        else:
            print("[AdaptiveLCM] 传统模式，无 LCM 会话报告")

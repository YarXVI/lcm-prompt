"""
LCM v2 统一日志系统
提供结构化日志、错误码体系和可配置的日志级别
"""
import logging
import sys
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime


class LCMErrorCode(str, Enum):
    """LCM 错误码体系"""
    # 存储相关 (1xxx)
    STORE_PERSISTENCE_FAILED = "LCM1001"
    STORE_LOAD_FAILED = "LCM1002"
    STORE_GRAPH_SAVE_FAILED = "LCM1003"
    STORE_GRAPH_LOAD_FAILED = "LCM1004"
    
    # 编排器相关 (2xxx)
    ORCH_STREAM_EXCEPTION = "LCM2001"
    ORCH_MAX_ROUNDS_EXCEEDED = "LCM2002"
    ORCH_CHUNK_NOT_FOUND = "LCM2003"
    
    # 检测器相关 (3xxx)
    DETECTOR_BUFFER_OVERFLOW = "LCM3001"
    
    # 客户端相关 (4xxx)
    CLIENT_LLM_ERROR = "LCM4001"
    
    # 通用 (9xxx)
    UNKNOWN_ERROR = "LCM9999"


class LCMLogger:
    """LCM 统一日志器"""
    
    _instance: Optional["LCMLogger"] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, name: str = "lcm", level: int = logging.INFO):
        if self._initialized:
            return
        
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        # 清除已有处理器，避免重复
        self._logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 格式化器
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        self._initialized = True
    
    def debug(self, msg: str, **kwargs):
        self._logger.debug(self._format(msg, **kwargs))
    
    def info(self, msg: str, **kwargs):
        self._logger.info(self._format(msg, **kwargs))
    
    def warning(self, msg: str, **kwargs):
        self._logger.warning(self._format(msg, **kwargs))
    
    def error(self, msg: str, code: Optional[LCMErrorCode] = None, **kwargs):
        formatted = self._format(msg, **kwargs)
        if code:
            formatted = f"[{code}] {formatted}"
        self._logger.error(formatted)
    
    def critical(self, msg: str, code: Optional[LCMErrorCode] = None, **kwargs):
        formatted = self._format(msg, **kwargs)
        if code:
            formatted = f"[{code}] {formatted}"
        self._logger.critical(formatted)
    
    @staticmethod
    def _format(msg: str, **kwargs) -> str:
        if kwargs:
            extras = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{msg} | {extras}"
        return msg
    
    def set_level(self, level: int):
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)


# 全局日志器实例
logger = LCMLogger()


def get_logger() -> LCMLogger:
    """获取全局日志器"""
    return logger

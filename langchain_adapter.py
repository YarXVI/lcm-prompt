"""
LCM v2 LangChain 适配器
将 LCM 集成到 LangChain 生态
"""
from typing import List, Dict, Any, Optional, Iterator
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from .lcm_types import ContextChunk
from .store import ChunkStoreV2
from .client import LCMClientV2


class LCMRetriever(BaseRetriever):
    """
    LCM Retriever for LangChain
    
    用法：
        store = ChunkStoreV2()
        store.add_chunk(ContextChunk(...))
        
        retriever = LCMRetriever(store=store, k=5)
        docs = retriever.invoke("查询内容")
    """

    store: ChunkStoreV2 = Field(..., description="LCM Chunk 存储")
    k: int = Field(default=5, description="返回文档数")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """获取相关文档"""
        # 使用 LCM 的搜索功能
        results = self.store.search(query, top_k=self.k)

        documents = []
        for chunk_id, score in results:
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                documents.append(Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "summary": chunk.summary,
                        "source": chunk.source,
                        "score": score,
                        "tokens": chunk.tokens,
                    },
                ))

        return documents


class LCMEmbeddings:
    """
    LCM Embeddings 包装器
    将 LCM chunk 的 embedding 暴露给 LangChain
    """

    def __init__(self, store: ChunkStoreV2):
        self.store = store

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        # 查找已有 chunk 的 embedding
        embeddings = []
        for text in texts:
            # 简单匹配：查找内容包含该文本的 chunk
            found = False
            for chunk_id in list(self.store._chunks.keys()):
                chunk = self.store.get_chunk(chunk_id)
                if chunk and text in chunk.content:
                    if chunk.embedding:
                        embeddings.append(chunk.embedding)
                        found = True
                        break
            if not found:
                # 返回零向量
                embeddings.append([0.0] * 384)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        # 查找匹配的 chunk
        for chunk_id in list(self.store._chunks.keys()):
            chunk = self.store.get_chunk(chunk_id)
            if chunk and text in chunk.content:
                if chunk.embedding:
                    return chunk.embedding
        return [0.0] * 384


class LCMTool:
    """
    LCM Tool for LangChain Agents
    
    允许 LangChain Agent 使用 LCM 加载 chunk
    """

    def __init__(self, client: LCMClientV2):
        self.client = client

    def load_chunk(self, chunk_id: str) -> str:
        """
        加载指定 chunk
        
        Args:
            chunk_id: Chunk ID
        
        Returns:
            Chunk 内容
        """
        chunk = self.client.store.get_chunk(chunk_id)
        if chunk:
            return chunk.content
        return f"Chunk {chunk_id} 未找到"

    def search_chunks(self, query: str) -> str:
        """
        搜索相关 chunks
        
        Args:
            query: 查询字符串
        
        Returns:
            搜索结果摘要
        """
        results = self.client.store.search(query)
        if not results:
            return "未找到相关 chunks"

        summaries = []
        for chunk_id, score in results[:5]:
            chunk = self.client.store.get_chunk(chunk_id)
            if chunk:
                summaries.append(f"- {chunk_id} (相关度: {score:.2f}): {chunk.summary}")

        return "\n".join(summaries)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取工具定义（用于 OpenAI function calling）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "load_chunk",
                    "description": "加载指定的上下文 chunk",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "Chunk 的唯一标识符",
                            },
                        },
                        "required": ["chunk_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_chunks",
                    "description": "搜索相关的上下文 chunks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

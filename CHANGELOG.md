# Changelog

All notable changes to the LCM Protocol project will be documented in this file.

## [2.0.0] - 2026-05-12

### Added
- **Core Protocol**: Lazy Context Materialization with sentinel-based chunk loading
- **Streaming Detection**: Real-time `[NEED_CHUNK:id]` detection in LLM output streams
- **Token Budget**: Intelligent context window management with utilization tracking
- **Chunk Dependency Graph**: DAG-based dependency resolution with topological sorting
- **Hybrid Mode**: Hot chunk direct injection + cold chunk LCM loading
- **Provider Router**: Automatic cloud vs local API detection and strategy switching
- **KV Cache Optimization**: Anthropic/DeepSeek prompt caching integration
- **Async I/O**: Asynchronous LLM calls and concurrent chunk loading
- **Multimodal Support**: Image, PDF, and audio chunk handling
- **Distributed Storage**: Multi-node chunk storage with consistent hashing
- **Protocol Versioning**: v1/v2/v3 compatibility with feature negotiation
- **SQLite Backend**: Optional SQLite persistence with FTS5 full-text search
- **HNSW Vector Index**: Approximate nearest neighbor search for semantic retrieval
- **Chunk Compression**: gzip/zstd compression for memory and disk optimization
- **Quality Evaluation**: Systematic answer quality assessment framework
- **Multi-Model Benchmark**: Cross-model latency and quality comparison
- **Multi-Agent Collaboration**: Shared index with deduplication across agents
- **Comprehensive Logging**: Structured logging with error code system
- **Thread Safety**: All operations protected by RLock

### Security
- Sentinel buffer overflow protection with safe truncation
- Concurrent access protection for all storage operations
- Graceful degradation on persistence corruption

### Performance
- LRU cache for hot chunks
- Batch loading and speculative prefetching
- Token estimation with tiktoken support (when available)
- Efficient JSONL persistence with incremental updates

## [1.0.0] - 2025-01-15

### Added
- Initial LCM protocol implementation
- Basic chunk storage and retrieval
- Simple sentinel detection
- In-memory operation only

from typing import List, Dict


class CacheAwarePrefixBuilder:

    def build_cached_messages(
        self,
        system_prompt: str,
        dynamic_messages: List[Dict[str, str]],
        chunk_summaries: str,
    ) -> List[Dict[str, str]]:
        static_prefix = f"{system_prompt}\n\n## Chunk Index\n{chunk_summaries}"
        return [
            {"role": "system", "content": static_prefix},
        ] + dynamic_messages

    def build_chunk_index(
        self,
        chunk_summaries: List[Dict[str, str]],
    ) -> str:
        if not chunk_summaries:
            return "[no chunks available]"
        lines = []
        for s in chunk_summaries:
            cid = s.get("chunk_id", "unknown")
            source = s.get("source", "unknown")
            tokens = s.get("tokens", 0)
            summary = s.get("summary", "")
            lines.append(
                f"- **{cid}** [{source}] ({tokens} tokens): {summary}"
            )
        return "\n".join(lines)

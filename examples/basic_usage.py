"""
Basic Usage Example
Demonstrates how to use LCM with a local model endpoint.
"""

import sys
import os
import httpx
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcm import ChunkStore, ContextChunk, LCMClient, build_initial_messages, load_env

load_env()

ENDPOINT = os.environ.get("LCM_ENDPOINT", "http://localhost:1234/v1")
MODEL = os.environ.get("LCM_MODEL", "local-model")

CODE_SNIPPETS = {
    "auth_service": """
class AuthService:
    SECRET_KEY = "sk-prod-2026-hardcoded-secret-key-do-not-commit"
    
    def authenticate(self, token):
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=['HS256'])
            return payload.get('user')
        except Exception as e:
            print(f"Auth error: {e}, token was: {token}")
            return None
""".strip(),
    "database_layer": """
class DatabaseLayer:
    DSN = "postgresql://admin:SuperSecret123@db.internal:5432/production"
    
    def get_user(self, user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        return self.db.execute(query)
    
    def search_products(self, keyword):
        query = f"SELECT name, price FROM products WHERE name LIKE '%{keyword}%'"
        return self.db.execute(query)
""".strip(),
    "file_handler": """
class FileHandler:
    UPLOAD_DIR = "/var/www/uploads"
    
    def save_file(self, filename, content):
        filepath = os.path.join(self.UPLOAD_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        os.chmod(filepath, 0o777)
        return filepath
""".strip(),
}


def make_stream_fn():
    """Create a stream function for the local LLM endpoint."""

    def stream_fn(messages):
        url = f"{ENDPOINT}/chat/completions"
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": True,
        }
        t0 = time.time()
        try:
            with httpx.Client(timeout=300) as client:
                with client.stream("POST", url, json=payload) as resp:
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"\n[ERROR: {e}]"

        elapsed = (time.time() - t0) * 1000
        print(f"\n--- Call completed in {elapsed:.0f}ms ---")

    return stream_fn


def main():
    store = ChunkStore()
    for name, code in CODE_SNIPPETS.items():
        store.add_chunk(ContextChunk(
            chunk_id=name,
            content=code,
            summary=f"{name}: {code.split(chr(10))[0]} - ~{len(code)//4} tokens",
            tokens=len(code) // 4,
            source="codebase",
        ))

    stream_fn = make_stream_fn()
    client = LCMClient(chunk_store=store, stream_fn=stream_fn, enable_prefetch=False)

    query = "Review all code for security vulnerabilities. List each finding concisely."

    print("=== LCM Code Security Review ===\n")
    for text in client.chat_stream(query):
        print(text, end="", flush=True)
    print("\n\n=== Done ===")


if __name__ == "__main__":
    main()
